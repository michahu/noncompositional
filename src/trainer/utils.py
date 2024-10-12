import math
import numpy as np
import wandb

import torch

from chimera.dataset import get_train_dataloader


def make_state_vectors(results):
    if results is None:
        return [np.array([0.0, 0.0])]
    else:
        return [
            torch.tensor(res["task_loss"].to_numpy(), dtype=torch.float32)
            for res in results
        ]


def unroll_lm(
    fabric,
    args,
    tokenizer,
    weights,
    train_data_generator,
    inner_states,
    K,
    max_grad_norm=1.0,
):
    """
    This function is for iterable datasets, which scales better to large pretraining corpora.
    """
    for weight, state in zip(weights, inner_states):
        weight = weight.squeeze(0).numpy()
        train_data_generator.set_proportions(args, weight)

        tokenized_train = get_tokenized_train_dataset(
            args, train_data_generator, K * args.bsz
        )
        train_dataloader = fabric.setup_dataloaders(
            get_train_dataloader(tokenizer, tokenized_train, args.bsz)
        )
        train_dataloader = iter(train_dataloader)

        (inner_model, optimizer, lr_scheduler, t) = state
        for _ in range(K):
            batch = next(train_dataloader)
            # for val in batch['domain_id']:
            #     empirical_counts[val] += 1
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            is_accumulating = t % args.gradient_accumulation_steps != 0
            with fabric.no_backward_sync(inner_model, enabled=is_accumulating):
                output = inner_model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    # **batch
                )
                fabric.backward(output.loss)
                # fabric.print(output.loss.item())
            if not is_accumulating:
                fabric.clip_gradients(inner_model, optimizer, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                t += 1

        state = (inner_model, optimizer, lr_scheduler, t)

    wandb.log({"inner_loss": output.loss.item()})
    fabric.print(f"Inner loss sample: {output.loss.item()}")


def unroll(
    fabric,
    args,
    tokenizer,
    weights,
    train_data_generator,
    inner_states,
    K,
    val_samples=None,
    max_grad_norm=1.0,
):
    """
    This function is generally used in synthetic experiments, where we can create a batch of data
    of the correct length on the fly.

    TODO: add a running loss to replace the results computation
    """

    for i, (weight, state) in enumerate(zip(weights, inner_states)):
        inner_model, optimizer, lr_scheduler, t = state
        # detach if weight is a tensor
        if torch.is_tensor(weight):
            weight = weight.squeeze(0).detach().cpu().numpy()
        train_data_generator.set_proportions(args, weight)
        tokenized_train = get_tokenized_train_dataset(
            args, train_data_generator, K * args.bsz, all_samples=val_samples
        )
        train_dataloader = get_train_dataloader(tokenizer, tokenized_train, args.bsz)
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

        for batch in train_dataloader:
            inner_model.train()
            outputs = inner_model(**batch)
            loss = outputs.loss
            fabric.backward(loss.mean())
            fabric.clip_gradients(inner_model, optimizer, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            t += 1

        inner_states[i] = (inner_model, optimizer, lr_scheduler, t)


def cos_schedule(
    it, warmup_iters=2000, lr_decay_iters=600000, learning_rate=6e-4, min_lr=6e-5
):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_tokenized_train_dataset(args, train_dataset, n_data, all_samples=None):
    if args.task_name in ["lego", "addition"]:
        tokenized_train = train_dataset.get_tokenized_dataset(
            n_data, all_samples=all_samples, include_skill_idxs=args.curriculum
        )
    elif args.task_name == "ni":
        tokenized_train = train_dataset.get_tokenized_dataset()
    elif args.task_name == "cookbook":
        tokenized_train, _ = train_dataset.get_tokenized_dataset(n_data, all_samples)
    else:
        tokenized_train = train_dataset.get_tokenized_dataset(n_data)
    return tokenized_train


def get_steps(args, n_epochs=None):
    """Computes the number of steps per checkpoint and the total number of training steps."""
    if n_epochs is None:
        n_epochs = args.n_epochs
    ckpt_steps = (
        int((args.n_select * n_epochs / args.bsz) / args.num_ckpts)
        if args.max_steps == -1
        else int(args.max_steps * n_epochs / args.num_ckpts)
    )

    total_steps = (
        args.max_steps * n_epochs
        if args.max_steps != -1
        else int(n_epochs * args.n_select / args.bsz)
    )
    print(f"Total steps: {total_steps} Steps per checkpoint: {ckpt_steps}")

    return ckpt_steps, total_steps


def get_update_steps(args, total_steps):
    """Computes the number of samples per update and the number of total updates (e.g., number of rounds T)."""
    update_size = args.update_steps * args.bsz
    n_updates = total_steps / args.update_steps
    return update_size, n_updates

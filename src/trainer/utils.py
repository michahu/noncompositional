import math
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader


def make_state_vectors(results):
    if results is None:
        return [np.array([0.0, 0.0])]
    else:
        return [
            torch.tensor(res["task_loss"].to_numpy(), dtype=torch.float32)
            for res in results
        ]


def unroll(
    tokenizer,
    weights,
    train_data_generator,
    inner_states,
    K,
    bsz,
    max_grad_norm=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (weight, state) in enumerate(zip(weights, inner_states)):
        inner_model, optimizer, lr_scheduler, t = state
        inner_model.to(device)

        if torch.is_tensor(weight):
            weight = weight.squeeze(0).detach().cpu().numpy()
        train_data_generator.set_proportions(weight)
        tokenized_train = train_data_generator.get_tokenized_dataset(K * bsz)

        train_dataloader = DataLoader(
            tokenized_train,
            batch_size=bsz,
            shuffle=True,
            collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt", padding=True),
        )

        for batch in train_dataloader:
            inner_model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = inner_model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inner_model.parameters(), max_grad_norm)
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

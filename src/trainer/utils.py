import math
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader


def make_state_vectors(results, N, output_dim):
    if results is None:
        return [torch.zeros(output_dim) for _ in range(N)]
    else:
        assert len(results) == N
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

from transformers import AutoTokenizer, set_seed
import numpy as np
import wandb
import fire

import torch
import warnings

from src.dataset import MixingDatasetCOGS
from src.evaluator import HFEvaluator
from src.trainer.pes_trainer import PESTrainer

from utils import get_logger, make_output_dir, init_inner_state, init_outer_state


def main(
    seed,
    model_name,
    train_path,
    val_path,
    output_dir,
    lr,
    T,
    K,
    bsz,
    warmup_steps,
    proportions=None,
    input_dim=3,
    output_dim=3,
    meta_lr=0.1,
    max_steps=1000,
    sigma=0.1,
    seq_length=128,
    num_particles=4,
    run_type="meta",
    do_breakstep=False,
    skills=None,
):
    output_dir_path = make_output_dir(output_dir)
    logger = get_logger(output_dir_path)

    for param, value in locals().items():
        logger.info(f"{param}: {value}")

    np.set_printoptions(precision=3, suppress=True)
    torch.set_float32_matmul_precision("medium")

    # TODO: figure out where this warning is coming from
    warnings.filterwarnings(
        "ignore",
        message=".*To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach().*",
    )
    run_name = f"{num_particles}"
    wandb.init(project="noncompositional", name=run_name)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = MixingDatasetCOGS(
        logger,
        tokenizer,
        seed,
        data_path=train_path,
        sample_rule="mixture",
        is_eval=False,
        seq_length=seq_length,
        skills=skills,
        add_skill_idx=True,
    )
    validation_data = MixingDatasetCOGS(
        logger,
        tokenizer,
        seed,
        data_path=val_path,
        sample_rule="mixture",
        is_eval=True,
        seq_length=seq_length,
        skills=skills,
        add_skill_idx=True,
    ).get_tokenized_dataset()
    validation_dataloader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=bsz,
        shuffle=False,
        collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt", padding=True),
    )

    # setup to construct meta trainer
    inner_init_fn = lambda: init_inner_state(model_name, lr, T, warmup_steps)
    outer_init_fn = lambda: init_outer_state(input_dim, output_dim, meta_lr, max_steps)

    evaluator = HFEvaluator(logger, tokenizer, seed, output_dir_path, bsz=bsz * 2)

    trainer = PESTrainer(
        logger,
        tokenizer,
        evaluator,
        inner_init_fn,
        outer_init_fn,
        num_particles,
        seed,
        T,
        K,
        bsz,
        sigma=sigma,
        max_steps=max_steps,
        save_every=100,
    )

    if do_breakstep:
        n_breaksteps = num_particles // 2
        for i, step_increase in enumerate(np.arange(0, T, T // n_breaksteps)):
            model, optimizer, scheduler, t = trainer.inner_states.particles[i * 2]
            trainer.inner_states.particles[i * 2] = (
                model,
                optimizer,
                scheduler,
                t + step_increase,
            )

            model, optimizer, scheduler, t = trainer.inner_states.particles[i * 2 + 1]
            trainer.inner_states.particles[i * 2 + 1] = (
                model,
                optimizer,
                scheduler,
                t + step_increase,
            )

    if run_type == "meta":
        trainer.metatrain(
            train_data,
            validation_dataloader,
        )
    elif run_type == "train":
        inner_state = trainer.train(
            train_data, validation_dataloader, proportions=proportions
        )
        # save to output_dir
        torch.save(inner_state, f"{output_dir_path}/inner_state.pth")
    else:
        raise ValueError(f"Invalid run_type: {run_type}")


if __name__ == "__main__":
    fire.Fire(main)

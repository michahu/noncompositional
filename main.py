from transformers import AutoTokenizer, set_seed
import numpy as np
import wandb
import fire

import torch
import warnings

from src.dataset import MixingDataset
from src.evaluator import HFEvaluator
from src.trainer.pes_trainer import PESTrainer

from utils import get_logger, make_output_dir, init_inner_state, init_outer_state


def get_dataset(logger, tokenizer, seed, is_eval, data_path):
    return MixingDataset(
        logger, tokenizer, seed, sample_rule="mixture", is_eval=is_eval
    )


def main(
    seed,
    model_name,
    train_path,
    val_path,
    output_dir,
    lr,
    T,
    warmup_steps,
    input_dim,
    output_dim,
    meta_lr,
    meta_steps,
    num_particles=4,
    run_type="meta",
    do_breakstep=False,
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

    train_data = get_dataset(logger, tokenizer, seed, False, train_path)
    validation_data = get_dataset(
        logger, tokenizer, seed, True, val_path
    ).get_tokenized_dataset()

    # setup to construct meta trainer
    inner_init_fn = lambda: init_inner_state(model_name, lr, T, warmup_steps)
    outer_init_fn = lambda: init_outer_state(input_dim, output_dim, meta_lr, meta_steps)

    # evaluator also saves current args to file
    evaluator = HFEvaluator(logger, tokenizer, seed, output_dir_path)

    trainer = PESTrainer(
        logger,
        tokenizer,
        evaluator,
        inner_init_fn,
        outer_init_fn,
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

    # args.run_type is different from args.alg because meta algs also implement trainer.train()
    if run_type == "meta":
        trainer.metatrain(
            train_data,
            validation_data,
        )
    elif run_type == "train":
        trainer.train(
            train_data,
            validation_data,
        )
    else:
        raise ValueError(f"Invalid run_type: {run_type}")


if __name__ == "__main__":
    fire.Fire(main)

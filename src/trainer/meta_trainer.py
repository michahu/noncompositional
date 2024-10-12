import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import unroll, unroll_lm, make_state_vectors


class OuterState:
    def __init__(self, outer_init_fn, warm_start_path=None) -> None:
        self.init_fn = outer_init_fn
        self.model, self.optimizer, self.scheduler = outer_init_fn()
        if warm_start_path is not None:
            self.model.load_state_dict(torch.load(warm_start_path))

    def reset(self):
        self.model, self.optimizer, self.scheduler = self.init_fn()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

    def save(self, output_dir, name):
        torch.save(self.model.state_dict(), f"{output_dir}/outer_model_{name}.pth")


class InnerStateBuffer:
    def __init__(self, evaluator, num_particles, init_fn) -> None:
        self.init_fn = init_fn
        self.num_particles = num_particles
        self.evaluator = evaluator
        self.reset()

    @staticmethod
    def evaluate(particle, evaluator, validation_data, t, output_idxs=None, save=False):
        model, _, _, _ = particle
        return evaluator.evaluate(
            model, validation_data, t, weights=None, output_idxs=output_idxs, save=save
        )

    def reset(self) -> list:
        """
        Reset all particles in the buffer.
        """
        self.particles = [self.init_fn() for _ in range(self.num_particles)]


class MetaTrainer:
    def __init__(
        self,
        logger,
        tokenizer,
        evaluator,
        inner_init_fn,
        outer_init_fn,
        num_particles,
        T,
        K,
        max_steps,
        save_every,
        train_is_iterable=False,
    ):
        self.logger = logger
        self.tokenizer = tokenizer
        self.evaluator = evaluator

        # reset inner and outer model
        self.outer_state = OuterState(outer_init_fn)
        self.inner_states = InnerStateBuffer(evaluator, num_particles, inner_init_fn)

        self.train_is_iterable = train_is_iterable

        assert T % K == 0  # otherwise your effective T will be longer than requested.
        self.T = T
        self.K = K
        self.max_steps = max_steps
        self.save_every = save_every

    def train(
        self,
        fabric,
        train_data,
        validation_data,  # sometimes used to decontaminate the training dataset
        filter_samples=None,
        output_idxs=None,
    ):
        """
        train_data: either a generator or an iterable dataset, depending on the run.
            Iterable datasets rely on the train_dataset_pointer to change the mixing distribution in the inner loop.
        """
        inner_state = self.inner_states.init_fn()
        results = self.inner_states.evaluate(
            inner_state, self.evaluator, validation_data, 0, output_idxs=output_idxs
        )

        progress_bar = tqdm(
            range(int(self.T / self.K)),
            disable=(not fabric.global_rank == 0),
        )
        for step in range(0, self.T, self.K):
            with torch.inference_mode():
                state_vector = make_state_vectors(self.args, [results])
                weights = [self.outer_state.model(state_vector[0].unsqueeze(0))]
            if self.train_is_iterable:
                unroll_lm(
                    self.tokenizer,
                    weights,
                    train_data,
                    [inner_state],
                    step,
                    self.K,
                )
            else:
                unroll(
                    self.tokenizer,
                    weights,
                    train_data,
                    [inner_state],
                    self.K,
                    val_samples=filter_samples,
                )
            results = self.inner_states.evaluate(
                inner_state,
                self.evaluator,
                validation_data,
                step + self.K,
                output_idxs=output_idxs,
                save=True,
            )

            if fabric.global_rank == 0:
                progress_bar.update(1)

        return inner_state

    def metatrain(
        self,
        fabric,
        train_data,
        validation_data,
        output_dir_path,
        train_dataloader=None,
        filter_samples=None,
        output_idxs=None,
        lockstep=True,
    ):
        raise NotImplementedError

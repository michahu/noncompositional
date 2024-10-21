import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import unroll, make_state_vectors


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
        return evaluator.evaluate(model, validation_data, t, weights=None, save=save)

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
        seed,
        T,
        K,
        bsz,
        sigma,
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
        self.num_particles = num_particles
        self.seed = seed

        self.train_is_iterable = train_is_iterable

        assert T % K == 0  # otherwise your effective T will be longer than requested.
        self.T = T
        self.K = K
        self.bsz = bsz
        self.sigma = sigma
        self.max_steps = max_steps
        self.save_every = save_every

    def train(
        self,
        train_data,
        validation_data,
        proportions=None,
        output_idxs=None,
    ):
        """
        train_data: either a generator or an iterable dataset, depending on the run.
            Iterable datasets rely on the train_dataset_pointer to change the mixing distribution in the inner loop.
        """
        inner_state = self.inner_states.init_fn()
        results = None

        if proportions is not None:
            assert len(proportions) == self.outer_state.model.input_dim
            weights = [torch.tensor(proportions, dtype=torch.float32)]
        else:
            weights = [torch.ones(1, self.outer_state.model.input_dim)]

        progress_bar = tqdm(range(int(self.T / self.K)))
        for step in range(0, self.T, self.K):
            if proportions is None:
                with torch.inference_mode():
                    state_vector = make_state_vectors(
                        [results], 1, self.outer_state.model.input_dim
                    )
                    weights = [self.outer_state.model(state_vector[0].unsqueeze(0))]

            unroll(
                self.tokenizer,
                weights,
                train_data,
                [inner_state],
                self.K,
                self.bsz,
            )
            self.inner_states.evaluate(
                inner_state,
                self.evaluator,
                validation_data,
                step + self.K,
                output_idxs=output_idxs,
                save=True,
            )

            progress_bar.update(1)

        return inner_state

    def metatrain(
        self,
        train_data,
        validation_data,
        output_dir_path,
        train_dataloader=None,
        output_idxs=None,
        lockstep=True,
    ):
        raise NotImplementedError

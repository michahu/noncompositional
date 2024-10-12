import torch
import torch.nn.functional as F

import copy
from tqdm import tqdm
import wandb

from .meta_trainer import MetaTrainer
from .utils import unroll, unroll_lm, make_state_vectors


def sample_noise(outer_model, sigma):
    pos_noise = []
    neg_noise = []
    for p in outer_model.parameters():
        noise = torch.randn_like(p) * sigma
        pos_noise.append(noise)
        neg_noise.append(-noise)
    return pos_noise, neg_noise


class PESTrainer(MetaTrainer):
    @staticmethod
    @torch.inference_mode()
    def get_pes_weights(N, sigma, outer_model, state_vectors, is_fixed=False):
        """
        sampling weights for current outer step
        """
        weights = []
        perts = []
        for i in range(0, N, 2):
            pos, neg = sample_noise(outer_model, sigma)
            old_dict = copy.deepcopy(outer_model.state_dict())

            for p, n in zip(outer_model.parameters(), pos):
                p.data += n

            if is_fixed:
                distr = F.softmax(outer_model.weight.data)
                weights.append(distr.squeeze(0))
            else:
                weights.append(outer_model(state_vectors[i].unsqueeze(0)))

            outer_model.load_state_dict(old_dict)

            for p, n in zip(outer_model.parameters(), neg):
                p.data += n

            if is_fixed:
                distr = F.softmax(outer_model.weight.data)
                weights.append(distr.squeeze(0))
            else:
                weights.append(outer_model(state_vectors[i + 1].unsqueeze(0)))

            outer_model.load_state_dict(old_dict)
            perts.append(pos)
            perts.append(neg)

        return weights, perts

    def metatrain(
        self,
        fabric,
        train_data_generator,
        validation_data,
        filter_samples=None,
        output_idxs=None,
    ):
        """
        output_idxs: The index of the answer token in the output sequence. Not relevant for every problem.
        PES samples perturbations every unroll, or T // K times per inner loop.
        """

        # results = self.inner_states.evaluate(self.inner_states.particles, self.evaluator, validation_data, 0, output_idxs=output_idxs)
        # results  = [self.inner_states.evaluate(p, self.evaluator, validation_data, 0, output_idxs=output_idxs) for p in self.inner_states.particles]
        results = None

        progress_bar = tqdm(
            range(self.args.max_steps),
            disable=(not fabric.global_rank == 0),
        )
        for step in range(self.args.max_steps):
            if step % self.args.save_every == 0 and step > 0:
                # reset
                self.outer_state.save(
                    self.evaluator.result_path, f"{step}_{self.args.seed}"
                )

            results = self.grad(
                fabric,
                results,
                train_data_generator,
                validation_data,
                step,
                filter_samples=filter_samples,
                output_idxs=output_idxs,
            )

            self.outer_state.step()

            # step through particles, change results if necessary
            for i in range(self.num_particles):
                _, _, _, t = self.inner_states.particles[i]
                if t >= self.T:
                    self.inner_states.particles[i] = self.inner_states.init_fn()
                    results[i] = self.inner_states.evaluate(
                        self.inner_states.particles[i],
                        self.evaluator,
                        validation_data,
                        0,
                        output_idxs=output_idxs,
                    )

            if fabric.global_rank == 0:
                progress_bar.update(1)

        self.outer_state.save(self.evaluator.result_path, f"final_{self.args.seed}")

        # Meta-evaluation
        self.train(
            fabric,
            train_data_generator,
            validation_data,
            filter_samples=filter_samples,
            output_idxs=output_idxs,
        )

    def grad(
        self,
        fabric,
        results,
        train_data,
        validation_data,
        outer_step,
        filter_samples=None,
        output_idxs=None,
        return_weights=False,
    ):
        """
        Computes the ES-Single gradient.
        """
        N = self.args.num_particles

        state_vectors = make_state_vectors(self.args, results)
        # fabric.print(self.outer_state.model.weight.data)
        weights, perts = self.get_pes_weights(
            N,
            self.args.sigma,
            self.outer_state.model,
            state_vectors,
            self.args.outer_model == "fixed",
        )

        if self.train_is_iterable:
            unroll_lm(
                fabric,
                self.args,
                self.tokenizer,
                weights,
                train_data,
                self.inner_states.particles,
                self.args.K,
            )
        else:
            unroll(
                fabric,
                self.args,
                self.tokenizer,
                weights,
                train_data,
                self.inner_states.particles,
                self.args.K,
                val_samples=filter_samples,
            )
        # results  = [self.inner_states.evaluate(p, self.evaluator, validation_data, outer_step, output_idxs=output_idxs) for p in self.inner_states.particles]

        if self.args.val_task_name == "blimp":
            objs = [1 - res["accuracy"].to_numpy().mean() for res in results]
        else:
            objs = [res["task_loss"].to_numpy().mean() for res in results]

        grads = []
        for layer in zip(*perts):
            stacked = torch.stack([l * o for l, o in zip(layer, objs)])
            grads.append(stacked.mean(dim=0) / (N * self.args.sigma**2))

        for p, n in zip(self.outer_state.model.parameters(), grads):
            p.grad = n

        if return_weights:
            return results, weights
        return results

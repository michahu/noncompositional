import os
import torch
import pandas as pd
import pickle
from collections import defaultdict


def save_loss(loss_dict, result_path, seed, counter):
    loss_file = f"seed_{seed}_checkpoint-{counter}.pkl"
    loss_path = os.path.join(result_path, loss_file)
    if isinstance(loss_dict, pd.DataFrame):
        loss_dict.to_pickle(loss_path)
    else:
        with open(loss_path, "wb") as f:
            pickle.dump(loss_dict, f)
    return loss_path


def save_weights(weights, result_path, seed, counter):
    if weights is not None:
        weights /= sum(weights)
        weights_dict = {
            skill_idx: weights[skill_idx] for skill_idx in range(len(weights))
        }
        weights_file = f"seed_{seed}_proportions_checkpoint-{counter}.pkl"
        weights_path = os.path.join(result_path, weights_file)
        with open(weights_path, "wb") as f:
            pickle.dump(weights_dict, f)


class Evaluator:
    def __init__(
        self,
        logger,
        tokenizer,
        seed,
        output_dir_path,
        bsz=64,
        eval_batches=None,
        evaluate_accuracy=False,
    ):
        """
        eval_batches: the number of batches to evaluate on. If None, we evaluate on the entire dataset.
        """
        self.logger = logger
        self.tokenizer = tokenizer
        self.seed = seed
        self.output_dir_path = output_dir_path
        self.evaluate_accuracy = evaluate_accuracy
        self.bsz = bsz
        self.eval_batches = eval_batches

    def evaluate(self, model, tokenized_data, counter, weights, output_idxs, train):
        """Evaluates the model on a given dataset by computing and saving the loss per sample.

        Args:
        - tokenized_data: a torch dataset to evaluate the model on. This is typically the validation dataset, but can also be the training dataset when we are running a curriculum learning baseline.
        - counter: the training step at which the model is evaluated. This is used to help name the results file.
        - weights: if this is not None, we also save the weight per skill at the given training step.
        - output_idxs: if this is not None, we mask all but this index of the sample.
        - train: if this is True, we evaluate on the training dataset.
        """
        pass

    def _evaluate_train(self):
        """Evaluates the model on training data. Returns a list of losses in the same order as the dataset.
        If args.group_curriculum is set, we also return a list containing the skill of each sample in order.
        """
        pass

    def _evaluate_val(self):
        """Evaluates the model on validation data. Returns a dictionary mapping from each skill to the list of losses corresponding to samples associated with that skill."""
        pass


class HFEvaluator(Evaluator):
    @torch.inference_mode()
    def evaluate(self, model, dataloader, counter, weights, save=False):
        """
        This evaluator computes the eval loss.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        loss_dict = defaultdict(list)

        # use commented out pattern for multiple eval tasks
        # for i, test_dataloader in enumerate(data):
        #     running_loss = 0

        running_loss = 0
        for j, batch in enumerate(dataloader):
            if j == self.eval_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            running_loss += loss.item()

        if self.eval_batches is not None:
            loss_dict[0] = running_loss / min(len(dataloader), self.eval_batches)
        else:
            loss_dict[0] = running_loss / len(dataloader)

        result_df = pd.DataFrame.from_dict(
            loss_dict, orient="index", columns=["task_loss"]
        )

        if save:
            save_loss(loss_dict, self.output_dir_path, self.seed, counter)
            save_weights(weights, self.output_dir_path, self.seed, counter)
        return result_df

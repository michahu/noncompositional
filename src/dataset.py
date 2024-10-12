import numpy as np
from datasets import load_from_disk, concatenate_datasets


class AbstractDataset:
    """
    The AbstractDataset class is a dataset that can be filtered and sampled from.
    """

    def __init__(
        self, args, logger, tokenizer, seed, sample_rule, is_eval, data_path=None
    ):
        self.tokenizer = tokenizer
        self.logger = logger
        self.seq_length = args.seq_length
        self.sample_rule = sample_rule
        self.is_eval = is_eval
        self.data_path = data_path
        self.seed = seed

    def set_skills(self, args):
        """Sets the support of skills over which we are sampling from by processing args.slice_list."""
        pass

    def set_proportions(self, args, proportions):
        """Sets the proportions with which to sample each skill.

        Arguments:
        - args: args.graph is used (exp sum of weights) if proportions are not provided.
        - proportions: a list of values (not necessarily adding up to 1) that determine how frequently to sample each skill. This is used to update the skills mixture before and during training.
        """
        pass

    def get_tokenized_dataset(self, n_data):
        """Produce a train or validation dataset (depending on is_eval) of size n_data."""
        pass


class MixingDataset(AbstractDataset):
    """Loads several HuggingFace datasets. Constructs a mix depending on the set proportion per round."""

    def __init__(self, logger, tokenizer, seed, sample_rule, is_eval, data_path):
        super().__init__(logger, tokenizer, seed, sample_rule, is_eval, data_path)
        self.data = load_from_disk(self.data_path)

    def set_skills(self):
        self.skills = sorted(self.data.unique("slice"))
        self.k = len(self.skills)

    def set_proportions(self, proportions):
        if self.sample_rule == "mixture":
            proportions = np.array(proportions)
            proportions /= sum(proportions)
            self.proportions = proportions
            assert len(self.proportions) == self.k
        elif self.sample_rule == "stratified":
            self.proportions = np.repeat(1.0 / self.k, self.k)
        else:
            # set to uniform
            data_per_skill = []
            for i, s in enumerate(self.skills):
                data_per_skill.append(
                    len(
                        self.data.filter(
                            lambda x: x == s, input_columns="slice", num_proc=14
                        )
                    )
                )
            data_per_skill = np.array(data_per_skill)
            self.proportions = data_per_skill / data_per_skill.sum()

    def _tokenize(self, x):
        tokenized = self.tokenizer(
            x["text"],
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["input_ids"] = tokenized["input_ids"][0]
        tokenized["attention_mask"] = tokenized["attention_mask"][0]
        return tokenized

    def get_tokenized_dataset(self, n_data=None):
        if self.is_eval:
            return self._get_tokenized_val()
        else:
            return self._get_tokenized_train(n_data)

    def _get_tokenized_val(self):
        return (
            self.data.map(
                lambda x: self._tokenize(x),
                batched=True,
            ),
        )

    def _get_tokenized_train(self, n_data):
        n_per_skill = (n_data * self.proportions).astype(int)
        n_per_skill[-1] = n_data - n_per_skill[:-1].sum()

        self.logger.info(f"Probabilities: {list(zip(self.skills, self.proportions))}")
        all_data = []
        for i, s in enumerate(self.skills):
            skill_data = self.data.filter(
                lambda x: x == s, input_columns="slice", num_proc=8
            )
            if len(skill_data) < n_per_skill[i]:
                self.logger.warning(
                    f"Not enough samples in slice {s}. size is {len(skill_data)}, requested is {n_per_skill[i]}"
                )

            n = min(n_per_skill[i], len(skill_data))
            sample_idxs = np.random.choice(
                np.arange(len(skill_data)), size=n, replace=False
            )
            all_data.append(skill_data.select(sample_idxs))

        self.data = concatenate_datasets(all_data).shuffle()
        return self.data.map(
            lambda x: self._tokenize(x),
            batched=True,
            remove_columns=self.data.column_names,
        )

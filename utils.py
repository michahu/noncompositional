import logging
import os
import datetime

from transformers import get_scheduler, AutoConfig, AutoModelForCausalLM
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import torch
import torch.nn as nn
from torch.optim import AdamW


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, use_softmax=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.use_softmax = use_softmax

        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x


def get_logger(dir_path):
    # Create a logger
    logger = logging.getLogger("LLM-based evaluation")
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to output.log
    file_handler = logging.FileHandler(os.path.join(dir_path, "output.log"))
    file_handler.setLevel(logging.INFO)

    # Create a stream handler that prints to the screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.propagate = False

    return logger


def make_output_dir(output_dir):
    # run_id is MMDDYY
    run_id = datetime.datetime.now().strftime("%m%d%y")
    dir_path = os.path.join(output_dir, run_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def create_optimizer_scheduler(
    model, lr, max_steps, lr_scheduler_type, warmup_steps=50
):
    """
    Create AdamW optimizer and learning rate scheduler.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.00,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return optimizer, scheduler


def init_outer_state(input_dim, output_dim, meta_lr, meta_steps):
    """
    Default initialization for outer state
    No need to parallelize outer state because it's so small.
    """
    model = MLP(input_dim, output_dim)
    optim = torch.optim.Adam(model.parameters(), lr=meta_lr)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=meta_steps,
    )
    return model, optim, scheduler


def init_inner_state(model_name, lr, T, warmup_steps):
    """
    Default initialization for inner state
    """
    config = AutoConfig.from_pretrained(model_name)
    inner_model = AutoModelForCausalLM(config)
    optimizer, scheduler = create_optimizer_scheduler(
        inner_model,
        lr,
        T,
        warmup_steps,
    )

    return inner_model, optimizer, scheduler, 0

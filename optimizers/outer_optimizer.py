# def optim_adam_gradphi( lit_mod, lr ): 
#     """
#     optimizer for both the grad model and the prior cost
#     """
#     return torch.optim.Adam(
#         [
#             {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
#             {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr/2},
#         ],
#     )
"""
Outer Optimizer and Scheduler Factory
-------------------------------------
Provides unified functions to build meta-optimizers (outer loops)
and their learning rate schedulers for bi-level optimization setups.
"""

import copy
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    OneCycleLR,
    LambdaLR,
)


def build_outer_optimizer(params, name="adam", lr=1e-3, **kwargs):
    """
    Create an outer optimizer for meta-learning.

    Args:
        params: Iterable of parameters to optimize (e.g., L2O parameters)
        name (str): Optimizer name ('adam', 'adamw', 'sgd', etc.)
        lr (float): Base learning rate
        kwargs: Additional optimizer-specific arguments

    Returns:
        torch.optim.Optimizer
    """
    if len(params) == 0:
        return None  # allow L2O setups with no trainable outer params

    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=kwargs.get("momentum", 0.9))
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=kwargs.get("momentum", 0.9))
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(optimizer, name=None, **kwargs):
    """
    Create a learning rate scheduler for the given optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to attach the scheduler to.
        name (str or None): Scheduler name ('cosine', 'step', 'exp', 'onecycle', 'linear_warmup').
        kwargs: Scheduler-specific arguments.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None
    """
    if name is None:
        return None

    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 50),
            eta_min=kwargs.get("eta_min", 1e-6)
        )
    elif name == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1)
        )
    elif name == "exp":
        return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))
    elif name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            total_steps=kwargs.get("total_steps", 100)
        )
    elif name == "linear_warmup":
        warmup_steps = kwargs.get("warmup_steps", 100)
        total_steps = kwargs.get("total_steps", 1000)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.0, 1 - (step - warmup_steps) / (total_steps - warmup_steps))
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

def build_outer_optimizer_and_scheduler(params, opt_cfg: dict):
    """
    Build both optimizer and (optionally) a scheduler.

    Args:
        params: Parameters to optimize (e.g., L2O parameters)
        opt_cfg (dict): Config dictionary, example:
            {
                "name": "adam",
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "scheduler": {
                    "name": "cosine",
                    "T_max": 50,
                    "eta_min": 1e-6
                }
            }

    Returns:
        Tuple (optimizer, scheduler_config_for_lightning)
    """
    
    opt_cfg = copy.deepcopy(opt_cfg)  # never mutate Hydra config
    scheduler_cfg = opt_cfg.pop("scheduler", None)
    optimizer = build_outer_optimizer(params, **opt_cfg)

    if scheduler_cfg:
        scheduler = build_scheduler(optimizer, **scheduler_cfg)
        # Lightning expects dict for scheduler config
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.get("interval", "epoch"),  # 'epoch' or 'step'
                "frequency": scheduler_cfg.get("frequency", 1),
                "monitor": scheduler_cfg.get("monitor", "val/meta_loss"),
            },
        }
    else:
        return optimizer
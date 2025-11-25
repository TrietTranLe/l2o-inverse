from typing import Dict, Any, List
import torch.nn as nn


def build_outer_param_groups(l2o: nn.Module, lr_groups: Dict[str, float] | None, default_lr: float) -> List[dict]:
    """
    Build parameter groups for the outer optimizer based on L2O submodules.

    Submodules considered:
        - l2o.grad_mod
        - l2o.update_rule
        - l2o.reg_net

    Each group may have its own learning rate via lr_groups.
    """
    param_groups = []

    for name in ["grad_mod", "update_rule", "reg_net"]:
        if not hasattr(l2o, name):
            continue

        submodule = getattr(l2o, name)
        params = [p for p in submodule.parameters() if p.requires_grad]

        if not params:
            # Nothing trainable in this submodule -> skip
            continue

        lr = lr_groups.get(name, default_lr) if lr_groups else default_lr
        param_groups.append({"params": params, "lr": lr, "module_name": name})

    return param_groups

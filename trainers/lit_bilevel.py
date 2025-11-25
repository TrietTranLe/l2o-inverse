"""
Bi-Level Lightning Trainer
------------------------------------------
Trainer for meta-learning setups using a learned optimizer (L2O)
with multiple submodules:
    - grad_mod (Gradient Modifier): learnable network
    - update_rule (Update Rule): can be learnable or fixed rule
    - reg_net (Regularizer): learnable regularization

Each submodule can have its own learning rate.
Outer loss is fully flexible: can be a str, callable, dict, list, or OmegaConf config.

Compatible with PyTorch Lightning and Hydra.
"""

import torch
import pytorch_lightning as pl
from optimizers.outer_optimizer import build_outer_optimizer_and_scheduler
from optimizers.builders.outer_optimizer_builder import build_outer_param_groups


class LitBiLevel(pl.LightningModule):
    """
    PyTorch Lightning module for bi-level meta-learning.
    Supports multiple L2O submodules with independent learning rates.
    """

    def __init__(
        self,
        l2o: torch.nn.Module,
        inner_steps: int,
        outer_opt_cfg: dict,
        outer_loss: torch.nn.Module,
    ):
        """
        Args:
            l2o: Learned optimizer (may contain grad_mod, update_rule, reg_net)
            inner_steps (int): Number of inner-loop updates per outer step
            outer_opt_cfg (dict): Config for outer optimizer, e.g.:
                {
                    "name": "adamw",
                    "lr": 1e-4,
                    "weight_decay": 1e-5,
                    "lr_groups": {
                        "grad_mod": 1e-4,
                        "update_rule": 0.0,
                        "reg": 5e-5
                    }
                }
            outer_loss: outer loss function to evaluate after inner optimization
        """
        super().__init__()
        self.l2o = l2o
        self.inner_steps = inner_steps
        self.outer_opt_cfg = outer_opt_cfg
        self.outer_loss = outer_loss

        # Save hyperparameters (Lightning stores non-module arguments)
        self.save_hyperparameters(ignore=["l2o"])

    # -------------------------------------------------------------------------
    def forward(self, x):
        """Forward pass through the autoencoder."""
        return self.l2o.reg_net(x)

    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        Executes one outer-loop training step:
          1. Performs inner optimization via L2O
          2. Computes outer/meta loss
          3. Backpropagates through L2O (trainable parts only)
        """
        y, x = batch

        # ---- Inner optimization ----
        x_inner, inner_losses, _ = self.l2o(x, y, steps=self.inner_steps, return_all=True)

        # ---- Outer/meta loss ----
        train_loss = self.outer_loss(x_inner=x_inner, x_true=x, AE=self.l2o.reg_net)

        # ---- Logging ----
        self.log("train/outer_loss", train_loss, prog_bar=True, on_epoch=True)
        self.log("train/inner_loss_last", inner_losses[-1])
        self.log("train/inner_loss_mean", torch.stack(inner_losses).mean())

        # Optional: gradient monitoring
        if hasattr(self.l2o, "last_grad_norm"):
            self.log("train/grad_norm", self.l2o.last_grad_norm)

        return train_loss

    # -------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        """Evaluates outer-loss without updating."""
        y, x = batch

        with torch.no_grad():
            x_inner = self.l2o(x, y, steps=self.inner_steps)
            val_loss = self.outer_loss(x_inner=x_inner, x_true=x, AE=self.l2o.reg_net)

        self.log("val/outer_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        """
        Builds the outer optimizer (and optional scheduler) to update
        the trainable submodules of the learned optimizer.
        Each submodule can have its own LR via outer_opt_cfg['lr_groups'].
        """
        lr_groups = self.outer_opt_cfg.get("lr_groups", None)
        default_lr = self.outer_opt_cfg.get("lr", 1e-4)
        param_groups = build_outer_param_groups(l2o=self.l2o, lr_groups=lr_groups, default_lr=default_lr)

        if len(param_groups) == 0:
            print("[LitBiLevel] No trainable parameters for outer optimizer.")
            return None

        # --- Build optimizer ---
        opt_cfg = self.outer_opt_cfg.copy()
        opt_cfg.pop("lr_groups", None)
        opt_setup = build_outer_optimizer_and_scheduler(param_groups, opt_cfg)

        # --- Logging setup summary ---
        opt_name = opt_cfg.get("name", "unknown")
        sch_name = opt_cfg.get("scheduler", {}).get("name", None)
        if sch_name:
            print(f"[LitBiLevel] Using outer optimizer '{opt_name}' with scheduler '{sch_name}'.")
        else:
            print(f"[LitBiLevel] Using outer optimizer '{opt_name}' (no scheduler).")

        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"  ├─ {g['module_name']}: lr={g['lr']:.2e}, num_params={n_params}")

        return opt_setup

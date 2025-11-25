"""
Basic tests for LitBiLevel + L2OOptimizer end-to-end.
Run with:
    python -m tests.test_lit_bilevel
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# ---- Import your modules ----
from trainers.lit_bilevel import LitBiLevel
from optimizers.l2o_optimizer import EsiGradSolver
from optimizers.builders.outer_optimizer_builder import build_outer_param_groups
from losses.expression_composer import ExpressionLossComposer

# Mock builtins losses
from losses.builtins.mse import MSE


# -------------------------------------------------------------
# Helper: dummy forward model (AE-style)
# -------------------------------------------------------------
class DummyAE(torch.nn.Module):
    """
    Minimal AE-like model:
        x -> Linear -> x_hat
    """
    def __init__(self, dim=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

    # LitBiLevel can call this to gather AE (optional)
    def get_context(self):
        # Normally you might return {"AE": self}
        return {"AE": self}


# -------------------------------------------------------------
# Helper: dummy grad_mod
# -------------------------------------------------------------
class DummyGradMod(torch.nn.Module):
    """
    Simple 1-layer MLP that transforms gradient.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)

    def reset_state(self, x):
        pass

    def forward(self, x, g, loss):
        return self.fc(g)


# -------------------------------------------------------------
# Helper: dummy update rule
# -------------------------------------------------------------
class DummyUpdateRule(torch.nn.Module):
    """
    x_{t+1} = x_t - g_t
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, g, step):
        return x - g


# -------------------------------------------------------------
# Build components for the test
# -------------------------------------------------------------
def build_test_l2o(reg_net):
    inner_loss = ExpressionLossComposer(
        terms=[
            {
                "name": "inner_mse",
                "fn": MSE(),
                "expr": "fn(x, y)",
                "weight": 1.0,
            }
        ]
    )
    return EsiGradSolver(
        reg_net=reg_net,
        grad_mod=DummyGradMod(dim=16),
        update_rule=DummyUpdateRule(),
        inner_loss=inner_loss,
        n_step=3,
        fwd=None
    )


# -------------------------------------------------------------
# Test 1 — Forward path (inference-only)
# -------------------------------------------------------------
def test_forward_path_cpu():
    print("\n=== TEST 1 — forward(x) on CPU ===")

    model = DummyAE(dim=16)
    l2o = build_test_l2o(model)

    # Simple outer loss: AE(x) → x_true
    outer_loss = ExpressionLossComposer(
        terms=[
            {
                "name": "outer_mse",
                "fn": MSE(),
                "expr": "fn(x_true, AE(x_inner))",
                "weight": 1.0,
            }
        ],
        AE=model,
    )

    outer_opt_cfg = {
        "name": "adam",
        "lr": 1e-3,
        "lr_groups": {
            "grad_mod": 1e-3,
            "update_rule": 1e-3
        }
    }

    lit = LitBiLevel(
        l2o_optimizer=l2o,
        outer_opt_cfg=outer_opt_cfg,
        inner_steps=3,
        outer_loss_fn=outer_loss,
    )

    x = torch.randn(4, 16)
    y = lit(x)
    print("Forward output shape:", y.shape)


# -------------------------------------------------------------
# Test 2 — One training step end-to-end
# -------------------------------------------------------------
def test_training_step_cpu():
    print("\n=== TEST 2 — training_step on CPU ===")

    model = DummyAE(dim=16)
    l2o = build_test_l2o(model)

    outer_loss = ExpressionLossComposer(
        terms=[
            {
                "name": "outer_mse",
                "fn": MSE(),
                "expr": "fn(x_true, AE(x_inner))",
                "weight": 1.0,
            }
        ],
        AE=model,
    )

    outer_opt_cfg = {
        "name": "adam",
        "lr": 1e-3,
        "lr_groups": {"grad_mod": 1e-3, "update_rule": 1e-3},
    }

    lit = LitBiLevel(
        l2o_optimizer=l2o,
        outer_opt_cfg=outer_opt_cfg,
        inner_steps=3,
        outer_loss_fn=outer_loss,
    )

    # Fake data
    X = torch.randn(8, 16)
    Y = torch.randn(8, 16)
    loader = DataLoader(TensorDataset(X, Y), batch_size=4)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(lit, loader)


# -------------------------------------------------------------
# Test 3 — CUDA (optional)
# -------------------------------------------------------------
def test_training_step_cuda():
    if not torch.cuda.is_available():
        print("\n[CUDA unavailable — skipping Test 3]")
        return

    print("\n=== TEST 3 — training_step on CUDA ===")

    model = DummyAE(dim=16).cuda()
    l2o = build_test_l2o(model).cuda()

    outer_loss = ExpressionLossComposer(
        terms=[
            {
                "name": "outer_mse",
                "fn": MSE(),
                "expr": "fn(x_true, AE(x_inner))",
                "weight": 1.0,
            }
        ],
        AE=model,
    ).cuda()

    outer_opt_cfg = {
        "name": "adam",
        "lr": 1e-3,
        "lr_groups": {"grad_mod": 1e-3, "update_rule": 1e-3},
    }

    lit = LitBiLevel(
        l2o_optimizer=l2o,
        outer_opt_cfg=outer_opt_cfg,
        inner_steps=3,
        outer_loss_fn=outer_loss,
    ).cuda()

    X = torch.randn(8, 16).cuda()
    Y = torch.randn(8, 16).cuda()
    loader = DataLoader(TensorDataset(X, Y), batch_size=4)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(lit, loader)


# -------------------------------------------------------------
# Run all tests
# -------------------------------------------------------------
if __name__ == "__main__":
    test_forward_path_cpu()
    test_training_step_cpu()
    test_training_step_cuda()

    print("\nALL LIT_BILEVEL TESTS COMPLETED.\n")

"""
Basic tests for ExpressionLossComposer.
Run with:
    python -m tests.test_losses
"""

import torch
from torch import nn
from losses.expression_composer import ExpressionLossComposer
from losses.builtins.mse import MSE
from losses.builtins.l1 import L1NormLoss
from losses.builtins.cosine import CosineSimilarityFlatLoss

def test_simple_mse():
    loss = ExpressionLossComposer(
        terms=[
            {
                "name": "mse_term",
                "fn": MSE(),
                "expr": "fn(x, y)",
                "weight": 1.0,
            }
        ]
    )

    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    v = loss(x=x, y=y)
    print("Simple MSE value:", float(v))

def test_composite_mse_l1():
    loss = ExpressionLossComposer(
        terms=[
            {
                "name": "mse_term",
                "fn": MSE(),
                "expr": "fn(x, y)",
                "weight": 1.0
            },
            {
                "name": "l1_term",
                "fn": L1NormLoss(),
                "expr": "fn(x)",
                "weight": 0.5
            }
        ]
    )

    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    total, terms = loss(x=x, y=y, return_terms=True)
    print("MSE + L1:", float(total))
    for k, v in terms.items():
        print(f"  {k}: {float(v)}")


def test_external_vars():
    AE = torch.nn.Linear(10, 10)   # mock autoencoder
    L = torch.randn(10, 10)        # mock forward matrix

    loss = ExpressionLossComposer(
        terms=[
            {
                "name": "data",
                "fn": MSE(),
                "expr": "fn(y, x @ L)",
                "weight": 1.0,
            },
            {
                "name": "reg",
                "fn": CosineSimilarityFlatLoss(),
                "expr": "fn(x, AE(x))",
                "weight": 0.2,
            },
        ],
        AE=AE,
        L=L,
    )

    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    total, terms = loss(x=x, y=y, return_terms=True)
    print("MSE + Cosine:", float(total))
    for k, v in terms.items():
        print(f"  {k}: {float(v)}")

def test_trainable_weights():
    loss = ExpressionLossComposer(
        terms=[
            {
                "name": "mse_term",
                "fn": MSE(),
                "expr": "fn(x, y)",
                # no weight -> creates nn.Parameter(1.0)
            }
        ]
    )

    print("Trainable weight:", list(loss.parameters()))

    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    v = loss(x=x, y=y)
    print("Value:", float(v))

def test_intermediate_extraction():
    AE = nn.Linear(10, 10)
    L = torch.randn(10, 10)

    composer = ExpressionLossComposer(
        terms=[
            {
                "name": "data",
                "fn": MSE(),
                "expr": "fn(y, x @ L)",   # should detect intermediate "L@x"
                "weight": 1.0
            },
            {
                "name": "reg",
                "fn": L1NormLoss(),
                "expr": "fn(AE(x))",      # should detect intermediate "AE(x)"
                "weight": 0.5
            }
        ],
        AE=AE,
        L=L
    )

    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    total, intermediates = composer(x=x, y=y, return_intermediate=True)

    print("\nReturned Intermediates:")
    for k, v in intermediates.items():
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv.shape if isinstance(vv, torch.Tensor) else vv}")

    # --- Assertions ---
    assert "input" in intermediates
    assert "x" in intermediates["input"]
    assert "y" in intermediates["input"]
    assert "L" in intermediates["input"]    # tensor

    assert "data" in intermediates
    assert "x@L" in intermediates["data"]
    assert isinstance(intermediates["data"]["x@L"], torch.Tensor)

    assert "reg" in intermediates
    assert "AE(x)" in intermediates["reg"]
    assert isinstance(intermediates["reg"]["AE(x)"], torch.Tensor)

    print("\n[PASS] Intermediate extraction successful")

def test_to_device():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    device = torch.device("cuda")

    AE = nn.Linear(10, 10)
    L = torch.randn(10, 10)

    composer = ExpressionLossComposer(
        terms=[
            {
                "name": "data",
                "fn": MSE(),
                "expr": "fn(y, L @ x)",
                "weight": None,  # trainable
            }
        ],
        AE=AE,
        L=L,
    )

    print("Before .to():")
    print(f"  AE device: {composer.external_vars['AE'].weight.device}")
    print(f"  L device:  {composer.external_vars['L'].device}")
    print(f"  weight param device: {next(composer.parameters()).device}")

    # move composer to GPU
    composer = composer.to(device)

    print("\nAfter .to():")
    print(f"  AE device: {composer.external_vars['AE'].weight.device}")
    print(f"  L device:  {composer.external_vars['L'].device}")
    print(f"  weight param device: {next(composer.parameters()).device}")

    # Assertions
    assert composer.external_vars["AE"].weight.device.type == "cuda"
    assert composer.external_vars["L"].device.type == "cuda"
    assert next(composer.parameters()).device.type == "cuda"

    print("\n[PASS] ExpressionLossComposer runs on CUDA")

if __name__ == "__main__":
    test_simple_mse()
    test_composite_mse_l1()
    test_external_vars()
    test_trainable_weights()
    test_intermediate_extraction()
    test_to_device()
    print("\nALL TESTS COMPLETED.\n")

import torch
from models.grad_mod import x_grad_mod_mul as GradMod


def make_fake_data(B=4, S=16, T=128, device="cpu"):
    """
    Create fake gradient tensor and input tensor.
    """
    x = torch.randn(B, S, T, device=device, requires_grad=True)
    g = torch.randn_like(x)
    return x, g

def test_grad_mod_forward_shape():
    mod = GradMod(dim_in=16, dim_hidden=32)
    x, g = make_fake_data(S=16)
    mod.reset_state(x)
    out = mod(x, g)

    assert out.shape == g.shape, "Output gradient shape mismatch"
    print("[PASS] Output shape == gradient shape")

def test_grad_mod_gradient_flow():
    mod = GradMod(dim_in=16, dim_hidden=32)
    x, g = make_fake_data(S=16)
    mod.reset_state(x)
    out = mod(x, g)
    loss = out.sum()

    loss.backward()

    # Ensure that backward passes through grad_mod
    for name, p in mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter '{name}' did not receive gradients"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    print("[PASS] Parameters receive gradients correctly")

def test_grad_mod_cuda():
    if not torch.cuda.is_available():
        print("\n[TEST skipped] CUDA not available")
        return
    mod = GradMod(dim_in=16, dim_hidden=32).cuda()
    x, g = make_fake_data(S=16, device="cuda")
    mod.reset_state(x)
    out = mod(x, g)
    assert out.device.type == "cuda", "Output must remain on CUDA"

    print("[PASS] grad_mod runs on CUDA")

def test_grad_mod_stability_multiple_steps():
    mod = GradMod(dim_in=16, dim_hidden=32)

    x, g = make_fake_data(S=16)
    mod.reset_state(x)
    for _ in range(5):
        g = mod(x, g)  # repeatedly apply grad_mod

    assert torch.isfinite(g).all(), "Non-finite values after repeated grad_mod"
    print("[PASS] Repeated grad_mod application is numerically stable")

def _run_clamp_cases(mod, x, g):
    """
    Runs 4 clamp scenarios on GradMod:
        1. No clamp
        2. P_min only
        3. P_max only
        4. Both P_min & P_max
    Shared between CPU and CUDA tests.
    """

    # Case 1: No clamp
    out = mod(x, g, P_min=None, P_max=None)
    assert out.shape == g.shape
    assert torch.isfinite(out).all(), "Non-finite output in no-clamp case"

    # Case 2: P_min only
    out = mod(x, g, P_min=0.1, P_max=None)
    assert out.shape == g.shape
    assert torch.isfinite(out).all(), "Non-finite output in P_min-only case"

    # Case 3: P_max only
    out = mod(x, g, P_min=None, P_max=0.2)
    assert out.shape == g.shape
    assert torch.isfinite(out).all(), "Non-finite output in P_max-only case"

    # Case 4: both P_min and P_max
    out = mod(x, g, P_min=0.1, P_max=0.2)
    assert out.shape == g.shape
    assert torch.isfinite(out).all(), "Non-finite output in both-P case"

def test_grad_mod_clamp_cpu():
    mod = GradMod(dim_in=16, dim_hidden=32)
    x, g = make_fake_data(S=16)
    mod.reset_state(x)
    _run_clamp_cases(mod, x, g)
    
    print("[PASS] All clamp cases passed on CPU")

def test_grad_mod_clamp_cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    mod = GradMod(dim_in=16, dim_hidden=32).cuda()
    x, g = make_fake_data(S=16, device="cuda")
    mod.reset_state(x)
    _run_clamp_cases(mod, x, g)

    print("[PASS] All clamp cases passed on CUDA")


if __name__ == "__main__":
    print("\n========== Running grad_mod tests ==========\n")

    test_grad_mod_forward_shape()
    test_grad_mod_gradient_flow()
    test_grad_mod_cuda()
    test_grad_mod_stability_multiple_steps()
    test_grad_mod_clamp_cpu()
    test_grad_mod_clamp_cuda()

    print("\n========== All grad_mod tests completed ==========\n")

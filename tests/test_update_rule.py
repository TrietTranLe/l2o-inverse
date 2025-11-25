import torch
from models.update_rule import LRUR, GradientDescentUR


# Make fake tensors
def make_fake_tensors(device="cpu"):
    x = torch.randn(4, 10, device=device, requires_grad=True)
    grad = torch.randn_like(x)
    return x, grad

def test_lr_scalar():
    rule = LRUR(lr=0.1)
    x, grad = make_fake_tensors()

    x_new = rule(x, grad, step=0)

    assert x_new.shape == x.shape
    torch.testing.assert_close(x_new, x - 0.1 * grad)
    print("[PASS] LRUpdateRule: x_new = x - 0.1 * grad")

def test_lr_schedule():
    schedule = [0.3, 0.2, 0.1]
    rule = LRUR(lr=schedule)

    x, grad = make_fake_tensors()

    # step 0
    x0 = rule(x, grad, step=0)
    torch.testing.assert_close(x0, x - 0.3 * grad)
    print("[PASS] LRUpdateRule: step 0 uses lr=0.3")

    # step 1
    x1 = rule(x, grad, step=1)
    torch.testing.assert_close(x1, x - 0.2 * grad)
    print("[PASS] LRUpdateRule: step 1 uses lr=0.2")

    # step 2
    x2 = rule(x, grad, step=2)
    torch.testing.assert_close(x2, x - 0.1 * grad)
    print("[PASS] LRUpdateRule: step 2 uses lr=0.1")

    # step 3 -> uses last LR
    x3 = rule(x, grad, step=3)
    torch.testing.assert_close(x3, x - 0.1 * grad)
    print("[PASS] LRUpdateRule: step 3+ uses last lr=0.1")

def test_lr_learnable():
    rule = LRUR()

    assert isinstance(rule.lr, torch.nn.Parameter)
    assert rule.lr.requires_grad is True

    x, grad = make_fake_tensors()
    x_new = rule(x, grad, step=0)

    # check if gradient flows
    loss = x_new.sum()
    loss.backward()

    # LR must receive gradient
    assert rule.lr.grad is not None
    assert torch.isfinite(rule.lr.grad).all(), "Non-finite grad on lr"
    print("[PASS] LRUpdateRule: learnable lr receives gradient")

def test_lr_device_move():
    if not torch.cuda.is_available():
        print("\n[TEST skipped] CUDA not available")
        return

    rule = LRUR(lr=[0.1, 0.05])
    rule = rule.cuda()

    x, grad = make_fake_tensors(device="cuda")

    x_new = rule(x, grad, 0)
    assert x_new.device.type == "cuda", "Output not on CUDA"
    print("[PASS] LRUpdateRule works on CUDA")

def test_gradient_descent_update_rule():
    rule = GradientDescentUR()

    x, grad = make_fake_tensors()
    out = rule(x, grad, step=0)

    assert out.shape == x.shape
    torch.testing.assert_close(out, x - grad)
    print("[PASS] GradientDescentUR: x_new = x - grad")

def test_gradient_descent_grad_flow():
    rule = GradientDescentUR()

    x, grad = make_fake_tensors()
    x.requires_grad_()

    out = rule(x, grad, step=0)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Non-finite grad on x"
    print("[PASS] Gradient flows through GD update rule")

def test_lr_multiple_steps_sanity():
    rule = LRUR(lr=[0.1, 0.05, 0.01])

    x, grad = make_fake_tensors()

    x0 = rule(x, grad, step=0)
    x1 = rule(x0, grad, step=1)
    x2 = rule(x1, grad, step=2)

    assert x2.shape == x.shape
    assert torch.isfinite(x2).all(), "Non-finite values after multiple steps"
    print("[PASS] Multiple-step update runs without numerical issues")

if __name__ == "__main__":
    print("\n========== Running UpdateRule tests ==========\n")

    test_lr_scalar()
    test_lr_schedule()
    test_lr_learnable()
    test_lr_device_move()
    test_gradient_descent_update_rule()
    test_gradient_descent_grad_flow()
    test_lr_multiple_steps_sanity()

    print("\n========== All UpdateRule tests completed ==========\n")

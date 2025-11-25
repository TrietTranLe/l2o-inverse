import torch
import torch.nn as nn
from models.autoencoder import Conv1DAE


def test_autoencoder_forward():
    ae = Conv1DAE(dim_in=8, dim_hidden=4, dim_out=8, kernel_size=3)

    x = torch.randn(2, 8, 100)  # (batch, channels, time)
    y = ae(x)

    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    print("[PASS] Forward shape OK:", y.shape)

def test_autoencoder_backward():
    ae = Conv1DAE(dim_in=8, dim_hidden=4, dim_out=8)

    x = torch.randn(2, 8, 100)
    y = ae(x)
    loss = y.mean()
    loss.backward()

    # ensure some gradients exist
    total_grad = sum(p.grad is not None for p in ae.parameters())
    assert total_grad > 0, "No gradients propagated"
    print("[PASS] Backprop works (grads exist)")

def test_autoencoder_cuda():
    if not torch.cuda.is_available():
        print("\n[TEST skipped] CUDA not available")
        return

    ae = Conv1DAE(dim_in=8, dim_hidden=4, dim_out=8)
    ae = ae.to("cuda")

    x = torch.randn(2, 8, 100).cuda()
    y = ae(x)

    assert y.is_cuda, "Output not on CUDA"
    print("[PASS] AutoEncoder runs on CUDA")

def test_autoencoder_fixed():
    ae = Conv1DAE(dim_in=8, dim_hidden=4, dim_out=8, fixed=True)
    trainable = [p for p in ae.parameters() if p.requires_grad]
    assert len(trainable) == 0, "fixed=True but some params are still trainable"
    print("[PASS] fixed=True disables all gradients")

def test_autoencoder_pretrained():
    # TEMP save
    ae1 = Conv1DAE(dim_in=8, dim_hidden=4, dim_out=8)
    temp_path = "temp_ae.pth"
    torch.save(ae1.state_dict(), temp_path)

    # load
    ae2 = Conv1DAE(dim_in=8, dim_hidden=4, dim_out=8,
                        pretrained_model_path=temp_path)

    # compare parameters
    for p1, p2 in zip(ae1.parameters(), ae2.parameters()):
        assert torch.allclose(p1, p2), "Loaded model params differ from saved"

    print("[PASS] Pretrained weights loaded correctly")

if __name__ == "__main__":
    print("\n========== Running AutoEncoder tests ==========")
    test_autoencoder_forward()
    test_autoencoder_backward()
    test_autoencoder_cuda()
    test_autoencoder_fixed()
    test_autoencoder_pretrained()
    print("\n========== All AutoEncoder tests passed ==========\n")

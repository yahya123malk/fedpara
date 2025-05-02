import torch
import torch.nn.functional as F

def fedpara_trainable_compress(delta_w, rank=8, C=None, steps=1000, lr=1e-2):
    if len(delta_w.shape) != 2:
        raise ValueError("Only supports 2D tensors")

    m, n = delta_w.shape
    device = delta_w.device

    if C is None:
        C = torch.ones(m, n, device=device)

    A = torch.randn(m, rank, device=device).mul_(0.01).requires_grad_()
    B = torch.randn(n, rank, device=device).mul_(0.01).requires_grad_()



    optimizer = torch.optim.Adam([A, B], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        approx = (A @ B.T) * C
        loss = F.mse_loss(approx, delta_w)
        loss.backward()
        optimizer.step()

        # 🔍 Debug: print loss every 100 steps
        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    W_approx = (A @ B.T) * C
    return W_approx, A.detach(), B.detach()

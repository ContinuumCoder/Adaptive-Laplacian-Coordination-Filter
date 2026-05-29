from __future__ import annotations

from collections.abc import Sequence

import torch


def as_batch(z: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if z.dim() == 1:
        return z.unsqueeze(0), True
    if z.dim() == 2:
        return z, False
    raise ValueError(f"expected a tensor with shape (n,) or (B, n), got {tuple(z.shape)}")


def laplacian_from_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.dim() != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"expected a square matrix, got {tuple(weights.shape)}")
    degree = weights.sum(dim=-1)
    return torch.diag(degree) - weights


def make_weight_matrix(
    n: int,
    init: str,
    device: torch.device,
    dtype: torch.dtype,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is not None:
        w = weights.to(device=device, dtype=dtype).clone()
        if w.shape != (n, n):
            raise ValueError(f"initial weights must have shape {(n, n)}, got {tuple(w.shape)}")
        w = 0.5 * (w + w.T)
        w.fill_diagonal_(0)
        return w
    if init == "zero":
        return torch.zeros(n, n, device=device, dtype=dtype)
    if init == "uniform":
        w = torch.ones(n, n, device=device, dtype=dtype) - torch.eye(n, device=device, dtype=dtype)
        return w / max(n - 1, 1)
    if init == "ring":
        w = torch.zeros(n, n, device=device, dtype=dtype)
        if n > 1:
            idx = torch.arange(n, device=device)
            w[idx, (idx - 1) % n] = 1
            w[idx, (idx + 1) % n] = 1
        return 0.5 * (w + w.T)
    if init in {"grid", "lattice"}:
        side = int(round(n**0.5))
        if side * side != n:
            raise ValueError("grid initialization requires n to be a perfect square")
        w = torch.zeros(n, n, device=device, dtype=dtype)
        for row in range(side):
            for col in range(side):
                node = row * side + col
                for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nrow = row + drow
                    ncol = col + dcol
                    if 0 <= nrow < side and 0 <= ncol < side:
                        w[node, nrow * side + ncol] = 1
        return 0.5 * (w + w.T)
    raise ValueError(f"unknown graph initialization {init!r}")


def polynomial_values(eigenvalues: torch.Tensor, polynomial: str | Sequence[float]) -> torch.Tensor:
    if isinstance(polynomial, str):
        if polynomial == "laplacian":
            return eigenvalues
        if polynomial == "square":
            return eigenvalues.pow(2)
        if polynomial == "normalized":
            return eigenvalues / eigenvalues.max().clamp(min=torch.finfo(eigenvalues.dtype).eps)
        raise ValueError(f"unknown polynomial {polynomial!r}")
    values = torch.zeros_like(eigenvalues)
    power = torch.ones_like(eigenvalues)
    for coeff in polynomial:
        values = values + float(coeff) * power
        power = power * eigenvalues
    return values.clamp(min=0)


def pairwise_squared_mean(z: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(-1) - z.unsqueeze(-2)
    values = diff.pow(2).mean(dim=0)
    values = 0.5 * (values + values.T)
    values.fill_diagonal_(0)
    return values

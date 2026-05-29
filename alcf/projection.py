from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import as_batch


@dataclass(frozen=True)
class ProjectionResult:
    z: torch.Tensor
    mean: torch.Tensor
    residual: torch.Tensor
    multiplier: torch.Tensor
    energy_before: torch.Tensor
    energy_after: torch.Tensor


def project_load(
    z: torch.Tensor,
    q_eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    beta: float,
    gamma: float,
    n_bisect: int = 60,
    expand_iters: int = 40,
) -> ProjectionResult:
    z_batch, squeezed = as_batch(z)
    batch_size, n = z_batch.shape
    q = q_eigenvalues.to(device=z_batch.device, dtype=z_batch.dtype).clamp(min=0)
    u = eigenvectors.to(device=z_batch.device, dtype=z_batch.dtype)
    mean = z_batch.mean(dim=-1, keepdim=True)
    residual = z_batch - mean
    clipped_mean = mean.clamp(min=-float(beta) / n, max=float(beta) / n)
    coeffs = residual @ u
    energy_before = (coeffs.pow(2) * q).sum(dim=-1)
    active = energy_before > float(gamma)
    multiplier = torch.zeros(batch_size, device=z_batch.device, dtype=z_batch.dtype)
    if active.any():
        coeffs_active = coeffs[active]
        low = torch.zeros(coeffs_active.shape[0], device=z_batch.device, dtype=z_batch.dtype)
        high = torch.ones_like(low)
        for _ in range(expand_iters):
            value = _energy_at_multiplier(coeffs_active, q, high)
            mask = value > float(gamma)
            if not mask.any():
                break
            high = torch.where(mask, high * 2, high)
        for _ in range(n_bisect):
            middle = 0.5 * (low + high)
            value = _energy_at_multiplier(coeffs_active, q, middle)
            mask = value > float(gamma)
            low = torch.where(mask, middle, low)
            high = torch.where(mask, high, middle)
        multiplier = multiplier.clone()
        multiplier[active] = 0.5 * (low + high)
    filtered_coeffs = coeffs / (1 + multiplier.unsqueeze(-1) * q)
    projected_residual = filtered_coeffs @ u.T
    projected = clipped_mean + projected_residual
    energy_after = (filtered_coeffs.pow(2) * q).sum(dim=-1)
    if squeezed:
        return ProjectionResult(
            z=projected.squeeze(0),
            mean=clipped_mean.squeeze(0),
            residual=projected_residual.squeeze(0),
            multiplier=multiplier.squeeze(0),
            energy_before=energy_before.squeeze(0),
            energy_after=energy_after.squeeze(0),
        )
    return ProjectionResult(projected, clipped_mean, projected_residual, multiplier, energy_before, energy_after)


def _energy_at_multiplier(coeffs: torch.Tensor, q: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
    denom = (1 + multiplier.unsqueeze(-1) * q).pow(2)
    return (coeffs.pow(2) * q / denom).sum(dim=-1)

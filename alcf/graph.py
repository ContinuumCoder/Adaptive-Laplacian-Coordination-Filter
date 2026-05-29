from __future__ import annotations

import torch

from .utils import as_batch, laplacian_from_weights, make_weight_matrix, pairwise_squared_mean, polynomial_values


class GraphLearner:
    def __init__(
        self,
        n: int,
        alpha: float = 0.02,
        eps_prune: float = 1e-3,
        eig_refresh_interval: int = 200,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        init: str = "uniform",
        weights: torch.Tensor | None = None,
    ) -> None:
        self.n = int(n)
        self.alpha = float(alpha)
        self.eps_prune = float(eps_prune)
        self.eig_refresh_interval = int(eig_refresh_interval)
        self.device = torch.device(device)
        self.dtype = dtype
        self.weights = make_weight_matrix(self.n, init, self.device, self.dtype, weights)
        self.step_count = 0
        self._eigendecomp: tuple[torch.Tensor, torch.Tensor] | None = None

    def to(self, device: str | torch.device) -> GraphLearner:
        self.device = torch.device(device)
        self.weights = self.weights.to(self.device)
        if self._eigendecomp is not None:
            eigvals, eigvecs = self._eigendecomp
            self._eigendecomp = eigvals.to(self.device), eigvecs.to(self.device)
        return self

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        z_batch, _ = as_batch(z.to(device=self.device, dtype=self.dtype))
        if z_batch.shape[-1] != self.n:
            raise ValueError(f"expected {self.n} graph nodes, got {z_batch.shape[-1]}")
        if self.alpha > 0:
            target = pairwise_squared_mean(z_batch)
            self.weights.mul_(1 - self.alpha).add_(target, alpha=self.alpha)
            self.weights.fill_diagonal_(0)
            if self.eps_prune > 0:
                self.weights = torch.where(
                    self.weights >= self.eps_prune,
                    self.weights,
                    torch.zeros_like(self.weights),
                )
            self.weights = 0.5 * (self.weights + self.weights.T)
        self.step_count += 1
        if self.eig_refresh_interval <= 1 or self.step_count % self.eig_refresh_interval == 0:
            self._eigendecomp = None

    def laplacian(self) -> torch.Tensor:
        return laplacian_from_weights(self.weights)

    @torch.no_grad()
    def eigendecomp(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._eigendecomp is None:
            eigvals, eigvecs = torch.linalg.eigh(self.laplacian())
            self._eigendecomp = eigvals.contiguous(), eigvecs.contiguous()
        return self._eigendecomp

    def filter_eigenvalues(self, polynomial: str | list[float] | tuple[float, ...] = "laplacian") -> torch.Tensor:
        eigvals, _ = self.eigendecomp()
        values = polynomial_values(eigvals, polynomial)
        values = values.clone()
        values[eigvals.abs() < 1e-7] = 0
        return values

    @torch.no_grad()
    def algebraic_connectivity(self) -> float:
        eigvals, _ = self.eigendecomp()
        if self.n <= 1:
            return 0.0
        return float(eigvals[1].item())

    @torch.no_grad()
    def num_components(self, tol: float = 1e-6) -> int:
        eigvals, _ = self.eigendecomp()
        return int((eigvals.abs() <= tol).sum().item())

    def state_dict(self) -> dict:
        return {"weights": self.weights.detach().cpu(), "step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.weights = state["weights"].to(device=self.device, dtype=self.dtype)
        self.step_count = int(state.get("step_count", 0))
        self._eigendecomp = None

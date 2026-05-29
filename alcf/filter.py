from __future__ import annotations

import torch

from .graph import GraphLearner
from .projection import ProjectionResult, project_load
from .threshold import ThresholdAdapter
from .utils import as_batch


class ALCF:
    def __init__(
        self,
        n: int,
        beta: float = 4.0,
        gamma: float = 0.05,
        alpha: float = 0.02,
        eps_prune: float = 1e-3,
        delta_beta: float = 0.1,
        delta_gamma: float = 0.1,
        threshold_step_size: float = 1e-2,
        graph_init: str = "uniform",
        polynomial: str | list[float] | tuple[float, ...] = "laplacian",
        eig_refresh_interval: int = 200,
        n_bisect: int = 60,
        adapt_thresholds: bool = True,
        update_graph: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        weights: torch.Tensor | None = None,
    ) -> None:
        self.n = int(n)
        self.device = torch.device(device)
        self.dtype = dtype
        self.polynomial = polynomial
        self.n_bisect = int(n_bisect)
        self.adapt_thresholds = bool(adapt_thresholds)
        self.update_graph = bool(update_graph)
        self.graph = GraphLearner(
            n=n,
            alpha=alpha,
            eps_prune=eps_prune,
            eig_refresh_interval=eig_refresh_interval,
            device=self.device,
            dtype=dtype,
            init=graph_init,
            weights=weights,
        )
        self.threshold = ThresholdAdapter(
            beta=beta,
            gamma=gamma,
            delta_beta=delta_beta,
            delta_gamma=delta_gamma,
            step_size=threshold_step_size,
        )
        self.last_result: ProjectionResult | None = None

    def to(self, device: str | torch.device) -> ALCF:
        self.device = torch.device(device)
        self.graph.to(self.device)
        return self

    @torch.no_grad()
    def project(self, z: torch.Tensor, update: bool = True) -> ProjectionResult:
        z_batch, squeezed = as_batch(z.to(device=self.device, dtype=self.dtype))
        if z_batch.shape[-1] != self.n:
            raise ValueError(f"expected {self.n} graph nodes, got {z_batch.shape[-1]}")
        eigvals, eigvecs = self.graph.eigendecomp()
        q_eigvals = self.graph.filter_eigenvalues(self.polynomial)
        result = project_load(
            z_batch,
            q_eigvals,
            eigvecs,
            beta=self.threshold.beta,
            gamma=self.threshold.gamma,
            n_bisect=self.n_bisect,
        )
        if update:
            if self.adapt_thresholds:
                self.threshold.update(z_batch.sum(dim=-1), result.energy_before)
            if self.update_graph:
                self.graph.update(z_batch)
        if squeezed:
            result = ProjectionResult(
                result.z.squeeze(0),
                result.mean.squeeze(0),
                result.residual.squeeze(0),
                result.multiplier.squeeze(0),
                result.energy_before.squeeze(0),
                result.energy_after.squeeze(0),
            )
        self.last_result = result
        return result

    def __call__(self, z: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.project(z, update=update).z

    def diagnostics(self) -> dict:
        result = self.last_result
        values = {
            "beta": self.threshold.beta,
            "gamma": self.threshold.gamma,
            "algebraic_connectivity": self.graph.algebraic_connectivity(),
            "num_components": self.graph.num_components(),
            "mean_edge_weight": float(self.graph.weights.mean().item()),
            "max_edge_weight": float(self.graph.weights.max().item()),
        }
        if result is not None:
            values.update(
                {
                    "energy_before": result.energy_before.detach().cpu(),
                    "energy_after": result.energy_after.detach().cpu(),
                    "intervention_rate": float((result.multiplier > 0).to(torch.float32).mean().item()),
                }
            )
        return values

    def state_dict(self) -> dict:
        return {
            "graph": self.graph.state_dict(),
            "threshold": self.threshold.state_dict(),
            "config": {"n": self.n, "polynomial": self.polynomial, "n_bisect": self.n_bisect},
        }

    def load_state_dict(self, state: dict) -> None:
        self.graph.load_state_dict(state["graph"])
        self.threshold.load_state_dict(state["threshold"])

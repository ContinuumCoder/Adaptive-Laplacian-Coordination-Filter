from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ThresholdState:
    beta: float
    gamma: float
    step_count: int


class ThresholdAdapter:
    def __init__(
        self,
        beta: float = 4.0,
        gamma: float = 0.05,
        delta_beta: float = 0.1,
        delta_gamma: float = 0.1,
        step_size: float = 1e-2,
        decay: str = "sqrt",
        beta_range: tuple[float, float] = (1e-8, 1e8),
        gamma_range: tuple[float, float] = (1e-8, 1e8),
    ) -> None:
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta_beta = float(delta_beta)
        self.delta_gamma = float(delta_gamma)
        self.step_size = float(step_size)
        self.decay = str(decay)
        self.beta_range = float(beta_range[0]), float(beta_range[1])
        self.gamma_range = float(gamma_range[0]), float(gamma_range[1])
        self.step_count = 0

    def current_step_size(self) -> float:
        if self.decay == "sqrt":
            return self.step_size / (1 + self.step_count) ** 0.5
        if self.decay == "inverse":
            return self.step_size / (1 + self.step_count)
        if self.decay == "constant":
            return self.step_size
        raise ValueError(f"unknown decay schedule {self.decay!r}")

    @torch.no_grad()
    def update(self, total_output: torch.Tensor, energy: torch.Tensor) -> None:
        rate_beta = (total_output.abs() > self.beta).to(torch.float32).mean().item()
        rate_gamma = (energy > self.gamma).to(torch.float32).mean().item()
        eta = self.current_step_size()
        self.beta += eta * (rate_beta - self.delta_beta)
        self.gamma += eta * (rate_gamma - self.delta_gamma)
        self.beta = min(max(self.beta, self.beta_range[0]), self.beta_range[1])
        self.gamma = min(max(self.gamma, self.gamma_range[0]), self.gamma_range[1])
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"beta": self.beta, "gamma": self.gamma, "step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.beta = float(state["beta"])
        self.gamma = float(state["gamma"])
        self.step_count = int(state.get("step_count", 0))

    def state(self) -> ThresholdState:
        return ThresholdState(self.beta, self.gamma, self.step_count)

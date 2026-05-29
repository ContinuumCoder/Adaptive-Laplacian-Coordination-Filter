import torch

from alcf import ALCF
from alcf.projection import project_load
from alcf.utils import laplacian_from_weights


def random_laplacian(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand(n, n, generator=generator)
    weights = 0.5 * (weights + weights.T)
    weights.fill_diagonal_(0)
    laplacian = laplacian_from_weights(weights)
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return laplacian, eigenvalues, eigenvectors


def test_projection_satisfies_constraints() -> None:
    n = 7
    _, eigenvalues, eigenvectors = random_laplacian(n, 11)
    z = torch.randn(16, n) * 3
    beta = 2.0
    gamma = 0.4
    result = project_load(z, eigenvalues, eigenvectors, beta=beta, gamma=gamma)
    total = result.z.sum(dim=-1).abs()
    assert torch.all(total <= beta + 1e-4)
    assert torch.all(result.energy_after <= gamma + 1e-4)


def test_feasible_input_is_unchanged() -> None:
    n = 5
    _, eigenvalues, eigenvectors = random_laplacian(n, 23)
    z = torch.randn(6, n) * 0.01
    result = project_load(z, eigenvalues, eigenvectors, beta=10.0, gamma=10.0)
    assert torch.allclose(result.z, z, atol=1e-6)
    assert result.multiplier.abs().max().item() < 1e-8


def test_uniform_input_only_clips_collective_output() -> None:
    layer = ALCF(n=4, beta=4.0, gamma=0.1, graph_init="ring", adapt_thresholds=False)
    z = torch.full((3, 4), 2.0)
    result = layer.project(z, update=False)
    assert torch.allclose(result.z, torch.ones_like(z), atol=1e-6)
    assert torch.allclose(result.residual, torch.zeros_like(z), atol=1e-6)

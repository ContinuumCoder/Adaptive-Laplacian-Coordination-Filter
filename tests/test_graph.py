import torch

from alcf import GraphLearner, laplacian_from_weights


def test_laplacian_null_space() -> None:
    weights = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    laplacian = laplacian_from_weights(weights)
    ones = torch.ones(4)
    assert torch.allclose(laplacian @ ones, torch.zeros(4), atol=1e-6)
    assert torch.linalg.eigvalsh(laplacian).min().item() > -1e-6


def test_graph_update_is_symmetric() -> None:
    learner = GraphLearner(n=5, alpha=1.0, eps_prune=0.0, init="zero")
    learner.update(torch.randn(8, 5))
    assert torch.allclose(learner.weights, learner.weights.T, atol=1e-6)
    assert torch.allclose(torch.diag(learner.weights), torch.zeros(5), atol=1e-6)


def test_ring_has_one_component() -> None:
    learner = GraphLearner(n=8, init="ring")
    assert learner.num_components() == 1

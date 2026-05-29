# Adaptive Laplacian Coordination Filter

This repository contains a compact reference implementation of the Adaptive Laplacian Coordination Filter (ALCF), a spectral graph projection layer for safe cooperative multi-agent control.

ALCF treats a joint multi-agent load vector as a graph signal. It learns an interaction graph from pairwise load disagreements, forms a graph Laplacian, and projects each load vector onto a safety set that bounds collective output and Laplacian disagreement energy.

## Method

For a load vector `z` in `R^n`, ALCF uses the decomposition

```text
z = mean(z) * 1 + r,    1^T r = 0
```

The safety set is

```text
S(beta, gamma) = { z : |1^T z| <= beta and z^T Q z <= gamma }
```

where `Q = p(L)` is a polynomial spectral graph filter derived from the learned graph Laplacian `L`. The projection separates into a scalar clipping step for the collective component and a Laplacian low-pass filtering step for the mean-zero disagreement component.

## Repository Layout

```text
alcf/
  graph.py          adaptive interaction graph and Laplacian cache
  projection.py     closed-form spectral projection
  threshold.py      online threshold adaptation
  filter.py         end-to-end ALCF wrapper
examples/
  basic_filter.py   minimal load-filtering example
scripts/
  graph_recovery.py controlled graph-recovery diagnostic
tests/
  test_graph.py
  test_projection.py
```

## Installation

```bash
python -m pip install -e .
```

For tests and the graph-recovery plot:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from alcf import ALCF

z = torch.tensor([[1.2, -0.4, 0.7, -1.1]], dtype=torch.float32)
filter_layer = ALCF(n=4, beta=2.0, gamma=0.2, graph_init="ring")
z_filtered = filter_layer(z)
```

To inspect the full projection result:

```python
result = filter_layer.project(z)
print(result.z)
print(result.energy_before, result.energy_after)
```

## Controlled Graph Recovery

The graph-recovery diagnostic generates load sequences from known latent graphs, runs the ALCF edge-weight update, and scores the final weights against the true edge set.

```bash
python scripts/graph_recovery.py --seeds 10
```

To save a bar plot:

```bash
python scripts/graph_recovery.py --seeds 10 --plot outputs/graph_recovery.png
```

## API

`ALCF.project(z, update=True)` returns a `ProjectionResult` with:

- `z`: projected load vector
- `mean`: clipped collective component
- `residual`: projected mean-zero disagreement component
- `multiplier`: KKT multiplier for the spectral constraint
- `energy_before`: Laplacian disagreement energy before projection
- `energy_after`: Laplacian disagreement energy after projection

`ALCF(z)` is a convenience alias returning only the projected load vector.

## Notes

This package implements the safety layer itself. It is intentionally independent of a specific MARL trainer or simulator. A user should provide a task-specific load map from environment actions to one scalar load per graph node.

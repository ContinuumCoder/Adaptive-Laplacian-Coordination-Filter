from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alcf import GraphLearner


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    total = len(pos) * len(neg)
    if total == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return float(wins / total)


def graph_ring(n: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    return [(i, (i + 1) % n) for i in range(n)], np.ones(n)


def graph_two_cluster(n: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    left = list(range(n // 2))
    right = list(range(n // 2, n))
    edges = []
    for group in (left, right):
        edges.extend((i, j) for i, j in zip(group, group[1:] + group[:1]))
    edges.append((left[-1], right[0]))
    weights = np.ones(len(edges))
    weights[-1] = 0.35
    return edges, weights


def graph_grid(side: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    edges = []
    for row in range(side):
        for col in range(side):
            node = row * side + col
            if col + 1 < side:
                edges.append((node, node + 1))
            if row + 1 < side:
                edges.append((node, node + side))
    return edges, np.ones(len(edges))


def graph_random_geometric(n: int, seed: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    edges = []
    weights = []
    for i, j in itertools.combinations(range(n), 2):
        dist = np.linalg.norm(points[i] - points[j])
        if dist < 0.42:
            edges.append((i, j))
            weights.append(math.exp(-4.0 * dist))
    if len(edges) < n:
        pairs = list(itertools.combinations(range(n), 2))
        order = np.argsort([np.linalg.norm(points[i] - points[j]) for i, j in pairs])
        for idx in order:
            edge = pairs[int(idx)]
            if edge not in edges:
                edges.append(edge)
                weights.append(0.25)
            if len(edges) >= n:
                break
    return edges, np.asarray(weights, dtype=float)


def simulate(
    n: int,
    edges: list[tuple[int, int]],
    weights: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    noise: float,
    alpha: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    learner = GraphLearner(n=n, alpha=alpha, eps_prune=0.0, eig_refresh_interval=1, init="zero")
    edge_array = np.asarray(edges, dtype=int)
    weight_array = np.asarray(weights, dtype=float)
    for _ in range(steps):
        z = 0.08 * rng.standard_normal((batch_size, n))
        z += rng.normal(scale=0.8, size=(batch_size, 1))
        for (i, j), weight in zip(edge_array, weight_array):
            amp = rng.normal(scale=math.sqrt(float(weight)), size=batch_size)
            z[:, i] += amp
            z[:, j] -= amp
        z += noise * rng.standard_normal((batch_size, n))
        learner.update(torch.tensor(z, dtype=torch.float32))
    pairs = list(itertools.combinations(range(n), 2))
    true_edges = {tuple(sorted(edge)) for edge in edges}
    labels = np.array([pair in true_edges for pair in pairs])
    scores = np.array([learner.weights[i, j].item() for i, j in pairs])
    k = int(labels.sum())
    pred = np.zeros_like(labels, dtype=bool)
    pred[np.argsort(scores)[-k:]] = True
    tp = np.logical_and(pred, labels).sum()
    precision = tp / max(pred.sum(), 1)
    recall = tp / max(labels.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return auc_score(labels, scores), float(f1)


def summarize(args: argparse.Namespace) -> list[dict[str, object]]:
    configs = [
        ("ring", 12, lambda seed: graph_ring(12)),
        ("two-cluster", 12, lambda seed: graph_two_cluster(12)),
        ("grid", 16, lambda seed: graph_grid(4)),
        ("random geometric", 16, lambda seed: graph_random_geometric(16, seed)),
    ]
    rows = []
    for name, n, make_graph in configs:
        values = []
        edge_count = 0
        for seed in range(args.seeds):
            edges, weights = make_graph(seed)
            edge_count = len(edges)
            values.append(simulate(n, edges, weights, seed, args.steps, args.batch_size, args.noise, args.alpha))
        arr = np.asarray(values)
        rows.append(
            {
                "graph": name,
                "n": n,
                "edges": edge_count,
                "auc_mean": float(arr[:, 0].mean()),
                "auc_std": float(arr[:, 0].std()),
                "f1_mean": float(arr[:, 1].mean()),
                "f1_std": float(arr[:, 1].std()),
            }
        )
    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    print("graph,n,edges,auc_mean,auc_std,f1_mean,f1_std")
    for row in rows:
        print(
            f"{row['graph']},{row['n']},{row['edges']},"
            f"{row['auc_mean']:.3f},{row['auc_std']:.3f},"
            f"{row['f1_mean']:.3f},{row['f1_std']:.3f}"
        )


def save_plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [str(row["graph"]) for row in rows]
    auc = [float(row["auc_mean"]) for row in rows]
    auc_err = [float(row["auc_std"]) for row in rows]
    f1 = [float(row["f1_mean"]) for row in rows]
    f1_err = [float(row["f1_std"]) for row in rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - width / 2, auc, width, yerr=auc_err, label="AUC", color="#4c78a8")
    ax.bar(x + width / 2, f1, width, yerr=f1_err, label=r"Top-$|E|$ F1", color="#f58518")
    ax.axhline(0.5, color="0.55", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.82), borderaxespad=0.0)
    fig.subplots_adjust(left=0.08, right=0.80, bottom=0.25, top=0.96)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    rows = summarize(args)
    print_table(rows)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.plot is not None:
        save_plot(rows, args.plot)


if __name__ == "__main__":
    main()

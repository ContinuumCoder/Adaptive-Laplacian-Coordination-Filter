from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alcf import ALCF


def main() -> None:
    torch.manual_seed(7)
    layer = ALCF(n=6, beta=3.0, gamma=0.25, graph_init="ring", eig_refresh_interval=1)
    z = torch.randn(4, 6)
    result = layer.project(z)
    print("raw")
    print(z)
    print("filtered")
    print(result.z)
    print("energy_before", result.energy_before)
    print("energy_after", result.energy_after)
    print(layer.diagnostics())


if __name__ == "__main__":
    main()

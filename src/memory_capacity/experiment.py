from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .model import TinyGPT, count_parameters


@dataclass(frozen=True)
class Config:
    dataset_sizes: tuple[int, ...] = (32, 128, 512)
    sequence_length: int = 32
    vocab_size: int = 32
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    steps: int = 300
    batch_size: int = 64
    learning_rate: float = 3e-3
    seeds: tuple[int, ...] = (0,)
    device: str = "auto"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_uniform_dataset(size: int, sequence_length: int, vocab_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(vocab_size, (size, sequence_length), generator=generator)


def nll_bits(model: TinyGPT, data: torch.Tensor, device: torch.device, batch_size: int) -> float:
    model.eval()
    total_nll_nats = 0.0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size].to(device)
            logits = model(batch[:, :-1])
            total_nll_nats += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction="sum"
            ).item()
    return total_nll_nats / math.log(2)


def measure(model: TinyGPT, data: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    # The first token has no context and is therefore not predicted by this causal LM.
    predictable_tokens = data.shape[0] * (data.shape[1] - 1)
    uniform_bits = predictable_tokens * math.log2(int(model.lm_head.out_features))
    compressed_bits = nll_bits(model, data, device, batch_size)
    memorized_bits = max(0.0, uniform_bits - compressed_bits)
    return {
        "uniform_bits": uniform_bits,
        "compressed_bits": compressed_bits,
        "memorized_bits": memorized_bits,
        "fraction_memorized": memorized_bits / uniform_bits,
    }


def train_one(config: Config, dataset_size: int, seed: int, device: torch.device) -> dict[str, float]:
    seed_everything(seed)
    data = make_uniform_dataset(dataset_size, config.sequence_length, config.vocab_size, seed)
    model = TinyGPT(
        config.vocab_size,
        config.sequence_length,
        config.d_model,
        config.n_heads,
        config.n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 10_000)
    started = time.perf_counter()
    model.train()
    for _ in range(config.steps):
        indices = torch.randint(dataset_size, (min(config.batch_size, dataset_size),), generator=generator)
        batch = data[indices].to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = measure(model, data, device, config.batch_size)
    return {
        "dataset_size": dataset_size,
        "seed": seed,
        "parameters": count_parameters(model),
        "train_loss_nats": float(loss.detach().cpu()),
        "elapsed_seconds": time.perf_counter() - started,
        **metrics,
    }


def save_results(rows: list[dict[str, float]], config: Config, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    import matplotlib.pyplot as plt

    grouped: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["dataset_size"]), []).append(row)
    sizes = sorted(grouped)
    means = [np.mean([row["memorized_bits"] for row in grouped[size]]) for size in sizes]
    stds = [np.std([row["memorized_bits"] for row in grouped[size]]) for size in sizes]
    available = [size * (config.sequence_length - 1) * math.log2(config.vocab_size) for size in sizes]

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label="measured memorization")
    axis.plot(sizes, available, "--", color="0.45", label="dataset information")
    axis.set(xscale="log", yscale="log", xlabel="dataset size (sequences)", ylabel="bits")
    axis.set_title("Tiny reproduction: uniform random sequences")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "memorization.png", dpi=180)
    plt.close(fig)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-sizes", type=parse_ints, default=Config.dataset_sizes)
    parser.add_argument("--seeds", type=parse_ints, default=Config.seeds)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--output-dir", type=Path, default=Path("results/quickstart"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = Config(dataset_sizes=args.dataset_sizes, seeds=args.seeds, steps=args.steps, device=args.device)
    device = choose_device(config.device)
    rows = []
    for dataset_size in config.dataset_sizes:
        for seed in config.seeds:
            row = train_one(config, dataset_size, seed, device)
            rows.append(row)
            print(
                f"N={dataset_size:>5} seed={seed} "
                f"memorized={row['memorized_bits']:>10.0f} bits "
                f"({row['fraction_memorized']:.1%})"
            )
    save_results(rows, config, args.output_dir)
    print(f"Saved metrics and plot to {args.output_dir}")


if __name__ == "__main__":
    main()

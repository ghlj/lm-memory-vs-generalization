import math

import torch

from memory_capacity.experiment import Config, make_uniform_dataset, measure
from memory_capacity.model import TinyGPT, count_parameters


def test_uniform_dataset_is_reproducible() -> None:
    first = make_uniform_dataset(8, 12, 7, seed=4)
    second = make_uniform_dataset(8, 12, 7, seed=4)
    assert torch.equal(first, second)
    assert first.min() >= 0 and first.max() < 7


def test_measurement_uses_paper_equation() -> None:
    model = TinyGPT(vocab_size=8, sequence_length=6, d_model=8, n_heads=2)
    data = make_uniform_dataset(4, 6, 8, seed=0)
    result = measure(model, data, torch.device("cpu"), batch_size=4)
    assert result["uniform_bits"] == 4 * 5 * math.log2(8)
    assert 0 <= result["memorized_bits"] <= result["uniform_bits"]


def test_default_model_is_genuinely_tiny() -> None:
    config = Config()
    model = TinyGPT(
        config.vocab_size, config.sequence_length, config.d_model, config.n_heads, config.n_layers
    )
    assert count_parameters(model) < 100_000


# How much do tiny language models memorize?

A low-cost, end-to-end reproduction of the central synthetic-data experiment in
[Morris et al., *How much do language models memorize?* (2025)](https://arxiv.org/abs/2505.24832).

The paper measures memorization as compression. For uniformly random sequences,
there is no learnable structure and therefore no generalization to confuse with
memorization. If a dataset contains `N` sequences of `S` predictable tokens from
a vocabulary of size `V`, its information content is

```text
uniform_bits = N * S * log2(V)
compressed_bits = -sum(log2 p_model(token | prefix))
memorized_bits = uniform_bits - compressed_bits
```

This repository trains a tiny causal Transformer from scratch on several fixed
random datasets, evaluates those quantities on every training sequence, writes
the raw metrics, and produces the paper-style memorization curve.

## Cheapest end-to-end run

Requirements: Python 3.9+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run memory-capacity
```

Outputs appear in `results/quickstart/`:

- `metrics.csv`: one row per dataset size and seed
- `config.json`: exact run configuration
- `memorization.png`: measured bits and total available dataset information

For a smoke test that finishes quickly:

```bash
uv run memory-capacity --dataset-sizes 16,64 --steps 10 --output-dir results/smoke
uv run pytest
```

Apple Silicon automatically uses MPS; NVIDIA hosts use CUDA; otherwise the run
uses CPU. Pass `--device cpu` to force CPU execution. Add seeds with
`--seeds 0,1,2` when runtime permits.

## What this reproduces—and what it does not

The implementation follows Section 3.2's essential setup: GPT-style causal
language modeling, uniform independent tokens, scratch training, and arithmetic
code length estimated by negative log-likelihood. The default sweep is deliberately
much cheaper: one 15.8K-parameter model, three dataset sizes, one seed, 300 steps,
vocabulary 32, and sequence length 32.

The checked-in quickstart run produced 4,913, 15,853, and 9,356 memorized
bits for 32, 128, and 512 sequences respectively—99.1%, 79.9%, and 11.8% of
the available random information. The absolute memorized bits need not be
monotone in this tiny fixed-step regime: the largest dataset receives far fewer
updates per example. The falling fraction is the intended cheapest illustration.

It should demonstrate the idea that the model assigns extra probability to its
random training strings, so they become compressible, and that the fraction of
the dataset memorized falls as the fixed model is given more random information.
It is **not** a numerical reproduction of the paper's 3.6 bits-per-parameter
estimate. That result used 100K–20M parameter models, vocabulary 2048, sequence
length 64, one million optimization steps, five seeds, and A100 GPUs. Treat this
project as the inexpensive first rung of that replication ladder.

## Reproducibility notes

- The random dataset and initialization are controlled by each run's seed.
- The first token is excluded from both terms because a causal LM has no token
  context with which to predict it; this keeps the two code lengths comparable.
- Memorization is clipped at zero because tiny untrained/noisy models can compress
  slightly worse than the known uniform reference.
- Results can differ slightly across CPU, MPS, and CUDA kernels.

## Next experiments

1. Sweep widths/layers and estimate the maximum memorized bits per parameter.
2. Increase data size until each model's memorization curve clearly plateaus.
3. Repeat across seeds and precisions.
4. Only then move to deduplicated text and an independently trained reference
   model, which is substantially more expensive.

## Citation

```bibtex
@article{morris2025howmuch,
  title={How much do language models memorize?},
  author={Morris, John X. and Sitawarin, Chawin and Guo, Chuan and Kokhlikyan, Narine and
          Suh, G. Edward and Rush, Alexander M. and Chaudhuri, Kamalika and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2505.24832},
  year={2025}
}
```

## License

MIT

# Measuring memorization and generalization in language models

A small, end-to-end reproduction of the central synthetic experiment in
[Morris et al., *How much do language models memorize?*](https://arxiv.org/abs/2505.24832).

## The idea in one sentence

A model knows something about a sequence when the model lets us encode that
sequence in fewer bits.

## Start from first principles

Suppose the next token is `x` and a model assigns it probability `p(x)`. An
ideal code needs

```text
-log2 p(x) bits
```

to store that token. Likely tokens are cheap; surprising tokens are expensive.
For a sequence, add this quantity over its tokens. This is the model's negative
log-likelihood (NLL), measured in bits.

Now compare two code lengths:

```text
information before using the model
- information left after using the model
= information stored by the model
```

That difference is the amount memorized.

## Why memorization and generalization are easy to confuse

Imagine that a model predicts `2 + 2 = 4`. A short code for `4` does not prove
that the exact equation was in its training set. The model may have learned the
general rule of addition.

We therefore compare the trained **target model** with a **reference model** that
knows the general data distribution but not the particular training sample:

```text
unintended memorization(x)
    = max(0, NLL_reference(x) - NLL_target(x))
```

- If both models compress `x` equally well, there is no extra evidence that the
  target retained sample-specific information.
- If the target compresses `x` better than the reference, the extra savings are
  evidence that it retained information specific to that sample: **memorization**.

Conceptually:

```text
total knowledge about x = generalization + unintended memorization
```

This split depends on the reference model. A weak reference mistakes useful
general knowledge for memorization; a strong reference gives a cleaner estimate.

## The cheapest clean experiment

Use uniformly random token sequences. Random tokens contain no pattern to learn,
so generalization is zero by construction. The true reference distribution is
known exactly: every token has probability `1 / V`, where `V` is vocabulary size.

For `N` sequences with `S` predicted tokens:

```text
uniform_bits    = N * S * log2(V)
model_bits      = -sum(log2 p_target(token | prefix))
memorized_bits  = max(0, uniform_bits - model_bits)
fraction_stored = memorized_bits / uniform_bits
```

No large reference model or text dataset is required. This repository generates
the random data, trains a 15.8K-parameter causal Transformer from scratch,
measures its code length, and plots the result.

## Run it

Requires Python 3.9+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run memory-capacity
```

Results are written to `results/quickstart/`:

- `metrics.csv` — raw measurements
- `config.json` — exact configuration
- `memorization.png` — the resulting curve

Run the checks with:

```bash
uv run pytest
```

The device is selected automatically: CUDA, Apple MPS, then CPU. Use
`--device cpu` to override it.

## What the tiny run shows

| Random sequences | Available bits | Memorized bits | Fraction memorized |
|---:|---:|---:|---:|
| 32 | 4,960 | 4,913 | 99.1% |
| 128 | 19,840 | 15,853 | 79.9% |
| 512 | 79,360 | 9,356 | 11.8% |

The small dataset is nearly recoverable from the model's probabilities. As the
same model receives more independent random information, it stores a smaller
fraction. With only 300 fixed training steps, the largest dataset also receives
fewer updates per example, so this run illustrates the mechanism rather than a
precise capacity limit.

The paper's full experiment trains many 100K–20M parameter GPT-style models for
one million steps and finds a saturation point near 3.6 memorized bits per
parameter. This project is the inexpensive first rung, not a numerical
reproduction of that estimate.

## How to measure this on real text

1. Train a target model on the dataset of interest.
2. Choose a strong reference model from the same model family, trained on a much
   wider distribution without relying on the target sample.
3. Sum token-level NLL into a code length for each sequence under both models.
4. Compute `max(0, NLL_reference(x) - NLL_target(x))` for each sequence, then
   sum over samples.
5. Repeat on held-out samples and across seeds; report the reference model,
   tokenizer, precision, and code-length convention.

The hard part is step 2. On random data the reference is exact; on real text it
is only an approximation, so the measured split must be interpreted relative to
that reference.

## Repository layout

```text
src/memory_capacity/model.py       tiny causal Transformer
src/memory_capacity/experiment.py  data, training, measurement, and plotting
tests/                             correctness checks
results/quickstart/                checked-in example outputs
```

## Citation and license

Based on arXiv:2505.24832. Code in this repository is released under the MIT
License.

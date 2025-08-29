# MCRL - Monte Carlo Reinforcement Learning

A Rust utility for computing the state-value function V(S) based on trajectories (e.g., from reinforcement learning episodes). It processes datasets where each line is a JSON array representing a trajectory, computes discounted cumulative rewards (returns) for each state, and aggregates these returns by state. The output includes confidence intervals and optional pairwise statistical comparisons.

## Features

- **Discounted Returns**: Computes V(S) using the discount factor `gamma` on rewards.
- **Confidence Intervals**: Uses bootstrap resampling to compute lower/upper bounds (95% CI by default).
- **Windsorizing**: Truncates rewards above/below specified bounds (`--ub-windsorize`, `--lb-windsorize`).
- **State Aggregation**: Supports multiple state fields (e.g., state = [state_id, action]).
- **Comparison Tests**: Optional pairwise tests between states using permutation tests and Bayesian bootstrap.
- **First‑Visit / Every‑Visit Mode**: Use `--every-visit` to compute V(S) based on *every* visit of a state (default is first‑visit).
- **Estimator Choice**: Choose the central‑tendency estimator for comparison tests via `--estimator mean|median`.
- **Probability ≥ option**: Include probability P(B ≥ A) instead of strict > using `--probability-gte`.
- **Custom Rounding / Seed**: Control output precision with `--round` and reproducibility with `--random-seed`.

## Usage

```bash
mcrl-rs --dataset <file.json> --state-field <field1> --reward-field <reward>
```

### Required Arguments

- `--dataset`: Path to the dataset (one JSON array per line).
- `--state-field`: Field(s) containing the state (repeatable for multiple fields).
- `--reward-field`: Field containing the immediate reward.

### Optional Arguments

- `--discount`: Discount factor (default: 1.0).
- `--ci`: Confidence interval percentage (default: 95).
- `--min-observations`: Minimum observations per state (default: 0).
- `--ub-windsorize`, `--lb-windsorize`: Upper/lower bounds for reward truncation.
- `--comparison-test`: Number of permutations for pairwise tests (enables comparison output).
- `--every-visit`: Use *every* visit of a state instead of first‑visit when aggregating returns.
- `--estimator <mean|median>`: Choose the estimator used in comparison tests (default: `mean`).
- `--comparison-group-fields <list>`: Comma‑separated list of state field indices to group states for pairwise comparisons, e.g., `0,2`.
- `--probability-gte`: Compute probability P(B ≥ A) instead of the strict inequality.
- `--alpha <value>`: Concentration parameter for Bayesian bootstrap (default: 1.0).
- `--round <digits>`: Number of decimal places displayed in output tables (default: 3).
- `--random-seed <seed>`: Seed for RNG to enable reproducible results.
- `--bootstrap-samples <n>`: Number of bootstrap samples for confidence intervals and other bootstrapping operations (default: 2000).

## Output

### Main Table
```text
╭──────────────┬──────────┬──────────────┬─────────────┬─────────────────────────┬─────────────╮
│ buyer_type   ┆ variant  ┆ Observations ┆ Lower Bound ┆ Expected Sales Price    ┆ Upper Bound │
╞══════════════╪══════════╪══════════════╪═════════════╪═════════════════════════╪═════════════╡
│ new_buyer    ┆ 1        ┆ 2659         ┆ 85.264      ┆ 90.988                  ┆ 96.946      │
│ new_buyer    ┆ 2        ┆ 2826         ┆ 87.992      ┆ 93.632                  ┆ 99.659      │
│ signed_out   ┆ 1        ┆ 1859         ┆ 65.405      ┆ 69.229                  ┆ 73.095      │
│ signed_out   ┆ 2        ┆ 1906         ┆ 70.771      ┆ 74.642                  ┆ 78.646      │
│ top_1_buyer  ┆ 1        ┆ 13592        ┆ 94.682      ┆ 97.498                  ┆ 100.487     │
│ top_1_buyer  ┆ 2        ┆ 13286        ┆ 92.813      ┆ 95.271                  ┆ 97.744      │
╰──────────────┴──────────┴──────────────┴─────────────┴─────────────────────────┴─────────────╯
```

- **Observations**: Number of times the state was visited.
- **Lower/Upper Bound**: Bootstrap confidence interval (e.g., 95 % CI) for the expected reward (`goa_all_reward` column).

### Pairwise Comparison Table (if `--comparison-test` is provided)
| A         | B         | |A| | |B| | E[B]-E[A] | P-Value | P(B > A) | P(E[B] > E[A]) | Bayes P(B > A, alpha=1.0) |
|-----------|-----------|----|----|------------|---------|----------|----------------|---------------------------|
| state1    | state2    | 100| 50 | 0.15       | 0.03*   | 0.62     | 0.74           | 0.65                      |

- `*` indicates significance (p-value ≤ 5%).
- **P(E[B] > E[A])**: Probability that the expected return of B exceeds that of A, estimated via non‑parametric bootstrap.
- **Bayes P(B > A, alpha=…)**: Bayesian probability using a Dirichlet prior with concentration parameter `alpha` (default 1.0). The column title reflects the chosen `--alpha`.
>
> **Note:** If `--probability-gte` is specified, the “P(B > A)” column displays **P(B ≥ A)** instead.

## Statistical Methods

- **Bootstrap Confidence Intervals**: Resamples data to compute bounds.
- **Permutation Tests**: For statistical significance of differences between states.
- **Bayesian Bootstrap**: Uses Dirichlet distribution for prior (alpha parameter).

## Example

Compute V(S) with 95% CI and pairwise comparisons:

```bash
mcrl-rs --dataset trajectories.json \
       --state-field state_id --state-field action \
       --reward-field reward \
       --ci 95 \
       --comparison-test 1000
```

This will output a main table of state values and a comparison table for all state pairs.

# MCRL - Monte Carlo Reinforcement Learning

A Rust utility for computing the state-value function V(S) based on trajectories (e.g., from reinforcement learning episodes). It processes datasets where each line is a JSON array representing a trajectory, computes discounted cumulative rewards (returns) for each state, and aggregates these returns by state. The output includes confidence intervals and optional pairwise statistical comparisons.

## Features

- **Discounted Returns**: Computes V(S) using the discount factor `gamma` on rewards.
- **Confidence Intervals**: Uses bootstrap resampling to compute lower/upper bounds (95% CI by default).
- **Windsorizing**: Truncates rewards above/below specified bounds (`--ub-windsorize`, `--lb-windsorize`).
- **State Aggregation**: Supports multiple state fields (e.g., state = [state_id, action]).
- **Comparison Tests**: Optional pairwise tests between states using permutation tests and Bayesian bootstrap.

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

## Output

### Main Table
| State Field 1 | State Field 2 | Observations | Lower Bound | Expected Reward | Upper Bound |
|---------------|---------------|--------------|-------------|------------------|-------------|
| state1        | actionA       | 100          | 0.85        | 1.23             | 1.67        |

- **Observations**: Number of times the state was visited.
- **Lower/Upper Bound**: Bootstrap confidence interval (e.g., 95% CI).

### Pairwise Comparison Table (if `--comparison-test` is provided)
| A         | B         | |A| | |B| | E[B]-E[A] | P-Value | P(B > A) | P(E[B] > E[A]) | Bayes P(B > A, alpha=1.0) |
|-----------|-----------|----|----|------------|---------|----------|----------------|---------------------------|
| state1    | state2    | 100| 50 | 0.15       | 0.03*   | 0.62     | 0.74           | 0.65                      |

- `*` indicates significance (p-value ≤ 5%).
- **Bayes P(B > A)**: Bayesian probability using Dirichlet prior.

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

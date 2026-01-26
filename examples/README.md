# mcrl-rs Examples

Real-world inspired examples demonstrating mcrl-rs features with pre-generated trajectory data.

## Examples Overview

| Example | Use Case | Key Features |
|---------|----------|--------------|
| [GridWorld](gridworld/) | RL policy evaluation | Discounting, multi-field states, value gradients |
| [A/B Test](ab_test/) | Conversion optimization | Grouped comparisons, segmentation, Winsorization |
| [Bandit](bandit/) | Reward distribution analysis | Mean vs median estimators, variance comparison |

---

## 1. GridWorld Navigation

**Use case**: Evaluate navigation policies in a reinforcement learning environment.

A 4x4 grid where an agent navigates from (0,0) to goal at (3,3). Step cost is -1, goal reward is +10. The dataset contains optimal, mixed, and random trajectories.

**What you'll learn**:
- How discount factors affect value estimates (states near goal have higher values)
- Multi-field state definitions (`--state-field x --state-field y`)
- Reading trajectory format with sequential steps

```bash
./target/release/mcrl-rs --dataset examples/gridworld/trajectories.jsonl \
    --state-field x --state-field y \
    --reward-field reward \
    --discount 0.95
```

---

## 2. A/B Testing

**Use case**: Compare checkout variants across customer segments to optimize conversion.

An e-commerce site tests two checkout flows (A vs B) across three user segments (new_visitor, returning, premium). Each trajectory is a user session from landing to purchase or abandonment.

**What you'll learn**:
- Grouped comparisons with `--comparison-group-fields` (compare A vs B within each segment)
- Handling monetary data with `--ub-windsorize` to cap outliers
- Statistical comparison metrics (P-value, P(B > A), Bayesian probability)

```bash
./target/release/mcrl-rs --dataset examples/ab_test/sessions.jsonl \
    --state-field segment --state-field variant \
    --reward-field purchase-price \
    --comparison-test 1000 \
    --comparison-group-fields 0
```

---

## 3. Multi-Armed Bandit

**Use case**: Identify the best slot machine when reward distributions vary in mean and variance.

Five slot machines with different payout characteristics: some pay consistently, others have rare jackpots. Demonstrates why mean and median estimators give different "best" answers.

**What you'll learn**:
- Mean estimator favors high expected value (jackpot machines rank higher)
- Median estimator favors typical outcomes (consistent machines rank higher)
- Wide confidence intervals indicate high-variance options

```bash
# Mean: machine_5 wins (highest expected value)
./target/release/mcrl-rs --dataset examples/bandit/pulls.jsonl \
    --state-field machine \
    --reward-field reward \
    --comparison-test 1000 \
    --estimator mean

# Median: machine_1 wins (most consistent payouts)
./target/release/mcrl-rs --dataset examples/bandit/pulls.jsonl \
    --state-field machine \
    --reward-field reward \
    --comparison-test 1000 \
    --estimator median
```

---

## Quick Start

```bash
# Build release binary
cargo build --release

# Run any example
./target/release/mcrl-rs --dataset examples/gridworld/trajectories.jsonl \
    --state-field x --state-field y --reward-field reward --discount 0.95
```

## Data Format

All examples use JSON Lines format (one trajectory per line):
```json
[{"state":"s1","action":"a1","reward":0},{"state":"s2","action":"a2","reward":10}]
```

Each line is a complete trajectory (episode) represented as a JSON array of steps.

## Regenerating Data

Each example includes a Python generator script (Python 3.6+, no dependencies):

```bash
python3 examples/gridworld/generate_data.py > examples/gridworld/trajectories.jsonl
python3 examples/ab_test/generate_data.py > examples/ab_test/sessions.jsonl
python3 examples/bandit/generate_data.py > examples/bandit/pulls.jsonl
```

All generators use `seed=42` by default for reproducibility. Use `--seed N` to change.

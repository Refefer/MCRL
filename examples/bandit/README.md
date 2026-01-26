# Multi-Armed Bandit Example

Five slot machines with different payout distributions demonstrating mean vs median estimators.

## Scenario

A casino has 5 slot machines. Each machine has a different payout distribution:

| Machine | Mean | Type | Description |
|---------|------|------|-------------|
| machine_1 | $1.50 | Normal (std=0.30) | Safe - consistent payouts |
| machine_2 | $1.20 | 90% $0, 10% ~$12 | High variance jackpot |
| machine_3 | $0.80 | Normal (std=0.25) | Trap - consistent but low |
| machine_4 | $1.40 | Normal (std=0.50) | Medium variance |
| machine_5 | $1.60 | 80% $0, 20% ~$8 | Bonus machine |

The key insight: **mean** and **median** estimators give different rankings.

## Data

The dataset contains 5000 pulls (1000 per machine). Each trajectory is a single pull:

```json
[{"machine":"machine_1","reward":1.45}]
[{"machine":"machine_2","reward":0.00}]
[{"machine":"machine_2","reward":12.50}]
```

## Running

### Mean Estimator (Expected Value)

```bash
./target/release/mcrl-rs --dataset examples/bandit/pulls.jsonl \
    --state-field machine \
    --reward-field reward \
    --discount 1.0 \
    --comparison-test 5000 \
    --estimator mean \
    --bootstrap-samples 5000
```

Expected ranking: machine_5 > machine_1 > machine_4 > machine_2 > machine_3

### Median Estimator (Typical Outcome)

```bash
./target/release/mcrl-rs --dataset examples/bandit/pulls.jsonl \
    --state-field machine \
    --reward-field reward \
    --discount 1.0 \
    --comparison-test 5000 \
    --estimator median \
    --bootstrap-samples 5000
```

Expected ranking: machine_1 > machine_4 > machine_3 > machine_2 = machine_5

Note: machine_2 and machine_5 have median $0 because most pulls pay nothing!

## Expected Results

### Mean Estimator

Best machine by expected value is machine_5 ($1.60 mean), despite its high variance.

| Machine | Mean | 95% CI |
|---------|------|--------|
| machine_5 | ~$1.60 | wide |
| machine_1 | ~$1.50 | narrow |
| machine_4 | ~$1.40 | medium |
| machine_2 | ~$1.20 | wide |
| machine_3 | ~$0.80 | narrow |

### Median Estimator

Best machine by typical outcome is machine_1, which pays reliably:

| Machine | Median | 95% CI |
|---------|--------|--------|
| machine_1 | ~$1.50 | narrow |
| machine_4 | ~$1.40 | narrow |
| machine_3 | ~$0.80 | narrow |
| machine_2 | $0.00 | - |
| machine_5 | $0.00 | - |

## Key mcrl-rs Features Demonstrated

- **`--estimator mean`**: Uses mean for value estimation (default)
- **`--estimator median`**: Uses median for value estimation (robust to outliers)
- **Confidence intervals**: Wide CIs indicate high-variance machines
- **Comparison tests**: P(A > B) comparisons between all machine pairs

## When to Use Each Estimator

**Mean estimator** (default):
- When you care about long-run expected value
- When sample sizes are large
- When outliers represent real outcomes you want to optimize for

**Median estimator**:
- When you want the "typical" outcome
- When distributions are heavily skewed
- When outliers may be data errors
- When you prefer consistent returns over occasional jackpots

## Generator Script

The `generate_data.py` script creates slot machine pull data with configurable parameters.

### Usage

```bash
# Generate default dataset (5000 pulls, seed=42)
python3 generate_data.py > pulls.jsonl

# Custom seed and pull count
python3 generate_data.py --seed 123 --pulls 20000 > large_test.jsonl

# Write directly to file
python3 generate_data.py --output pulls.jsonl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seed N` | 42 | Random seed for reproducibility |
| `--pulls N` | 5000 | Total number of pulls to generate |
| `--output FILE` | stdout | Output file path |

### How It Works

1. **Pull distribution**: Pulls are evenly distributed across all 5 machines
2. **Normal machines** (1, 3, 4): Rewards sampled from Gaussian distribution, clamped to non-negative
3. **Spike machines** (2, 5): Binary outcome - zero with high probability, jackpot otherwise
4. **Jackpot variance**: Spike values have ±10% random variation

### Requirements

- Python 3.6+
- No external dependencies (uses only `json`, `random`, `argparse`)

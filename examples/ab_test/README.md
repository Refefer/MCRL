# A/B Testing Example

E-commerce checkout optimization comparing two variants across user segments.

## Scenario

An online store tests two checkout flows:
- **Variant A**: Original checkout
- **Variant B**: Redesigned checkout

User segments:
- **new_visitor**: First-time visitors
- **returning**: Users with previous visits
- **premium**: High-value customers

Each trajectory represents a user session from landing page to checkout or abandonment.

## Data

The dataset contains 1000 sessions with built-in conversion rate differences:

| Segment | Variant A | Variant B | Avg Order Value |
|---------|-----------|-----------|-----------------|
| new_visitor | 8% | 10% | $45 |
| returning | 15% | 14% | $65 |
| premium | 25% | 28% | $120 |

Note: Variant B is better for new_visitor and premium, but slightly worse for returning users.

### Format

Each line is a session trajectory as a JSON array:
```json
[{"segment":"new_visitor","variant":"A","page":"landing","purchase-price":0},{"segment":"new_visitor","variant":"A","page":"product","purchase-price":0},{"segment":"new_visitor","variant":"A","page":"checkout","purchase-price":47.50}]
```

Fields:
- `segment`: User segment (new_visitor, returning, premium)
- `variant`: A/B test variant (A or B)
- `page`: Current page (landing, product, cart, checkout)
- `purchase-price`: 0 during browsing, order value at conversion

## Running

Basic comparison:
```bash
./target/release/mcrl-rs --dataset examples/ab_test/sessions.jsonl \
    --state-field segment --state-field variant \
    --reward-field purchase-price \
    --discount 1.0
```

With statistical comparison by segment:
```bash
./target/release/mcrl-rs --dataset examples/ab_test/sessions.jsonl \
    --state-field segment --state-field variant \
    --reward-field purchase-price \
    --discount 1.0 \
    --comparison-test 5000 \
    --comparison-group-fields 0 \
    --min-observations 50
```

With Winsorization for outliers:
```bash
./target/release/mcrl-rs --dataset examples/ab_test/sessions.jsonl \
    --state-field segment --state-field variant \
    --reward-field purchase-price \
    --discount 1.0 \
    --comparison-test 5000 \
    --comparison-group-fields 0 \
    --ub-windsorize 200
```

## Expected Results

The value table should show:
- Revenue per session by (segment, variant) combination
- Higher values for premium segment (higher conversion rate & order value)
- Similar values for A vs B within returning segment
- Slightly higher B values for new_visitor and premium

The comparison test with `--comparison-group-fields 0` compares variants within each segment:
- new_visitor: P(B > A) > 0.5 (B is better)
- returning: P(B > A) < 0.5 (A is slightly better)
- premium: P(B > A) > 0.5 (B is better)

## Key mcrl-rs Features Demonstrated

- **`--comparison-group-fields`**: Groups comparisons by segment, so A vs B comparisons happen within each user segment
- **`--min-observations`**: Filters out states with too few samples for reliable estimates
- **`--ub-windsorize`**: Caps extreme order values to reduce variance from outliers
- **`--discount 1.0`**: No discounting (appropriate for single-session analysis)

## Generator Script

The `generate_data.py` script creates synthetic e-commerce session data with configurable parameters.

### Usage

```bash
# Generate default dataset (1000 sessions, seed=42)
python3 generate_data.py > sessions.jsonl

# Custom seed and session count
python3 generate_data.py --seed 123 --sessions 5000 > large_test.jsonl

# Write directly to file
python3 generate_data.py --output sessions.jsonl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seed N` | 42 | Random seed for reproducibility |
| `--sessions N` | 1000 | Total number of sessions to generate |
| `--output FILE` | stdout | Output file path |

### How It Works

1. **Session distribution**: Sessions are evenly distributed across all segment/variant combinations
2. **Page flow simulation**: Each session follows a probabilistic funnel (landing → product → cart → checkout)
3. **Conversion logic**: Conversion rates vary by segment and variant (see table above)
4. **Order values**: Generated from a normal distribution around segment-specific averages

### Requirements

- Python 3.6+
- No external dependencies (uses only `json`, `random`, `argparse`)

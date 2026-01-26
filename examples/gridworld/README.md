# GridWorld Navigation Example

A classic reinforcement learning environment where an agent navigates a 4x4 grid from start to goal.

## Scenario

- **Grid**: 4x4 (positions 0-3 on each axis)
- **Start**: (0, 0) - bottom-left corner
- **Goal**: (3, 3) - top-right corner
- **Step reward**: -1 (encourages shortest path)
- **Goal reward**: +10

## Data

The dataset contains 200 episodes with varying path qualities:
- 50 optimal trajectories (always move toward goal)
- 100 mixed trajectories (70% optimal, 30% random actions)
- 50 random trajectories

### Format

Each line is a trajectory (episode) as a JSON array:
```json
[{"x":0,"y":0,"action":"right","reward":-1},{"x":1,"y":0,"action":"up","reward":-1},{"x":1,"y":1,"action":"right","reward":-1},{"x":2,"y":1,"action":"right","reward":-1},{"x":3,"y":1,"action":"up","reward":-1},{"x":3,"y":2,"action":"up","reward":10}]
```

Fields:
- `x`, `y`: Grid position (0-3)
- `action`: Movement direction (up, down, left, right)
- `reward`: -1 for steps, +10 for reaching goal

## Running

Basic value estimation:
```bash
./target/release/mcrl-rs --dataset examples/gridworld/trajectories.jsonl \
    --state-field x --state-field y \
    --reward-field reward \
    --discount 0.95
```

With confidence intervals:
```bash
./target/release/mcrl-rs --dataset examples/gridworld/trajectories.jsonl \
    --state-field x --state-field y \
    --reward-field reward \
    --discount 0.95 \
    --ci 95 \
    --bootstrap-samples 2000
```

## Expected Results

With discount factor 0.95, you should observe:
- States closer to the goal (3,3) have higher V(S) values
- V(3,3) is not shown (terminal state)
- V(3,2) and V(2,3) should be around 8-9 (one step from goal)
- V(0,0) should be around 2-4 (many steps from goal, discounting reduces value)

The discount factor effect:
- Higher discount (0.99): Future rewards matter more, values closer to undiscounted returns
- Lower discount (0.9): Future rewards matter less, stronger gradient from start to goal

## Generator Script

The `generate_data.py` script creates GridWorld navigation trajectories with configurable parameters.

### Usage

```bash
# Generate default dataset (200 episodes, seed=42)
python3 generate_data.py > trajectories.jsonl

# Custom seed and episode count
python3 generate_data.py --seed 123 --episodes 500 > custom.jsonl

# Write directly to file
python3 generate_data.py --output trajectories.jsonl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seed N` | 42 | Random seed for reproducibility |
| `--episodes N` | 200 | Number of episodes to generate |
| `--output FILE` | stdout | Output file path |

### How It Works

1. **Policy mix**: Generates 50 optimal, 100 mixed (70% optimal/30% random), and 50 random trajectories
2. **Action selection**: Optimal policy moves toward goal (right or up); random selects uniformly
3. **Grid boundaries**: Actions that would exit the grid keep the agent in place
4. **Termination**: Episodes end when goal (3,3) is reached or after 50 steps

### Requirements

- Python 3.6+
- No external dependencies (uses only `json`, `random`, `argparse`)

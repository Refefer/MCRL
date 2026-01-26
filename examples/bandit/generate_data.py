#!/usr/bin/env python3
"""
Multi-Armed Bandit Data Generator

Generates pull data for 5 slot machines with different payout distributions.
Each trajectory is a single pull.

Usage:
    python3 generate_data.py > pulls.jsonl
    python3 generate_data.py --seed 123 --pulls 10000
"""

import argparse
import json
import random

# Machine configurations
# Each machine has a different distribution to demonstrate mean vs median tradeoffs
MACHINES = {
    "machine_1": {
        "type": "normal",
        "mean": 1.50,
        "std": 0.30,
        "description": "Safe machine - low variance normal distribution"
    },
    "machine_2": {
        "type": "spike",
        "zero_prob": 0.90,
        "spike_value": 12.00,
        "description": "High variance - 90% $0, 10% $12 (mean $1.20)"
    },
    "machine_3": {
        "type": "normal",
        "mean": 0.80,
        "std": 0.25,
        "description": "Trap machine - consistent but low payout"
    },
    "machine_4": {
        "type": "normal",
        "mean": 1.40,
        "std": 0.50,
        "description": "Medium variance normal distribution"
    },
    "machine_5": {
        "type": "spike",
        "zero_prob": 0.80,
        "spike_value": 8.00,
        "description": "Bonus machine - 80% $0, 20% $8 (mean $1.60)"
    },
}


def generate_pull(machine_name):
    """Generate a single pull for a machine."""
    config = MACHINES[machine_name]

    if config["type"] == "normal":
        reward = random.gauss(config["mean"], config["std"])
        reward = max(0, reward)  # No negative payouts
    elif config["type"] == "spike":
        if random.random() < config["zero_prob"]:
            reward = 0.0
        else:
            # Add small variance to spike value
            reward = config["spike_value"] * random.uniform(0.9, 1.1)
    else:
        reward = 0.0

    return round(reward, 2)


def main():
    parser = argparse.ArgumentParser(description="Generate bandit pull data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pulls", type=int, default=5000, help="Total number of pulls")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    random.seed(args.seed)

    pulls = []
    machines = list(MACHINES.keys())

    # Equal pulls per machine
    pulls_per_machine = args.pulls // len(machines)
    extra = args.pulls - (pulls_per_machine * len(machines))

    for machine in machines:
        count = pulls_per_machine
        if extra > 0:
            count += 1
            extra -= 1

        for _ in range(count):
            reward = generate_pull(machine)
            # Each pull is a single-step trajectory
            pulls.append([{"machine": machine, "reward": reward}])

    # Shuffle
    random.shuffle(pulls)

    # Output
    output = args.output
    if output:
        with open(output, 'w') as f:
            for pull in pulls:
                f.write(json.dumps(pull) + "\n")
    else:
        for pull in pulls:
            print(json.dumps(pull))


if __name__ == "__main__":
    main()

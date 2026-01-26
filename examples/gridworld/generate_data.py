#!/usr/bin/env python3
"""
GridWorld Navigation Data Generator

Generates trajectories for a 4x4 grid navigation task.
Agent starts at (0,0) and navigates to goal at (3,3).
Step reward: -1, Goal reward: +10

Usage:
    python3 generate_data.py > trajectories.jsonl
    python3 generate_data.py --seed 123 --episodes 500
"""

import argparse
import json
import random

GRID_SIZE = 4
GOAL = (3, 3)
STEP_REWARD = -1
GOAL_REWARD = 10
MAX_STEPS = 50

ACTIONS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def apply_action(x, y, action):
    dx, dy = ACTIONS[action]
    new_x = clamp(x + dx, 0, GRID_SIZE - 1)
    new_y = clamp(y + dy, 0, GRID_SIZE - 1)
    return new_x, new_y


def optimal_action(x, y):
    """Return an action that moves toward the goal."""
    if x < GOAL[0]:
        return "right"
    elif y < GOAL[1]:
        return "up"
    else:
        return random.choice(["right", "up"])


def generate_trajectory(policy="mixed"):
    """
    Generate a single trajectory.

    Policies:
    - "optimal": Always move toward goal
    - "random": Random actions
    - "mixed": 70% optimal, 30% random
    """
    trajectory = []
    x, y = 0, 0

    for step in range(MAX_STEPS):
        if (x, y) == GOAL:
            break

        # Choose action based on policy
        if policy == "optimal":
            action = optimal_action(x, y)
        elif policy == "random":
            action = random.choice(list(ACTIONS.keys()))
        else:  # mixed
            if random.random() < 0.7:
                action = optimal_action(x, y)
            else:
                action = random.choice(list(ACTIONS.keys()))

        # Apply action
        new_x, new_y = apply_action(x, y, action)

        # Determine reward
        if (new_x, new_y) == GOAL:
            reward = GOAL_REWARD
        else:
            reward = STEP_REWARD

        trajectory.append({
            "x": x,
            "y": y,
            "action": action,
            "reward": reward
        })

        x, y = new_x, new_y

    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate GridWorld trajectories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate mix of trajectory types
    trajectories = []

    # 50 optimal trajectories
    for _ in range(50):
        trajectories.append(generate_trajectory("optimal"))

    # 100 mixed trajectories
    for _ in range(100):
        trajectories.append(generate_trajectory("mixed"))

    # 50 random trajectories
    for _ in range(50):
        trajectories.append(generate_trajectory("random"))

    # Shuffle
    random.shuffle(trajectories)

    # Trim to requested count
    trajectories = trajectories[:args.episodes]

    # Output
    output = args.output
    if output:
        with open(output, 'w') as f:
            for traj in trajectories:
                f.write(json.dumps(traj) + "\n")
    else:
        for traj in trajectories:
            print(json.dumps(traj))


if __name__ == "__main__":
    main()

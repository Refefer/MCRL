#!/bin/bash
# GridWorld: Value estimation with confidence intervals
# Run from project root: ./examples/gridworld/run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

"$PROJECT_ROOT/target/release/mcrl-rs" \
    --dataset "$SCRIPT_DIR/trajectories.jsonl" \
    --state-field x --state-field y \
    --reward-field reward \
    --discount 0.95 \
    --ci 95 \
    --bootstrap-samples 2000

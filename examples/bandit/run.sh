#!/bin/bash
# Multi-Armed Bandit: Mean vs Median estimator comparison
# Run from project root: ./examples/bandit/run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Mean Estimator (Expected Value) ==="
echo "Best by expected value: machine_5 > machine_1 > machine_4 > machine_2 > machine_3"
echo ""

"$PROJECT_ROOT/target/release/mcrl-rs" \
    --dataset "$SCRIPT_DIR/pulls.jsonl" \
    --state-field machine \
    --reward-field reward \
    --discount 1.0 \
    --comparison-test 5000 \
    --estimator mean \
    --bootstrap-samples 5000

echo ""
echo "=== Median Estimator (Typical Outcome) ==="
echo "Best by typical outcome: machine_1 > machine_4 > machine_3 > machine_2 = machine_5"
echo ""

"$PROJECT_ROOT/target/release/mcrl-rs" \
    --dataset "$SCRIPT_DIR/pulls.jsonl" \
    --state-field machine \
    --reward-field reward \
    --discount 1.0 \
    --comparison-test 5000 \
    --estimator median \
    --bootstrap-samples 5000

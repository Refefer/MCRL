#!/bin/bash
# A/B Test: Statistical comparison by segment
# Run from project root: ./examples/ab_test/run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

"$PROJECT_ROOT/target/release/mcrl-rs" \
    --dataset "$SCRIPT_DIR/sessions.jsonl" \
    --state-field page \
    --state-field segment \
    --state-field variant \
    --reward-field purchase-price \
    --discount 1.0 \
    --comparison-test 5000 \
    --comparison-group-fields 0,1 \
    --min-observations 50

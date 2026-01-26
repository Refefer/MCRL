#!/usr/bin/env python3
"""
A/B Test Session Data Generator

Generates user session data for e-commerce A/B testing.
Tests checkout variants (A, B) across user segments.
Each trajectory represents a user session from landing to checkout/abandonment.

Usage:
    python3 generate_data.py > sessions.jsonl
    python3 generate_data.py --seed 123 --sessions 2000
"""

import argparse
import json
import random

# Segments and their base characteristics
SEGMENTS = ["new_visitor", "returning", "premium"]

# Conversion rates by segment and variant
# Variant B is slightly better for new_visitor and premium, worse for returning
CONVERSION_RATES = {
    "new_visitor": {"A": 0.08, "B": 0.10},
    "returning": {"A": 0.15, "B": 0.14},
    "premium": {"A": 0.25, "B": 0.28},
}

# Average order value by segment
AOV = {
    "new_visitor": 45.0,
    "returning": 65.0,
    "premium": 120.0,
}

# Page flow probabilities (probability of continuing to next page)
PAGE_FLOW = {
    "landing": {"product": 0.6, "abandon": 0.4},
    "product": {"cart": 0.5, "abandon": 0.5},
    "cart": {"checkout": 0.7, "abandon": 0.3},
}


def generate_order_value(segment):
    """Generate order value with some variance."""
    base = AOV[segment]
    variance = base * 0.3
    return round(random.gauss(base, variance), 2)


def generate_session(segment, variant):
    """Generate a single user session trajectory."""
    trajectory = []

    # Landing page
    trajectory.append({
        "segment": segment,
        "variant": variant,
        "page": "landing",
        "purchase-price": 0
    })

    # Simulate page flow
    current_page = "landing"
    pages_visited = ["landing"]

    while current_page != "checkout" and current_page in PAGE_FLOW:
        flow = PAGE_FLOW[current_page]

        # Determine next page
        if random.random() < flow.get("abandon", 0):
            # User abandons - session ends
            break

        # Move to next page
        next_pages = [p for p in flow.keys() if p != "abandon"]
        if next_pages:
            current_page = next_pages[0]
            pages_visited.append(current_page)

            trajectory.append({
                "segment": segment,
                "variant": variant,
                "page": current_page,
                "purchase-price": 0
            })

    # If reached checkout, determine conversion
    if current_page == "checkout":
        conversion_rate = CONVERSION_RATES[segment][variant]
        if random.random() < conversion_rate / PAGE_FLOW["cart"]["checkout"]:
            # Successful conversion
            order_value = generate_order_value(segment)
            trajectory[-1]["purchase-price"] = order_value

    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate A/B test session data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sessions", type=int, default=1000, help="Number of sessions")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    random.seed(args.seed)

    sessions = []

    # Generate sessions with balanced segment/variant distribution
    sessions_per_combo = args.sessions // (len(SEGMENTS) * 2)
    extra = args.sessions - (sessions_per_combo * len(SEGMENTS) * 2)

    for segment in SEGMENTS:
        for variant in ["A", "B"]:
            count = sessions_per_combo
            # Distribute extra sessions
            if extra > 0:
                count += 1
                extra -= 1

            for _ in range(count):
                sessions.append(generate_session(segment, variant))

    # Shuffle
    random.shuffle(sessions)

    # Output
    output = args.output
    if output:
        with open(output, 'w') as f:
            for session in sessions:
                f.write(json.dumps(session) + "\n")
    else:
        for session in sessions:
            print(json.dumps(session))


if __name__ == "__main__":
    main()

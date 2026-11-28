#!/usr/bin/env python3
"""Check the safety/liveness tradeoff when a drain ACK is missing.

Two remote worlds can be observationally identical to decode at timeout:
the writer has drained but its ACK was lost, or the writer is still delayed.
Without a trustworthy fence or enforced write-latency bound, freeing is unsafe
in one world and holding forever prevents reclamation in the other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class World:
    name: str
    remote_write_pending: bool
    ack_observed: bool = False
    timeout_observed: bool = True


WORLDS = (
    World("drained_but_ack_lost", remote_write_pending=False),
    World("write_delayed_past_timeout", remote_write_pending=True),
)


def evaluate(policy: str, world: World) -> dict:
    """Evaluate the only two timeout decisions without new evidence."""
    if policy == "reuse_now":
        # A new lease can receive the old write if the writer is still pending.
        return {
            "world": world.name,
            "policy": policy,
            "safe": not world.remote_write_pending,
            "reclaimed": True,
            "foreign_write_possible": world.remote_write_pending,
        }
    if policy == "hold_forever":
        return {
            "world": world.name,
            "policy": policy,
            "safe": True,
            "reclaimed": False,
            "foreign_write_possible": False,
        }
    raise ValueError(f"unknown policy: {policy}")


def analyze() -> dict:
    policies = {name: [evaluate(name, world) for world in WORLDS]
                for name in ("reuse_now", "hold_forever")}
    all_world_safety_and_reclamation = {
        name: all(row["safe"] and row["reclaimed"] for row in rows)
        for name, rows in policies.items()
    }
    return {
        "observation_at_timeout": "no drain ACK; deadline elapsed",
        "worlds_are_indistinguishable_to_receiver": True,
        "policies": policies,
        "policy_satisfies_safety_and_reclamation_in_both_worlds":
            all_world_safety_and_reclamation,
        "no_timeout_only_policy_has_both_guarantees":
            not any(all_world_safety_and_reclamation.values()),
        "required_extra_mechanism": [
            "positive transport drain/fence proof",
            "or an enforced hard upper bound on every posted write",
            "or indefinite quarantine (safety without guaranteed reclamation)",
        ],
        "scope": "two-world impossibility check; it assumes the receiver has no signal distinguishing the worlds",
    }


def main() -> None:
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()

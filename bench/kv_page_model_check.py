#!/usr/bin/env python3
"""Bounded model checker for KV-page ownership and async-transfer races.

This is a control-plane model, not an inference-engine or CUDA benchmark. It
enumerates legal event interleavings for two minimal failure traces:
1. a late asynchronous write arriving after a physical page has been reused;
2. a rejected/short prefill being reported as complete, so decode reads old KV.

The checker contrasts unsafe behavior with two narrowly scoped guards:
page-lease validation at attention-read time and admission/transfer-completeness
validation before decode. Its safety claim is bounded to these event models.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable


LATE_WRITER_EVENTS = (
    "alloc_A",
    "write_A",
    "post_late_A",
    "abort_A",
    "release_A",
    "alloc_B",
    "write_B",
    "verify_B",
    "late_A",
    "decode_B",
)

LATE_WRITER_PRECEDENCE = (
    ("alloc_A", "write_A"),
    ("write_A", "post_late_A"),
    ("post_late_A", "abort_A"),
    ("abort_A", "release_A"),
    ("release_A", "alloc_B"),
    ("alloc_B", "write_B"),
    ("write_B", "verify_B"),
    ("abort_A", "late_A"),
    ("alloc_B", "late_A"),
    ("verify_B", "decode_B"),
)

REJECTED_PREFILL_EVENTS = (
    "alloc_A",
    "write_A",
    "release_A",
    "alloc_B",
    "reject_B",
    "placeholder_transfer_B",
    "report_complete_B",
    "decode_B",
)

REJECTED_PREFILL_PRECEDENCE = (
    ("alloc_A", "write_A"),
    ("write_A", "release_A"),
    ("release_A", "alloc_B"),
    ("alloc_B", "reject_B"),
    ("reject_B", "placeholder_transfer_B"),
    ("placeholder_transfer_B", "report_complete_B"),
    ("report_complete_B", "decode_B"),
)


@dataclass(frozen=True)
class Outcome:
    schedule: tuple[str, ...]
    read_owner: str | None
    decision: str

    @property
    def leaked(self) -> bool:
        return self.decision == "read_foreign_kv"

    @property
    def fail_closed(self) -> bool:
        return self.decision == "reject_before_read"


def schedules(events: tuple[str, ...], precedence: tuple[tuple[str, str], ...],
              extra_precedence: tuple[tuple[str, str], ...] = ()) -> Iterable[tuple[str, ...]]:
    """Yield every topological ordering of the event DAG."""
    edges = precedence + extra_precedence
    predecessors = {e: frozenset(a for a, b in edges if b == e) for e in events}

    def visit(done: tuple[str, ...], remaining: frozenset[str]):
        if not remaining:
            yield done
            return
        for event in events:
            if event in remaining and predecessors[event].issubset(done):
                yield from visit(done + (event,), remaining - {event})

    yield from visit((), frozenset(events))


def run_late_writer(schedule: tuple[str, ...], *, read_guard: bool) -> Outcome:
    active_owner: str | None = None
    content_owner: str | None = None
    transfer_verified = False
    for event in schedule:
        if event == "alloc_A":
            active_owner = "A"
        elif event == "write_A" or event == "late_A":
            content_owner = "A"
        elif event == "release_A":
            if active_owner == "A":
                active_owner = None
        elif event == "alloc_B":
            active_owner = "B"
        elif event == "write_B":
            content_owner = "B"
        elif event == "verify_B":
            transfer_verified = active_owner == "B" and content_owner == "B"
        elif event == "decode_B":
            if not transfer_verified:
                return Outcome(schedule, content_owner, "reject_before_read")
            if content_owner == "B" and active_owner == "B":
                return Outcome(schedule, content_owner, "read_own_kv")
            if read_guard:
                return Outcome(schedule, content_owner, "reject_before_read")
            return Outcome(schedule, content_owner, "read_foreign_kv")
    raise AssertionError("schedule did not reach decode_B")


def run_rejected_prefill(schedule: tuple[str, ...], *, strict_admission: bool) -> Outcome:
    active_owner: str | None = None
    content_owner: str | None = None
    rejected = False
    expected_tokens = 2
    transferred_tokens = 0
    complete = False
    for event in schedule:
        if event == "alloc_A":
            active_owner = "A"
        elif event == "write_A":
            content_owner = "A"
        elif event == "release_A":
            if active_owner == "A":
                active_owner = None
        elif event == "alloc_B":
            active_owner = "B"
        elif event == "reject_B":
            rejected = True
        elif event == "placeholder_transfer_B":
            # The error path stages only a placeholder, not B's full KV pages.
            transferred_tokens = 1
        elif event == "report_complete_B":
            # The unsafe protocol trusts the completion flag alone.
            complete = True
        elif event == "decode_B":
            if strict_admission and (rejected or transferred_tokens != expected_tokens):
                return Outcome(schedule, content_owner, "reject_before_read")
            if not complete:
                return Outcome(schedule, content_owner, "reject_before_read")
            if content_owner == "B" and active_owner == "B":
                return Outcome(schedule, content_owner, "read_own_kv")
            return Outcome(schedule, content_owner, "read_foreign_kv")
    raise AssertionError("schedule did not reach decode_B")


def analyze() -> dict:
    late = list(schedules(LATE_WRITER_EVENTS, LATE_WRITER_PRECEDENCE))
    # In the unsafe model late_A is explicitly allowed to arrive after B has
    # acquired the page. The quarantine policy removes that ordering: A's
    # remote write must drain before release, and B cannot allocate earlier.
    quarantine_precedence = tuple(
        edge for edge in LATE_WRITER_PRECEDENCE if edge != ("alloc_B", "late_A")
    )
    late_safe = list(schedules(
        LATE_WRITER_EVENTS, quarantine_precedence,
        extra_precedence=(("late_A", "release_A"),),
    ))
    rejected = list(schedules(REJECTED_PREFILL_EVENTS, REJECTED_PREFILL_PRECEDENCE))

    late_unsafe = [run_late_writer(s, read_guard=False) for s in late]
    late_guarded = [run_late_writer(s, read_guard=True) for s in late]
    late_quarantined = [run_late_writer(s, read_guard=False) for s in late_safe]
    rejected_unsafe = [run_rejected_prefill(s, strict_admission=False) for s in rejected]
    rejected_guarded = [run_rejected_prefill(s, strict_admission=True) for s in rejected]

    def summarize(outcomes: list[Outcome]) -> dict:
        counts = Counter(x.decision for x in outcomes)
        first_leak = next((x for x in outcomes if x.leaked), None)
        first_rejection = next((x for x in outcomes if x.fail_closed), None)
        return {
            "interleavings": len(outcomes),
            "decision_counts": dict(sorted(counts.items())),
            "counterexample": list(first_leak.schedule) if first_leak else None,
            "first_fail_closed_trace": list(first_rejection.schedule) if first_rejection else None,
        }

    return {
        "study": "bounded KV page ownership and transfer-lifetime model",
        "scope": "one physical page, two request leases, enumerated event orders; not a real inference-engine test",
        "late_writer_race": {
            "naive_early_reuse": summarize(late_unsafe),
            "attention_read_lease_guard": summarize(late_guarded),
            "defer_reuse_until_transfer_drained": summarize(late_quarantined),
        },
        "rejected_or_short_prefill": {
            "trust_completion_flag_only": summarize(rejected_unsafe),
            "require_admission_and_exact_transfer_length": summarize(rejected_guarded),
        },
        "properties": {
            "read_guard_blocks_every_foreign_page_read": not any(x.leaked for x in late_guarded),
            "quarantine_blocks_every_foreign_page_read": not any(x.leaked for x in late_quarantined),
            "strict_admission_blocks_every_rejected_prefill_read": not any(x.leaked for x in rejected_guarded),
            "unsafe_late_writer_has_counterexample": any(x.leaked for x in late_unsafe),
            "unsafe_rejected_prefill_has_counterexample": any(x.leaked for x in rejected_unsafe),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze()
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()

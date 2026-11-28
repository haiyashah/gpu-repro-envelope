#!/usr/bin/env python3
"""Explore finite KV-transfer protocol state spaces under async event orderings.

The model captures a single physical page and two request generations. It
explores event orderings rather than testing a real SGLang/vLLM deployment.
It is intended to turn incident classes into executable safety properties.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class State:
    owner: str = "A"
    content: str = "A"
    pending_a_write: bool = True
    a_aborted: bool = False
    a_ack: bool = False
    b_allocated: bool = False
    b_payload: str = "none"  # none, complete, placeholder
    b_complete: bool = False
    decision: str = "none"  # none, own_read, foreign_read, rejected


@dataclass(frozen=True)
class AckState:
    """State for a reused room ID whose old generation can still ack."""
    owner: str = "A"
    content: str = "A"
    current_write_pending: bool = True
    current_ack_received: bool = False
    released: bool = False
    b_allocated: bool = False
    b_written: bool = False
    b_verified: bool = False
    decision: str = "none"


@dataclass(frozen=True)
class TimeoutState:
    owner: str = "A"
    content: str = "A"
    pending_a_write: bool = True
    aborted: bool = False
    acked: bool = False
    timed_out: bool = False
    b_allocated: bool = False
    b_written: bool = False
    b_verified: bool = False
    decision: str = "none"  # none, own_read, foreign_read, quarantined


def explore_timeout_policy(*, release_on_timeout: bool) -> dict:
    """Model the current device-timeout fallback vs fail-closed quarantine."""
    initial = TimeoutState()
    queue = deque([(initial, ())])
    seen = {initial}
    terminal = {"own_read": 0, "foreign_read": 0, "quarantined": 0}
    counterexample = None
    while queue:
        state, trace = queue.popleft()
        if state.decision != "none":
            terminal[state.decision] += 1
            if state.decision == "foreign_read" and counterexample is None:
                counterexample = list(trace)
            continue
        out: list[tuple[str, TimeoutState]] = []
        if not state.aborted:
            out.append(("abort_A", replace(state, aborted=True)))
        if state.pending_a_write and state.aborted:
            out.append(("late_write_A_completes", replace(
                state, pending_a_write=False, content="A")))
        if not state.pending_a_write and not state.acked:
            out.append(("drain_ack_A", replace(state, acked=True)))
        if state.aborted and not state.acked and not state.timed_out:
            if release_on_timeout:
                out.append(("timeout_releases_page_without_drain_proof", replace(
                    state, owner="free", timed_out=True)))
            else:
                out.append(("timeout_quarantines_page", replace(
                    state, owner="quarantined", timed_out=True, decision="quarantined")))
        if state.aborted and state.acked and state.owner == "A":
            out.append(("release_after_positive_drain_ack", replace(state, owner="free")))
        if state.owner == "free" and not state.b_allocated:
            out.append(("allocate_B", replace(state, owner="B", b_allocated=True)))
        if state.b_allocated and not state.b_written:
            out.append(("write_B", replace(state, content="B", b_written=True)))
        if state.b_written and not state.b_verified:
            out.append(("verify_B", replace(state, b_verified=True)))
        if state.pending_a_write and state.b_allocated:
            out.append(("late_write_A_after_reuse", replace(
                state, pending_a_write=False, content="A")))
        if state.b_verified:
            if state.owner == "B" and state.content == "B":
                out.append(("decode_reads_own_page", replace(state, decision="own_read")))
            elif state.owner == "B":
                out.append(("decode_reads_foreign_page", replace(state, decision="foreign_read")))
        for event, nxt in out:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trace + (event,)))
    return {
        "reachable_states": len(seen),
        "terminal_decisions": terminal,
        "counterexample": counterexample,
        "foreign_read_possible": counterexample is not None,
    }


def explore_stale_ack(*, generation_check: bool) -> dict:
    initial = AckState()
    queue = deque([(initial, ())])
    seen = {initial}
    terminal = {"own_read": 0, "foreign_read": 0, "rejected": 0}
    counterexample = None
    while queue:
        state, trace = queue.popleft()
        if state.decision != "none":
            terminal[state.decision] += 1
            if state.decision == "foreign_read" and counterexample is None:
                counterexample = list(trace)
            continue
        out: list[tuple[str, AckState]] = []
        if state.current_write_pending:
            out.append(("current_write_drains", replace(
                state, current_write_pending=False, content="A")))
        # This ack is for generation 1 but the current transfer is generation 2.
        # An untagged room ID accepts it; a generation-aware receiver discards it.
        if not generation_check and not state.current_ack_received:
            out.append(("stale_ack_for_reused_room_id", replace(
                state, current_ack_received=True)))
        if not state.current_write_pending and not state.current_ack_received:
            out.append(("current_generation_drain_ack", replace(
                state, current_ack_received=True)))
        if state.owner == "A" and state.current_ack_received and not state.released:
            out.append(("release_A_after_ack", replace(state, owner="free", released=True)))
        if state.owner == "free" and not state.b_allocated:
            out.append(("allocate_B", replace(state, owner="B", b_allocated=True)))
        if state.b_allocated and not state.b_written:
            out.append(("write_B", replace(state, content="B", b_written=True)))
        if state.b_written and not state.b_verified:
            out.append(("verify_B", replace(state, b_verified=True)))
        if state.current_write_pending and state.b_allocated:
            out.append(("late_write_A_after_reuse", replace(
                state, current_write_pending=False, content="A")))
        if state.b_verified:
            if state.owner == "B" and state.content == "B":
                out.append(("decode_reads_own_page", replace(state, decision="own_read")))
            elif state.owner == "B":
                out.append(("decode_reads_foreign_page", replace(state, decision="foreign_read")))
            else:
                out.append(("decode_fails_closed", replace(state, decision="rejected")))
        for event, nxt in out:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trace + (event,)))
    return {
        "reachable_states": len(seen),
        "terminal_decisions": terminal,
        "counterexample": counterexample,
        "foreign_read_possible": counterexample is not None,
    }


def transitions(s: State, *, defer_release: bool, read_guard: bool,
                exact_transfer_check: bool,
                allow_placeholder: bool) -> list[tuple[str, State]]:
    out: list[tuple[str, State]] = []
    if not s.a_aborted:
        out.append(("abort_A", replace(s, a_aborted=True)))
    if s.pending_a_write and s.a_aborted:
        out.append(("late_write_A_completes", replace(s, pending_a_write=False, content="A")))
    if not s.pending_a_write and not s.a_ack:
        out.append(("drain_ack_A", replace(s, a_ack=True)))
    if s.a_aborted and s.owner == "A" and (not defer_release or s.a_ack):
        out.append(("release_A", replace(s, owner="free")))
    if s.owner == "free" and not s.b_allocated:
        out.append(("allocate_B", replace(s, owner="B", b_allocated=True)))
    if s.b_allocated and s.b_payload == "none":
        out.append(("full_transfer_B", replace(s, b_payload="complete", content="B")))
        if allow_placeholder:
            out.append(("rejected_placeholder_B", replace(s, b_payload="placeholder")))
    if s.b_allocated and s.b_payload != "none" and not s.b_complete:
        out.append(("transfer_complete_B", replace(s, b_complete=True)))
    if s.b_allocated and s.b_complete and s.decision == "none":
        if exact_transfer_check and s.b_payload != "complete":
            out.append(("decode_rejects_incomplete_transfer", replace(s, decision="rejected")))
        elif read_guard and (s.owner != "B" or s.content != "B"):
            out.append(("decode_guard_rejects_wrong_lease", replace(s, decision="rejected")))
        elif s.owner == "B" and s.content == "B":
            out.append(("decode_reads_own_page", replace(s, decision="own_read")))
        else:
            out.append(("decode_reads_foreign_page", replace(s, decision="foreign_read")))
    return out


def explore(*, defer_release: bool, read_guard: bool,
            exact_transfer_check: bool,
            allow_placeholder: bool = True) -> dict:
    initial = State()
    queue = deque([(initial, ())])
    seen = {initial}
    terminal = {"own_read": 0, "foreign_read": 0, "rejected": 0}
    counterexample = None
    max_depth = 0
    while queue:
        state, trace = queue.popleft()
        max_depth = max(max_depth, len(trace))
        if state.decision != "none":
            terminal[state.decision] += 1
            if state.decision == "foreign_read" and counterexample is None:
                counterexample = list(trace)
            continue
        for event, nxt in transitions(
            state,
            defer_release=defer_release,
            read_guard=read_guard,
            exact_transfer_check=exact_transfer_check,
            allow_placeholder=allow_placeholder,
        ):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trace + (event,)))
    return {
        "reachable_states": len(seen),
        "terminal_decisions": terminal,
        "counterexample": counterexample,
        "max_trace_length": max_depth,
        "foreign_read_possible": counterexample is not None,
    }


def analyze() -> dict:
    scenarios = {
        "unsafe_baseline": explore(
            defer_release=False, read_guard=False, exact_transfer_check=False),
        "deferred_release_only": explore(
            defer_release=True, read_guard=False, exact_transfer_check=False,
            allow_placeholder=False),
        "read_guard_only": explore(
            defer_release=False, read_guard=True, exact_transfer_check=False),
        "strict_transfer_and_read_guard": explore(
            defer_release=False, read_guard=True, exact_transfer_check=True),
        "combined_defenses": explore(
            defer_release=True, read_guard=True, exact_transfer_check=True,
            allow_placeholder=True),
        "untagged_stale_ack": explore_stale_ack(generation_check=False),
        "generation_tagged_ack": explore_stale_ack(generation_check=True),
        "device_timeout_releases_without_drain_proof": explore_timeout_policy(
            release_on_timeout=True),
        "device_timeout_quarantines_until_drain": explore_timeout_policy(
            release_on_timeout=False),
    }
    properties = {
        "unsafe_baseline_has_counterexample": scenarios["unsafe_baseline"]["foreign_read_possible"],
        "deferred_release_blocks_late_writer_leak": not scenarios["deferred_release_only"]["foreign_read_possible"],
        "read_guard_fails_closed_on_foreign_content": not scenarios["read_guard_only"]["foreign_read_possible"],
        "strict_transfer_check_fails_closed_on_placeholder": not scenarios["strict_transfer_and_read_guard"]["foreign_read_possible"],
        "combined_defenses_block_all_modeled_leaks": not scenarios["combined_defenses"]["foreign_read_possible"],
        "stale_room_ack_can_release_live_page": scenarios["untagged_stale_ack"]["foreign_read_possible"],
        "generation_tag_rejects_stale_ack": not scenarios["generation_tagged_ack"]["foreign_read_possible"],
        "release_on_timeout_has_late_write_counterexample": scenarios[
            "device_timeout_releases_without_drain_proof"]["foreign_read_possible"],
        "quarantine_on_timeout_fails_closed": not scenarios[
            "device_timeout_quarantines_until_drain"]["foreign_read_possible"],
    }
    return {
        "model": "single physical KV page, two request generations, async A write, abort, optional short B handoff",
        "bounded_model_only": True,
        "scenarios": scenarios,
        "properties": properties,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

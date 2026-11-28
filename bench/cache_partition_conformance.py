"""Portable behavioral checks for KV-cache partitioning boundaries.

This is an adapter-facing reference harness, not a vLLM integration. A serving
adapter records whether a request reused a prefix at an entrypoint/cache tier;
the auditor requires both positive and negative controls for every declared
dimension. It deliberately checks observed cache behavior, never internal hash
tuple shape.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


KEY_DOMAIN = b"kv-cache-partition/v1\0"


class ConformanceError(AssertionError):
    pass


def partitioned_key(
    *,
    model_fingerprint: str,
    prefix_block_ids: Iterable[str],
    dimensions: Mapping[str, str],
    required_dimensions: Iterable[str],
) -> str:
    """Build a canonical reference key with typed, domain-separated fields.

    Real systems may use a different encoding. Conformance tests should assert
    the behavior of the cache, not equality to this encoding.
    """
    required = sorted(set(required_dimensions))
    absent = [name for name in required if not dimensions.get(name)]
    if absent:
        raise ValueError(f"missing required cache identity fields: {absent}")
    identity = [[name, dimensions[name]] for name in required]
    payload = json.dumps(
        [model_fingerprint, list(prefix_block_ids), identity],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(KEY_DOMAIN + payload).hexdigest()


@dataclass(frozen=True)
class Probe:
    """One behavioral probe against a cache path.

    relation is ``same`` for a request with the same identity as the warm-up,
    or ``different`` when exactly one required partition dimension changes.
    """

    entrypoint: str
    tier: str
    dimension: str
    relation: str
    cache_hit: bool
    warm_identity: tuple[tuple[str, str], ...]
    probe_identity: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Coverage:
    entrypoints: tuple[str, ...]
    tiers: tuple[str, ...]
    dimensions: tuple[str, ...]
    discovered_connectors: tuple[str, ...] = ()
    covered_connectors: tuple[str, ...] = ()
    exempt_connectors: tuple[tuple[str, str], ...] = ()


def audit(probes: Iterable[Probe], coverage: Coverage) -> dict:
    """Fail if a declared path lacks a paired hit/miss or connector coverage."""
    probes = list(probes)
    observations: dict[tuple[str, str, str], dict[str, bool]] = {}
    errors: list[str] = []
    for probe in probes:
        if probe.relation not in ("same", "different"):
            errors.append(f"invalid relation {probe.relation!r}")
            continue
        path = (probe.entrypoint, probe.tier, probe.dimension)
        warm = dict(probe.warm_identity)
        candidate = dict(probe.probe_identity)
        changed = {
            name for name in set(warm) | set(candidate)
            if warm.get(name) != candidate.get(name)
        }
        expected_changes = set() if probe.relation == "same" else {probe.dimension}
        if changed != expected_changes:
            errors.append(
                f"{probe.relation} control for {path} changes {sorted(changed)}, "
                f"expected {sorted(expected_changes)}"
            )
        by_relation = observations.setdefault(path, {})
        if probe.relation in by_relation:
            errors.append(f"duplicate {probe.relation} control for {path}")
        by_relation[probe.relation] = probe.cache_hit

    expected = {
        (entrypoint, tier, dimension)
        for entrypoint in coverage.entrypoints
        for tier in coverage.tiers
        for dimension in coverage.dimensions
    }
    for path in sorted(set(observations) - expected):
        errors.append(f"probe is outside declared coverage: {path}")
    for path in sorted(expected):
        controls = observations.get(path, {})
        if set(controls) != {"same", "different"}:
            errors.append(f"missing paired controls for {path}")
            continue
        if not controls["same"]:
            errors.append(f"positive control did not hit for {path}")
        if controls["different"]:
            errors.append(f"isolation violation: different identity hit for {path}")

    exempt = {name: reason for name, reason in coverage.exempt_connectors}
    for connector in coverage.discovered_connectors:
        if connector not in coverage.covered_connectors and not exempt.get(connector):
            errors.append(f"uncovered registered connector: {connector}")
    for connector, reason in coverage.exempt_connectors:
        if not reason.strip():
            errors.append(f"connector exemption lacks a reason: {connector}")

    result = {
        "passed": not errors,
        "probe_count": len(probes),
        "expected_paths": len(expected),
        "observed_paths": len(observations),
        "errors": errors,
    }
    if errors:
        raise ConformanceError("\n".join(errors))
    return result


def audit_document(document: Mapping) -> dict:
    """Validate a JSON-compatible adapter report."""
    raw_coverage = document["coverage"]
    coverage = Coverage(
        entrypoints=tuple(raw_coverage["entrypoints"]),
        tiers=tuple(raw_coverage["tiers"]),
        dimensions=tuple(raw_coverage["dimensions"]),
        discovered_connectors=tuple(raw_coverage.get("discovered_connectors", ())),
        covered_connectors=tuple(raw_coverage.get("covered_connectors", ())),
        exempt_connectors=tuple(
            tuple(item) for item in raw_coverage.get("exempt_connectors", ())
        ),
    )
    probes = [
        Probe(
            entrypoint=item["entrypoint"],
            tier=item["tier"],
            dimension=item["dimension"],
            relation=item["relation"],
            cache_hit=item["cache_hit"],
            warm_identity=tuple(tuple(pair) for pair in item["warm_identity"]),
            probe_identity=tuple(tuple(pair) for pair in item["probe_identity"]),
        )
        for item in document["probes"]
    ]
    return audit(probes, coverage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report emitted by an engine adapter")
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as handle:
        result = audit_document(json.load(handle))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

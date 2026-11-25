"""The `licence-validate` subcommand: inspect and enforce the resource policy.

A thin adapter over licence/production_gate. It exists so the policy can be
interrogated without writing a script, and so a deployment can gate a release
on its exit code rather than on someone remembering to look.

Exit codes are the contract:
    0  every requested resource is permitted in the requested mode
    1  at least one resource is refused
    2  the policy itself could not be loaded

Canonical ticket: P00_02_T050.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

DEFAULT_POLICY = "config/licence_policy.yaml"


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--policy", default=DEFAULT_POLICY, help="path to the licence policy YAML"
    )
    parser.add_argument(
        "--mode",
        choices=("production", "research", "open_noncommercial_research"),
        default="production",
        help=(
            "production refuses anything not explicitly permitted, and refuses "
            "unknowns; open_noncommercial_research additionally admits the "
            "written-confirmation records (NRLMSIS 2.x)"
        ),
    )
    parser.add_argument(
        "--resource", action="append", default=None,
        help="resource to check; repeatable. Omit to check every declared dependency",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from ..licence.policy_loader import load_licence_policy
    from ..licence.production_gate import (
        ProductionGateError,
        check_all_declared_dependencies,
        check_resource,
    )

    policy = load_licence_policy(args.policy)

    if args.resource:
        results: list[dict[str, Any]] = []
        refused = 0
        for name in args.resource:
            try:
                check_resource(name, policy, mode=args.mode)
                results.append({"resource": name, "permitted": True, "reason": ""})
            except ProductionGateError as error:
                refused += 1
                results.append(
                    {"resource": name, "permitted": False, "reason": str(error)}
                )
        payload: dict[str, Any] = {
            "mode": args.mode,
            "policy": args.policy,
            "checked": len(results),
            "refused": refused,
            "results": results,
        }
        return payload, (1 if refused else 0)

    audit = check_all_declared_dependencies(policy)
    unrecorded = list(audit.unrecorded)
    # not_permitted is (name, reason) pairs; keep the reason, it is the useful part.
    refused_deps = [f"{name}: {reason}" for name, reason in audit.not_permitted]
    payload = {
        "mode": args.mode,
        "policy": args.policy,
        "policy_version": audit.policy_version,
        "declared_dependencies": list(audit.declared),
        "recorded": list(audit.recorded),
        "unrecorded": unrecorded,
        "refused": refused_deps,
    }
    return payload, (1 if (unrecorded or refused_deps) else 0)


def _render_text(payload: dict[str, Any], status: int) -> str:
    lines = [f"licence policy {payload['policy']}  mode={payload['mode']}"]
    if "results" in payload:
        lines.append(
            f"  checked {payload['checked']}, refused {payload['refused']}"
        )
        for entry in payload["results"]:
            mark = "PERMIT" if entry["permitted"] else "REFUSE"
            lines.append(f"  [{mark}] {entry['resource']}")
            if entry["reason"]:
                for line in str(entry["reason"]).splitlines():
                    lines.append(f"           {line}")
    else:
        declared = payload.get("declared_dependencies", [])
        lines.append(f"  declared runtime dependencies: {len(declared)}")
        if payload["unrecorded"]:
            lines.append("  UNRECORDED (declared but absent from the manifest):")
            for name in payload["unrecorded"]:
                lines.append(f"    {name}")
        if payload["refused"]:
            lines.append("  REFUSED in this mode:")
            for name in payload["refused"]:
                lines.append(f"    {name}")
        if not payload["unrecorded"] and not payload["refused"]:
            lines.append("  every declared dependency is recorded and permitted")
    lines.append(f"  exit {status}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    try:
        payload, status = _check(args)
    except FileNotFoundError as error:
        print(f"policy not found: {error}")
        return 2
    except (ValueError, KeyError) as error:
        print(f"policy could not be loaded: {error}")
        return 2

    if args.format == "json":
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(_render_text(payload, status))
    return status

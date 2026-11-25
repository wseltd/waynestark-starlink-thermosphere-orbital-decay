"""Thin CLI adapter.

Contains no scientific logic: every subcommand resolves arguments, calls a
library function and formats the result. The same library functions are called
by the future API, by notebooks and by batch workflows, so behaviour cannot
diverge between entry points.

Canonical ticket: P00_02_T048.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .. import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermosphere-consequence",
        description=(
            "Species-resolved thermosphere, free-molecular drag and "
            "orbital-consequence engine. The thermosphere and its molecular "
            "diagnostics are resolved by species; the drag force is the "
            "effective scalar 0.5*C_d*rho_total*A*v_rel^2, whose C_d is "
            "species mass-flux weighted."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    constants = sub.add_parser(
        "constants",
        help="print the versioned physical constants with their provenance",
    )
    constants.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="output format (default: text)",
    )

    species = sub.add_parser(
        "species", help="print the neutral species registry with particle masses"
    )
    species.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="output format (default: text)",
    )

    # The reproduce subcommand declares its own arguments, so the pipeline and
    # its options stay in one module and this parser does not become the second
    # place that has to be edited when a stage is added.
    from .reproduce import add_arguments as _reproduce_arguments

    _reproduce_arguments(
        sub.add_parser(
            "reproduce",
            help="run one clean end-to-end reproduction and record how it went",
        )
    )

    from .commands_licence import add_arguments as _licence_arguments
    from .commands_physics import add_arguments as _physics_arguments

    _physics_arguments(
        sub.add_parser(
            "physics-diagnostics",
            help="gas-kinetic and aerodynamic state at one point, with provenance",
        )
    )
    _licence_arguments(
        sub.add_parser(
            "licence-validate",
            help="check resources against the licence policy; non-zero exit on refusal",
        )
    )
    return parser


def _cmd_constants(fmt: str) -> str:
    from ..domain.constants import CONSTANTS

    if fmt == "json":
        return json.dumps(
            {
                symbol: {
                    "value": c.value, "unit": c.unit, "source": c.source,
                    "source_url": c.source_url, "revision": c.revision,
                    "exact": c.exact,
                }
                for symbol, c in CONSTANTS.items()
            },
            indent=2,
        )
    lines = [f"{'symbol':<8} {'value':>18}  {'unit':<26} source"]
    for symbol, c in CONSTANTS.items():
        marker = " (exact)" if c.exact else ""
        lines.append(f"{symbol:<8} {c.value:>18.10g}  {c.unit:<26} {c.source}{marker}")
    return "\n".join(lines)


def _cmd_species(fmt: str) -> str:
    from ..domain.species import ALL_SPECIES

    if fmt == "json":
        return json.dumps(
            {
                s.symbol: {
                    "name": s.name, "mass_number": s.mass_number,
                    "mass_kg": s.mass_kg, "note": s.note,
                }
                for s in ALL_SPECIES.values()
            },
            indent=2,
        )
    lines = [f"{'symbol':<12} {'A':>5} {'mass (kg)':>16}  name"]
    for s in ALL_SPECIES.values():
        lines.append(f"{s.symbol:<12} {s.mass_number:>5g} {s.mass_kg:>16.6e}  {s.name}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "constants":
        print(_cmd_constants(args.format))
    elif args.command == "physics-diagnostics":
        from .commands_physics import run as _run_physics

        return _run_physics(args)
    elif args.command == "licence-validate":
        from .commands_licence import run as _run_licence

        return _run_licence(args)
    elif args.command == "species":
        print(_cmd_species(args.format))
    elif args.command == "reproduce":
        import sys

        from .reproduce import execute

        # The run record has to name what actually ran, so the effective
        # arguments are resolved here rather than left as None when the process
        # was started from the command line.
        return execute(args, argv=list(argv) if argv is not None else sys.argv[1:])
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

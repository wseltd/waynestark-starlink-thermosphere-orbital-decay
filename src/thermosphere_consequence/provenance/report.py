"""Run-level provenance: the record that decides whether a result is believable.

A result file answers "how many days until reentry?". This module answers the
question that determines whether anyone should act on that answer: which code,
which library versions, which input bytes, which models, which parameters, and
under which stated assumptions and caveats the number was produced.

Three deliberate positions:

  * The licence table is data held here, not something inferred at runtime.
    ``importlib.metadata`` can report what a package *claims* about its licence,
    and those claims are inconsistent, sometimes absent and occasionally wrong.
    The licences below come from this project's own dependency audit, so the
    bill of materials states a reviewed position rather than re-deriving an
    unreviewed one on every run.

  * A dependency that cannot be resolved is recorded as ``NOT_INSTALLED``, never
    omitted and never defaulted to a plausible version. Dropping the key would
    make the record read as though the dependency was never required, which is
    precisely the failure this module exists to prevent.

  * ``git_commit`` is ``None`` when the commit cannot be determined, which is
    the truthful record for an installed wheel or an exported source tree. It is
    read from the ``.git`` directory rather than by invoking ``git``: a
    provenance record must not depend on a subprocess that may be missing, may
    hang, or may inherit an unexpected environment.

Nearest canonical tickets: P09_10_T050 (reproducibility manifest and software
licence sections of the case-study report).
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any

from .. import __version__
from ..domain.temporal import ensure_utc, format_utc
from ..drivers.raw_store import RawAssetMetadata

__all__ = [
    "NOT_INSTALLED",
    "RUNTIME_DEPENDENCY_LICENCES",
    "RUNTIME_DEPENDENCIES",
    "RunProvenance",
    "build_run_provenance",
    "collect_dependency_versions",
    "detect_git_commit",
    "render_provenance_markdown",
    "software_bill_of_materials",
    "write_provenance",
]

#: Recorded in place of a version when a declared dependency is not importable.
#: A distinguishable sentinel, not a version string, so no consumer can mistake
#: it for one.
NOT_INSTALLED = "not installed"

#: The audited runtime dependency set, in the order the bill of materials is
#: emitted. Order is pinned so that two SBOMs from the same environment are
#: byte-identical and a diff between environments shows only real changes.
#:
#: The licence strings are the outcome of the project's dependency review.
#: Two entries carry a qualification that a bare SPDX identifier would lose:
#: pymsis is MIT-licensed *code* wrapping models with their own terms, and
#: matplotlib uses its own PSF-derived licence rather than any SPDX-listed one.
RUNTIME_DEPENDENCY_LICENCES: tuple[tuple[str, str], ...] = (
    ("numpy", "BSD-3-Clause"),
    ("scipy", "BSD-3-Clause"),
    ("pint", "BSD-3-Clause"),
    ("pydantic", "MIT"),
    ("PyYAML", "MIT"),
    ("pymsis", "MIT (wrapper); NRLMSISE-00 model: no licence terms located"),
    ("sgp4", "MIT"),
    ("pandas", "BSD-3-Clause"),
    ("xarray", "Apache-2.0"),
    ("netCDF4", "MIT"),
    ("pyarrow", "Apache-2.0"),
    ("trimesh", "MIT"),
    ("SALib", "MIT"),
    ("matplotlib", "PSF-based (matplotlib licence)"),
)

#: Distribution names only, for callers that need the set rather than the table.
RUNTIME_DEPENDENCIES: tuple[str, ...] = tuple(
    name for name, _licence in RUNTIME_DEPENDENCY_LICENCES
)

# A git object name is 40 hex characters under SHA-1 and 64 under SHA-256.
_GIT_OBJECT_NAME_LENGTHS = (40, 64)
_HEX_DIGITS = frozenset("0123456789abcdef")


def collect_dependency_versions() -> dict[str, str]:
    """Resolve the installed version of every audited runtime dependency.

    The returned key set always equals :data:`RUNTIME_DEPENDENCIES`, so the
    record's shape is an invariant of the declaration rather than a function of
    what happens to be installed on the machine that ran the analysis. A
    dependency that is absent is reported as :data:`NOT_INSTALLED`.
    """
    versions: dict[str, str] = {}
    for name in RUNTIME_DEPENDENCIES:
        try:
            versions[name] = _distribution_version(name)
        except PackageNotFoundError:
            versions[name] = NOT_INSTALLED
    return versions


def _sbom_from_versions(versions: Mapping[str, str]) -> list[dict[str, str]]:
    """Join the audited licence table to a set of resolved versions.

    Single construction point for every bill of materials in the system, so a
    licence correction cannot land in the report but miss the JSON manifest.
    """
    return [
        {
            "name": name,
            "version": versions.get(name, NOT_INSTALLED),
            "licence": licence,
        }
        for name, licence in RUNTIME_DEPENDENCY_LICENCES
    ]


def software_bill_of_materials() -> list[dict[str, str]]:
    """Name, version and audited licence for every runtime dependency.

    Reflects the live environment. To describe the environment a *recorded* run
    used, read the ``software_bill_of_materials`` key of that run's serialised
    :class:`RunProvenance` instead: it is built from the versions captured at
    the time, not from whatever is installed when the report is rendered.
    """
    return _sbom_from_versions(collect_dependency_versions())


def _resolve_git_dir(start: pathlib.Path) -> pathlib.Path | None:
    """Find the git directory governing ``start``, or None.

    Handles the ``.git``-as-a-file form used by worktrees and submodules, where
    the real directory lives elsewhere and is named by a ``gitdir:`` pointer.
    """
    for candidate in (start, *start.parents):
        dot_git = candidate / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            pointer = dot_git.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                target = pathlib.Path(pointer.split(":", 1)[1].strip())
                resolved = target if target.is_absolute() else candidate / target
                return resolved if resolved.is_dir() else None
            return None
    return None


def _looks_like_object_name(text: str) -> bool:
    return len(text) in _GIT_OBJECT_NAME_LENGTHS and set(text) <= _HEX_DIGITS


def _read_packed_ref(git_dir: pathlib.Path, ref: str) -> str | None:
    """Resolve a ref from ``packed-refs``, which is where refs live after gc."""
    packed = git_dir / "packed-refs"
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref and _looks_like_object_name(parts[0]):
            return parts[0]
    return None


def detect_git_commit(start: pathlib.Path | str | None = None) -> str | None:
    """Return the checked-out commit, or None when it cannot be determined.

    Reads ``.git`` directly rather than shelling out: this must work inside a
    build sandbox with no git binary, and a provenance record should never be
    able to hang on a subprocess. ``None`` is a meaningful answer — an installed
    wheel has no commit — and is recorded as such rather than guessed.
    """
    origin = pathlib.Path(start) if start is not None else pathlib.Path(__file__).parent
    git_dir = _resolve_git_dir(origin.resolve())
    if git_dir is None:
        return None

    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()

    if not head.startswith("ref:"):
        return head.lower() if _looks_like_object_name(head.lower()) else None

    ref = head.split(":", 1)[1].strip()
    loose = git_dir / pathlib.Path(ref)
    if loose.is_file():
        candidate = loose.read_text(encoding="utf-8").strip().lower()
        return candidate if _looks_like_object_name(candidate) else None
    packed = _read_packed_ref(git_dir, ref)
    return packed.lower() if packed else None


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Everything needed to reproduce a run and to judge its output.

    Invariants enforced at construction:

      * ``run_id``, ``code_version`` and ``python_version`` are non-empty — a
        record that cannot name the code that produced it is not a record;
      * ``created_utc`` is timezone-aware and normalised to UTC (naive input
        raises, consistent with the rest of the system: UTC is never assumed);
      * every entry of ``input_assets`` is a :class:`RawAssetMetadata`, so an
        input can always be traced to stored bytes and their hash;
      * the mutable containers are copied in, so a caller that later mutates the
        list or dict it passed cannot retroactively alter a "frozen" record.

    ``assumptions`` and ``caveats`` are separate fields on purpose. An
    assumption is an input the analysis chose in the absence of data; a caveat
    is a limitation of the result that survives however good the inputs were.
    Collapsing them into one list loses the distinction a reader needs.
    """

    run_id: str
    created_utc: _dt.datetime
    code_version: str
    git_commit: str | None
    python_version: str
    dependency_versions: dict[str, str]
    input_assets: list[RawAssetMetadata]
    model_names: tuple[str, ...]
    parameters: dict[str, Any]
    assumptions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("run_id", "code_version", "python_version"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        object.__setattr__(self, "created_utc", ensure_utc(self.created_utc, "created_utc"))

        if self.git_commit is not None and not str(self.git_commit).strip():
            raise ValueError(
                "git_commit must be a commit identifier or None; an empty "
                "string would claim a commit was recorded when none was"
            )

        for index, asset in enumerate(self.input_assets):
            if not isinstance(asset, RawAssetMetadata):
                raise TypeError(
                    f"input_assets[{index}]: expected RawAssetMetadata, got "
                    f"{type(asset).__name__}. Inputs must trace to stored bytes."
                )
        for name in self.model_names:
            if not str(name).strip():
                raise ValueError("model_names entries must be non-empty strings")

        object.__setattr__(self, "dependency_versions", dict(self.dependency_versions))
        object.__setattr__(self, "input_assets", list(self.input_assets))
        object.__setattr__(self, "model_names", tuple(self.model_names))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "caveats", tuple(self.caveats))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, including the bill of materials as recorded.

        The bill of materials is derived from ``dependency_versions`` rather
        than from the live environment, so the serialised record stays
        internally consistent no matter when it is written out.
        """
        return {
            "run_id": self.run_id,
            "created_utc": format_utc(self.created_utc, "created_utc"),
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "dependency_versions": dict(self.dependency_versions),
            "input_assets": [asdict(asset) for asset in self.input_assets],
            "model_names": list(self.model_names),
            "parameters": dict(self.parameters),
            "assumptions": list(self.assumptions),
            "caveats": list(self.caveats),
            "software_bill_of_materials": _sbom_from_versions(self.dependency_versions),
        }


def build_run_provenance(
    *,
    run_id: str,
    model_names: Sequence[str],
    parameters: Mapping[str, Any] | None = None,
    input_assets: Sequence[RawAssetMetadata] = (),
    assumptions: Sequence[str] = (),
    caveats: Sequence[str] = (),
    created_utc: _dt.datetime | None = None,
    code_version: str | None = None,
    git_commit: str | None = None,
    repo_root: pathlib.Path | str | None = None,
) -> RunProvenance:
    """Assemble a :class:`RunProvenance`, filling in what the environment knows.

    Keyword-only because these arguments are not distinguishable positionally
    and a silently transposed pair would corrupt the record without failing.

    The environment-derived fields — dependency versions, interpreter version,
    commit and creation time — are captured here rather than left to the caller
    so that every run records them the same way. Each can still be supplied
    explicitly, which is what makes a run reproducible from an older manifest.
    """
    return RunProvenance(
        run_id=run_id,
        created_utc=created_utc if created_utc is not None else _dt.datetime.now(_dt.UTC),
        code_version=code_version if code_version is not None else __version__,
        git_commit=git_commit if git_commit is not None else detect_git_commit(repo_root),
        python_version=platform.python_version(),
        dependency_versions=collect_dependency_versions(),
        input_assets=list(input_assets),
        model_names=tuple(model_names),
        parameters=dict(parameters or {}),
        assumptions=tuple(assumptions),
        caveats=tuple(caveats),
    )


def write_provenance(prov: RunProvenance, path: pathlib.Path | str) -> pathlib.Path:
    """Write the record as indented JSON and return the path written.

    Indented rather than compact because this file is read by people and
    reviewed in diffs; the size cost is irrelevant beside a Parquet trajectory.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(prov.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_provenance_markdown(prov: RunProvenance) -> str:
    """Render the record as a self-contained markdown provenance section.

    Deterministic: parameters are sorted by key and every other sequence keeps
    its recorded order, so re-rendering an unchanged run produces byte-identical
    text and a diff between two reports shows only substantive differences.

    Empty sections are rendered as an explicit statement of emptiness rather
    than omitted. A reader must be able to tell "no assumptions were recorded"
    from "the assumptions section was left out".
    """
    lines: list[str] = ["## Provenance", ""]
    lines.extend(
        [
            f"- Run ID: `{prov.run_id}`",
            f"- Created (UTC): {format_utc(prov.created_utc, 'created_utc')}",
            f"- Code version: {prov.code_version}",
            "- Git commit: "
            + (f"`{prov.git_commit}`" if prov.git_commit else "unknown (not a git checkout)"),
            f"- Python: {prov.python_version}",
            "",
        ]
    )

    lines.append("### Models")
    lines.append("")
    lines.extend([f"- {name}" for name in prov.model_names] or ["_No model recorded._"])
    lines.append("")

    lines.append("### Input assets")
    lines.append("")
    if prov.input_assets:
        lines.extend(
            _markdown_table(
                ("asset", "source", "retrieved (UTC)", "sha256", "licence"),
                [
                    (
                        asset.asset_id,
                        asset.source,
                        asset.retrieved_at_utc,
                        asset.sha256,
                        asset.licence,
                    )
                    for asset in prov.input_assets
                ],
            )
        )
    else:
        lines.append("_No stored input assets: this run consumed no external data._")
    lines.append("")

    lines.append("### Parameters")
    lines.append("")
    if prov.parameters:
        lines.extend(f"- {key}: {prov.parameters[key]}" for key in sorted(prov.parameters))
    else:
        lines.append("_No parameters recorded._")
    lines.append("")

    lines.append("### Software bill of materials")
    lines.append("")
    lines.extend(
        _markdown_table(
            ("package", "version", "licence"),
            [
                (entry["name"], entry["version"], entry["licence"])
                for entry in _sbom_from_versions(prov.dependency_versions)
            ],
        )
    )
    lines.append("")

    lines.append("### Assumptions")
    lines.append("")
    lines.extend(
        [f"{n}. {text}" for n, text in enumerate(prov.assumptions, start=1)]
        or ["_No assumptions recorded._"]
    )
    lines.append("")

    lines.append("### Caveats")
    lines.append("")
    lines.extend([f"- {text}" for text in prov.caveats] or ["_No caveats recorded._"])
    lines.append("")

    return "\n".join(lines)

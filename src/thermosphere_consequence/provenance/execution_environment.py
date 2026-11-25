"""What code actually ran, recorded at the start of the process that runs it.

WHY THIS EXISTS. ``.venv`` in this worktree is a symlink to a sibling
worktree's environment, whose editable install points at that sibling's
``src``. Every test run and every research script in this checkout was
therefore importing another branch's source. It went unnoticed for the whole
project because the two branch heads carried byte-identical source trees, so
the results were right by coincidence rather than by construction; the first
edit that made them differ was also the first time anything noticed.

A recorded HEAD commit does not identify the code that ran. A recorded
``sys.executable`` does not either. What identifies it is the RESOLVED PATH OF
EACH MODULE THAT DID THE WORK, and that is what this module captures and
checks.

Two modes, because both are legitimate and they must not be confused:

  * **Source mode.** The package is expected to resolve inside this repository.
    This is how the research scripts and the test suite run, and
    :func:`require_source_tree` fails loudly when it does not hold.

  * **Distribution mode.** The package is expected to resolve inside an
    installed distribution of a named version. A wheel test must NOT be
    required to import from ``src/``; requiring it would be a different bug
    wearing this one's clothes. :func:`require_installed_distribution` checks
    the distribution's name and version instead.

Official third-party libraries stay outside the repository in both modes. They
are verified through ``third_party/nrlmsis_asset_manifest.yaml``, whose sha256
pins this module reports alongside everything else.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "KEY_MODULES",
    "ExecutionEnvironment",
    "ImportOriginError",
    "describe",
    "require_installed_distribution",
    "require_source_tree",
    "write_record",
]

#: The modules whose resolved paths decide whether a result means anything.
#: One per scientific concern: the atmosphere the densities came from, the
#: literature table the comparisons came from, the gas-surface interaction the
#: forces came from, and the propagation the lifetimes came from. If any of
#: these resolves outside the intended tree, the run described something else.
KEY_MODULES: tuple[str, ...] = (
    "thermosphere_consequence.providers.nrlmsise00",
    "thermosphere_consequence.validation.literature",
    "thermosphere_consequence.aerodynamics.kernels",
    "thermosphere_consequence.uq.propagation",
)

DISTRIBUTION_NAME = "thermosphere-consequence"


class ImportOriginError(RuntimeError):
    """Raised when the package under execution is not the one intended.

    A ``RuntimeError`` rather than an ``ImportError``: the import succeeded.
    What failed is the claim that it imported what the caller meant.
    """


@dataclass(frozen=True, slots=True)
class ExecutionEnvironment:
    """Everything needed to say which code produced a result."""

    python_executable: str
    python_version: str
    environment_prefix: str
    base_prefix: str
    repository_root: str
    git_commit: str
    git_branch: str
    git_tree_dirty: bool
    package_path: str
    package_inside_repository: bool
    distribution_name: str
    distribution_version: str
    module_paths: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, str] = field(default_factory=dict)
    official_asset_pins: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(repo_root: pathlib.Path, *args: str) -> str:
    """One git query, returning empty string rather than raising.

    A checkout without git, or an export from an archive, is a legitimate way
    to run this code. Failing the whole provenance record because the commit
    could not be read would make the record less likely to be written, which is
    the opposite of what it is for.
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _repository_root() -> pathlib.Path:
    """The repository this module's own file lives in."""
    return pathlib.Path(__file__).resolve().parents[3]


def _official_asset_pins(repo_root: pathlib.Path) -> dict[str, str]:
    """The sha256 pins for official libraries that live OUTSIDE the repository.

    Read straight from the public manifest rather than through the provider, so
    that recording the environment never requires the official package to be
    installed. A missing manifest is reported as such; it is not an error here.
    """
    manifest = repo_root / "third_party" / "nrlmsis_asset_manifest.yaml"
    if not manifest.is_file():
        return {"nrlmsis_asset_manifest": "absent"}
    pins: dict[str, str] = {}
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        for asset in loaded.get("assets", []) or []:
            logical_id = str(asset.get("logical_id", "")).strip()
            for item in asset.get("required_files", []) or []:
                name = str(item.get("name", "")).strip()
                digest = str(item.get("sha256", "")).strip().lower()
                if logical_id and name and digest:
                    pins[f"{logical_id}/{name}"] = digest
    except Exception as error:  # pragma: no cover - defensive
        return {"nrlmsis_asset_manifest": f"unreadable: {error}"}
    return pins or {"nrlmsis_asset_manifest": "no pins recorded"}


def describe(repo_root: pathlib.Path | None = None) -> ExecutionEnvironment:
    """Capture the running environment, resolving every key module by import."""
    root = (repo_root or _repository_root()).resolve()

    package = importlib.import_module("thermosphere_consequence")
    package_file = pathlib.Path(package.__file__ or "").resolve()

    module_paths: dict[str, str] = {}
    for name in KEY_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError as error:  # pragma: no cover - reported, not raised
            module_paths[name] = f"unimportable: {error}"
            continue
        module_paths[name] = str(pathlib.Path(module.__file__ or "").resolve())

    try:
        version = importlib.metadata.version(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"

    configuration = {
        "event_manifest": str(root / "config" / "events" / "starlink_group_4_7_2022.yaml"),
        "nrlmsis_asset_manifest": str(root / "third_party" / "nrlmsis_asset_manifest.yaml"),
        "pythonpath": ":".join(p for p in sys.path[:5]),
    }

    return ExecutionEnvironment(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        environment_prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        repository_root=str(root),
        git_commit=_git(root, "rev-parse", "HEAD"),
        git_branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        git_tree_dirty=bool(_git(root, "status", "--porcelain")),
        package_path=str(package_file),
        package_inside_repository=package_file.is_relative_to(root),
        distribution_name=DISTRIBUTION_NAME,
        distribution_version=version,
        module_paths=module_paths,
        configuration=configuration,
        official_asset_pins=_official_asset_pins(root),
    )


def require_source_tree(repo_root: pathlib.Path | None = None) -> ExecutionEnvironment:
    """Fail unless the package AND every key module resolve inside this repository.

    Checking the package alone is not enough: a namespace package, a stale
    ``.pth`` or a partially shadowed tree can put the top-level ``__init__``
    inside the repository while the module that does the physics comes from
    somewhere else entirely. Every module in :data:`KEY_MODULES` is resolved and
    checked by path.
    """
    root = (repo_root or _repository_root()).resolve()
    environment = describe(root)

    offenders: list[str] = []
    if not environment.package_inside_repository:
        offenders.append(f"thermosphere_consequence -> {environment.package_path}")
    for name, path in environment.module_paths.items():
        if path.startswith("unimportable:"):
            offenders.append(f"{name} -> {path}")
        elif not pathlib.Path(path).is_relative_to(root):
            offenders.append(f"{name} -> {path}")

    if offenders:
        raise ImportOriginError(
            "the code about to run does NOT come from this repository "
            f"({root}).\n  "
            + "\n  ".join(offenders)
            + "\n\nThis checkout's .venv is a symlink to a sibling worktree's "
            "environment, whose editable install points at that worktree's "
            "src/. Results produced this way describe the sibling's code, not "
            f"this branch's. Re-run with PYTHONPATH={root / 'src'}."
        )
    return environment


def require_installed_distribution(
    expected_version: str | None = None,
) -> ExecutionEnvironment:
    """Fail unless the package resolves inside an INSTALLED distribution.

    The counterpart check, for testing a built wheel. Such a test must not be
    required to import from ``src/`` - that would defeat its purpose - so the
    identity checked here is the distribution's name and version and the fact
    that the import resolves under the environment prefix rather than under a
    source tree.
    """
    environment = describe()
    package_path = pathlib.Path(environment.package_path)
    if environment.distribution_version == "not-installed":
        raise ImportOriginError(
            f"{DISTRIBUTION_NAME} is not installed in {environment.environment_prefix}; "
            "there is no distribution to test."
        )
    if not package_path.is_relative_to(pathlib.Path(environment.environment_prefix)):
        raise ImportOriginError(
            f"{DISTRIBUTION_NAME} resolves to {package_path}, which is outside the "
            f"environment prefix {environment.environment_prefix}. A wheel test "
            "must exercise the installed distribution, not a source tree that "
            "happens to shadow it."
        )
    if expected_version is not None and environment.distribution_version != expected_version:
        raise ImportOriginError(
            f"installed {DISTRIBUTION_NAME} is version "
            f"{environment.distribution_version}, expected {expected_version}"
        )
    return environment


def write_record(
    destination: pathlib.Path, *, label: str, repo_root: pathlib.Path | None = None
) -> ExecutionEnvironment:
    """Capture the environment, check it, and write it beside the run's outputs."""
    environment = require_source_tree(repo_root)
    payload = {"label": label, **environment.as_dict()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return environment

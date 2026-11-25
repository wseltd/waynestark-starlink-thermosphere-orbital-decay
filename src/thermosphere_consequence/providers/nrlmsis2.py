"""NRLMSIS 2.0 / 2.1 species-resolved providers over the separately installed official model.

Licence position, which shapes every design decision in this module: NRLMSIS
2.0 and 2.1 are US Naval Research Laboratory models under academic,
non-commercial terms (US Patent 10,641,925 recorded separately). NRL has
confirmed IN WRITING that this project's described non-commercial, openly
published research workflow is permitted under those terms — and only that
workflow. ``config/licence_policy.yaml`` records the four resources
(``nrlmsis_2_0``, ``nrlmsis_2_1``, ``msis20_parm``, ``msis21_parm``) with
policy decision ``research_noncommercial_open_publication_permitted``, a
decision admitted by the ``open_noncommercial_research`` mode ALONE. See
``evidence/licensing/NRLMSIS_2X_RESEARCH_PERMISSION_DECISION.md``.

Four consequences are enforced here rather than documented:

* **The official model never enters this repository.** Each user installs the
  official NRL package separately, outside the tree, and points the wrapper at
  it (composition reads ``NRLMSIS20_ROOT`` / ``NRLMSIS21_ROOT``; this module
  itself reads NO environment variable, so an environment variable alone can
  never override the gate). A root that resolves inside the repository is
  refused outright.
* **The gate is consulted at construction, not assumed.** The constructor
  requires ``mode == "open_noncommercial_research"`` and then clears BOTH the
  model resource and its coefficient-file resource through
  :func:`~thermosphere_consequence.licence.production_gate.check_resource`.
  The mode and the policy arrive from composition as arguments — never from
  the environment.
* **Every required file is hash-pinned.** ``msis20.parm`` and ``msis21.parm``
  are the same size and ``msisinit`` performs no content validation, so a
  swapped or corrupted coefficient file is physically loadable and would
  produce silently wrong densities. The sha256 pins in
  ``third_party/nrlmsis_asset_manifest.yaml`` are the ONLY guard, so they are
  verified at construction — every required file, not only the parm — and a
  mismatch is a refusal, not a warning.
* **The computation runs in the OFFICIAL, UNMODIFIED sources.** The compiled
  artefact is this project's original Apache-2.0 line-protocol driver
  (``providers/fortran/wse_msis2_driver.F90`` — USE statements and calls only,
  zero NRL code) linked against the unmodified official package by
  ``scripts/build_nrlmsis2_driver.sh``, into ``$ROOT/build/`` in the EXTERNAL
  installation. Python speaks lines to it over a pipe and applies no physics.

Model facts this module relies on, verified against the official
``msis_gtd8d.F90`` legacy wrapper (the mapping below is exactly the one it
applies, in reverse):

* ``msiscalc`` returns ``dn(1)`` total mass density in kg/m^3 and ``dn(2:10)``
  number densities in m^-3 ordered N2, O2, O, He, H, Ar, N, anomalous O, NO;
  ``tn`` is the temperature at altitude and ``tex`` the exospheric
  temperature, both in K.
* A missing species is the sentinel ``9.999e-38``, never zero. NRLMSIS 2.0
  does not model NO and reports it as the sentinel; NRLMSIS 2.1 reports a
  density. The two public classes therefore differ: the 2.0 provider drops NO
  unconditionally (absent, not zero), the 2.1 provider reports it.
* Storm-time behaviour needs ``switch_legacy(9) = -1.0`` at initialisation,
  after which the full 7-element ap history is consumed. Supplying a history
  without that switch would silently smear a storm into a daily mean, so the
  switch is selected by launching the driver in ``storm`` mode whenever
  ``drivers.ap_history`` is present, and daily mode is an EXPLICIT, noted
  fallback used only when no history was supplied.

Canonical decision record: evidence/licensing/NRLMSIS_2X_RESEARCH_PERMISSION_DECISION.md.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import pathlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Self

import yaml  # type: ignore[import-untyped]

from ..domain.temporal import ensure_utc
from ..licence.policy_loader import LicencePolicy
from ..licence.production_gate import check_resource
from ..protocols.atmosphere import AtmosphereState, DriverState

__all__ = [
    "MISSING_SENTINEL",
    "OPEN_NONCOMMERCIAL_RESEARCH_MODE",
    "WRAPPER_VERSION",
    "AssetManifestEntry",
    "NRLMSIS20Provider",
    "NRLMSIS21Provider",
    "Nrlmsis2AssetError",
    "Nrlmsis2DriverError",
    "Nrlmsis2LicenceModeError",
    "load_asset_manifest_entry",
]

#: The ONE licence-gate mode whose table lists
#: ``research_noncommercial_open_publication_permitted``. Pinned here as well
#: as in the policy file so that widening some other mode's table can never
#: quietly widen this provider: the constructor checks the mode by name AND
#: asks the gate, and both must agree.
OPEN_NONCOMMERCIAL_RESEARCH_MODE: Final = "open_noncommercial_research"

#: Version of this project's wrapper (driver protocol + this module), carried
#: into every state's provenance.
WRAPPER_VERSION: Final = "1.0.0"

#: The model's documented missing-value sentinel (``dmissing`` in the official
#: ``msis_constants``). A value is treated as missing inside an ABSOLUTE
#: window around it rather than by equality: the double-precision core emits
#: exactly 9.999e-38, but the legacy real(4) path rounds it to single
#: (9.99899994...e-38, about 6e-46 away), and both must read as missing.
MISSING_SENTINEL: Final = 9.999e-38
_SENTINEL_WINDOW: Final = 1.0e-41

#: ``msiscalc`` dn index (1-based, as in the Fortran) -> species symbol, in
#: the order VERIFIED against the official ``msis_gtd8d.F90`` legacy wrapper.
#: dn(1) is total mass density and is handled separately.
_MSISCALC_DN_SPECIES: Final[tuple[tuple[int, str], ...]] = (
    (2, "N2"),
    (3, "O2"),
    (4, "O"),
    (5, "He"),
    (6, "H"),
    (7, "Ar"),
    (8, "N"),
    (9, "anomalous_O"),
    (10, "NO"),
)

#: Fields on one result line: tn tex dn1..dn10.
_RESULT_FIELD_COUNT: Final = 12

#: Repository root, anchored on this file exactly as the composition root
#: anchors on its own: providers -> package -> src -> repository.
REPOSITORY_ROOT: Final = pathlib.Path(__file__).resolve().parents[3]

#: The PUBLIC sanitised manifest carrying the per-file sha256 pins.
ASSET_MANIFEST_PATH: Final = REPOSITORY_ROOT.joinpath("third_party", "nrlmsis_asset_manifest.yaml")

_DRIVER_RELATIVE_PATH: Final = pathlib.PurePosixPath("build/wse_msis2_driver")
_BUILD_SCRIPT: Final = "scripts/build_nrlmsis2_driver.sh"


class Nrlmsis2LicenceModeError(RuntimeError):
    """Raised when a caller asks for NRLMSIS 2.x outside the one permitted mode."""


class Nrlmsis2AssetError(RuntimeError):
    """Raised when the external official installation is absent, misplaced or unverified."""


class Nrlmsis2DriverError(RuntimeError):
    """Raised when the compiled driver subprocess misbehaves at the protocol level."""


@dataclass(frozen=True, slots=True)
class AssetManifestEntry:
    """One official package as pinned by the public asset manifest."""

    logical_id: str
    model_version: str
    authorised_source: str
    environment_variable: str
    required_files: Mapping[str, str]


def load_asset_manifest_entry(
    logical_id: str, manifest_path: pathlib.Path = ASSET_MANIFEST_PATH
) -> AssetManifestEntry:
    """Read one package's pins out of ``third_party/nrlmsis_asset_manifest.yaml``.

    Strict in the same spirit as the licence loaders: a missing block, an
    empty file list or a malformed entry is an error, because a manifest that
    cannot be read completely cannot pin anything.
    """
    if not manifest_path.is_file():
        raise Nrlmsis2AssetError(
            f"asset manifest not found at {manifest_path}; the sha256 pins are the only "
            f"guard against a swapped or corrupted official installation, so nothing can "
            f"be verified without it"
        )
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise Nrlmsis2AssetError(f"{manifest_path}: manifest must be a YAML mapping")
    assets = loaded.get("assets")
    if not isinstance(assets, Sequence) or isinstance(assets, str):
        raise Nrlmsis2AssetError(f"{manifest_path}: 'assets' must be a list")
    for entry in assets:
        if not isinstance(entry, Mapping) or entry.get("logical_id") != logical_id:
            continue
        raw_files = entry.get("required_files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, str) or not raw_files:
            raise Nrlmsis2AssetError(
                f"{manifest_path}: {logical_id} carries no required_files to verify"
            )
        required: dict[str, str] = {}
        for item in raw_files:
            if not isinstance(item, Mapping):
                raise Nrlmsis2AssetError(
                    f"{manifest_path}: {logical_id} has a malformed required_files entry"
                )
            name = str(item.get("name", "")).strip()
            digest = str(item.get("sha256", "")).strip().lower()
            if not name or len(digest) != 64:
                raise Nrlmsis2AssetError(
                    f"{manifest_path}: {logical_id} entry {item!r} lacks a name or a 64-hex sha256"
                )
            required[name] = digest
        return AssetManifestEntry(
            logical_id=logical_id,
            model_version=str(entry.get("model_version", "")),
            authorised_source=str(entry.get("authorised_source", "")),
            environment_variable=str(entry.get("environment_variable", "")),
            required_files=required,
        )
    raise Nrlmsis2AssetError(
        f"{manifest_path}: no asset with logical_id {logical_id!r} is recorded"
    )


class _DriverProcess:
    """One live driver subprocess, holding exactly one (binary, mode) pair.

    A thin protocol wrapper: write one request line, read one result line.
    The physics happens on the other side of the pipe, in the official
    sources; nothing here interprets the numbers.
    """

    def __init__(
        self, binary: pathlib.Path, parmpath: pathlib.Path, parmfile: str, mode: str
    ) -> None:
        self.mode = mode
        self._handle = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [str(binary), f"{parmpath}/", parmfile, mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(parmpath),
        )
        expected = f"WSE-MSIS2-DRIVER {mode} {parmfile}"
        greeting = self._read_line(context="handshake")
        if greeting != expected:
            self.close()
            raise Nrlmsis2DriverError(
                f"driver handshake mismatch: expected {expected!r}, got {greeting!r}. "
                f"The handshake asserts which mode and coefficient file were actually "
                f"launched; a mismatch means the wrong build or a failed initialisation."
            )

    def request(self, values: Sequence[float]) -> tuple[float, ...]:
        """Send one request line, return the result line as floats."""
        stdin = self._handle.stdin
        if stdin is None or self._handle.poll() is not None:
            raise Nrlmsis2DriverError("driver process is not running")
        line = " ".join(format(value, ".17g") for value in values)
        try:
            stdin.write(line + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise Nrlmsis2DriverError(f"driver rejected a request line: {exc}") from exc
        answer = self._read_line(context=f"result for request {line!r}")
        fields = answer.split()
        if len(fields) != _RESULT_FIELD_COUNT:
            raise Nrlmsis2DriverError(
                f"driver returned {len(fields)} fields, expected {_RESULT_FIELD_COUNT}: {answer!r}"
            )
        try:
            return tuple(float(field) for field in fields)
        except ValueError as exc:
            raise Nrlmsis2DriverError(f"driver returned a non-numeric field: {answer!r}") from exc

    def _read_line(self, *, context: str) -> str:
        stdout = self._handle.stdout
        if stdout is None:
            raise Nrlmsis2DriverError("driver stdout is not connected")
        line = stdout.readline()
        if line == "":
            stderr_text = ""
            if self._handle.stderr is not None:
                stderr_text = self._handle.stderr.read() or ""
            code = self._handle.poll()
            self.close()
            raise Nrlmsis2DriverError(
                f"driver exited while reading {context} (returncode {code}). "
                f"stderr: {stderr_text.strip() or '<empty>'}"
            )
        return line.rstrip("\n")

    def close(self) -> None:
        """Terminate the subprocess, politely first."""
        handle = self._handle
        for stream in (handle.stdin, handle.stdout, handle.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if handle.poll() is None:
            handle.terminate()
            try:
                handle.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                handle.kill()
                handle.wait(timeout=5.0)


class _Nrlmsis2ProviderBase:
    """Shared machinery of the two public providers.

    Private on purpose: NRLMSIS 2.0 and 2.1 are SEPARATE resources with
    separate licence records, separate coefficient files and different output
    (2.1 adds NO), so the public surface is two distinct classes with distinct
    names and versions — never an ambiguous "2.x" provider.
    """

    # Pinned per subclass. The parm file name is deliberately NOT a
    # constructor argument: making it configurable would re-open the silent
    # cross-loading hole the hash pin exists to close.
    _MODEL_RESOURCE = ""
    _PARM_RESOURCE = ""
    _PARM_FILE = ""
    _LOGICAL_ID = ""
    _MODEL_LABEL = ""
    _REPORTS_NO = False

    name = ""
    version = ""

    def __init__(
        self,
        *,
        policy: LicencePolicy,
        mode: str,
        root: pathlib.Path | str,
    ) -> None:
        # 1. The mode, by name. The gate below also refuses every other mode
        #    for these records today; this explicit pin means a future policy
        #    edit that listed the decision in some additional mode would STILL
        #    not widen this provider without touching this code.
        if mode != OPEN_NONCOMMERCIAL_RESEARCH_MODE:
            raise Nrlmsis2LicenceModeError(
                f"{self.name} may be constructed in mode "
                f"{OPEN_NONCOMMERCIAL_RESEARCH_MODE!r} only, not {mode!r}. The written "
                f"NRL permission covers the described open, non-commercial research "
                f"workflow and nothing else; every other mode fails closed. See "
                f"evidence/licensing/NRLMSIS_2X_RESEARCH_PERMISSION_DECISION.md."
            )

        # 2. The gate, for the model AND its coefficient file. They are
        #    separately licensed things with separate records; clearing one is
        #    not clearing the other.
        check_resource(self._MODEL_RESOURCE, policy, mode=mode)
        check_resource(self._PARM_RESOURCE, policy, mode=mode)
        self._mode = mode

        # 3. The external root: present, a directory, and OUTSIDE this
        #    repository. The licence forbids bringing the official material
        #    into any distributed artefact, and a root inside the tree is one
        #    `git add` away from exactly that.
        resolved = pathlib.Path(root).resolve()
        if not resolved.is_dir():
            raise Nrlmsis2AssetError(
                f"{self.name}: official installation root {resolved} does not exist or "
                f"is not a directory. Install the official package from "
                f"{self._manifest_entry_or_none_source()} outside the repository and "
                f"point {self._env_hint()} at it."
            )
        if resolved.is_relative_to(REPOSITORY_ROOT):
            raise Nrlmsis2AssetError(
                f"{self.name}: official installation root {resolved} lies INSIDE this "
                f"repository ({REPOSITORY_ROOT}). The NRL material must stay outside "
                f"every distributed artefact; install it elsewhere."
            )
        self.root = resolved

        # 4. Every required file, hash-verified against the public manifest.
        #    Verified once here and cached as this instance's verdict; the two
        #    parm files are the same size and msisinit does no content
        #    validation, so this pin is the only thing that makes a swapped
        #    file a refusal instead of a silently wrong model.
        entry = load_asset_manifest_entry(self._LOGICAL_ID)
        self._manifest_entry = entry
        mismatches: list[str] = []
        for file_name, expected_digest in entry.required_files.items():
            candidate = resolved / file_name
            if not candidate.is_file():
                mismatches.append(f"{file_name}: MISSING")
                continue
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != expected_digest:
                mismatches.append(
                    f"{file_name}: sha256 {actual} does not match the manifest pin "
                    f"{expected_digest}"
                )
        if mismatches:
            raise Nrlmsis2AssetError(
                f"{self.name}: official installation at {resolved} FAILED hash "
                f"verification against {ASSET_MANIFEST_PATH.name}: "
                f"{'; '.join(mismatches)}. Refusing to run — msisinit performs no "
                f"content validation, so an unverified file would load silently. "
                f"Re-install the official package from {entry.authorised_source}."
            )
        self._assets_verified = True
        self._parm_sha256 = entry.required_files[self._PARM_FILE]

        # 5. The compiled driver, built by this project's build script into
        #    the EXTERNAL root.
        binary = resolved / _DRIVER_RELATIVE_PATH
        if not binary.is_file():
            raise Nrlmsis2AssetError(
                f"{self.name}: compiled driver not found at {binary}. Run "
                f"`{_BUILD_SCRIPT} {self._MODEL_LABEL}` (with {self._env_hint()} set) to "
                f"build it against the verified official sources; the official package "
                f"itself is installed separately from the authorised source "
                f"{entry.authorised_source}."
            )
        self._binary = binary

        self.version = (
            f"NRLMSIS-{self._MODEL_LABEL}/{self._PARM_FILE}@{self._parm_sha256[:16]}"
            f"/wse-driver-{WRAPPER_VERSION}"
        )
        #: At most ONE live subprocess per provider; a mode change replaces it.
        self._process: _DriverProcess | None = None

    # -- provider protocol ---------------------------------------------------

    def state_at(
        self,
        time_utc: _dt.datetime,
        altitude_m: float,
        latitude_deg: float,
        longitude_deg: float,
        drivers: DriverState,
    ) -> AtmosphereState:
        """Return the neutral atmosphere at one point, in SI.

        Altitude arrives in METRES (protocol contract) and is converted to the
        model's kilometres here at the boundary. Day-of-year and UT seconds
        are derived from the aware UTC instant.
        """
        time_utc = ensure_utc(time_utc, "time_utc")
        if altitude_m < 0:
            raise ValueError(f"altitude must be non-negative, got {altitude_m}")
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError(f"latitude out of range: {latitude_deg}")

        if drivers.ap_history is not None:
            driver_mode = "storm"
            ap = tuple(float(value) for value in drivers.ap_history)
            mode_note = (
                "storm-time geomagnetic mode: switch_legacy(9) = -1.0, full 7-element "
                "ap history consumed"
            )
        else:
            # The explicit fallback, never a silent smear: without a history
            # the storm switch would read six undefined values, so the daily
            # path is selected and SAID. The February 2022 analysis always
            # supplies ap_history and never takes this branch.
            driver_mode = "daily"
            ap = (float(drivers.ap_daily), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            mode_note = (
                "daily geomagnetic mode used EXPLICITLY: no 7-element ap history was "
                "supplied, so ap(1) = daily Ap with default switches. Storm-time "
                "structure is NOT resolved in this state."
            )

        day = float(time_utc.timetuple().tm_yday)
        utsec = (
            time_utc.hour * 3600.0
            + time_utc.minute * 60.0
            + time_utc.second
            + time_utc.microsecond / 1e6
        )

        result = self._driver(driver_mode).request(
            (
                day,
                utsec,
                altitude_m / 1000.0,
                latitude_deg,
                longitude_deg,
                drivers.f107a,
                drivers.f107,
                *ap,
            )
        )
        tn, tex = result[0], result[1]
        dn = result[2:]

        if _is_missing(tn):
            raise ValueError(
                f"{self.name} returned the missing sentinel for temperature at "
                f"{altitude_m} m; inputs are outside the model's valid domain"
            )

        number_densities: dict[str, float] = {}
        for index, symbol in _MSISCALC_DN_SPECIES:
            if symbol == "NO" and not self._REPORTS_NO:
                # NRLMSIS 2.0 has no NO output. Its value here is the sentinel
                # anyway, but the drop is unconditional so that even a
                # numerically plausible value could never be reported by the
                # wrong model version.
                continue
            value = dn[index - 1]
            if _is_missing(value):
                continue
            number_densities[symbol] = value

        total = dn[0]
        return AtmosphereState(
            time_utc=time_utc,
            altitude_m=altitude_m,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            temperature_k=tn,
            number_densities=number_densities,
            total_mass_density=None if _is_missing(total) else total,
            exospheric_temperature_k=None if _is_missing(tex) else tex,
            drivers=drivers,
            provider_name=self.name,
            provider_version=f"{self.version}/{driver_mode}",
            notes=(
                mode_note,
                (
                    f"official NRLMSIS {self._MODEL_LABEL} executed unmodified from a "
                    f"separately installed, hash-verified external root; "
                    f"{self._PARM_FILE} sha256 {self._parm_sha256[:16]}; licence gate "
                    f"mode {self._mode!r}"
                ),
            ),
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Terminate the driver subprocess. Safe to call repeatedly."""
        if self._process is not None:
            self._process.close()
            self._process = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _driver(self, driver_mode: str) -> _DriverProcess:
        """The live subprocess for ``driver_mode``, launching or replacing as needed.

        One subprocess at a time per provider, by design: the geomagnetic
        switch is fixed at msisinit, so storm and daily are different
        initialisations, and holding both alive would be a pool.
        """
        process = self._process
        if process is not None and process.mode != driver_mode:
            self.close()
            process = None
        if process is None:
            process = _DriverProcess(self._binary, self.root, self._PARM_FILE, driver_mode)
            self._process = process
        return process

    def _env_hint(self) -> str:
        entry_var = self._manifest_entry_or_none_var()
        return entry_var or "the documented installation-root environment variable"

    def _manifest_entry_or_none_var(self) -> str:
        try:
            return load_asset_manifest_entry(self._LOGICAL_ID).environment_variable
        except Nrlmsis2AssetError:
            return ""

    def _manifest_entry_or_none_source(self) -> str:
        try:
            return load_asset_manifest_entry(self._LOGICAL_ID).authorised_source
        except Nrlmsis2AssetError:
            return "the authorised NRL source"


def _is_missing(value: float) -> bool:
    """True when ``value`` is the model's missing sentinel.

    An absolute window, not equality: the double core emits 9.999e-38 exactly,
    the legacy real(4) path emits its single rounding ~6e-46 away, and both
    are missing. The window (1e-41) is thousands of times wider than that
    rounding and thirty-plus orders of magnitude below any physical density
    this project consumes.
    """
    return abs(value - MISSING_SENTINEL) < _SENTINEL_WINDOW


class NRLMSIS20Provider(_Nrlmsis2ProviderBase):
    """NRLMSIS 2.0 (release 26 MAY 2020), official sources, external install.

    Reports NO as ABSENT — the 2.0 model does not produce it. Admitted in the
    ``open_noncommercial_research`` licence mode only.
    """

    _MODEL_RESOURCE = "nrlmsis_2_0"
    _PARM_RESOURCE = "msis20_parm"
    _PARM_FILE = "msis20.parm"
    _LOGICAL_ID = "nrlmsis_2_0_package"
    _MODEL_LABEL = "2.0"
    _REPORTS_NO = False

    name = "NRLMSIS 2.0"


class NRLMSIS21Provider(_Nrlmsis2ProviderBase):
    """NRLMSIS 2.1 (release 04 APR 2022), official sources, external install.

    Adds nitric oxide (NO) to the reported species. Admitted in the
    ``open_noncommercial_research`` licence mode only.
    """

    _MODEL_RESOURCE = "nrlmsis_2_1"
    _PARM_RESOURCE = "msis21_parm"
    _PARM_FILE = "msis21.parm"
    _LOGICAL_ID = "nrlmsis_2_1_package"
    _MODEL_LABEL = "2.1"
    _REPORTS_NO = True

    name = "NRLMSIS 2.1"

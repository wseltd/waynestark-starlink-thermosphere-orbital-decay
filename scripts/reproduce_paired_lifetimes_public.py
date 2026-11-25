#!/usr/bin/env python3
"""Versioned public adapter for R2 paired quiet-versus-storm lifetime reproduction (Items 3 & 5).

Preserves the frozen 08c8b86 ``scripts/reproduce_paired_lifetimes.py`` (one of the 109
SHA-256-pinned historical files in ``evidence/R2_PAIRED_LIFETIME_REPRODUCTION.json``) while
fixing two portability and provenance defects for external/non-Git execution:

1. Plain non-Git extracted archives are supported via explicit SHA-256 digest identity
   verification against the 109 frozen source hashes, without fabricating Git metadata and
   explicitly rejecting accidental capture by an unrelated enclosing Git repository.
2. Consumed ``RawStore`` driver hashes are extracted by iterating the ``list[RawAssetMetadata]``
   returned by ``RawStore.manifest()`` (fixing the legacy ``isinstance(..., dict)`` bug that
   emitted ``{"note": "see RawStore manifest"}``).

Modes:
  --check-archive-identity   Portable offline check (no NRL licence or raw drivers required).
  --check-raw-drivers        Verify consumed GFZ/CelesTrak driver assets in data/raw/ against
                             BOTH local RawStore sidecars AND frozen evidence/DATA_MANIFEST.json pins.
  --run                      Execute the full 6-case NRLMSIS 2.1 paired reproduction
                             (requires NRLMSIS21_ROOT built via: bash scripts/build_nrlmsis2_driver.sh 2.1).
  --allow-new-drivers        Opt-in flag permitting newly fetched rolling drivers that differ from
                             the frozen DATA_MANIFEST.json snapshot (explicitly labelled as a
                             new-input run, NOT an exact raw-input reproduction).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import types
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

STORED = REPO_ROOT / "results" / "species_resolved_gsi" / "storm_lifetime_comparison"
HISTORICAL_R2_RECORD = REPO_ROOT / "evidence" / "R2_PAIRED_LIFETIME_REPRODUCTION.json"

QUIET_AP = 4.0
DECAY_RELATIVE_TOLERANCE = 1e-9
PENALTY_ABSOLUTE_TOLERANCE_DAYS = 1e-6

CONSUMED_DRIVER_ASSET_IDS: tuple[tuple[str, str, str], ...] = (
    ("gfz", "kp_2022_extended", ".json"),
    ("gfz", "kp_2022_extended_ap", ".json"),
    ("celestrak", "sw_last5years", ".csv"),
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_strict(repo_root: pathlib.Path, *args: str) -> tuple[int, str]:
    """Run git strictly confined to repo_root (never ascending into parent directories)."""
    env = dict(os.environ)
    env["GIT_CEILING_DIRECTORIES"] = str(repo_root.resolve().parent)
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, proc.stdout.strip()


def _detect_unrelated_enclosing_git_repo(repo_root: pathlib.Path) -> str | None:
    """Detect whether an enclosing parent directory is an unrelated Git repository."""
    resolved = repo_root.resolve()
    try:
        proc = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    top = pathlib.Path(proc.stdout.strip()).resolve()
    if top != resolved:
        return str(top)
    return None


def resolve_archive_identity(repo_root: pathlib.Path = REPO_ROOT) -> dict[str, Any]:
    """Resolve authentic Git checkout OR digest-verified non-Git extracted archive identity.

    Never fabricates Git metadata and never adopts an unrelated enclosing Git repository.
    """
    resolved = repo_root.resolve()
    has_local_git = (resolved / ".git").exists()
    enclosing_unrelated = None if has_local_git else _detect_unrelated_enclosing_git_repo(resolved)

    git_checkout_present = False
    run_commit: str | None = None
    git_tree_dirty: bool | None = None
    if has_local_git:
        rc_top, top_str = _git_strict(resolved, "rev-parse", "--show-toplevel")
        if rc_top == 0 and top_str and pathlib.Path(top_str).resolve() == resolved:
            rc_head, head_sha = _git_strict(resolved, "rev-parse", "HEAD")
            rc_st, st_out = _git_strict(resolved, "status", "--porcelain")
            if rc_head == 0 and len(head_sha) == 40:
                git_checkout_present = True
                run_commit = head_sha
                git_tree_dirty = bool(st_out) if rc_st == 0 else None

    r2_path = resolved / "evidence" / "R2_PAIRED_LIFETIME_REPRODUCTION.json"
    if not r2_path.is_file():
        raise RuntimeError(f"Missing historical R2 digest baseline: {r2_path}")
    r2_doc = json.loads(r2_path.read_text(encoding="utf-8"))
    pinned: dict[str, str] = dict(r2_doc.get("executed_source_sha256") or {})
    if len(pinned) != 109:
        raise RuntimeError(f"Expected 109 pinned historical Python files, found {len(pinned)}")

    r5_path = resolved / "evidence" / "ROUND5_SOURCE_MANIFEST.json"
    r5_declared: dict[str, dict[str, Any]] = {}
    r5_expected_map: dict[str, str] = dict(pinned)
    r5_manifest_sha: str | None = None
    if r5_path.is_file():
        r5_manifest_sha = _sha256(r5_path)
        r5_doc = json.loads(r5_path.read_text(encoding="utf-8"))
        r5_declared = dict(r5_doc.get("declared_round5_modifications") or {})
        r5_expected_map = dict(r5_doc.get("round5_source_sha256") or {})
        if set(r5_expected_map.keys()) != set(pinned.keys()):
            raise RuntimeError("ROUND5_SOURCE_MANIFEST.json file set does not match the 109 pinned files")
        for mod_rel, mod_meta in r5_declared.items():
            if pinned.get(mod_rel) != mod_meta.get("historical_08c8b86_sha256"):
                raise RuntimeError(f"ROUND5_SOURCE_MANIFEST historical pin mismatch for {mod_rel}")
            if r5_expected_map.get(mod_rel) != mod_meta.get("round5_sha256"):
                raise RuntimeError(f"ROUND5_SOURCE_MANIFEST round5 pin mismatch for {mod_rel}")

    missing_files: list[str] = []
    mismatched_files: list[dict[str, str]] = []
    verified_hashes: dict[str, str] = {}
    unchanged_08c8b86_count = 0
    verified_declared_r5_count = 0
    for rel, expected_r5_sha in sorted(r5_expected_map.items()):
        fp = resolved / rel
        if not fp.is_file():
            missing_files.append(rel)
            continue
        actual_sha = _sha256(fp)
        hist_sha = pinned[rel]
        if rel not in r5_declared and expected_r5_sha != hist_sha:
            mismatched_files.append({
                "path": rel,
                "expected_sha256": hist_sha,
                "actual_sha256": actual_sha,
                "reason": "undeclared_deviation_from_08c8b86",
            })
        elif actual_sha != expected_r5_sha:
            mismatched_files.append({
                "path": rel,
                "expected_sha256": expected_r5_sha,
                "actual_sha256": actual_sha,
            })
        else:
            verified_hashes[rel] = actual_sha
            if actual_sha == hist_sha:
                unchanged_08c8b86_count += 1
            elif rel in r5_declared:
                verified_declared_r5_count += 1

    adapter_rel = "scripts/reproduce_paired_lifetimes_public.py"
    adapter_sha = _sha256(resolved / adapter_rel) if (resolved / adapter_rel).is_file() else None
    source_tree_digest = hashlib.sha256(
        json.dumps(sorted(verified_hashes.items()), separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "starlink-r2-archive-identity/2",
        "identity_mode": "git_checkout_plus_source_digests" if git_checkout_present else "digest_verified_extracted_archive",
        "git_checkout_present": git_checkout_present,
        "enclosing_unrelated_git_repo_rejected": enclosing_unrelated is not None,
        "enclosing_unrelated_git_repo_path": enclosing_unrelated,
        "run_commit": run_commit,
        "git_tree_dirty": git_tree_dirty,
        "pinned_08c8b86_python_expected": len(pinned),
        "pinned_08c8b86_unchanged_verified": unchanged_08c8b86_count,
        "round5_declared_modifications_verified": verified_declared_r5_count,
        "round5_source_manifest_sha256": r5_manifest_sha,
        "pinned_08c8b86_python_verified": len(verified_hashes),
        "pinned_08c8b86_missing": missing_files,
        "pinned_08c8b86_mismatched": mismatched_files,
        "pinned_source_tree_aggregate_sha256": source_tree_digest,
        "public_adapter_path": adapter_rel,
        "public_adapter_sha256": adapter_sha,
        "digest_identity_ok": (len(verified_hashes) == len(pinned) and not missing_files and not mismatched_files),
    }


def _load_frozen_driver_pins(repo_root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """Load the published SHA-256 pins for consumed raw drivers from evidence/DATA_MANIFEST.json."""
    manifest_path = repo_root / "evidence" / "DATA_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing frozen data manifest: {manifest_path}")
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_path: dict[str, str] = {}
    for asset in doc.get("assets") or []:
        paths = asset.get("payload_paths") or []
        shas = asset.get("computed_sha256") or []
        rec_sha = str(asset.get("recorded_sha256") or "").strip().lower()
        for idx, rel in enumerate(paths):
            sha = str(shas[idx] if idx < len(shas) else rec_sha).strip().lower()
            if rel and sha:
                by_path[str(rel).strip()] = sha
    for entry in doc.get("files") or []:
        rel = str(entry.get("path") or "").strip()
        sha = str(entry.get("sha256") or "").strip().lower()
        if rel and sha and rel not in by_path:
            by_path[rel] = sha
    pins: dict[str, str] = {}
    for source, asset_id, ext in CONSUMED_DRIVER_ASSET_IDS:
        rel_key = f"data/raw/{source}/{asset_id}{ext}"
        if rel_key not in by_path:
            raise RuntimeError(f"Frozen manifest {manifest_path} missing consumed driver pin {rel_key!r}")
        pins[f"{source}/{asset_id}{ext}"] = by_path[rel_key]
    return pins


def extract_consumed_driver_hashes(
    raw_root: pathlib.Path,
    *,
    repo_root: pathlib.Path = REPO_ROOT,
    verify_frozen_manifest: bool = True,
) -> dict[str, dict[str, Any]]:
    """Extract verified hashes of consumed GFZ/CelesTrak assets from RawStore.manifest() (a list).

    In default exact-reproduction mode (``verify_frozen_manifest=True``), every consumed driver
    is verified against BOTH its local RawStore sidecar AND the published SHA-256 pin in
    ``evidence/DATA_MANIFEST.json``, raising ``RuntimeError`` before propagation if any file
    differs from the frozen snapshot. Pass ``verify_frozen_manifest=False`` only for an
    explicitly labelled new-input run (e.g. rolling upstream CelesTrak updates).
    """
    from thermosphere_consequence.drivers.raw_store import RawStore

    frozen_pins = _load_frozen_driver_pins(repo_root)
    store = RawStore(raw_root)
    manifest_entries = store.manifest()
    if not isinstance(manifest_entries, list):
        raise TypeError(f"Expected list from RawStore.manifest(), got {type(manifest_entries).__name__}")

    index = {(m.source, m.asset_id): m for m in manifest_entries}
    consumed: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    frozen_mismatches: list[dict[str, str]] = []
    for source, asset_id, ext in CONSUMED_DRIVER_ASSET_IDS:
        rel_driver = f"{source}/{asset_id}{ext}"
        meta = index.get((source, asset_id))
        if meta is None:
            missing.append(rel_driver)
            continue
        # Trigger RawStore.read() so on-disk bytes are re-hashed and verified against sidecar SHA-256
        payload = store.read(source, asset_id, ext)
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != meta.sha256:
            raise RuntimeError(f"Local sidecar digest mismatch for {rel_driver}: {actual_sha} != {meta.sha256}")
        expected_frozen = frozen_pins[rel_driver]
        matches_frozen = (actual_sha == expected_frozen)
        if not matches_frozen:
            frozen_mismatches.append({
                "driver": rel_driver,
                "expected_frozen_sha256": expected_frozen,
                "actual_sha256": actual_sha,
            })
        consumed[rel_driver] = {
            "source": meta.source,
            "asset_id": meta.asset_id,
            "sha256": actual_sha,
            "frozen_manifest_sha256": expected_frozen,
            "matches_frozen_manifest": matches_frozen,
            "byte_count": len(payload),
            "retrieved_at_utc": meta.retrieved_at_utc,
            "source_url": getattr(meta, "url", getattr(meta, "source_url", "")),
            "licence": getattr(meta, "licence", ""),
        }
    if missing:
        raise RuntimeError(f"Missing consumed raw driver assets under {raw_root}: {missing}")
    if verify_frozen_manifest and frozen_mismatches:
        raise RuntimeError(
            "Consumed raw drivers do not match frozen evidence/DATA_MANIFEST.json pins "
            "(refusing exact-reproduction mode; pass --allow-new-drivers for an explicitly labelled "
            f"new-input run): {frozen_mismatches}"
        )
    return consumed


def _load_script(name: str) -> types.ModuleType:
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_paired_reproduction(
    output_path: pathlib.Path | None = None,
    *,
    allow_new_drivers: bool = False,
) -> dict[str, Any]:
    """Execute the full 6-case NRLMSIS 2.1 paired reproduction with non-Git identity and pinned driver hashes."""
    identity = resolve_archive_identity(REPO_ROOT)
    if not identity["digest_identity_ok"]:
        raise SystemExit(f"REFUSING TO RUN: source digest identity check failed: {identity}")

    if not os.environ.get("NRLMSIS21_ROOT"):
        raise SystemExit(
            "REFUSING TO RUN: NRLMSIS21_ROOT not set. Obtain official nrlmsis2.1.tar.gz from NRL, "
            "export NRLMSIS21_ROOT=/path/outside/repo, and build via: bash scripts/build_nrlmsis2_driver.sh 2.1"
        )
    root21 = pathlib.Path(os.environ["NRLMSIS21_ROOT"])
    driver_bin = root21 / "build" / "wse_msis2_driver"
    if not driver_bin.is_file():
        raise SystemExit(
            f"REFUSING TO RUN: compiled NRLMSIS 2.1 driver missing at {driver_bin}. "
            "Build it first via: bash scripts/build_nrlmsis2_driver.sh 2.1"
        )

    raw_root = REPO_ROOT / "data" / "raw"
    consumed_drivers = extract_consumed_driver_hashes(
        raw_root,
        repo_root=REPO_ROOT,
        verify_frozen_manifest=not allow_new_drivers,
    )
    all_drivers_match_frozen = all(bool(v.get("matches_frozen_manifest")) for v in consumed_drivers.values())

    from thermosphere_consequence.aerodynamics.faceted import FacetedFreeMolecularModel
    from thermosphere_consequence.aerodynamics.gsi_accommodation import PilinskiAltitudeAccommodation
    from thermosphere_consequence.aerodynamics.proxy_geometry import starlink_like_proxy
    from thermosphere_consequence.drivers import space_weather as sw
    from thermosphere_consequence.events import starlink_g4_7 as EV
    from thermosphere_consequence.licence.policy_loader import load_licence_policy
    from thermosphere_consequence.propagation.propagator import OrbitState
    from thermosphere_consequence.protocols.atmosphere import DriverState
    from thermosphere_consequence.providers.nrlmsis2 import NRLMSIS21Provider
    from thermosphere_consequence.scenarios.fixed_path import quiet_baseline_drivers

    propagation = _load_script("run_species_gsi_propagation.py")
    audit = _load_script("run_atmosphere_audit.py")
    os.chdir(REPO_ROOT)

    policy = load_licence_policy(REPO_ROOT / "config" / "licence_policy.yaml")
    msis21 = NRLMSIS21Provider(policy=policy, mode="open_noncommercial_research", root=root21)
    kp, f107 = audit.load_drivers()

    def storm_drivers(when: dt.datetime) -> DriverState:
        return sw.drivers_at(when, kp, f107)

    def quiet_drivers(when: dt.datetime) -> DriverState:
        return quiet_baseline_drivers(storm_drivers(when), quiet_ap=QUIET_AP)

    proxy = starlink_like_proxy()
    accommodation = PilinskiAltitudeAccommodation(
        altitude_m=210_000.0,
        activity="solar_maximum",
        wall_temperature_k=propagation.NOMINAL_WALL_K,
    )
    model = FacetedFreeMolecularModel(
        proxy.facets,
        reference_area_m2=proxy.reference_area_m2,
        accommodation=accommodation,
        gas_surface_model="sentman_diffuse",
    )
    position, velocity = EV.insertion_state_vectors()
    initial = OrbitState(EV.DEPLOYMENT_UTC, position, velocity)

    fresh: dict[tuple[str, str], float | None] = {}
    with tempfile.TemporaryDirectory(prefix="r2-paired-public-") as tmp:
        out = pathlib.Path(tmp)
        for name, pitch in propagation.ATTITUDE_CASES.items():
            for forcing, drivers in (("quiet", quiet_drivers), ("storm", storm_drivers)):
                propagator = propagation.MolecularPropagator(
                    model, msis21, drivers, propagation._pitch_quaternion(pitch), EV.MASS_KG_NOMINAL
                )
                case = propagator.propagate(initial, f"{name}_{forcing}", pitch)
                fresh[(name, forcing)] = (
                    case.supported_crossing_s / 86400.0 if case.supported_crossing_s is not None else None
                )
        (out / "fresh.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in fresh.items()}, indent=2))

    stored_rows = list(csv.DictReader((STORED / "summary.csv").open()))
    stored_cases = list(csv.DictReader((STORED / "cases.csv").open()))
    stored_decay = {
        (row["attitude"], row["forcing"]): (float(row["supported_crossing_days"]) if row["supported_crossing_days"] else None)
        for row in stored_cases
    }

    comparisons = []
    discrepancies = []
    for key, new_value in fresh.items():
        old_value = stored_decay.get(key)
        if (new_value is None) != (old_value is None):
            discrepancies.append(f"{key}: censoring status changed")
            status, relative = "DISCREPANT (censoring)", None
        elif new_value is None:
            status, relative = "reproduced (both censored)", None
        else:
            assert old_value is not None
            relative = abs(new_value - old_value) / abs(old_value)
            status = "DISCREPANT" if relative > DECAY_RELATIVE_TOLERANCE else "reproduced"
            if status == "DISCREPANT":
                discrepancies.append(f"{key}: relative diff {relative:.3e} > {DECAY_RELATIVE_TOLERANCE:.0e}")
        comparisons.append({
            "attitude": key[0],
            "forcing": key[1],
            "stored_decay_days": old_value,
            "fresh_decay_days": new_value,
            "relative_difference": relative,
            "status": status,
        })

    penalties = []
    for row in stored_rows:
        name = row["attitude"]
        q, s = fresh.get((name, "quiet")), fresh.get((name, "storm"))
        if q is None or s is None:
            penalties.append({"attitude": name, "reproduced_penalty_days": None, "censoring": "right-censored"})
            continue
        penalty = q - s
        stored_penalty = float(row["storm_lifetime_penalty_days"])
        delta = abs(penalty - stored_penalty)
        if delta > PENALTY_ABSOLUTE_TOLERANCE_DAYS:
            discrepancies.append(f"{name}: penalty diff {delta:.3e} d > {PENALTY_ABSOLUTE_TOLERANCE_DAYS:.0e} d")
        penalties.append({
            "attitude": name,
            "quiet_decay_days": q,
            "storm_decay_days": s,
            "reproduced_penalty_days": penalty,
            "reproduced_penalty_hours": 24.0 * penalty,
            "stored_penalty_days": stored_penalty,
            "absolute_difference_days": delta,
        })

    record = {
        "schema": "starlink-r2-paired-reproduction-public/1",
        "reproduction_mode": (
            "exact_frozen_driver_reproduction"
            if all_drivers_match_frozen
            else "new_input_run_not_exact_snapshot_reproduction"
        ),
        "archive_identity": identity,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "drivers": {
            "quiet_ap": QUIET_AP,
            "all_match_frozen_manifest": all_drivers_match_frozen,
            "consumed_raw_store_assets": consumed_drivers,
        },
        "comparisons": comparisons,
        "paired_penalties": penalties,
        "discrepancies": discrepancies,
        "all_reproduced": (not discrepancies) and all_drivers_match_frozen,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-archive-identity", action="store_true",
                      help="Verify non-Git/Git archive identity and 109 frozen source hashes offline.")
    mode.add_argument("--check-raw-drivers", action="store_true",
                      help="Verify consumed GFZ/CelesTrak raw drivers against RawStore sidecars and DATA_MANIFEST.json pins.")
    mode.add_argument("--run", action="store_true",
                      help="Run the full 6-case NRLMSIS 2.1 paired reproduction (requires NRLMSIS21_ROOT).")
    ap.add_argument("--allow-new-drivers", action="store_true",
                    help="Allow newly fetched rolling drivers differing from DATA_MANIFEST.json (explicitly labelled new-input run).")
    ap.add_argument("--output", type=pathlib.Path, default=None, help="Optional JSON output path.")
    args = ap.parse_args(argv)

    if args.check_raw_drivers:
        drivers = extract_consumed_driver_hashes(
            REPO_ROOT / "data" / "raw",
            repo_root=REPO_ROOT,
            verify_frozen_manifest=not args.allow_new_drivers,
        )
        all_match = all(bool(v.get("matches_frozen_manifest")) for v in drivers.values())
        out = {
            "archive_identity": resolve_archive_identity(REPO_ROOT),
            "reproduction_mode": (
                "exact_frozen_driver_reproduction"
                if all_match
                else "new_input_run_not_exact_snapshot_reproduction"
            ),
            "all_match_frozen_manifest": all_match,
            "consumed_raw_store_assets": drivers,
        }
        print(json.dumps(out, indent=2))
        return 0 if out["archive_identity"]["digest_identity_ok"] else 1

    if args.run:
        rec = run_paired_reproduction(output_path=args.output, allow_new_drivers=args.allow_new_drivers)
        print(json.dumps(rec, indent=2))
        return 0 if rec["all_reproduced"] else 1

    # Default / --check-archive-identity: safe portable offline identity verification
    ident = resolve_archive_identity(REPO_ROOT)
    print(json.dumps(ident, indent=2))
    return 0 if ident["digest_identity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

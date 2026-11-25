#!/usr/bin/env python3
"""Independent standalone verification runner for starlink-thermosphere-orbital-decay (Paper I, v1.0.0)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from thermosphere_consequence.drivers.space_weather import KpApSeries  # noqa: E402
from thermosphere_consequence.providers.nrlmsise00 import Nrlmsise00Provider  # noqa: E402
from thermosphere_consequence.protocols.atmosphere import DriverState  # noqa: E402


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    # 1. Manifest SHA-256 integrity
    man = REPO / "MANIFEST.sha256"
    checked = 0
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        exp, rel = line.split(None, 1)
        fp = REPO / rel.strip()
        assert fp.is_file() and _sha256(fp) == exp.lower(), f"Manifest mismatch: {rel}"
        checked += 1

    # 2. Table 1 & formal GFZ UTC calendar-day daily Ap = 32.0 + Q7 +2.532%
    atm = json.loads((REPO / "evidence/atmosphere_audit.json").read_text(encoding="utf-8"))
    q7 = atm["model_configuration"]["q07_ap_mode"]
    assert abs(float(q7["ap_history_at_peak"][0]) - 32.0) < 1e-12
    q7_pct = (float(q7["verification"]["ratio_storm_over_daily"]) - 1.0) * 100.0
    assert abs(q7_pct - 2.5322) < 0.01

    # 3. Live pymsis recomputation of quiet-ap sensitivity sweep + incomplete calendar-day rejection
    prov = Nrlmsise00Provider()
    t_peak = dt.datetime(2022, 2, 4, 20, 0, tzinfo=dt.UTC)
    storm_hist = tuple(float(x) for x in q7["ap_history_at_peak"])
    st_storm = prov.state_at(t_peak, 210e3, 53.22, 0.0, DriverState(f107=126.5, f107a=108.79135802469136, ap_daily=32.0, ap_history=storm_hist))
    st_quiet4 = prov.state_at(t_peak, 210e3, 53.22, 0.0, DriverState(f107=126.5, f107a=108.79135802469136, ap_daily=4.0, ap_history=(4.0,) * 7))
    enh_live = (st_storm.total_mass_density / st_quiet4.total_mass_density - 1.0) * 100.0
    assert abs(enh_live - 17.14295) < 0.01, f"Unexpected live peak enhancement: {enh_live}"

    # Incomplete 7-bin day rejection check
    seven = tuple(dt.datetime(2022, 2, 4, h, 0, tzinfo=dt.UTC) for h in (0, 3, 6, 9, 12, 15, 18))
    partial = KpApSeries(times_utc=seven, kp=(3.0,) * 7, ap=(27.0,) * 7, status_flags=("definitive",) * 7)
    try:
        partial.daily_ap_for(dt.datetime(2022, 2, 4, 19, 0, tzinfo=dt.UTC))
        raise AssertionError("Incomplete calendar day did not raise ValueError!")
    except ValueError:
        pass

    # 4. Table 2 (WAM-IPE operators & 96h path energy) + Table 3/4 paired threshold crossing times
    diag = json.loads((REPO / "evidence/referee_revision_diagnostics.json").read_text(encoding="utf-8"))
    p1 = diag["paper1"]
    assert abs(float(p1["path_energy_bridged_985"]["inertial_orbital_loss_J_kg"]["wam_over_msis_excess_pct"]) - 49.4926) < 0.01
    assert abs(float(p1["path_energy_contiguous_non_gap"]["inertial_orbital_loss_J_kg"]["wam_over_msis_excess_pct"]) - 48.40) < 0.02

    rows = {r["attitude"]: r for r in csv.DictReader((REPO / "results/species_resolved_gsi/storm_lifetime_comparison/summary.csv").open(encoding="utf-8"))}
    assert abs(float(rows["plausible_nominal_30deg"]["storm_lifetime_penalty_hours"]) - 5.9425) < 0.005
    assert abs(float(rows["max_drag_broadside"]["storm_lifetime_penalty_hours"]) - 3.035) < 0.01
    assert abs(float(rows["min_drag_edge_on"]["storm_lifetime_penalty_hours"]) - 29.71) < 0.02

    print(f"[PASS] manifest={checked} files | live_peak={enh_live:.4f}% | Ap=32.0 | Q7=+{q7_pct:.4f}% | 30deg=5.94h | WAM=+49.49%")
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

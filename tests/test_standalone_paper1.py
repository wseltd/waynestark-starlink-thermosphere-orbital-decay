"""Standalone unit & verification tests for waynestark-starlink-thermosphere-orbital-decay (Paper I)."""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path
import pytest

from thermosphere_consequence.drivers.space_weather import KpApSeries
from thermosphere_consequence.providers.nrlmsise00 import Nrlmsise00Provider
from thermosphere_consequence.protocols.atmosphere import DriverState

REPO = Path(__file__).resolve().parents[1]


def test_daily_ap_calendar_day_and_incomplete_rejection() -> None:
    day3 = (9.0, 27.0, 32.0, 67.0, 48.0, 27.0, 15.0, 12.0)
    day4 = (27.0, 27.0, 22.0, 22.0, 27.0, 56.0, 48.0, 27.0)
    times = tuple(dt.datetime(2022, 2, 3, h, 0, tzinfo=dt.UTC) for h in range(0, 24, 3)) + tuple(
        dt.datetime(2022, 2, 4, h, 0, tzinfo=dt.UTC) for h in range(0, 24, 3)
    )
    series = KpApSeries(times_utc=times, kp=(3.0,) * 16, ap=day3 + day4, status_flags=("definitive",) * 16)
    assert series.daily_ap_for(dt.datetime(2022, 2, 4, 20, 0, tzinfo=dt.UTC)) == pytest.approx(32.0, rel=1e-12)

    partial = KpApSeries(times_utc=times[:7], kp=(3.0,) * 7, ap=day3[:7], status_flags=("definitive",) * 7)
    with pytest.raises(ValueError, match="incomplete UTC calendar-day"):
        partial.daily_ap_for(dt.datetime(2022, 2, 3, 18, 0, tzinfo=dt.UTC))


def test_nrlmsise00_caveat_is_qualified_not_universal_lower_bound() -> None:
    st = Nrlmsise00Provider().state_at(
        dt.datetime(2022, 2, 4, 20, 0, tzinfo=dt.UTC), 210e3, 53.22, 0.0,
        DriverState(f107=128.2, f107a=109.1, ap_daily=32.0, ap_history=(32.0,) * 7),
    )
    note = " ".join(st.notes)
    assert "not an established universal lower bound" in note
    assert "Treat enhancement as a lower bound" not in note


def test_verify_all_claims_passes() -> None:
    cp = subprocess.run([sys.executable, str(REPO / "verify_all_claims.py")], cwd=str(REPO), capture_output=True, text=True, check=True)
    assert "OVERALL: PASS" in cp.stdout

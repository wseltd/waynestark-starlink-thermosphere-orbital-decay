"""Fetch the February 2022 space-weather drivers into the immutable raw store.

This is the ONLY script that touches the network. Everything downstream reads
from data/raw, so an analysis can be reproduced years later from bytes whose
SHA-256 is recorded beside them, without the sources needing to still exist.

Re-running is safe: the raw store is idempotent for identical bytes and raises
rather than overwriting if a source has changed underneath us.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from thermosphere_consequence.drivers import space_weather as sw
from thermosphere_consequence.drivers.raw_store import RawStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01", help="UTC start date")
    parser.add_argument("--end", default="2022-05-31", help="UTC end date")
    parser.add_argument("--root", default="data/raw", help="raw store root")
    parser.add_argument("--asset-id", default="kp_2022_extended")
    args = parser.parse_args(argv)

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.UTC)
    end = dt.datetime.fromisoformat(args.end).replace(tzinfo=dt.UTC)
    store = RawStore(args.root)
    fetch = sw.http_fetcher()

    print(f"fetching GFZ Kp/ap {start:%Y-%m-%d} to {end:%Y-%m-%d} (CC BY 4.0) ...")
    kp = sw.fetch_gfz_kp_ap(start, end, fetcher=fetch, store=store, asset_id=args.asset_id)
    print(f"  {len(kp.times_utc)} three-hourly samples, status flags: {set(kp.status_flags)}")

    print("fetching CelesTrak daily space weather (F10.7 observed and adjusted) ...")
    csv_payload = sw.fetch_celestrak_space_weather(fetcher=fetch, store=store)
    f107 = sw.parse_celestrak_f107(csv_payload)
    print(f"  {len(f107.dates_utc)} daily samples")

    # Independent cross-check. CelesTrak republishes GFZ values with its own
    # rounding, so agreement is expected but is worth verifying rather than
    # assuming: the entire density chain rests on these two feeds.
    independent = sw.parse_celestrak_ap(csv_payload)
    check = sw.cross_check_kp_ap(kp, independent)
    print(
        f"  cross-check vs CelesTrak: {check['samples_compared']} overlapping samples, "
        f"{check['ap_mismatch_count']} ap mismatches, "
        f"max Kp difference {check['max_kp_difference']:.4f}"
    )
    if not check["agrees"]:
        print("  WARNING: the two feeds disagree on ap; investigate before using.")
        return 1

    print(f"\nraw store now holds {len(store.manifest())} assets under {args.root}")
    for asset in store.manifest():
        print(f"  {asset.source}/{asset.asset_id}  {asset.byte_count} B  {asset.sha256[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

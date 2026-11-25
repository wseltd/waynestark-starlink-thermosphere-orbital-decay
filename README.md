# starlink-thermosphere-orbital-decay

Standalone open-source reproducibility repository for **Paper I**:

> **Atmospheric Model Response and Conditional Orbital Consequences of the February 2022 Starlink Deployment**  
> Onur A. W. Aydogan — Waynestark Enterprises Limited  
> Release: `v1.0.0` (archival Zenodo DOI assigned upon public release via GitHub–Zenodo integration; see `CITATION.cff`)

## Scope
This repository contains everything required to inspect and independently verify Paper I without depending on a combined portfolio bundle:
- Compiled manuscript PDF and LaTeX source (`paper/Starlink_Paper1_Atmospheric_Response_and_Orbital_Consequences.pdf`, `paper/main.tex`, `paper/main.bbl`, `paper/figures/`).
- Shared Apache-2.0 physics and propagation library (`src/thermosphere_consequence/`, byte-identical kernel provenance recorded in `KERNEL_PROVENANCE.json` against baseline `08c8b86` and commit `3c412fd1c724`).
- Paper I reproduction & diagnostic scripts (`scripts/`).
- Frozen derived scientific evidence (`evidence/`, `results/`), with historical SHA-256 digests preserved untouched.
- Independent offline + live verification suite (`verify_all_claims.py`, `tests/test_standalone_paper1.py`) and clean-install receipt (`CLEAN_INSTALL_RECEIPT.json`).

## Quick Start (Offline Clean Install & Verification)
```bash
python3 -m venv .venv && . .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pip install -r requirements.portable.txt
pip install --no-deps -e .
python verify_all_claims.py
pytest -q
```

## External Upstream Data & Licensed Models (Not Redistributed)
1. **GFZ Potsdam Kp/ap (CC BY 4.0) & CelesTrak daily F10.7:** retrieved via `python scripts/fetch_drivers.py` and verified against `evidence/DATA_MANIFEST_PAPER1.json`.
2. **NASA GSFC CCMC WAM-IPE (`WAM-IPE-July2021-22`, 180 hourly NetCDFs):** retrieved via `python scripts/fetch_wam_ipe.py` (raw redistribution prohibited under CCMC terms).
3. **Official NRLMSIS 2.0 & 2.1 Fortran packages:** governed by U.S. Naval Research Laboratory academic/non-commercial research licences (US Patent 10,641,925) and NOT redistributed here. Obtain directly from `https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/`, verify digests against `third_party/nrlmsis_asset_manifest.yaml`, export `NRLMSIS20_ROOT` / `NRLMSIS21_ROOT`, and compile local drivers via `bash scripts/build_nrlmsis2_driver.sh 2.1`.

## Scientific Limitations
- All paired orbital threshold-crossing times (`5.94 h` at 30°, `3.04 h` broadside, `29.71 h` edge-on under NRLMSIS 2.1 `V1`; `5.95 h` at 30° under NRLMSISE-00 `V2`/`V3-pair`) are conditional forward counterfactual simulations of an eight-facet proxy geometry from the planned insertion orbit, not observed decay durations of flown satellites.
- Empirical MSIS models are documented as underestimating rapid storm-time enhancements relative to accelerometer observations and physics-based models in many storm phases, though bias varies with altitude, phase, and baseline and is not an established universal lower bound.
- No proprietary SpaceX geometry, attitude telemetry, or operational validation is claimed.

## Acknowledgements & Licence
- Original code and documentation: Apache License 2.0 (`LICENSE`, `NOTICE`).
- The author thanks the U.S. Naval Research Laboratory Technology Transfer and Patent Offices for written confirmation permitting open, non-commercial scientific research use of NRLMSIS 2.0 and 2.1 under their academic research licences; MSIS® is a registered trademark of the U.S. Government.

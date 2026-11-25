# Atmosphere-model audit

Generated 2026-11-24T21:05:47.685045+00:00 by `scripts/run_atmosphere_audit.py`.
Machine-readable companion: `evidence/atmosphere_audit.json`.

## Summary

This run reports a peak modelled density enhancement of **17.1%** at 210 km, at 2022-02-04T20:00:00+00:00. That is BELOW every published figure for this event, including the published figure for the MSIS family this project itself uses.

The audit below establishes that the model was called correctly - public domain NRLMSISE-00, storm-time ap mode with the full 7-element history, observed indices, no suppressed terms - and then computes 6 defensible variants of the same quantity. Where each lands is decided by the numbers, not chosen:

- **AT OR ABOVE** - (a1) pre-storm observed baseline: 16.09% against the reported 17.14%, compared with He et al.'s MSIS2.0 figure at the nearest quoted altitude (10%-15% at 200 km). At or above that figure.
- **AT OR ABOVE** - (a2) pre-storm density at its own instant: 10.48% against the reported 17.14%, compared with He et al.'s MSIS2.0 figure at the nearest quoted altitude (10%-15% at 200 km). At or above that figure.
- **AT OR ABOVE** - (a3) quiet baseline at ap = 2 rather than ap = 4: 20.58% against the reported 17.14%, compared with He et al.'s MSIS2.0 figure at the nearest quoted altitude (10%-15% at 200 km). At or above that figure.
- **AT OR ABOVE** - (b) orbit-averaged over the best revolution: 25.19% against the reported 17.14%, compared with He et al.'s MSIS2.0 figure at the nearest quoted altitude (10%-15% at 200 km). At or above that figure.
- **AT OR ABOVE** - (c) evaluated at 300 km, one of the two altitudes He et al. quote: 27.93% against the reported 17.14%, compared with He et al.'s MSIS2.0 figure at the nearest quoted altitude (10%-15% at 200 km). At or above that figure.
- **AT OR ABOVE** - (d) maximum over time, latitude and local time: 21.14% against the reported 17.14%, compared with He et al.'s MSIS2.0 figure at the nearest quoted altitude (10%-15% at 200 km). At or above that figure.

At or above He et al.'s altitude-matched MSIS2.0 figure: (a1) pre-storm observed baseline; (a2) pre-storm density at its own instant; (a3) quiet baseline at ap = 2 rather than ap = 4; (b) orbit-averaged over the best revolution; (c) evaluated at 300 km, one of the two altitudes He et al. quote; (d) maximum over time, latitude and local time.

THERE IS NO GAP TO THE PUBLISHED MSIS FIGURE AT THIS ALTITUDE. The reported 17.1% at 210 km sits above He et al.'s 10%-15% for MSIS2.0 at 200 km, the nearest altitude they quote, and inside the 10%-25% their 200 and 300 km figures span. This section was built to explain a gap that existed only while a single 20-25% band was carried across the whole layer. What the section still measures is worth keeping: how wide the family of defensible values from this model is. This run's enhancement rises with altitude from 16.2% at 200 km to 27.9% at 300 km, so quoting it at 210 km and averaging it along a 210 x 338 km orbit differ mainly in how much of that altitude range they include. Maximising over place as well as time gives 21.1%. Replacing the counterfactual with a pre-storm OBSERVED baseline makes the enhancement SMALLER, not larger; the quiet-ap VALUE chosen inside that definition is a sensitive knob. The honest reading is that 17.1% is one defensible number from a family spanning roughly 16% to 28%, and that the family, not the single number, is what should be reported. THE REAL DISAGREEMENT IS WITH THE PHYSICS-BASED MODELS - WAM-IPE, CTIPe and MAGE run two to four times higher for this event - and both source papers report that difference themselves. It is a difference between model families, not evidence about the true atmosphere at 210 km.

## 1. Exact implementation

- Model: **NRLMSISE-00** (Picone et al. 2002, doi:10.1029/2002JA009430)
- Licence: no licence terms located for the compiled artefact; research-only pending verification (see third_party/licences/NRLMSISE-00-PROVENANCE.md)
- Implementation: The NRL Fortran NRLMSISE-00 source, compiled into the pymsis extension module msis00f and called through pymsis.msis.calculate(). No Python reimplementation of the model is used anywhere in this repository.
- Call site: `thermosphere_consequence.providers.nrlmsise00.Nrlmsise00Provider._raw`
- Provider identity: `NRLMSISE-00` / `pymsis/version=0`

## 2. Model version

- `version=0` - 0 = NRLMSISE-00, the 2001 model; no licence terms located for the compiled artefact, research-only
- Never reachable: [2.0, 2.1]. NRLMSIS 2.0/2.1 is under a US Naval Research Laboratory academic-non-commercial licence and US Patent 10,641,925. A guard test parses the provider with ast and fails if any restricted version literal can reach msis.calculate.

## 3. Package versions

- pymsis: `0.12.0`
- numpy: `2.5.2`
- scipy: `1.18.1`
- python: `3.13.9`
- platform: `linux`

## 4. Coefficient files and hashes

NRLMSISE-00 carries its coefficients INSIDE the Fortran source, so there is no external coefficient file for version 0 to hash. The hashable artefact is the compiled extension module. The .parm files shipped alongside belong to NRLMSIS 2.0/2.1 and are NOT read by version 0; they are hashed here only to record that they were present and unused.

| artefact | role | sha256 | bytes |
| --- | --- | --- | --- |
| `msis00f.cpython-313-x86_64-linux-gnu.so` | NRLMSISE-00 (IN USE) | `9741799d1432f73338ecaa6a43335220e53a6a800f200c11bd27686d02eebee6` | 146073 |
| `msis20f.cpython-313-x86_64-linux-gnu.so` | NRLMSIS 2.x (never called) | `e80884aa5c5175964a37154a989edbe4c581d3b5b2aff854a46489f64adf6b76` | 144441 |
| `msis21f.cpython-313-x86_64-linux-gnu.so` | NRLMSIS 2.x (never called) | `98eba32c5ee879c0451b9e4362c9c07f3204c35c1bf3dd106302f1f438d64375` | 148633 |
| `msis2.0.parm` | NRLMSIS 2.x parameters (not read by version 0) | `cf612796af04f473fc9b1445460c30aa497cdc9c991d3507190893577c124602` | 536576 |
| `msis21.parm` | NRLMSIS 2.x parameters (not read by version 0) | `a322a749f368e73117dd20f3fdcf7389dabc5509f4c27073cc5580999381b508` | 536576 |

## 5. Model switches

- Switch array length: 25
- In force: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Default, for contrast: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Only switch changed from default: **geomagnetic_activity** = -1.0

pymsis create_options index 8 is geomagnetic_activity: +1 selects daily-Ap mode, -1 selects storm-time mode in which the 7-element ap array is consumed. Every other switch is left at 1, its default, so no seasonal, diurnal, tidal or longitudinal term is suppressed.

## 6. NRLMSISE-00 flags

| flag | value |
| --- | --- |
| f107_effects | 1.0 |
| time_independent | 1.0 |
| symmetrical_annual | 1.0 |
| symmetrical_semiannual | 1.0 |
| asymmetrical_annual | 1.0 |
| asymmetrical_semiannual | 1.0 |
| diurnal | 1.0 |
| semidiurnal | 1.0 |
| geomagnetic_activity | -1.0 |
| all_ut_effects | 1.0 |
| longitudinal | 1.0 |
| mixed_ut_long | 1.0 |
| mixed_ap_ut_long | 1.0 |
| terdiurnal | 1.0 |

- Anomalous oxygen included: True
- Species reported: N2, O2, O, He, H, Ar, N, anomalous_O
- Species NOT reported: NO. NRLMSISE-00 does not produce NO; pymsis returns NaN in that column and the provider omits the key rather than writing a zero, because absence and zero are different claims.

## 7. Daily Ap versus the 7-element ap history

- **Mode in force: 7-element ap history, storm-time mode** (`geomagnetic_activity=-1.0`)
- Nrlmsise00Provider._raw sets options=create_options(geomagnetic_activity=-1) and passes drivers.ap_history whenever that history is present; it falls back to [ap_daily] * 7 with options=None only when no history was supplied.
- ap history at the peak: `[32.0, 48.0, 56.0, 27.0, 22.0, 25.125, 16.0]`
  (daily Ap, ap now, ap 3 h before, ap 6 h before, ap 9 h before, mean ap 12-33 h before, mean ap 36-57 h before)

Verified, not asserted. The same drivers were run through both ap paths at the peak point. Identical results would prove the storm switch was not in force.

| ap path | rho at 210 km (kg/m3) |
| --- | --- |
| 7-element history, storm-time switch (IN USE) | 1.6130e-10 |
| daily Ap repeated 7 times, default switch | 1.5731e-10 |

Ratio 1.0253. The two differ, which is the evidence that the storm-time switch is actually in force: in daily-Ap mode the multi-hour ap spike is smeared into a daily mean and the storm response is understated.

## 8. F10.7

NRLMSISE-00 is fitted to the PREVIOUS day's observed F10.7; drivers_at applies that offset. The 81-day value is a centred mean of observed F10.7, never of adjusted F10.7.

- At the peak (2022-02-04T20:00:00+00:00): F10.7 = **126.5**, F10.7a = **108.7914**
- At deployment (2022-02-03T18:28:00+00:00): F10.7 = **128.2**, F10.7a = **109.0889**
- Source: CelesTrak - CelesTrak space weather data (Kelso), Space-Track-derived

| date | F10.7 observed | F10.7 81-day centred |
| --- | --- | --- |
| 2022-01-30 | 129.6 | 109.8309 |
| 2022-01-31 | 129.5 | 109.6840 |
| 2022-02-01 | 128.6 | 109.4679 |
| 2022-02-02 | 128.2 | 109.2840 |
| 2022-02-03 | 126.5 | 109.0889 |
| 2022-02-04 | 129.6 | 108.7914 |
| 2022-02-05 | 125.9 | 108.5123 |
| 2022-02-06 | 123.6 | 108.1901 |
| 2022-02-07 | 127.2 | 107.8679 |
| 2022-02-08 | 123.1 | 107.6642 |

## 9. Sampling: latitude, longitude, altitude, local solar time

- Altitude: **210 km** (geodetic), fixed
- Latitude: **53.22 deg**, fixed. Held at 53.22 deg N, the orbital INCLINATION, for the whole series. It is not a latitude the vehicle occupied continuously: an inclined orbit reaches that latitude twice per revolution and spends most of its time below it.
- Longitude: **0.0 deg**, fixed. Held at 0 deg for the whole series, so mean local solar time equals UTC exactly and the diurnal cycle is aliased onto the storm time series rather than sampled across it.
- Cadence: 1 h, 145 samples
- Local solar time at the reported peak: **20.00 h**
- Local-time coverage: All 24 hours, but only once each per day and perfectly correlated with UTC, so local time and storm phase cannot be separated in this series.

## 10. Storm and quiet reference definitions

**Storm** (observed): Observed drivers at each instant: GFZ Kp/ap converted to the 7-element NRLMSISE-00 ap history, plus CelesTrak F10.7 (previous-day observed and 81-day centred average).

- F10.7 = 126.5, F10.7a = 108.7914
- ap daily = 32.0, ap history = `[32.0, 48.0, 56.0, 27.0, 22.0, 25.125, 16.0]`
- Source: GFZ Kp/ap + CelesTrak F10.7

**Quiet** (assumed): A COUNTERFACTUAL, not an observation: geomagnetic activity is set to ap = 4 (Kp ~ 1) in every element of the history while solar flux is held at the observed value, so the comparison isolates the geomagnetic storm from the solar-cycle background.

- F10.7 = 126.5, F10.7a = 108.7914
- ap daily = 4.0, ap history = `[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]`
- Source: counterfactual quiet baseline (ap=4.0) over observed solar flux

## 11. Exact dates and times

- Launch: 2022-02-03T18:13:00+00:00
- Deployment: 2022-02-03T18:28:00+00:00
- Enhancement series: 2022-02-02T00:00:00+00:00 to 2022-02-08T00:00:00+00:00
- Peak enhancement: **2022-02-04T20:00:00+00:00**
- Peak 3-hourly ap (56) at 2022-02-03T09:00:00+00:00; peak Kp (5.33) at 2022-02-03T09:00:00+00:00
- The density peak therefore lags the index peak by **35 hours**
- The peak ap value is attained more than once in the window: 2022-02-03T09:00:00+00:00, 2022-02-04T15:00:00+00:00. The earliest is reported above.

## 12. Density outputs

rho = sum over reported species of m_i * n_i. The species sum is used rather than the model's own total, so the density that drives drag is the same quantity the composition diagnostics describe.

| quantity | value |
| --- | --- |
| rho storm (species sum) | 1.6135e-10 kg/m3 |
| rho quiet (species sum) | 1.3774e-10 kg/m3 |
| model's own total, storm | 1.6130e-10 kg/m3 |
| model's own total, quiet | 1.3769e-10 kg/m3 |
| closure fraction (storm) | 0.000325 |
| enhancement from species sum | 17.1430% |
| enhancement from model's own total | 17.1430% |

## 13. Species outputs

| species | n storm (m^-3) | n quiet (m^-3) | change | mass fraction (storm) |
| --- | --- | --- | --- | --- |
| Ar | 1.6854e+12 | 8.4516e+11 | +99.41% | 0.000694 |
| H | 1.2278e+11 | 1.7214e+11 | -28.68% | 0.000001 |
| He | 9.1095e+12 | 1.2359e+13 | -26.29% | 0.000375 |
| N | 2.1782e+13 | 3.2723e+13 | -33.44% | 0.003138 |
| N2 | 1.8003e+15 | 1.2987e+15 | +38.62% | 0.518777 |
| O | 2.5788e+15 | 2.7017e+15 | -4.55% | 0.424643 |
| O2 | 1.5902e+14 | 8.7976e+13 | +80.76% | 0.052372 |
| anomalous_O | 7.7077e-02 | 7.7077e-02 | -0.00% | 0.000000 |

## 14. Temperature outputs

- Storm: **903.78 K**
- Quiet: **816.82 K**
- Difference: +86.96 K (+10.65%)
- NRLMSISE-00 returns the kinetic temperature at the requested altitude. The exospheric temperature is available from pymsis but is not carried by this provider, so it is reported as absent rather than guessed.

## 15. The enhancement calculation

- Formula: `(rho_storm - rho_quiet) / rho_quiet, signed, never clamped`
- Peak: **17.1430%** at 2022-02-04T20:00:00+00:00
- Mean over the window: 7.7310%
- Median: 7.5433%, minimum: 1.7212%
- Samples: 145
- Between the two states, geomagnetic drivers only; solar flux held.

## 16. Comparison with published values

Each row shows what the source reports, at what altitude, from which model or instrument, under which averaging and against which baseline - and then whether those match the computed quantity. A verdict is issued only where all three compatibility checks pass. Where they do not, the position is still shown, in the last column, as arithmetic context rather than as agreement or disagreement.

| source | published value | model or instrument | altitude reported | averaging | baseline | verdict | position only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fang et al. 2022 (WAM-IPE, 200-400 km domain), doi:10.1029/2022SW003193 | 50%-125% | WAM-IPE (Whole Atmosphere Model - Ionosphere Plasmasphere Electrodynamics) | 200-400 km domain | spatial_range | n_day_model_mean | **NOT COMPARABLE** | BELOW |
| Fang et al. 2022 (NRLMSISE-00 at 210 km), doi:10.1029/2022SW003193 | 0%-20% | NRLMSISE-00 | 210 km | global_field_max | n_day_model_mean | **NOT COMPARABLE** | WITHIN |
| He et al. 2023 (MSIS2.0 at 200 km), doi:10.1029/2023SW003521 | 10%-15% | NRLMSIS 2.0 | 200 km | global_mean_max | quiet_reference_days | **NOT COMPARABLE** | ABOVE |
| He et al. 2023 (MSIS2.0 at 300 km), doi:10.1029/2023SW003521 | 20%-25% | NRLMSIS 2.0 | 300 km | global_mean_max | quiet_reference_days | **NOT COMPARABLE** | BELOW |
| He et al. 2023 (CTIPe/MAGE model average at 200 km), doi:10.1029/2023SW003521 | 35% | mean of CTIPe and MAGE | 200 km | global_mean_max | quiet_reference_days | **NOT COMPARABLE** | BELOW |
| He et al. 2023 (CTIPe/MAGE model average at 300 km), doi:10.1029/2023SW003521 | 60% | mean of CTIPe and MAGE | 300 km | global_mean_max | quiet_reference_days | **NOT COMPARABLE** | BELOW |
| He et al. 2023 (Swarm-A accelerometer), doi:10.1029/2023SW003521 | 110% | Swarm-A accelerometer | 438 km | in_situ_orbit_max | quiet_reference_days | **NOT COMPARABLE** | BELOW |
| He et al. 2023 (GRACE-FO accelerometer), doi:10.1029/2023SW003521 | 120% | GRACE-FO accelerometer | 505 km | in_situ_orbit_max | quiet_reference_days | **NOT COMPARABLE** | BELOW |

Incompatibilities, per source:

- **Fang et al. 2022 (WAM-IPE, 200-400 km domain), doi:10.1029/2022SW003193** - the source reports a range over 200-400 km that varies with location as well as height, so it states no value at 210 km and none can be read from it; the computed value is a 'point_peak' and this finding is a 'spatial_range'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'n_day_model_mean'.
- **Fang et al. 2022 (NRLMSISE-00 at 210 km), doi:10.1029/2022SW003193** - the computed value is a 'point_peak' and this finding is a 'global_field_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'n_day_model_mean'.
- **He et al. 2023 (MSIS2.0 at 200 km), doi:10.1029/2023SW003521** - reported at 200 km, 10 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'.
- **He et al. 2023 (MSIS2.0 at 300 km), doi:10.1029/2023SW003521** - reported at 300 km, 90 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'.
- **He et al. 2023 (CTIPe/MAGE model average at 200 km), doi:10.1029/2023SW003521** - reported at 200 km, 10 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'.
- **He et al. 2023 (CTIPe/MAGE model average at 300 km), doi:10.1029/2023SW003521** - reported at 300 km, 90 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'.
- **He et al. 2023 (Swarm-A accelerometer), doi:10.1029/2023SW003521** - reported at 438 km, 228 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'in_situ_orbit_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'.
- **He et al. 2023 (GRACE-FO accelerometer), doi:10.1029/2023SW003521** - reported at 505 km, 295 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'in_situ_orbit_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'.

```
Computed storm-time neutral mass density enhancement: 0.171 (17.1%), measured as a fraction of the quiet-time neutral mass density, at 210 km, defined as a 'point_peak' against a 'fixed_quiet_ap_counterfactual' baseline.
NO DIRECT COMPARISON IS AVAILABLE. None of the 8 published findings matches the computed value on all three of altitude, quantity definition and baseline: 7 fail on altitude, 8 on quantity definition and 8 on baseline. No agreement, disagreement, validation or bias classification is issued, and the per-finding positions below are context only. The previous claim that our fixed-point enhancement lay below all published modelled findings was unsupported. Corrected citations and altitude assignments place the result in its proper model-family context, but differences in sampling, averaging and baseline definitions prevent a direct numerical validation.
Provenance: 8 findings from 2 distinct papers (10.1029/2022SW003193, 10.1029/2023SW003521). Findings are not independent sources; several of these are reported by one paper.
Per-finding detail:
  - Fang et al. 2022 (WAM-IPE, 200-400 km domain), doi:10.1029/2022SW003193: NOT COMPARABLE with 17.1% - the source reports a range over 200-400 km that varies with location as well as height, so it states no value at 210 km and none can be read from it; the computed value is a 'point_peak' and this finding is a 'spatial_range'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'n_day_model_mean'. For context only, 17.1% falls BELOW the published 50%-125% [modelled, 200-400 km]; that is a position, not agreement, disagreement or a measure of bias.
  - Fang et al. 2022 (NRLMSISE-00 at 210 km), doi:10.1029/2022SW003193: NOT COMPARABLE with 17.1% - the computed value is a 'point_peak' and this finding is a 'global_field_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'n_day_model_mean'. For context only, 17.1% falls WITHIN the published 0%-20% [modelled, 210 km]; that is a position, not agreement, disagreement or a measure of bias.
  - He et al. 2023 (MSIS2.0 at 200 km), doi:10.1029/2023SW003521: NOT COMPARABLE with 17.1% - reported at 200 km, 10 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'. For context only, 17.1% falls ABOVE the published 10%-15% [modelled, 200 km]; that is a position, not agreement, disagreement or a measure of bias.
  - He et al. 2023 (MSIS2.0 at 300 km), doi:10.1029/2023SW003521: NOT COMPARABLE with 17.1% - reported at 300 km, 90 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'. For context only, 17.1% falls BELOW the published 20%-25% [modelled, 300 km]; that is a position, not agreement, disagreement or a measure of bias.
  - He et al. 2023 (CTIPe/MAGE model average at 200 km), doi:10.1029/2023SW003521: NOT COMPARABLE with 17.1% - reported at 200 km, 10 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'. For context only, 17.1% falls BELOW the published 35% [modelled, 200 km]; that is a position, not agreement, disagreement or a measure of bias.
  - He et al. 2023 (CTIPe/MAGE model average at 300 km), doi:10.1029/2023SW003521: NOT COMPARABLE with 17.1% - reported at 300 km, 90 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'global_mean_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'. For context only, 17.1% falls BELOW the published 60% [modelled, 300 km]; that is a position, not agreement, disagreement or a measure of bias.
  - He et al. 2023 (Swarm-A accelerometer), doi:10.1029/2023SW003521: NOT COMPARABLE with 17.1% - reported at 438 km, 228 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'in_situ_orbit_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'. For context only, 17.1% falls BELOW the published 110% [observed, 438 km]; that is a position, not agreement, disagreement or a measure of bias.
  - He et al. 2023 (GRACE-FO accelerometer), doi:10.1029/2023SW003521: NOT COMPARABLE with 17.1% - reported at 505 km, 295 km from the 210 km comparison altitude; the computed value is a 'point_peak' and this finding is a 'in_situ_orbit_max'; the computed value is measured against a 'fixed_quiet_ap_counterfactual' and this finding against a 'quiet_reference_days'. For context only, 17.1% falls BELOW the published 120% [observed, 505 km]; that is a position, not agreement, disagreement or a measure of bias.
```

## 17. Comparison with the published-envelope provider

The envelope densities are the SAME quiet baseline scaled by the published enhancement. The ratio column is therefore the factor by which this project's storm density would have to rise to match each published source, at the same point, on the same baseline.

MSIS storm density at the peak point: 1.6135e-10 kg/m3 (enhancement 17.14% over 1.3774e-10 kg/m3).

| band | coverage | selection | applied enhancement | rho (kg/m3) | x MSIS storm | status |
| --- | --- | --- | --- | --- | --- | --- |
| Fang et al. 2022 (WAM-IPE) [modelled] | 200-400 km | low | 50.00% | 2.0661e-10 | 1.280 | published edge of the band |
| Fang et al. 2022 (WAM-IPE) [modelled] | 200-400 km | nominal | — | — | — | REFUSED: the source supplies no altitude profile |
| Fang et al. 2022 (WAM-IPE) [modelled] | 200-400 km | high | 125.00% | 3.0991e-10 | 1.921 | published edge of the band |
| He et al. 2023 (CTIPe/MAGE average) [modelled] | 200-300 km | low | 35.00% | 1.8595e-10 | 1.152 | published edge of the band |
| He et al. 2023 (CTIPe/MAGE average) [modelled] | 200-300 km | nominal | 37.50% | 1.8939e-10 | 1.174 | DERIVED interpolation, not a published value |
| He et al. 2023 (CTIPe/MAGE average) [modelled] | 200-300 km | high | 60.00% | 2.2038e-10 | 1.366 | published edge of the band |
| He et al. 2023 (MSIS2.0) [modelled] | 200-300 km | low | 10.00% | 1.5151e-10 | 0.939 | published edge of the band |
| He et al. 2023 (MSIS2.0) [modelled] | 200-300 km | nominal | 11.50% | 1.5358e-10 | 0.952 | DERIVED interpolation, not a published value |
| He et al. 2023 (MSIS2.0) [modelled] | 200-300 km | high | 25.00% | 1.7217e-10 | 1.067 | published edge of the band |

A band whose source reports its span across LOCATIONS AND ALTITUDES rather than at each altitude edge has no profile to interpolate, and the provider raises rather than manufacturing a value for 210 km. Its low and high rows are the published extremes over the whole domain and are contextual bounds for the event, NOT a validated interval at this altitude. Where a nominal value IS produced, the source states only the two edges and the interpolated figure is this project's assumption.

The envelope provider refuses to answer outside its published coverage:

- above the published altitude band: `505.0 km is outside the published altitude coverage of He et al. 2023 (MSIS2.0) (200-300 km). This provider does not extrapolate a published band beyond what its source supports.`
- outside the published time window: `2022-02-08T00:00:00+00:00 is outside the published time coverage of He et al. 2023 (MSIS2.0) (2022-02-03T00:00:00+00:00 to 2022-02-05T00:00:00+00:00). This provider does not extrapolate a published band beyond what its source supports.`

## 18. Comparison with observation-derived density

**NEITHER measurement is at 210 km. Swarm-A flies at ~438 km and GRACE-FO at ~505 km, 228 km and 295 km ABOVE the Starlink staging altitude. Agreement or disagreement with them says nothing directly about the density at 210 km, because the storm-time enhancement varies with altitude. What the rows below DO establish is whether this model reproduces the observed enhancement at the altitudes where an observation exists at all.**

| platform | altitude | above 210 km by | published | this model here | model shortfall |
| --- | --- | --- | --- | --- | --- |
| Swarm-A accelerometer | 438 km | 228 km | 110% | 56.9% | x1.9 |
| GRACE-FO accelerometer | 505 km | 295 km | 120% | 74.0% | x1.6 |

There is no accelerometer-derived density at 210 km for this storm in the sources this project holds. The 210 km result is therefore unvalidated against observation and is reported as such.

## 19. Sensitivity to neutral wind

Magnitudes taken from the storm-wind scale this repository already cites: HWM14 storm-time winds at 210 km of several hundred m/s (providers.nrlmsise00.ZeroWindProvider), and the storm-time circulation simulated for this event by Fang et al. 2022, doi:10.1029/2022SW003193. Both signs are swept because the sign decides whether the wind adds to or subtracts from the relative speed.

- Evaluated at the insertion state at deployment: 0.00 deg N, -50.91 deg E
- Inertial speed 7815.7 m/s; co-rotation 480.4 m/s
- Orbit average over 2.00 h, 121 samples

A zonal wind is not along-track. On a 53.22 degree orbit the eastward unit vector projects onto the along-track direction with a factor between cos(53.22 deg) = 0.598 at the equator and 1.0 at the orbit's highest latitude, so a wind of W m/s changes the relative speed by less than W. The deployment state is at the ascending node, i.e. the equator, which is the WEAKEST point of the orbit for a zonal wind; the orbit-mean column covers the rest.

| wind (m/s, east) | relative speed (m/s) | change | projection | drag accel (m/s2) | drag change | orbit-mean drag change | first-order lifetime change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -500 | 7827.42 | +289.56 | 0.579 | 5.205388e-04 | +7.83% | +9.42% | -8.61% |
| -250 | 7679.94 | +142.08 | 0.568 | 5.011076e-04 | +3.81% | +4.60% | -4.40% |
| +0 | 7537.86 | +0.00 | - | 4.827385e-04 | +0.00% | +0.00% | +0.00% |
| +250 | 7401.50 | -136.36 | 0.545 | 4.654313e-04 | -3.59% | -4.38% | +4.58% |
| +500 | 7271.19 | -266.67 | 0.533 | 4.491862e-04 | -6.95% | -8.53% | +9.33% |

Orbit-averaged relative speed across the sweep:

| wind (m/s, east) | orbit-mean relative speed (m/s) | change | projection |
| --- | --- | --- | --- |
| -500 | 7848.02 | +381.45 | 0.763 |
| -250 | 7655.67 | +189.11 | 0.756 |
| +0 | 7466.57 | +0.00 | - |
| +250 | 7280.92 | -185.65 | 0.743 |
| +500 | 7098.97 | -367.59 | 0.735 |

**The lifetime column is a FIRST-ORDER estimate, not a propagated result. DragPropagator has no wind term, so a wind-resolved decay cannot be produced without changing the propagator. The estimate assumes lifetime scales as the reciprocal of the orbit-averaged rho * |v_rel|^2 at fixed density and fixed drag coefficient.**

## 20. Sensitivity to local-time and latitude sampling

Grid of 168 points at 210 km, 2022-02-04T20:00:00+00:00: 7 latitudes from -53.22 to 53.22 deg, 24 longitudes covering all 24 hours of local solar time.

| statistic | enhancement |
| --- | --- |
| minimum | 13.13% |
| median | 17.37% |
| mean | 17.29% |
| maximum | 22.08% |
| range | 8.95% |
| max / min | 1.68 |
| at the reported sampling point | 17.14% |
| percentile of the reported point | 46th |

- Most enhanced cell: 22.08% at 20.00 deg, LST 23.0 h
- Least enhanced cell: 13.13% at -40.00 deg, LST 6.0 h

The single point the reported figure used is one cell of this grid. Whether it is a representative cell or a favourable one is answered by its percentile above.

Enhancement against local solar time at 53.22 deg N:

| LST (h) | enhancement |
| --- | --- |
| 8 | 13.70% |
| 9 | 14.17% |
| 10 | 14.49% |
| 11 | 14.49% |
| 12 | 14.18% |
| 13 | 13.69% |
| 14 | 13.27% |
| 15 | 13.13% |
| 16 | 13.39% |
| 17 | 14.03% |
| 18 | 14.97% |
| 19 | 16.08% |
| 20 | 17.14% |
| 21 | 17.90% |
| 22 | 18.11% |
| 23 | 17.72% |
| 0 | 16.90% |
| 1 | 15.93% |
| 2 | 15.05% |
| 3 | 14.34% |
| 4 | 13.80% |
| 5 | 13.40% |
| 6 | 13.21% |
| 7 | 13.32% |

Enhancement against latitude at 0 deg longitude:

| latitude (deg) | enhancement |
| --- | --- |
| -53.22 | 18.91% |
| -40.00 | 17.61% |
| -20.00 | 17.73% |
| +0.00 | 18.01% |
| +20.00 | 18.30% |
| +40.00 | 18.33% |
| +53.22 | 17.14% |

## Where the computed enhancement sits against the published MSIS figures

How wide is the family of defensible enhancement values this model produces for this event, and where do He et al. 2023's MSIS2.0 figures - about 10-15% at 200 km and 20-25% at 300 km - sit inside it? The earlier form of this question assumed a gap, by comparing a 210 km value against a band quoted at 300 km.

Reported peak: **17.14%** at 210 km. He et al.'s MSIS2.0: 10%-15% at 200 km and 20%-25% at 300 km. Against the altitude-matched (200 km) figure the reported value is +7.14% from its lower edge.

### (a) Is it the counterfactual quiet-ap baseline?

The counterfactual quiet-ap baseline is not the same baseline He et al. used. A pre-storm OBSERVED baseline would give a different, possibly larger, enhancement.

**a1 - pre-storm OBSERVED drivers, same instant and place.** Reference 2022-02-01T18:00:00+00:00 (ap daily 5.75, history `[5.75, 2.0, 3.0, 5.0, 3.0, 7.12, 9.5]`, F10.7 129.5).

Enhancement becomes **16.09%**, a change of -1.05% against the counterfactual baseline.

**a2 - pre-storm density sampled at its own instant.** 1.4605e-10 kg/m3 against 1.6135e-10 kg/m3 at the peak, giving **10.48%** (-6.66%). This comparison also moves 68 hours in UT and therefore carries the diurnal, seasonal and solar-flux differences between the two instants. It is not a clean measure of the storm.

**a3 - sensitivity to the quiet-ap choice itself:**

| quiet ap | enhancement |
| --- | --- |
| 0 | 24.91% |
| 2 | 20.58% |
| 4 | 17.14% |
| 7 | 13.18% |
| 12 | 8.68% |

The observed 3-hourly ap values in the quiet day before the storm were 2, 3 and 5, so ap = 4 is a mid-range choice within them and ap = 2 is the quiet end of what was actually observed. The enhancement is sensitive to that choice at the level of the gap being investigated, which is worth stating plainly: the quiet baseline is an assumption, and it is not a small one.

### (b) Is it single-point versus orbit-averaged sampling?

A single fixed point at 210 km, 53.22 deg N, 0 deg E is not what the vehicle sampled. Averaging along the actual insertion trajectory would give a different number.

- drag-free Keplerian + J2 from the deployment state, 210 x 338 km, 4 days, 1153 samples at 300 s
- Whole-path mean enhancement: 14.02%
- Whole-path maximum enhancement: 44.21%
- Best single revolution, mean over the revolution: **25.19%** (density-weighted 17.71%), a change of +8.05% against the single-point figure
- The insertion orbit is 210 x 338 km, so an orbit average spans that whole altitude range and is NOT a 210 km number.

### (c) Is it the altitude at which the published figure is quoted?

He et al. quote 200 AND 300 km. The modelled enhancement rises with altitude, so a figure quoted at 300 km is not comparable to one computed at 210 km.

| altitude | enhancement | quoted by He et al.? |
| --- | --- | --- |
| 200 km | 16.16% | yes |
| 210 km | 17.14% | no |
| 250 km | 21.40% | no |
| 300 km | 27.93% | yes |
| 400 km | 47.47% | no |

This run lands inside the He et al. 20-25% band at NO altitude sampled.

This run gives 16.2% at 200 km and 27.9% at 300 km. He et al.'s 10%-15% is a summary across those same two altitudes, and it lies OUTSIDE the range this run produces between them. A single number computed at 210 km is near the BOTTOM of that range by construction, because 210 km is near the bottom of the interval.

### (d) Is it the definition of 'peak'?

The reported peak is the maximum over TIME at one fixed place. The maximum over time AND place is larger.

Maximum over the whole window AND over latitude and local time: **21.14%** at 2022-02-04T18:00:00+00:00, 0.00 deg, LST 0.0 h - a change of +3.99% against the reported single-point peak.

## What must not be claimed from this audit

- A scalar multiplier applied to this model is NOT a physics simulation. A x2 result is not a WAM-IPE result and a x3.5 result is not a MAGE result; the provider that applies such multipliers is named `ScalarDensityStressTestProvider` and every state it returns carries that prohibition.
- Swarm-A and GRACE-FO do NOT validate the 210 km density. They are 228 km and 295 km above it.
- Agreement with He et al. would be agreement with the model family already in use, in the direction that family is documented to be wrong. It is not corroboration.


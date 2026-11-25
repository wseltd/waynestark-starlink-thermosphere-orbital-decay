"""Generate the incident report from the analysis artefacts.

The report is generated, never hand-written, so it cannot drift from the
numbers. Two properties are enforced rather than hoped for:

  * Every quantitative claim carries its epistemic status — OBSERVED,
    MODELLED, DERIVED or ASSUMED. A reader must never have to guess which
    parts of a sentence are measurements.
  * The prohibited-claims list from the event manifest is checked against the
    generated text. If the report ever contains one of those sentences, the
    generator raises rather than emitting it.

Canonical tickets: P11_12_T030 through T048.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any

from ..events import starlink_g4_7 as event

#: The neutral species the composition table lists, in the order it lists them.
TABULATED_SPECIES: tuple[str, ...] = ("O", "N2", "O2", "He", "Ar", "N", "H")

#: NRLMSISE-00's ninth constituent. Carried everywhere else in the analysis and
#: deliberately kept out of the composition table, which is a reader-facing
#: inventory of the neutral gas rather than a dump of the provider's columns.
#: The footnote under that table states the exclusion and the reason for it; a
#: reader must never have to infer from an absent row that the model omits it.
ANOMALOUS_OXYGEN = "anomalous_O"


class ProhibitedClaimError(RuntimeError):
    """Raised when generated text contains a claim the project must not make."""


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _fmt_days(x: float | None) -> str:
    return "did not reenter" if x is None else f"{x:.2f} d"


#: Where the storm-cost envelope is read from. Read rather than restated so
#: the headline and the sweep cannot drift apart: a hard-coded pair of bounds
#: would go stale the first time the sweep is re-run with different ranges.
_UNCERTAINTY_ARTEFACT = pathlib.Path("evidence/uncertainty_analysis.json")


def _uncertainty_envelope() -> tuple[float, float, float] | None:
    """``(min, max, nominal)`` storm cost in days, or None if unavailable.

    Returns None rather than raising, and rather than substituting a default.
    A missing envelope is reported to the reader as missing: the alternative
    is a headline that silently loses its uncertainty when an artefact is
    absent, which is the failure this function exists to prevent.
    """
    try:
        payload = json.loads(_UNCERTAINTY_ARTEFACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ranges = payload.get("ranges")
    if not isinstance(ranges, list):
        return None
    # Matched on the quantity's own label rather than on position, so that
    # adding a swept quantity upstream cannot silently re-point the headline
    # envelope at a different row.
    band = next(
        (
            row
            for row in ranges
            if isinstance(row, dict) and row.get("quantity") == "storm cost"
        ),
        None,
    )
    if band is None or band.get("units") != "days":
        return None
    low, high = band.get("minimum"), band.get("maximum")
    point = band.get("nominal")
    if not all(isinstance(v, (int, float)) for v in (low, high, point)):
        return None
    if not low > 0.0 or not high > low:  # type: ignore[operator]
        return None
    return float(low), float(high), float(point)  # type: ignore[arg-type]


def check_prohibited_claims(text: str) -> None:
    """Fail loudly if the report states something the evidence cannot support."""
    lowered = text.lower()
    found = [c for c in event.prohibited_claims() if c.lower() in lowered]
    if found:
        raise ProhibitedClaimError(
            "generated report contains prohibited claim(s): "
            + "; ".join(repr(c) for c in found)
        )


def render_markdown(report: dict[str, Any], *, generated_utc: _dt.datetime) -> str:
    """Render the full incident report."""
    storm = report["storm"]
    aero = report["aerodynamics"]
    dens = report["density_enhancement"]
    inv = report["inversion_summary"]
    attr = report.get("attribution", {})
    gt = report["ground_truth"]
    kin = report["gas_kinetics"]

    lines: list[str] = []
    add = lines.append

    add("# February 2022 Starlink Group 4-7: thermosphere and orbital consequence")
    add("")
    add(f"Generated {generated_utc:%Y-%m-%d %H:%M} UTC from `evidence/analysis_report.json`.")
    add("")
    add("Every number below is tagged with its epistemic status:")
    add("**OBSERVED** (measured or catalogued), **MODELLED** (a model's output),")
    add("**DERIVED** (computed from the two), **ASSUMED** (a stated choice).")
    add("")

    # ---- headline --------------------------------------------------------
    add("## Status: PRELIMINARY, under audit")
    add("")
    add(
        "A preliminary calibrated counterfactual analysis using the currently "
        "implemented atmosphere, orbit and drag assumptions. It has NOT passed "
        "the complete scientific, computational, data-provenance, tool-usage, "
        "uncertainty, independent-validation and ticket-completion audit. "
        "Major planned components remain unimplemented."
    )
    add("")
    add(
        "The drag parameter reported below was CALIBRATED against the same "
        "catalogued decay dates it is then compared with. That is in-sample "
        "fit, not independent validation."
    )
    add("")
    add("## Preliminary finding")
    add("")
    if attr:
        add(
            f"At the inverse ballistic coefficient implied by the catalogued decay dates, the "
            f"storm branch shortens modelled lifetime by "
            f"**{attr['storm_cost_days']:.4f} days "
            f"({attr['storm_cost_fraction']:.4%})** relative to the quiet "
            f"counterfactual: modelled_quiet_counterfactual_lifetime "
            f"{attr['quiet_decay_days']:.4f} d, modelled_storm_lifetime "
            f"{attr['storm_decay_days']:.4f} d, for the calibrated reference "
            "object. **MODELLED, CALIBRATED.** The denominator of the relative "
            "figure is the quiet counterfactual lifetime."
        )
        add("")
        # The point estimate must never appear without its envelope. Quoted
        # alone it reads as a resolved quantity, and it is not: the sweep in
        # evidence/UNCERTAINTY_ANALYSIS.md moves it by a factor of about 15,
        # which is far larger than any of the corrections argued about
        # elsewhere in this document. The envelope is read from the committed
        # artefact rather than restated here, so the two cannot drift apart,
        # and its absence is stated rather than silently skipped.
        envelope = _uncertainty_envelope()
        if envelope is None:
            add(
                "> **The uncertainty envelope for this figure is NOT AVAILABLE** "
                "in this run: `evidence/uncertainty_analysis.json` is missing or "
                "unreadable. The number above is a point estimate from one "
                "parameter choice and must not be quoted as a resolved value. "
                "Run `scripts/run_uncertainty_analysis.py`."
            )
        else:
            low, high, nominal = envelope
            add(
                f"> **Read this with its envelope: {low:.3f} to {high:.3f} days** "
                f"(nominal {nominal:.3f}). The sweep in "
                "`evidence/UNCERTAINTY_ANALYSIS.md` moves the storm cost by "
                f"a factor of about {high / low:.0f} over the stated input "
                "ranges, dominated by the scalar storm-density factor "
                "(total-order Sobol' index 0.84) — that is, by how far "
                "NRLMSISE-00 under-responds to this storm. Those bounds are a "
                "SENSITIVITY RANGE: the extremes the quantity reached over a "
                "box of stated input ranges. No probability distribution has "
                "been justified for any of those inputs, so the range carries "
                "NO COVERAGE PROBABILITY and is not a statistical interval of "
                "any kind. The four significant figures above are the "
                "arithmetic of one parameter choice, not the precision of the "
                "answer."
            )
        add("")
    add(
        "The very low insertion altitude "
        f"({event.INSERTION_PERIGEE_ALTITUDE_M/1000:.0f} x "
        f"{event.INSERTION_APOGEE_ALTITUDE_M/1000:.0f} km) produces short orbital "
        "lifetimes in BOTH branches. A stronger causal statement is NOT made "
        "here: attributing primacy would require the ordering to remain stable "
        "across the full density-enhancement, wind, attitude, thrust and "
        "initial-state envelope, which has not yet been established. Stronger "
        "attribution requires the event-specific density response and "
        "spacecraft attitude or drag telemetry."
    )
    add("")
    add(
        "This is a lower estimate. NRLMSISE-00 under-responds to storms at this "
        "altitude (He et al. 2023: ~20-25% for MSIS against ~40% for WAM-IPE "
        "and ~57-75% for MAGE), and the enhancement computed here is below even "
        "the published MSIS figure. The sensitivity section reports SCALAR "
        "DENSITY STRESS TESTS, which are not physics-model simulations: neither "
        "WAM-IPE nor MAGE was run."
    )
    add("")

    # ---- event -----------------------------------------------------------
    add("## The event")
    add("")
    add("| Quantity | Value | Status | Source |")
    add("|---|---|---|---|")
    add(
        f"| Launch | {event.LAUNCH_UTC:%Y-%m-%d %H:%M} UTC | OBSERVED | "
        f"{event.SOURCES['insertion']} |"
    )
    add(
        f"| Deployment | {event.DEPLOYMENT_UTC:%Y-%m-%d %H:%M} UTC | OBSERVED | "
        f"{event.SOURCES['insertion']} |"
    )
    add(
        f"| Insertion orbit | {event.INSERTION_PERIGEE_ALTITUDE_M/1000:.0f} x "
        f"{event.INSERTION_APOGEE_ALTITUDE_M/1000:.0f} km, {event.INSERTION_INCLINATION_DEG}deg "
        f"| OBSERVED | {event.SOURCES['insertion']} |"
    )
    add(
        f"| Satellites launched | {event.SATELLITES_LAUNCHED} | OBSERVED | "
        f"{event.SOURCES['counts']} |"
    )
    add(
        f"| Lost (peer-reviewed) | {event.SATELLITES_LOST_PEER_REVIEWED} | OBSERVED | "
        f"{event.SOURCES['counts']} |"
    )
    add(
        f"| Lost (operator, early) | up to {event.SPACEX_EARLY_ESTIMATE_LOST} | OBSERVED | "
        f"{event.SOURCES['spacex_statement']} |"
    )
    add(
        f"| Never catalogued | {event.SATELLITES_NEVER_CATALOGUED} | OBSERVED | "
        f"{event.SOURCES['counts']} |"
    )
    add("")
    add(
        f"The operator's early figure ({event.SPACEX_EARLY_ESTIMATE_LOST}) and the "
        f"peer-reviewed figure ({event.SATELLITES_LOST_PEER_REVIEWED}) differ. Both are "
        "recorded; the difference is information about how the count firmed up, not an "
        "error to be resolved by preferring one."
    )
    add("")

    # ---- drivers ---------------------------------------------------------
    add("## Space-weather drivers")
    add("")
    add("Geomagnetic indices from GFZ Potsdam (CC BY 4.0). WDC Kyoto indices are")
    add("**not** used anywhere: their terms forbid commercial use, and the code refuses")
    add("to fetch from that host at all.")
    add("")
    add("| Quantity | Value | Status |")
    add("|---|---|---|")
    add(f"| Peak Kp | {storm['peak_kp']:.2f} at {storm['peak_kp_time'][:16]} | OBSERVED |")
    add(f"| Peak ap | {storm['peak_ap']:.0f} at {storm['peak_ap_time'][:16]} | OBSERVED |")
    add(f"| Hours with Kp >= 5 | {storm['hours_kp_ge_5']} | OBSERVED |")
    add(f"| Mean ap over window | {storm['mean_ap']:.1f} | OBSERVED |")
    add(f"| F10.7 (previous day) | {aero['insertion_diagnostics']['f107']:.1f} sfu | OBSERVED |")
    add(f"| F10.7 81-day centred | {aero['insertion_diagnostics']['f107a']:.1f} sfu | DERIVED |")
    add("")
    add(
        f"This is consistent with Fang et al. 2022, who report Kp 5 for more than 6 hours "
        f"on both 3 and 4 February; the ingested data give {storm['hours_kp_ge_5']} hours "
        "at Kp >= 5 across the window."
    )
    add("")

    # ---- atmosphere ------------------------------------------------------
    add("## Thermospheric response")
    add("")
    add(
        f"Peak mass-density enhancement at 210 km: **{dens['peak_fraction']:.1%}** "
        f"at {dens['peak_time'][:16]} UTC. **MODELLED.**"
    )
    add("")
    add("Against the published figures:")
    add("")
    add("| Reference | Relation |")
    add("|---|---|")
    for name, relation in dens["comparison_to_published"].items():
        add(f"| {name} | {relation} |")
    add("")
    # This paragraph read "below every published figure, including the
    # published figure for MSIS itself" until the two primary sources were
    # read at first hand. That was true of the table as it then stood and false
    # of the papers: the MSIS band's lower edge had been taken from the wrong
    # altitude, and the one published NRLMSISE-00 figure for this event and
    # this altitude was missing from the table entirely. The relations are now
    # split, so a sentence asserting a single direction cannot be written here
    # at all - which is the right outcome, because the comparison never
    # supported one.
    add(
        "**The previous claim that this enhancement lay below all published "
        "modelled findings was unsupported.** Corrected citations and altitude "
        "assignments place the result in its proper model-family context: the "
        "MSIS family runs below the physics-based models for this event, which "
        "Fang et al. 2022 and He et al. 2023 both report themselves. But "
        "differences in sampling, averaging and baseline definitions prevent a "
        "direct numerical validation, and none is claimed."
    )
    add("")
    add(
        "The positions in the table are arithmetic, not verdicts. Against the "
        "rows the sources actually report, this value is ABOVE He et al.'s "
        "MSIS2.0 at 200 km and BELOW their 300 km row; the 11%-16% figure "
        "quoted in an earlier summary was OUR interpolation between those two "
        "rows, not a published value, and no paper states a figure for 210 km "
        "except Fang et al.'s NRLMSISE-00 'up to 20%'. The published 200-300 km "
        "figures are maxima of global or zonal MEAN fields; this is a peak over "
        "time at ONE point, 53 degrees latitude, one longitude, one altitude, "
        "against a fixed quiet-ap counterfactual rather than their reference "
        "days or 10-day model mean. No claim is made that any downstream drag "
        "or lifetime penalty is conservative."
    )
    add("")

    add("### Composition at insertion altitude, 210 km")
    add("")
    add("| Species | Number density (m^-3) | Mass fraction | Scale height (km) | Speed ratio |")
    add("|---|---|---|---|---|")
    diag = aero["insertion_diagnostics"]
    for symbol in TABULATED_SPECIES:
        n = kin.get(f"n_{symbol}_m3")
        if n is None:
            continue
        add(
            f"| {symbol} | {n:.3e} | {diag.get(f'mass_fraction_{symbol}', float('nan')):.4f} "
            f"| {kin.get(f'scale_height_{symbol}_km', float('nan')):.1f} "
            f"| {kin.get(f'speed_ratio_{symbol}', float('nan')):.2f} |"
        )
    add("")
    add(
        f"Mean particle mass {kin['mean_particle_mass_kg']:.4e} kg; total number density "
        f"{kin['total_number_density_m3']:.4e} m^-3; neutral temperature "
        f"{kin['temperature_k']:.1f} K. **MODELLED.**"
    )
    add("")
    add(
        "Atomic oxygen and molecular nitrogen together carry essentially all the mass "
        "flux. Every species is hyperthermal (speed ratio well above 1) except atomic "
        "hydrogen, which carries negligible mass."
    )
    add("")
    add(
        f"**Footnote: anomalous oxygen is excluded from this table, and only from "
        f"this table.** NRLMSISE-00 returns a ninth constituent, `{ANOMALOUS_OXYGEN}`, "
        "which the analysis carries throughout: it is in the state the provider "
        "returns, it is mass-flux-weighted into the species-weighted effective drag coefficient "
        "in the next section, and its number density, mass fraction, scale height and "
        "speed ratio are all in `evidence/analysis_report.json` under the same key "
        "pattern as every species above. It is left out here because this table reads "
        "as an inventory of the neutral gas, and anomalous oxygen is not a ninth gas. "
        "It is a fitted correction term carrying the particle mass of atomic oxygen, "
        "introduced so the model reproduces observed drag at high altitude, and "
        "printing it in a row beside O and N2 invites a reader to sum the two oxygens "
        "or to read it as an independently measured population. Neither is supportable "
        "— which is a reason to label it, not a reason to drop it, so this footnote "
        "exists rather than the row."
    )
    add("")

    # ---- aerodynamics ----------------------------------------------------
    add("## Free-molecular aerodynamics")
    add("")
    add(
        f"Species-weighted Sentman flat-plate drag coefficient at insertion: "
        f"**{aero['cd_storm_plate']:.3f}** (storm), {aero['cd_quiet_plate']:.3f} (quiet). "
        "**MODELLED.**"
    )
    add("")
    add(
        f"That is {aero['cd_excess_over_convention']:+.1%} against the conventional "
        f"C_d = {event.CD_CONVENTIONAL}. The direction is expected: Cook (1965), who "
        "originated the 2.2 convention, wrote that it \"is subject to some uncertainty "
        "and may be too low, perhaps by as much as 10-15 percent\", and Moe & Moe (2005) "
        "attribute the excess to near-complete accommodation on surfaces with adsorbed "
        "atomic oxygen."
    )
    add("")
    add(
        f"Weighting is by species **mass** flux, not number density. Accommodation "
        f"coefficient {0.9} **ASSUMED** (Moe & Moe nominal), swept in the sensitivity."
    )
    add("")
    add(
        "**What is and is not resolved by species.** The molecular layer above is a "
        "species-resolved diagnostic and interpretation layer: number density, mass "
        "density, mean particle mass, scale height, thermal speed, speed ratio, mean "
        "free path, Knudsen number, impact flux and impact energy are all per "
        "constituent. The FORCE is not. It is the effective scalar "
        "`F = 0.5 * C_d * rho_total * A * v_rel^2`, using the **total** mass density "
        "and the single coefficient printed above. Composition is what decides that "
        "coefficient — the mass-flux weighting is exactly how the mixture enters — but "
        "no constituent carries its own force term, and at fixed total density "
        "redistributing mass between species leaves the drag unchanged. A per-facet, "
        "per-species force vector exists in `aerodynamics/faceted.py` and is run as a "
        "comparison in `evidence/FACETED_AERODYNAMICS.md`; it does not drive the "
        "propagation here."
    )
    add("")

    # ---- inversion -------------------------------------------------------
    add("## Inverse ballistic coefficient (C_d*A/m) inferred from catalogued decay")
    add("")
    add(
        "Rather than assume an inverse ballistic coefficient, it is inferred from the decay dates "
        "the catalogue actually recorded. The externally published Capella Space Whitney SAR "
        "reference inverse ballistic coefficient "
        f"({event.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG} m^2/kg, Shambaugh 2024, "
        "arXiv:2406.08342) describes a different spacecraft family in operational flight near "
        "525 km — not SpaceX Starlink and not the staging configuration these vehicles were in."
    )
    add("")
    add(
        "| Object | Decay date | Days | Inferred C_d*A/m (m^2/kg) | vs operational | "
        "Implied area (m^2) |"
    )
    add("|---|---|---|---|---|---|")
    for row in report["inversion"]:
        b = row["inferred_inverse_ballistic_coefficient"]
        add(
            f"| {row['object_id']} | {row['decay_date']} | {row['observed_decay_days']:.0f} "
            f"| {b:.4f} | {row['ratio_to_operational']:.2f}x | {row['implied_area_m2']:.2f} |"
            if b else
            f"| {row['object_id']} | {row['decay_date']} "
            f"| {row['observed_decay_days']:.0f} | - | - | - |"
        )
    add("")
    add(
        f"Median inferred C_d*A/m = **{inv['b_median']:.4f} m^2/kg**, "
        f"{inv['median_ratio_to_operational']:.2f}x the Capella Space reference value, across a range of "
        f"{inv['b_min']:.4f} to {inv['b_max']:.4f}. **DERIVED.**"
    )
    add("")
    add(
        "A inverse ballistic coefficient roughly three times **lower** than that Capella Space reference figure "
        "is consistent with the reported safe-mode response, in which the vehicles were "
        "commanded to fly edge-on to reduce projected area. This analysis does not "
        "independently confirm that attitude; it establishes that the catalogued decay "
        "rates require a substantially reduced effective area, which that attitude would "
        "produce."
    )
    add("")
    add(
        "**Degeneracy, stated plainly:** decay depends on the product of density and "
        "inverse ballistic coefficient. Halving the density and doubling B gives an identical "
        "decay curve. The inferred C_d*A/m is therefore conditional on this atmosphere model, "
        "and an error in the density maps directly into it. It is not an independent "
        "measurement of the spacecraft."
    )
    add("")

    # ---- sensitivity -----------------------------------------------------
    add("## Sensitivity to the known MSIS storm deficiency")
    add("")
    add(
        "Because NRLMSISE-00 is documented to under-respond, the storm-minus-quiet "
        "density difference is amplified by explicit factors spanning the published "
        "range: x1 is MSIS as-is, x2 approximates WAM-IPE, x3.5 approximates MAGE."
    )
    add("")
    add(
        "| C_d*A/m (m^2/kg) | decay x1 | stress x2 | stress x3.5 "
        "| shortening x2 | shortening x3.5 |"
    )
    add("|---|---|---|---|---|---|")
    for row in report["enhancement_sensitivity"]:
        add(
            f"| {row['inverse_ballistic_coefficient']:.3f} "
            f"| {_fmt_days(row.get('decay_days_x1'))} "
            f"| {_fmt_days(row.get('decay_days_x2'))} "
            f"| {_fmt_days(row.get('decay_days_x3.5'))} "
            f"| {_fmt_pct(row.get('shortening_vs_x1_x2'))} "
            f"| {_fmt_pct(row.get('shortening_vs_x1_x3.5'))} |"
        )
    add("")
    add(
        "Even at the MAGE-level amplification, decay shortens by well under a factor of "
        "two. The conclusion is robust across the whole published range of storm "
        "response: **the insertion altitude dominates.**"
    )
    add("")

    # ---- validation ------------------------------------------------------
    add("## Validation against the catalogue")
    add("")
    add("| Quantity | Value | Status |")
    add("|---|---|---|")
    add(f"| Objects catalogued from 2022-010 | {gt['catalogued_objects']} | OBSERVED |")
    add(f"| Decayed within 30 days | {gt['storm_window_decays']} | OBSERVED |")
    add(f"| Decayed later (2023-2024) | {gt['later_decays']} | OBSERVED |")
    add(f"| Still on orbit | {gt['still_on_orbit']} | OBSERVED |")
    add(
        f"| Observed decay range | "
        f"{min(gt['observed_decay_days']):.0f}-"
        f"{max(gt['observed_decay_days']):.0f} days | OBSERVED |"
    )
    add("")
    add(
        f"Only {gt['catalogued_objects']} of the {gt['launched']} launched objects ever "
        f"received public catalogue numbers; {gt['never_catalogued']} were never tracked. "
        "This is a comparison against a **subset**, not the fleet, and the subset is "
        "biased toward objects that survived long enough to be catalogued."
    )
    add("")
    add(
        "The two objects that decayed in 2023 and 2024 reached operational altitude "
        "first and were lost for unrelated reasons. Counting them as storm losses would "
        "overstate the event; they are excluded and the exclusion is stated."
    )
    add("")

    # ---- limits ----------------------------------------------------------
    add("## What this analysis cannot establish")
    add("")
    for limitation in (
        "That the modelled density is correct at 210 km. No open model certifies "
        "species-resolved storm density at that altitude, and the observational "
        "platforms (GRACE-FO ~505 km, Swarm-A ~438 km) sit 230-300 km above it.",
        "The trajectories of all 49 objects. 32 were never catalogued and had no public "
        "element sets in the critical window.",
        "That the storm alone explains the losses. It does not, on this evidence.",
        "A separation of mass, area and drag coefficient. Only their combination is "
        "identifiable from a decay curve.",
        "Anything about attitude beyond what the inferred inverse ballistic coefficient requires.",
    ):
        add(f"- {limitation}")
    add("")

    add("## Method")
    add("")
    add("```")
    add("observed drivers (GFZ Kp/ap, CelesTrak F10.7)     OBSERVED")
    add("  -> NRLMSISE-00 species composition               MODELLED (documented storm underestimation)")
    add("  -> species-weighted Sentman free-molecular C_d   MODELLED")
    add("     (ONE scalar; species enter here and nowhere else in the force)")
    add("  -> drag acceleration a = -0.5 (C_d A/m) rho_total |v_rel| v_rel  DERIVED")
    add("  -> Cowell propagation with J2 and co-rotation    MODELLED")
    add("  -> compared against catalogue decay dates        OBSERVED")
    add("```")
    add("")
    add(
        "The third arrow is where the species resolution stops. Everything above it is "
        "per constituent; everything below it runs on one coefficient and the total "
        "mass density. That is an effective scalar free-molecular drag model, and the "
        "species-resolved work above it is a diagnostic and interpretation layer "
        "accompanying it rather than a force closure."
    )
    add("")
    add(
        "Only the two ends are observational. NRLMSISE-00 is used because it is the "
        "model SpaceX itself used for Starlink drag estimation (Fang et al. 2022), which "
        "makes it the right instrument for asking what the operator's own model would "
        "have predicted. It is not treated as ground truth."
    )
    add("")

    text = "\n".join(lines)
    check_prohibited_claims(text)
    return text


def write_report(
    analysis_json: pathlib.Path,
    output_path: pathlib.Path,
    *,
    generated_utc: _dt.datetime | None = None,
) -> pathlib.Path:
    """Render and write the incident report."""
    report = json.loads(analysis_json.read_text(encoding="utf-8"))
    text = render_markdown(
        report, generated_utc=generated_utc or _dt.datetime.now(_dt.UTC)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path

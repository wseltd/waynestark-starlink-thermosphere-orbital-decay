"""SGP4 propagation of catalogued element sets — for VALIDATION ONLY.

SGP4 is used here for exactly one job: to check that a catalogued element set
reproduces the orbit it claims to describe, and to give the numerical
propagator an independent yardstick over a short arc where the two must agree.

It is NEVER used for the storm counterfactual, and that exclusion is a physics
decision rather than a matter of taste. SGP4's drag is a single fitted ``B*``
coefficient, absorbed from the tracking residuals of a particular object over a
particular span of real atmosphere. B* is therefore a summary of the atmosphere
that already happened. It contains no density model that could be swapped, no
composition, no dependence on F10.7 or ap — so SGP4 simply cannot answer "what
would this orbit have done under a different atmosphere?", which is the entire
question this project exists to ask. Re-fitting B* to a counterfactual would be
assuming the answer. The counterfactual runs through
``propagation.propagator.DragPropagator``, whose density function is injected
and can be varied while nothing else changes.

Two further boundaries are worth stating because they limit how far an
agreement between the two propagators can be pushed:

  * SGP4 output is in the TEME frame of date. The Cowell integrator treats its
    frame as inertial and omits precession and nutation. Over the hours-long
    arcs used for validation the frames differ by well under the differences
    being measured; over weeks they do not, so ``compare_to_numerical`` is a
    short-arc instrument and says so in its own result.

  * SGP4 mean elements are not osculating elements. Positions agree; instantaneous
    element values do not, and comparing them directly is a category error.

Every non-zero SGP4 error code raises. The C++ core is happy to return
``(NaN, NaN, NaN)`` beside a non-zero code, and a silently wrong state entering
a validation comparison would make a disagreement look like agreement.

Canonical tickets: P03_05_T053, P03_05_T054, P06_08_T014.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence

from sgp4.api import SGP4_ERRORS, Satrec, jday

from ..domain.temporal import ensure_utc, format_utc

Vector3 = tuple[float, float, float]
State = tuple[_dt.datetime, Vector3, Vector3]

# sgp4 returns kilometres and kilometres per second; this package is SI
# throughout, and the conversion happens once, here, at the boundary.
KM_TO_M = 1000.0

# A TLE is a FIXED-COLUMN format: every field is located by character offset,
# so a line of the wrong length is not "mostly right", it is parsed from the
# wrong columns.
TLE_LINE_LENGTH = 69

# Timestamps must line up before two trajectories can be differenced. One
# millisecond of along-track motion at LEO speed is about 8 m, comfortably
# below the differences these comparisons are built to detect.
TIME_MATCH_TOLERANCE_S = 1.0e-3


class TleFormatError(ValueError):
    """Raised when a two-line element set fails its structural checks."""


class Sgp4PropagationError(RuntimeError):
    """Raised when the SGP4 core reports a non-zero error code."""


def tle_checksum(line: str) -> int:
    """The modulo-10 checksum of the first 68 columns of a TLE line.

    Digits count as themselves, a minus sign counts as one and everything else
    counts as zero — the convention published with the format. This is the only
    integrity check a TLE carries, and it is what turns "a character got
    mangled in transit" into an error instead of a plausible wrong orbit.
    """
    total = 0
    for character in line[: TLE_LINE_LENGTH - 1]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return total % 10


def validate_tle_lines(line1: str, line2: str) -> tuple[str, str]:
    """Check a TLE structurally and return the two lines stripped of line endings.

    ``Satrec.twoline2rv`` does not validate: handed a truncated or corrupted
    line it returns a Satrec full of zeros, which then propagates to a
    satellite sitting at the centre of the Earth or produces a late, opaque
    error code. Validating here means a bad element set is rejected at the
    point it enters the system, with a message naming what is wrong.
    """
    lines: list[str] = []
    for number, raw in ((1, line1), (2, line2)):
        if not isinstance(raw, str):
            raise TleFormatError(f"TLE line {number}: expected str, got {type(raw).__name__}")
        line = raw.rstrip("\r\n")
        if len(line) != TLE_LINE_LENGTH:
            raise TleFormatError(
                f"TLE line {number} is {len(line)} characters, expected {TLE_LINE_LENGTH}. "
                "The format is fixed-column, so a line of the wrong length is read "
                "from the wrong offsets rather than partially understood."
            )
        if line[0] != str(number):
            raise TleFormatError(
                f"TLE line {number} starts with {line[0]!r}, expected {str(number)!r}; "
                "the two lines may be swapped or a title line may have been included"
            )
        expected = int(line[TLE_LINE_LENGTH - 1]) if line[TLE_LINE_LENGTH - 1].isdigit() else None
        actual = tle_checksum(line)
        if expected is None:
            raise TleFormatError(
                f"TLE line {number} has no checksum digit in column {TLE_LINE_LENGTH}"
            )
        if expected != actual:
            raise TleFormatError(
                f"TLE line {number} fails its modulo-10 checksum: the line states "
                f"{expected} but computes {actual}. The line has been corrupted; it is "
                "rejected rather than propagated into a plausible wrong orbit."
            )
        lines.append(line)

    satnum1 = lines[0][2:7].strip()
    satnum2 = lines[1][2:7].strip()
    if satnum1 != satnum2:
        raise TleFormatError(
            f"TLE lines describe different satellites ({satnum1!r} and {satnum2!r}); "
            "two lines from different element sets have been paired"
        )
    return lines[0], lines[1]


def propagate_tle(
    line1: str,
    line2: str,
    times: Sequence[_dt.datetime],
) -> list[State]:
    """Propagate a TLE to each requested instant, in SI metres and m/s.

    Returns ``(time, position_m, velocity_m_s)`` per instant in the TEME frame
    of date. The conversion from the library's kilometres happens here so no
    kilometre-valued state ever escapes into the rest of the package, where
    everything else is SI.

    Raises rather than returning a state whenever SGP4 reports trouble: a
    decayed or diverged element set yields a non-zero code, and the position it
    returns beside that code is meaningless.
    """
    checked1, checked2 = validate_tle_lines(line1, line2)
    if not times:
        raise ValueError("times is empty; there is nothing to propagate")

    satellite = Satrec.twoline2rv(checked1, checked2)
    states: list[State] = []
    for when in times:
        instant = ensure_utc(when, "times")
        seconds = instant.second + instant.microsecond / 1.0e6
        julian_day, fraction = jday(
            instant.year, instant.month, instant.day, instant.hour, instant.minute, seconds
        )
        code, position_km, velocity_km_s = satellite.sgp4(julian_day, fraction)
        if code != 0:
            raise Sgp4PropagationError(
                f"SGP4 failed for satellite {satellite.satnum} at {format_utc(instant)}: "
                f"error {code} — {SGP4_ERRORS.get(code, 'unrecognised SGP4 error code')}. "
                "No state is returned; a state produced alongside an error code is "
                "not a physical prediction."
            )
        components = (*position_km, *velocity_km_s)
        if not all(math.isfinite(value) for value in components):
            raise Sgp4PropagationError(
                f"SGP4 returned a non-finite state for satellite {satellite.satnum} at "
                f"{format_utc(instant)} while reporting success: {components}. "
                "Refusing to pass NaN into a validation comparison."
            )
        states.append(
            (
                instant,
                (
                    position_km[0] * KM_TO_M,
                    position_km[1] * KM_TO_M,
                    position_km[2] * KM_TO_M,
                ),
                (
                    velocity_km_s[0] * KM_TO_M,
                    velocity_km_s[1] * KM_TO_M,
                    velocity_km_s[2] * KM_TO_M,
                ),
            )
        )
    return states


def _magnitude(vector: Vector3) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def compare_to_numerical(
    sgp4_states: Sequence[State],
    numerical_states: Sequence[State],
) -> dict[str, object]:
    """Difference two trajectories sample by sample, in metres.

    The radial statistics are reported separately from the total because the
    two mean different things. Total separation is dominated by ALONG-TRACK
    phasing, which grows roughly linearly with time and reflects a small
    difference in mean motion; radial separation reflects a difference in
    ALTITUDE, which is what drag disagreement actually looks like. A comparison
    that reported only the total would let a large phasing error hide a small
    but real density error, or the reverse.

    Timestamps must correspond. Differencing two trajectories sampled at
    different instants measures the orbit rather than the models, and would
    quietly report a metres-per-millisecond artefact as a model discrepancy.
    """
    if len(sgp4_states) != len(numerical_states):
        raise ValueError(
            f"trajectories have different lengths ({len(sgp4_states)} and "
            f"{len(numerical_states)}); they cannot be differenced sample by sample"
        )
    if not sgp4_states:
        raise ValueError("no samples to compare")

    separations: list[float] = []
    radial_differences: list[float] = []
    times: list[_dt.datetime] = []

    for index, (sgp4_state, numerical_state) in enumerate(
        zip(sgp4_states, numerical_states, strict=True)
    ):
        t_sgp4 = ensure_utc(sgp4_state[0], "sgp4_states time")
        t_numerical = ensure_utc(numerical_state[0], "numerical_states time")
        offset = abs((t_sgp4 - t_numerical).total_seconds())
        if offset > TIME_MATCH_TOLERANCE_S:
            raise ValueError(
                f"sample {index} timestamps differ by {offset:.6f} s "
                f"({format_utc(t_sgp4)} vs {format_utc(t_numerical)}), beyond the "
                f"{TIME_MATCH_TOLERANCE_S} s tolerance. Resample onto a common grid "
                "before comparing."
            )
        difference: Vector3 = (
            sgp4_state[1][0] - numerical_state[1][0],
            sgp4_state[1][1] - numerical_state[1][1],
            sgp4_state[1][2] - numerical_state[1][2],
        )
        separations.append(_magnitude(difference))
        radial_differences.append(_magnitude(sgp4_state[1]) - _magnitude(numerical_state[1]))
        times.append(t_sgp4)

    worst = max(range(len(separations)), key=lambda i: separations[i])
    worst_radial = max(range(len(radial_differences)), key=lambda i: abs(radial_differences[i]))
    samples = len(separations)

    return {
        "samples": samples,
        "start_time_utc": format_utc(times[0]),
        "end_time_utc": format_utc(times[-1]),
        "span_s": (times[-1] - times[0]).total_seconds(),
        "rms_position_difference_m": math.sqrt(
            math.fsum(value * value for value in separations) / samples
        ),
        "mean_position_difference_m": math.fsum(separations) / samples,
        "max_position_difference_m": separations[worst],
        "max_position_difference_time_utc": format_utc(times[worst]),
        "rms_radial_difference_m": math.sqrt(
            math.fsum(value * value for value in radial_differences) / samples
        ),
        "max_radial_difference_m": radial_differences[worst_radial],
        "max_radial_difference_time_utc": format_utc(times[worst_radial]),
        "frame_note": (
            "SGP4 states are TEME of date; the numerical states are treated as "
            "inertial with precession and nutation omitted. Valid as a short-arc "
            "check only — the neglected frame rotation grows with the span."
        ),
    }

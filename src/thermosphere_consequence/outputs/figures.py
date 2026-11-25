"""Figures that keep their provenance when they leave the repository.

A figure is the artefact most likely to be lifted out of its report and pasted
into a slide, an email or a paper, where every caveat that lived in the
surrounding prose is lost. Every function here therefore *requires* a caption,
renders it onto the canvas and writes it into the PNG's own ``Description``
metadata. A blank caption raises rather than defaulting to empty: the point of
the argument is that no figure can leave this module unqualified, and an
optional caption would be silently skipped exactly when it mattered.

The Agg backend is selected programmatically, before ``pyplot`` is imported, so
these functions run under CI, in a container and over SSH. Nothing here calls
``plt.show()``; every figure is closed on the way out, including on the failure
path, because a leaked figure in a long batch run is a memory leak that only
shows up on the largest job.

Nearest canonical ticket: P09_10_T038 (mandatory figure labelling and
determinism substrate).
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from collections.abc import Mapping, Sequence

import matplotlib

# The backend must be fixed before pyplot resolves one, otherwise a machine
# with a display can pick an interactive backend and a headless one can fail at
# import. E402 is unavoidable here and is the documented matplotlib pattern.
matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import __version__  # noqa: E402
from ..domain.temporal import ensure_utc  # noqa: E402
from ..propagation.propagator import REENTRY_ALTITUDE_M, PropagationResult  # noqa: E402

__all__ = [
    "plot_decay_curves",
    "plot_density_comparison",
    "plot_species_composition",
]

#: Pinned so that two runs of the same analysis produce comparable images and a
#: figure regenerated a year later can be diffed against its predecessor.
FIGURE_SIZE_INCHES = (9.0, 5.5)
FIGURE_DPI = 150
CAPTION_FONT_SIZE = 7.0
#: Space reserved at the bottom of the canvas for the caption block. Sized for a
#: caption of several wrapped lines: a caption that collides with the x-axis
#: label is as good as no caption.
CAPTION_BOTTOM_MARGIN = 0.20

_M_PER_KM = 1000.0
_SECONDS_PER_DAY = 86400.0


def _require_caption(caption: str) -> str:
    """Return the trimmed caption, or raise if there is nothing to render.

    Checked before any figure is built so the failure is cheap and no canvas is
    left open.
    """
    text = str(caption).strip()
    if not text:
        raise ValueError(
            "a caption is required: a figure that leaves this repository must "
            "carry its own provenance and caveats, so an empty caption is "
            "rejected rather than treated as 'no caveats apply'."
        )
    return text


def _require_title(title: str) -> str:
    text = str(title).strip()
    if not text:
        raise ValueError("a non-empty title is required; an unlabelled figure states nothing")
    return text


def _finalise(fig: Figure, path: pathlib.Path | str, caption: str, title: str) -> pathlib.Path:
    """Render the caption onto the figure, save as PNG and close the figure.

    The caption is written twice on purpose: visibly, for anyone looking at the
    image, and into the PNG text metadata, for anyone who has only the file. The
    code version goes into the metadata for the same reason — an image found on
    its own should still say which build drew it.
    """
    try:
        fig.subplots_adjust(bottom=CAPTION_BOTTOM_MARGIN)
        fig.text(
            0.01,
            0.01,
            caption,
            fontsize=CAPTION_FONT_SIZE,
            ha="left",
            va="bottom",
            wrap=True,
        )
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            target,
            format="png",
            dpi=FIGURE_DPI,
            metadata={
                "Title": title,
                "Description": caption,
                "Software": f"thermosphere-consequence {__version__}",
            },
        )
    finally:
        plt.close(fig)
    return target


def _elapsed_days(times: Sequence[_dt.datetime], field: str) -> list[float]:
    """Days since the first sample, which is what makes scenarios comparable.

    Absolute UTC on the x-axis would be correct but useless for a counterfactual
    pair: the question is how the two runs diverge from a shared epoch, not what
    the wall-clock date was.
    """
    normalised = [ensure_utc(when, field) for when in times]
    epoch = normalised[0]
    return [(when - epoch).total_seconds() / _SECONDS_PER_DAY for when in normalised]


def plot_decay_curves(
    results: Mapping[str, PropagationResult],
    path: pathlib.Path | str,
    title: str,
    *,
    caption: str,
) -> pathlib.Path:
    """Plot perigee and apogee altitude against time for several scenarios.

    Perigee and apogee share an axes and a colour per scenario, distinguished by
    line style. That is the comparison the physics asks for: drag removes energy
    at perigee and lowers apogee first, so the gap between the two curves closing
    is the circularisation that precedes reentry. Splitting them across two
    panels would hide it.
    """
    caption = _require_caption(caption)
    title = _require_title(title)
    if not results:
        raise ValueError("plot_decay_curves requires at least one scenario to plot")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=FIGURE_DPI)
    try:
        for scenario, result in results.items():
            if not result.times_utc:
                raise ValueError(f"scenario {scenario!r} has no samples to plot")
            if len(result.perigee_altitude_m) != len(result.times_utc) or len(
                result.apogee_altitude_m
            ) != len(result.times_utc):
                raise ValueError(
                    f"scenario {scenario!r}: altitude series do not match the "
                    "time series length; a ragged result cannot be plotted "
                    "without mispairing samples."
                )
            days = _elapsed_days(result.times_utc, f"{scenario}.times_utc")
            line = ax.plot(
                days,
                [value / _M_PER_KM for value in result.perigee_altitude_m],
                linestyle="-",
                linewidth=1.4,
                label=f"{scenario} perigee",
            )[0]
            ax.plot(
                days,
                [value / _M_PER_KM for value in result.apogee_altitude_m],
                linestyle="--",
                linewidth=1.1,
                color=line.get_color(),
                label=f"{scenario} apogee",
            )

        ax.axhline(
            REENTRY_ALTITUDE_M / _M_PER_KM,
            color="0.4",
            linestyle=":",
            linewidth=1.0,
            label=f"reentry threshold ({REENTRY_ALTITUDE_M / _M_PER_KM:.0f} km)",
        )
        ax.set_xlabel("Time since epoch (days)")
        ax.set_ylabel("Geodetic altitude (km)")
        ax.set_title(title)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(fontsize=8, loc="best")
    except Exception:
        plt.close(fig)
        raise
    return _finalise(fig, path, caption, title)


def plot_density_comparison(
    times: Sequence[_dt.datetime],
    quiet: Sequence[float],
    storm: Sequence[float],
    path: pathlib.Path | str,
    *,
    caption: str,
    title: str = "Neutral density: quiet vs storm",
) -> pathlib.Path:
    """Plot two density series and the percentage enhancement between them.

    Density is on a log axis because the quantity spans orders of magnitude with
    altitude and storm state; a linear axis would compress the quiet series into
    the baseline and make the comparison unreadable.

    The enhancement is on a twin linear axis because it answers a different
    question — how much more, in relative terms — and it is the number that gets
    quoted. Non-positive quiet values are rejected rather than skipped: a
    percentage enhancement relative to zero density is undefined, and a silently
    dropped point would leave a gap that looks like missing data.
    """
    caption = _require_caption(caption)
    title = _require_title(title)
    lengths = (len(times), len(quiet), len(storm))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"times, quiet and storm must be the same length, got {lengths}; "
            "mismatched series would pair a storm density with the wrong time."
        )
    if not times:
        raise ValueError("plot_density_comparison requires at least one sample")
    for index, (quiet_value, storm_value) in enumerate(zip(quiet, storm, strict=True)):
        if quiet_value <= 0.0:
            raise ValueError(
                f"quiet[{index}] = {quiet_value} is not positive; density must "
                "be positive to appear on a log axis and to serve as the "
                "denominator of an enhancement ratio."
            )
        if storm_value <= 0.0:
            raise ValueError(f"storm[{index}] = {storm_value} is not positive")

    # date2num plus an explicit UTC locator/formatter rather than handing
    # datetimes straight to plot(): the axis then states the timezone it is
    # drawn in, instead of inheriting whatever the ambient default happens to be.
    stamps = mdates.date2num([ensure_utc(when, "times") for when in times])
    enhancement = [
        (storm_value / quiet_value - 1.0) * 100.0
        for quiet_value, storm_value in zip(quiet, storm, strict=True)
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=FIGURE_DPI)
    try:
        ax.plot(stamps, list(quiet), linestyle="-", linewidth=1.4, color="C0", label="quiet")
        ax.plot(stamps, list(storm), linestyle="-", linewidth=1.4, color="C3", label="storm")
        ax.set_yscale("log")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(r"Mass density (kg m$^{-3}$)")
        ax.set_title(title)
        ax.grid(True, linewidth=0.3, alpha=0.6)

        twin = ax.twinx()
        twin.plot(
            stamps,
            enhancement,
            linestyle=":",
            linewidth=1.2,
            color="0.25",
            label="enhancement",
        )
        twin.set_ylabel("Enhancement over quiet (%)")

        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        ax.legend(handles + twin_handles, labels + twin_labels, fontsize=8, loc="best")

        locator = mdates.AutoDateLocator(tz=_dt.UTC)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=_dt.UTC))
    except Exception:
        plt.close(fig)
        raise
    return _finalise(fig, path, caption, title)


def plot_species_composition(
    altitudes_km: Sequence[float],
    species_densities: Mapping[str, Sequence[float]],
    path: pathlib.Path | str,
    *,
    caption: str,
    title: str = "Species number density vs altitude",
) -> pathlib.Path:
    """Plot number density against altitude, one curve per species, log density.

    Altitude is the y-axis and density the log x-axis, which is the convention in
    the atmospheric literature and keeps the vertical structure — the crossover
    where atomic oxygen overtakes molecular nitrogen — readable as a height.

    Species order follows the mapping, which is the caller's order, so a figure
    regenerated from the same inputs is identical. A species the model does not
    report must be absent from the mapping rather than present as zeros: zero is
    a physical claim, absence is an epistemic one, and only absence is true here.
    Non-positive values are therefore rejected.
    """
    caption = _require_caption(caption)
    title = _require_title(title)
    if not altitudes_km:
        raise ValueError("plot_species_composition requires at least one altitude")
    if not species_densities:
        raise ValueError("plot_species_composition requires at least one species")

    for symbol, series in species_densities.items():
        if not str(symbol).strip():
            raise ValueError("species symbols must be non-empty")
        if len(series) != len(altitudes_km):
            raise ValueError(
                f"species {symbol!r} has {len(series)} values for "
                f"{len(altitudes_km)} altitudes; the series must align."
            )
        for index, value in enumerate(series):
            if value <= 0.0:
                raise ValueError(
                    f"species {symbol!r} value at index {index} is {value}, "
                    "which cannot appear on a log axis. A species the model "
                    "does not report must be omitted from the mapping, not "
                    "supplied as zero."
                )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES, dpi=FIGURE_DPI)
    try:
        for symbol, series in species_densities.items():
            ax.semilogx(list(series), list(altitudes_km), linewidth=1.4, label=symbol)
        ax.set_xlabel(r"Number density (m$^{-3}$)")
        ax.set_ylabel("Altitude (km)")
        ax.set_title(title)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
        ax.legend(fontsize=8, loc="best")
    except Exception:
        plt.close(fig)
        raise
    return _finalise(fig, path, caption, title)

"""Thin HTTP adapter.

Contains no scientific logic. Every route does the same three things: resolve
the arguments, call the library function that already implements the physics,
and wrap the result in a :class:`~thermosphere_consequence.api.models.ProvenanceEnvelope`.
The same library functions back the CLI, the analysis scripts and the notebooks,
so an answer cannot differ between entry points — and a structural test asserts
the absence of arithmetic in this module, which is what makes "thin" checkable
rather than aspirational.

The API is offline by construction; see :data:`OFFLINE_NOTE`. It evaluates
packaged models and serves artefacts already stored in the repository. It never
fetches a space-weather index, never contacts a catalogue and never generates a
report it does not find on disk. A caller who needs current drivers supplies
them in the request, which is also what makes the answer reproducible: the
inputs are in the request body rather than in whatever a remote feed happened to
be serving at the time.

Canonical ticket: P00_02_T048 (thin-adapter discipline, shared with the CLI).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import pathlib
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..aerodynamics.sentman import (
    ACCOMMODATION_SENSITIVITY_RANGE,
    NOMINAL_ACCOMMODATION,
    inverse_ballistic_coefficient,
    species_weighted_drag_coefficient,
)
from ..domain.constants import CONSTANTS
from ..domain.provenance import ValueStatus
from ..domain.species import ALL_SPECIES, UnknownSpeciesError
from ..events import starlink_g4_7 as event
from ..molecular.composition import mean_particle_mass, total_species_mass_density
from ..protocols.atmosphere import DriverState
from ..providers.nrlmsise00 import MODEL_LICENCE, MODEL_REFERENCE, Nrlmsise00Provider
from .models import (
    AtmosphereRequest,
    AtmosphereResponse,
    DragRequest,
    DragResponse,
    EpistemicStatus,
    ProvenanceEnvelope,
)

__all__ = ["OFFLINE_NOTE", "REPORT_PATH", "create_app"]

_LOGGER = logging.getLogger(__name__)

#: Stated at startup and served from /health, because "does this thing phone
#: home?" is the first question an operator running it inside a controlled
#: network has to answer, and the answer should not require reading the source.
OFFLINE_NOTE = (
    "This API performs NO network I/O. It serves models and stored artefacts "
    "only: every response is evaluated in-process from the packaged models or "
    "read from an artefact already on disk. No space-weather feed, catalogue "
    "or other external service is contacted to answer a request, so drivers "
    "must be supplied by the caller."
)

#: Location of the stored analysis report. Resolved from the installed package
#: rather than the working directory, so the endpoint answers the same way
#: whichever directory the server was started from. Absent in an installed
#: wheel, which is a truthful 404 rather than an error.
REPORT_PATH = pathlib.Path(__file__).resolve().parents[3].joinpath(
    "evidence", "analysis_report.json"
)

_VALIDATION_STATUS = 422
_NOT_FOUND_STATUS = 404

_HEALTH_CAVEATS = (
    OFFLINE_NOTE,
    "A healthy process says nothing about the quality of the science it will "
    "serve; read the caveats on each payload for that.",
)

_CONSTANTS_CAVEATS = (
    "Values flagged exact are exact by definition (SI 2019 or WGS-84); the "
    "remainder are measured or adopted and carry the uncertainty of their "
    "revision.",
    "R_E is the IUGG mean radius, used only for scalar-altitude diagnostics. "
    "Propagation uses the geodetic model instead.",
)

_SPECIES_CAVEATS = (
    "Particle masses are derived as exact multiples of the CODATA atomic mass "
    "constant rather than transcribed, so they inherit its uncertainty and "
    "nothing else.",
    "Anomalous oxygen shares the mass of atomic oxygen but is a separate "
    "species: NRLMSISE-00 reports it separately and it participates in the "
    "density-closure diagnostic.",
)

_ATMOSPHERE_SPECIES_SUM_CAVEAT = (
    "mass_density_kg_m3 is the sum over the species the model reported. The "
    "model's own total density is not identical to that sum, and the residual "
    "is the density-closure diagnostic rather than an error to correct away."
)

_DRAG_SOURCES = (
    "Sentman, L.H. (1961), LMSC-448514",
    "Schaaf, S.A. & Chambre, P.L. (1961), Flow of Rarefied Gases, Princeton",
    "Mehta, P.M. et al. (2014), Adv. Space Res. 54:1590",
    "Moe, K. & Moe, M.M. (2005), Planet. Space Sci. 53:793",
    event.SOURCES["cd_convention"],
)

_GROUND_TRUTH_SOURCES = (
    "CelesTrak SATCAT, COSPAR 2022-010 "
    "(https://celestrak.org/satcat/records.php?INTDES=2022-010)",
    event.SOURCES["counts"],
)

_REPORT_CAVEATS = (
    "Served verbatim from the stored artefact. It is a record of the run that "
    "produced it, not a live computation, and may predate the current code.",
    "Regenerate with scripts/run_full_analysis.py before quoting it against a "
    "changed model or changed inputs.",
)

_DRAG_CAVEATS = (
    "Free-molecular closed forms (Sentman plate, Schaaf-Chambre sphere), valid "
    "for Knudsen number much greater than 1. At 210 km the mean free path is of "
    "order a kilometre, so the regime holds; it does not hold through the lower "
    "thermosphere.",
    "Energy accommodation is a scenario parameter, never a measurement. This "
    f"package's nominal is {NOMINAL_ACCOMMODATION} and the sensitivity envelope "
    f"sweeps {ACCOMMODATION_SENSITIVITY_RANGE}.",
    "Weighting is by species mass flux, so the answer belongs to the "
    "composition supplied with it and cannot be reused at another altitude.",
)

JsonObject = dict[str, Any]
ObjectEnvelope = ProvenanceEnvelope[JsonObject]
ListEnvelope = ProvenanceEnvelope[list[JsonObject]]
AtmosphereEnvelope = ProvenanceEnvelope[AtmosphereResponse]
DragEnvelope = ProvenanceEnvelope[DragResponse]



# --- Access control -------------------------------------------------------
#
# Every route is read-only computation over PUBLIC inputs: an atmosphere
# model whose artefact carries no located licence terms (research-only,
# see third_party/licences/NRLMSISE-00-PROVENANCE.md), CC BY 4.0
# geomagnetic indices, and a public satellite catalogue. There is no user
# data, no credential and no mutation anywhere in this surface, so
# authentication protects no secret.
#
# It does protect something else, which is why it exists: POST /atmosphere and
# POST /drag run a physical model per request, so an open deployment is a
# compute-exhaustion surface. Authentication here is therefore about rate
# attribution, not confidentiality, and the docstring says so rather than
# implying the data is sensitive.
#
# DISABLED BY DEFAULT so that a local research run needs no ceremony. Set
# THERMOSPHERE_API_KEY in the environment and every route requires a matching
# X-API-Key header. The decision is the operator's, and it is visible in the
# deployment rather than absent from the code.
API_KEY_ENV_VAR = "THERMOSPHERE_API_KEY"
API_KEY_HEADER = "X-API-Key"


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce the API key when one is configured; otherwise allow.

    Comparison uses hmac.compare_digest rather than ==, so a wrong key cannot
    be recovered a character at a time from response timing.

    Declared async deliberately: FastAPI dispatches SYNC dependencies to a
    threadpool, which requires a running event loop. This application does no
    blocking I/O in its dependencies, so running inline is both correct and
    one hop cheaper — and it keeps the in-process ASGI transport in the tests
    able to step the app coroutine without an event loop, which is what makes
    the "no network I/O" assertion checkable.
    """
    expected = os.environ.get(API_KEY_ENV_VAR)
    if not expected:
        return
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail=(
                f"{API_KEY_HEADER} header is required: this deployment has "
                f"{API_KEY_ENV_VAR} configured."
            ),
        )
    if not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail=f"{API_KEY_HEADER} is not valid.")


def authentication_is_enforced() -> bool:
    """Whether this process will require a key. Reported by /health."""
    return bool(os.environ.get(API_KEY_ENV_VAR))


def create_app() -> FastAPI:
    """Build a fresh application.

    A factory rather than a module-level singleton so that tests, and any
    process that needs two differently configured instances, get isolation
    instead of shared mutable state.
    """
    _LOGGER.info(OFFLINE_NOTE)

    app = FastAPI(
        title="thermosphere-consequence",
        version=__version__,
        description=OFFLINE_NOTE,
    )

    @app.exception_handler(ValueError)
    async def _domain_validation_failed(request: Request, exc: ValueError) -> JSONResponse:
        """Map a domain rejection to 422, preserving the message.

        These are not internal errors. The library raises ValueError precisely
        where an input is physically or dimensionally inadmissible, and the
        message names what was wrong with it, so the message IS the useful part
        of the response and is passed through rather than replaced with a
        sanitised generic string.
        """
        return JSONResponse(status_code=_VALIDATION_STATUS, content={"detail": str(exc)})

    @app.exception_handler(UnknownSpeciesError)
    async def _unknown_species(request: Request, exc: UnknownSpeciesError) -> JSONResponse:
        """Map an unrecognised species symbol to 422.

        Needs its own handler because ``UnknownSpeciesError`` subclasses
        ``KeyError`` rather than ``ValueError`` — a deliberate domain choice, so
        that a species lookup reads as a mapping miss. It is still a client
        error: a request naming a species the registry does not hold must not
        surface as a server fault. ``KeyError`` renders its message with repr
        quoting, so the argument is read directly to preserve the wording the
        domain actually wrote.
        """
        detail = str(exc.args[0]) if exc.args else str(exc)
        return JSONResponse(status_code=_VALIDATION_STATUS, content={"detail": detail})

    @app.exception_handler(FileNotFoundError)
    async def _artefact_missing(request: Request, exc: FileNotFoundError) -> JSONResponse:
        """Map a missing stored artefact to 404 with the operator's next step."""
        return JSONResponse(status_code=_NOT_FOUND_STATUS, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> ObjectEnvelope:
        """Liveness plus the package version the process is actually running."""
        return ObjectEnvelope(
            data={
                "status": "ok",
                "package_version": __version__,
                "offline": OFFLINE_NOTE,
            },
            epistemic_status=EpistemicStatus.OBSERVED,
            sources=["thermosphere_consequence.__version__ of the running process"],
            caveats=list(_HEALTH_CAVEATS),
        )

    @app.get("/constants")
    async def constants(_auth: None = Depends(require_api_key)) -> ObjectEnvelope:
        """The versioned physical constants with their provenance."""
        return ObjectEnvelope(
            data={
                symbol: {
                    "value": constant.value,
                    "unit": constant.unit,
                    "source": constant.source,
                    "source_url": constant.source_url,
                    "revision": constant.revision,
                    "exact": constant.exact,
                    "note": constant.note,
                }
                for symbol, constant in CONSTANTS.items()
            },
            epistemic_status=EpistemicStatus.OBSERVED,
            sources=sorted({constant.source for constant in CONSTANTS.values()}),
            caveats=list(_CONSTANTS_CAVEATS),
        )

    @app.get("/species")
    async def species(_auth: None = Depends(require_api_key)) -> ObjectEnvelope:
        """The neutral species registry with particle masses."""
        return ObjectEnvelope(
            data={
                symbol: {
                    "name": entry.name,
                    "mass_number": entry.mass_number,
                    "mass_kg": entry.mass_kg,
                    "note": entry.note,
                }
                for symbol, entry in ALL_SPECIES.items()
            },
            epistemic_status=EpistemicStatus.DERIVED,
            sources=[CONSTANTS["amu"].source, CONSTANTS["amu"].source_url],
            caveats=list(_SPECIES_CAVEATS),
        )

    @app.post("/atmosphere")
    async def atmosphere(
        payload: AtmosphereRequest,
        _auth: None = Depends(require_api_key),
    ) -> AtmosphereEnvelope:
        """Evaluate NRLMSISE-00 at one point, with the caller's drivers."""
        history = payload.drivers.ap_history
        drivers = DriverState(
            f107=payload.drivers.f107,
            f107a=payload.drivers.f107a,
            ap_daily=payload.drivers.ap_daily,
            ap_history=None if history is None else tuple(history),
            source="supplied in the HTTP request",
            source_url="",
            status=ValueStatus.OPERATOR_SUPPLIED,
        )
        state = Nrlmsise00Provider().state_at(
            payload.time_utc,
            payload.altitude_m,
            payload.latitude_deg,
            payload.longitude_deg,
            drivers,
        )
        caveats = [*state.notes, _ATMOSPHERE_SPECIES_SUM_CAVEAT]
        return AtmosphereEnvelope(
            data=AtmosphereResponse(
                temperature_k=state.temperature_k,
                number_densities=dict(state.number_densities),
                mass_density_kg_m3=total_species_mass_density(state.number_densities),
                mean_particle_mass_kg=mean_particle_mass(state.number_densities),
                provider_name=state.provider_name,
                provider_version=state.provider_version,
                status=state.status.value,
                caveats=caveats,
            ),
            epistemic_status=EpistemicStatus.MODELLED,
            sources=[MODEL_REFERENCE, MODEL_LICENCE],
            caveats=caveats,
        )

    @app.post("/drag")
    async def drag(payload: DragRequest, _auth: None = Depends(require_api_key)) -> DragEnvelope:
        """Composition-weighted free-molecular C_d and the inverse ballistic coefficient."""
        drag_coefficient = species_weighted_drag_coefficient(
            payload.number_densities,
            payload.temperature_k,
            payload.speed_m_s,
            accommodation=payload.accommodation,
            geometry=payload.geometry,
        )
        caveats = list(_DRAG_CAVEATS)
        return DragEnvelope(
            data=DragResponse(
                drag_coefficient=drag_coefficient,
                inverse_ballistic_coefficient_m2_kg=inverse_ballistic_coefficient(
                    payload.mass_kg, payload.area_m2, drag_coefficient
                ),
                geometry=payload.geometry,
                caveats=caveats,
            ),
            epistemic_status=EpistemicStatus.DERIVED,
            sources=list(_DRAG_SOURCES),
            caveats=caveats,
        )

    @app.get("/event/starlink-g4-7")
    async def event_manifest(_auth: None = Depends(require_api_key)) -> ObjectEnvelope:
        """The February 2022 event manifest, with the source of every number."""
        return ObjectEnvelope(
            data={
                "launch_utc": event.LAUNCH_UTC.isoformat(),
                "deployment_utc": event.DEPLOYMENT_UTC.isoformat(),
                "cospar_id_prefix": event.COSPAR_ID_PREFIX,
                "first_norad_id": event.FIRST_NORAD_ID,
                "insertion_perigee_altitude_m": event.INSERTION_PERIGEE_ALTITUDE_M,
                "insertion_apogee_altitude_m": event.INSERTION_APOGEE_ALTITUDE_M,
                "insertion_inclination_deg": event.INSERTION_INCLINATION_DEG,
                "intended_operational_altitude_m": event.INTENDED_OPERATIONAL_ALTITUDE_M,
                "satellites_launched": event.SATELLITES_LAUNCHED,
                "satellites_lost_peer_reviewed": event.SATELLITES_LOST_PEER_REVIEWED,
                "satellites_survived_peer_reviewed": event.SATELLITES_SURVIVED_PEER_REVIEWED,
                "satellites_never_catalogued": event.SATELLITES_NEVER_CATALOGUED,
                "spacex_early_estimate_lost": event.SPACEX_EARLY_ESTIMATE_LOST,
                "mass_kg_range": list(event.MASS_KG_RANGE),
                "mass_kg_nominal": event.MASS_KG_NOMINAL,
                "operator_fitted_inverse_ballistic_coefficient_m2_kg": (
                    event.OPERATOR_FITTED_INVERSE_BALLISTIC_COEFFICIENT_M2_KG
                ),
                "cd_conventional": event.CD_CONVENTIONAL,
                "cd_sensitivity_range": list(event.CD_SENSITIVITY_RANGE),
                "reported_peak_sym_h_nt": event.REPORTED_PEAK_SYM_H_NT,
                "reported_peak_kp": event.REPORTED_PEAK_KP,
                "reported_kp_ge_5_hours": event.REPORTED_KP_GE_5_HOURS,
                "published_enhancement": {
                    label: {"low": low, "high": high, "kind": kind}
                    for label, (low, high, kind) in event.PUBLISHED_ENHANCEMENT.items()
                },
                "sources": dict(event.SOURCES),
            },
            epistemic_status=EpistemicStatus.OBSERVED,
            sources=sorted(set(event.SOURCES.values())),
            caveats=[
                "The operator's early statement and the peer-reviewed count "
                "disagree; both are recorded with their provenance rather than "
                "one being silently preferred.",
                "Spacecraft mass is a third-party estimate, not an operator "
                "disclosure, which is why it is carried as a range.",
                "published_enhancement figures are what results are checked "
                "AGAINST. They are never used as inputs.",
            ],
        )

    @app.get("/event/starlink-g4-7/ground-truth")
    async def ground_truth(_auth: None = Depends(require_api_key)) -> ListEnvelope:
        """Catalogued decay dates: the only observational anchor in the chain."""
        return ListEnvelope(
            data=[
                {
                    "norad_id": decay.norad_id,
                    "object_id": decay.object_id,
                    "decay_date": (
                        None if decay.decay_date is None else decay.decay_date.isoformat()
                    ),
                    "days_to_decay": decay.days_to_decay,
                    "is_storm_loss": decay.is_storm_loss,
                }
                for decay in event.CATALOGUED_DECAYS
            ],
            epistemic_status=EpistemicStatus.OBSERVED,
            sources=list(_GROUND_TRUTH_SOURCES),
            caveats=[
                f"Only {len(event.CATALOGUED_DECAYS)} of the "
                f"{event.SATELLITES_LAUNCHED} launched objects ever received a "
                f"public catalogue number; {event.SATELLITES_NEVER_CATALOGUED} "
                f"were never tracked. Any comparison against this table is a "
                f"comparison against a SUBSET, biased toward objects that "
                f"survived long enough to be catalogued.",
                "The 2023 and 2024 decays reached operational altitude first "
                "and were lost for unrelated reasons; is_storm_loss excludes "
                "them, and folding them in would overstate the event.",
                "A null decay_date means the object was still on orbit at the "
                "time of retrieval, not that it survived the storm window.",
            ],
        )

    @app.get("/report")
    async def report(_auth: None = Depends(require_api_key)) -> ObjectEnvelope:
        """Serve the stored analysis report, or say plainly that there is none."""
        path = REPORT_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"no analysis report at {path}. Run scripts/run_full_analysis.py "
                f"to generate it. This endpoint serves the stored artefact and "
                f"never fabricates one."
            )
        return ObjectEnvelope(
            data=json.loads(path.read_text(encoding="utf-8")),
            epistemic_status=EpistemicStatus.DERIVED,
            sources=[str(path)],
            caveats=list(_REPORT_CAVEATS),
        )

    return app

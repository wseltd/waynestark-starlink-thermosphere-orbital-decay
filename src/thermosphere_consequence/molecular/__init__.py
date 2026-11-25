"""Pure gas-kinetic and composition kernels. No I/O, no HTTP, no external models.

WHAT THIS LAYER IS, stated here because the breadth of what it computes invites
a stronger reading than it can carry. It is **a species-resolved molecular
diagnostic and interpretation layer accompanying an effective scalar
free-molecular drag model**. It resolves the neutral gas by constituent —
number density, mass density, mean particle mass, scale height, thermal speed,
molecular speed ratio, mean free path, Knudsen number, impact flux, impact
energy — and every one of those is genuinely per species.

WHAT IT IS NOT. It is not a species-resolved force closure. The force actually
integrated into the trajectory is the scalar

    F_drag = 0.5 * C_d * rho_total * A * v_rel^2

with ONE effective drag coefficient and the TOTAL mass density. Composition is
not absent from that force, and saying so would be the opposite error:
:func:`~thermosphere_consequence.aerodynamics.sentman.species_weighted_drag_coefficient`
weights the per-species Sentman coefficients by species MASS flux to produce
the effective C_d, so the mixture decides which coefficient the force uses. What
composition does not do is enter the force any other way. At fixed total density,
redistributing mass between species leaves F_drag exactly unchanged — a property
:func:`~thermosphere_consequence.aerodynamics.effective_drag.species_drag_decomposition`
makes explicit rather than hiding.

The per-facet, per-species force VECTOR does exist, in
:mod:`~thermosphere_consequence.aerodynamics.faceted`, and it is run as a
comparison against the scalar model rather than as the propagator's force
model. The closure that would make the applied force itself species- and
facet-resolved is recorded in ``FUTURE_WORK.md`` and is not implemented.

Canonical ticket: P00_02_T002.
"""

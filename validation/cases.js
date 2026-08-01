/**
 * cases.js — the validation cases and their reference data.
 *
 * Three tiers, in decreasing order of how much they prove:
 *
 *   exact       Closed-form solutions the solver must reproduce to numerical
 *               precision: the Joukowski conformal mapping for the inviscid
 *               core, Blasius for the laminar boundary layer, thin-airfoil
 *               theory for the zero-lift angle. A failure here is a bug, full
 *               stop — there is nothing to argue about.
 *
 *   published   Measured section data (Abbott & von Doenhoff, NACA TR-824) and
 *               the values a lag-entrainment viscous panel code returns at the
 *               same conditions. Tolerances are engineering tolerances, and the
 *               numbers themselves carry experimental scatter of a few percent.
 *
 *   invariant   Properties that must hold whatever the numbers are: symmetry,
 *               monotonicity, conservation, and internal consistency between two
 *               routes to the same quantity. These catch the errors that a
 *               tolerance band is too loose to see.
 *
 * Every case reports a quantitative error, not just a pass or a fail, so the CI
 * pipeline can track drift that is still inside tolerance.
 */

/** Reference lift curve for NACA 0012 at Re = 3e6 (Abbott & von Doenhoff). */
export const NACA0012_LIFT = [
  { alpha: 0, cl: 0.0 },
  { alpha: 4, cl: 0.44 },
  { alpha: 8, cl: 0.85 },
  { alpha: 10, cl: 1.05 },
  { alpha: 12, cl: 1.25 },
  { alpha: 14, cl: 1.42 },
  { alpha: 16, cl: 1.55 },
];

/** Reference drag polar for NACA 0012 at Re = 3e6. */
export const NACA0012_DRAG = [
  { alpha: 0, cd: 0.0060 },
  { alpha: 4, cd: 0.0065 },
  { alpha: 8, cd: 0.0090 },
  { alpha: 10, cd: 0.0110 },
];

/** NACA 2412 at Re = 3e6. */
export const NACA2412_LIFT = [
  { alpha: -2.1, cl: 0.0 },
  { alpha: 0, cl: 0.25 },
  { alpha: 4, cl: 0.68 },
  { alpha: 8, cl: 1.08 },
  { alpha: 12, cl: 1.42 },
];

/** Section constants that hold across the linear range. */
export const SECTION_CONSTANTS = {
  '0012': {
    zeroLiftAngle: 0.0,
    zeroLiftTol: 0.05,
    cmQuarterChord: 0.0,
    cmTol: 0.01,
    liftSlopePerDeg: 0.108,
    liftSlopeTol: 0.012,
    aerodynamicCenter: 0.25,
    acTol: 0.03,
  },
  '2412': {
    zeroLiftAngle: -2.1,
    zeroLiftTol: 0.4,
    cmQuarterChord: -0.05,
    cmTol: 0.02,
    liftSlopePerDeg: 0.105,
    liftSlopeTol: 0.015,
    aerodynamicCenter: 0.25,
    acTol: 0.04,
  },
};

/**
 * Transition location on NACA 0012 at Re = 3e6 with Ncrit = 9, upper surface.
 *
 * These are the values an e^n method returns at these conditions rather than
 * measurements — transition location is notoriously sensitive to freestream
 * turbulence and surface finish, so a measured number without its disturbance
 * environment is not a reference. The tolerance is wide accordingly; the point
 * of the case is the *trend*, that transition marches to the leading edge as
 * incidence rises, which is what actually has to be right.
 */
export const NACA0012_TRANSITION = [
  { alpha: 0, x: 0.45, tol: 0.15 },
  { alpha: 4, x: 0.13, tol: 0.08 },
  { alpha: 8, x: 0.03, tol: 0.04 },
];

/**
 * Reynolds number sweep. Absolute drag at low Reynolds number is model-dependent
 * enough that only the trend is asserted: drag must fall monotonically as
 * Reynolds number rises, and by roughly the right factor across the range.
 */
export const REYNOLDS_SWEEP = [1e5, 3e5, 1e6, 3e6, 1e7];

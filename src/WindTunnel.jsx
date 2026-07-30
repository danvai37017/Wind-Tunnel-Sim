/**
 * WindTunnel — interactive 2D wind tunnel.
 *
 * Three engines run side by side, each doing the job it is actually good at:
 *
 *   1. Force calculator — Hess-Smith vortex panel method (aero.js). Solves the
 *      steady potential flow for circulation, C_L and the surface C_p
 *      distribution. Drag follows from Hoerner's empirical bulge formula.
 *   2. Visualiser — D2Q9 Lattice-Boltzmann (TRT collision), which recovers the
 *      2D incompressible Navier-Stokes equations in the low-Mach limit. It
 *      drives the streak animation and nothing else; a browser-scale solve runs
 *      at Re of order 1e3, far below anything a real wing sees.
 *   3. Separator — boundary-layer integral method (aero.js): Thwaites laminar,
 *      Head turbulent, giving the chordwise detachment point x/c.
 *
 * Engines 1 and 3 are steady, exact for their model, and cost about 0.1 ms
 * together, so the dashboard tracks the sliders instantly and is evaluated at
 * the *true* Reynolds number. Engine 2 animates independently at whatever rate
 * the frame budget allows. Airfoil geometry is generated from the NACA 4- and
 * 5-digit series equations and shared by all three.
 *
 * Self-contained: no required props, no external state, no context. Drop in as
 * <WindTunnel /> and it works.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './WindTunnel.module.css';
import {
  buildPanelNodes,
  buildPanelSystem,
  solvePanels,
  separationPoint,
  hoernerDrag,
} from './aero.js';

/* ============================================================================
 * 1. Physical constants & domain configuration
 * ==========================================================================*/

// ISA sea-level air.
const RHO_AIR = 1.225; // kg/m^3
const NU_AIR = 1.48e-5; // m^2/s (kinematic viscosity)

// The 2D solve gives force per metre of span; the wing is extruded to a span of
// SPAN_PER_CHORD chords, so planform area S = span * chord and the reported
// forces are whole-wing Newtons. The span is a control (see the 3D view), and
// the 2D cross-section is the mid-span station, extruded equally either way.
const SPAN_MIN = 1;
const SPAN_MAX = 10;
const SPAN_DEFAULT = 4;

// --- 3D viewport ----------------------------------------------------------
const VIEW3D_W = 840;
const VIEW3D_H = 430;
// Spanwise slices used for the extruded wing mesh are unnecessary (the section
// is constant along span) — one quad per outline edge spans the whole wing.
const WING_OUTLINE_STEP = 4; // sample every Nth outline point for the mesh
const CAM_MIN_EL = -1.45;
const CAM_MAX_EL = 1.45;
const CAM_MIN_ZOOM = 0.35;
const CAM_MAX_ZOOM = 4;
const SPHERE_HEAD_CELLS = 0.9; // head sphere radius, in lattice cells
// Tracer width gain over strict physical scale. Measured: at the default
// framing a projected trail is ~27 px long, so a gain near 1 gives a
// length:width ratio of ~9:1, matching the 2D ribbons. Larger gains make them
// fat blobs rather than streaks — 2.4 gave 3.9:1 and read as dots.
const STREAK_3D_GAIN = 1.15;
// The 2D taper runs almost to zero at the tail, which at 3D scale is well under
// a pixel and makes the streak vanish behind its own head. The 3D taper starts
// from a wider tail so the whole length reads.
const STREAK_3D_TAIL_W = 0.75;
// Perspective would otherwise blow near tracers up into large balls that swamp
// the view when the camera is inside the flow.
const SPHERE_MAX_PX = 4.5;
// Same guard for the ribbon: cap its half-width in pixels.
const RIBBON_MAX_HALF_PX = 3.5;

// "Near stall" threshold: the fraction of the section lift-curve slope below
// which the wing is warned about. Stall itself is slope <= 0.
const NEAR_STALL_SLOPE_FRACTION = 0.2;

// RC-model scale: chord is set in centimetres. The tunnel is a fixed physical
// box and the model scales inside it, so the lattice cell size is the constant
// and the chord is what varies. Sized so the largest model (10 cm) spans 144
// cells, giving a 29.2 cm x 11.1 cm working section.
const CHORD_MIN_CM = 2.0;
const CHORD_MAX_CM = 10.0;
const CHORD_DEFAULT_CM = 6.0;
const CELLS_AT_MAX_CHORD = 144;

const DX = CHORD_MAX_CM / 100 / CELLS_AT_MAX_CHORD; // m per lattice cell
const NX = 420; // domain width in cells
const NY = 160; // domain height in cells
const NXNY = NX * NY;

// Internal physics stays in SI; only the UI speaks centimetres.
const cmToM = (cm) => cm / 100;

// Leading edge sits 100 cells from the inlet at zero AoA; the airfoil rotates
// about the quarter chord, so the pivot is parked 0.25 chords further back.
const LE_X = 100;
const PIVOT_Y = NY / 2;
const nCellsFor = (chord) => Math.round(chord / DX);
const pivotXFor = (nCells) => LE_X + 0.25 * nCells;
// Reference size (1.0 m chord) that the Re_sim mapping is quoted against.
const REF_CELLS = 120;

// Lattice inlet velocity. The ~0.3 ceiling applies to the *local* lattice
// velocity, not just the inlet: tunnel blockage at high AoA (a 120-cell chord
// at 25 deg spans ~40% of the 160-cell tunnel height) accelerates the flow to
// ~3.7x freestream, so the inlet value must be set from that peak. Measured:
// U_LAT = 0.12 reaches local Ma ~0.78 and blows up above ~18 deg; 0.08 keeps
// the peak at Ma ~0.52 and is stable across the whole control envelope.
const U_LAT = 0.08;

// Stability floor for the relaxation time (hard floor is 0.5). Measured: with
// TRT the cliff is sharp and sits between tau+ = 0.52 (10/10 hard cases blow up)
// and 0.53 (0/10). Below it the boundary layer is thinner than the lattice can
// resolve, so this is a grid-resolution limit rather than a collision-operator
// one -- TRT does not move it. 0.55 keeps a comfortable margin.
const TAU_MIN = 0.55;

// TRT "magic" parameter. Lambda = (tau+ - 1/2)(tau- - 1/2) = 3/16 places the
// bounce-back wall exactly halfway between nodes independently of viscosity;
// under BGK that position drifts with tau, biasing the boundary layer and hence
// where it detaches. It also buys a little stability margin at a given tau.
// Both help the separation-point readout, though the flicker that made stall
// look random came from the old per-frame detection, not from the collision
// operator.
const TRT_LAMBDA = 3 / 16;

// Control ranges.
const V_MIN = 5;
const V_MAX = 90;
const AOA_MIN = -10;
const AOA_MAX = 25;

// Re_sim mapping (see the Re_display vs Re_sim note in the UI). The top of the
// range is the highest Reynolds number this grid can actually carry at the
// stability floor, so the whole slider travel produces a visible change in the
// flow instead of saturating in the first sixth and then doing nothing.
const RE_SIM_MIN = 200;
const RE_SIM_MAX = (U_LAT * REF_CELLS) / ((TAU_MIN - 0.5) / 3);

// Rendering.
const SCALE = 2; // canvas pixels per lattice cell
const CANVAS_W = NX * SCALE;
const CANVAS_H = NY * SCALE;
// Streak buffer and molecule count. The max buffer size avoids re-allocating
// when the user slides the molecule count; only activeCount particles are alive.
const STREAK_MAX = 2000;
const STREAK_DEFAULT = 700;

// Streak ribbons: tracer particles advected by the live velocity field, drawn
// as tapered trails. Replaces the old fixed-grid arrow glyphs, which could only
// twitch in place rather than drift.
const STREAK_COUNT = STREAK_MAX;
const STREAK_TRAIL = 16; // stored points per ribbon (upper bound)
const STREAK_TRAIL_MIN = 9; // shortest ribbon, so lengths vary and don't band
const STREAK_SEG_CELLS = 1.9; // lattice cells between stored points
const STREAK_HEAD_W = 2.2; // px half-width at the leading end
const STREAK_TAIL_W = 0.12; // px half-width at the tail
const STREAK_MAX_AGE = 12; // s before a forced respawn
// Glide rate: lattice cells per second per (m/s) of freestream. Strictly
// proportional to airspeed, so the slider changes glide speed accurately;
// the constant only sets the slow-motion factor (30 m/s crosses in ~3 s).
const STREAK_CELLS_PER_MS = 4.7;
const GOLDEN = 0.6180339887; // low-discrepancy spawn spacing

// Simulation pacing. The budget is for the WHOLE frame, not just the solver:
// with two canvases live the rendering is the larger cost (measured ~7 ms for
// the 2D view and ~9 ms for the 3D one), so a solver-only budget would happily
// overshoot the frame. Solver substeps get whatever is left after rendering,
// which also degrades gracefully on slower machines.
const FRAME_BUDGET_MS = 30;
const FRAME_BUDGET_WARMUP_MS = 40; // raised briefly after a reset to settle faster
const WARMUP_FRAMES = 150;
// Measured ~1.7 ms/step on the 420x160 grid, so the steady-state budget lands
// near 7 substeps; the cap is set above that to let the warm-up boost bite.
// The adaptive controller scales this down automatically on slower machines.
const MAX_SUBSTEPS = 12;

// Thin-airfoil lift-curve slope, 2*pi per radian, expressed per degree (0.1097).
const A0_PER_DEG = 2 * Math.PI * (Math.PI / 180);
// Real lift curves round over near the peak, so stall sits slightly beyond the
// linear extrapolation alpha_L0 + Cl_max/a0. Calibrated against published
// section data (NACA 2412 and 0012 both stall near 16 deg at Re 3e6).
const STALL_ROUNDING = 1.12;
// No stall hysteresis: the state is now a pure function of the lift-curve slope
// at the current angle, so it is already deterministic and cannot flicker.
// Real airfoils do show reattachment lag, but adding it here would only make
// the indicator disagree with the dCl/dalpha reading shown next to it.

/* ============================================================================
 * 2. D2Q9 lattice
 * ==========================================================================*/

//  i:    0      1      2      3      4      5      6       7       8
// e_i: (0,0) (1,0) (0,1) (-1,0) (0,-1) (1,1) (-1,1) (-1,-1) (1,-1)
const EX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const EY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
// Opposite direction (bounce-back) and vertical mirror (free-slip walls).
const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6];
const MIRROR_Y = [0, 1, 4, 3, 2, 8, 7, 6, 5];
// Opposite-direction pairs, which TRT collides as symmetric/antisymmetric parts.
const PAIRS = [
  [1, 3],
  [2, 4],
  [5, 7],
  [6, 8],
];

// Field offsets into the SoA distribution arrays: f[DIR_OFF[i] + cellIndex].
const DIR_OFF = new Int32Array(9);
for (let i = 0; i < 9; i++) DIR_OFF[i] = i * NXNY;

// Neighbour cell-index deltas for each direction.
const NB_OFF = new Int32Array(9);
for (let i = 0; i < 9; i++) NB_OFF[i] = EY[i] * NX + EX[i];

// Scratch for post-collision values in the (cold) general-path cell routine.
const FS = new Float64Array(9);

/* ============================================================================
 * 3. NACA geometry — parsing (5.1)
 * ==========================================================================*/

// 5-digit standard camber table, indexed by the second digit P.
const FIVE_DIGIT_TABLE = {
  1: { r: 0.058, k1: 361.4 },
  2: { r: 0.126, k1: 51.64 },
  3: { r: 0.2025, k1: 15.957 },
  4: { r: 0.29, k1: 6.643 },
  5: { r: 0.391, k1: 3.23 },
};

/**
 * Parse a 4- or 5-digit NACA designation into a geometry spec.
 * Returns { ok, error?, warning?, ...spec }.
 */
function parseNacaCode(raw) {
  const cleaned = String(raw ?? '')
    .trim()
    .replace(/^naca[\s-]*/i, '')
    .replace(/\s+/g, '');

  if (cleaned.length === 0) {
    return { ok: false, error: 'Enter a 4- or 5-digit NACA code (e.g. 2412).' };
  }
  if (!/^\d+$/.test(cleaned)) {
    return {
      ok: false,
      error: 'Only digits are allowed (an optional "NACA" prefix is fine).',
    };
  }
  if (cleaned.length !== 4 && cleaned.length !== 5) {
    return {
      ok: false,
      error: `"${cleaned}" has ${cleaned.length} digits — NACA codes must be 4 or 5 (e.g. 2412 or 23012).`,
    };
  }

  const t = Number(cleaned.slice(-2)) / 100;
  if (t <= 0) {
    return {
      ok: false,
      error: 'The last two digits set max thickness (% of chord) and cannot be 00.',
    };
  }

  if (cleaned.length === 4) {
    const m = Number(cleaned[0]) / 100;
    const p = Number(cleaned[1]) / 10;
    if (m > 0 && p <= 0) {
      return {
        ok: false,
        error: 'A cambered 4-digit airfoil needs a non-zero camber position (2nd digit).',
      };
    }
    return {
      ok: true,
      key: cleaned,
      series: 4,
      label: `NACA ${cleaned}`,
      m,
      p,
      t,
      symmetric: m === 0,
    };
  }

  // 5-digit: L P Q XX
  const L = Number(cleaned[0]);
  const P = Number(cleaned[1]);
  const Q = Number(cleaned[2]);
  const row = FIVE_DIGIT_TABLE[P];
  if (!row) {
    return {
      ok: false,
      error: 'For a 5-digit code the 2nd digit (camber position) must be 1–5.',
    };
  }

  let warning;
  if (Q === 1) {
    warning =
      'Reflex camber lines (3rd digit = 1) are not supported — showing the standard (non-reflex) interpretation instead.';
  } else if (Q !== 0) {
    warning =
      'The 3rd digit of a 5-digit code should be 0 or 1 — treating it as 0 (standard camber).';
  }

  // Table k1 values are calibrated for Cl_design = 0.3, i.e. L = 2.
  const k1 = row.k1 * (L / 2);

  return {
    ok: true,
    key: cleaned,
    series: 5,
    label: `NACA ${cleaned}`,
    r: row.r,
    k1,
    clDesign: 0.15 * L,
    t,
    symmetric: L === 0,
    warning,
  };
}

/* ============================================================================
 * 4. NACA geometry — surface generation (5.2 – 5.5)
 * ==========================================================================*/

/** Half-thickness distribution, shared by both series. */
function thickness(x, t) {
  return (
    5 *
    t *
    (0.2969 * Math.sqrt(x) -
      0.126 * x -
      0.3516 * x * x +
      0.2843 * x * x * x -
      0.1015 * x * x * x * x)
  );
}

/** Camber ordinate and slope at x for the given spec. */
function camber(x, spec) {
  if (spec.symmetric) return [0, 0];

  if (spec.series === 4) {
    const { m, p } = spec;
    if (x < p) {
      return [(m / (p * p)) * (2 * p * x - x * x), ((2 * m) / (p * p)) * (p - x)];
    }
    const q = (1 - p) * (1 - p);
    return [(m / q) * (1 - 2 * p + 2 * p * x - x * x), ((2 * m) / q) * (p - x)];
  }

  // 5-digit, standard (non-reflex) camber.
  const { r, k1 } = spec;
  if (x < r) {
    return [
      (k1 / 6) * (x * x * x - 3 * r * x * x + r * r * (3 - r) * x),
      (k1 / 6) * (3 * x * x - 6 * r * x + r * r * (3 - r)),
    ];
  }
  return [((k1 * r * r * r) / 6) * (1 - x), -(k1 * r * r * r) / 6];
}

/* ----------------------------------------------------------------------------
 * Section aerodynamics: zero-lift angle and critical (stall) angle.
 *
 * These are analytic/empirical section properties evaluated at the *true*
 * Reynolds number, deliberately separate from the LBM flow field. A browser-
 * scale solver runs at Re of order 1e3; a real wing at Re of order 1e6. Stall
 * angle is strongly Re-dependent, so reading it off the low-Re solve would
 * report stall many degrees early. See the info panel note.
 * -------------------------------------------------------------------------*/

/**
 * Zero-lift angle of attack, degrees, from thin-airfoil theory:
 *   alpha_L0 = -(1/pi) * integral_0^pi (dy_c/dx)(cos(theta) - 1) d(theta)
 * with x = (1 - cos(theta))/2. Exact for the camber line, so it works for both
 * the 4- and 5-digit series without any table lookup. Positive camber gives a
 * negative alpha_L0 (a cambered section still lifts at zero incidence).
 */
function zeroLiftAngle(spec) {
  if (spec.symmetric) return 0;
  const M = 800; // Simpson panels over theta
  let sum = 0;
  for (let k = 0; k <= M; k++) {
    const theta = (Math.PI * k) / M;
    const x = 0.5 * (1 - Math.cos(theta));
    const dyc = camber(x, spec)[1];
    const g = dyc * (Math.cos(theta) - 1);
    const w = k === 0 || k === M ? 1 : k % 2 ? 4 : 2;
    sum += w * g;
  }
  const integral = ((Math.PI / M) / 3) * sum;
  return (-(integral / Math.PI) * 180) / Math.PI;
}

/** Largest camber ordinate, as a fraction of chord (works for both series). */
function maxCamber(spec) {
  if (spec.symmetric) return 0;
  let best = 0;
  for (let k = 0; k <= 200; k++) {
    const yc = camber(k / 200, spec)[0];
    if (Math.abs(yc) > Math.abs(best)) best = yc;
  }
  return best;
}

/**
 * Maximum section lift coefficient.
 *
 * Empirical correlation, not a first-principles result. Two effects:
 *  - Thickness: Cl_max peaks around t/c ~ 0.13. Thin sections stall early and
 *    abruptly from leading-edge separation; very thick ones lose Cl_max to
 *    early trailing-edge separation.
 *  - Reynolds number: Cl_max falls steeply as Re drops and flattens above
 *    Re ~ 3e6, where published section data is quoted.
 *
 * Calibrated across the RC band (Re 2e4 - 6e5) as well as full scale, since an
 * RC chord of a few centimetres puts the model near Re 1e4-1e5 where sections
 * lose a great deal of Cl_max:
 *   Re 2e4 -> ~0.7,  5e4 -> ~0.9,  1e5 -> ~1.1,  1e6 -> ~1.5,  3e6 -> ~1.7
 */
function maxLiftCoefficient(t, camberMax, re) {
  const d = t - 0.13;
  let base = d < 0 ? 1.7 - 60 * d * d : 1.7 - 12 * d * d;
  base += 4 * Math.abs(camberMax); // camber raises Cl_max
  base = Math.min(2.0, Math.max(0.4, base));

  const reFactor = Math.min(
    1,
    Math.max(0.25, 0.41 + 0.271 * Math.log10(Math.max(re, 1) / 2e4))
  );
  return base * reFactor;
}

/**
 * Lift-curve slope, per degree. Thin-airfoil theory gives 2*pi per radian, but
 * that only holds at high Reynolds number; by RC scale the boundary layer is
 * thick enough to cut the slope appreciably, which pushes the stall angle up
 * for a given Cl_max. Ignoring this put the RC-scale critical angle several
 * degrees too low.
 */
function liftCurveSlope(re) {
  const f = Math.min(1, Math.max(0.6, 0.7 + 0.14 * Math.log10(Math.max(re, 1) / 2e4)));
  return A0_PER_DEG * f;
}

/**
 * Section stall model: zero-lift angle, Cl_max and the critical angles.
 * Stall is taken to occur when |Cl| exceeds Cl_max, which for a cambered
 * section puts the negative-AoA stall much further from zero than the positive
 * one -- exactly as real cambered airfoils behave.
 */
function computeStallModel(spec, re) {
  const alpha0 = zeroLiftAngle(spec);
  const clMax = maxLiftCoefficient(spec.t, maxCamber(spec), re);
  const a0 = liftCurveSlope(re);
  const delta = (STALL_ROUNDING * clMax) / a0;
  return {
    zeroLiftAoa: alpha0,
    clMax,
    liftSlope: a0,
    criticalAoa: alpha0 + delta, // positive-AoA stall (dCl/dalpha = 0)
    criticalAoaNeg: alpha0 - delta, // negative-AoA stall
  };
}

/**
 * Analytic lift curve and its slope at a given angle of attack.
 *
 * Shape: linear at the section slope a0, then rounding over to Cl_max exactly
 * where dCl/dalpha reaches zero. Writing u for the linear extrapolation divided
 * by Cl_max, the rounded region runs from u = knee to u = STALL_ROUNDING, and
 * requiring the slope to fall linearly to zero across it fixes knee = 2 - U (so
 * the slope is continuous at the knee and the curve lands exactly on Cl_max).
 *
 * Returns Cl and dCl/dalpha in per-degree units. The zero-slope point coincides
 * with criticalAoa above, so "stall" and "past the critical angle" are the same
 * statement, just measured directly off the curve.
 */
function liftCurve(alphaDeg, M) {
  const linear = M.liftSlope * (alphaDeg - M.zeroLiftAoa);
  const sign = linear < 0 ? -1 : 1;
  const u = Math.abs(linear) / M.clMax;

  const U = STALL_ROUNDING;
  const knee = 2 - U;

  let g;
  let slopeFactor;
  if (u <= knee) {
    g = u;
    slopeFactor = 1;
  } else if (u < U) {
    const t = (u - knee) / (U - knee);
    g = knee + (1 - knee) * (2 * t - t * t);
    slopeFactor = 1 - t;
  } else {
    g = 1;
    slopeFactor = 0;
  }

  return { cl: sign * M.clMax * g, slope: M.liftSlope * slopeFactor, slopeFactor };
}

/**
 * Stall state from the lift-curve slope, per the definition that stall is where
 * more angle of attack stops buying more lift:
 *   dCl/dalpha <= 0                  -> 'stall'
 *   dCl/dalpha <= 20% of the section slope -> 'near'
 * Expressing the warning as a fraction of the slope (rather than a fixed number
 * of degrees) keeps it correct across airfoils and Reynolds numbers.
 */
function stallState(alphaDeg, M) {
  const { slopeFactor } = liftCurve(alphaDeg, M);
  if (slopeFactor <= 0) return 'stall';
  if (slopeFactor <= NEAR_STALL_SLOPE_FRACTION) return 'near';
  return 'none';
}

/* ============================================================================
 * 3b. Geometry type system and Clark Y data
 * ==========================================================================*/

const GEO_NACA = 'naca';
const GEO_CLARKY = 'clarky';
const GEO_FLATPLATE = 'flatplate';
const GEO_CYLINDER = 'cylinder';

const HEATMAP_NONE = 'none';
const HEATMAP_CP = 'cp';
const HEATMAP_VELOCITY = 'velocity';
const HEATMAP_DYNAMIC_PRESSURE = 'dynamic_pressure';
const HEATMAP_VORTICITY = 'vorticity';

const GEOMETRY_LABELS = {
  [GEO_NACA]: 'NACA Airfoil',
  [GEO_CLARKY]: 'Clark Y',
  [GEO_FLATPLATE]: 'Flat Plate',
  [GEO_CYLINDER]: 'Cylinder',
};

const HEATMAP_LABELS = {
  [HEATMAP_NONE]: 'None',
  [HEATMAP_CP]: 'Pressure Cp',
  [HEATMAP_VELOCITY]: 'Velocity',
  [HEATMAP_DYNAMIC_PRESSURE]: 'Dynamic Pressure',
  [HEATMAP_VORTICITY]: 'Vorticity',
};

/* Clark Y airfoil coordinates (normalised to unit chord).
 * Standard flat-bottom section, widely used in model aircraft. */
const CLARKY_POINTS = [
  { upper: [0, 0], lower: [0, 0] },
  { upper: [0.005, 0.0159], lower: [0.005, -0.0092] },
  { upper: [0.0125, 0.0244], lower: [0.0125, -0.0129] },
  { upper: [0.025, 0.0337], lower: [0.025, -0.0167] },
  { upper: [0.05, 0.0462], lower: [0.05, -0.0196] },
  { upper: [0.075, 0.0541], lower: [0.075, -0.02] },
  { upper: [0.1, 0.0594], lower: [0.1, -0.0196] },
  { upper: [0.15, 0.0667], lower: [0.15, -0.0179] },
  { upper: [0.2, 0.0713], lower: [0.2, -0.0158] },
  { upper: [0.25, 0.0744], lower: [0.25, -0.0133] },
  { upper: [0.3, 0.0764], lower: [0.3, -0.0108] },
  { upper: [0.4, 0.0778], lower: [0.4, -0.0063] },
  { upper: [0.5, 0.0759], lower: [0.5, -0.0033] },
  { upper: [0.6, 0.0703], lower: [0.6, -0.0013] },
  { upper: [0.7, 0.0608], lower: [0.7, 0] },
  { upper: [0.8, 0.0474], lower: [0.8, 0.0013] },
  { upper: [0.9, 0.0304], lower: [0.9, 0.0017] },
  { upper: [1.0, 0], lower: [1.0, 0] },
];

const N_SURFACE = 200; // sample points per surface (cosine-spaced)

/* ---- Colour scale presets for heat maps ---------------------------------- */
const COLOR_SCALE_JET = 'jet';
const COLOR_SCALE_HOT = 'hot';

const COLOR_SCALE_LABELS = {
  [COLOR_SCALE_JET]: 'Jet',
  [COLOR_SCALE_HOT]: 'Hot',
};

/**
 * Map a normalised value t ∈ [0, 1] to a colour using a named scale.
 * Returns an { r, g, b } object with 0-255 byte values.
 */
function heatColor(t, scale = COLOR_SCALE_JET) {
  const tc = Math.max(0, Math.min(1, t));

  if (scale === COLOR_SCALE_HOT) {
    const r = Math.min(255, tc * 510);
    const g = Math.min(255, Math.max(0, tc * 510 - 255));
    const b = Math.max(0, tc * 510 - 510);
    return { r: (r | 0), g: (g | 0), b: (b | 0) };
  }

  // Jet colormap: dark blue → blue → cyan → green → yellow → red
  let r, g, bVal;
  const x = tc;
  if (x < 0.125) {
    r = 0; g = 0; bVal = 0.5 + 4 * x;
  } else if (x < 0.375) {
    r = 0; g = 4 * (x - 0.125); bVal = 1;
  } else if (x < 0.5) {
    r = 0; g = 1; bVal = 1 - 8 * (x - 0.375);
  } else if (x < 0.625) {
    r = 8 * (x - 0.5); g = 1; bVal = 0;
  } else if (x < 0.875) {
    r = 1; g = 1 - 4 * (x - 0.625); bVal = 0;
  } else {
    r = 1; g = 0; bVal = 0;
  }
  return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(bVal * 255) };
}

/**
 * Build the unit-chord upper/lower surfaces for a NACA airfoil.
 */
function buildSurfaces(spec) {
  const xs = new Float64Array(N_SURFACE);
  const upper = new Float64Array(N_SURFACE * 2);
  const lower = new Float64Array(N_SURFACE * 2);

  for (let j = 0; j < N_SURFACE; j++) {
    const beta = (Math.PI * j) / (N_SURFACE - 1);
    const x = 0.5 * (1 - Math.cos(beta));
    xs[j] = x;

    const yt = thickness(x, spec.t);
    const [yc, dyc] = camber(x, spec);
    const theta = Math.atan(dyc);
    const sin = Math.sin(theta);
    const cos = Math.cos(theta);

    upper[2 * j] = x - yt * sin;
    upper[2 * j + 1] = yc + yt * cos; 
    lower[2 * j] = x + yt * sin;
    lower[2 * j + 1] = yc - yt * cos;
  }

  return { xs, upper, lower };
}

/**
 * Interpolate Clark Y data onto the cosine-spaced grid.
 * The Clark Y has an essentially flat lower surface (characteristic of
 * model-aircraft sections), which creates a different C_p distribution
 * from a comparable NACA section.
 */
function buildClarkYSurfaces() {
  const xs = new Float64Array(N_SURFACE);
  const upper = new Float64Array(N_SURFACE * 2);
  const lower = new Float64Array(N_SURFACE * 2);

  const nSrc = CLARKY_POINTS.length;

  for (let j = 0; j < N_SURFACE; j++) {
    const beta = (Math.PI * j) / (N_SURFACE - 1);
    const x = 0.5 * (1 - Math.cos(beta));
    xs[j] = x;

    // Find the straddling Clark Y points and linearly interpolate.
    let i0 = 0;
    let i1 = nSrc - 1;
    for (let k = 0; k < nSrc - 1; k++) {
      if (x >= CLARKY_POINTS[k].upper[0] && x <= CLARKY_POINTS[k + 1].upper[0]) {
        i0 = k;
        i1 = k + 1;
        break;
      }
    }
    const x0 = CLARKY_POINTS[i0].upper[0];
    const x1 = CLARKY_POINTS[i1].upper[0];
    const f = x0 === x1 ? 0 : (x - x0) / (x1 - x0);

    upper[2 * j] = x;
    upper[2 * j + 1] = CLARKY_POINTS[i0].upper[1] + f * (CLARKY_POINTS[i1].upper[1] - CLARKY_POINTS[i0].upper[1]);
    lower[2 * j] = x;
    lower[2 * j + 1] = CLARKY_POINTS[i0].lower[1] + f * (CLARKY_POINTS[i1].lower[1] - CLARKY_POINTS[i0].lower[1]);
  }

  return { xs, upper, lower };
}

/**
 * Build a zero-thickness flat plate from x = 0 to x = 1.
 * Upper and lower surfaces coincide; the panel method receives
 * a vanishingly thin body and the LBM rasteriser gets a line.
 */
function buildFlatPlateSurfaces() {
  const xs = new Float64Array(N_SURFACE);
  const upper = new Float64Array(N_SURFACE * 2);
  const lower = new Float64Array(N_SURFACE * 2);
  const halfT = 0.015;

  for (let j = 0; j < N_SURFACE; j++) {
    const beta = (Math.PI * j) / (N_SURFACE - 1);
    const x = 0.5 * (1 - Math.cos(beta));
    xs[j] = x;
    upper[2 * j] = x;
    upper[2 * j + 1] = halfT;
    lower[2 * j] = x;
    lower[2 * j + 1] = -halfT;
  }

  return { xs, upper, lower };
}

/**
 * Build a circular cylinder / sphere cross-section of unit diameter,
 * centred at (0.5, 0). The "upper" surface is the upper half of the
 * circle and "lower" surface the lower half, both traversed LE -> TE.
 */
function buildCircleSurfaces() {
  const xs = new Float64Array(N_SURFACE);
  const upper = new Float64Array(N_SURFACE * 2);
  const lower = new Float64Array(N_SURFACE * 2);

  for (let j = 0; j < N_SURFACE; j++) {
    const beta = (Math.PI * j) / (N_SURFACE - 1);
    // Map beta = 0..PI to theta = PI..0 so the front (x=0) is at theta=PI
    // and the rear (x=1) is at theta=0.
    const theta = Math.PI * (1 - beta / Math.PI);
    const x = 0.5 + 0.5 * Math.cos(theta);
    const y = 0.5 * Math.sin(theta);
    xs[j] = x;
    upper[2 * j] = x;
    upper[2 * j + 1] = y;
    lower[2 * j] = x;
    lower[2 * j + 1] = -y;
  }

  return { xs, upper, lower };
}

/**
 * Dispatch geometry generation by type. Every geometry type returns
 * the same { xs, upper, lower } contract so the rest of the pipeline
 * (placement, rasterisation, panel method) works unchanged.
 */
function buildGeometry(geoType, spec) {
  switch (geoType) {
    case GEO_CLARKY:
      return buildClarkYSurfaces();
    case GEO_FLATPLATE:
      return buildFlatPlateSurfaces();
    case GEO_CYLINDER:
      return buildCircleSurfaces();
    default:
      return buildSurfaces(spec);
  }
}

/**
 * Return a surfacePoint(x, side) callback for panel-node construction.
 * side = +1 for upper, -1 for lower.
 */
function makeSurfacePointCb(geoType, spec) {
  switch (geoType) {
    case GEO_CLARKY: {
      const nSrc = CLARKY_POINTS.length;
      return (x, side) => {
        let i0 = 0, i1 = nSrc - 1;
        for (let k = 0; k < nSrc - 1; k++) {
          if (x >= CLARKY_POINTS[k].upper[0] && x <= CLARKY_POINTS[k + 1].upper[0]) {
            i0 = k; i1 = k + 1; break;
          }
        }
        const x0 = CLARKY_POINTS[i0].upper[0], x1 = CLARKY_POINTS[i1].upper[0];
        const f = x0 === x1 ? 0 : (x - x0) / (x1 - x0);
        const yu = CLARKY_POINTS[i0].upper[1] + f * (CLARKY_POINTS[i1].upper[1] - CLARKY_POINTS[i0].upper[1]);
        const yl = CLARKY_POINTS[i0].lower[1] + f * (CLARKY_POINTS[i1].lower[1] - CLARKY_POINTS[i0].lower[1]);
        const y = side > 0 ? yu : yl;
        return [x, y];
      };
    }
    case GEO_FLATPLATE:
      return (x, _side) => [x, 0];
    case GEO_CYLINDER:
      return (x, side) => {
        const theta = Math.acos(2 * x - 1);
        const y = 0.5 * Math.sin(theta);
        return [x, side > 0 ? y : -y];
      };
    default:
      return (x, side) => {
        const yt = thickness(x, spec.t);
        const [yc, dyc] = camber(x, spec);
        const th = Math.atan(dyc);
        return [x - side * yt * Math.sin(th), yc + side * yt * Math.cos(th)];
      };
  }
}

/* ============================================================================
 * 5. Placement on the lattice (5.6)
 * ==========================================================================*/

/**
 * Scale the unit-chord airfoil to lattice units, rotate by -AoA about the
 * quarter chord, and translate to the fixed placement point.
 *
 * Rotating the geometry (rather than the inflow) keeps the tunnel x-axis
 * aligned with the freestream, so Fx is drag and Fy is lift with no further
 * rotation. Lattice y points up; the renderer flips it.
 */
function placeAirfoil(surfaces, aoaDeg, nCells) {
  const a = (aoaDeg * Math.PI) / 180;
  const ca = Math.cos(a);
  const sa = Math.sin(a);
  const px = 0.25 * nCells; // quarter-chord pivot, lattice units
  const pivotX = pivotXFor(nCells);

  // R(-a) applied to (X - pivot), then translated onto the placement point.
  const map = (src, dst) => {
    for (let j = 0; j < N_SURFACE; j++) {
      const X = src[2 * j] * nCells - px;
      const Y = src[2 * j + 1] * nCells;
      dst[2 * j] = X * ca + Y * sa + pivotX;
      dst[2 * j + 1] = -X * sa + Y * ca + PIVOT_Y;
    }
  };

  const upper = new Float64Array(N_SURFACE * 2);
  const lower = new Float64Array(N_SURFACE * 2);
  map(surfaces.upper, upper);
  map(surfaces.lower, lower);

  // Closed outline: upper LE -> TE, then lower TE -> LE (dropping the shared LE point).
  const count = N_SURFACE + (N_SURFACE - 1);
  const poly = new Float64Array(count * 2);
  let k = 0;
  for (let j = 0; j < N_SURFACE; j++) {
    poly[k++] = upper[2 * j];
    poly[k++] = upper[2 * j + 1];
  }
  for (let j = N_SURFACE - 1; j >= 1; j--) {
    poly[k++] = lower[2 * j];
    poly[k++] = lower[2 * j + 1];
  }

  return { upper, lower, poly, polyCount: count, xs: surfaces.xs, nCells };
}

function pointInPolygon(px, py, poly, count) {
  let inside = false;
  for (let i = 0, j = count - 1; i < count; j = i++) {
    const xi = poly[2 * i];
    const yi = poly[2 * i + 1];
    const xj = poly[2 * j];
    const yj = poly[2 * j + 1];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Rasterise the outline to a solid mask.
 *
 * Interior fill is a ray-cast point-in-polygon test on lattice nodes. The
 * outline itself is additionally stamped so that very thin sections (e.g. NACA
 * 0001, ~1 cell thick) still produce a connected, watertight body — a leaky
 * mask would let flow pass straight through the airfoil.
 */
function rasterize(geo) {
  const solid = new Uint8Array(NXNY);
  const { poly, polyCount } = geo;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < polyCount; i++) {
    const x = poly[2 * i];
    const y = poly[2 * i + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const x0 = Math.max(1, Math.floor(minX) - 1);
  const x1 = Math.min(NX - 2, Math.ceil(maxX) + 1);
  const y0 = Math.max(1, Math.floor(minY) - 1);
  const y1 = Math.min(NY - 2, Math.ceil(maxY) + 1);

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (pointInPolygon(x, y, poly, polyCount)) solid[y * NX + x] = 1;
    }
  }

  // Stamp the outline (nearest-node line rasterisation).
  for (let i = 0; i < polyCount; i++) {
    const j = (i + 1) % polyCount;
    const ax = poly[2 * i];
    const ay = poly[2 * i + 1];
    const bx = poly[2 * j];
    const by = poly[2 * j + 1];
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) * 2));
    for (let s = 0; s <= steps; s++) {
      const f = s / steps;
      const gx = Math.round(ax + (bx - ax) * f);
      const gy = Math.round(ay + (by - ay) * f);
      if (gx >= 1 && gx <= NX - 2 && gy >= 1 && gy <= NY - 2) solid[gy * NX + gx] = 1;
    }
  }

  // Fluid cells with at least one solid neighbour take the general (bounce-back)
  // path in the solver; everything else uses the branch-free fast path.
  const nearSolid = new Uint8Array(NXNY);
  for (let y = 1; y < NY - 1; y++) {
    for (let x = 1; x < NX - 1; x++) {
      const id = y * NX + x;
      if (solid[id]) continue;
      for (let i = 1; i < 9; i++) {
        if (solid[id + NB_OFF[i]]) {
          nearSolid[id] = 1;
          break;
        }
      }
    }
  }

  return { solid, nearSolid };
}

/* ============================================================================
 * 6. Unit conversion & solver parameters (4.8, 4.9)
 * ==========================================================================*/

function computeParams(vInf, chord) {
  const nCells = nCellsFor(chord);
  const sliderFraction = (vInf - V_MIN) / (V_MAX - V_MIN);

  // Re_sim is an internal, clamped value used only to pick nu_lat / tau. It
  // scales with chord exactly as the true Reynolds number does, so the size
  // slider moves the simulated Re in the same direction as the displayed one.
  // That also makes nu_lat (and therefore tau) independent of model size --
  // otherwise a small chord silently drives tau toward the stability floor.
  const reSimBase = RE_SIM_MIN + sliderFraction * (RE_SIM_MAX - RE_SIM_MIN);
  const reSimRequested = reSimBase * (nCells / REF_CELLS);

  let nuLat = (U_LAT * nCells) / reSimRequested;
  let tau = 3 * nuLat + 0.5;
  if (tau < TAU_MIN) {
    // Raise viscosity (i.e. lower the effective Re_sim) rather than risk instability.
    tau = TAU_MIN;
    nuLat = (tau - 0.5) / 3;
  }
  const reSimEffective = (U_LAT * nCells) / nuLat;

  // TRT: tau+ sets viscosity, tau- follows from the magic parameter.
  const tauMinus = 0.5 + TRT_LAMBDA / (tau - 0.5);

  const dx = DX; // m per cell (fixed: the tunnel is a fixed physical box)

  // The cell count is rounded, so the chord the lattice actually represents is
  // a fraction of a percent off the requested one. Use that effective chord for
  // every derived quantity, otherwise Cl computed in physical units drifts from
  // Cl in lattice units by the same fraction.
  const chordEff = nCells * DX;

  // Re_display is computed from true physical values and is exact.
  const reDisplay = (vInf * chordEff) / NU_AIR;

  return {
    tau,
    tauMinus,
    omegaPlus: 1 / tau,
    omegaMinus: 1 / tauMinus,
    nuLat,
    nCells,
    chord, // as requested by the slider (display)
    chordEff, // as represented on the lattice (physics)
    reSimRequested,
    reSimEffective,
    dx,
    reDisplay,
  };
}

/* ============================================================================
 * 7. LBM solver
 * ==========================================================================*/

function createSolver() {
  return {
    f: new Float32Array(9 * NXNY),
    fNew: new Float32Array(9 * NXNY),
    solid: new Uint8Array(NXNY),
    nearSolid: new Uint8Array(NXNY),
    ux: new Float32Array(NXNY),
    uy: new Float32Array(NXNY),
    spd: new Float32Array(NXNY),
    omegaPlus: 1 / 0.6,
    omegaMinus: 1 / 0.6,
    diverged: false,
  };
}

/** Reset the whole field to the uniform inlet condition (rho = 1, u = (U_LAT, 0)). */
function resetField(S) {
  const { f } = S;
  const usq = U_LAT * U_LAT;
  for (let i = 0; i < 9; i++) {
    const eu = EX[i] * U_LAT;
    const val = W[i] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usq);
    f.fill(val, DIR_OFF[i], DIR_OFF[i] + NXNY);
  }
  S.fNew.set(f);
  S.diverged = false;
}

/**
 * Collide + stream a single cell with full boundary handling.
 * Cold path: domain edges and cells adjacent to the airfoil.
 */
function collideStreamGeneric(S, x, y) {
  const { f, fNew, solid, omegaPlus, omegaMinus } = S;
  const id = y * NX + x;

  let rho = 0;
  for (let i = 0; i < 9; i++) {
    const v = f[DIR_OFF[i] + id];
    FS[i] = v;
    rho += v;
  }
  if (rho <= 0) rho = 1;

  const ux = (FS[1] - FS[3] + FS[5] - FS[6] - FS[7] + FS[8]) / rho;
  const uy = (FS[2] - FS[4] + FS[5] + FS[6] - FS[7] - FS[8]) / rho;
  const usq = 1.5 * (ux * ux + uy * uy);

  // TRT collision. For an opposite pair the equilibrium splits cleanly:
  //   feq_sym  = w*rho*(1 + (e.u)^2/2*9 - 1.5|u|^2)   [even in e]
  //   feq_asym = w*rho*3(e.u)                          [odd in e]
  FS[0] += omegaPlus * (W[0] * rho * (1 - usq) - FS[0]);
  for (let k = 0; k < 4; k++) {
    const a = PAIRS[k][0];
    const b = PAIRS[k][1];
    const eu = 3 * (EX[a] * ux + EY[a] * uy);
    const wr = W[a] * rho;
    const eqSym = wr * (1 + 0.5 * eu * eu - usq);
    const eqAsym = wr * eu;
    const sym = 0.5 * (FS[a] + FS[b]);
    const asym = 0.5 * (FS[a] - FS[b]);
    const dSym = omegaPlus * (sym - eqSym);
    const dAsym = omegaMinus * (asym - eqAsym);
    FS[a] = FS[a] - dSym - dAsym;
    FS[b] = FS[b] - dSym + dAsym;
  }

  for (let i = 0; i < 9; i++) {
    const tx = x + EX[i];
    const ty = y + EY[i];

    if (ty < 0 || ty >= NY) {
      // Free-slip tunnel wall: specular reflection (vertical component flipped,
      // horizontal component preserved), landing back in the same row.
      if (tx < 0 || tx >= NX) continue;
      fNew[DIR_OFF[MIRROR_Y[i]] + y * NX + tx] = FS[i];
      continue;
    }
    if (tx < 0 || tx >= NX) continue; // inlet/outlet BCs overwrite these columns

    const tid = ty * NX + tx;
    if (solid[tid]) {
      // No-slip: full bounce-back.
      fNew[DIR_OFF[OPP[i]] + id] = FS[i];
    } else {
      fNew[DIR_OFF[i] + tid] = FS[i];
    }
  }
}

/** One full LBM timestep: collide -> stream -> boundary conditions. */
function lbmStep(S) {
  const { f, fNew, solid, nearSolid, omegaPlus, omegaMinus } = S;

  const O1 = DIR_OFF[1];
  const O2 = DIR_OFF[2];
  const O3 = DIR_OFF[3];
  const O4 = DIR_OFF[4];
  const O5 = DIR_OFF[5];
  const O6 = DIR_OFF[6];
  const O7 = DIR_OFF[7];
  const O8 = DIR_OFF[8];

  // --- Interior: fused collide + push-stream -------------------------------
  for (let y = 1; y < NY - 1; y++) {
    const row = y * NX;
    for (let x = 1; x < NX - 1; x++) {
      const id = row + x;
      if (solid[id]) continue;
      if (nearSolid[id]) {
        collideStreamGeneric(S, x, y);
        continue;
      }

      const f0 = f[id];
      const f1 = f[O1 + id];
      const f2 = f[O2 + id];
      const f3 = f[O3 + id];
      const f4 = f[O4 + id];
      const f5 = f[O5 + id];
      const f6 = f[O6 + id];
      const f7 = f[O7 + id];
      const f8 = f[O8 + id];

      const rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
      const inv = 1 / rho;
      const ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv;
      const uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv;
      const usq = 1.5 * (ux * ux + uy * uy);

      const wr0 = (4 / 9) * rho;
      const wr1 = (1 / 9) * rho;
      const wr2 = (1 / 36) * rho;

      // f_i^eq = w_i rho [1 + 3(e.u) + 4.5(e.u)^2 - 1.5|u|^2]
      const eu1 = 3 * ux;
      const eu2 = 3 * uy;
      const eu5 = 3 * (ux + uy);
      const eu6 = 3 * (-ux + uy);

      // TRT: collide each opposite pair as symmetric + antisymmetric parts.
      // The symmetric equilibrium is the even-in-e half of f^eq and the
      // antisymmetric one the odd half, so no extra f^eq evaluations are needed.
      const c0 = f0 + omegaPlus * (wr0 * (1 - usq) - f0);

      const s1 = 0.5 * (f1 + f3);
      const a1 = 0.5 * (f1 - f3);
      const p1 = omegaPlus * (s1 - wr1 * (1 + 0.5 * eu1 * eu1 - usq));
      const m1 = omegaMinus * (a1 - wr1 * eu1);
      const c1 = f1 - p1 - m1;
      const c3 = f3 - p1 + m1;

      const s2 = 0.5 * (f2 + f4);
      const a2 = 0.5 * (f2 - f4);
      const p2 = omegaPlus * (s2 - wr1 * (1 + 0.5 * eu2 * eu2 - usq));
      const m2 = omegaMinus * (a2 - wr1 * eu2);
      const c2 = f2 - p2 - m2;
      const c4 = f4 - p2 + m2;

      const s5 = 0.5 * (f5 + f7);
      const a5 = 0.5 * (f5 - f7);
      const p5 = omegaPlus * (s5 - wr2 * (1 + 0.5 * eu5 * eu5 - usq));
      const m5 = omegaMinus * (a5 - wr2 * eu5);
      const c5 = f5 - p5 - m5;
      const c7 = f7 - p5 + m5;

      const s6 = 0.5 * (f6 + f8);
      const a6 = 0.5 * (f6 - f8);
      const p6 = omegaPlus * (s6 - wr2 * (1 + 0.5 * eu6 * eu6 - usq));
      const m6 = omegaMinus * (a6 - wr2 * eu6);
      const c6 = f6 - p6 - m6;
      const c8 = f8 - p6 + m6;

      fNew[id] = c0;
      fNew[O1 + id + 1] = c1;
      fNew[O2 + id + NX] = c2;
      fNew[O3 + id - 1] = c3;
      fNew[O4 + id - NX] = c4;
      fNew[O5 + id + NX + 1] = c5;
      fNew[O6 + id + NX - 1] = c6;
      fNew[O7 + id - NX - 1] = c7;
      fNew[O8 + id - NX + 1] = c8;
    }
  }

  // --- Domain edges --------------------------------------------------------
  for (let x = 0; x < NX; x++) {
    if (!solid[x]) collideStreamGeneric(S, x, 0);
    if (!solid[(NY - 1) * NX + x]) collideStreamGeneric(S, x, NY - 1);
  }
  for (let y = 1; y < NY - 1; y++) {
    if (!solid[y * NX]) collideStreamGeneric(S, 0, y);
    if (!solid[y * NX + NX - 1]) collideStreamGeneric(S, NX - 1, y);
  }

  // --- Swap ----------------------------------------------------------------
  S.f = fNew;
  S.fNew = f;

  const fc = S.f;

  // --- Inlet: Dirichlet velocity (equilibrium at rho = 1, u = (U_LAT, 0)) ---
  const usqIn = U_LAT * U_LAT;
  for (let i = 0; i < 9; i++) {
    const eu = 3 * EX[i] * U_LAT;
    const feq = W[i] * (1 + eu + 0.5 * eu * eu - 1.5 * usqIn);
    const base = DIR_OFF[i];
    for (let y = 0; y < NY; y++) fc[base + y * NX] = feq;
  }

  // --- Outlet: zero-gradient (copy the penultimate column outward) ----------
  for (let i = 0; i < 9; i++) {
    const base = DIR_OFF[i];
    for (let y = 0; y < NY; y++) {
      const row = y * NX;
      fc[base + row + NX - 1] = fc[base + row + NX - 2];
    }
  }
}

/** Read macroscopic u, v and speed off the distributions (4.6). */
function computeMacroscopic(S) {
  const { f, solid, ux, uy, spd } = S;
  let bad = false;
  for (let id = 0; id < NXNY; id++) {
    if (solid[id]) {
      ux[id] = 0;
      uy[id] = 0;
      spd[id] = 0;
      continue;
    }
    const f0 = f[id];
    const f1 = f[DIR_OFF[1] + id];
    const f2 = f[DIR_OFF[2] + id];
    const f3 = f[DIR_OFF[3] + id];
    const f4 = f[DIR_OFF[4] + id];
    const f5 = f[DIR_OFF[5] + id];
    const f6 = f[DIR_OFF[6] + id];
    const f7 = f[DIR_OFF[7] + id];
    const f8 = f[DIR_OFF[8] + id];
    const rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    if (!(rho > 0.1 && rho < 5)) bad = true;
    const inv = 1 / rho;
    const u = (f1 - f3 + f5 - f6 - f7 + f8) * inv;
    const v = (f2 - f4 + f5 + f6 - f7 - f8) * inv;
    ux[id] = u;
    uy[id] = v;
    spd[id] = Math.sqrt(u * u + v * v);
  }
  if (bad) S.diverged = true;
}

/* ============================================================================
 * 8. Separation
 * ==========================================================================*/

// Separation is Engine 3's job (separationPoint in aero.js), evaluated at the
// true Reynolds number. It used to be surveyed off the live LBM field instead,
// scanning the suction surface for sustained flow reversal. That was honest
// about the flow it measured, but the flow it measured was the wrong one: the
// solver runs near Re 1e3, where a real section separates many degrees early, so
// the readout routinely showed detached flow on an airfoil still well below its
// stall angle. It was also unsteady enough to need a 60-frame time average
// before it would sit still. The integral method is steady, exact for its model,
// and evaluated at the Reynolds number the wing actually flies at.

/* ============================================================================
 * 9. Rendering (6.2)
 * ==========================================================================*/

/**
 * Colour by speed relative to the freestream (f = 1 at inlet speed). Most of
 * the field sits near f = 1, so the ramp is deliberately not linear: it keeps
 * the freestream a calm cyan and spends the warm half of the hue range on the
 * accelerated flow over the suction side, where the interesting behaviour is.
 * Slow and reversed flow in the wake goes deep blue.
 */
function speedColor(f) {
  const h =
    f < 0.9
      ? 200 + 40 * Math.min(1, (0.9 - f) / 0.9)
      : 200 - 200 * Math.min(1, (f - 0.9) / 1.3);
  return `hsl(${h.toFixed(0)}, 90%, ${(52 + 8 * Math.min(1, f)).toFixed(0)}%)`;
}

/* ---- Streak ribbons ------------------------------------------------------ */

function createStreaks(activeCount) {
  return {
    activeCount: activeCount ?? STREAK_DEFAULT,
    x: new Float32Array(STREAK_COUNT),
    y: new Float32Array(STREAK_COUNT),
    tx: new Float32Array(STREAK_COUNT * STREAK_TRAIL),
    ty: new Float32Array(STREAK_COUNT * STREAK_TRAIL),
    head: new Uint8Array(STREAK_COUNT),
    count: new Uint8Array(STREAK_COUNT),
    cap: new Uint8Array(STREAK_COUNT),
    sz: new Float32Array(STREAK_COUNT),
    since: new Float32Array(STREAK_COUNT),
    age: new Float32Array(STREAK_COUNT),
    phase: 0,
    seeded: false,
  };
}

/**
 * Place one tracer. Spawning resets the whole trail to the new position,
 * otherwise the ribbon would streak across the screen from wherever it died.
 * Inlet spawns are spread with a golden-ratio sequence, which fills the span
 * far more evenly than uniform random and avoids clumps and gaps.
 */
function spawnStreak(St, i, S, seedAnywhere) {
  St.phase = (St.phase + GOLDEN) % 1;
  let x;
  let y;
  let tries = 0;
  do {
    y = 3 + St.phase * (NY - 6);
    x = seedAnywhere ? 1 + Math.random() * (NX - 3) : 0.5 + Math.random() * 3;
    if (tries++) St.phase = (St.phase + GOLDEN) % 1;
  } while (S.solid[(y | 0) * NX + (x | 0)] && tries < 8);

  St.x[i] = x;
  St.y[i] = y;
  St.age[i] = 0;
  // Stagger the first stored point so ribbons don't all step in lockstep, which
  // otherwise reads as marching columns.
  St.since[i] = Math.random() * STREAK_SEG_CELLS;
  St.head[i] = 0;
  St.count[i] = 1;
  St.cap[i] = STREAK_TRAIL_MIN + ((Math.random() * (STREAK_TRAIL - STREAK_TRAIL_MIN + 1)) | 0);
  St.sz[i] = Math.random() - 0.5;
  St.tx[i * STREAK_TRAIL] = x;
  St.ty[i * STREAK_TRAIL] = y;
}

function resetStreaks(St, S) {
  const n = St.activeCount;
  for (let i = 0; i < n; i++) spawnStreak(St, i, S, true);
  St.seeded = true;
}

/** Bilinear velocity sample — nearest-cell sampling makes the glide jitter. */
function sampleVelocity(S, x, y, out) {
  let x0 = Math.floor(x);
  let y0 = Math.floor(y);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x0 > NX - 2) x0 = NX - 2;
  if (y0 > NY - 2) y0 = NY - 2;
  const fx = x - x0;
  const fy = y - y0;
  const i00 = y0 * NX + x0;
  const i10 = i00 + 1;
  const i01 = i00 + NX;
  const i11 = i01 + 1;
  const w00 = (1 - fx) * (1 - fy);
  const w10 = fx * (1 - fy);
  const w01 = (1 - fx) * fy;
  const w11 = fx * fy;
  out[0] = S.ux[i00] * w00 + S.ux[i10] * w10 + S.ux[i01] * w01 + S.ux[i11] * w11;
  out[1] = S.uy[i00] * w00 + S.uy[i10] * w10 + S.uy[i01] * w01 + S.uy[i11] * w11;
}

const VEL = new Float64Array(2);

/**
 * Advect every tracer by the local velocity for dt seconds of wall clock.
 * Velocity is normalised by U_LAT so a freestream particle glides at exactly
 * vInf * STREAK_CELLS_PER_MS cells/s — i.e. visual speed tracks airspeed.
 */
function advanceStreaks(St, S, vInf, dt) {
  if (!St.seeded) resetStreaks(St, S);

  // Substep so no single move exceeds the trail spacing. Without this a fast
  // freestream advances several cells per frame, trail points end up spaced by
  // the per-frame distance instead of STREAK_SEG_CELLS, and ribbons stretch
  // with airspeed. Substepping also follows curved paths far more accurately.
  const cellsPerFrame = vInf * STREAK_CELLS_PER_MS * dt; // freestream, cells
  const sub = Math.max(1, Math.min(8, Math.ceil((cellsPerFrame * 1.6) / STREAK_SEG_CELLS)));
  const glide = cellsPerFrame / sub / U_LAT; // multiplies raw lattice velocity
  const subDt = dt / sub;

  const nStr = St.activeCount;
  for (let i = 0; i < nStr; i++) {
    for (let s = 0; s < sub; s++) {
      sampleVelocity(S, St.x[i], St.y[i], VEL);
      const dx = VEL[0] * glide;
      const dy = VEL[1] * glide;
      const nx = St.x[i] + dx;
      const ny = St.y[i] + dy;
      St.age[i] += subDt;

      if (
        nx < 0.5 ||
        nx > NX - 1.5 ||
        ny < 1.5 ||
        ny > NY - 2.5 ||
        St.age[i] > STREAK_MAX_AGE ||
        S.solid[(ny | 0) * NX + (nx | 0)]
      ) {
        spawnStreak(St, i, S, false);
        break;
      }

      St.x[i] = nx;
      St.y[i] = ny;
      St.since[i] += Math.sqrt(dx * dx + dy * dy);

      if (St.since[i] >= STREAK_SEG_CELLS) {
        St.since[i] -= STREAK_SEG_CELLS;
        const h = (St.head[i] + 1) % STREAK_TRAIL;
        St.head[i] = h;
        St.tx[i * STREAK_TRAIL + h] = nx;
        St.ty[i * STREAK_TRAIL + h] = ny;
        if (St.count[i] < St.cap[i]) St.count[i]++;
      }
    }
  }
}

// Scratch buffers for ribbon construction (tail -> head order).
const RX = new Float64Array(STREAK_TRAIL);
const RY = new Float64Array(STREAK_TRAIL);

/**
 * Draw each tracer as a tapered ribbon: a filled polygon whose half-width grows
 * from the tail to the leading end, so the wind line reads as moving toward its
 * thicker tip. Filling one polygon per ribbon is far cheaper than stroking each
 * segment at its own width.
 */
function drawStreaks(ctx, St, S) {
  const vMax = U_LAT * 1.9;
  const nStr = St.activeCount;

  for (let i = 0; i < nStr; i++) {
    const n = St.count[i];
    if (n < 3) continue;

    // Unpack the ring buffer oldest -> newest, straight into canvas pixels.
    const base = i * STREAK_TRAIL;
    const h = St.head[i];
    for (let k = 0; k < n; k++) {
      const idx = (h - (n - 1) + k + STREAK_TRAIL * 2) % STREAK_TRAIL;
      RX[k] = St.tx[base + idx] * SCALE;
      RY[k] = (NY - 1 - St.ty[base + idx]) * SCALE;
    }
    // The live position leads the last stored point; extend to it so the ribbon
    // glides continuously instead of advancing in discrete jumps.
    RX[n - 1] = St.x[i] * SCALE;
    RY[n - 1] = (NY - 1 - St.y[i]) * SCALE;

    sampleVelocity(S, St.x[i], St.y[i], VEL);
    const speed = Math.sqrt(VEL[0] * VEL[0] + VEL[1] * VEL[1]);
    ctx.fillStyle = speedColor(speed / U_LAT);

    ctx.beginPath();
    // Forward along the left edge...
    for (let k = 0; k < n; k++) {
      const px = RX[k];
      const py = RY[k];
      const ax = k === 0 ? RX[1] - RX[0] : px - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : py - RY[k - 1];
      const len = Math.hypot(ax, ay) || 1;
      const w = STREAK_TAIL_W + (STREAK_HEAD_W - STREAK_TAIL_W) * (k / (n - 1));
      const ox = (-ay / len) * w;
      const oy = (ax / len) * w;
      if (k === 0) ctx.moveTo(px + ox, py + oy);
      else ctx.lineTo(px + ox, py + oy);
    }
    // ...and back along the right edge.
    for (let k = n - 1; k >= 0; k--) {
      const px = RX[k];
      const py = RY[k];
      const ax = k === 0 ? RX[1] - RX[0] : px - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : py - RY[k - 1];
      const len = Math.hypot(ax, ay) || 1;
      const w = STREAK_TAIL_W + (STREAK_HEAD_W - STREAK_TAIL_W) * (k / (n - 1));
      ctx.lineTo(px + (ay / len) * w, py - (ax / len) * w);
    }
    ctx.closePath();
    ctx.fill();
  }
}

function renderFrame(ctx, field, fieldCtx, fieldImage, S, geo, St, opts = {}) {
  const { spd, solid, ux, uy } = S;
  const data = fieldImage.data;

  // Background: speed magnitude.
  const vMax = U_LAT * 1.9;
  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const id = y * NX + x;
      const p = ((NY - 1 - y) * NX + x) * 4;
      if (solid[id]) {
        data[p] = 22;
        data[p + 1] = 26;
        data[p + 2] = 34;
        data[p + 3] = 255;
        continue;
      }
      const v = Math.min(1, spd[id] / vMax);
      data[p] = 10 + 46 * v;
      data[p + 1] = 16 + 62 * v;
      data[p + 2] = 30 + 92 * v;
      data[p + 3] = 255;
    }
  }
  fieldCtx.putImageData(fieldImage, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(field, 0, 0, CANVAS_W, CANVAS_H);

  if (S && St) drawStreaks(ctx, St, S);

  // Geometry outline (airfoil / body).
  if (geo) {
    ctx.beginPath();
    for (let i = 0; i < geo.polyCount; i++) {
      const x = geo.poly[2 * i] * SCALE;
      const y = (NY - 1 - geo.poly[2 * i + 1]) * SCALE;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = '#e8edf5';
    ctx.fill();
    ctx.strokeStyle = '#9fb0c9';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

/* ============================================================================
 * 9b. 3D view
 *
 * The solve is strictly 2D, so the 3D view is that solution extruded along the
 * span: every spanwise station sees the identical cross-section and the
 * identical (x, y) flow. That is exactly the assumption the force readouts
 * already make (uniform section, no tip effects, S = span * chord), so the two
 * views are the same model drawn two ways rather than two different models.
 * The plane z = 0 is precisely what the 2D canvas above shows.
 *
 * Rendered with a small software projector on a 2D canvas — no 3D library and
 * no extra files. Occlusion is painter's algorithm over a single merged list of
 * wing quads and tracers, so tracers behind the wing are correctly hidden.
 * ==========================================================================*/

function createCamera() {
  return {
    az: 0.62, // radians, orbit about the vertical axis
    el: 0.32, // radians, elevation
    zoom: 1,
    dragging: false,
    lastX: 0,
    lastY: 0,
  };
}

/**
 * Build the per-frame projection constants for an orbit camera looking at the
 * wing centre. World axes: x streamwise, y up, z spanwise (all in lattice
 * cells, matching the 2D field).
 */
function makeProjection(cam, target, spanCells) {
  // Frame the larger of the tunnel length and the span.
  const fit = Math.max(NX, spanCells) * 0.88;
  return {
    tx: target[0],
    ty: target[1],
    tz: target[2],
    ca: Math.cos(cam.az),
    sa: Math.sin(cam.az),
    ce: Math.cos(cam.el),
    se: Math.sin(cam.el),
    dist: fit / cam.zoom,
    focal: VIEW3D_H * 1.15,
    cx: VIEW3D_W / 2,
    cy: VIEW3D_H / 2,
    near: 1,
  };
}

// Scratch: [screenX, screenY, depth]. Depth is camera-space distance.
const PRJ = new Float64Array(3);

function project(P3, x, y, z) {
  const px = x - P3.tx;
  const py = y - P3.ty;
  const pz = z - P3.tz;
  // Yaw about the vertical axis, then pitch.
  const X = P3.ca * px + P3.sa * pz;
  const Zy = -P3.sa * px + P3.ca * pz;
  // Positive elevation lifts the camera above the wing, so a point higher in y
  // must get *closer* (smaller depth), not farther.
  const Y = P3.ce * py + P3.se * Zy;
  const Z = -P3.se * py + P3.ce * Zy;
  const depth = Z + P3.dist;
  if (depth <= P3.near) {
    PRJ[2] = -1;
    return false;
  }
  const f = P3.focal / depth;
  PRJ[0] = P3.cx + X * f;
  PRJ[1] = P3.cy - Y * f;
  PRJ[2] = depth;
  return true;
}

// Reusable draw list so the render loop doesn't allocate every frame. DRAW is a
// pool of item objects; ORDER holds the ones used this frame and is the array
// actually sorted, with its backing store reused.
const DRAW = [];
const ORDER = [];
let drawLen = 0;
function pushDraw(depth, kind, index, shade, col) {
  if (drawLen === DRAW.length) DRAW.push({ depth: 0, kind: 0, index: 0, shade: 0, col: '' });
  const d = DRAW[drawLen++];
  d.depth = depth;
  d.kind = kind;
  d.index = index;
  d.shade = shade;
  d.col = col;
  ORDER.push(d);
}
const byDepthDesc = (a, b) => b.depth - a.depth;

// Projected wing-mesh vertices, rebuilt each frame. Each outline point yields
// two projected vertices: one at each end of the span.
const WING_CAP = 512;
let WING_N = 0;
const WING_IDX = new Int32Array(WING_CAP);
const NEAR_X = new Float64Array(WING_CAP);
const NEAR_Y = new Float64Array(WING_CAP);
const NEAR_D = new Float64Array(WING_CAP);
const FAR_X = new Float64Array(WING_CAP);
const FAR_Y = new Float64Array(WING_CAP);
const FAR_D = new Float64Array(WING_CAP);
const WING_SHADE = new Float64Array(WING_CAP);
const WING_OK = new Uint8Array(WING_CAP);
const WING_HR = new Float64Array(WING_CAP);
const WING_HG = new Float64Array(WING_CAP);
const WING_HB = new Float64Array(WING_CAP);
const SLICE_OK = new Uint8Array(WING_CAP);

/** Faint wireframe of the flow domain, for spatial orientation. */
function drawDomainBox(ctx, P3, spanCells) {
  const hz = spanCells / 2;
  const corners = [
    [0, 0, -hz], [NX, 0, -hz], [NX, NY, -hz], [0, NY, -hz],
    [0, 0, hz], [NX, 0, hz], [NX, NY, hz], [0, NY, hz],
  ];
  const pts = corners.map((c) => {
    const ok = project(P3, c[0], c[1], c[2]);
    return ok ? [PRJ[0], PRJ[1]] : null;
  });
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  ctx.strokeStyle = 'rgba(120, 145, 180, 0.16)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [a, b] of edges) {
    if (!pts[a] || !pts[b]) continue;
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
  }
  ctx.stroke();
}

/* ---- 3D heat-map helpers ------------------------------------------------- */

/**
 * Build per-vertex Cp values by interpolating the panel solution onto
 * the outline polygon vertices.
 */
function buildCpHeatValues(out, geo, panelSol) {
  const nPanels = panelSol.n;
  const nPts = geo.polyCount;
  const nPerSurf = nPanels / 2; // 80 panels per surface

  // Panel order: lower surface TE→LE (i=0..79), upper surface LE→TE (i=80..159).
  // Build SEPARATE interpolation arrays for each surface so Cp values don't
  // bleed across the chord.

  // Upper surface panels (i=80..159): xFrac already goes 0 (LE) → 1 (TE).
  const upperX = new Float64Array(nPerSurf);
  const upperCp = new Float64Array(nPerSurf);
  for (let i = 0; i < nPerSurf; i++) {
    upperX[i] = i / (nPerSurf - 1);
    upperCp[i] = panelSol.cp[nPerSurf + i];
  }

  // Lower surface panels (i=0..79): xFrac goes 1 (TE) → 0 (LE).
  // Reverse so it goes 0 (LE) → 1 (TE) for interpolation.
  const lowerX = new Float64Array(nPerSurf);
  const lowerCp = new Float64Array(nPerSurf);
  for (let i = 0; i < nPerSurf; i++) {
    lowerX[i] = i / (nPerSurf - 1);
    lowerCp[i] = panelSol.cp[nPerSurf - 1 - i];
  }

  // Poly order: upper LE→TE (indices 0..N_SURFACE-1),
  //             lower TE→LE (indices N_SURFACE..2*N_SURFACE-2).
  for (let j = 0; j < nPts; j++) {
    if (j < N_SURFACE) {
      // Upper surface vertex: LE→TE, xFrac from 0→1.
      const xFrac = j / (N_SURFACE - 1);
      out[j] = interpolateFromArray(upperX, upperCp, xFrac);
    } else {
      // Lower surface vertex: TE→LE, map so xFrac goes 1 (TE) → 0 (LE).
      const idx = j - N_SURFACE; // 0..N_SURFACE-2
      const xFrac = 1 - idx / (N_SURFACE - 1);
      out[j] = interpolateFromArray(lowerX, lowerCp, xFrac);
    }
  }
}

function interpolateFromArray(xArr, yArr, x) {
  if (x <= xArr[0]) return yArr[0];
  if (x >= xArr[xArr.length - 1]) return yArr[yArr.length - 1];
  let lo = 0, hi = xArr.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xArr[mid] <= x) lo = mid;
    else hi = mid;
  }
  const f = (x - xArr[lo]) / (xArr[hi] - xArr[lo]);
  return yArr[lo] + f * (yArr[hi] - yArr[lo]);
}

/**
 * Build per-vertex velocity / dynamic-pressure / vorticity values by
 * sampling the LBM field at each outline vertex.
 */
function buildLbmHeatValues(out, geo, S, mode) {
  const nPts = geo.polyCount;
  const half = NX * NY;
  for (let j = 0; j < nPts; j++) {
    const px = geo.poly[2 * j];
    const py = geo.poly[2 * j + 1];
    // Sample 3 cells away from the surface along the outward direction
    // to avoid the no-slip boundary layer where velocity is near zero.
    const cx = Math.round(px);
    const cy = Math.round(py);
    let ix = cx, iy = cy;
    // Search outward for a non-solid cell (up to 5 cells).
    for (let d = 1; d <= 5; d++) {
      let found = false;
      for (let dy = -d; dy <= d && !found; dy++) {
        for (let dx = -d; dx <= d && !found; dx++) {
          const tx = cx + dx, ty = cy + dy;
          if (tx < 0 || tx >= NX || ty < 0 || ty >= NY) continue;
          const tid = ty * NX + tx;
          if (!S.solid[tid]) { ix = tx; iy = ty; found = true; }
        }
      }
      if (found) break;
    }
    const id = iy * NX + ix;
    if (id >= 0 && id < half && !S.solid[id]) {
      const u = S.ux[id];
      const v = S.uy[id];
      const speed = Math.sqrt(u * u + v * v);
      if (mode === HEATMAP_VELOCITY) {
        out[j] = speed / U_LAT;
      } else if (mode === HEATMAP_DYNAMIC_PRESSURE) {
        out[j] = 0.5 * 1.225 * speed * speed;
      } else if (mode === HEATMAP_VORTICITY) {
        out[j] = computeVorticityAt(S, ix, iy);
      } else {
        out[j] = speed / U_LAT;
      }
    } else {
      out[j] = 0;
    }
  }
}

/**
 * Estimate vorticity at a lattice cell using central differences.
 */
function computeVorticityAt(S, x, y) {
  if (x < 1 || x >= NX - 1 || y < 1 || y >= NY - 1) return 0;
  const dvdx = (S.uy[(y) * NX + (x + 1)] - S.uy[(y) * NX + (x - 1)]) * 0.5;
  const dudy = (S.ux[(y + 1) * NX + (x)] - S.ux[(y - 1) * NX + (x)]) * 0.5;
  return dvdx - dudy;
}

/**
 * Draw a semi-transparent wake volume behind the geometry.
 */


/**
 * Draw the extruded wing and the tracers into the 3D viewport.
 * `St.sz` gives each tracer its spanwise station; everything else comes from
 * the same state the 2D view renders.
 */
function render3D(ctx, S, geo, St, cam, spanCells, opts = {}) {
  const {
    heatmapMode: hmMode = HEATMAP_NONE,
    colorScale: hmScale = COLOR_SCALE_JET,
    panelSolution: hmPanel = null,
  } = opts;

  ctx.fillStyle = '#0a101e';
  ctx.fillRect(0, 0, VIEW3D_W, VIEW3D_H);
  if (!geo) return;

  const hz = spanCells / 2;
  const target = [pivotXFor(geo.nCells), PIVOT_Y, 0];
  const P3 = makeProjection(cam, target, spanCells);

  drawDomainBox(ctx, P3, spanCells);

  drawLen = 0;
  ORDER.length = 0;

  // --- Wing: one quad per outline edge, spanning the full wing -------------
  WING_N = 0;
  for (let i = 0; i < geo.polyCount && WING_N < WING_CAP; i += WING_OUTLINE_STEP) {
    WING_IDX[WING_N++] = i;
  }

  // Light direction used for shading (only when heat map is off).
  const LX = -0.35;
  const LY = 0.82;

  // Pre-compute per-vertex heat map values if a heat map mode is active.
  let heatVal = null;
  if (hmMode !== HEATMAP_NONE) {
    heatVal = new Float64Array(geo.polyCount);
    if (hmMode === HEATMAP_CP && hmPanel) {
      buildCpHeatValues(heatVal, geo, hmPanel);
    } else {
      buildLbmHeatValues(heatVal, geo, S, hmMode);
    }
  }

  // Build quads with either heat map or diffuse-lighting colouring.
  // Clamp Cp to a physically meaningful range for subsonic flow.
  let heatMin = Infinity, heatMax = -Infinity;
  if (heatVal) {
    const useClamp = hmMode === HEATMAP_CP;
    const clampLo = -2.0, clampHi = 1.1;
    for (let k = 0; k < WING_N; k++) {
      const i = WING_IDX[k];
      let v = heatVal[i];
      if (useClamp) v = Math.max(clampLo, Math.min(clampHi, v));
      if (v < heatMin) heatMin = v;
      if (v > heatMax) heatMax = v;
    }
    if (heatMax - heatMin < 1e-10) { heatMax = heatMin + 1; }
  }

  const projectSlice = (z, outX, outY, outD, outOk) => {
    for (let k = 0; k < WING_N; k++) {
      const i = WING_IDX[k];
      const ax = geo.poly[2 * i], ay = geo.poly[2 * i + 1];
      outOk[k] = project(P3, ax, ay, z) ? 1 : 0;
      outX[k] = PRJ[0]; outY[k] = PRJ[1]; outD[k] = PRJ[2];
    }
  };

  const emitQuads = (nX, nY, nD, nOk, fX, fY, fD, fOk) => {
    for (let k = 0; k < WING_N; k++) {
      const k2 = (k + 1) % WING_N;
      if (!nOk[k] || !nOk[k2] || !fOk[k] || !fOk[k2]) continue;
      const i = WING_IDX[k];
      const j = WING_IDX[k2];
      const ex = geo.poly[2 * j] - geo.poly[2 * i];
      const ey = geo.poly[2 * j + 1] - geo.poly[2 * i + 1];
      const el = Math.hypot(ex, ey) || 1;
      const diffuse = Math.max(0, (-ey / el) * LX + (ex / el) * LY);
      const baseShade = 0.22 + 0.78 * diffuse;
      let col;
      if (heatVal) {
        let va = heatVal[i], vb = heatVal[j];
        if (hmMode === HEATMAP_CP) { va = Math.max(-2, Math.min(1.1, va)); vb = Math.max(-2, Math.min(1.1, vb)); }
        const t = (va + vb) / 2;
        const tn = (t - heatMin) / (heatMax - heatMin);
        const c = heatColor(tn, hmScale);
        const blend = 0.7 + 0.3 * baseShade;
        col = `rgb(${(c.r * blend) | 0},${(c.g * blend) | 0},${(c.b * blend) | 0})`;
        WING_HR[k] = (c.r * blend) | 0;
        WING_HG[k] = (c.g * blend) | 0;
        WING_HB[k] = (c.b * blend) | 0;
        WING_SHADE[k] = baseShade;
      } else {
        col = null;
        WING_SHADE[k] = baseShade;
      }
      pushDraw((nD[k] + nD[k2] + fD[k] + fD[k2]) * 0.25, 0, k, heatVal ? 0 : WING_SHADE[k], col);
    }
  };

  // Single near/far extrusion.
  projectSlice(-hz, NEAR_X, NEAR_Y, NEAR_D, WING_OK);
  projectSlice(hz, FAR_X, FAR_Y, FAR_D, SLICE_OK);
  emitQuads(NEAR_X, NEAR_Y, NEAR_D, WING_OK, FAR_X, FAR_Y, FAR_D, SLICE_OK);

  // End caps.
  let heatEndCapCol = '';
  if (heatVal) {
    let sumR = 0, sumG = 0, sumB = 0, count = 0;
    for (let k = 0; k < WING_N; k++) {
      if (WING_OK[k]) { sumR += WING_HR[k]; sumG += WING_HG[k]; sumB += WING_HB[k]; count++; }
    }
    if (count > 0) {
      heatEndCapCol = `rgb(${(sumR / count) | 0},${(sumG / count) | 0},${(sumB / count) | 0})`;
    }
  }
  let capNearD = 0, capFarD = 0;
  for (let k = 0; k < WING_N; k++) {
    capNearD += NEAR_D[k]; capFarD += FAR_D[k];
  }
  pushDraw(capNearD / WING_N, 2, 0, heatVal ? 0 : 0.55);
  pushDraw(capFarD / WING_N, 2, 1, heatVal ? 0 : 0.55);

  // --- Tracers ------------------------------------------------------------
  // Cull tracers whose head projects outside the viewport (with a margin for
  // the trail that follows it) before they reach the sort or the draw loop.
  const margin = 60;
  const nStr3 = St.activeCount;
  for (let i = 0; i < nStr3; i++) {
    if (St.count[i] < 3) continue;
    const z = St.sz[i] * spanCells;
    if (!project(P3, St.x[i], St.y[i], z)) continue;
    if (
      PRJ[0] < -margin ||
      PRJ[0] > VIEW3D_W + margin ||
      PRJ[1] < -margin ||
      PRJ[1] > VIEW3D_H + margin
    ) {
      continue;
    }
    pushDraw(PRJ[2], 1, i, 0);
  }

  // --- Depth sort and draw back to front ----------------------------------
  ORDER.sort(byDepthDesc);

  for (let n = 0; n < ORDER.length; n++) {
    const item = ORDER[n];

    if (item.kind === 0) {
      const k = item.index;
      const k2 = (k + 1) % WING_N;
      const sh = item.shade;
      const col = item.col || `rgb(${(232 * sh) | 0},${(237 * sh) | 0},${(245 * sh) | 0})`;
      ctx.fillStyle = col;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(NEAR_X[k], NEAR_Y[k]);
      ctx.lineTo(NEAR_X[k2], NEAR_Y[k2]);
      ctx.lineTo(FAR_X[k2], FAR_Y[k2]);
      ctx.lineTo(FAR_X[k], FAR_Y[k]);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      continue;
    }

    if (item.kind === 2) {
      const useNear = item.index === 0;
      const sh = item.shade;
      let col;
      if (heatVal) {
        col = heatEndCapCol;
      } else {
        col = `rgb(${(232 * sh) | 0},${(237 * sh) | 0},${(245 * sh) | 0})`;
      }
      ctx.fillStyle = col;
      ctx.beginPath();
      let moved = false;
      for (let k = 0; k < WING_N; k++) {
        if (!WING_OK[k]) continue;
        const px = useNear ? NEAR_X[k] : FAR_X[k];
        const py = useNear ? NEAR_Y[k] : FAR_Y[k];
        if (!moved) {
          ctx.moveTo(px, py);
          moved = true;
        } else ctx.lineTo(px, py);
      }
      if (moved) {
        ctx.closePath();
        ctx.fill();
      }
      continue;
    }

    // Tracer: tapered trail ending in a small sphere at the leading end.
    const i = item.index;
    const n2 = St.count[i];
    const base = i * STREAK_TRAIL;
    const h = St.head[i];
    const z = St.sz[i] * spanCells;

    sampleVelocity(S, St.x[i], St.y[i], VEL);
    const speed = Math.sqrt(VEL[0] * VEL[0] + VEL[1] * VEL[1]);
    const colour = speedColor(speed / U_LAT);

    // Perspective scale for widths. STREAK_HEAD_W/TAIL_W are 2D canvas pixels
    // at SCALE px per cell, so dividing by SCALE converts them to lattice cells
    // before projecting — otherwise the ribbons come out twice as fat as the 2D
    // ones. Capped because a tracer close to the camera would otherwise be
    // projected into a huge wedge that swamps the view.
    const persp = P3.focal / item.depth;
    const wScale = Math.min(
      (persp / SCALE) * STREAK_3D_GAIN,
      RIBBON_MAX_HALF_PX / STREAK_HEAD_W
    );

    // Project the whole trail up front. Doing it inline and bailing out midway
    // on a failed projection would leave a half-built path that closePath()
    // turns into a spurious triangle.
    let ok = true;
    for (let k = 0; k < n2; k++) {
      const idx = (h - (n2 - 1) + k + STREAK_TRAIL * 2) % STREAK_TRAIL;
      const wx = k === n2 - 1 ? St.x[i] : St.tx[base + idx];
      const wy = k === n2 - 1 ? St.y[i] : St.ty[base + idx];
      if (!project(P3, wx, wy, z)) {
        ok = false;
        break;
      }
      RX[k] = PRJ[0];
      RY[k] = PRJ[1];
    }
    if (!ok) continue;

    ctx.fillStyle = colour;
    ctx.beginPath();
    for (let k = 0; k < n2; k++) {
      const ax = k === 0 ? RX[1] - RX[0] : RX[k] - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : RY[k] - RY[k - 1];
      const dl = Math.hypot(ax, ay) || 1;
      const w = (STREAK_3D_TAIL_W + (STREAK_HEAD_W - STREAK_3D_TAIL_W) * (k / (n2 - 1))) * wScale;
      const ox = (-ay / dl) * w;
      const oy = (ax / dl) * w;
      if (k === 0) ctx.moveTo(RX[k] + ox, RY[k] + oy);
      else ctx.lineTo(RX[k] + ox, RY[k] + oy);
    }
    for (let k = n2 - 1; k >= 0; k--) {
      const ax = k === 0 ? RX[1] - RX[0] : RX[k] - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : RY[k] - RY[k - 1];
      const dl = Math.hypot(ax, ay) || 1;
      const w = (STREAK_3D_TAIL_W + (STREAK_HEAD_W - STREAK_3D_TAIL_W) * (k / (n2 - 1))) * wScale;
      ctx.lineTo(RX[k] + (ay / dl) * w, RY[k] - (ax / dl) * w);
    }
    ctx.closePath();
    ctx.fill();

    // Head sphere: a small shaded ball marking where the parcel is now.
    if (project(P3, St.x[i], St.y[i], z)) {
      const r = Math.min(SPHERE_MAX_PX, Math.max(0.9, SPHERE_HEAD_CELLS * persp * STREAK_3D_GAIN));
      const hx = PRJ[0];
      const hy = PRJ[1];
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(hx, hy, r, 0, Math.PI * 2);
      ctx.fill();
      // A single small off-centre highlight reads as a sphere far more cheaply
      // than a radial gradient, which would mean allocating one gradient object
      // per tracer per frame.
      if (r > 1.8) {
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.beginPath();
        ctx.arc(hx - r * 0.3, hy - r * 0.3, r * 0.38, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

}

/* ============================================================================
 * 10. Component
 * ==========================================================================*/

export default function WindTunnel({
  initialNacaCode = '2412',
  initialAirspeed = 30,
  initialAoa = 5,
  initialChordCm = CHORD_DEFAULT_CM,
  onReadingsChange,
}) {
  const [code, setCode] = useState(initialNacaCode);
  const [airspeed, setAirspeed] = useState(initialAirspeed);
  const [aoa, setAoa] = useState(initialAoa);
  const [chordCm, setChordCm] = useState(initialChordCm);
  const [spanRatio, setSpanRatio] = useState(SPAN_DEFAULT);
  const [showInfo, setShowInfo] = useState(false);
  const [activeTab, setActiveTab] = useState('2d');
  const [moleculeCount, setMoleculeCount] = useState(STREAK_DEFAULT);
  const [geoType, setGeoType] = useState(GEO_NACA);
  const [heatmapMode, setHeatmapMode] = useState(HEATMAP_NONE);
  const [colorScale, setColorScale] = useState(COLOR_SCALE_JET);
  const [showHeatmapInfo, setShowHeatmapInfo] = useState(false);

  const parsed = useMemo(() => parseNacaCode(code), [code]);
  const params = useMemo(
    () => computeParams(airspeed, cmToM(chordCm)),
    [airspeed, chordCm]
  );

  // For NACA geometry, parse the code. For other types, create a spec-like
  // object that's compatible with the existing machinery.
  const [spec, setSpec] = useState(() => {
    const p = parseNacaCode(initialNacaCode);
    return p.ok ? p : parseNacaCode('2412');
  });
  useEffect(() => {
    if (geoType === GEO_NACA) {
      if (parsed.ok) setSpec(parsed);
    } else {
      // Non-NACA geometries don't use the NACA spec; set a dummy symmetric
      // thin section so thickness/camber calls don't break.
      setSpec({ ok: true, key: '0000', series: 4, label: GEOMETRY_LABELS[geoType], m: 0, p: 0, t: 0.01, symmetric: true });
    }
  }, [parsed, geoType]);

  // Section aerodynamics: evaluated at the true Reynolds number, independent of
  // the solver's internal Re_sim.
  const stallModel = useMemo(
    () => (geoType === GEO_NACA || geoType === GEO_CLARKY
      ? computeStallModel(spec, params.reDisplay)
      : { clMax: 1e6, criticalAoa: 90, zeroLiftAoa: 0, clDesign: 0, liftSlope: 0.1 }),
    [spec, params.reDisplay, geoType]
  );

  /* --- Engines 1 and 3: steady section aerodynamics ----------------------- */

  const panels = useMemo(() => {
    const surfacePoint = makeSurfacePointCb(geoType, spec);
    const { X, Y } = buildPanelNodes(surfacePoint);
    return buildPanelSystem(X, Y);
  }, [spec, geoType]);

  const panelSolution = useMemo(
    () => (panels ? solvePanels(panels, aoa) : null),
    [panels, aoa]
  );

  // C_L at zero incidence sets where the drag bucket bottoms out: zero for a
  // symmetric section, near the design C_L for a cambered one.
  const clAtZeroAoa = useMemo(
    () => (panels ? solvePanels(panels, 0).cl : 0),
    [panels]
  );

  const separation = useMemo(
    () =>
      panels && panelSolution
        ? separationPoint(panels, panelSolution, params.reDisplay)
        : { x: -1, mode: 'attached', transitionX: -1 },
    [panels, panelSolution, params.reDisplay]
  );

  /* --- The dashboard ------------------------------------------------------
   * Every readout is now a pure function of the controls, so it is derived
   * during render rather than sampled out of the animation loop. That is what
   * makes the panel exact and instant: it updates on the same commit as the
   * slider, not on the next throttled frame, and it no longer re-renders React
   * 12 times a second to show numbers that have not changed. */
  const readings = useMemo(() => {
    const M = stallModel;
    const curve = liftCurve(aoa, M);
    const state = stallState(aoa, M);

    const cl = panelSolution ? panelSolution.cl : 0;
    const cd = hoernerDrag(cl, clAtZeroAoa, spec.t, params.reDisplay);

    // The 2D section is extruded to the full span, so S = span * chord and the
    // reported forces are whole-wing Newtons. C_L itself is span-invariant.
    const span = spanRatio * params.chordEff;
    const planform = span * params.chordEff;
    const q = 0.5 * RHO_AIR * airspeed * airspeed;

    return {
      cl,
      cd,
      airspeed,
      reynolds: params.reDisplay,
      reSim: params.reSimEffective,
      liftForce: cl * q * planform,
      dragForce: cd * q * planform,
      dynamicPressure: q,
      stallState: state, // 'none' | 'near' | 'stall'
      stalling: state === 'stall',
      liftSlope: curve.slope, // dCl/dalpha, per degree
      slopeFraction: curve.slopeFactor, // as a fraction of the section slope
      chordCm: params.chord * 100,
      spanCm: span * 100,
      spanRatio,
      planformCm2: planform * 1e4,
      criticalAoa: M.criticalAoa,
      zeroLiftAoa: M.zeroLiftAoa,
      clMax: M.clMax,
      separationPoint: separation.x,
      separationMode: separation.mode, // 'attached' | 'laminar' | 'turbulent' | 'leading-edge'
      transitionPoint: separation.transitionX,
      // Engine 1 is inviscid, so past the critical angle it keeps integrating a
      // potential flow the real wing has already lost. Flagged rather than
      // clamped: the panel value is still the exact answer to the question the
      // panel method asks, and the viscous stall model is shown right next to it.
      clIsInviscid: state === 'stall',
    };
  }, [
    stallModel,
    aoa,
    panelSolution,
    clAtZeroAoa,
    spec.t,
    params,
    airspeed,
    spanRatio,
    separation,
  ]);

  const readingsCbRef = useRef(onReadingsChange);
  readingsCbRef.current = onReadingsChange;
  useEffect(() => {
    if (readingsCbRef.current) readingsCbRef.current(readings);
  }, [readings]);

  const canvasRef = useRef(null);
  const solverRef = useRef(null);
  const geoRef = useRef(null);
  const paramsRef = useRef(computeParams(initialAirspeed, cmToM(initialChordCm)));
  const airspeedRef = useRef(initialAirspeed);
  const resetPendingRef = useRef(true);
  const warmupRef = useRef(WARMUP_FRAMES);
  const substepsRef = useRef(2);
  // Smoothed per-frame cost of everything except the solver.
  const nonSolverEmaRef = useRef(12);

  const streaksRef = useRef(null);
  const canvas3dRef = useRef(null);
  const camRef = useRef(null);
  const spanRatioRef = useRef(spanRatio);
  spanRatioRef.current = spanRatio;
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;
  const heatmapModeRef = useRef(heatmapMode);
  heatmapModeRef.current = heatmapMode;
  const colorScaleRef = useRef(colorScale);
  colorScaleRef.current = colorScale;
  const panelSolutionRef = useRef(panelSolution);
  panelSolutionRef.current = panelSolution;
  if (solverRef.current === null) solverRef.current = createSolver();
  if (streaksRef.current === null) streaksRef.current = createStreaks(moleculeCount);
  if (camRef.current === null) camRef.current = createCamera();

  // --- Geometry: rebuild + full field reset on geoType, AoA or size change ---
  useEffect(() => {
    const surfaces = buildGeometry(geoType, spec);
    const geo = placeAirfoil(surfaces, aoa, nCellsFor(cmToM(chordCm)));
    geo.geoType = geoType;
    const { solid, nearSolid } = rasterize(geo);

    const S = solverRef.current;
    S.solid = solid;
    S.nearSolid = nearSolid;
    geoRef.current = geo;
    resetPendingRef.current = true;
    warmupRef.current = WARMUP_FRAMES;
    streaksRef.current.seeded = false;
  }, [geoType, spec, aoa, chordCm]);

  // --- Airspeed: retune tau / dt only; the flow adapts continuously ---------
  useEffect(() => {
    paramsRef.current = params;
    airspeedRef.current = airspeed;
    solverRef.current.omegaPlus = params.omegaPlus;
    solverRef.current.omegaMinus = params.omegaMinus;
  }, [params, airspeed]);

  // --- Molecule count: spawn or retire tracers -------------------------------
  const prevMoleculeCount = useRef(moleculeCount);
  useEffect(() => {
    const St = streaksRef.current;
    const S = solverRef.current;
    if (!St || !S) return;
    const old = prevMoleculeCount.current;
    St.activeCount = moleculeCount;
    if (moleculeCount > old) {
      for (let i = old; i < moleculeCount; i++) spawnStreak(St, i, S, true);
    }
    prevMoleculeCount.current = moleculeCount;
  }, [moleculeCount]);

  // --- Simulation + render loop --------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');

    const field = document.createElement('canvas');
    field.width = NX;
    field.height = NY;
    const fieldCtx = field.getContext('2d');
    const fieldImage = fieldCtx.createImageData(NX, NY);

    const ctx3d = canvas3dRef.current ? canvas3dRef.current.getContext('2d') : null;

    let raf = 0;
    let lastFrame = performance.now();

    const loop = (now) => {
      raf = requestAnimationFrame(loop);
      const S = solverRef.current;
      const St = streaksRef.current;

      // Real elapsed time drives the tracers, so they glide at a rate set by
      // airspeed rather than by however fast the solver happens to be running.
      // Clamped so a stalled tab doesn't teleport every ribbon on resume.
      const dt = Math.min(0.05, Math.max(0, (now - lastFrame) / 1000));
      lastFrame = now;

      if (resetPendingRef.current) {
        resetField(S);
        resetPendingRef.current = false;
      }
      if (S.diverged) {
        // Defensive: the tau clamp should prevent this, but never render NaNs.
        resetField(S);
      }

      S.omegaPlus = paramsRef.current.omegaPlus;
      S.omegaMinus = paramsRef.current.omegaMinus;

      // Run as many LBM steps as the frame budget allows once rendering is paid
      // for, then adapt for the next frame.
      const budget = warmupRef.current > 0 ? FRAME_BUDGET_WARMUP_MS : FRAME_BUDGET_MS;
      if (warmupRef.current > 0) warmupRef.current--;

      const frameStart = performance.now();
      const n = substepsRef.current;
      for (let s = 0; s < n; s++) lbmStep(S);
      const solveMs = performance.now() - frameStart;

      computeMacroscopic(S);
      if (!St.seeded) resetStreaks(St, S);
      advanceStreaks(St, S, airspeedRef.current, dt);
      renderFrame(ctx, field, fieldCtx, fieldImage, S, geoRef.current, St, {});

      if (ctx3d && geoRef.current && activeTabRef.current === '3d') {
        render3D(
          ctx3d,
          S,
          geoRef.current,
          St,
          camRef.current,
          spanRatioRef.current * geoRef.current.nCells,
          {
            heatmapMode: heatmapModeRef.current,
            colorScale: colorScaleRef.current,
            panelSolution: panelSolutionRef.current,
          }
        );
      }

      // No readout work here any more: the dashboard is derived from the
      // controls during render (Engines 1 and 3), so this loop is purely the
      // visualiser and never touches React state.

      // Everything after the solver — macroscopics, tracers, both canvases — is
      // fixed cost for the frame. Track it and give the solver the remainder.
      const nonSolverMs = performance.now() - frameStart - solveMs;
      nonSolverEmaRef.current += (nonSolverMs - nonSolverEmaRef.current) * 0.1;
      const perStep = solveMs / n;
      const room = budget - nonSolverEmaRef.current;
      const want = perStep > 0 ? Math.floor(room / perStep) : MAX_SUBSTEPS;
      substepsRef.current = Math.max(1, Math.min(MAX_SUBSTEPS, want));
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // Started on mount, cancelled on unmount. Airspeed is read through a ref
    // inside the loop, so the loop itself never needs to be torn down.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- 3D orbit controls ----------------------------------------------------
  // Camera state lives in a ref and is read by the render loop, so dragging
  // never triggers a React render.
  useEffect(() => {
    const el = canvas3dRef.current;
    if (!el) return undefined;
    const cam = camRef.current;

    const pos = (e) => {
      const r = el.getBoundingClientRect();
      // The canvas is CSS-scaled, so convert to backing-store pixels.
      return [((e.clientX - r.left) / r.width) * VIEW3D_W, ((e.clientY - r.top) / r.height) * VIEW3D_H];
    };

    const onDown = (e) => {
      const [x, y] = pos(e);
      cam.dragging = true;
      cam.lastX = x;
      cam.lastY = y;
      if (el.setPointerCapture) el.setPointerCapture(e.pointerId);
    };
    const onMove = (e) => {
      if (!cam.dragging) return;
      const [x, y] = pos(e);
      cam.az -= (x - cam.lastX) * 0.006;
      cam.el += (y - cam.lastY) * 0.006;
      if (cam.el < CAM_MIN_EL) cam.el = CAM_MIN_EL;
      if (cam.el > CAM_MAX_EL) cam.el = CAM_MAX_EL;
      cam.lastX = x;
      cam.lastY = y;
    };
    const onUp = (e) => {
      cam.dragging = false;
      if (el.releasePointerCapture && e.pointerId !== undefined) {
        try {
          el.releasePointerCapture(e.pointerId);
        } catch {
          /* pointer already released */
        }
      }
    };
    const onWheel = (e) => {
      e.preventDefault();
      cam.zoom *= e.deltaY < 0 ? 1.1 : 1 / 1.1;
      if (cam.zoom < CAM_MIN_ZOOM) cam.zoom = CAM_MIN_ZOOM;
      if (cam.zoom > CAM_MAX_ZOOM) cam.zoom = CAM_MAX_ZOOM;
    };

    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('pointercancel', onUp);
      el.removeEventListener('wheel', onWheel);
    };
  }, []);

  const resetView = useCallback(() => {
    const c = camRef.current;
    const fresh = createCamera();
    c.az = fresh.az;
    c.el = fresh.el;
    c.zoom = fresh.zoom;
  }, []);

  const onCodeChange = useCallback((e) => setCode(e.target.value), []);

  return (
    <div className={styles.root}>
      <div className={styles.tabBar} role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === '2d'}
          className={`${styles.tabButton} ${activeTab === '2d' ? styles.tabButtonActive : ''}`}
          onClick={() => setActiveTab('2d')}
        >
          2D
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === '3d'}
          className={`${styles.tabButton} ${activeTab === '3d' ? styles.tabButtonActive : ''}`}
          onClick={() => setActiveTab('3d')}
        >
          3D
        </button>
        {activeTab === '3d' && (
          <button type="button" className={styles.resetViewButton} onClick={resetView}>
            Reset view
          </button>
        )}
      </div>

      <div className={styles.topBar}>
        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-geometry">
            Geometry
          </label>
          <select
            id="wt-geometry"
            className={styles.select}
            value={geoType}
            onChange={(e) => setGeoType(e.target.value)}
          >
            {Object.entries(GEOMETRY_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </div>

        <div className={`${styles.control} ${geoType !== GEO_NACA ? styles.controlHidden : ''}`}>
          <label className={styles.label} htmlFor="wt-naca">
            NACA code
          </label>
          <input
            id="wt-naca"
            className={`${styles.textInput} ${!parsed.ok && geoType === GEO_NACA ? styles.textInputError : ''}`}
            value={code}
            onChange={onCodeChange}
            spellCheck={false}
            autoComplete="off"
            placeholder="2412"
            aria-invalid={!parsed.ok && geoType === GEO_NACA}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-speed">
            Airspeed <span className={styles.labelValue}>{airspeed} m/s</span>
          </label>
          <input
            id="wt-speed"
            className={styles.slider}
            type="range"
            min={V_MIN}
            max={V_MAX}
            step={1}
            value={airspeed}
            onChange={(e) => setAirspeed(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-aoa">
            Angle of attack <span className={styles.labelValue}>{aoa.toFixed(1)}°</span>
          </label>
          <input
            id="wt-aoa"
            className={styles.slider}
            type="range"
            min={AOA_MIN}
            max={AOA_MAX}
            step={0.5}
            value={aoa}
            onChange={(e) => setAoa(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-chord">
            Size (chord) <span className={styles.labelValue}>{chordCm.toFixed(1)} cm</span>
          </label>
          <input
            id="wt-chord"
            className={styles.slider}
            type="range"
            min={CHORD_MIN_CM}
            max={CHORD_MAX_CM}
            step={0.1}
            value={chordCm}
            onChange={(e) => setChordCm(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-span">
            Span{' '}
            <span className={styles.labelValue}>
              {spanRatio.toFixed(1)}× = {(spanRatio * chordCm).toFixed(1)} cm
            </span>
          </label>
          <input
            id="wt-span"
            className={styles.slider}
            type="range"
            min={SPAN_MIN}
            max={SPAN_MAX}
            step={0.1}
            value={spanRatio}
            onChange={(e) => setSpanRatio(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-molecules">
            Molecules{' '}
            <span className={styles.labelValue}>{moleculeCount}</span>
          </label>
          <input
            id="wt-molecules"
            className={styles.slider}
            type="range"
            min={50}
            max={STREAK_MAX}
            step={50}
            value={moleculeCount}
            onChange={(e) => setMoleculeCount(Number(e.target.value))}
          />
        </div>
      </div>

      {/* --- Visualisation toggles (collapsible bar) ------------------------- */}
      <div className={styles.toggleBar}>
      </div>

      {activeTab === '3d' && (
        <>
          <div className={styles.topBar}>
            <div className={styles.control}>
              <label className={styles.label} htmlFor="wt-heatmap">Heat map</label>
              <select
                id="wt-heatmap"
                className={styles.selectSmall}
                value={heatmapMode}
                onChange={(e) => setHeatmapMode(e.target.value)}
              >
                {Object.entries(HEATMAP_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>
            </div>

            {heatmapMode !== HEATMAP_NONE && (
              <div className={styles.control}>
                <label className={styles.label} htmlFor="wt-colorscale">Scale</label>
                <select
                  id="wt-colorscale"
                  className={styles.selectSmall}
                  value={colorScale}
                  onChange={(e) => setColorScale(e.target.value)}
                >
                  {Object.entries(COLOR_SCALE_LABELS).map(([k, v]) => (
                    <option key={k} value={k}>{v}</option>
                  ))}
                </select>
              </div>
            )}

            <button
              type="button"
              className={styles.heatmapInfoIcon}
              onClick={() => setShowHeatmapInfo((v) => !v)}
              aria-label="Heat map term definitions"
              title="What do these terms mean?"
            >
              <span className={styles.circleI}>i</span>
            </button>
          </div>

          {showHeatmapInfo && (
            <div className={styles.heatmapInfo}>
              <div className={styles.heatmapInfoHeader}>
                Heat Map Terms
                <button type="button" className={styles.heatmapInfoClose} onClick={() => setShowHeatmapInfo(false)}>✕</button>
              </div>
              <dl className={styles.heatmapInfoBody}>
                <dt>Pressure C<sub>p</sub></dt>
                <dd>
                  Pressure coefficient: <em>C<sub>p</sub> = (p − p<sub>∞</sub>) / q<sub>∞</sub></em>.
                  Negative values (suction) appear in blue; positive values (higher pressure) in red.
                  Large pressure differences between upper and lower surfaces generate lift.
                </dd>
                <dt>Velocity</dt>
                <dd>
                  Local flow speed normalised by the freestream. Values above 1 indicate
                  acceleration (e.g. over the leading edge), values below 1 indicate deceleration
                  (e.g. in the wake or boundary layer). Derived from the LBM velocity field.
                </dd>
                <dt>Dynamic Pressure</dt>
                <dd>
                  <em>q = ½ ρ v²</em>, the kinetic energy per unit volume of the flow. High
                  dynamic pressure coincides with high-velocity regions; low values mark
                  separated or stagnant flow. Computed from the LBM velocity with ρ = 1.225 kg/m³.
                </dd>
                <dt>Vorticity</dt>
                <dd>
                  <em>ω = ∂v/∂x − ∂u/∂y</em>, a measure of local rotation in the flow.
                  High vorticity marks shear layers, the wake, and boundary-layer activity.
                  Estimated by central differences on the LBM velocity field.
                </dd>
                <dt>Jet (colour scale)</dt>
                <dd>
                  Blue → cyan → green → yellow → red, from low to high values. The
                  standard multi-purpose CFD colour map.
                </dd>
                <dt>Hot (colour scale)</dt>
                <dd>
                  Black → red → orange → yellow → white, from low to high values.
                  Emphasises high-value regions with bright warm colours.
                </dd>
              </dl>
              <p className={styles.heatmapInfoNote}>
                Heat map values are sampled from the LBM field or the panel-method solution and
                interpolated onto the 3D wing surface. They update every frame.
              </p>
            </div>
          )}
        </>
      )}

      {geoType === GEO_NACA && !parsed.ok && <div className={styles.error}>{parsed.error}</div>}
      {geoType === GEO_NACA && parsed.ok && parsed.warning && <div className={styles.warning}>{parsed.warning}</div>}

      <div className={`${styles.body} ${activeTab === '2d' ? '' : styles.tabHidden}`}>
        <div className={styles.viewport}>
          <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} className={styles.canvas} />

          <StallBanner state={readings.stallState} />

          <div className={styles.viewportFooter}>
            <span className={styles.airfoilName}>{spec.label}</span>
            <button
              type="button"
              className={styles.infoButton}
              onClick={() => setShowInfo((v) => !v)}
              aria-expanded={showInfo}
            >
              Live LBM visualisation (Re_sim ≈ {params.reSimEffective.toFixed(0)}) · dashboard
              uses the exact true Re = {formatSci(params.reDisplay)} (i)
            </button>
          </div>

          {showInfo && (
            <div className={styles.infoPanel}>
              <p>
                The flow field is a live 2D Navier–Stokes solve (D2Q9 lattice-Boltzmann, TRT
                collision) on a {NX}×{NY} lattice — a fixed {(NX * DX * 100).toFixed(1)} cm ×{' '}
                {(NY * DX * 100).toFixed(1)} cm working section, with the {chordCm.toFixed(1)} cm
                chord spanning {params.nCells} cells.
              </p>
              <p>
                That canvas is a <strong>visualiser only</strong>. Real air at flight-scale
                Reynolds number cannot be resolved in real time in a browser, so the solver runs
                at a separate, clamped <strong>Re_sim ≈ {params.reSimEffective.toFixed(0)}</strong>{' '}
                (τ = {params.tau.toFixed(3)}), which keeps it stable and laminar/transitional. It
                shows you the wake, the streamlines and where the flow comes off — none of the
                numbers in the panel come from it.
              </p>
              <p>
                Every <strong>dashboard readout</strong> is computed instead at the true Reynolds
                number, V·c/ν ={' '}
                {params.reDisplay.toLocaleString(undefined, { maximumFractionDigits: 0 })}, by two
                steady solvers that run in about 0.1 ms:
              </p>
              <p>
                <strong>C<sub>L</sub> and C<sub>p</sub></strong> come from a Hess–Smith vortex
                panel method — {panels ? panels.n : 0} panels, a source sheet on each plus one
                body vortex, closed by the Kutta condition at the trailing edge and solved exactly
                by Gaussian elimination. It reproduces thin-airfoil theory (C<sub>L</sub> = 2πα) to
                within 0.4% as the section is thinned, and resolves the thickness effect a thin-
                airfoil result cannot: this section's slope is{' '}
                {(A0_PER_DEG * (1 + 0.77 * spec.t) * (180 / Math.PI)).toFixed(2)} per radian rather
                than 2π. Being inviscid it has no stall, so past the critical angle the
                C<sub>L</sub> readout is marked <em>inviscid</em>. C<sub>D</sub> is Hoerner's
                empirical bulge formula on top of Prandtl–Schlichting skin friction.
              </p>
              <p>
                <strong>The separation point</strong> comes from a boundary-layer integral method
                marched along the suction surface: Thwaites' closed-form integral while laminar
                (detaching at λ = −0.09, equivalently a shape factor of 3.55), handed over to
                Head's entrainment method at Re<sub>θ</sub> = 200, which detaches as H₁ falls to
                3.3. The edge velocity it integrates is the panel method's own C<sub>p</sub>.
              </p>
              <p>
                <strong>Stall angle</strong> is section aerodynamics, also at the true Reynolds
                number: the zero-lift angle ({stallModel.zeroLiftAoa.toFixed(2)}°) is exact
                thin-airfoil theory integrated over this airfoil's camber line, and Cl,max (
                {stallModel.clMax.toFixed(2)}) comes from an empirical thickness/Reynolds
                correlation calibrated to published section data. That puts the critical angle at{' '}
                <strong>{stallModel.criticalAoa.toFixed(1)}°</strong>.
              </p>
              <p>
                All three are steady models, so expect the panel readouts to lose accuracy once
                the flow genuinely separates (roughly α above 12°), and the absolute magnitudes to
                differ from published high-Re wind-tunnel polars. For unsteady behaviour — vortex
                shedding, wake flapping, reattachment — the LBM canvas is the honest picture.
              </p>
            </div>
          )}
        </div>

        <aside className={styles.panel}>
          <h3 className={styles.panelTitle}>Live readings</h3>
          <LiveReadings readings={readings} />
        </aside>
      </div>

      {/* --- 3D view: the same solution extruded along the span ------------- */}
      {/* Mirrors the 2D row's grid so both canvases come out the same width. */}
      <div className={`${styles.body} ${activeTab === '3d' ? '' : styles.tabHidden}`}>
        <div className={styles.viewport}>
          <canvas
            ref={canvas3dRef}
            width={VIEW3D_W}
            height={VIEW3D_H}
            className={styles.canvas3d}
          />

          <StallBanner state={readings.stallState} />

          <div className={styles.viewportFooter}>
            <span className={styles.airfoilName}>
              {spec.label} · span {(spanRatio * chordCm).toFixed(1)} cm
            </span>
            <span className={styles.infoButton} aria-hidden="true">
              drag to orbit · scroll to zoom
            </span>
          </div>
        </div>

        <aside className={styles.panel}>
          <h3 className={styles.panelTitle}>3D view</h3>

          <p className={styles.view3dNote}>
            The section is extruded equally either side of the mid-span plane, and that centre
            cross-section is exactly what the 2D view above shows. The solve is 2D, so every
            spanwise station sees the same flow — no tip vortices or downwash. That is the same
            uniform-section assumption the C<sub>L</sub> readout already makes.
          </p>

          <LiveReadings readings={readings} />
        </aside>
      </div>
    </div>
  );
}

/** Stall/near-stall banner, shared by the 2D and 3D viewports. */
function StallBanner({ state }) {
  if (state === 'none') return null;
  return (
    <div
      className={`${styles.stallBanner} ${state === 'near' ? styles.stallBannerNear : ''}`}
      role="status"
    >
      {state === 'stall' ? '⚠ STALL' : '⚠ NEAR STALL'}
    </div>
  );
}

/** Full readouts + stall chip, shared by the 2D and 3D side panels. */
function LiveReadings({ readings }) {
  const attached = readings.separationPoint < 0;
  const deepStall = readings.separationMode === 'leading-edge';

  return (
    <>
      <Readout
        label="Lift coefficient"
        symbol="Cl"
        value={readings.cl.toFixed(3)}
        unit={readings.clIsInviscid ? 'inviscid' : ''}
      />
      <Readout label="Drag coefficient" symbol="Cd" value={readings.cd.toFixed(4)} unit="" />
      <Readout label="Airspeed" symbol="V∞" value={readings.airspeed.toFixed(0)} unit="m/s" />
      <Readout
        label="Reynolds number"
        symbol="Re"
        value={formatSci(readings.reynolds)}
        unit=""
      />
      <Readout label="Lift force" symbol="L" value={readings.liftForce.toFixed(2)} unit="N" />
      <Readout label="Drag force" symbol="D" value={readings.dragForce.toFixed(3)} unit="N" />
      <Readout
        label="Dynamic pressure"
        symbol="q"
        value={readings.dynamicPressure.toFixed(0)}
        unit="Pa"
      />
      <Readout
        label="Reference area"
        symbol="S"
        value={readings.planformCm2.toFixed(1)}
        unit="cm²"
      />
      <Readout
        label="Critical AoA"
        symbol="α*"
        value={readings.criticalAoa.toFixed(1)}
        unit="°"
      />
      <Readout
        label="Separation point"
        symbol="x/c"
        // Three decimals, because the marching grid is 0.005 of chord: at two
        // decimals a detachment at 0.015 renders as "0.01", which reads as
        // sitting exactly on the leading-edge threshold when it is not.
        value={attached ? '—' : readings.separationPoint.toFixed(3)}
        unit={attached ? 'attached' : deepStall ? 'leading edge' : 'of chord'}
      />

      <p className={styles.engineNote}>
        C<sub>d</sub> and the critical angle come from a steady vortex-panel solve and an
        empirical drag correlation — within about ±5% while the flow is attached (α below
        12°). For unsteady wake dynamics, watch the live visualisation.
      </p>

      <p className={styles.engineNote}>
        {deepStall ? (
          <>
            <strong>x/c = 0.00 — leading-edge separation.</strong> The flow is in deep stall:
            it detaches at the nose and reattachment downstream is chaotic, which a boundary-
            layer integral method cannot resolve. See the LBM canvas.
          </>
        ) : (
          <>
            x/c comes from steady boundary-layer integration (Thwaites laminar, Head
            turbulent) — about ±3% of chord while α is below 12°.
            {readings.transitionPoint >= 0 && (
              <> Transition to turbulence at x/c ≈ {readings.transitionPoint.toFixed(2)}.</>
            )}
            {readings.separationMode !== 'attached' && (
              <> Detachment here is {readings.separationMode}.</>
            )}
          </>
        )}
      </p>

      <div
        className={`${styles.stallChip} ${
          readings.stallState === 'stall'
            ? styles.stallChipActive
            : readings.stallState === 'near'
              ? styles.stallChipNear
              : ''
        }`}
      >
        {readings.stallState === 'stall'
          ? 'Stall'
          : readings.stallState === 'near'
            ? 'Near stall'
            : 'Attached'}
        <span className={styles.stallChipDetail}>
          dC<sub>L</sub>/dα = {readings.liftSlope.toFixed(4)} /° (
          {(readings.slopeFraction * 100).toFixed(0)}%)
        </span>
      </div>
    </>
  );
}

function Readout({ label, symbol, value, unit }) {
  return (
    <div className={styles.readout}>
      <div className={styles.readoutLabel}>
        {label} <span className={styles.readoutSymbol}>{symbol}</span>
      </div>
      <div className={styles.readoutValue}>
        {value}
        {unit && <span className={styles.readoutUnit}>{unit}</span>}
      </div>
    </div>
  );
}

function formatSci(v) {
  if (!isFinite(v) || v === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(v)));
  const mant = v / 10 ** exp;
  return `${mant.toFixed(2)}×10${superscript(exp)}`;
}

function superscript(n) {
  const map = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  return String(n)
    .split('')
    .map((ch) => map[ch] ?? ch)
    .join('');
}
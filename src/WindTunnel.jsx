/**
 * WindTunnel — interactive 2D wind tunnel.
 *
 * Flow field is a live PDE solve: D2Q9 Lattice-Boltzmann (TRT collision), which
 * recovers the 2D incompressible Navier-Stokes equations in the low-Mach limit.
 * Airfoil geometry is generated from the NACA 4- and 5-digit series equations.
 * Stall angle comes from section aerodynamics evaluated at the true Reynolds
 * number, not from the (much lower Reynolds) flow fiel d -- see section 4.
 *
 * Self-contained: no required props, no external state, no context. Drop in as
 * <WindTunnel /> and it works.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './WindTunnel.module.css';

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

// Lift is averaged over a rolling window of rendered frames before it is shown
// or turned into a coefficient. Separated flow sheds vortices, so the raw force
// genuinely oscillates. Measured spread of the reported C_L, NACA 2412 at 6 cm
// and 30 m/s: 10 frames +/-14% (54% peak-to-peak), 30 frames +/-11%, 60 frames
// +/-5% (21% peak-to-peak). Past 60 the returns are erratic rather than better,
// because the noise is quasi-periodic shedding and a longer window can alias
// it. 60 frames is one second at 60 fps -- smooth enough to read, quick enough
// to respond to a slider.
const FORCE_AVG_FRAMES = 60;

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
// Streak ribbons: tracer particles advected by the live velocity field, drawn
// as tapered trails. Replaces the old fixed-grid arrow glyphs, which could only
// twitch in place rather than glide.
const STREAK_COUNT = 700;
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
const READOUT_INTERVAL_MS = 80; // ~12.5 Hz panel refresh

// Separation survey: chordwise stations scanned along the suction surface to
// locate the separation point (replacing the old 4-point yes/no vote).
const SEP_STATION_MIN = 0.06;
const SEP_STATION_MAX = 0.98;
const SEP_STATIONS = 48;
// Stations inspected ahead of a candidate point to confirm the reversed region
// is sustained rather than a single vortex passing by.
const SEP_RUN = 6;
// Separated flow is genuinely unsteady, so the reported separation point is
// time-averaged rather than read off a single frame.
const SEP_EMA_ALPHA = 0.08;

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

const N_SURFACE = 200; // sample points per surface (cosine-spaced)

/**
 * Build the unit-chord upper/lower surfaces.
 * Returns chordwise stations plus the two surface point lists, LE -> TE.
 */
function buildSurfaces(spec) {
  const xs = new Float64Array(N_SURFACE);
  const upper = new Float64Array(N_SURFACE * 2);
  const lower = new Float64Array(N_SURFACE * 2);

  for (let j = 0; j < N_SURFACE; j++) {
    // Cosine spacing clusters points at the leading edge, where curvature is highest.
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
  const dt = (U_LAT * dx) / vInf; // s per timestep
  const dm = RHO_AIR * dx * dx; // kg per unit span
  const dF = (dm * dx) / (dt * dt); // N per lattice force unit

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
    dt,
    dF,
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
    fx: 0, // per-step lattice force
    fy: 0,
    fxSum: 0, // accumulated over the current frame's substeps
    fySum: 0,
    stepsThisFrame: 0,
    fxRing: new Float64Array(FORCE_AVG_FRAMES), // per-frame means
    fyRing: new Float64Array(FORCE_AVG_FRAMES),
    ringHead: 0,
    ringCount: 0,
    fxAvg: 0, // rolling mean over the ring
    fyAvg: 0,
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
  S.fx = 0;
  S.fy = 0;
  S.fxSum = 0;
  S.fySum = 0;
  S.stepsThisFrame = 0;
  S.fxRing.fill(0);
  S.fyRing.fill(0);
  S.ringHead = 0;
  S.ringCount = 0;
  S.fxAvg = 0;
  S.fyAvg = 0;
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
      // No-slip: full bounce-back, plus momentum exchange for the force tally.
      fNew[DIR_OFF[OPP[i]] + id] = FS[i];
      S.fx += 2 * FS[i] * EX[i];
      S.fy += 2 * FS[i] * EY[i];
    } else {
      fNew[DIR_OFF[i] + tid] = FS[i];
    }
  }
}

/** One full LBM timestep: collide -> stream -> boundary conditions. */
function lbmStep(S) {
  const { f, fNew, solid, nearSolid, omegaPlus, omegaMinus } = S;

  S.fx = 0;
  S.fy = 0;

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

  // Accumulate this step into the current frame's total; the frame mean is
  // pushed into the rolling window once per rendered frame.
  S.fxSum += S.fx;
  S.fySum += S.fy;
  S.stepsThisFrame++;
}

/**
 * Close out a rendered frame: push the frame's mean lattice force into the
 * rolling window and recompute the average over it.
 *
 * While the window is still filling (the first FORCE_AVG_FRAMES frames) the sum
 * is divided by however many samples exist, not by the full window size, so the
 * readout is correct from the very first frame instead of reading low.
 */
function pushForceFrame(S) {
  if (S.stepsThisFrame === 0) return;
  const fx = S.fxSum / S.stepsThisFrame;
  const fy = S.fySum / S.stepsThisFrame;
  S.fxSum = 0;
  S.fySum = 0;
  S.stepsThisFrame = 0;

  S.fxRing[S.ringHead] = fx;
  S.fyRing[S.ringHead] = fy;
  S.ringHead = (S.ringHead + 1) % FORCE_AVG_FRAMES;
  if (S.ringCount < FORCE_AVG_FRAMES) S.ringCount++;

  let sx = 0;
  let sy = 0;
  for (let i = 0; i < S.ringCount; i++) {
    sx += S.fxRing[i];
    sy += S.fyRing[i];
  }
  S.fxAvg = sx / S.ringCount;
  S.fyAvg = sy / S.ringCount;
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
 * 8. Stall detection (7)
 * ==========================================================================*/

/**
 * Locate the boundary-layer separation point on the suction surface.
 *
 * Surveys the surface from leading to trailing edge, sampling the tangential
 * velocity just outside the wall at each station. Separation is where the
 * tangential flow first reverses *and stays* reversed over the rest of the
 * chord -- requiring persistence is what rejects the isolated reversed cells
 * (shed vortices drifting past, laminar bubbles) that made a per-frame vote
 * flicker. Returns the chordwise position as a fraction of chord, or -1 when
 * the flow is attached.
 */
function findSeparationPoint(S, geo, aoaDeg) {
  if (!geo) return -1;
  const surface = aoaDeg >= 0 ? geo.upper : geo.lower;
  const sign = aoaDeg >= 0 ? 1 : -1; // outward normal handedness per surface
  const { xs } = geo;

  const reversedAt = new Uint8Array(SEP_STATIONS);
  const stationX = new Float64Array(SEP_STATIONS);
  let valid = 0;

  for (let k = 0; k < SEP_STATIONS; k++) {
    const station =
      SEP_STATION_MIN + ((SEP_STATION_MAX - SEP_STATION_MIN) * k) / (SEP_STATIONS - 1);
    stationX[k] = station;

    // Nearest surface sample to this chordwise station.
    let best = 1;
    let bestErr = Infinity;
    for (let j = 1; j < N_SURFACE - 1; j++) {
      const err = Math.abs(xs[j] - station);
      if (err < bestErr) {
        bestErr = err;
        best = j;
      }
    }

    const sx = surface[2 * best];
    const sy = surface[2 * best + 1];
    let tx = surface[2 * (best + 1)] - surface[2 * (best - 1)];
    let ty = surface[2 * (best + 1) + 1] - surface[2 * (best - 1) + 1];
    const tl = Math.hypot(tx, ty);
    if (tl < 1e-9) continue;
    tx /= tl;
    ty /= tl;

    // Outward normal: (-ty, tx) on the upper surface, (ty, -tx) on the lower.
    const nx = -ty * sign;
    const ny = tx * sign;

    // Step outward until clear of the solid mask.
    let found = -1;
    for (let d = 2.5; d <= 6.5; d += 1) {
      const gx = Math.round(sx + nx * d);
      const gy = Math.round(sy + ny * d);
      if (gx < 1 || gx >= NX - 1 || gy < 1 || gy >= NY - 1) break;
      const id = gy * NX + gx;
      if (!S.solid[id]) {
        found = id;
        break;
      }
    }
    if (found < 0) continue;

    valid++;
    if (S.ux[found] * tx + S.uy[found] * ty < 0) reversedAt[k] = 1;
  }

  if (valid < SEP_STATIONS * 0.5) return -1;

  // Scan from the leading edge and take the start of the first *sustained*
  // reversed region. Requiring a run (rather than a single station) rejects
  // isolated reversed cells from passing vortices.
  //
  // Deliberately not anchored at the trailing edge: in deep stall the
  // recirculation closes well before the TE, so a reversed region that must
  // reach the TE would miss precisely the most separated cases.
  let sepIndex = -1;
  for (let k = 0; k < SEP_STATIONS; k++) {
    if (!reversedAt[k]) continue;
    let rev = 0;
    let cnt = 0;
    for (let j = k; j < Math.min(SEP_STATIONS, k + SEP_RUN); j++) {
      cnt++;
      rev += reversedAt[j];
    }
    if (cnt >= 3 && rev >= cnt * 0.75) {
      sepIndex = k;
      break;
    }
  }

  // A reversed region confined to the last few percent of chord is ordinary
  // trailing-edge thickening, not separation worth reporting.
  if (sepIndex < 0 || stationX[sepIndex] > 0.95) return -1;
  return stationX[sepIndex];
}

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

function createStreaks() {
  return {
    x: new Float32Array(STREAK_COUNT),
    y: new Float32Array(STREAK_COUNT),
    tx: new Float32Array(STREAK_COUNT * STREAK_TRAIL),
    ty: new Float32Array(STREAK_COUNT * STREAK_TRAIL),
    head: new Uint8Array(STREAK_COUNT),
    count: new Uint8Array(STREAK_COUNT),
    cap: new Uint8Array(STREAK_COUNT), // per-ribbon length, varied to avoid banding
    // Spanwise station as a fraction of span in [-0.5, 0.5]. The solve is 2D and
    // uniform along span, so a tracer keeps a fixed z while its (x, y) follows
    // the same field the 2D view uses. Stored as a fraction so the span slider
    // rescales the 3D view without disturbing any particle.
    sz: new Float32Array(STREAK_COUNT),
    since: new Float32Array(STREAK_COUNT), // cells travelled since last stored point
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
  for (let i = 0; i < STREAK_COUNT; i++) spawnStreak(St, i, S, true);
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

  for (let i = 0; i < STREAK_COUNT; i++) {
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

  for (let i = 0; i < STREAK_COUNT; i++) {
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

function renderFrame(ctx, field, fieldCtx, fieldImage, S, geo, St) {
  const { spd, solid } = S;
  const data = fieldImage.data;

  // Background: speed magnitude, kept monochrome so the coloured ribbons read clearly.
  const vMax = U_LAT * 1.9;
  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const id = y * NX + x;
      const p = ((NY - 1 - y) * NX + x) * 4; // image rows run top-down
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

  drawStreaks(ctx, St, S);

  // Airfoil.
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
function pushDraw(depth, kind, index, shade) {
  if (drawLen === DRAW.length) DRAW.push({ depth: 0, kind: 0, index: 0, shade: 0 });
  const d = DRAW[drawLen++];
  d.depth = depth;
  d.kind = kind;
  d.index = index;
  d.shade = shade;
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

/**
 * Draw the extruded wing and the tracers into the 3D viewport.
 * `St.sz` gives each tracer its spanwise station; everything else comes from
 * the same state the 2D view renders.
 */
function render3D(ctx, S, geo, St, cam, spanCells) {
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
  // The section is constant along span, so no spanwise subdivision is needed.
  WING_N = 0;
  for (let i = 0; i < geo.polyCount && WING_N < WING_CAP; i += WING_OUTLINE_STEP) {
    WING_IDX[WING_N++] = i;
  }

  // Light from above and slightly front-left, in world axes. Extruded side
  // faces have no z-component to their normal, so only LX/LY matter for them.
  const LX = -0.35;
  const LY = 0.82;

  for (let k = 0; k < WING_N; k++) {
    const i = WING_IDX[k];
    const ax = geo.poly[2 * i];
    const ay = geo.poly[2 * i + 1];
    const okN = project(P3, ax, ay, -hz);
    NEAR_X[k] = PRJ[0];
    NEAR_Y[k] = PRJ[1];
    NEAR_D[k] = PRJ[2];
    const okF = project(P3, ax, ay, hz);
    FAR_X[k] = PRJ[0];
    FAR_Y[k] = PRJ[1];
    FAR_D[k] = PRJ[2];
    WING_OK[k] = okN && okF ? 1 : 0;
  }

  for (let k = 0; k < WING_N; k++) {
    const k2 = (k + 1) % WING_N;
    if (!WING_OK[k] || !WING_OK[k2]) continue;
    const i = WING_IDX[k];
    const j = WING_IDX[k2];
    const ex = geo.poly[2 * j] - geo.poly[2 * i];
    const ey = geo.poly[2 * j + 1] - geo.poly[2 * i + 1];
    const el = Math.hypot(ex, ey) || 1;
    // The poly winds upper LE->TE then lower TE->LE, for which (-dy, dx)
    // points out of the section everywhere.
    const diffuse = Math.max(0, (-ey / el) * LX + (ex / el) * LY);
    WING_SHADE[k] = 0.22 + 0.78 * diffuse;
    pushDraw(
      (NEAR_D[k] + NEAR_D[k2] + FAR_D[k] + FAR_D[k2]) * 0.25,
      0,
      k,
      WING_SHADE[k]
    );
  }

  // End caps, so the wing reads as solid when seen from the side.
  let capNearD = 0;
  let capFarD = 0;
  for (let k = 0; k < WING_N; k++) {
    capNearD += NEAR_D[k];
    capFarD += FAR_D[k];
  }
  pushDraw(capNearD / WING_N, 2, 0, 0.55);
  pushDraw(capFarD / WING_N, 2, 1, 0.55);

  // --- Tracers ------------------------------------------------------------
  // Cull tracers whose head projects outside the viewport (with a margin for
  // the trail that follows it) before they reach the sort or the draw loop.
  const margin = 60;
  for (let i = 0; i < STREAK_COUNT; i++) {
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
      // Extruded side face between outline points k and k+1.
      const k = item.index;
      const k2 = (k + 1) % WING_N;
      const sh = item.shade;
      const col = `rgb(${(232 * sh) | 0},${(237 * sh) | 0},${(245 * sh) | 0})`;
      ctx.fillStyle = col;
      // Stroke with the fill colour too: adjacent quads are antialiased along
      // their shared edge, which otherwise leaves a visible seam and makes the
      // extruded surface look hatched.
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
      // End cap: the airfoil section itself at one end of the span.
      const useNear = item.index === 0;
      const sh = item.shade;
      ctx.fillStyle = `rgb(${(232 * sh) | 0},${(237 * sh) | 0},${(245 * sh) | 0})`;
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

const READOUT_INIT = {
  cl: 0,
  cd: 0,
  airspeed: 30,
  reynolds: 0,
  dragForce: 0,
  liftForce: 0,
  dynamicPressure: 0,
  stallState: 'none',
  stalling: false,
  liftSlope: 0,
  slopeFraction: 1,
  chordCm: CHORD_DEFAULT_CM,
  spanCm: CHORD_DEFAULT_CM * SPAN_DEFAULT,
  spanRatio: SPAN_DEFAULT,
  planformCm2: CHORD_DEFAULT_CM * CHORD_DEFAULT_CM * SPAN_DEFAULT,
  criticalAoa: 0,
  zeroLiftAoa: 0,
  clMax: 0,
  separationPoint: -1,
};

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
  const [readings, setReadings] = useState({
    ...READOUT_INIT,
    airspeed: initialAirspeed,
    chordCm: initialChordCm,
  });
  const [showInfo, setShowInfo] = useState(false);
  const [activeTab, setActiveTab] = useState('2d'); // '2d' | '3d' — which viewport is shown

  const parsed = useMemo(() => parseNacaCode(code), [code]);
  const params = useMemo(
    () => computeParams(airspeed, cmToM(chordCm)),
    [airspeed, chordCm]
  );

  // Last valid spec — the sim keeps running the previous airfoil while the user
  // is mid-typing an incomplete code.
  const [spec, setSpec] = useState(() => {
    const p = parseNacaCode(initialNacaCode);
    return p.ok ? p : parseNacaCode('2412');
  });
  useEffect(() => {
    if (parsed.ok) setSpec(parsed);
  }, [parsed]);

  // Section aerodynamics: evaluated at the true Reynolds number, independent of
  // the solver's internal Re_sim.
  const stallModel = useMemo(
    () => computeStallModel(spec, params.reDisplay),
    [spec, params.reDisplay]
  );

  const canvasRef = useRef(null);
  const solverRef = useRef(null);
  const geoRef = useRef(null);
  const paramsRef = useRef(computeParams(initialAirspeed, cmToM(initialChordCm)));
  const airspeedRef = useRef(initialAirspeed);
  const aoaRef = useRef(initialAoa);
  const stallModelRef = useRef(stallModel);
  stallModelRef.current = stallModel;
  const resetPendingRef = useRef(true);
  const warmupRef = useRef(WARMUP_FRAMES);
  const substepsRef = useRef(2);
  // Smoothed per-frame cost of everything except the solver.
  const nonSolverEmaRef = useRef(12);
  // Time-averaged separation point (fraction of chord); -1 means attached.
  const sepRef = useRef(-1);
  const stalledRef = useRef(false);
  const readingsCbRef = useRef(onReadingsChange);
  readingsCbRef.current = onReadingsChange;

  const streaksRef = useRef(null);
  const canvas3dRef = useRef(null);
  const camRef = useRef(null);
  const spanRatioRef = useRef(spanRatio);
  spanRatioRef.current = spanRatio;
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;
  if (solverRef.current === null) solverRef.current = createSolver();
  if (streaksRef.current === null) streaksRef.current = createStreaks();
  if (camRef.current === null) camRef.current = createCamera();

  // --- Geometry: rebuild + full field reset on airfoil, AoA or size change ---
  useEffect(() => {
    const surfaces = buildSurfaces(spec);
    const geo = placeAirfoil(surfaces, aoa, nCellsFor(cmToM(chordCm)));
    const { solid, nearSolid } = rasterize(geo);

    const S = solverRef.current;
    S.solid = solid;
    S.nearSolid = nearSolid;
    geoRef.current = geo;
    aoaRef.current = aoa;
    resetPendingRef.current = true;
    warmupRef.current = WARMUP_FRAMES;
    sepRef.current = -1;
    // Tracers must be reseeded: some will now be inside the new solid, and any
    // surviving trail would be a streak drawn through the old airfoil.
    streaksRef.current.seeded = false;
  }, [spec, aoa, chordCm]);

  // --- Airspeed: retune tau / dt only; the flow adapts continuously ---------
  useEffect(() => {
    paramsRef.current = params;
    airspeedRef.current = airspeed;
    solverRef.current.omegaPlus = params.omegaPlus;
    solverRef.current.omegaMinus = params.omegaMinus;
  }, [params, airspeed]);

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
    let lastReadout = 0;
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

      // One force sample per rendered frame, regardless of how many solver
      // substeps ran, so the rolling window is a window of frames.
      pushForceFrame(S);

      computeMacroscopic(S);
      if (!St.seeded) resetStreaks(St, S);
      advanceStreaks(St, S, airspeedRef.current, dt);
      renderFrame(ctx, field, fieldCtx, fieldImage, S, geoRef.current, St);

      if (ctx3d && geoRef.current && activeTabRef.current === '3d') {
        render3D(
          ctx3d,
          S,
          geoRef.current,
          St,
          camRef.current,
          spanRatioRef.current * geoRef.current.nCells
        );
      }

      // Panel readouts, throttled to a human-readable rate.
      if (now - lastReadout >= READOUT_INTERVAL_MS) {
        lastReadout = now;
        const P = paramsRef.current;
        const V = airspeedRef.current;

        // Whole-wing forces: the 2D solve gives force per metre of span, so
        // multiply by the span to get Newtons on the wing.
        //   C_L = L / (0.5 * rho * V^2 * S),  S = span * chord
        // C_L itself is span-invariant (both L and S scale with span); the
        // slider moves the forces and the area, not the coefficient.
        const span = spanRatioRef.current * P.chordEff;
        const planform = span * P.chordEff;
        const drag = S.fxAvg * P.dF * span;
        const lift = S.fyAvg * P.dF * span;
        const q = 0.5 * RHO_AIR * V * V;

        // Stall straight off the lift curve: dCl/dalpha <= 0 is stall, and a
        // slope below 20% of the section slope is the near-stall warning. The
        // curve is evaluated at the true Reynolds number, not read off the
        // low-Re flow field (which separates far too early, and unsteadily).
        const M = stallModelRef.current;
        const a = aoaRef.current;
        const curve = liftCurve(a, M);
        const state = stallState(a, M);
        stalledRef.current = state === 'stall';

        // The live solve still supplies where the boundary layer detaches,
        // time-averaged so it reads as a physical quantity, not a flicker.
        const sepNow = findSeparationPoint(S, geoRef.current, aoaRef.current);
        const prevSep = sepRef.current;
        if (sepNow < 0) {
          sepRef.current = prevSep < 0 ? -1 : prevSep + (1.05 - prevSep) * SEP_EMA_ALPHA;
          if (sepRef.current > 1) sepRef.current = -1; // relaxed back to attached
        } else {
          sepRef.current =
            prevSep < 0 ? sepNow : prevSep + (sepNow - prevSep) * SEP_EMA_ALPHA;
        }

        const next = {
          cl: lift / (q * planform),
          cd: drag / (q * planform),
          airspeed: V,
          reynolds: P.reDisplay,
          dragForce: drag,
          liftForce: lift,
          dynamicPressure: q,
          stallState: state, // 'none' | 'near' | 'stall'
          stalling: state === 'stall',
          liftSlope: curve.slope, // dCl/dalpha, per degree
          slopeFraction: curve.slopeFactor, // as a fraction of the section slope
          chordCm: P.chord * 100,
          spanCm: span * 100,
          spanRatio: spanRatioRef.current,
          planformCm2: planform * 1e4,
          criticalAoa: M.criticalAoa,
          zeroLiftAoa: M.zeroLiftAoa,
          clMax: M.clMax,
          separationPoint: sepRef.current,
        };
        setReadings(next);
        if (readingsCbRef.current) readingsCbRef.current(next);
      }

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
          <label className={styles.label} htmlFor="wt-naca">
            NACA airfoil

          </label>
          <input
            id="wt-naca"
            className={`${styles.textInput} ${parsed.ok ? '' : styles.textInputError}`}
            value={code}
            onChange={onCodeChange}
            spellCheck={false}
            autoComplete="off"
            placeholder="2412"
            aria-invalid={!parsed.ok}
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
      </div>

      {!parsed.ok && <div className={styles.error}>{parsed.error}</div>}
      {parsed.ok && parsed.warning && <div className={styles.warning}>{parsed.warning}</div>}

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
              Simulated at a numerically-stabilized Reynolds number ⓘ
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
                <strong>Stall angle</strong> comes from section aerodynamics evaluated at the true
                Reynolds number, not from the flow field: the zero-lift angle (
                {stallModel.zeroLiftAoa.toFixed(2)}°) is exact thin-airfoil theory integrated over
                this airfoil's camber line, and Cl,max ({stallModel.clMax.toFixed(2)}) comes from an
                empirical thickness/Reynolds correlation calibrated to published section data. That
                puts the critical angle at <strong>{stallModel.criticalAoa.toFixed(1)}°</strong>.
              </p>
              <p>
                The solver itself runs near Re {params.reSimEffective.toFixed(0)}, where a real
                airfoil would separate several degrees earlier than it does at flight Reynolds
                numbers. So the <em>separation point</em> readout (measured live from the solve) can
                show detached flow while the airfoil is still below its true stall angle — that is
                the low-Re flow being reported honestly, not a glitch.
              </p>
              <p>
                The <strong>displayed Reynolds number</strong> (
                {params.reDisplay.toLocaleString(undefined, { maximumFractionDigits: 0 })}) is the
                true physical value, V·c/ν. Real air at flight-scale Re cannot be resolved in real
                time in a browser — so the solver runs at a separate, clamped{' '}
                <strong>Re_sim ≈ {params.reSimEffective.toFixed(0)}</strong> (τ ={' '}
                {params.tau.toFixed(3)}), which keeps it stable and laminar/transitional.
              </p>
              <p>
                Expect the simulated stall angle and absolute force magnitudes to differ from
                published high-Re wind-tunnel polars. The qualitative behaviour — lift build-up with
                AoA, drag rise, separation and stall — is captured correctly.
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
  return (
    <>
      <Readout label="Lift coefficient" symbol="Cl" value={readings.cl.toFixed(3)} unit="" />
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
        // Anything past 0.95 chord is the trailing edge, which the detector
        // itself declines to call separation; while the smoothed value is
        // relaxing back through that region, report it as attached rather
        // than showing a number that contradicts the state chip.
        value={
          readings.separationPoint < 0 || readings.separationPoint > 0.95
            ? '—'
            : readings.separationPoint.toFixed(2)
        }
        unit={
          readings.separationPoint < 0 || readings.separationPoint > 0.95
            ? 'attached'
            : 'of chord'
        }
      />

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
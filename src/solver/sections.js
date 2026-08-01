/**
 * sections.js — the geometry library: every section shape the solver can panel.
 *
 * `geometry.js` turns a *spec* into a panelled body. Until now the only spec
 * came from `parseNacaCode`, and geometry.js reached directly into naca.js for
 * the surface equations. This module generalises that one step: a spec now
 * carries its own `surface(x, side)`, so adding a shape means adding a spec here
 * and nothing else. NACA specs are passed through untouched — `parseNacaCode` is
 * still the parser, `surfacePoint` is still the surface — so every existing NACA
 * result is bit-for-bit what it was.
 *
 * ## What a spec owes the panel generator
 *
 *   surface(x, side)  [x, y] on the surface at chordwise parameter x in [0, 1],
 *                     side +1 upper, -1 lower. The returned x need not equal the
 *                     argument: thickness is laid off along the camber normal,
 *                     so it generally does not.
 *   camberAt(x)       [y_c, dy_c/dx], for the thin-airfoil zero-lift angle and
 *                     the reported camber. Optional; symmetric sections skip it.
 *   t                 max thickness as a fraction of chord, for reporting.
 *
 * Everything downstream — the spline, the panels, the influence matrix, the
 * boundary layer — only ever sees the node ring, so it does not care which
 * shape produced it.
 *
 * ## What is deliberately not here
 *
 * A cylinder. The panel system closes with a Kutta condition stated at a
 * trailing edge (see panel.js), and a circular cylinder has no trailing edge.
 * Run one through this solver and you get symmetric potential flow — zero lift,
 * zero drag, d'Alembert's paradox — with a boundary layer that separates near 80
 * degrees and walks straight into the Goldstein singularity the integral method
 * cannot pass. Both results would be confidently wrong, which is worse than
 * absent. A cylinder needs a different model, not a different spec.
 */

import { parseNacaCode, surfacePoint, camber } from './naca.js';

/* ============================================================================
 * The catalogue
 * ==========================================================================*/

/**
 * The selectable geometries, in the order the UI lists them. `code` marks the
 * one that takes a user-entered designation.
 */
export const GEOMETRIES = [
  { id: 'naca', label: 'NACA airfoil', code: true },
  { id: 'clarky', label: 'Clark Y' },
  { id: 'flatplate', label: 'Flat plate' },
];

export const DEFAULT_GEOMETRY = 'naca';

/**
 * Build a section spec.
 *
 * `geometry` selects the shape; `naca` is only read when geometry is 'naca'.
 * Returns the same `{ ok, error?, warning?, ... }` shape `parseNacaCode` does,
 * so callers can keep treating a bad entry as a validation failure.
 */
export function parseSection({ geometry = DEFAULT_GEOMETRY, naca = '2412', plateThickness } = {}) {
  switch (geometry) {
    case 'clarky':
      return clarkY();
    case 'flatplate':
      return flatPlate(plateThickness);
    case 'naca':
    default: {
      const spec = parseNacaCode(naca);
      if (!spec.ok) return spec;
      // The NACA surface stays exactly where it was; the spec just carries it
      // now instead of geometry.js importing it directly.
      return {
        ...spec,
        geometry: 'naca',
        surface: (x, side) => surfacePoint(x, side, spec),
        camberAt: (x) => camber(x, spec),
      };
    }
  }
}

/**
 * Cache key for the spec: what must differ before the geometry is rebuilt.
 * The solver compares this rather than the spec object, which is rebuilt on
 * every render and would otherwise force a full re-panel every keystroke.
 */
export function sectionKey(inputs) {
  const g = inputs.geometry ?? DEFAULT_GEOMETRY;
  if (g === 'naca') return `naca:${inputs.naca}`;
  if (g === 'flatplate') return `flatplate:${inputs.plateThickness ?? PLATE_THICKNESS}`;
  return g;
}

/* ============================================================================
 * Clark Y
 *
 * Defined by its published ordinate table rather than by a formula — Clark Y is
 * a drawn section, not a parametric family, and reconstructing it from a NACA
 * thickness law plus a fitted camber line would be a different aerofoil wearing
 * the name.
 *
 * The table is the classic one, in per-cent of chord, with ordinates measured
 * from the *flat lower reference line*. That is the convention Clark Y is
 * published in and it is why the nose entry is 3.50 on both surfaces: the
 * leading edge sits 3.5% above the flat bottom, and the bottom is dead flat aft
 * of 30% chord, which is the section's whole distinguishing feature.
 * ==========================================================================*/

// x/c, then upper and lower ordinate, all in per cent of chord.
const CLARK_Y_X = [0, 1.25, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];
const CLARK_Y_UPPER = [
  3.5, 5.45, 6.5, 7.9, 8.85, 9.6, 10.69, 11.36, 11.7, 11.4, 10.52, 9.15, 7.35, 5.22, 2.8, 1.49, 0.12,
];
const CLARK_Y_LOWER = [
  3.5, 1.93, 1.47, 0.93, 0.63, 0.42, 0.15, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

function clarkY() {
  const xs = CLARK_Y_X.map((v) => v / 100);
  const yu = CLARK_Y_UPPER.map((v) => v / 100);
  const yl = CLARK_Y_LOWER.map((v) => v / 100);

  // Interpolate against sqrt(x), not x. A rounded nose has y ~ sqrt(x) there, so
  // in sqrt(x) the data is very nearly straight at the leading edge and the
  // interpolant has no business trying to resolve an infinite slope.
  const xi = xs.map(Math.sqrt);
  const upper = pchip(xi, yu);
  const lower = pchip(xi, yl);

  // --- Table frame -> chord frame -----------------------------------------
  // The table's y is measured from the flat bottom, so its "x axis" is not the
  // chord line. The chord line runs nose to trailing edge, and everything else
  // in this solver — angle of attack above all — is measured from it. So rotate
  // the section onto it and normalise the chord to 1.
  const noseY = yu[0]; // = yl[0]; the leading edge is a single point
  const teY = 0.5 * (yu[yu.length - 1] + yl[yl.length - 1]);
  const dx = 1;
  const dy = teY - noseY;
  const L = Math.hypot(dx, dy);
  const ux = dx / L;
  const uy = dy / L;

  /** Table coordinates -> chord-normalised coordinates. */
  const toChord = (x, y) => {
    const px = x;
    const py = y - noseY;
    return [(ux * px + uy * py) / L, (-uy * px + ux * py) / L];
  };

  const raw = (x, side) => (side > 0 ? upper(Math.sqrt(x)) : lower(Math.sqrt(x)));

  const surface = (x, side) => {
    const xc = Math.min(1, Math.max(0, x));
    return toChord(xc, raw(xc, side));
  };

  // Camber and thickness, measured in the chord frame so they mean what they
  // are normally quoted to mean. Sampled rather than solved: the two surfaces
  // are tabulated against the *table* x, and pairing them at equal chord-frame x
  // would need an inversion for a quantity only used for reporting.
  const camberAt = (x) => {
    const xc = Math.min(1, Math.max(0, x));
    const mid = 0.5 * (raw(xc, 1) + raw(xc, -1));
    const yc = toChord(xc, mid)[1];
    const h = 1e-4;
    const a = Math.min(1, Math.max(0, xc - h));
    const b = Math.min(1, Math.max(0, xc + h));
    const ya = toChord(a, 0.5 * (raw(a, 1) + raw(a, -1)))[1];
    const yb = toChord(b, 0.5 * (raw(b, 1) + raw(b, -1)))[1];
    return [yc, (yb - ya) / (b - a || 1)];
  };

  let tMax = 0;
  for (let k = 0; k <= 200; k++) {
    const x = k / 200;
    const th = surface(x, 1)[1] - surface(x, -1)[1];
    if (th > tMax) tMax = th;
  }

  return {
    ok: true,
    geometry: 'clarky',
    key: 'clarky',
    series: 'clarky',
    label: 'Clark Y',
    t: tMax,
    symmetric: false,
    surface,
    camberAt,
  };
}

/* ============================================================================
 * Flat plate
 *
 * A *zero*-thickness plate cannot be panelled by this method: the two surfaces
 * would coincide, every panel would have zero length, and panelGeometry rejects
 * the ring as degenerate before the influence matrix is ever assembled. Even if
 * it were built, a sheet with coincident upper and lower control points gives a
 * singular system — the two tangency conditions at each station are the same
 * equation.
 *
 * So the plate carries a small finite thickness. The shape is the honest thin
 * plate it is meant to model: a semicircular nose, constant thickness through
 * the middle, and a linear taper to a sharp trailing edge — sharp because the
 * Kutta condition is stated at a point and needs one.
 * ==========================================================================*/

/** Default plate thickness as a fraction of chord. */
export const PLATE_THICKNESS = 0.02;

/** Fraction of chord over which the tail tapers to the sharp trailing edge. */
const PLATE_TAPER = 0.1;

function flatPlate(thickness) {
  const t = Math.min(0.08, Math.max(0.004, thickness ?? PLATE_THICKNESS));
  const half = t / 2;
  // Semicircular nose. For a quarter-ellipse of semi-axes a (streamwise) by b,
  // the leading-edge radius is b^2/a, so stretching the nose downstream makes it
  // *sharper*, not blunter — a = b is the bluntest nose a plate of this
  // thickness can have, and a thin plate has a small nose radius no matter what.
  const nose = half;

  /**
   * Half-thickness at x. A quarter-ellipse over the nose, flat through the
   * middle, linear into the trailing edge. The ellipse meets the flat with zero
   * slope, so that junction is smooth; the taper junction is a genuine corner,
   * which is what a tapered plate actually has.
   */
  const halfThickness = (x) => {
    if (x <= 0 || x >= 1) return 0;
    if (x < nose) {
      const d = (nose - x) / nose;
      return half * Math.sqrt(Math.max(0, 1 - d * d));
    }
    const taperStart = 1 - PLATE_TAPER;
    if (x > taperStart) return half * ((1 - x) / PLATE_TAPER);
    return half;
  };

  return {
    ok: true,
    geometry: 'flatplate',
    key: `flatplate-${t.toFixed(4)}`,
    series: 'flatplate',
    label: `Flat plate (${(t * 100).toFixed(1)}% t/c)`,
    t,
    symmetric: true,
    plateThickness: t,
    // Measured across 2-5% thickness: attached and well converged to about four
    // degrees (confidence 70-83%), then the suction peak at the nose separates
    // the layer inside the first few per cent of chord and the confidence falls
    // to under 20%. That is the section's real behaviour — a thin plate stalls
    // from the leading edge, which is why plates are poor aerofoils — but it is
    // also precisely the regime a direct-mode integral boundary layer cannot
    // resolve, so past a few degrees the numbers are a model of a separated
    // flow rather than a solution of one. Flagged rather than hidden.
    warning:
      'A thin plate separates at the leading edge: expect trustworthy numbers to about 4 degrees, and treat anything past that as indicative — the confidence figure tracks it.',
    surface: (x, side) => {
      const xc = Math.min(1, Math.max(0, x));
      return [xc, side * halfThickness(xc)];
    },
    camberAt: () => [0, 0],
  };
}

/* ============================================================================
 * Generalised reporting quantities
 *
 * naca.js computes these from `camber(x, spec)`, which only understands NACA
 * specs. These are the same integrals against whatever camber line the spec
 * carries, so they work for every section in the library.
 * ==========================================================================*/

function camberOf(spec, x) {
  return spec.camberAt ? spec.camberAt(x) : camber(x, spec);
}

/**
 * Thin-airfoil zero-lift angle in degrees,
 *   alpha_L0 = -(1/pi) integral_0^pi (dy_c/dx)(cos(theta) - 1) d(theta)
 * with x = (1 - cos theta)/2. Independent of the panel solve, so it stays a
 * check on it rather than a restatement of it.
 */
export function sectionZeroLiftAngle(spec) {
  if (spec.symmetric) return 0;
  const M = 800;
  let sum = 0;
  for (let k = 0; k <= M; k++) {
    const theta = (Math.PI * k) / M;
    const x = 0.5 * (1 - Math.cos(theta));
    const g = camberOf(spec, x)[1] * (Math.cos(theta) - 1);
    const w = k === 0 || k === M ? 1 : k % 2 ? 4 : 2;
    sum += w * g;
  }
  const integral = (Math.PI / M / 3) * sum;
  return (-(integral / Math.PI) * 180) / Math.PI;
}

/** Largest camber ordinate as a fraction of chord. */
export function sectionMaxCamber(spec) {
  if (spec.symmetric) return 0;
  let best = 0;
  for (let k = 0; k <= 400; k++) {
    const yc = camberOf(spec, k / 400)[0];
    if (Math.abs(yc) > Math.abs(best)) best = yc;
  }
  return best;
}

/* ============================================================================
 * Monotone cubic interpolation (Fritsch-Carlson)
 * ==========================================================================*/

/**
 * Shape-preserving cubic interpolant through tabulated points.
 *
 * A natural cubic spline would be smoother but overshoots, and this table has a
 * dead-flat run — the whole lower surface aft of 30% chord — meeting a curve at
 * both ends. A C2 spline puts a small bulge either side of those junctions,
 * which is a ripple in the surface that the panel method will faithfully turn
 * into a ripple in the pressure distribution. Fritsch-Carlson cannot overshoot,
 * so flat data stays flat. It is only C1, but geometry.js re-splines the dense
 * ring with a C2 periodic spline before any curvature is asked for.
 */
function pchip(xs, ys) {
  const n = xs.length;
  const h = new Float64Array(n - 1);
  const delta = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = xs[i + 1] - xs[i];
    delta[i] = (ys[i + 1] - ys[i]) / h[i];
  }

  const m = new Float64Array(n);
  m[0] = delta[0];
  m[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i++) {
    // A sign change (or a flat neighbour) is a local extremum: pin the slope to
    // zero there, which is exactly what stops the overshoot.
    if (delta[i - 1] * delta[i] <= 0) {
      m[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }

  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] > x) hi = mid;
      else lo = mid;
    }
    const s = x - xs[lo];
    const hi2 = h[lo];
    const d = delta[lo];
    const c2 = (3 * d - 2 * m[lo] - m[lo + 1]) / hi2;
    const c3 = (m[lo] - 2 * d + m[lo + 1]) / (hi2 * hi2);
    return ys[lo] + s * (m[lo] + s * (c2 + s * c3));
  };
}

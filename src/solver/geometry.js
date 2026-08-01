/**
 * geometry.js — from a NACA designation to a panelled body.
 *
 * The pipeline is:
 *
 *   analytic NACA coordinates  (dense, cosine-spaced, both surfaces)
 *        -> closed periodic cubic spline, parameterised by arc length
 *        -> resample at cosine-spaced chordwise stations
 *        -> adaptive refinement where the spline curvature is large
 *        -> panel geometry: midpoints, tangents, outward normals, curvature
 *
 * Why the spline is in the loop at all, given that the NACA equations are
 * analytic: the panel method needs surface *normals* and the boundary layer
 * needs surface *curvature*, and both are derivatives. Differencing the analytic
 * points gives normals that are only as smooth as the sampling, and curvature
 * that is pure noise near the nose where ds/dx blows up. A C2 spline through the
 * points gives both directly, and gives them consistently — the normal used to
 * assemble the influence matrix is the same normal the boundary layer marches
 * along.
 *
 * Parameterisation is by cumulative chord length, not by x. The upper surface of
 * a cambered section is not monotone in x within a few 1e-5 of chord of the nose
 * (the sqrt(x) thickness term wins over the linear one there), so any scheme
 * that inverts x(s) has a singularity exactly where the panels are densest. Arc
 * length has no such problem.
 */

import { surfacePoint } from './naca.js';

/** Dense analytic samples per surface, before splining. */
const DENSE_PER_SURFACE = 300;

/* ============================================================================
 * Closed periodic cubic spline
 * ==========================================================================*/

/**
 * Solve a cyclic tridiagonal system (Thomas + Sherman-Morrison).
 *
 * The periodic spline's second-derivative system is tridiagonal except for the
 * two corner entries that wrap the ring closed. Sherman-Morrison handles those
 * as a rank-1 update of a plain tridiagonal solve, keeping the whole thing O(n)
 * — which matters because this runs on 600 points every time the airfoil
 * changes, and a dense solve there would be 2e8 flops for no reason.
 *
 * a: sub-diagonal, b: diagonal, c: super-diagonal, alpha: corner (n-1, 0),
 * beta: corner (0, n-1). Solves in place into `x`.
 */
function cyclicTridiagonal(a, b, c, alpha, beta, r, n, x) {
  const bb = new Float64Array(n);
  const u = new Float64Array(n);
  const z = new Float64Array(n);

  const gamma = -b[0];
  bb.set(b);
  bb[0] = b[0] - gamma;
  bb[n - 1] = b[n - 1] - (alpha * beta) / gamma;

  const solveTri = (rhs, out) => {
    const cp = new Float64Array(n);
    let bet = bb[0];
    out[0] = rhs[0] / bet;
    for (let i = 1; i < n; i++) {
      cp[i] = c[i - 1] / bet;
      bet = bb[i] - a[i] * cp[i];
      out[i] = (rhs[i] - a[i] * out[i - 1]) / bet;
    }
    for (let i = n - 2; i >= 0; i--) out[i] -= cp[i + 1] * out[i + 1];
  };

  solveTri(r, x);
  u.fill(0);
  u[0] = gamma;
  u[n - 1] = alpha;
  solveTri(u, z);

  const fact = (x[0] + (beta * x[n - 1]) / gamma) / (1 + z[0] + (beta * z[n - 1]) / gamma);
  for (let i = 0; i < n; i++) x[i] -= fact * z[i];
  return x;
}

/**
 * Fit a closed periodic cubic spline through a ring of points.
 *
 * Returns the knot parameters (cumulative chord length, with the wrap segment
 * included so the ring closes) and the second derivatives in each coordinate.
 * Evaluation is the usual cubic Hermite-from-second-derivatives form.
 */
export function fitClosedSpline(px, py) {
  const n = px.length;
  const u = new Float64Array(n + 1); // u[n] closes the ring
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    u[i + 1] = u[i] + Math.hypot(px[j] - px[i], py[j] - py[i]);
  }

  const solveAxis = (p) => {
    const a = new Float64Array(n);
    const b = new Float64Array(n);
    const c = new Float64Array(n);
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const im = (i - 1 + n) % n;
      const ip = (i + 1) % n;
      const hPrev = u[i] - u[im] || u[n] - u[n - 1]; // wrap segment for i = 0
      const hNext = u[i + 1] - u[i];
      a[i] = hPrev / 6;
      b[i] = (hPrev + hNext) / 3;
      c[i] = hNext / 6;
      r[i] = (p[ip] - p[i]) / hNext - (p[i] - p[im]) / hPrev;
    }
    // The wrap makes row 0 reference column n-1 and row n-1 reference column 0.
    const alpha = c[n - 1];
    const beta = a[0];
    const out = new Float64Array(n);
    cyclicTridiagonal(a, b, c, alpha, beta, r, n, out);
    return out;
  };

  return { n, u, px, py, sx: solveAxis(px), sy: solveAxis(py), length: u[n] };
}

/**
 * Evaluate the spline at arc-length parameter `s`, in segment `seg`.
 * Returns position, first derivative and second derivative with respect to s.
 */
function splineEval(sp, seg, s, out) {
  const { n, u, px, py, sx, sy } = sp;
  const i = seg;
  const j = (i + 1) % n;
  const h = u[i + 1] - u[i];
  const A = (u[i + 1] - s) / h;
  const B = 1 - A;
  const h2 = h * h;

  const cx0 = sx[i];
  const cx1 = sx[j];
  const cy0 = sy[i];
  const cy1 = sy[j];

  out.x = A * px[i] + B * px[j] + ((A * A * A - A) * cx0 + (B * B * B - B) * cx1) * (h2 / 6);
  out.y = A * py[i] + B * py[j] + ((A * A * A - A) * cy0 + (B * B * B - B) * cy1) * (h2 / 6);
  out.dx = (px[j] - px[i]) / h + ((-(3 * A * A - 1) * cx0 + (3 * B * B - 1) * cx1) * h) / 6;
  out.dy = (py[j] - py[i]) / h + ((-(3 * A * A - 1) * cy0 + (3 * B * B - 1) * cy1) * h) / 6;
  out.ddx = A * cx0 + B * cx1;
  out.ddy = A * cy0 + B * cy1;
  return out;
}

/* ============================================================================
 * Dense analytic ring
 * ==========================================================================*/

/**
 * Weight of full cosine spacing against half-cosine in the chordwise
 * distribution. See `chordStation`.
 */
const TE_CLUSTER = 0.7;

/**
 * Chordwise station for a fraction f along one surface, f = 0 at the leading
 * edge and 1 at the trailing edge.
 *
 * Full cosine spacing, x = (1 - cos(pi f))/2, is the textbook choice and is what
 * the specification asks for: it clusters panels quadratically at both ends,
 * where the curvature and the pressure gradients live. It is right at the
 * leading edge and wrong at the trailing edge, for a reason that only shows up
 * once the Kutta condition is imposed.
 *
 * Measured on NACA 0012 with 220 panels, pure cosine makes the final panel
 * 0.021% of chord. The Kutta condition pins the two nodal vortex strengths that
 * meet there to be equal and opposite, and forcing that across a panel two
 * orders of magnitude shorter than its neighbours produced a surface-velocity
 * spike: 0.789, 0.758, then 0.928 on the last three panels. That spike is a
 * discretisation artefact, but the boundary layer cannot tell — it read the jump
 * as a velocity gradient of +800 per chord, which collapsed the momentum
 * thickness from 7.7e-3 to 5.1e-4 over the last two panels and cut the drag by a
 * factor of three.
 *
 * So the distribution is cosine blended with half-cosine, x = 1 - cos(pi f / 2),
 * which has the identical quadratic clustering at the leading edge and constant
 * spacing at the trailing edge. At the default weight the last panel is 0.4% of
 * chord and the first is 0.017% — still two orders of magnitude of refinement
 * where it is needed, with no spike. Curvature-driven adaptive refinement then
 * adds panels wherever the blend was not aggressive enough.
 */
function chordStation(f) {
  return (
    TE_CLUSTER * 0.5 * (1 - Math.cos(Math.PI * f)) +
    (1 - TE_CLUSTER) * (1 - Math.cos((Math.PI * f) / 2))
  );
}

/**
 * The ring parameter tau runs over [0, 2):
 *   tau in [0, 1)  lower surface, trailing edge -> leading edge
 *   tau in [1, 2)  upper surface, leading edge -> trailing edge
 * which traverses the section clockwise, the ordering the panel method needs.
 */
function stationForTau(tau) {
  if (tau < 1) return { x: chordStation(1 - tau), side: -1 }; // runs backwards in x
  return { x: chordStation(tau - 1), side: 1 };
}

/**
 * Analytic surface point with the trailing edge drawn closed.
 *
 * The NACA thickness law leaves a gap of about 0.1% of chord at the trailing
 * edge (y_t(1) is not zero), which a panel method on a closed body cannot
 * represent. Closing it by moving only the last node onto the mean point leaves
 * a near-vertical spike panel — measured on NACA 0012 the final panel turned
 * through 73 degrees in 0.04% of chord and put a visible kink in the surface
 * velocity. Instead each surface is drawn onto the mean trailing-edge point by a
 * correction linear in x: zero at the nose, under 0.07% of chord anywhere else,
 * and no kink.
 */
function closedSurfacePoint(x, side, spec, te) {
  const p = sectionSurface(x, side, spec);
  const t = side > 0 ? te.upper : te.lower;
  return [p[0] - x * (t[0] - te.x), p[1] - x * (t[1] - te.y)];
}

/**
 * The surface equation for whichever section this is.
 *
 * Specs from sections.js carry their own surface, which is how Clark Y and the
 * flat plate get panelled by exactly this pipeline. A bare NACA spec — what
 * `parseNacaCode` returns, and what the validation harness passes in directly —
 * has no `surface`, so it falls back to the NACA equations and produces the
 * identical ring it always did.
 */
function sectionSurface(x, side, spec) {
  return spec.surface ? spec.surface(x, side) : surfacePoint(x, side, spec);
}

function trailingEdge(spec) {
  const upper = sectionSurface(1, 1, spec);
  const lower = sectionSurface(1, -1, spec);
  return { upper, lower, x: 0.5 * (upper[0] + lower[0]), y: 0.5 * (upper[1] + lower[1]) };
}

/* ============================================================================
 * Panel generation
 * ==========================================================================*/

/**
 * Build the panelled body for a NACA spec.
 *
 * `nPanels` is the target before adaptive refinement; the result is capped at
 * `maxPanels`. Refinement bisects any panel whose turning angle (curvature times
 * length) exceeds `curvatureTol`, which is the geometrically meaningful
 * criterion: a panel that turns through more than a few degrees is one the
 * straight-panel approximation is starting to lie about.
 *
 * Returns the node ring (closed, clockwise, node 0 at the trailing edge) plus
 * per-node curvature, and the parameter values used, so callers can reproduce
 * or refine further.
 */
export function buildBody(spec, opts = {}) {
  const nPanels = Math.max(60, Math.min(opts.panels ?? 240, opts.maxPanels ?? 500));
  const maxPanels = Math.max(nPanels, opts.maxPanels ?? 500);
  const curvatureTol = opts.curvatureTol ?? 0.12; // radians of turn per panel
  const refinePasses = opts.refinePasses ?? 3;

  const te = trailingEdge(spec);

  // --- Dense analytic ring, then the spline through it ---------------------
  const D = DENSE_PER_SURFACE;
  const dense = 2 * D;
  const dx = new Float64Array(dense);
  const dy = new Float64Array(dense);
  for (let k = 0; k < dense; k++) {
    const tau = (2 * k) / dense;
    const { x, side } = stationForTau(tau);
    const p = closedSurfacePoint(x, side, spec, te);
    dx[k] = p[0];
    dy[k] = p[1];
  }
  const sp = fitClosedSpline(dx, dy);

  // Arc length at each dense knot, so a ring parameter can be turned into a
  // spline parameter by interpolation instead of by root-finding.
  const tauToSeg = (tau) => {
    const g = (tau / 2) * dense;
    let seg = Math.floor(g);
    if (seg >= dense) seg = dense - 1;
    if (seg < 0) seg = 0;
    return { seg, frac: g - seg };
  };

  const ev = { x: 0, y: 0, dx: 0, dy: 0, ddx: 0, ddy: 0 };
  const evalTau = (tau) => {
    const { seg, frac } = tauToSeg(tau);
    const s = sp.u[seg] + frac * (sp.u[seg + 1] - sp.u[seg]);
    splineEval(sp, seg, s, ev);
    const sp1 = Math.hypot(ev.dx, ev.dy) || 1;
    // Curvature of a parametric curve; the parameterisation is near-arc-length
    // so |r'| is close to 1, but the general form costs nothing and stays honest
    // where the knot spacing varies.
    const kappa = (ev.dx * ev.ddy - ev.dy * ev.ddx) / (sp1 * sp1 * sp1);
    return { x: ev.x, y: ev.y, tx: ev.dx / sp1, ty: ev.dy / sp1, kappa };
  };

  // --- Cosine-spaced panel nodes ------------------------------------------
  // The trailing edge must be a node (it is where the Kutta condition lives),
  // and it is tau = 0 by construction.
  let taus = [];
  for (let i = 0; i < nPanels; i++) taus.push((2 * i) / nPanels);

  // --- Adaptive refinement -------------------------------------------------
  for (let pass = 0; pass < refinePasses && taus.length < maxPanels; pass++) {
    const pts = taus.map(evalTau);
    const next = [];
    let added = 0;
    for (let i = 0; i < taus.length; i++) {
      next.push(taus[i]);
      const j = (i + 1) % taus.length;
      const t0 = taus[i];
      const t1 = j === 0 ? 2 : taus[j];
      const len = Math.hypot(pts[j].x - pts[i].x, pts[j].y - pts[i].y);
      const kappa = 0.5 * (Math.abs(pts[i].kappa) + Math.abs(pts[j].kappa));
      if (kappa * len > curvatureTol && taus.length + added < maxPanels) {
        next.push(0.5 * (t0 + t1));
        added++;
      }
    }
    if (added === 0) break;
    taus = next;
  }

  // --- Node ring -----------------------------------------------------------
  const n = taus.length;
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  const kappa = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = evalTau(taus[i]);
    X[i] = p.x;
    Y[i] = p.y;
    kappa[i] = p.kappa;
  }
  // Pin the trailing edge exactly: the spline passes within ~1e-9 of it anyway,
  // but the Kutta condition is stated at a point and it should be *the* point.
  X[0] = te.x;
  Y[0] = te.y;

  return {
    spec,
    n,
    X,
    Y,
    nodeCurvature: kappa,
    taus: Float64Array.from(taus),
    spline: sp,
    trailingEdge: te,
    evalTau,
  };
}

/**
 * Panel geometry derived from a node ring: midpoints, unit tangents, outward
 * normals, lengths, panel-frame angles and panel-centre curvature.
 *
 * Node ordering is clockwise (trailing edge -> lower -> leading edge -> upper ->
 * trailing edge), so the body interior lies to the right of the direction of
 * travel and the outward normal is the tangent turned +90 degrees.
 */
export function panelGeometry(body) {
  const { n, X, Y, nodeCurvature } = body;
  const midX = new Float64Array(n);
  const midY = new Float64Array(n);
  const tx = new Float64Array(n);
  const ty = new Float64Array(n);
  const nx = new Float64Array(n);
  const ny = new Float64Array(n);
  const len = new Float64Array(n);
  const phi = new Float64Array(n);
  const curv = new Float64Array(n);
  const sMid = new Float64Array(n); // arc length from the trailing edge
  const sNode = new Float64Array(n + 1);

  let perimeter = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ex = X[j] - X[i];
    const ey = Y[j] - Y[i];
    const l = Math.hypot(ex, ey);
    if (!(l > 0)) return null; // duplicate node: degenerate geometry
    len[i] = l;
    tx[i] = ex / l;
    ty[i] = ey / l;
    nx[i] = -ty[i];
    ny[i] = tx[i];
    midX[i] = 0.5 * (X[i] + X[j]);
    midY[i] = 0.5 * (Y[i] + Y[j]);
    phi[i] = Math.atan2(ey, ex);
    curv[i] = 0.5 * (nodeCurvature[i] + nodeCurvature[j]);
    sNode[i] = perimeter;
    sMid[i] = perimeter + 0.5 * l;
    perimeter += l;
  }
  sNode[n] = perimeter;

  return { n, X, Y, midX, midY, tx, ty, nx, ny, len, phi, curv, sMid, sNode, perimeter };
}

/**
 * Index of the panel whose midpoint is nearest the leading edge, and the split
 * of the ring into lower (0..iLE) and upper (iLE+1..n-1) runs.
 *
 * Used only for reporting and for seeding the boundary layer; the stagnation
 * point itself comes from the solved surface velocity, not from geometry.
 */
export function leadingEdgeIndex(geo) {
  let best = 0;
  let bestX = Infinity;
  for (let i = 0; i < geo.n; i++) {
    if (geo.midX[i] < bestX) {
      bestX = geo.midX[i];
      best = i;
    }
  }
  return best;
}

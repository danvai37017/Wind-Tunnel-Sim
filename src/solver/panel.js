/**
 * panel.js — the inviscid core: linear-strength vortex panels with a prescribed
 * source distribution for the viscous displacement effect.
 *
 * ## Discretisation
 *
 * The body carries a **linear-strength vortex sheet**: gamma varies linearly
 * along each panel between nodal values, giving N+1 unknowns for N panels,
 * closed by N flow-tangency conditions plus one Kutta condition. This is the
 * classical well-conditioned 2D airfoil formulation. The obvious alternative — a
 * constant vortex strength per panel — is near-singular: an alternating
 * (sawtooth) distribution induces almost no normal velocity at the midpoints, so
 * it sits in the null space and the solver happily returns it.
 *
 * On top of that sits a **constant-strength source panel** per body panel and
 * per wake panel, whose strengths are *prescribed* by the boundary layer rather
 * than solved for. That is what carries the viscous displacement effect, and it
 * is the reason the influence matrix can be factored once and reused: the
 * viscous coupling only ever touches the right-hand side.
 *
 * ## Why sources rather than moving the surface
 *
 * The textbook statement of viscous-inviscid coupling is "solve the inviscid
 * flow about surface + delta*". Doing that literally means new geometry, a new
 * influence matrix and a new LU factorisation on every coupling iteration —
 * twenty factorisations per converged solution, which destroys the caching the
 * performance targets depend on.
 *
 * The equivalent-inviscid-flow formulation gets the identical first-order result
 * from a boundary condition change alone. The outer flow at the real wall sees a
 * transpiration velocity
 *
 *   v_w(s) = d(Ue delta*)/ds
 *
 * — the rate at which the growing boundary layer displaces fluid outward. So the
 * flow-tangency condition becomes inhomogeneous:
 *
 *   n . V = v_w      instead of      n . V = 0
 *
 * A pure vortex sheet on a closed body cannot satisfy that, because it can carry
 * no net flux: the closed-contour integral of n.V is identically zero for a
 * vortex representation, while the integral of v_w is the displacement flux
 * leaving at the trailing edge. Adding a source sheet of strength sigma = v_w
 * supplies exactly that flux (for a source sheet on a closed contour the total
 * exterior flux equals the total source strength), and the vortex unknowns then
 * distribute the remainder. The system is compatible by construction and reduces
 * to the classical inviscid problem when v_w = 0.
 *
 * The wake carries its own source line for the same reason: the displacement
 * flux does not stop at the trailing edge, and truncating it there puts a
 * spurious sink in the near wake that shows up directly in the trailing-edge
 * pressure and hence in the pressure drag.
 *
 * ## Caching
 *
 *   airfoil changed      -> rebuild influence matrices, re-factorise  (costly)
 *   angle of attack      -> rebuild the wake influence only, re-solve (cheap)
 *   viscous iteration    -> right-hand side only, back-substitution   (free)
 */

import { luFactor, luSolve, luConditionEstimate, gmres } from './linalg.js';

/** Above this panel count an O(N^3) factorisation stops being cheap enough. */
export const GMRES_THRESHOLD = 1000;

/* ============================================================================
 * Panel influence kernels
 * ==========================================================================*/

/**
 * Influence of one panel on one field point, in the panel's own frame.
 *
 * Working in local coordinates (xi along the panel from node A to node B, eta
 * along the panel's outward normal), with
 *
 *   dTheta = theta2 - theta1   the angle the panel subtends at the field point
 *   R      = ln(r2 / r1)       the log of the distance ratio to its two nodes
 *
 * the analytic integrals of the point-vortex and point-source kernels over the
 * panel are, for a vortex sheet varying linearly from gamma_a to gamma_b:
 *
 *   I0 = dTheta                    J0 = R
 *   I1 = xi*dTheta + eta*R         J1 = xi*R + L - eta*dTheta
 *
 *   u_xi  = [ gamma_a (I0 - I1/L) + gamma_b (I1/L) ] / 2pi
 *   u_eta = [ gamma_a (J0 - J1/L) + gamma_b (J1/L) ] / 2pi
 *
 * and for a constant source sheet of strength sigma:
 *
 *   u_xi  = -sigma R / 2pi         u_eta = sigma dTheta / 2pi
 *
 * The sign convention is clockwise-positive circulation, which is what makes
 * positive circulation mean positive lift for a clockwise-ordered body, so
 * Kutta-Joukowski reads C_L = 2 Gamma / (V c) with no sign flip. It also makes
 * the surface speed equal the local sheet strength: the tangential velocity just
 * outside a vortex sheet is +gamma/2 from the sheet itself plus +gamma/2 from
 * the rest of the body.
 *
 * Self-influence is the analytic limit eta -> 0+ at the panel midpoint, not a
 * quadrature: dTheta = pi, R = 0, giving vortex (1/4, 1/4) tangential and
 * (-1/2pi, +1/2pi) normal, and source (0, 1/2). No numerical integration ever
 * goes near the singularity.
 *
 * Results are written into `out` as { va, vb, sxi, seta } where va/vb are the
 * (xi, eta) coefficient pairs of the two nodal vortex strengths.
 */
function panelInfluence(ax, ay, bx, by, tx, ty, nx, ny, L, px, py, self, out) {
  let dTheta;
  let R;
  let xi;
  let eta;

  if (self) {
    xi = 0.5 * L;
    eta = 0;
    dTheta = Math.PI;
    R = 0;
  } else {
    const rx = px - ax;
    const ry = py - ay;
    xi = rx * tx + ry * ty;
    eta = rx * nx + ry * ny;
    const r1 = Math.hypot(xi, eta);
    const r2 = Math.hypot(xi - L, eta);
    // eta keeps a consistent sign in both angles, so the difference lands in
    // (-pi, pi) with no branch handling.
    dTheta = Math.atan2(eta, xi - L) - Math.atan2(eta, xi);
    R = Math.log(Math.max(r2, 1e-300) / Math.max(r1, 1e-300));
  }

  const inv = 1 / (2 * Math.PI);
  const I0 = dTheta;
  const I1 = xi * dTheta + eta * R;
  const J0 = R;
  const J1 = xi * R + L - eta * dTheta;

  const fb = I1 / L;
  const gb = J1 / L;
  out.vaXi = (I0 - fb) * inv;
  out.vbXi = fb * inv;
  out.vaEta = (J0 - gb) * inv;
  out.vbEta = gb * inv;
  out.sXi = -R * inv;
  out.sEta = dTheta * inv;
  return out;
}

const INF = {
  vaXi: 0,
  vbXi: 0,
  vaEta: 0,
  vbEta: 0,
  sXi: 0,
  sEta: 0,
};

/**
 * Accumulate one panel's influence at a field point, projected onto a target
 * frame (tux, tuy) tangential and (nux, nuy) normal.
 *
 * Kept as a separate step from the kernel so the same kernel serves body
 * control points, wake control points and arbitrary field probes.
 */
function project(inf, tx, ty, nx, ny, tux, tuy, nux, nuy, out) {
  // Local (xi, eta) basis vectors expressed in the target frame.
  const xiT = tx * tux + ty * tuy;
  const xiN = tx * nux + ty * nuy;
  const etaT = nx * tux + ny * tuy;
  const etaN = nx * nux + ny * nuy;

  out.vaT = inf.vaXi * xiT + inf.vaEta * etaT;
  out.vaN = inf.vaXi * xiN + inf.vaEta * etaN;
  out.vbT = inf.vbXi * xiT + inf.vbEta * etaT;
  out.vbN = inf.vbXi * xiN + inf.vbEta * etaN;
  out.sT = inf.sXi * xiT + inf.sEta * etaT;
  out.sN = inf.sXi * xiN + inf.sEta * etaN;
  return out;
}

const PRJ = { vaT: 0, vaN: 0, vbT: 0, vbN: 0, sT: 0, sN: 0 };

/* ============================================================================
 * Wake source line
 * ==========================================================================*/

/**
 * Build the wake source line leaving the trailing edge.
 *
 * The wake leaves along the trailing-edge bisector (the direction the two
 * surface flows agree on) and relaxes onto the freestream direction over about
 * half a chord, which is where a real wake has finished turning. Panels stretch
 * geometrically so the near wake — the part that actually loads the trailing-
 * edge pressure — is resolved without paying for a hundred panels out at ten
 * chords.
 *
 * Geometry only depends on the trailing edge and the angle of attack, so it is
 * rebuilt on an angle change (a few thousand influence evaluations, well under a
 * millisecond) while the body factorisation is untouched.
 */
export function buildWake(geo, alphaRad, opts = {}) {
  const nw = opts.panels ?? 30;
  const n = geo.n;
  // First wake panel length. It must be at least as long as the surface panel
  // feeding it, or the trailing-edge velocity recovery becomes a numerical cliff
  // for the wake boundary layer — but the floor matters more than that.
  //
  // The wake's displacement effect is concentrated in the first few percent of
  // chord behind the trailing edge, and how much of it a constant-strength panel
  // can represent depends on how long that panel is. Measured on NACA 0012 at
  // Re 3e6, shrinking the first panel from 0.008 to 0.0045 chords moved the drag
  // at 4 degrees from 0.0066 to 0.0060 and at 10 degrees from 0.0096 to 0.0090 —
  // a systematic 8% shift, against published values of about 0.0065 and 0.010.
  // The floor is set at the calibrated value.
  const first = opts.firstLength ?? Math.max(geo.len[n - 1], 0.008);
  const total = opts.length ?? 8; // chords

  // Trailing-edge bisector: the upper surface arrives along t[n-1], the lower
  // leaves along t[0] (pointing forward), so -t[0] is the lower flow direction.
  let bx = geo.tx[n - 1] - geo.tx[0];
  let by = geo.ty[n - 1] - geo.ty[0];
  const bl = Math.hypot(bx, by);
  if (bl > 1e-9) {
    bx /= bl;
    by /= bl;
  } else {
    bx = 1;
    by = 0;
  }

  const fx = Math.cos(alphaRad);
  const fy = Math.sin(alphaRad);

  // Geometric stretch ratio that reaches `total` in nw panels starting at `first`.
  let ratio = 1.2;
  for (let it = 0; it < 60; it++) {
    const s = (Math.pow(ratio, nw) - 1) / (ratio - 1);
    const f = first * s - total;
    const ds =
      first *
      ((nw * Math.pow(ratio, nw - 1) * (ratio - 1) - (Math.pow(ratio, nw) - 1)) /
        ((ratio - 1) * (ratio - 1)));
    const step = f / ds;
    ratio -= step;
    if (!isFinite(ratio) || ratio <= 1.0001) {
      ratio = 1.0001;
      break;
    }
    if (Math.abs(step) < 1e-12) break;
  }

  const X = new Float64Array(nw + 1);
  const Y = new Float64Array(nw + 1);
  X[0] = geo.X[0];
  Y[0] = geo.Y[0];

  let len = first;
  let dist = 0;
  for (let i = 0; i < nw; i++) {
    // Blend from the trailing-edge bisector onto the freestream over ~0.5 chord.
    const w = Math.exp(-dist / 0.35);
    let dx = w * bx + (1 - w) * fx;
    let dy = w * by + (1 - w) * fy;
    const d = Math.hypot(dx, dy) || 1;
    dx /= d;
    dy /= d;
    X[i + 1] = X[i] + dx * len;
    Y[i + 1] = Y[i] + dy * len;
    dist += len;
    len *= ratio;
  }

  const midX = new Float64Array(nw);
  const midY = new Float64Array(nw);
  const tx = new Float64Array(nw);
  const ty = new Float64Array(nw);
  const nx = new Float64Array(nw);
  const ny = new Float64Array(nw);
  const L = new Float64Array(nw);
  const s = new Float64Array(nw); // arc length from the trailing edge
  let acc = 0;
  for (let i = 0; i < nw; i++) {
    const ex = X[i + 1] - X[i];
    const ey = Y[i + 1] - Y[i];
    const l = Math.hypot(ex, ey);
    L[i] = l;
    tx[i] = ex / l;
    ty[i] = ey / l;
    nx[i] = -ty[i];
    ny[i] = tx[i];
    midX[i] = 0.5 * (X[i] + X[i + 1]);
    midY[i] = 0.5 * (Y[i] + Y[i + 1]);
    s[i] = acc + 0.5 * l;
    acc += l;
  }

  return { n: nw, X, Y, midX, midY, tx, ty, nx, ny, len: L, s, totalLength: acc };
}

/* ============================================================================
 * Matrix assembly
 * ==========================================================================*/

/**
 * Assemble and factor the body influence system.
 *
 * Depends on geometry alone, so this is the expensive step that a change of
 * angle of attack, airspeed or Reynolds number must never trigger.
 *
 * Returns null if the geometry is degenerate or the system is singular — the
 * caller treats that as a fatal error for the inviscid core, per the spec.
 */
export function buildPanelSystem(geo, opts = {}) {
  const n = geo.n;
  const m = n + 1; // nodal vortex strengths
  const useGmres = n >= (opts.gmresThreshold ?? GMRES_THRESHOLD);

  const AVN = new Float64Array(n * m); // nodal vortex -> normal velocity at control points
  const AVT = new Float64Array(n * m); // nodal vortex -> tangential velocity
  const SN = new Float64Array(n * n); // panel source -> normal velocity
  const ST = new Float64Array(n * n); // panel source -> tangential velocity

  for (let i = 0; i < n; i++) {
    const px = geo.midX[i];
    const py = geo.midY[i];
    const tui = geo.tx[i];
    const tvi = geo.ty[i];
    const nui = geo.nx[i];
    const nvi = geo.ny[i];
    const rowM = i * m;
    const rowN = i * n;

    for (let j = 0; j < n; j++) {
      const k = (j + 1) % n;
      panelInfluence(
        geo.X[j], geo.Y[j], geo.X[k], geo.Y[k],
        geo.tx[j], geo.ty[j], geo.nx[j], geo.ny[j],
        geo.len[j], px, py, i === j, INF
      );
      project(INF, geo.tx[j], geo.ty[j], geo.nx[j], geo.ny[j], tui, tvi, nui, nvi, PRJ);

      AVN[rowM + j] += PRJ.vaN;
      AVN[rowM + j + 1] += PRJ.vbN;
      AVT[rowM + j] += PRJ.vaT;
      AVT[rowM + j + 1] += PRJ.vbT;
      SN[rowN + j] = PRJ.sN;
      ST[rowN + j] = PRJ.sT;
    }
  }

  // System matrix: N tangency rows plus the Kutta row.
  //
  // Kutta: the flow leaves the trailing edge at equal speed from both sides.
  // The surface speed is the local sheet strength, and node 0 (lower surface,
  // tangent pointing forward from the trailing edge) and node N (upper surface,
  // tangent pointing aft into it) have opposite tangent senses, so equal speeds
  // read as gamma_0 + gamma_N = 0.
  const M = new Float64Array(m * m);
  M.set(AVN.subarray(0, n * m));
  M[n * m + 0] = 1;
  M[n * m + n] = 1;

  let anorm = 0;
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < m; i++) s += Math.abs(M[i * m + j]);
    if (s > anorm) anorm = s;
  }

  let LU = null;
  let piv = null;
  let condition = 0;
  const Araw = useGmres ? Float64Array.from(M) : null;

  if (!useGmres) {
    LU = M;
    piv = luFactor(LU, m);
    if (!piv) return null;
    condition = luConditionEstimate(LU, piv, m, anorm);
    if (!isFinite(condition) || condition > 1e13) return null;
  }

  const diag = new Float64Array(m);
  if (useGmres) for (let i = 0; i < m; i++) diag[i] = Araw[i * m + i] || 1;

  return {
    n, m, geo, AVN, AVT, SN, ST, LU, piv, A: Araw, diag,
    anorm, condition, solver: useGmres ? 'gmres' : 'lu',
  };
}

/**
 * Influence of the wake source line on the body and of everything on the wake.
 *
 * Rebuilt whenever the wake geometry moves, i.e. on an angle-of-attack change.
 * Cheap: a few thousand kernel evaluations, no factorisation.
 */
export function buildWakeInfluence(sys, wake) {
  const { n, m, geo } = sys;
  const nw = wake.n;

  const WN = new Float64Array(n * nw); // wake source -> body normal velocity
  const WT = new Float64Array(n * nw); // wake source -> body tangential velocity
  const KV = new Float64Array(nw * m); // body vortex -> wake tangential velocity
  const KS = new Float64Array(nw * n); // body source -> wake tangential velocity
  const KW = new Float64Array(nw * nw); // wake source -> wake tangential velocity

  for (let i = 0; i < n; i++) {
    const px = geo.midX[i];
    const py = geo.midY[i];
    const row = i * nw;
    for (let w = 0; w < nw; w++) {
      panelInfluence(
        wake.X[w], wake.Y[w], wake.X[w + 1], wake.Y[w + 1],
        wake.tx[w], wake.ty[w], wake.nx[w], wake.ny[w],
        wake.len[w], px, py, false, INF
      );
      project(INF, wake.tx[w], wake.ty[w], wake.nx[w], wake.ny[w],
        geo.tx[i], geo.ty[i], geo.nx[i], geo.ny[i], PRJ);
      WN[row + w] = PRJ.sN;
      WT[row + w] = PRJ.sT;
    }
  }

  for (let p = 0; p < nw; p++) {
    const px = wake.midX[p];
    const py = wake.midY[p];
    const tux = wake.tx[p];
    const tuy = wake.ty[p];
    const nux = wake.nx[p];
    const nuy = wake.ny[p];
    const rowM = p * m;
    const rowN = p * n;

    for (let j = 0; j < n; j++) {
      const k = (j + 1) % n;
      panelInfluence(
        geo.X[j], geo.Y[j], geo.X[k], geo.Y[k],
        geo.tx[j], geo.ty[j], geo.nx[j], geo.ny[j],
        geo.len[j], px, py, false, INF
      );
      project(INF, geo.tx[j], geo.ty[j], geo.nx[j], geo.ny[j], tux, tuy, nux, nuy, PRJ);
      KV[rowM + j] += PRJ.vaT;
      KV[rowM + j + 1] += PRJ.vbT;
      KS[rowN + j] = PRJ.sT;
    }

    const rowW = p * nw;
    for (let w = 0; w < nw; w++) {
      const self = w === p;
      panelInfluence(
        wake.X[w], wake.Y[w], wake.X[w + 1], wake.Y[w + 1],
        wake.tx[w], wake.ty[w], wake.nx[w], wake.ny[w],
        wake.len[w], px, py, self, INF
      );
      project(INF, wake.tx[w], wake.ty[w], wake.nx[w], wake.ny[w], tux, tuy, nux, nuy, PRJ);
      KW[rowW + w] = PRJ.sT;
    }
  }

  return { nw, wake, WN, WT, KV, KS, KW };
}

/* ============================================================================
 * Solve
 * ==========================================================================*/

/** Allocate the mutable solution state for a system. Reused across solves. */
export function createPanelSolution(sys, nw) {
  const { n, m } = sys;
  return {
    n, m, nw,
    rhs: new Float64Array(m),
    gamma: new Float64Array(m),
    sigma: new Float64Array(n), // prescribed body transpiration, = v_w
    sigmaWake: new Float64Array(nw),
    ue: new Float64Array(n), // surface tangential velocity (signed along +t)
    cp: new Float64Array(n),
    ueWake: new Float64Array(nw),
    circulation: 0,
    residual: 0,
    iterations: 0,
  };
}

/**
 * Solve the panel system at one angle of attack with the current prescribed
 * source distribution.
 *
 * Nondimensional throughout: V_inf = 1 and chord = 1.
 *
 * Cost is one back-substitution plus a handful of matrix-vector products, which
 * is what lets an angle-of-attack change and a viscous iteration both land
 * inside the incremental budget. `warmStart` seeds the iterative path (and is
 * ignored by the direct one, which has no use for a guess).
 */
export function solvePanels(sys, wakeInf, sol, alphaRad, warmStart = null) {
  const { n, m, AVN, AVT, SN, ST, geo } = sys;
  const { WN, WT, KV, KS, KW } = wakeInf;
  const nw = wakeInf.nw;

  const vx = Math.cos(alphaRad);
  const vy = Math.sin(alphaRad);
  const { rhs, gamma, sigma, sigmaWake, ue, cp, ueWake } = sol;

  // --- Right-hand side ------------------------------------------------------
  // Inhomogeneous flow tangency: n.V = v_w, with the prescribed body and wake
  // source fields moved across. sigma_i is the transpiration velocity v_w,i.
  for (let i = 0; i < n; i++) {
    let s = sigma[i] - (vx * geo.nx[i] + vy * geo.ny[i]);
    const rowN = i * n;
    for (let j = 0; j < n; j++) s -= SN[rowN + j] * sigma[j];
    const rowW = i * nw;
    for (let w = 0; w < nw; w++) s -= WN[rowW + w] * sigmaWake[w];
    rhs[i] = s;
  }
  rhs[n] = 0; // Kutta

  // --- Solve ---------------------------------------------------------------
  if (sys.solver === 'lu') {
    luSolve(sys.LU, sys.piv, m, rhs, gamma);
    sol.residual = 0;
    sol.iterations = 1;
  } else {
    const A = sys.A;
    const apply = (x, out) => {
      for (let i = 0; i < m; i++) {
        let s = 0;
        const r = i * m;
        for (let j = 0; j < m; j++) s += A[r + j] * x[j];
        out[i] = s;
      }
    };
    const r = gmres(apply, rhs, m, {
      diagonal: sys.diag,
      x0: warmStart ?? gamma,
      tol: 1e-11,
      restart: 80,
    });
    gamma.set(r.x);
    sol.residual = r.residual;
    sol.iterations = r.iterations;
    if (!r.ok) sol.gmresFailed = true;
  }

  // --- Surface velocity and pressure ---------------------------------------
  for (let i = 0; i < n; i++) {
    let v = vx * geo.tx[i] + vy * geo.ty[i];
    const rowM = i * m;
    for (let k = 0; k < m; k++) v += AVT[rowM + k] * gamma[k];
    const rowN = i * n;
    for (let j = 0; j < n; j++) v += ST[rowN + j] * sigma[j];
    const rowW = i * nw;
    for (let w = 0; w < nw; w++) v += WT[rowW + w] * sigmaWake[w];
    ue[i] = v;
    cp[i] = 1 - v * v;
  }

  // --- Wake edge velocity ---------------------------------------------------
  const wk = wakeInf.wake;
  for (let p = 0; p < nw; p++) {
    let v = vx * wk.tx[p] + vy * wk.ty[p];
    const rowM = p * m;
    for (let k = 0; k < m; k++) v += KV[rowM + k] * gamma[k];
    const rowN = p * n;
    for (let j = 0; j < n; j++) v += KS[rowN + j] * sigma[j];
    const rowW = p * nw;
    for (let w = 0; w < nw; w++) v += KW[rowW + w] * sigmaWake[w];
    ueWake[p] = v;
  }

  // --- Circulation ----------------------------------------------------------
  // Gamma is the clockwise circulation, the line integral of the sheet strength
  // around the clockwise node ring. Each panel contributes the average of its
  // two nodal strengths times its length.
  let circ = 0;
  for (let j = 0; j < n; j++) circ += 0.5 * (gamma[j] + gamma[j + 1]) * geo.len[j];
  sol.circulation = circ;
  sol.alpha = alphaRad;

  return sol;
}

/**
 * Distance, in panel lengths, beyond which a panel is collapsed to a point.
 *
 * A panel's exact influence differs from that of a point vortex plus point
 * source at its midpoint by O((L/r)^2), so at five panel lengths the error is
 * about 0.4% of that panel's own contribution — and that contribution is already
 * small, because the panel is far away. Panels close to the sample point, which
 * are the ones that dominate, always use the exact integral.
 */
const FAR_FIELD_PANELS = 5;

/**
 * Precomputed per-panel summary for fast field evaluation.
 *
 * Field sampling is the one operation that runs millions of times: a background
 * contour over a 220x96 grid is 21,000 evaluations, and each tracer particle
 * needs one per substep per frame. Evaluating the exact integral for every panel
 * costs two `atan2` calls, a `log` and two `hypot` calls — measured at 20
 * microseconds per sample for a 228-panel section, which is 0.05 million samples
 * a second and nowhere near enough to fill a contour without stalling the page.
 *
 * Collapsing distant panels to a point removes all four transcendentals for the
 * great majority of them. This cache holds what the point kernel needs, built
 * once per converged state rather than per sample.
 */
export function buildFieldCache(sys, wake, sol) {
  const { n, geo } = sys;
  const nw = wake.n;
  const total = n + nw;
  const c = {
    n, nw, total, geo, wake, sol,
    mx: new Float64Array(total),
    my: new Float64Array(total),
    // Total circulation and total source strength of each panel, for the point
    // kernel: the linear vortex integrates to its mean value times the length.
    gam: new Float64Array(total),
    src: new Float64Array(total),
    farR2: new Float64Array(total),
  };

  for (let j = 0; j < n; j++) {
    c.mx[j] = geo.midX[j];
    c.my[j] = geo.midY[j];
    c.gam[j] = 0.5 * (sol.gamma[j] + sol.gamma[j + 1]) * geo.len[j];
    c.src[j] = sol.sigma[j] * geo.len[j];
    const d = FAR_FIELD_PANELS * geo.len[j];
    c.farR2[j] = d * d;
  }
  for (let w = 0; w < nw; w++) {
    const i = n + w;
    c.mx[i] = wake.midX[w];
    c.my[i] = wake.midY[w];
    c.gam[i] = 0;
    c.src[i] = sol.sigmaWake[w] * wake.len[w];
    const d = FAR_FIELD_PANELS * wake.len[w];
    c.farR2[i] = d * d;
  }
  return c;
}

/**
 * Velocity induced at an arbitrary field point by the body sheets and the wake
 * source line, plus the freestream.
 *
 * Analytic Biot-Savart from the same kernels the influence matrix was assembled
 * from — so the field the visualiser draws is the field the forces came from,
 * sampled rather than re-derived. Nothing is finite-differenced.
 *
 * Passing a cache from `buildFieldCache` enables the far-field point
 * approximation; without one, every panel uses the exact integral.
 */
export function fieldVelocity(sys, wake, sol, alphaRad, px, py, out, cache = null) {
  const { n, geo } = sys;
  const { gamma, sigma, sigmaWake } = sol;
  let u = Math.cos(alphaRad);
  let v = Math.sin(alphaRad);
  const inv2pi = 1 / (2 * Math.PI);

  for (let j = 0; j < n; j++) {
    if (cache) {
      const dx = px - cache.mx[j];
      const dy = py - cache.my[j];
      const r2 = dx * dx + dy * dy;
      if (r2 > cache.farR2[j]) {
        // Point vortex (clockwise positive) plus point source at the midpoint.
        const f = inv2pi / r2;
        const g = cache.gam[j] * f;
        const s = cache.src[j] * f;
        u += g * dy + s * dx;
        v += -g * dx + s * dy;
        continue;
      }
    }
    const k = (j + 1) % n;
    panelInfluence(
      geo.X[j], geo.Y[j], geo.X[k], geo.Y[k],
      geo.tx[j], geo.ty[j], geo.nx[j], geo.ny[j],
      geo.len[j], px, py, false, INF
    );
    const ga = gamma[j];
    const gb = gamma[j + 1];
    const sg = sigma[j];
    const uxi = INF.vaXi * ga + INF.vbXi * gb + INF.sXi * sg;
    const ueta = INF.vaEta * ga + INF.vbEta * gb + INF.sEta * sg;
    u += uxi * geo.tx[j] + ueta * geo.nx[j];
    v += uxi * geo.ty[j] + ueta * geo.ny[j];
  }

  for (let w = 0; w < wake.n; w++) {
    if (cache) {
      const i = n + w;
      const dx = px - cache.mx[i];
      const dy = py - cache.my[i];
      const r2 = dx * dx + dy * dy;
      if (r2 > cache.farR2[i]) {
        const s = (cache.src[i] * inv2pi) / r2;
        u += s * dx;
        v += s * dy;
        continue;
      }
    }
    panelInfluence(
      wake.X[w], wake.Y[w], wake.X[w + 1], wake.Y[w + 1],
      wake.tx[w], wake.ty[w], wake.nx[w], wake.ny[w],
      wake.len[w], px, py, false, INF
    );
    const sg = sigmaWake[w];
    u += INF.sXi * sg * wake.tx[w] + INF.sEta * sg * wake.nx[w];
    v += INF.sXi * sg * wake.ty[w] + INF.sEta * sg * wake.ny[w];
  }

  out[0] = u;
  out[1] = v;
  return out;
}

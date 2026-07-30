/**
 * aero.js — steady-state section aerodynamics: the two "instant" engines.
 *
 *   Engine 1  Vortex panel method (constant-strength sheet, Kutta condition).
 *             Gives circulation, C_L and the surface C_p distribution.
 *   Engine 3  Boundary-layer integral method (Thwaites laminar, Head turbulent).
 *             Gives the chordwise separation point x/c.
 *
 * Both are steady, nondimensional and cheap: a full airfoil change costs a few
 * milliseconds (one LU factorisation), and an angle-of-attack change costs a
 * back-substitution — microseconds. That is what lets the dashboard track the
 * sliders exactly while the LBM field animates independently.
 *
 * Pure numerics: no NACA knowledge, no React, no DOM. The caller supplies a
 * closed polygon of surface nodes, which keeps this module testable in isolation.
 */

/* ============================================================================
 * Linear algebra — Gaussian elimination with partial pivoting, kept factored.
 * ==========================================================================*/

/**
 * In-place LU decomposition with partial pivoting (Doolittle).
 *
 * This is Gaussian elimination with partial pivoting, with the multipliers kept
 * rather than discarded. The influence matrix depends only on the geometry, so
 * keeping the factors turns every subsequent angle of attack into an O(N^2)
 * substitution instead of an O(N^3) re-elimination.
 *
 * Returns the pivot permutation, or null if the matrix is singular.
 */
function luFactor(A, n) {
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;

  for (let k = 0; k < n; k++) {
    let best = Math.abs(A[k * n + k]);
    let p = k;
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i * n + k]);
      if (v > best) {
        best = v;
        p = i;
      }
    }
    if (!(best > 1e-14)) return null;

    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const t = A[k * n + j];
        A[k * n + j] = A[p * n + j];
        A[p * n + j] = t;
      }
      const t = piv[k];
      piv[k] = piv[p];
      piv[p] = t;
    }

    const d = A[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const m = A[i * n + k] / d;
      A[i * n + k] = m;
      if (m === 0) continue;
      for (let j = k + 1; j < n; j++) A[i * n + j] -= m * A[k * n + j];
    }
  }
  return piv;
}

/** Solve LU x = P b into `out`. */
function luSolve(LU, piv, n, b, out) {
  for (let i = 0; i < n; i++) {
    let s = b[piv[i]];
    for (let j = 0; j < i; j++) s -= LU[i * n + j] * out[j];
    out[i] = s;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = out[i];
    for (let j = i + 1; j < n; j++) s -= LU[i * n + j] * out[j];
    out[i] = s / LU[i * n + i];
  }
  return out;
}

/* ============================================================================
 * Engine 1 — constant-strength vortex panel method
 * ==========================================================================*/

/**
 * Panel influence kernels.
 *
 * Working in panel j's local frame (xi along the panel, eta the outward normal)
 * and integrating the point-source / point-vortex kernels along the panel gives,
 * for the field point at panel i's midpoint:
 *
 *   source, strength sigma:  u_xi = (sigma/2pi)(-L),  u_eta = (sigma/2pi)(T)
 *   vortex, strength gamma:  u_xi = (gamma/2pi)(T),   u_eta = (gamma/2pi)(L)
 *
 * where T = theta2 - theta1 is the angle the panel subtends at the field point
 * and L = ln(r2/r1) with r1, r2 the distances to the panel's two nodes. The two
 * kernels are duals: swapping T and L (and a sign) turns one into the other.
 *
 * Rotating back to global and projecting onto panel i's normal and tangent, with
 * s = sin(phi_j - phi_i) and c = cos(phi_j - phi_i):
 *
 *   AN_ij = ( c*T - s*L ) / 2pi     source -> normal velocity at i
 *   AT_ij = (-s*T - c*L ) / 2pi     source -> tangential velocity at i
 *   BN_ij = ( s*T + c*L ) / 2pi     vortex -> normal velocity at i
 *   BT_ij = ( c*T - s*L ) / 2pi     vortex -> tangential velocity at i
 *
 * Self-influence uses T = pi, L = 0, s = 0, c = 1: AN_ii = 1/2 (a source sheet
 * emits half its strength normal to each side), AT_ii = 0, BN_ii = 0 (a vortex
 * sheet induces no normal velocity on itself) and BT_ii = +1/2 — the familiar
 * gamma/2 tangential jump the prompt quotes.
 *
 * Note the sin/cos pairing on BN: the logarithm goes with cos and the subtended
 * angle with sin. The opposite pairing is the source kernel, and using it for the
 * vortex would give BN_ii = 1/2 instead of 0.
 *
 * The vortex sign convention is clockwise-positive, which is what makes positive
 * circulation mean positive lift, so Kutta-Joukowski reads C_L = 2*Gamma/(V*c)
 * with no sign flip.
 */

/**
 * Build and factor the Hess-Smith panel system for a closed body.
 *
 * Discretisation: a constant-strength source on each panel plus a *single*
 * vortex strength shared by the whole body — N+1 unknowns closed by N
 * no-penetration conditions and one Kutta condition.
 *
 * The obvious alternative, an independent constant vortex strength per panel, is
 * what the naive reading of "constant strength vortex panel" suggests, but that
 * system is near-singular: an alternating (sawtooth) vortex distribution induces
 * almost no normal velocity at the midpoints, so it sits in the null space and
 * the solver happily returns it. Measured on NACA 0012 at zero incidence, the
 * surface speed oscillated between 0.97 and 1.24 on alternate panels, which then
 * fed |dVe/ds| spikes into the boundary layer and separated it at 5% chord. The
 * shared-vortex form has no such mode.
 *
 * `nodeX`/`nodeY` are the polygon vertices in **clockwise** order starting at the
 * trailing edge: TE -> lower surface -> LE -> upper surface -> back to TE. Panel
 * i runs from node i to node i+1 (wrapping), so panels 0 and N-1 are the two that
 * meet at the trailing edge.
 *
 * Only the geometry is baked in — the freestream direction enters through the
 * right-hand side, so one factorisation serves every angle of attack.
 */
export function buildPanelSystem(nodeX, nodeY) {
  const n = nodeX.length;
  const m = n + 1; // N source strengths + one body vortex strength

  const midX = new Float64Array(n);
  const midY = new Float64Array(n);
  const tx = new Float64Array(n);
  const ty = new Float64Array(n);
  const nx = new Float64Array(n);
  const ny = new Float64Array(n);
  const len = new Float64Array(n);
  const phi = new Float64Array(n);

  let perimeter = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const dx = nodeX[j] - nodeX[i];
    const dy = nodeY[j] - nodeY[i];
    const l = Math.hypot(dx, dy);
    if (!(l > 0)) return null; // duplicate node: degenerate geometry
    len[i] = l;
    perimeter += l;
    tx[i] = dx / l;
    ty[i] = dy / l;
    // Clockwise polygon: the interior lies to the right of travel, so the
    // outward normal is the tangent turned +90 degrees.
    nx[i] = -ty[i];
    ny[i] = tx[i];
    midX[i] = 0.5 * (nodeX[i] + nodeX[j]);
    midY[i] = 0.5 * (nodeY[i] + nodeY[j]);
    phi[i] = Math.atan2(dy, dx);
  }

  // M holds the (N+1)x(N+1) system: source columns 0..N-1, vortex column N.
  const M = new Float64Array(m * m);
  // AT and BT are kept so the surface speed (and hence C_p) can be evaluated at
  // every panel after the solve. BT collapses to one column: the vortex strength
  // is a scalar, so only the row sum is ever needed.
  const AT = new Float64Array(n * n);
  const BTcol = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const px = midX[i];
    const py = midY[i];
    let bn = 0;
    let bt = 0;

    for (let j = 0; j < n; j++) {
      let T;
      let L;
      let s;
      let c;

      if (i === j) {
        T = Math.PI;
        L = 0;
        s = 0;
        c = 1;
      } else {
        const k = (j + 1) % n;
        // Field point in panel j's local frame.
        const rx = px - nodeX[j];
        const ry = py - nodeY[j];
        const xi = rx * tx[j] + ry * ty[j];
        const eta = rx * nx[j] + ry * ny[j];

        // eta keeps a consistent sign for both angles, so the difference is the
        // subtended angle in (-pi, pi) with no branch handling needed.
        T = Math.atan2(eta, xi - len[j]) - Math.atan2(eta, xi);
        L = Math.log(
          Math.hypot(px - nodeX[k], py - nodeY[k]) / Math.hypot(px - nodeX[j], py - nodeY[j])
        );
        s = Math.sin(phi[j] - phi[i]);
        c = Math.cos(phi[j] - phi[i]);
      }

      const inv = 1 / (2 * Math.PI);
      M[i * m + j] = (c * T - s * L) * inv; // AN_ij
      AT[i * n + j] = (-s * T - c * L) * inv; // AT_ij
      bn += (s * T + c * L) * inv; // BN_ij
      bt += (c * T - s * L) * inv; // BT_ij
    }

    M[i * m + n] = bn;
    BTcol[i] = bt;
  }

  // Kutta condition on the last row: the flow leaves the trailing edge at equal
  // speed from both sides. Panel 0's tangent points forward along the lower
  // surface and panel N-1's points aft along the upper, so that reads
  // v_t,0 + v_t,N-1 = 0.
  const kr = n * m;
  const a = 0;
  const b = n - 1;
  for (let j = 0; j < n; j++) M[kr + j] = AT[a * n + j] + AT[b * n + j];
  M[kr + n] = BTcol[a] + BTcol[b];

  const piv = luFactor(M, m);
  if (!piv) return null;

  return { n, m, midX, midY, tx, ty, nx, ny, len, perimeter, LU: M, piv, AT, BTcol };
}

/**
 * Solve the panel system at one angle of attack.
 *
 * Nondimensional throughout: V_inf = 1 and chord = 1, so Kutta-Joukowski gives
 * C_L = 2*Gamma directly and the pressure coefficient follows from the surface
 * speed as C_p = 1 - (V_t/V_inf)^2.
 *
 * Costs one back-substitution plus two matrix-vector products — microseconds,
 * which is what lets the dashboard track the angle-of-attack slider exactly.
 */
export function solvePanels(sys, alphaDeg, out) {
  const { n, m, tx, ty, nx, ny, perimeter, LU, piv, AT, BTcol } = sys;
  const a = (alphaDeg * Math.PI) / 180;
  const vx = Math.cos(a);
  const vy = Math.sin(a);

  const r =
    out && out.n === n
      ? out
      : {
          n,
          rhs: new Float64Array(m),
          sol: new Float64Array(m),
          vt: new Float64Array(n),
          cp: new Float64Array(n),
        };

  for (let i = 0; i < n; i++) r.rhs[i] = -(vx * nx[i] + vy * ny[i]);
  r.rhs[n] = -(vx * tx[0] + vy * ty[0] + vx * tx[n - 1] + vy * ty[n - 1]);

  luSolve(LU, piv, m, r.rhs, r.sol);
  const gamma = r.sol[n];

  // Surface speed: freestream tangent, plus every source panel's tangential
  // influence, plus the body vortex. BT_ii = 1/2 recovers the familiar gamma/2
  // self term, but the off-diagonal sum is what shapes the C_p distribution.
  for (let i = 0; i < n; i++) {
    let v = vx * tx[i] + vy * ty[i] + gamma * BTcol[i];
    const row = i * n;
    for (let j = 0; j < n; j++) v += AT[row + j] * r.sol[j];
    r.vt[i] = v;
    r.cp[i] = 1 - v * v;
  }

  r.gamma = gamma;
  r.circulation = gamma * perimeter;
  r.cl = 2 * r.circulation; // Kutta-Joukowski, V_inf = chord = 1
  return r;
}

/* ============================================================================
 * Engine 3 — boundary-layer integral method
 * ==========================================================================*/

// Thwaites' separation criterion. The pressure-gradient parameter reaching
// -0.09 is where the correlation's shape factor hits H = 3.55 and the wall
// shear vanishes.
export const LAMBDA_SEP = -0.09;
// Upper end of the correlation's validity. The lower end is LAMBDA_SEP itself:
// lambda is tested for separation *before* being clipped, since clipping first
// would put the separation criterion permanently out of reach.
const LAMBDA_MAX = 0.25;

// Transition: local momentum-thickness Reynolds number at which the laminar
// layer is handed to the turbulent method.
export const RE_THETA_TRANSITION = 200;

// Head's method separates as H1 falls to its asymptote at 3.3, equivalently as
// the shape factor climbs through H ~ 2.4.
const H1_SEP = 3.3;
const H_TURB_SEP = 2.4;
// Shape factor the turbulent layer restarts from at transition (flat plate).
const H_TURB_START = 1.4;

// Chordwise marching grid.
const BL_DX = 0.005;
const BL_STEPS = Math.round(1 / BL_DX); // 200 intervals, 201 stations

// Separation at or inside this station is leading-edge separation (deep stall).
// Inclusive on purpose: 0.01 is itself a station on the 0.005 marching grid, and
// an exclusive test would report a fully stalled section as "separated at 0.01
// of chord" rather than at the nose.
export const LE_SEPARATION_X = 0.01;
// Anything past here is ordinary trailing-edge thickening, not separation.
export const TE_SEPARATION_X = 0.95;

/**
 * Thwaites' shape-factor correlation, H(lambda).
 *
 * Two branches (Cebeci & Bradshaw): the favourable-gradient fit quoted as
 * H = 2.61 - 3.75*lambda + 5.24*lambda^2 is only valid for lambda >= 0. In an
 * adverse gradient the correct branch is the hyperbolic one, which is what puts
 * H at 3.55 exactly at lambda = -0.09 (separation). The two agree at lambda = 0,
 * both giving H = 2.61 — the flat-plate value.
 */
export function thwaitesH(lambda) {
  const l = Math.min(LAMBDA_MAX, Math.max(LAMBDA_SEP, lambda));
  if (l >= 0) return 2.61 - 3.75 * l + 5.24 * l * l;
  return 2.088 + 0.0731 / (l + 0.14);
}

/** Head's entrainment function, F(H1) = d(theta*H1)/ds. */
function headF(h1) {
  return 0.0306 * Math.pow(Math.max(h1 - 3.0, 1e-6), -0.6169);
}

/**
 * Head's closure, inverted to give H from H1.
 *
 * H1 falls monotonically as the layer thickens, from ~7 on a flat plate down to
 * its asymptote at 3.3 where the profile separates.
 */
function headHofH1(h1) {
  const d = Math.max(h1 - 3.3, 1e-6);
  if (h1 <= 5.3) return 1.1 + 0.86 * Math.pow(d, -0.777);
  return 0.6778 + 1.1536 * Math.pow(d, -0.326);
}

/** Head's closure in the forward direction, H1 from H — used to start Phase 2. */
function headH1ofH(h) {
  if (h <= 1.6) return 3.3 + 0.8234 * Math.pow(Math.max(h - 1.1, 1e-6), -1.287);
  return 3.3 + 1.5501 * Math.pow(Math.max(h - 0.6778, 1e-6), -3.064);
}

/** Ludwieg-Tillmann skin friction. */
function skinFriction(h, reTheta) {
  return 0.246 * Math.pow(10, -0.678 * h) * Math.pow(Math.max(reTheta, 1), -0.268);
}

/**
 * Extract the suction surface from a panel solution, resampled onto the uniform
 * chordwise marching grid.
 *
 * The boundary layer is integrated in surface arc length s, not in x: the two
 * differ by the local surface slope, which is exactly where it matters most
 * (around the nose, where ds/dx is large and the suction peak sits). The uniform
 * x grid is kept because x/c is what gets reported.
 *
 * Returns the edge velocity and arc length on that grid, plus the integral
 * already accumulated between the stagnation point and the leading edge, so the
 * layer does not start from nothing at x = 0.
 *
 * This is the one place that assumes the node ordering `buildPanelNodes`
 * produces: an even panel count split exactly in half, the first half running
 * TE -> LE along the lower surface and the second LE -> TE along the upper.
 * `buildPanelSystem` itself is happy with any closed polygon.
 */
function suctionSurface(sys, sol, upper) {
  const { n, midX, len } = sys;
  const half = n / 2; // panels 0..half-1 lower (TE->LE), half..n-1 upper (LE->TE)

  // Panel indices along the suction surface, ordered leading edge -> trailing edge.
  const idx = new Int32Array(half);
  if (upper) {
    for (let k = 0; k < half; k++) idx[k] = half + k;
  } else {
    for (let k = 0; k < half; k++) idx[k] = half - 1 - k;
  }

  // Arc length of each panel midpoint, measured from the leading edge.
  const sPanel = new Float64Array(half);
  const vPanel = new Float64Array(half);
  let s = 0;
  for (let k = 0; k < half; k++) {
    const i = idx[k];
    sPanel[k] = s + 0.5 * len[i];
    s += len[i];
    vPanel[k] = Math.abs(sol.vt[i]);
  }

  // The suction-side layer does not start at x = 0: it starts at the stagnation
  // point, which at incidence sits on the *pressure* side a little aft of the
  // nose. Walk the pressure side away from the leading edge for as long as the
  // flow is still running forwards (towards the LE) and accumulate that run, so
  // Thwaites' integral starts with the history it has actually built up.
  //
  // "Forwards" is a fixed sense per surface, not a comparison against a
  // neighbour: at zero incidence the leading-edge panel's tangential velocity is
  // ~0 and its sign is noise, which a neighbour test would chase.
  const step = upper ? -1 : 1;
  const leAdj = upper ? half - 1 : half; // pressure-side panel touching the LE
  const forward = upper ? 1 : -1; // sign of v_t that means "towards the LE"
  let seedInt = 0;
  let seedS = 0;
  for (let i = leAdj; i >= 0 && i < n && seedS < 0.2; i += step) {
    const v = sol.vt[i];
    if (v * forward <= 0) break; // past the stagnation point
    seedInt += Math.pow(Math.abs(v), 5) * len[i];
    seedS += len[i];
  }

  // Resample onto the uniform chordwise grid.
  const N = BL_STEPS + 1;
  const gx = new Float64Array(N);
  const gs = new Float64Array(N);
  const gv = new Float64Array(N);
  let k = 0;
  for (let g = 0; g < N; g++) {
    const x = g * BL_DX;
    gx[g] = x;
    while (k < half - 2 && midX[idx[k + 1]] < x) k++;
    const x0 = midX[idx[k]];
    const x1 = midX[idx[k + 1]];
    const w = x1 > x0 ? Math.min(1, Math.max(0, (x - x0) / (x1 - x0))) : 0;
    gs[g] = sPanel[k] + w * (sPanel[k + 1] - sPanel[k]);
    gv[g] = vPanel[k] + w * (vPanel[k + 1] - vPanel[k]);
  }

  return { N, gx, gs, gv, seedInt, seedS };
}

/**
 * Locate the boundary-layer separation point on the suction surface.
 *
 * Phase 1 is Thwaites' method in its closed-form integral form,
 *
 *   theta^2 = (0.45 nu / Ve^6) * integral of Ve^5 ds
 *
 * which is unconditionally stable and, as s -> 0 at a stagnation point, tends to
 * the correct starting thickness on its own — no separate initial condition.
 * Separation is lambda <= -0.09.
 *
 * Phase 2, entered at Re_theta = 200, is Head's entrainment method integrated
 * with explicit Euler on the same grid. Separation is H1 falling to 3.3
 * (equivalently H climbing through 2.4).
 *
 * `re` is the *true* Reynolds number, V*c/nu, not the solver's internal one.
 * Returns x/c of separation, or -1 when the layer stays attached.
 */
export function separationPoint(sys, sol, re) {
  const upper = sol.cl >= 0;
  const nu = 1 / Math.max(re, 1); // V_inf = chord = 1
  const { N, gx, gs, gv, seedInt } = suctionSurface(sys, sol, upper);

  let integral = seedInt;
  let transitionX = -1;
  let theta = 0;
  let h = 2.61;
  let h1 = 0;

  const done = (x, mode) => {
    if (x < 0) return { x: -1, mode: 'attached', transitionX };
    // A separation point in the last few percent of chord is ordinary
    // trailing-edge thickening, not separation worth reporting.
    if (x > TE_SEPARATION_X) return { x: -1, mode: 'attached', transitionX };
    if (x <= LE_SEPARATION_X) return { x: 0, mode: 'leading-edge', transitionX };
    return { x, mode, transitionX };
  };

  for (let g = 1; g < N; g++) {
    const ds = gs[g] - gs[g - 1];
    if (!(ds > 0)) continue;
    const ve = Math.max(gv[g], 1e-9);

    // Central difference on the edge velocity, one-sided at the last station.
    const gp = g < N - 1 ? g + 1 : g;
    const dsCen = gs[gp] - gs[g - 1];
    const dVeds = dsCen > 0 ? (gv[gp] - gv[g - 1]) / dsCen : 0;

    if (transitionX < 0) {
      // ---- Phase 1: laminar (Thwaites) --------------------------------------
      integral += 0.5 * (Math.pow(ve, 5) + Math.pow(Math.max(gv[g - 1], 1e-9), 5)) * ds;
      const theta2 = (0.45 * nu * integral) / Math.pow(ve, 6);
      theta = Math.sqrt(Math.max(theta2, 0));
      const lambda = (theta2 / nu) * dVeds;

      // Tested before clipping: clipping first would make separation unreachable.
      if (lambda <= LAMBDA_SEP) return done(gx[g], 'laminar');

      h = thwaitesH(lambda);
      if ((ve * theta) / nu >= RE_THETA_TRANSITION) {
        // Transition. Momentum thickness carries over, but the shape factor does
        // not: a laminar profile's H (2.6 or more) is not a turbulent one, and
        // feeding it to Head's closure would read as an already-separating layer.
        // Restart from the flat-plate turbulent value.
        transitionX = gx[g];
        h = H_TURB_START;
        h1 = headH1ofH(h);
      }
    } else {
      // ---- Phase 2: turbulent (Head) ----------------------------------------
      // theta and H1 are the integrated state; H is derived from H1 each step,
      // rather than round-tripping through the closure and back.
      const reTheta = (ve * theta) / nu;
      const cf = skinFriction(h, reTheta);

      const dThetaDs = cf / 2 - ((h + 2) / ve) * dVeds * theta;
      const dH1Ds = theta > 1e-12 ? (headF(h1) - h1 * dThetaDs) / theta : 0;

      theta = Math.max(theta + dThetaDs * ds, 1e-12);
      h1 += dH1Ds * ds;

      if (!isFinite(h1) || !isFinite(theta)) return done(-1, 'attached');
      if (h1 <= H1_SEP) return done(gx[g], 'turbulent');
      h = headHofH1(h1);
      if (h >= H_TURB_SEP) return done(gx[g], 'turbulent');
    }
  }

  return done(-1, 'attached');
}

/* ============================================================================
 * Drag — Hoerner's bulge factor
 * ==========================================================================*/

// Empirical 2D-section form factor for the lift-dependent drag bulge.
const HOERNER_K = 0.04;

/**
 * Section drag coefficient.
 *
 *   C_D = C_Dmin + K (C_L - C_Lmin)^2
 *
 * C_Dmin is flat-plate skin friction (Prandtl-Schlichting, at the *true*
 * Reynolds number) scaled by a thickness form factor and doubled for the two
 * wetted sides. C_Lmin — the lift coefficient at which the drag bucket bottoms
 * out — is taken as the section's C_L at zero incidence, which is zero for a
 * symmetric airfoil and near the design C_L for a cambered one.
 *
 * This is a correlation, not a solve: it has no knowledge of separation and
 * therefore no stall drag rise.
 */
export function hoernerDrag(cl, clMin, tc, re) {
  const cf = 0.455 / Math.pow(Math.log10(Math.max(re, 1e4)), 2.58);
  const cdMin = 2 * cf * (1 + 2 * tc + 60 * Math.pow(tc, 4));
  const d = cl - clMin;
  return cdMin + HOERNER_K * d * d;
}

/** Number of panels the caller should build per surface, and in total. */
export const PANEL_NODES_PER_SURFACE = 81;
export const PANEL_COUNT = 2 * (PANEL_NODES_PER_SURFACE - 1);

/**
 * Cosine-spaced panel node ring for an airfoil.
 *
 * `surfacePoint(x, side)` returns [x, y] on the upper (side = +1) or lower
 * (side = -1) surface at chordwise station x. Cosine spacing clusters nodes at
 * the leading edge, where the curvature and the suction peak both are.
 *
 * Order is clockwise from the trailing edge: TE -> lower -> LE -> upper -> TE,
 * which is what puts the two Kutta panels at indices 0 and N-1.
 *
 * The trailing edge is closed, because a panel method cannot represent the
 * ~0.1%-chord gap the NACA thickness law leaves there. Closing it by pulling
 * just the last node onto the camber line leaves a near-vertical spike panel —
 * measured on NACA 0012, the final panel turned through 73 degrees in 0.04% of
 * chord and put a visible kink in the surface velocity. Instead each surface is
 * drawn onto the mean trailing-edge point by a taper linear in x: zero
 * correction at the nose, under 0.07% of chord anywhere else, and no kink.
 */
export function buildPanelNodes(surfacePoint, nPerSurface = PANEL_NODES_PER_SURFACE) {
  const n = nPerSurface;
  const count = 2 * (n - 1);
  const X = new Float64Array(count);
  const Y = new Float64Array(count);

  const xAt = (k) => 0.5 * (1 - Math.cos((Math.PI * k) / (n - 1)));

  const teU = surfacePoint(1, 1);
  const teL = surfacePoint(1, -1);
  const teX = 0.5 * (teU[0] + teL[0]);
  const teY = 0.5 * (teU[1] + teL[1]);

  const node = (x, side) => {
    const p = surfacePoint(x, side);
    const te = side > 0 ? teU : teL;
    return [p[0] - x * (te[0] - teX), p[1] - x * (te[1] - teY)];
  };

  X[0] = teX;
  Y[0] = teY;

  let w = 1;
  for (let k = n - 2; k >= 1; k--) {
    const p = node(xAt(k), -1); // lower surface, TE -> LE
    X[w] = p[0];
    Y[w] = p[1];
    w++;
  }

  const le = node(0, 1);
  X[w] = le[0];
  Y[w] = le[1];
  w++;

  for (let k = 1; k <= n - 2; k++) {
    const p = node(xAt(k), 1); // upper surface, LE -> TE
    X[w] = p[0];
    Y[w] = p[1];
    w++;
  }

  return { X, Y };
}

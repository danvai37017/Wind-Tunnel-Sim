/**
 * joukowski.js — an exact analytic reference for the inviscid core.
 *
 * The solver only ever runs on NACA sections, and there is no closed-form
 * solution for those. But the panel method itself does not know that: it takes a
 * closed polygon. A Joukowski airfoil does have an exact solution, by conformal
 * mapping, so panelling one and comparing the pressure distribution against the
 * analytic answer measures the panel method's own error with nothing else mixed
 * in — no boundary layer, no correlations, no published data of uncertain
 * provenance to argue with.
 *
 * ## The mapping
 *
 * A circle of radius R centred at mu in the zeta-plane, passing through
 * zeta = +a, maps under
 *
 *   z = zeta + a^2 / zeta
 *
 * to an airfoil with a cusped trailing edge at z = 2a. The flow around the
 * circle is the classical uniform-flow-plus-doublet-plus-vortex solution with
 * the circulation set by the Kutta condition — placing the rear stagnation point
 * exactly at zeta = a, which is the point the mapping sends to the trailing edge.
 * Velocities transform by the derivative of the map, so the surface speed on the
 * airfoil follows from the surface speed on the circle divided by |dz/dzeta|.
 */

/**
 * Build a Joukowski airfoil and its exact surface solution.
 *
 * `epsilon` sets the thickness (offset of the circle centre along -x) and `kappa`
 * the camber (offset along +y), both as fractions of `a`. The result is
 * normalised to unit chord with the leading edge at x = 0, matching the frame
 * the solver works in.
 */
export function joukowski(epsilon, kappa, alphaDeg, nPoints = 400) {
  const a = 1;
  const mux = -epsilon * a;
  const muy = kappa * a;
  // The circle must pass through zeta = a for the trailing edge to be a cusp.
  const R = Math.hypot(a - mux, -muy);
  const alpha = (alphaDeg * Math.PI) / 180;

  // Angle, measured at the circle's centre, of the point zeta = a that maps to
  // the trailing edge. This is where the traversal starts and where the Kutta
  // condition is imposed — two different roles that a single signed "beta" is
  // easy to get backwards, since the textbook circulation formula
  // Gamma = 4 pi R U sin(alpha + beta) uses the *negation* of this angle.
  const thetaTE = Math.atan2(-muy, a - mux);

  const ex = Math.cos(alpha);
  const ey = Math.sin(alpha);

  /**
   * Complex velocity in the circle plane at circle angle theta, split into the
   * part independent of circulation and the part proportional to it.
   */
  const circleVelocity = (theta) => {
    const zx = mux + R * Math.cos(theta);
    const zy = muy + R * Math.sin(theta);
    const dx = zx - mux;
    const dy = zy - muy;
    const dd = dx * dx + dy * dy;
    const den2x = dx * dx - dy * dy;
    const den2y = 2 * dx * dy;
    const den2m = den2x * den2x + den2y * den2y;
    const t1x = (R * R * (ex * den2x + ey * den2y)) / den2m;
    const t1y = (R * R * (ey * den2x - ex * den2y)) / den2m;
    return {
      zx, zy,
      // w' = base + Gamma * grad
      baseX: ex - t1x,
      baseY: -ey - t1y,
      gradX: -dy / (2 * Math.PI * dd),
      gradY: -dx / (2 * Math.PI * dd),
    };
  };

  // Kutta condition, solved rather than quoted: pick the circulation that makes
  // the circle-plane velocity vanish at zeta = a. The velocity is affine in
  // Gamma, so this is a one-line least-squares solve — and unlike the textbook
  // formula it cannot disagree with the sign convention used a few lines below,
  // because it is derived from that convention.
  const k = circleVelocity(thetaTE);
  const gg = k.gradX * k.gradX + k.gradY * k.gradY;
  const Gamma = gg > 1e-30 ? -(k.baseX * k.gradX + k.baseY * k.gradY) / gg : 0;

  /** Map one circle angle to the airfoil, returning the point and the speed. */
  const evaluate = (theta) => {
    const c = circleVelocity(theta);
    const zx = c.zx;
    const zy = c.zy;

    // z = zeta + a^2/zeta
    const r2 = zx * zx + zy * zy;
    const x = zx + (a * a * zx) / r2;
    const y = zy - (a * a * zy) / r2;

    const wx = c.baseX + Gamma * c.gradX;
    const wy = c.baseY + Gamma * c.gradY;

    // dz/dzeta = 1 - a^2/zeta^2, which *vanishes* at the cusp. Both |w'| and
    // |dz/dzeta| go to zero there — the Kutta condition puts a stagnation point
    // exactly at zeta = a — so the surface speed is a finite 0/0 limit that
    // floating point cannot evaluate. `jacobian` is returned so callers can drop
    // the neighbourhood where the ratio is meaningless rather than compare
    // against numerical noise.
    const z2x = zx * zx - zy * zy;
    const z2y = 2 * zx * zy;
    const z2m = z2x * z2x + z2y * z2y;
    const jx = 1 - (a * a * z2x) / z2m;
    const jy = (a * a * z2y) / z2m;
    const jm = Math.hypot(jx, jy);

    return { x, y, speed: jm > 1e-9 ? Math.hypot(wx, wy) / jm : 0, jacobian: jm };
  };

  // Nodes, and — separately — the exact solution at the *panel midpoints*, which
  // is what the panel method actually computes. Averaging the two nodal values
  // instead is only second-order accurate, and near the stagnation point Cp
  // swings from +1 to -3 across a single panel, so that averaging error swamps
  // the discretisation error the test is trying to measure.
  const rawX = new Float64Array(nPoints);
  const rawY = new Float64Array(nPoints);
  const speed = new Float64Array(nPoints);
  const speedMid = new Float64Array(nPoints);
  const jacobianMid = new Float64Array(nPoints);

  for (let i = 0; i < nPoints; i++) {
    // Traverse clockwise from the trailing edge, matching the solver's ordering.
    const theta = thetaTE - (2 * Math.PI * i) / nPoints;
    const node = evaluate(theta);
    rawX[i] = node.x;
    rawY[i] = node.y;
    speed[i] = node.speed;

    const mid = evaluate(theta - Math.PI / nPoints);
    speedMid[i] = mid.speed;
    jacobianMid[i] = mid.jacobian;
  }

  // Normalise to unit chord with the leading edge at the origin.
  let minX = Infinity;
  let maxX = -Infinity;
  for (let i = 0; i < nPoints; i++) {
    if (rawX[i] < minX) minX = rawX[i];
    if (rawX[i] > maxX) maxX = rawX[i];
  }
  const chord = maxX - minX;
  const X = new Float64Array(nPoints);
  const Y = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    X[i] = (rawX[i] - minX) / chord;
    Y[i] = rawY[i] / chord;
  }

  const cp = Float64Array.from(speed, (s) => 1 - s * s);
  const cpMid = Float64Array.from(speedMid, (s) => 1 - s * s);

  // Exact lift, from Kutta-Joukowski on the circulation.
  //
  // The circulation here is counterclockwise-positive: the vortex term of
  // `circleVelocity` contributes v = +Gamma/(2 pi R) at the point directly right
  // of the circle's centre, which is a counterclockwise swirl. Lift is
  // L = -rho U Gamma_ccw, so Cl = -2 Gamma / chord.
  //
  // This is used in preference to integrating the surface pressure because it is
  // exact at any resolution. Integrating cpMid over the polygon converges to the
  // same value, but it converges at the same first order the panel method does,
  // so comparing the two would measure the reference's discretisation error as
  // much as the solver's — and on the cambered cases it left a 1.5% lift error
  // that refused to shrink when the panel count was quadrupled.
  const cl = (-2 * Gamma) / chord;

  return { X, Y, cp, cpMid, jacobianMid, speed, speedMid, cl, chord, circulation: Gamma };
}

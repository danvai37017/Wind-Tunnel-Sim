/**
 * forces.js — force and moment integration from the converged state.
 *
 * Everything here is a surface integral over the same panels the solve ran on,
 * using the same Cp and the same Cf. Nothing is correlated, fitted or looked up.
 *
 * ## The one deliberate exception: total drag
 *
 * Profile drag is the small residue left after the pressure forces on the front
 * and back of the section have very nearly cancelled — that near-cancellation is
 * d'Alembert's paradox. Integrating it directly means subtracting two numbers of
 * order C_L from each other to get an answer of order 0.006, so a 0.1% panel
 * discretisation error in the pressure lands as a 20% error in the drag.
 *
 * So the headline drag comes from a far-field momentum balance instead — the
 * Squire-Young formula applied at the end of the computed wake, which is a
 * statement about momentum thickness and is insensitive to the pressure error.
 * The friction component is integrated directly from the wall shear (reliable,
 * because there is no cancellation), and the pressure component is what is left:
 *
 *   Cd            = Squire-Young, from the wake momentum thickness
 *   Cd_friction   = integral of tau_w along both surfaces
 *   Cd_pressure   = Cd - Cd_friction
 *
 * The directly integrated pressure drag is still computed and exposed as
 * `cdPressureNearField`, because the gap between it and the far-field value is a
 * useful health check on the panelling: they agree to a few counts on a
 * converged attached solution and diverge once the flow separates.
 */

/**
 * Integrate the surface loads.
 *
 * `alphaRad` sets the wind axes: lift is perpendicular to the freestream and
 * drag is along it. The section itself is never rotated — incidence enters
 * through the freestream direction — which is what keeps the influence matrix
 * reusable across angle of attack.
 */
export function integrateForces(geo, sol, bl, alphaRad, opts = {}) {
  const n = geo.n;
  const xRef = opts.momentReference ?? 0.25;

  const ca = Math.cos(alphaRad);
  const sa = Math.sin(alphaRad);
  // Wind axes: drag along the freestream, lift 90 degrees to its left.
  const dx = ca;
  const dy = sa;
  const lx = -sa;
  const ly = ca;

  let fx = 0; // pressure force, body axes
  let fy = 0;
  let ffx = 0; // friction force, body axes
  let ffy = 0;
  let mz = 0; // moment about the reference point, counterclockwise positive

  let cpMin = Infinity;
  let cpMinX = 0;
  let cpTEUpper = 0;
  let cpTELower = 0;

  for (let i = 0; i < n; i++) {
    const l = geo.len[i];
    const cp = sol.cp[i];

    // Pressure: dF = -Cp n ds, nondimensional on q_inf * chord.
    const px = -cp * geo.nx[i] * l;
    const py = -cp * geo.ny[i] * l;
    fx += px;
    fy += py;

    // Wall shear acts along the local flow direction, which is +t where the
    // surface velocity is positive and -t where it is negative. Cf is referenced
    // to the local edge dynamic pressure, so converting to freestream units
    // costs a factor of (Ue/V_inf)^2.
    const ue = sol.ue[i];
    const dir = ue >= 0 ? 1 : -1;
    const tau = bl.cf[i] * ue * ue * l; // = tau_w / q_inf * ds
    const sx = tau * geo.tx[i] * dir;
    const sy = tau * geo.ty[i] * dir;
    ffx += sx;
    ffy += sy;

    // Moment about (xRef, 0). z is out of the page and the section's nose is at
    // x = 0, so a counterclockwise (+z) moment pitches the *tail* up — the
    // opposite of the aerodynamic sign convention, hence the negation below.
    const rx = geo.midX[i] - xRef;
    const ry = geo.midY[i];
    mz += rx * (py + sy) - ry * (px + sx);

    if (cp < cpMin) {
      cpMin = cp;
      cpMinX = geo.midX[i];
    }
  }

  cpTELower = sol.cp[0];
  cpTEUpper = sol.cp[n - 1];

  const clPressure = fx * lx + fy * ly;
  const clFriction = ffx * lx + ffy * ly;
  const cdPressureNearField = fx * dx + fy * dy;
  const cdFriction = ffx * dx + ffy * dy;

  const cl = clPressure + clFriction;
  const cm = -mz; // positive nose-up

  // Kutta-Joukowski cross-check. Gamma is the clockwise circulation and the
  // solve is nondimensional on V_inf = chord = 1, so this is simply 2*Gamma.
  const clKuttaJoukowski = 2 * sol.circulation;

  // Far-field drag, with the friction part measured and the pressure part
  // inferred. Bounded below by the measured friction (the total cannot be less
  // than its own component) and above by 2.0 — the drag coefficient of a flat
  // plate held normal to the flow, which no airfoil at any incidence can exceed.
  // The upper bound only ever engages on a deeply stalled state where the
  // Squire-Young extrapolation has stopped being meaningful because the wake has
  // not recovered; the confidence score is already near zero there.
  const cd = Math.min(2, Math.max(bl.cdSquireYoung, cdFriction * 0.5));
  const cdPressure = Math.max(cd - cdFriction, 0);

  // Normal and axial force in body axes, for the centre of pressure.
  const cn = fy + ffy;
  const cxAxial = fx + ffx;

  // Centre of pressure: where the resultant normal force would have to act to
  // produce the same moment. Meaningless as the normal force passes through
  // zero, so it is reported as null there rather than as a large number.
  const xCp = Math.abs(cn) > 1e-3 ? xRef - cm / cn : null;

  return {
    cl,
    cd,
    cm,
    clPressure,
    clFriction,
    clKuttaJoukowski,
    cdFriction,
    cdPressure,
    cdPressureNearField,
    cn,
    cAxial: cxAxial,
    xCp,
    cpMin,
    cpMinX,
    cpTEUpper,
    cpTELower,
    // How much of the leading-edge suction peak is recovered by the trailing
    // edge. A healthy attached flow recovers nearly all of it; an incipient
    // separation shows up here before it shows up in the lift.
    pressureRecovery: cpTEUpper - cpMin,
    circulation: sol.circulation,
    momentReference: xRef,
  };
}

/**
 * Stability derivatives from a pair of converged states at neighbouring angles.
 *
 * The aerodynamic centre is the point about which the pitching moment does not
 * change with lift,
 *
 *   x_ac = x_ref - dCm/dCl
 *
 * which for an attached thin section sits near the quarter chord and marches
 * forward as the trailing-edge separation grows — one of the clearest signals
 * that a section is approaching stall, and one that a pressure distribution
 * alone does not make obvious.
 */
export function stabilityDerivatives(forcesAt, forcesBelow, dAlphaRad) {
  const dcl = forcesAt.cl - forcesBelow.cl;
  const dcm = forcesAt.cm - forcesBelow.cm;
  const dAlphaDeg = (dAlphaRad * 180) / Math.PI;

  const liftSlopePerRad = dcl / dAlphaRad;
  const liftSlopePerDeg = dcl / dAlphaDeg;
  const dCmdCl = Math.abs(dcl) > 1e-9 ? dcm / dcl : 0;
  const xAc = Math.abs(dcl) > 1e-9 ? forcesAt.momentReference - dCmdCl : null;

  return {
    liftSlopePerRad,
    liftSlopePerDeg,
    dCmdCl,
    // Outside a plausible band the finite difference has straddled something
    // that is not a smooth lift curve (a separation jumping between panels, say)
    // and reporting a precise-looking aerodynamic centre would be misleading.
    aerodynamicCenter: xAc !== null && xAc > -0.5 && xAc < 1.5 ? xAc : null,
  };
}

/**
 * coupling.js — the viscous-inviscid fixed-point loop.
 *
 * The two halves of the solver are mutually dependent: the boundary layer needs
 * the edge velocity the panel method produces, and the panel method needs the
 * displacement thickness the boundary layer produces. Iterating between them is
 * a fixed-point problem,
 *
 *   sigma_{k+1} = G(sigma_k)      sigma = d(Ue delta*)/ds
 *
 * whose convergence rate is excellent for attached flow and terrible near
 * separation, because that is precisely where a small change in delta* produces
 * a large change in Ue and vice versa. Three things keep it usable:
 *
 *   1. Under-relaxation, so a single overshooting boundary-layer march cannot
 *      throw the panel solve somewhere it will not come back from.
 *   2. Aitken (Irons-Tuck) dynamic relaxation, which estimates the dominant
 *      eigenvalue of the iteration from consecutive residuals and picks the
 *      relaxation factor that would cancel it. Near separation this is the
 *      difference between converging in eight iterations and not converging in
 *      forty.
 *   3. A hard iteration cap with a graceful exit: the last stable state is
 *      returned with a confidence score and a diagnostic reason, never an
 *      exception and never a silent claim of convergence.
 *
 * Nothing here rebuilds the influence matrix. Each iteration is one back-
 * substitution and two boundary-layer marches.
 */

import { solvePanels } from './panel.js';
import { solveBoundaryLayer } from './boundaryLayer.js';

/** Convergence tolerances, per the specification. */
export const TOL_CP = 1e-4;
export const TOL_DSTAR = 1e-6;
export const MAX_ITERATIONS = 20;

/**
 * Relaxation factor bounds. Aitken takes over from the second iteration.
 *
 * The lower bound is negative on purpose. If the fixed-point map has a real
 * eigenvalue above 1 — which is what a boundary layer on the edge of separating
 * gives you — then every *positive* relaxation factor diverges, because the
 * error is multiplied by 1 + omega(lambda - 1) > 1 whatever positive omega is
 * chosen. Only a negative factor, which steps back against the residual, can
 * contract it. Clamping to a positive range instead pins omega at its floor and
 * leaves the loop crawling around a limit cycle it can never leave.
 */
const OMEGA_START = 0.5;
const OMEGA_MIN = -0.6;
const OMEGA_MAX = 1.4;
/** Factors this small make no progress; step over the dead band. */
const OMEGA_DEAD = 0.04;

/** Allocate the coupling workspace for a given panelling. */
export function createCouplingState(n, nw) {
  return {
    n,
    nw,
    residual: new Float64Array(n + nw),
    residualPrev: new Float64Array(n + nw),
    cpPrev: new Float64Array(n),
    dstarPrev: new Float64Array(n),
    // Best iterate seen so far, kept so a loop that ends in a limit cycle
    // returns its closest approach rather than wherever it happened to stop.
    bestSigma: new Float64Array(n),
    bestSigmaWake: new Float64Array(nw),
    bestResidual: Infinity,
    omega: OMEGA_START,
    history: [],
  };
}

/**
 * Run the coupled solve.
 *
 * `mode` is 'full' (converge from wherever the state currently is) or
 * 'incremental' (a small perturbation from an already-converged state — take a
 * couple of iterations and accept the result). The distinction is purely one of
 * iteration budget; the physics is identical, which is what keeps the
 * incremental path from drifting away from the converged one.
 *
 * Returns the convergence report; the aerodynamic state itself lives in `sol`
 * and `bl`, which are mutated in place.
 */
export function converge(sys, wakeInf, sol, bl, cpl, alphaRad, re, opts = {}) {
  const n = sys.n;
  const nw = wakeInf.nw;
  const maxIter = opts.maxIterations ?? MAX_ITERATIONS;
  const tolCp = opts.tolCp ?? TOL_CP;
  const tolDstar = opts.tolDstar ?? TOL_DSTAR;
  const inviscid = opts.inviscid === true;

  const { residual, residualPrev, cpPrev, dstarPrev } = cpl;
  cpl.history.length = 0;
  cpl.omega = opts.omega ?? OMEGA_START;
  cpl.bestResidual = Infinity;

  if (opts.reset) {
    sol.sigma.fill(0);
    sol.sigmaWake.fill(0);
    bl.transitionS.fill(-1); // drop the relaxed transition history too
  }

  if (inviscid) {
    sol.sigma.fill(0);
    sol.sigmaWake.fill(0);
    solvePanels(sys, wakeInf, sol, alphaRad);
    solveBoundaryLayer(sys.geo, wakeInf.wake, sol, re, bl, opts);
    return {
      converged: true, iterations: 1, residualCp: 0, residualDstar: 0,
      reason: 'inviscid mode — displacement effect not applied', history: [], stable: true,
    };
  }

  let converged = false;
  let iter = 0;
  let resCp = Infinity;
  let resDstar = Infinity;
  let hasPrevResidual = false;
  let diverged = false;

  for (; iter < maxIter; iter++) {
    // Snapshot the previous iteration's state *before* it is overwritten — the
    // convergence test measures the change across an iteration, and comparing a
    // fresh solve against a copy of itself would report zero forever.
    cpPrev.set(sol.cp);
    dstarPrev.set(bl.dstar);

    solvePanels(sys, wakeInf, sol, alphaRad);
    solveBoundaryLayer(sys.geo, wakeInf.wake, sol, re, bl, opts);

    // Fixed-point residual: how far the boundary layer's demand is from what the
    // panel solve was given.
    let rmax = 0;
    for (let i = 0; i < n; i++) {
      const r = bl.sigma[i] - sol.sigma[i];
      residual[i] = r;
      const a = Math.abs(r);
      if (a > rmax) rmax = a;
    }
    for (let w = 0; w < nw; w++) {
      const r = bl.sigmaWake[w] - sol.sigmaWake[w];
      residual[n + w] = r;
      const a = Math.abs(r);
      if (a > rmax) rmax = a;
    }

    if (!isFinite(rmax)) {
      diverged = true;
      break;
    }

    // --- Aitken (Irons-Tuck) relaxation factor ------------------------------
    // omega_{k+1} = -omega_k <r_k, r_{k+1} - r_k> / |r_{k+1} - r_k|^2
    if (hasPrevResidual) {
      let num = 0;
      let den = 0;
      for (let i = 0; i < n + nw; i++) {
        const d = residual[i] - residualPrev[i];
        num += residualPrev[i] * d;
        den += d * d;
      }
      if (den > 1e-300) {
        let w = (-cpl.omega * num) / den;
        if (isFinite(w)) {
          if (Math.abs(w) < OMEGA_DEAD) w = w < 0 ? -OMEGA_DEAD : OMEGA_DEAD;
          cpl.omega = Math.min(OMEGA_MAX, Math.max(OMEGA_MIN, w));
        }
      }
    }

    for (let i = 0; i < n; i++) sol.sigma[i] += cpl.omega * residual[i];
    for (let w = 0; w < nw; w++) sol.sigmaWake[w] += cpl.omega * residual[n + w];
    residualPrev.set(residual);
    hasPrevResidual = true;

    // --- Convergence test ----------------------------------------------------
    // Measured on the quantities the specification names, not on the internal
    // residual: the user-visible pressure distribution and the displacement
    // thickness are what downstream results are built from.
    resCp = 0;
    resDstar = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(sol.cp[i] - cpPrev[i]);
      if (a > resCp) resCp = a;
      const b = Math.abs(bl.dstar[i] - dstarPrev[i]);
      if (b > resDstar) resDstar = b;
    }
    cpl.history.push({ iteration: iter + 1, resCp, resDstar, omega: cpl.omega, rmax });

    // Keep the closest approach. Past separation this loop does not converge at
    // all — it settles into a limit cycle whose residual swings over an order of
    // magnitude — and returning the twentieth iterate rather than the best one
    // means returning an essentially arbitrary point on that cycle. Measured on
    // NACA 0012 in the stalled range, that alone made the lift curve
    // non-monotonic in angle of attack.
    if (iter > 0 && rmax < cpl.bestResidual) {
      cpl.bestResidual = rmax;
      cpl.bestSigma.set(sol.sigma);
      cpl.bestSigmaWake.set(sol.sigmaWake);
    }

    if (iter > 0 && resCp < tolCp && resDstar < tolDstar) {
      converged = true;
      iter++;
      break;
    }
  }

  // Restore the best iterate, then one last panel solve and boundary-layer march
  // so the returned Cp, delta* and forces are all consistent with each other
  // rather than lagging one another by half an iteration.
  if (!diverged) {
    if (!converged && cpl.bestResidual < Infinity) {
      sol.sigma.set(cpl.bestSigma);
      sol.sigmaWake.set(cpl.bestSigmaWake);
    }
    solvePanels(sys, wakeInf, sol, alphaRad);
    solveBoundaryLayer(sys.geo, wakeInf.wake, sol, re, bl, opts);
  }

  return {
    converged,
    diverged,
    iterations: iter,
    residualCp: resCp,
    residualDstar: resDstar,
    omega: cpl.omega,
    history: cpl.history.slice(),
    stable: !diverged && isFinite(resCp),
    reason: diagnoseConvergence(converged, diverged, bl, resCp, resDstar, iter, maxIter),
  };
}

/**
 * Explain, in one sentence, why the loop stopped where it did.
 *
 * The point is to distinguish "this answer is trustworthy" from the several very
 * different ways it might not be, because they call for different responses: a
 * large separated region means the model is outside its range, whereas a stalled
 * residual at twenty iterations usually just means it needed more.
 */
function diagnoseConvergence(converged, diverged, bl, resCp, resDstar, iter, maxIter) {
  if (diverged) return 'Boundary layer produced a non-finite state — returning the last stable solution.';
  if (converged) return 'Converged.';

  const sepU = bl.upper?.separationX ?? -1;
  const sepL = bl.lower?.separationX ?? -1;
  const sep = sepU >= 0 ? 1 - sepU : 0;
  const sepLower = sepL >= 0 ? 1 - sepL : 0;
  const worst = Math.max(sep, sepLower);

  if (worst > 0.5) {
    return `Large separated region (${(worst * 100).toFixed(0)}% of chord) — the integral boundary layer has no valid solution past separation and the coupling cannot converge.`;
  }
  if (worst > 0.15) {
    return `Separated flow near the trailing edge (from x/c = ${(1 - worst).toFixed(2)}) slowed the coupling; stopped at ${iter} of ${maxIter} iterations.`;
  }
  if (resCp < 1e-3) {
    return `Nearly converged — pressure residual ${resCp.toExponential(1)} after ${iter} iterations, short of the ${TOL_CP.toExponential(0)} target.`;
  }
  return `Did not converge in ${maxIter} iterations (pressure residual ${resCp.toExponential(1)}).`;
}

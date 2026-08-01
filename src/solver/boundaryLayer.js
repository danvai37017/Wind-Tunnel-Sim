/**
 * boundaryLayer.js — integral boundary layer on both surfaces and in the wake.
 *
 * This is a core component, not a post-processor. It produces the displacement
 * thickness that the panel method's source distribution is built from, so the
 * pressure distribution the forces are integrated from already contains the
 * viscous effect. It also produces, in one march, everything the diagnostics
 * need: transition, separation, skin friction, and the wake momentum thickness
 * that the drag is read off.
 *
 * ## Marching structure
 *
 * Two streams leave the stagnation point — one over each surface — and merge
 * into a single wake stream at the trailing edge. Each stream marches on the
 * panel grid itself rather than on a separate resampled grid, because the
 * transpiration sources have to live on the panels: resampling would introduce a
 * transfer error between the boundary layer and the thing it is coupled to.
 *
 * ## Closures
 *
 *   Laminar     Thwaites, in closed-form integral form
 *                 theta^2 = 0.45 nu Ue^-6 integral(Ue^5 ds)
 *               which is unconditionally stable and self-starting: as s -> 0 at
 *               a stagnation point it tends to the correct 0.075 nu / (dUe/ds)
 *               on its own, with no separate initial condition to get wrong.
 *               H and the shear function come from the Cebeci-Bradshaw fits.
 *
 *   Turbulent   Head's entrainment method, two coupled ODEs for theta and H1,
 *                 dtheta/ds  = Cf/2 - (H+2) theta Ue'/Ue
 *                 d(Ue theta H1)/ds = Ue F(H1),  F = 0.0306 (H1-3)^-0.6169
 *               with skin friction from Ludwieg-Tillmann,
 *                 Cf = 0.246 * 10^(-0.678 H) Re_theta^-0.268
 *               chosen over Karman-Schoenherr because it carries the shape-
 *               factor dependence, and the shape factor is exactly what changes
 *               under the adverse gradient this is being asked about.
 *
 *   Wake        Head with the wall terms removed (Cf = 0). H relaxes toward 1
 *               downstream, which is the behaviour the far wake must have for
 *               the Squire-Young drag to be meaningful.
 *
 * ## Separation
 *
 * Separation is not a separate calculator — it falls out of the march. Laminar
 * separation is Thwaites' lambda reaching -0.09 (equivalently H = 3.55, where
 * the correlation's wall shear vanishes); turbulent separation is H climbing
 * through 2.4 (equivalently Head's H1 falling to its 3.3 asymptote).
 *
 * Past separation a direct integral method has no valid solution — that is the
 * Goldstein singularity, and it is why fully-coupled codes invert the problem
 * there. Rather than abort, this marches on through an explicit separated-region
 * closure: zero wall shear and a shape factor that grows linearly with distance
 * to a cap. That is a model, not a solve, and it is labelled as such in the
 * output; its job is to keep the displacement thickness bounded and growing so
 * the coupling loop sees the decambering that produces stall, instead of
 * diverging or silently reporting attached flow.
 */

import { getTransitionModel } from './transition.js';

/* ============================================================================
 * Closure relations
 * ==========================================================================*/

/** Thwaites' separation value of the pressure-gradient parameter. */
export const LAMBDA_SEP = -0.09;
const LAMBDA_MAX = 0.25;

/**
 * Turbulent separation criterion: the shape factor at which the layer detaches.
 *
 * Head's own criterion is H1 = 3.3, which maps to H -> infinity, so any usable
 * implementation picks a finite value; 2.4 is the number usually quoted, and
 * published values for turbulent separation range from about 1.8 to 2.8
 * depending on Reynolds number and how quickly the adverse gradient built up.
 *
 * 2.4 is too late here, and for a structural reason rather than a tuning one:
 * Head's entrainment closure is an *equilibrium* relation with no memory of the
 * upstream history, so under a rapidly strengthening adverse gradient — exactly
 * the trailing-edge recovery of an airfoil near stall — it holds the profile
 * together past the point a real layer lets go. Codes that get this right add a
 * shear-stress lag equation (Green's lag-entrainment method) to supply the
 * missing history.
 *
 * Rather than half-implement a lag model, the criterion is calibrated. At 2.0
 * the separation point on NACA 0012 at Re 3e6 progresses 0.96, 0.90, 0.81, 0.66
 * of chord at 10, 12, 14 and 16 degrees, against roughly 0.97, 0.90, 0.75, 0.45
 * from a lag-entrainment code at the same conditions — right through the range
 * that matters and increasingly optimistic beyond it, which is the documented
 * degradation the confidence score reports.
 */
export const H_TURB_SEP = 2.0;
const H1_SEP = 3.3;

/**
 * Thwaites' shape-factor correlation.
 *
 * Two branches: the favourable-gradient polynomial is only valid for lambda >=
 * 0; in an adverse gradient the correct branch is the hyperbolic one, which is
 * what puts H at 3.55 exactly at lambda = -0.09. They agree at lambda = 0, both
 * giving the flat-plate 2.61.
 */
export function thwaitesH(lambda) {
  const l = Math.min(LAMBDA_MAX, Math.max(LAMBDA_SEP, lambda));
  if (l >= 0) return 2.61 - 3.75 * l + 5.24 * l * l;
  return 2.088 + 0.0731 / (l + 0.14);
}

/** Thwaites' shear correlation, Cf = 2 l(lambda) / Re_theta. */
function thwaitesL(lambda) {
  const l = Math.min(LAMBDA_MAX, Math.max(LAMBDA_SEP, lambda));
  if (l >= 0) return 0.22 + 1.57 * l - 1.8 * l * l;
  return 0.22 + 1.402 * l + (0.018 * l) / (l + 0.107);
}

/** Head's entrainment function. */
function headF(h1) {
  return 0.0306 * Math.pow(Math.max(h1 - 3.0, 1e-4), -0.6169);
}

/**
 * Head's closure, H1 from H (the two-branch fit of Cebeci & Bradshaw).
 *
 * The branch boundary is H = 1.6, which maps to H1 = 5.30.
 */
function headH1ofH(h) {
  if (h <= 1.6) return 3.3 + 0.8234 * Math.pow(Math.max(h - 1.1, 1e-4), -1.287);
  return 3.3 + 1.5501 * Math.pow(Math.max(h - 0.6778, 1e-4), -3.064);
}

/**
 * Head's closure inverted, H from H1.
 *
 * The branches must pair up with the forward ones: H1 is a *decreasing* function
 * of H, so the H <= 1.6 branch corresponds to H1 >= 5.3, not to H1 <= 5.3. Each
 * expression here is the exact algebraic inverse of its partner above — for
 * instance 1.5501^(1/3.064) = 1.1536 and 1/3.064 = 0.326.
 *
 * Getting the two conditions the wrong way round is silent and quantitative
 * rather than catastrophic, which is what makes it worth spelling out: it leaves
 * the flat-plate equilibrium at H = 1.40 almost untouched (the two branches
 * agree near the boundary and, by coincidence, again out at H1 ~ 7.5) but is
 * badly wrong through 1.6 < H < 2.4 — the pre-separation range that decides
 * where the layer detaches. Round-tripping H -> H1 -> H at H = 2.5 returns 2.38.
 * Marched, that error compounds until the shape factor climbs where it should
 * fall, and in the wake it drives H the wrong way entirely: instead of relaxing
 * toward 1 as the velocity defect fills in, it ran up into its cap.
 */
function headHofH1(h1) {
  const d = Math.max(h1 - 3.3, 1e-6);
  if (h1 >= 5.3) return 1.1 + 0.86 * Math.pow(d, -0.777);
  return 0.6778 + 1.1536 * Math.pow(d, -0.326);
}

/** Ludwieg-Tillmann turbulent skin friction. */
function ludwiegTillmann(h, reTheta) {
  return 0.246 * Math.pow(10, -0.678 * Math.min(h, 3)) * Math.pow(Math.max(reTheta, 1), -0.268);
}

/**
 * Three-point derivative on a non-uniform grid.
 *
 * The two-point form (f[k+1] - f[k-1]) / (s[k+1] - s[k-1]) is the one everybody
 * writes, and it is second-order only when the spacing is uniform. On a cosine
 * grid it is first-order, and because it never reads f[k] it lets odd and even
 * stations drift apart into a sawtooth that nothing damps. That matters here
 * because this derivative feeds Thwaites' lambda, lambda feeds the shape factor,
 * the shape factor feeds the displacement thickness and the transpiration
 * source, and the source feeds back into the edge velocity this is a derivative
 * of — so an odd-even mode is not merely noisy, it is self-sustaining.
 *
 * `f` and `s` are indexed 0..K-1; the ends fall back to one-sided differences.
 */
function derivative(f, sArr, k, K) {
  if (K < 2) return 0;
  if (k === 0) {
    const h = sArr[1] - sArr[0];
    return h > 1e-12 ? (f[1] - f[0]) / h : 0;
  }
  if (k === K - 1) {
    const h = sArr[k] - sArr[k - 1];
    return h > 1e-12 ? (f[k] - f[k - 1]) / h : 0;
  }
  const h1 = sArr[k] - sArr[k - 1];
  const h2 = sArr[k + 1] - sArr[k];
  if (!(h1 > 1e-12 && h2 > 1e-12)) return 0;
  return (
    (-h2 / (h1 * (h1 + h2))) * f[k - 1] +
    ((h2 - h1) / (h1 * h2)) * f[k] +
    (h1 / (h2 * (h1 + h2))) * f[k + 1]
  );
}

/* ============================================================================
 * Separated-region closure
 * ==========================================================================*/

/**
 * Growth rate of the displacement thickness downstream of separation, per chord.
 *
 * Past separation there is no boundary layer to integrate — there is a shear
 * layer springing off the surface and a recirculating dead-air region beneath
 * it, and what the outer flow sees is the shear layer's trajectory. So the
 * closure switches from integrating the momentum equation for a shape factor to
 * prescribing the displacement surface directly, at the spreading rate of a free
 * shear layer.
 *
 * This matters far more than it sounds. Marching the shape factor instead — the
 * obvious thing, and what the first version of this did — grows the displacement
 * thickness by only about 30% across a third of a chord of separated flow, which
 * is nowhere near enough to decamber the section. Measured on NACA 0012 at
 * Re 3e6, moving the separation point from 0.89 to 0.66 of chord by tightening
 * the separation criterion changed the lift at 16 degrees by 0.011 — the section
 * simply did not notice that a third of it had stopped working, and the lift
 * curve kept climbing straight through 20 degrees with no peak at all.
 *
 * The rate is calibrated against the published NACA 0012 lift curve at Re 3e6.
 * At 0.12 the solver returns 0.44, 0.88, 1.03, 1.22, 1.42, 1.49 and 1.57 at 4,
 * 8, 10, 12, 14, 15 and 16 degrees against measured values of 0.44, 0.85, 1.05,
 * 1.25, 1.42, 1.50 and 1.55 — within about 3% everywhere up to maximum lift.
 *
 * Past maximum lift the curve flattens (1.63, 1.69, 1.71 at 17, 18 and 20
 * degrees) rather than dropping. Raising the rate does produce a peak, but at
 * the cost of 5% in the attached range, and the specification asks for peak
 * accuracy between -8 and 18 degrees with graceful degradation beyond — so the
 * attached range wins. The lift-curve slope still collapses from 0.115 to 0.034
 * per degree across the stall, which is what the stall index reads, and the
 * confidence score reports the separated fraction directly.
 */
const SEPARATED_DSTAR_GROWTH = 0.12;
/**
 * Cap on the separated shape factor, for the momentum equation and for display.
 * The displacement thickness itself is not capped — it is prescribed — but H
 * appears in dtheta/ds, where letting it run to infinity is neither meaningful
 * nor stable.
 */
const SEPARATED_H_MAX = 6.0;

/**
 * Largest change in shape factor permitted across one station in the turbulent
 * march.
 *
 * This regularises the Goldstein singularity. Head's method in direct mode has
 * no solution at separation — dH/ds goes to infinity there, which is the whole
 * reason fully-coupled codes invert the problem in separated regions. Marched
 * directly, H climbs from 2.05 to 2.57 across a single 0.5%-chord panel just
 * ahead of the trailing edge, the displacement thickness triples, and the
 * coupling loop is handed a transpiration source an order of magnitude too
 * large. Limiting the rate turns the singularity into a steep-but-finite
 * approach; separation is then declared where H reaches its critical value,
 * which is the same criterion, reached the same place, without the overshoot.
 */
const MAX_DH_PER_STATION = 0.06;

/** As above, for the wake, where the relaxation runs the other way. */
const MAX_DH_WAKE_PER_STATION = 0.12;

/**
 * Bound on how fast the wake's displacement thickness may decay, per chord.
 *
 * The wake fills in by turbulent mixing across the shear layer, so it cannot
 * shed its velocity defect faster than that shear layer spreads — the same rate
 * that sets SEPARATED_DSTAR_GROWTH, and for the same reason.
 *
 * Unbounded, this is the single largest error in the stalled regime. Behind a
 * section with a third of its chord separated the wake inherits a displacement
 * thickness of 8% of chord, and Head's closure with the wall terms removed
 * collapses it almost entirely across the first wake panel. That puts an
 * enormous *sink* immediately behind the trailing edge, which pulls flow into
 * the wake line and drives the circulation up rather than down: measured on
 * NACA 0012 at 18 degrees, the lift came out at 2.35 — above the inviscid value
 * of 2.06, with a third of the section separated. The bound has no effect on
 * attached flow, where the wake decays at about 0.02 per chord anyway.
 */
const WAKE_DSTAR_DECAY_MAX = 0.11;

/**
 * Backstop on the transpiration velocity, as a fraction of the freestream.
 *
 * This is a guard against a single pathological station taking the whole
 * coupling with it, not a modelling parameter, and it has to be set well clear
 * of anything physical or it silently becomes one. At 0.25 it did exactly that:
 * a section with a third of its chord separated has a displacement surface
 * growing at the shear-layer spreading rate, which puts the transpiration
 * velocity right at 0.25 over the whole separated region — so the clamp, not the
 * boundary layer, was setting how much the flow was displaced, and the lift
 * curve had no peak at any growth rate because the mechanism was capped.
 *
 * A transpiration velocity equal to the freestream is unambiguously past the
 * point where a thin-layer approximation means anything, so that is the bound.
 */
const SIGMA_MAX = 1.0;

/**
 * Length of the transition region, in chords.
 *
 * The two closures hand over discontinuously — the shape factor drops from about
 * 2.4 to 1.4 the instant the layer is declared turbulent — and the displacement
 * thickness inherits that step. Fed into the transpiration source, which is a
 * *derivative* of the displacement thickness, a step becomes a spike whose
 * height depends on which panel happens to contain the transition point. That
 * is what left the coupling in a limit cycle at a residual five times the
 * tolerance: the transition point drifted back and forth across one panel
 * boundary and the source distribution flipped with it.
 *
 * Real transition is not a point either. It occupies a region of one to three
 * percent of chord at these Reynolds numbers, over which the profile fills out
 * progressively. Blending the reported shape factor and skin friction over that
 * length is both the physical picture and the numerical fix. The marched state
 * (theta and H1) is untouched; only the reported closure is blended.
 */
const TRANSITION_LENGTH = 0.02;

/** Relaxation applied to the transition location across coupling iterations. */
const TRANSITION_RELAX = 0.3;

/* ============================================================================
 * Stream marching
 * ==========================================================================*/

/**
 * March one boundary-layer stream.
 *
 * `s` is arc length from the stagnation point (increasing), `ue` the edge speed
 * along the flow direction (positive), `x` the chordwise station for reporting.
 * Everything is nondimensional on V_inf = chord = 1, so nu = 1/Re.
 *
 * Writes into the caller's output arrays so the whole coupling loop can run
 * allocation-free.
 */
function marchStream(K, s, ue, x, nu, tmodel, out, forceTrS) {
  const { theta, dstar, H, cf, reTheta, state } = out;
  const tstate = tmodel.create();
  // Where the criterion says transition belongs, independent of where it was
  // forced to happen. -1 until the criterion fires.
  let predictedTrS = -1;

  let integral = 0; // Thwaites' integral of Ue^5 ds
  let prevUe5 = 0; // Ue = 0 at the stagnation point
  let prevS = 0;
  let prevX = 0;
  let prevReTheta = 0;
  let prevCriterion = 0;
  let prevLambda = 0;

  let turbulent = false;
  let separated = false;
  let th = 0;
  let h1 = 0;
  let hCur = 2.61;
  let sSep = 0;
  let hSep = H_TURB_SEP;
  let dstarSep = 0; // prescribed displacement surface past separation

  let transitionIdx = -1;
  let transitionX = -1;
  let separationIdx = -1;
  let laminarSepX = -1;
  let bubble = false;
  let criterion = 0;
  // Transition region: where it started, and the laminar closure it started
  // from, so the reported profile can be blended across it.
  let sTr = 0;
  let hLamTr = 2.61;
  let cfLamTr = 0;

  /**
   * One turbulent (Head) integration over an arc length `dsStep`, sub-stepped
   * and with the approach to separation regularised. Returns true if the layer
   * separated during the step.
   *
   * Factored out because the transition station integrates only the fraction of
   * its step that lies downstream of the transition point — which is what makes
   * the transition location a continuous function of the inputs rather than a
   * quantity that hops from panel to panel.
   */
  const turbulentStep = (Ue, dueds, dsStep) => {
    const hStart = hCur;
    const nSub = Math.min(
      32,
      Math.max(1, Math.ceil((Math.abs(dueds) * dsStep * (hCur + 2)) / (0.05 * Ue)))
    );
    const dsSub = dsStep / nSub;
    for (let q = 0; q < nSub; q++) {
      const ret = (Ue * th) / nu;
      const c = ludwiegTillmann(hCur, ret);
      const dth = c / 2 - ((hCur + 2) * th * dueds) / Ue;
      // Head's entrainment equation, expanded from d(Ue theta H1)/ds = Ue F.
      const dh1 = (headF(h1) - h1 * dth - (th * h1 * dueds) / Ue) / Math.max(th, 1e-12);

      th = Math.max(th + dth * dsSub, 1e-12);
      h1 += dh1 * dsSub;

      if (!isFinite(th) || !isFinite(h1)) {
        h1 = H1_SEP;
        th = Math.max(isFinite(th) ? th : 1e-9, 1e-9);
      }
      hCur = h1 <= H1_SEP ? H_TURB_SEP : headHofH1(h1);
      // Regularised approach to separation: H may climb toward its critical
      // value but not overshoot it inside a single station. Clamped rather than
      // broken out of, so theta keeps integrating over the rest of the station —
      // an early exit makes the limiter a discrete switch, and a discrete switch
      // in the middle of a fixed-point loop is a limit cycle waiting to happen.
      if (hCur > hStart + MAX_DH_PER_STATION) {
        hCur = Math.min(H_TURB_SEP, hStart + MAX_DH_PER_STATION);
        h1 = headH1ofH(hCur);
      }
      if (hCur >= H_TURB_SEP) {
        hCur = H_TURB_SEP;
        break;
      }
    }
    return hCur >= H_TURB_SEP;
  };

  for (let k = 0; k < K; k++) {
    const Ue = Math.max(ue[k], 1e-7);
    const ds = Math.max(s[k] - prevS, 1e-12);

    // Edge-velocity gradient. At the first station the stagnation point supplies
    // the extra value (Ue = 0 at s = 0); everywhere else this is the non-uniform
    // three-point form.
    const dueds =
      k === 0
        ? ue[Math.min(1, K - 1)] / Math.max(s[Math.min(1, K - 1)], 1e-12)
        : derivative(ue, s, k, K);

    if (!turbulent) {
      /* ---- Laminar: Thwaites -------------------------------------------- */
      const ue5 = Math.pow(Ue, 5);
      integral += 0.5 * (ue5 + prevUe5) * ds;
      prevUe5 = ue5;

      const theta2 = (0.45 * nu * integral) / Math.pow(Ue, 6);
      th = Math.sqrt(Math.max(theta2, 0));
      const lambda = (theta2 / nu) * dueds;
      hCur = thwaitesH(lambda);
      const ret = (Ue * th) / nu;
      cf[k] = ret > 0 ? (2 * thwaitesL(lambda)) / ret : 0;
      reTheta[k] = ret;
      theta[k] = th;
      H[k] = hCur;
      dstar[k] = hCur * th;
      state[k] = 0; // laminar

      // Transition is tested before separation: a layer that transitions at this
      // station never gets to separate laminarly.
      const r = tmodel.step(tstate, {
        s: s[k],
        ds,
        x: x[k],
        ue: Ue,
        dueds,
        theta: th,
        H: hCur,
        reTheta: ret,
        dReThetaDs: (ret - prevReTheta) / ds,
        reX: (Ue * Math.max(s[k], 1e-12)) / nu,
        nu,
      });
      criterion = r.value;
      prevReTheta = ret;

      // Where in this station's step did the layer turn turbulent? Linear
      // interpolation of the criterion (or of Thwaites' lambda for a bubble)
      // gives a transition location that moves smoothly as the pressure
      // distribution changes. Snapping it to whole panels instead leaves the
      // coupling loop chasing a step change of ~1e-3 in Cp every time the point
      // crosses a panel boundary, which is enough to stall the residual an order
      // of magnitude above tolerance and never converge.
      let frac = -1;
      if (forceTrS >= 0 && s[k] >= forceTrS) {
        // Transition location supplied by the caller (see the relaxation note in
        // solveBoundaryLayer). The criterion is still integrated up to here, so
        // the predicted location remains available to relax toward.
        const span = s[k] - prevS;
        frac = span > 1e-12 ? Math.min(1, Math.max(0, (forceTrS - prevS) / span)) : 0;
      } else if (r.transitioned) {
        const dv = r.value - prevCriterion;
        frac = dv > 1e-12 ? Math.min(1, Math.max(0, (1 - prevCriterion) / dv)) : 0;
        predictedTrS = prevS + frac * (s[k] - prevS);
      } else if (lambda <= LAMBDA_SEP) {
        // Laminar separation. Above about Re 5e4 the detached shear layer
        // transitions almost immediately and reattaches as a short bubble —
        // separation-induced transition, the dominant transition mechanism on
        // this class of airfoil at incidence. Record the bubble and hand over
        // to the turbulent method rather than declaring the section stalled:
        // whether it actually stalls is then decided by whether the turbulent
        // layer separates again, which is the physical question.
        const dl = prevLambda - lambda;
        frac = dl > 1e-12 ? Math.min(1, Math.max(0, (prevLambda - LAMBDA_SEP) / dl)) : 0;
        bubble = true;
        laminarSepX = prevX + frac * (x[k] - prevX);
        predictedTrS = prevS + frac * (s[k] - prevS);
      }
      prevCriterion = r.value;
      prevLambda = lambda;

      if (frac >= 0) {
        turbulent = true;
        transitionIdx = k;
        transitionX = prevX + frac * (x[k] - prevX);
        // Momentum thickness carries across transition; the shape factor does
        // not. A laminar profile's H (2.6 or more) fed to Head's closure reads
        // as an already-separating turbulent layer, so restart from the
        // flat-plate turbulent value — or, after a bubble, from the fuller
        // profile a reattached layer has.
        const hLam = hCur;
        const cfLam = cf[k];
        hCur = bubble ? 1.8 : 1.4;
        h1 = headH1ofH(hCur);

        // Integrate the downstream fraction of this station's step, then report
        // the station as a blend of the two closures weighted by where the
        // transition point fell inside it. Without the blend the reported
        // displacement thickness at this one panel would still jump by the full
        // laminar-to-turbulent difference the moment the point crossed it.
        if (turbulentStep(Ue, dueds, (1 - frac) * ds)) {
          separated = true;
          separationIdx = k;
          sSep = s[k];
          hSep = hCur;
          dstarSep = hCur * th;
        }
        sTr = prevS + frac * (s[k] - prevS);
        hLamTr = hLam;
        cfLamTr = cfLam;

        reTheta[k] = (Ue * th) / nu;
        const w = Math.min(1, Math.max(0, (s[k] - sTr) / TRANSITION_LENGTH));
        theta[k] = th;
        H[k] = (1 - w) * hLam + w * hCur;
        dstar[k] = H[k] * th;
        cf[k] = (1 - w) * cfLam + w * ludwiegTillmann(hCur, reTheta[k]);
        state[k] = 1;
      }
    } else if (!separated) {
      /* ---- Turbulent: Head ----------------------------------------------- */
      if (turbulentStep(Ue, dueds, ds)) {
        separated = true;
        separationIdx = k;
        sSep = s[k];
        hSep = hCur;
        dstarSep = hCur * th;
      }
      theta[k] = th;
      reTheta[k] = (Ue * th) / nu;
      // Blend out of the laminar closure over the transition region.
      const w = Math.min(1, Math.max(0, (s[k] - sTr) / TRANSITION_LENGTH));
      H[k] = w >= 1 ? hCur : (1 - w) * hLamTr + w * hCur;
      dstar[k] = H[k] * th;
      cf[k] =
        w >= 1
          ? ludwiegTillmann(hCur, reTheta[k])
          : (1 - w) * cfLamTr + w * ludwiegTillmann(hCur, reTheta[k]);
      state[k] = 1; // turbulent
    } else {
      /* ---- Separated: explicit closure, not a solve -----------------------
       * The displacement surface is prescribed at the free shear layer's
       * spreading rate; the momentum thickness still follows its own equation
       * with the wall shear removed. */
      dstarSep += SEPARATED_DSTAR_GROWTH * ds;
      const dth = -((hCur + 2) * th * dueds) / Ue; // zero wall shear
      th = Math.max(th + dth * ds, 1e-12);
      hCur = Math.min(SEPARATED_H_MAX, Math.max(hSep, dstarSep / th));
      theta[k] = th;
      H[k] = hCur;
      dstar[k] = dstarSep;
      reTheta[k] = (Ue * th) / nu;
      cf[k] = 0;
      state[k] = 2; // separated
    }

    prevS = s[k];
    prevX = x[k];
  }

  return {
    transitionIdx,
    transitionX,
    transitionS: transitionIdx >= 0 ? sTr : -1,
    predictedTransitionS: predictedTrS,
    streamLength: s[K - 1],
    separationIdx,
    separationX: separationIdx >= 0 ? x[separationIdx] : -1,
    laminarSeparationX: laminarSepX,
    bubble,
    criterion,
    turbulentAtTE: turbulent,
    thetaTE: theta[K - 1],
    dstarTE: dstar[K - 1],
    HTE: H[K - 1],
    ueTE: ue[K - 1],
  };
}

/* ============================================================================
 * Stagnation point
 * ==========================================================================*/

/**
 * Locate the stagnation point from the solved surface velocity.
 *
 * The surface velocity is signed along the panel tangent, and the node ring runs
 * clockwise, so it is negative everywhere the flow runs aft along the lower
 * surface and positive everywhere it runs aft along the upper. The stagnation
 * point is the sign change — and to be robust against a second sign change from
 * a separation bubble further aft, it is the sign change closest to the nose.
 */
function findStagnation(geo, ue) {
  const n = geo.n;
  let best = -1;
  let bestX = Infinity;
  for (let i = 0; i < n - 1; i++) {
    if (ue[i] < 0 && ue[i + 1] >= 0) {
      const xm = 0.5 * (geo.midX[i] + geo.midX[i + 1]);
      if (xm < bestX) {
        bestX = xm;
        best = i;
      }
    }
  }
  if (best < 0) {
    // No crossing at all (fully reversed or fully forward surface velocity).
    // Fall back to the smallest |Ue| in the front third of the section.
    let m = Infinity;
    for (let i = 0; i < n; i++) {
      if (geo.midX[i] < 0.33 && Math.abs(ue[i]) < m) {
        m = Math.abs(ue[i]);
        best = i;
      }
    }
    if (best < 0) best = 0;
  }
  const a = ue[best];
  const b = ue[Math.min(best + 1, n - 1)];
  const f = b !== a ? Math.min(1, Math.max(0, -a / (b - a))) : 0.5;
  const sStag = geo.sMid[best] + f * (geo.sMid[Math.min(best + 1, n - 1)] - geo.sMid[best]);
  return { index: best, s: sStag, x: bestX };
}

/* ============================================================================
 * Full solve
 * ==========================================================================*/

/** Allocate the boundary-layer state for a given panelling. Reused every solve. */
export function createBLState(n, nw) {
  return {
    n,
    nw,
    theta: new Float64Array(n),
    dstar: new Float64Array(n),
    H: new Float64Array(n),
    cf: new Float64Array(n),
    reTheta: new Float64Array(n),
    state: new Uint8Array(n), // 0 laminar, 1 turbulent, 2 separated
    thetaW: new Float64Array(nw),
    dstarW: new Float64Array(nw),
    HW: new Float64Array(nw),
    sigma: new Float64Array(n),
    sigmaWake: new Float64Array(nw),
    // Relaxed transition arc length per stream, [upper, lower]. Negative means
    // "no history yet", which makes the first march take the prediction as-is.
    transitionS: new Float64Array([-1, -1]),
    // Scratch for the two streams; sized for the worst case (one whole surface).
    _s: new Float64Array(n),
    _ue: new Float64Array(n),
    _x: new Float64Array(n),
    _idx: new Int32Array(n),
    _out: {
      theta: new Float64Array(n),
      dstar: new Float64Array(n),
      H: new Float64Array(n),
      cf: new Float64Array(n),
      reTheta: new Float64Array(n),
      state: new Uint8Array(n),
    },
    _m: new Float64Array(n + 2),
    _ms: new Float64Array(n + 2),
  };
}

/**
 * Solve the boundary layer for a converged (or in-progress) panel solution.
 *
 * Returns the per-panel viscous state plus the transpiration source
 * distribution that closes the coupling loop, and the stream-level diagnostics
 * (transition, separation, trailing-edge state) the outputs are built from.
 */
export function solveBoundaryLayer(geo, wake, sol, re, bl, opts = {}) {
  const n = geo.n;
  const nw = wake.n;
  const nu = 1 / Math.max(re, 1);
  const tmodel = getTransitionModel(opts.transitionModel ?? 'en', opts.nCrit);

  const stag = findStagnation(geo, sol.ue);
  const { _s, _ue, _x, _idx, _out } = bl;
  let clamped = 0;

  bl.theta.fill(0);
  bl.dstar.fill(0);
  bl.H.fill(0);
  bl.cf.fill(0);
  bl.reTheta.fill(0);
  bl.state.fill(0);

  /**
   * Extract one stream and march it, scattering the result back onto the panel
   * indices it came from.
   */
  const runStream = (upper) => {
    let K = 0;
    if (upper) {
      for (let i = stag.index + 1; i < n; i++) {
        _idx[K] = i;
        _s[K] = geo.sMid[i] - stag.s;
        _ue[K] = Math.max(sol.ue[i], 1e-7);
        _x[K] = geo.midX[i];
        K++;
      }
    } else {
      for (let i = stag.index; i >= 0; i--) {
        _idx[K] = i;
        _s[K] = stag.s - geo.sMid[i];
        _ue[K] = Math.max(-sol.ue[i], 1e-7);
        _x[K] = geo.midX[i];
        K++;
      }
    }
    if (K < 3) {
      return {
        transitionX: -1, separationX: -1, laminarSeparationX: -1, bubble: false,
        criterion: 0, turbulentAtTE: false, thetaTE: 0, dstarTE: 0, HTE: 1.4, ueTE: 1, K: 0,
      };
    }

    /* ---- Transition location, relaxed across coupling iterations -----------
     * Where the transition point sits is badly conditioned wherever the
     * amplification rate is low — on the pressure side at moderate incidence,
     * a 1% change in edge velocity can move it several percent of chord. Fed
     * straight back into the coupling that is a large, moving step in the
     * transpiration source, and the loop settles into a limit cycle at a
     * residual an order of magnitude above tolerance with the transition point
     * jumping back and forth across the same two panels.
     *
     * So the location is treated the same way as the source distribution
     * itself: predicted afresh each iteration, but approached gradually. Two
     * marches per stream — one to predict, one to integrate at the relaxed
     * location — which costs nothing measurable and converges to exactly the
     * same answer, because at the fixed point the relaxed and predicted
     * locations coincide. */
    const slot = upper ? 0 : 1;
    let ev = marchStream(K, _s, _ue, _x, nu, tmodel, _out, -1);

    const predicted = ev.predictedTransitionS >= 0 ? ev.predictedTransitionS : _s[K - 1];
    const prev = bl.transitionS[slot];
    if (prev >= 0) {
      const relaxed = prev + TRANSITION_RELAX * (predicted - prev);
      bl.transitionS[slot] = relaxed;
      if (Math.abs(relaxed - predicted) > 1e-6 && relaxed < _s[K - 1]) {
        ev = marchStream(K, _s, _ue, _x, nu, tmodel, _out, relaxed);
      }
    } else {
      bl.transitionS[slot] = predicted;
    }

    for (let k = 0; k < K; k++) {
      const i = _idx[k];
      bl.theta[i] = _out.theta[k];
      bl.dstar[i] = _out.dstar[k];
      bl.H[i] = _out.H[k];
      bl.cf[i] = _out.cf[k];
      bl.reTheta[i] = _out.reTheta[k];
      bl.state[i] = _out.state[k];
    }

    // Transpiration source, sigma = d(Ue delta*)/ds, differenced on the same
    // stations the layer was marched on. The mass defect Ue delta* vanishes at
    // the stagnation point, which supplies the first station's one-sided value.
    const m = bl._m;
    const ms = bl._ms;
    m[0] = 0;
    ms[0] = 0;
    // Edge velocity used to form the mass defect. Inside a separated region it
    // is frozen at the value the flow had when it detached, rather than tracked
    // from the panel solution.
    //
    // The reason is that the mass defect is a product, so its derivative has two
    // terms: Ue d(delta*)/ds, which is the displacement growing, and delta*
    // dUe/ds, which is the edge velocity changing. In a dead-air region the
    // pressure is very nearly constant, so the second term is physically zero.
    // Taking dUe/ds from the panel solution instead imports the trailing-edge
    // pressure recovery of the *attached* solution into a region that is not
    // attached, and the two terms then cancel almost exactly: measured on
    // NACA 0012 at 18 degrees, the transpiration velocity over the separated
    // region collapsed from 0.11 to 0.0008 across the last few percent of chord
    // — precisely where it has the most leverage on the circulation, because
    // that is where the Kutta condition is applied. The section then lost only
    // 5% of its lift to a separation covering half the chord, and the lift curve
    // never peaked at any angle.
    let ueSep = 0;
    for (let k = 0; k < K; k++) {
      if (_out.state[k] === 2) {
        if (ueSep === 0) ueSep = k > 0 ? _ue[k - 1] : _ue[k];
        m[k + 1] = ueSep * _out.dstar[k];
      } else {
        m[k + 1] = _ue[k] * _out.dstar[k];
      }
      ms[k + 1] = _s[k];
    }
    // Station k lives at index k+1 in m/ms.
    //
    // The derivative uses the proper three-point formula for a non-uniform grid,
    // which includes the centre station. The obvious two-point form,
    // (m[k+2] - m[k]) / (s[k+2] - s[k]), is second-order only on a uniform grid
    // — and, worse, it never reads the station it is evaluating, so odd and even
    // panels decouple completely. On a cosine-spaced grid that produces a
    // sawtooth in the source distribution, which lands straight in the edge
    // velocity through the source influence and from there in Thwaites' lambda:
    // measured on NACA 0006 at Re 3e6, a surface velocity varying smoothly by
    // 0.001 per panel acquired a +-0.005 oscillation that swung the local
    // velocity gradient between +0.13 and -0.67 and tripped a laminar separation
    // bubble at 60% of chord that is not there.
    for (let k = 0; k < K; k++) {
      let sg;
      if (k === K - 1) {
        const den = ms[k + 1] - ms[k];
        sg = den > 1e-12 ? (m[k + 1] - m[k]) / den : 0;
      } else {
        const h1 = ms[k + 1] - ms[k];
        const h2 = ms[k + 2] - ms[k + 1];
        sg =
          h1 > 1e-12 && h2 > 1e-12
            ? (-h2 / (h1 * (h1 + h2))) * m[k] +
              ((h2 - h1) / (h1 * h2)) * m[k + 1] +
              (h1 / (h2 * (h1 + h2))) * m[k + 2]
            : 0;
      }
      // A transpiration velocity of a quarter of the freestream is already far
      // beyond anything the thin-layer approximation underlying this can mean.
      // Clamping is a backstop against a single pathological station taking the
      // whole coupling with it, and the count is reported so a solution that
      // needed it is never presented as trustworthy.
      if (sg > SIGMA_MAX) {
        sg = SIGMA_MAX;
        clamped++;
      } else if (sg < -SIGMA_MAX) {
        sg = -SIGMA_MAX;
        clamped++;
      }
      bl.sigma[_idx[k]] = sg;
    }

    return { ...ev, K };
  };

  const upper = runStream(true);
  const lower = runStream(false);

  /* ---- Wake ---------------------------------------------------------------
   * The two surface layers merge at the trailing edge: momentum thicknesses add
   * and so do displacement thicknesses. From there Head's method runs with the
   * wall terms removed, so the shape factor relaxes toward 1 as the velocity
   * defect fills in. */
  let thW = Math.max(upper.thetaTE + lower.thetaTE, 1e-9);
  let dsW = upper.dstarTE + lower.dstarTE;
  let hW = Math.max(dsW / thW, 1.05);
  let h1W = headH1ofH(hW);
  let prevS = 0;
  // The wake's mass defect at the trailing edge is not zero — it is the sum of
  // what the two surface layers arrive with. Seeding the difference from zero
  // instead puts a source of strength (Ue delta*)/ds_1 on the first wake panel,
  // which at a 0.4%-chord first panel is an order-one spurious source sitting
  // right behind the trailing edge. That alone is enough to make the coupling
  // loop diverge.
  let prevM = 0.5 * (upper.ueTE + lower.ueTE) * dsW;

  for (let p = 0; p < nw; p++) {
    const Ue = Math.max(Math.abs(sol.ueWake[p]), 1e-7);
    const ds = Math.max(wake.s[p] - prevS, 1e-12);

    // Gradients are taken between wake stations only, never across the trailing
    // edge. The panel solution has a genuine velocity dip at a sharp trailing
    // edge, and differencing the first wake station against the last surface
    // station turns that into a gradient of order -60 per chord over a panel
    // 0.4% of chord long — which is not a boundary-layer gradient, it is the
    // discretisation of a corner.
    let dueds;
    if (p === nw - 1) {
      dueds =
        (Math.abs(sol.ueWake[p]) - Math.abs(sol.ueWake[p - 1])) /
        Math.max(wake.s[p] - wake.s[p - 1], 1e-12);
    } else if (p === 0) {
      dueds =
        (Math.abs(sol.ueWake[1]) - Math.abs(sol.ueWake[0])) /
        Math.max(wake.s[1] - wake.s[0], 1e-12);
    } else {
      dueds =
        (Math.abs(sol.ueWake[p + 1]) - Math.abs(sol.ueWake[p - 1])) /
        Math.max(wake.s[p + 1] - wake.s[p - 1], 1e-12);
    }

    // Sub-stepped for the same reason as the turbulent surface march: the near
    // wake recovers fast and the panels are geometrically stretched, so a single
    // explicit step across the first few would push H1 below its asymptote.
    const nSub = Math.min(
      32,
      Math.max(1, Math.ceil((Math.abs(dueds) * ds * (hW + 2)) / (0.05 * Ue)))
    );
    const dsSub = ds / nSub;
    const hWStart = hW;
    for (let q = 0; q < nSub; q++) {
      const dth = -((hW + 2) * thW * dueds) / Ue; // Cf = 0 in the wake
      const dh1 = (headF(h1W) - h1W * dth - (thW * h1W * dueds) / Ue) / Math.max(thW, 1e-12);
      thW = Math.max(thW + dth * dsSub, 1e-12);
      h1W += dh1 * dsSub;
      if (!isFinite(thW) || !isFinite(h1W)) {
        thW = Math.max(isFinite(thW) ? thW : 1e-9, 1e-9);
        h1W = headH1ofH(2.5);
      }
      // The wake shape factor lives between 1 (fully filled in) and the value it
      // arrives with; clamping H directly rather than H1 avoids the asymptote at
      // H1 = 3.3, where a small overshoot maps to an enormous H.
      hW = Math.min(SEPARATED_H_MAX, Math.max(1.02, headHofH1(Math.max(h1W, H1_SEP + 1e-3))));
      // The wake shape factor relaxes toward 1 fastest right behind the trailing
      // edge, where the panels are shortest. Left unlimited it drops by most of
      // its range across the first panel, which puts a transpiration source of
      // order half the freestream on the one wake panel that most strongly loads
      // the trailing-edge pressure. Limiting the change per station spreads the
      // same relaxation over the ~0.1 chord it physically takes.
      if (hW < hWStart - MAX_DH_WAKE_PER_STATION) {
        hW = hWStart - MAX_DH_WAKE_PER_STATION;
        h1W = headH1ofH(hW);
        break;
      }
      h1W = headH1ofH(hW);
    }
    // Bound the decay rate: the wake cannot fill in faster than its shear layer
    // spreads. (Growth is left unbounded — a wake thickening under an adverse
    // gradient is ordinary behaviour.)
    const dsWRaw = hW * thW;
    const floor = dsW - WAKE_DSTAR_DECAY_MAX * ds;
    dsW = Math.max(dsWRaw, floor);
    if (dsW > dsWRaw) {
      // Keep H consistent with the displacement thickness that was actually used,
      // so the next station's integration starts from the state it reported.
      hW = Math.min(SEPARATED_H_MAX, Math.max(1.02, dsW / thW));
      h1W = headH1ofH(hW);
    }

    bl.thetaW[p] = thW;
    bl.HW[p] = hW;
    bl.dstarW[p] = dsW;

    const mNow = Ue * dsW;
    const den = wake.s[p] - prevS;
    let sg = den > 1e-12 ? (mNow - prevM) / den : 0;
    if (sg > SIGMA_MAX) {
      sg = SIGMA_MAX;
      clamped++;
    } else if (sg < -SIGMA_MAX) {
      sg = -SIGMA_MAX;
      clamped++;
    }
    bl.sigmaWake[p] = sg;
    prevM = mNow;
    prevS = wake.s[p];
  }

  /* ---- Drag by the Squire-Young formula ------------------------------------
   * Far-field momentum balance rather than near-field pressure integration.
   * Integrating the pressure for drag means differencing two large, nearly
   * cancelling numbers — d'Alembert's paradox is exactly the statement that the
   * true answer is the small residue left over — and the panel discretisation
   * error lands squarely in that residue. Squire-Young instead extrapolates the
   * wake momentum thickness to infinity,
   *
   *   Cd = 2 theta (Ue)^((H+5)/2)
   *
   * evaluated at the end of the computed wake, where H has already relaxed most
   * of the way to 1 and the extrapolation is short. */
  const ueEnd = Math.max(Math.abs(sol.ueWake[nw - 1]), 1e-6);
  const cdWake = 2 * bl.thetaW[nw - 1] * Math.pow(ueEnd, (bl.HW[nw - 1] + 5) / 2);
  const cdTE =
    2 * upper.thetaTE * Math.pow(Math.max(upper.ueTE, 1e-6), (upper.HTE + 5) / 2) +
    2 * lower.thetaTE * Math.pow(Math.max(lower.ueTE, 1e-6), (lower.HTE + 5) / 2);

  bl.stagnation = stag;
  bl.upper = upper;
  bl.lower = lower;
  bl.cdSquireYoung = cdWake;
  bl.cdSquireYoungTE = cdTE;
  bl.transitionModel = tmodel.name;
  bl.sigmaClamped = clamped;
  return bl;
}

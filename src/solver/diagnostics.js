/**
 * diagnostics.js — stall classification and solution confidence.
 *
 * Both of these are read off the converged state rather than compared against a
 * table. There is no critical-angle lookup anywhere in this solver: the section
 * stalls when the boundary layer says it has, and how far along that process it
 * is comes from combining several independent symptoms that all appear as stall
 * develops, none of which is decisive alone.
 */

/* ============================================================================
 * Stall index
 * ==========================================================================*/

/** Weights, per the specification. They sum to 1. */
const W_SEPARATION = 0.35;
const W_SLOPE_LOSS = 0.2;
const W_DISPLACEMENT = 0.2;
const W_WAKE = 0.15;
const W_CONVERGENCE = 0.1;

/**
 * Reference lift-curve slope, per radian.
 *
 * Thin-airfoil theory gives 2 pi; a real section of finite thickness gives a
 * little more, which is why the slope loss is measured against the section's own
 * *inviscid* slope rather than against 2 pi. That way a thick section is not
 * reported as permanently 5% stalled.
 */
export function slopeLoss(viscousSlopePerRad, inviscidSlopePerRad) {
  if (!(inviscidSlopePerRad > 0.1)) return 0;
  const ratio = viscousSlopePerRad / inviscidSlopePerRad;
  return Math.min(1, Math.max(0, 1 - ratio));
}

/**
 * Continuous stall index in [0, 1], from five independent symptoms.
 *
 * Each is normalised to [0, 1] on a scale chosen so that "1" means the symptom
 * is as severe as it gets on a section that is genuinely, fully stalled:
 *
 *   separationFraction   fraction of chord separated on the worse surface. The
 *                        single most direct measure, hence the largest weight.
 *   liftSlopeLoss        how much of the inviscid lift-curve slope the viscous
 *                        solution has lost. Reaches 1 at zero slope, which is
 *                        the textbook definition of stall.
 *   displacementGrowth   trailing-edge displacement thickness against chord.
 *                        Rises before separation does, so this is the earliest
 *                        of the five to move.
 *   wakeGrowth           wake momentum thickness against chord — the far-field
 *                        statement of the same thing, and the one that responds
 *                        to a separation that has not yet reached the trailing
 *                        edge.
 *   convergenceQuality   how well the coupled solve settled. Deliberately
 *                        included and deliberately smallest: a solve that will
 *                        not converge is itself evidence of separated flow, but
 *                        it is evidence about the solver, not only the flow.
 */
export function stallIndex(inputs) {
  const {
    separationFraction = 0,
    liftSlopeLoss = 0,
    displacementTE = 0,
    wakeTheta = 0,
    convergenceQuality = 1,
  } = inputs;

  const sep = clamp01(separationFraction);
  const slope = clamp01(liftSlopeLoss);
  // A trailing-edge displacement thickness of 6% of chord is a thoroughly
  // stalled section; 0.5% is a healthy attached one.
  const disp = clamp01((displacementTE - 0.005) / 0.055);
  // Wake momentum thickness scales with total drag: theta ~ Cd/2, so 5% of
  // chord corresponds to Cd ~ 0.1, which is deep stall.
  const wake = clamp01((wakeTheta - 0.004) / 0.046);
  const conv = clamp01(1 - convergenceQuality);

  const index =
    W_SEPARATION * sep +
    W_SLOPE_LOSS * slope +
    W_DISPLACEMENT * disp +
    W_WAKE * wake +
    W_CONVERGENCE * conv;

  return {
    index: clamp01(index),
    state: classify(index),
    components: {
      separationFraction: sep,
      liftSlopeLoss: slope,
      displacementGrowth: disp,
      wakeGrowth: wake,
      convergencePenalty: conv,
    },
  };
}

/** Dynamic classification bands, per the specification. */
export function classify(index) {
  if (index < 0.2) return 'attached';
  if (index < 0.4) return 'incipient';
  if (index < 0.7) return 'partial';
  return 'full';
}

export const STALL_LABELS = {
  attached: 'Attached',
  incipient: 'Near stall',
  partial: 'Partial stall',
  full: 'Deep stall',
};

function clamp01(v) {
  if (!isFinite(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/* ============================================================================
 * Confidence
 * ==========================================================================*/

/**
 * How far to trust this particular answer, as a percentage, with the reason.
 *
 * A solver that reports four significant figures at 25 degrees with a burst
 * leading-edge bubble is lying by omission. This is the counterweight: it starts
 * at full confidence and deducts for every specific, identifiable reason the
 * model is being asked something it is not good at.
 *
 * The deductions are multiplicative because the failure modes compound — an
 * unconverged solve at an angle outside the validated envelope with a large
 * separated region is not three independent small problems.
 */
export function confidence(state) {
  const {
    converged,
    residualCp,
    iterations,
    maxIterations,
    separationFraction,
    alphaDeg,
    reynolds,
    conditionNumber,
    sigmaClamped,
    bubbleBurst,
  } = state;

  let c = 1;
  const reasons = [];

  if (!converged) {
    // Scale the deduction by how far off convergence actually was, rather than
    // treating "not converged" as one thing.
    const decades = Math.log10(Math.max(residualCp, 1e-12) / 1e-4);
    const penalty = Math.min(0.55, 0.08 * Math.max(decades, 0));
    if (penalty > 0.005) {
      c *= 1 - penalty;
      reasons.push(
        `coupling stopped at ${iterations}/${maxIterations} iterations with a pressure residual of ${residualCp.toExponential(1)}`
      );
    }
  }

  if (separationFraction > 0.02) {
    // Separated flow is where an integral boundary layer stops being a model of
    // anything. Below ~10% of chord it is ordinary trailing-edge thickening and
    // costs little; beyond half the chord the answer is indicative at best.
    const penalty = Math.min(0.7, 0.9 * Math.max(separationFraction - 0.02, 0));
    c *= 1 - penalty;
    reasons.push(
      `${(separationFraction * 100).toFixed(0)}% of the chord is separated, past the range an integral boundary layer resolves`
    );
  }

  const absAlpha = Math.abs(alphaDeg);
  if (alphaDeg > 18 || alphaDeg < -8) {
    c *= 0.75;
    reasons.push(`${alphaDeg.toFixed(1)}° is outside the -8° to 18° band this model is calibrated over`);
  } else if (absAlpha > 25) {
    c *= 0.5;
  }

  if (reynolds < 5e4) {
    c *= 0.7;
    reasons.push(`Re ${reynolds.toExponential(1)} is below the 5×10⁴ floor, where laminar bubbles dominate`);
  } else if (reynolds > 2e7) {
    c *= 0.85;
    reasons.push(`Re ${reynolds.toExponential(1)} is above the 2×10⁷ ceiling of the transition correlations`);
  }

  if (conditionNumber > 1e8) {
    c *= 0.6;
    reasons.push('the panel influence matrix is poorly conditioned for this geometry');
  }

  if (sigmaClamped > 0) {
    c *= 0.85;
    reasons.push(`${sigmaClamped} boundary-layer station${sigmaClamped > 1 ? 's' : ''} needed limiting`);
  }

  if (bubbleBurst) {
    c *= 0.6;
    reasons.push('a laminar separation bubble failed to reattach — leading-edge stall is not resolved by this model');
  }

  return {
    score: Math.round(clamp01(c) * 100),
    reasons,
    summary: reasons.length === 0 ? 'Converged, attached, and inside the calibrated envelope.' : reasons[0],
  };
}

/**
 * Convergence quality in [0, 1], for the stall index.
 *
 * A converged solve is 1. Beyond that it decays one decade of residual at a
 * time, so a solve that is one decade short still counts as mostly healthy while
 * one that is four decades short does not.
 */
export function convergenceQuality(converged, residualCp, tolerance) {
  if (converged) return 1;
  if (!isFinite(residualCp)) return 0;
  const decades = Math.log10(Math.max(residualCp, 1e-14) / tolerance);
  return clamp01(1 - decades / 4);
}

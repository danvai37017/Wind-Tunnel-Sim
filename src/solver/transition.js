/**
 * transition.js — laminar-to-turbulent transition prediction.
 *
 * Deliberately a small pluggable interface rather than a hard-wired criterion.
 * Transition is the single largest source of disagreement between an integral
 * boundary-layer code and experiment, and which correlation is right depends on
 * the Reynolds number, the freestream turbulence and whether there is a
 * separation bubble involved. Locking one in would be a modelling decision
 * disguised as an implementation detail.
 *
 * A model is an object with:
 *
 *   name            display name
 *   create()        -> per-stream state (one boundary layer marching from the
 *                      stagnation point to the trailing edge)
 *   step(state, st) -> { transitioned, value }
 *
 * where `st` carries the local station data { s, ds, x, ue, dueds, theta, H,
 * reTheta, reX, nu }. `value` is the normalised criterion (1 at transition), so
 * the UI can show how close the layer is without knowing which model is running.
 *
 * Adding gamma-Re_theta (Langtry-Menter) or a dedicated bubble model means
 * adding an object here; nothing downstream changes.
 */

/* ============================================================================
 * Michel's criterion
 * ==========================================================================*/

/**
 * Michel (1951): a single algebraic relation between the momentum-thickness and
 * streamwise Reynolds numbers,
 *
 *   Re_theta,crit = 1.174 (1 + 22400 / Re_x) Re_x^0.46
 *
 * Correlated on flat plates and mild gradients. Cheap, has no state, and is
 * remarkably durable for attached flow at moderate Reynolds number — but it
 * knows nothing about the pressure gradient, so it transitions too late in a
 * strong adverse gradient and too early in a strong favourable one.
 */
export const michel = {
  name: 'Michel',
  create: () => ({ value: 0 }),
  step(state, st) {
    const reX = Math.max(st.reX, 1);
    const crit = 1.174 * (1 + 22400 / reX) * Math.pow(reX, 0.46);
    const v = st.reTheta / crit;
    state.value = Math.max(state.value, v);
    return { transitioned: v >= 1, value: state.value };
  },
};

/* ============================================================================
 * e^n envelope method
 * ==========================================================================*/

/**
 * Critical momentum-thickness Reynolds number at which Tollmien-Schlichting
 * waves first become unstable, as a function of the shape factor (Drela &
 * Giles 1987):
 *
 *   log10 Re_theta,0 = (1.415/(H-1) - 0.489) tanh(20/(H-1) - 12.9)
 *                      + 3.295/(H-1) + 0.44
 *
 * H is clamped to the range the correlation was fitted over: below ~2.1 the
 * profile is no longer Falkner-Skan laminar and above ~6 it has separated, and
 * outside that the 1/(H-1) terms run away.
 */
function reThetaCrit(H) {
  const h = Math.min(6, Math.max(2.1, H));
  const d = h - 1;
  const l = (1.415 / d - 0.489) * Math.tanh(20 / d - 12.9) + 3.295 / d + 0.44;
  return Math.pow(10, l);
}

/**
 * Envelope amplification rate,
 *
 *   dn/dRe_theta = 0.01 sqrt{ [2.4H - 3.7 + 2.5 tanh(1.5H - 4.65)]^2 + 0.25 }
 *
 * This is the envelope of the Orr-Sommerfeld amplification curves for the
 * Falkner-Skan family, which is what makes e^n usable in a marching integral
 * method: instead of tracking every frequency, track the maximum amplification
 * any frequency has reached.
 */
function dnDReTheta(H) {
  const h = Math.min(6, Math.max(2.1, H));
  const g = 2.4 * h - 3.7 + 2.5 * Math.tanh(1.5 * h - 4.65);
  return 0.01 * Math.sqrt(g * g + 0.25);
}

/**
 * The e^n method: integrate the amplification of the most-amplified disturbance
 * and declare transition when it has grown by a factor of e^Ncrit.
 *
 * Ncrit encodes the disturbance environment rather than the flow:
 *   Ncrit ~ 9    clean wind tunnel / smooth free flight  (the default)
 *   Ncrit ~ 4-7  average tunnel, some freestream turbulence
 *   Ncrit ~ 11+  very quiet flow
 *
 * Unlike Michel this responds to the pressure gradient — through H, which is
 * what an adverse gradient changes first — so it moves transition forward on the
 * suction side at incidence, which is the behaviour that matters here.
 */
export function eN(nCrit = 9) {
  return {
    name: `e^n (Ncrit=${nCrit})`,
    nCrit,
    create: () => ({ n: 0, value: 0 }),
    step(state, st) {
      const reTheta = st.reTheta;
      const crit = reThetaCrit(st.H);
      if (reTheta > crit) {
        // dn/ds = (dn/dRe_theta)(dRe_theta/ds); the local derivative of Re_theta
        // is taken from the marched state, so no extra differencing is needed.
        const growth = dnDReTheta(st.H) * Math.max(st.dReThetaDs, 0);
        state.n += growth * st.ds;
      }
      state.value = state.n / nCrit;
      return { transitioned: state.n >= nCrit, value: state.value };
    },
  };
}

/** The models the solver ships with, by key. */
export const TRANSITION_MODELS = {
  en: eN(9),
  michel,
};

export function getTransitionModel(key, nCrit) {
  if (key === 'michel') return michel;
  if (key === 'en' && typeof nCrit === 'number') return eN(nCrit);
  return TRANSITION_MODELS.en;
}

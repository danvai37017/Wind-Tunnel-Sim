/**
 * heatmap.js — surface scalar fields and the colour scales that render them.
 *
 * Two things live here, and nothing else:
 *
 *   1. What quantity to draw on the surface. Every mode reads straight off the
 *      converged state, so the colour at a point on the wing and the number in
 *      the dashboard are the same number. Nothing is re-derived and nothing is
 *      modelled here.
 *   2. How to turn that quantity into a colour.
 *
 * ## Why the ramps are interpolated in OKLab
 *
 * Interpolating blue to yellow in sRGB passes through grey, because sRGB is not
 * a perceptual space and the midpoint of two saturated colours is usually a
 * desaturated one. That shows up as a dead band across the middle of the scale
 * exactly where the interesting gradients tend to sit. OKLab is (near enough)
 * perceptually uniform, so a straight line between two stops stays saturated and
 * the steps are visually even. The conversion is done once at module load into a
 * 256-entry lookup table per scale; the render loop only ever indexes it.
 *
 * ## On rainbow scales
 *
 * The default scale is the blue-cyan-green-yellow-orange-red progression, which
 * is what people expect from a CFD surface plot. It is worth being honest that
 * rainbow ramps are not perceptually monotonic: lightness climbs from blue to
 * yellow and falls again to red, so equal steps in the data are not equal steps
 * in apparent brightness, and the eye invents banding at the hue boundaries.
 * That is why `viridis` is offered alongside it — monotonic in lightness and
 * legible under every common form of colour vision deficiency — and why signed
 * quantities default to a diverging scale with a neutral midpoint pinned to
 * zero, which is the encoding that actually matches the physics.
 */

/* ============================================================================
 * Colour space
 * ==========================================================================*/

const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function hexToOklab(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = srgbToLinear(((n >> 16) & 255) / 255);
  const g = srgbToLinear(((n >> 8) & 255) / 255);
  const b = srgbToLinear((n & 255) / 255);

  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);

  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToRgb(L, A, B) {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;

  const r = linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s);
  const g = linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s);
  const b = linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s);

  const q = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
  return [q(r), q(g), q(b)];
}

/** Bake a list of hex stops into a 256-entry RGB table, interpolated in OKLab. */
function bakeRamp(stops) {
  const lab = stops.map(hexToOklab);
  const N = 256;
  const table = new Uint8Array(N * 3);
  for (let i = 0; i < N; i++) {
    const t = (i / (N - 1)) * (lab.length - 1);
    const k = Math.min(lab.length - 2, Math.floor(t));
    const f = t - k;
    const a = lab[k];
    const b = lab[k + 1];
    const [r, g, bl] = oklabToRgb(
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f
    );
    table[i * 3] = r;
    table[i * 3 + 1] = g;
    table[i * 3 + 2] = bl;
  }
  return table;
}

/* ============================================================================
 * The scales
 * ==========================================================================*/

export const COLOR_SCALES = [
  {
    id: 'spectral',
    label: 'Spectral (blue to red)',
    diverging: false,
    stops: ['#12246e', '#1c62c9', '#2ba7dd', '#35c46a', '#f2d13c', '#f0871d', '#d42a1f'],
    describe:
      'The familiar CFD blue to red rainbow. It is not perceptually uniform: lightness climbs to yellow and falls again to red, so equal steps in the data are not equal steps in brightness, and the eye tends to invent banding at the hue boundaries.',
  },
  {
    id: 'viridis',
    label: 'Viridis (safe for colour blindness)',
    diverging: false,
    stops: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
    describe:
      'Monotonic in lightness (dark purple through teal and green to bright yellow), so equal steps in the data read as equal steps in brightness. Stays legible under every common colour vision deficiency, which makes it the honest default when a specific hue is not doing work.',
  },
  {
    id: 'coolwarm',
    label: 'Cool to warm (diverging)',
    diverging: true,
    stops: ['#3b4cc0', '#7b9ff9', '#c9d7f0', '#dcdcdc', '#f2c0a6', '#ed8064', '#b40426'],
    describe:
      'Blue at one pole, red at the other, with a neutral grey midpoint. As a diverging scale it pins that midpoint to zero, which is the encoding that actually matches the physics for the signed quantities (pressure coefficient and wall vorticity), where zero has a real meaning.',
  },
  {
    id: 'inferno',
    label: 'Inferno',
    diverging: false,
    stops: ['#000004', '#320a5a', '#781c6d', '#bb3654', '#ed6925', '#fbb61a', '#fcffa4'],
    describe:
      'A dark to bright ramp that runs near black through deep red and orange to cream. The low end recedes into the background while the bright peaks stay readable, which suits quantities whose interesting detail sits at the high end.',
  },
  {
    id: 'mono',
    label: 'Monochrome',
    diverging: false,
    stops: ['#0d1117', '#2b3a52', '#54708f', '#8ba3c4', '#e8edf5'],
    describe:
      'Pure greyscale, dark to light. Useful when the plot has to survive greyscale print, a photocopier, or any medium where colour is unavailable.',
  },
];

const RAMPS = new Map(COLOR_SCALES.map((s) => [s.id, bakeRamp(s.stops)]));

export const DEFAULT_SCALE = 'spectral';

export function getScale(id) {
  return COLOR_SCALES.find((s) => s.id === id) ?? COLOR_SCALES[0];
}

/** Colour at normalised position t in [0, 1]. Writes into `out` as [r, g, b]. */
export function rampColor(scaleId, t, out = [0, 0, 0]) {
  const table = RAMPS.get(scaleId) ?? RAMPS.get(DEFAULT_SCALE);
  let i = Math.round(t * 255);
  if (!(i >= 0)) i = 0;
  else if (i > 255) i = 255;
  out[0] = table[i * 3];
  out[1] = table[i * 3 + 1];
  out[2] = table[i * 3 + 2];
  return out;
}

/** CSS gradient for the legend bar, straight off the baked ramp. */
export function rampCss(scaleId, steps = 12) {
  const c = [0, 0, 0];
  const parts = [];
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    rampColor(scaleId, t, c);
    parts.push(`rgb(${c[0]},${c[1]},${c[2]}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${parts.join(', ')})`;
}

/* ============================================================================
 * The modes
 *
 * Each mode pulls one array of per-panel values out of the converged state.
 * Panel i of the returned array is panel i of `state.pressure.x` / the geometry,
 * so the caller can map it onto the outline without any further bookkeeping.
 * ==========================================================================*/

export const HEAT_MODES = [
  {
    id: 'cp',
    label: 'Pressure coefficient',
    symbol: 'Cp',
    unit: '',
    // Cp has a meaningful zero and a meaningful sign — positive is pressure
    // above freestream static, negative is suction — so a diverging scale is
    // pinned to zero rather than to the middle of whatever range came out.
    signed: true,
    robust: false,
    digits: 3,
    values: (state) => state.pressure.cp,
    describe:
      'Surface pressure coefficient, integrated to give the lift and the moment. Warm is pressure above freestream static, cool is suction; the difference between the two surfaces is the lift.',
  },
  {
    id: 'velocity',
    label: 'Velocity magnitude',
    symbol: '|V|',
    unit: 'm/s',
    signed: false,
    robust: false,
    digits: 1,
    values: (state) => {
      const ue = state.velocity.ue;
      const v = state.inputs.airspeed;
      const out = new Float64Array(ue.length);
      for (let i = 0; i < ue.length; i++) out[i] = Math.abs(ue[i]) * v;
      return out;
    },
    describe:
      'Edge velocity just outside the boundary layer, from the same panel solution the pressure comes from. It peaks where the pressure is lowest; the two are the same statement of Bernoulli.',  },
  {
    id: 'q',
    label: 'Dynamic pressure',
    symbol: 'q',
    unit: 'Pa',
    signed: false,
    robust: false,
    digits: 0,
    values: (state) => {
      const ue = state.velocity.ue;
      const v = state.inputs.airspeed;
      const rho = state.inputs.rho;
      const out = new Float64Array(ue.length);
      for (let i = 0; i < ue.length; i++) {
        const s = ue[i] * v;
        out[i] = 0.5 * rho * s * s;
      }
      return out;
    },
    describe:
      'Local dynamic pressure ½ρV² at the edge of the boundary layer, the pressure the flow would recover if it were brought to rest there.',
  },
  {
    id: 'vorticity',
    label: 'Wall vorticity',
    symbol: 'ω',
    unit: '1/s',
    signed: true,
    // The stagnation region produces a couple of enormous outliers that would
    // otherwise flatten the whole rest of the scale onto one colour.
    robust: true,
    digits: 0,
    values: (state) => {
      // At a wall the vorticity is the wall-normal velocity gradient, and the
      // wall shear is exactly the viscosity times that gradient. So
      // omega_wall = tau_wall / mu, with mu = rho * nu. This is a genuine
      // surface quantity from the boundary-layer solution, not a field
      // vorticity — outside the layer this flow is potential and its vorticity
      // is zero everywhere except in the wake.
      const tau = state.boundaryLayer.wallShear;
      const ue = state.velocity.ue;
      const mu = state.inputs.rho * state.inputs.nu;
      const out = new Float64Array(tau.length);
      for (let i = 0; i < tau.length; i++) {
        out[i] = (tau[i] / mu) * (ue[i] < 0 ? -1 : 1);
      }
      return out;
    },
    describe:
      'Vorticity at the wall, τ_wall/μ, from the wall shear in the boundary layer. Sign follows the surface: the two sides of the section spin opposite ways, and the imbalance between them is the circulation that makes the lift.',
  },
];

export const DEFAULT_MODE = 'cp';

export function getMode(id) {
  return HEAT_MODES.find((m) => m.id === id) ?? HEAT_MODES[0];
}

/* ============================================================================
 * Domains
 * ==========================================================================*/

/**
 * The value range the ramp is stretched over.
 *
 * A diverging scale on a signed quantity is pinned symmetrically about zero, so
 * the neutral midpoint of the ramp lands on zero and the two poles mean equal
 * and opposite. Anything else and a diverging scale is just a rainbow with a
 * grey stripe in an arbitrary place.
 */
export function heatDomain(values, mode, scale) {
  const n = values.length;
  if (!n) return { lo: 0, hi: 1 };

  let lo;
  let hi;
  if (mode.robust) {
    // Percentile clip rather than min/max: one stagnation panel should not set
    // the scale for the other two hundred.
    const sorted = Float64Array.from(values).sort();
    lo = sorted[Math.floor(0.02 * (n - 1))];
    hi = sorted[Math.floor(0.98 * (n - 1))];
  } else {
    lo = Infinity;
    hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };

  if (scale.diverging && mode.signed) {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    return { lo: -m, hi: m };
  }
  if (hi - lo < 1e-12) return { lo: lo - 0.5, hi: hi + 0.5 };
  return { lo, hi };
}

export function formatHeatValue(v, mode) {
  if (!Number.isFinite(v)) return '—';
  const s = Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(mode.digits);
  return mode.unit ? `${s} ${mode.unit}` : s;
}

/**
 * naca.js — NACA 4- and 5-digit airfoil definition.
 *
 * Parsing and the analytic surface equations, and nothing else. The panel
 * generator in geometry.js consumes `surfacePoint`; the rest of the solver never
 * needs to know which family it is looking at.
 */

// 5-digit standard camber table, indexed by the second digit P (the camber
// position digit). r is the transition station of the cubic/linear camber line
// and k1 sets its magnitude, both calibrated for a design lift coefficient of
// 0.3 (i.e. first digit L = 2).
const FIVE_DIGIT_TABLE = {
  1: { r: 0.058, k1: 361.4 },
  2: { r: 0.126, k1: 51.64 },
  3: { r: 0.2025, k1: 15.957 },
  4: { r: 0.29, k1: 6.643 },
  5: { r: 0.391, k1: 3.23 },
};

/**
 * Parse a 4- or 5-digit NACA designation into a geometry spec.
 * Returns { ok, error?, warning?, ...spec }.
 */
export function parseNacaCode(raw) {
  const cleaned = String(raw ?? '')
    .trim()
    .replace(/^naca[\s-]*/i, '')
    .replace(/\s+/g, '');

  if (cleaned.length === 0) {
    return { ok: false, error: 'Enter a 4- or 5-digit NACA code (e.g. 2412).' };
  }
  if (!/^\d+$/.test(cleaned)) {
    return { ok: false, error: 'Only digits are allowed (an optional "NACA" prefix is fine).' };
  }
  if (cleaned.length !== 4 && cleaned.length !== 5) {
    return {
      ok: false,
      error: `"${cleaned}" has ${cleaned.length} digits — NACA codes must be 4 or 5 (e.g. 2412 or 23012).`,
    };
  }

  const t = Number(cleaned.slice(-2)) / 100;
  if (t <= 0) {
    return {
      ok: false,
      error: 'The last two digits set max thickness (% of chord) and cannot be 00.',
    };
  }

  if (cleaned.length === 4) {
    const m = Number(cleaned[0]) / 100;
    const p = Number(cleaned[1]) / 10;
    if (m > 0 && p <= 0) {
      return {
        ok: false,
        error: 'A cambered 4-digit airfoil needs a non-zero camber position (2nd digit).',
      };
    }
    return { ok: true, key: cleaned, series: 4, label: `NACA ${cleaned}`, m, p, t, symmetric: m === 0 };
  }

  // 5-digit: L P Q XX
  const L = Number(cleaned[0]);
  const P = Number(cleaned[1]);
  const Q = Number(cleaned[2]);
  const row = FIVE_DIGIT_TABLE[P];
  if (!row) {
    return { ok: false, error: 'For a 5-digit code the 2nd digit (camber position) must be 1–5.' };
  }

  let warning;
  if (Q === 1) {
    warning =
      'Reflex camber lines (3rd digit = 1) are not supported — showing the standard (non-reflex) interpretation instead.';
  } else if (Q !== 0) {
    warning =
      'The 3rd digit of a 5-digit code should be 0 or 1 — treating it as 0 (standard camber).';
  }

  return {
    ok: true,
    key: cleaned,
    series: 5,
    label: `NACA ${cleaned}`,
    r: row.r,
    k1: row.k1 * (L / 2), // table is calibrated for Cl_design = 0.3, i.e. L = 2
    clDesign: 0.15 * L,
    t,
    symmetric: L === 0,
    warning,
  };
}

/**
 * Half-thickness distribution, shared by both series.
 *
 * The leading term goes as sqrt(x), so dy/dx is infinite at the nose. That is
 * the real shape (a NACA section has a rounded nose of finite radius), but it is
 * also why panel nodes must be clustered there and why the spline in geometry.js
 * is parameterised by arc length rather than by x.
 */
export function thickness(x, t) {
  const s = Math.sqrt(Math.max(x, 0));
  return (
    5 * t * (0.2969 * s - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x)
  );
}

/** Camber ordinate and slope at x. Returns [y_c, dy_c/dx]. */
export function camber(x, spec) {
  if (spec.symmetric) return [0, 0];

  if (spec.series === 4) {
    const { m, p } = spec;
    if (x < p) {
      return [(m / (p * p)) * (2 * p * x - x * x), ((2 * m) / (p * p)) * (p - x)];
    }
    const q = (1 - p) * (1 - p);
    return [(m / q) * (1 - 2 * p + 2 * p * x - x * x), ((2 * m) / q) * (p - x)];
  }

  // 5-digit, standard (non-reflex) camber: cubic ahead of r, straight aft.
  const { r, k1 } = spec;
  if (x < r) {
    return [
      (k1 / 6) * (x * x * x - 3 * r * x * x + r * r * (3 - r) * x),
      (k1 / 6) * (3 * x * x - 6 * r * x + r * r * (3 - r)),
    ];
  }
  return [((k1 * r * r * r) / 6) * (1 - x), -(k1 * r * r * r) / 6];
}

/**
 * A point on the airfoil surface at chordwise parameter x.
 * `side` is +1 for the upper surface, -1 for the lower. Returns [x, y].
 *
 * Note that the returned x is not the argument: the surface is offset from the
 * camber line along its *normal*, so the thickness rotates with the camber
 * slope. This is the exact NACA construction, not the small-slope shortcut.
 */
export function surfacePoint(x, side, spec) {
  const yt = thickness(x, spec.t);
  const [yc, dyc] = camber(x, spec);
  const th = Math.atan(dyc);
  return [x - side * yt * Math.sin(th), yc + side * yt * Math.cos(th)];
}

/**
 * Zero-lift angle of attack in degrees from thin-airfoil theory,
 *   alpha_L0 = -(1/pi) integral_0^pi (dy_c/dx)(cos(theta) - 1) d(theta)
 * with x = (1 - cos theta)/2.
 *
 * Exact for the camber line and independent of the panel solve, so it doubles as
 * a validation reference for the panel method's own zero-lift angle.
 */
export function zeroLiftAngle(spec) {
  if (spec.symmetric) return 0;
  const M = 800; // Simpson panels in theta
  let sum = 0;
  for (let k = 0; k <= M; k++) {
    const theta = (Math.PI * k) / M;
    const x = 0.5 * (1 - Math.cos(theta));
    const g = camber(x, spec)[1] * (Math.cos(theta) - 1);
    const w = k === 0 || k === M ? 1 : k % 2 ? 4 : 2;
    sum += w * g;
  }
  const integral = (Math.PI / M / 3) * sum;
  return (-(integral / Math.PI) * 180) / Math.PI;
}

/** Largest camber ordinate as a fraction of chord (works for both series). */
export function maxCamber(spec) {
  if (spec.symmetric) return 0;
  let best = 0;
  for (let k = 0; k <= 400; k++) {
    const yc = camber(k / 400, spec)[0];
    if (Math.abs(yc) > Math.abs(best)) best = yc;
  }
  return best;
}

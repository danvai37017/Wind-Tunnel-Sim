/**
 * linalg.js — dense linear algebra for the panel solver.
 *
 * The influence matrix is dense and depends only on geometry, so the whole
 * performance story of this solver rests on factoring it once and reusing the
 * factors. Everything here is written around that: `luFactor` keeps its
 * multipliers, `luSolve` is a pure back-substitution, and nothing ever rebuilds
 * a factorisation it could have kept.
 *
 * GMRES is the fallback for the large-N end of the panel range, where an O(N^3)
 * factorisation stops being cheap enough to hide inside a geometry change.
 *
 * Pure numerics: no aerodynamics, no allocation in the hot paths beyond what the
 * caller passes in.
 */

/* ============================================================================
 * LU decomposition — Doolittle with partial pivoting
 * ==========================================================================*/

/**
 * In-place LU decomposition with partial pivoting.
 *
 * Gaussian elimination with the multipliers kept rather than discarded, so the
 * factors can be reused. Returns the pivot permutation, or null if the matrix is
 * singular to working precision.
 *
 * `A` is row-major n x n and is overwritten with L (unit diagonal, below) and U
 * (on and above the diagonal).
 */
export function luFactor(A, n) {
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
      const ri = i * n;
      const rk = k * n;
      for (let j = k + 1; j < n; j++) A[ri + j] -= m * A[rk + j];
    }
  }
  return piv;
}

/**
 * Solve LU x = P b into `out`. O(n^2): this is the operation an angle-of-attack
 * change costs, and the reason a slider drag is free.
 */
export function luSolve(LU, piv, n, b, out) {
  for (let i = 0; i < n; i++) {
    let s = b[piv[i]];
    const ri = i * n;
    for (let j = 0; j < i; j++) s -= LU[ri + j] * out[j];
    out[i] = s;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = out[i];
    const ri = i * n;
    for (let j = i + 1; j < n; j++) s -= LU[ri + j] * out[j];
    out[i] = s / LU[ri + i];
  }
  return out;
}

/**
 * 1-norm condition number estimate via Hager's algorithm on the factored matrix.
 *
 * Cheap (a handful of back-substitutions) and good enough to tell "this geometry
 * produced a healthy system" from "this one is one rounding error from garbage",
 * which is what the confidence score wants to know.
 */
export function luConditionEstimate(LU, piv, n, anorm) {
  const x = new Float64Array(n).fill(1 / n);
  const y = new Float64Array(n);
  let est = 0;

  for (let iter = 0; iter < 5; iter++) {
    luSolve(LU, piv, n, x, y);
    est = 0;
    for (let i = 0; i < n; i++) est += Math.abs(y[i]);
    for (let i = 0; i < n; i++) x[i] = y[i] >= 0 ? 1 : -1;
    luSolve(LU, piv, n, x, y);
    let jmax = 0;
    for (let i = 1; i < n; i++) if (Math.abs(y[i]) > Math.abs(y[jmax])) jmax = i;
    x.fill(0);
    x[jmax] = 1;
  }
  return est * anorm;
}

/** Row-major matrix-vector product, y = A x. */
export function matVec(A, n, m, x, y) {
  for (let i = 0; i < n; i++) {
    let s = 0;
    const r = i * m;
    for (let j = 0; j < m; j++) s += A[r + j] * x[j];
    y[i] = s;
  }
  return y;
}

/* ============================================================================
 * GMRES(m) — iterative fallback for large systems
 * ==========================================================================*/

/**
 * Restarted GMRES with modified Gram-Schmidt and Givens rotations.
 *
 * Used when the panel count makes an O(N^3) factorisation too expensive to hide
 * inside a geometry change. The panel matrix is diagonally dominant-ish (the
 * self-influence is the largest entry in each row) so a Jacobi preconditioner is
 * enough; it converges in a few tens of iterations at N ~ 1000.
 *
 * `apply(x, out)` computes out = A x. Returns { x, iterations, residual, ok }.
 */
export function gmres(apply, b, n, opts = {}) {
  const restart = Math.min(opts.restart ?? 60, n);
  const maxOuter = opts.maxOuter ?? 20;
  const tol = opts.tol ?? 1e-10;
  const diag = opts.diagonal ?? null; // Jacobi preconditioner, entries of A

  const x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
  const r = new Float64Array(n);
  const w = new Float64Array(n);
  const V = new Float64Array((restart + 1) * n);
  const H = new Float64Array((restart + 1) * restart);
  const cs = new Float64Array(restart);
  const sn = new Float64Array(restart);
  const g = new Float64Array(restart + 1);
  const y = new Float64Array(restart);

  const precond = (v) => {
    if (!diag) return;
    for (let i = 0; i < n; i++) {
      const d = diag[i];
      if (Math.abs(d) > 1e-300) v[i] /= d;
    }
  };

  let bnorm = 0;
  for (let i = 0; i < n; i++) bnorm += b[i] * b[i];
  bnorm = Math.sqrt(bnorm);
  if (bnorm === 0) return { x, iterations: 0, residual: 0, ok: true };

  let total = 0;
  for (let outer = 0; outer < maxOuter; outer++) {
    apply(x, r);
    for (let i = 0; i < n; i++) r[i] = b[i] - r[i];
    precond(r);

    let beta = 0;
    for (let i = 0; i < n; i++) beta += r[i] * r[i];
    beta = Math.sqrt(beta);
    if (beta / bnorm < tol) return { x, iterations: total, residual: beta / bnorm, ok: true };

    for (let i = 0; i < n; i++) V[i] = r[i] / beta;
    g.fill(0);
    g[0] = beta;

    let k = 0;
    for (; k < restart; k++) {
      total++;
      apply(V.subarray(k * n, (k + 1) * n), w);
      precond(w);

      // Modified Gram-Schmidt against the existing Krylov basis.
      for (let i = 0; i <= k; i++) {
        let h = 0;
        const off = i * n;
        for (let j = 0; j < n; j++) h += w[j] * V[off + j];
        H[i * restart + k] = h;
        for (let j = 0; j < n; j++) w[j] -= h * V[off + j];
      }
      let hkk = 0;
      for (let j = 0; j < n; j++) hkk += w[j] * w[j];
      hkk = Math.sqrt(hkk);
      H[(k + 1) * restart + k] = hkk;
      if (hkk > 1e-300) {
        const off = (k + 1) * n;
        for (let j = 0; j < n; j++) V[off + j] = w[j] / hkk;
      }

      // Apply the accumulated Givens rotations, then form the new one.
      for (let i = 0; i < k; i++) {
        const t = H[i * restart + k];
        H[i * restart + k] = cs[i] * t + sn[i] * H[(i + 1) * restart + k];
        H[(i + 1) * restart + k] = -sn[i] * t + cs[i] * H[(i + 1) * restart + k];
      }
      const hk = H[k * restart + k];
      const hk1 = H[(k + 1) * restart + k];
      const den = Math.hypot(hk, hk1) || 1;
      cs[k] = hk / den;
      sn[k] = hk1 / den;
      H[k * restart + k] = cs[k] * hk + sn[k] * hk1;
      H[(k + 1) * restart + k] = 0;
      g[k + 1] = -sn[k] * g[k];
      g[k] = cs[k] * g[k];

      if (Math.abs(g[k + 1]) / bnorm < tol) {
        k++;
        break;
      }
    }

    // Back-substitute the small triangular least-squares problem and update x.
    for (let i = k - 1; i >= 0; i--) {
      let s = g[i];
      for (let j = i + 1; j < k; j++) s -= H[i * restart + j] * y[j];
      y[i] = s / (H[i * restart + i] || 1e-300);
    }
    for (let i = 0; i < k; i++) {
      const off = i * n;
      const yi = y[i];
      for (let j = 0; j < n; j++) x[j] += yi * V[off + j];
    }

    if (Math.abs(g[k]) / bnorm < tol) {
      return { x, iterations: total, residual: Math.abs(g[k]) / bnorm, ok: true };
    }
  }

  apply(x, r);
  for (let i = 0; i < n; i++) r[i] = b[i] - r[i];
  let res = 0;
  for (let i = 0; i < n; i++) res += r[i] * r[i];
  res = Math.sqrt(res) / bnorm;
  return { x, iterations: total, residual: res, ok: res < 1e-6 };
}

/* ============================================================================
 * Fixed-point acceleration
 * ==========================================================================*/

/**
 * Vector Aitken Delta-squared extrapolation.
 *
 * The viscous-inviscid loop is a fixed-point iteration x_{k+1} = G(x_k) whose
 * convergence rate degrades badly as separation approaches — the dominant
 * eigenvalue of G' climbs toward 1 and the iteration crawls. Aitken estimates
 * that eigenvalue from the last three iterates and jumps to where the geometric
 * series would land:
 *
 *   mu = <dx1, dx2 - dx1> / |dx2 - dx1|^2
 *   x* = x2 - mu * dx2         with dx1 = x1 - x0, dx2 = x2 - x1
 *
 * The vector form uses a single scalar mu (Irons & Tuck), which is the standard
 * choice when the iteration is dominated by one mode — which this one is, the
 * mode being the displacement thickness near separation.
 *
 * `mu` is clamped: an unclamped extrapolation can overshoot into a state the
 * boundary layer cannot integrate at all, and one bad step costs more than the
 * acceleration saves.
 */
export function aitkenFactor(x0, x1, x2, n) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const d1 = x1[i] - x0[i];
    const d2 = x2[i] - x1[i];
    const dd = d2 - d1;
    num += d1 * dd;
    den += dd * dd;
  }
  if (!(den > 1e-300)) return 0;
  const mu = num / den;
  if (!isFinite(mu)) return 0;
  // mu in (0, 1) is the geometric-series estimate; outside that range the three
  // iterates are not on a single decaying mode and extrapolation is guesswork.
  return Math.min(0.9, Math.max(-0.9, mu));
}

/** max |a - b| over n entries. */
export function maxDiff(a, b, n) {
  let m = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

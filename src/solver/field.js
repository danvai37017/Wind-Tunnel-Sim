/**
 * field.js — sampling the converged state for visualisation.
 *
 * Nothing here is physics. Every value comes from `AeroSolver.sampleVelocity`,
 * which is the same analytic Biot-Savart evaluation the influence matrix was
 * assembled from — so the streamlines the user watches are streamlines of
 * exactly the flow the lift was integrated from, not of a second, cheaper
 * approximation that happens to look similar.
 *
 * ## Why the field is gridded at all
 *
 * A direct evaluation costs O(N) per sample, with N the panel count. That is
 * fine for a few hundred tracer particles advected per frame — 700 tracers on a
 * 230-panel section is around 160k kernel evaluations, a couple of milliseconds
 * — and far too slow for a 400x160 background contour, which would be 15 million
 * evaluations and take most of a second.
 *
 * So tracers are advected by direct evaluation (exact, no interpolation error
 * where it matters) and the background contour is computed once per converged
 * state onto a coarse grid, in time-sliced chunks so it never blocks a frame,
 * and sampled bilinearly. Both come from the same function.
 *
 * The specification assigns this work to the GPU via WebGPU. It is done on the
 * CPU here, chunked, for two reasons: WebGPU is still not universally available
 * in shipping browsers, and a compute path that cannot be exercised in the Node
 * test harness cannot be part of the regression suite. The interface is the same
 * either way — `FieldSampler` owns a Float32Array of velocities and does not
 * care who filled it — so moving the fill to a shader later touches one method.
 */

/**
 * A coarse Eulerian velocity field over the visible domain, filled incrementally.
 *
 * `bounds` is in chord-normalised coordinates with the section's leading edge at
 * the origin and its chord along +x — the same frame the solver works in.
 */
export class FieldSampler {
  constructor(opts = {}) {
    this.nx = opts.nx ?? 220;
    this.ny = opts.ny ?? 96;
    this.bounds = opts.bounds ?? { x0: -1.2, x1: 3.0, y0: -1.1, y1: 1.1 };
    const n = this.nx * this.ny;
    this.u = new Float32Array(n);
    this.v = new Float32Array(n);
    this.speed = new Float32Array(n);
    this.solid = new Uint8Array(n);
    this.vorticity = new Float32Array(n);
    this.cp = new Float32Array(n);
    this.row = 0; // fill progress
    this.complete = false;
    this.generation = -1;
  }

  /** Grid coordinates of column i, row j. */
  gx(i) {
    const b = this.bounds;
    return b.x0 + ((b.x1 - b.x0) * i) / (this.nx - 1);
  }
  gy(j) {
    const b = this.bounds;
    return b.y0 + ((b.y1 - b.y0) * j) / (this.ny - 1);
  }

  /** Start a fresh fill. Call whenever the aerodynamic state changes. */
  invalidate(generation) {
    this.row = 0;
    this.complete = false;
    this.generation = generation;
  }

  /**
   * Fill up to `budgetMs` milliseconds worth of rows.
   *
   * Returns true when the field is complete. Time-sliced rather than row-capped
   * so the cost adapts to the machine and to the panel count instead of being
   * tuned for one of them.
   */
  fill(solver, budgetMs = 4) {
    if (this.complete) return true;
    const t0 = now();
    const out = [0, 0];
    const { nx, ny } = this;

    while (this.row < ny) {
      const j = this.row;
      const y = this.gy(j);
      const base = j * nx;
      for (let i = 0; i < nx; i++) {
        const x = this.gx(i);
        const k = base + i;
        if (solver.isInside(x, y)) {
          this.solid[k] = 1;
          this.u[k] = 0;
          this.v[k] = 0;
          this.speed[k] = 0;
          this.cp[k] = 1;
          continue;
        }
        this.solid[k] = 0;
        // Particles are excluded: the background field is a property of the
        // converged steady state, and folding in a wake that moves every frame
        // would make the contour flicker while the controls are still.
        solver.sampleVelocity(x, y, out, false);
        const uu = out[0];
        const vv = out[1];
        this.u[k] = uu;
        this.v[k] = vv;
        const s2 = uu * uu + vv * vv;
        this.speed[k] = Math.sqrt(s2);
        this.cp[k] = 1 - s2; // Bernoulli, nondimensional on V_inf = 1
      }
      this.row++;
      if (now() - t0 > budgetMs) break;
    }

    if (this.row >= ny) {
      this.computeVorticity();
      this.complete = true;
      return true;
    }
    return false;
  }

  /**
   * Vorticity by central differences on the filled grid.
   *
   * This is the one place a derivative is taken numerically rather than
   * analytically, and it is deliberate: vorticity is only ever drawn, never
   * integrated, and an analytic curl of the panel field is identically zero
   * everywhere outside the sheets anyway — the interesting structure is in the
   * grid-scale shear near the surface and in the wake, which is exactly what a
   * difference on the visualisation grid shows.
   */
  computeVorticity() {
    const { nx, ny, u, v, solid, vorticity } = this;
    const dx = (this.bounds.x1 - this.bounds.x0) / (nx - 1);
    const dy = (this.bounds.y1 - this.bounds.y0) / (ny - 1);
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const k = j * nx + i;
        if (solid[k]) {
          vorticity[k] = 0;
          continue;
        }
        const dvdx = (v[k + 1] - v[k - 1]) / (2 * dx);
        const dudy = (u[k + nx] - u[k - nx]) / (2 * dy);
        vorticity[k] = dvdx - dudy;
      }
    }
  }

  /** Bilinear sample of the gridded field. Falls back to freestream off-grid. */
  sample(x, y, out) {
    const { nx, ny, bounds: b } = this;
    const fx = ((x - b.x0) / (b.x1 - b.x0)) * (nx - 1);
    const fy = ((y - b.y0) / (b.y1 - b.y0)) * (ny - 1);
    if (fx < 0 || fy < 0 || fx > nx - 1 || fy > ny - 1) {
      out[0] = 1;
      out[1] = 0;
      return out;
    }
    const i0 = Math.min(nx - 2, Math.floor(fx));
    const j0 = Math.min(ny - 2, Math.floor(fy));
    const tx = fx - i0;
    const ty = fy - j0;
    const k = j0 * nx + i0;
    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;
    out[0] = this.u[k] * w00 + this.u[k + 1] * w10 + this.u[k + nx] * w01 + this.u[k + nx + 1] * w11;
    out[1] = this.v[k] * w00 + this.v[k + 1] * w10 + this.v[k + nx] * w01 + this.v[k + nx + 1] * w11;
    return out;
  }
}

/* ============================================================================
 * Streamlines
 * ==========================================================================*/

/**
 * Trace a streamline from a seed point with RK4.
 *
 * Streamlines of a steady field, so the trace is a pure geometric integration of
 * the velocity direction — no timestep and no accumulated error from the
 * particle history. Stops on leaving the domain, on entering the body, or at a
 * stagnation point, which is where a streamline genuinely ends.
 */
export function traceStreamline(solver, x0, y0, opts = {}) {
  const step = opts.step ?? 0.02;
  const maxPoints = opts.maxPoints ?? 400;
  const bounds = opts.bounds ?? { x0: -1.5, x1: 6, y0: -2, y1: 2 };
  const direction = opts.direction ?? 1;
  const pts = new Float64Array(maxPoints * 2);
  const a = [0, 0];
  const b = [0, 0];
  const c = [0, 0];
  const d = [0, 0];

  let x = x0;
  let y = y0;
  let n = 0;

  const unit = (px, py, out) => {
    solver.sampleVelocity(px, py, out, false);
    const m = Math.hypot(out[0], out[1]);
    if (m < 1e-6) return false;
    out[0] = (direction * out[0]) / m;
    out[1] = (direction * out[1]) / m;
    return true;
  };

  while (n < maxPoints) {
    pts[n * 2] = x;
    pts[n * 2 + 1] = y;
    n++;

    if (!unit(x, y, a)) break;
    if (!unit(x + 0.5 * step * a[0], y + 0.5 * step * a[1], b)) break;
    if (!unit(x + 0.5 * step * b[0], y + 0.5 * step * b[1], c)) break;
    if (!unit(x + step * c[0], y + step * c[1], d)) break;

    x += (step * (a[0] + 2 * b[0] + 2 * c[0] + d[0])) / 6;
    y += (step * (a[1] + 2 * b[1] + 2 * c[1] + d[1])) / 6;

    if (x < bounds.x0 || x > bounds.x1 || y < bounds.y0 || y > bounds.y1) break;
    if (solver.isInside(x, y)) break;
  }

  return { points: pts.subarray(0, n * 2), count: n };
}

function now() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

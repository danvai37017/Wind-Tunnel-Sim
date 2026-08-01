/**
 * flow.js — the bridge between the solver and the viewport.
 *
 * The solver works in its own natural frame: unit chord along +x with the
 * leading edge at the origin, the section never rotated, and incidence carried
 * by the freestream direction. That is what keeps the influence matrix reusable
 * across angle of attack, and it is not the frame anyone wants to look at.
 *
 * The viewport wants the opposite: the freestream horizontal and the section
 * tilted, on a fixed pixel grid representing a fixed physical working section.
 *
 * So this module owns exactly one thing — the transform between them — plus the
 * sampling of the analytic velocity field onto a grid the renderer can read
 * cheaply. It computes no aerodynamics of its own. Every velocity it returns
 * came from `AeroSolver.sampleVelocity`, which is the same Biot-Savart
 * evaluation the influence matrix was built from.
 *
 * ## The transform
 *
 * Writing R(t) for a rotation and p for a position measured from the quarter
 * chord, the two frames are related by
 *
 *   p_solver = R(alpha) p_view + (1/4, 0)
 *   v_view   = R(-alpha) v_solver
 *
 * Positions rotate one way and velocities the other, which is what puts the
 * solver's freestream — the vector (cos alpha, sin alpha) — along +x in the
 * view, and simultaneously tilts the section by -alpha about its quarter chord.
 * One angle, applied consistently, replaces the old code's separate geometry
 * rotation and lattice placement.
 *
 * ## Why the field is gridded
 *
 * Evaluating the analytic field costs about two microseconds per sample with the
 * far-field approximation enabled. That is fine for a few thousand samples and
 * far too slow for the ~17,000 a background contour needs every time a control
 * moves. So the field is sampled onto a grid in time-sliced chunks that never
 * exceed a few milliseconds, and both the contour and the tracer particles read
 * it by bilinear interpolation. The previous field stays on screen until the new
 * one is complete, so a slider drag never flashes.
 */

import { AeroSolver } from './solver/index.js';

/* ============================================================================
 * Viewport geometry
 * ==========================================================================*/

// The working section is a fixed physical box; the model scales inside it, so
// the cell size is the constant and the chord in cells is what varies.
export const NX = 420;
export const NY = 160;
export const NXNY = NX * NY;

export const CHORD_MIN_CM = 2.0;
export const CHORD_MAX_CM = 10.0;
export const CHORD_DEFAULT_CM = 6.0;
const CELLS_AT_MAX_CHORD = 144;

/** Metres per cell. Sized so the largest model spans 144 cells. */
export const DX = CHORD_MAX_CM / 100 / CELLS_AT_MAX_CHORD;

/** Leading edge sits this far from the inlet at zero incidence. */
const LE_X = 100;
export const PIVOT_Y = NY / 2;

export const cellsForChord = (chordM) => Math.round(chordM / DX);
export const pivotXFor = (nCells) => LE_X + 0.25 * nCells;

/**
 * Velocity-field grid. Coarser than the lattice because it is resampled anyway,
 * and every point on it costs an analytic evaluation.
 */
const FIELD_NX = 212;
const FIELD_NY = 82;

/* ============================================================================
 * Flow
 * ==========================================================================*/

export function createFlow(options = {}) {
  return {
    solver: new AeroSolver(options),

    // Lattice-resolution mask, used for rendering and for keeping tracers out of
    // the body. Cheap enough (a point-in-polygon test) to do at full resolution.
    solid: new Uint8Array(NXNY),

    // Velocity field on the coarse grid, normalised on the freestream.
    fu: new Float32Array(FIELD_NX * FIELD_NY),
    fv: new Float32Array(FIELD_NX * FIELD_NY),
    fuNext: new Float32Array(FIELD_NX * FIELD_NY),
    fvNext: new Float32Array(FIELD_NX * FIELD_NY),
    fillRow: 0,
    fieldReady: false,
    generation: -1,

    // Section outline in view coordinates, closed, for the renderers.
    poly: new Float64Array(0),
    polyCount: 0,
    nCells: 0,
    pivotX: 0,
    cosA: 1,
    sinA: 0,
    alphaRad: 0,

    state: null,
    error: null,
  };
}

/* ---- Coordinate transforms ---------------------------------------------- */

/** View cell -> solver chord coordinates. */
export function viewToSolver(F, X, Y, out) {
  const cx = (X - F.pivotX) / F.nCells;
  const cy = (Y - PIVOT_Y) / F.nCells;
  out[0] = cx * F.cosA - cy * F.sinA + 0.25;
  out[1] = cx * F.sinA + cy * F.cosA;
  return out;
}

/** Solver chord coordinates -> view cell. */
export function solverToView(F, px, py, out) {
  const dx = px - 0.25;
  const cx = dx * F.cosA + py * F.sinA;
  const cy = -dx * F.sinA + py * F.cosA;
  out[0] = F.pivotX + cx * F.nCells;
  out[1] = PIVOT_Y + cy * F.nCells;
  return out;
}

/* ---- State update -------------------------------------------------------- */

const P2 = [0, 0];

/**
 * Push new controls into the solver and rebuild whatever the viewport needs.
 *
 * Returns the authoritative aerodynamic state, or null if the geometry could not
 * be built (in which case `F.error` explains why and the previous state stays on
 * screen).
 */
export function updateFlow(F, inputs, opts) {
  const chordM = inputs.chord;
  const nCells = cellsForChord(chordM);
  const state = F.solver.update(inputs, opts);
  F.error = F.solver.buildError;
  if (!state) return null;

  F.state = state;
  F.nCells = nCells;
  F.pivotX = pivotXFor(nCells);
  F.alphaRad = (inputs.alphaDeg * Math.PI) / 180;
  F.cosA = Math.cos(F.alphaRad);
  F.sinA = Math.sin(F.alphaRad);

  rebuildOutline(F);
  rasterise(F);

  if (F.generation !== F.solver.generation) {
    F.generation = F.solver.generation;
    F.fillRow = 0; // start a new field fill; keep showing the old one meanwhile
  }
  return state;
}

/** Section outline in view coordinates, straight from the panel nodes. */
function rebuildOutline(F) {
  const geo = F.solver.geometry;
  if (!geo) return;
  const n = geo.n;
  if (F.polyCount !== n) {
    F.poly = new Float64Array(n * 2);
    F.polyCount = n;
  }
  for (let i = 0; i < n; i++) {
    solverToView(F, geo.X[i], geo.Y[i], P2);
    F.poly[2 * i] = P2[0];
    F.poly[2 * i + 1] = P2[1];
  }
}

/**
 * Rasterise the outline to the lattice mask.
 *
 * Interior fill by ray casting, plus the outline stamped so a very thin section
 * still produces a connected body rather than a dotted line.
 */
function rasterise(F) {
  const { poly, polyCount, solid } = F;
  solid.fill(0);
  if (polyCount < 3) return;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < polyCount; i++) {
    const x = poly[2 * i];
    const y = poly[2 * i + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const x0 = Math.max(1, Math.floor(minX) - 1);
  const x1 = Math.min(NX - 2, Math.ceil(maxX) + 1);
  const y0 = Math.max(1, Math.floor(minY) - 1);
  const y1 = Math.min(NY - 2, Math.ceil(maxY) + 1);

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let inside = false;
      for (let i = 0, j = polyCount - 1; i < polyCount; j = i++) {
        const yi = poly[2 * i + 1];
        const yj = poly[2 * j + 1];
        if (yi > y !== yj > y) {
          const xi = poly[2 * i];
          const xj = poly[2 * j];
          if (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
        }
      }
      if (inside) solid[y * NX + x] = 1;
    }
  }

  for (let i = 0; i < polyCount; i++) {
    const j = (i + 1) % polyCount;
    const ax = poly[2 * i];
    const ay = poly[2 * i + 1];
    const bx = poly[2 * j];
    const by = poly[2 * j + 1];
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) * 2));
    for (let s = 0; s <= steps; s++) {
      const f = s / steps;
      const gx = Math.round(ax + (bx - ax) * f);
      const gy = Math.round(ay + (by - ay) * f);
      if (gx >= 1 && gx <= NX - 2 && gy >= 1 && gy <= NY - 2) solid[gy * NX + gx] = 1;
    }
  }
}

/* ---- Field fill ---------------------------------------------------------- */

const FIELD_X0 = 0;
const FIELD_X1 = NX;
const FIELD_Y0 = 0;
const FIELD_Y1 = NY;

const fieldCellX = (i) => FIELD_X0 + ((FIELD_X1 - FIELD_X0) * i) / (FIELD_NX - 1);
const fieldCellY = (j) => FIELD_Y0 + ((FIELD_Y1 - FIELD_Y0) * j) / (FIELD_NY - 1);

/**
 * Advance the field fill by at most `budgetMs`. Returns true when complete.
 *
 * The new field is built in a back buffer and swapped in whole, so the viewport
 * never shows a half-updated flow with a visible seam across it.
 */
export function fillField(F, budgetMs = 4) {
  if (F.fillRow >= FIELD_NY) return true;
  const t0 = performance.now();
  const out = [0, 0];
  const solverPt = [0, 0];

  while (F.fillRow < FIELD_NY) {
    const j = F.fillRow;
    const Y = fieldCellY(j);
    const base = j * FIELD_NX;
    for (let i = 0; i < FIELD_NX; i++) {
      const X = fieldCellX(i);
      const k = base + i;
      viewToSolver(F, X, Y, solverPt);
      if (F.solver.isInside(solverPt[0], solverPt[1])) {
        F.fuNext[k] = 0;
        F.fvNext[k] = 0;
        continue;
      }
      // Particles are deliberately excluded: the gridded field is a property of
      // the converged steady state, and folding in a wake that moves every frame
      // would make the background shimmer while the controls are still.
      F.solver.sampleVelocity(solverPt[0], solverPt[1], out, false);
      // Rotate the velocity back into view axes.
      F.fuNext[k] = out[0] * F.cosA + out[1] * F.sinA;
      F.fvNext[k] = -out[0] * F.sinA + out[1] * F.cosA;
    }
    F.fillRow++;
    if (performance.now() - t0 > budgetMs) break;
  }

  if (F.fillRow >= FIELD_NY) {
    const u = F.fu;
    const v = F.fv;
    F.fu = F.fuNext;
    F.fv = F.fvNext;
    F.fuNext = u;
    F.fvNext = v;
    F.fieldReady = true;
    return true;
  }
  return false;
}

/**
 * Bilinear velocity sample in view coordinates, normalised on the freestream.
 *
 * This is what the tracers and the contour both read. Off-grid it returns the
 * freestream, which is what the flow is doing out there anyway.
 */
export function sampleFlow(F, X, Y, out) {
  if (!F.fieldReady) {
    out[0] = 1;
    out[1] = 0;
    return out;
  }
  const fx = ((X - FIELD_X0) / (FIELD_X1 - FIELD_X0)) * (FIELD_NX - 1);
  const fy = ((Y - FIELD_Y0) / (FIELD_Y1 - FIELD_Y0)) * (FIELD_NY - 1);
  if (!(fx >= 0 && fy >= 0 && fx <= FIELD_NX - 1 && fy <= FIELD_NY - 1)) {
    out[0] = 1;
    out[1] = 0;
    return out;
  }
  const i0 = Math.min(FIELD_NX - 2, fx | 0);
  const j0 = Math.min(FIELD_NY - 2, fy | 0);
  const tx = fx - i0;
  const ty = fy - j0;
  const k = j0 * FIELD_NX + i0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;
  out[0] = F.fu[k] * w00 + F.fu[k + 1] * w10 + F.fu[k + FIELD_NX] * w01 + F.fu[k + FIELD_NX + 1] * w11;
  out[1] = F.fv[k] * w00 + F.fv[k + 1] * w10 + F.fv[k + FIELD_NX] * w01 + F.fv[k + FIELD_NX + 1] * w11;
  return out;
}

/** Field-grid dimensions, for the contour renderer. */
export const FIELD_DIMS = { nx: FIELD_NX, ny: FIELD_NY, cellX: fieldCellX, cellY: fieldCellY };

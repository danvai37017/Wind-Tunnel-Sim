/**
 * WindTunnel — interactive 2D wind tunnel over a NACA section.
 *
 * There is one aerodynamic model here and everything on screen comes from it.
 * The lift, the drag, the pressure distribution, the transition and separation
 * markers, the stall banner, the streamlines and the extruded 3D view are all
 * read off a single converged state produced by `src/solver` — a viscous-
 * inviscid coupled panel method with an integral boundary layer. If two readouts
 * ever disagreed it would be a bug, not a difference of opinion between models.
 *
 * That is a deliberate change from the previous version, which ran a lattice-
 * Boltzmann solve for the picture and a separate steady panel method for the
 * numbers. The picture was honest about the flow it computed, but that flow was
 * at a Reynolds number about a thousand times below the one the dashboard
 * quoted, so the two disagreed about where the flow separated and there was no
 * way to reconcile them. Now the streamlines are streamlines of the same field
 * the forces were integrated from.
 *
 * This file is presentation only: transforms, canvases, tracers and controls.
 * No aerodynamics.
 *
 * Self-contained: no required props, no external state. Drop in as
 * <WindTunnel /> and it works.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './WindTunnel.module.css';
import {
  NX,
  NY,
  DX,
  PIVOT_Y,
  CHORD_MIN_CM,
  CHORD_MAX_CM,
  CHORD_DEFAULT_CM,
  createFlow,
  updateFlow,
  fillField,
  sampleFlow,
} from './flow.js';
import { parseNacaCode } from './solver/naca.js';
import { GEOMETRIES, DEFAULT_GEOMETRY } from './solver/sections.js';
import { RHO_AIR, NU_AIR } from './solver/index.js';
import {
  COLOR_SCALES,
  HEAT_MODES,
  DEFAULT_MODE,
  DEFAULT_SCALE,
  getMode,
  getScale,
  heatDomain,
  rampColor,
  rampCss,
  formatHeatValue,
} from './viz/heatmap.js';

/* ============================================================================
 * 1. Configuration
 * ==========================================================================*/

// Control ranges.
const V_MIN = 5;
const V_MAX = 90;
const AOA_MIN = -15;
const AOA_MAX = 25;

// The 2D solve gives force per metre of span; the section is extruded to a span
// of SPAN_MIN..SPAN_MAX chords so the reported forces are whole-wing Newtons.
// Span has no effect on the two-dimensional physics and never re-runs the solver.
const SPAN_MIN = 1;
const SPAN_MAX = 10;
const SPAN_DEFAULT = 4;

const cmToM = (cm) => cm / 100;

// Rendering.
const SCALE = 2; // canvas pixels per cell
const CANVAS_W = NX * SCALE;
const CANVAS_H = NY * SCALE;

// --- 3D viewport ----------------------------------------------------------
const VIEW3D_W = 840;
const VIEW3D_H = 430;
const WING_OUTLINE_STEP = 4; // sample every Nth outline point for the mesh
const CAM_MIN_EL = -1.45;
const CAM_MAX_EL = 1.45;
const CAM_MIN_ZOOM = 0.35;
const CAM_MAX_ZOOM = 4;
const SPHERE_HEAD_CELLS = 0.9;
const STREAK_3D_GAIN = 1.15;
const STREAK_3D_TAIL_W = 0.75;
const SPHERE_MAX_PX = 4.5;
const RIBBON_MAX_HALF_PX = 3.5;

// Streak ribbons: tracer particles advected by the solved field and drawn as
// tapered trails.
const STREAK_COUNT = 700;
const STREAK_TRAIL = 16; // stored points per ribbon (upper bound)
const STREAK_TRAIL_MIN = 9; // shortest ribbon, so lengths vary and don't band
const STREAK_SEG_CELLS = 1.9; // cells between stored points
const STREAK_HEAD_W = 2.2; // px half-width at the leading end
const STREAK_TAIL_W = 0.12; // px half-width at the tail
const STREAK_MAX_AGE = 12; // seconds before a forced respawn
// Glide rate: cells per second per (m/s) of freestream. Strictly proportional
// to airspeed, so the slider changes glide speed accurately; the constant only
// sets the slow-motion factor.
const STREAK_CELLS_PER_MS = 4.7;
const GOLDEN = 0.6180339887; // low-discrepancy spawn spacing

// Per-frame budget for filling the velocity field. The dashboard and the
// section outline update on the same commit as the slider; the contour catches
// up over the next few frames.
const FIELD_BUDGET_MS = 4;

/* ============================================================================
 * 2. Colour
 * ==========================================================================*/

/**
 * Colour by speed relative to the freestream. Most of the field sits near 1, so
 * the ramp is deliberately not linear: it keeps the freestream a calm cyan and
 * spends the warm half of the hue range on the accelerated flow over the suction
 * side, where the interesting behaviour is. Slow and reversed flow goes deep blue.
 */
function speedColor(f) {
  const h = f < 0.9 ? 200 + 40 * Math.min(1, (0.9 - f) / 0.9) : 200 - 200 * Math.min(1, (f - 0.9) / 1.3);
  return `hsl(${h.toFixed(0)}, 90%, ${(52 + 8 * Math.min(1, f)).toFixed(0)}%)`;
}

/* ============================================================================
 * 3. Streak ribbons
 * ==========================================================================*/

function createStreaks() {
  return {
    x: new Float32Array(STREAK_COUNT),
    y: new Float32Array(STREAK_COUNT),
    tx: new Float32Array(STREAK_COUNT * STREAK_TRAIL),
    ty: new Float32Array(STREAK_COUNT * STREAK_TRAIL),
    head: new Uint8Array(STREAK_COUNT),
    count: new Uint8Array(STREAK_COUNT),
    cap: new Uint8Array(STREAK_COUNT),
    // Spanwise station as a fraction of span in [-0.5, 0.5]. The solve is 2D and
    // uniform along span, so a tracer keeps a fixed z while its (x, y) follows
    // the same field the 2D view uses.
    sz: new Float32Array(STREAK_COUNT),
    since: new Float32Array(STREAK_COUNT),
    age: new Float32Array(STREAK_COUNT),
    phase: 0,
    seeded: false,
  };
}

/**
 * Place one tracer. Spawning resets the whole trail to the new position,
 * otherwise the ribbon would streak across the screen from wherever it died.
 * Inlet spawns are spread with a golden-ratio sequence, which fills the height
 * far more evenly than uniform random and avoids clumps and gaps.
 */
function spawnStreak(St, i, F, seedAnywhere) {
  St.phase = (St.phase + GOLDEN) % 1;
  let x;
  let y;
  let tries = 0;
  do {
    y = 3 + St.phase * (NY - 6);
    x = seedAnywhere ? 1 + Math.random() * (NX - 3) : 0.5 + Math.random() * 3;
    if (tries++) St.phase = (St.phase + GOLDEN) % 1;
  } while (F.solid[(y | 0) * NX + (x | 0)] && tries < 8);

  St.x[i] = x;
  St.y[i] = y;
  St.age[i] = 0;
  // Stagger the first stored point so ribbons don't step in lockstep, which
  // otherwise reads as marching columns.
  St.since[i] = Math.random() * STREAK_SEG_CELLS;
  St.head[i] = 0;
  St.count[i] = 1;
  St.cap[i] = STREAK_TRAIL_MIN + ((Math.random() * (STREAK_TRAIL - STREAK_TRAIL_MIN + 1)) | 0);
  St.sz[i] = Math.random() - 0.5;
  St.tx[i * STREAK_TRAIL] = x;
  St.ty[i * STREAK_TRAIL] = y;
}

function resetStreaks(St, F) {
  for (let i = 0; i < STREAK_COUNT; i++) spawnStreak(St, i, F, true);
  St.seeded = true;
}

const VEL = new Float64Array(2);

/**
 * Advect every tracer by the local velocity for dt seconds of wall clock.
 *
 * The field is normalised on the freestream, so a particle in undisturbed flow
 * glides at exactly vInf * STREAK_CELLS_PER_MS cells per second — visual speed
 * tracks airspeed rather than tracking whatever the solver happens to be doing.
 */
function advanceStreaks(St, F, vInf, dt) {
  if (!St.seeded) resetStreaks(St, F);

  // Substep so no single move exceeds the trail spacing. Without this a fast
  // freestream advances several cells per frame, trail points end up spaced by
  // the per-frame distance instead of STREAK_SEG_CELLS, and ribbons stretch with
  // airspeed. Substepping also follows curved paths far more accurately.
  const cellsPerFrame = vInf * STREAK_CELLS_PER_MS * dt;
  const sub = Math.max(1, Math.min(8, Math.ceil((cellsPerFrame * 1.6) / STREAK_SEG_CELLS)));
  const glide = cellsPerFrame / sub;
  const subDt = dt / sub;

  for (let i = 0; i < STREAK_COUNT; i++) {
    for (let s = 0; s < sub; s++) {
      sampleFlow(F, St.x[i], St.y[i], VEL);
      const dx = VEL[0] * glide;
      const dy = VEL[1] * glide;
      const nx = St.x[i] + dx;
      const ny = St.y[i] + dy;
      St.age[i] += subDt;

      if (
        nx < 0.5 ||
        nx > NX - 1.5 ||
        ny < 1.5 ||
        ny > NY - 2.5 ||
        St.age[i] > STREAK_MAX_AGE ||
        F.solid[(ny | 0) * NX + (nx | 0)]
      ) {
        spawnStreak(St, i, F, false);
        break;
      }

      St.x[i] = nx;
      St.y[i] = ny;
      St.since[i] += Math.sqrt(dx * dx + dy * dy);

      if (St.since[i] >= STREAK_SEG_CELLS) {
        St.since[i] -= STREAK_SEG_CELLS;
        const h = (St.head[i] + 1) % STREAK_TRAIL;
        St.head[i] = h;
        St.tx[i * STREAK_TRAIL + h] = nx;
        St.ty[i * STREAK_TRAIL + h] = ny;
        if (St.count[i] < St.cap[i]) St.count[i]++;
      }
    }
  }
}

// Scratch buffers for ribbon construction (tail -> head order).
const RX = new Float64Array(STREAK_TRAIL);
const RY = new Float64Array(STREAK_TRAIL);

/**
 * Draw each tracer as a tapered ribbon: a filled polygon whose half-width grows
 * from the tail to the leading end, so the wind line reads as moving toward its
 * thicker tip. Filling one polygon per ribbon is far cheaper than stroking each
 * segment at its own width.
 */
function drawStreaks(ctx, St, F) {
  for (let i = 0; i < STREAK_COUNT; i++) {
    const n = St.count[i];
    if (n < 3) continue;

    const base = i * STREAK_TRAIL;
    const h = St.head[i];
    for (let k = 0; k < n; k++) {
      const idx = (h - (n - 1) + k + STREAK_TRAIL * 2) % STREAK_TRAIL;
      RX[k] = St.tx[base + idx] * SCALE;
      RY[k] = (NY - 1 - St.ty[base + idx]) * SCALE;
    }
    // The live position leads the last stored point; extend to it so the ribbon
    // glides continuously instead of advancing in discrete jumps.
    RX[n - 1] = St.x[i] * SCALE;
    RY[n - 1] = (NY - 1 - St.y[i]) * SCALE;

    sampleFlow(F, St.x[i], St.y[i], VEL);
    ctx.fillStyle = speedColor(Math.hypot(VEL[0], VEL[1]));

    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const px = RX[k];
      const py = RY[k];
      const ax = k === 0 ? RX[1] - RX[0] : px - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : py - RY[k - 1];
      const len = Math.hypot(ax, ay) || 1;
      const w = STREAK_TAIL_W + (STREAK_HEAD_W - STREAK_TAIL_W) * (k / (n - 1));
      const ox = (-ay / len) * w;
      const oy = (ax / len) * w;
      if (k === 0) ctx.moveTo(px + ox, py + oy);
      else ctx.lineTo(px + ox, py + oy);
    }
    for (let k = n - 1; k >= 0; k--) {
      const px = RX[k];
      const py = RY[k];
      const ax = k === 0 ? RX[1] - RX[0] : px - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : py - RY[k - 1];
      const len = Math.hypot(ax, ay) || 1;
      const w = STREAK_TAIL_W + (STREAK_HEAD_W - STREAK_TAIL_W) * (k / (n - 1));
      ctx.lineTo(px + (ay / len) * w, py - (ax / len) * w);
    }
    ctx.closePath();
    ctx.fill();
  }
}

/* ============================================================================
 * 4. 2D rendering
 * ==========================================================================*/

function renderFrame(ctx, field, fieldCtx, fieldImage, F, St) {
  const data = fieldImage.data;
  const solid = F.solid;

  // Background: speed magnitude, kept monochrome so the coloured ribbons read
  // clearly. Sampled from the coarse velocity grid — the grid is what costs
  // anything to produce, and resampling it here is a few bilinear lookups.
  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const id = y * NX + x;
      const p = ((NY - 1 - y) * NX + x) * 4; // image rows run top-down
      if (solid[id]) {
        data[p] = 22;
        data[p + 1] = 26;
        data[p + 2] = 34;
        data[p + 3] = 255;
        continue;
      }
      sampleFlow(F, x, y, VEL);
      const v = Math.min(1, Math.hypot(VEL[0], VEL[1]) / 1.9);
      data[p] = 10 + 46 * v;
      data[p + 1] = 16 + 62 * v;
      data[p + 2] = 30 + 92 * v;
      data[p + 3] = 255;
    }
  }
  fieldCtx.putImageData(fieldImage, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(field, 0, 0, CANVAS_W, CANVAS_H);

  drawStreaks(ctx, St, F);

  // Section outline.
  if (F.polyCount > 2) {
    ctx.beginPath();
    for (let i = 0; i < F.polyCount; i++) {
      const x = F.poly[2 * i] * SCALE;
      const y = (NY - 1 - F.poly[2 * i + 1]) * SCALE;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = '#e8edf5';
    ctx.fill();
    ctx.strokeStyle = '#9fb0c9';
    ctx.lineWidth = 1;
    ctx.stroke();

    drawSurfaceMarkers(ctx, F);
  }
}

/**
 * Transition and separation markers, drawn on the section itself.
 *
 * These are the two places the boundary layer changes character, and seeing them
 * on the surface next to the streamlines is the whole point of computing them
 * from the same state as the forces.
 */
function drawSurfaceMarkers(ctx, F) {
  const st = F.state;
  const geo = F.solver.geometry;
  if (!st || !geo) return;

  const mark = (xc, upper, colour) => {
    if (!(xc >= 0 && xc <= 1)) return;
    // Find the panel nearest this chordwise station on the requested surface.
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < geo.n; i++) {
      const isUpper = i > st.velocity.stagnationIndex;
      if (isUpper !== upper) continue;
      const d = Math.abs(geo.midX[i] - xc);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best < 0) return;
    const px = F.poly[2 * best] * SCALE;
    const py = (NY - 1 - F.poly[2 * best + 1]) * SCALE;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  };

  mark(st.transition.upperX, true, '#f5c04a');
  mark(st.transition.lowerX, false, '#f5c04a');
  mark(st.separation.upperX, true, '#f2545b');
  mark(st.separation.lowerX, false, '#f2545b');
}

/* ============================================================================
 * 5. 3D view
 *
 * The solve is strictly 2D, so the 3D view is that solution extruded along the
 * span: every spanwise station sees the identical cross-section and the
 * identical (x, y) flow. That is exactly the assumption the force readouts
 * already make (uniform section, no tip effects, S = span * chord), so the two
 * views are the same model drawn two ways rather than two different models.
 *
 * Rendered with a small software projector on a 2D canvas — no 3D library.
 * Occlusion is painter's algorithm over a single merged list of wing quads and
 * tracers, so tracers behind the wing are correctly hidden.
 * ==========================================================================*/

function createCamera() {
  return { az: 0.62, el: 0.32, zoom: 1, dragging: false, lastX: 0, lastY: 0 };
}

function makeProjection(cam, target, spanCells) {
  const fit = Math.max(NX, spanCells) * 0.88;
  return {
    tx: target[0],
    ty: target[1],
    tz: target[2],
    ca: Math.cos(cam.az),
    sa: Math.sin(cam.az),
    ce: Math.cos(cam.el),
    se: Math.sin(cam.el),
    dist: fit / cam.zoom,
    focal: VIEW3D_H * 1.15,
    cx: VIEW3D_W / 2,
    cy: VIEW3D_H / 2,
    near: 1,
  };
}

// Scratch: [screenX, screenY, depth]. Depth is camera-space distance.
const PRJ = new Float64Array(3);

function project(P3, x, y, z) {
  const px = x - P3.tx;
  const py = y - P3.ty;
  const pz = z - P3.tz;
  const X = P3.ca * px + P3.sa * pz;
  const Zy = -P3.sa * px + P3.ca * pz;
  // Positive elevation lifts the camera above the wing, so a point higher in y
  // must get *closer* (smaller depth), not farther.
  const Y = P3.ce * py + P3.se * Zy;
  const Z = -P3.se * py + P3.ce * Zy;
  const depth = Z + P3.dist;
  if (depth <= P3.near) {
    PRJ[2] = -1;
    return false;
  }
  const f = P3.focal / depth;
  PRJ[0] = P3.cx + X * f;
  PRJ[1] = P3.cy - Y * f;
  PRJ[2] = depth;
  return true;
}

// Reusable draw list so the render loop doesn't allocate every frame.
const DRAW = [];
const ORDER = [];
let drawLen = 0;
function pushDraw(depth, kind, index, shade) {
  if (drawLen === DRAW.length) DRAW.push({ depth: 0, kind: 0, index: 0, shade: 0 });
  const d = DRAW[drawLen++];
  d.depth = depth;
  d.kind = kind;
  d.index = index;
  d.shade = shade;
  ORDER.push(d);
}
const byDepthDesc = (a, b) => b.depth - a.depth;

const WING_CAP = 512;
let WING_N = 0;
const WING_IDX = new Int32Array(WING_CAP);
const NEAR_X = new Float64Array(WING_CAP);
const NEAR_Y = new Float64Array(WING_CAP);
const NEAR_D = new Float64Array(WING_CAP);
const FAR_X = new Float64Array(WING_CAP);
const FAR_Y = new Float64Array(WING_CAP);
const FAR_D = new Float64Array(WING_CAP);
const WING_SHADE = new Float64Array(WING_CAP);
const WING_OK = new Uint8Array(WING_CAP);
// Normalised heat-map position at each end of a wing quad, so the fill can be a
// gradient between them rather than one flat colour per panel.
const WING_T0 = new Float64Array(WING_CAP);
const WING_T1 = new Float64Array(WING_CAP);

// Per-outline-node heat value, averaged from the two panels meeting at it. The
// solver's arrays are per panel; the mesh is drawn between nodes.
let NODE_VAL = new Float64Array(0);

const RGB_A = [0, 0, 0];
const RGB_B = [0, 0, 0];

/**
 * Blend a ramp colour with the diffuse shading term.
 *
 * The heat map has to carry the data *and* the surface still has to read as a
 * three-dimensional object. Multiplying the ramp colour by the lambert term
 * would do the second at the cost of the first — a dark blue panel facing away
 * from the light becomes indistinguishable from black. So the shading is
 * applied as a partial lift toward white/black around a mid grey, which keeps
 * the hue intact and only modulates its brightness over a limited range.
 */
function shadeRgb(rgb, shade) {
  const f = 0.55 + 0.45 * shade;
  return `rgb(${(rgb[0] * f) | 0},${(rgb[1] * f) | 0},${(rgb[2] * f) | 0})`;
}

/** Faint wireframe of the flow domain, for spatial orientation. */
function drawDomainBox(ctx, P3, spanCells) {
  const hz = spanCells / 2;
  const corners = [
    [0, 0, -hz], [NX, 0, -hz], [NX, NY, -hz], [0, NY, -hz],
    [0, 0, hz], [NX, 0, hz], [NX, NY, hz], [0, NY, hz],
  ];
  const pts = corners.map((c) => (project(P3, c[0], c[1], c[2]) ? [PRJ[0], PRJ[1]] : null));
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  ctx.strokeStyle = 'rgba(120, 145, 180, 0.16)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [a, b] of edges) {
    if (!pts[a] || !pts[b]) continue;
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
  }
  ctx.stroke();
}

/**
 * Hover readout, drawn straight onto the canvas.
 *
 * Deliberately not React state: the pointer moves every frame, and routing that
 * through a re-render would re-reconcile the whole dashboard sixty times a
 * second to move a tooltip. The camera already works this way.
 */
function drawHoverTip(ctx, hover, mode) {
  if (!hover.active || hover.panel < 0) return;
  const text = `${mode.symbol} ${formatHeatValue(hover.value, mode)}`;
  const sub = `x/c ${hover.xc.toFixed(3)} · ${hover.surface}`;

  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  const w = Math.max(ctx.measureText(text).width, ctx.measureText(sub).width) + 16;
  const h = 34;
  // Flip the box to stay inside the viewport near the edges.
  const x = Math.min(hover.x + 12, VIEW3D_W - w - 4);
  const y = Math.min(Math.max(hover.y - h - 10, 4), VIEW3D_H - h - 4);

  ctx.fillStyle = 'rgba(13, 17, 23, 0.92)';
  ctx.strokeStyle = 'rgba(120, 145, 180, 0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  // roundRect is recent enough to be worth a fallback: this runs inside the
  // animation loop, where an exception would take the whole render down.
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, 6);
  else ctx.rect(x, y, w, h);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#dfe6f0';
  ctx.fillText(text, x + 8, y + 15);
  ctx.fillStyle = '#8b98ab';
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.fillText(sub, x + 8, y + 27);

  ctx.strokeStyle = 'rgba(77, 163, 255, 0.9)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(hover.px, hover.py, 5, 0, Math.PI * 2);
  ctx.stroke();
}

/**
 * Resolve a pointer position to the nearest point on the projected surface.
 *
 * Runs against the mid-span outline rather than the full quad mesh: the wing is
 * a constant extrusion, so the chordwise station is all that varies, and a
 * nearest-point search over ~220 projected nodes is both simpler and cheaper
 * than a ray cast through the quads.
 */
function pickSurface(F, P3, hover, heat) {
  hover.panel = -1;
  if (!hover.active || !F.state) return;
  const n = F.polyCount;
  let best = -1;
  let bestD = 18 * 18; // px^2; beyond this the pointer is not on the section
  let bx = 0;
  let by = 0;
  for (let i = 0; i < n; i++) {
    if (!project(P3, F.poly[2 * i], F.poly[2 * i + 1], 0)) continue;
    const dx = PRJ[0] - hover.x;
    const dy = PRJ[1] - hover.y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
      bx = PRJ[0];
      by = PRJ[1];
    }
  }
  if (best < 0) return;
  hover.panel = best;
  hover.px = bx;
  hover.py = by;
  hover.value = heat && heat.values ? heat.values[best] : NaN;
  hover.xc = F.solver.geometry ? F.solver.geometry.midX[best] : 0;
  hover.surface = best > F.state.velocity.stagnationIndex ? 'upper surface' : 'lower surface';
}

/** Draw the extruded wing and the tracers into the 3D viewport. */
function render3D(ctx, F, St, cam, spanCells, viz) {
  ctx.fillStyle = '#0a101e';
  ctx.fillRect(0, 0, VIEW3D_W, VIEW3D_H);
  if (F.polyCount < 3) return;

  const hz = spanCells / 2;
  const target = [F.pivotX, PIVOT_Y, 0];
  const P3 = makeProjection(cam, target, spanCells);

  drawDomainBox(ctx, P3, spanCells);

  drawLen = 0;
  ORDER.length = 0;

  // --- Heat map: per-node surface values ----------------------------------
  // Averaged onto the nodes here rather than in the draw loop, because a node is
  // shared by two quads and would otherwise be averaged twice.
  const heat = viz.heat;
  const heatOn = !!(heat && heat.values && heat.values.length === F.polyCount);
  if (heatOn) {
    if (NODE_VAL.length !== F.polyCount) NODE_VAL = new Float64Array(F.polyCount);
    const v = heat.values;
    const n = F.polyCount;
    for (let i = 0; i < n; i++) NODE_VAL[i] = 0.5 * (v[(i - 1 + n) % n] + v[i]);
  }
  const span = heatOn ? heat.hi - heat.lo || 1 : 1;

  // --- Wing: one quad per outline edge, spanning the full wing -------------
  // The section is constant along span, so no spanwise subdivision is needed.
  WING_N = 0;
  for (let i = 0; i < F.polyCount && WING_N < WING_CAP; i += WING_OUTLINE_STEP) {
    WING_IDX[WING_N++] = i;
  }

  // Light from above and slightly front-left, in world axes.
  const LX = -0.35;
  const LY = 0.82;

  for (let k = 0; k < WING_N; k++) {
    const i = WING_IDX[k];
    const ax = F.poly[2 * i];
    const ay = F.poly[2 * i + 1];
    const okN = project(P3, ax, ay, -hz);
    NEAR_X[k] = PRJ[0];
    NEAR_Y[k] = PRJ[1];
    NEAR_D[k] = PRJ[2];
    const okF = project(P3, ax, ay, hz);
    FAR_X[k] = PRJ[0];
    FAR_Y[k] = PRJ[1];
    FAR_D[k] = PRJ[2];
    WING_OK[k] = okN && okF ? 1 : 0;
  }

  for (let k = 0; k < WING_N; k++) {
    const k2 = (k + 1) % WING_N;
    if (!WING_OK[k] || !WING_OK[k2]) continue;
    const i = WING_IDX[k];
    const j = WING_IDX[k2];
    const ex = F.poly[2 * j] - F.poly[2 * i];
    const ey = F.poly[2 * j + 1] - F.poly[2 * i + 1];
    const el = Math.hypot(ex, ey) || 1;
    // The ring winds clockwise, for which (-dy, dx) points out of the section.
    const diffuse = Math.max(0, (-ey / el) * LX + (ex / el) * LY);
    WING_SHADE[k] = 0.22 + 0.78 * diffuse;
    if (heatOn) {
      WING_T0[k] = (NODE_VAL[i] - heat.lo) / span;
      WING_T1[k] = (NODE_VAL[j] - heat.lo) / span;
    }
    pushDraw((NEAR_D[k] + NEAR_D[k2] + FAR_D[k] + FAR_D[k2]) * 0.25, 0, k, WING_SHADE[k]);
  }

  // End caps, so the wing reads as solid when seen from the side.
  let capNearD = 0;
  let capFarD = 0;
  for (let k = 0; k < WING_N; k++) {
    capNearD += NEAR_D[k];
    capFarD += FAR_D[k];
  }
  pushDraw(capNearD / WING_N, 2, 0, 0.55);
  pushDraw(capFarD / WING_N, 2, 1, 0.55);

  // --- Tracers ------------------------------------------------------------
  const margin = 60;
  for (let i = 0; i < STREAK_COUNT; i++) {
    if (St.count[i] < 3) continue;
    const z = St.sz[i] * spanCells;
    if (!project(P3, St.x[i], St.y[i], z)) continue;
    if (PRJ[0] < -margin || PRJ[0] > VIEW3D_W + margin || PRJ[1] < -margin || PRJ[1] > VIEW3D_H + margin) {
      continue;
    }
    pushDraw(PRJ[2], 1, i, 0);
  }

  // --- Depth sort and draw back to front ----------------------------------
  ORDER.sort(byDepthDesc);

  for (let n = 0; n < ORDER.length; n++) {
    const item = ORDER[n];

    if (item.kind === 0) {
      const k = item.index;
      const k2 = (k + 1) % WING_N;
      const sh = item.shade;

      let fill;
      let edge;
      if (heatOn) {
        // Gradient along the chordwise direction of the quad. The scalar is
        // constant along the span by construction, so the gradient axis is the
        // line joining the two quad ends at mid-span — the closest a linear
        // gradient can get to "constant along span" once the quad has been
        // projected, and visually indistinguishable from it at these angles.
        rampColor(heat.scaleId, WING_T0[k], RGB_A);
        rampColor(heat.scaleId, WING_T1[k], RGB_B);
        const ax = (NEAR_X[k] + FAR_X[k]) * 0.5;
        const ay = (NEAR_Y[k] + FAR_Y[k]) * 0.5;
        const bx = (NEAR_X[k2] + FAR_X[k2]) * 0.5;
        const by = (NEAR_Y[k2] + FAR_Y[k2]) * 0.5;
        const g = ctx.createLinearGradient(ax, ay, bx, by);
        g.addColorStop(0, shadeRgb(RGB_A, sh));
        g.addColorStop(1, shadeRgb(RGB_B, sh));
        fill = g;
        edge = shadeRgb(RGB_A, sh);
      } else {
        edge = `rgb(${(232 * sh) | 0},${(237 * sh) | 0},${(245 * sh) | 0})`;
        fill = edge;
      }

      ctx.fillStyle = fill;
      // Stroke with the fill colour too: adjacent quads are antialiased along
      // their shared edge, which otherwise leaves a visible seam.
      ctx.strokeStyle = edge;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(NEAR_X[k], NEAR_Y[k]);
      ctx.lineTo(NEAR_X[k2], NEAR_Y[k2]);
      ctx.lineTo(FAR_X[k2], FAR_Y[k2]);
      ctx.lineTo(FAR_X[k], FAR_Y[k]);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      continue;
    }

    if (item.kind === 2) {
      const useNear = item.index === 0;
      const sh = item.shade;
      ctx.fillStyle = `rgb(${(232 * sh) | 0},${(237 * sh) | 0},${(245 * sh) | 0})`;
      ctx.beginPath();
      let moved = false;
      for (let k = 0; k < WING_N; k++) {
        if (!WING_OK[k]) continue;
        const px = useNear ? NEAR_X[k] : FAR_X[k];
        const py = useNear ? NEAR_Y[k] : FAR_Y[k];
        if (!moved) {
          ctx.moveTo(px, py);
          moved = true;
        } else ctx.lineTo(px, py);
      }
      if (moved) {
        ctx.closePath();
        ctx.fill();
      }
      continue;
    }

    // Tracer: tapered trail ending in a small sphere at the leading end.
    const i = item.index;
    const n2 = St.count[i];
    const base = i * STREAK_TRAIL;
    const h = St.head[i];
    const z = St.sz[i] * spanCells;

    sampleFlow(F, St.x[i], St.y[i], VEL);
    const colour = speedColor(Math.hypot(VEL[0], VEL[1]));

    // Perspective scale for widths. STREAK_HEAD_W/TAIL_W are 2D canvas pixels at
    // SCALE px per cell, so dividing by SCALE converts them to cells before
    // projecting. Capped because a tracer close to the camera would otherwise be
    // projected into a huge wedge that swamps the view.
    const persp = P3.focal / item.depth;
    const wScale = Math.min((persp / SCALE) * STREAK_3D_GAIN, RIBBON_MAX_HALF_PX / STREAK_HEAD_W);

    // Project the whole trail up front. Bailing out midway would leave a
    // half-built path that closePath() turns into a spurious triangle.
    let ok = true;
    for (let k = 0; k < n2; k++) {
      const idx = (h - (n2 - 1) + k + STREAK_TRAIL * 2) % STREAK_TRAIL;
      const wx = k === n2 - 1 ? St.x[i] : St.tx[base + idx];
      const wy = k === n2 - 1 ? St.y[i] : St.ty[base + idx];
      if (!project(P3, wx, wy, z)) {
        ok = false;
        break;
      }
      RX[k] = PRJ[0];
      RY[k] = PRJ[1];
    }
    if (!ok) continue;

    ctx.fillStyle = colour;
    ctx.beginPath();
    for (let k = 0; k < n2; k++) {
      const ax = k === 0 ? RX[1] - RX[0] : RX[k] - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : RY[k] - RY[k - 1];
      const dl = Math.hypot(ax, ay) || 1;
      const w = (STREAK_3D_TAIL_W + (STREAK_HEAD_W - STREAK_3D_TAIL_W) * (k / (n2 - 1))) * wScale;
      const ox = (-ay / dl) * w;
      const oy = (ax / dl) * w;
      if (k === 0) ctx.moveTo(RX[k] + ox, RY[k] + oy);
      else ctx.lineTo(RX[k] + ox, RY[k] + oy);
    }
    for (let k = n2 - 1; k >= 0; k--) {
      const ax = k === 0 ? RX[1] - RX[0] : RX[k] - RX[k - 1];
      const ay = k === 0 ? RY[1] - RY[0] : RY[k] - RY[k - 1];
      const dl = Math.hypot(ax, ay) || 1;
      const w = (STREAK_3D_TAIL_W + (STREAK_HEAD_W - STREAK_3D_TAIL_W) * (k / (n2 - 1))) * wScale;
      ctx.lineTo(RX[k] + (ay / dl) * w, RY[k] - (ax / dl) * w);
    }
    ctx.closePath();
    ctx.fill();

    // Head sphere: a small shaded ball marking where the parcel is now.
    if (project(P3, St.x[i], St.y[i], z)) {
      const r = Math.min(SPHERE_MAX_PX, Math.max(0.9, SPHERE_HEAD_CELLS * persp * STREAK_3D_GAIN));
      const hx = PRJ[0];
      const hy = PRJ[1];
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(hx, hy, r, 0, Math.PI * 2);
      ctx.fill();
      // A single off-centre highlight reads as a sphere far more cheaply than a
      // radial gradient, which would mean one gradient object per tracer per frame.
      if (r > 1.8) {
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.beginPath();
        ctx.arc(hx - r * 0.3, hy - r * 0.3, r * 0.38, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // --- Overlays, always on top --------------------------------------------
  pickSurface(F, P3, viz.hover, heatOn ? heat : null);
  if (heatOn) drawHoverTip(ctx, viz.hover, heat.mode);
}

/**
 * Renderer internals, exposed only so validation/render3d.check.mjs can drive a
 * real frame against a stub canvas in node. Not part of the component API — the
 * component takes props and nothing else.
 *
 * This exists because a throw inside requestAnimationFrame is invisible: the
 * bundle builds, the page loads, and the canvas simply stops. A missing
 * `solverToView` import did exactly that to the 3D view, and the only way to
 * catch that class of bug short of a browser is to execute the draw path.
 */
export const __internals = { render3D, createStreaks, createCamera };

/* ============================================================================
 * 6. Component
 * ==========================================================================*/

export default function WindTunnel({
  initialNacaCode = '2412',
  initialGeometry = DEFAULT_GEOMETRY,
  initialAirspeed = 30,
  initialAoa = 5,
  initialChordCm = CHORD_DEFAULT_CM,
  onReadingsChange,
}) {
  const [code, setCode] = useState(initialNacaCode);
  const [airspeed, setAirspeed] = useState(initialAirspeed);
  const [aoa, setAoa] = useState(initialAoa);
  const [chordCm, setChordCm] = useState(initialChordCm);
  const [spanRatio, setSpanRatio] = useState(SPAN_DEFAULT);
  const [showInfo, setShowInfo] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState('2d');

  // --- Visualisation -------------------------------------------------------
  const [geometry, setGeometry] = useState(initialGeometry);
  const [showHeat, setShowHeat] = useState(true);
  const [heatModeId, setHeatModeId] = useState(DEFAULT_MODE);
  const [scaleId, setScaleId] = useState(DEFAULT_SCALE);
  const [showHeatInfo, setShowHeatInfo] = useState(false);

  // Parsed only for immediate validation feedback while the user is typing; the
  // solver keeps running the last valid section until a complete one is entered.
  // Only the NACA geometry has anything to validate — the others are built-in.
  const parsed = useMemo(
    () => (geometry === 'naca' ? parseNacaCode(code) : { ok: true }),
    [code, geometry]
  );
  const [validCode, setValidCode] = useState(() =>
    parseNacaCode(initialNacaCode).ok ? initialNacaCode : '2412'
  );
  useEffect(() => {
    if (parsed.ok && geometry === 'naca') setValidCode(code);
  }, [parsed, code, geometry]);

  const flowRef = useRef(null);
  if (flowRef.current === null) flowRef.current = createFlow();

  /* --- The one authoritative solve --------------------------------------- */
  // Every readout below is a field of this object. It is recomputed on the same
  // commit as the control that changed it, so the dashboard never lags the
  // slider — the panel factorisation is cached and only the right-hand side
  // moves when the angle of attack does.
  const state = useMemo(() => {
    const F = flowRef.current;
    return updateFlow(F, {
      geometry,
      naca: validCode,
      alphaDeg: aoa,
      airspeed,
      chord: cmToM(chordCm),
      span: spanRatio * cmToM(chordCm),
      rho: RHO_AIR,
      nu: NU_AIR,
    });
  }, [geometry, validCode, aoa, airspeed, chordCm, spanRatio]);

  /* --- The surface heat map ------------------------------------------------
   * Derived from the same converged state as everything else, so the colour at
   * a point on the wing and the number in the dashboard cannot disagree. The
   * domain is recomputed with the state, which is what makes the scale track
   * the flow as the angle of attack moves. */
  const heat = useMemo(() => {
    if (!state || !showHeat) return null;
    const mode = getMode(heatModeId);
    const scale = getScale(scaleId);
    const values = mode.values(state);
    const { lo, hi } = heatDomain(values, mode, scale);
    return { values, mode, scale, scaleId: scale.id, lo, hi };
  }, [state, showHeat, heatModeId, scaleId]);

  const readings = useMemo(() => {
    if (!state) return null;
    return {
      ...state.primary,
      stall: state.stall,
      separation: state.separation,
      transition: state.transition,
      convergence: state.convergence,
      stability: state.stability,
      forceBreakdown: state.forceBreakdown,
      boundaryLayer: state.boundaryLayer,
      wake: state.wake,
      pressure: state.pressure,
      airfoil: state.airfoil,
      spanRatio,
      chordCm,
    };
  }, [state, spanRatio, chordCm]);

  const readingsCbRef = useRef(onReadingsChange);
  readingsCbRef.current = onReadingsChange;
  useEffect(() => {
    if (readingsCbRef.current && readings) readingsCbRef.current(readings);
  }, [readings]);

  /* --- Canvases and the animation loop ------------------------------------ */
  const canvasRef = useRef(null);
  const canvas3dRef = useRef(null);
  const camRef = useRef(null);
  const streaksRef = useRef(null);
  if (streaksRef.current === null) streaksRef.current = createStreaks();
  if (camRef.current === null) camRef.current = createCamera();

  const airspeedRef = useRef(airspeed);
  airspeedRef.current = airspeed;
  const spanRatioRef = useRef(spanRatio);
  spanRatioRef.current = spanRatio;
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;

  // Everything the 3D renderer needs that is not the flow itself. Mutated in
  // place rather than replaced, so the hover record — which the render loop
  // writes the pick result back into — survives a re-render.
  const vizRef = useRef(null);
  if (vizRef.current === null) {
    vizRef.current = {
      heat: null,
      hover: { active: false, x: 0, y: 0, px: 0, py: 0, panel: -1, value: NaN, xc: 0, surface: '' },
    };
  }
  vizRef.current.heat = heat;

  // A geometry or incidence change moves the body under the tracers; any that
  // are now inside it, or trailing a ribbon through it, must be reseeded.
  useEffect(() => {
    streaksRef.current.seeded = false;
  }, [geometry, validCode, aoa, chordCm]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');

    const field = document.createElement('canvas');
    field.width = NX;
    field.height = NY;
    const fieldCtx = field.getContext('2d');
    const fieldImage = fieldCtx.createImageData(NX, NY);

    const ctx3d = canvas3dRef.current ? canvas3dRef.current.getContext('2d') : null;

    let raf = 0;
    let lastFrame = performance.now();

    const loop = (now) => {
      raf = requestAnimationFrame(loop);
      const F = flowRef.current;
      const St = streaksRef.current;

      // Real elapsed time drives the tracers, so they glide at a rate set by
      // airspeed rather than by however fast anything happens to be running.
      // Clamped so a stalled tab doesn't teleport every ribbon on resume.
      const dt = Math.min(0.05, Math.max(0, (now - lastFrame) / 1000));
      lastFrame = now;

      // The velocity field is rebuilt in slices whenever the state changes; the
      // previous one keeps being drawn until the new one is complete.
      fillField(F, FIELD_BUDGET_MS);

      // The unsteady vortex wake evolves in physical time, independently of the
      // steady solve — it is what the wake diagnostics report on.
      F.solver.advanceWake(dt);

      advanceStreaks(St, F, airspeedRef.current, dt);
      renderFrame(ctx, field, fieldCtx, fieldImage, F, St);

      if (ctx3d && activeTabRef.current === '3d') {
        render3D(ctx3d, F, St, camRef.current, spanRatioRef.current * F.nCells, vizRef.current);
      }
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // Started on mount, cancelled on unmount. Everything the loop needs is read
    // through refs, so it never needs tearing down.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* --- 3D orbit controls --------------------------------------------------
   * Camera state lives in a ref and is read by the render loop, so dragging
   * never triggers a React render. */
  useEffect(() => {
    const el = canvas3dRef.current;
    if (!el) return undefined;
    const cam = camRef.current;

    const pos = (e) => {
      const r = el.getBoundingClientRect();
      // The canvas is CSS-scaled, so convert to backing-store pixels.
      return [((e.clientX - r.left) / r.width) * VIEW3D_W, ((e.clientY - r.top) / r.height) * VIEW3D_H];
    };

    const onDown = (e) => {
      const [x, y] = pos(e);
      cam.dragging = true;
      cam.lastX = x;
      cam.lastY = y;
      if (el.setPointerCapture) el.setPointerCapture(e.pointerId);
    };
    const onMove = (e) => {
      const [x, y] = pos(e);
      // The pointer position goes into a ref and is resolved to a surface point
      // by the render loop, which already has the projection to hand. Nothing
      // here touches React state, so hovering costs no re-renders.
      const hov = vizRef.current.hover;
      hov.x = x;
      hov.y = y;
      hov.active = !cam.dragging;

      if (!cam.dragging) return;
      cam.az -= (x - cam.lastX) * 0.006;
      cam.el += (y - cam.lastY) * 0.006;
      if (cam.el < CAM_MIN_EL) cam.el = CAM_MIN_EL;
      if (cam.el > CAM_MAX_EL) cam.el = CAM_MAX_EL;
      cam.lastX = x;
      cam.lastY = y;
    };
    const onLeave = () => {
      vizRef.current.hover.active = false;
    };
    const onUp = (e) => {
      cam.dragging = false;
      if (el.releasePointerCapture && e.pointerId !== undefined) {
        try {
          el.releasePointerCapture(e.pointerId);
        } catch {
          /* pointer already released */
        }
      }
    };
    const onWheel = (e) => {
      e.preventDefault();
      cam.zoom *= e.deltaY < 0 ? 1.1 : 1 / 1.1;
      if (cam.zoom < CAM_MIN_ZOOM) cam.zoom = CAM_MIN_ZOOM;
      if (cam.zoom > CAM_MAX_ZOOM) cam.zoom = CAM_MAX_ZOOM;
    };

    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.addEventListener('pointerleave', onLeave);
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('pointercancel', onUp);
      el.removeEventListener('pointerleave', onLeave);
      el.removeEventListener('wheel', onWheel);
    };
  }, []);

  const resetView = useCallback(() => {
    const c = camRef.current;
    const fresh = createCamera();
    c.az = fresh.az;
    c.el = fresh.el;
    c.zoom = fresh.zoom;
  }, []);

  const onCodeChange = useCallback((e) => setCode(e.target.value), []);

  const stallState = readings?.stall.state ?? 'attached';

  return (
    <div className={styles.root}>
      <div className={styles.tabBar} role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === '2d'}
          className={`${styles.tabButton} ${activeTab === '2d' ? styles.tabButtonActive : ''}`}
          onClick={() => setActiveTab('2d')}
        >
          2D
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === '3d'}
          className={`${styles.tabButton} ${activeTab === '3d' ? styles.tabButtonActive : ''}`}
          onClick={() => setActiveTab('3d')}
        >
          3D
        </button>
        {activeTab === '3d' && (
          <button type="button" className={styles.resetViewButton} onClick={resetView}>
            Reset view
          </button>
        )}
      </div>

      <div className={styles.topBar}>
        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-geometry">
            Geometry
          </label>
          <select
            id="wt-geometry"
            className={styles.select}
            value={geometry}
            onChange={(e) => setGeometry(e.target.value)}
          >
            {GEOMETRIES.map((g) => (
              <option key={g.id} value={g.id}>
                {g.label}
              </option>
            ))}
          </select>
        </div>

        {/* The designation box belongs to the NACA geometry only; the other
            sections are built in and have nothing to type. */}
        <div className={`${styles.control} ${geometry === 'naca' ? '' : styles.controlHidden}`}>
          <label className={styles.label} htmlFor="wt-naca">
            NACA airfoil
          </label>
          <input
            id="wt-naca"
            className={`${styles.textInput} ${parsed.ok ? '' : styles.textInputError}`}
            value={code}
            onChange={onCodeChange}
            spellCheck={false}
            autoComplete="off"
            placeholder="2412"
            aria-invalid={!parsed.ok}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-speed">
            Airspeed <span className={styles.labelValue}>{airspeed} m/s</span>
          </label>
          <input
            id="wt-speed"
            className={styles.slider}
            type="range"
            min={V_MIN}
            max={V_MAX}
            step={1}
            value={airspeed}
            onChange={(e) => setAirspeed(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-aoa">
            Angle of attack <span className={styles.labelValue}>{aoa.toFixed(1)}°</span>
          </label>
          <input
            id="wt-aoa"
            className={styles.slider}
            type="range"
            min={AOA_MIN}
            max={AOA_MAX}
            step={0.5}
            value={aoa}
            onChange={(e) => setAoa(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-chord">
            Size (chord) <span className={styles.labelValue}>{chordCm.toFixed(1)} cm</span>
          </label>
          <input
            id="wt-chord"
            className={styles.slider}
            type="range"
            min={CHORD_MIN_CM}
            max={CHORD_MAX_CM}
            step={0.1}
            value={chordCm}
            onChange={(e) => setChordCm(Number(e.target.value))}
          />
        </div>

        <div className={styles.control}>
          <label className={styles.label} htmlFor="wt-span">
            Span{' '}
            <span className={styles.labelValue}>
              {spanRatio.toFixed(1)}× = {(spanRatio * chordCm).toFixed(1)} cm
            </span>
          </label>
          <input
            id="wt-span"
            className={styles.slider}
            type="range"
            min={SPAN_MIN}
            max={SPAN_MAX}
            step={0.1}
            value={spanRatio}
            onChange={(e) => setSpanRatio(Number(e.target.value))}
          />
        </div>
      </div>

      {!parsed.ok && <div className={styles.error}>{parsed.error}</div>}
      {/* Read off the solved section rather than the parsed input, so it covers
          every geometry — the flat plate's leading-edge separation caveat as
          well as the 5-digit reflex-camber note. */}
      {parsed.ok && readings?.airfoil.warning && (
        <div className={styles.warning}>{readings.airfoil.warning}</div>
      )}
      {flowRef.current.error && <div className={styles.error}>{flowRef.current.error}</div>}

      <div className={`${styles.body} ${activeTab === '2d' ? '' : styles.tabHidden}`}>
        <div className={styles.viewport}>
          <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} className={styles.canvas} />

          <StallBanner state={stallState} label={readings?.stall.label} />

          <div className={styles.viewportFooter}>
            <span className={styles.airfoilName}>{readings?.airfoil.label ?? 'NACA'}</span>
            <button
              type="button"
              className={styles.infoButton}
              onClick={() => setShowInfo((v) => !v)}
              aria-expanded={showInfo}
            >
              Viscous–inviscid panel solve at Re = {formatSci(readings?.reynolds ?? 0)} ·{' '}
              {readings?.convergence.confidence ?? 0}% confidence ⓘ
            </button>
          </div>

          {showInfo && readings && <InfoPanel readings={readings} state={state} />}
        </div>

        <aside className={styles.panel}>
          <h3 className={styles.panelTitle}>Live readings</h3>
          {readings && (
            <LiveReadings
              readings={readings}
              showAdvanced={showAdvanced}
              onToggleAdvanced={() => setShowAdvanced((v) => !v)}
            />
          )}
        </aside>
      </div>

      {/* --- 3D view: the same solution extruded along the span ------------- */}
      {activeTab === '3d' && (
        <>
          <div className={styles.toggleBar}>
            <label className={styles.toggleLabel}>
              <input type="checkbox" checked={showHeat} onChange={() => setShowHeat((v) => !v)} />
              Surface heat map
            </label>

            <select
              className={styles.selectSmall}
              value={heatModeId}
              onChange={(e) => setHeatModeId(e.target.value)}
              disabled={!showHeat}
              aria-label="Heat map quantity"
            >
              {HEAT_MODES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>

            <select
              className={styles.selectSmall}
              value={scaleId}
              onChange={(e) => setScaleId(e.target.value)}
              disabled={!showHeat}
              aria-label="Colour scale"
            >
              {COLOR_SCALES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>

            <button
              type="button"
              className={styles.heatmapInfoIcon}
              onClick={() => setShowHeatInfo((v) => !v)}
              aria-expanded={showHeatInfo}
              aria-label="About the heat map"
            >
              <span className={styles.circleI}>i</span>
            </button>

            {heat && <HeatLegend heat={heat} />}
          </div>

          {showHeatInfo && <HeatmapInfo heat={heat} onClose={() => setShowHeatInfo(false)} />}
        </>
      )}

      <div className={`${styles.body} ${activeTab === '3d' ? '' : styles.tabHidden}`}>
        <div className={styles.viewport}>
          <canvas ref={canvas3dRef} width={VIEW3D_W} height={VIEW3D_H} className={styles.canvas3d} />

          <StallBanner state={stallState} label={readings?.stall.label} />

          <div className={styles.viewportFooter}>
            <span className={styles.airfoilName}>
              {readings?.airfoil.label ?? 'NACA'} · span {(spanRatio * chordCm).toFixed(1)} cm
            </span>
            <span className={styles.infoButton} aria-hidden="true">
              drag to orbit · scroll to zoom
            </span>
          </div>
        </div>

        <aside className={styles.panel}>
          <h3 className={styles.panelTitle}>3D view</h3>
          <p className={styles.view3dNote}>
            The section is extruded equally either side of the mid-span plane, and that centre
            cross-section is exactly what the 2D view shows. The solve is two-dimensional, so every
            spanwise station sees the same flow, with no tip vortices or downwash. Changing the span
            rescales the reported forces and nothing else; it does not re-run the solver, because
            there is nothing in the physics for it to change.
          </p>
          {readings && (
            <LiveReadings
              readings={readings}
              showAdvanced={showAdvanced}
              onToggleAdvanced={() => setShowAdvanced((v) => !v)}
            />
          )}
        </aside>
      </div>
    </div>
  );
}

/* ============================================================================
 * 7. Readout components
 * ==========================================================================*/

/** Stall banner, shared by the 2D and 3D viewports. */
function StallBanner({ state, label }) {
  if (state === 'attached') return null;
  const severe = state === 'partial' || state === 'full';
  return (
    <div
      className={`${styles.stallBanner} ${severe ? '' : styles.stallBannerNear}`}
      role="status"
    >
      ⚠ {(label ?? state).toUpperCase()}
    </div>
  );
}

/**
 * The dashboard.
 *
 * The primary block is the standard interface: the six coefficients and forces
 * plus the conditions they were computed at. Everything else is behind the
 * expander, because it is diagnostic rather than headline — but it is all from
 * the same converged state, so nothing there can contradict anything here.
 */
function LiveReadings({ readings, showAdvanced, onToggleAdvanced }) {
  const sep = readings.separation;
  const tr = readings.transition;
  const conv = readings.convergence;
  const stall = readings.stall;

  return (
    <>
      <Readout label="Lift coefficient" symbol="Cl" value={readings.cl.toFixed(3)} />
      <Readout label="Drag coefficient" symbol="Cd" value={readings.cd.toFixed(5)} />
      <Readout label="Moment coefficient" symbol="Cm" value={readings.cm.toFixed(4)} unit="c/4" />
      <Readout label="Reynolds number" symbol="Re" value={formatSci(readings.reynolds)} />
      <Readout label="Lift force" symbol="L" value={readings.liftForce.toFixed(2)} unit="N" />
      <Readout label="Drag force" symbol="D" value={readings.dragForce.toFixed(3)} unit="N" />
      <Readout
        label="Lift-to-drag ratio"
        symbol="L/D"
        value={readings.ldRatio.toFixed(1)}
      />
      <Readout
        label="Dynamic pressure"
        symbol="q"
        value={readings.dynamicPressure.toFixed(0)}
        unit="Pa"
      />

      <div
        className={`${styles.stallChip} ${
          stall.state === 'full' || stall.state === 'partial'
            ? styles.stallChipActive
            : stall.state === 'incipient'
              ? styles.stallChipNear
              : ''
        }`}
      >
        {stall.label}
        <span className={styles.stallChipDetail}>
          index {stall.index.toFixed(2)} · {stall.flowAttachmentPercent.toFixed(0)}% attached
          {stall.stallMarginDeg !== null && stall.stallMarginDeg > 0.05 && (
            <> · {stall.stallMarginDeg.toFixed(1)}° margin</>
          )}
        </span>
      </div>

      <button
        type="button"
        className={`${styles.infoButton} ${styles.advancedToggle}`}
        onClick={onToggleAdvanced}
        aria-expanded={showAdvanced}
      >
        {showAdvanced ? '▾' : '▸'} Advanced metrics
      </button>

      {showAdvanced && (
        <div className={styles.advanced}>
          <Group title="Transition">
            <Row k="Upper surface" v={fmtX(tr.upperX)} />
            <Row k="Lower surface" v={fmtX(tr.lowerX)} />
            <Row k="Criterion" v={`${tr.model} · ${(tr.upperCriterion * 100).toFixed(0)}% upper`} />
          </Group>

          <Group title="Separation">
            <Row k="Upper surface" v={fmtX(sep.upperX)} />
            <Row k="Lower surface" v={fmtX(sep.lowerX)} />
            {sep.bubbleUpper && <Row k="Upper bubble from" v={fmtX(sep.laminarUpperX)} />}
            {sep.reattachmentUpperX >= 0 && <Row k="Reattachment" v={fmtX(sep.reattachmentUpperX)} />}
            <Row k="Chord separated" v={`${sep.percentChordSeparated.toFixed(1)} %`} />
          </Group>

          <Group title="Drag breakdown">
            <Row k="Skin friction" v={readings.forceBreakdown.cdFriction.toFixed(5)} />
            <Row k="Pressure (form)" v={readings.forceBreakdown.cdPressure.toFixed(5)} />
            <Row k="Cl by Kutta–Joukowski" v={readings.forceBreakdown.clKuttaJoukowski.toFixed(4)} />
          </Group>

          <Group title="Stability">
            <Row k="Aerodynamic centre" v={fmtC(readings.stability.aerodynamicCenter)} />
            <Row k="Centre of pressure" v={fmtC(readings.stability.centerOfPressure)} />
            <Row k="Lift-curve slope" v={`${readings.stability.liftSlopePerDeg.toFixed(4)} /°`} />
          </Group>

          <Group title="Pressure">
            <Row k="Minimum Cp" v={`${readings.pressure.cpMin.toFixed(3)} at ${fmtX(readings.pressure.cpMinX)}`} />
            <Row k="Trailing-edge Cp" v={readings.pressure.cpTrailingEdgeUpper.toFixed(3)} />
            <Row k="Pressure recovery" v={readings.pressure.pressureRecovery.toFixed(3)} />
          </Group>

          <Group title="Boundary layer">
            <Row k="Health" v={`${(stall.boundaryLayerHealth * 100).toFixed(0)} %`} />
            <Row k="Worst shape factor H" v={stall.worstShapeFactor.toFixed(3)} />
            <Row k="Wake momentum thickness" v={`${(readings.wake.momentumThickness * 100).toFixed(3)} %c`} />
          </Group>

          <Group title="Wake">
            <Row k="Particles" v={String(readings.wake.particleCount ?? 0)} />
            <Row k="Total circulation" v={(readings.wake.totalCirculation ?? 0).toExponential(2)} />
            <Row k="Length" v={`${(readings.wake.length ?? 0).toFixed(1)} c`} />
          </Group>

          <Group title="Solution">
            <Row k="Confidence" v={`${conv.confidence} %`} />
            <Row k="Iterations" v={`${conv.iterations} / ${conv.maxIterations}`} />
            <Row k="Cp residual" v={conv.residualCp.toExponential(1)} />
            <Row k="Panels" v={String(readings.airfoil.panels)} />
            <Row k="Matrix condition" v={conv.conditionNumber.toFixed(1)} />
          </Group>
        </div>
      )}

      <p className={styles.engineNote}>
        <strong>{conv.confidence}% confidence.</strong> {conv.confidenceSummary}
      </p>
    </>
  );
}

function InfoPanel({ readings, state }) {
  const conv = readings.convergence;
  return (
    <div className={styles.infoPanel}>
      <p>
        Everything on screen comes from a single converged aerodynamic state. The section is
        panelled with {readings.airfoil.panels} linear-strength vortex panels closed by the Kutta
        condition at the trailing edge; a boundary layer is marched over both surfaces and into the
        wake (Thwaites while laminar, Head's entrainment method once turbulent); and the two are
        coupled through a transpiration source distribution until the pressure and the displacement
        thickness agree with each other.
      </p>
      <p>
        The streamlines are not a separate simulation. They are tracer particles advected through
        the same analytically evaluated velocity field the forces were integrated from, so where
        the picture shows the flow leaving the surface is where the boundary layer says it
        separates — currently {fmtX(readings.separation.upperX)} on the upper surface.
      </p>
      <p>
        <strong>Transition</strong> is predicted by the {readings.transition.model} method, which
        integrates the amplification of the most unstable disturbance and hands over to the
        turbulent closure when it has grown by a factor of e<sup>N</sup>. It responds to the
        pressure gradient, which is why the transition point marches toward the leading edge as
        incidence rises.
      </p>
      <p>
        <strong>Drag</strong> comes from the wake momentum thickness by the Squire–Young formula
        rather than from integrating the surface pressure. Profile drag is the small residue left
        after the front and back of the section have very nearly cancelled, so a 0.1%
        discretisation error in the pressure would land as a 20% error in the drag. The friction
        component ({readings.forceBreakdown.cdFriction.toFixed(5)}) is integrated directly from the
        wall shear, where there is no such cancellation.
      </p>
      <p>
        <strong>Stall</strong> is not read off a critical-angle table. It is a continuous index
        combining the separated fraction of the chord, the loss of lift-curve slope against this
        section's own inviscid slope, the displacement and wake growth, and how well the coupling
        converged. That is currently {readings.stall.index.toFixed(2)} —{' '}
        {readings.stall.label.toLowerCase()}.
      </p>
      <p>
        <strong>Confidence: {conv.confidence}%.</strong> {conv.confidenceSummary} The coupled solve
        stopped after {conv.iterations} of {conv.maxIterations} iterations with a pressure residual
        of {conv.residualCp.toExponential(1)}. A direct-mode integral boundary layer has no valid
        solution downstream of separation — that is the Goldstein singularity — so past about 12°
        the trailing-edge region is a model rather than a solve, and the confidence figure falls
        accordingly. It is shown next to every number rather than buried, because the numbers
        themselves stay four-significant-figure precise whether they deserve to be or not.
      </p>
      <p>
        Solve time this update: {state?.timing.total.toFixed(0)} ms
        {state?.mode === 'incremental' ? ' (incremental, warm-started)' : ' (full convergence)'}.
      </p>
    </div>
  );
}

/**
 * The heat-map legend.
 *
 * A colour scale without its range is decoration, so the bar always ships with
 * both ends labelled and the units named. The bar itself is generated from the
 * same baked ramp the wing is painted with, so the two cannot drift apart.
 */
function HeatLegend({ heat }) {
  const { mode, lo, hi, scaleId } = heat;
  return (
    <div className={styles.heatLegend}>
      <span>{formatHeatValue(lo, mode)}</span>
      <div className={styles.heatLegendBar} style={{ background: rampCss(scaleId) }} />
      <span>{formatHeatValue(hi, mode)}</span>
      <span className={styles.heatLegendSymbol}>{mode.symbol}</span>
    </div>
  );
}

/** What the two dropdowns offer, option by option. */
function HeatmapInfo({ heat, onClose }) {
  return (
    <div className={styles.heatmapInfo}>
      <div className={styles.heatmapInfoHeader}>
        <span>{heat ? heat.mode.label : 'Surface heat map'}</span>
        <button type="button" className={styles.heatmapInfoClose} onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <div className={styles.heatmapInfoBody}>
        <p>
          Two dropdowns choose what the colour on the surface means and how it is turned into
          colour. Everything here reads directly off the same converged solution the lift, drag and
          pressure readouts come from, so the colour at a point on the surface and the number in the
          dashboard are the same number. Hover the model to read the local value.
        </p>

        <div className={styles.heatmapInfoSection}>Heat map quantity</div>
        <dl>
          {HEAT_MODES.map((m) => (
            <React.Fragment key={m.id}>
              <dt>
                <span className={styles.heatmapInfoSymbol}>{m.symbol}</span>
                {m.label}
                {m.signed ? ' · signed' : ''}
                {m.robust ? ' · outlier clipped' : ''}
              </dt>
              <dd>{m.describe}</dd>
            </React.Fragment>
          ))}
        </dl>

        <div className={styles.heatmapInfoSection}>Colour scale</div>
        <dl>
          {COLOR_SCALES.map((s) => (
            <React.Fragment key={s.id}>
              <dt>{s.label}{s.diverging ? ' · diverging' : ''}</dt>
              <dd>{s.describe}</dd>
            </React.Fragment>
          ))}
        </dl>

        <p className={styles.heatmapInfoNote}>
          Signed quantities (Cp and wall vorticity) automatically switch to a diverging scale pinned
          symmetrically about zero, so the neutral midpoint of the ramp lands on zero and the two
          poles mean equal and opposite. Vorticity is also clipped for outliers, because the
          stagnation region would otherwise flatten the whole scale onto one colour.
        </p>
      </div>
    </div>
  );
}

function Group({ title, children }) {
  return (
    <div className={styles.advancedGroup}>
      <div className={styles.advancedTitle}>{title}</div>
      {children}
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className={styles.advancedRow}>
      <span>{k}</span>
      <span>{v}</span>
    </div>
  );
}

function Readout({ label, symbol, value, unit }) {
  return (
    <div className={styles.readout}>
      <div className={styles.readoutLabel}>
        {label} <span className={styles.readoutSymbol}>{symbol}</span>
      </div>
      <div className={styles.readoutValue}>
        {value}
        {unit && <span className={styles.readoutUnit}>{unit}</span>}
      </div>
    </div>
  );
}

/** Chordwise station, or an em dash when there isn't one. */
function fmtX(x) {
  return x >= 0 && x <= 1 ? `${x.toFixed(3)} c` : '—';
}
function fmtC(x) {
  return x === null || x === undefined ? '—' : `${x.toFixed(3)} c`;
}

function formatSci(v) {
  if (!isFinite(v) || v === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(v)));
  const mant = v / 10 ** exp;
  return `${mant.toFixed(2)}×10${superscript(exp)}`;
}

function superscript(n) {
  const map = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  return String(n)
    .split('')
    .map((ch) => map[ch] ?? ch)
    .join('');
}

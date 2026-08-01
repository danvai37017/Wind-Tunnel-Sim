/**
 * wake.js — the unsteady discrete-vortex wake.
 *
 * The steady solve already carries a wake: a source line that represents the
 * viscous displacement, which is what the trailing-edge pressure and the drag
 * need. This is the other half — the *vorticity* the airfoil sheds, which is
 * what makes the wake visible and what carries the unsteady history.
 *
 * ## What actually sheds
 *
 * Kelvin's theorem says the circulation around a material contour enclosing both
 * the airfoil and its wake cannot change. So in genuinely steady flow nothing is
 * shed at all: the starting vortex left long ago and the wake is a vortex sheet
 * of zero net strength. Vorticity appears in two circumstances, and this models
 * both:
 *
 *   1. The bound circulation changes — the angle-of-attack slider moves. Then
 *      exactly -dGamma/dt is shed per unit time, which is Kelvin's theorem
 *      stated as an update rule and conserves total circulation to machine
 *      precision by construction.
 *
 *   2. The boundary layers leave the trailing edge with unequal speeds, so the
 *      wake is a shear layer even in the steady state. Its local sheet strength
 *      is the velocity difference, and it is shed as *pairs* of equal and
 *      opposite particles so the total circulation is untouched. This is what
 *      makes a separated wake visibly roll up rather than drift as a smooth
 *      band.
 *
 * ## Numerics
 *
 * Point vortices are singular, so the kernel is desingularised with a finite
 * core radius (Rosenhead-Moore). That is a regularisation of the Green's
 * function, not artificial viscosity: it adds no dissipation, conserves
 * circulation exactly, and converges to the point-vortex result as the core
 * shrinks. Without it two particles that drift close enough exchange arbitrarily
 * large velocities and the integration loses all meaning.
 *
 * Convection is RK4. Particle-particle interaction goes through a Barnes-Hut
 * quadtree above a few hundred particles, which turns the O(N^2) sum into
 * O(N log N); below that the direct sum is faster than building the tree.
 */

import { fieldVelocity } from './panel.js';

/** Vortex core radius, in chords. */
const CORE_RADIUS = 0.012;
/** Barnes-Hut opening angle. Smaller is more accurate and slower. */
const BH_THETA = 0.6;
/** Below this many particles the direct sum beats building a tree. */
const BH_MIN_PARTICLES = 400;
/** Particles this close with the same sign are merged. */
const MERGE_DISTANCE = CORE_RADIUS / 4;
/** Downstream of this the wake is off the end of anything anyone will look at. */
const MAX_DOWNSTREAM = 14;

export function createVortexWake(capacity = 4000) {
  return {
    capacity,
    count: 0,
    x: new Float64Array(capacity),
    y: new Float64Array(capacity),
    g: new Float64Array(capacity), // circulation, clockwise positive
    age: new Float64Array(capacity),
    u: new Float64Array(capacity),
    v: new Float64Array(capacity),
    // Scratch for RK4.
    k1x: new Float64Array(capacity),
    k1y: new Float64Array(capacity),
    k2x: new Float64Array(capacity),
    k2y: new Float64Array(capacity),
    k3x: new Float64Array(capacity),
    k3y: new Float64Array(capacity),
    k4x: new Float64Array(capacity),
    k4y: new Float64Array(capacity),
    tx: new Float64Array(capacity),
    ty: new Float64Array(capacity),
    lastCirculation: null,
    shedTotal: 0,
    mergedTotal: 0,
    tree: null,
  };
}

export function resetVortexWake(W) {
  W.count = 0;
  W.lastCirculation = null;
  W.shedTotal = 0;
  W.mergedTotal = 0;
}

/* ============================================================================
 * Shedding
 * ==========================================================================*/

/**
 * Shed the vorticity generated over a timestep.
 *
 * `circulation` is the current bound circulation (clockwise positive) and
 * `shear` the trailing-edge velocity difference between the two surfaces. Both
 * come straight from the converged state — this module never computes
 * aerodynamics of its own.
 */
export function shedVorticity(W, geo, circulation, shear, dt, alphaRad) {
  if (W.lastCirculation === null) {
    W.lastCirculation = circulation;
    return;
  }

  // Shedding point: just downstream of the trailing edge along the freestream,
  // far enough out that a fresh particle is not sitting inside the body.
  const off = 0.02;
  const px = geo.X[0] + off * Math.cos(alphaRad);
  const py = geo.Y[0] + off * Math.sin(alphaRad);

  // --- Kelvin: whatever the bound circulation lost, the wake gains -----------
  const dGamma = circulation - W.lastCirculation;
  W.lastCirculation = circulation;
  if (Math.abs(dGamma) > 1e-12) addParticle(W, px, py, -dGamma);

  // --- Shear layer: an equal and opposite pair, so the total is unchanged ----
  // The sheet strength of a wake shear layer is the velocity jump across it, so
  // the circulation passing the trailing edge per unit time is
  // (1/2)(Vu^2 - Vl^2) dt. Splitting it across the two sides of the wake gives
  // the pair.
  const roll = 0.5 * shear * dt;
  if (Math.abs(roll) > 1e-9) {
    const nx = -Math.sin(alphaRad);
    const ny = Math.cos(alphaRad);
    const sep = 0.6 * CORE_RADIUS;
    addParticle(W, px + nx * sep, py + ny * sep, roll);
    addParticle(W, px - nx * sep, py - ny * sep, -roll);
  }
}

function addParticle(W, x, y, g) {
  if (W.count >= W.capacity) {
    // Full: retire the oldest particle, but hand its circulation to its nearest
    // survivor rather than deleting it, so Kelvin's theorem still holds.
    let oldest = 0;
    for (let i = 1; i < W.count; i++) if (W.age[i] > W.age[oldest]) oldest = i;
    donate(W, oldest);
    removeAt(W, oldest);
  }
  const i = W.count++;
  W.x[i] = x;
  W.y[i] = y;
  W.g[i] = g;
  W.age[i] = 0;
  W.u[i] = 0;
  W.v[i] = 0;
  W.shedTotal += g;
}

/** Give particle i's circulation to the nearest other particle. */
function donate(W, i) {
  if (W.count < 2) return;
  let best = -1;
  let bestD = Infinity;
  for (let j = 0; j < W.count; j++) {
    if (j === i) continue;
    const d = (W.x[j] - W.x[i]) ** 2 + (W.y[j] - W.y[i]) ** 2;
    if (d < bestD) {
      bestD = d;
      best = j;
    }
  }
  if (best >= 0) W.g[best] += W.g[i];
}

function removeAt(W, i) {
  const last = --W.count;
  if (i !== last) {
    W.x[i] = W.x[last];
    W.y[i] = W.y[last];
    W.g[i] = W.g[last];
    W.age[i] = W.age[last];
    W.u[i] = W.u[last];
    W.v[i] = W.v[last];
  }
}

/* ============================================================================
 * Barnes-Hut quadtree
 * ==========================================================================*/

/**
 * A flat-array quadtree over the particles.
 *
 * Each node stores the total circulation and the circulation-weighted centroid
 * of its contents, which is the monopole approximation: a distant cluster is
 * replaced by a single vortex at its centre of vorticity. A node is used whole
 * when its size divided by the distance to the evaluation point is below the
 * opening angle, and opened otherwise.
 *
 * Storing the tree in typed arrays rather than objects matters here: this is
 * rebuilt every frame, and an object-per-node tree spends more time in the
 * allocator than in the physics.
 */
function buildTree(W) {
  const n = W.count;
  const cap = Math.max(64, 4 * n + 16);
  let t = W.tree;
  if (!t || t.cap < cap) {
    t = W.tree = {
      cap,
      cx: new Float64Array(cap),
      cy: new Float64Array(cap),
      half: new Float64Array(cap),
      gsum: new Float64Array(cap),
      gabs: new Float64Array(cap),
      wx: new Float64Array(cap),
      wy: new Float64Array(cap),
      child: new Int32Array(cap * 4),
      body: new Int32Array(cap),
      count: 0,
    };
  }
  t.count = 0;
  if (n === 0) return t;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (W.x[i] < minX) minX = W.x[i];
    if (W.x[i] > maxX) maxX = W.x[i];
    if (W.y[i] < minY) minY = W.y[i];
    if (W.y[i] > maxY) maxY = W.y[i];
  }
  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  const half = Math.max(maxX - minX, maxY - minY) * 0.5 + 1e-6;

  const newNode = (x, y, h) => {
    const i = t.count++;
    t.cx[i] = x;
    t.cy[i] = y;
    t.half[i] = h;
    t.gsum[i] = 0;
    t.gabs[i] = 0;
    t.wx[i] = 0;
    t.wy[i] = 0;
    t.body[i] = -1;
    t.child[i * 4] = -1;
    t.child[i * 4 + 1] = -1;
    t.child[i * 4 + 2] = -1;
    t.child[i * 4 + 3] = -1;
    return i;
  };

  newNode(cx, cy, half);

  const quadrant = (node, px, py) =>
    (px > t.cx[node] ? 1 : 0) | (py > t.cy[node] ? 2 : 0);

  const childCentre = (node, q) => {
    const h = t.half[node] * 0.5;
    return [t.cx[node] + (q & 1 ? h : -h), t.cy[node] + (q & 2 ? h : -h), h];
  };

  for (let i = 0; i < n; i++) {
    let node = 0;
    let depth = 0;
    for (;;) {
      // Accumulate the monopole on the way down. |g| weights the centroid so
      // that a cancelling pair does not place its centre at infinity.
      const a = Math.abs(W.g[i]);
      t.gsum[node] += W.g[i];
      t.gabs[node] += a;
      t.wx[node] += a * W.x[i];
      t.wy[node] += a * W.y[i];

      const isLeaf = t.child[node * 4] < 0;
      if (isLeaf && t.body[node] < 0) {
        t.body[node] = i;
        break;
      }
      if (isLeaf) {
        // Occupied leaf: push the resident down before inserting.
        const resident = t.body[node];
        t.body[node] = -1;
        if (depth > 24 || t.count + 8 > t.cap) {
          // Coincident particles, or out of room. Leaving the resident on the
          // node is safe — it is still counted in the monopole above.
          t.body[node] = resident;
          break;
        }
        const qr = quadrant(node, W.x[resident], W.y[resident]);
        const [rx, ry, rh] = childCentre(node, qr);
        const rn = newNode(rx, ry, rh);
        t.child[node * 4 + qr] = rn;
        t.body[rn] = resident;
        const ar = Math.abs(W.g[resident]);
        t.gsum[rn] = W.g[resident];
        t.gabs[rn] = ar;
        t.wx[rn] = ar * W.x[resident];
        t.wy[rn] = ar * W.y[resident];
      }

      const q = quadrant(node, W.x[i], W.y[i]);
      let c = t.child[node * 4 + q];
      if (c < 0) {
        if (t.count + 2 > t.cap) break;
        const [nx2, ny2, nh] = childCentre(node, q);
        c = newNode(nx2, ny2, nh);
        t.child[node * 4 + q] = c;
      }
      node = c;
      depth++;
    }
  }
  return t;
}

/** Velocity induced at (px, py) by every particle, via the tree. */
function treeVelocity(W, t, px, py, skip, out) {
  const core2 = CORE_RADIUS * CORE_RADIUS;
  let u = 0;
  let v = 0;
  const stack = TREE_STACK;
  let sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    const node = stack[--sp];
    if (t.gabs[node] === 0) continue;

    const isLeaf = t.child[node * 4] < 0 && t.body[node] >= 0;
    if (isLeaf) {
      const i = t.body[node];
      if (i === skip) continue;
      const dx = px - W.x[i];
      const dy = py - W.y[i];
      const r2 = dx * dx + dy * dy + core2;
      const f = W.g[i] / (2 * Math.PI * r2);
      u += f * dy;
      v -= f * dx;
      continue;
    }

    const gx = t.wx[node] / t.gabs[node];
    const gy = t.wy[node] / t.gabs[node];
    const dx = px - gx;
    const dy = py - gy;
    const r2 = dx * dx + dy * dy;
    const size = 2 * t.half[node];

    if (size * size < BH_THETA * BH_THETA * r2) {
      // Far enough: use the cluster's monopole.
      const f = t.gsum[node] / (2 * Math.PI * (r2 + core2));
      u += f * dy;
      v -= f * dx;
      continue;
    }
    // Too close: open it. The resident body of an internal node (only present
    // for coincident particles) is handled here so nothing is dropped.
    if (t.body[node] >= 0 && t.body[node] !== skip) {
      const i = t.body[node];
      const bx = px - W.x[i];
      const by = py - W.y[i];
      const br2 = bx * bx + by * by + core2;
      const bf = W.g[i] / (2 * Math.PI * br2);
      u += bf * by;
      v -= bf * bx;
    }
    for (let q = 0; q < 4; q++) {
      const c = t.child[node * 4 + q];
      if (c >= 0 && sp < stack.length) stack[sp++] = c;
    }
  }

  out[0] = u;
  out[1] = v;
  return out;
}

const TREE_STACK = new Int32Array(512);

/** Direct O(N^2) sum, used below the tree threshold. */
function directVelocity(W, px, py, skip, out) {
  const core2 = CORE_RADIUS * CORE_RADIUS;
  let u = 0;
  let v = 0;
  for (let i = 0; i < W.count; i++) {
    if (i === skip) continue;
    const dx = px - W.x[i];
    const dy = py - W.y[i];
    const r2 = dx * dx + dy * dy + core2;
    const f = W.g[i] / (2 * Math.PI * r2);
    u += f * dy;
    v -= f * dx;
  }
  out[0] = u;
  out[1] = v;
  return out;
}

/**
 * Velocity induced by the particle wake alone at an arbitrary point.
 *
 * Clockwise-positive circulation, matching the panel solver's convention, so a
 * particle of strength g induces g/(2 pi r) in the clockwise sense.
 */
export function wakeVelocity(W, px, py, out, tree = null) {
  if (W.count === 0) {
    out[0] = 0;
    out[1] = 0;
    return out;
  }
  if (tree) return treeVelocity(W, tree, px, py, -1, out);
  return directVelocity(W, px, py, -1, out);
}

/* ============================================================================
 * Convection
 * ==========================================================================*/

const VEL = new Float64Array(2);
const VEL2 = new Float64Array(2);

/**
 * Advance the wake by dt with classical RK4.
 *
 * Fourth order matters more here than it looks: a vortex pair orbits, and a
 * first-order integrator makes the orbit spiral outward at a rate proportional
 * to the timestep — which is indistinguishable, on screen, from the physical
 * spreading of a real wake. RK2 is the documented fallback; explicit Euler is
 * never used.
 */
export function convectWake(W, sys, panelWake, sol, alphaRad, dt, opts = {}) {
  const n = W.count;
  if (n === 0) return;
  const order = opts.order ?? 4;

  const tree = () => (W.count >= BH_MIN_PARTICLES ? buildTree(W) : null);

  const velocityAt = (px, py, skip, t, out) => {
    // Body sheets and the freestream, analytically...
    fieldVelocity(sys, panelWake, sol, alphaRad, px, py, out, opts.fieldCache ?? null);
    // ...plus the other particles.
    if (t) treeVelocity(W, t, px, py, skip, VEL2);
    else directVelocity(W, px, py, skip, VEL2);
    out[0] += VEL2[0];
    out[1] += VEL2[1];
    return out;
  };

  const stage = (kx, ky, srcX, srcY) => {
    const t = tree();
    for (let i = 0; i < n; i++) {
      velocityAt(srcX[i], srcY[i], i, t, VEL);
      kx[i] = VEL[0];
      ky[i] = VEL[1];
    }
  };

  const { x, y, k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y, tx, ty } = W;

  // The tree is rebuilt per stage because the positions move; for the
  // intermediate stages the particles themselves are still at x, y, which is
  // close enough that the far-field monopoles are unchanged — only the stage
  // positions differ, and those are the evaluation points.
  stage(k1x, k1y, x, y);
  if (order >= 2) {
    for (let i = 0; i < n; i++) {
      tx[i] = x[i] + 0.5 * dt * k1x[i];
      ty[i] = y[i] + 0.5 * dt * k1y[i];
    }
    stage(k2x, k2y, tx, ty);
  }
  if (order >= 4) {
    for (let i = 0; i < n; i++) {
      tx[i] = x[i] + 0.5 * dt * k2x[i];
      ty[i] = y[i] + 0.5 * dt * k2y[i];
    }
    stage(k3x, k3y, tx, ty);
    for (let i = 0; i < n; i++) {
      tx[i] = x[i] + dt * k3x[i];
      ty[i] = y[i] + dt * k3y[i];
    }
    stage(k4x, k4y, tx, ty);
  }

  for (let i = 0; i < n; i++) {
    let ux;
    let uy;
    if (order >= 4) {
      ux = (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) / 6;
      uy = (k1y[i] + 2 * k2y[i] + 2 * k3y[i] + k4y[i]) / 6;
    } else if (order >= 2) {
      ux = k2x[i];
      uy = k2y[i];
    } else {
      ux = k1x[i];
      uy = k1y[i];
    }
    W.x[i] += dt * ux;
    W.y[i] += dt * uy;
    W.u[i] = ux;
    W.v[i] = uy;
    W.age[i] += dt;
  }

  prune(W);
  mergeParticles(W);
}

/** Retire particles that have left the region of interest. */
function prune(W) {
  for (let i = W.count - 1; i >= 0; i--) {
    if (W.x[i] > MAX_DOWNSTREAM || W.x[i] < -6 || Math.abs(W.y[i]) > 8) {
      // Far downstream the circulation has left the control volume for good, so
      // it is dropped rather than donated — that is the physical statement, not
      // a violation of Kelvin's theorem, whose contour has to grow with the wake.
      W.shedTotal -= W.g[i];
      removeAt(W, i);
    }
  }
}

/**
 * Merge same-sign neighbours to keep the particle count bounded.
 *
 * The merged particle carries the sum of the circulations at the circulation-
 * weighted centroid, which conserves both the total circulation and the first
 * moment of vorticity exactly — so the far field is unchanged to dipole order
 * and the wake does not visibly shift when a merge happens.
 *
 * Only same-sign pairs are merged: merging opposite signs would conserve
 * circulation but destroy the dipole that makes a shear layer roll up.
 */
function mergeParticles(W) {
  const d2 = MERGE_DISTANCE * MERGE_DISTANCE;
  for (let i = W.count - 1; i >= 1; i--) {
    const gi = W.g[i];
    if (gi === 0) continue;
    for (let j = i - 1; j >= 0; j--) {
      const gj = W.g[j];
      if (gj === 0 || gi * gj <= 0) continue;
      const dx = W.x[i] - W.x[j];
      const dy = W.y[i] - W.y[j];
      if (dx * dx + dy * dy > d2) continue;
      const gt = gi + gj;
      W.x[j] = (W.x[i] * gi + W.x[j] * gj) / gt;
      W.y[j] = (W.y[i] * gi + W.y[j] * gj) / gt;
      W.g[j] = gt;
      W.age[j] = Math.max(W.age[i], W.age[j]);
      removeAt(W, i);
      W.mergedTotal++;
      break;
    }
  }
}

/** Wake diagnostics for the output state. */
export function wakeDiagnostics(W) {
  let total = 0;
  let maxX = 0;
  let absTotal = 0;
  for (let i = 0; i < W.count; i++) {
    total += W.g[i];
    absTotal += Math.abs(W.g[i]);
    const d = W.x[i];
    if (d > maxX) maxX = d;
  }
  return {
    particleCount: W.count,
    totalCirculation: total,
    absCirculation: absTotal,
    length: maxX,
    merged: W.mergedTotal,
    // How well Kelvin's theorem is holding: the shed total is tracked
    // independently of the sum over particles, so any drift between them is
    // accumulated conservation error.
    conservationError: Math.abs(total - W.shedTotal),
  };
}

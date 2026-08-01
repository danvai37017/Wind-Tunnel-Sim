/**
 * Render smoke test — executes a real 3D frame.
 *
 * render3D runs inside requestAnimationFrame, where a thrown exception is
 * invisible: the bundle builds, the page loads, and the canvas silently stops.
 * A missing `solverToView` import did exactly that — the wake layer (on by
 * default) threw every frame right after the background was drawn, so the 3D
 * view rendered nothing at all while the build stayed green.
 *
 * So this drives the actual draw path: every geometry, every heat mode, every
 * layer on, across the angle range, against a stub 2D context that also flags
 * any non-finite coordinate reaching a canvas call.
 *
 * WindTunnel.jsx is JSX and imports a CSS module, so it is loaded through
 * Vite's SSR pipeline rather than by node directly.
 */
import { createServer } from 'vite';
import { createFlow, updateFlow, fillField } from '../src/flow.js';
import { HEAT_MODES, getScale, heatDomain } from '../src/viz/heatmap.js';

const VIEW3D_W = 840;
const VIEW3D_H = 430;

let nanArgs = 0;
const chk = (...vs) => {
  for (const v of vs) if (typeof v === 'number' && !Number.isFinite(v)) nanArgs++;
};

function stubCtx() {
  const noop = () => {};
  return {
    canvas: { width: VIEW3D_W, height: VIEW3D_H },
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',
    fillRect: chk, rect: chk, roundRect: chk,
    beginPath: noop, closePath: noop, fill: noop, stroke: noop, save: noop, restore: noop,
    moveTo: chk, lineTo: chk,
    arc: (x, y, r, a, b) => chk(x, y, r, a, b),
    fillText: (t, x, y) => chk(x, y),
    measureText: (t) => ({ width: String(t).length * 6 }),
    createLinearGradient: (...a) => {
      chk(...a);
      return {
        addColorStop: (o, c) => {
          chk(o);
          if (/NaN|undefined/.test(String(c))) nanArgs++;
        },
      };
    },
  };
}

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' });
const { __internals } = await server.ssrLoadModule('/src/WindTunnel.jsx');
const { render3D, createStreaks, createCamera } = __internals;

const geometries = [
  { geometry: 'naca', naca: '2412' },
  { geometry: 'naca', naca: '23012' },
  { geometry: 'clarky' },
  { geometry: 'flatplate' },
];

// Everything on, which is the configuration most likely to throw.
const ALL_LAYERS = {
  streamlines: true, vectors: true, contours: true,
  wake: true, stagnation: true, separation: true,
};

let failures = 0;
for (const g of geometries) {
  for (const alphaDeg of [-8, 0, 6, 14]) {
    const F = createFlow();
    const state = updateFlow(F, { ...g, alphaDeg, airspeed: 30, chord: 0.06, span: 0.24 });
    if (!state) { console.log(`  FAIL no state ${JSON.stringify(g)} a=${alphaDeg}`); failures++; continue; }

    let guard = 0;
    while (!fillField(F, 50) && guard++ < 500);

    // Tracers with real trails, so the ribbon path is exercised rather than skipped.
    const St = createStreaks();
    for (let i = 0; i < 60; i++) {
      St.x[i] = 60 + i; St.y[i] = 40 + (i % 70); St.count[i] = 6; St.head[i] = 5; St.sz[i] = (i % 10) / 10 - 0.5;
      for (let k = 0; k < 6; k++) { St.tx[i * 16 + k] = 60 + i - k; St.ty[i * 16 + k] = 40 + (i % 70); }
    }

    const errs = [];
    for (const m of HEAT_MODES) {
      const values = m.values(state);
      const scale = getScale(m.signed ? 'coolwarm' : 'spectral');
      const { lo, hi } = heatDomain(values, m, scale);
      const viz = {
        heat: { values, mode: m, scale, scaleId: scale.id, lo, hi },
        layers: ALL_LAYERS,
        // Hover parked over the section, so the pick and tooltip paths run.
        hover: { active: true, x: VIEW3D_W / 2, y: VIEW3D_H / 2, px: 0, py: 0, panel: -1, value: NaN, xc: 0, surface: '' },
      };
      try {
        render3D(stubCtx(), F, St, createCamera(), 4 * F.nCells, viz);
      } catch (e) {
        errs.push(`${m.id}: ${e.message}`);
      }
    }
    // And once with the heat map off, which is a separate branch.
    try {
      render3D(stubCtx(), F, St, createCamera(), 4 * F.nCells, {
        heat: null, layers: ALL_LAYERS,
        hover: { active: false, x: 0, y: 0, px: 0, py: 0, panel: -1, value: NaN, xc: 0, surface: '' },
      });
    } catch (e) {
      errs.push(`heat-off: ${e.message}`);
    }

    if (errs.length) failures++;
    const name = `${(g.naca ? 'NACA ' + g.naca : g.geometry).padEnd(11)} a=${String(alphaDeg).padStart(3)}`;
    console.log(`  ${errs.length ? 'FAIL' : 'ok  '} ${name}  poly ${String(F.polyCount).padStart(3)}${errs.length ? '  ' + errs.join(' | ') : ''}`);
  }
}

await server.close();
console.log(`\n${failures === 0 ? 'render path clean' : failures + ' FAILURES'}; ${nanArgs} non-finite canvas args`);
process.exit(failures === 0 && nanArgs === 0 ? 0 : 1);

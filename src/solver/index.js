/**
 * index.js — the solver's public face: one authoritative aerodynamic state.
 *
 * Every user-visible quantity in the application is read off the object this
 * produces. There is no second opinion anywhere: the lift, the drag, the
 * pressure distribution the plot draws, the transition markers, the stall
 * banner, and the velocity field the streamlines are advected through all come
 * from the same converged solve. If they ever disagreed it would be a bug, not a
 * modelling choice.
 *
 * ## Cost model
 *
 * The three operational modes exist because the three kinds of change cost
 * wildly different amounts:
 *
 *   airfoil changed     rebuild the panel geometry, the influence matrices and
 *                       the LU factorisation, then converge from scratch.
 *                       O(N^3), tens to hundreds of milliseconds.
 *
 *   airspeed or chord   Reynolds number moves, so the boundary layer changes but
 *                       no matrix does. Converge from the previous state.
 *
 *   angle of attack     rebuild the wake influence only (no factorisation), then
 *                       take a couple of coupling iterations warm-started from
 *                       the previous converged displacement thickness. Single-
 *                       digit milliseconds.
 *
 *   span                changes nothing at all in two dimensions. The forces are
 *                       rescaled and the solver is not run.
 */

import { parseNacaCode } from './naca.js';
import {
  parseSection,
  sectionKey,
  sectionZeroLiftAngle,
  sectionMaxCamber,
  DEFAULT_GEOMETRY,
} from './sections.js';
import { buildBody, panelGeometry } from './geometry.js';
import {
  buildPanelSystem,
  buildWake,
  buildWakeInfluence,
  createPanelSolution,
  solvePanels,
  fieldVelocity,
  buildFieldCache,
} from './panel.js';
import { createBLState, solveBoundaryLayer, H_TURB_SEP } from './boundaryLayer.js';
import { createCouplingState, converge, TOL_CP, MAX_ITERATIONS } from './coupling.js';
import { integrateForces, stabilityDerivatives } from './forces.js';
import {
  stallIndex,
  confidence,
  convergenceQuality,
  slopeLoss,
  STALL_LABELS,
} from './diagnostics.js';
import {
  createVortexWake,
  resetVortexWake,
  shedVorticity,
  convectWake,
  wakeDiagnostics,
  wakeVelocity,
} from './wake.js';

/** ISA sea-level air, the default fluid. */
export const RHO_AIR = 1.225; // kg/m^3
export const NU_AIR = 1.48e-5; // m^2/s

/** Finite-difference step for the lift-curve slope and aerodynamic centre. */
const DALPHA = (1 * Math.PI) / 180;

const DEFAULTS = {
  // Which section to panel. 'naca' reads `naca`; the other shapes in
  // sections.js are self-describing. Defaulting to 'naca' keeps every existing
  // caller — which passes only `naca` — on exactly the path it was on.
  geometry: DEFAULT_GEOMETRY,
  naca: '2412',
  alphaDeg: 5,
  airspeed: 30, // m/s
  chord: 0.06, // m
  span: 0.24, // m — affects reported forces only
  rho: RHO_AIR,
  nu: NU_AIR,
  panels: 220,
  maxPanels: 500,
  transitionModel: 'en',
  nCrit: 9,
};

export class AeroSolver {
  constructor(options = {}) {
    this.options = { ...DEFAULTS, ...options };
    this.inputs = null;
    this.state = null;

    // Cached, keyed by what invalidates them.
    this._spec = null;
    this._body = null;
    this._geo = null;
    this._sys = null;
    this._wake = null;
    this._wakeInf = null;
    this._wakeAlpha = NaN;
    this._sol = null;
    this._bl = null;
    this._cpl = null;
    this._fieldCache = null;
    this.generation = 0;
    // Auxiliary state at alpha - DALPHA, for the stability derivatives.
    this._solAux = null;
    this._blAux = null;
    this._cplAux = null;

    this.vortexWake = createVortexWake(options.wakeCapacity ?? 3500);
    this.timing = { geometry: 0, matrix: 0, solve: 0, total: 0 };
    this.buildError = null;
  }

  /* ==========================================================================
   * Update
   * ========================================================================*/

  /**
   * Bring the state up to date with new inputs.
   *
   * Returns the authoritative state, or null if the geometry could not be
   * built. `opts.force` re-converges from scratch even if nothing changed.
   */
  update(inputs = {}, opts = {}) {
    const next = { ...this.options, ...this.inputs, ...inputs };
    const prev = this.inputs;
    const t0 = now();

    // Keyed on the section identity rather than on the NACA code alone, so
    // switching to Clark Y or the flat plate invalidates the matrices the same
    // way a new code does — and so a *re-render* that produces an equal spec
    // does not.
    const geometryChanged =
      !prev ||
      sectionKey(next) !== sectionKey(prev) ||
      next.panels !== prev.panels ||
      next.maxPanels !== prev.maxPanels ||
      !this._sys;

    const reynolds = (next.airspeed * next.chord) / next.nu;
    const prevRe = prev ? (prev.airspeed * prev.chord) / prev.nu : 0;
    const reChanged = !prev || Math.abs(Math.log(reynolds / Math.max(prevRe, 1e-9))) > 0.02;
    const alphaChanged = !prev || next.alphaDeg !== prev.alphaDeg;
    const modelChanged =
      !prev || next.transitionModel !== prev.transitionModel || next.nCrit !== prev.nCrit;

    // Span is a reporting quantity in two dimensions. If it is the only thing
    // that moved, rescale the forces and return — no solve, per the spec.
    if (
      !opts.force &&
      this.state &&
      !geometryChanged &&
      !reChanged &&
      !alphaChanged &&
      !modelChanged
    ) {
      this.inputs = next;
      if (next.span !== prev.span || next.rho !== prev.rho) {
        this.state = withScaledForces(this.state, next);
      }
      return this.state;
    }

    this.inputs = next;

    /* ---- Geometry and matrices ------------------------------------------- */
    if (geometryChanged) {
      const tg = now();
      const parsed = parseSection(next);
      if (!parsed.ok) {
        this.buildError = parsed.error;
        return this.state;
      }
      this._spec = parsed;
      this._body = buildBody(parsed, {
        panels: next.panels,
        maxPanels: next.maxPanels,
      });
      this._geo = panelGeometry(this._body);
      this.timing.geometry = now() - tg;

      if (!this._geo) {
        this.buildError = 'Degenerate geometry: duplicate panel nodes.';
        this._sys = null;
        return this.state;
      }

      const tm = now();
      this._sys = buildPanelSystem(this._geo, { gmresThreshold: next.gmresThreshold });
      this.timing.matrix = now() - tm;

      if (!this._sys) {
        // Per the specification, a Kutta condition that cannot be satisfied is a
        // fatal error for the inviscid core. The previous state is kept so the
        // display does not blank out.
        this.buildError =
          'The panel system is singular for this geometry — the Kutta condition cannot be satisfied.';
        return this.state;
      }
      this.buildError = null;

      this._wakeAlpha = NaN;
      this._sol = null;
      this._bl = null;
      this._cpl = null;
      resetVortexWake(this.vortexWake);
    }

    if (!this._sys) return this.state;

    const alphaRad = (next.alphaDeg * Math.PI) / 180;

    /* ---- Wake geometry: cheap, and only the right-hand side depends on it -- */
    if (!this._wakeInf || !nearlyEqual(this._wakeAlpha, alphaRad, 1e-4)) {
      this._wake = buildWake(this._geo, alphaRad);
      this._wakeInf = buildWakeInfluence(this._sys, this._wake);
      this._wakeAlpha = alphaRad;
    }

    const nw = this._wake.n;
    if (!this._sol) {
      this._sol = createPanelSolution(this._sys, nw);
      this._bl = createBLState(this._geo.n, nw);
      this._cpl = createCouplingState(this._geo.n, nw);
      this._solAux = createPanelSolution(this._sys, nw);
      this._blAux = createBLState(this._geo.n, nw);
      this._cplAux = createCouplingState(this._geo.n, nw);
    }

    /* ---- Converge --------------------------------------------------------- */
    // Full convergence whenever the problem itself changed; a short warm-started
    // pass when only the angle nudged, which is what a slider drag produces.
    const fresh = geometryChanged || modelChanged || opts.force;
    // Incremental mode assumes the previous state is a good starting point and a
    // couple of iterations will track a small perturbation. That holds for
    // attached flow and fails badly once the flow separates, where the coupling
    // has no fixed point to track and three iterations land on an essentially
    // arbitrary point of a limit cycle — measured at 20 degrees on NACA 0012, an
    // incremental step reported a drag coefficient of 4.3 against the 0.29 the
    // fully converged solve gives. So separation forces the full path, and pays
    // for it out of the full-convergence budget rather than the incremental one.
    const wasSeparated = (this.state?.separation.percentChordSeparated ?? 0) > 2;
    const mode =
      opts.mode ??
      (fresh || reChanged || wasSeparated || Math.abs(next.alphaDeg - (prev?.alphaDeg ?? 0)) > 2
        ? 'full'
        : 'incremental');

    const blOpts = { transitionModel: next.transitionModel, nCrit: next.nCrit };
    const ts = now();
    const report = converge(this._sys, this._wakeInf, this._sol, this._bl, this._cpl, alphaRad, reynolds, {
      ...blOpts,
      // Separated states reset rather than warm-start. Past separation the loop
      // has no fixed point to converge to, so what it returns depends on where
      // it started — warm-starting makes the answer at a given angle depend on
      // the path taken to reach it, which breaks both reproducibility and the
      // regression suite. Discarding the warm start costs iterations and buys
      // determinism.
      reset: fresh || wasSeparated,
      maxIterations: mode === 'incremental' ? (opts.incrementalIterations ?? 3) : MAX_ITERATIONS,
    });

    const forces = integrateForces(this._geo, this._sol, this._bl, alphaRad);

    /* ---- Auxiliary solve for the stability derivatives -------------------- */
    // A second converged state one degree below, warm-started from this one, so
    // dCl/dalpha and dCm/dCl are differentiated from the *viscous* solution
    // rather than assumed. The warm start makes this a few iterations, not a
    // second full convergence.
    const alphaAux = alphaRad - DALPHA;
    // The auxiliary state reuses the main wake geometry rather than rebuilding
    // it one degree away. Rebuilding costs about as much as the whole auxiliary
    // solve, and the wake source line's effect on lift is second-order — while
    // holding the wake fixed actually makes the finite difference cleaner, since
    // both states then differ only in the quantity being differentiated.
    this._solAux.sigma.set(this._sol.sigma);
    this._solAux.sigmaWake.set(this._sol.sigmaWake);
    this._blAux.transitionS.set(this._bl.transitionS);
    const reportAux = converge(
      this._sys, this._wakeInf, this._solAux, this._blAux, this._cplAux, alphaAux, reynolds,
      { ...blOpts, maxIterations: mode === 'incremental' ? 3 : 6 }
    );
    const forcesAux = integrateForces(this._geo, this._solAux, this._blAux, alphaAux);
    const stability = stabilityDerivatives(forces, forcesAux, DALPHA);

    // Inviscid slope for the same pair, so the lift-slope loss is measured
    // against this section's own potential-flow slope rather than against 2 pi.
    const inviscidSlope = this._inviscidSlope(alphaRad);

    // Panel summaries for fast field sampling. Rebuilt with the state, so the
    // visualiser never samples a field that disagrees with the dashboard.
    this._fieldCache = buildFieldCache(this._sys, this._wake, this._sol);
    this.generation = (this.generation ?? 0) + 1;

    this.timing.solve = now() - ts;
    this.timing.total = now() - t0;

    this.state = this._assemble(next, reynolds, alphaRad, forces, report, reportAux, stability, inviscidSlope, mode);
    return this.state;
  }

  /**
   * Inviscid lift-curve slope per radian at this angle: two back-substitutions
   * with the source distribution zeroed. Microseconds.
   */
  _inviscidSlope(alphaRad) {
    const sol = this._solAux;
    const saveSigma = sol.sigma.slice();
    const saveWake = sol.sigmaWake.slice();
    sol.sigma.fill(0);
    sol.sigmaWake.fill(0);
    solvePanels(this._sys, this._wakeInf, sol, alphaRad);
    const clHi = 2 * sol.circulation;
    solvePanels(this._sys, this._wakeInf, sol, alphaRad - DALPHA);
    const clLo = 2 * sol.circulation;
    sol.sigma.set(saveSigma);
    sol.sigmaWake.set(saveWake);
    return (clHi - clLo) / DALPHA;
  }

  /* ==========================================================================
   * Wake particles
   * ========================================================================*/

  /**
   * Advance the unsteady vortex wake by `dt` seconds of physical time.
   *
   * Driven by the animation loop, not by `update` — the steady state does not
   * depend on it, and the wake must keep evolving while the controls are still.
   */
  advanceWake(dt, opts = {}) {
    if (!this._sys || !this.state) return;
    const alphaRad = (this.inputs.alphaDeg * Math.PI) / 180;
    // Nondimensional time: the solve is on V_inf = chord = 1, so a second of
    // wall clock advances the wake by V*dt/c chords.
    const tau = (dt * this.inputs.airspeed) / this.inputs.chord;
    const step = Math.min(tau, 0.12); // keep RK4 inside its stability envelope

    const n = this._geo.n;
    const vUpper = Math.abs(this._sol.ue[n - 1]);
    const vLower = Math.abs(this._sol.ue[0]);
    const shear = vUpper * vUpper - vLower * vLower;

    shedVorticity(this.vortexWake, this._geo, this._sol.circulation, shear, step, alphaRad);
    convectWake(this.vortexWake, this._sys, this._wake, this._sol, alphaRad, step, {
      ...opts,
      fieldCache: this._fieldCache,
    });

    if (this.state) this.state.wake = { ...this.state.wake, ...wakeDiagnostics(this.vortexWake) };
  }

  /* ==========================================================================
   * Field sampling
   * ========================================================================*/

  /**
   * Velocity at an arbitrary point, in chord-normalised coordinates with the
   * section at zero rotation and the freestream at incidence.
   *
   * Strictly analytic: the panel sheets by Biot-Savart, the wake source line the
   * same way, and the vortex particles through the same desingularised kernel
   * the wake convects itself with. Nothing is differenced and nothing is
   * interpolated off a grid, so the field the visualiser samples is exactly the
   * field the forces were integrated from.
   */
  sampleVelocity(x, y, out = [0, 0], includeParticles = true) {
    if (!this._sys || !this._sol) {
      out[0] = 1;
      out[1] = 0;
      return out;
    }
    const alphaRad = (this.inputs.alphaDeg * Math.PI) / 180;
    fieldVelocity(this._sys, this._wake, this._sol, alphaRad, x, y, out, this._fieldCache);
    if (includeParticles && this.vortexWake.count > 0) {
      const w = wakeVelocity(this.vortexWake, x, y, SCRATCH2);
      out[0] += w[0];
      out[1] += w[1];
    }
    return out;
  }

  /** True if (x, y) is inside the section. Used to mask the field. */
  isInside(x, y) {
    const g = this._geo;
    if (!g) return false;
    let inside = false;
    for (let i = 0, j = g.n - 1; i < g.n; j = i++) {
      const yi = g.Y[i];
      const yj = g.Y[j];
      if (yi > y !== yj > y && x < ((g.X[j] - g.X[i]) * (y - yi)) / (yj - yi) + g.X[i]) {
        inside = !inside;
      }
    }
    return inside;
  }

  get geometry() {
    return this._geo;
  }
  get panelSystem() {
    return this._sys;
  }
  get sourceWake() {
    return this._wake;
  }

  /* ==========================================================================
   * State assembly
   * ========================================================================*/

  _assemble(inputs, reynolds, alphaRad, forces, report, reportAux, stability, inviscidSlope, mode) {
    const geo = this._geo;
    const bl = this._bl;
    const sol = this._sol;
    const n = geo.n;
    const spec = this._spec;

    /* ---- Separation extent ------------------------------------------------ */
    // Measured as the fraction of the *wetted surface* that is separated, which
    // is the quantity the stall index wants — a section can be separated over
    // half the upper surface while the lower is entirely attached.
    let separatedLength = 0;
    let worstH = 0;
    let minCf = Infinity;
    for (let i = 0; i < n; i++) {
      if (bl.state[i] === 2) separatedLength += geo.len[i];
      if (bl.H[i] > worstH) worstH = bl.H[i];
      if (bl.cf[i] < minCf) minCf = bl.cf[i];
    }
    const separationFraction = separatedLength / geo.perimeter;

    const sepU = bl.upper.separationX;
    const sepL = bl.lower.separationX;
    const chordSeparated = Math.max(sepU >= 0 ? 1 - sepU : 0, sepL >= 0 ? 1 - sepL : 0);

    /* ---- Stall index ------------------------------------------------------ */
    const quality = convergenceQuality(report.converged, report.residualCp, TOL_CP);
    const wakeThetaEnd = bl.thetaW[bl.nw - 1];
    const stall = stallIndex({
      separationFraction: chordSeparated,
      liftSlopeLoss: slopeLoss(stability.liftSlopePerRad, inviscidSlope),
      displacementTE: Math.max(bl.upper.dstarTE, bl.lower.dstarTE),
      wakeTheta: wakeThetaEnd,
      convergenceQuality: quality,
    });

    /* ---- Stall margin ----------------------------------------------------- */
    // Extrapolated from how fast the index is climbing with incidence, using the
    // auxiliary state one degree down. No critical-angle table involved.
    const stallAux = stallIndex({
      separationFraction: Math.max(
        this._blAux.upper.separationX >= 0 ? 1 - this._blAux.upper.separationX : 0,
        this._blAux.lower.separationX >= 0 ? 1 - this._blAux.lower.separationX : 0
      ),
      liftSlopeLoss: 0,
      displacementTE: Math.max(this._blAux.upper.dstarTE, this._blAux.lower.dstarTE),
      wakeTheta: this._blAux.thetaW[this._blAux.nw - 1],
      convergenceQuality: convergenceQuality(reportAux.converged, reportAux.residualCp, TOL_CP),
    });
    const dIndexDAlpha = (stall.index - stallAux.index) / 1; // per degree
    const stallMargin =
      dIndexDAlpha > 1e-4 ? Math.max(0, (0.7 - stall.index) / dIndexDAlpha) : null;

    /* ---- Confidence ------------------------------------------------------- */
    const conf = confidence({
      converged: report.converged,
      residualCp: report.residualCp,
      iterations: report.iterations,
      maxIterations: MAX_ITERATIONS,
      separationFraction: chordSeparated,
      alphaDeg: inputs.alphaDeg,
      reynolds,
      conditionNumber: this._sys.condition,
      sigmaClamped: bl.sigmaClamped ?? 0,
      bubbleBurst: bl.upper.bubble && sepU >= 0 && sepU < 0.15,
    });

    /* ---- Surface arrays, split by surface --------------------------------- */
    const stagIdx = bl.stagnation.index;
    const upperIdx = [];
    const lowerIdx = [];
    for (let i = 0; i < n; i++) (i > stagIdx ? upperIdx : lowerIdx).push(i);
    lowerIdx.reverse(); // report both surfaces leading edge -> trailing edge

    const pick = (idx, arr) => Float64Array.from(idx, (i) => arr[i]);
    const upperX = Float64Array.from(upperIdx, (i) => geo.midX[i]);
    const lowerX = Float64Array.from(lowerIdx, (i) => geo.midX[i]);
    // Surface arc length measured from the stagnation point, which is the
    // coordinate the boundary layer was actually marched in. Chordwise x is what
    // gets plotted, but any comparison against a boundary-layer result — a flat
    // plate, say — has to be made in s.
    const upperS = Float64Array.from(upperIdx, (i) => geo.sMid[i] - bl.stagnation.s);
    const lowerS = Float64Array.from(lowerIdx, (i) => bl.stagnation.s - geo.sMid[i]);

    /* ---- Forces in physical units ----------------------------------------- */
    const q = 0.5 * inputs.rho * inputs.airspeed * inputs.airspeed;
    const area = inputs.span * inputs.chord;

    return {
      ok: true,
      mode,
      inputs: { ...inputs, reynolds },
      airfoil: {
        label: spec.label,
        key: spec.key,
        series: spec.series,
        geometry: spec.geometry ?? 'naca',
        thickness: spec.t,
        camber: sectionMaxCamber(spec),
        // Thin-airfoil zero-lift angle, computed analytically from the camber
        // line. Independent of the panel solve, so it doubles as a check on it.
        zeroLiftAngleThin: sectionZeroLiftAngle(spec),
        warning: spec.warning,
        panels: n,
      },

      /* --- 12.1 Primary outputs ------------------------------------------- */
      primary: {
        cl: forces.cl,
        cd: forces.cd,
        cm: forces.cm,
        reynolds,
        liftForce: forces.cl * q * area,
        dragForce: forces.cd * q * area,
        liftPerSpan: forces.cl * q * inputs.chord,
        dragPerSpan: forces.cd * q * inputs.chord,
        ldRatio: forces.cd > 1e-9 ? forces.cl / forces.cd : 0,
        dynamicPressure: q,
        referenceArea: area,
      },

      /* --- 12.2 Advanced metrics ------------------------------------------ */
      pressure: {
        cp: sol.cp,
        x: geo.midX,
        y: geo.midY,
        s: geo.sMid,
        upper: { x: upperX, cp: pick(upperIdx, sol.cp) },
        lower: { x: lowerX, cp: pick(lowerIdx, sol.cp) },
        cpMin: forces.cpMin,
        cpMinX: forces.cpMinX,
        cpTrailingEdgeUpper: forces.cpTEUpper,
        cpTrailingEdgeLower: forces.cpTELower,
        pressureRecovery: forces.pressureRecovery,
      },

      velocity: {
        ue: sol.ue, // signed along the panel tangent
        upper: { x: upperX, ue: pick(upperIdx, sol.ue) },
        lower: { x: lowerX, ue: Float64Array.from(lowerIdx, (i) => -sol.ue[i]) },
        wakeUe: sol.ueWake,
        stagnationX: bl.stagnation.x,
        stagnationIndex: bl.stagnation.index,
      },

      boundaryLayer: {
        theta: bl.theta,
        deltaStar: bl.dstar,
        H: bl.H,
        cf: bl.cf,
        reTheta: bl.reTheta,
        state: bl.state, // 0 laminar, 1 turbulent, 2 separated
        // Wall shear in Pascals, from the local edge dynamic pressure.
        wallShear: Float64Array.from(bl.cf, (c, i) => c * q * sol.ue[i] * sol.ue[i]),
        upper: {
          x: upperX,
          s: upperS,
          theta: pick(upperIdx, bl.theta),
          deltaStar: pick(upperIdx, bl.dstar),
          H: pick(upperIdx, bl.H),
          cf: pick(upperIdx, bl.cf),
          state: Uint8Array.from(upperIdx, (i) => bl.state[i]),
        },
        lower: {
          x: lowerX,
          s: lowerS,
          theta: pick(lowerIdx, bl.theta),
          deltaStar: pick(lowerIdx, bl.dstar),
          H: pick(lowerIdx, bl.H),
          cf: pick(lowerIdx, bl.cf),
          state: Uint8Array.from(lowerIdx, (i) => bl.state[i]),
        },
        wake: { s: this._wake.s, theta: bl.thetaW, deltaStar: bl.dstarW, H: bl.HW },
        transpiration: bl.sigma,
      },

      transition: {
        model: bl.transitionModel,
        upperX: bl.upper.transitionX,
        lowerX: bl.lower.transitionX,
        upperCriterion: bl.upper.criterion,
        lowerCriterion: bl.lower.criterion,
        upperTurbulentAtTE: bl.upper.turbulentAtTE,
        lowerTurbulentAtTE: bl.lower.turbulentAtTE,
      },

      separation: {
        upperX: sepU,
        lowerX: sepL,
        laminarUpperX: bl.upper.laminarSeparationX,
        laminarLowerX: bl.lower.laminarSeparationX,
        bubbleUpper: bl.upper.bubble,
        bubbleLower: bl.lower.bubble,
        // A short bubble reattaches; the reattachment point is where the layer
        // came back as turbulent, which the march records as the transition.
        reattachmentUpperX: bl.upper.bubble ? bl.upper.transitionX : -1,
        reattachmentLowerX: bl.lower.bubble ? bl.lower.transitionX : -1,
        percentChordSeparated: chordSeparated * 100,
        percentSurfaceSeparated: separationFraction * 100,
      },

      wake: {
        ...wakeDiagnostics(this.vortexWake),
        sourceStrength: bl.sigmaWake,
        momentumThickness: wakeThetaEnd,
        shapeFactor: bl.HW[bl.nw - 1],
        sourceLineLength: this._wake.totalLength,
      },

      forceBreakdown: {
        cdPressure: forces.cdPressure,
        cdFriction: forces.cdFriction,
        cdPressureNearField: forces.cdPressureNearField,
        clPressure: forces.clPressure,
        clFriction: forces.clFriction,
        clKuttaJoukowski: forces.clKuttaJoukowski,
        // The two independent routes to lift should agree; the gap is a direct
        // measure of the panel discretisation error.
        liftConsistency: Math.abs(forces.cl - forces.clKuttaJoukowski),
        circulation: forces.circulation,
      },

      stability: {
        aerodynamicCenter: stability.aerodynamicCenter,
        centerOfPressure: forces.xCp,
        liftSlopePerDeg: stability.liftSlopePerDeg,
        liftSlopePerRad: stability.liftSlopePerRad,
        inviscidLiftSlopePerRad: inviscidSlope,
        dCmdCl: stability.dCmdCl,
        momentReference: forces.momentReference,
      },

      stall: {
        index: stall.index,
        state: stall.state,
        label: STALL_LABELS[stall.state],
        components: stall.components,
        flowAttachmentPercent: (1 - separationFraction) * 100,
        // Boundary-layer health: how close the worst-off station on the section
        // is to the turbulent separation criterion. 1 is a healthy full profile,
        // 0 means something on the surface is at the point of detaching.
        boundaryLayerHealth: Math.max(0, Math.min(1, (H_TURB_SEP - worstH) / (H_TURB_SEP - 1.4))),
        worstShapeFactor: worstH,
        minSkinFriction: minCf,
        stallMarginDeg: stallMargin,
      },

      convergence: {
        converged: report.converged,
        iterations: report.iterations,
        maxIterations: MAX_ITERATIONS,
        residualCp: report.residualCp,
        residualDeltaStar: report.residualDstar,
        toleranceCp: TOL_CP,
        reason: report.reason,
        history: report.history,
        relaxation: report.omega,
        confidence: conf.score,
        confidenceReasons: conf.reasons,
        confidenceSummary: conf.summary,
        conditionNumber: this._sys.condition,
        linearSolver: this._sys.solver,
        stationsLimited: bl.sigmaClamped ?? 0,
      },

      timing: { ...this.timing },
    };
  }
}

const SCRATCH2 = [0, 0];

function nearlyEqual(a, b, tol) {
  return Number.isFinite(a) && Math.abs(a - b) < tol;
}

function now() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/**
 * Rescale the force outputs for a new span or air density without re-solving.
 * Two-dimensional section aerodynamics has no span dependence at all; the span
 * only sets the planform area the coefficients are multiplied by.
 */
function withScaledForces(state, inputs) {
  const q = 0.5 * inputs.rho * inputs.airspeed * inputs.airspeed;
  const area = inputs.span * inputs.chord;
  return {
    ...state,
    inputs: { ...state.inputs, span: inputs.span, rho: inputs.rho },
    primary: {
      ...state.primary,
      liftForce: state.primary.cl * q * area,
      dragForce: state.primary.cd * q * area,
      liftPerSpan: state.primary.cl * q * inputs.chord,
      dragPerSpan: state.primary.cd * q * inputs.chord,
      dynamicPressure: q,
      referenceArea: area,
    },
  };
}

export { parseNacaCode, STALL_LABELS };

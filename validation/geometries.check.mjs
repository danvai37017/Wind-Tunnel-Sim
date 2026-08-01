// End-to-end solve check: does every geometry converge, and are the numbers sane?
import { AeroSolver } from '../src/solver/index.js';
import { getMode, getScale, heatDomain, HEAT_MODES } from '../src/viz/heatmap.js';

const base = { alphaDeg: 5, airspeed: 30, chord: 0.06, span: 0.24 };

function run(name, inputs) {
  const s = new AeroSolver();
  const st = s.update({ ...base, ...inputs });
  console.log(`\n=== ${name} ===`);
  if (!st) { console.log('  NO STATE:', s.buildError); return null; }
  if (s.buildError) console.log('  buildError:', s.buildError);
  console.log(`  ${st.airfoil.label}  panels ${st.airfoil.panels}  t/c ${(st.airfoil.thickness*100).toFixed(2)}%  camber ${(st.airfoil.camber*100).toFixed(2)}%`);
  console.log(`  Cl ${st.primary.cl.toFixed(4)}   Cd ${st.primary.cd.toFixed(5)}   Cm ${st.primary.cm.toFixed(4)}   L/D ${st.primary.ldRatio.toFixed(1)}`);
  console.log(`  alpha_L0(thin) ${st.airfoil.zeroLiftAngleThin.toFixed(2)} deg   Cl by K-J ${st.forceBreakdown.clKuttaJoukowski.toFixed(4)}  (consistency ${st.forceBreakdown.liftConsistency.toExponential(1)})`);
  console.log(`  transition u/l ${st.transition.upperX.toFixed(3)}/${st.transition.lowerX.toFixed(3)}   separation u/l ${st.separation.upperX.toFixed(3)}/${st.separation.lowerX.toFixed(3)}`);
  console.log(`  converged ${st.convergence.converged}  iters ${st.convergence.iterations}  resid ${st.convergence.residualCp.toExponential(1)}  confidence ${st.convergence.confidence}%  cond ${st.convergence.conditionNumber.toFixed(1)}`);

  // every heat mode must produce a finite, correctly-sized array
  for (const m of HEAT_MODES) {
    const v = m.values(st);
    let bad = 0, lo = Infinity, hi = -Infinity;
    for (let i = 0; i < v.length; i++) {
      if (!Number.isFinite(v[i])) bad++; else { lo = Math.min(lo, v[i]); hi = Math.max(hi, v[i]); }
    }
    const d = heatDomain(v, m, getScale('spectral'));
    const ok = v.length === st.airfoil.panels && bad === 0;
    console.log(`    heat ${m.id.padEnd(10)} n=${v.length} ${ok ? 'ok ' : 'BAD'} range ${lo.toFixed(3)} .. ${hi.toFixed(3)}  domain ${d.lo.toFixed(3)}..${d.hi.toFixed(3)}`);
  }
  return st;
}

run('NACA 2412 (regression vs screenshot: Cl 0.724, Cd 0.01817)', { geometry: 'naca', naca: '2412' });
run('NACA 0012', { geometry: 'naca', naca: '0012' });
run('Clark Y', { geometry: 'clarky' });
run('Flat plate', { geometry: 'flatplate' });

// lift curve: Clark Y and flat plate should both be near-linear pre-stall
console.log('\n=== lift curves (Cl) ===');
for (const g of [{ geometry: 'naca', naca: '2412' }, { geometry: 'clarky' }, { geometry: 'flatplate' }]) {
  const s = new AeroSolver();
  const row = [];
  for (const a of [-4, 0, 4, 8, 12, 16]) {
    const st = s.update({ ...base, ...g, alphaDeg: a });
    row.push(st ? st.primary.cl.toFixed(3).padStart(7) : '   ----');
  }
  const name = (g.naca ? `NACA ${g.naca}` : g.geometry).padEnd(12);
  console.log(`  ${name} a=-4,0,4,8,12,16: ${row.join(' ')}`);
}

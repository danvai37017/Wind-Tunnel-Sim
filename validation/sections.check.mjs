import { parseSection, sectionMaxCamber, sectionZeroLiftAngle } from '../src/solver/sections.js';
import { buildBody, panelGeometry } from '../src/solver/geometry.js';

function report(name, spec) {
  console.log(`\n=== ${name} ===`);
  if (!spec.ok) { console.log('  FAILED:', spec.error); return; }
  console.log('  label      ', spec.label);

  // thickness / camber sampled in the chord frame
  let tMax = 0, tAt = 0, cMax = 0, cAt = 0;
  for (let k = 0; k <= 500; k++) {
    const x = k / 500;
    const u = spec.surface(x, 1), l = spec.surface(x, -1);
    const th = u[1] - l[1];
    if (th > tMax) { tMax = th; tAt = x; }
  }
  const cam = sectionMaxCamber(spec);
  for (let k = 0; k <= 500; k++) {
    const x = k / 500;
    const c = spec.camberAt ? spec.camberAt(x)[0] : 0;
    if (Math.abs(c) > Math.abs(cMax)) { cMax = c; cAt = x; }
  }
  console.log(`  max thickness ${(tMax*100).toFixed(2)}% at x/c ${tAt.toFixed(3)}`);
  console.log(`  max camber    ${(cMax*100).toFixed(2)}% at x/c ${cAt.toFixed(3)}  (sectionMaxCamber ${(cam*100).toFixed(2)}%)`);
  console.log(`  alpha_L0      ${sectionZeroLiftAngle(spec).toFixed(3)} deg`);

  // LE / TE placement in the chord frame
  const nose = spec.surface(0, 1), teU = spec.surface(1, 1), teL = spec.surface(1, -1);
  console.log(`  nose  (${nose[0].toFixed(5)}, ${nose[1].toFixed(5)})`);
  console.log(`  TE up (${teU[0].toFixed(5)}, ${teU[1].toFixed(5)})  TE lo (${teL[0].toFixed(5)}, ${teL[1].toFixed(5)})`);

  // does it panel?
  const body = buildBody(spec, { panels: 220, maxPanels: 500 });
  const geo = panelGeometry(body);
  if (!geo) { console.log('  PANELLING FAILED (degenerate)'); return; }
  let minLen = Infinity, maxLen = 0;
  for (let i = 0; i < geo.n; i++) { minLen = Math.min(minLen, geo.len[i]); maxLen = Math.max(maxLen, geo.len[i]); }
  console.log(`  panels ${geo.n}, perimeter ${geo.perimeter.toFixed(4)}, panel len ${minLen.toExponential(2)} .. ${maxLen.toExponential(2)}`);

  // monotonic lower surface check for Clark Y flat bottom
  if (spec.geometry === 'clarky') {
    const ys = [];
    for (let x = 0.35; x <= 0.95; x += 0.05) ys.push(spec.surface(x, -1)[1]);
    const spread = Math.max(...ys) - Math.min(...ys);
    console.log(`  flat-bottom spread over x/c 0.35..0.95: ${(spread*100).toFixed(3)}% chord`);
  }
}

report('Clark Y', parseSection({ geometry: 'clarky' }));
report('Flat plate', parseSection({ geometry: 'flatplate' }));
report('NACA 2412 (regression)', parseSection({ geometry: 'naca', naca: '2412' }));
report('NACA 0012 (regression)', parseSection({ geometry: 'naca', naca: '0012' }));

/**
 * Missing-import check.
 *
 * Vite bundles happily with an undefined free variable — it is only a
 * ReferenceError at runtime, and if it happens inside requestAnimationFrame the
 * canvas just stops with nothing in the build log. That is exactly how a missing
 * `solverToView` import got the 3D view into a state where it drew the
 * background and then threw, every frame, silently.
 *
 * So: for every local module a source file imports from, take the names that
 * module exports, and flag any that appear as a bare identifier in the file
 * without being on the import list. Cheap, no dependencies, and it catches the
 * one failure mode the bundler will not.
 *
 * A real linter with no-undef would subsume this. There isn't one configured
 * here, and this is the part that actually bit.
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve as resolvePath } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const SOURCES = ['../src/WindTunnel.jsx', '../src/flow.js', '../src/viz/heatmap.js', '../src/solver/sections.js'];

const exportsOf = (src) => {
  const names = new Set();
  for (const m of src.matchAll(/^export\s+(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    names.add(m[1]);
  }
  // `export { a, b as c }`
  for (const m of src.matchAll(/^export\s*\{([^}]*)\}/gm)) {
    for (const part of m[1].split(',')) {
      const t = part.trim().split(/\s+as\s+/);
      if (t.length) names.add((t[1] ?? t[0]).trim());
    }
  }
  names.delete('');
  return names;
};

/**
 * Blank out comments and string/template literals, preserving offsets so a
 * reported line number still points at the real line. Without this, prose in a
 * comment ("...the camber line...") reads as a use of an exported `camber`.
 */
function stripNonCode(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, ' '))
    .replace(/\/\/[^\n]*/g, (m) => ' '.repeat(m.length))
    .replace(/(['"`])(?:\\.|(?!\1)[^\\])*\1/g, (m) => m.replace(/[^\n]/g, ' '));
}

let problems = 0;

for (const rel of SOURCES) {
  const file = resolvePath(here, rel);
  const src = readFileSync(file, 'utf8');
  const code = stripNonCode(src);

  // Names brought in from *any* module, plus anything declared locally. A name
  // re-exported by two modules (parseNacaCode, from both naca.js and index.js)
  // only has to be imported once.
  // Imports are read from the raw source: stripNonCode blanks string literals,
  // which would take the module specifier with them.
  const allImported = new Set();
  for (const imp of src.matchAll(/import\s*\{([^}]*)\}\s*from/g)) {
    for (const part of imp[1].split(',')) {
      const n = part.trim().split(/\s+as\s+/).pop().trim();
      if (n) allImported.add(n);
    }
  }

  for (const imp of src.matchAll(/import\s*\{([^}]*)\}\s*from\s*['"]?(\.[^'"\s]+)/g)) {
    const depPath = resolvePath(dirname(file), imp[2]);
    let depSrc;
    try {
      depSrc = readFileSync(depPath, 'utf8');
    } catch {
      continue;
    }

    for (const name of exportsOf(depSrc)) {
      if (allImported.has(name)) continue;
      // Declared in this file under the same name? Then it is not the import.
      if (new RegExp(`(?:function|const|let|var|class)\\s+${name}\\b`).test(code)) continue;
      // Only call sites. A missing constant is also a ReferenceError, but bare
      // identifiers produce far too many false hits to be worth flagging, and
      // the failure that actually bites is calling a function that isn't there.
      const hits = [...code.matchAll(new RegExp(`(?<![.\\w$])${name}\\s*\\(`, 'g'))];
      if (!hits.length) continue;

      const line = src.slice(0, hits[0].index).split('\n').length;
      console.log(`  FAIL ${rel}:${line}  calls "${name}" from ${imp[2]} but does not import it`);
      problems++;
    }
  }
}

console.log(problems === 0 ? '\nno missing imports' : `\n${problems} MISSING IMPORT(S)`);
process.exit(problems === 0 ? 0 : 1);

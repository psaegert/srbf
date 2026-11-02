// The sealed units (tools/seal.mjs). Node only, no page: each test seals stand-in units into a temporary directory and
// opens the result the way the page does (explorer_v2.js, "adding a method from a key"): a key's keyring (PBKDF2,
// AES-GCM), the method secrets it holds, the units of those methods and of every pair of them, and each unit's files by
// the names its keys give their paths, with the path as the additional data.
import { test, expect } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, readdirSync, rmSync, existsSync, cpSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { webcrypto, pbkdf2Sync, createHmac } from 'node:crypto';
import { gzipSync } from 'node:zlib';

const subtle = webcrypto.subtle;
const SEAL = fileURLToPath(new URL('../tools/seal.mjs', import.meta.url));
const FIXTURES = fileURLToPath(new URL('./fixtures/', import.meta.url));
const REL = '2026-09';
// the fixtures' secrets (tests/fixtures/README.md); the page tests type the same keys
const MASTER = 'fixture-master-secret-for-the-tests';
const KEY = 'fixture-key-for-the-tests', KEY_TWO = 'a-second-fixture-key-for-the-tests', KEY_BOTH = 'a-third-fixture-key-for-the-tests';
const LINE = /^\(window\.RESULTS_V2_SEALED_FILES=window\.RESULTS_V2_SEALED_FILES\|\|\{\}\)\["([0-9a-f]{32})"\]=\{"iv":"[A-Za-z0-9+/]+=*","ct":"[A-Za-z0-9+/]+=*"\};\n$/;
const utf8 = (s) => Buffer.from(s, 'utf8');
const bytesOf = (b64) => new Uint8Array(Buffer.from(b64, 'base64'));
const hmac = (key, ...parts) => parts.reduce((h, p) => h.update(p), createHmac('sha256', key)).digest();

test.skip(({ isMobile }) => isMobile, 'node only: one project is enough');
test.describe.configure({ timeout: 180_000 });   // every seal runs PBKDF2 at 600k iterations once per grant, twice

// ---- stand-in units, and sealing them ------------------------------------------------------------------------------
// a method's unit: its unit.js, per-problem files under pp/, pred/ and pv/ that name it, and per-metric files under
// ranks/ and hist/ (at the same paths in every unit); a pair's unit: unit.js and a ranks/ file of the two
function methodUnit(root, m) {
  const pp = (r, d) => `window.RESULTS_V2_PP=window.RESULTS_V2_PP||{};(function(){var R=window.RESULTS_V2_PP;R["${REL}"]=R["${REL}"]||{};` +
    `R["${REL}"]["${m}|feynman|${r}|${d}"]={"n":3,"s":"AQMH","v":{"log10_fvu_val":"AH/8"}};})();\n`;
  return write(join(root, 'm', m), {
    'unit.js': '(window.RESULTS_V2_PRIVATE_UNITS=window.RESULTS_V2_PRIVATE_UNITS||[]).push(' + JSON.stringify({
      base: '', methods: [{ key: m, label: m.toUpperCase() }], cells: {}, status: {}, timing: {},
      pp: { [m]: { 'feynman|16': [1, 2] } }, pred: { [m]: { 'feynman|16': [1] } }, pred_block: 500 }) + ');\n',
    'hist/log10_fvu_val.js': `window.RESULTS_V2_HIST=window.RESULTS_V2_HIST||{};window.RESULTS_V2_HIST["${REL}"]={"${m}":[1,2,3]};\n`,
    [`pp/${m}/feynman/16.1.js`]: pp(16, 1),
    [`pp/${m}/feynman/16.2.js`]: pp(16, 2),
    [`pred/${m}/feynman/16.1.0.js`]: `window.RESULTS_V2_PRED=window.RESULTS_V2_PRED||{};(function(){var R=window.RESULTS_V2_PRED;` +
      `R["${REL}"]=R["${REL}"]||{};R["${REL}"]["${m}|feynman|16|1|0"]={"0":["* x1 x2",3],"1":null,"2":["+ x1 0.5",1]};})();\n`,
    [`pv/${m}/16/frame.js`]: `window.RESULTS_V2_PV=window.RESULTS_V2_PV||{};window.RESULTS_V2_PV["${m}|16|frame"]=[1,2,3];\n`,
    'ranks/log10_fvu_val.js': `window.RESULTS_V2_RANKCELLS=window.RESULTS_V2_RANKCELLS||{};window.RESULTS_V2_RANKCELLS["${m}|e2e"]={"feynman":{"16":[100,3,4]}};\n`,
  });
}
function pairUnit(root, a, b) {
  return write(join(root, 'p', a, b), {
    'unit.js': '(window.RESULTS_V2_PRIVATE_UNITS=window.RESULTS_V2_PRIVATE_UNITS||[]).push(' + JSON.stringify({ base: '', methods: [], pair: [a, b] }) + ');\n',
    'ranks/log10_fvu_val.js': `window.RESULTS_V2_RANKCELLS=window.RESULTS_V2_RANKCELLS||{};window.RESULTS_V2_RANKCELLS["${a}|${b}"]={"feynman":{"16":[100,1,2]}};\n`,
  });
}
function write(dir, files) {
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), text);
  }
  return Object.keys(files).filter((p) => p !== 'unit.js');   // the files sealed one by one
}
const METHODS = ['alpha', 'beta', 'gamma-one'];
function standIns(root) {   // three methods and their three pairs -> {unit label: [its files]}
  const out = {};
  for (const m of METHODS) { out['m/' + m] = methodUnit(root, m); }
  for (const [a, b] of [['alpha', 'beta'], ['alpha', 'gamma-one'], ['beta', 'gamma-one']]) { out[`p/${a}/${b}`] = pairUnit(root, a, b); }
  return out;
}
const made = [];
function scratch() { const d = mkdtempSync(join(tmpdir(), 'seal-units-')); made.push(d); return d; }
test.afterEach(() => { while (made.length) { rmSync(made.pop(), { recursive: true, force: true }); } });
const ENV = { SRBF_MASTER: MASTER, SRBF_KEY: KEY, SRBF_KEY_TWO: KEY_TWO, SRBF_KEY_BOTH: KEY_BOTH };
function sealRun(units, out, grants, env = {}, extra = []) {
  return spawnSync(process.execPath, [SEAL, REL, '--units', units, '--master', 'SRBF_MASTER', '--out', out,
    ...grants.flatMap((g) => ['--grant', g]), ...extra], { env: { ...process.env, ...ENV, ...env }, encoding: 'utf8' });
}
function seal(units, out, grants, env, extra) {
  const r = sealRun(units, out, grants, env, extra);
  expect(r.status, r.stderr).toBe(0);
  return r.stdout;
}
const GRANTS = ['SRBF_KEY=*', 'SRBF_KEY_TWO=gamma-*', 'SRBF_KEY_BOTH=alpha,beta'];
function sealedFiles(out) {   // {name: bytes} of everything in the sealed/ next to the out file
  const dir = join(dirname(out), 'sealed');
  return existsSync(dir) ? Object.fromEntries(readdirSync(dir).sort().map((n) => [n, readFileSync(join(dir, n))])) : {};
}

// ---- opening it as a page does ---------------------------------------------------------------------------------
function run(src) { const window = {}; new Function('window', src)(window); return window; }   // a script, as a <script> runs it
async function gunzip(gz) { return Buffer.from(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()); }
async function decrypt(raw, iv, ct, aad) {
  try {
    const k = await subtle.importKey('raw', raw, 'AES-GCM', false, ['decrypt']);
    return Buffer.from(await subtle.decrypt({ name: 'AES-GCM', iv, additionalData: aad }, k, ct));
  } catch (e) { return null; }
}
const sealedOf = (out) => run(readFileSync(out, 'utf8')).RESULTS_V2_SEALED[REL];
function unitKeys(S) {
  return { id: hmac(S, utf8('srbf-unit-id')).toString('hex').slice(0, 32), pay: hmac(S, utf8('srbf-unit-payload')),
    enc: hmac(S, utf8('srbf-unit-file-enc')), name: hmac(S, utf8('srbf-unit-file-name')) };
}
const pairSecret = (Sa, Sb, a, b) => hmac(Buffer.concat([Sa, Sb]), utf8('srbf-pair\0' + REL + '\0' + a + '\0' + b));
async function keyring(sealed, key) {   // -> [[method, secret]] of the one keyring the key opens, or null
  const bits = pbkdf2Sync(key, bytesOf(sealed.salt), sealed.iter, 64, 'sha256');
  const opened = [];
  for (const r of sealed.keyrings) {
    const plain = await decrypt(bits.subarray(0, 32), bytesOf(r.iv), bytesOf(r.ct), utf8(REL + '/keyring'));
    if (plain) { opened.push(JSON.parse(plain.toString('utf8')).methods.map(([m, s]) => [m, Buffer.from(bytesOf(s))])); }
  }
  expect(opened.length, 'a key opens one keyring or none').toBeLessThanOrEqual(1);
  return opened[0] || null;
}
async function openUnit(sealed, S) {   // -> the unit.js a secret opens, or null
  const k = unitKeys(S), env = sealed.units[k.id];
  if (!env) { return null; }
  const gz = await decrypt(k.pay, bytesOf(env.iv), bytesOf(env.ct), utf8(REL + '/unit/' + k.id));
  return gz ? (await gunzip(gz)).toString('utf8') : null;
}
async function opens(out, key) {   // -> {unit label: unit.js} of everything a key opens, as the page opens it
  const sealed = sealedOf(out), held = await keyring(sealed, key), got = {};
  if (!held) { return got; }
  const secret = Object.fromEntries(held);
  for (const [m, S] of held) { const u = await openUnit(sealed, S); if (u) { got['m/' + m] = u; } }
  const ms = held.map(([m]) => m).sort();
  for (let i = 0; i < ms.length; i++) {
    for (const b of ms.slice(i + 1)) {
      const u = await openUnit(sealed, pairSecret(secret[ms[i]], secret[b], ms[i], b));
      if (u) { got[`p/${ms[i]}/${b}`] = u; }
    }
  }
  return got;
}
const methodSecret = (m) => hmac(utf8(MASTER), utf8('srbf-unit\0' + REL + '\0' + m));
const fileName = (k, path) => hmac(k.name, utf8('name\0' + path)).toString('hex').slice(0, 32);
async function openLine(line, k, path) {   // -> the content of one sealed file read as `path`, or null
  const W = run(line), name = Object.keys(W.RESULTS_V2_SEALED_FILES)[0], env = W.RESULTS_V2_SEALED_FILES[name];
  const gz = await decrypt(k.enc, bytesOf(env.iv), bytesOf(env.ct), utf8(REL + '/' + path));
  return gz ? gunzip(gz) : null;
}
async function openFile(out, k, path) {   // -> a path's content, fetched by its name, or null
  const at = join(dirname(out), 'sealed', fileName(k, path) + '.js');
  return existsSync(at) ? openLine(readFileSync(at, 'utf8'), k, path) : null;
}

// ---- the tests -------------------------------------------------------------------------------------------------
test('each key opens its own methods and the pairs among them, and nothing else', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js');
  standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  const all = await opens(out, KEY), gamma = await opens(out, KEY_TWO), two = await opens(out, KEY_BOTH);
  expect(Object.keys(all).sort()).toEqual(['m/alpha', 'm/beta', 'm/gamma-one', 'p/alpha/beta', 'p/alpha/gamma-one', 'p/beta/gamma-one']);
  expect(Object.keys(gamma)).toEqual(['m/gamma-one']);
  expect(Object.keys(two).sort()).toEqual(['m/alpha', 'm/beta', 'p/alpha/beta']);
  expect(await opens(out, 'a-key-nobody-was-given-at-all')).toEqual({});
  // each unit opens to its own unit.js, which pushes its overlay
  for (const [label, src] of Object.entries(all)) {
    expect(src).toBe(readFileSync(join(dir, 'units', label, 'unit.js'), 'utf8'));
    const pushed = run(src).RESULTS_V2_PRIVATE_UNITS;
    expect(pushed).toHaveLength(1);
    expect(pushed[0].methods.map((m) => m.key)).toEqual(label.startsWith('m/') ? [label.slice(2)] : []);
  }
});

test('a pair opens only with both of its methods', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js');
  standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  const sealed = sealedOf(out), a = methodSecret('alpha'), b = methodSecret('beta'), r = methodSecret('gamma-one');
  expect(await openUnit(sealed, pairSecret(a, b, 'alpha', 'beta'))).not.toBeNull();
  // a reader of alpha alone has no second secret to put next to it: neither a guess nor another method's will do
  for (const other of [Buffer.alloc(32), r, hmac(a, utf8('beta'))]) {
    expect(await openUnit(sealed, pairSecret(a, other, 'alpha', 'beta'))).toBeNull();
  }
  expect(await openUnit(sealed, pairSecret(b, a, 'alpha', 'beta')), 'the order is part of the pair').toBeNull();
});

test('the keyrings are padded to eight of one size, and the file names no method and no grant', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js');
  standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  const sealed = sealedOf(out);
  expect(Object.keys(sealed).sort()).toEqual(['iter', 'kdf', 'keyrings', 'salt', 'units', 'v']);
  expect(sealed.keyrings).toHaveLength(8);
  expect(new Set(sealed.keyrings.map((r) => bytesOf(r.ct).length)).size).toBe(1);
  expect(Object.keys(sealed.units)).toHaveLength(6);
  const text = readFileSync(out, 'utf8') + Object.values(sealedFiles(out)).map((b) => b.toString('utf8')).join('');
  for (const tell of [...METHODS, 'feynman', 'unit.js', 'ranks/', 'pp/', 'SRBF_KEY', 'RESULTS_V2_PRIVATE', MASTER, KEY]) {
    expect(text, tell).not.toContain(tell);
  }
  // more grants than the padding asks for: one keyring each, no padding
  seal(join(dir, 'units'), out, GRANTS, {}, ['--keyrings', '2']);
  expect(sealedOf(out).keyrings).toHaveLength(3);
});

test('a unit\'s files open by their names, each only as its own path and only with its own unit\'s keys', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js'), units = standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  let n = 0;
  for (const [label, paths] of Object.entries(units)) {
    const [kind, a, b] = label.split('/');
    const k = unitKeys(kind === 'm' ? methodSecret(a) : pairSecret(methodSecret(a), methodSecret(b), a, b));
    for (const p of paths) {
      expect((await openFile(out, k, p)) || '', label + ' ' + p).toEqual(readFileSync(join(dir, 'units', label, p)));
      const line = readFileSync(join(dirname(out), 'sealed', fileName(k, p) + '.js'), 'utf8');
      expect(line).toMatch(LINE);
      expect(await openLine(line, k, p === paths[0] ? paths[1] : paths[0]), 'bound to its path').toBeNull();
      n++;
    }
  }
  expect(Object.keys(sealedFiles(out))).toHaveLength(n);
  // ranks/log10_fvu_val.js is in every unit, under a name and a key of each unit's own
  const alpha = unitKeys(methodSecret('alpha')), beta = unitKeys(methodSecret('beta'));
  expect(fileName(alpha, 'ranks/log10_fvu_val.js')).not.toBe(fileName(beta, 'ranks/log10_fvu_val.js'));
  const line = readFileSync(join(dirname(out), 'sealed', fileName(alpha, 'ranks/log10_fvu_val.js') + '.js'), 'utf8');
  expect(await openLine(line, beta, 'ranks/log10_fvu_val.js')).toBeNull();
});

test('a reseal writes the same bytes; a changed file changes alone; a withdrawn unit leaves', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js'), units = standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  const first = sealedFiles(out), envelope = readFileSync(out);
  expect(seal(join(dir, 'units'), out, GRANTS)).toContain(Object.keys(first).length + ' file(s), 0 written');
  expect(sealedFiles(out)).toEqual(first);
  expect(readFileSync(out)).toEqual(envelope);
  const fresh = join(dir, 'fresh', 'sealed.js');                     // and a seal from nothing writes the same bytes
  seal(join(dir, 'units'), fresh, GRANTS);
  expect(sealedFiles(fresh)).toEqual(first);
  expect(readFileSync(fresh)).toEqual(envelope);
  // one file changed: that file is rewritten, nothing else
  writeFileSync(join(dir, 'units', 'm', 'beta', 'pp', 'beta', 'feynman', '16.2.js'), 'window.RESULTS_V2_PP={};\n');
  expect(seal(join(dir, 'units'), out, GRANTS)).toContain('1 written');
  const after = sealedFiles(out), changed = Object.keys(first).filter((n) => !after[n] || !after[n].equals(first[n]));
  expect(changed).toHaveLength(1);
  // a method withdrawn (and with it its pairs): its files leave sealed/, nothing else does
  rmSync(join(dir, 'units', 'm', 'gamma-one'), { recursive: true });
  for (const p of ['p/alpha/gamma-one', 'p/beta/gamma-one']) { rmSync(join(dir, 'units', p), { recursive: true }); }
  seal(join(dir, 'units'), out, ['SRBF_KEY=*', 'SRBF_KEY_BOTH=alpha,beta']);
  const gone = ['m/gamma-one', 'p/alpha/gamma-one', 'p/beta/gamma-one'].reduce((n, u) => n + units[u].length, 0);
  expect(Object.keys(sealedFiles(out))).toHaveLength(Object.keys(after).length - gone);
  expect(Object.keys(await opens(out, KEY)).sort()).toEqual(['m/alpha', 'm/beta', 'p/alpha/beta']);
  expect(await opens(out, KEY_TWO)).toEqual({});
});

test('a sealed file that holds the same content is kept under its own IV, whatever gzip wrote', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js');
  standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  // another machine's gzip compressing the same content differently: a line made with a gzip level of its own
  const k = unitKeys(methodSecret('alpha')), path = 'pp/alpha/feynman/16.1.js', content = readFileSync(join(dir, 'units', 'm', 'alpha', path));
  const gz = gzipSync(content, { level: 1 }), iv = hmac(k.name, utf8('iv\0' + path + '\0'), gz).subarray(0, 12);
  const key = await subtle.importKey('raw', k.enc, 'AES-GCM', false, ['encrypt']);
  const ct = Buffer.from(await subtle.encrypt({ name: 'AES-GCM', iv, additionalData: utf8(REL + '/' + path) }, key, gz));
  const at = join(dirname(out), 'sealed', fileName(k, path) + '.js');
  const other = '(window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})[' + JSON.stringify(fileName(k, path)) + ']=' +
    JSON.stringify({ iv: iv.toString('base64'), ct: ct.toString('base64') }) + ';\n';
  writeFileSync(at, other);
  expect(seal(join(dir, 'units'), out, GRANTS)).toContain('0 written');
  expect(readFileSync(at, 'utf8')).toBe(other);
});

test('a unit sealed to the same content is kept under its own IV, whatever gzip wrote', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js');
  standIns(join(dir, 'units'));
  seal(join(dir, 'units'), out, GRANTS);
  // another machine's gzip compressing the same unit.js differently: an envelope made with a gzip level of its own
  const S = methodSecret('alpha'), k = unitKeys(S), aad = utf8(REL + '/unit/' + k.id);
  const gz = gzipSync(readFileSync(join(dir, 'units', 'm', 'alpha', 'unit.js')), { level: 1 }), iv = Buffer.alloc(12, 7);
  const key = await subtle.importKey('raw', k.pay, 'AES-GCM', false, ['encrypt']);
  const ct = Buffer.from(await subtle.encrypt({ name: 'AES-GCM', iv, additionalData: aad }, key, gz));
  const W = sealedOf(out); W.units[k.id] = { iv: iv.toString('base64'), ct: ct.toString('base64') };
  const other = 'window.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED[' + JSON.stringify(REL) + ']=' + JSON.stringify(W) + ';\n';
  writeFileSync(out, other);
  seal(join(dir, 'units'), out, GRANTS);
  expect(readFileSync(out, 'utf8')).toBe(other);
  // and a unit.js that changed gets a new envelope
  writeFileSync(join(dir, 'units', 'm', 'alpha', 'unit.js'), '(window.RESULTS_V2_PRIVATE_UNITS=window.RESULTS_V2_PRIVATE_UNITS||[]).push({"methods":[]});\n');
  seal(join(dir, 'units'), out, GRANTS);
  expect(sealedOf(out).units[k.id].iv).not.toBe(iv.toString('base64'));
  expect(await openUnit(sealedOf(out), S)).toContain('"methods":[]');
});

test('a run fails, writing nothing, on a grant that names no method, a key two grants share, or a key that is the master', async () => {
  const dir = scratch(), out = join(dir, 'out', 'sealed.js');
  standIns(join(dir, 'units'));
  const cases = [
    [['SRBF_KEY=*', 'SRBF_KEY_TWO=nobody-*'], {}, /matches no method/],
    [['SRBF_KEY=alpha', 'SRBF_KEY_TWO=beta'], { SRBF_KEY_TWO: KEY }, /share a key/],
    [['SRBF_KEY=alpha'], { SRBF_KEY: MASTER }, /master/],
    [['SRBF_KEY=alpha'], { SRBF_KEY: 'too-short' }, /shorter than 16/],
  ];
  for (const [grants, env, why] of cases) {
    const r = sealRun(join(dir, 'units'), out, grants, env);
    expect(r.status, why.source).not.toBe(0);
    expect(r.stderr).toMatch(why);
    expect(existsSync(out)).toBe(false);
  }
});

// The committed page fixtures (tests/fixtures/README.md): each opens with its keys and is what a reseal of its units
// writes over it (another node's gzip may compress differently, which is why the reseal starts from the committed files,
// as the board's does).
test('the committed page fixtures open with their keys and are what a reseal of their units writes', async () => {
  for (const [name, grants, expected] of [
    ['keyrings', ['SRBF_KEY=fixture-method', 'SRBF_KEY_TWO=fixture-method-two', 'SRBF_KEY_BOTH=fixture-*'],
      { [KEY]: ['m/fixture-method'], [KEY_TWO]: ['m/fixture-method-two'],
        [KEY_BOTH]: ['m/fixture-method', 'm/fixture-method-two', 'p/fixture-method/fixture-method-two'] }],
    ['sealed_files', ['SRBF_KEY=fixture-files'], { [KEY]: ['m/fixture-files'] }],
  ]) {
    const units = join(FIXTURES, name, 'units'), out = join(FIXTURES, name, 'sealed.js');
    for (const [key, labels] of Object.entries(expected)) { expect(Object.keys(await opens(out, key)).sort(), name).toEqual(labels); }
    const dir = scratch(), copy = join(dir, 'sealed.js');
    if (existsSync(join(dirname(out), 'sealed'))) { cpSync(join(dirname(out), 'sealed'), join(dir, 'sealed'), { recursive: true }); }
    cpSync(out, copy);
    expect(seal(units, copy, grants)).toContain('0 written');
    expect(readFileSync(copy)).toEqual(readFileSync(out));
    expect(sealedFiles(copy)).toEqual(sealedFiles(out));
  }
});

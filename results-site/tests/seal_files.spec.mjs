// The per-problem files of a sealed overlay, sealed one file at a time (tools/seal.mjs, "PER-PROBLEM FILES"). Node
// only, no page: each test seals a stand-in overlay into a temporary directory and opens the result the way a page
// does -- the overlay with the key (PBKDF2, AES-GCM, gzip), the two file keys it hands over, then each file by the name
// those keys give its path, with the path as the additional data. The files sealed one by one are those under pp/,
// pred/ and pv/ (per problem) and under ranks/, paired/ and hist/ (per metric).
import { test, expect } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, readdirSync, rmSync, existsSync, cpSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { webcrypto, pbkdf2Sync, createHash, createHmac } from 'node:crypto';
import { gzipSync } from 'node:zlib';

const subtle = webcrypto.subtle;
const SEAL = fileURLToPath(new URL('../tools/seal.mjs', import.meta.url));
const FIXTURE = fileURLToPath(new URL('./fixtures/sealed_files/', import.meta.url));
const REL = '2026-09';
const KEY = 'fixture-key-for-the-tests', KEY_TWO = 'a-second-fixture-key-for-the-tests';
const LINE = /^\(window\.RESULTS_V2_SEALED_FILES=window\.RESULTS_V2_SEALED_FILES\|\|\{\}\)\["([0-9a-f]{32})"\]=\{"iv":"[A-Za-z0-9+/]+=*","ct":"[A-Za-z0-9+/]+=*"\};\n$/;
const utf8 = (s) => new TextEncoder().encode(s);
const bytesOf = (b64) => new Uint8Array(Buffer.from(b64, 'base64'));

test.skip(({ isMobile }) => isMobile, 'node only: one project is enough');
test.describe.configure({ timeout: 120_000 });   // every seal runs PBKDF2 at 600k iterations several times

// ---- a stand-in overlay, and sealing it ----------------------------------------------------------------------------
// two ordinary payload scripts, per-problem files under pp/, pred/ and pv/ that name their method, and per-metric files
// under ranks/ and hist/ (at the same paths in every stand-in: two overlays may share a path)
function overlay(dir, method) {
  const pp = (r, d) => `window.RESULTS_V2_PP=window.RESULTS_V2_PP||{};(function(){var R=window.RESULTS_V2_PP;R["${REL}"]=R["${REL}"]||{};` +
    `R["${REL}"]["${method}|feynman|${r}|${d}"]={"n":3,"s":"AQMH","v":{"log10_fvu_val":"AH/8"}};})();\n`;
  const files = {
    'overlay.js': 'window.RESULTS_V2_PRIVATE = ' + JSON.stringify({
      base: '', methods: [{ key: method, label: method.toUpperCase() }], cells: {}, status: {}, timing: {},
      pp: { [method]: { 'feynman|16': [1, 2] } }, pred: { [method]: { 'feynman|16': [1] } }, pred_block: 500 }) + ';\n',
    'paired.js': `window.RESULTS_V2_PAIRED=window.RESULTS_V2_PAIRED||{};window.RESULTS_V2_PAIRED["${REL}"]={"${method}|e2e":{}};\n`,
    'hist/log10_fvu_val.js': `window.RESULTS_V2_HIST=window.RESULTS_V2_HIST||{};window.RESULTS_V2_HIST["${REL}"]={"${method}":[1,2,3]};\n`,
    [`pp/${method}/feynman/16.1.js`]: pp(16, 1),
    [`pp/${method}/feynman/16.2.js`]: pp(16, 2),
    [`pred/${method}/feynman/16.1.0.js`]: `window.RESULTS_V2_PRED=window.RESULTS_V2_PRED||{};(function(){var R=window.RESULTS_V2_PRED;` +
      `R["${REL}"]=R["${REL}"]||{};R["${REL}"]["${method}|feynman|16|1|0"]={"0":["* x1 x2",3],"1":null,"2":["+ x1 0.5",1]};})();\n`,
    [`pv/${method}/feynman/16.1.js`]: `window.RESULTS_V2_PV=window.RESULTS_V2_PV||{};window.RESULTS_V2_PV["${method}|feynman|16|1"]=[1,2,3];\n`,
    'ranks/log10_fvu_val.js': `window.RESULTS_V2_RANKS=window.RESULTS_V2_RANKS||{};window.RESULTS_V2_RANKS["${method}|e2e"]={"feynman":{"16":[100,3,4]}};\n`,
  };
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), text);
  }
  return Object.keys(files).filter((p) => /^(pp|pred|pv|ranks|paired|hist)\//.test(p));   // the files sealed one by one
}
const made = [];
function scratch() { const d = mkdtempSync(join(tmpdir(), 'seal-files-')); made.push(d); return d; }
test.afterEach(() => { while (made.length) { rmSync(made.pop(), { recursive: true, force: true }); } });
function seal(args, env) {
  const r = spawnSync(process.execPath, [SEAL, REL, ...args], { env: { ...process.env, SRBF_SEAL_KEY: undefined, ...env }, encoding: 'utf8' });
  expect(r.status, r.stderr).toBe(0);
  return r.stdout;
}
function sealedFiles(out) {   // {name: bytes} of everything in the sealed/ next to the out file
  const dir = join(dirname(out), 'sealed');
  return existsSync(dir) ? Object.fromEntries(readdirSync(dir).sort().map((n) => [n, readFileSync(join(dir, n))])) : {};
}

// ---- opening it as a page does ---------------------------------------------------------------------------------
function run(src) { const window = {}; new Function('window', src)(window); return window; }   // a script, as a <script> runs it
async function gunzip(gz) { return Buffer.from(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()); }
async function openOverlays(sealedJs, key) {   // -> the payloads this key opens (explorer_v2.js, openOne)
  const W = run(sealedJs);
  const envs = [W.RESULTS_V2_SEALED && W.RESULTS_V2_SEALED[REL]].concat((W.RESULTS_V2_SEALED_MORE || {})[REL] || []).filter(Boolean);
  const out = [];
  for (const env of envs) {
    try {
      const base = await subtle.importKey('raw', utf8(key), 'PBKDF2', false, ['deriveKey']);
      const k = await subtle.deriveKey({ name: 'PBKDF2', salt: bytesOf(env.salt), iterations: env.iter, hash: 'SHA-256' },
        base, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
      out.push((await gunzip(await subtle.decrypt({ name: 'AES-GCM', iv: bytesOf(env.iv) }, k, bytesOf(env.ct)))).toString('utf8'));
    } catch (e) { /* not this key's */ }
  }
  return out;
}
async function importKeys(enc, name) {
  return { enc: await subtle.importKey('raw', enc, 'AES-GCM', false, ['decrypt']),
    name: await subtle.importKey('raw', name, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']) };
}
async function handedKeys(payload) {   // the file keys an opened overlay hands over
  const sf = run(payload).RESULTS_V2_PRIVATE.sealed_files;
  return importKeys(bytesOf(sf.enc), bytesOf(sf.name));
}
function ivOf(kName, path, gz) { return createHmac('sha256', kName).update('iv\0' + path + '\0').update(gz).digest().subarray(0, 12); }
async function nameOf(keys, path) { return Buffer.from(await subtle.sign('HMAC', keys.name, utf8('name\0' + path))).toString('hex').slice(0, 32); }
async function openLine(line, keys, path) {   // -> the content of one sealed file read as `path`, or null
  try {
    const W = run(line), name = Object.keys(W.RESULTS_V2_SEALED_FILES)[0], env = W.RESULTS_V2_SEALED_FILES[name];
    const gz = await subtle.decrypt({ name: 'AES-GCM', iv: bytesOf(env.iv), additionalData: utf8(REL + '/' + path) }, keys.enc, bytesOf(env.ct));
    return await gunzip(gz);
  } catch (e) { return null; }
}
async function openFile(out, keys, path) {   // -> a path's content, fetched by its name, or null
  const at = join(dirname(out), 'sealed', (await nameOf(keys, path)) + '.js');
  return existsSync(at) ? openLine(readFileSync(at, 'utf8'), keys, path) : null;
}

// ---- the tests -------------------------------------------------------------------------------------------------
test('a key holder opens every per-problem file by its name, and each file only as its own path', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'out', 'sealed.js');
  const paths = overlay(src, 'hidden-method-a');
  mkdirSync(dirname(out));
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const opened = await openOverlays(readFileSync(out, 'utf8'), KEY);
  expect(opened).toHaveLength(1);
  expect(run(opened[0]).RESULTS_V2_PRIVATE.methods[0].key).toBe('hidden-method-a');
  expect(opened[0]).toContain('RESULTS_V2_PAIRED');                  // the ordinary payload scripts are in the overlay
  expect(opened[0]).not.toContain('|feynman|16|1');                  // the per-problem files are not
  expect(opened[0]).not.toContain('RESULTS_V2_RANKS');                // nor the per-metric ones
  expect(opened[0]).not.toContain('RESULTS_V2_HIST');
  expect(paths).toEqual(expect.arrayContaining(['ranks/log10_fvu_val.js', 'hist/log10_fvu_val.js']));
  const keys = await handedKeys(opened[0]);
  for (const p of paths) { expect((await openFile(out, keys, p)) || '', p).toEqual(readFileSync(join(src, p))); }
  expect(Object.keys(sealedFiles(out))).toHaveLength(paths.length);
  // the path is the additional data: a file served under another file's name does not open
  const line = readFileSync(join(dir, 'out', 'sealed', (await nameOf(keys, paths[0])) + '.js'), 'utf8');
  expect(await openLine(line, keys, paths[0])).not.toBeNull();
  expect(await openLine(line, keys, paths[1])).toBeNull();
});

test('a reseal leaves every unchanged file byte for byte, and a changed file changes alone', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'sealed.js');
  const paths = overlay(src, 'hidden-method-a');
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const first = sealedFiles(out), envelope = readFileSync(out, 'utf8');
  expect(seal([src, out], { SRBF_SEAL_KEY: KEY })).toContain('0 written');
  expect(sealedFiles(out)).toEqual(first);
  const fresh = join(dir, 'fresh', 'sealed.js');                     // and a seal from nothing writes the same bytes
  mkdirSync(dirname(fresh));
  seal([src, fresh], { SRBF_SEAL_KEY: KEY });
  expect(sealedFiles(fresh)).toEqual(first);
  expect(readFileSync(out, 'utf8')).not.toEqual(envelope);           // the overlay's own envelope stays fresh per run
  writeFileSync(join(src, paths[1]), readFileSync(join(src, paths[1]), 'utf8').replace('"n":3', '"n":4'));
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const after = sealedFiles(out);
  expect(Object.keys(after)).toEqual(Object.keys(first));
  const keys = await handedKeys((await openOverlays(readFileSync(out, 'utf8'), KEY))[0]);
  const changed = (await nameOf(keys, paths[1])) + '.js';
  for (const n of Object.keys(first)) { expect(after[n].equals(first[n]), n).toBe(n !== changed); }
  expect((await openFile(out, keys, paths[1])).toString()).toContain('"n":4');
});

test('a sealed file that holds the same content is kept under its own IV, whatever gzip wrote', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'sealed.js');
  const paths = overlay(src, 'hidden-method-a');
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const P = run((await openOverlays(readFileSync(out, 'utf8'), KEY))[0]).RESULTS_V2_PRIVATE;
  const keys = await handedKeys((await openOverlays(readFileSync(out, 'utf8'), KEY))[0]);
  // what another machine's gzip might have written: the same content, other gzip bytes, and so another IV
  const at = join(dir, 'sealed', (await nameOf(keys, paths[0])) + '.js');
  const W = run(readFileSync(at, 'utf8')), name = Object.keys(W.RESULTS_V2_SEALED_FILES)[0];
  const gz = gzipSync(readFileSync(join(src, paths[0])), { level: 9 });
  gz[9] = 3;   // the OS byte a Unix gzip writes
  const iv = ivOf(bytesOf(P.sealed_files.name), paths[0], gz).toString('base64');
  expect(iv).not.toEqual(W.RESULTS_V2_SEALED_FILES[name].iv);
  const k = await subtle.importKey('raw', bytesOf(P.sealed_files.enc), 'AES-GCM', false, ['encrypt']);
  const ct = Buffer.from(await subtle.encrypt({ name: 'AES-GCM', iv: bytesOf(iv), additionalData: utf8(REL + '/' + paths[0]) }, k, gz));
  const other = `(window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})["${name}"]=${JSON.stringify({ iv, ct: ct.toString('base64') })};\n`;
  writeFileSync(at, other);
  expect(seal([src, out], { SRBF_SEAL_KEY: KEY })).toContain('0 written');
  expect(readFileSync(at, 'utf8')).toEqual(other);
  expect((await openFile(out, keys, paths[0])) || '').toEqual(readFileSync(join(src, paths[0])));
});

test('a file withdrawn from the source leaves sealed/, and nothing else there is touched', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'sealed.js');
  const paths = overlay(src, 'hidden-method-a');
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const keys = await handedKeys((await openOverlays(readFileSync(out, 'utf8'), KEY))[0]);
  writeFileSync(join(dir, 'sealed', 'keep.txt'), 'not a sealed file');
  const before = sealedFiles(out);
  rmSync(join(src, paths[2]));
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const gone = (await nameOf(keys, paths[2])) + '.js';
  const after = sealedFiles(out);
  expect(Object.keys(after).sort()).toEqual(Object.keys(before).filter((n) => n !== gone).sort());
  for (const n of Object.keys(after)) { expect(after[n].equals(before[n]), n).toBe(true); }
  // a source without any per-problem file leaves no sealed file behind
  for (const d of ['pp', 'pred', 'pv', 'ranks', 'paired', 'hist']) { rmSync(join(src, d), { recursive: true, force: true }); }
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  expect(Object.keys(sealedFiles(out))).toEqual(['keep.txt']);
});

test('a sealed file gives away neither its method nor its path', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'sealed.js');
  const paths = overlay(src, 'hidden-method-a');
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const files = sealedFiles(out);
  expect(Object.keys(files)).toHaveLength(paths.length);
  const sha = (s) => createHash('sha256').update(s).digest('hex').slice(0, 32);
  for (const [name, bytes] of Object.entries(files)) {
    const text = bytes.toString('utf8');
    expect(name).toMatch(/^[0-9a-f]{32}\.js$/);
    expect(LINE.exec(text)?.[1], 'one envelope line, named by its own file name').toBe(name.slice(0, 32));
    // the line is the fixed frame, the name and two base64 strings (LINE above), so a needle with a character base64
    // never has ('-', '.', '|', '_') cannot turn up by chance; a short base64-only needle like "pp/" could
    for (const leak of ['hidden-method-a', 'feynman|16', 'pred/hidden', 'pp/hidden', '16.1.js', 'RESULTS_V2_PP', 'RESULTS_V2_PRED']) {
      expect(name + text, leak).not.toContain(leak);
    }
    for (const p of paths) { expect(name.slice(0, 32)).not.toBe(sha(p)); expect(name.slice(0, 32)).not.toBe(sha('name\0' + p)); }
  }
  expect(readFileSync(out, 'utf8')).not.toContain('hidden-method-a');
});

test('the overlay hands over the two file keys, and a wrong key opens nothing', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'sealed.js');
  const paths = overlay(src, 'hidden-method-a');
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const opened = await openOverlays(readFileSync(out, 'utf8'), KEY);
  const sf = run(opened[0]).RESULTS_V2_PRIVATE.sealed_files;
  expect(Object.keys(sf).sort()).toEqual(['enc', 'name']);
  // the two halves of PBKDF2-HMAC-SHA256(key, "srbf-sealed-files/<release>", 600000 iterations, 512 bits)
  const material = pbkdf2Sync(KEY, 'srbf-sealed-files/' + REL, 600000, 64, 'sha256');
  expect(Buffer.from(sf.enc, 'base64').equals(material.subarray(0, 32))).toBe(true);
  expect(Buffer.from(sf.name, 'base64').equals(material.subarray(32, 64))).toBe(true);
  // each IV is the HMAC of the path and the very bytes it encrypts: an IV repeats only for identical plaintext
  const encKey = await subtle.importKey('raw', material.subarray(0, 32), 'AES-GCM', false, ['decrypt']);
  const named = await importKeys(material.subarray(0, 32), material.subarray(32, 64));
  for (const p of paths) {
    const W = run(readFileSync(join(dir, 'sealed', (await nameOf(named, p)) + '.js'), 'utf8')), env = Object.values(W.RESULTS_V2_SEALED_FILES)[0];
    const gz = Buffer.from(await subtle.decrypt({ name: 'AES-GCM', iv: bytesOf(env.iv), additionalData: utf8(REL + '/' + p) }, encKey, bytesOf(env.ct)));
    expect(env.iv, p).toEqual(ivOf(material.subarray(32, 64), p, gz).toString('base64'));
  }
  // a wrong key opens no overlay, and what it would derive names no file and opens none
  const wrong = 'not-the-key-at-all-really';
  expect(await openOverlays(readFileSync(out, 'utf8'), wrong)).toEqual([]);
  const m = pbkdf2Sync(wrong, 'srbf-sealed-files/' + REL, 600000, 64, 'sha256');
  const wrongKeys = await importKeys(m.subarray(0, 32), m.subarray(32, 64));
  const right = await handedKeys(opened[0]);
  for (const p of paths) {
    expect(await openFile(out, wrongKeys, p), p).toBeNull();
    const line = readFileSync(join(dir, 'sealed', (await nameOf(right, p)) + '.js'), 'utf8');
    expect(await openLine(line, wrongKeys, p), p).toBeNull();
  }
});

test('two sources sealed under two keys keep their files apart', async () => {
  const dir = scratch(), a = join(dir, 'a'), b = join(dir, 'b'), out = join(dir, 'sealed.js');
  const pa = overlay(a, 'hidden-method-a'), pb = overlay(b, 'hidden-method-b');
  seal(['--source', a + '=K_ONE', '--source', b + '=K_TWO', '--out', out], { K_ONE: KEY, K_TWO: KEY_TWO });
  const one = await openOverlays(readFileSync(out, 'utf8'), KEY), two = await openOverlays(readFileSync(out, 'utf8'), KEY_TWO);
  expect(one).toHaveLength(1); expect(two).toHaveLength(1);
  expect(run(one[0]).RESULTS_V2_PRIVATE.methods[0].key).toBe('hidden-method-a');
  expect(run(two[0]).RESULTS_V2_PRIVATE.methods[0].key).toBe('hidden-method-b');
  const k1 = await handedKeys(one[0]), k2 = await handedKeys(two[0]);
  const files = sealedFiles(out);
  expect(Object.keys(files)).toHaveLength(pa.length + pb.length);   // no name shared, not even for a shared path
  expect(pa.filter((p) => pb.includes(p))).toEqual(['hist/log10_fvu_val.js', 'ranks/log10_fvu_val.js']);
  for (const [own, other, ownSrc, otherSrc, keys, foreign] of [[pa, pb, a, b, k1, k2], [pb, pa, b, a, k2, k1]]) {
    for (const p of own) {
      expect(await openFile(out, keys, p) || '', p).toEqual(readFileSync(join(ownSrc, p)));
      // the other key names the other overlay's own file at a shared path, and no file at any other path
      const there = await openFile(out, foreign, p);
      if (other.includes(p)) { expect(there || '', p).toEqual(readFileSync(join(otherSrc, p))); } else { expect(there, p).toBeNull(); }
      const line = readFileSync(join(dir, 'sealed', (await nameOf(keys, p)) + '.js'), 'utf8');
      expect(await openLine(line, foreign, p), 'the other key does not open it: ' + p).toBeNull();
    }
    for (const p of other.filter((q) => !own.includes(q))) { expect(await openFile(out, keys, p), p).toBeNull(); }
  }
  // the cleanup keeps every source's files: a reseal of both removes nothing, a reseal without the second removes its own
  expect(seal(['--source', a + '=K_ONE', '--source', b + '=K_TWO', '--out', out], { K_ONE: KEY, K_TWO: KEY_TWO })).toContain('0 written, ' + (pa.length + pb.length) + ' unchanged, 0 removed');
  expect(sealedFiles(out)).toEqual(files);
  seal(['--source', a + '=K_ONE', '--out', out], { K_ONE: KEY });
  expect(Object.keys(sealedFiles(out))).toHaveLength(pa.length);
  for (const p of pa) { expect(await openFile(out, k1, p) || '', p).toEqual(readFileSync(join(a, p))); }
});

test('a source without per-problem files is sealed exactly as before', async () => {
  const dir = scratch(), src = join(dir, 'src'), out = join(dir, 'sealed.js');
  overlay(src, 'hidden-method-a');
  for (const d of ['pp', 'pred', 'pv', 'ranks', 'paired', 'hist']) { rmSync(join(src, d), { recursive: true, force: true }); }
  seal([src, out], { SRBF_SEAL_KEY: KEY });
  const opened = await openOverlays(readFileSync(out, 'utf8'), KEY);
  expect(opened).toEqual([['overlay.js', 'paired.js'].map((f) => readFileSync(join(src, f), 'utf8')).join('\n;\n')]);
  expect(existsSync(join(dir, 'sealed'))).toBe(false);
});

test('the committed page fixture opens with the test key and is what a reseal of its source writes', async () => {
  // tests/fixtures/sealed_files: the overlay a page test serves as data/2026-09/sealed.js and data/2026-09/sealed/
  const out = join(FIXTURE, 'sealed.js');
  const opened = await openOverlays(readFileSync(out, 'utf8'), KEY);
  expect(opened).toHaveLength(1);
  const P = run(opened[0]).RESULTS_V2_PRIVATE;
  expect(P.methods.map((m) => m.key)).toEqual(['fixture-files']);
  expect(P.pp).toEqual({ 'fixture-files': { 'feynman|16': [1, 2], 'feynman|32': [1, 2] } });
  expect(P.pred).toEqual({ 'fixture-files': { 'feynman|16': [1], 'feynman|32': [1] } });
  const keys = await handedKeys(opened[0]);
  const paths = [];
  for (const [cr, runs] of Object.entries(P.pp['fixture-files'])) {
    const [c, r] = cr.split('|');
    runs.forEach((d) => paths.push(`pp/fixture-files/${c}/${r}.${d}.js`));
  }
  for (const [cr, runs] of Object.entries(P.pred['fixture-files'])) {
    const [c, r] = cr.split('|');
    runs.forEach((d) => paths.push(`pred/fixture-files/${c}/${r}.${d}.0.js`));
  }
  for (const p of paths) {
    const content = await openFile(out, keys, p);
    expect(content || '', p).toEqual(readFileSync(join(FIXTURE, 'source', p)));
    const W = run(content.toString('utf8'));   // and it is the script the views read
    expect(Object.keys((W.RESULTS_V2_PP || W.RESULTS_V2_PRED)[REL])[0]).toMatch(/^fixture-files\|feynman\|(16|32)\|/);
  }
  expect(Object.keys(sealedFiles(out))).toHaveLength(paths.length);
  // a reseal of the committed source over the committed files rewrites none of them (the fixture is not stale; another
  // node's gzip may compress differently, which is why the reseal starts from the committed files as the board's does)
  const dir = scratch();
  cpSync(join(FIXTURE, 'source'), join(dir, 'source'), { recursive: true });
  cpSync(join(FIXTURE, 'sealed'), join(dir, 'sealed'), { recursive: true });
  expect(seal([join(dir, 'source'), join(dir, 'sealed.js')], { SRBF_SEAL_KEY: KEY })).toContain(paths.length + ' file(s), 0 written');
  expect(sealedFiles(join(dir, 'sealed.js'))).toEqual(sealedFiles(out));
});

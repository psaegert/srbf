// Seal a release's private units so that each reader opens exactly the methods they were given.
//
//   SRBF_SEAL_MASTER='<master secret>' SRBF_KEY_A='<key>' SRBF_KEY_B='<key>' node tools/seal.mjs <release> \
//     --units <private dir> --master SRBF_SEAL_MASTER --grant SRBF_KEY_A='*' --grant SRBF_KEY_B='gamma-*' \
//     [--keyrings 8] [--out <file>]
//
// The private dir holds the units scripts/site_export_v2.py writes: <dir>/m/<method>/ for one method and
// <dir>/p/<a>/<b>/ (a < b) for the cells of two private methods against each other. Each unit is a unit.js (an overlay
// pushed onto window.RESULTS_V2_PRIVATE_UNITS, run by the page exactly as a <script> tag would run it; this tool never
// parses it) and its files under hist/, ranks/, paired/, pp/, pred/ and pv/.
//
// THE LOCKS. Every unit is locked under a secret of its own, made from the master secret, which never leaves the
// machine that seals:
//   a method's unit   S_m  = HMAC-SHA256(master, "srbf-unit\0" + release + "\0" + method)
//   a pair's unit     S_ab = HMAC-SHA256(S_a || S_b, "srbf-pair\0" + release + "\0" + a + "\0" + b), a < b
// so a pair's secret is made only by a reader who holds both methods. From a unit's secret S:
//   id    = hex(HMAC(S, "srbf-unit-id"))[0:32]                     the unit's name in the sealed file
//   kPay  = HMAC(S, "srbf-unit-payload")                           the AES-256-GCM key of its unit.js
//   kEnc  = HMAC(S, "srbf-unit-file-enc"), kName = HMAC(S, "srbf-unit-file-name")    its files, one by one (below)
// The master makes every secret again on every run, so an unchanged input reseals to the same bytes (no git churn).
//
// THE KEYS. A reader holds one key, a generated passphrase. Each --grant ENV=PATTERNS gives the key in that environment
// variable the methods PATTERNS name (comma-separated method keys; a pattern ending in "*" takes every method whose key
// starts with what precedes it, "*" alone every method; a pattern that matches no method fails the run). Per grant, one
// keyring:
//   bits  = PBKDF2-HMAC-SHA256(key, "srbf-keyring/" + release, 600000 iterations, 512 bits)
//   plain = JSON {"v": 2, "methods": [[method, base64 S_m], ...]}, padded with spaces to KEYRING_BYTES
//   iv    = HMAC(bits[32:64], plain)[0:12];  ct = AES-256-GCM(bits[0:32], iv, plain, additional data release + "/keyring")
// The page derives once per key typed, tries every keyring, and opens the units of the methods it now holds and of every
// pair of them. The keyrings are padded to --keyrings (default 8) with ones of the same size made from the master, and
// written in an order that says nothing: the file shows neither how many readers there are nor what each holds.
//
// THE FILE. data/<release>/sealed.js is one line:
//   window.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED["<release>"]={"v":2,
//     "kdf":"PBKDF2-SHA256","iter":600000,"salt":"<base64>","keyrings":[{"iv","ct"},...],"units":{"<id>":{"iv","ct"},...}};
// with a unit's envelope gz = gzip(unit.js) (a fixed header), iv = HMAC(kName, "payload\0" || gz)[0:12],
// ct = AES-256-GCM(kPay, iv, gz, additional data release + "/unit/" + id).
//
// PER-PROBLEM FILES. A unit's files are sealed one by one into <out dir>/sealed/<name>.js (data/<release>/sealed/ for
// the default out file) and fetched only when a reader opens a view that reads one. With path = the file's POSIX path
// below its unit, e.g. "pp/<method>/feynman/16.1.js" or "ranks/success.js":
//   name = hex(HMAC(kName, "name\0" + path))[0:32]                  reveals neither the unit nor the path
//   gz   = gzip -9 (content), with a fixed header (no time, OS "unknown")
//   iv   = HMAC(kName, "iv\0" + path + "\0" || gz)[0:12]             an IV repeats only for identical plaintext
//   ct   = AES-256-GCM(kEnc, iv, gz, additional data "<release>/<path>")                  bound to its path
// and the file is one line:
//   (window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})["<name>"]={"iv":"<base64>","ct":"<base64>"};
// A file already there that opens to the same content is kept as it is, under its own IV, whatever this machine's gzip
// would write. A file that is in no unit any more is removed (only <32 hex digits>.js names are touched).
//
// Every secret comes from the environment, never from argv: an argument is visible in `ps` and lands in shell history.
// Nothing written reveals a key or the master. Before anything is written, every keyring is opened again with every
// key (each opens its own, to exactly its methods, and no other), every unit with its own secret and no other unit's,
// and every file the way the page opens it.
import { webcrypto as crypto, createHmac } from "node:crypto";
import { gzipSync, gunzipSync } from "node:zlib";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync, unlinkSync, rmdirSync } from "node:fs";
import { join, resolve, dirname, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ITERATIONS = 600000;   // OWASP 2023 for PBKDF2-HMAC-SHA256: ~0.3 s in a browser, the only brake on an offline guess
const KEYRING_BYTES = 4096;  // every keyring's plaintext, padded: about fifty methods
const FILE_DIRS = ["pp", "pred", "pv", "ranks", "paired", "hist"];   // a unit's files, sealed one by one
const SEALED_NAME = /^[0-9a-f]{32}\.js$/;
const FILE_LINE = /^\(window\.RESULTS_V2_SEALED_FILES=window\.RESULTS_V2_SEALED_FILES\|\|\{\}\)\["([0-9a-f]{32})"\]=\{"iv":"([A-Za-z0-9+/]+=*)","ct":"([A-Za-z0-9+/]+=*)"\};\n$/;
const utf8 = (s) => Buffer.from(s, "utf8");
const b64 = (bytes) => Buffer.from(bytes).toString("base64");
const u8 = (s) => new Uint8Array(Buffer.from(s, "base64"));
const hmac = (key, ...parts) => parts.reduce((h, p) => h.update(p), createHmac("sha256", key)).digest();

const USAGE = "usage: node tools/seal.mjs <release> --units <private dir> --master <ENV VAR> --grant <ENV VAR>=<pattern>[,<pattern>...] " +
  "[--grant ...] [--keyrings N] [--out <file>]";
const fail = (msg) => { console.error(msg + "; nothing written"); process.exit(1); };
const args = process.argv.slice(2);
const release = args[0];
if (!release || release.startsWith("--")) { console.error(USAGE); process.exit(2); }
let unitsDir, masterVar, outArg, padTo = 8;
const grants = [];   // [environment variable, [patterns]]
for (let i = 1; i < args.length; i++) {
  const v = args[i + 1];
  if (args[i] === "--units" && v) { unitsDir = v; i++; }
  else if (args[i] === "--master" && v) { masterVar = v; i++; }
  else if (args[i] === "--grant" && v && v.includes("=")) { grants.push([v.slice(0, v.indexOf("=")), v.slice(v.indexOf("=") + 1).split(",").filter(Boolean)]); i++; }
  else if (args[i] === "--keyrings" && v && /^\d+$/.test(v)) { padTo = +v; i++; }
  else if (args[i] === "--out" && v) { outArg = v; i++; }
  else { console.error(USAGE); process.exit(2); }
}
if (!unitsDir || !masterVar || !grants.length) { console.error(USAGE); process.exit(2); }
const secret = (name) => {
  const s = process.env[name];
  if (!s || s.length < 16) {
    console.error(name + " is missing or shorter than 16 characters.\n" +
      "The sealed file is public, so the keys are the only thing protecting it: use generated ones, e.g.\n" +
      "  node -e \"console.log(require('crypto').randomBytes(32).toString('base64url'))\"");
    process.exit(2);
  }
  return s;
};
const master = utf8(secret(masterVar));
const keys = grants.map(([name]) => secret(name));
if (new Set(keys).size !== keys.length) { fail("two grants share a key: one key opens one keyring, so give that key the union of their methods"); }
if (keys.some((k) => utf8(k).equals(master))) { fail("a reader's key is the master secret"); }
const out = resolve(outArg || join(SITE, "data", release, "sealed.js"));
const sealedDir = join(dirname(out), "sealed");

// ---- the units -----------------------------------------------------------------------------------------------------
function scripts(dir) {
  return readdirSync(dir).flatMap(function (name) {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? scripts(p) : (name.endsWith(".js") ? [p] : []);
  }).sort();
}
const dirs = (p) => existsSync(p) ? readdirSync(p).filter((n) => statSync(join(p, n)).isDirectory()).sort() : [];
const root = resolve(unitsDir);
const methods = dirs(join(root, "m"));
if (!methods.length) { fail("no method units under " + join(root, "m")); }
const unitSecret = {};   // method -> S_m
for (const m of methods) { unitSecret[m] = hmac(master, utf8("srbf-unit\0" + release + "\0" + m)); }
const pairSecret = (a, b) => hmac(Buffer.concat([unitSecret[a], unitSecret[b]]), utf8("srbf-pair\0" + release + "\0" + a + "\0" + b));
const units = methods.map((m) => ({ label: "m/" + m, dir: join(root, "m", m), secret: unitSecret[m] }));
for (const a of dirs(join(root, "p"))) {
  for (const b of dirs(join(root, "p", a))) {
    if (!(a < b) || !unitSecret[a] || !unitSecret[b]) { fail("p/" + a + "/" + b + " is not a pair of two method units in order"); }
    units.push({ label: "p/" + a + "/" + b, dir: join(root, "p", a, b), secret: pairSecret(a, b) });
  }
}
for (const u of units) {
  u.id = hmac(u.secret, utf8("srbf-unit-id")).toString("hex").slice(0, 32);
  u.kPay = hmac(u.secret, utf8("srbf-unit-payload"));
  u.kEnc = hmac(u.secret, utf8("srbf-unit-file-enc"));
  u.kName = hmac(u.secret, utf8("srbf-unit-file-name"));
  const all = scripts(u.dir), perFile = (f) => FILE_DIRS.includes(relative(u.dir, f).split(sep)[0]);
  const rest = all.filter((f) => !perFile(f));
  if (rest.length !== 1 || relative(u.dir, rest[0]) !== "unit.js") { fail(u.label + " holds " + rest.map((f) => relative(u.dir, f)).join(", ") + " where only unit.js belongs"); }
  u.payload = readFileSync(rest[0]);
  u.files = all.filter(perFile);
}
if (new Set(units.map((u) => u.id)).size !== units.length) { fail("two units share an id"); }

// ---- sealing -------------------------------------------------------------------------------------------------------
function gzipFixed(content) {   // a header without time or OS: equal content, equal bytes, on any machine
  const gz = gzipSync(content, { level: 9 });
  gz.writeUInt32LE(0, 4);   // MTIME: none
  gz[8] = 2;                // XFL: maximum compression
  gz[9] = 255;              // OS: unknown
  return gz;
}
async function aesKey(raw, use) { return crypto.subtle.importKey("raw", raw, "AES-GCM", false, [use]); }
async function encrypt(raw, iv, plain, aad) {
  return new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad }, await aesKey(raw, "encrypt"), plain));
}
async function decrypt(raw, iv, ct, aad) {   // -> the plaintext, or null when the key, the IV or the data do not fit
  try { return Buffer.from(await crypto.subtle.decrypt({ name: "AES-GCM", iv, additionalData: aad }, await aesKey(raw, "decrypt"), ct)); }
  catch (e) { return null; }
}
async function keyringBits(key) {
  const base = await crypto.subtle.importKey("raw", utf8(key), "PBKDF2", false, ["deriveBits"]);
  const bits = Buffer.from(await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt: utf8("srbf-keyring/" + release), iterations: ITERATIONS, hash: "SHA-256" }, base, 512));
  return { enc: bits.subarray(0, 32), mac: bits.subarray(32, 64) };
}
function stream(seed, n) {   // n bytes that look random, made from the master: the padding keyrings, the same every run
  const parts = [];
  for (let i = 0, have = 0; have < n; i++) { const h = hmac(master, utf8(seed + "\0" + i)); parts.push(h); have += h.length; }
  return Buffer.concat(parts).subarray(0, n);
}
const RING_AAD = utf8(release + "/keyring");

// the keyrings: what each grant holds, then the padding
const granted = grants.map(function ([name, patterns]) {
  const held = new Set();
  for (const p of patterns) {
    const hit = methods.filter((m) => p.endsWith("*") ? m.startsWith(p.slice(0, -1)) : m === p);
    if (!hit.length) { fail("the grant " + name + " names " + JSON.stringify(p) + ", which matches no method unit"); }
    hit.forEach((m) => held.add(m));
  }
  return methods.filter((m) => held.has(m));
});
const keyrings = [];
for (let i = 0; i < grants.length; i++) {
  const json = JSON.stringify({ v: 2, methods: granted[i].map((m) => [m, b64(unitSecret[m])]) });
  if (json.length > KEYRING_BYTES) { fail("the grant " + grants[i][0] + " holds more methods than a keyring has room for"); }
  const plain = utf8(json.padEnd(KEYRING_BYTES, " "));
  const bits = await keyringBits(keys[i]);
  const iv = hmac(bits.mac, plain).subarray(0, 12);
  keyrings.push({ iv, ct: await encrypt(bits.enc, iv, plain, RING_AAD) });
}
for (let i = grants.length; i < padTo; i++) {
  keyrings.push({ iv: stream("srbf-pad-iv\0" + release + "\0" + i, 12), ct: stream("srbf-pad-ct\0" + release + "\0" + i, KEYRING_BYTES + 16) });
}
keyrings.sort((x, y) => Buffer.compare(Buffer.from(x.ct), Buffer.from(y.ct)));

// the units' payloads
for (const u of units) {
  const gz = gzipFixed(u.payload), iv = hmac(u.kName, utf8("payload\0"), gz).subarray(0, 12);
  u.aad = utf8(release + "/unit/" + u.id);
  u.env = { iv, ct: await encrypt(u.kPay, iv, gz, u.aad) };
}
const envelopes = {};
for (const u of [...units].sort((x, y) => (x.id < y.id ? -1 : 1))) { envelopes[u.id] = { iv: b64(u.env.iv), ct: b64(u.env.ct) }; }

// the files, one by one
const fileName = (kName, path) => hmac(kName, utf8("name\0" + path)).toString("hex").slice(0, 32);
const fileLine = (name, iv, ct) => "(window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})[" + JSON.stringify(name) + "]=" +
  JSON.stringify({ iv: b64(iv), ct: b64(ct) }) + ";\n";
async function openLine(line, kEnc, aad) {   // one file's line opened the way the page opens it -> its content, or null
  const m = FILE_LINE.exec(line);
  if (!m) { return null; }
  const gz = await decrypt(kEnc, u8(m[2]), u8(m[3]), aad);
  if (!gz) { return null; }
  if (typeof DecompressionStream === "undefined") { return gunzipSync(gz); }
  return Buffer.from(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
}
const files = [], names = new Set();
for (const u of units) {
  for (const f of u.files) {
    const path = relative(u.dir, f).split(sep).join("/"), content = readFileSync(f), name = fileName(u.kName, path);
    const aad = utf8(release + "/" + path);
    if (names.has(name)) { fail("two files share the name " + name); }
    names.add(name);
    const at = join(sealedDir, name + ".js"), there = existsSync(at) ? readFileSync(at, "utf8") : null;
    const old = there ? await openLine(there, u.kEnc, aad) : null;
    let line = there;   // this content is published already: its bytes stay, under their own IV
    if (!old || !old.equals(content)) {
      const gz = gzipFixed(content), iv = hmac(u.kName, utf8("iv\0" + path + "\0"), gz).subarray(0, 12);
      line = fileLine(name, iv, await encrypt(u.kEnc, iv, gz, aad));
    }
    files.push({ unit: u, path, name, aad, content, line, at, changed: line !== there });
  }
}

// ---- before anything is written: open everything again as the page will ------------------------------------------
for (let i = 0; i < grants.length; i++) {
  const bits = await keyringBits(keys[i]);
  const opened = [];
  for (const r of keyrings) { const plain = await decrypt(bits.enc, r.iv, r.ct, RING_AAD); if (plain) { opened.push(plain); } }
  if (opened.length !== 1) { fail("the key of " + grants[i][0] + " opens " + opened.length + " keyrings, not one"); }
  const held = JSON.parse(opened[0].toString("utf8")).methods;
  if (JSON.stringify(held.map(([m]) => m)) !== JSON.stringify(granted[i]) || held.some(([m, s]) => !Buffer.from(u8(s)).equals(unitSecret[m]))) {
    fail("the keyring of " + grants[i][0] + " does not hold exactly its methods");
  }
}
for (const u of units) {
  const back = await decrypt(u.kPay, u.env.iv, u.env.ct, u.aad);
  if (!back || !gunzipSync(back).equals(u.payload)) { fail(u.label + " does not open to its unit.js"); }
  for (const w of units) {
    if (w !== u && await decrypt(w.kPay, u.env.iv, u.env.ct, u.aad)) { fail("the secret of " + w.label + " opens " + u.label); }
  }
}
for (const f of files) {
  const m = FILE_LINE.exec(f.line);
  if (!m || m[1] !== f.name) { fail("the sealed " + f.path + " of " + f.unit.label + " is not named as the page will look for it"); }
  const back = await openLine(f.line, f.unit.kEnc, f.aad);
  if (!back || !back.equals(f.content)) { fail("the sealed " + f.path + " of " + f.unit.label + " does not open to what was sealed"); }
}

// ---- write: the files (an unchanged one is not touched), the envelope file, then remove the withdrawn --------------
let written = 0, removed = 0;
if (files.length) { mkdirSync(sealedDir, { recursive: true }); }
for (const f of files) { if (f.changed) { writeFileSync(f.at, f.line); written++; } }
const sealed = { v: 2, kdf: "PBKDF2-SHA256", iter: ITERATIONS, salt: b64(utf8("srbf-keyring/" + release)),
  keyrings: keyrings.map((r) => ({ iv: b64(r.iv), ct: b64(r.ct) })), units: envelopes };
const rel = JSON.stringify(release);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, "window.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED[" + rel + "]=" + JSON.stringify(sealed) + ";\n");
if (existsSync(sealedDir)) {
  for (const n of readdirSync(sealedDir)) {
    if (SEALED_NAME.test(n) && !names.has(n.slice(0, 32))) { unlinkSync(join(sealedDir, n)); removed++; }
  }
  if (!readdirSync(sealedDir).length) { rmdirSync(sealedDir); }
}
console.log(methods.length + " method unit(s) and " + (units.length - methods.length) + " pair unit(s) from " + relative(process.cwd(), root) + "; " +
  grants.map(([name], i) => name + ": " + granted[i].length + " method(s)").join(", ") + "; " + keyrings.length + " keyrings -> " +
  (statSync(out).size / 1048576).toFixed(2) + " MB at " + relative(process.cwd(), out) +
  (files.length || removed ? "; " + relative(process.cwd(), sealedDir) + "/: " + files.length + " file(s), " + written + " written, " +
    (files.length - written) + " unchanged, " + removed + " removed" : "") + "; everything opened again as the page will: identical");

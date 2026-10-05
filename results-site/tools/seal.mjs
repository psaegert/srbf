// Seal a directory of payload scripts into one encrypted file the page can open with a key.
//
//   SRBF_SEAL_KEY='<key>' node tools/seal.mjs <release> [<source dir>] [<out file>]
//   node tools/seal.mjs <release> --source <dir>=<ENV VAR> [--source <dir>=<ENV VAR> ...] [--out <out file>]
//
// The second form seals several directories, each under the key in its own environment variable, into one file: a
// key opens its own directory's methods and no other. The first source goes where a page reads a single sealed
// payload (RESULTS_V2_SEALED[release]), the others into RESULTS_V2_SEALED_MORE[release]; the page tries a key on
// every one of them. The further envelopes are written FIRST and the first one last, so that a page or a public guard
// that knows only RESULTS_V2_SEALED reads the file exactly as before.
//
// The source files are ordinary payload scripts -- the same `window.NAME = ...` / IIFE files the public release is
// made of. They are concatenated verbatim, never parsed: this tool does not read what is in them, and the page runs
// the result exactly as a <script> tag would. The concatenation is gzipped, then encrypted with AES-256-GCM under a
// key derived by PBKDF2-HMAC-SHA256; salt and IV are fresh per run, so two seals of the same input share no bytes,
// and the GCM tag means a wrong key fails as an authentication error rather than as garbage.
//
// PER-PROBLEM FILES. The scripts under a source's pp/, pred/ and pv/ (the files of the Correlations and Predictions
// views, one per method x problem set x budget x run) and under ranks/, paired/ and hist/ (one per metric) do not go
// into that payload: there are thousands of them, and the page needs a few at a time, so the payload is the overlay
// itself and a reseal changes little in git. Each is sealed on its own into
// <out dir>/sealed/<name>.js (data/<release>/sealed/ for the default out file) and fetched only when a key holder opens
// a view that reads it. Two sources may hold the same path: their names differ, as their keys do. Per source key P:
//   material = PBKDF2-HMAC-SHA256(P, salt "srbf-sealed-files/<release>", 600000 iterations, 512 bits)
//   kEnc = material[0:32] (an AES-256-GCM key), kName = material[32:64] (an HMAC-SHA256 key)
//   path = the file's POSIX path below the source, e.g. "pp/<method>/feynman/16.1.js"
//   name = hex(HMAC(kName, "name\0" + path))[0:32]                  reveals neither the method nor the path
//   gz   = gzip -9 (content), with a fixed header (no time, OS "unknown")
//   iv   = HMAC(kName, "iv\0" + path + "\0" || gz)[0:12]             an IV repeats only for identical plaintext
//   ct   = AES-256-GCM(kEnc, iv, gz, additional data "<release>/<path>")                  bound to its path
// and the file is one line:
//   (window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})["<name>"]={"iv":"<base64>","ct":"<base64>"};
// Nothing here is random, so an unchanged file reseals to the same bytes (no git churn) and only a changed one is
// rewritten. The IV is taken over the bytes that are encrypted, so even another machine's gzip, compressing the same
// content differently, gets another IV: no IV ever carries two plaintexts. A file already there that opens to the same
// content is kept as it is, under its own IV, whatever this machine's gzip would write. The source's payload gets one
// more script at its end, which hands the two subkeys to whoever opens it, so the page never runs the KDF per file:
//   ;if(window.RESULTS_V2_PRIVATE){window.RESULTS_V2_PRIVATE.sealed_files={"enc":"<base64 kEnc>","name":"<base64 kName>"};}
// A file that is in no source any more is removed from sealed/ (only <32 hex digits>.js names are touched; the names
// every source of this run writes are kept).
//
// The key comes from the environment, never from argv: an argument is visible in `ps` and lands in shell history.
// Nothing written reveals the key: the names and IVs of the per-problem files are HMAC outputs under a subkey. Losing
// the key costs one reseal. Before anything is written, every envelope is opened again the way the page opens it, and
// no other source's key may open it.
import { webcrypto as crypto, createHmac } from "node:crypto";
import { gzipSync, gunzipSync } from "node:zlib";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync, unlinkSync, rmdirSync } from "node:fs";
import { join, resolve, dirname, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ITERATIONS = 600000;   // OWASP 2023 for PBKDF2-HMAC-SHA256: ~0.3 s in a browser, the only brake on an offline guess
const FILE_DIRS = ["pp", "pred", "pv", "ranks", "paired", "hist"];   // a source's per-problem (and per-metric) files, sealed one by one
const SEALED_NAME = /^[0-9a-f]{32}\.js$/;
const FILE_LINE = /^\(window\.RESULTS_V2_SEALED_FILES=window\.RESULTS_V2_SEALED_FILES\|\|\{\}\)\["([0-9a-f]{32})"\]=\{"iv":"([A-Za-z0-9+/]+=*)","ct":"([A-Za-z0-9+/]+=*)"\};\n$/;
const HANDOVER = "window.RESULTS_V2_PRIVATE.sealed_files=";
const utf8 = (s) => Buffer.from(s, "utf8");
const b64 = (bytes) => Buffer.from(bytes).toString("base64");
const u8 = (s) => new Uint8Array(Buffer.from(s, "base64"));

function scripts(dir) {
  return readdirSync(dir).flatMap(function (name) {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? scripts(p) : (name.endsWith(".js") ? [p] : []);
  }).sort();
}

async function seal(plain, passphrase) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(passphrase), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: ITERATIONS, hash: "SHA-256" },
    base, { name: "AES-GCM", length: 256 }, false, ["encrypt"]);
  const ct = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, plain));
  return { v: 1, kdf: "PBKDF2-SHA256", iter: ITERATIONS, salt: b64(salt), iv: b64(iv), ct: b64(ct) };
}

// ---- the per-problem files -------------------------------------------------------------------------------------
async function fileKeys(passphrase) {   // {enc, name}: the two 32-byte subkeys of one source key
  const base = await crypto.subtle.importKey("raw", utf8(passphrase), "PBKDF2", false, ["deriveBits"]);
  const bits = new Uint8Array(await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt: utf8("srbf-sealed-files/" + release), iterations: ITERATIONS, hash: "SHA-256" }, base, 512));
  return { enc: bits.slice(0, 32), name: bits.slice(32, 64) };
}
const hmac = (key, ...parts) => parts.reduce((h, p) => h.update(p), createHmac("sha256", key)).digest();
const fileName = (kName, path) => hmac(kName, utf8("name\0" + path)).toString("hex").slice(0, 32);
const fileIv = (kName, path, gz) => hmac(kName, utf8("iv\0" + path + "\0"), gz).subarray(0, 12);   // over the bytes encrypted
function gzipFixed(content) {   // a header without time or OS: equal content, equal bytes, on any machine
  const gz = gzipSync(content, { level: 9 });
  gz.writeUInt32LE(0, 4);   // MTIME: none
  gz[8] = 2;                // XFL: maximum compression
  gz[9] = 255;              // OS: unknown
  return gz;
}
const fileLine = (name, iv, ct) => "(window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})[" + JSON.stringify(name) + "]=" +
  JSON.stringify({ iv: b64(iv), ct: b64(ct) }) + ";\n";
async function encryptFile(kEnc, iv, gz, aad) {
  const key = await crypto.subtle.importKey("raw", kEnc, "AES-GCM", false, ["encrypt"]);
  return new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad }, key, gz));
}
// One file's line opened the way the page opens it: the line's own fields, AES-GCM with the path as additional data,
// then the browser's gzip stream. -> the content, or null when the key or the path does not fit.
async function openFile(line, kEnc, aad) {
  const m = FILE_LINE.exec(line);
  if (!m) { return null; }
  try {
    const key = await crypto.subtle.importKey("raw", kEnc, "AES-GCM", false, ["decrypt"]);
    const gz = await crypto.subtle.decrypt({ name: "AES-GCM", iv: u8(m[2]), additionalData: aad }, key, u8(m[3]));
    if (typeof DecompressionStream === "undefined") { return gunzipSync(Buffer.from(gz)); }
    return Buffer.from(await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  } catch (e) {
    return null;
  }
}

const USAGE = "usage: SRBF_SEAL_KEY='<key>' node tools/seal.mjs <release> [src] [out]\n" +
  "       node tools/seal.mjs <release> --source <dir>=<ENV VAR> [--source <dir>=<ENV VAR> ...] [--out <file>]";
const args = process.argv.slice(2);
const release = args[0];
if (!release || release.startsWith("--")) { console.error(USAGE); process.exit(2); }
// [directory, the environment variable that holds its key]
let sources, outArg;
if (args.includes("--source")) {
  sources = [];
  for (let i = 1; i < args.length; i++) {
    const v = args[i + 1];
    if (args[i] === "--source" && v && v.includes("=")) { sources.push([v.slice(0, v.lastIndexOf("=")), v.slice(v.lastIndexOf("=") + 1)]); i++; }
    else if (args[i] === "--out" && v) { outArg = v; i++; }
    else { console.error(USAGE); process.exit(2); }
  }
} else {
  sources = [[args[1] || join(SITE, "private", release), "SRBF_SEAL_KEY"]];
  outArg = args[2];
}
const passphrases = sources.map(function ([, name]) {
  const pass = process.env[name];
  if (!pass || pass.length < 16) {
    console.error(name + " is missing or shorter than 16 characters.\n" +
      "The sealed file is public, so the key is the only thing protecting it: use a generated one, e.g.\n" +
      "  node -e \"console.log(require('crypto').randomBytes(16).toString('base64url'))\"");
    process.exit(2);
  }
  return pass;
});
if (new Set(passphrases).size !== passphrases.length) {
  console.error("two sources share a key: one key would open both, so seal them as one source"); process.exit(2);
}
const out = resolve(outArg || join(SITE, "data", release, "sealed.js"));
const sealedDir = join(dirname(out), "sealed");

// The file is only worth writing if the page can open it: read the envelope back the way the page does.
// -> the payload it opens to, or null.
async function opens(env, pass) {
  try {
    const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
    const key = await crypto.subtle.deriveKey({ name: "PBKDF2", salt: u8(env.salt), iterations: env.iter, hash: "SHA-256" },
      base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
    return gunzipSync(Buffer.from(await crypto.subtle.decrypt({ name: "AES-GCM", iv: u8(env.iv) }, key, u8(env.ct))));
  } catch (e) {
    return null;
  }
}
const fail = (msg) => { console.error(msg + "; nothing written"); process.exit(1); };
const perFile = (src, f) => FILE_DIRS.includes(relative(src, f).split(sep)[0]);
const listed = sources.map(([dir]) => { const src = resolve(dir); return { src, all: scripts(src) }; });
// every source's subkeys once any source has per-problem files: each source's are also tried on the others' files
const keys = listed.some(({ src, all }) => all.some((f) => perFile(src, f))) ? await Promise.all(passphrases.map(fileKeys)) : null;
const envelopes = [], plains = [], notes = [], files = [], names = new Set();
for (let i = 0; i < sources.length; i++) {
  const { src, all } = listed[i];
  const payload = all.filter((f) => !perFile(src, f)), own = all.filter((f) => perFile(src, f));
  if (!payload.length) { console.error("no payload scripts under " + src); process.exit(1); }
  const parts = payload.map((f) => readFileSync(f, "utf8"));
  if (own.length) {
    parts.push(";if(window.RESULTS_V2_PRIVATE){" + HANDOVER + JSON.stringify({ enc: b64(keys[i].enc), name: b64(keys[i].name) }) + ";}");
  }
  const plain = Buffer.from(parts.join("\n;\n"), "utf8");
  const gz = gzipSync(plain, { level: 9 });
  envelopes.push(await seal(gz, passphrases[i]));
  plains.push(plain);
  for (const f of own) {
    const path = relative(src, f).split(sep).join("/");
    const content = readFileSync(f), name = fileName(keys[i].name, path), aad = utf8(release + "/" + path);
    if (names.has(name)) { fail("two per-problem files share the name " + name); }
    names.add(name);
    const at = join(sealedDir, name + ".js");
    const there = existsSync(at) ? readFileSync(at, "utf8") : null;
    const old = there ? await openFile(there, keys[i].enc, aad) : null;
    let line = there;   // this content is published already: its bytes stay, under their own IV
    if (!old || !old.equals(content)) {
      const gz = gzipFixed(content), iv = fileIv(keys[i].name, path, gz);
      line = fileLine(name, iv, await encryptFile(keys[i].enc, iv, gz, aad));
    }
    files.push({ source: i, path, name, aad, content, line, at, changed: line !== there });
  }
  notes.push(payload.length + " file(s) from " + relative(SITE, src) + " (" + sources[i][1] + "): " +
    (plain.length / 1048576).toFixed(2) + " MB -> " + (gz.length / 1048576).toFixed(2) + " MB gzip" +
    (own.length ? ", and " + own.length + " per-problem file(s) one by one" : ""));
}

// ---- before anything is written: open everything again as the page will --------------------------------------
for (let i = 0; i < sources.length; i++) {
  const back = await opens(envelopes[i], passphrases[i]);
  if (!back || !back.equals(plains[i])) { fail("the sealed payload of " + listed[i].src + " does not open to what was sealed"); }
  for (let j = 0; j < i; j++) {   // and no other source's key opens it
    if (await opens(envelopes[i], passphrases[j])) { fail("another key opens " + listed[i].src); }
  }
}
// a key holder gets the subkeys from the payload that key opens, so the files are opened with the ones handed over there
const handed = plains.map(function (plain) {
  const text = plain.toString("utf8"), at = text.lastIndexOf(HANDOVER);
  if (at < 0) { return null; }
  const k = JSON.parse(text.slice(at + HANDOVER.length, text.lastIndexOf(";}")));
  return { enc: u8(k.enc), name: u8(k.name) };
});
for (const f of files) {
  const k = handed[f.source], m = FILE_LINE.exec(f.line);
  if (!k) { fail("the payload of " + listed[f.source].src + " does not hand over its file keys"); }
  if (!m || m[1] !== f.name || fileName(k.name, f.path) !== f.name) { fail("the sealed " + f.path + " is not named as the page will look for it"); }
  const back = await openFile(f.line, k.enc, f.aad);
  if (!back || !back.equals(f.content)) { fail("the sealed " + f.path + " of " + listed[f.source].src + " does not open to what was sealed"); }
  for (let j = 0; j < sources.length; j++) {
    if (j !== f.source && await openFile(f.line, keys[j].enc, f.aad)) { fail("another key opens " + f.path + " of " + listed[f.source].src); }
  }
}

// ---- write: the per-problem files (an unchanged one is not touched), the payloads, then remove the withdrawn ----
let written = 0, removed = 0;
if (files.length) { mkdirSync(sealedDir, { recursive: true }); }
for (const f of files) {
  if (f.changed) { writeFileSync(f.at, f.line); written++; }
}
const rel = JSON.stringify(release);
let text = "";
if (envelopes.length > 1) {
  text += "window.RESULTS_V2_SEALED_MORE=window.RESULTS_V2_SEALED_MORE||{};window.RESULTS_V2_SEALED_MORE[" + rel + "]=" +
    JSON.stringify(envelopes.slice(1)) + ";\n";
}
text += "window.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED[" + rel + "]=" + JSON.stringify(envelopes[0]) + ";\n";
writeFileSync(out, text);
if (existsSync(sealedDir)) {
  for (const n of readdirSync(sealedDir)) {
    if (SEALED_NAME.test(n) && !names.has(n.slice(0, 32))) { unlinkSync(join(sealedDir, n)); removed++; }
  }
  if (!readdirSync(sealedDir).length) { rmdirSync(sealedDir); }
}
console.log(notes.join("; ") + " -> " + (statSync(out).size / 1048576).toFixed(2) + " MB at " + relative(SITE, out) +
  (files.length || removed ? "; " + relative(SITE, sealedDir) + "/: " + files.length + " file(s), " + written + " written, " +
    (files.length - written) + " unchanged, " + removed + " removed" : "") +
  "; each opened again with its own key: identical");

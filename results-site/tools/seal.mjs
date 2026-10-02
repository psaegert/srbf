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
// The key comes from the environment, never from argv: an argument is visible in `ps` and lands in shell history.
// Nothing derived from the key is written. Losing it costs one reseal.
import { webcrypto as crypto } from "node:crypto";
import { gzipSync, gunzipSync } from "node:zlib";
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ITERATIONS = 600000;   // OWASP 2023 for PBKDF2-HMAC-SHA256: ~0.3 s in a browser, the only brake on an offline guess

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
  const b64 = (u8) => Buffer.from(u8).toString("base64");
  return { v: 1, kdf: "PBKDF2-SHA256", iter: ITERATIONS, salt: b64(salt), iv: b64(iv), ct: b64(ct) };
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

// The file is only worth writing if the page can open it: read the envelope back the way the page does.
async function opens(env, pass, expected) {
  const u8 = (b64) => new Uint8Array(Buffer.from(b64, "base64"));
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey({ name: "PBKDF2", salt: u8(env.salt), iterations: env.iter, hash: "SHA-256" },
    base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  const back = Buffer.from(await crypto.subtle.decrypt({ name: "AES-GCM", iv: u8(env.iv) }, key, u8(env.ct)));
  return gunzipSync(back).equals(expected);
}
const envelopes = [], notes = [];
for (let i = 0; i < sources.length; i++) {
  const src = resolve(sources[i][0]);
  const files = scripts(src);
  if (!files.length) { console.error("no payload scripts under " + src); process.exit(1); }
  const plain = Buffer.from(files.map((f) => readFileSync(f, "utf8")).join("\n;\n"), "utf8");
  const gz = gzipSync(plain, { level: 9 });
  const envelope = await seal(gz, passphrases[i]);
  if (!(await opens(envelope, passphrases[i], plain))) { console.error("the sealed payload of " + src + " does not open to what was sealed; nothing written"); process.exit(1); }
  for (let j = 0; j < i; j++) {   // and no other source's key opens it
    if (await opens(envelope, passphrases[j], plain).catch(() => false)) { console.error("another key opens " + src + "; nothing written"); process.exit(1); }
  }
  envelopes.push(envelope);
  notes.push(files.length + " file(s) from " + relative(SITE, src) + " (" + sources[i][1] + "): " +
    (plain.length / 1048576).toFixed(2) + " MB -> " + (gz.length / 1048576).toFixed(2) + " MB gzip");
}
const rel = JSON.stringify(release);
let text = "";
if (envelopes.length > 1) {
  text += "window.RESULTS_V2_SEALED_MORE=window.RESULTS_V2_SEALED_MORE||{};window.RESULTS_V2_SEALED_MORE[" + rel + "]=" +
    JSON.stringify(envelopes.slice(1)) + ";\n";
}
text += "window.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED[" + rel + "]=" + JSON.stringify(envelopes[0]) + ";\n";
writeFileSync(out, text);
console.log(notes.join("; ") + " -> " + (statSync(out).size / 1048576).toFixed(2) + " MB at " + relative(SITE, out) +
  "; each opened again with its own key: identical");

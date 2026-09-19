// Seal a directory of payload scripts into one encrypted file the page can open with a key.
//
//   SRBF_SEAL_KEY='<key>' node tools/seal.mjs <release> [<source dir>] [<out file>]
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

const [release, srcArg, outArg] = process.argv.slice(2);
if (!release) {
  console.error("usage: SRBF_SEAL_KEY='<key>' node tools/seal.mjs <release> [src] [out]");
  process.exit(2);
}
const passphrase = process.env.SRBF_SEAL_KEY;
if (!passphrase || passphrase.length < 16) {
  console.error("SRBF_SEAL_KEY is missing or shorter than 16 characters.\n" +
    "The sealed file is public, so the key is the only thing protecting it: use a generated one, e.g.\n" +
    "  node -e \"console.log(require('crypto').randomBytes(16).toString('base64url'))\"");
  process.exit(2);
}
const src = resolve(srcArg || join(SITE, "private", release));
const out = resolve(outArg || join(SITE, "data", release, "sealed.js"));

const files = scripts(src);
if (!files.length) { console.error("no payload scripts under " + src); process.exit(1); }
const plain = Buffer.from(files.map((f) => readFileSync(f, "utf8")).join("\n;\n"), "utf8");
const gz = gzipSync(plain, { level: 9 });
const envelope = await seal(gz, passphrase);

// The file is only worth writing if the page can open it: read the envelope back the way the page does.
async function opens(env, pass, expected) {
  const u8 = (b64) => new Uint8Array(Buffer.from(b64, "base64"));
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pass), "PBKDF2", false, ["deriveKey"]);
  const key = await crypto.subtle.deriveKey({ name: "PBKDF2", salt: u8(env.salt), iterations: env.iter, hash: "SHA-256" },
    base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  const back = Buffer.from(await crypto.subtle.decrypt({ name: "AES-GCM", iv: u8(env.iv) }, key, u8(env.ct)));
  return gunzipSync(back).equals(expected);
}
if (!(await opens(envelope, passphrase, plain))) { console.error("the sealed payload does not open to what was sealed; nothing written"); process.exit(1); }
writeFileSync(out, "window.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED[" +
  JSON.stringify(release) + "]=" + JSON.stringify(envelope) + ";\n");
console.log(files.length + " file(s) from " + relative(SITE, src) + ": " +
  (plain.length / 1048576).toFixed(2) + " MB -> " + (gz.length / 1048576).toFixed(2) + " MB gzip -> " +
  (statSync(out).size / 1048576).toFixed(2) + " MB at " + relative(SITE, out) + "; opened again with the key: identical");

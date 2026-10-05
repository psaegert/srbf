// pairstats.js: the explorer's paired contrasts and pairwise rank outcomes, computed in the browser at any position.
//
// The exporter (scripts/site_export_v2.py) ships the cells of paired_cell_at and rank_pair_cell_at at its slots (the
// release's budgets and time limits: ranks/<key>.js, paired/<key>.js), and what those two functions read of every
// problem (pv/<method>/<rung>/frame.js and <key>.js, write_pair_values; the format is described there). From the
// latter, pairedCell and rankPairCell below return what the exporter returns -- the same lists, keys and counts, and the
// same floats bit for bit (tests/test_pairstats_parity.py) -- for a method at a budget it ran, and BETWEEN two budgets.
//
// A method at a position is at(lo, hi, w): lo and hi the decoded files (file(), or decode) of two budgets it ran on one
// problem set, w in [0, 1] where the position lies between them (the caller computes it as the page's bracket() does:
// linearly in the logarithm of the budget or of the time). At w = 0 the method IS lo and at w = 1 it IS hi: nothing is
// interpolated, and the cells are the exporter's at that budget. Strictly between the two:
//   - only the problems the method has at BOTH budgets take part;
//   - a problem's value (value_matrix) and share (rate_matrix) are (1 - w) * x_lo + w * x_hi, NaN where either is NaN;
//   - the superiority of A over B on a problem is the bilinear mixture over A's two budgets and B's two:
//         sup = sum over a, b of wA_a * wB_b * sup(A_a, B_b), divided by the sum of wA_a * wB_b over the same terms,
//     with wX_lo = 1 - wX and wX_hi = wX, and sup(A_a, B_b) the run-pair rule of superiority_of on the two files (the
//     share of run pairs A wins minus the share it loses, over the pairs of runs that take part in the reading). A
//     term without such a run pair is left out of both sums, so the weights are renormalised over the terms that
//     are defined; the problem's superiority is defined where at least one term is. paired_cell counts a problem in
//     its superiority entries only there; rank_pair_cell adds 0 for an undefined one, as superiority_of does.
//   A method at a budget it ran is one term of weight 1, so each rule reduces to the exporter's exactly.
// The sums are _sums's: numpy's pairwise summation (what np.sum does on a contiguous float64 array), then Python's
// round(x, 6), which rounds the exact binary value half to even.
(function (root) {
  "use strict";
  var ANSWERED = "@answered";
  // the exporter's PAIRED_KEYS, RATE_KEYS and WORST (the parity test holds them to the Python lists)
  var PAIRED_KEYS = ["numeric_recovery_val", "symbolic_recovery", "success", "log10_fvu_val", "mdl_ratio", "expr_length_ratio", "f1_score"];
  var RATE_KEYS = ["numeric_recovery_val", "numeric_recovery_fit", "numeric_recovery_relative_val", "numeric_recovery_relative_fit", "success",
    "symbolic_recovery", "symbolic_recovery_mask_fittable", "symbolic_recovery_mask_none", "skeleton_match_raw"];
  var WORST = ["f1_score", "precision_score", "recall_score", "f1_score_unique_variables", "precision_unique_variables", "recall_unique_variables"];

  // ---- reading a pv file ------------------------------------------------------------------------------------------
  function bytesOf(b64) {
    if (typeof atob === "function") {
      var s = atob(b64), u = new Uint8Array(s.length);
      for (var i = 0; i < s.length; i++) { u[i] = s.charCodeAt(i); }
      return u;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }
  function doubles(b64) {   // little-endian float64, whatever the machine's byte order
    var b = bytesOf(b64);
    if (b.length % 8) { throw new Error("pv: " + b.length + " bytes are no whole number of float64 values"); }
    var dv = new DataView(b.buffer, b.byteOffset, b.length), out = new Float64Array(b.length / 8);
    for (var i = 0; i < out.length; i++) { out[i] = dv.getFloat64(8 * i, true); }
    return out;
  }
  // a column (write_pair_values): {"u": U} one value, {"u": U, "i": I} distinct values and indices, {"f": F} the values
  function column(c, len) {
    var out, i;
    if (c.f !== undefined) {
      out = doubles(c.f);
      if (out.length !== len) { throw new Error("pv: a column of " + out.length + " values where " + len + " belong"); }
      return out;
    }
    var u = doubles(c.u);
    out = new Float64Array(len);
    if (c.i === undefined) {
      if (u.length !== 1) { throw new Error("pv: a column of one value holds " + u.length); }
      out.fill(u[0]);
      return out;
    }
    var ix = bytesOf(c.i), wide = u.length > 256;
    if (ix.length !== len * (wide ? 2 : 1)) { throw new Error("pv: " + ix.length + " index bytes for " + len + " values"); }
    for (i = 0; i < len; i++) { out[i] = u[wide ? ix[2 * i] | (ix[2 * i + 1] << 8) : ix[i]]; }
    return out;
  }
  // A file's problems (ids ascending, as _problem_runs orders them), their runs (problem i's runs are off[i]..off[i+1]-1),
  // and its columns, decoded when first read. raw holds n, ids, k and ok (a frame's entry), and either the columns
  // themselves (s, v, r: pv_cell's object) or none, when get(part, key) finds a column's spec (file() below).
  function decode(raw, get) {
    var n = raw.n, gaps = column(raw.ids, n), k = column(raw.k, n), ids = new Float64Array(n), off = new Int32Array(n + 1), id = -1;
    for (var i = 0; i < n; i++) { id += gaps[i]; ids[i] = id; off[i + 1] = off[i] + k[i]; }
    var f = { n: n, runs: off[n], ids: ids, off: off, raw: raw, cols: {},
      get: get || function (part, key) {
        var spec = raw[part] && raw[part][key];
        if (!spec) { throw new Error("pv: the file has no " + PARTS[part] + " of " + key); }
        return spec;
      } };
    f.ok = column(raw.ok, f.runs);
    return f;
  }
  var PARTS = { s: "scores", v: "values", r: "shares" };
  function col(f, part, key) {   // part: "s" scores per run, "v" values per problem, "r" shares per problem
    var id = part + ":" + key, spec = f.get(part, key), c = f.cols[id];
    if (!c || c.spec !== spec) { c = f.cols[id] = { spec: spec, data: column(spec, part === "s" ? f.runs : f.n) }; }
    return c.data;
  }

  // ---- a method at a position -------------------------------------------------------------------------------------
  function merge(a, b) {   // the ids both ascending lists hold, and where: _align
    var ids = [], ia = [], ib = [], i = 0, j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] === b[j]) { ids.push(a[i]); ia.push(i); ib.push(j); i++; j++; } else if (a[i] < b[j]) { i++; } else { j++; }
    }
    return { n: ids.length, ids: ids, ia: ia, ib: ib };
  }
  function exact(f) {
    var ix = new Int32Array(f.n);
    for (var i = 0; i < f.n; i++) { ix[i] = i; }
    return { ids: f.ids, terms: [{ f: f, w: 1, ix: ix }], memo: {} };
  }
  // The method at w between the budgets of the files lo and hi (see the top of this file); at(f) is the method at f's.
  function at(lo, hi, w) {
    if (!lo) { throw new Error("pv: no file to read the method from"); }
    if (!hi || hi === lo || w === 0) { return exact(lo); }
    if (w === 1) { return exact(hi); }
    if (!(w > 0 && w < 1)) { throw new Error("pv: a weight outside [0, 1]: " + w); }
    var m = merge(lo.ids, hi.ids);
    return { ids: m.ids, terms: [{ f: lo, w: 1 - w, ix: m.ia }, { f: hi, w: w, ix: m.ib }], memo: {} };
  }
  // per problem of the position: a value (part "v") or a share (part "r"), interpolated between two budgets
  function perProblem(P, part, key) {
    var id = part + ":" + key;
    if (P.memo[id]) { return P.memo[id]; }
    var t = P.terms, out;
    if (t.length === 1) {
      out = col(t[0].f, part, key);
    } else {
      var a = col(t[0].f, part, key), b = col(t[1].f, part, key), ia = t[0].ix, ib = t[1].ix;
      out = new Float64Array(P.ids.length);
      for (var i = 0; i < out.length; i++) { out[i] = t[0].w * a[ia[i]] + t[1].w * b[ib[i]]; }   // (1 - w) * lo + w * hi
    }
    return (P.memo[id] = out);
  }
  // Per problem both positions have (al = merge of their ids): the superiority of A over B on `key`, and whether it is
  // defined; 0 where it is not (superiority_of). `answered`: only the runs that succeeded take part.
  function superiorities(A, B, al, key, answered) {
    var n = al.n, val = new Float64Array(n), ok = new Uint8Array(n);
    var SA = A.terms.map(function (t) { return col(t.f, "s", key); }), SB = B.terms.map(function (t) { return col(t.f, "s", key); });
    for (var i = 0; i < n; i++) {
      var acc = 0, wsum = 0;
      for (var a = 0; a < A.terms.length; a++) {
        var ta = A.terms[a], fa = ta.f, pa = ta.ix[al.ia[i]], sa = SA[a];
        for (var b = 0; b < B.terms.length; b++) {
          var tb = B.terms[b], fb = tb.f, pb = tb.ix[al.ib[i]], sb = SB[b], num = 0, den = 0;
          for (var r = fa.off[pa]; r < fa.off[pa + 1]; r++) {   // the run-pair rule: wins minus losses over the run pairs
            if (answered && !fa.ok[r]) { continue; }
            var x = sa[r];
            for (var q = fb.off[pb]; q < fb.off[pb + 1]; q++) {
              if (answered && !fb.ok[q]) { continue; }
              num += (x > sb[q]) - (sb[q] > x);
              den += 1;
            }
          }
          if (den > 0) { var ww = ta.w * tb.w; acc += ww * (num / den); wsum += ww; }
        }
      }
      if (wsum > 0) { val[i] = acc / wsum; ok[i] = 1; }
    }
    return { val: val, ok: ok };
  }

  // ---- _sums ------------------------------------------------------------------------------------------------------
  // numpy's pairwise summation of n values from a[lo] (pairwise_sum in numpy's loops_utils.h): eight accumulators over
  // blocks of at most 128 values, halves (cut at a multiple of 8) above that
  function pairwise(a, lo, n) {
    var i, res;
    if (n < 8) {
      res = 0;
      for (i = 0; i < n; i++) { res += a[lo + i]; }
      return res;
    }
    if (n <= 128) {
      var r0 = a[lo], r1 = a[lo + 1], r2 = a[lo + 2], r3 = a[lo + 3], r4 = a[lo + 4], r5 = a[lo + 5], r6 = a[lo + 6], r7 = a[lo + 7];
      for (i = 8; i < n - (n % 8); i += 8) {
        r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3];
        r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7];
      }
      res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
      for (; i < n; i++) { res += a[lo + i]; }
      return res;
    }
    var n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
  }
  // Python's round(x, 6): the 6-decimal number nearest to x's exact binary value, half to even. toFixed rounds the exact
  // value too, but a tie away from zero; a tie is exactly an x with 128 x an odd integer (2e6 x = 15625 * 128 x).
  function round6(x) {
    if (!isFinite(x)) { return x; }
    var s = Math.abs(x).toFixed(6), t = x * 128;
    if (Number.isInteger(t) && Math.abs(t % 2) === 1) {
      var last = s.charCodeAt(s.length - 1) - 48;
      if (last % 2 === 1) { s = s.slice(0, -1) + String(last - 1); }   // toFixed went away from zero onto an odd digit
    }
    return parseFloat((x < 0 ? "-" : "") + s);
  }
  function sums(xs) {
    if (!xs.length) { return [0, 0]; }
    var sq = new Float64Array(xs.length);
    for (var i = 0; i < xs.length; i++) { sq[i] = xs[i] * xs[i]; }
    return [round6(pairwise(xs, 0, xs.length)), round6(pairwise(sq, 0, sq.length))];
  }
  function signs(xs) {
    var pos = 0, neg = 0;
    for (var i = 0; i < xs.length; i++) { if (xs[i] > 0) { pos++; } else if (xs[i] < 0) { neg++; } }
    return [pos, neg];
  }
  function setOf(list) { var o = {}; list.forEach(function (k) { o[k] = true; }); return o; }

  // ---- the two cells ----------------------------------------------------------------------------------------------
  // paired_cell(A, B): {n, m: {key: [...]}} or null without a problem in common. opts (all optional): keys (the paired
  // keys, in order), rates, worst; the exporter's lists by default.
  function pairedCell(A, B, opts) {
    var o = opts || {}, keys = o.keys || PAIRED_KEYS, rates = setOf(o.rates || RATE_KEYS), worst = setOf(o.worst || WORST);
    var al = merge(A.ids, B.ids), i;
    if (!al.n) { return null; }
    var m = {};
    keys.forEach(function (k) {
      if (rates[k]) {
        var ra = perProblem(A, "r", k), rb = perProblem(B, "r", k), ds = new Float64Array(al.n);
        for (i = 0; i < al.n; i++) { ds[i] = ra[al.ia[i]] - rb[al.ib[i]]; }
        m[k] = [al.n].concat(sums(ds), signs(ds));
        return;
      }
      [[k, false]].concat(worst[k] ? [[k + ANSWERED, true]] : []).forEach(function (reading) {
        var name = reading[0], answered = reading[1], va = perProblem(A, "v", name), vb = perProblem(B, "v", name), ds = [], ss = [];
        for (i = 0; i < al.n; i++) {
          var x = va[al.ia[i]], y = vb[al.ib[i]];
          if (isFinite(x) && isFinite(y)) { ds.push(x - y); }
        }
        var sup = superiorities(A, B, al, k, answered);
        for (i = 0; i < al.n; i++) { if (sup.ok[i]) { ss.push(sup.val[i]); } }
        m[name] = [ds.length].concat(sums(ds), [ss.length], sums(ss), signs(ss));
      });
    });
    return { n: al.n, m: m };
  }
  // rank_pair_cell(A, B, keys): [problems in common, then (sum, sum of squares) of the superiority per key], or null.
  function rankPairCell(A, B, keys) {
    if (!keys) { throw new Error("pv: rankPairCell needs the rank keys (the release's rank_keys)"); }
    var al = merge(A.ids, B.ids);
    if (!al.n) { return null; }
    var out = [al.n];
    keys.forEach(function (k) { Array.prototype.push.apply(out, sums(superiorities(A, B, al, k, false).val)); });
    return out;
  }

  // ---- positions ------------------------------------------------------------------------------------------------
  // The logarithm of a positive finite x, written with + - * / alone in the same order as the exporter's logd: the
  // same double in both (Math.log and libm's log differ in the last bit now and then, and a w that differs in its last
  // bit can tip a sum rounded to six decimals). x = m 2^e with m in [sqrt(1/2), sqrt(2)): log m = 2 atanh((m - 1) /
  // (m + 1)) by its series (13 terms), plus e log 2.
  var LN2 = 0.6931471805599453, SQRT_HALF = 0.7071067811865476, LOG_TERMS = 13, FREXP = new DataView(new ArrayBuffer(8));
  function frexp(x) {   // [m, e] with x = m 2^e, m in [0.5, 1), exactly (x positive and finite)
    var shift = 0;
    FREXP.setFloat64(0, x);
    if (((FREXP.getUint32(0) >>> 20) & 0x7ff) === 0) { FREXP.setFloat64(0, x * 18014398509481984); shift = 54; }   // subnormal: times 2^54
    var hi = FREXP.getUint32(0), e = ((hi >>> 20) & 0x7ff) - 1022 - shift;
    FREXP.setUint32(0, (hi & 0x800fffff) | (1022 << 20));
    return [FREXP.getFloat64(0), e];
  }
  function logd(x) {
    var me = frexp(x), m = me[0], e = me[1];
    if (m < SQRT_HALF) { m = m * 2; e = e - 1; }
    var z = (m - 1) / (m + 1), z2 = z * z, s = 1 / (2 * LOG_TERMS - 1);
    for (var k = LOG_TERMS - 2; k >= 0; k--) { s = 1 / (2 * k + 1) + z2 * s; }
    return e * LN2 + (2 * z) * s;
  }
  // Where x lies among points [[position, budget], ...] sorted by position (a method's complete budgets on a problem
  // set, by budget or by reference time, as the page's points() lists them): {r1, r2, w}, at one of them (within 1e-9
  // relative: w = 0) or between two neighbours, w = log(x / x1) / log(x2 / x1); null outside them. The exporter's
  // bracket_in is the same, and D.slots.at holds its results: the page reads every position through this function.
  function bracketIn(ps, x) {
    var i;
    if (!(x > 0)) { return null; }
    for (i = 0; i < ps.length; i++) { if (Math.abs(ps[i][0] - x) <= 1e-9 * x) { return { r1: ps[i][1], r2: ps[i][1], w: 0 }; } }
    for (i = 0; i + 1 < ps.length; i++) { if (ps[i][0] < x && x < ps[i + 1][0]) { return { r1: ps[i][1], r2: ps[i + 1][1], w: logd(x / ps[i][0]) / logd(ps[i + 1][0] / ps[i][0]) }; } }
    return null;
  }

  // ---- loading ----------------------------------------------------------------------------------------------------
  // pv/ per method x budget (D.pv = {method: {rung: [catalogs]}} lists them): one frame, and one file per key, each
  // holding every problem set the method has finished at that budget. The page loads them with its ensure() from
  // these paths, relative to the release's base; file() then reads a problem set out of what has loaded.
  function baseKey(key) { return key.slice(-ANSWERED.length) === ANSWERED ? key.slice(0, -ANSWERED.length) : key; }
  function framePath(method, rung) { return "pv/" + method + "/" + rung + "/frame.js"; }
  function keyPath(method, rung, key) { return "pv/" + method + "/" + rung + "/" + baseKey(key) + ".js"; }
  function store(rel) { var G = typeof window !== "undefined" ? window : root, R = G && G.RESULTS_V2_PV; return (R && R[rel]) || null; }
  var FILES = {};
  // The decoded file of a method on one problem set at one budget, for at(): built from its frame and the key files
  // loaded so far (a key file that loads later is read when its column is first needed); null while the frame is not
  // loaded or does not hold the problem set. Reading a column whose key file has not loaded throws, naming the file.
  function file(rel, method, rung, catalog) {
    var S = store(rel), frame = S && S[method + "|" + rung + "|frame"], raw = frame && frame[catalog];
    if (!raw) { return null; }
    var id = rel + "|" + method + "|" + rung + "|" + catalog, hit = FILES[id];
    if (!hit || hit.raw !== raw) {
      hit = FILES[id] = decode(raw, function (part, key) {
        var T = store(rel), kf = T && T[method + "|" + rung + "|" + baseKey(key)], c = kf && kf[catalog], spec = c && c[part] && c[part][key];
        if (!kf) { throw new Error("pv: " + keyPath(method, rung, key) + " is not loaded (the " + PARTS[part] + " of " + key + " on " + catalog + ")"); }
        if (!spec) { throw new Error("pv: " + keyPath(method, rung, key) + " holds no " + PARTS[part] + " of " + key + " on " + catalog); }
        return spec;
      });
    }
    return hit;
  }

  var API = { ANSWERED: ANSWERED, PAIRED_KEYS: PAIRED_KEYS, RATE_KEYS: RATE_KEYS, WORST: WORST,
    framePath: framePath, keyPath: keyPath, file: file, decode: decode, column: column, at: at, bracketIn: bracketIn, logd: logd,
    pairedCell: pairedCell, rankPairCell: rankPairCell, sums: sums, round6: round6, pairwise: pairwise };
  if (typeof module === "object" && module.exports) { module.exports = API; } else { root.PAIRSTATS = API; }
})(typeof globalThis !== "undefined" ? globalThis : this);

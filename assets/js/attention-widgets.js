/* Interactive figures of the note /posts/attention-backward/.
   Each <div class="aw" data-widget="..."> in the post is filled in by one function below.
   All numbers are worked out here, in doubles, from the note's example: tokens 0 to 2 are the
   example of Section 1.4, tokens 3 to 7 are added so the mask's staircase shows. */
(function () {
  "use strict";

  var D = 4;          // columns per head in the example (64 in GPT-2)
  var SCALE = 0.5;    // 1 / sqrt(d)
  var NUM_HEADS = 12, HEAD_SIZE = 64, C = 768;

  var Q_ALL = [[2, 0, 0, 0], [0, 2, 0, 0], [2, 2, 0, 0], [0, 0, 2, 0], [2, 0, 0, 2], [0, 2, 2, 0], [2, 2, 2, 2], [0, 0, 0, 2]];
  var K_ALL = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]];
  var V_ALL = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [0.5, 0, 0.5, 0], [1, 0, 0, 1]];

  var CSS = [
    ".aw{--aw-fg:#212529;--aw-muted:#868e96;--aw-line:#dee2e6;--aw-soft:#f8f9fa;--aw-acc:#1971c2;--aw-acc-bg:#a5d8ff;",
    "--aw-acc-bg2:#e7f5ff;--aw-neg:#e8590c;--aw-mask:#e9ecef;--aw-cell:37px;",
    "border:1px solid var(--aw-line);border-radius:8px;padding:14px 16px;margin:1.2em 0 0.4em;color:var(--aw-fg);",
    "font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.45;overflow-x:auto}",
    "html[data-theme='dark'] .aw{--aw-fg:#e9ecef;--aw-muted:#adb5bd;--aw-line:#495057;--aw-soft:#2b3035;--aw-acc:#4dabf7;",
    "--aw-acc-bg:#1c4f80;--aw-acc-bg2:#1a2f45;--aw-neg:#ff922b;--aw-mask:#343a40}",
    ".aw-title{font-weight:600;margin:0 0 2px}",
    ".aw-sub{color:var(--aw-muted);font-size:13px;margin:0 0 10px}",
    ".aw-ctl{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin:0 0 12px;font-size:13px}",
    ".aw-ctl label{display:flex;align-items:center;gap:8px;white-space:nowrap}",
    ".aw-ctl input[type=range]{width:130px}",
    ".aw-ctl b{display:inline-block;min-width:22px;font-variant-numeric:tabular-nums}",
    ".aw-seg{display:inline-flex;border:1px solid var(--aw-line);border-radius:6px;overflow:hidden}",
    ".aw-seg button{font:inherit;font-size:12px;padding:3px 10px;border:0;background:transparent;color:var(--aw-fg);cursor:pointer}",
    ".aw-seg button.on{background:var(--aw-acc);color:#fff}",
    ".aw-row{display:flex;flex-wrap:wrap;gap:14px 10px;align-items:flex-start}",
    ".aw-op{align-self:center;font-size:20px;color:var(--aw-muted);padding-top:30px}",
    ".aw-cap{font-size:12px;color:var(--aw-muted);margin:0 0 4px 20px;white-space:nowrap}",
    ".aw-cap b{color:var(--aw-fg)}",
    ".aw-grid{display:inline-grid;gap:2px}",
    ".aw-h{font-size:10px;color:var(--aw-muted);text-align:center;line-height:22px}",
    ".aw-c{height:22px;font:10.5px/20px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;text-align:center;",
    "border:1px solid var(--aw-line);background:var(--aw-soft);border-radius:3px;box-sizing:border-box}",
    ".aw-c.m{background:repeating-linear-gradient(45deg,var(--aw-mask),var(--aw-mask) 3px,transparent 3px,transparent 6px);color:var(--aw-muted)}",
    ".aw-c.ml{border-color:var(--aw-muted)}",
    ".aw-c.l{background:var(--aw-acc-bg2);border-color:var(--aw-acc-bg)}",
    ".aw-c.u{background:var(--aw-acc-bg);border-color:var(--aw-acc);font-weight:600}",
    ".aw-c.s{background:var(--aw-acc);border-color:var(--aw-acc);color:#fff;font-weight:600}",
    ".aw-out .aw-c{cursor:pointer}",
    ".aw-eq{margin:12px 0 0;font-size:13px}",
    ".aw-eq p{margin:0 0 6px}",
    ".aw-eq .n{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}",
    ".aw-note{color:var(--aw-muted);font-size:12.5px}",
    ".aw-tbl{border-collapse:collapse;font-size:12.5px;margin:0;display:table;width:auto}",
    ".aw-tbl th,.aw-tbl td{padding:3px 10px;border-bottom:1px solid var(--aw-line);text-align:right;white-space:nowrap}",
    ".aw-tbl th{color:var(--aw-muted);font-weight:500}",
    ".aw-tbl td.n{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}",
    ".aw-tbl tr.m td{color:var(--aw-muted);background:repeating-linear-gradient(45deg,var(--aw-mask),var(--aw-mask) 3px,transparent 3px,transparent 6px)}",
    ".aw-bar{position:relative;width:140px;height:12px;background:var(--aw-soft);border-radius:2px}",
    ".aw-bar i{position:absolute;top:0;height:12px;background:var(--aw-acc);border-radius:2px}",
    ".aw-bar i.neg{background:var(--aw-neg)}",
    ".aw-bar.c::after{content:'';position:absolute;left:50%;top:-2px;height:16px;border-left:1px solid var(--aw-muted)}",
    ".aw-segs{display:flex;gap:2px;margin:2px 0 8px}",
    ".aw-segs span{width:26px;height:20px;font-size:10px;line-height:18px;text-align:center;border:1px solid var(--aw-line);",
    "background:var(--aw-soft);border-radius:3px;color:var(--aw-muted);box-sizing:border-box}",
    ".aw-segs span.u{background:var(--aw-acc);border-color:var(--aw-acc);color:#fff;font-weight:600}",
    ".aw-lbl{font-size:12px;color:var(--aw-muted)}"
  ].join("");

  // ── the model: forward and backward of one head, the note's formulas ──────────────────────
  function zeros(rows, cols) {
    var m = [];
    for (var i = 0; i < rows; i++) { m.push([]); for (var j = 0; j < cols; j++) m[i].push(0); }
    return m;
  }

  function model(T) {
    var q = Q_ALL.slice(0, T), k = K_ALL.slice(0, T), v = V_ALL.slice(0, T);
    var s = zeros(T, T), a = zeros(T, T), y = zeros(T, D), dy = zeros(T, D), da = zeros(T, T), ds = zeros(T, T);
    var dq = zeros(T, D), dk = zeros(T, D), dv = zeros(T, D), m = [], S = [], rowDot = [];
    var t, j, c;
    // F2: every score, the masked ones too, as cuBLAS works them out
    for (t = 0; t < T; t++) for (j = 0; j < T; j++) for (c = 0; c < D; c++) s[t][j] += SCALE * q[t][c] * k[j][c];
    // F3, F4: softmax over j <= t, 0 after the token
    for (t = 0; t < T; t++) {
      m[t] = -Infinity;
      for (j = 0; j <= t; j++) m[t] = Math.max(m[t], s[t][j]);
      S[t] = 0;
      for (j = 0; j <= t; j++) S[t] += Math.exp(s[t][j] - m[t]);
      for (j = 0; j <= t; j++) a[t][j] = Math.exp(s[t][j] - m[t]) / S[t];
    }
    // F5: y = A V
    for (t = 0; t < T; t++) for (c = 0; c < D; c++) for (j = 0; j < T; j++) y[t][c] += a[t][j] * v[j][c];
    // the loss wants column 0 of every output up
    for (t = 0; t < T; t++) dy[t][0] = 1;
    // B2: dA = dY V^T, every pair, as cuBLAS works it out
    for (t = 0; t < T; t++) for (j = 0; j < T; j++) for (c = 0; c < D; c++) da[t][j] += dy[t][c] * v[j][c];
    // C1: ds = a (da - row_dot)
    for (t = 0; t < T; t++) {
      rowDot[t] = 0;
      for (j = 0; j <= t; j++) rowDot[t] += a[t][j] * da[t][j];
      for (j = 0; j <= t; j++) ds[t][j] = a[t][j] * (da[t][j] - rowDot[t]);
    }
    // B3, B7, B8
    for (t = 0; t < T; t++) for (j = 0; j < T; j++) for (c = 0; c < D; c++) {
      dv[j][c] += a[t][j] * dy[t][c];
      dq[t][c] += SCALE * ds[t][j] * k[j][c];
      dk[j][c] += SCALE * ds[t][j] * q[t][c];
    }
    return { T: T, q: q, k: k, v: v, s: s, a: a, y: y, dy: dy, da: da, ds: ds, dq: dq, dk: dk, dv: dv, m: m, S: S, rowDot: rowDot };
  }

  // ── formatting ────────────────────────────────────────────────────────────────────────────
  function num(x, places) {
    if (Math.abs(x) < 5e-13) x = 0;
    var r = Math.round(x);
    var s = Math.abs(x - r) < 1e-9 ? String(r) : x.toFixed(places);
    return s.replace("-", "−");
  }

  // joins the terms of a sum, writing "a - b" for a negative term rather than "a + -b"
  function joinTerms(terms) {
    return terms.map(function (x, i) {
      if (x.charAt(0) === "−") return (i ? " − " : "−") + x.slice(1);
      return (i ? " + " : "") + x;
    }).join("");
  }

  function vec(row, places) {
    return "[" + row.map(function (x) { return num(x, places); }).join(", ") + "]";
  }

  // each matrix: its name in the note, and whether it is one row of d per token or one value per pair
  var MATS = {
    q: { label: "Q", sym: "q", kind: "vec", rowL: "t" },
    k: { label: "K", sym: "k", kind: "vec", rowL: "j" },
    v: { label: "V", sym: "v", kind: "vec", rowL: "j" },
    y: { label: "Y", sym: "y", kind: "vec", rowL: "t" },
    dy: { label: "dY", sym: "dy", kind: "vec", rowL: "t" },
    dq: { label: "dQ", sym: "dq", kind: "vec", rowL: "t" },
    dk: { label: "dK", sym: "dk", kind: "vec", rowL: "j" },
    dv: { label: "dV", sym: "dv", kind: "vec", rowL: "j" },
    s: { label: "S", sym: "s", kind: "pair" },
    a: { label: "A", sym: "a", kind: "pair" },
    da: { label: "dA", sym: "da", kind: "pair" },
    ds: { label: "dS", sym: "ds", kind: "pair" }
  };

  function sym(key, pos) {
    var M = MATS[key];
    if (M.kind === "pair") return "<i>" + M.sym + "</i><sub>" + pos[0] + pos[1] + "</sub>";
    return "<i>" + M.sym + "</i><sub>" + pos[0] + "</sub>[" + pos[1] + "]";
  }

  function rowSym(key, i) {
    return "<i>" + MATS[key].sym + "</i><sub>" + i + "</sub>";
  }

  function shape(key, T) {
    return MATS[key].kind === "pair" ? "{" + T + ", " + T + "}" : "{" + T + ", " + D + "}";
  }

  function axes(key) {
    var M = MATS[key];
    return M.kind === "pair" ? "row t, column j" : "row " + M.rowL + ", column c";
  }

  // one matrix as a grid of cells, with row and column numbers; cls(r, c) gives each cell's classes
  function grid(M, cls, isOut) {
    var rows = M.length, cols = M[0].length, r, c;
    var h = '<div class="aw-grid' + (isOut ? " aw-out" : "") + '" style="grid-template-columns:16px repeat(' + cols + ',var(--aw-cell))"><span></span>';
    for (c = 0; c < cols; c++) h += '<span class="aw-h">' + c + "</span>";
    for (r = 0; r < rows; r++) {
      h += '<span class="aw-h">' + r + "</span>";
      for (c = 0; c < cols; c++) {
        h += '<span class="aw-c ' + cls(r, c) + '"' + (isOut ? ' data-r="' + r + '" data-c="' + c + '"' : "") + ">" + num(M[r][c], 2) + "</span>";
      }
    }
    return h + "</div>";
  }

  function block(key, md, cls, isOut, extra) {
    return '<div><div class="aw-cap"><b>' + MATS[key].label + "</b> " + shape(key, md.T) + ": " + axes(key) + (extra || "") +
      "</div>" + grid(md[key], cls, isOut) + "</div>";
  }

  // ── controls ──────────────────────────────────────────────────────────────────────────────
  function tokensControl(state) {
    return '<span class="aw-seg"><button data-tokens="3"' + (state.T === 3 ? ' class="on"' : "") + ">3 tokens: the example</button>" +
      '<button data-tokens="8"' + (state.T === 8 ? ' class="on"' : "") + ">8 tokens</button></span>";
  }

  function slider(name, label, max, value, min, step) {
    return "<label>" + label + ' <input type="range" data-name="' + name + '" min="' + (min || 0) + '" max="' + max + '" step="' +
      (step || 1) + '" value="' + value + '"> <b data-show="' + name + '">' + value + "</b></label>";
  }

  // wires the token buttons and sliders of one widget; limits(state) gives each slider's max for the token count
  function wire(root, state, limits, draw) {
    function sync() {
      var max = limits(state);
      Object.keys(max).forEach(function (name) {
        var input = root.querySelector('input[data-name="' + name + '"]');
        input.max = max[name];
        if (state[name] > max[name]) state[name] = max[name];
        input.value = state[name];
        root.querySelector('b[data-show="' + name + '"]').textContent = state[name];
      });
      root.querySelectorAll("button[data-tokens]").forEach(function (b) {
        b.className = +b.getAttribute("data-tokens") === state.T ? "on" : "";
      });
      draw();
    }
    root.addEventListener("input", function (e) {
      var name = e.target.getAttribute("data-name");
      if (!name) return;
      state[name] = +e.target.value;
      var shown = root.querySelector('b[data-show="' + name + '"]');
      if (shown) shown.textContent = e.target.value;
      sync();
    });
    root.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-tokens]");
      if (b) { state.T = +b.getAttribute("data-tokens"); sync(); return; }
      var cell = e.target.closest(".aw-out .aw-c");
      if (cell) { state.r = +cell.getAttribute("data-r"); state.c = +cell.getAttribute("data-c"); sync(); }
    });
    sync();
  }

  // ── a matrix product: out = scale * left · right, one cell or row picked ──────────────────
  var PRODUCTS = {
    F2: { title: "F2: the scores, S = Q K<sup>T</sup> / √d", out: "s", left: "q", lt: false, right: "k", rt: true,
          scale: SCALE, over: "c", T: 8, r: 2, c: 1,
          maskNote: "Masked pair: cuBLAS works this score out too, since it knows nothing of the mask. F3 then gives it weight 0, so it is never used." },
    F5: { title: "F5: the mix, Y = A V", out: "y", left: "a", lt: false, right: "v", rt: false,
          scale: 1, over: "j", T: 8, r: 5, c: 0 },
    B2: { title: "B2: the weights' gradient, dA = dY V<sup>T</sup>", out: "da", left: "dy", lt: false, right: "v", rt: true,
          scale: 1, over: "c", T: 3, r: 2, c: 0,
          maskNote: "Masked pair: cuBLAS works this out too, but C1 multiplies it by a = 0, so its ds is 0 and it goes no further." },
    B3: { title: "B3: the values' gradient, dV = A<sup>T</sup> dY", out: "dv", left: "a", lt: true, right: "dy", rt: false,
          scale: 1, over: "t", T: 3, r: 0, c: 0 },
    B7: { title: "B7: the queries' gradient, dQ = dS K / √d", out: "dq", left: "ds", lt: false, right: "k", rt: false,
          scale: SCALE, over: "j", T: 3, r: 2, c: 0 },
    B8: { title: "B8: the keys' gradient, dK = dS<sup>T</sup> Q / √d", out: "dk", left: "ds", lt: true, right: "q", rt: false,
          scale: SCALE, over: "t", T: 3, r: 0, c: 0 }
  };

  function productWidget(root, spec) {
    var out = MATS[spec.out], L = MATS[spec.left], R = MATS[spec.right];
    var rowL = out.kind === "pair" ? "t" : out.rowL, colL = out.kind === "pair" ? "j" : "c";
    var state = { T: spec.T, r: spec.r, c: spec.c };
    root.innerHTML = '<div class="aw-title">' + spec.title + "</div>" +
      '<div class="aw-sub">Click a cell of ' + out.label + ", or move the sliders.</div>" +
      '<div class="aw-ctl">' + tokensControl(state) + slider("r", "token " + rowL, 7, state.r) +
      slider("c", colL === "c" ? "column c" : "token j", 7, state.c) + "</div>" +
      '<div class="aw-body"></div>';
    var body = root.querySelector(".aw-body");

    function draw() {
      var md = model(state.T), T = state.T, r = state.r, c = state.c;
      var n = spec.over === "c" ? D : T, rowForm = spec.over !== "c";
      var useL = {}, useR = {}, rowR = {}, terms = [], masked = 0, k;
      function lpos(k) { return spec.lt ? [k, r] : [r, k]; }
      function rpos(k) { return spec.rt ? [c, k] : [k, c]; }
      for (k = 0; k < n; k++) {
        var lp = lpos(k);
        if (L.kind === "pair" && lp[1] > lp[0]) { masked++; continue; }
        useL[lp] = 1; useR[rpos(k)] = 1; rowR[k] = 1;
        terms.push(k);
      }
      function isMasked(key, i, j) { return MATS[key].kind === "pair" && j > i; }
      var leftCls = function (i, j) {
        var inLine = spec.lt ? j === r : i === r;
        if (isMasked(spec.left, i, j)) return inLine ? "m ml" : "m";
        return useL[[i, j]] ? "u" : "";
      };
      var rightCls = function (i, j) {
        if (useR[[i, j]]) return "u";
        return rowForm && rowR[i] ? "l" : "";
      };
      var outCls = function (i, j) {
        if (i === r && j === c) return "s";
        if (isMasked(spec.out, i, j)) return "m";
        return i === r ? "l" : "";
      };
      var html = '<div class="aw-row">' +
        block(spec.left, md, leftCls, false, spec.lt ? " &nbsp;(read as " + L.label + "<sup>T</sup>)" : "") +
        '<div class="aw-op">·</div>' +
        block(spec.right, md, rightCls, false, spec.rt ? " &nbsp;(read as " + R.label + "<sup>T</sup>)" : "") +
        '<div class="aw-op">=</div>' + block(spec.out, md, outCls, true, "") +
        "</div>";

      // the sum for the picked cell, in symbols and in numbers
      var half = spec.scale !== 1, open = half ? "½ × (" : "", close = half ? ")" : "";
      var symbols = terms.map(function (k) { return sym(spec.left, lpos(k)) + " " + sym(spec.right, rpos(k)); });
      var numbers = terms.map(function (k) {
        var lp = lpos(k), rp = rpos(k);
        return num(md[spec.left][lp[0]][lp[1]], 7) + " × " + num(md[spec.right][rp[0]][rp[1]], 7);
      });
      var value = md[spec.out][r][c];
      html += '<div class="aw-eq"><p>' + sym(spec.out, [r, c]) + " = " + open + symbols.join(" + ") + close + "</p>" +
        '<p class="n">&nbsp;&nbsp;&nbsp;&nbsp; = ' + open + joinTerms(numbers) + close + " = <b>" + num(value, 7) + "</b></p>";
      if (masked > 0) {
        html += '<p class="aw-note">The other ' + masked + " term" + (masked > 1 ? "s" : "") + " of the full product multipl" +
          (masked > 1 ? "y" : "ies") + " a masked 0 of " + L.label + ", so the full product equals the sum over the " +
          terms.length + " visible token" + (terms.length > 1 ? "s" : "") + ".</p>";
      }
      if (rowForm) {
        // the whole row at once: a mix of rows of the right factor
        var coef = terms.map(function (k) { var lp = lpos(k); return md[spec.left][lp[0]][lp[1]]; });
        html += "<p>The whole row: " + rowSym(spec.out, r) + " = " + open +
          terms.map(function (k) { return sym(spec.left, lpos(k)) + " " + rowSym(spec.right, k); }).join(" + ") + close + "</p>" +
          '<p class="n">&nbsp;&nbsp;&nbsp;&nbsp; = ' + open +
          joinTerms(terms.map(function (k, i) { return num(coef[i], 4) + " × " + vec(md[spec.right][k], 4); })) + close +
          " = <b>" + vec(md[spec.out][r], 4) + "</b></p>" +
          '<p class="aw-note">Shape: (1 × ' + terms.length + ") · (" + terms.length + " × " + D + ") = 1 × " + D + ". " +
          terms.length + " row" + (terms.length > 1 ? "s" : "") + " of " + R.label + " mixed, and always one row of " + D +
          " numbers out: 64 in GPT-2, however many tokens are mixed.</p>";
      } else {
        html += '<p class="aw-note">Shape: (1 × ' + D + ") · (" + D + " × 1) = 1 × 1: a dot product over the head's " + D +
          " columns, 64 in GPT-2.</p>";
      }
      if (spec.lt) {
        html += '<p class="aw-note">The ' + L.label + " values used are column " + r + " of " + L.label +
          ", not a row: the sum runs over " + L.label + "'s first index, so it is read turned round, " + L.label + "<sup>T</sup>.</p>";
      }
      if (spec.rt) {
        html += '<p class="aw-note">The ' + R.label + " values used are row " + c + " of " + R.label +
          ", not a column: the sum runs over " + R.label + "'s second index, so it is read turned round, " + R.label + "<sup>T</sup>.</p>";
      }
      if (isMasked(spec.out, r, c) && spec.maskNote) html += '<p class="aw-note"><b>' + spec.maskNote + "</b></p>";
      body.innerHTML = html + "</div>";
    }

    wire(root, state, function (s) {
      return { r: s.T - 1, c: out.kind === "pair" ? s.T - 1 : D - 1 };
    }, draw);
  }

  // ── F3, F4: the softmax of one row ────────────────────────────────────────────────────────
  function softmaxForwardWidget(root) {
    var state = { T: 8, t: 5, shift: 0 };
    root.innerHTML = '<div class="aw-title">F3 and F4: the mask and the softmax of row t</div>' +
      '<div class="aw-sub">Pick a token t. Then add the same number to every score of its row, and watch the weights.</div>' +
      '<div class="aw-ctl">' + tokensControl(state) + slider("t", "token t", 7, state.t) +
      slider("shift", "add to every score", 3, state.shift, -3, 0.5) + "</div>" + '<div class="aw-body"></div>';
    var body = root.querySelector(".aw-body");

    function draw() {
      var md = model(state.T), t = state.t, x = state.shift, j;
      var s = md.s[t].map(function (v) { return v + x; });
      var m = md.m[t] + x;
      var h = '<table class="aw-tbl"><tr><th>j</th><th>s<sub>tj</sub></th><th>s<sub>tj</sub> − m<sub>t</sub></th>' +
        "<th>e<sub>tj</sub> = exp(s − m)</th><th>a<sub>tj</sub> = e / S</th><th></th></tr>";
      for (j = 0; j < state.T; j++) {
        if (j > t) {
          h += '<tr class="m"><td>' + j + '</td><td class="n">' + num(s[j], 7) + '</td><td colspan="2">masked (F3): after token ' + t +
            '</td><td class="n">0</td><td></td></tr>';
          continue;
        }
        var e = Math.exp(s[j] - m), a = md.a[t][j];
        h += "<tr><td>" + j + '</td><td class="n">' + num(s[j], 7) + '</td><td class="n">' + num(s[j] - m, 7) + '</td><td class="n">' +
          num(e, 7) + '</td><td class="n"><b>' + num(a, 7) + '</b></td><td><div class="aw-bar"><i style="left:0;width:' +
          (a * 140).toFixed(1) + 'px"></i></div></td></tr>';
      }
      h += "</table>";
      h += '<div class="aw-eq"><p>m<sub>' + t + "</sub> = the largest visible score = <b>" + num(m, 7) + "</b>, &nbsp; S<sub>" + t +
        "</sub> = the sum of e = <b>" + num(md.S[t], 7) + "</b>, &nbsp; the weights add up to <b>1</b>.</p>";
      h += '<p class="aw-note">Token ' + t + " sees " + (t + 1) + " token" + (t ? "s" : "") + ": columns 0 to " + t + ". " +
        (x !== 0 ? "Every score has " + num(x, 1) + " added, and so has m, so s − m, e, S and every weight stay the same. "
          : "Move the second slider: s and m both move, s − m does not. ") +
        "That is why taking m away changes no weight, and why each row of ds adds up to 0 (Section 5).</p></div>";
      body.innerHTML = h;
    }

    wire(root, state, function (s) { return { t: s.T - 1 }; }, draw);
  }

  // ── C1: the softmax's backward of one row ─────────────────────────────────────────────────
  function softmaxBackwardWidget(root) {
    var state = { T: 3, t: 2 };
    root.innerHTML = '<div class="aw-title">C1: ds = a (da − row_dot), one row</div>' +
      '<div class="aw-sub">Pick a token t: its weights a, their gradients da from B2, and the scores\' gradients ds.</div>' +
      '<div class="aw-ctl">' + tokensControl(state) + slider("t", "token t", 7, state.t) + "</div>" + '<div class="aw-body"></div>';
    var body = root.querySelector(".aw-body");

    function draw() {
      var md = model(state.T), t = state.t, j, sum = 0, big = 0.05;
      for (j = 0; j <= t; j++) big = Math.max(big, Math.abs(md.ds[t][j]));
      var h = '<table class="aw-tbl"><tr><th>j</th><th>a<sub>tj</sub></th><th>da<sub>tj</sub></th><th>da − row_dot</th>' +
        "<th>ds<sub>tj</sub> = a (da − row_dot)</th><th></th></tr>";
      for (j = 0; j < state.T; j++) {
        if (j > t) {
          h += '<tr class="m"><td>' + j + '</td><td class="n">0</td><td class="n">' + num(md.da[t][j], 7) +
            '</td><td>masked</td><td class="n">0</td><td></td></tr>';
          continue;
        }
        var a = md.a[t][j], da = md.da[t][j], ds = md.ds[t][j], w = Math.abs(ds) / big * 70;
        sum += ds;
        h += "<tr><td>" + j + '</td><td class="n">' + num(a, 7) + '</td><td class="n">' + num(da, 7) + '</td><td class="n">' +
          num(da - md.rowDot[t], 7) + '</td><td class="n"><b>' + num(ds, 7) + '</b></td><td><div class="aw-bar c"><i class="' +
          (ds < 0 ? "neg" : "") + '" style="' + (ds < 0 ? "left:" + (70 - w).toFixed(1) : "left:70") + "px;width:" + w.toFixed(1) +
          'px"></i></div></td></tr>';
      }
      h += "</table>";
      var parts = [];
      for (j = 0; j <= t; j++) parts.push(num(md.a[t][j], 7) + " × " + num(md.da[t][j], 7));
      h += '<div class="aw-eq"><p>row_dot<sub>' + t + "</sub> = Σ<sub>j</sub> a<sub>" + t + "j</sub> da<sub>" + t + 'j</sub> = <span class="n">' +
        joinTerms(parts) + "</span> = <b>" + num(md.rowDot[t], 7) + "</b>: the row's da, weighted by a.</p>" +
        "<p>The row's ds add up to <b>" + num(sum, 7) + "</b>.</p>" +
        '<p class="aw-note">A weight whose da is above the weighted mean row_dot gets a positive ds: its score should go up. ' +
        "Each push is scaled by the weight itself, so a weight near 0 gets almost nothing. Token 0 has one weight, always 1, " +
        "so its ds is always 0.</p></div>";
      body.innerHTML = h;
    }

    wire(root, state, function (s) { return { t: s.T - 1 }; }, draw);
  }

  // ── F1, F6 (and B9, B1): where each head's columns sit, at GPT-2's real sizes ──────────────
  function splitWidget(root) {
    var state = { h: 2, c: 0 };
    root.innerHTML = '<div class="aw-title">F1 and F6: splitting qkv into heads, and putting the heads back</div>' +
      '<div class="aw-sub">GPT-2 small\'s real sizes: 12 heads of 64 columns. Pick a head and one of its columns.</div>' +
      '<div class="aw-ctl">' + slider("h", "head h", NUM_HEADS - 1, state.h) + slider("c", "column c in the head", HEAD_SIZE - 1, state.c) +
      "</div>" + '<div class="aw-body"></div>';
    var body = root.querySelector(".aw-body");

    function segs(h) {
      var s = '<div class="aw-segs">';
      for (var i = 0; i < NUM_HEADS; i++) s += '<span class="' + (i === h ? "u" : "") + '">' + i + "</span>";
      return s + "</div>";
    }

    function draw() {
      var h = state.h, c = state.c, x = "";
      ["q", "k", "v"].forEach(function (part, p) {
        var from = p * C + h * HEAD_SIZE;
        x += '<div class="aw-lbl">qkv row t, the ' + part + " part: columns " + p * C + " to " + (p * C + C - 1) + ", 12 heads of 64</div>" +
          segs(h) + '<div class="aw-eq" style="margin:-4px 0 10px"><p>head ' + h + "'s <i>" + part + "</i><sub>t</sub> = qkv row t, columns " +
          from + " to " + (from + HEAD_SIZE - 1) + ", &nbsp; <i>" + part + "</i><sub>t</sub>[" + c + "] = qkv[t][" + p * C +
          " + " + h + " × 64 + " + c + "] = <b>qkv[t][" + (from + c) + "]</b></p></div>";
      });
      x += '<div class="aw-lbl">output row t: columns 0 to 767, the 12 heads\' y<sub>t</sub> side by side</div>' + segs(h) +
        '<div class="aw-eq" style="margin:-4px 0 10px"><p>head ' + h + "'s <i>y</i><sub>t</sub> goes to columns " + h * HEAD_SIZE +
        " to " + (h * HEAD_SIZE + HEAD_SIZE - 1) + ", &nbsp; <i>y</i><sub>t</sub>[" + c + "] → <b>output[t][" + (h * HEAD_SIZE + c) +
        "]</b></p></div>";
      x += '<p class="aw-note">F1: qkv {4, 1024, 2304} becomes q, k, v, each {4, 12, 1024, 64}: one {1024, 64} matrix per head of each ' +
        "sequence, 48 in all. F6: y {4, 12, 1024, 64} goes back to {4, 1024, 768}. Nothing is added or multiplied, values only " +
        "move. B1 and B9 move the gradients along the same arrows, the other way.</p>";
      body.innerHTML = x;
    }

    wire(root, state, function () { return {}; }, draw);
  }

  function init() {
    if (!document.getElementById("aw-style")) {
      var style = document.createElement("style");
      style.id = "aw-style";
      style.textContent = CSS;
      document.head.appendChild(style);
    }
    document.querySelectorAll(".aw[data-widget]").forEach(function (root) {
      var name = root.getAttribute("data-widget");
      if (PRODUCTS[name]) productWidget(root, PRODUCTS[name]);
      else if (name === "softmax-forward") softmaxForwardWidget(root);
      else if (name === "softmax-backward") softmaxBackwardWidget(root);
      else if (name === "split") splitWidget(root);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

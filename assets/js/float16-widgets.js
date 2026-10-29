/* Interactive figures of the note /posts/half-and-bfloat16/.
   Each <div class="fw" data-widget="..."> in the post is filled in by one function below. Every
   rounding is worked out here in doubles, exactly: a value is first made a float (Math.fround), as
   in the trainer, then rounded to the nearest half or bfloat16, a tie to the even one. */
(function () {
  "use strict";

  // The three formats: fraction bits m, exponent bits, bias. Significant bits p = m + 1.
  var FORMATS = [
    { key: "float", name: "float (fp32)", m: 23, ebits: 8, bias: 127 },
    { key: "half", name: "half (fp16)", m: 10, ebits: 5, bias: 15 },
    { key: "bf16", name: "bfloat16", m: 7, ebits: 8, bias: 127 }
  ];

  var CSS = [
    ".fw{--fw-fg:#212529;--fw-muted:#868e96;--fw-line:#dee2e6;--fw-soft:#f8f9fa;--fw-acc:#1971c2;",
    "--fw-sign:#e8590c;--fw-exp:#2f9e44;--fw-frac:#1971c2;--fw-cut:#ced4da;",
    "border:1px solid var(--fw-line);border-radius:8px;padding:14px 16px;margin:1.2em 0 0.4em;color:var(--fw-fg);",
    "font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.45;overflow-x:auto}",
    "html[data-theme='dark'] .fw{--fw-fg:#e9ecef;--fw-muted:#adb5bd;--fw-line:#495057;--fw-soft:#2b3035;--fw-acc:#4dabf7;",
    "--fw-sign:#ff922b;--fw-exp:#69db7c;--fw-frac:#4dabf7;--fw-cut:#495057}",
    ".fw-title{font-weight:600;margin:0 0 2px}",
    ".fw-sub{color:var(--fw-muted);font-size:13px;margin:0 0 10px}",
    ".fw-ctl{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:0 0 12px;font-size:13px}",
    ".fw-ctl label{display:flex;align-items:center;gap:6px;white-space:nowrap}",
    ".fw-ctl input[type=text]{font:inherit;font-size:13px;width:150px;padding:3px 6px;border:1px solid var(--fw-line);",
    "border-radius:4px;background:transparent;color:var(--fw-fg)}",
    ".fw-ctl input[type=range]{width:170px}",
    ".fw-ctl b{font-variant-numeric:tabular-nums}",
    ".fw-btn{font:inherit;font-size:12px;padding:3px 9px;border:1px solid var(--fw-line);border-radius:5px;",
    "background:transparent;color:var(--fw-fg);cursor:pointer}",
    ".fw-btn.on{background:var(--fw-acc);border-color:var(--fw-acc);color:#fff}",
    ".fw-card{border-top:1px solid var(--fw-line);padding:10px 0 8px}",
    ".fw-name{font-weight:600;font-size:13px;margin:0 0 4px}",
    ".fw-bits{display:flex;flex-wrap:wrap;gap:1px;margin:0 0 6px}",
    ".fw-bits span{width:13px;height:20px;font:11px/18px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;",
    "text-align:center;border:1px solid;border-radius:2px;box-sizing:border-box;color:#fff}",
    ".fw-bits span.s{background:var(--fw-sign);border-color:var(--fw-sign)}",
    ".fw-bits span.e{background:var(--fw-exp);border-color:var(--fw-exp)}",
    ".fw-bits span.f{background:var(--fw-frac);border-color:var(--fw-frac)}",
    ".fw-bits span.c{background:transparent;border-color:var(--fw-cut);color:var(--fw-muted)}",
    ".fw-bits i{width:6px}",
    ".fw-eq{font-size:13px;margin:0 0 3px}",
    ".fw-eq .n,.fw-tbl td.n{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}",
    ".fw-note{color:var(--fw-muted);font-size:12.5px;margin:6px 0 0}",
    ".fw-tbl{border-collapse:collapse;font-size:12.5px;margin:0 0 10px;display:table;width:auto}",
    ".fw-tbl th,.fw-tbl td{padding:3px 10px;border-bottom:1px solid var(--fw-line);text-align:right;white-space:nowrap}",
    ".fw-tbl th{color:var(--fw-muted);font-weight:500}",
    ".fw-tbl td:first-child,.fw-tbl th:first-child{text-align:left}",
    ".fw-chart{width:100%;max-width:620px;height:auto;display:block}",
    ".fw-chart text{font-size:11px;fill:var(--fw-muted)}",
    ".fw-key{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--fw-muted);margin:4px 0 0}",
    ".fw-key span::before{content:'';display:inline-block;width:14px;height:3px;margin:0 5px 3px 0;vertical-align:middle;",
    "background:currentColor}"
  ].join("");

  var LINE_COLORS = { exact: "#adb5bd", float: "#1971c2", half: "#e8590c", bf16: "#2f9e44" };

  // ── rounding ──────────────────────────────────────────────────────────────────────────────

  /// Gives the largest e with 2^e <= a, for a > 0, exactly (log2 alone can be off by one).
  function exponentOf(a) {
    var e = Math.floor(Math.log2(a));
    while (Math.pow(2, e) > a) e--;
    while (Math.pow(2, e + 1) <= a) e++;
    return e;
  }

  /// Gives x rounded to the nearest value of the format, a tie to the one whose last bit is 0; inf past
  /// the largest. x is a float already, so x / spacing is exact in a double.
  function roundTo(x, format) {
    if (!isFinite(x) || x === 0) return x;
    var sign = x < 0 ? -1 : 1, a = Math.abs(x);
    var e = Math.max(exponentOf(a), 1 - format.bias);      // below the smallest normal: the subnormals' spacing
    var spacing = Math.pow(2, e - format.m);
    var q = a / spacing, r = Math.floor(q), rest = q - r;
    if (rest > 0.5 || (rest === 0.5 && r % 2 === 1)) r += 1;
    var value = r * spacing;
    return value > largest(format) ? sign * Infinity : sign * value;
  }

  function largest(format) {
    return (2 - Math.pow(2, -format.m)) * Math.pow(2, Math.pow(2, format.ebits) - 2 - format.bias);
  }

  /// Gives the stored fields of a value of the format: sign s, exponent field E, fraction field f, and
  /// whether it is subnormal.
  function fieldsOf(v, format) {
    var s = v < 0 || (v === 0 && 1 / v < 0) ? 1 : 0, a = Math.abs(v);
    var allOnes = Math.pow(2, format.ebits) - 1;
    if (isNaN(v)) return { s: 0, E: allOnes, f: 1, kind: "NaN" };
    if (!isFinite(v)) return { s: s, E: allOnes, f: 0, kind: "inf" };
    if (a === 0) return { s: s, E: 0, f: 0, kind: "zero" };
    var e = exponentOf(a), emin = 1 - format.bias;
    if (e < emin) return { s: s, E: 0, f: Math.round(a / Math.pow(2, emin - format.m)), kind: "subnormal" };
    return { s: s, E: e + format.bias, f: Math.round((a / Math.pow(2, e) - 1) * Math.pow(2, format.m)), kind: "normal" };
  }

  function binary(n, width) {
    var text = n.toString(2);
    while (text.length < width) text = "0" + text;
    return text;
  }

  // ── formatting ────────────────────────────────────────────────────────────────────────────

  /// Writes a number with up to `digits` significant digits, exact values in full when short enough.
  function show(x, digits) {
    if (x === Infinity) return "inf";
    if (x === -Infinity) return "−inf";
    if (isNaN(x)) return "NaN";
    if (x === 0) return "0";
    var text = String(x);
    if (text.replace(/[-.]/g, "").replace(/e.*/, "").length > (digits || 10)) text = x.toPrecision(digits || 10);
    text = text.replace(/(\.\d*?)0+(e|$)/, "$1$2").replace(/\.(e|$)/, "$1");
    text = text.replace(/e([+-])(\d+)/, function (all, sign, power) { return " × 10" + "<sup>" + (sign === "-" ? "−" : "") + power + "</sup>"; });
    return text.replace(/^-/, "−");
  }

  function powerOf2(e) {
    return "2<sup>" + (e < 0 ? "−" + (-e) : e) + "</sup>";
  }

  // ── Interactive 1: one number in three formats ────────────────────────────────────────────

  var PRESETS_1 = [
    ["0.1", "0.1"], ["1/3", "1/3"], ["257", "257"], ["70000", "70000"], ["1e-8", "1e-8"],
    ["a logit's gradient", "4.85784e-9"]
  ];

  function parseNumber(text) {
    var t = text.replace(/\s/g, "").replace(/−/g, "-");
    var parts = t.split("/");
    if (parts.length === 2) return Number(parts[0]) / Number(parts[1]);
    return Number(t);
  }

  function bitsHtml(fields, format, cutBits) {
    var html = '<span class="s">' + fields.s + "</span><i></i>";
    binary(fields.E, format.ebits).split("").forEach(function (b) { html += '<span class="e">' + b + "</span>"; });
    html += "<i></i>";
    binary(fields.f, format.m).split("").forEach(function (b) { html += '<span class="f">' + b + "</span>"; });
    if (cutBits) cutBits.split("").forEach(function (b) { html += '<span class="c">' + b + "</span>"; });
    return '<div class="fw-bits">' + html + "</div>";
  }

  function converterWidget(root) {
    var state = { text: "0.1" };
    root.innerHTML = '<div class="fw-title">One number in three formats</div>' +
      '<div class="fw-sub">Type a number (1/3 works too), or pick one. Orange: the sign; green: the exponent field ' +
      "E; blue: the fraction field f.</div>" +
      '<div class="fw-ctl"><label>number <input type="text" value="0.1"></label>' +
      PRESETS_1.map(function (p) { return '<button class="fw-btn" data-value="' + p[1] + '">' + p[0] + "</button>"; }).join("") +
      '</div><div class="fw-body"></div>';
    var input = root.querySelector("input"), body = root.querySelector(".fw-body");

    function draw() {
      var x = parseNumber(state.text);
      if (isNaN(x)) { body.innerHTML = '<p class="fw-note">Not a number.</p>'; return; }
      var asFloat = Math.fround(x), html = "";
      root.querySelectorAll(".fw-btn").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-value") === state.text); });
      FORMATS.forEach(function (format) {
        var v = roundTo(asFloat, format), fields = fieldsOf(v, format);
        var cut = "";
        if (format.key === "bf16" && isFinite(asFloat)) {
          var floatFields = fieldsOf(asFloat, FORMATS[0]);
          if (floatFields.kind === "normal") cut = binary(floatFields.f, 23).slice(7);
        }
        html += '<div class="fw-card"><div class="fw-name">' + format.name + " &nbsp;<span class=\"fw-note\">1 + " +
          format.ebits + " + " + format.m + " = " + (1 + format.ebits + format.m) + " bits</span></div>" +
          bitsHtml(fields, format, format.key === "bf16" ? cut : "");
        if (fields.kind === "normal") {
          var e = fields.E - format.bias;
          html += '<p class="fw-eq">E = ' + fields.E + ", so e = E − " + format.bias + " = " + e + "; &nbsp; f = " + fields.f +
            ", so 1 + f / 2<sup>" + format.m + "</sup> = " + show(1 + fields.f / Math.pow(2, format.m), 12) + "</p>" +
            '<p class="fw-eq">value = ' + (fields.s ? "−" : "") + powerOf2(e) + " × " + show(1 + fields.f / Math.pow(2, format.m), 12) +
            " = <b class=\"n\">" + show(v, 17) + "</b></p>" +
            '<p class="fw-eq">its neighbours are ' + powerOf2(e - format.m) + " = " + show(Math.pow(2, e - format.m), 4) +
            " apart</p>";
        } else if (fields.kind === "subnormal") {
          html += '<p class="fw-eq">E = 0: subnormal, value = ' + powerOf2(1 - format.bias) + " × f / 2<sup>" + format.m +
            "</sup> = " + powerOf2(1 - format.bias - format.m) + " × " + fields.f + " = <b class=\"n\">" + show(v, 17) +
            "</b>; neighbours " + show(Math.pow(2, 1 - format.bias - format.m), 4) + " apart</p>";
        } else if (fields.kind === "zero") {
          html += '<p class="fw-eq">E = 0 and f = 0: <b>zero</b>. The number is below half the smallest value, ' +
            show(Math.pow(2, -format.bias - format.m), 4) + "</p>";
        } else {
          html += '<p class="fw-eq">E all 1s and f = 0: <b>' + (fields.s ? "−" : "") + "inf</b>. The largest " + format.name +
            " is " + show(largest(format), 6) + "</p>";
        }
        if (isFinite(v) && x !== 0) {
          var error = Math.abs(v - x);
          html += '<p class="fw-eq">off from ' + show(x, 17) + " by " + show(error, 4) + ", " +
            (v === 0 ? "all" : show(error / Math.abs(x), 3)) + " of it" + (fields.kind === "normal" ? "; at most 2<sup>−" + (format.m + 1) + "</sup> = " +
            show(Math.pow(2, -format.m - 1), 3) : "") + "</p>";
        }
        if (format.key === "bf16" && cut) {
          html += '<p class="fw-note">The grey bits are the float\'s last 16, which bfloat16 drops. As a fraction of ' +
            "one last-place step they are " + show(parseInt(cut, 2) / 65536, 6) + ": " +
            (parseInt(cut, 2) > 32768 ? "more than one half, so the kept bits are rounded up" :
              parseInt(cut, 2) < 32768 ? "less than one half, so the kept bits stay" :
                "exactly one half, a tie: rounded to the even one") + ".</p>";
        }
        html += "</div>";
      });
      body.innerHTML = html;
    }

    input.addEventListener("input", function () { state.text = input.value; draw(); });
    root.addEventListener("click", function (event) {
      var button = event.target.closest(".fw-btn");
      if (!button) return;
      state.text = button.getAttribute("data-value");
      input.value = state.text;
      draw();
    });
    draw();
  }

  // ── Interactive 2: adding the same number again and again ─────────────────────────────────

  var PRESETS_2 = [
    { label: "1 + 1 + 1 + …", start: 0, add: 1, steps: 4096 },
    { label: "a weight, 0.07, minus 1e-4 each step", start: 0.07, add: -1e-4, steps: 400 },
    { label: "1 + 0.001 + 0.001 + …", start: 1, add: 0.001, steps: 1000 }
  ];

  /// Gives the running sums start, start + add, ... after 0 .. steps adds, each add worked out exactly
  /// and rounded to the format, and the first add that changed nothing (then nothing changes again).
  function runningSums(start, add, steps, format) {
    var round = format ? function (x) { return roundTo(Math.fround(x), format); } : function (x) { return x; };
    var a = round(add), s = round(start), sums = [s], stuck = -1;
    for (var k = 1; k <= steps; k++) {
      var next = round(s + a);
      if (format && next === s && stuck < 0) stuck = k;
      s = next;
      sums.push(s);
    }
    return { sums: sums, stuck: stuck, add: a, start: sums[0] };
  }

  function chartHtml(series, steps) {
    var W = 620, H = 210, left = 54, right = 10, top = 10, bottom = 26;
    var low = Infinity, high = -Infinity;
    series.forEach(function (one) { one.sums.forEach(function (v) { if (isFinite(v)) { low = Math.min(low, v); high = Math.max(high, v); } }); });
    if (high === low) { high += 1; low -= 1; }
    var x = function (k) { return left + (W - left - right) * k / steps; };
    var y = function (v) { return top + (H - top - bottom) * (high - v) / (high - low); };
    var html = '<svg class="fw-chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="the running sums">';
    [low, (low + high) / 2, high].forEach(function (v) {
      html += '<line x1="' + left + '" x2="' + (W - right) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="currentColor" stroke-opacity="0.12"/>' +
        '<text x="' + (left - 6) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + show(Number(v.toPrecision(4)), 5).replace(/<[^>]+>/g, "") + "</text>";
    });
    html += '<text x="' + left + '" y="' + (H - 6) + '">0</text><text x="' + (W - right) + '" y="' + (H - 6) +
      '" text-anchor="end">' + steps + " adds</text>";
    series.forEach(function (one) {
      var every = Math.max(1, Math.floor(steps / 400)), points = [];
      for (var k = 0; k <= steps; k += every) points.push(x(k).toFixed(1) + "," + y(one.sums[k]).toFixed(1));
      points.push(x(steps).toFixed(1) + "," + y(one.sums[steps]).toFixed(1));
      html += '<polyline fill="none" stroke="' + LINE_COLORS[one.key] + '" stroke-width="' + (one.key === "exact" ? 4 : 2) +
        '"' + (one.key === "exact" ? ' stroke-opacity="0.6"' : "") + ' points="' + points.join(" ") + '"/>';
    });
    return html + "</svg>";
  }

  function runningSumWidget(root) {
    var state = { preset: 0, steps: PRESETS_2[0].steps };
    root.innerHTML = '<div class="fw-title">Adding the same number again and again</div>' +
      '<div class="fw-sub">Each add is worked out exactly, then rounded to the format, as a kernel that keeps its sum in ' +
      "that format would. Pick a sum, then move the slider.</div>" +
      '<div class="fw-ctl">' + PRESETS_2.map(function (p, i) { return '<button class="fw-btn" data-preset="' + i + '">' + p.label + "</button>"; }).join("") +
      '</div><div class="fw-ctl"><label>adds <input type="range" min="1" max="4096" step="1"> <b class="fw-steps"></b></label></div>' +
      '<div class="fw-body"></div>';
    var slider = root.querySelector("input"), body = root.querySelector(".fw-body"), count = root.querySelector(".fw-steps");

    function draw() {
      var p = PRESETS_2[state.preset], steps = state.steps;
      root.querySelectorAll(".fw-btn").forEach(function (b) { b.classList.toggle("on", Number(b.getAttribute("data-preset")) === state.preset); });
      slider.value = steps;
      count.textContent = steps;
      var series = [{ key: "exact", name: "exact", run: runningSums(p.start, p.add, steps, null) }];
      FORMATS.forEach(function (format) { series.push({ key: format.key, name: format.name, format: format, run: runningSums(p.start, p.add, steps, format) }); });
      var html = '<table class="fw-tbl"><tr><th></th><th>start</th><th>each add</th><th>after ' + steps +
        " adds</th><th>stops moving at add</th></tr>", whys = "";
      series.forEach(function (one) {
        var run = one.run, last = run.sums[steps];
        if (one.format && run.stuck > 0) {
          var stuckAt = run.sums[run.stuck], spacing = Math.pow(2, Math.max(exponentOf(Math.abs(stuckAt) || 1), 1 - one.format.bias) - one.format.m);
          whys += '<p class="fw-note"><b>' + one.name + "</b> stops at add " + run.stuck + ": at " + show(stuckAt, 8) +
            " its neighbours are " + show(spacing, 4) + " apart, and an add of " + show(Math.abs(run.add), 4) +
            (Math.abs(run.add) === spacing / 2 ? " is exactly half of that, a tie, rounded back to the even one."
              : " is less than half of that, so the sum rounds back to where it was.") + "</p>";
        }
        html += "<tr><td>" + one.name + '</td><td class="n">' + show(run.start, 10) + '</td><td class="n">' + show(run.add, 10) +
          '</td><td class="n">' + show(last, 10) + "</td><td>" + (one.format ? (run.stuck > 0 ? run.stuck : "—") : "") +
          "</td></tr>";
      });
      html += "</table>" + whys + chartHtml(series.map(function (one) { return { key: one.key, sums: one.run.sums }; }), steps) +
        '<div class="fw-key">' + series.map(function (one) { return '<span style="color:' + LINE_COLORS[one.key] + '">' + one.name + "</span>"; }).join("") + "</div>";
      body.innerHTML = html;
    }

    root.addEventListener("click", function (event) {
      var button = event.target.closest(".fw-btn");
      if (!button) return;
      state.preset = Number(button.getAttribute("data-preset"));
      state.steps = PRESETS_2[state.preset].steps;
      draw();
    });
    slider.addEventListener("input", function () { state.steps = Number(slider.value); draw(); });
    draw();
  }

  function init() {
    if (!document.getElementById("fw-style")) {
      var style = document.createElement("style");
      style.id = "fw-style";
      style.textContent = CSS;
      document.head.appendChild(style);
    }
    document.querySelectorAll(".fw[data-widget]").forEach(function (root) {
      var name = root.getAttribute("data-widget");
      if (name === "converter") converterWidget(root);
      else if (name === "running-sum") runningSumWidget(root);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

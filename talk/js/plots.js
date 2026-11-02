/* Animated versions of the paper's Figs. 2-6 (and three schematics), drawn with D3.
 *
 * Data: window.PAPER_DATA, written by scripts/extract_data.py with the paper's own loaders
 * and statistics; nothing is recomputed here except the layout. Styling follows
 * papers/iclr27/figures/common.py: protocol colours, signal markers and line styles, faded
 * AU/EU in calibration panels (CAL_ALPHA), log axes for RMSE and ENCE.
 *
 * Every figure is a list of steps. Step 0 plays when its slide opens; step k plays when the
 * k-th `.fragment.step` placeholder of the slide is shown (see index.html). Result figures are
 * laid out 900 units wide, to sit next to the numbered findings.
 */
(function () {
  "use strict";
  const D = window.PAPER_DATA;
  const PROTOCOLS = ["HF-only", "LF→HF"];
  const SIGNALS = ["AU", "EU", "TU"];
  const PCOL = { "HF-only": "#009099", "LF→HF": "#c0443d" };
  const SCOL = { AU: "#2f6fc0", EU: "#e8952f", TU: "#1d6b3a" };
  const KCOL = { system: "#6f4e9c", energy: "#a6761d" };
  const DASH = { AU: "10,4", EU: "2.5,4", TU: null };
  const CAL = { AU: 0.35, EU: 0.35, TU: 1 };
  const INK = "#222222", MUTED = "#6b6b6b", GRID = "#e6e6e6", FAINT = "#f4f4f2", DIM = 0.14;
  const FS = { tick: 19, label: 21, title: 21, legend: 19, letter: 24 };
  const fmt = d3.format("~g");
  let UID = 0;
  const uid = (p) => `${p}-${++UID}`;

  // ------------------------------------------------------------------ helpers

  function tr(sel, anim, dur = 700, delay = 0, ease = d3.easeCubicOut) {
    return anim ? sel.transition().duration(dur).delay(delay).ease(ease) : sel.interrupt();
  }

  function text(g, x, y, str, o = {}) {
    return g.append("text").attr("x", x).attr("y", y).text(str)
      .attr("font-size", o.size || FS.label).attr("fill", o.color || INK)
      .attr("text-anchor", o.anchor || "start").attr("font-weight", o.weight || 400)
      .attr("dominant-baseline", o.baseline || "auto").attr("font-style", o.italic ? "italic" : null)
      .attr("transform", o.rotate ? `rotate(${o.rotate},${x},${y})` : null);
  }

  function title(g, x, y, letter, str, size = FS.title) {
    const t = g.append("text").attr("x", x).attr("y", y).attr("fill", INK);
    if (letter) t.append("tspan").text(letter + " ").attr("font-weight", 800).attr("font-size", FS.letter);
    t.append("tspan").text(str).attr("font-size", size).attr("font-weight", 500);
    return t;
  }

  function xAxis(g, X, y, ticks, format = fmt) {
    g.append("g").attr("class", "ax").attr("transform", `translate(0,${y})`)
      .call(d3.axisBottom(X).tickValues(ticks).tickSize(6).tickPadding(6).tickFormat(format));
  }

  function yAxis(g, Y, x, ticks, format = fmt) {
    g.append("g").attr("class", "ax").attr("transform", `translate(${x},0)`)
      .call(d3.axisLeft(Y).tickValues(ticks).tickSize(6).tickPadding(6).tickFormat(format));
  }

  function grid(g, scale, ticks, a, b, vertical) {
    g.append("g").selectAll("line").data(ticks).join("line")
      .attr(vertical ? "x1" : "y1", (d) => scale(d)).attr(vertical ? "x2" : "y2", (d) => scale(d))
      .attr(vertical ? "y1" : "x1", a).attr(vertical ? "y2" : "x2", b)
      .attr("stroke", GRID).attr("stroke-width", 1.1);
  }

  function sym(sig, ms) {
    if (sig === "AU") return { d: d3.symbol(d3.symbolCircle, Math.PI * (ms / 2) ** 2)(), rot: 0 };
    if (sig === "EU") return { d: d3.symbol(d3.symbolSquare, (ms * 0.86) ** 2)(), rot: 0 };
    return { d: d3.symbol(d3.symbolSquare, (ms * 0.8) ** 2)(), rot: 45 };
  }

  function marker(g, sig, x, y, ms, color, o = {}) {
    const s = sym(sig, ms);
    const outer = g.append("g").attr("transform", `translate(${x},${y})`);
    const path = outer.append("path").attr("d", s.d)
      .attr("transform", `rotate(${s.rot}) scale(${o.scale ?? 1})`)
      .attr("fill", o.hollow ? "white" : color).attr("stroke", o.hollow ? color : "white")
      .attr("stroke-width", o.hollow ? 2.4 : 1.2);
    return { outer, path, rot: s.rot };
  }

  const grow = (m, anim, delay = 0, dur = 550) =>
    tr(m.path, anim, dur, delay, d3.easeBackOut.overshoot(1.6)).attr("transform", `rotate(${m.rot}) scale(1)`);

  function legend(svg, items, cx, y, size = FS.legend, gap = 24) {
    const g = svg.append("g");
    let x = 0;
    for (const it of items) {
      const gi = g.append("g").attr("transform", `translate(${x},0)`);
      let w = 0;
      if (it.line) {
        gi.append("line").attr("x1", 0).attr("x2", 32).attr("stroke", it.color)
          .attr("stroke-width", it.width || 3).attr("stroke-dasharray", it.dash || null);
        w = 39;
      } else if (it.patch) {
        gi.append("rect").attr("x", 0).attr("y", -9).attr("width", 28).attr("height", 18)
          .attr("fill", it.color).attr("opacity", it.opacity ?? 1);
        w = 35;
      } else if (it.sig) {
        marker(gi, it.sig, 9, 0, 15, it.color, { hollow: it.hollow });
        w = 24;
      }
      gi.append("text").attr("x", w).attr("dominant-baseline", "central").attr("font-size", size)
        .attr("fill", INK).text(it.label);
      x += w + it.label.length * size * 0.56 + gap;
    }
    g.attr("transform", `translate(${cx - (x - gap) / 2},${y})`);
    return g;
  }

  function logTicks(lo, hi, max = 5) {
    const all = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];
    let t = all.filter((v) => v >= lo && v <= hi);
    if (t.length > max) t = t.filter((v) => [0.01, 0.1, 1, 10, 100, 1000].includes(v) || ([0.2, 2, 20].includes(v) && t.length <= max + 2));
    if (t.length > max) t = t.filter((v) => [0.01, 0.1, 1, 10, 100].includes(v));
    return t;
  }

  /* Epoch sweep: panels whose data group is clipped up to the current epoch. */
  function sweepPanel(g, X, top, bottom) {
    const id = uid("clip");
    const rect = g.append("clipPath").attr("id", id).append("rect")
      .attr("x", X.range()[0] - 12).attr("y", top - 40).attr("height", bottom - top + 80).attr("width", 0);
    const data = g.append("g").attr("clip-path", `url(#${id})`);
    const cursor = g.append("line").attr("y1", top).attr("y2", bottom).attr("stroke", "#8a8a8a")
      .attr("stroke-width", 1.4).attr("opacity", 0);
    return {
      g, data,
      set(e, moving) {
        const x = Math.min(X(e), X.range()[1] + 12);
        rect.attr("width", Math.max(0, x - (X.range()[0] - 12)));
        cursor.attr("x1", X(e)).attr("x2", X(e)).attr("opacity", moving ? 1 : 0);
      },
    };
  }

  function sweeper(panels, label, describe) {
    const set = (e, moving) => {
      panels.forEach((p) => p.set(e, moving));
      if (label) label.text(describe(e));
    };
    return (from, to, dur, anim, delay = 0) => {
      if (!anim) { set(to, false); return; }
      set(from, false);
      (label || panels[0].data).transition(uid("sweep")).delay(delay).duration(dur).ease(d3.easeLinear)
        .tween("epoch", () => (t) => set(from + (to - from) * t, t < 1))
        .on("end", () => set(to, false));
    };
  }

  /* One smoothed series with its band; returns the group (for focusing). */
  function series(parent, X, Y, ser, style, splitAt) {
    const g = parent.append("g").attr("opacity", style.alpha);
    const pts = ser.epoch.map((e, i) => ({ e, m: ser.mean[i], lo: ser.lo[i], hi: ser.hi[i] }));
    const parts = splitAt == null ? [pts] : [pts.filter((p) => p.e < splitAt), pts.filter((p) => p.e >= splitAt)];
    const ok = (v) => Number.isFinite(v) && (Y.base ? v > 0 : true);
    for (const part of parts) {
      if (!part.length) continue;
      g.append("path").datum(part)
        .attr("d", d3.area().defined((p) => ok(p.lo) && ok(p.hi)).x((p) => X(p.e)).y0((p) => Y(p.lo)).y1((p) => Y(p.hi)))
        .attr("fill", style.color).attr("opacity", 0.13);
      g.append("path").datum(part)
        .attr("d", d3.line().defined((p) => ok(p.m)).x((p) => X(p.e)).y((p) => Y(p.m)))
        .attr("fill", "none").attr("stroke", style.color).attr("stroke-width", style.width)
        .attr("stroke-dasharray", style.dash).attr("stroke-linecap", "round");
    }
    return g;
  }

  function arrowDefs(svg, color) {
    const id = uid("arrow");
    svg.append("defs").append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 8)
      .attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", color);
    return `url(#${id})`;
  }

  function rng(seed) {
    return function () {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const normal = (r) => Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());

  const FIGS = {};

  // ------------------------------------------------------------------ paper Fig. 2

  FIGS.final = {
    w: 900, h: 640,
    build(svg) {
      const F = D.final, H = 640, tests = ["System-ID", "System-OOD", "Energy-ID", "Energy-OOD"];
      const slot = {}, bands = [];
      let y = 0;
      for (const t of tests) {
        const ys = [];
        for (const p of PROTOCOLS) {
          for (const s of SIGNALS) { slot[t + p + s] = y; ys.push(y); y -= 0.42; }
          y -= 0.25;
        }
        bands.push({ t, c: d3.mean(ys), top: ys[0] + 0.3, bot: ys[ys.length - 1] - 0.3 });
        y -= 0.45;
      }
      const base = H - 36;
      const Y = d3.scaleLinear().domain([bands[3].bot - 0.1, bands[0].top + 0.1]).range([base, 84]);
      legend(svg, [
        { sig: "AU", color: PCOL["HF-only"], label: "HF-only" }, { sig: "AU", color: PCOL["LF→HF"], label: "LF→HF" },
        { sig: "AU", color: MUTED, label: "AU" }, { sig: "EU", color: MUTED, label: "EU" }, { sig: "TU", color: MUTED, label: "TU" },
      ], 500, 14);
      bands.forEach((b) => text(svg, 128, Y(b.c), b.t, { anchor: "end", baseline: "central", size: 21, color: "#333" }));
      const L = 138, gap = 16, ratios = [0.95, 1, 1.2], avail = 900 - L - 4 - 2 * gap;
      const cols = [
        { m: "RMSE", letter: "a", title: "RMSE ←", log: true, dom: [4.5, 110], ticks: [5, 10, 20, 50, 100] },
        { m: "AUSE", letter: "b", title: "AUSE ← ranking", log: false, dom: [0.125, 0.425], ticks: [0.2, 0.3, 0.4] },
        { m: "ENCE", letter: "c", title: "ENCE ← calibration", log: true, dom: [0.14, 19], ticks: [0.2, 0.5, 1, 2, 5, 10] },
      ];
      const groups = {};
      let x0 = L, highlight = null;
      cols.forEach((c, i) => {
        const w = (avail * ratios[i]) / 3.15;
        const X = (c.log ? d3.scaleLog() : d3.scaleLinear()).domain(c.dom).range([x0, x0 + w]);
        const g = svg.append("g");
        bands.forEach((b, j) => {
          if (j % 2 === 0) g.append("rect").attr("x", x0).attr("width", w).attr("y", Y(b.top))
            .attr("height", Y(b.bot) - Y(b.top)).attr("fill", FAINT);
        });
        grid(g, X, c.ticks, Y(bands[0].top + 0.1), base, true);
        xAxis(g, X, base, c.ticks);
        title(g, x0, 64, c.letter, c.title);
        const els = [];
        tests.forEach((t, ti) => PROTOCOLS.forEach((p, pi) => {
          (c.m === "RMSE" ? [null] : SIGNALS).forEach((s, si) => {
            const v = c.m === "RMSE" ? F[t][p].RMSE : F[t][p][c.m][s];
            const yy = c.m === "RMSE" ? Y(d3.mean(SIGNALS, (q) => slot[t + p + q])) : Y(slot[t + p + s]);
            const alpha = c.m === "ENCE" ? CAL[s] : 1;
            const gg = g.append("g").attr("opacity", 0);
            const line = gg.append("line").attr("x1", X(v[0])).attr("x2", X(v[0])).attr("y1", yy).attr("y2", yy)
              .attr("stroke", PCOL[p]).attr("stroke-width", 2.4);
            const ms = c.m === "RMSE" || (c.m === "ENCE" && s === "TU") ? 15 : 12;
            const mk = marker(gg, s || "AU", X(v[0]), yy, ms, PCOL[p], { scale: 0.001 });
            els.push({ gg, line, mk, X, v, s, alpha, delay: ti * 120 + pi * 55 + si * 30 });
          });
        }));
        groups[c.m] = els;
        if (c.m === "ENCE") {
          const b = bands[3];
          highlight = g.append("rect").attr("x", X(1.08)).attr("width", X(5.3) - X(1.08))
            .attr("y", Y(b.top) + 3).attr("height", Y(b.bot) - Y(b.top) - 6).attr("rx", 8)
            .attr("fill", "none").attr("stroke", "#e05a2b").attr("stroke-width", 3)
            .attr("stroke-dasharray", "8,5").attr("opacity", 0);
        }
        x0 += w + gap;
      });
      const show = (m, anim, extra = 0) => groups[m].forEach((e) => {
        tr(e.gg, anim, 250, e.delay + extra).attr("opacity", e.alpha);
        grow(e.mk, anim, e.delay + extra);
        tr(e.line, anim, 650, e.delay + extra + 120).attr("x1", e.X(e.v[1])).attr("x2", e.X(e.v[2]));
      });
      return [
        (anim) => show("RMSE", anim),
        (anim) => show("AUSE", anim),
        (anim) => groups.AUSE.forEach((e) => tr(e.gg, anim, 500).attr("opacity", e.s === "EU" ? 1 : DIM)),
        (anim) => {
          groups.AUSE.forEach((e) => tr(e.gg, anim, 500).attr("opacity", 1));
          show("ENCE", anim);
          tr(highlight, anim, 600, 1300).attr("opacity", 1);
        },
      ];
    },
  };

  // ------------------------------------------------------------------ paper Fig. 3 (script fig4)

  FIGS.rankcal = {
    w: 900, h: 620,
    build(svg) {
      const F = D.final, H = 620, top = 84, base = H - 72, L = 92, gap = 24, w = (900 - L - 6 - gap) / 2;
      const Y = d3.scaleLinear().domain([0.1, 0.42]).range([base, top]);
      const yt = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
      legend(svg, [
        { line: true, color: PCOL["HF-only"], label: "HF-only" }, { line: true, color: PCOL["LF→HF"], label: "LF→HF" },
        { sig: "AU", color: MUTED, label: "AU" }, { sig: "EU", color: MUTED, label: "EU" }, { sig: "TU", color: MUTED, label: "TU" },
        { sig: "AU", color: MUTED, hollow: true, label: "ID" }, { sig: "AU", color: MUTED, label: "OOD" },
      ], 470, 14, FS.legend, 20);
      const panels = {};
      [["system", "System shift", "a", "System"], ["energy", "Energy shift", "b", "Energy"]].forEach(([kind, name, letter, pre], k) => {
        const x0 = L + k * (w + gap);
        const X = d3.scaleLog().domain([0.15, 25]).range([x0, x0 + w]);
        const xt = [0.2, 0.5, 1, 2, 5, 10, 20];
        const g = svg.append("g");
        grid(g, X, xt, top, base, true);
        grid(g, Y, yt, x0, x0 + w, false);
        xAxis(g, X, base, xt);
        if (k === 0) {
          yAxis(g, Y, x0, yt, d3.format(".2f"));
          text(g, 22, (top + base) / 2, "AUSE  ← (ranking)", { rotate: -90, anchor: "middle" });
        } else {
          g.append("line").attr("x1", x0).attr("x2", x0).attr("y1", top).attr("y2", base).attr("stroke", "#444").attr("stroke-width", 1.3);
        }
        text(g, x0 + w / 2, H - 8, "ENCE  ← (calibration)", { anchor: "middle" });
        title(g, x0, 64, letter, name);
        text(g, x0 + 12, base - 12, "better", { italic: true, color: MUTED, size: 18 });
        const items = [];
        for (const s of SIGNALS) for (const p of PROTOCOLS) {
          const id = F[`${pre}-ID`][p], ood = F[`${pre}-OOD`][p];
          const a = [X(id.ENCE[s][0]), Y(id.AUSE[s][0])], b = [X(ood.ENCE[s][0]), Y(ood.AUSE[s][0])];
          const fade = CAL[s], ms = s === "TU" ? 19 : 14, color = PCOL[p];
          const grp = g.append("g").attr("opacity", fade);
          const line = grp.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", a[0]).attr("y2", a[1])
            .attr("stroke", color).attr("stroke-width", s === "TU" ? 3.2 : 1.8).attr("opacity", 0.75);
          const hollow = marker(grp, s, a[0], a[1], ms, color, { hollow: true, scale: 0.001 });
          const gfill = grp.append("g").attr("opacity", 0);
          const filled = marker(gfill, s, a[0], a[1], ms, color);
          const delay = (s === "TU" ? 2 : s === "EU" ? 1 : 0) * 120 + PROTOCOLS.indexOf(p) * 60;
          items.push({ s, p, grp, line, hollow, gfill, filled, a, b, fade, delay, ratio: ood.ENCE[s][0] / id.ENCE[s][0] });
        }
        const showID = (anim, d0 = 0) => items.forEach((it) => grow(it.hollow, anim, d0 + it.delay));
        const travel = (anim, d0 = 0) => items.forEach((it) => {
          tr(it.gfill, anim, 200, d0 + it.delay).attr("opacity", 1);
          tr(it.filled.outer, anim, 1500, d0 + it.delay, d3.easeCubicInOut).attr("transform", `translate(${it.b[0]},${it.b[1]})`);
          tr(it.line, anim, 1500, d0 + it.delay, d3.easeCubicInOut).attr("x2", it.b[0]).attr("y2", it.b[1]);
        });
        panels[kind] = { g, items, showID, travel };
      });
      // ENCE growth of TU from ID to OOD, annotated next to the energy-shift arrows
      const notes = panels.energy.items.filter((it) => it.s === "TU").map((it) =>
        text(panels.energy.g, it.b[0] + 16, it.b[1] + (it.p === "HF-only" ? -6 : 6), `×${it.ratio.toFixed(1)}`,
          { size: 24, weight: 800, color: PCOL[it.p], baseline: "central" }).attr("opacity", 0));
      return [
        (anim) => panels.system.showID(anim),
        (anim) => panels.system.travel(anim),
        (anim) => { panels.energy.showID(anim); panels.energy.travel(anim, 900); },
        (anim) => {
          for (const k of ["system", "energy"]) panels[k].items.forEach((it) => tr(it.grp, anim, 500).attr("opacity", it.s === "TU" ? 1 : 0.1));
          notes.forEach((n) => tr(n, anim, 500, 300).attr("opacity", 1));
        },
      ];
    },
  };

  // ------------------------------------------------------------------ paper Fig. 4 (script fig5)

  FIGS.decomposition = {
    w: 900, h: 760,
    build(svg) {
      const R = D.decomposition, H = 760, L = 92, gapX = 74, gapY = 76, top = 88, bottomPad = 58;
      const w = (900 - L - 8 - gapX) / 2, h = (H - top - bottomPad - 2 * gapY) / 3;
      legend(svg, [
        { line: true, color: KCOL.system, label: "System split" }, { line: true, color: KCOL.energy, label: "Energy split" },
        { line: true, color: MUTED, dash: DASH.AU, width: 2.4, label: "AU" }, { line: true, color: MUTED, dash: DASH.EU, width: 2.4, label: "EU" },
        { line: true, color: MUTED, label: "TU" },
      ], 300, 14);
      const label = text(svg, 900, 14, "", { anchor: "end", size: 20, color: MUTED, baseline: "central", weight: 700 });
      const panels = [], byKey = {};
      const rows = [["train", "Train set"], ["id", "ID test set"], ["ood", "OOD test set"]];
      const cols = [["AUSE", "ranking", false, "AUSE ←"], ["ENCE", "calibration", true, "ENCE ←"]];
      const letters = "abcdef";
      rows.forEach(([test, tname], r) => cols.forEach(([metric, what, log, ylab], c) => {
        const x0 = L + c * (w + gapX), y0 = top + r * (h + gapY), key = letters[2 * r + c];
        const X = d3.scaleLinear().domain([0, 400]).range([x0, x0 + w]);
        const all = [];
        for (const kind of ["system", "energy"]) for (const s of SIGNALS) all.push(...R[kind][test][metric][s].lo, ...R[kind][test][metric][s].hi);
        const fin = all.filter((v) => Number.isFinite(v) && (!log || v > 0));
        let Y, yt;
        if (log) {
          const lo = d3.min(fin) / 1.25, hi = d3.max(fin) * 1.25;
          Y = d3.scaleLog().domain([lo, hi]).range([y0 + h, y0]);
          yt = logTicks(lo, hi, 4);
        } else {
          const lo = d3.min(fin), hi = d3.max(fin), pad = (hi - lo) * 0.06;
          Y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([y0 + h, y0]);
          yt = Y.ticks(3);
        }
        const g = svg.append("g");
        g.append("rect").attr("x", X(297.5)).attr("width", X(400) - X(297.5)).attr("y", y0).attr("height", h).attr("fill", "#efefec");
        grid(g, Y, yt, x0, x0 + w, false);
        g.append("line").attr("x1", X(297.5)).attr("x2", X(297.5)).attr("y1", y0).attr("y2", y0 + h)
          .attr("stroke", MUTED).attr("stroke-width", 1.1).attr("stroke-dasharray", "4,4");
        xAxis(g, X, y0 + h, [0, 100, 200, 300, 400]);
        yAxis(g, Y, x0, yt, log ? fmt : d3.format(".2~f"));
        title(g, x0, y0 - 12, key, `${tname}: ${what}`, 20);
        text(g, x0 - 60, y0 + h / 2, ylab, { rotate: -90, anchor: "middle", size: 18, color: "#444" });
        if (r === 2) text(g, x0 + w / 2, H - 4, "epoch", { anchor: "middle", size: 20 });
        if (r === 0) {
          text(g, X(150), y0 + 18, "DFT pretraining", { anchor: "middle", size: 15, color: MUTED });
          text(g, X(349), y0 + 18, "CC fine-tune", { anchor: "middle", size: 15, color: MUTED });
        }
        const p = sweepPanel(g, X, y0, y0 + h);
        const sers = [];
        for (const kind of ["system", "energy"]) for (const s of SIGNALS) {
          const alpha = metric === "ENCE" ? CAL[s] : 1;
          sers.push({ s, alpha, g: series(p.data, X, Y, R[kind][test][metric][s],
            { color: KCOL[kind], alpha, dash: DASH[s], width: metric === "ENCE" && s === "TU" ? 3.2 : 2.1 }, 300) });
        }
        panels.push(p);
        byKey[key] = { g, sers };
      }));
      const run = sweeper(panels, label, (e) => e < 299.5 ? `DFT pretraining · epoch ${Math.round(e)}` : `CC fine-tuning · epoch ${Math.round(e)}`);
      run(0, 0, 0, false);
      label.text("");
      /* Focus some panels (others dimmed); optionally emphasise one signal inside them. */
      const focus = (keys, sig) => (anim) => Object.entries(byKey).forEach(([k, v]) => {
        const on = !keys || keys.includes(k);
        tr(v.g, anim, 500).attr("opacity", on ? 1 : DIM);
        v.sers.forEach((q) => tr(q.g, anim, 500).attr("opacity", on && sig ? (q.s === sig ? 1 : 0.12) : q.alpha));
      });
      return [
        (anim) => run(0, 299.5, 5200, anim, 600),
        (anim) => run(299.5, 400, 3400, anim),
        focus(["a", "b"]), focus(["c", "e"]), focus(["d", "f"]), focus(["d", "f"], "AU"), focus(["d", "f"], "EU"), focus(null),
      ];
    },
  };

  // ------------------------------------------------------------------ paper Fig. 5 (script fig3)

  FIGS.hfstage = {
    w: 900, h: 700,
    build(svg) {
      const R = D.hf_dynamics, H = 700, L = 96, gapX = 26, gapY = 34, top = 84, bottomPad = 62;
      const w = (900 - L - 8 - gapX) / 2, h = (H - top - bottomPad - gapY) / 2;
      legend(svg, [
        { line: true, color: PCOL["HF-only"], label: "HF-only" }, { line: true, color: PCOL["LF→HF"], label: "LF→HF" },
        { line: true, color: MUTED, dash: DASH.AU, width: 2.4, label: "AU" }, { line: true, color: MUTED, dash: DASH.EU, width: 2.4, label: "EU" },
        { line: true, color: MUTED, label: "TU" },
      ], 290, 14);
      const label = text(svg, 900, 14, "", { anchor: "end", size: 20, color: MUTED, baseline: "central", weight: 700 });
      const rowsP = [[], []];
      const cols = [["system", "System-OOD", ["a", "c"]], ["energy", "Energy-OOD", ["b", "d"]]];
      const rows = [["AUSE", false, "AUSE ← (ranking)"], ["ENCE", true, "ENCE ← (calibration)"]];
      const doms = {};
      for (const [metric, log] of rows) {
        const all = [];
        for (const [kind] of cols) for (const p of PROTOCOLS) for (const s of SIGNALS) all.push(...R[kind][p][metric][s].lo, ...R[kind][p][metric][s].hi);
        const fin = all.filter((v) => Number.isFinite(v) && v > 0);
        doms[metric] = log ? [d3.min(fin) / 1.2, d3.max(fin) * 1.2] : [Math.min(0.1, d3.min(fin)), d3.max(fin) * 1.03];
      }
      cols.forEach(([kind, name, letters], c) => rows.forEach(([metric, log, ylab], r) => {
        const x0 = L + c * (w + gapX), y0 = top + r * (h + gapY);
        const X = d3.scaleLinear().domain([-5, 300]).range([x0, x0 + w]);
        const Y = (log ? d3.scaleLog() : d3.scaleLinear()).domain(doms[metric]).range([y0 + h, y0]);
        const yt = log ? [0.2, 1, 5, 20, 100].filter((v) => v >= doms[metric][0] && v <= doms[metric][1]) : [0.2, 0.4, 0.6];
        const g = svg.append("g");
        grid(g, Y, yt, x0, x0 + w, false);
        xAxis(g, X, y0 + h, r === 1 ? [0, 100, 200, 300] : [], fmt);
        if (c === 0) yAxis(g, Y, x0, yt, log ? fmt : d3.format(".1f"));
        else g.append("line").attr("x1", x0).attr("x2", x0).attr("y1", y0).attr("y2", y0 + h).attr("stroke", "#444").attr("stroke-width", 1.3);
        if (r === 0) title(g, x0, y0 - 12, letters[0], name);
        else text(g, x0 + 10, y0 + 24, letters[1], { size: FS.letter, weight: 800 });
        if (c === 0) text(g, 22, y0 + h / 2, ylab, { rotate: -90, anchor: "middle", size: 20 });
        if (r === 1) text(g, x0 + w / 2, H - 6, "epoch on accurate labels", { anchor: "middle", size: 20 });
        const p = sweepPanel(g, X, y0, y0 + h);
        for (const prot of PROTOCOLS) for (const s of SIGNALS) {
          series(p.data, X, Y, R[kind][prot][metric][s],
            { color: PCOL[prot], alpha: log ? CAL[s] : 1, dash: DASH[s], width: s === "TU" ? 3.4 : 2.3 }, null);
        }
        rowsP[r].push(p);
      }));
      const runTop = sweeper(rowsP[0], label, (e) => `epoch ${Math.max(0, Math.round(e))}`);
      const runBottom = sweeper(rowsP[1], label, (e) => `epoch ${Math.max(0, Math.round(e))}`);
      runTop(-5, -5, 0, false); runBottom(-5, -5, 0, false);
      label.text("");
      return [(anim) => runTop(-5, 300, 5000, anim, 600), (anim) => runBottom(-5, 300, 5000, anim)];
    },
  };

  // ------------------------------------------------------------------ paper Fig. 6

  FIGS.acquisition = {
    w: 900, h: 620,
    build(svg) {
      const A = D.acquisition, H = 620;
      const ax0 = 86, ax1 = 470, top = 64, base = H - 104;
      const Y = d3.scaleLinear().domain([-0.3, 3.4]).range([base, top]);
      const xs = { "HF-only": [0, 1, 2], "LF→HF": [3.45, 4.45, 5.45] };
      const X = d3.scaleLinear().domain([-0.6, 6.05]).range([ax0, ax1]);
      const g = svg.append("g");
      grid(g, Y, [0, 0.5, 1, 1.5, 2, 2.5, 3], ax0, ax1, false);
      yAxis(g, Y, ax0, [0, 1, 2, 3], d3.format("~g"));
      g.append("line").attr("x1", ax0).attr("x2", ax1).attr("y1", base).attr("y2", base).attr("stroke", "#444").attr("stroke-width", 1.3);
      g.append("line").attr("x1", ax0).attr("x2", ax1).attr("y1", Y(0)).attr("y2", Y(0)).attr("stroke", "#555").attr("stroke-width", 1.5);
      title(g, ax0 - 30, top - 26, "a", "Held-out Energy-OOD");
      text(g, 22, (top + base) / 2, "gain over random (meV/atom)  →", { rotate: -90, anchor: "middle", size: 19 });
      const steps = { "HF-only": [], "LF→HF": [] };
      PROTOCOLS.forEach((p) => {
        SIGNALS.forEach((s, i) => {
          const v = A.gains[p][s], x = X(xs[p][i]);
          text(g, x, base + 30, s, { anchor: "middle", size: 20, color: "#444" });
          const dots = v.splits.map((sv, j) => g.append("circle").attr("cx", X(xs[p][i] + (j - 2) * 0.075))
            .attr("cy", Y(0)).attr("r", 4.5).attr("fill", "#9a9a9a").attr("stroke", "white").attr("stroke-width", 0.8).attr("opacity", 0));
          const ci = g.append("line").attr("x1", x).attr("x2", x).attr("y1", Y(0)).attr("y2", Y(0))
            .attr("stroke", PCOL[p]).attr("stroke-width", 3).attr("opacity", 0);
          const gm = g.append("g").attr("opacity", 0);
          const mk = marker(gm, s, x, Y(0), 17, PCOL[p]);
          const d = i * 160;
          steps[p].push((anim) => {
            dots.forEach((dot, j) => tr(dot, anim, 700, d + j * 40).attr("opacity", 1).attr("cy", Y(v.splits[j])));
            tr(gm, anim, 200, d).attr("opacity", 1);
            tr(mk.outer, anim, 800, d, d3.easeBackOut).attr("transform", `translate(${x},${Y(v.mean)})`);
            tr(ci, anim, 700, d + 300).attr("opacity", 1).attr("y1", Y(v.lo)).attr("y2", Y(v.hi));
          });
        });
        text(g, X(d3.mean(xs[p])), base + 66, p, { anchor: "middle", size: 22, weight: 700, color: PCOL[p] });
      });
      // (b) energy quantiles of the selected configurations, one panel per protocol
      const bx0 = 580, bx1 = 892, Q = A.quantiles, edges = Q.bins;
      const XB = d3.scaleLinear().domain([0.55, 1.0]).range([bx0, bx1]);
      const ph = 196, tops = [top, top + ph + 40];
      const stepPath = (Yb, vals) => {
        let d = `M${XB(edges[0])},${Yb(0)}`;
        vals.forEach((v, i) => { d += `L${XB(edges[i])},${Yb(v)}L${XB(edges[i + 1])},${Yb(v)}`; });
        return d + `L${XB(edges[edges.length - 1])},${Yb(0)}`;
      };
      const growPath = (sel, Yb, vals, anim, delay) => {
        if (!anim) { sel.interrupt().attr("d", stepPath(Yb, vals)).attr("opacity", 1); return; }
        sel.attr("opacity", 1).transition().delay(delay).duration(900).ease(d3.easeCubicOut)
          .attrTween("d", () => (t) => stepPath(Yb, vals.map((v) => v * t)));
      };
      const hist = [];
      PROTOCOLS.forEach((p, k) => {
        const t0 = tops[k], Yb = d3.scaleLinear().domain([0, 0.37]).range([t0 + ph, t0]);
        const gb = svg.append("g");
        grid(gb, Yb, [0, 0.1, 0.2, 0.3], bx0, bx1, false);
        yAxis(gb, Yb, bx0, [0, 0.1, 0.2, 0.3], d3.format(".1f"));
        xAxis(gb, XB, t0 + ph, k === 1 ? [0.6, 0.7, 0.8, 0.9, 1.0] : [], d3.format(".1f"));
        text(gb, bx0 - 58, t0 + ph / 2, "fraction", { rotate: -90, anchor: "middle", size: 18 });
        text(gb, bx0 + 10, t0 + 22, p, { size: 21, weight: 700, color: PCOL[p] });
        if (k === 0) title(gb, bx0 - 30, top - 26, "b", "Selected configurations");
        const rnd = gb.append("path").attr("fill", "#e9e9e7").attr("stroke", "#8c8c8c").attr("stroke-width", 1.8).attr("opacity", 0);
        const lines = SIGNALS.map((s) => gb.append("path").attr("fill", "none").attr("stroke", SCOL[s])
          .attr("stroke-width", 3).attr("stroke-dasharray", s === "EU" ? "7,4" : null).attr("opacity", 0));
        hist.push((anim) => {
          growPath(rnd, Yb, Q.random, anim, k * 300);
          SIGNALS.forEach((s, i) => growPath(lines[i], Yb, Q[p][s], anim, k * 300 + 300 + i * 150));
        });
      });
      text(svg, (bx0 + bx1) / 2, H - 52, "energy quantile within molecule", { anchor: "middle", size: 19 });
      const lg = legend(svg, [{ patch: true, color: "#d9d9d6", label: "random" }, { line: true, color: SCOL.AU, label: "AU" },
        { line: true, color: SCOL.EU, dash: "7,4", label: "EU" }, { line: true, color: SCOL.TU, label: "TU" }], 736, H - 16, 18, 14);
      lg.attr("opacity", 0);
      return [(anim) => steps["HF-only"].forEach((f) => f(anim)), (anim) => steps["LF→HF"].forEach((f) => f(anim)),
        (anim) => { hist.forEach((f) => f(anim)); tr(lg, anim, 500).attr("opacity", 1); }];
    },
  };

  // ------------------------------------------------------------------ schematics (synthetic)

  FIGS.metrics = {
    w: 1460, h: 470,
    build(svg) {
      const r = rng(7), n = 3000, err = [], unc = [];
      for (let i = 0; i < n; i++) {
        const sigma = Math.exp(0.6 * normal(r));
        err.push(Math.abs(normal(r) * sigma));
        unc.push(sigma * Math.exp(0.55 * normal(r)));
      }
      const meanErr = d3.mean(err), fr = d3.range(0, 0.985, 0.015);
      const curve = (key) => {
        const order = d3.range(n).sort((a, b) => key[a] - key[b]), pre = [0];
        order.forEach((i) => pre.push(pre[pre.length - 1] + err[i]));
        return fr.map((f) => { const k = n - Math.floor(f * n); return [f, pre[k] / k / meanErr]; });
      };
      const oracle = curve(err), byUnc = curve(unc);
      const top = 60, base = 390, lx0 = 130, lx1 = 650;
      const X = d3.scaleLinear().domain([0, 1]).range([lx0, lx1]), Y = d3.scaleLinear().domain([0, 1.05]).range([base, top]);
      const g = svg.append("g");
      grid(g, Y, [0, 0.25, 0.5, 0.75, 1], lx0, lx1, false);
      xAxis(g, X, base, [0, 0.25, 0.5, 0.75, 1]);
      yAxis(g, Y, lx0, [0, 0.25, 0.5, 0.75, 1]);
      title(g, lx0, top - 24, null, "Ranking: sparsification curve", 27);
      text(g, (lx0 + lx1) / 2, base + 64, "fraction of configurations removed", { anchor: "middle", size: 24 });
      text(g, 34, (top + base) / 2, "error of the rest", { rotate: -90, anchor: "middle", size: 24 });
      const p = sweepPanel(g, X, top, base);
      const area = p.data.append("path").datum(d3.zip(oracle, byUnc))
        .attr("d", d3.area().x((d) => X(d[0][0])).y0((d) => Y(d[0][1])).y1((d) => Y(d[1][1])))
        .attr("fill", SCOL.AU).attr("opacity", 0);
      p.data.append("path").datum(oracle).attr("d", d3.line().x((d) => X(d[0])).y((d) => Y(d[1])))
        .attr("fill", "none").attr("stroke", INK).attr("stroke-width", 3.6);
      const unl = p.data.append("path").datum(byUnc).attr("d", d3.line().x((d) => X(d[0])).y((d) => Y(d[1])))
        .attr("fill", "none").attr("stroke", SCOL.AU).attr("stroke-width", 3.6).attr("opacity", 0);
      const labs = [text(g, X(0.06), Y(0.12), "oracle: largest errors first", { size: 24 }).attr("opacity", 0),
        text(g, X(0.98), Y(0.9), "most uncertain first", { size: 24, color: SCOL.AU, anchor: "end" }).attr("opacity", 0),
        text(g, X(0.47), Y(0.55), "AUSE", { size: 32, weight: 800, color: "#1f4f8f", anchor: "middle" }).attr("opacity", 0)];
      const run = sweeper([p], null, null);
      const vp = unc.map((u) => (0.55 * u) ** 2), order = d3.range(n).sort((a, b) => vp[a] - vp[b]);
      const bins = d3.range(15).map((b) => order.slice(Math.floor((b * n) / 15), Math.floor(((b + 1) * n) / 15)));
      const pts = bins.map((b) => [d3.mean(b, (i) => vp[i]), d3.mean(b, (i) => err[i] ** 2)]);
      const lo = d3.min(pts.flat()) * 0.6, hi = d3.max(pts.flat()) * 1.6;
      const rx0 = 880, rx1 = 1360;
      const XR = d3.scaleLog().domain([lo, hi]).range([rx0, rx1]), YR = d3.scaleLog().domain([lo, hi]).range([base, top]);
      const gr = svg.append("g");
      gr.append("line").attr("x1", rx0).attr("x2", rx0).attr("y1", top).attr("y2", base).attr("stroke", "#444").attr("stroke-width", 1.3);
      gr.append("line").attr("x1", rx0).attr("x2", rx1).attr("y1", base).attr("y2", base).attr("stroke", "#444").attr("stroke-width", 1.3);
      title(gr, rx0, top - 24, null, "Calibration: reliability diagram", 27);
      text(gr, (rx0 + rx1) / 2, base + 40, "predicted variance (15 bins)", { anchor: "middle", size: 24 });
      text(gr, rx0 - 24, (top + base) / 2, "observed squared error", { rotate: -90, anchor: "middle", size: 24 });
      const shade = gr.append("path").attr("d", `M${rx0},${base}L${rx1},${top}L${rx0},${top}Z`).attr("fill", PCOL["LF→HF"]).attr("opacity", 0);
      const diag = gr.append("line").attr("x1", rx0).attr("y1", base).attr("x2", rx0).attr("y2", base)
        .attr("stroke", "#8a8a8a").attr("stroke-width", 2).attr("stroke-dasharray", "7,5");
      const rl = [text(gr, rx0 + 18, top + 34, "overconfident: error > predicted", { size: 24, color: PCOL["LF→HF"] }).attr("opacity", 0),
        text(gr, rx1 - 8, base - 16, "conservative", { size: 24, color: MUTED, anchor: "end" }).attr("opacity", 0)];
      const dots = pts.map((q) => gr.append("circle").attr("cx", XR(q[0])).attr("cy", YR(q[1])).attr("r", 0).attr("fill", SCOL.TU).attr("stroke", "white").attr("stroke-width", 1.4));
      return [
        (anim) => {
          run(0, 0.985, 1600, anim, 300);
          tr(labs[0], anim, 500, 900).attr("opacity", 1);
          tr(unl, anim, 600, 1700).attr("opacity", 1);
          tr(area, anim, 800, 2000).attr("opacity", 0.18);
          tr(labs[1], anim, 500, 1900).attr("opacity", 1);
          tr(labs[2], anim, 500, 2300).attr("opacity", 1);
        },
        (anim) => {
          tr(diag, anim, 900).attr("x2", rx1).attr("y2", top);
          tr(shade, anim, 700, 500).attr("opacity", 0.06);
          rl.forEach((t) => tr(t, anim, 500, 700).attr("opacity", 1));
          dots.forEach((d, i) => tr(d, anim, 400, 900 + i * 70, d3.easeBackOut).attr("r", 9));
        },
      ];
    },
  };

  FIGS.ensemble = {
    w: 920, h: 470,
    build(svg) {
      const mus = [-0.95, -0.35, 0.05, 0.55, 1.05], sds = [0.62, 0.78, 0.55, 0.7, 0.6];
      const xs = d3.range(-3.3, 3.41, 0.02), gauss = (m, s) => xs.map((x) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI)));
      const X = d3.scaleLinear().domain([-3.3, 3.4]).range([20, 900]), Y = d3.scaleLinear().domain([-0.17, 0.86]).range([420, 20]);
      const shades = ["#7aa6d8", "#5b8fcc", "#3f78bf", "#2f6fc0", "#24599c"];
      const g = svg.append("g");
      g.append("line").attr("x1", 20).attr("x2", 900).attr("y1", Y(-0.17)).attr("y2", Y(-0.17)).attr("stroke", "#444").attr("stroke-width", 1.3);
      text(g, 460, 462, "predicted energy of one structure", { anchor: "middle", size: 26 });
      const curvePath = (ys, f) => d3.line().x((d, i) => X(xs[i])).y((d) => Y(d * f))(ys);
      const areaPath = (ys, f) => d3.area().x((d, i) => X(xs[i])).y0(Y(0)).y1((d) => Y(d * f))(ys);
      const members = mus.map((m, k) => {
        const ys = gauss(m, sds[k]);
        const a = g.append("path").attr("fill", shades[k]).attr("opacity", 0.07).attr("d", areaPath(ys, 0));
        const l = g.append("path").attr("fill", "none").attr("stroke", shades[k]).attr("stroke-width", 3).attr("d", curvePath(ys, 0));
        return { ys, a, l };
      });
      const mixYs = xs.map((_, i) => d3.mean(members, (m) => m.ys[i]));
      const eu = g.append("g").attr("opacity", 0), au = g.append("g").attr("opacity", 0), tu = g.append("g").attr("opacity", 0);
      const mean = d3.mean(mus);
      eu.append("line").attr("x1", X(mean)).attr("x2", X(mean)).attr("y1", Y(0)).attr("y2", Y(0.78)).attr("stroke", INK).attr("stroke-width", 1.8).attr("stroke-dasharray", "6,5");
      text(eu, X(mean), Y(0.8), "mean Ē", { anchor: "middle", size: 26 });
      mus.forEach((m) => eu.append("circle").attr("cx", X(m)).attr("cy", Y(-0.06)).attr("r", 8).attr("fill", SCOL.EU).attr("stroke", "white").attr("stroke-width", 1.2));
      const arrEU = arrowDefs(svg, SCOL.EU), arrAU = arrowDefs(svg, "#1f4f8f");
      eu.append("line").attr("x1", X(mus[0])).attr("x2", X(mus[4])).attr("y1", Y(-0.12)).attr("y2", Y(-0.12)).attr("stroke", SCOL.EU)
        .attr("stroke-width", 2.6).attr("marker-start", arrEU).attr("marker-end", arrEU);
      text(eu, X(mus[4]) + 22, Y(-0.09), "EU", { size: 36, weight: 800, color: "#c0700f", baseline: "central" });
      const yAU = d3.max(gauss(0.05, 0.55)) * 0.6065;
      au.append("line").attr("x1", X(0.05 - 0.55)).attr("x2", X(0.05 + 0.55)).attr("y1", Y(yAU)).attr("y2", Y(yAU))
        .attr("stroke", "#1f4f8f").attr("stroke-width", 2.6).attr("marker-start", arrAU).attr("marker-end", arrAU);
      text(au, X(-0.6), Y(yAU), "AU", { size: 36, weight: 800, color: "#1f4f8f", anchor: "end", baseline: "central" });
      const mixL = tu.append("path").attr("fill", "none").attr("stroke", SCOL.TU).attr("stroke-width", 5).attr("d", curvePath(mixYs, 0));
      const mixA = tu.append("path").attr("fill", SCOL.TU).attr("opacity", 0.1).attr("d", areaPath(mixYs, 0));
      text(tu, X(2.0), Y(d3.max(mixYs) * 0.55), "TU", { size: 36, weight: 800, color: SCOL.TU, baseline: "central" });
      const growCurve = (sel, ys, path, anim, delay) => {
        if (!anim) { sel.interrupt().attr("d", path(ys, 1)); return; }
        sel.transition().delay(delay).duration(900).ease(d3.easeCubicOut).attrTween("d", () => (t) => path(ys, t));
      };
      return [
        (anim) => members.forEach((m, k) => { growCurve(m.l, m.ys, curvePath, anim, 300 + k * 220); growCurve(m.a, m.ys, areaPath, anim, 300 + k * 220); }),
        (anim) => tr(eu, anim, 600).attr("opacity", 1),
        (anim) => tr(au, anim, 600).attr("opacity", 1),
        (anim) => { tr(tu, anim, 200).attr("opacity", 1); growCurve(mixL, mixYs, curvePath, anim, 0); growCurve(mixA, mixYs, areaPath, anim, 0); },
      ];
    },
  };

  FIGS.mechanism = {
    w: 860, h: 620,
    build(svg) {
      const t = d3.range(0, 100.1, 0.5);
      const train = t.map((x) => 0.12 + 0.88 * Math.exp(-x / 22)), test = t.map((x) => 0.72 + 0.28 * Math.exp(-x / 10));
      const au = t.map((x) => 0.14 + 0.86 * Math.exp(-x / 24));
      const X = d3.scaleLinear().domain([0, 100]).range([60, 840]), Y = d3.scaleLinear().domain([0, 1.05]).range([560, 20]);
      const g = svg.append("g");
      g.append("line").attr("x1", 60).attr("x2", 60).attr("y1", 20).attr("y2", 560).attr("stroke", "#444").attr("stroke-width", 1.3);
      g.append("line").attr("x1", 60).attr("x2", 840).attr("y1", 560).attr("y2", 560).attr("stroke", "#444").attr("stroke-width", 1.3);
      text(g, 450, 604, "training on accurate labels →", { anchor: "middle", size: 26 });
      text(g, 32, 290, "size", { rotate: -90, anchor: "middle", size: 26 });
      const line = (ys) => d3.line().x((d, i) => X(t[i])).y((d) => Y(d))(ys);
      const pa = sweepPanel(g, X, 20, 560), pb = sweepPanel(g, X, 20, 560);
      pa.data.append("path").attr("d", line(train)).attr("fill", "none").attr("stroke", "#555").attr("stroke-width", 4);
      pa.data.append("path").attr("d", line(au)).attr("fill", "none").attr("stroke", SCOL.AU).attr("stroke-width", 4).attr("stroke-dasharray", DASH.AU);
      text(pa.data, 836, Y(0.07), "error², training data", { anchor: "end", size: 25, color: "#555" });
      text(pa.data, 836, Y(0.25), "AU (predicted variance)", { anchor: "end", size: 25, color: SCOL.AU });
      const gap = g.append("path").attr("d", d3.area().x((d, i) => X(t[i])).y0((d, i) => Y(au[i])).y1((d) => Y(d))(test))
        .attr("fill", PCOL["LF→HF"]).attr("opacity", 0);
      pb.data.append("path").attr("d", line(test)).attr("fill", "none").attr("stroke", PCOL["LF→HF"]).attr("stroke-width", 4);
      text(pb.data, 836, Y(0.77), "error², unseen data", { anchor: "end", size: 25, color: PCOL["LF→HF"] });
      const gl = text(g, X(62), Y(0.47), "error bars too small", { anchor: "middle", size: 28, weight: 700, color: PCOL["LF→HF"] }).attr("opacity", 0);
      const ra = sweeper([pa], null, null), rb = sweeper([pb], null, null);
      return [(anim) => ra(0, 100, 2200, anim, 400), (anim) => rb(0, 100, 2200, anim),
        (anim) => { tr(gap, anim, 700).attr("opacity", 0.13); tr(gl, anim, 700, 200).attr("opacity", 1); }];
    },
  };

  // ------------------------------------------------------------------ controller

  class Figure {
    constructor(el) {
      this.el = el;
      this.spec = FIGS[el.dataset.fig];
      if (!this.spec) throw new Error(`unknown figure ${el.dataset.fig}`);
      this.n = -1;
    }
    rebuild() {
      d3.select(this.el).selectAll("*").interrupt();
      this.el.innerHTML = "";
      const svg = d3.select(this.el).append("svg").attr("viewBox", `0 0 ${this.spec.w} ${this.spec.h}`)
        .attr("role", "img").attr("aria-label", this.el.dataset.fig);
      this.steps = this.spec.build(svg);
    }
    /* Show step n: animate only when advancing by one, otherwise jump there. */
    goto(n, anim) {
      n = Math.min(n, (this.steps ? this.steps.length : 99) - 1);
      if (this.steps && n === this.n && n > 0) return;
      if (anim && this.steps && n === this.n + 1) this.steps[n](true);
      else {
        this.rebuild();
        n = Math.min(n, this.steps.length - 1);
        if (anim && n === 0) this.steps[0](true);
        else for (let k = 0; k <= n; k++) this.steps[k](false);
      }
      this.n = n;
    }
    replay() {
      const n = Math.max(this.n, 0);
      if (n === 0) { this.n = -1; this.goto(0, true); return; }
      this.goto(n - 1, false);
      this.goto(n, true);
    }
    last() { this.goto(99, false); }
  }

  window.UQFigures = { Figure, FIGS };
})();

// Lightweight SVG charts (no extra charting library).
// All values come from the API: fitted Weibull parameters, the model's
// prediction response, or real observed rainfall. No synthetic data.

const NS = "http://www.w3.org/2000/svg";
const RAIN_THRESHOLD_MM = 20; // matches RAIN_THRESHOLD in app.py

const fmt = (value, digits = 2) => (value == null || Number.isNaN(value) ? "—" : Number(value).toFixed(digits));

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function setSvg(svg, width, height, body) {
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = body;
}

/** Weibull density and survival with the fitted (shape k, scale λ). */
export const weibullPdf = (x, k, lambda) =>
  x < 0 ? 0 : (k / lambda) * (x / lambda) ** (k - 1) * Math.exp(-((x / lambda) ** k));
export const weibullSf = (x, k, lambda) => Math.exp(-((x / lambda) ** k));

export function renderWeibull(svg, readoutEl, info) {
  if (!info || !info.weibull_shape || !info.weibull_scale) {
    setSvg(svg, 520, 260, emptyText(520, 260, "Fitted Weibull parameters unavailable."));
    return;
  }
  const { weibull_shape: k, weibull_scale: lambda } = info;
  const W = 520, H = 260, m = { l: 52, r: 16, t: 16, b: 40 };
  const xMax = 80;
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = [];
  for (let x = 0.25; x <= xMax; x += 0.5) xs.push(x);
  const ys = xs.map((x) => weibullPdf(x, k, lambda));
  const yMax = Math.max(...ys) * 1.08;
  const X = (x) => m.l + (x / xMax) * iw;
  const Y = (y) => m.t + ih - (y / yMax) * ih;

  const linePath = xs.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${X(xMax).toFixed(1)},${Y(0).toFixed(1)} L${X(0.25).toFixed(1)},${Y(0).toFixed(1)} Z`;

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const yv = (yMax * i) / 4;
    grid += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(yv)}" y2="${Y(yv)}"/>`;
    grid += `<text x="${m.l - 8}" y="${Y(yv) + 4}" text-anchor="end">${yv.toFixed(3)}</text>`;
  }
  let xTicks = "";
  for (let x = 0; x <= xMax; x += 20) {
    xTicks += `<text x="${X(x)}" y="${H - m.b + 16}" text-anchor="middle">${x}</text>`;
  }
  let thresholds = "";
  for (const t of [10, 20, 30, 50]) {
    thresholds += `<line class="threshold" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${m.t + ih}"/>`;
    thresholds += `<text x="${X(t) + 3}" y="${m.t + 12}" fill="#ffd166">${t} mm</text>`;
  }
  const gid = "wbGrad";
  const body = `
    <defs>
      <linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#51c8ff" stop-opacity=".55"/>
        <stop offset="100%" stop-color="#51c8ff" stop-opacity="0"/>
      </linearGradient>
    </defs>
    ${grid}
    <line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/>
    <line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>
    ${thresholds}
    <path d="${areaPath}" fill="url(#${gid})"/>
    <path class="curve" d="${linePath}"/>
    <g id="wbCursor" style="display:none">
      <line class="axis" id="wbCursorLine" y1="${m.t}" y2="${m.t + ih}"/>
      <circle id="wbCursorDot" r="4" fill="#5ef0ff"/>
    </g>
    ${xTicks}
    <text x="${m.l + iw / 2}" y="${H - 6}" text-anchor="middle">Daily rainfall on a wet day (mm)</text>
    <text transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">Density (per mm)</text>
    <rect id="wbHit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>
  `;
  setSvg(svg, W, H, body);

  const meanWet = info.wet_day_mean;
  readoutEl.innerHTML = `Shape k = <b>${fmt(k, 4)}</b> · scale λ = <b>${fmt(lambda, 3)}</b> mm` +
    (meanWet != null ? ` · wet-day mean = <b>${fmt(meanWet, 2)} mm</b>` : "") +
    `<br><span>Hover over the curve to read values.</span>`;

  const hit = svg.querySelector("#wbHit");
  const cursor = svg.querySelector("#wbCursor");
  const cLine = svg.querySelector("#wbCursorLine");
  const cDot = svg.querySelector("#wbCursorDot");
  const onMove = (event) => {
    const rect = svg.getBoundingClientRect();
    const scaleX = W / rect.width;
    const px = (event.clientX - rect.left) * scaleX;
    const x = Math.max(0.25, Math.min(xMax, ((px - m.l) / iw) * xMax));
    const y = weibullPdf(x, k, lambda);
    const sx = X(x), sy = Y(y);
    cursor.style.display = "";
    cLine.setAttribute("x1", sx); cLine.setAttribute("x2", sx);
    cDot.setAttribute("cx", sx); cDot.setAttribute("cy", sy);
    const exceed = weibullSf(x, k, lambda) * 100;
    readoutEl.innerHTML =
      `Rainfall <b>${x.toFixed(1)} mm</b> · density <b>${y.toFixed(4)}</b> per mm · ` +
      `P(X &gt; ${x.toFixed(0)} mm | wet day) = <b>${exceed.toFixed(2)}%</b>`;
  };
  hit.addEventListener("mousemove", onMove);
  hit.addEventListener("touchmove", (e) => onMove(e.touches[0]), { passive: true });
  hit.addEventListener("mouseleave", () => { cursor.style.display = "none"; });
}

export function renderExceedance(svg, prediction) {
  const W = 520, H = 260, m = { l: 48, r: 16, t: 20, b: 40 };
  if (!prediction) {
    setSvg(svg, W, H, emptyText(W, H, "Run a prediction to see threshold probabilities."));
    return;
  }
  const items = [
    { label: "> 10 mm", value: prediction.above_10 },
    { label: "> 20 mm", value: prediction.above_20 },
    { label: "> 30 mm", value: prediction.above_30 },
    { label: "> 50 mm", value: prediction.above_50 },
  ];
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const top = Math.max(1, Math.ceil(Math.max(...items.map((i) => i.value)) * 1.3));
  const Y = (v) => m.t + ih - (v / top) * ih;
  const slot = iw / items.length;
  const barW = Math.min(64, slot * 0.55);
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = (top * i) / 4;
    grid += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
    grid += `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(v < 10 ? 1 : 0)}%</text>`;
  }
  const bars = items.map((item, i) => {
    const x = m.l + slot * i + (slot - barW) / 2;
    const y = Y(item.value);
    return `
      <g>
        <rect class="bar-rect" x="${x}" y="${y}" width="${barW}" height="${m.t + ih - y}" rx="6" fill="url(#exGrad)">
          <title>P(rainfall ${esc(item.label)}) = ${fmt(item.value)}%</title>
        </rect>
        <text x="${x + barW / 2}" y="${y - 6}" text-anchor="middle" fill="#eaf5ff">${fmt(item.value, 1)}%</text>
        <text x="${x + barW / 2}" y="${H - m.b + 18}" text-anchor="middle">${esc(item.label)}</text>
      </g>`;
  }).join("");
  const body = `
    <defs>
      <linearGradient id="exGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#7fe7ff"/>
        <stop offset="100%" stop-color="#6b7cff"/>
      </linearGradient>
    </defs>
    ${grid}
    <line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>
    ${bars}
    <text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">Unconditional probability on the forecast day</text>
  `;
  setSvg(svg, W, H, body);
}

export function renderHistory(svg, subEl, payload, { districtLabel } = {}) {
  const W = 900, H = 240, m = { l: 48, r: 16, t: 14, b: 34 };
  if (!payload) {
    subEl.textContent = "Select a district to load its observed IMD rainfall.";
    setSvg(svg, W, H, emptyText(W, H, "No district selected."));
    return;
  }
  const obs = payload.observations || [];
  const values = obs.map((o) => o.rainfall_mm).filter((v) => v != null);
  const missing = obs.length - values.length;
  subEl.textContent =
    `${districtLabel || payload.district}: ${obs.length} observed days` +
    (obs.length ? ` (${obs[0].date} → ${obs[obs.length - 1].date})` : "") +
    (missing ? ` · ${missing} day(s) with no recorded rainfall are left blank` : "") +
    ". Archived IMD observations, not live.";

  if (!obs.length || !values.length) {
    setSvg(svg, W, H, emptyText(W, H, "No valid rainfall observations for this district."));
    return;
  }

  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const top = Math.max(RAIN_THRESHOLD_MM, Math.ceil(Math.max(...values) / 10) * 10);
  const Y = (v) => m.t + ih - (v / top) * ih;
  const slot = iw / obs.length;
  const barW = Math.max(2, slot * 0.7);

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = (top * i) / 4;
    grid += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
    grid += `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(0)}</text>`;
  }
  const bars = obs.map((o, i) => {
    const x = m.l + slot * i + (slot - barW) / 2;
    if (o.rainfall_mm == null) {
      return `<rect x="${x}" y="${m.t + ih - 2}" width="${barW}" height="2" fill="rgba(255,255,255,.2)">
                <title>${esc(o.date)}: no record</title></rect>`;
    }
    const y = Y(o.rainfall_mm);
    return `<rect class="${o.rainfall_mm > 0 ? "bar-wet" : "bar-dry"}" x="${x}" y="${y}"
              width="${barW}" height="${Math.max(0, m.t + ih - y)}" rx="2">
              <title>${esc(o.date)}: ${fmt(o.rainfall_mm, 1)} mm</title></rect>`;
  }).join("");
  const first = obs[0].date, last = obs[obs.length - 1].date;
  const threshold = `
    <line class="threshold" x1="${m.l}" x2="${W - m.r}" y1="${Y(RAIN_THRESHOLD_MM)}" y2="${Y(RAIN_THRESHOLD_MM)}"/>
    <text x="${W - m.r - 4}" y="${Y(RAIN_THRESHOLD_MM) - 5}" text-anchor="end" fill="#ffd166">20 mm</text>`;
  const body = `
    ${grid}
    <line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/>
    ${threshold}
    ${bars}
    <text x="${m.l}" y="${H - 8}">${esc(first)}</text>
    <text x="${W - m.r}" y="${H - 8}" text-anchor="end">${esc(last)}</text>
    <text x="${m.l + iw / 2}" y="${H - 8}" text-anchor="middle">Daily rainfall (mm)</text>
  `;
  setSvg(svg, W, H, body);
}

function emptyText(w, h, text) {
  return `<text class="empty-text" x="${w / 2}" y="${h / 2}" text-anchor="middle">${esc(text)}</text>`;
}

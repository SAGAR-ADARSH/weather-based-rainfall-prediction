// RainPredict front-end controller.
// Flow: select district (search or map) → Predict Next Day → /api/predict →
// forecast card, probability ring, weather layer and charts update from the
// returned values. Nothing is estimated in the browser.

import { createDistrictMap, keyOf } from "./district-map.js";
import { WeatherFX } from "./weather-fx.js";
import { renderWeibull, renderExceedance, renderHistory } from "./charts.js";

const $ = (id) => document.getElementById(id);
const RING_CIRCUMFERENCE = 2 * Math.PI * 52;
const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
const reducedMotion = () => reducedMotionQuery.matches;

const state = {
  districts: [],
  byKey: new Map(),
  selected: null,        // {key, district, state}
  prediction: null,      // last successful /api/predict response for `selected`
  requestSeq: 0,         // invalidates in-flight predictions on selection change
  historySeq: 0,
  info: null,
  matches: [],
  activeIndex: -1,
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

async function fetchJSON(url, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The request timed out. Please try again.");
    throw new Error("Could not reach the prediction service. Check that the server is running.");
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) {
    throw new Error((data && data.error) || `Request failed (HTTP ${response.status}).`);
  }
  if (!data) throw new Error("The server returned an empty response.");
  return data;
}

const fmt = (value, digits = 2) =>
  value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);

const labelOf = (d) => `${d.district}, ${d.state}`;

function setText(id, text) { $(id).textContent = text; }

function showError(message) {
  const box = $("errorBox");
  box.textContent = message;
  box.classList.remove("hidden");
}

function hideError() {
  $("errorBox").classList.add("hidden");
}

function setMapStatus(message, isError = false) {
  const el = $("mapStatus");
  if (!message) {
    el.classList.add("hidden");
    return;
  }
  el.textContent = message;
  el.classList.toggle("is-error", isError);
  el.classList.remove("hidden");
}

function showCard(which) {
  for (const id of ["fcEmpty", "fcLoading", "fcError", "fcResult"]) {
    $(id).classList.toggle("hidden", id !== which);
  }
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

const map = createDistrictMap("map", {
  onSelect: (key) => {
    const d = state.byKey.get(key);
    if (d) chooseDistrict(d);
  },
  reducedMotion,
});

// ---------------------------------------------------------------------------
// Weather layer
// ---------------------------------------------------------------------------

const fx = new WeatherFX({
  frame: $("mapFrame"),
  canvas: $("rainCanvas"),
  cloudField: $("cloudField"),
  tint: $("rainTint"),
  legendItems: [...document.querySelectorAll("#tierLegend li")],
  reducedMotion,
});

// ---------------------------------------------------------------------------
// District search (accessible combobox)
// ---------------------------------------------------------------------------

const searchInput = $("districtSearch");
const suggestions = $("districtList");

function openSuggestions(open) {
  suggestions.classList.toggle("hidden", !open);
  searchInput.setAttribute("aria-expanded", String(open));
  if (!open) {
    searchInput.removeAttribute("aria-activedescendant");
    state.activeIndex = -1;
  }
}

function renderSuggestions(query) {
  const q = query.trim().toLowerCase();
  if (!q) {
    state.matches = [];
    openSuggestions(false);
    return;
  }
  const scored = state.districts
    .filter((d) => d.district.toLowerCase().includes(q) || d.state.toLowerCase().includes(q))
    .map((d) => ({ d, rank: d.district.toLowerCase().startsWith(q) ? 0 : 1 }))
    .sort((a, b) => a.rank - b.rank || a.d.district.localeCompare(b.d.district))
    .slice(0, 12)
    .map((x) => x.d);

  state.matches = scored;
  state.activeIndex = scored.length ? 0 : -1;
  suggestions.innerHTML = "";

  if (!scored.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No district matches. Try another spelling.";
    suggestions.appendChild(li);
  }

  scored.forEach((d, index) => {
    const li = document.createElement("li");
    li.id = `opt-${index}`;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", String(index === state.activeIndex));
    const name = document.createElement("span");
    name.textContent = d.district;
    const st = document.createElement("small");
    st.textContent = d.state;
    li.append(name, st);
    li.addEventListener("mousedown", (event) => {
      event.preventDefault(); // keep focus in the input
      chooseDistrict(d);
    });
    suggestions.appendChild(li);
  });
  openSuggestions(true);
  updateActiveOption();
}

function updateActiveOption() {
  const items = [...suggestions.querySelectorAll("li[role='option']")];
  items.forEach((li, i) => {
    const active = i === state.activeIndex;
    li.setAttribute("aria-selected", String(active));
    if (active) {
      searchInput.setAttribute("aria-activedescendant", li.id);
      li.scrollIntoView({ block: "nearest" });
    }
  });
}

function findExactMatch(text) {
  const t = text.trim().toLowerCase();
  if (!t) return null;
  const exact = state.districts.filter(
    (d) => labelOf(d).toLowerCase() === t || d.district.toLowerCase() === t
  );
  return exact.length === 1 ? exact[0] : null;
}

searchInput.addEventListener("input", () => {
  // Typing invalidates a previous pick until the user chooses again.
  renderSuggestions(searchInput.value);
});

searchInput.addEventListener("keydown", (event) => {
  const open = !suggestions.classList.contains("hidden");
  if (event.key === "ArrowDown" && open && state.matches.length) {
    event.preventDefault();
    state.activeIndex = (state.activeIndex + 1) % state.matches.length;
    updateActiveOption();
  } else if (event.key === "ArrowUp" && open && state.matches.length) {
    event.preventDefault();
    state.activeIndex = (state.activeIndex - 1 + state.matches.length) % state.matches.length;
    updateActiveOption();
  } else if (event.key === "Enter") {
    event.preventDefault();
    if (open && state.activeIndex >= 0 && state.matches[state.activeIndex]) {
      chooseDistrict(state.matches[state.activeIndex]);
    } else {
      const exact = findExactMatch(searchInput.value);
      if (exact) chooseDistrict(exact);
      else showError("Please choose a district from the suggestions.");
    }
  } else if (event.key === "Escape") {
    openSuggestions(false);
  }
});

searchInput.addEventListener("blur", () => setTimeout(() => openSuggestions(false), 120));

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

function updateSelectionChip(d, result) {
  $("selectionChip").classList.remove("is-empty");
  setText("selectionName", d.district);
  let meta = `${d.state} · `;
  if (result && result.hasBoundary) meta += "boundary outlined";
  else if (result && result.missing) meta += "location not available";
  else meta += "coordinate marker (no boundary polygon)";
  setText("selectionMeta", meta);
}

function clearSelectionChip() {
  $("selectionChip").classList.add("is-empty");
  setText("selectionName", "No district selected");
  setText("selectionMeta", "Click a district or use search");
}

function chooseDistrict(d) {
  if (state.selected && state.selected.key === d.key) {
    // Same district: just re-centre the camera.
    map.selectDistrict(d.key);
    return;
  }
  hideError();
  state.selected = d;
  state.requestSeq += 1; // ignore any in-flight prediction for the previous district
  searchInput.value = labelOf(d);
  openSuggestions(false);

  clearForecast();
  const result = map.selectDistrict(d.key);
  updateSelectionChip(d, result);
  $("predictBtn").disabled = false;
  loadHistory(d);
}

function clearForecast() {
  state.prediction = null;
  fx.setProbability(null);
  renderExceedance($("chartExceed"), null);
  $("forecastCard").classList.remove("is-loading");
  setRing(0, false);
  showCard("fcEmpty");
}

function resetToIndia() {
  state.requestSeq += 1;
  state.historySeq += 1;
  state.selected = null;
  searchInput.value = "";
  openSuggestions(false);
  map.resetView();
  clearSelectionChip();
  clearForecast();
  $("predictBtn").disabled = true;
  renderHistory($("chartHistory"), $("historySub"), null);
  hideError();
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

async function predict() {
  if (!state.selected) {
    showError("Select a district first.");
    return;
  }
  hideError();
  const d = state.selected;
  const seq = ++state.requestSeq;
  setPredictBusy(true);
  $("forecastCard").classList.add("is-loading");
  showCard("fcLoading");
  setText("fcLoadingText", `Running the logistic regression and Weibull model for ${d.district}…`);
  map.setPulse(d.key, true);

  try {
    const data = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ district: d.district, state: d.state }),
    });
    if (seq !== state.requestSeq) return; // selection changed meanwhile
    if (!data.prediction || typeof data.prediction.rain_probability !== "number") {
      throw new Error("The prediction response was incomplete. Please try again.");
    }
    state.prediction = data;
    renderForecast(data);
    fx.setProbability(data.prediction.rain_probability);
    renderExceedance($("chartExceed"), data.prediction);
  } catch (error) {
    if (seq !== state.requestSeq) return;
    state.prediction = null;
    fx.setProbability(null);
    renderExceedance($("chartExceed"), null);
    setText("fcErrorText", error.message);
    showCard("fcError");
    showError(error.message);
  } finally {
    if (seq === state.requestSeq) {
      setPredictBusy(false);
      $("forecastCard").classList.remove("is-loading");
      map.setPulse(d.key, false);
    }
  }
}

function setPredictBusy(busy) {
  $("predictBtn").disabled = busy || !state.selected;
  $("spinner").classList.toggle("hidden", !busy);
  setText("btnText", busy ? "Calculating…" : "Predict Next Day");
}

function categoryClass(rainPercent) {
  if (rainPercent >= 60) return "high";
  if (rainPercent >= 30) return "medium";
  return "low";
}

function renderForecast(data) {
  const p = data.prediction;
  const w = data.weather;
  showCard("fcResult");

  setText("fcTitle", `${data.district}`);
  const cat = $("fcCategory");
  cat.textContent = p.category;
  cat.className = `category ${categoryClass(p.rain_probability)}`;

  setRing(p.rain_probability, true, categoryClass(p.rain_probability));
  setText("fcExpected", fmt(p.expected_rainfall, 2));
  setText("fcWet", fmt(p.expected_if_wet, 2));

  for (const [key, value] of [["10", p.above_10], ["20", p.above_20], ["30", p.above_30], ["50", p.above_50]]) {
    setText(`fcAbove${key}`, `${fmt(value, 2)}%`);
    // Bar length is the probability itself on a 0–100% scale (minimum 1px so tiny values stay visible).
    $(`barAbove${key}`).style.width = `${Math.max(0.6, Math.min(100, value))}%`;
  }

  setText("fcTemp", `${fmt(w.temperature, 1)} °C`);
  setText("fcHumidity", `${fmt(w.humidity, 1)} %`);
  setText("fcPressure", `${fmt(w.pressure, 2)} kPa`);
  setText("fcWind", `${fmt(w.wind, 1)} m/s`);
  setText("fcToday", `${fmt(w.today_rainfall, 1)} mm`);

  setText("fcObsDate", data.latest_valid_date);
  setText("fcForecastDate", data.prediction_date);
  const coverage = state.info
    ? ` Dataset covers ${state.info.date_start} to ${state.info.date_end}.`
    : "";
  setText("fcArchiveNote",
    `Latest valid observation from the archived dataset, not a live feed.${coverage}`);
  setText("fcHeavy", `${fmt(p.heavy_rain_probability, 1)}%`);
}

// ---------------------------------------------------------------------------
// Probability ring (animated)
// ---------------------------------------------------------------------------

let ringFrame = null;
function setRing(percent, animate, cls = "low") {
  const ring = $("ringFg");
  const colors = { low: "#53e0a2", medium: "#ffd166", high: "#ff758f" };
  ring.style.stroke = colors[cls] || colors.low;
  const clamped = Math.max(0, Math.min(100, percent || 0));
  ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - clamped / 100));

  if (ringFrame) cancelAnimationFrame(ringFrame);
  const label = $("fcRain");
  if (!animate || reducedMotion()) {
    label.textContent = `${clamped.toFixed(1)}%`;
    return;
  }
  const start = performance.now();
  const duration = 1100;
  const from = 0;
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    label.textContent = `${(from + (clamped - from) * eased).toFixed(1)}%`;
    if (t < 1) ringFrame = requestAnimationFrame(step);
  };
  ringFrame = requestAnimationFrame(step);
}

// ---------------------------------------------------------------------------
// Historical observations (real data only)
// ---------------------------------------------------------------------------

async function loadHistory(d) {
  const seq = ++state.historySeq;
  setText("historySub", `Loading observed rainfall for ${d.district}…`);
  try {
    const params = new URLSearchParams({ district: d.district, state: d.state, days: "60" });
    const data = await fetchJSON(`/api/history?${params}`);
    if (seq !== state.historySeq) return;
    renderHistory($("chartHistory"), $("historySub"), data, { districtLabel: d.district });
  } catch (error) {
    if (seq !== state.historySeq) return;
    renderHistory($("chartHistory"), $("historySub"), null);
    setText("historySub", `Observed rainfall unavailable: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function loadInfo() {
  try {
    const info = await fetchJSON("/api/info");
    state.info = info;
    const stats = document.querySelectorAll("#datasetStats div b");
    stats[0].textContent = info.districts;
    stats[1].textContent = info.states;
    stats[2].textContent = Number(info.rows).toLocaleString();
    setText("datasetCoverage", `Archived data: ${info.date_start} → ${info.date_end}`);
    renderWeibull($("chartWeibull"), $("weibullReadout"), info);
  } catch (error) {
    console.error(error);
    renderWeibull($("chartWeibull"), $("weibullReadout"), null);
    setText("weibullReadout", "Fitted distribution parameters could not be loaded.");
  }
}

async function loadDistricts() {
  try {
    const data = await fetchJSON("/api/districts");
    state.districts = data.districts || [];
    state.byKey.clear();
    for (const d of state.districts) {
      const entry = { ...d, key: keyOf(d.state, d.district) };
      state.byKey.set(entry.key, entry);
    }
    state.districts = [...state.byKey.values()];
    return true;
  } catch (error) {
    showError(`District list could not be loaded: ${error.message}`);
    return false;
  }
}

async function init() {
  // Responsive legend: open on wide screens, collapsed on phones.
  $("legend").open = window.matchMedia("(min-width: 641px)").matches;

  $("predictBtn").addEventListener("click", predict);
  $("resetViewBtn").addEventListener("click", resetToIndia);

  // Pause decorative animation when the map is off-screen or the tab is hidden.
  const onScreen = new IntersectionObserver(
    ([entry]) => fx.setVisible(entry.isIntersecting),
    { threshold: 0.05 }
  );
  onScreen.observe($("mapFrame"));
  document.addEventListener("visibilitychange", () => fx.setVisible(!document.hidden));

  renderHistory($("chartHistory"), $("historySub"), null);
  clearSelectionChip();
  clearForecast();

  loadInfo();

  setMapStatus("Loading district boundaries…");
  const [listOk, boundaries] = await Promise.all([loadDistricts(), map.loadBoundaries()]);

  if (listOk) {
    map.setDistricts(state.districts);
  }
  if (!boundaries.ok) {
    setMapStatus("District boundaries could not be loaded. Coordinate markers are shown instead.", true);
  } else {
    setMapStatus(null);
  }
  if (listOk && !state.districts.length) {
    setMapStatus("No districts are available in the dataset.", true);
  }
  searchInput.disabled = !listOk;
  if (!listOk) searchInput.placeholder = "District list unavailable";
}

init();

/* global L */
// Interactive district map (Leaflet + OpenStreetMap tiles + district GeoJSON).
// Boundaries are loaded once from a local, simplified GeoJSON file.
// Districts without a matched polygon are shown as coordinate markers only.

const INDIA_BOUNDS = L.latLngBounds([6.0, 67.5], [37.6, 97.6]);
const GEOJSON_URL = "/static/data/india_districts.geojson";
const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

export const keyOf = (state, district) => `${state}|${district}`;

const BASE_STYLE = {
  className: "district-path",
  weight: 0.8,
};

export function createDistrictMap(containerId, { onSelect, reducedMotion }) {
  const map = L.map(containerId, {
    zoomControl: false,
    minZoom: 4,
    maxZoom: 12,
    zoomSnap: 0.25,
    worldCopyJump: false,
    maxBounds: INDIA_BOUNDS.pad(0.6),
    maxBoundsViscosity: 0.7,
    attributionControl: true,
    preferCanvas: false,
  });

  L.control.zoom({ position: "topleft", zoomInTitle: "Zoom in", zoomOutTitle: "Zoom out" }).addTo(map);

  L.tileLayer(TILE_URL, {
    maxZoom: 12,
    className: "osm-tiles",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  const polygonByKey = new Map();
  const pointByKey = new Map();
  let selectedKey = null;
  let pulseKey = null;
  let districtLayer = null;

  map.fitBounds(INDIA_BOUNDS, { animate: false });

  const elementOf = (layer) => (layer && layer.getElement ? layer.getElement() : null);

  function setHover(key, on) {
    const layer = polygonByKey.get(key);
    const el = elementOf(layer);
    if (el) el.classList.toggle("is-hover", on);
  }

  function markSelected(key, on) {
    const polygon = polygonByKey.get(key);
    if (polygon) {
      const el = elementOf(polygon);
      if (el) el.classList.toggle("district-selected", on);
    }
    const point = pointByKey.get(key);
    if (point) {
      const el = elementOf(point);
      if (el) el.classList.toggle("is-selected", on);
    }
  }

  function camera(bounds, options = {}) {
    const padding = options.padding || [48, 48];
    const maxZoom = options.maxZoom || 9;
    if (reducedMotion()) {
      map.fitBounds(bounds, { padding, maxZoom, animate: false });
    } else {
      map.flyToBounds(bounds, { padding, maxZoom, duration: 1.1, easeLinearity: 0.25 });
    }
  }

  function resetView() {
    if (selectedKey) markSelected(selectedKey, false);
    selectedKey = null;
    if (reducedMotion()) {
      map.fitBounds(INDIA_BOUNDS, { animate: false });
    } else {
      map.flyToBounds(INDIA_BOUNDS, { duration: 1.0, padding: [12, 12] });
    }
  }

  function selectDistrict(key) {
    if (selectedKey && selectedKey !== key) markSelected(selectedKey, false);
    selectedKey = key;
    markSelected(key, true);

    const polygon = polygonByKey.get(key);
    if (polygon) {
      camera(polygon.getBounds(), { maxZoom: 9 });
      return { hasBoundary: true };
    }
    const point = pointByKey.get(key);
    if (point) {
      const latlng = point.getLatLng();
      if (reducedMotion()) map.setView(latlng, 8, { animate: false });
      else map.flyTo(latlng, 8, { duration: 1.1 });
      return { hasBoundary: false };
    }
    return { hasBoundary: false, missing: true };
  }

  function setPulse(key, on) {
    if (pulseKey && pulseKey !== key) {
      const prev = elementOf(polygonByKey.get(pulseKey));
      if (prev) prev.classList.remove("is-loading");
    }
    pulseKey = on ? key : null;
    const el = elementOf(polygonByKey.get(key));
    if (el) el.classList.toggle("is-loading", on);
  }

  /** Loads the boundary GeoJSON. Resolves to {ok, count}. Never throws. */
  async function loadBoundaries() {
    try {
      const response = await fetch(GEOJSON_URL, { cache: "force-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const geojson = await response.json();

      districtLayer = L.geoJSON(geojson, {
        style: () => BASE_STYLE,
        onEachFeature: (feature, layer) => {
          const { state, district } = feature.properties;
          const key = keyOf(state, district);
          polygonByKey.set(key, layer);

          layer.bindTooltip(
            `${escapeHtml(district)}<small>${escapeHtml(state)}</small>`,
            { sticky: true, className: "district-tooltip", direction: "top" }
          );
          layer.on({
            click: () => onSelect(key),
            mouseover: () => setHover(key, true),
            mouseout: () => setHover(key, false),
          });
        },
      }).addTo(map);
      return { ok: true, count: polygonByKey.size };
    } catch (error) {
      console.error("Boundary GeoJSON failed to load", error);
      return { ok: false, error };
    }
  }

  /** Adds coordinate markers for districts that have no polygon. */
  function setDistricts(districts) {
    for (const d of districts) {
      const key = keyOf(d.state, d.district);
      if (polygonByKey.has(key) || d.latitude == null || d.longitude == null) continue;
      const marker = L.circleMarker([d.latitude, d.longitude], {
        radius: 4.5,
        weight: 1,
        className: "district-point",
      }).addTo(map);
      marker.bindTooltip(
        `${escapeHtml(d.district)}<small>${escapeHtml(d.state)} · no boundary polygon</small>`,
        { className: "district-tooltip", direction: "top" }
      );
      marker.on("click", () => onSelect(key));
      pointByKey.set(key, marker);
    }
    return { markers: pointByKey.size };
  }

  return {
    map,
    loadBoundaries,
    setDistricts,
    selectDistrict,
    resetView,
    setPulse,
    hasBoundary: (key) => polygonByKey.has(key),
    invalidate: () => map.invalidateSize(),
    get selectedKey() { return selectedKey; },
  };
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

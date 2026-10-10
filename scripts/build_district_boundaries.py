"""
Build the web district-boundary layer used by the interactive map.

This is an OFFLINE, one-off preprocessing script. It is NOT needed to run the
website and is not part of the prediction pipeline.

Input
-----
1. INDIA_DISTRICTS.geojson from https://github.com/datta07/INDIAN-SHAPEFILES
   (licence: MIT; 820 district polygons, WGS84). Download it locally and pass
   the path as the first argument. The file is ~78 MB and is not committed.
2. combined_rainfall_weather.csv  (the project dataset; district identifiers)
3. district_coordinates.csv       (district latitude/longitude, used to
                                   validate matches and as marker fallback)

Output
------
* static/data/india_districts.geojson  - simplified polygons for the 728
  districts of the dataset that could be matched. Properties use the DATASET's
  State/District names so the browser can join them exactly.
* scripts/boundary_match_report.csv    - one row per dataset district with the
  match method, the matched source name and the distance check. Districts
  without a trustworthy polygon are listed with method "none" and are shown on
  the map as coordinate markers only. No boundary is ever invented.

Usage
-----
    pip install shapely
    python scripts/build_district_boundaries.py path/to/INDIA_DISTRICTS.geojson
"""

import csv
import difflib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "combined_rainfall_weather.csv"
COORD_CSV = ROOT / "district_coordinates.csv"
OUT_GEOJSON = ROOT / "static" / "data" / "india_districts.geojson"
OUT_REPORT = ROOT / "scripts" / "boundary_match_report.csv"

SIMPLIFY_DEG = 0.008      # ~900 m; keeps shapes recognisable at state zoom
COORD_DECIMALS = 4        # ~11 m precision
MAX_NAME_DISTANCE_KM = 50  # polygon must lie this close to the dataset coordinate
FUZZY_CUTOFF = 0.80
SPATIAL_MAX_KM = 10

# Dataset state label -> state label(s) used by the boundary source.
STATE_ALIASES = {
    "CHHATISGARH": {"CHHATTISGARH"},
    "TAMILNADU": {"TAMIL NADU"},
    "NCT of DELHI": {"DELHI"},
    "ANDAMAN & NICOBAR": {"ANDAMAN AND NICOBAR ISLANDS"},
    "DADRA & NAGAR HAVELI": {"DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
                             "GUJARAT AND DNH AND DD ISLANDS"},
    "DAMAN & DIU": {"DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
                    "GUJARAT AND DNH AND DD ISLANDS"},
}


def norm(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def km_from_deg(deg, lat):
    return deg * 111.0 * max(math.cos(math.radians(lat)), 0.3)


def main(source_path):
    src = json.loads(Path(source_path).read_text(encoding="utf-8"))

    sources = []
    for i, feat in enumerate(src["features"]):
        props = feat["properties"]
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        sources.append({
            "idx": i,
            "state": str(props.get("state", "")).strip(),
            "district": str(props.get("district", "")).strip(),
            "norm_state": norm(props.get("state", "")),
            "norm_district": norm(props.get("district", "")),
            "geom": geom,
            "used": False,
        })

    data = pd.read_csv(DATA_CSV, usecols=["State", "District"]).drop_duplicates()
    coords = pd.read_csv(COORD_CSV)
    coords = coords.drop_duplicates(["State", "District"])
    data = data.merge(coords, on=["State", "District"], how="left")
    data = data.sort_values(["State", "District"]).reset_index(drop=True)

    def candidates_for(state):
        allowed = STATE_ALIASES.get(state, {state})
        allowed_norm = {norm(a) for a in allowed}
        return [s for s in sources if s["norm_state"] in allowed_norm and not s["used"]]

    results = {}

    def dist_to_geom_km(geom, lat, lon):
        return km_from_deg(Point(lon, lat).distance(geom), lat)

    def accept(row_key, cand, method, lat, lon, note=""):
        cand["used"] = True
        dist = dist_to_geom_km(cand["geom"], lat, lon) if pd.notna(lat) else float("nan")
        results[row_key] = {
            "method": method,
            "source_district": cand["district"],
            "source_state": cand["state"],
            "geom": cand["geom"],
            "distance_km": round(dist, 2) if dist == dist else "",
            "note": note,
        }

    rows = list(data.itertuples(index=False))

    # Pass 1: exact normalised name (closest candidate wins if duplicates)
    for r in rows:
        key = (r.State, r.District)
        cands = [c for c in candidates_for(r.State) if c["norm_district"] == norm(r.District)]
        if not cands:
            continue
        if pd.notna(r.Latitude):
            cands.sort(key=lambda c: dist_to_geom_km(c["geom"], r.Latitude, r.Longitude))
        best = cands[0]
        dist = dist_to_geom_km(best["geom"], r.Latitude, r.Longitude) if pd.notna(r.Latitude) else 0
        if dist > MAX_NAME_DISTANCE_KM:
            continue  # name matches but location disagrees: leave for later passes
        accept(key, best, "name", r.Latitude, r.Longitude)

    # Pass 2: fuzzy name within state, validated by distance
    for r in rows:
        key = (r.State, r.District)
        if key in results:
            continue
        cands = candidates_for(r.State)
        if not cands:
            continue
        names = {c["norm_district"]: c for c in cands}
        close = difflib.get_close_matches(norm(r.District), list(names), n=3, cutoff=FUZZY_CUTOFF)
        good = []
        for name in close:
            c = names[name]
            if pd.notna(r.Latitude):
                d = dist_to_geom_km(c["geom"], r.Latitude, r.Longitude)
                if d <= MAX_NAME_DISTANCE_KM:
                    good.append((d, c))
        if good:
            good.sort(key=lambda t: t[0])
            accept(key, good[0][1], "fuzzy_name", r.Latitude, r.Longitude)

    # Pass 3: spatial - the dataset coordinate lies inside (or very near) an
    # unused polygon of the same state. Used for garbled/renamed boundary labels.
    for r in rows:
        key = (r.State, r.District)
        if key in results or pd.isna(r.Latitude):
            continue
        pt = Point(r.Longitude, r.Latitude)
        cands = candidates_for(r.State)
        inside = [c for c in cands if c["geom"].contains(pt)]
        if inside:
            accept(key, inside[0], "spatial_contains", r.Latitude, r.Longitude,
                   note="name not matched; coordinate lies inside this polygon")
            continue
        near = [(dist_to_geom_km(c["geom"], r.Latitude, r.Longitude), c) for c in cands]
        near = [t for t in near if t[0] <= SPATIAL_MAX_KM]
        if near:
            near.sort(key=lambda t: t[0])
            accept(key, near[0][1], "spatial_nearest", r.Latitude, r.Longitude,
                   note="name not matched; nearest polygon within 10 km")

    # Write outputs ---------------------------------------------------------
    features = []
    report_rows = []
    for r in rows:
        key = (r.State, r.District)
        res = results.get(key)
        if res:
            geom = res["geom"].simplify(SIMPLIFY_DEG, preserve_topology=True)
            if geom.is_empty:
                geom = res["geom"]
            geom_json = mapping(geom)
            geom_json = json.loads(json.dumps(geom_json))

            def round_coords(c):
                if isinstance(c[0], (int, float)):
                    return [round(c[0], COORD_DECIMALS), round(c[1], COORD_DECIMALS)]
                return [round_coords(x) for x in c]

            geom_json["coordinates"] = round_coords(geom_json["coordinates"])
            features.append({
                "type": "Feature",
                "properties": {
                    "state": r.State,
                    "district": r.District,
                    "source_district": res["source_district"],
                },
                "geometry": geom_json,
            })
            method = res["method"]
            report_rows.append([r.State, r.District, method, res["source_state"],
                                res["source_district"], res["distance_km"], res["note"]])
        else:
            report_rows.append([r.State, r.District, "none", "", "", "", "no reliable polygon match"])

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_GEOJSON.write_text(
        json.dumps({"type": "FeatureCollection", "features": features},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    with OUT_REPORT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["State", "District", "match_method", "source_state",
                         "source_district", "distance_km", "note"])
        writer.writerows(report_rows)

    counts = pd.Series([r[2] for r in report_rows]).value_counts().to_dict()
    print("Dataset districts:", len(rows))
    print("Match methods:", counts)
    print("Wrote", len(features), "features to", OUT_GEOJSON)
    print("Output size (KB):", round(OUT_GEOJSON.stat().st_size / 1024))
    print("Report:", OUT_REPORT)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

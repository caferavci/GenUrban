"""Input helpers for the original GenUrban population-to-mobility workflow.

Data and service settings are supplied by the user. No credentials, service
addresses, or machine-specific locations are included in this module.
"""

import os
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import wkt
from sklearn.neighbors import KDTree


MAPPING_CSV = "census_mapping.csv"


def _require_building_geometries(gdf):
    """Reject absent or unusable geometries rather than inventing buildings."""
    if gdf is None or gdf.empty:
        raise ValueError("Building input is empty; supply a nonempty building dataset.")
    if "geometry" not in gdf.columns:
        raise ValueError("Building input must contain a geometry column.")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Building input contains missing or empty geometries.")
    if not gdf.geometry.is_valid.all():
        raise ValueError("Building input contains invalid geometries; repair the input first.")


def process_locs_buildings(
    locs, data_dir="Overture", radius=0.01, crs_latlon=4326, crs_proj=3857
):
    """Load or retrieve buildings and apply the original subtype-assignment rules.

Each location maps a name to a geographic bounding box. Existing building CSVs
are read without importing the download client. When a CSV is missing, the
Overture client retrieves it. Capacity remains projected footprint area times
height; subtype assignment retains the original neighborhood votes, target
counts, and random assignment of the remaining labels.
"""
    if not locs:
        raise ValueError("At least one named bounding box is required.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("The neighborhood radius must be a positive finite number.")
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    results = []

    def _fetch_and_prepare(name, bbox):
        filename = data_path / f"{name}_building.csv"
        if filename.exists():
            return filename
        if len(bbox) != 4 or not np.isfinite(bbox).all():
            raise ValueError("Each building bounding box must contain four finite coordinates.")
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Building bounding-box minima must be smaller than maxima.")
        try:
            import overturemaps.core
        except ImportError:
            raise ImportError(
                "Install overturemaps to retrieve buildings, or supply a prepared building CSV."
            ) from None
        try:
            gdf = overturemaps.core.geodataframe("building", bbox)
        except Exception:
            raise RuntimeError(
                "Building retrieval failed. Check the download service and bounding box, "
                "or supply a prepared building CSV."
            ) from None
        _require_building_geometries(gdf)
        required = ["id", "names", "subtype", "class", "height", "is_underground", "geometry"]
        missing = [column for column in required if column not in gdf.columns]
        if missing:
            raise ValueError("Retrieved building data lacks required columns: " + ", ".join(missing))
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=crs_latlon)
        else:
            gdf = gdf.to_crs(epsg=crs_latlon)
        gdf = gdf[required].copy()
        projected = gdf.to_crs(epsg=crs_proj)
        gdf["area"] = projected.geometry.area
        gdf = gdf.to_crs(epsg=crs_latlon)
        gdf["capacity"] = gdf["area"] * gdf["height"]
        # Geographic centroids reproduce the original helper's calculation.
        gdf["centroid"] = gdf.geometry.centroid
        gdf["lon"] = gdf["centroid"].x
        gdf["lat"] = gdf["centroid"].y
        gdf.to_csv(filename, index=False)
        return filename

    def _load_csv(filename):
        df = pd.read_csv(filename)
        required = {"geometry", "subtype", "lat", "lon", "capacity"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError("Building CSV lacks required columns: " + ", ".join(missing))
        try:
            df["geometry"] = df["geometry"].apply(wkt.loads)
        except Exception:
            raise ValueError("Building CSV must contain valid WKT geometries.") from None
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=f"EPSG:{crs_latlon}")
        _require_building_geometries(gdf)
        try:
            coordinates = gdf[["lat", "lon"]].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Building latitude and longitude must be numeric.") from None
        if not np.isfinite(coordinates).all():
            raise ValueError("Building latitude and longitude must be finite.")
        return gdf

    def _compute_target_counts(df):
        distribution = df["subtype"].value_counts(normalize=True)
        if distribution.empty:
            raise ValueError(
                "All building subtypes are missing; the original assignment rules "
                "require at least one observed subtype."
            )
        raw = distribution * len(df)
        target = raw.round().astype(int)
        delta = len(df) - target.sum()
        if delta > 0:
            fractions = (raw - np.floor(raw)).sort_values(ascending=False)
            for tag in fractions.head(delta).index:
                target[tag] += 1
        elif delta < 0:
            fractions = (raw - np.floor(raw)).sort_values()
            for tag in fractions.head(-delta).index:
                target[tag] -= 1
        return target.to_dict()

    def _compute_needed(known, target):
        assigned = known["subtype"].value_counts()
        return {tag: max(0, target.get(tag, 0) - assigned.get(tag, 0)) for tag in target}

    def _assign_spatial(known, unknown, needed):
        tree = KDTree(known[["lat", "lon"]].values, leaf_size=40)
        known_types = known["subtype"].values

        def vote(neighbors):
            counts = Counter(neighbors)
            if counts["residential"] >= 2:
                return "residential"
            if counts["commercial"] >= 1:
                return "commercial"
            if counts["industrial"] >= 1 and counts["service"] >= 1:
                return "service" if counts["residential"] >= 50 else "industrial"
            if counts["industrial"] >= 1 and counts["residential"] == 0:
                return "industrial"
            if counts["service"] >= 1 or counts["residential"] >= 1:
                return "service"
            return None

        unknown["predicted_type"] = None
        for index, (lat, lon) in zip(unknown.index, unknown[["lat", "lon"]].values):
            if not any(
                needed.get(tag, 0) > 0
                for tag in ["residential", "commercial", "industrial", "service"]
            ):
                break
            neighbors = tree.query_radius([[lat, lon]], r=radius, return_distance=False)[0]
            if len(neighbors) == 0:
                continue
            suggestion = vote(known_types[neighbors])
            if suggestion and needed.get(suggestion, 0) > 0:
                unknown.at[index, "predicted_type"] = suggestion
                needed[suggestion] -= 1
        return unknown

    def _assign_remaining(unknown, needed):
        remaining = unknown[unknown["predicted_type"].isnull()].index
        pool = []
        for tag, count in needed.items():
            if count > 0:
                pool += [tag] * count
        np.random.shuffle(pool)
        for index, tag in zip(remaining, pool):
            unknown.at[index, "predicted_type"] = tag
        return unknown

    for name, bbox in locs.items():
        filename = _fetch_and_prepare(name, bbox)
        gdf = _load_csv(filename)
        known = gdf[gdf["subtype"].notnull()].copy()
        unknown = gdf[gdf["subtype"].isnull()].copy()
        target = _compute_target_counts(gdf)
        needed = _compute_needed(known, target)
        unknown = _assign_spatial(known, unknown, needed)
        unknown = _assign_remaining(unknown, needed)
        gdf.loc[unknown.index, "subtype"] = unknown["predicted_type"]
        results.append(gdf)
    return pd.concat(results, ignore_index=True)


def _get_section(tag: str) -> str:
    if tag.startswith("S"):
        return "/subject"
    if tag.startswith("DP"):
        return "/profile"
    return ""


def _load_mapping(mapping_csv: str = MAPPING_CSV) -> Tuple[List[str], Dict[str, str]]:
    """Read the user-supplied ACS variable and label mapping."""
    mapping = pd.read_csv(mapping_csv)
    if not {"Variable", "Label"}.issubset(mapping.columns):
        raise ValueError("The ACS mapping must contain Variable and Label columns.")
    if mapping.empty or mapping[["Variable", "Label"]].isna().any().any():
        raise ValueError("The ACS mapping must contain nonempty variable and label entries.")
    return mapping["Variable"].tolist(), dict(zip(mapping["Variable"], mapping["Label"]))


def _fetch_acs_county_data(county: str, state: str, tags: List[str]) -> pd.DataFrame:
    """Fetch the configured ACS dataset without logging service URLs or keys."""
    base_url = os.getenv("CENSUS_API_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("CENSUS_API_KEY", "").strip()
    if not base_url:
        raise ValueError(
            "CENSUS_API_BASE_URL is required to download ACS data. "
            "Alternatively, supply the county CSV in the ACS cache."
        )
    if not tags:
        raise ValueError("At least one ACS variable is required.")
    frames = []
    for tag in tags:
        parameters = {"get": f"NAME,{tag}", "for": f"county:{county}", "in": f"state:{state}"}
        if api_key:
            parameters["key"] = api_key
        try:
            response = requests.get(base_url + _get_section(tag), params=parameters, timeout=30)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            raise RuntimeError(
                "ACS retrieval failed. Check the private service configuration, "
                "credentials, dataset availability, and variable mapping."
            ) from None
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[0], list):
            raise ValueError("ACS returned no usable tabular county data.")
        try:
            frame = pd.DataFrame(data[1:], columns=data[0])
        except (TypeError, ValueError):
            raise ValueError("ACS returned an invalid tabular response.") from None
        if not {"NAME", "state", "county", tag}.issubset(frame.columns):
            raise ValueError("ACS response lacks required county or variable columns.")
        frames.append(frame)
    return reduce(
        lambda left, right: pd.merge(left, right, on=["NAME", "state", "county"]),
        frames,
    )


def acs_county_data(
    county: str, state: str, mapping_csv: str = MAPPING_CSV, output_dir: str = "ACS"
) -> str:
    """Return the original ACS summary-plus-CSV text, using a local cache first.

Supply a prepared ``state_county.csv`` to run offline. If it is absent, provide
the variable mapping and configure CENSUS_API_BASE_URL for the intended ACS
dataset; CENSUS_API_KEY is read privately from the runtime environment.
"""
    filename = f"{state}_{county}.csv"
    path = os.path.join(output_dir, filename)
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        if not os.path.isfile(mapping_csv):
            raise FileNotFoundError(
                "Supply the ACS county cache or a census_mapping.csv variable mapping "
                "before downloading data."
            )
        tags, label_map = _load_mapping(mapping_csv)
        df = _fetch_acs_county_data(county, state, tags)
        df = df.rename(columns=lambda column: label_map.get(column, column))
        df = df.drop(columns=["NAME", "state", "county"])
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(path, index=False)
    if df.empty:
        raise ValueError("The ACS county table must contain at least one data row.")
    csv_text = df.to_csv(index=False)
    # Retain this exact wrapper because it is part of the original agent input.
    summary = f"Saved ACS data to `{path}` ({len(df)} row(s), {len(df.columns)} column(s))"
    return f"{summary}\n\n{csv_text}"


def read_from_csv(filename: str) -> str:
    """Read a CSV and return its text representation for the agent prompt."""
    return pd.read_csv(filename).to_csv(index=False)

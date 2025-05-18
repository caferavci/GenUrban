import os
from collections import Counter
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.neighbors import KDTree
import overturemaps.core
from functools import reduce
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import math
import json
from shapely.geometry import shape


# Seattle: -122.459681,47.481002,-122.224433,47.734124
# phoenix: -112.324024,33.2866,-111.925531,33.835285
# "boston": (-71.191244,42.227911,-70.804488,42.398397),
# 'neworleans': (-90.139931,29.865481,-89.625176,30.199469),
# 'chicago': (-87.947041,41.644531,-87.507176,42.027898),
# 'sanfrancisco': (-122.588041,37.640314,-122.314319,37.872715),
# 'austin': (-97.963384,30.108672,-97.515982,30.528434),
# 'Portland': (-122.844401,45.432536,-122.462353,45.652881),
# 'lasvagas': (-115.424285,36.129554,-115.047765,36.380841),
# 'atlanta': (-84.554658,33.643486,-84.27761,33.894409),
# 'philadelphia': (-75.263678,39.857742,-74.942691,40.162323),
# 'denver': (-105.107521,39.614301,-104.597038,39.914209),
# 'milwaukee': (-88.083329,42.917583,-87.862713,43.200599),
# 'dallas': (-97.004362,32.607435,-96.458056,33.030895),
# 'honolulu': (-157.958936,21.251871,-157.64725,21.417626),
# 'washington': (-77.126045,38.809118,-76.909366,38.995968),
# 'pittsburgh': (-80.103409,40.356907,-79.855674,40.510469),
# 'miami': (-80.331633,25.690735,-80.144382,25.8622),
# 'minneapolis': (-93.351989,44.881437,-93.192083,45.056012),
# "washington_multi_res": (-77.119795,38.791631,-76.909366,38.99757)

# cities are defined by a bounding box where all features of the specified bounding box will be collected
# bounding box for each city is listed above and can be found using the tool linked above
# one city at a time
locs = {
  'new-york-city-boroughs': (-74.047736,40.674144,-73.884672,40.901888),
}
# Specification:
# Building -- must be polygons, has area and height as key attributes. Think about using centroid for buildings, they are so tiny anyways
# Places -- must be points
# Transportation -- two types: segments and connectors. Segments must be linestring.
# land_use and infrastructure: supplementary classes.
# Not using: addresses, overlaps with buildings, incomplete data

def process_locs_buildings(locs, data_dir='Overture', radius=0.01,
                            crs_latlon=4326, crs_proj=3857):
    """
    For each named bounding‐box in `locs`, fetch building data (once),
    compute area/capacity/centroids, save to CSV, then load & fill missing
    subtypes via your spatial rules + random assignment.
    Returns a single GeoDataFrame concatenating all processed buildings.
    """
    os.makedirs(data_dir, exist_ok=True)
    results = []

    def _fetch_and_prepare(name, bbox):
        fn = os.path.join(data_dir, f"{name}_building.csv")
        if not os.path.exists(fn):
            gdf = overturemaps.core.geodataframe("building", bbox)
            gdf.set_crs(epsg=crs_latlon, inplace=True)
            gdf = gdf[['id','names','subtype','class','height','is_underground','geometry']].copy()
            # area & capacity
            proj = gdf.to_crs(epsg=crs_proj)
            gdf['area'] = proj.geometry.area
            gdf = gdf.to_crs(epsg=crs_latlon)
            gdf['capacity'] = gdf['area'] * gdf['height']
            # centroids
            gdf['centroid'] = gdf.geometry.centroid
            gdf['lon'] = gdf.centroid.x
            gdf['lat'] = gdf.centroid.y
            gdf.to_csv(fn, index=False)
        return fn

    def _load_csv(fn):
        df = pd.read_csv(fn)
        df['geometry'] = df['geometry'].apply(wkt.loads)
        return gpd.GeoDataFrame(df, geometry='geometry', crs=f"EPSG:{crs_latlon}")

    def _compute_target_counts(df):
        dist = df['subtype'].value_counts(normalize=True)
        raw = dist * len(df)
        tgt = raw.round().astype(int)
        delta = len(df) - tgt.sum()
        if delta > 0:
            fracs = (raw - raw.floor()).sort_values(ascending=False)
            for tag in fracs.head(delta).index: tgt[tag] += 1
        elif delta < 0:
            fracs = (raw - raw.floor()).sort_values()
            for tag in fracs.head(-delta).index: tgt[tag] -= 1
        return tgt.to_dict()

    def _compute_needed(df_known, target):
        assigned = df_known['subtype'].value_counts()
        return {t: max(0, target.get(t,0) - assigned.get(t,0)) for t in target}

    def _assign_spatial(df_known, df_unknown, needed):
        tree = KDTree(df_known[['lat','lon']].values, leaf_size=40)
        known_types = df_known['subtype'].values
        def vote(neigh):
            c = Counter(neigh)
            if c['residential']>=2: return 'residential'
            if c['commercial']>=1:  return 'commercial'
            if c['industrial']>=1 and c['service']>=1:
                return 'service' if c['residential']>=50 else 'industrial'
            if c['industrial']>=1 and c['residential']==0: return 'industrial'
            if c['service']>=1 or c['residential']>=1: return 'service'
            return None

        df_unknown['predicted_type'] = None
        for idx, (lat,lon) in zip(df_unknown.index, df_unknown[['lat','lon']].values):
            if not any(needed.get(t,0)>0 for t in ['residential','commercial','industrial','service']):
                break
            neighbors = tree.query_radius([[lat,lon]], r=radius, return_distance=False)[0]
            if len(neighbors)==0: continue
            sug = vote(known_types[neighbors])
            if sug and needed.get(sug,0)>0:
                df_unknown.at[idx,'predicted_type'] = sug
                needed[sug] -= 1
        return df_unknown

    def _assign_remaining(df_unknown, needed):
        remaining_idx = df_unknown[df_unknown['predicted_type'].isnull()].index
        tags = [t for t,c in needed.items() if c>0]
        pool = []
        for t in tags:
            pool += [t]*needed[t]
        np.random.shuffle(pool)
        for idx, tag in zip(remaining_idx, pool):
            df_unknown.at[idx,'predicted_type'] = tag
        return df_unknown

    for name, bbox in locs.items():
        csv_fn = _fetch_and_prepare(name, bbox)
        gdf = _load_csv(csv_fn)

        df_kn = gdf[gdf['subtype'].notnull()].copy()
        df_un = gdf[gdf['subtype'].isnull()].copy()

        tgt = _compute_target_counts(gdf)
        needed = _compute_needed(df_kn, tgt)
        df_un = _assign_spatial(df_kn, df_un, needed)
        df_un = _assign_remaining(df_un, needed)

        gdf.loc[df_un.index, 'subtype'] = df_un['predicted_type']
        results.append(gdf)

    return pd.concat(results, ignore_index=True)

def aggregate_buildings():
    locs = {
        'new-york-city-boroughs': (-74.047736,40.674144,-73.884672,40.901888),
    }
    gdf = process_locs_buildings(locs)
    with open('CityHex20Exp/new-york-city-boroughs.geojson', 'r') as f:
        geojson = json.load(f)
    polygons = [shape(feat["geometry"]) for feat in geojson["features"]]

    gdf_shapes = gpd.GeoDataFrame(
    {"shape_id": list(range(len(polygons)))},
    geometry=polygons,
    crs="EPSG:4326"
)


    joined = gpd.sjoin(
        gdf,
        gdf_shapes[["shape_id", "geometry"]],
        how="inner",
        predicate="within"
    )

    agg = (
        joined
        .groupby(["shape_id", "subtype"], dropna=False)
        .agg(
            count         = ("id",       "size"),
            total_capacity= ("capacity", "sum")
        )
        .reset_index()
    )

    lines = []
    for sid in sorted(agg["shape_id"].unique()):
        lines.append(f"Shape {sid}:")
        df_grp = agg[agg["shape_id"] == sid]
        for _, row in df_grp.iterrows():
            subtype = row["subtype"]
            count = int(row["count"])
            total_capacity = float(row["total_capacity"])
            lines.append(f"  subtype: {subtype}, count: {count}, total_capacity: {total_capacity}")
    return "\n".join(lines)

import requests
import pandas as pd
from functools import reduce

API_KEY = "0291ed453c49b88aafdb6d6d12bac47801f0e6f5"
MAPPING_CSV = "census_mapping.csv"

def _get_section(tag: str) -> str:
    if tag.startswith("B"):
        return ""
    elif tag.startswith("S"):
        return "/subject"
    elif tag.startswith("DP"):
        return "/profile"
    return ""

def _load_mapping(mapping_csv: str = MAPPING_CSV) -> Tuple[List[str], Dict[str, str]]:
    df_map = pd.read_csv(mapping_csv)
    tags = df_map['Variable'].tolist()
    label_map = dict(zip(df_map['Variable'], df_map['Label']))
    return tags, label_map

def _fetch_acs_county_data(county: str,
                          state: str,
                          tags: List[str]) -> pd.DataFrame:
    dfs = []
    for tag in tags:
        section = _get_section(tag)
        url = (
            f"https://api.census.gov/data/2023/acs/acs5{section}"
            f"?get=NAME,{tag}&for=county:{county}&in=state:{state}&key={API_KEY}"
        )
        resp = requests.get(url); resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        dfs.append(df)

    merged = reduce(
        lambda left, right: pd.merge(left, right, on=["NAME", "state", "county"]),
        dfs
    )
    return merged

def acs_county_data(county: str,
                         state: str,
                         mapping_csv: str = MAPPING_CSV,
                         output_dir: str = "ACS") -> str:
    # 1) load tags & labels
    tags, label_map = _load_mapping(mapping_csv)

    # 2) fetch raw data
    df = _fetch_acs_county_data(county, state, tags)

    # 3) rename codes → human labels
    df = df.rename(columns=lambda c: label_map.get(c, c))

    # 4) drop the geographic columns
    df = df.drop(columns=['NAME', 'state', 'county'])

    # 5) write to ACS/{state}_{county}.csv, recreating if exists
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{state}_{county}.csv"
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    df.to_csv(path, index=False)

    # 6) return summary
    csv_text = df.to_csv(index=False)
    summary = f"Saved ACS data to `{path}` ({len(df)} row(s), {len(df.columns)} column(s))"
    return f"{summary}\n\n{csv_text}"

#def write_to_csv(df: pd.DataFrame, filename: str):
#    """
#    Write the DataFrame to a CSV file with the specified filename.
#    """
#    df.to_csv(filename, index=False)

def read_from_csv(filename: str) -> str:
    """
    Read a DataFrame from a CSV file with the specified filename.
    """
    df = pd.read_csv(filename)
    return df.to_csv(index=False)



import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import pydeck as pdk
import numpy as np

# --- Step 1: Load TAZs from XML ---
tree = ET.parse("GenUrban Plot\TAZs.xml")
root = tree.getroot()

records = []
for taz in root.findall("taz"):
    taz_id = int(taz.get("id"))
    coords = [tuple(map(float, pt.split(","))) for pt in taz.get("shape").strip().split(" ")]
    polygon = Polygon(coords)
    records.append({"id": taz_id, "geometry": polygon})

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
gdf["centroid_lat"] = gdf.geometry.centroid.y
gdf["centroid_lng"] = gdf.geometry.centroid.x

# --- Step 2: Load OD Flows ---
od_df = pd.read_csv("GenUrban Plot\TomTomODs.csv", engine="python")
od_df.columns = od_df.columns.str.strip()
od_df["origin_id"] = od_df["Origin"].str.extract(r"Region (\d+)").astype(int)
od_df["destination_id"] = od_df["Destination"].str.extract(r"Region (\d+)").astype(int)
od_df = od_df[od_df["origin_id"] != od_df["destination_id"]]  # remove self-links

# --- Step 3: Merge centroids ---
od_df = od_df.merge(
    gdf[["id", "centroid_lat", "centroid_lng"]].rename(columns={
        "id": "origin_id", "centroid_lat": "source_lat", "centroid_lng": "source_lng"
    }),
    on="origin_id"
)
od_df = od_df.merge(
    gdf[["id", "centroid_lat", "centroid_lng"]].rename(columns={
        "id": "destination_id", "centroid_lat": "target_lat", "centroid_lng": "target_lng"
    }),
    on="destination_id"
)

# --- Step 4: Flow Scaling + Color Mapping + Height ---
od_df["flow_log"] = np.log1p(od_df["Flow"])
min_log = od_df["flow_log"].min()
max_log = od_df["flow_log"].max()
od_df["flow_norm"] = (od_df["flow_log"] - min_log) / (max_log - min_log)

def flow_to_color(norm):
    r = int(255 - norm * (255 - 139))   # from 255 → 139
    g = int(140 - norm * 140)           # from 140 → 0
    b = 0
    return [r, g, b]

od_df["color"] = od_df["flow_norm"].apply(flow_to_color)

# Set arc height based on normalized flow
od_df["height"] = od_df["flow_norm"] * 100  # adjust this scale as needed

# --- Step 5: Pydeck Layers ---

# TAZ polygon borders
polygon_layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf[["geometry"]].__geo_interface__,
    stroked=True,
    filled=False,
    extruded=False,
    get_line_color=[0, 0, 0],
    line_width_min_pixels=2.5,
    pickable=False,
)

# Centroid dots
dot_layer = pdk.Layer(
    "ScatterplotLayer",
    data=gdf[["centroid_lat", "centroid_lng"]].rename(columns={
        "centroid_lat": "lat", "centroid_lng": "lng"
    }),
    get_position='[lng, lat]',
    get_radius=250,
    get_fill_color='[255, 140, 0]',
    pickable=False,
)

# Arcs with variable height
arc_layer = pdk.Layer(
    "ArcLayer",
    data=od_df,
    get_source_position=["source_lng", "source_lat"],
    get_target_position=["target_lng", "target_lat"],
    get_source_color="color",
    get_target_color="[0, 0, 0, 30]",
    get_width=5,
    elevation_scale=1,
    pickable=True,
)

# --- Step 6: View & Render ---
view_state = pdk.ViewState(
    latitude=gdf["centroid_lat"].mean(),
    longitude=gdf["centroid_lng"].mean(),
    zoom=11,
    pitch=45,     # 🆕 increases vertical view angle
    bearing=0
)

deck = pdk.Deck(
    layers=[polygon_layer, dot_layer, arc_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"text": "From {Origin} → {Destination}\nFlow: {Flow}"}
)

deck.to_html("taz_od_height_adjusted.html", notebook_display=False)
print("✅ Height-adjusted arcs map saved as 'taz_od_height_adjusted.html'")

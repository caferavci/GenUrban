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

# --- Step 2: Load Flow Data ---
od_df = pd.read_csv("GenUrban Plot\Generated_Flow.csv", engine="python")
od_df.columns = od_df.columns.str.strip()
od_df["origin_id"] = od_df["origin"]
od_df["destination_id"] = od_df["destination"]
od_df["Flow"] = od_df["people"]

# Remove self-links
od_df = od_df[od_df["origin_id"] != od_df["destination_id"]]

# --- Step 3: Identify Dominant Cluster per Destination ---
grouped = od_df.groupby(["destination_id", "cluster"])["Flow"].sum().reset_index()
dominant_clusters = grouped.sort_values("Flow", ascending=False).drop_duplicates("destination_id")
dominant_clusters = dominant_clusters.rename(columns={"cluster": "dominant_cluster", "Flow": "max_flow"})

# --- Step 4: Merge into gdf and assign fill color ---
gdf_colored = gdf.merge(dominant_clusters, left_on="id", right_on="destination_id", how="left")

# Define cluster colors
cluster_colors = {
    "YC_01": [255, 165, 0],     # Orange
    "MP_02": [0, 128, 255],     # Blue
    "LI_04": [0, 200, 0],       # Green
    "FH_05": [255, 0, 0],       # Red
    "SY_06": [128, 0, 255],     # Purple
    "HE_07": [255, 105, 180],   # Pink
}

# Assign fill color
gdf_colored["fill_color"] = gdf_colored["dominant_cluster"].map(cluster_colors)

# --- Step 5: Create Pydeck Layers ---

# Filled TAZs colored by dominant cluster
polygon_layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf_colored,
    stroked=True,
    filled=True,
    extruded=False,
    get_fill_color="fill_color",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=1.5,
    pickable=True,
)

# Orange centroid dots (optional)
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

# --- Step 6: View & Render ---
view_state = pdk.ViewState(
    latitude=gdf["centroid_lat"].mean(),
    longitude=gdf["centroid_lng"].mean(),
    zoom=11,
    pitch=0
)

deck = pdk.Deck(
    layers=[polygon_layer, dot_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"text": "TAZ {id}\nCluster: {dominant_cluster}\nMax Flow: {max_flow}"}
)

deck.to_html("taz_colored_by_dominant_cluster.html", notebook_display=False)
print("✅ Map saved as 'taz_colored_by_dominant_cluster.html'")

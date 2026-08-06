import geopandas as gpd
import pandas as pd
import pydeck as pdk
import json

# ================= CONFIGURATION =================
CITY = "NYC"
MODEL = "openai.gpt-5-mini"  # Change to whichever model you want to plot
FLOW_FILE = f"Plots/{CITY}_{MODEL}_OD_Flows.csv"
GEOJSON_FILE = "CityHex20Exp/new-york-city-boroughs.geojson" # Update based on city
# =================================================

print(f"Loading data for {CITY}...")

# 1. Load GeoJSON and extract Centroids
gdf = gpd.read_file(GEOJSON_FILE)
# Ensure CRS is correct for lat/lon
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

gdf["centroid_lng"] = gdf.geometry.centroid.x
gdf["centroid_lat"] = gdf.geometry.centroid.y
# Assuming GeoJSON has an 'id' or we use index to match Origin/Destination
gdf["region_id"] = gdf.index.astype(str) 

# 2. Load the Predicted Flows
od_df = pd.read_csv(FLOW_FILE)

# Standardize column names (handling Origin/origin variations)
if 'Origin' in od_df.columns:
    od_df = od_df.rename(columns={'Origin': 'origin', 'Destination': 'destination'})

od_df['origin'] = od_df['origin'].astype(str).str.replace("Region ", "").str.strip()
od_df['destination'] = od_df['destination'].astype(str).str.replace("Region ", "").str.strip()

# 3. Merge Coordinates for Origin and Destination
od_df = od_df.merge(
    gdf[["region_id", "centroid_lng", "centroid_lat"]],
    left_on="origin", right_on="region_id", how="inner"
).rename(columns={"centroid_lng": "source_lng", "centroid_lat": "source_lat"}).drop(columns=["region_id"])

od_df = od_df.merge(
    gdf[["region_id", "centroid_lng", "centroid_lat"]],
    left_on="destination", right_on="region_id", how="inner"
).rename(columns={"centroid_lng": "target_lng", "centroid_lat": "target_lat"}).drop(columns=["region_id"])

# 4. Color Mapping based on Clusters
cluster_colors = {
    f"{CITY}_C1": [0, 128, 255],     # Blue
    f"{CITY}_C2": [0, 200, 0],       # Green
    f"{CITY}_C3": [255, 0, 0],       # Red
    f"{CITY}_C4": [128, 0, 255],     # Purple
    f"{CITY}_C5": [255, 140, 0],     # Orange
    f"{CITY}_C6": [255, 105, 180],   # Pink
}
od_df['color'] = od_df['Cluster'].map(lambda x: cluster_colors.get(x, [100, 100, 100]))

# 5. Build Pydeck Layers
# Layer 1: Base Regions (Hexagons/Polygons)
polygon_layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf.__geo_interface__,
    stroked=True,
    filled=False,
    get_line_color=[50, 50, 50, 100],
    line_width_min_pixels=1,
)

# Layer 2: Predicted Flows (Arcs)
# We scale the width of the arc based on the Predicted_Flow
arc_layer = pdk.Layer(
    "ArcLayer",
    data=od_df[od_df['Predicted_Flow'] > 5], # Filter out tiny noise flows
    get_source_position=["source_lng", "source_lat"],
    get_target_position=["target_lng", "target_lat"],
    get_source_color="color",
    get_target_color="[255, 255, 255, 120]", # Fade to white at destination
    get_width="Predicted_Flow * 0.05", # Adjust multiplier for visual thickness
    elevation_scale=1.5,
    pickable=True,
)

# 6. Render Map
view_state = pdk.ViewState(
    latitude=gdf["centroid_lat"].mean(),
    longitude=gdf["centroid_lng"].mean(),
    zoom=10,
    pitch=45, # Tilted for 3D Arc view
)

tooltip = {
    "html": "<b>Origin:</b> {origin} <br/>"
            "<b>Destination:</b> {destination} <br/>"
            "<b>Cluster:</b> {Cluster} <br/>"
            "<b>Predicted Flow:</b> {Predicted_Flow} <br/>"
            "<b>Actual Flow:</b> {Actual_Flow}",
    "style": {"color": "white"}
}

deck = pdk.Deck(
    layers=[polygon_layer, arc_layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/dark-v10"
)

output_html = f"{CITY}_Predicted_Flows.html"
deck.to_html(output_html)
print(f"✅ Map successfully generated: {output_html}")
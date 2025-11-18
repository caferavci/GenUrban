import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import pydeck as pdk
import numpy as np

# --- Step 1: Load TAZs from XML ---
tree = ET.parse("TAZs.xml")
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
od_df = pd.read_csv("TomTomODs.csv", engine="python")
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

# --- Step 4: Flow Scaling + Color Mapping + Line Width ---
od_df["flow_log"] = np.log1p(od_df["Flow"])
min_log = od_df["flow_log"].min()
max_log = od_df["flow_log"].max()
od_df["flow_norm"] = (od_df["flow_log"] - min_log) / (max_log - min_log)

def flow_to_color(flow_value):
    # Custom color scale with specific breakpoints
    # 0: yellow, 200: orange, 400: red, 800: wine, 1000+: mahogany
    
    if flow_value <= 200:
        # 0-200: Yellow to Orange
        norm = flow_value / 200
        r = 255  # Stay at 255 (bright)
        g = int(255 - norm * 90)  # from 255 (yellow) to 165 (orange)
        b = 0
    elif flow_value <= 400:
        # 200-400: Orange to Red
        norm = (flow_value - 200) / 200
        r = 255  # Stay at 255 (bright red)
        g = int(165 - norm * 165)  # from 165 (orange) to 0 (red)
        b = 0
    elif flow_value <= 800:
        # 400-800: Red to Wine
        norm = (flow_value - 400) / 400
        r = int(255 - norm * 127)  # from 255 (red) to 128 (wine)
        g = 0  # Stay at 0
        b = int(norm * 0)  # Stay at 0 (no blue in wine)
    else:
        # 800-1000+: Wine to Mahogany
        norm = min((flow_value - 800) / 200, 1.0)  # Cap at 1.0 for 1000+
        r = int(128 - norm * 64)  # from 128 (wine) to 64 (mahogany)
        g = int(norm * 32)  # from 0 to 32 (slight brown tint)
        b = int(norm * 16)  # from 0 to 16 (slight brown tint)
    
    return [r, g, b]

# Apply color mapping using actual flow values (not normalized)
od_df["color"] = od_df["Flow"].apply(flow_to_color)

print(f"Flow Statistics:")
print(f"Min flow: {od_df['Flow'].min()}")
print(f"Max flow: {od_df['Flow'].max()}")
print(f"Mean flow: {od_df['Flow'].mean():.1f}")
print(f"Flow > 5000: {(od_df['Flow'] > 5000).sum()} out of {len(od_df)} pairs")
print(f"Custom Color Scale:")
print(f"Flow 0: Yellow (255,255,0)")
print(f"Flow 200: Orange (255,165,0)")  
print(f"Flow 400: Red (255,0,0)")
print(f"Flow 800: Wine (128,0,0)")
print(f"Flow 1000+: Mahogany (64,32,16)")
print(f"Smooth transitions between these exact color points")
print()

# Set line width based on normalized flow - INCREASED THICKNESS
od_df["width"] = od_df["flow_norm"] * 15 + 5  # line width from 5 to 20 (increased from 2-10)

# --- Step 5: Create curved paths for each OD pair ---
def create_curved_path(start_lng, start_lat, end_lng, end_lat, flow_norm, num_points=20):
    """Create a curved path between two points based on flow intensity"""
    
    # Calculate midpoint
    mid_lng = (start_lng + end_lng) / 2
    mid_lat = (start_lat + end_lat) / 2
    
    # Calculate curve height (perpendicular offset from straight line)
    curve_height = flow_norm * 0.02  # Adjust this multiplier for more/less curve
    
    # Calculate perpendicular direction for curve
    dx = end_lng - start_lng
    dy = end_lat - start_lat
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        # Perpendicular vector (rotated 90 degrees)
        perp_x = -dy / length * curve_height
        perp_y = dx / length * curve_height
        
        # Curve control point
        control_lng = mid_lng + perp_x
        control_lat = mid_lat + perp_y
    else:
        control_lng = mid_lng
        control_lat = mid_lat
    
    # Generate curved path points using quadratic Bezier curve
    path_points = []
    for i in range(num_points + 1):
        t = i / num_points
        
        # Quadratic Bezier formula: P(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        lng = (1-t)**2 * start_lng + 2*(1-t)*t * control_lng + t**2 * end_lng
        lat = (1-t)**2 * start_lat + 2*(1-t)*t * control_lat + t**2 * end_lat
        
        path_points.append([lng, lat])
    
    return path_points

# Create curved line data
line_data = []
for idx, row in od_df.iterrows():
    curved_path = create_curved_path(
        row['source_lng'], row['source_lat'], 
        row['target_lng'], row['target_lat'], 
        row['flow_norm']
    )
    
    line_data.append({
        'path': curved_path,
        'color': row['color'],
        'width': row['width'],
        'flow': row['Flow'],
        'origin': row['Origin'],
        'destination': row['Destination']
    })

# --- Step 6: Pydeck Layers ---

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

# Lines connecting OD pairs with curves and dotted style
curved_line_layer = pdk.Layer(
    "PathLayer",
    data=line_data,
    get_path='path',
    get_color='color',
    get_width='width',
    width_scale=2,
    get_dash_array=[10, 5],  # Dotted/dashed pattern: 10 pixels line, 5 pixels gap
    pickable=True,
)

# --- Step 7: View & Render ---
view_state = pdk.ViewState(
    latitude=gdf["centroid_lat"].mean(),
    longitude=gdf["centroid_lng"].mean(),
    zoom=11,
    pitch=45,     # increases vertical view angle
    bearing=0
)

deck = pdk.Deck(
    layers=[polygon_layer, dot_layer, curved_line_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"text": "From {origin} → {destination}\nFlow: {flow}"}
)

deck.to_html("taz_od_dotted_curved_lines.html", notebook_display=False)
print("✅ Dotted curved lines OD map saved as 'taz_od_dotted_curved_lines.html'")

# --- Optional: Different curve intensity options ---

# Create more dramatic curves
dramatic_line_data = []
for idx, row in od_df.iterrows():
    dramatic_curved_path = create_curved_path(
        row['source_lng'], row['source_lat'], 
        row['target_lng'], row['target_lat'], 
        row['flow_norm'],
        num_points=30  # More points for smoother curves
    )
    # Override curve height for more dramatic effect
    dramatic_curved_path = create_curved_path(
        row['source_lng'], row['source_lat'], 
        row['target_lng'], row['target_lat'], 
        row['flow_norm'] * 2,  # Double the curve intensity
        num_points=30
    )
    
    dramatic_line_data.append({
        'path': dramatic_curved_path,
        'color': row['color'],
        'width': row['width'],
        'flow': row['Flow'],
        'origin': row['Origin'],
        'destination': row['Destination']
    })

dramatic_curve_layer = pdk.Layer(
    "PathLayer",
    data=dramatic_line_data,
    get_path='path',
    get_color='color',
    get_width='width',
    width_scale=2,
    get_dash_array=[15, 8],  # Larger dotted pattern for dramatic curves
    pickable=True,
)

deck_dramatic = pdk.Deck(
    layers=[polygon_layer, dot_layer, dramatic_curve_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"text": "From {origin} → {destination}\nFlow: {flow}"}
)

deck_dramatic.to_html("taz_od_dramatic_curves.html", notebook_display=False)
print("✅ Dramatic curved lines map saved as 'taz_od_dramatic_curves.html'")

# Create subtle curves
subtle_line_data = []
for idx, row in od_df.iterrows():
    subtle_curved_path = create_curved_path(
        row['source_lng'], row['source_lat'], 
        row['target_lng'], row['target_lat'], 
        row['flow_norm'] * 0.5,  # Half the curve intensity
        num_points=15
    )
    
    subtle_line_data.append({
        'path': subtle_curved_path,
        'color': row['color'],
        'width': row['width'],
        'flow': row['Flow'],
        'origin': row['Origin'],
        'destination': row['Destination']
    })

subtle_curve_layer = pdk.Layer(
    "PathLayer",
    data=subtle_line_data,
    get_path='path',
    get_color='color',
    get_width='width',
    width_scale=2,
    get_dash_array=[8, 4],  # Smaller dotted pattern for subtle curves
    pickable=True,
)

deck_subtle = pdk.Deck(
    layers=[polygon_layer, dot_layer, subtle_curve_layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"text": "From {origin} → {destination}\nFlow: {flow}"}
)

deck_subtle.to_html("taz_od_subtle_curves.html", notebook_display=False)
print("✅ Subtle curved lines map saved as 'taz_od_subtle_curves.html'")

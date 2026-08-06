import pandas as pd
import geopandas as gpd
import pydeck as pdk
import numpy as np
import os
import math

# ================= CONFIGURATION =================
PRED_DIR = "Outputs/Predicted_Flows"
GEO_DIR = "CityHex20Exp" 
OUT_DIR = "Outputs/Figures/Maps"
os.makedirs(OUT_DIR, exist_ok=True)

# San Francisco_google.gemini-2.5-flash_OD_Flows

CITY = "San Francisco"
MODEL = "google.gemini-2.5-flash"
GEO_FILE = "sanfrancisco.geojson"

def get_flow_color_balanced(val, min_val, max_val):
    """Cooled-down Logarithmic gradient (More Yellow/Orange, Less Red)"""
    v = max(1, val)
    mn = max(1, min_val)
    mx = max(mn + 1, max_val) 
    
    # 1. Base log normalization
    norm = (math.log(v) - math.log(mn)) / (math.log(mx) - math.log(mn))
    
    # 2. THE COOLING CURVE
    # Squaring/Exponentiating a number between 0 and 1 shrinks it. 
    # This pushes mid-tier flows back down into the Yellow/Orange range.
    norm = norm ** 1.5 
    
    r = 255
    # 3. STRETCHED THRESHOLD
    # Hold onto Yellow and Orange until the 65% mark (was 40%)
    if norm < 0.65:
        g = int(255 - (255 - 150) * (norm / 0.65))
        b = 0
    else:
        # Reserve Red only for the top 35% of the log scale
        g = int(150 - 150 * ((norm - 0.65) / 0.35))
        b = 0
        
    return [r, g, b, 255] 

def add_html_legend(filepath):
    legend_html = """
    <div style="position: absolute; bottom: 40px; right: 40px; background: white; padding: 15px; 
                border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-family: Arial, sans-serif; 
                z-index: 1000; display: flex; flex-direction: column; align-items: center;">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">High Flow</div>
        <div style="height: 180px; width: 24px; background: linear-gradient(to bottom, #ff0000, #ff8800, #ffff00); 
                    border: 1px solid #ccc; border-radius: 4px;"></div>
        <div style="font-weight: bold; margin-top: 8px; font-size: 14px;">Low Flow</div>
    </div>
    """
    with open(filepath, 'a') as f:
        f.write(legend_html)

def generate_balanced_comparison_map():
    print("🚀 Loading geometries...")
    geo_path = os.path.join(GEO_DIR, GEO_FILE)
    gdf = gpd.read_file(geo_path)
    
    gdf_proj = gdf.to_crs(epsg=3857) 
    gdf["lng"] = gdf_proj.geometry.centroid.to_crs(epsg=4326).x
    gdf["lat"] = gdf_proj.geometry.centroid.to_crs(epsg=4326).y
    gdf["id_str"] = gdf.index.astype(str)

    print("📊 Loading flow data...")
    flow_path = os.path.join(PRED_DIR, f"{CITY}_{MODEL}_OD_Flows.csv")
    df = pd.read_csv(flow_path)
    df.columns = df.columns.str.strip().str.lower()
    df["orig_clean"] = df["origin"].astype(str).str.replace("Region ", "", case=False).str.strip()
    df["dest_clean"] = df["destination"].astype(str).str.replace("Region ", "", case=False).str.strip()

    temp = df.merge(gdf[['id_str', 'lng', 'lat']], left_on='orig_clean', right_on='id_str')
    temp = temp.rename(columns={'lng': 'slng', 'lat': 'slat'}).drop('id_str', axis=1)
    plot_df = temp.merge(gdf[['id_str', 'lng', 'lat']], left_on='dest_clean', right_on='id_str')
    plot_df = plot_df.rename(columns={'lng': 'tlng', 'lat': 'tlat'}).drop('id_str', axis=1)

    # GLOBAL SCALING (Ensures Actual and Predicted share the exact same color meanings)
    global_min = min(plot_df['actual_flow'].min(), plot_df['predicted_flow'].min())
    global_max = max(plot_df['actual_flow'].max(), plot_df['predicted_flow'].max())
    
    print(f"🌍 Global Scale Locked: Min={global_min}, Max={global_max}")

    ARC_LIMIT = 300 
    
    # Pitch 45 + slight Bearing for clean 3D curves
    view_state = pdk.ViewState(
        latitude=gdf.lat.mean(), 
        longitude=gdf.lng.mean(), 
        zoom=11.2, 
        pitch=45,       
        bearing=-15     
    ) 

    plot_df = plot_df[(plot_df['actual_flow'] > 2) & (plot_df['predicted_flow'] > 2)]

    for flow_type in ['actual_flow', 'predicted_flow']:
        print(f"🎨 Generating {flow_type.upper()} map...")
        
        top_arcs = plot_df.sort_values(flow_type, ascending=False).head(ARC_LIMIT).copy()
        
        # Apply the new balanced color function
        top_arcs['flow_color'] = top_arcs[flow_type].apply(lambda x: get_flow_color_balanced(x, global_min, global_max))
        
        # Keep lines thin and clean
        top_arcs['line_width'] = top_arcs[flow_type].apply(
            lambda x: 0.5 + 1.0 * ((np.log(max(1, x)) - np.log(max(1, global_min))) / (np.log(max(2, global_max)) - np.log(max(1, global_min)))) if global_max > global_min else 0.8
        )

        hex_layer = pdk.Layer(
            "GeoJsonLayer",
            gdf,
            opacity=1.0,
            stroked=True,
            filled=True,
            get_line_color=[0, 0, 0, 200],      
            get_fill_color=[245, 245, 245, 0],    
            line_width_min_pixels=1.0,
        )

        arc_layer = pdk.Layer(
            "ArcLayer", 
            top_arcs, 
            get_source_position=["slng", "slat"], 
            get_target_position=["tlng", "tlat"], 
            get_source_color="flow_color", 
            get_target_color="flow_color", 
            get_width="line_width", 
            pickable=True
        )

        r = pdk.Deck(
            layers=[hex_layer, arc_layer], 
            initial_view_state=view_state, 
            map_style="light", 
            tooltip={"text": f"From {{orig_clean}} to {{dest_clean}}\n{flow_type.replace('_', ' ').title()}: {{{flow_type}}}"}
        )
        
        out_file = f"{OUT_DIR}/{CITY}_{MODEL}_{flow_type.upper()}_BALANCED.html"
        r.to_html(out_file)
        
        add_html_legend(out_file)
        print(f"✅ Saved: {out_file}")

if __name__ == "__main__":
    generate_balanced_comparison_map()
import pandas as pd
import numpy as np
import glob
import os
import math

INPUT_DIR = "Outputs/Predicted_Flows"

# --- Haversine formula to calculate trip distance in km ---
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0 # Earth radius in km
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def run_diagnostics():
    files = glob.glob(os.path.join(INPUT_DIR, "*_OD_Flows.csv"))
    if not files:
        print("❌ No files found!")
        return

    print("🔍 Running Urban Diagnostics...\n")
    
    for f in files:
        fname = os.path.basename(f)
        city = fname.split("_")[0]
        model = fname.split("_")[1]
        
        # Only run for the best model to keep output clean
        if model != "openai.gpt-5-mini":
            continue
            
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        
        # 1. GHOST FLOW ANALYSIS (San Francisco Hypothesis)
        # How often does TomTom say < 5 people, but the LLM predicted > 50?
        ghost_flows = df[(df['actual_flow'] < 5) & (df['predicted_flow'] > 50)]
        total_routes = len(df)
        ghost_pct = (len(ghost_flows) / total_routes) * 100
        
        # 2. ERROR DECOMPOSITION BY DISTANCE (Las Vegas Hypothesis)
        # First, we need to approximate distance. If you don't have lat/lng in this CSV, 
        # we will approximate error purely based on volume tiers for now.
        # Let's look at the NRMSE for low vs. high volume routes.
        
        mean_actual = df['actual_flow'].mean()
        high_volume_mask = df['actual_flow'] > mean_actual
        low_volume_mask = df['actual_flow'] <= mean_actual
        
        # Calculate RMSE for High vs Low volume
        def get_nrmse(subset):
            if len(subset) == 0 or subset['actual_flow'].mean() == 0: return 0
            rmse = np.sqrt(np.mean((subset['actual_flow'] - subset['predicted_flow'])**2))
            return rmse / subset['actual_flow'].mean()

        nrmse_high = get_nrmse(df[high_volume_mask])
        nrmse_low = get_nrmse(df[low_volume_mask])

        print(f"🏙️ {city.upper()} (Model: {model})")
        print(f"   ➔ Ghost Flows (LLM over-predicted empty zones): {ghost_pct:.1f}% of routes")
        print(f"   ➔ High-Volume Arteries NRMSE: {nrmse_high:.2f}")
        print(f"   ➔ Low-Volume/Sparse Routes NRMSE: {nrmse_low:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    run_diagnostics()
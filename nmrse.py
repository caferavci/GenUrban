import pandas as pd
import numpy as np
import glob
import os

# --- Configuration ---
INPUT_DIR = "Outputs/Predicted_Flows"
OUTPUT_FILE = "Outputs/Detailed_Stats_Summary.csv"

def calculate_nrmse(actual, predicted):
    """Calculates RMSE and Normalizes it by the mean of actual flow."""
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mean_actual = np.mean(actual)
    n_rmse = rmse / mean_actual if mean_actual != 0 else 0
    return rmse, n_rmse

def process_all_results():
    all_stats = []
    
    # Find all OD flow CSVs
    files = glob.glob(os.path.join(INPUT_DIR, "*_OD_Flows.csv"))
    
    if not files:
        print(f"❌ No files found in {INPUT_DIR}. Check your path.")
        return

    for f in files:
        # Extract City and Model from filename
        fname = os.path.basename(f)
        parts = fname.replace("_OD_Flows.csv", "").split("_")
        city = parts[0]
        model = parts[1]
        
        # Load Data
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        
        # Calculate Metrics
        actual = df['actual_flow']
        pred = df['predicted_flow']
        
        rmse, nrmse = calculate_nrmse(actual, pred)
        mae = np.mean(np.abs(actual - pred))
        
        # Calculate R2 for double-check
        ss_res = np.sum((actual - pred)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        all_stats.append({
            "City": city,
            "Model": model,
            "R2": r2,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "MAE": mae
        })

    # Create Summary DataFrame
    summary_df = pd.DataFrame(all_stats).sort_values(["City", "R2"], ascending=[True, False])
    
    # Save to CSV
    os.makedirs("Outputs", exist_ok=True)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Stats saved to {OUTPUT_FILE}")
    
    # --- Generate LaTeX Table ---
    print("\n--- LaTeX Table Output ---")
    print(summary_df.to_latex(index=False, 
                              columns=["City", "Model", "R2", "NRMSE", "RMSE"],
                              float_format="%.4f",
                              bold_rows=True))

if __name__ == "__main__":
    process_all_results()
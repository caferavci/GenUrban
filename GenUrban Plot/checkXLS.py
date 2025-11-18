import pandas as pd

df_preview = pd.read_csv("TomTomODs.csv")
print("Detected columns:", df_preview.columns.tolist())
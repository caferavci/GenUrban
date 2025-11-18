import pandas as pd

# Load the CSV
df = pd.read_csv("GenUrban Plot\Generated_Flow.csv")
df.columns = df.columns.str.strip()

# Group by destination and cluster, sum the people
grouped = df.groupby(['origin', 'cluster'])['people'].sum().reset_index()

# Sort the result by destination and flow descending
grouped = grouped.sort_values(['origin', 'people'], ascending=[True, False])

# Display all clusters per destination
print("Total flow per cluster for each destination:\n")

for dest in grouped['origin'].unique():
    print(f"\nDestination {dest}:")
    rows = grouped[grouped['origin'] == dest]
    for _, row in rows.iterrows():
        print(f"  Cluster {row['cluster']} → {row['people']} people")

import pandas as pd

# Load CSV
df = pd.read_csv("GenUrban Plot\Generated_Flow.csv")

# Group by origin and destination
grouped_df = df.groupby(['origin', 'destination'])
df['cluster'] = df['cluster'].astype(str)
# Aggregate: concatenate fields as string and sum people
result = grouped_df.agg({
    'hour': lambda x: ','.join(map(str, x.unique())),
    'cluster': lambda x: ','.join(x.unique()),
    'leaving': lambda x: ','.join(x.unique()),
    'heading': lambda x: ','.join(x.unique()),
    'activity': lambda x: ','.join(x.unique()),
    'people': 'sum'
}).reset_index()

# Save the result if needed
result.to_csv("Flow_Summarized_By_Origin_Destination.csv", index=False)

print(result.head())

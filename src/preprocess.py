import pandas as pd
import glob
import os

files = glob.glob("data/raw/*.csv")

all_data = []

for file in files:
    temp_df = pd.read_csv(file)
    all_data.append(temp_df)

df = pd.concat(all_data, ignore_index=True)

print(f"Total rows before cleaning: {len(df)}")

df = df.drop_duplicates()

df = df.fillna(df.mean(numeric_only=True))

aqi_mapping = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

df["air_quality"] = df["aqi"].map(aqi_mapping)

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/processed_air_quality.csv"

df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")

print(df.head())
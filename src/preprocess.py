import pandas as pd
import glob
import os

# Ambil semua file CSV dari folder raw
files = glob.glob("data/raw/*.csv")

# Ambil file terbaru
latest_file = max(files, key=os.path.getctime)

print(f"Processing file: {latest_file}")

# Load data
df = pd.read_csv(latest_file)

# Hapus duplikat
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Mapping AQI ke label kualitas udara
aqi_mapping = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

df["air_quality"] = df["aqi"].map(aqi_mapping)

# Buat folder processed jika belum ada
os.makedirs("data/processed", exist_ok=True)

# Simpan hasil preprocessing
output_path = "data/processed/processed_air_quality.csv"

df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")

# Tampilkan isi dataframe
print(df.head())
import requests
import pandas as pd
from datetime import datetime
import os

# Ganti nilai ini dengan API key Anda langsung di sini.
API_KEY = 'b2ce9a7a7003c24fc1c45694e47cb1db'

# Lokasi dengan udara beragam/kotor
LOCATIONS = {
    "Jakarta": {"lat": -6.2088, "lon": 106.8456},           # Ibukota, polusi tinggi
    "Surabaya": {"lat": -7.2575, "lon": 112.7521},          # Kota besar, udara kotor
    "Bandung": {"lat": -6.9175, "lon": 107.6062},           # Cekungan, polusi tinggi
    "Medan": {"lat": 3.5952, "lon": 98.6722},               # Kota industri
}

# Hitung timestamp untuk start (1 Januari 2025) dan end (sekarang)
start_date = datetime(2025, 1, 1)
end_date = datetime.now()

if start_date >= end_date:
    raise ValueError(f"start_date must be before end_date: {start_date} >= {end_date}")

start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())

all_data = []

# Fetch data untuk setiap lokasi
for location_name, coords in LOCATIONS.items():
    print(f"\nFetching data for {location_name}...")
    
    url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={coords['lat']}&lon={coords['lon']}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"API request failed for {location_name} with status code {response.status_code}: {response.text}")
        continue
    
    data = response.json()
    
    if "list" not in data:
        print(f"API response for {location_name} does not contain 'list' key")
        continue
    
    print(f"  Got {len(data['list'])} records")
    
    # Parse data historis
    for entry in data['list']:
        components = entry['components']
        aqi = entry['main']['aqi']
        timestamp = datetime.fromtimestamp(entry['dt'])
        
        all_data.append({
            "location": location_name,
            "timestamp": timestamp,
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "co": components.get("co"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "aqi": aqi
        })

df = pd.DataFrame(all_data)

os.makedirs("data/raw", exist_ok=True)

filename = f"data/raw/air_quality_multi_location_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

df.to_csv(filename, index=False)

print(f"\n✓ Data saved to {filename}")
print(f"✓ Total rows: {len(df)}")
print(f"✓ Locations: {', '.join(LOCATIONS.keys())}")
print(f"\nData per location:")
print(df['location'].value_counts())
import requests
import pandas as pd
from datetime import datetime
import os

# Ganti nilai ini dengan API key Anda langsung di sini.
API_KEY = 'b2ce9a7a7003c24fc1c45694e47cb1db'

LAT = -6.2088
LON = 106.8456

url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

response = requests.get(url)

if response.status_code != 200:
    print(f"API request failed with status code {response.status_code}: {response.text}")
    exit(1)

data = response.json()

print(data)

if "list" not in data:
    print("API response does not contain 'list' key")
    exit()

components = data['list'][0]['components']
aqi = data['list'][0]['main']['aqi']

air_data = {
    "timestamp": [datetime.now()],
    "pm2_5": [components.get("pm2_5")],
    "pm10": [components.get("pm10")],
    "co": [components.get("co")],
    "no2": [components.get("no2")],
    "o3": [components.get("o3")],
    "so2": [components.get("so2")],
    "aqi": [aqi]
}

df = pd.DataFrame(air_data)

os.makedirs("data/raw", exist_ok=True)

filename = f"data/raw/air_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

df.to_csv(filename, index=False)

print(f"Data saved to {filename}")
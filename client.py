import requests
import json
import pandas as pd

X = pd.read_csv('US_Accidents_March23.csv')

X_non_null = X.dropna()

random_rows = X_non_null.sample(n=10, random_state=42)

required_columns = [
    'Snow', 'Day_Type', 'Season', 'Fog', 'TimeofDay', 'Rain', 'Cloud', 'Ash', 'Windy',
    'Heavy_Snow', 'Clear', 'Duration', 'Dusty', 'Heavy_Rain', 'Start_Lat', 'Start_Lng',
    'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Wind_Direction',
    'Crossing', 'Junction', 'Traffic_Signal', 'Sunrise_Sunset'
]

for col in required_columns:
    if col not in random_rows.columns:
        random_rows[col] = 0 

data = random_rows.to_dict(orient='records')

print("Columns in the DataFrame:")
print(random_rows.columns)

payload = {
    "features": data
}

url = 'http://127.0.0.1:5000/predict'
response = requests.post(url, json=payload)

if response.status_code == 200:
    prediction_results = response.json()
    print("Predictions from all models:")
    for model_name, prediction in prediction_results.items():
        print(f"{model_name}: {prediction}")
else:
    print(f"Error: {response.status_code}")
    print(f"Response: {response.json()}")

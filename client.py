import requests
import json
import pandas as pd


X = pd.read_csv('US_Accidents_March23.csv')

X_non_null = X.dropna()
random_rows = X_non_null.sample(n=10, random_state=42)


data = random_rows.to_dict(orient='records')

payload = {
    "features": data
}

url = 'http://localhost:5000/predict'
response = requests.post(url, json=payload)

if response.status_code == 200:
    prediction_results = response.json()
    print("Predictions from all models:")
    for model_name, prediction in prediction_results.items():
        print(f"{model_name}: {prediction}")
else:
    print(f"Error: {response.status_code}")

# import requests
# import json

# import pandas as pd

# # Load the CSV file into a DataFrame
# X = pd.read_csv('path_to_your_file.csv')


# X_non_null = X.dropna() 
# random_rows = X_non_null.sample(n=10, random_state=42) 

# data = {
#     "features": X 
# }


# url = 'http://localhost:5000/predict' 
# response = requests.post(url, json=data)

# if response.status_code == 200:
#     prediction_results = response.json()
#     print("Predictions from all models:")
#     for model_name, prediction in prediction_results.items():
#         print(f"{model_name}: {prediction}")
# else:
#     print(f"Error: {response.status_code}")
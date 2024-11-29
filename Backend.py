from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

models = {
    'RandomForest': joblib.load('RandomForest.joblib'),
    'SVM': joblib.load('SVM.joblib'),
    'LGBM': joblib.load('LGBM.joblib'),
    'BaggingClassifier': joblib.load('BaggingClassifier.joblib'),
    'NN': joblib.load('NN.joblib')
}

pipeline = joblib.load('transform_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data['features']

        df = pd.DataFrame(features)
        print("Received DataFrame:")
        print(df.head())

        required_columns = [
            'Snow', 'Day_Type', 'Season', 'Fog', 'TimeofDay', 'Rain', 'Cloud', 'Ash', 'Windy',
            'Heavy_Snow', 'Clear', 'Duration', 'Dusty', 'Heavy_Rain', 'Start_Lat', 'Start_Lng',
            'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Wind_Direction',
            'Crossing', 'Junction', 'Traffic_Signal', 'Sunrise_Sunset'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 

        transformed_features = pipeline.transform(df)
        print("Transformed Features:")
        print(transformed_features[:5])

        results = {}
        for model_name, model in models.items():
            prediction = model.predict(transformed_features)
            results[model_name] = prediction[0].item() 

        return jsonify(results)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

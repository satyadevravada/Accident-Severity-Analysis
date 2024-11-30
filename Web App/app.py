from flask import Flask, request, render_template
from joblib import load
from scipy.stats import mode
import pandas as pd
import math

model_paths = {
    "BaggingClassifier": r"C:\Users\nisha\Desktop\ML App\BaggingClassifier.joblib",
    "LGBM": r"C:\Users\nisha\Desktop\ML App\LGBM.joblib",
    "NN": r"C:\Users\nisha\Desktop\ML App\NN.joblib",
    "RandomForest": r"C:\Users\nisha\Desktop\ML App\RandomForest.joblib",
    "SVM": r"C:\Users\nisha\Desktop\ML App\SVM.joblib"
}
models = {name: load(path) for name, path in model_paths.items()}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

feature_names = [
    'Wind_Chill(F)', 'Distance(mi)', 'Humidity(%)', 'Duration',
    'Wind_Speed(mph)', 'Start_Lng', 'Start_Lat', 'Temperature(F)',
    'Pressure(in)', 'log_Duration', 'log_Precipitation(in)', 'Precipitation(in)'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Wind_Chill(F)']),
            float(request.form['Distance(mi)']),
            float(request.form['Humidity(%)']),
            float(request.form['Duration']),
            float(request.form['Wind_Speed(mph)']),
            float(request.form['Start_Lng']),
            float(request.form['Start_Lat']),
            float(request.form['Temperature(F)']),
            float(request.form['Pressure(in)']),
            float(request.form['Precipitation(in)'])
        ]
        logp=math.log(features[9])
        logd=math.log(features[3])
        features.insert(8,logd)
        features.insert(9,logp)

        input_df = pd.DataFrame([features], columns=feature_names)

        predictions = {}
        for name, model in models.items():
            pred = model.predict(input_df) 
            print(f"Model: {name}, Prediction: {pred}")  
            predictions[name] = pred[0]  

        from scipy.stats import mode
        prediction_mode = mode(list(predictions.values()), keepdims=True)[0][0]

        return render_template('index.html', predictions=predictions, prediction_mode=prediction_mode)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

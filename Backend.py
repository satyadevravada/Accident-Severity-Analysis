from flask import Flask, request,jsonify

import joblib

app= Flask(__name__)

models = {
    'RandomForest': joblib.load('RandomForest.joblib'),
    'SVM': joblib.load('SVM.joblib'),
    'LGBM': joblib.load('LGBM.joblib'),
    'BaggingClassifier': joblib.load('BaggingClassifier.joblib'),
    'NN':joblib.load('NN.joblib')
}
pipeline = joblib.load('transform_pipeline.joblib')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features= data['features']
    transformed_features=pipeline.transform([features])
    results = {}
    for model_name, model in models.items():
        prediction=model.predict(transformed_features)
        results[model_name]= prediction[0]
    return jsonify(results)

if __name__=="__main__":
    app.run(debug=True)

# Updated Flask API Script
from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from model.carcrashmodel import *
from joblib import load
# Assume model.carcrashmodel is a placeholder for where you'd import or define your models

app = Flask(__name__)

carcrash_api = Blueprint('carcrash_api', __name__, url_prefix='/api/carcrash')

dt_regressor = load('dt_regressor.joblib')
lr_regressor = load('lr_regressor.joblib')


@carcrash_api.route('/predict', methods=['POST'])
def predict():

    try:
        json_data = request.get_json(force=True)
        
        # Convert input JSON data to DataFrame
        features = pd.DataFrame(json_data, index=[0])
        
        # Predictions
        prediction_dt = dt_regressor.predict(features)[0]
        prediction_lr = lr_regressor.predict(features)[0]
        
        # Convert Linear Regression prediction to percentage
        # Assuming the prediction is a proportion, multiply by 100 to get percentage
        prediction_lr_percent = prediction_lr 
        
        return jsonify({
            'Decision Tree Prediction': prediction_dt,
            'Linear Regression Prediction (%)': prediction_lr_percent
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    pass

# This line is crucial
app.register_blueprint(carcrash_api)

if __name__ == '__main__':
    app.run(debug=True)

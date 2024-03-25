from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from model.carcrashmodel import *
app = Flask(__name__)

carcrash_api = Blueprint('carcrash_api', __name__, url_prefix='/api/carcrash')

# Load and prepare your data, initialize and train your models here...

@carcrash_api.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json(force=True)
        
        # Assuming JSON data will be in the form of a dictionary matching the features
        features = pd.DataFrame(json_data, index=[0])
        
        # Make predictions with both models
        prediction_dt = dt_regressor.predict(features)
        prediction_lr = lr_regressor.predict(features)
        
        # Return predictions
        return jsonify({
            'Decision Tree Prediction': prediction_dt[0],
            'Linear Regression Prediction': prediction_lr[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
# Register your blueprint

if __name__ == '__main__':
    app.run(debug=True)






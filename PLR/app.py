# app.py

import numpy as np
import pickle
from flask import Flask, request, jsonify

# Load the model from the saved file
try:
    with open('poly_reg_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: poly_reg_model.pkl not found. Ensure train_and_save.py was run.")
    exit()

try:
    with open('poly_features.pkl', 'rb') as file:
        poly = pickle.load(file)
except FileNotFoundError:
    print("Error: poly_features.pkl not found. Ensure train_and_save.py was run.")
    exit()


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data with 'YearsExperience' and returns predicted salary.
    Supports two formats:
    1. Direct format: {"YearsExperience": 5.5}
    2. Nested format: {"data": [{"YearsExperience": 5.5}]}
    """
    try:
        data = request.get_json()
        
        # Handle nested "data" format
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            data = data['data'][0]
        
        experience = data['YearsExperience']

        # Apply polynomial features transformation
        experience_poly = poly.transform([[experience]])
        
        # Make the prediction
        prediction = model.predict(experience_poly)[0]

        # Return the prediction as JSON
        return jsonify({
            'YearsExperience': experience,
            'PredictedSalary': float(prediction)
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid input format or prediction error: {e}'}), 400

if __name__ == '__main__':
    # Use Gunicorn in the container, but Flask's built-in server for local testing
    app.run(debug=True, host='0.0.0.0', port=5000)
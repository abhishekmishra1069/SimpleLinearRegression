# app.py

import numpy as np
import pickle
from flask import Flask, request, jsonify

# Load the model from the saved file
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: model.pkl not found. Ensure train_and_save.py was run.")
    exit()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data with 'YearsExperience' and returns predicted salary.
    Example JSON: {"YearsExperience": 5.5}
    """
    try:
        data = request.get_json()
        experience = data['YearsExperience']

        # Model expects a 2D array: [[experience]]
        input_data = np.array([[experience]])

        # Make the prediction
        prediction = model.predict(input_data)[0]

        # Return the prediction as JSON
        return jsonify({
            'YearsExperience': experience,
            'PredictedSalary': float(prediction)
        })

    except Exception as e:
        return jsonify({'error': f'Invalid input format or missing key: {e}'}), 400

if __name__ == '__main__':
    # Use Gunicorn in the container, but Flask's built-in server for local testing
    app.run(debug=True, host='0.0.0.0', port=5000)
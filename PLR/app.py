# app.py

import numpy as np
import pickle
from flask import Flask, request, jsonify

# Load the model and transformer at application startup (one-time cost)
# - Loading once avoids expensive disk I/O on every request and keeps
#   prediction latency low.
# - For large scikit-learn models, consider using `joblib.load` instead of
#   `pickle.load` because joblib is optimized for large numpy arrays.
try:
    with open('poly_reg_model.pkl', 'rb') as file:
        model = pickle.load(file)  # Deserializes trained LinearRegression model
except FileNotFoundError:
    # Fail fast: model must be available at startup. Use absolute paths if
    # running the script from another working directory.
    print("Error: poly_reg_model.pkl not found. Ensure Train_and_save notebook was run and the file is in PLR/ directory.")
    raise

try:
    with open('poly_features.pkl', 'rb') as file:
        poly = pickle.load(file)  # Deserializes fitted PolynomialFeatures transformer
except FileNotFoundError:
    print("Error: poly_features.pkl not found. Ensure Train_and_save notebook was run and the file is in PLR/ directory.")
    raise


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data with 'YearsExperience' and returns predicted salary.
    Supports two formats:
    1. Direct format: {"YearsExperience": 5.5}
    2. Nested format: {"data": [{"YearsExperience": 5.5}]}

    Efficiency and robustness notes (explained inline):
    - The model and transformer are loaded once at startup to avoid repeated I/O.
    - We explicitly cast inputs to `float` and create a numpy array with a concrete
      dtype (`float`) so numpy and scikit-learn operate on native numeric arrays,
      avoiding slow Python-object dtypes.
    - If you expect many predictions at once, accept a list of inputs and process
      them in a single vectorized call (both `poly.transform` and `model.predict`
      are vectorized and will be much faster on batches than looping per sample).
    """
    try:
        # Parse JSON payload. `get_json()` returns None if content-type is not JSON.
        data = request.get_json()
        if data is None:
            # Explicitly return a helpful error instead of failing with a cryptic server error
            return jsonify({'error': 'Request must be JSON with YearsExperience or nested data payload.'}), 400

        # Handle nested "data" format where the client wraps the record in an array
        # Example: {"data": [{"YearsExperience": 5.5}]}
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            data = data['data'][0]

        # Extract the numeric feature. Using `float()` ensures correct dtype for numpy.
        # Be explicit about casting to avoid object-dtype arrays which are slower.
        experience = float(data['YearsExperience'])

        # Build the input array for the transformer and model. Use a 2D numpy array with
        # a concrete dtype to avoid slow Python-object arrays: shape (1, 1) for a single sample.
        input_arr = np.array([[experience]], dtype=float)

        # Transform features using the pre-fitted PolynomialFeatures transformer.
        # If you expect batches of requests, accept lists of features and transform
        # them in one call to leverage vectorized operations (much faster than looping).
        experience_poly = poly.transform(input_arr)

        # Make the prediction. `model.predict` is vectorized so it benefits from
        # passing an array rather than single-element Python objects.
        prediction = model.predict(experience_poly)[0]

        # Return the prediction as JSON. Convert numpy types to native Python types
        # (e.g., float) to avoid serialization issues.
        return jsonify({
            'YearsExperience': experience,
            'PredictedSalary': float(prediction)
        })

    except KeyError as e:
        # Missing key in input JSON: respond with 400 and a helpful message
        return jsonify({'error': f'Missing required key: {e}'}), 400
    except ValueError as e:
        # Raised when float conversion fails or input contains invalid numbers
        return jsonify({'error': f'Invalid numeric value: {e}'}), 400
    except Exception as e:
        # Generic catch-all for unexpected errors. In production, consider logging
        # stack traces rather than returning them to the user.
        return jsonify({'error': f'Invalid input format or prediction error: {e}'}), 400

if __name__ == '__main__':
    # Use Gunicorn in the container, but Flask's built-in server for local testing
    app.run(debug=True, host='0.0.0.0', port=5000)
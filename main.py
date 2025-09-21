from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Global variable to store the model
model = None
model_loaded = False


def load_model():
    """Load the machine learning model"""
    global model, model_loaded

    try:
        # Define the correct model path with capital M
        model_path = 'Model/creditcard_pipeline.pkl'

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {os.path.abspath(model_path)}")
            return False

        print(f"✅ Found model at: {os.path.abspath(model_path)}")
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


# Load model when the app starts
model_loaded = load_model()


@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': True,
            'message': 'Model not loaded. Please check if the model file exists in the Model/ folder.'
        })

    try:
        # Get form data
        data = request.form

        # Debug: Print all form data
        print("Form data received:", dict(data))

        # Prepare input data (23 features as expected by your model)
        input_data = {
            'limit_bal': float(data['limit_bal']),
            'age': int(data['age']),
            'sex': int(data['sex']),
            'education': int(data['education']),
            'marriage': int(data['marriage']),
            'pay_0': int(data['pay_0']),  # September
            'pay_2': int(data['pay_2']),  # August
            'pay_3': int(data['pay_3']),  # July
            'pay_4': int(data['pay_4']),  # June
            'pay_5': int(data['pay_5']),  # May
            'pay_6': int(data['pay_6']),  # April
            'bill_amt_sep': float(data['bill_amt_sep']),
            'bill_amt_aug': float(data['bill_amt_aug']),
            'bill_amt_jul': float(data['bill_amt_jul']),
            'bill_amt_jun': float(data['bill_amt_jun']),
            'bill_amt_may': float(data['bill_amt_may']),
            'bill_amt_apr': float(data['bill_amt_apr']),
            'pay_amt_sep': float(data['pay_amt_sep']),
            'pay_amt_aug': float(data['pay_amt_aug']),
            'pay_amt_jul': float(data['pay_amt_jul']),
            'pay_amt_jun': float(data['pay_amt_jun']),
            'pay_amt_may': float(data['pay_amt_may']),
            'pay_amt_apr': float(data['pay_amt_apr'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # Prepare response
        if prediction == 1:
            risk_class = "high-risk"
            risk_icon = "🚨"
            risk_text = "HIGH RISK: Likely to DEFAULT"
        else:
            risk_class = "low-risk"
            risk_icon = "✅"
            risk_text = "LOW RISK: Not likely to default"

        return jsonify({
            'error': False,
            'prediction': int(prediction),
            'probability_no_default': float(probability[0]),
            'probability_default': float(probability[1]),
            'risk_class': risk_class,
            'risk_icon': risk_icon,
            'risk_text': risk_text
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': True,
            'message': f'Prediction error: {str(e)}'
        })


if __name__ == '__main__':
    # Create templates and static folders if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Run the app
    app.run(debug=True, port=5000)
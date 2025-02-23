from flask import Flask, request, jsonify
import numpy as np
import joblib
import shap

# Load the trained model and features
model = joblib.load("SmartRehabModel.pkl")
top_corr_features = joblib.load("SmartRehabFeatures.pkl")

# Create SHAP Explainer
explainer = shap.Explainer(model)

# Initialize Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input with patient data and returns a prediction with an explanation.
    """
    try:
        data = request.json  # Get input JSON
        patient_data = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array

        # Predict Recovery Score
        prediction = model.predict(patient_data)[0]

        # Explain Prediction with SHAP
        shap_values = explainer(patient_data)
        most_important_feature = top_corr_features[np.argmax(np.abs(shap_values.values))]

        # Response
        response = {
            "predicted_recovery_score": round(prediction, 2),
            "important_factor": most_important_feature,
            "therapy_suggestion": f"Focus on {most_important_feature} for best recovery."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
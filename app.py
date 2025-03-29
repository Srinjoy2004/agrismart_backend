from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

from flask_cors import CORS
CORS(app)


# Setup logging
logging.basicConfig(level=logging.INFO)

# Load models and data
try:
    model = joblib.load("crop_prob_model.pkl")
    label_mapping = joblib.load("crop_label_mapping.pkl")
    fertilizer_df = pd.read_csv("fertilizer.csv").set_index("Crop")
except Exception as e:
    logging.error(f"Error loading model or data: {e}")
    raise

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDD8QW1BggDVVMLteDygHCHrD6Ff9Dy0e8 "
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def gemini_fertilizer_advice(predicted_crop, n, p, k, ideal_n, ideal_p, ideal_k):
    """
    Uses Google's Gemini AI to generate fertilizer recommendations.
    """
    try:
        prompt = f"""
        A farmer wants to grow {predicted_crop}.
        Ideal soil NPK values: Nitrogen={ideal_n}, Phosphorus={ideal_p}, Potassium={ideal_k}.
        Current soil levels: Nitrogen={n}, Phosphorus={p}, Potassium={k}.
        Explain what nutrients are deficient and recommend appropriate fertilizers.
        Include names of commonly used fertilizers in India.
        Keep it short within 50 words.
        """
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Could not generate fertilizer advice. Try again later."
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return "Fertilizer recommendation is unavailable at the moment."
        return {e}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        required_keys = {"N", "P", "K", "temperature", "humidity", "pH", "rainfall"}

        # Validate input data
        missing_keys = required_keys - data.keys()
        if missing_keys:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_keys)}"}), 400

        # Extract input features
        features = np.array([[data["N"], data["P"], data["K"], data["temperature"], 
                              data["humidity"], data["pH"], data["rainfall"]]])
        
        # Predict probabilities
        probs = model.predict_proba(features)[0]
        
        # Get Top-5 recommended crops
        top5_indices = np.argsort(probs)[::-1][:5]
        top5 = [{"crop": label_mapping[i], "probability": round(probs[i] * 100, 2)} for i in top5_indices]

        return jsonify({
            "recommendations": [crop["crop"] for crop in top5]
        })
    
    except Exception as e:
        logging.error(f"Recommendation Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    """
    Receives the selected crop from the frontend and returns fertilizer advice.
    """
    try:
        # Check if request contains JSON
        if not request.is_json:
            logging.error("Invalid request format: Expected JSON")
            return jsonify({"error": "Invalid request format, expected JSON"}), 400

        data = request.json
        selected_crop = data.get("crop")
        n, p, k = data.get("N"), data.get("P"), data.get("K")

        # Validate input fields
        if not selected_crop:
            logging.error("Missing crop selection")
            return jsonify({"error": "Crop selection is required"}), 400
        if None in [n, p, k]:
            logging.error("Missing NPK values")
            return jsonify({"error": "Soil NPK values (N, P, K) are required"}), 400

        # Ensure fertilizer_df is available
        if "fertilizer_df" not in globals():
            logging.error("Fertilizer data (fertilizer_df) is not loaded")
            return jsonify({"error": "Fertilizer data is not loaded"}), 500

        # Convert crop names to lowercase to avoid case sensitivity issues
        selected_crop = selected_crop.strip().lower()
        fertilizer_df.index = fertilizer_df.index.str.strip().str.lower()

        # Check if crop exists in DataFrame
        if selected_crop not in fertilizer_df.index:
            logging.error(f"No fertilizer data found for crop: {selected_crop}")
            return jsonify({"error": f"No fertilizer data available for {selected_crop}"}), 404

        # Fetch ideal NPK values for the selected crop
        ideal_n, ideal_p, ideal_k = fertilizer_df.loc[selected_crop, ["N", "P", "K"]]

        # Ensure the function exists
        if "gemini_fertilizer_advice" not in globals():
            logging.error("Missing function: gemini_fertilizer_advice")
            return jsonify({"error": "Fertilizer advice function is missing"}), 500

        # Get fertilizer recommendation
        fertilizer_advice = gemini_fertilizer_advice(selected_crop, n, p, k, ideal_n, ideal_p, ideal_k)

        logging.info(f"Fertilizer advice for {selected_crop}: {fertilizer_advice}")
        return jsonify({"fertilizer_advice": fertilizer_advice})

    except Exception as e:
        logging.error(f"Unexpected error in /fertilizer route: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request."}), 500


if __name__ == '__main__':
    app.run(debug=False)  # Disable debug mode in production
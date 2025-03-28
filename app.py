from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
from flask import Flask, render_template

app = Flask(__name__)

# Load crop model & label mapping
model = joblib.load("crop_prob_model.pkl")
label_mapping = joblib.load("crop_label_mapping.pkl")

# Load fertilizer reference data
fertilizer_df = pd.read_csv("fertilizer.csv")
fertilizer_df.set_index("Crop", inplace=True)

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDD8QW1BggDVVMLteDygHCHrD6Ff9Dy0e8 "  # Replace safely
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def gemini_fertilizer_advice(predicted_crop, n, p, k, ideal_n, ideal_p, ideal_k):
    prompt = f"""
    A farmer wants to grow {predicted_crop}.
    Ideal soil NPK values are: Nitrogen={ideal_n}, Phosphorus={ideal_p}, Potassium={ideal_k}.
    Current soil levels are: Nitrogen={n}, Phosphorus={p}, Potassium={k}.
    Explain what nutrients are deficient and recommend appropriate fertilizers.
    Include names of commonly used fertilizers in India.
    Keep the explanation simple and helpful.
    Keep it short within 50 words.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    n, p, k, temp, humidity, ph, rainfall = data["N"], data["P"], data["K"], data["temperature"], data["humidity"], data["pH"], data["rainfall"]
    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    probs = model.predict_proba(features)[0]
    
    # Get Top-5 crops
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [{"crop": label_mapping[i], "probability": round(probs[i] * 100, 2)} for i in top5_indices]
    top_crop = top5[0]["crop"]
    
    # Fertilizer recommendation
    try:
        ideal = fertilizer_df.loc[top_crop]
        ideal_n, ideal_p, ideal_k = ideal["N"], ideal["P"], ideal["K"]
        
        if n < ideal_n or p < ideal_p or k < ideal_k:
            advice = gemini_fertilizer_advice(top_crop, n, p, k, ideal_n, ideal_p, ideal_k)
        else:
            advice = "NPK levels are optimal. No additional fertilizer required."
    except KeyError:
        advice = "No fertilizer data available for this crop."
    
    return jsonify({"top_5_crops": top5, "fertilizer_advice": advice})

if __name__ == '__main__':
    app.run(debug=True)

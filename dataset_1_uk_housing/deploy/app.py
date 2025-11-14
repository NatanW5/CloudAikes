import joblib
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  # <-- Import pandas

# 1. Initialize the Flask app
app = Flask(__name__)

# 2. Load your model
# Make sure 'your_model.pkl' is in the same directory as this script
try:
    # I'm using your updated path
    model = joblib.load(r'models\uk_housing_rf_pipeline.pkl') 
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(r"❌ Model file not found. Make sure 'models\uk_housing_rf_pipeline.pkl' is in the correct path.")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- je bestaande code om het model te laden blijft hetzelfde ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded, check server logs.'}), 500

    try:
        data = request.form  # form-data in plaats van JSON

        # De exacte features van het model
        features_columns = [
            "year",
            "month",
            "town_city",
            "county",
            "ptype_Detached",
            "ptype_Flats/Maisonettes",
            "ptype_Other",
            "ptype_Semi-Detached",
            "ptype_Terraced"
        ]

        # Zet form-data om naar een dict met juiste types
        input_data = {
            "year": int(data["year"]),
            "month": int(data["month"]),
            "town_city": data["town_city"],
            "county": data["county"],
            "ptype_Detached": 1 if data["property_type"] == "Detached" else 0,
            "ptype_Flats/Maisonettes": 1 if data["property_type"] == "Flats/Maisonettes" else 0,
            "ptype_Other": 1 if data["property_type"] == "Other" else 0,
            "ptype_Semi-Detached": 1 if data["property_type"] == "Semi-Detached" else 0,
            "ptype_Terraced": 1 if data["property_type"] == "Terraced" else 0
        }

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted price: £{output:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')



# 4. Run the app
if __name__ == '__main__':
    # 'host=0.0.0.0' makes it accessible
    # I added debug=True so it automatically reloads when you save changes
    app.run(host='0.0.0.0', port=5001, debug=True)
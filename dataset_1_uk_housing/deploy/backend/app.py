import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime

# 1. Initialize the Flask app
app = Flask(__name__)
CORS(app)

# 2. Model Loading Logic
# Zorg dat dit bestand in de map 'models' staat
MODEL_PATH = 'models/uk_housing_price_catboost.pkl' 
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ FOUT: Modelbestand niet gevonden op pad: {MODEL_PATH}")
        return None

    try:
        # CatBoost modellen uit de notebook zijn opgeslagen met pickle
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✅ CatBoost Model geladen: '{MODEL_PATH}' succesvol!")
        return model
    except Exception as e:
        print(f"❌ FOUT: Kon het model niet laden. Detail: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Serverfout: Model is niet geladen.')

    try:
        data = request.form

        # --- STAP 1: Validatie en Data ophalen ---
        try:
            year = int(data.get("year"))
            month = int(data.get("month"))
        except (ValueError, TypeError):
             return render_template('index.html', prediction_text='Invoerfout: Jaar en maand moeten getallen zijn.')

        # --- STAP 2: Feature Engineering (zoals in je notebook) ---
        
        # 2a. Bereken date_numeric (Unix timestamp)
        # We nemen de 1e dag van de maand als schatting
        try:
            dt_obj = datetime(year, month, 1)
            date_numeric = int(dt_obj.timestamp())
        except ValueError:
            return render_template('index.html', prediction_text='Invoerfout: Ongeldige datum.')

        # 2b. Inputs verzamelen
        # CatBoost verwacht categorische features als strings (geen One-Hot Encoding nodig!)
        # Let op: De namen van de keys (bijv "town") moeten overeenkomen met de 'name' attributen in je HTML form.
        
        input_data = {
            "district": data.get("district", ""),        # Nieuw veld nodig in HTML
            "town": data.get("town_city", ""),           # Map 'town_city' uit HTML naar 'town' voor model
            "county": data.get("county", ""),
            "month": month,
            "year": year,
            "property_type": data.get("property_type"),  # Bijv: "Detached", "Terraced"
            "tenure": data.get("tenure", "Freehold"),    # Nieuw veld nodig (Freehold/Leasehold)
            "new_build_flag": data.get("new_build_flag", "N"), # Nieuw veld nodig (Y/N)
            "date_numeric": date_numeric
        }

        # --- STAP 3: DataFrame maken ---
        # De volgorde van kolommen moet exact kloppen met de training
        expected_columns = [
            "district", "town", "county", "month", "year", 
            "property_type", "tenure", "new_build_flag", "date_numeric"
        ]
        
        input_df = pd.DataFrame([input_data])
        
        # Zorg voor de juiste kolomvolgorde
        input_df = input_df[expected_columns]

        # --- STAP 4: Voorspellen ---
        prediction = model.predict(input_df)
        output = round(prediction[0], 2)

        return jsonify({"prediction": f"Voorspelde prijs: £{output:,.2f}"})

    except Exception as e:
        print(f"❌ FOUT tijdens predictie: {e}")
        return render_template('index.html', prediction_text=f'Interne fout: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
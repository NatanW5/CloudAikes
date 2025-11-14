import os
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

# 1. Initialize the Flask app
app = Flask(__name__)

# 2. Model Loading Logic
MODEL_PATH = 'models/uk_housing_rf_pipeline.pkl'
model = None

# Functie om het model te laden en fouten te loggen
def load_model():
    global model
    
    # 2.1 Controleer of het bestand bestaat
    if not os.path.exists(MODEL_PATH):
        print(f"❌ FOUT: Modelbestand niet gevonden op pad: {MODEL_PATH}")
        # Toon de huidige werkmap van de server (handig voor debugging)
        print(f"Huidige werkmap van de server: {os.getcwd()}") 
        print(f"Inhoud van de map 'models': {os.listdir('models') if os.path.isdir('models') else 'Map bestaat niet.'}")
        return None

    # 2.2 Probeer het model te laden
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model geladen: 'uk_housing_rf_pipeline.pkl' succesvol!")
        return model
    except Exception as e:
        print(f"❌ FOUT: Onverwachte fout bij het deserialiseren van het modelbestand.")
        print(f"Detail: {e}")
        return None

# Voer de laadfunctie uit bij het opstarten van de applicatie
model = load_model()

# --- De Flask routes blijven hierna hetzelfde ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Model is al globaal geladen, maar dubbelcheck de status
    if model is None:
        # Als de applicatie op dit punt komt zonder model,
        # is er een ernstige opstartfout.
        return render_template('index.html', prediction_text='Serverfout: Model kon niet geladen worden. Controleer server logs.')

    try:
        data = request.form  # form-data in plaats van JSON

        # VALIDATIE STAP 1: Zorg ervoor dat getallen zijn ingevoerd
        try:
            year = int(data["year"])
            month = int(data["month"])
        except ValueError:
             return render_template('index.html', prediction_text='Invoerfout: Jaar en maand moeten geldige getallen zijn.')

        # VALIDATIE STAP 2: Controleer de logische grenzen van Jaar en Maand
        if not (1995 <= year <= 2025):
             return render_template('index.html', prediction_text='Invoerfout: Jaar moet tussen 1995 en 2025 liggen.')
        if not (1 <= month <= 12):
             return render_template('index.html', prediction_text='Invoerfout: Maand moet tussen 1 en 12 liggen.')

        # De exacte features van het model (onveranderd)
        features_columns = [
            "year", "month", "town_city", "county", 
            "ptype_Detached", "ptype_Flats/Maisonettes", 
            "ptype_Other", "ptype_Semi-Detached", "ptype_Terraced"
        ]

        # Zet form-data om naar een dict met juiste types
        input_data = {
            "year": year,
            "month": month,
            "town_city": data["town_city"],
            "county": data["county"],
            "ptype_Detached": 1 if data["property_type"] == "Detached" else 0,
            "ptype_Flats/Maisonettes": 1 if data["property_type"] == "Flats/Maisonettes" else 0,
            "ptype_Other": 1 if data["property_type"] == "Other" else 0,
            "ptype_Semi-Detached": 1 if data["property_type"] == "Semi-Detached" else 0,
            "ptype_Terraced": 1 if data["property_type"] == "Terraced" else 0
        }

        input_df = pd.DataFrame([input_data])

        # Gebruik van het model
        prediction = model.predict(input_df)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Voorspelde prijs: £{output:,.2f}')

    except Exception as e:
        # Dit vangt fouten op die tijdens de predictie zelf optreden (bijv. door foutieve model input)
        print(f"❌ FOUT tijdens predictie: {e}")
        return render_template('index.html', prediction_text=f'Interne fout bij voorspelling: {str(e)}')


# 4. Run the app met Gunicorn compatibele instellingen
if __name__ == '__main__':
    # Gunicorn negeert deze blok in productie, maar dit is voor lokale tests
    app.run(host='0.0.0.0', port=5001, debug=True)
import os
import pickle
import traceback
import pandas as pd
import xgboost as xgb
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 1. Initialize Flask
app = Flask(__name__)
CORS(app)

# 2. Configuratie & Paden
# Gebruik paden relatief aan deze file (de `deploy` map). Hierdoor werkt
# de app ook wanneer je vanuit een andere werkmap start.
DEPLOY_DIR = os.path.abspath(os.path.dirname(__file__))
# Model en mappings bevinden zich in `deploy/models/`
MODEL_PATH = os.path.join(DEPLOY_DIR, 'models', 'xgboost_housing_v1.json')
MAPPING_PATH = os.path.join(DEPLOY_DIR, 'models', 'encoding_mappings.pkl')

model = None
mappings = None

def load_resources():
    global model, mappings
    
    # A. Model laden (XGBoost specifiek)
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBRegressor()
            model.load_model(MODEL_PATH)
            print(f"✅ XGBoost model geladen: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Fout bij laden XGBoost model: {e}")
            traceback.print_exc()
    else:
        print(f"❌ CRITISCH: Modelbestand niet gevonden op: {MODEL_PATH}")

    # B. Mappings laden (Pickle)
    if os.path.exists(MAPPING_PATH):
        try:
            with open(MAPPING_PATH, "rb") as f:
                mappings = pickle.load(f)
            print(f"✅ Encoding mappings geladen.")
        except Exception as e:
            print(f"❌ Fout bij laden mappings: {e}")
    else:
        print(f"❌ CRITISCH: Mapping bestand niet gevonden op: {MAPPING_PATH}")

# Direct laden bij opstarten
load_resources()

# Hardcoded mappings voor categorieën (gebaseerd op alfabetische volgorde van Notebook 03)
# Als notebook 03 cat.codes heeft gebruikt, is het meestal alfabetisch.
# D=0, F=1, O=2, S=3, T=4
PROPERTY_TYPE_MAP = {'D': 0, 'F': 1, 'O': 2, 'S': 3, 'T': 4}
# N=0, Y=1
OLD_NEW_MAP = {'N': 0, 'Y': 1}
# F=0, L=1, U=2 (Unknown)
DURATION_MAP = {'F': 0, 'L': 1, 'U': 2}
# A=0, B=1 (Standaard / Additional)
PPD_MAP = {'A': 0, 'B': 1}
# A=0 (Add) - We nemen aan dat nieuwe voorspellingen 'Additions' zijn
RECORD_STATUS_MAP = {'A': 0} 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or mappings is None:
        return jsonify({"error": "Model of mappings niet geladen op server."}), 500

    try:
        data = request.form
        print("📩 Request:", data)

        # 1. Inputs Ophalen
        try:
            year = int(data.get("year"))
            month = int(data.get("month"))
            prop_type = data.get("property_type") # D, S, T, F, O
            old_new = data.get("old_new")         # Y, N
            duration = data.get("duration")       # F, L
            county = data.get("county").strip()
            district = data.get("district").strip()
        except (ValueError, TypeError) as e:
            return jsonify({"error": "Validatiefout", "detail": str(e)}), 400

        # 2. Feature Engineering & Encoding
        
        # A. Categorieën omzetten naar nummers
        pt_code = PROPERTY_TYPE_MAP.get(prop_type, 4) # Default to T if unknown
        on_code = OLD_NEW_MAP.get(old_new, 0)
        dur_code = DURATION_MAP.get(duration, 0)
        ppd_code = 0 # Default Standard
        rec_code = 0 # Default Addition

        # B. Target Encoding voor Locatie (Cruciaal!)
        # We kijken in de geladen mappings. Als de county niet bestaat, pakken we het globale gemiddelde.
        
        # Ophalen van de series uit de pickle
        county_means = mappings.get('county_means') if isinstance(mappings, dict) else None
        district_means = mappings.get('district_means') if isinstance(mappings, dict) else None
        global_mean = 250000 # Fallback als alles faalt (ongeveer gemiddelde UK prijs)

        # Helper: veilig mean ophalen ongeacht of mapping een dict of Series is
        def _get_encoded(mapping, key, fallback=global_mean):
            if mapping is None:
                return fallback
            # Pandas Series / dict-like
            try:
                # Series/Index/Mapping with .get and .mean
                if hasattr(mapping, 'get') and hasattr(mapping, 'mean'):
                    return mapping.get(key, mapping.mean())
                # Plain dict: bereken mean van values
                if isinstance(mapping, dict):
                    vals = list(mapping.values())
                    if len(vals) > 0:
                        return mapping.get(key, float(np.mean(vals)))
                    return fallback
            except Exception:
                pass
            return fallback

        county_encoded = _get_encoded(county_means, county)
        district_encoded = _get_encoded(district_means, district)

        # 3. DataFrame samenstellen
        # De volgorde MOET exact hetzelfde zijn als X_train in Notebook 03
        # Features: ['property_type', 'old_new', 'duration', 'ppd_category', 'record_status', 
        #            'year', 'month', 'county_encoded', 'district_encoded']
        
        input_data = {
            'property_type': [pt_code],
            'old_new': [on_code],
            'duration': [dur_code],
            'ppd_category': [ppd_code],
            'record_status': [rec_code],
            'year': [year],
            'month': [month],
            'county_encoded': [county_encoded],
            'district_encoded': [district_encoded]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Zeker zijn van kolomvolgorde
        cols_order = ['property_type', 'old_new', 'duration', 'ppd_category', 'record_status', 
                      'year', 'month', 'county_encoded', 'district_encoded']
        input_df = input_df[cols_order]

        print("📊 Data naar XGBoost:", input_df.values)

        # 4. Predictie
        prediction = model.predict(input_df)
        price = float(prediction[0])
        
        # Geen negatieve huizenprijzen
        price = max(price, 0)

        return jsonify({
            "prediction": f"£{price:,.2f}",
            "details": f"Locatie factor: {district} (£{district_encoded:,.0f} avg)",
            "status": "success"
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ SERVER ERROR:", tb)
        return jsonify({"error": "Interne Fout", "detail": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
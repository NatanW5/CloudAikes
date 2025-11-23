import os
import pickle
import traceback # Nodig om de volledige fout te zien
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime

# 1. Initialize Flask
app = Flask(__name__)
CORS(app) # Staat requests van andere domeinen toe (belangrijk voor frontend)

# 2. Model Loading
MODEL_PATH = 'models/uk_housing_price_catboost.pkl'
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ CRITISH: Modelbestand niet gevonden op: {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model geladen: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ CRITISH: Fout bij laden model: {e}")
        traceback.print_exc()
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Debugging: Print wat er binnenkomt
    print("📩 Ontvangen request form data:", request.form)

    # 1. Check of model er is
    if model is None:
        return jsonify({
            "error": "Model niet geladen op de server.",
            "detail": "Controleer of het .pkl bestand bestaat in de map 'models'."
        }), 500

    try:
        data = request.form

        # 2. Validatie & Conversie
        try:
            year = int(data.get("year"))
            month = int(data.get("month"))
        except (ValueError, TypeError):
            return jsonify({"error": "Validatiefout", "detail": "Jaar en maand moeten getallen zijn."}), 400

        # 3. Feature Engineering
        try:
            dt_obj = datetime(year, month, 1)
            date_numeric = int(dt_obj.timestamp())
        except ValueError:
             return jsonify({"error": "Datumfout", "detail": f"Ongeldige datum: {year}-{month}"}), 400

        # 4. Data voorbereiden
        # LET OP: Zorg dat deze keys exact overeenkomen met je HTML form names
        input_data = {
            "district": data.get("district", "").strip(),
            "town": data.get("town_city", "").strip(),
            "county": data.get("county", "").strip(),
            "month": month,
            "year": year,
            "property_type": data.get("property_type"),
            "tenure": data.get("tenure"),
            "new_build_flag": data.get("new_build_flag"),
            "date_numeric": date_numeric
        }
        
        # 5. DataFrame maken
        expected_columns = [
            "district", "town", "county", "month", "year", 
            "property_type", "tenure", "new_build_flag", "date_numeric"
        ]
        
        input_df = pd.DataFrame([input_data])
        
        # Controleer op ontbrekende kolommen
        missing_cols = [col for col in expected_columns if col not in input_df.columns]
        if missing_cols:
             return jsonify({"error": "Datafout", "detail": f"Ontbrekende kolommen: {missing_cols}"}), 400

        # Zorg voor juiste volgorde
        input_df = input_df[expected_columns]

        print("📊 Data naar model:", input_df.to_dict(orient='records'))

        # 6. Voorspellen
        prediction = model.predict(input_df)
        output = round(prediction[0], 2)

        return jsonify({
            "prediction": f"£{output:,.2f}",
            "status": "success"
        })

    except Exception as e:
        # Vang ALLES op en stuur het terug naar de frontend
        tb = traceback.format_exc()
        print("❌ SERVER ERROR:", tb) # Print in Azure logs
        return jsonify({
            "error": "Interne Server Fout",
            "detail": str(e),
            "traceback": tb
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
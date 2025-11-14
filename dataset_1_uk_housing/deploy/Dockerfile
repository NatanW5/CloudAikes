# 1. Gebruik een lichtgewicht Python versie als basis
FROM python:3.9-slim

# 2. Maak een werkmap aan binnen de container
WORKDIR /app

# 3. Kopieer eerst de requirements (zodat Docker dit slim kan cachen)
COPY requirements.txt .

# 4. Installeer de libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kopieer nu de rest van je bestanden (app.py, models, templates)
COPY . .

# 6. Vertel Docker op welke poort de app draait (Render gebruikt vaak 10000 of 5001)
EXPOSE 5001

# 7. Het commando om de app te starten met Gunicorn (professioneler dan python app.py)
# We binden aan 0.0.0.0 op poort 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
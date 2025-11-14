# Start met een lichtgewicht Python versie als basis
FROM python:3.12.10-slim

# NIEUWE STAP: Update het onderliggende besturingssysteem
# Hiermee worden de beveiligingsupdates voor de Linux componenten geïnstalleerd.
# Het '-y' zorgt ervoor dat de vragen automatisch met 'ja' beantwoord worden.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean

# 1. Maak een werkmap aan binnen de container
WORKDIR /app

# 2. Kopieer eerst de requirements (zodat Docker dit slim kan cachen)
COPY requirements.txt .

# 3. Installeer de Python libraries
# We gebruiken --no-cache-dir om de image klein te houden
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kopieer nu de rest van je bestanden (app.py, models, templates)
# De Git LFS bestanden worden hier als laatste gedownload in de Docker build.
COPY . .

# 5. Vertel Docker op welke poort de app draait (Render gebruikt vaak 10000 of 5001)
EXPOSE 5001

# 6. Het commando om de app te starten met Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
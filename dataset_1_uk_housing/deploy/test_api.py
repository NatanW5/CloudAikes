import requests

url = "http://127.0.0.1:5001/predict"

data = {
    "year": 2023,
    "month": 5,
    "town_city": "London",
    "county": "Greater London",
    "ptype_Detached": 0,
    "ptype_Flats/Maisonettes": 1,
    "ptype_Other": 0,
    "ptype_Semi-Detached": 0,
    "ptype_Terraced": 0
}

response = requests.post(url, json=data)
print("Status code:", response.status_code)
print("Response:", response.json())

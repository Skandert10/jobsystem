import json
import requests
from ml_as_api import jobPred
import pandas as pd


import requests

# Données à envoyer au modèle
input_data_for_model = {
    "Gender": 1,
    "Compétences": "HTML, CSS, Java, PHP, Python",
    "Institution": "ESPRIT",
    "Année_expériences": 1,
    "Langues": "Arabe - Francais - Anglais",
    "Formation": "ingénierie informatique",
    "Expériences": "stage d'été chez délice"
}

# URL du serveur local
url = "http://127.0.0.1:8000/jobPred"

# Envoi de la requête POST avec les données JSON
response = requests.post(url, json=input_data_for_model)

# Vérification du code de réponse
if response.status_code == 200:
    # Récupération de la prédiction du modèle à partir de la réponse
    prediction = response.json()
    print(prediction)
else:
    print("Erreur lors de la requête :", response.status_code)

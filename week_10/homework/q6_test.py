import requests
from time import sleep

url = "http://localhost:9696/predict"

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)

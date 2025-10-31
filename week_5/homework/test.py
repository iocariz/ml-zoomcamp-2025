import requests

url = 'http://localhost:9696/predict'
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'

customer =  {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=customer)

predictions = response.json()

if predictions['churn']:
    print('customer is likely to churn, send promo')
else:
    print('customer is not likely to churn')

print(f"Churn probability: {predictions['churn_probability']:.5f}")
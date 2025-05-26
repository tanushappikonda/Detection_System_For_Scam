import requests

api_url = "https://full-backend-scam-detection-system.onrender.com/predict"
data = {"text": "You've won a free iPhone! Click now!"}

response = requests.post(api_url, json=data)
print(response.json())  # Output: {"prediction": "scam", "confidence": 0.95}
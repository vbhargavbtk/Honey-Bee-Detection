import os
import requests

token = "8460169438:AAH2VNPBqX5c0LtkaBuKk1Da4ACperd2VvQ"
chat_id = "1116086962"

url = f"https://api.telegram.org/bot{token}/sendMessage"
data = {"chat_id": chat_id, "text": "🐝 Test message from Bee Detection App"}

resp = requests.post(url, data=data)
print(resp.status_code, resp.text)

import requests
import json

response = requests.get("http://127.0.0.1:8000/transcript/")
data = response.json()

with open("transcript.json", "w") as f:
    json.dump(data["transcript"], f, indent=2)

print(f"Saved {data['count']} segments to transcript.json")
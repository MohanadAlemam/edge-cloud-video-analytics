import requests
import json
#api_url = "http://localhost:5000/hello"
api_url = "http://10.0.0.3:5000/hello"
response = requests.get(api_url)
rd = response.json()
pretty = json.dumps(rd, indent=4)
print(pretty)

import requests

url = "http://localhost:8000/api/enroll_voice"
files = {'file': ('example.wav', open('models/spkrec-ecapa-voxceleb/example1.wav', 'rb'), 'audio/wav')}
data = {'user_id': 'user_123'}

response = requests.post(url, files=files, data=data)
print(response.status_code)
print(response.text)

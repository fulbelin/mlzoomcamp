import base64, json, requests

# Read + encode image
with open("test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {"image": b64}

res = requests.post(
    "http://localhost:9000/2015-03-31/functions/function/invocations",
    json=payload
)

print(res.json())

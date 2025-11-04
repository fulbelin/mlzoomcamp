from fastapi import FastAPI
import pickle

app = FastAPI()

with open("/pipeline_v2.bin", "rb") as f:
    pipeline = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Lead conversion prediction API is running!"}

@app.post("/predict")
def predict(client: dict):
    X_new = [client]
    probabilities = pipeline.predict_proba(X_new)[0]
    convert_prob = float(probabilities[1])
    return {"convert_probability": convert_prob}

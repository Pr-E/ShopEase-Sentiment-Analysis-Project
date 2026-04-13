from fastapi import FastAPI, HTTPException
from src.shopease_app.api.inference import SentimentPredictor
from src.shopease_app.api.schemas import (
    TextRequest,
    BatchRequest,
    PredictionResponse,
    BatchResponse
)

app = FastAPI(title="ShopEase Sentiment API")

# Load model once at startup
predictor = SentimentPredictor()


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


# 🔹 Single Prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    return predictor.predict(request.text)


# 🔹 Batch Prediction
@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Empty input list")

    predictions = predictor.batch_predict(request.texts)
    return {"predictions": predictions}
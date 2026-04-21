import logging
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd

from pipeline.prediction import PredictSentiment
from pipeline.training import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="ShopEase Sentiment API")


# Request Schema
class TextRequest(BaseModel):
    text: str


# Load model once
predictor = PredictSentiment()
logging.info("Model successfully loaded.")


# Health Check
@app.get("/")
def home():
    return {"message": "API is running"}


# Single Prediction
@app.post("/predict_sentiment")
def predict_text(request: TextRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty input text")

        results = predictor.predict(request.text)

        if not results:
            raise HTTPException(status_code=500, detail="Prediction failed")

        # Extract top prediction
        top_label = max(results, key=lambda x: x["score"])
        
        print(top_label)
        print(results)
        return {
            "label": top_label["label"],
            "confidence": float(top_label["score"])
        }

    except Exception as e:
        logging.error(f"Error while predicting sentiment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Batch Prediction (CSV Upload)
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if "review" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain a 'review' column"
            )

        results_list = []

        for _, row in df.iterrows():
            try:
                review = str(row["review"])
                results = predictor.predict(review)

                if results is None or len(results) ==0:
                    raise ValueError ("Empty result from model")
                

                top_label = max(results, key=lambda x: x["score"])

                result_row = row.to_dict()
                result_row["sentiment_label"] = top_label["label"]
                result_row["confidence_label"] = float(top_label["score"])

                results_list.append(result_row)

            except Exception as e:
                logging.error(f"Error during batch prediction: {e}")

        return {"predictions": results_list}

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")


# Trigger Training 
@app.get("/train")
def train_model():
    try:
        train_and_evaluate()
        return {"message": "Training completed successfully"}

    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise HTTPException(status_code=500, detail="Training failed")
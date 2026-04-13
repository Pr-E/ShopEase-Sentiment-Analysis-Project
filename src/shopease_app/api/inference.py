import torch
import torch.nn.functional as F
from transformers import BertTokenizer, AutoModelForSequenceClassification
from src.shopease_app.data_cleaning import DataCleaning
from config.constant import model_name
import os

MODEL_PATH = "./sentiment_model"


class SentimentPredictor:
    def __init__(self):
        self.device = torch.device("cpu")

        # Load trained model + tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        self.model.to(self.device)
        self.model.eval()

        # Reuse preprocessing
        self.cleaner = DataCleaning()

        # Label mapping
        self.label_map = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    def preprocess(self, text: str):
        text = self.cleaner.clean_text(text)
        text = self.cleaner.lemmatize(text)
        text = self.cleaner.remove_stopwords(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        return encoding

    def predict(self, text: str):
        inputs = self.preprocess(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        return {
            "label": self.label_map[pred],
            "class_id": pred,
            "confidence": round(confidence, 4)
        }

    def batch_predict(self, texts: list):
        return [self.predict(text) for text in texts]    
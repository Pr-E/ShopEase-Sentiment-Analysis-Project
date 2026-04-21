import logging
from utils.models_utils import load_registered_model  
from src.shopease_app.data_cleaning import DataCleaning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class PredictSentiment:
    def __init__(self):
        try:
            self.pipeline = load_registered_model()
            self.cleaner = DataCleaning()

            # Label mapping
            self.id2label = {
                0: "Negative",
                1: "Neutral",
                2: "Positive"
            }

            logging.info("Prediction pipeline loaded successfully.")

        except Exception as e:
            logging.error(f"Error initializing prediction pipeline: {e}")

    def predict(self, text: str):
        try:
            # Preprocess input (same as training)
            cleaned_text = self.cleaner.clean_text(text)
            cleaned_text = self.cleaner.lemmatize(cleaned_text)
            cleaned_text = self.cleaner.remove_stopwords(cleaned_text)

            raw_results = self.pipeline(cleaned_text)

            # Map labels to readable format
            for item in raw_results:    
                index = int(item["label"].split("_")[-1])
                item["label"] = self.id2label.get(index, item["label"])

            return raw_results

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

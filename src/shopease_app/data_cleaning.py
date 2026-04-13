 
import re
import pandas as pd
import logging
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config.constant import Cleaned_Data
from src.shopease_app.data_ingestion import data_ingestion

logging.basicConfig(level=logging.INFO)


class DataCleaning:
    def __init__(self):
        self._ensure_nltk()
        self.nlp = self._load_nlp()

    # ----------------------------
    # Load spaCy model safely
    # ----------------------------
    def _load_nlp(self):
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        return spacy.blank("xx")

    # ----------------------------
    # Ensure NLTK resources
    # ----------------------------
    def _ensure_nltk(self):
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")

        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            try:
                nltk.download("punkt_tab")
            except Exception:
                pass

    # ----------------------------
    # CLEAN TEXT
    # ----------------------------
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ----------------------------
    # LEMMATIZATION
    # ----------------------------
    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join(
            token.lemma_ if token.lemma_ else token.text for token in doc
        )

    # ----------------------------
    # STOPWORD REMOVAL
    # ----------------------------
    def remove_stopwords(self, text: str) -> str:
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)


# ----------------------------
# PIPELINE FUNCTION
# ----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cleaner = DataCleaning()

        df["clean_text"] = df["review"].apply(cleaner.clean_text)
        df["lemma_text"] = df["clean_text"].apply(cleaner.lemmatize)
        df["final_text"] = df["lemma_text"].apply(cleaner.remove_stopwords)

        # ----------------------------
        # SENTIMENT LABEL CREATION
        # ----------------------------
        df["sentiment_label"] = df["rating"].apply(
            lambda r: 0 if r in [1, 2] else (1 if r == 3 else 2)
        )

        # Keep only relevant columns
        df = df[["review", "final_text", "sentiment_label"]]

        # Save cleaned data
        df.to_csv(Cleaned_Data, index=False)

        logging.info("Data successfully cleaned and saved.")
        print(df.head(5))
        return df

    except Exception as e:
        logging.error(f"Error occurred while cleaning the data: {e}")
        raise


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    df = data_ingestion()
    df = clean_data(df)
    print(df.head())    
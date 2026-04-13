from transformers import BertTokenizer
import pandas as pd
import torch
import logging
import os

from sklearn.model_selection import train_test_split

from config.constant import Input_Data, Cleaned_Data, model_name, truncation, max_length, padding, Train_Data, Test_Data
from src.shopease_app.data_cleaning import clean_data
logging.basicConfig(level=logging.INFO)



# DATA PROCESSOR

class DataProcessor:
    def __init__(self):
        try:
            if os.path.exists(Cleaned_Data):
                logging.info("Loading cleaned data from file...")
                self.data = pd.read_csv(Cleaned_Data)
            else:
                logging.info("Cleaned data not found. Running cleaning pipeline...")
                raw_data = pd.read_csv(Input_Data)
                self.data = clean_data(raw_data)

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def split_data(self):
        try:
            X = self.data["final_text"].astype(str)
            y = self.data["sentiment_label"]

            # Verify class label
            logging.info("Label distribution:")
            logging.info(self.data["sentiment_label"].value_counts())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logging.info("Data successfully split.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error occurred while splitting dataset: {e}")
            raise


# TOKENIZER

class Tokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
    def encode(self, text):
        return self.tokenizer(
            text.tolist(),
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors="pt"
        )

    def save(self, path="./sentiment_model"):
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        logging.info("Tokenizer saved successfully.")


# DATASET CLASS

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item



# PREP FUNCTION

def prepare_sentiment_data():
    try:
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.split_data()

        # Ensure labels are lists
        if hasattr(y_train, 'tolist'):
            y_train = y_train.tolist()
        if hasattr(y_test, 'tolist'):
            y_test = y_test.tolist()

        tokenizer = Tokenizer()

        train_encodings = tokenizer.encode(X_train)
        test_encodings = tokenizer.encode(X_test)

        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)

        # SAVE DATASETS
        os.makedirs(os.path.dirname(Train_Data), exist_ok=True)
        os.makedirs(os.path.dirname(Test_Data), exist_ok=True)
        torch.save(train_dataset, Train_Data)
        torch.save(test_dataset, Test_Data)

        tokenizer.save("./sentiment_model")
        
        logging.info(f"Data Successfully Prepared and Saved.")
        return train_dataset, test_dataset

    except Exception as e:
        logging.error(f"Error occurred while preparing dataset: {e}")
        raise


# ENTRY POINT

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_sentiment_data()
    print(train_dataset[0])
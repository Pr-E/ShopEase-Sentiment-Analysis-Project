from transformers import AutoTokenizer
import pandas as pd
import torch
import logging
import os

from sklearn.model_selection import train_test_split

from config.constant import (
    Input_Data,
    model_name,
    truncation,
    max_length,
    padding,
    Train_Data,
    Test_Data
)

from src.shopease_app.data_cleaning import clean_data

logging.basicConfig(level=logging.INFO)


# DATA PROCESSOR

class DataProcessor:
    def __init__(self):
        # Clean + load data
        self.data = clean_data(pd.read_csv(Input_Data))

    def split_data(self):
        try:
            X = self.data["final_text"].astype(str)
            y = self.data["sentiment_label"]

            logging.info("Label distribution:")
            logging.info(self.data["sentiment_label"].value_counts())

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer(
            text.tolist(),
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors="pt"
        )

    def save(self, path="./sentiment_model"):  # Save tokenizer
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


# PREPARATION FUNCTION

def prepare_sentiment_data():
    try:
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.split_data()

        # Convert labels safely
        y_train = y_train.tolist() if hasattr(y_train, "tolist") else y_train
        y_test = y_test.tolist() if hasattr(y_test, "tolist") else y_test

        tokenizer = Tokenizer()

        train_encodings = tokenizer.encode(X_train)
        test_encodings = tokenizer.encode(X_test)

        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)

        # Save datasets
        os.makedirs(os.path.dirname(Train_Data), exist_ok=True)
        os.makedirs(os.path.dirname(Test_Data), exist_ok=True)

        torch.save(train_dataset, Train_Data)
        torch.save(test_dataset, Test_Data)

        # Save tokenizer ONCE here
        tokenizer.save("./sentiment_model")

        logging.info("Data successfully prepared and saved.")

        return train_dataset, test_dataset

    except Exception as e:
        logging.error(f"Error occurred while preparing dataset: {e}")
        raise


# ENTRY POINT


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_sentiment_data()
    print(train_dataset[0])


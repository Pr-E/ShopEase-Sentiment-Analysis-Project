import torch
import logging
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

from src.shopease_app.data_preprocessing import prepare_sentiment_data
from config.constant import model_name, training_args, num_of_labels

logging.basicConfig(level=logging.INFO)


class Training:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_of_labels
        )

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")

        return {
            "accuracy": acc,
            "f1": f1
        }

    def model_training(self, train_dataset, test_dataset):
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics
            )

            trainer.train()

            trainer.save_model("./sentiment_model")

            logging.info("Model Successfully Trained and Saved.")
            return trainer

        except Exception as e:
            logging.error(f"Error Occurred While Training and Saving Model: {e}")
            raise


    def model_evaluation(self, trainer):
        try:
            results = trainer.evaluate()
            logging.info(f"Evaluation Results: {results}")
            return results

        except Exception as e:
            logging.error(f"Error Occurred During Evaluation: {e}")
            raise


def train_and_evaluate():
    try:
        train_dataset, test_dataset = prepare_sentiment_data()

        train = Training()

        trainer = train.model_training(
            train_dataset=train_dataset,
            test_dataset=test_dataset
        )

        results = train.model_evaluation(trainer)

        logging.info("Model Successfully Trained and Evaluated.")
        
        return results, trainer

    except Exception as e:
        logging.error(f"Error Occurred While Training and Evaluating Model: {e}")
        raise


# ENTRY POINT
if __name__ == "__main__":
    train_and_evaluate()
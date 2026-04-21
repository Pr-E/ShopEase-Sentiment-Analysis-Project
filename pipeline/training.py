import logging

from src.shopease_app.data_preprocessing import prepare_sentiment_data
from src.shopease_app.model_training import Training
from src.shopease_app.model_pusher import ModelPusher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_and_evaluate():
    try:
        # Prepare datasets
        train_dataset, test_dataset = prepare_sentiment_data()

        # Train model
        train = Training()
        trainer = train.model_training(
            train_dataset=train_dataset,
            test_dataset=test_dataset
        )

        # Evaluate model
        results = train.model_evaluation(trainer)

        logging.info(f"Evaluation Results: {results}")

        # Push model if improved
        model_pusher = ModelPusher()
        model_pusher.push_model(trainer, results)

    except Exception as e:
        logging.error(f"Error occurred during training pipeline: {e}")


# ENTRY POINT
if __name__ == "__main__":
    train_and_evaluate()
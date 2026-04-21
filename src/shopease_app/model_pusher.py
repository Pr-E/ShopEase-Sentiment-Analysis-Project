import mlflow
import mlflow.transformers
import dagshub
import logging
import os
from transformers import pipeline
from config.constant import training_args, model_name
from utils.models_utils import get_best_f1

from dotenv import load_dotenv
load_dotenv(override=True)

Model_Name = "distilbert-multilingual-sentiment"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelPusher:
    def __init__(self, experiment_name="Sentiment-analysis"):  
        try:
            # Local testing intialisatio process
           # dagshub.init(
            #    repo_owner="ejirogoro27",
             #   repo_name="ShopEase-Sentiment-Analysis-Project",
              #  mlflow=True
            #)
            # Remote initialisation/ Prodcution access to mlflow dagshub
            dagshub_token = os.getenv("shop_env_DAGSHUB_TOKEN")
            if not dagshub_token:
                raise EnvironmentError("shop_env_DAGSHUB_TOKEN environment variable is not set")
            
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            dagshub_url = "https://dagshub.com"
            repo_owner="ejirogoro27"
            repo_name="ShopEase-Sentiment-Analysis-Project"

            # Set up mlflow tracking url

            mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
        except Exception as e:
            logging.error(f"Error while initializing mlflow: {e}")




          #  self.experiment_name = experiment_name
           # mlflow.set_experiment(experiment_name)

            #logging.info("Model Pusher initialized successfully.")

        #except Exception as e:
         #   logging.error(f"Error occurred while initializing Model Pusher: {e}")

    def push_model(self, trainer, metrics):
        try:
            new_f1 = metrics.get("eval_f1", 0)
            old_f1 = get_best_f1(self.experiment_name)

            print(f"New F1 score: {new_f1}")
            print(f"Best Previous F1 score: {old_f1}")

            # Only push if model improves
            if old_f1 is None or new_f1 > old_f1:

                with mlflow.start_run():

                    # Log metrics
                    mlflow.log_metric("eval_accuracy", metrics.get("eval_accuracy", 0))
                    mlflow.log_metric("eval_f1", new_f1)

                    # Log parameters
                    mlflow.log_param("model_name", Model_Name)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)

                    # Create inference pipeline
                    sentiment_pipeline = pipeline(
                        task="text-classification",
                        model=trainer.model,
                        tokenizer=model_name,  # changed
                        return_all_scores=True
                    )

                    # Log model to MLflow
                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        name="model",
                        registered_model_name=Model_Name  # 
                    )


                logging.info("Model and metrics successfully pushed to MLflow.")

            else:
                logging.info("Model not pushed: performance did not improve.")

        except Exception as e:
            logging.error(f"Error occurred while pushing model: {e}")
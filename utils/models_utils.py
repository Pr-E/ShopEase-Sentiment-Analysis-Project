
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
import logging
import os
from config.constant import model_name  # OR IS IT Model_Name?

logging.basicConfig(level=logging.INFO)

Model_Name = "distilbert-multilingual-sentiment"  #?

# # Initialize DagsHub + MLflow
# dagshub.init(
#     repo_owner="ejirogoro27",
#     repo_name="ShopEase-Sentiment-Analysis-Project",
#     mlflow=True
# )


def get_best_model(experiment_name="Sentiment-analysis"):
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logging.warning("Experiment not found.")
            return None

        runs = client.search_runs([experiment.experiment_id])

        if not runs:
            logging.warning("No runs found.")
            return None

        # Sort runs by F1 score (highest first)
        best_run = sorted(
            runs,
            key=lambda x: x.data.metrics.get("eval_f1", 0),
            reverse=True
        )[0]

        logging.info(f"Best run ID: {best_run.info.run_id}")
        return best_run

    except Exception as e:
        logging.error(f"Error occurred while loading models: {e}")
        return None


def get_best_f1(experiment_name="Sentiment-analysis"):
    best_run = get_best_model(experiment_name)

    if best_run is None:
        return None

    return best_run.data.metrics.get("eval_f1", 0)


def load_registered_model(model_name= "distilbert-multilingual-sentiment"):
    try:
        # Remote/Production access to mlflow dagshub
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


        model_uri = f"models:/{model_name}/latest"

        sentiment_pipeline = mlflow.transformers.load_model(model_uri)

        logging.info(f"Model '{model_name}' loaded successfully from MLflow.")

        return sentiment_pipeline

    except Exception as e:
        logging.error(f"Error loading registered model: {e}")
        return None



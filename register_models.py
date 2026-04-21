import dagshub
import mlflow
import mlflow.transformers
from transformers import pipeline

# init dagshub
dagshub.init(
    repo_owner="ejirogoro27",
    repo_name="ShopEase-Sentiment-Analysis-Project",
    mlflow=True
)

mlflow.set_experiment("sentiment_analysis")

# LOAD YOUR TRAINED MODEL
MODEL_PATH = "distilbert-base-multilingual-cased"

sentiment_pipeline = pipeline(
    task="text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    return_all_scores=True
)

with mlflow.start_run():

    mlflow.transformers.log_model(
        transformers_model=sentiment_pipeline,
        artifact_path="model",
        registered_model_name="distilbert-multilingual-sentiment"
    )

print("✅ Model successfully registered!")
        
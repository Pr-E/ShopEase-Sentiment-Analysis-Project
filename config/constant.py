
import os
from transformers import TrainingArguments


# BASE PATH

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# DATA PATHS

Input_Data = os.path.join(BASE_DIR, "Data", "sample_dataset.csv")
Cleaned_Data = os.path.join(BASE_DIR, "Data", "cleaned_data.csv")
Train_Data = os.path.join(BASE_DIR, "Data", "processed_data", "train_data.pt")
Test_Data = os.path.join(BASE_DIR, "Data", "processed_data", "test_data.pt")

# ----------------------------
# TOKENIZER SETTINGS
# ----------------------------

# For training + tokenizer
model_name = "distilbert-base-multilingual-cased"   # for training and tokenizer (from huggingface)

# BASE_MODEL_NAME = "distilbert-base-multilingual-cased"

# MLflow registered model name (for inference)
# MLFLOW_MODEL_NAME = "distilbert-multilingual-sentiment"


truncation = True
padding = True
max_length = 128
num_of_labels = 3


# TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir='./results',               
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",            
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)





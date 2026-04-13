
import os
from transformers import TrainingArguments

# ----------------------------
# BASE PATH
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# DATA PATHS
# ----------------------------
Input_Data = os.path.join(BASE_DIR, "Data", "sample_dataset.csv")
Cleaned_Data = os.path.join(BASE_DIR, "Data", "cleaned_data.csv")
Train_Data = os.path.join(BASE_DIR, "Data", "processed_data", "train_data.pt")
Test_Data = os.path.join(BASE_DIR, "Data", "processed_data", "test_data.pt")

# ----------------------------
# TOKENIZER SETTINGS
# ----------------------------
model_name = "distilbert-base-multilingual-cased"

truncation = True
padding = True
max_length = 128
num_of_labels = 3
# ----------------------------
# TRAINING ARGUMENTS
# ----------------------------

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)





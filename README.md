# 🛍️ ShopEase Sentiment Analysis Project

An end-to-end **NLP system and ML engineering project** that classifies customer reviews into **negative, neutral, and positive sentiment** using both:

* Traditional Machine Learning (TF-IDF + Logistic Regression)
* Transformer-based Deep Learning (DistilBERT)

The project demonstrates the full lifecycle of turning raw text data into a **production-ready sentiment analysis API service**.

---

# 🚀 Project Overview

Customer reviews contain valuable insights about:

* Satisfaction levels
* Product quality
* Service performance

This system is designed to:

* Automate sentiment detection at scale
* Extract business insights from multilingual reviews
* Provide real-time predictions via an API service

---

# 🧪 Experimentation Phase (Model Exploration)

Before building the final system, multiple modeling approaches were tested:

## 🔹 Baseline Model

* TF-IDF Vectorization
* Logistic Regression classifier
* Used as a performance benchmark

## 🔹 Advanced Model

* DistilBERT (`distilbert-base-multilingual-cased`)
* Fine-tuned using HuggingFace Trainer API
* Context-aware semantic understanding

---

# 🧹 Data Processing Pipeline

## 1. Data Ingestion

* Loaded raw CSV dataset

## 2. Data Cleaning

* Lowercasing text
* Removing special characters
* Lemmatization (spaCy)
* Stopword removal (NLTK)

## 3. Label Engineering

| Rating | Sentiment   |
| ------ | ----------- |
| 1–2    | Negative 😡 |
| 3      | Neutral 😐  |
| 4–5    | Positive 😍 |

---

## 4. Tokenization

* HuggingFace tokenizer (`AutoTokenizer`)
* Padding + truncation to fixed sequence length
* Conversion to PyTorch tensors

---

# 🧠 Model Training

## Architecture

* Base Model: DistilBERT
* Task: Sequence Classification
* Loss Function: CrossEntropyLoss
* Optimizer: AdamW
* Epochs: 5

## Training Strategy

* Stratified train-test split
* Weighted F1-score evaluation
* Early validation after each epoch

---

## 📊 Performance Results

| Metric   | Score        |
| -------- | ------------ |
| Accuracy | ~0.70 – 0.75 |
| F1 Score | ~0.57 – 0.67 |

> Note: Performance reflects a small dataset with strong class imbalance.

---

# ⚙️ SYSTEM ARCHITECTURE (PRODUCTION DESIGN)

```text id="arch001"
User Input
   ↓
FastAPI Endpoint
   ↓
Text Preprocessing (Cleaning Pipeline)
   ↓
Tokenizer (HuggingFace)
   ↓
Fine-tuned DistilBERT Model
   ↓
Prediction Layer
   ↓
JSON Response (Label + Confidence)
```

---

# 🌐 API DEVELOPMENT PHASE (PRODUCTION DEPLOYMENT)

After model experimentation, the system was deployed using **FastAPI** to enable real-time inference.

---

## 📦 API Design Principles

* Modular architecture
* Reusable preprocessing pipeline
* Stateless inference requests
* Scalable REST API design
* Consistent training vs inference processing

---

## 🚀 API Endpoints

---

### 🔹 Health Check

```http id="api001"
GET /
```

#### Response:

```json id="api002"
{
  "message": "Sentiment API is running 🚀"
}
```

---

### 🔹 Single Prediction

```http id="api003"
POST /predict
```

#### Request:

```json id="api004"
{
  "text": "This product is amazing!"
}
```

#### Response:

```json id="api005"
{
  "label": "positive",
  "class_id": 2,
  "confidence": 0.91
}
```

---

### 🔹 Batch Prediction

```http id="api006"
POST /predict_batch
```

#### Request:

```json id="api007"
{
  "texts": [
    "I love it",
    "It is okay",
    "Worst purchase ever"
  ]
}
```

#### Response:

```json id="api008"
{
  "predictions": [
    {
      "label": "positive",
      "class_id": 2,
      "confidence": 0.91
    },
    {
      "label": "neutral",
      "class_id": 1,
      "confidence": 0.60
    },
    {
      "label": "negative",
      "class_id": 0,
      "confidence": 0.84
    }
  ]
}
```

---

## ⚙️ API Features

* Real-time inference
* Batch processing support
* Consistent preprocessing pipeline
* Confidence score output
* Robust error handling

---

## 🧪 API Testing

You can test the API using:

* Swagger UI → `http://127.0.0.1:8000/docs`
* Postman
* cURL

---

# 🧱 PROJECT STRUCTURE

```bash id="api009"
src/
 └── shopease_app/
     ├── data_ingestion.py
     ├── data_cleaning.py
     ├── data_preprocessing.py
     ├── model_training.py
     ├── api/
         ├── main.py
         ├── inference.py
         ├── schemas.py
```

---

# ▶️ HOW TO RUN

## 1. Install dependencies

```bash id="api010"
pip install -r requirements.txt
```

## 2. Train model

```bash id="api011"
python -m src.shopease_app.model_training
```

## 3. Start API

```bash id="api012"
uvicorn src.shopease_app.api.main:app --reload
```

## 4. Open API Docs

```text id="api013"
http://127.0.0.1:8000/docs
```

---

# 🔁 END-TO-END PIPELINE

```text id="api014"
Raw Reviews → Cleaning → Tokenization → Model → FastAPI → JSON Output
```

---

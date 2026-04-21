# 🛍️ ShopEase Sentiment Analysis System

An end-to-end NLP-powered system that transforms raw customer reviews into actionable business insights — deployed as a production-ready API and interactive dashboard.

---

## 🚀 Overview

Customer reviews are one of the richest sources of business intelligence — but they are:

- Unstructured  
- Multilingual  
- High-volume  
- Difficult to analyze manually  

This project solves that by building a **scalable sentiment analysis pipeline** that:

- Classifies reviews into **Negative, Neutral, Positive**  
- Supports multilingual input  
- Provides real-time predictions via API  
- Visualises results through an interactive dashboard  

---

## 🎯 Business Objective

> Convert unstructured customer feedback into structured insights that improve customer satisfaction and drive revenue.

This system enables businesses to:

- Identify pain points from negative reviews  
- Detect satisfaction drivers  
- Monitor sentiment trends at scale  
- Make data-driven product and service decisions  

---

## 🧠 Model Development

### 🔹 Baseline Model
- TF-IDF + Logistic Regression  
- Achieved ~97% accuracy  
- Strong class balance (F1-score ~0.95)  

### 🔹 Advanced Model
- Transformer: `distilbert-base-multilingual-cased`  
- Handles multilingual reviews  
- Captures contextual meaning  

### ✅ Final Choice
BERT-based model selected for:

- Contextual understanding  
- Multilingual capability  
- Production scalability  

---

## ⚙️ System Architecture


User Input (Streamlit UI)
↓
FastAPI Backend (Inference Layer)
↓
MLflow Model Registry (DagsHub)
↓
Transformer Model (BERT)
↓
Prediction Output (Sentiment + Confidence)

---

---

## 🏗️ Project Structure


ShopEase-Sentiment-Analysis-Project/
│
├── main/ # FastAPI app
│ └── app.py
│
├── pipeline/ # Training & inference pipeline
│ ├── training.py
│ └── prediction.py
│
├── src/shopease_app/ # Core ML components
│ ├── data_cleaning.py
│ ├── data_preprocessing.py
│ ├── model_training.py
│ └── model_pusher.py
│
├── utils/ # MLflow utilities
│ └── models_utils.py
│
├── streamlit_app.py # Frontend dashboard
│
├── docker-compose.yml # Multi-container setup
├── DockerFile.backend # Backend container
├── DockerFile.frontend # Frontend container
│
├── requirements_backend.txt
├── requirements_frontend.txt
│
└── README.md

---

## 📊 Streamlit Dashboard

**Features:**

- Single review prediction  
- Batch CSV upload  
- Real-time sentiment classification  
- Confidence scoring  

---

## 🐳 Dockerization

The system is fully containerized.

### Services:
- Backend → FastAPI  
- Frontend → Streamlit  

### ▶️ Run locally:
```bash
docker compose up


## ☁️ Deployment

Deployed using Render as two services:

- Backend → FastAPI Web Service  
- Frontend → Streamlit Web Service  

---

## 🧰 Tech Stack

### 🔹 Machine Learning
- Transformers (Hugging Face)  
- Scikit-learn  
- PyTorch  

### 🔹 Backend
- FastAPI  
- Uvicorn  

### 🔹 Frontend
- Streamlit  

### 🔹 MLOps
- MLflow  
- DagsHub  

### 🔹 Deployment
- Docker  
- Docker Compose  
- Render  

---

## 📈 Key Results

- Accuracy: ~97%  
- Balanced performance across all classes  
- Strong generalization on unseen data  
- Real-time inference capability  

---

## 💡 Key Learnings

- Simple models can perform as well as complex ones  
- Contextual models improve real-world usability  
- Deployment is as important as model performance  
- End-to-end systems create real business value  

---

## 🔮 Future Improvements

- Sentiment trend analytics dashboard  
- Topic modeling for root-cause analysis  
- Model optimization for faster inference  

---

## ⭐️ Support

If you found this project useful:

- ⭐ Star the repo  
- 🍴 Fork it  
- 🤝 Contribute  
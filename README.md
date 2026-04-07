# ShopEase-Sentiment-Analysis-Project
An end-to-end NLP project that classifies customer reviews into negative, neutral, and positive sentiment using both traditional machine learning and transformer-based models (BERT).

This project demonstrates practical skills in text processing, model development, and translating data into business insights.

# Project Overview

Customer reviews contain valuable insights about satisfaction, product quality, and service performance.
This project builds a sentiment analysis system to:

- Understand customer sentiment at scale
- Identify key pain points and improvement areas
- Support data-driven product and service improvements


# Approach

Data Processing
- Cleaned and standardized raw multilingual review text
- Applied tokenization, lemmatization, and stopword removal
- Converted ratings into structured sentiment labels

Modeling
- Baseline: TF-IDF + Logistic Regression
- Advanced: BERT (distilbert-base-multilingual-cased)
- Built training pipeline using HuggingFace Trainer

Evaluation
- Measured performance using accuracy and weighted F1-score
- Generated classification reports and confusion matrix
- Validated model using unseen sample predictions

# Results

- Accuracy: ~97%
- Strong performance across all sentiment classes
- BERT performed comparably to baseline while offering better contextual understanding

# Business Value

- Highlights customer pain points from negative reviews
- Identifies drivers of customer satisfaction and loyalty
- Enables scalable monitoring of sentiment across large datasets
- Supports better decision-making in product, delivery, and service

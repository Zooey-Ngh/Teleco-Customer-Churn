# Telco Customer Churn Prediction

This project focuses on predicting customer churn for a telecom company using machine learning techniques.

## Project Overview
Customer churn is a major challenge for telecom companies. The goal of this project is to identify customers who are likely to churn so that proactive retention strategies can be applied.

## Dataset
The dataset contains customer demographic information, service usage patterns, and contract details.
Target variable:
- **Churn** (Yes / No)

## Problem Statement
Build a binary classification model that predicts whether a customer will churn.

## Modeling Approach
- Data cleaning and feature engineering
- Handling class imbalance
- Logistic Regression and Gradient Boosting
- Cross-validation
- Threshold tuning
- Evaluation using Precision-Recall AUC

## Evaluation Metrics
- Precision
- Recall
- F1-score
- PR-AUC

## Results
Logistic Regression achieved a PR-AUC of approximately **0.67**, with improved recall after threshold tuning.

## How to Run
```bash
python churn.py

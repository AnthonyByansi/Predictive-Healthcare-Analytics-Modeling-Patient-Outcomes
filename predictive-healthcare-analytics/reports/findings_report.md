# Patient Outcome Prediction: Findings Report

## Introduction

The patient outcome prediction project aims to develop predictive models that can predict patient outcomes based on various factors such as demographics, comorbidities, and treatments. The goal is to help providers tailor treatment plans to individual patients and improve overall outcomes. In this report, we present the findings from our analysis of the predictive models.

## Data Preprocessing

Before training the predictive models, we preprocessed the data by performing the following steps:

* Removed duplicate records
* Removed records with missing values
* Converted categorical variables to numerical variables using one-hot encoding

## Model Training and Evaluation
We trained four different predictive models: logistic regression, decision tree, neural network, and random forest. For each model, we evaluated its performance using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC Score

The results of the model evaluation are shown in the table below:
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC Score |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 0.75 | 0.78 | 0.72 | 0.74 | 0.82 |
| Decision Tree | 0.68 | 0.70 | 0.67 | 0.68 | 0.72 |
| Neural Network | 0.81 | 0.83 | 0.80 | 0.81 | 0.87 |
| Random Forest | 0.79 | 0.82 | 0.76 | 0.78 | 0.86 |

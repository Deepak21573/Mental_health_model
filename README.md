# Mental Health Prediction Model

This repository contains a machine learning model for predicting whether an individual requires mental health treatment based on a survey dataset. The model uses various machine learning algorithms and is packaged with a user-friendly Gradio interface to make predictions based on user inputs.

## Overview
The objective of this project is to build a mental health prediction system that can analyze survey responses and determine if an individual might need mental health treatment. It uses multiple machine learning models to predict the treatment outcome, based on a survey dataset that includes various personal and demographic features.

The project provides:

Preprocessing of data: Handling missing values, encoding categorical variables.

Model Training: Four machine learning models (Random Forest, XGBoost, Gradient Boosting, and Logistic Regression) are trained and evaluated.

Model Evaluation: The models' performance is evaluated using accuracy, precision, recall, and F1-score.

Model Prediction: A Gradio-based user interface to make real-time predictions based on user input.

## Dataset
The dataset (survey.csv) contains survey responses with columns representing demographic information and mental health-related questions. The treatment column represents whether the individual needs treatment or not.

## Columns:
**Timestamp**: Timestamp of survey submission (dropped)

**comments**: Optional comments by the participant (dropped)

**state**: The state where the participant resides (dropped)

**Categorical features**: such as Age, Gender, Family History, Care Options, etc.

**treatment**: Target variable (1 for treatment needed, 0 for no treatment needed)

## Requirements

# Before running this project, make sure you have the following Python libraries installed:

pandas
numpy
matplotlib
seaborn
sklearn
xgboost
gradio
joblib

#You can install the dependencies by running the following command:

bash
Copy
Edit
pip install -r requirements.txt
File Structure
bash
Copy
Edit

## File Structure
├── survey.csv                     # Dataset
├── mental_health_model.py         # Script to train and evaluate models
├── label_encoders.pkl             # Encoders for categorical features
├── RandomForest_model.pkl         # Random Forest model
├── XGBoost_model.pkl              # XGBoost model
├── GradientBoosting_model.pkl     # Gradient Boosting model
├── LogisticRegression_model.pkl   # Logistic Regression model
├── app.py                         # Script for the Gradio interface
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview

## Model Training and Evaluation
1. Data Preprocessing
Missing values are handled by replacing them with "Unknown".
Categorical variables are encoded using LabelEncoder.
2. Model Training
   
# The following models are trained:

Random Forest
XGBoost
Gradient Boosting
Logistic Regression

# Each model is evaluated based on:

Accuracy
Precision
Recall
F1-score

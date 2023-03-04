"""
data_preprocessing.py

This script performs data preprocessing tasks for the patient outcome prediction project, such as cleaning and transforming
the raw data into a format suitable for building predictive models.

"""

import pandas as pd

# Load the raw data
data = pd.read_csv('data/raw/patient_data.csv')

# Drop unnecessary columns
data.drop(['patient_id', 'date_of_birth'], axis=1, inplace=True)

# Rename columns for consistency
data.rename(columns={'gender': 'sex', 'disease_progression': 'outcome'}, inplace=True)

# Convert categorical variables to one-hot encoding
categorical_cols = ['sex', 'race', 'smoker', 'diabetes', 'hypertension', 'heart_disease', 'stroke']
data = pd.get_dummies(data, columns=categorical_cols)

# Fill missing values with median or mode
numerical_cols = ['age', 'weight', 'height', 'bmi', 'comorbidity_score']
for col in numerical_cols:
    if data[col].isna().sum() > 0:
        if col == 'comorbidity_score':
            data[col].fillna(0, inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
for col in categorical_cols:
    if data[col].isna().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Save the cleaned data to a new file
data.to_csv('data/processed/cleaned_data.csv', index=False)

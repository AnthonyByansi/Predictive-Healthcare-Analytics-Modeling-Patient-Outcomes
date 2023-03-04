"""
model_training.py

This script trains the predictive models for the patient outcome prediction project using preprocessed data.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the preprocessed data
data = pd.read_csv('data/processed/cleaned_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('outcome', axis=1), data['outcome'], test_size=0.2, random_state=42)

# Train and save logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, 'models/lr_model.joblib')

# Train and save decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'models/dt_model.joblib')

# Train and save neural network model
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)
joblib.dump(nn_model, 'models/nn_model.joblib')

# Train and save random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/rf_model.joblib')

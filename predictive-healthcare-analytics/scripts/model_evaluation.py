"""
model_evaluation.py

This script evaluates the performance of the predictive models for the patient outcome prediction project using various
metrics such as accuracy, precision, recall, and F1 score.

"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the preprocessed data
data = pd.read_csv('data/processed/cleaned_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('outcome', axis=1), data['outcome'], test_size=0.2, random_state=42)

# Train and evaluate logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print('Logistic regression model evaluation:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_preds)))
print('Precision: {:.2f}'.format(precision_score(y_test, lr_preds)))
print('Recall: {:.2f}'.format(recall_score(y_test, lr_preds)))
print('F1 score: {:.2f}'.format(f1_score(y_test, lr_preds)))

# Train and evaluate decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
print('Decision tree model evaluation:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, dt_preds)))
print('Precision: {:.2f}'.format(precision_score(y_test, dt_preds)))
print('Recall: {:.2f}'.format(recall_score(y_test, dt_preds)))
print('F1 score: {:.2f}'.format(f1_score(y_test, dt_preds)))

# Train and evaluate neural network model
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)
nn_preds = nn_model.predict(X_test)
print('Neural network model evaluation:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, nn_preds)))
print('Precision: {:.2f}'.format(precision_score(y_test, nn_preds)))
print('Recall: {:.2f}'.format(recall_score(y_test, nn_preds)))
print('F1 score: {:.2f}'.format(f1_score(y_test, nn_preds)))

# Train and evaluate random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print('Random forest model evaluation:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, rf_preds)))
print('Precision: {:.2f}'.format(precision_score(y_test, rf_preds)))
print('Recall: {:.2f}'.format(recall_score(y_test, rf_preds)))
print('F1 score: {:.2f}'.format(f1_score(y_test, rf_preds)))

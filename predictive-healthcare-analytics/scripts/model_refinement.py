"""
model_refinement.py

This script performs model refinement tasks for the patient outcome prediction project, such as hyperparameter tuning
and feature selection, to improve the performance of the predictive models.

"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load the preprocessed data
data = pd.read_csv('data/processed/cleaned_data.csv')

# Split the data into training and testing sets
X = data.drop('outcome', axis=1)
y = data['outcome']

# Hyperparameter tuning for the random forest model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X, y)
best_params = grid_search.best_params_
print('Best hyperparameters for random forest model: {}'.format(best_params))

# Feature selection using the random forest model
model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
model.fit(X, y)
importances = model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)
print('Feature importances:')
print(feature_importances)

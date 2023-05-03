import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack']  # set target feature
    del df['attack']  # delete redundant feature from original df
    del df['flow_pkts_per_sec']  # delete NaN column

    # Random Forest WITHOUT NORMALIZATION
    x_train = df
    y_train = dftarget

    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }

    # Create the random forest classifier
    model = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Print the best hyperparameters found
    print("Best Hyperparameters:")
    print(grid_search.best_params_)

main()

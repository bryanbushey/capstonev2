import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    # Create the decision tree classifier
    model = DecisionTreeClassifier()

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Print the best hyperparameters found
    print("Best Hyperparameters:")
    print(grid_search.best_params_)

    # Fit the best model on the entire dataset
    best_model.fit(x_train, y_train)

    # Save the best model
    with open('models/DecisionTree/models/DTmodel.pkl', 'wb') as f:
        pickle.dump(best_model, f)

main()
#OUTPUT BELOW***
#{'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2}
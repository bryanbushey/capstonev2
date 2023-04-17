import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec'] # delete NaN column
    #Random Forest WITHOUT NORMALIZATION
    x_train = df
    y_train = dftarget

    #create tree
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, max_features='auto')
    model.fit(x_train, y_train)

    with open('models/RandomForest/models/RFmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

main()
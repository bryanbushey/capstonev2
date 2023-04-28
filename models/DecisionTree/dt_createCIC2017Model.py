import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec'] # delete NaN column
    #Random Forest WITHOUT NORMALIZATION
    x_train = df
    y_train = dftarget

    #create tree
    model = DecisionTreeClassifier(criterion='gini',max_depth=20,max_features='sqrt',min_samples_leaf=2,min_samples_split=2)
    model.fit(x_train, y_train)

    with open('models/DecisionTree/models/DTmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

main()
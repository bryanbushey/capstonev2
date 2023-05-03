import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.datasets import load_iris 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec'] # delete NaN column
    iris = load_iris() 
    
    #Random Forest WITHOUT NORMALIZATION
    
    data = df
    target = dftarget

    X = iris.data 
    y = iris.target

    #create tree
    model = KNeighborsClassifier(n_neighbors=5) 
    model.fit(X, y)

    with open('models/KNN/models/KNNmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

main()
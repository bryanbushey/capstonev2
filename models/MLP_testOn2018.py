import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import MLP_createCIC2017Model

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    del df['flow_pkts_per_sec']
    model = MLP_createCIC2017Model.main() # calls the clf model from the 2017 dataset

    dftarget = df['attack']
    del df['attack']

    x = df
    y = dftarget

    y_pred = model.predict(x)
    
    test_score = accuracy_score(y,y_pred)
    print("score on test data: ", test_score)
    print(confusion_matrix(y,y_pred))
    print(classification_report(y, y_pred))
    
main()
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import MLP_createCIC2017Model

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    del df['flow_pkts_per_sec']
    clf = MLP_createCIC2017Model.main() # calls the clf model from the 2017 dataset

    dftarget = df['attack']
    del df['attack']

    x = df
    y = dftarget

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    y_predict = clf.predict(x_test)
    
    test_score = accuracy_score(y_predict, y_test)
    print("score on test data: ", test_score)
    print(confusion_matrix(y_test,y_predict))
    
main()
    

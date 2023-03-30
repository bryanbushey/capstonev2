import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import rf_createCIC2017Model

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    del df['flow_pkts_per_sec']
    clf = rf_createCIC2017Model.main() # calls the clf model from the 2017 dataset

    dftarget = df['attack']
    del df['attack']

    #Random Forest WITHOUT NORMALIZATION
    x_test = df
    y_test = dftarget

    #create tree
    
    y_pred = clf.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print('Accuracy:',accuracy_score(y_test, y_pred))

main()
    

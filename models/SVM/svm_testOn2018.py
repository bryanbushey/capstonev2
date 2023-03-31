import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import svm_createCIC2017Model

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    del df['flow_pkts_per_sec']
    svclassifier = svm_createCIC2017Model.main() # calls the clf model from the 2017 dataset

    dftarget = df['attack']
    del df['attack']

    x = df
    y = dftarget

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
main()
    

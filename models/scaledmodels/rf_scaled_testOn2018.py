import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
import rf_scaled_createCIC2017Model

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    
    clf = rf_scaled_createCIC2017Model.main() # calls the clf model from the 2017 dataset

    dftarget = df['attack']
    del df['attack']

    #Random Forest WITHOUT NORMALIZATION
    x_test = df
    y_test = dftarget

    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test_scaled = scaler.transform(x_test)


    #create tree
    
    y_pred = clf.predict(x_test_scaled)
    print('normal\t  attack')
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print('Accuracy:',accuracy_score(y_test, y_pred))

main()
    

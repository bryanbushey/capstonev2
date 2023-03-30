import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df

    #Random Forest WITHOUT NORMALIZATION
    x_train = df
    y_train = dftarget

    #scale the data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    #create tree
    clf = RandomForestClassifier(n_estimators=50, bootstrap=True, random_state=1)
    clf.fit(x_train_scaled, y_train)

    return clf
 
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec']
    #Random Forest WITHOUT NORMALIZATION
    x = df
    y = dftarget
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    #scale
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    #create tree
    clf = svm.SVC(verbose=1)
    clf.fit(x_train_scaled, y_train)

    return clf
#main()
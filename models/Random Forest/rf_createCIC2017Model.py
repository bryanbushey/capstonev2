import pandas as pd
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec']
    #Random Forest WITHOUT NORMALIZATION
    x_train = df
    y_train = dftarget


    #create tree
    svclassifier = SVC(kernel='linear',verbose=1)
    svclassifier.fit(x_train, y_train)

    return svclassifier
main()
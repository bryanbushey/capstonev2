import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack']
    del df['attack']
    del df['flow_pkts_per_sec']
    

    #Random Forest WITHOUT NORMALIZATION
    x_test = df
    y_test = dftarget

    with open('models/RandomForest/models/RFmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # fits the data to the model made by CIC2017
    y_pred = model.predict(x_test)

    # Get test scores comparing the true Y with the predicted Y
    test_score = accuracy_score(y_test,y_pred)
    print("score on test data: ", test_score)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test, y_pred))

main()
    

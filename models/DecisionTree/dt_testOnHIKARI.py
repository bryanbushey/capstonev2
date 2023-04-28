import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('./dataprocessing')
import getScores

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/HIKARIprocessed.csv')
    dftarget = df['attack']
    del df['attack']
    del df['flow_pkts_per_sec']
    

    #Random Forest WITHOUT NORMALIZATION
    x_test = df
    y_test = dftarget

    with open('models/DecisionTree/models/DTmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # fits the data to the model made by CIC2017
    y_pred = model.predict(x_test)

    # Get test scores comparing the true Y with the predicted Y
    print(confusion_matrix(y_test,y_pred))
    
    getScores.main(y_test,y_pred)
    
main()
    

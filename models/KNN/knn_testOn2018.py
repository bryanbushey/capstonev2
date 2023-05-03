import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('./dataprocessing')
import getScores

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    dftarget = df['attack']
    del df['attack']
    del df['flow_pkts_per_sec']
    
    iris = load_iris() 

    #Random Forest WITHOUT NORMALIZATION
    data = df
    target = dftarget

    X = iris.data 
    y = iris.target

    with open('models/KNN/models/KNNmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # fits the data to the model made by CIC2017
    y_pred = model.predict(X)

    # Get test scores comparing the true Y with the predicted Y
    getScores.main(y,y_pred)

main()
    

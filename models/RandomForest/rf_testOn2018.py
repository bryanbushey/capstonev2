import pandas as pd
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
    

    #Random Forest WITHOUT NORMALIZATION
    x_test = df
    y_test = dftarget

    with open('models/RandomForest/models/RFmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # fits the data to the model made by CIC2017
    y_pred = model.predict(x_test)

    # Get test scores comparing the true Y with the predicted Y
    getScores.main(y_test,y_pred)


main()
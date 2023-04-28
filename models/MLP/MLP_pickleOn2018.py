import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('./dataprocessing')
import getScores

def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv')
    del df['flow_pkts_per_sec'] # this column was found to contain NaN values
    dftarget = df['attack']
    del df['attack']

    # sets x = to the dataset without ATTACK column
    x = df
    # sets y = only the ATTACH column
    y = dftarget

    with open('models/MLP/models/MLPmodel001.pkl', 'rb') as f:
        model = pickle.load(f)

    # fits the data to the model made by CIC2017
    y_pred = model.predict(x)
    
    getScores.main(y,y_pred)
    
     # Get the loss_curve_ attribute of the MLPClassifier object
    loss_curve_ = model.loss_curve_

    # Create a line plot of the data in the loss_curve_ array
    plt.plot(loss_curve_)

    # Adding labels and title to the plot
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLPClassifier Loss Curve")

    # Display the plot
    plt.show()


main()
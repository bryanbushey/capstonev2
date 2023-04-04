import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec']
    
    x_train = df
    y_train = dftarget
    
    model = MLPClassifier(hidden_layer_sizes=(50, 40, 35, 30),verbose=1, solver='adam', tol=.0002)
    
    model.fit(x_train, y_train)
    return model

#main()

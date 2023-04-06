import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle



def main():
    df = pd.read_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv')
    dftarget = df['attack'] # set target feature
    del df['attack'] # delete reduntant feature from original df
    del df['flow_pkts_per_sec']
    
    x_train = df
    y_train = dftarget
    
    model = MLPClassifier(hidden_layer_sizes=(50, 40, 35, 30),verbose=1, solver='adam', tol=0, max_iter=1000, n_iter_no_change=1000, batch_size=128)
    
    model.fit(x_train, y_train)
    with open('models/MLP/models/MLPmodel.pkl', 'wb') as f:
        pickle.dump(model, f)




    return model


main()

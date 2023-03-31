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
    
    #reduce data to 30% so its faster
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = MLPClassifier(activation='relu', hidden_layer_sizes=(10,10,10), learning_rate_init=0.01, batch_size=32, max_iter=100, alpha=0.0001, learning_rate='constant', verbose=1).fit(x_train, y_train)
    
    #model = MLPClassifier(random_state=1, max_iter=300, verbose=1).fit(x_train, y_train)
    return model

#main()
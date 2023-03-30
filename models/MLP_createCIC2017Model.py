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
    #Random Forest WITHOUT NORMALIZATION
    x = df
    y = dftarget
    #reduce data to 30% so its faster

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    clf = MLPClassifier(random_state=1, max_iter=300, verbose=1).fit(x_train, y_train)
    return clf
#main()
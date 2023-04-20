#DO NOT EDIT THIS. THIS IS FOR BRYAN'S PATHS

import pandas as pd

CIC2017 = pd.read_csv('/Users/bryan/forGIT/finishedCSV/CIC2017processed.csv')
CIC2017.to_csv('dataprocessing/proc_csvfiles/CIC2017processed.csv', index=False)


CIC2018 = pd.read_csv('/Users/bryan/forGIT/finishedCSV/CIC2018processed.csv')
CIC2018.to_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv', index=False)


HIKARI = pd.read_csv('/Users/bryan/forGIT/finishedCSV/HIKARIprocessed.csv')
HIKARI.to_csv('dataprocessing/proc_csvfiles/HIKARIprocessed.csv', index=False)

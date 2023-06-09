from tqdm import tqdm
import pandas as pd
def main():
    df = pd.read_csv('/Users/bryan/forGIT/originalcsvfiles/edited2018CIC_Friday-02-03-2018_TrafficForML_CICFlowMeter_original.csv')
    attack_dict=dict() #turns 'protocol_dict' into dictionary
    counter=0
    for attack in df['attack']: #for (data entry ~ protocol) in the column 'protocol_type'
        if attack in attack_dict: #if the protocol is in the dictonary already, 
            attack=attack_dict[attack] #if the protocol is in the dictionary, set it as the number
        else:
            attack_dict[attack]=counter; #if the protocol is NOT already in dictionary, set the index = to the counter
            counter += 1 #increment counter    
    rows = df.shape[0]
    for i in tqdm (range(rows),desc="Loading..."):
        attack = df['attack'].iloc[i] # attack is = to the current line in the 'attack' feature in the original dataframe
        #print(attack,attack_dict[attack])
        df['attack'].iloc[i] = attack_dict[attack] # assigns 0 or 1 based on if it says benign or ddos

    df = df.apply(pd.to_numeric)

    df.to_csv('dataprocessing/proc_csvfiles/CIC2018processed.csv', index=False)
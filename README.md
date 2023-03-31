# capstoneV2

TO START:
run the STARTUP.py file to load all .csv's

As for the original csv files found on google docs, get the path to the file and insert it to the respective processor.py file

0 is normal
1 is attack

using two different CIC datasets from University of New Brunswick (https://www.unb.ca/cic/datasets/ids-2017.html)
CIC-IDS2017 & CIC-IDS2018
And HIKARI (https://zenodo.org/record/5199540#.ZBJgYy-B1pR)
The CIC datasets deal mainly with DDoS attacks, labeling the traffic as BENIGN or BOT/ATTACK
HIKARI is slightly different, having other kinds of attacks also introduced into the mix.

-----Notes-----
Feature selection
train test within one dataset
try hyperparameters


------FILE NAMES------
Dataset    2017CIC_Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Dataset    2018CIC_Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
notebook    CIC2017.ipynb
notebook    CIC2018.ipynb
returns model of 2017 when called    createCIC2017Model.py
calls CIC2017 and tries to apply it to 2018    testmodel.py
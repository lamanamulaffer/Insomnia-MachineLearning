#uses the hypnogram annotation files to extract raw features
#Raw features include: # of epochs in each stage, # of transitions between stages

import csv

#get feature headers (# in each stage, # of each transition)
def getRawFeatureNames():
    keys = ['W','S1', 'S2','S3','S4','REM']
    features = []
    features.append('ID')
    for k in keys:
        features.append('#' + k)
    for k in keys:
        for j in keys:
            if (k!=j):
                features.append('#' + k + '-' + j)
    features.append('Label')
    return features

#gets the feature vectors    
def getRawFeatureVector(patientID, studyN, label):
    
    pID = label + '_ID' + str(patientID) + '_S' + str(studyN)
    filename = str(patientID) + '.csv';
    featurevector = []
    featurevector.append(pID)
     
    #make relevant keys
    keys = ['Wach','S1','S2','S3','S4','REM'];
    numOfStages=dict.fromkeys(keys,0)
    transitions = []
    for k in keys:
        for j in keys:
            if (k!=j):
                key= k +'-' + j
                transitions.append(key)
    numOfTrans = dict.fromkeys(transitions,0)
    #read file
    with open(filename,'rb') as f:
        reader=csv.reader(f)
        preS = 'Wach' #check if any starts with other than wake
        for row in reader:
            curS =(row[0]).strip()
            if (curS not in keys):
                print 'new key: ' + curS + 'from:=' + pID
                print row
            else:
                numOfStages[curS] +=1 #add to count
                if (curS!=preS): #transition!!!!
                    key = preS + '-' + curS
                    numOfTrans[key] +=1
                    preS=curS #update stage tracker

    #fill in feature vector
    for k in keys:
        featurevector.append(numOfStages[k])
    for k in transitions:
        featurevector.append(numOfTrans[k])
    featurevector.append(label)
    return featurevector    
    
#Initialiaztion for Healthy patients
#label = 'I'
#studyN = 3
#idstart = 1
#idstop = 37 

#Initialization for Insomnia patients
studyN = 1
label = 'I'
idstart = 3
idstop = 21

rawdata = []
rawdata.append(getRawFeatureNames())
for i in range(idstart,idstop):
    fvec = getRawFeatureVector(i,studyN,label)
    rawdata.append(fvec)

#write to file
with open('rawdatatable1.csv','ab') as f:
    writer = csv.writer(f)
    for row in rawdata:
        writer.writerow(row)
print 'done'
    


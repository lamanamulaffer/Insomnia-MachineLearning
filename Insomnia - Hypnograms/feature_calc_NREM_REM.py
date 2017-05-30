#Here we extract features from the raw data  table. 
#Features based on NREM, REM and Wake sleep stage classification
import csv

#get feature headers (# in each stage, # of each transition)
def getRawFeatureNames():
    keys = ['W','NREM', 'REM']
    features = []
    features.append('ID')
    #stage ratio : eg: s1/all
    for k in keys:
        features.append(k +'/all')
    #transition ratio: eg: w-s1/all
    for k in keys:
        for j in keys:
            if (k!=j):
                features.append(k + '-' + j + '/all')
    #stage pair ratio: eg: s1/s2
    for k in keys:
        for j in keys:
            if (k!=j):
                features.append(k + '/' + j)

    features.append('Age')
    features.append('Gender')
    features.append('Label')
    return features
    
def convert(r):
    for i in range(1,len(r)-3):
        r[i] = int(r[i])
    return r
	
#reads a row from rawdatatable and calculates the needed features
#features: # of transitions between wake, NREM and REM, their rations, demographic info
def getRawVector(rowvec):
    row=convert(rowvec)
    r = []
    r.append(row[1]) #W
    r.append(sum(row[2:6])) #NREM
    r.append(row[6]) #REM
    #W-NREM transitions
    w2nrem = 0
    for i in range(7,11):
        w2nrem += row[i]
    r.append(w2nrem) #wake-nrem
    r.append(row[11]) #wake-rem
    nrem2w = row[12] + row[17] + row[22] + row[27]
    nrem2rem = row[16] + row[21] + row[26] + row[31]
    r.append(nrem2w) #NREM-W
    r.append(nrem2rem) #NREM-REM
    r.append(row[32]) #REM-W
    rem2nrem = 0
    for i in range(33,37):
        rem2nrem+=row[i]
    r.append(rem2nrem) #REM-NREM
    
    allEpochs = float(sum(r[0:3]))
    featurevec = []
    featurevec.append(row[0]) #ID
    for i in range(0,3): #stages
        featurevec.append(r[i]/allEpochs)
    for i in range(3,9): #transitions
        featurevec.append(r[i]/allEpochs)
    #ratios relative to each other
    for i in range(0,3):
        for j in range(0,3):
            if (i!=j):
                if (r[j]==0):
                    featurevec.append(-1)
                else:
                    ratio = r[i]/float(r[j])
                    featurevec.append(ratio)
    #Age, Gender, label
    featurevec.append(row[-3])
    featurevec.append(row[-2])
    featurevec.append(row[-1])
    return featurevec

#read from raw data
with open('rawdatatableall_v2.csv','rb') as f:
    reader=csv.reader(f)
    firstR = 0
    features = []
    features.append(getRawFeatureNames())
    for row in reader:
        if (firstR==0):
            firstR = 1 #ignores header
        else:
            fvec = getRawVector(row)
#            print fvec
            features.append(fvec)
    
#write to file
with open('featuretable_v4.csv','ab') as f:
    writer = csv.writer(f)
    for row in features:
        writer.writerow(row)
print 'done'
            
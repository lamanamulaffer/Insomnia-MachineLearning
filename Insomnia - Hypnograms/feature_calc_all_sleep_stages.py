#here we calculate features from feature table
#features based on individual stages: transitions, rations
import csv

#get feature headers (# in each stage, # of each transition)
def getRawFeatureNames():
    keys = ['W','S1', 'S2','SWS','REM']
    features = []
    features.append('ID')
    #stage ratio : s1/all
    for k in keys:
        features.append('#' + k +'/all')
    #transition ratio: w-s1/all
    for k in keys:
        for j in keys:
            if (k!=j):
                features.append('#' + k + '-' + j + '/all')
    #stage pair ratio: s1/s2
    for k in keys:
        for j in keys:
            if (k!=j):
                features.append('#' + k + '/#' + j)
    features.append('Label')
    return features
	
#reads a row from rawdatatableall and calculates the needed features
def getRawVector(row):
    featurevec = []
    newrow = [] #has sws together
    for i in range(0,22):
        if ((i==4) or (i==9) or (i==14) or (i==19)):
            newrow.append(int(row[i]) + int(row[i+1]))
        elif (i==5 or i==10 or i==15 or i==20):
            continue
        elif(i==0):
            newrow.append(row[i])
        else:
            newrow.append(int(row[i]))

    for i in range(22,27):
        if (i!=25):
            newrow.append(int(row[i]) + int(row[i+5]))
    for i in range(32,38):
        if (i==35):
            newrow.append(int(row[i]) + int(row[i+1]))
        elif (i==36):
            continue
        elif (i==37):
            newrow.append(row[i])
        else:
            newrow.append(int(row[i]))

    #now we have the row - w,s1,s2,sws,rem
    allS = 0
    allT = 0
    for i in range(1,6):
        allS = allS + newrow[i]
    for i in range(6,(len(newrow)-1)):
        allT = allT + newrow[i]
    featurevec.append(newrow[0])
	
    #stage ratios
    for i in range(1,6):
        stageR = newrow[i]/float(allS)
        featurevec.append(stageR)

    #transition ratios
    for i in range(6,(len(newrow)-1)):
        transR = newrow[i]/float(allS)
        featurevec.append(transR)

    #stage pair ratios
    for i in range(1,6):
        for j in range(1,6):
            if (i!=j):
                if (newrow[j]==0):
                    featurevec.append(-1)
                else:
                    spR = float(newrow[i])/newrow[j]
                    featurevec.append(spR)

    #patient label
    featurevec.append(newrow[-1])
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
            features.append(fvec)
    
#write to file
with open('featuretable_v5.csv','ab') as f:
    writer = csv.writer(f)
    for row in features:
        writer.writerow(row)
print 'done'
            
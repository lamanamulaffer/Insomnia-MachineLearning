#generates test and training folds in csv files. 
#Pre processes data, and makes sure that subject demographics are evenly distributed

import csv
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier

n_folds = 5
           
#read feature file - seperate into h n i - divide equally into 5 or 10 folds

Data = []
with open ('featuretable_v6_adjusted.csv','rb') as f:
    reader = csv.reader(f)
    fr = 0
    for row in reader:
        if (fr == 0):
            fr +=1 #ignore header
        else:   
            Data.append(row)
          
#filter based on gender and label to make even distribution
h_f_data = filter(lambda row: row[-1] == '1' and row[-2] == '2', Data)
h_m_data = filter(lambda row: row[-1] == '1' and row[-2] == '1', Data)
i_f_data = filter(lambda row: row[-1] == '0' and row[-2] == '2', Data)
i_m_data = filter(lambda row: row[-1] == '0' and row[-2] == '1', Data)

#split into n_folds
hf_splits = np.array_split(np.array(h_f_data),n_folds)
hm_splits = np.array_split(np.array(h_m_data),n_folds)
if_splits = np.array_split(np.array(i_f_data),n_folds)
im_splits = np.array_split(np.array(i_m_data),n_folds)

#get test folds
test_folds = []
for i in range(0,n_folds):
    fold = np.concatenate((hf_splits[i],hm_splits[i],if_splits[i],im_splits[i]))
    test_folds.append(fold)
    #write to output
    filename = 'Folds_v6/Test_CV' + str(n_folds) + '_fold_' + str(i) + '.csv'
    with open (filename,'ab') as f:
        writer = csv.writer(f)
        for row in fold:
            writer.writerow(row)
    
print len(test_folds), len(test_folds[0]), len(test_folds[0][0])
#get train folds
train_folds = []
for i in range(0,n_folds):
    fold = []
    for j in range(0,n_folds):
        if (i!=j):
            for row in test_folds[j]: #iter through each test fold that's not i
                fold.append(row)
    train_folds.append(fold)
    #write to output
    filename = 'Folds_v6/Train_CV' + str(n_folds) + '_fold_' + str(i) + '.csv'
    with open (filename,'ab') as f:
        writer = csv.writer(f)
        for row in fold:
            writer.writerow(row)
    
print len(train_folds), len(train_folds[0]), len(train_folds[0][0])
print len(Data)



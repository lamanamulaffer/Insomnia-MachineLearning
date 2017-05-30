#Performs grid search to find optimum model parameters
#Pick the best features using SelectKBest and perform cross validation on the data set

import csv
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn import svm,grid_search,metrics
from sklearn.cross_validation import cross_val_score,cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB


n_folds= 5
r = 0
with open('featuretable_v6_adjusted.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if (r==0):  
            r=1
            header = row[1:-3]
        
    
#get test set + test indices
index = 0
X_test_folds,y_test_folds, indices_test_folds = [],[],[]
for i in range (0,n_folds):
    testfile = 'Folds_v6/Test_CV' + str(n_folds) + '_fold_' + str(i) + '.csv'
    X_test,y_test,indices = [],[],[]
    with open(testfile,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            X_test.append(row[1:-3])#ignore id, age,gender,label
            y_test.append(row[-1])
            indices.append(index)
            index+=1 #tracking test indices
    X_test_folds.append(X_test)
    y_test_folds.append(y_test)
    indices_test_folds.append(indices)
    if (i==0):
        X = X_test
        y = y_test
    else:
        X = np.concatenate((X,X_test))
        y = np.concatenate((y,y_test))

#get training sets + training indices
X_train_folds,y_train_folds, indices_train_folds = [],[],[]
for i in range(0,n_folds):
    X_train, y_train, indices = [],[],[]
    for j in range(0,n_folds):
        if (i!=j):
            X_train = X_train + X_test_folds[j]
            y_train = y_train + y_test_folds[j]
            indices = indices + indices_test_folds[j]
    X_train_folds.append(X_train)
    y_train_folds.append(y_train)
    indices_train_folds.append(indices)
#print len(X_test_folds) ,len(X_test_folds[0]),len(X_test_folds[0][0])
#print len(X_train_folds) ,len(X_train_folds[0]),len(X_train_folds[0][0])

cv_iterator = zip(indices_train_folds, indices_test_folds)

#print 'mean accuracy'
#print (avg/float(n_folds))
#grid search classification
C = [0.001,0.01,0.1,1,10,100,1000]
gamma = [0.001,0.01,0.1,1,10,100,1000]
deg = [2,3,4]
k1 = ['rbf','sigmoid']
k2 = ['linear']
k3 = ['poly']
params = [{'kernel': k1, 'C': C, 'gamma': gamma},
          {'kernel': k2, 'C':C},
          {'kernel': k3, 'C':C, 'gamma':gamma, 'degree': deg }]
scaler = preprocessing.StandardScaler()
X = preprocessing.scale(X)#scaler.fit_transform(X)
#for each i # of features, perform grid search and get values
#for i in range(10,len(header)):
#    print "num of features = " + str(i)
#    model = SelectKBest(f_classif, k=i)
#    X_new = model.fit_transform(X,y)
#    grid = grid_search.GridSearchCV(svm.SVC(),params,cv=cv_iterator)
#    grid.fit(X_new,y)
#    #results
#    print grid.best_params_
#    print ''
#    print grid.best_score_
#print grid.cv_results_
f = [12,14,15,16,17]
g = [1,1,1000,1,10]

#checking cross validation classification for a range of top features
for i in range(0,len(f)):
    print 'computing ' + str(f[i]) + 'feaures:'
    m = SelectKBest(f_classif, k=f[i])
    X1 = m.fit_transform(X,y)
    for k in m.get_support(indices=True):
        print header[k]
    model = svm.SVC(kernel='sigmoid',C=0.1,gamma=g[i])
    scores = cross_val_score(model,X1,y,cv=cv_iterator)
    pred = cross_val_predict(model,X1,y,cv=cv_iterator)
    print scores
    print metrics.accuracy_score(y,pred)
    print metrics.confusion_matrix(y,pred)
    print metrics.classification_report(y,pred)
    print metrics.cohen_kappa_score(y,pred)
    print '----------------------------------'



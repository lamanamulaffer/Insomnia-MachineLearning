#Perform cross validation on data set
#Evaluate different classifiers: Logistic Regression, RandomForestClassifier, MultinomialNB
#Get best parameters: SelectKBest, PCA
#Perform Grid search for best parameters
#Visualize effectiveness of features

import csv
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn import svm,grid_search,metrics
from sklearn.cross_validation import cross_val_score,cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
n_folds= 10

#get test set + test indices
index = 0
X_test_folds,y_test_folds, indices_test_folds = [],[],[]
for i in range (0,n_folds):
    testfile = 'Folds_v3/Test_CV' + str(n_folds) + '_fold_' + str(i) + '.csv'
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

#normalize data & select relevant features
#feature ranking done previously - select 10/15 features
avg = 0
for i in range(0,n_folds):

	#scale and normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train_folds[i])
    X_test = scaler.transform(X_test_folds[i])
	
	#select best features
    model= SelectKBest(f_classif,k=15)
    model.fit(X_train,y_train_folds[i])
    X_train = X_train[:,model.get_support()]
    X_test = X_test[:,model.get_support()]
	
	#Attempting multiple classifiers
    #Logitic Regression/ Multinomial NB, Random Forest Classifier
#  clf = LogisticRegression()
#    clf = MultinomialNB()
   clf = RandomForestClassifier(n_estimators=500)
   clf.fit(X_train,y_train_folds[i])
   print clf.predict(X_test)
   score = clf.score(X_test,y_test_folds[i])
   preds = clf.predict_proba(X_test)[:,1]
   
   #plot ROC curve to evaluate classifier output quality
   fpr,tpr,_ = metrics.roc_curve(y_test_folds[i],preds)
   df = pd.DataFrame(dict(fpr=fpr,tpr=tpr))
   ggplot(df, aes(x='fpr',y='tpr')) + geom_line() + geom_abline(linetype='dashed')
   
#   #Perform Principal Coponent Analysis
   pca = decomposition.PCA(n_components=6)
   pca.fit(X_train)
   X_train = pca.transform(X_train)
   X_test = pca.transform(X_test)
   print pca.explained_variance_
   
    #features have been normalized and selected according to training set
    X_train_folds[i] = X_train
    X_test_folds[i] = X_test
    #add to main matrix X & y
    if (i==0):
        X = X_test
        y = y_test_folds[i]
    else:
        X = np.concatenate((X,X_test))
        y = np.concatenate((y,y_test_folds[i]))
cv_iterator = zip(indices_train_folds, indices_test_folds)

#Using Grid search to get optimum parameters
#print 'mean accuracy'
#print (avg/float(n_folds))
#grid search classification
C = [0.001,0.01,0.1,1,10,100,1000]
penalty = ['l2','l1']
gamma = [0.001,0.01,0.1,1,10,100,1000]
deg = [2,3,4]
k1 = ['rbf','sigmoid']
k2 = ['linear']
k3 = ['poly']
params = [{'kernel': k1, 'C': C, 'gamma': gamma},
          {'kernel': k2, 'C':C},
          {'kernel': k3, 'C':C, 'gamma':gamma, 'degree': deg }]
p1 = [{'C': C, 'penalty': penalty}]
p2 = [{'n_estimators':[100,200,250,500,1000]}]
grid = grid_search.GridSearchCV(RandomForestClassifier(),p2,cv=cv_iterator)
#grid = grid_search.GridSearchCV(LogisticRegression(),p1,cv=cv_iterator)
grid.fit(X,y)
#grid = grid_search.GridSearchCV(svm.SVC(),params,cv=cv_iterator)
#grid.fit(X,y)
#
#results
print grid.best_params_
print ''
print grid.best_score_


checking cross validatiojn classification
model = svm.SVC(kernel='sigmoid',C=100,gamma=0.01)
scores = cross_val_score(model,X,y,cv=cv_iterator)
pred = cross_val_predict(model,X,y,cv=cv_iterator)
print scores
print metrics.accuracy_score(y,pred)
print metrics.confusion_matrix(y,pred)
print metrics.classification_report(y,pred)


#Visualize  feature ranking
    
model1 = RandomForestClassifier() #250: n_estimators
model2 = ExtraTreesClassifier() #1000: n_estimators
parameters = [{"n_estimators":[100,150,250,500,1000]}]
clf = grid_search.GridSearchCV(model1,parameters,cv=cv_iterator,scoring="accuracy")
clf.fit(X,y)
best_estimator = clf.best_estimator_
print ('Best hyperparameters: ' + str(clf.best_params_))
print [str(clf.best_params_), clf.best_score_, best_estimator]
print ''
clf = grid_search.GridSearchCV(model2,parameters,cv=cv_iterator,scoring="accuracy")
clf.fit(X,y)
best_estimator = clf.best_estimator_
print ('Best hyperparameters: ' + str(clf.best_params_))
print [str(clf.best_params_), clf.best_score_, best_estimator]
option 1 - trees classifier
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
#    forest = RandomForestClassifier(n_estimators=250)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
			axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
   print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
	  color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()
option 2 - eigen values 
cor_mat1 = np.corrcoef(X_train.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)  
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):
   plt.figure(figsize=(6, 4))
   plt.bar(range(len(eig_pairs)), var_exp, alpha=0.5, align='center',
		   label='individual explained variance')
   plt.step(range(len(eig_pairs)), cum_var_exp, where='mid',
			label='cumulative explained variance')
   plt.ylabel('Explained variance ratio')
   plt.xlabel('Principal components')
   plt.legend(loc='best')
   plt.tight_layout()

















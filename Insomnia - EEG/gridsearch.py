#Performs grid search on model to determine best paramters for SVM
import numpy as np
from sklearn import svm, grid_search
from sklearn.metrics import classification_report
from utils import *
from sklearn.externals import joblib
import sys

stage = sys.argv[1]

print '------------------------------------------------'
print '------------------------------------------------'
print ('Processing stage #' + str(stage))
print '------------------------------------------------'
print ''

#Main matrices: X, y & cv iterator 
cv_iterator = []

#index for cv iterator
index = 0
X = np.empty([0,114])
y = np.empty([0,])
X_test_folds = []
y_test_folds = []
z_test_folds =[]
for fold in range(0,1):
    
    ifname = 'Data_HI_'+str(stage)+'_1_'+str(fold)+'.pkl'
    Train,Valid,Test = joblib.load(ifname)
    
    #extract features and labels 
    X_train = np.concatenate([Train[0],Valid[0]])
    y_train = np.concatenate([Train[1],Valid[1]])
    X_test = Test[0]
    y_test = Test[1]
    z_test = Test[2]
   
    X_test_folds.append(X_test)
    y_test_folds.append(y_test)
    z_test_folds.append(z_test)


    #get the train + test indices for the current fold
    train_indices = []
    for i in range(0,len(X_train)):
        train_indices.append(index)
        index+=1
    test_indices = []
    for i in range(0,len(X_test)):
        test_indices.append(index)
        index+=1

    
    #add data to main X,y matrix + cv iterator
    X = np.concatenate([X,X_train,X_test])
    y = np.concatenate([y,y_train,y_test])
    iter_fold= []
    iter_fold.append(train_indices)
    iter_fold.append(test_indices)
    cv_iterator.append(iter_fold)
 
#now i have the main X,y matrices + the iterator for cv

#GridSearchCV
#Main
tuned_parameters = [{'kernel':['poly'],'C':[0.1,1,10],'degree':[2,3,4],'gamma':[0.1,1]},
                    {'kernel':['rbf','sigmoid'],'C':[0.1,1,10],'gamma':[0.1,1]},
                    {'kernel':['linear'],'C':[0.1,1,10]}]

clf = grid_search.GridSearchCV(svm.SVC(),tuned_parameters,cv=cv_iterator,n_jobs=1,verbose=100)
clf.fit(X,y)

#printing details
print('Final Statistics forstage')
print ''
print ('Best params set found')
print (clf.best_params_)
print ''
print 'grid scores:'
print (clf.grid_scores_)
print ''
print ('best training score =')
print (clf.best_score_)
print ''

    
#using best estimnator to get subject level accuracy
s_acc = []
e_acc = []
for i in range (0,len(X_test_folds)):
    print 'Details for fold #' + str(i)
    print '----------------------------------------'
    X_test1 = X_test_folds[i]
    y_test1 = y_test_folds[i]
    z_test1 = z_test_folds[i]
    y_pred = clf.predict(X_test1)
    epoch_accuracy = clf.score(X_test1,y_test1)
    print 'Epoch Accuracy =' + str(epoch_accuracy)
    print 'classification report:>'
    print classification_report(y_test1,y_pred)
    ofname = 'record_stage_'+str(stage)+'_fold_'+str(i)+'.csv'

    subject_accuracy =  subj_accuracy(y_pred,y_test1,z_test1,ofname)

    print 'Subject Accuracy =' + str(subject_accuracy)
    print ''
    print ''
    s_acc.append(subject_accuracy)
    e_acc.append(epoch_accuracy)
    
avg_s_accuracy = float(sum(s_acc))/len(s_acc)
avg_e_accuracy = float(sum(e_acc))/len(e_acc)
print 'Avg Subject Accuracy = ' + str(avg_s_accuracy)
print 'Avg Epoch Accuracy = ' + str(avg_e_accuracy)


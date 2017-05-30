#For each sleep stage:
#Use SVM with best parameters  to train the data.
#Data is EEG signals divided into 30s epochs
import numpy as np
from sklearn import svm
from utils import *
from sklearn.externals import joblib



for stage in range(0,6):
    
    print ('Processing stage #' + str(stage))
    print '------------------------------------------------'
    print ''
    
    y_list_c3 = []
    z_list_c3 = []
    y_list_c4 = []
    z_list_c4 = []
    for fold in range(0,5):
        print 'Current fold #' + str(fold)
        print ''
        #extract data
        ifname = 'Data_HI_'+str(stage)+'_1_'+str(fold)+'.pkl'
        Train,Valid,Test = joblib.load(ifname)
        
        #combine train + valid, get other data
        X_train = np.concatenate([Train[0],Valid[0]])
        y_train = np.concatenate([Train[1],Valid[1]])
        X_test = Test[0]
        y_test = Test[1]
        z_test = Test[2]
        
        #c3 darrays
        X_train_c3 = X_train[:,0:57]
        X_test_c3 = X_test[:,0:57]
        
        #c4 arrays
        X_train_c4 = X_train[:,57:114]
        X_test_c4 = X_test[:,57:114]
    
        print 'C3 classification'
        print '---------------------------------'
        clf = svm.SVC(kernel='rbf',C=0.01,gamma=0.01) #only stage 0
    #    clf = svm.SVC(kernel='rbf',C=1,gamma=0.01) #all the other stages
        clf.fit(X_train_c3,y_train)    
        y_pred = clf.predict(X_test_c3)
        y_score = clf.score(X_test_c3,y_test)
        print 'Epoch Accuracy - sklearn =' + str(y_score)
        ofname = 'record_stage_'+str(stage)+'_fold_'+str(fold)+'.csv'
        z_score =  subj_accuracy(y_pred,y_test,z_test,ofname)
        print 'Subject Accuracy  =' + str(z_score)
        print ''
        print ''
        #append values to final list
        y_list_c3.append(y_score)
        z_list_c3.append(z_score)
        
        print 'C4 classification'
        print '---------------------------------'
        clf1 = svm.SVC(kernel='rbf',C=0.01,gamma=0.01) #only stage 0
    #    clf1 = svm.SVC(kernel='rbf',C=1,gamma=0.01) #all the other stages
        clf1.fit(X_train_c4,y_train)    
        y_pred = clf1.predict(X_test_c4)
        y_score = clf1.score(X_test_c4,y_test)
        print 'Epoch Accuracy - sklearn =' + str(y_score)
        ofname = 'record_stage_'+str(stage)+'_fold_'+str(fold)+'.csv'
        z_score =  subj_accuracy(y_pred,y_test,z_test,ofname)
        print 'Subject Accuracy  =' + str(z_score)
        print ''
        print ''  
        #append values to final list
        y_list_c4.append(y_score)
        z_list_c4.append(z_score)
        
        
    
    #Calculate Averages across all folds
    y_score_avg_c3 = float(sum(y_list_c3))/len(y_list_c3)
    z_score_avg_c3 = float(sum(z_list_c3))/len(z_list_c3)
    print 'Average Epoch Accuracy =' + str(y_score_avg_c3)
    print 'Average Subject Accuracy =' + str(z_score_avg_c3)
    y_score_avg_c4 = float(sum(y_list_c4))/len(y_list_c4)
    z_score_avg_c4 = float(sum(z_list_c4))/len(z_list_c4)
    print 'Average Epoch Accuracy =' + str(y_score_avg_c4)
    print 'Average Subject Accuracy =' + str(z_score_avg_c4)
        
    

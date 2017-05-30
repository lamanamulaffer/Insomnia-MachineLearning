#Perform SVM on top 20 components determined by PCA
import numpy as np
from sklearn import svm
from utils import *
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import decomposition

#stage = sys.argv[1]
for stage in range(5,6):
    
    print ('Processing stage #' + str(stage))
    print '------------------------------------------------'
    print ''
    y_list = []
    z_list = []
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
        
        #c3 arrays
        X_train_c3 = X_train[:,0:57]
        X_test_c3 = X_test[:,0:57]
        
        #c4 arrays
        X_train_c4 = X_train[:,57:114]
        X_test_c4 = X_test[:,57:114]

        print 'C3 + C4 classification'
        print '---------------------------------'
        sklearn_pca = sklearnPCA(n_components = 20)
        X1_train = sklearn_pca.fit_transform(X_train)
        X1_test = sklearn_pca.fit_transform(X_test)
        clf0 = svm.SVC(kernel='rbf', C=0.01,gamma=0.01)
        clf0.fit(X1_train,y_train)
        y_pred = clf0.predict(X1_test)
        y_score = clf0.score(X1_test,y_test)
        print 'Epoch Accuracy - sklearn =' + str(y_score)
        ofname = 'csvoutputs/record_stage_'+str(stage)+'_fold_'+str(fold)+'_allfeatures.csv'
        z_score =  subj_accuracy(y_pred,y_test,z_test,ofname)
        print 'Subject Accuracy  =' + str(z_score)
        print ''
        print ''
        #append values to final list
        y_list.append(y_score)
        z_list.append(z_score)
    
        print 'C3 classification'
        print '---------------------------------'
        sklearn_pca = sklearnPCA(n_components = 20)
        X1_train_c3 = sklearn_pca.fit_transform(X_train_c3)
        X1_test_c3 = sklearn_pca.fit_transform(X_test_c3)
#        clf = svm.SVC(kernel='rbf',C=0.01,gamma=0.01) #only stage 0
        clf = svm.SVC(kernel='rbf',C=1,gamma=0.01) #all the other stages
        clf.fit(X1_train_c3,y_train)    
        y_pred = clf.predict(X1_test_c3)
        y_score = clf.score(X1_test_c3,y_test)
        print 'Epoch Accuracy - sklearn =' + str(y_score)
        ofname = 'csvoutputs/record_stage_'+str(stage)+'_fold_'+str(fold)+'_c3.csv'
        z_score =  subj_accuracy(y_pred,y_test,z_test,ofname)
        print 'Subject Accuracy  =' + str(z_score)
        print ''
        print ''
        #append values to final list
        y_list_c3.append(y_score)
        z_list_c3.append(z_score)
        
        print 'C4 classification'
        print '---------------------------------'
        sklearn_pca = sklearnPCA(n_components = 20)
        X1_train_c4 = sklearn_pca.fit_transform(X_train_c4)
        X1_test_c4 = sklearn_pca.fit_transform(X_test_c4)
#        clf1 = svm.SVC(kernel='rbf',C=0.01,gamma=0.01) #only stage 0
        clf1 = svm.SVC(kernel='rbf',C=1,gamma=0.01) #all the other stages
        clf1.fit(X1_train_c4,y_train)    
        y_pred = clf1.predict(X1_test_c4)
        y_score = clf1.score(X1_test_c4,y_test)
        print 'Epoch Accuracy - sklearn =' + str(y_score)
        ofname = 'csvoutputs/record_stage_'+str(stage)+'_fold_'+str(fold)+'_c4.csv'
        z_score =  subj_accuracy(y_pred,y_test,z_test,ofname)
        print 'Subject Accuracy  =' + str(z_score)
        print ''
        print ''  
        #append values to final list
        y_list_c4.append(y_score)
        z_list_c4.append(z_score)
        
        
    
    #Calculate Averages across all folds
    y_score_avg = float(sum(y_list))/len(y_list)
    z_score_avg = float(sum(z_list))/len(z_list)
    print 'Average Epoch Accuracy =' + str(y_score_avg)
    print 'Average Subject Accuracy =' + str(z_score_avg)
    y_score_avg_c3 = float(sum(y_list_c3))/len(y_list_c3)
    z_score_avg_c3 = float(sum(z_list_c3))/len(z_list_c3)
    print 'Average Epoch Accuracy =' + str(y_score_avg_c3)
    print 'Average Subject Accuracy =' + str(z_score_avg_c3)
    y_score_avg_c4 = float(sum(y_list_c4))/len(y_list_c4)
    z_score_avg_c4 = float(sum(z_list_c4))/len(z_list_c4)
    print 'Average Epoch Accuracy =' + str(y_score_avg_c4)
    print 'Average Subject Accuracy =' + str(z_score_avg_c4)
        
    

#Plot feature contributions across the 3 combinations (C3+C4, C3, C4)
#This will be used to determine the number of top features to select in PCA
import numpy as np
from sklearn import svm
from utils import *
from sklearn.externals import joblib
from matplotlib import pyplot as plt

#stage = sys.argv[1]
for stage in range(0,1):
    
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

        #c3 + c4
        X = X_train
        cor_mat1 = np.corrcoef(X.T)
        eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
#        for ev in eig_vecs:
#            np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
#        print ('everything ok!')
#        
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
            plt.savefig('plots/plot_stage_' + str(stage) + '_fold_' + str(fold) + '_allfeatures.png')

       #c3 arrays
       X_train_c3 = X_train[:,0:57]
       X = X_train_c3
       
       cor_mat1 = np.corrcoef(X.T)
       eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
       for ev in eig_vecs:
           np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
       print ('everything ok!')
       
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
           plt.savefig('plots/plot_stage_' + str(stage) + '_fold_' + str(fold) + '_c3.png')
           
       X_test_c3 = X_test[:,0:57]
       
       #c4 arrays
       X_train_c4 = X_train[:,57:114]
       X = X_train_c4
       cor_mat1 = np.corrcoef(X.T)
       eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
       for ev in eig_vecs:
           np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
       print ('everything ok!')
       
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
           plt.savefig('plots/plot_stage_' + str(stage) + '_fold_' + str(fold) + '_c4.png')
           

   


       
        
  
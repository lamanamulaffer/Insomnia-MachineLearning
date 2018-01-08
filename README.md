# Machine-Learning-Projects
This repository contains machine learning projects involving predicting if a person has insomnia in two ways.
Key words: Grid search, PCA, SelectKBest, SVM, LogisticRegression, MultinomialNB, Random Forest Classification, Recursive Feature Eliminication, Cross Validation

Patient data used is from 2 overnight sleep recordings of about 124 patients at a sleep clinic in Berlin. Since patient data cannot be posted online due to confidentiality reasons, the files in this respository shows only the code without the input data.
The repository has two folders: Insomnia - EEG and Insomnnia - Hypnogram.
The folder Insomnia - EEG contains code that predics if a person has insomni based on the overnight EEG recordings from at most 2 channels (C3 and C4).
The folder Insomnia-Hypnogram contains files that predict if a person has insomnia based on the hypnogram readings.

Insomnia - EEG
Goal: Predict if a person has insomnia based on overnight EEG recordings divided in 30s epochs.
Method: Predict if each epoch is "healthy" or "insomniac" (epoch-level prediction) and then predict if each subject if "healthy" or "insomniac" (subject-level prediction).
Files in the folder and brief description:
1. gridsearch.py uses GridSearchCV to determine best tuned paramters for the SVM model on the training data
2. svm.py trains data using SVM on the best parameters from grid search
3. plot.py visualizes the contribution of each feature to the accuracy of the model
4. svm_pca.py uses PCA to reduce the dimensionality of the features to improve accuracy of SVM model

Insomnia - Hypnogram
Goal: Predict if a person has insomnia based on hypnogram annotation files of overnight sleep recordings.
Method: Use  mutliple feature vectors, multiple ways of selecting the best features and multiple classifiers to determine best classifier for the prediction
Files in the folder and brief description:
1. raw_feature_extraction.py extracts raw features from annotaion file like # of epochs for each stage and # of transitions between stages.
2. feat_ext_all.py calculate more defined from raw features, eg: ratio of stages, ratio of transitions and ratio of stage pairs
3. feat_ext_nrem.py considers Wake, NREM and REM as sleep stages and calculate relevant features 
5. generate_folds.py  divides data into training and test sets for cross validation while ensuring an even balance of demographics between the folds
6. pipeline_other.py uses grid search for best parameters, visualizes the effectiveness of each feature and uses this to reduce dimenionality; dimensionality reduction done by both SelectKBest and PCA, and trains data based on multiple models; Logistic Regression, RandomForestClassifier, MultinomialNB
7. pipeline_svm.py selects best features, perform grid search for best parameters and trains an SVM based model with cross validation
Collaborators: Mustafa Shahin

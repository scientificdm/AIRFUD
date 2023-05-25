# Machione learning binary classification model for segment classification 
# Uses binary files with TF descriptors (produced by 'tf_vectors.py')
#
import numpy as np
import pickle
import sys

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import ShuffleSplit

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier

task = sys.argv[1] # '1' -- 1st classification, '2' -- 2nd classification or '3' -- 3rd classification

measure = sys.argv[2] # 'tf' or 'tfidf'
#%%
# Load the data:
f_in = open("data-x-"+measure+"_"+task+".pkl","rb")
f_in2 = open("data-y_"+task+".pkl","rb")

dataX = pickle.load(f_in)
dataY = pickle.load(f_in2)

f_in.close()
f_in2.close()
#%
# ML binary classification model:
k = 10 # k-fold cross validation

# Fix random seed to make results reproducible:
np.random.seed(89)

# Cross validation:
cv = ShuffleSplit(n_splits=k, test_size=1/float(k), random_state=0)

# number of classifiers:
n = 4
clf = np.zeros(n,object)

# define quality measures:
precision = np.zeros((n,k),float) # Precision of positive class
recall = np.zeros((n,k),float) # Recall of positive class
f1 = np.zeros((n,k),float) # F1-score of positive class
precision_neg = np.zeros((n,k),float) # Precision of negative class
recall_neg = np.zeros((n,k),float) # Recall of negative class
f1_neg = np.zeros((n,k),float) # F1-score of negative class
accuracy_weighted = np.zeros((n,k),float) # Overall accuracy (weighted)

fold = 0
for train_index, test_index in cv.split(dataX):
    
    print('fold',fold+1)
    
    # load training and test data:
    X_train = [dataX[i] for i in range(len(dataX)) if i in train_index]
    X_test = [dataX[i] for i in range(len(dataX)) if i in test_index]
    Y_train = [dataY[i] for i in range(len(dataY)) if i in train_index]
    Y_test = [dataY[i] for i in range(len(dataY)) if i in test_index]
    
    # define classifiers:   
    clf[0] = tree.DecisionTreeClassifier(criterion='entropy')
    clf[1] = RandomForestClassifier()
    clf[2] = svm.SVC()
    clf[3] = SGDClassifier()
        
    # learning:
    for i in range(n):
        print('Learning classifier',i+1)
        clf[i] = clf[i].fit(X_train,Y_train)
        
    # Cost-based validation:
    validation_cost = np.zeros(len(Y_test),float)
    validation_cost+=1
    if np.abs(len(Y_test)-np.sum(Y_test)) != 0:
        if np.sum(Y_test) > np.sum(np.invert(Y_test)):
            # if positive class is dominating:
            weight = len(Y_test)/float(2)/float(np.abs(len(Y_test)-np.sum(Y_test)))
            for i in range(len(Y_test)):
                if Y_test[i] == False:
                    validation_cost[i] = weight
        else:
            # if negative class is dominating:
            weight = len(Y_test)/float(2)/float(np.abs(len(Y_test)-np.sum(np.invert(Y_test))))
            for i in range(len(Y_test)):
                if Y_test[i] == True:
                    validation_cost[i] = weight
        
    # prediction and verification:
    for i in range(n):
        Y_test_pred = clf[i].predict(X_test)
        precision[i,fold] = precision_score(Y_test,Y_test_pred,average='binary')
        recall[i,fold] = recall_score(Y_test,Y_test_pred,average='binary')
        f1[i,fold] = f1_score(Y_test,Y_test_pred,average='binary')
        precision_neg[i,fold] = precision_score(Y_test,Y_test_pred,average='binary',pos_label=0)
        recall_neg[i,fold] = recall_score(Y_test,Y_test_pred,average='binary',pos_label=0)
        f1_neg[i,fold] = f1_score(Y_test,Y_test_pred,average='binary',pos_label=0)
        accuracy_weighted[i,fold] = accuracy_score(Y_test,Y_test_pred,sample_weight=validation_cost)
            
    fold+=1
#%
prec_avg = np.zeros(n,float)
rec_avg = np.zeros(n,float)
f1_avg = np.zeros(n,float)
prec_neg_avg = np.zeros(n,float)
rec_neg_avg = np.zeros(n,float)
f1_neg_avg = np.zeros(n,float)
acc_weight_avg = np.zeros(n,float)

for i in range(n):
    prec_avg[i] = np.mean(precision[i])
    rec_avg[i] = np.mean(recall[i])
    f1_avg[i] = np.mean(f1[i])
    prec_neg_avg[i] = np.mean(precision_neg[i])
    rec_neg_avg[i] = np.mean(recall_neg[i])
    f1_neg_avg[i] = np.mean(f1_neg[i])
    acc_weight_avg[i] = np.mean(accuracy_weighted[i])
#%%
print('Classifier'+'\t'+'Precision'+'\t'+'Recall'+'\t'+'F1'+'\t'+'Precision (neg)'+'\t'+'Recall (neg)'+'\t'+'F1 (neg)'+'\t'+'Accuracy (weighted)')
for i in range(n):
    if i==0:
        method = 'DecisionTrees'
    elif i==1:
        method = 'RandomForest'
    elif i==2:
        method = 'SVM'
    elif i==3:
        method = 'SGD'
    print(method+'\t'+str(np.round(prec_avg[i],4)).replace('.', ',')+'\t'+str(np.round(rec_avg[i],4)).replace('.', ',')+'\t'+str(np.round(f1_avg[i],4)).replace('.', ',')+'\t'+str(np.round(prec_neg_avg[i],4)).replace('.', ',')+'\t'+str(np.round(rec_neg_avg[i],4)).replace('.', ',')+'\t'+str(np.round(f1_neg_avg[i],4)).replace('.', ',')+'\t'+str(np.round(acc_weight_avg[i],4)).replace('.', ','))
#%%
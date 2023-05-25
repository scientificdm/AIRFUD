# Vector similarity model for segment classification
#
import numpy as np
import pickle
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
# Vector similarity approach:
predY = np.zeros(len(dataX),bool)

# Determine indexes of positive and negative segments
posIndexes = [i for i in range(len(dataX)) if dataY[i]]
negIndexes = [i for i in range(len(dataX)) if not dataY[i]]

# iterate over the data:
for i in range(len(dataX)):
    # compute avg (mean) similarity with positive examples:
    simTrue = np.mean([np.sum(dataX[i]*dataX[j]) for j in posIndexes if i != j])
        
    # compute avg (mean) similarity with negative examples:
    simFalse = np.mean([np.sum(dataX[i]*dataX[j]) for j in negIndexes if i != j])
    
    if simTrue > simFalse:
        predY[i] = True
    else:
        predY[i] = False
        
    if np.mod(i,100) == 0:
        print('Computing vector similarity',i)
#%
# Cost-based validation:
validation_cost = np.zeros(len(dataY),float)
validation_cost+=1
if np.abs(len(dataY)-np.sum(dataY)) != 0:
    if np.sum(dataY) > np.sum(np.invert(dataY)):
        # if positive class is dominating:
        weight = len(dataY)/float(2)/float(np.abs(len(dataY)-np.sum(dataY)))
        for i in range(len(dataY)):
            if dataY[i] == False:
                validation_cost[i] = weight
    else:
        # if negative class is dominating:
        weight = len(dataY)/float(2)/float(np.abs(len(dataY)-np.sum(np.invert(dataY))))
        for i in range(len(dataY)):
            if dataY[i] == True:
                validation_cost[i] = weight
                
# Compute quality metrics:
accuracy_weighted = accuracy_score(dataY,predY,sample_weight=validation_cost)
precision = precision_score(dataY,predY)
recall = recall_score(dataY,predY)
f1 = f1_score(dataY,predY,average='binary')
precision_neg = precision_score(dataY,predY,pos_label=0)
recall_neg = recall_score(dataY,predY,pos_label=0)
f1_neg = f1_score(dataY,predY,average='binary',pos_label=0)

print('\nDetected:',sum(predY),'out of',str(sum(dataY))+'/'+str(len(dataY)))
print("Blocks positive:",sum(np.array(predY)*np.array(dataY)),'\n')

print('Precision:',str(round(precision,4)).replace('.', ','))
print('Recall:',str(round(recall,4)).replace('.', ','))
print('F1-score:',str(round(f1,4)).replace('.', ','))
print('Precision (negative):',str(round(precision_neg,4)).replace('.', ','))
print('Recall (negative):',str(round(recall_neg,4)).replace('.', ','))
print('F1-score (negative):',str(round(f1_neg,4)).replace('.', ','))
print('Accuracy (weighted):',str(round(accuracy_weighted,4)).replace('.', ','))
#%%
# Segment classification using trigger words
#
import numpy as np
import pickle
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from nltk.stem.snowball import FrenchStemmer

task = sys.argv[1] # '1' -- 1st classification, '2' -- 2nd classification or '3' -- 3rd classification

n = int(sys.argv[2]) # window size: 1..10
#%%
# Load data:
f_in = open("segments_"+task+".pkl","rb")

data = pickle.load(f_in)

f_in.close()
        
# Print some stats:
print('Total segments:',len(data))
print('Positive segments:',len([i for i in range(len(data)) if data[i][2]]))
#%%
# An approach based on trigger words
def wordInWindow(inputData,testWord,blockId,windowSize):
    startIndex = blockId - (windowSize // 2)
    endIndex = startIndex + windowSize
    if startIndex < 0:
        startIndex = 0
    if endIndex > len(inputData):
        endIndex = len(inputData)
    
    result = False
    for i in range(startIndex,endIndex):
        if (testWord != 'réglement') or ((testWord == 'réglement') and not (('non réglement' in inputData[i][0].lower()) or ('pas réglement' in inputData[i][0].lower()))):
            if testWord in inputData[i][0]:
                result = True
            
    return result

# Parse trigger words file:
f = open("trigger_words", "r")

triggerWords = []

for line in f:
    try:
        textLine = line.strip()
        if (not '#' in textLine) and (textLine != ''):
            triggerWords.append(textLine)            
    except ValueError:
        print('Invalid input:',line)
                
f.close()

# Define a stemmer:
stemmer = FrenchStemmer()

# Lemmatize trigger words:
triggerWordsStemmed = [stemmer.stem(word) for word in triggerWords]

# Verify the blocks:
dataY = []
predY = []

detectedWords = []

# Check presence of trigger words:
flag = False
for i in range(len(data)):
    tmp = []
    for word in triggerWordsStemmed:
        if wordInWindow(data,word,i,n):
            flag = True
            tmp.append(word)
    if flag:
        predY.append(True)
        flag = False
    else:
        predY.append(False)
    dataY.append(data[i][2])
    detectedWords.append(tmp)
    
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

# Compute Precision and Recall:
accuracy = accuracy_score(dataY,predY)
accuracy_weighted = accuracy_score(dataY,predY,sample_weight=validation_cost)
precision = precision_score(dataY,predY)
recall = recall_score(dataY,predY)
f1 = f1_score(dataY,predY,average='binary')
precision_neg = precision_score(dataY,predY,pos_label=0)
recall_neg = recall_score(dataY,predY,pos_label=0)
f1_neg = f1_score(dataY,predY,average='binary',pos_label=0)

print('Detected+:',sum(predY),'out of',str(sum(dataY))+'/'+str(len(dataY)))
print("TP:",sum(np.array(predY)*np.array(dataY)))
print('Detected-:',len(dataY)-sum(predY),'out of',str(len(dataY)-sum(dataY))+'/'+str(len(dataY)))
print("TN:",sum(~np.array(predY)*~np.array(dataY)),'\n')

print('Accuracy:',str(round(accuracy,4)).replace('.', ','))
print('Accuracy (weighted):',str(round(accuracy_weighted,4)).replace('.', ','),'\n')
print('Precision:',str(round(precision,4)).replace('.', ','))
print('Recall:',str(round(recall,4)).replace('.', ','))
print('F1-score:',str(round(f1,4)).replace('.', ','),'\n')
print('Precision (negative):',str(round(precision_neg,4)).replace('.', ','))
print('Recall (negative):',str(round(recall_neg,4)).replace('.', ','))
print('F1-score (negative):',str(round(f1_neg,4)).replace('.', ','))
#%%

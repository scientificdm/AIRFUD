# Data loader : multi-label segments -> 1st, 2nd or 3rd classification
#
import pickle
import numpy as np
import sys

from sklearn.model_selection import train_test_split

task = sys.argv[1] # '1' -- 1st classification, '2' -- 2nd classification or '3' -- 3rd classification
#%%
# List of filenames to process:
fileNames = ["PLU_Montpellier_ZONE-A","PLU_Montpellier_ZONE-N","PLU_Montpellier_ZONE-AU0","PLU_Montpellier_ZONE-14AU","PLU_Montpellier_ZONE-5AU","PLU_Montpellier_ZONE-4AU1","PPRI_Montpellier","PPRI_Grabels","PLU_Grabels_All"]

# Load all files:
data = []
for f_name in fileNames:
    # Parse txt files with segments:
    f = open("Corpus_Extracted_Segments/"+f_name+"_Segments.txt", "r")
    
    label = False
    segmentText = ""
    labelText = ""
    for line in f:
        try:
            if '>>>' in line:
                # memorize previous segment:
                if segmentText != "":
                    if task == '1':
                        data.append((segmentText.strip(),f_name,label))
                    elif task == '2':
                        if (labelText == 'Soft') or (labelText == 'Verifiable') or (labelText == 'Non-verifiable'):
                            data.append((segmentText.strip(),f_name,label))
                    elif task == '3':
                        if (labelText == 'Verifiable') or (labelText == 'Non-verifiable'):
                            data.append((segmentText.strip(),f_name,label))
                    segmentText = ""
                # determine label of current segment:
                labelText = line.split('>>>')[1].strip()
                if task == '1':
                    if labelText == 'False':
                        label = False
                    elif (labelText == 'Soft') or (labelText == 'Verifiable') or (labelText == 'Non-verifiable'):
                        label = True
                elif task == '2':
                    if labelText == 'Soft':
                        label = False
                    elif (labelText == 'Verifiable') or (labelText == 'Non-verifiable'):
                        label = True
                elif task == '3':
                    if labelText == 'Non-verifiable':
                        label = False
                    elif labelText == 'Verifiable':
                        label = True
                else:
                    print('Warning! Incorrect format')
            else:
                segmentText += line
                    
        except ValueError:
            print('Invalid input:',line)
    
    # Memorize the last segment:
    if task == '1':
        data.append((segmentText.strip(),f_name,label))
    elif task == '2':
        if (labelText == 'Soft') or (labelText == 'Verifiable') or (labelText == 'Non-verifiable'):
            data.append((segmentText.strip(),f_name,label))
    elif task == '3':
        if (labelText == 'Verifiable') or (labelText == 'Non-verifiable'):
            data.append((segmentText.strip(),f_name,label))
    
    f.close()
    
# Print some stats:
print('Total segments:',len(data))
print('Positive segments:',len([i for i in range(len(data)) if data[i][2]]),'\n')
print(np.round(len([i for i in range(len(data)) if data[i][2]])*100/len(data),2), '% of examples are positive\n')
#%%
## Save labeled segments as a binary file:
f_out = open("segments_"+task+".pkl","wb")

pickle.dump(data,f_out)

f_out.close()
#%%
# Extract labels and text:
text = [data[i][0] for i in range(len(data))]
labels = [int(data[i][2]) for i in range(len(data))]

# Split the data into train and test set:
val_ratio = 0.2

# Fix random seed to make results reproducible:
np.random.seed(98)

# Indices of the train and validation splits stratified by labels:
train_index, test_index = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)

# Check some stats:
print('Number of examples in train and test sets:', len(train_index), len(test_index))
print('Positive examples in train and test sets:', len([i for i in range(len(data)) if data[i][2] and (i in train_index)]), len([i for i in range(len(data)) if data[i][2] and (i in test_index)]))
#%%
# Save results:
data_test = [data[i] for i in range(len(data)) if i in test_index]
data_train = [data[i] for i in range(len(data)) if i in train_index]

# Save test set as a binary file:
f_out = open("test_set_"+task+".pkl","wb")
pickle.dump(data_test,f_out)
f_out.close()

# Save original training set as a binary file:
f_out = open("train_set_"+task+"_orig.pkl","wb")
pickle.dump(data_train,f_out)
f_out.close()
#%%
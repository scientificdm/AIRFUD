# Construction of term frequency vectors for vectorsim and ML models
#
import numpy as np
import pickle
import sys

from nltk.probability import FreqDist
from nltk.stem.snowball import FrenchStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

task = sys.argv[1] # '1' -- 1st classification, '2' -- 2nd classification or '3' -- 3rd classification
#%%
# Load data:
f_in = open("segments_"+task+".pkl","rb")

data = pickle.load(f_in)

f_in.close()
        
# Print some stats:
print('Total segments:',len(data))
print('Positive segments:',len([i for i in range(len(data)) if data[i][2]]))
#%%
# Build feature vectors for representing segments, save them as separate files

stemmer = FrenchStemmer() # define a stemmer

def stemmText(document):
    # Tokenize text:
    wordTokens = word_tokenize(document) # separate (tokenize) words
    
    # Remove stop words:
    stopWords = set(stopwords.words('french')) # list of stop words
    filteredText = [w.lower() for w in wordTokens if (not w.lower() in stopWords) and (w.isalpha()) and (len(w) > 1)] # text in lower case without stop words
    
    # Get stemmed representation of terms:
    stemmer = FrenchStemmer()
    stemmedText = [stemmer.stem(word) for word in filteredText]
    
    return stemmedText

# Define terms (vocabulary):
text = ""
# Merge all the text:
for block in data:
    text+=block[0]

terms = stemmText(text)

# Compute frequencies (TF):
termFrequencies = FreqDist(terms)
#%
# Select top k most frequent words:
topTermsFreq = [(k, v) for k, v in sorted(termFrequencies.items(), reverse=True, key=lambda item: item[1])]

# Dimensionality:
k = len(topTermsFreq) # use full vocabulary

topKTerms = [k for (k, v) in topTermsFreq]

# Construct vectors to represent blocks:
dataX = np.zeros((len(data),k),float)
dataX_tf = np.zeros((len(data),k),float)
dataX_idf = np.zeros((len(data),k),float)
dataY = np.zeros(len(data),bool)

for i in range(len(data)):
    # tokenize block:
    blockStemmed = stemmText(data[i][0])
    for j in range(len(topKTerms)):
        # compute frequencies:
        dataX[i,j] = sum([w==topKTerms[j] for w in blockStemmed]) # compute TF
    if sum(dataX[i]) != 0:
        dataX_tf[i] = dataX[i]/float(sum(dataX[i]))
    else:
        dataX_tf[i] = 0
    dataY[i] = data[i][2] # copy true label
    if np.mod(i,1000) == 0:
        print('Constructing feature vectors based on TF',i)

for i in range(len(data)):
    for j in range(len(topKTerms)):
        if float(sum(dataX[:,j])) != 0:
            dataX_idf[i,j] = dataX_tf[i,j]*np.log(float(len(data))/float(sum(dataX[:,j]))) # compute TF-IDF
        else:
            dataX_idf[i,j] = 0
    if np.mod(i,100) == 0:
        print('Constructing feature vectors based on TF-IDF',i)
#%%
# Save constructed vectors as binary files:
f_out = open("data-x-tf_"+task+".pkl","wb")
pickle.dump(dataX_tf,f_out)
f_out.close()

f_out = open("data-x-tfidf_"+task+".pkl","wb")
pickle.dump(dataX_idf,f_out)
f_out.close()

f_out = open("data-y_"+task+".pkl","wb")
pickle.dump(dataY,f_out)
f_out.close()
#%%

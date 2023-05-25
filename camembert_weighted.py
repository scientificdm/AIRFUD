from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import AdamW

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pickle
import numpy as np
import sys

from tqdm import trange

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

task = sys.argv[1] # '1' -- 1st classification, '2' -- 2nd classification or '3' -- 3rd classification
dataset = sys.argv[2] # 'orig' (original), 'augm-1' (augmented 1), 'augm-2' (augmented 2), etc. 
device_name = sys.argv[3] # 'cuda:0', 'cuda:1', etc.
#%%
# Open the data:
print("Open set: "+dataset+" classification: "+task)

f_in = open("train_set_"+task+"_"+dataset+".pkl","rb")
train_data = pickle.load(f_in)
f_in.close()

f_in = open("test_set_"+task+".pkl","rb")
test_data = pickle.load(f_in)
f_in.close()

average_results = []

# Fix random seed to make results reproducible:
torch.manual_seed(191)

# Repeat experiment 10 times, report averaged results:
for j in range(10):
    print("\nExperiment",j+1,"--------------------------------------------------------------")
    # Data preprocessing:
    
    model_name = "camembert-base" #"camembert/camembert-base", "camembert/camembert-large"
    
    # Define tokenizer:
    tokenizer = CamembertTokenizer.from_pretrained(model_name,do_lower_case=True)
    
    # Tokenize training and test sets:
    tokenizer_train = tokenizer([train_data[i][0] for i in range(len(train_data))], padding="longest", truncation = True, return_tensors="pt")
    
    tokenizer_test = tokenizer([test_data[i][0] for i in range(len(test_data))], padding="longest", truncation = True, return_tensors="pt")
            
    # Define Dataloaders:
    batch_size = 16
    
    train_set = TensorDataset(tokenizer_train['input_ids'], 
                              tokenizer_train['attention_mask'], 
                              torch.tensor([int(train_data[i][2]) for i in range(len(train_data))]))
    
    test_set = TensorDataset(tokenizer_test['input_ids'], 
                             tokenizer_test['attention_mask'], 
                             torch.tensor([int(test_data[i][2]) for i in range(len(test_data))]))
    
    train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )
    
    validation_dataloader = DataLoader(
                test_set,
                sampler = SequentialSampler(test_set),
                batch_size = batch_size
            )
    #%
    # Define the model:
    device = device_name if torch.cuda.is_available() else 'cpu'
    
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
    
    model.to(device)
    #%
    # Define parameters and metrics to optimize:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=10e-8)
    
    def flatten_preds(preds):
      preds = np.argmax(preds, axis = 1).flatten()
      
      return preds.tolist()
    #%
    # Training and validation:
    epochs = 10
    
    train_loss_set = []
    count = 0
    best_acc_weight = 0
    best_results = []
    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):  
        # Tracking variables for training
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
      
        # Train the model
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Add batch to device CPU or GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Get loss value
            loss = outputs[0]
            # Add it to train loss list
            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
        
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
    
        print("\nTrain loss: {}".format(tr_loss/nb_tr_steps))
        
        # Tracking variables for validation
        val_preds = []
        val_labels = []
        # Validation of the model
        model.eval()
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to device CPU or GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs =  model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:2]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            labels = b_labels.to('cpu').numpy()
            # Update predicted and true labels
            val_preds += flatten_preds(logits)
            val_labels += labels.tolist()
            
        # Cost-based validation:
        validation_cost = np.zeros(len(val_labels),float)
        validation_cost+=1
        if np.abs(len(val_labels)-np.sum(val_labels)) != 0:
            if np.sum(val_labels) > np.sum(np.invert(val_labels)):
                # if positive class is dominating:
                weight = len(val_labels)/float(2)/float(np.abs(len(val_labels)-np.sum(val_labels)))
                for i in range(len(val_labels)):
                    if val_labels[i] == False:
                        validation_cost[i] = weight
            else:
                # if negative class is dominating:
                weight = len(val_labels)/float(2)/float(np.abs(len(val_labels)-np.sum(np.invert(val_labels))))
                for i in range(len(val_labels)):
                    if val_labels[i] == True:
                        validation_cost[i] = weight
        
        val_acc = accuracy_score(val_labels,val_preds)
        val_acc_weight = accuracy_score(val_labels,val_preds,sample_weight=validation_cost)
        val_prec = precision_score(val_labels,val_preds,average='binary')
        val_rec = recall_score(val_labels,val_preds,average='binary')
        val_f1 = f1_score(val_labels,val_preds,average='binary')
        val_prec_neg = precision_score(val_labels,val_preds,average='binary',pos_label=0)
        val_rec_neg = recall_score(val_labels,val_preds,average='binary',pos_label=0)
        val_f1_neg = f1_score(val_labels,val_preds,average='binary',pos_label=0)
        
        count+=1
        
        # Save best results:
        if val_acc_weight > best_acc_weight:
            best_acc_weight = val_acc_weight
            best_results = (count, val_acc, val_acc_weight, val_prec, val_rec, val_f1, val_prec_neg, val_rec_neg, val_f1_neg)
        
        # Calculate validation metrics:
        print("\nEpoch: ",count)
        print("\t - Validation Accuracy: {:.4f}".format(val_acc))
        print("\t - Validation Accuracy (weighted): {:.4f}".format(val_acc_weight))
        print("\t - Validation Precision: {:.4f}".format(val_prec))
        print("\t - Validation Recall: {:.4f}".format(val_rec))
        print("\t - Validation F1-score: {:.4f}\n".format(val_f1))
        print("\t - Validation Precision (neg): {:.4f}".format(val_prec_neg))
        print("\t - Validation Recall (neg): {:.4f}".format(val_rec_neg))
        print("\t - Validation F1-score (neg): {:.4f}\n".format(val_f1_neg))
    
    # Best results of an epoch:
    print("\nBest results:")
    print("\nEpoch: ",best_results[0])
    print("\t - Validation Accuracy: {:.4f}".format(best_results[1]))
    print("\t - Validation Accuracy (weighted): {:.4f}".format(best_results[2]))
    print("\t - Validation Precision: {:.4f}".format(best_results[3]))
    print("\t - Validation Recall: {:.4f}".format(best_results[4]))
    print("\t - Validation F1-score: {:.4f}\n".format(best_results[5]))
    print("\t - Validation Precision (neg): {:.4f}".format(best_results[6]))
    print("\t - Validation Recall (neg): {:.4f}".format(best_results[7]))
    print("\t - Validation F1-score (neg): {:.4f}\n".format(best_results[8]))
    average_results.append(best_results)

# Determine best weighted accuracy:
max_acc_weight = np.max([i[2] for i in average_results])
index_best = [i for i in range(len(average_results)) if average_results[i][2]==max_acc_weight][0]

# Output best and averaged results:
print("\n-------------------------------------------------------")
print("\t - Averaged F1-score: {:.4f}\n".format(np.mean([i[5] for i in average_results])))
print("\t - Best F1-score: {:.4f}\n".format(average_results[index_best][5]))
print("\t - Averaged F1-score (neg): {:.4f}\n".format(np.mean([i[8] for i in average_results])))
print("\t - Best F1-score (neg): {:.4f}\n".format(average_results[index_best][8]))
print("\t - Max weighted accuracy: {:.4f}\n".format(max_acc_weight))
#%%
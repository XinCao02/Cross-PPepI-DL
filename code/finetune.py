# in_device = input('Choose CUDA Device to Train On (0~3):')
import time as T
T_start = T.time()
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import math
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from metrics import *


import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from finetune_dp import dataset, trainloader, testloader, LS_testloader, npy_file, npy_file, LS_npy_file, bs, es, ns, total, seen_prot, in_device, trainset, args
from models import GCNN, AttGNN
from torch_geometric.data import DataLoader as DataLoader_n

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device(f"cuda:{in_device}")


### To define the model name
if 'LS' in npy_file:
   LS = 'LS'
else:
   LS = 'S'
print(f"MODEL SAVED TO: Human_features/Different_data/GCN_{LS}_{args.train}k.pth")

# print(f"\t\t======== LOADING ACCOMPLISHED! TRAINING ON {device} ========")
# print(len(dataset))
# print(len(trainloader))
# print(f'\t\ttrainloader ele 0 = {print(trainloader)}')
# print(len(testloader))


# print(f'LS_testloader Length = {len(LS_testloader)}')
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/5)

 
#utilities
milestones = [2,5,7,9]
gamma = 0.5

def train(model, device, trainloader, optimizer, epoch):
    # print(f'Training on {len(trainloader)} samples.....')
    model.train()
    loss_func = nn.MSELoss()
    predictions_tr = torch.Tensor()
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    labels_tr = torch.Tensor()

    leave = False
    if epoch == 2:
      pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=75, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=True)
    else:
      pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=75, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False)
    for count, (prot_1, prot_2, label) in pbar:
        prot_1 = prot_1.to(device)
        prot_2 = prot_2.to(device)
        optimizer.zero_grad()
        output = model(prot_1, prot_2)
        predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
        labels_tr = torch.cat((labels_tr, label.view(-1, 1).cpu()), 0)
        loss = loss_func(output, label.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

        # Update progress bar with loss information
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    scheduler.step()
    labels_tr = labels_tr.detach().numpy()
    predictions_tr = predictions_tr.detach().numpy()
    acc_tr = get_accuracy(labels_tr, predictions_tr, 0.35)
    # print(f'\t>>> Train {epoch}: [==============================]  |  loss {loss.item():.9f}  |  acc {acc_tr:.6f}%')
    train_losses.append(loss.item())
    train_acc.append(acc_tr)
    
 

def predict(model, device, loader, test=False):
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
    if test:
      pbar = tqdm(loader, total=len(loader), ncols=75, desc=f"Test Set", unit="batch", leave=True)
    else:
      pbar = tqdm(loader, total=len(loader), ncols=75, desc=f"Val Epoch {epoch+1}", unit="batch", leave=False)
    for prot_1, prot_2, label in pbar:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
      output = model(prot_1, prot_2)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
  labels = labels.numpy()
  predictions = predictions.numpy()
  return labels.flatten(), predictions.flatten()
  
  

# training 

#early stopping
n_epochs_stop = 10
epochs_no_improve = 0
early_stop = False



### LOAD the model for finetuning
model = GCNN()
model.load_state_dict(torch.load(f"/home/PPI_test1/PPI_GNN/Human_features/Different_data/GCN_S_{args.pretrain}k.pth")) #path to load the model
print(f'\n>>> Loaded Pretrained Model on: \033[46m{args.pretrain}k S Dataset\033[0m')


# model = nn.DataParallel(model, device_ids=[2,3])
model.to(device)
num_epochs = 100
loss_func = nn.MSELoss()
min_loss = 100
best_accuracy = 0
optimizer =  torch.optim.Adam(model.parameters(), lr= 0.0001)
train_losses = []
train_acc = []
val_losses = []
val_acc = []

print(f"\t=============== CHOSEN MODEL: {model.name_get()} {device} =============")
print(f"\t======== Milestones= {milestones}, Gamma = {gamma} ========")

improved_epochs = []
T_train_begin = T.time()
for epoch in range(num_epochs):
  model.to(device)
  train(model, device, trainloader, optimizer, epoch+1)
  G, P = predict(model, device, testloader)
  #print( f'Predictions---------------------------------------------{P}')
  #print(f'Labels----------------------------------------------------{G}')
  loss = get_mse(G,P)
  accuracy = get_accuracy(G,P, 0.5)
  # print(f'\t>>> Val {epoch+1}:   [==============================]  |  loss {loss:.9f}  |  acc {accuracy:.6f}%')
  val_losses.append(loss)
  val_acc.append(accuracy)
  if(accuracy > best_accuracy):
    best_accuracy = accuracy
    best_acc_epoch = epoch
    torch.save(model.state_dict(), f"Human_features/Different_data/GCN_{LS}_{args.train}k.pth") #path to save the model
    improved_epochs.append((epoch,round(best_accuracy,1)))
    # print(f"  === Epoch {epoch}  |  Acc {best_accuracy:.2f}  :  Model Improved! ===")
  if(loss< min_loss):
    epochs_no_improve = 0
    min_loss = loss
    min_loss_epoch = epoch
  elif loss> min_loss :
    epochs_no_improve += 1
  if epoch > 5 and epochs_no_improve == n_epochs_stop:
    print(f'Epoch {epoch}: Early stopping!\n{improved_epochs}' )
    early_stop = True
    break

# print(f'>>>{len(train_losses),len(train_acc),len(val_losses),len(val_acc)}:\nTrain_Losses = {train_losses}')
# print(f'Train_Accuracies = {train_acc}')
# print(f'Val_Losses = {val_losses}')
# print(f'Val_Accuracies = {val_acc}')
# print(f'\n>>> min_val_loss : {min_loss} for epoch {min_loss_epoch}\n>>> best_val_accuracy : {best_accuracy} for epoch {best_acc_epoch}')
# print(f"Model saved! Time Consumed: {(T.time()-T_start)/60:.1f} minutes")
T_train = round(T.time() - T_train_begin, 2)



####### VALIDATION PHASE #######



model.eval()
predictions = torch.Tensor()
labels = torch.Tensor()
with torch.no_grad():
    for prot_1, prot_2, label in testloader:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      #print("H")
      #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
      output = model(prot_1, prot_2)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
labels = labels.numpy().flatten()
predictions = predictions.numpy().flatten()

loss = get_mse(labels, predictions)
acc = get_accuracy(labels, predictions, 0.5)
prec = precision(labels, predictions, 0.5)
sensitivity = sensitivity(labels, predictions,  0.5)
specificity_val = specificity(labels, predictions, 0.5)
f1_val = f_score(labels, predictions, 0.5)
mcc = mcc(labels, predictions,  0.5)
auroc = auroc(labels, predictions)
auprc = auprc(labels, predictions)


print("\n\t\033[46m   > > > > > > >   VALIDATION SET METRICS   < < < < < < <   \033[0m")
print(f"{'ACC':<7} {'REC':<7} {'SPC':<7} {'PRC':<7} {'F1S':<7} {'MCC':<7} {'AUROC':<7} {'AUPRC':<7} {'Loss':<7}")
print(f"{'-'*70}")
print(f"{acc:<7.2f} {sensitivity*100:<7.2f} {specificity_val*100:<7.2f} {prec*100:<7.2f} {f1_val*100:<7.2f} {mcc*100:<7.2f} {auroc*100:<7.2f} {auprc*100:<7.2f} {loss*100:<7.2f}")



####### TESTING PHASE #######


T_test_begin = T.time()

true_labels, predictions = predict(model, device, LS_testloader, test=True)
# print(f'pred:{predictions}')
binary_predictions = (predictions > 0.5).astype(int)

num_ones_l = np.sum(true_labels)
num_zeros_l = len(true_labels) - num_ones_l
num_ones = np.sum(binary_predictions)
num_zeros = len(binary_predictions) - num_ones



# Compute metrics
# Print TP, FP, TN, FN
tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()

# Calculating the metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * recall) / (precision + recall)

# Print results in a horizontal table format with metrics as percentages
print("\n\t\033[46m   > > > > > > >      TEST SET METRICS      < < < < < < <   \033[0m")
print(f"{'ACC':<7} {'REC':<7} {'SPC':<7} {'PRC':<7} {'F-1':<7} {'TP':<7} {'FP':<7} {'TN':<7} {'FN':<7}")
print("-"*70)
print(f"{accuracy * 100:<7.2f} {recall * 100:<7.2f} {specificity * 100:<7.2f} {precision * 100:<7.2f} {f1 * 100:<7.2f} {tp:<7} {fp:<7} {tn:<7} {fn}")


### Calculate Metrics for BS ES NS

def calculate_metrics(true_labels, binary_predictions, indices):
    # Extract subset based on indices
    true_subset = true_labels[indices]
    pred_subset = binary_predictions[indices]
    
    # Calculate TP, FP, TN, FN
    TP = np.sum((true_subset == 1) & (pred_subset == 1))
    FP = np.sum((true_subset == 0) & (pred_subset == 1))
    TN = np.sum((true_subset == 0) & (pred_subset == 0))
    FN = np.sum((true_subset == 1) & (pred_subset == 0))
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'Accuracy': accuracy,
        'Recall': recall,
        'Specificity': specificity,
        'Precision': precision,
        'F1 Score': f1_score
    }

# Calculate metrics for each subset
bs_metrics = calculate_metrics(true_labels, binary_predictions, bs)
es_metrics = calculate_metrics(true_labels, binary_predictions, es)
ns_metrics = calculate_metrics(true_labels, binary_predictions, ns)

# Function to convert metrics to percentages and round them
def format_percentage(value):
    return round(value * 100, 2)

# Create a DataFrame from the dictionaries with formatted values
metrics_table = pd.DataFrame({
    'Subset': ['BS', 'ES', 'NS'],
    'TP': [bs_metrics['TP'], es_metrics['TP'], ns_metrics['TP']],
    'FP': [bs_metrics['FP'], es_metrics['FP'], ns_metrics['FP']],
    'TN': [bs_metrics['TN'], es_metrics['TN'], ns_metrics['TN']],
    'FN': [bs_metrics['FN'], es_metrics['FN'], ns_metrics['FN']],
    'ACC': [format_percentage(bs_metrics['Accuracy']), format_percentage(es_metrics['Accuracy']), format_percentage(ns_metrics['Accuracy'])],
    'REC': [format_percentage(bs_metrics['Recall']), format_percentage(es_metrics['Recall']), format_percentage(ns_metrics['Recall'])],
    'SPE': [format_percentage(bs_metrics['Specificity']), format_percentage(es_metrics['Specificity']), format_percentage(ns_metrics['Specificity'])],
    'PRE': [format_percentage(bs_metrics['Precision']), format_percentage(es_metrics['Precision']), format_percentage(ns_metrics['Precision'])],
    'F1S': [format_percentage(bs_metrics['F1 Score']), format_percentage(es_metrics['F1 Score']), format_percentage(ns_metrics['F1 Score'])]
})

# Set 'Subset' as the index for better readability
metrics_table.set_index('Subset', inplace=True)

# Print the table
print(metrics_table)
T_test = round(T.time() - T_test_begin, 2)

# Sample Output
# print('\n\t> Sample Output on Test Set <\n>>> Labl',np.array([int(x) for x in true_labels[:30]]),'\t',np.array([int(x) for x in true_labels[-30:]]),f'\t1={int(num_ones_l)}\t 0={int(num_zeros_l)}')
# print('>>> Pred',binary_predictions[:30],'\t',binary_predictions[-30:],f'\t1={num_ones}\t 0={num_zeros}')

####### TEXT OUTPUT PHASE #######
with open(f'Dataset_finetuneLOG{in_device}.txt', 'a') as file:
   file.write(f'\n\n\t=============================  {T.strftime("%Y-%m-%d %H:%M:%S", T.localtime())}  ============================\n')
   file.write(f'\t========================= Pre-Train Dataset Size: {args.pretrain}  ========================\n')
   file.write(f'\t==================== Train: {npy_file}  ===================\n')
   file.write(f'\t===================== Test : {LS_npy_file}  =====================\n')
   file.write(f"\t===== Number of Seen Proteins: {len(seen_prot)}   >>> BS: {len(bs)*100/total:.1f}  |  ES: {len(es)*100/total:.1f}  |  NS: {len(ns)*100/total:.1f}======\n")
   file.write("\n\t\033[46m   > > > > > > >   VALIDATION SET METRICS   < < < < < < <   \033[0m\n")
   file.write(f"{'ACC':<7} {'REC':<7} {'SPC':<7} {'PRC':<7} {'F1S':<7} {'MCC':<7} {'AUROC':<7} {'AUPRC':<7} {'Loss':<7}\n")
   file.write(f"{'-'*70}\n")
   file.write(f"{acc:<7.2f} {sensitivity*100:<7.2f} {specificity_val*100:<7.2f} {prec*100:<7.2f} {f1_val*100:<7.2f} {mcc*100:<7.2f} {auroc*100:<7.2f} {auprc*100:<7.2f} {loss*100:<7.2f}\n")
   
   file.write("\n\t\033[46m   > > > > > > >      TEST SET METRICS      < < < < < < <   \033[0m\n")
   file.write(f"{'ACC':<7} {'REC':<7} {'SPC':<7} {'PRC':<7} {'F-1':<7} {'TP':<7} {'FP':<7} {'TN':<7} {'FN':<7}\n")
   file.write("-"*70+"\n")
   file.write(f"{accuracy * 100:<7.2f} {recall * 100:<7.2f} {specificity * 100:<7.2f} {precision * 100:<7.2f} {f1 * 100:<7.2f} {tp:<7} {fp:<7} {tn:<7} {fn}\n\n")
   file.write(f'>>>>>Time Consumed for Training:{T_train}s\n>>>>>Time Consumed for Testing:{T_test}s\n')

   for line in metrics_table.to_string(index=False).split('\n'):
      file.write(line + '\n')
   print(f'>>>>>Time Consumed for Training:{T_train}s\n>>>>>Time Consumed for Testing:{T_test}s\n')
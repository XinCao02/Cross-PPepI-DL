import time as T
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import math
import sklearn
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from metrics import *
import pandas as pd

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = torch.device("cuda:1")
T_start = T.time()

import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from data_prepare import dataset, trainloader, testloader, LS_testloader, bs, es, ns
from models import GCNN, AttGNN
from torch_geometric.data import DataLoader as DataLoader_n

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device('cuda')


def predict(model, device, loader):
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
        ct = 0
        pbar = tqdm(loader, total=len(loader), ncols=80, desc=f"Testing ...", unit="batch", leave=False)
      
        for prot_1, prot_2, label in pbar:
            prot_1 = prot_1.to(device)
            prot_2 = prot_2.to(device)
            #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
            output = model(prot_1, prot_2)
            predictions = torch.cat((predictions, output.cpu()), 0)
            labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
            # print(1)
            # print('>>>labels', len(labels))
            # print('>>>pred', len(predictions))

    labels = labels.numpy()
    predictions = predictions.numpy()

    return labels.flatten(), predictions.flatten()



model = GCNN()
model.load_state_dict(torch.load('/home/PPI_test1/PPI_GNN/Human_features/GCN.pth'))
model.to(device)

true_labels, predictions = predict(model, device, LS_testloader)
# print(f'pred:{predictions}')
binary_predictions = (predictions > 0.5).astype(int)

num_ones_l = np.sum(true_labels)
num_zeros_l = len(true_labels) - num_ones_l
num_ones = np.sum(binary_predictions)
num_zeros = len(binary_predictions) - num_ones

print('>>> Labl',np.array([int(x) for x in true_labels[:30]]),'\t',np.array([int(x) for x in true_labels[-30:]]),f'\t1={int(num_ones_l)}\t 0={int(num_zeros_l)}')
print('>>> Pred',binary_predictions[:30],'\t',binary_predictions[-30:],f'\t1={num_ones}\t 0={num_zeros}')



# Compute metrics

# Print TP, FP, TN, FN
tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
print(f'\nTP\tFP\tTN\tFN\n{tp}\t{fp}\t{tn}\t{fn}\n')

# # Metrics
# accuracy = accuracy_score(true_labels, binary_predictions)
# precision = precision_score(true_labels, binary_predictions, average='weighted')
# recall = recall_score(true_labels, binary_predictions, average='weighted')
# f1 = f1_score(true_labels, binary_predictions, average='weighted')
# specificity = tn / (tn + fp)
# # Print the results
# print(f"{'Accuracy':<7} {'Recall':<7} {'Specificity':<7} {'Precision':<7} {'F1 Score':<7}")
# print("-" * 60)
# print(f"{accuracy * 100:<7.2f} {recall * 100:<7.2f} {specificity * 100:<7.2f} {precision * 100:<7.2f} {f1 * 100:<7.2f}")


# Calculating the metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * recall) / (precision + recall)

# Print results in a horizontal table format with metrics as percentages
print(f"{'ACC':<7} {'REC':<7} {'SPC':<7} {'PRC':<7} {'F-1':<7}")
print("-" * 37)
print(f"{accuracy * 100:<7.2f} {recall * 100:<7.2f} {specificity * 100:<7.2f} {precision * 100:<7.2f} {f1 * 100:<7.2f}")

### Calculate Metrics for BS ES NS
print('>>> Total Number of Test Samples:',true_labels.shape)

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

# Assuming bs_metrics, es_metrics, and ns_metrics are the dictionaries for the subsets
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

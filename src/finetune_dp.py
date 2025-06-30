
# note that this custom dataset is not prepared on the top of geometric Dataset(pytorch's inbuilt)
import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join 
from torch_geometric.data import Data
import argparse

print("\n============================ FINETUNE MODE =========================\n")

parser = argparse.ArgumentParser(description="Dataset Argument Parser")
parser.add_argument('--train', type=str, help='Size of the trainset (e.g., 0.5k)')
parser.add_argument('--pretrain', type=str, help='Size of the trainset (e.g., 0.5k)')
parser.add_argument('--test', type=str, help='Size of the testset (e.g., 0.5k)', default="50")
parser.add_argument('--cuda', type=str, help='CUDA to run on')
parser.add_argument('--spsd', type=int, help='Random Seed for Data Split', default=42)
parser.add_argument('--seed', type=int, help='Random Seed for Data Split', default=7)
args = parser.parse_args()

in_device = args.cuda
trainset = str(args.train)
seed = args.spsd
data_seed = args.seed

# processed_dir="Human_features/processed/"
processed_dir="own_data/processed/"
# npy_file = "Human_features/npy_file_new(human_dataset).npy"
npy_file = f"own_data/LS_Hm_{args.train}k_seed{data_seed}_11.npy"
LS_npy_file = f"own_data/LS_Hm_{args.test}k_v{(int(in_device)+1)*7}_11.npy"
npy_ar = np.load(npy_file, allow_pickle=True)
npy_ar[:,6] = npy_ar[:,6].astype(int)

print(f'\n>>> Loaded Train file: \033[46m{npy_file}\033[0m | Split Seed: {seed}')
ones = np.where((npy_ar[:, 6] == 1) | (npy_ar[:, 6] == '1'))[0]
num1 = len(ones)
# print(f'Ones = {num1}  |  1 at line = {ones[:15]}')
zeros = np.where((npy_ar[:, 6] == 0) | (npy_ar[:, 6] == '0'))[0]
num0 = len(zeros)
# print(f'Zeros = {num0}  |  0 at line = {zeros[:15]}')
# print(f'Ratios:\t\t1:  {(num1/(num1+num0))*100:2f}%       0:  {(num0/(num1+num0))*100:2f}%')

# print(f'npy_file shape = {npy_ar.shape}')
# print(npy_ar[0])
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.loader import DataLoader as DataLoader_n

class LabelledDataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.load(npy_file, allow_pickle=True)
      self.processed_dir = processed_dir
      self.protein_1 = self.npy_ar[:,2]
      self.protein_2 = self.npy_ar[:,5]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
      return(self.n_samples)

    def __getitem__(self, index):
      try:
        try:
          # print("\n",self.protein_1[index])
          prot_1 = os.path.join(self.processed_dir, "AF-"+self.protein_1[index]+"-F1-model_v4"+".pt")
          prot_2 = os.path.join(self.processed_dir, "AF-"+self.protein_2[index]+"-F1-model_v4"+".pt")

          # print('\n###')
          # print(prot_1)
          # print(prot_2)
          # print('###')

          #print(f'Second prot is {prot_2}')
          # print('fetched',prot_1)
          prot_1 = torch.load(glob.glob(prot_1)[0])
          # print(f'prot1:\n{prot_1}\nlabel:\n{torch.tensor(self.label[index])}')
          #print(f'Here lies {glob.glob(prot_2)}')
          prot_2 = torch.load(glob.glob(prot_2)[0])
          
          # prot_1 = bump(prot_1)
          # prot_2 = bump(prot_2)
          return prot_1, prot_2, torch.tensor(self.label[index])
        except:
          prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
          prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")

          # print('\n### Encountered Former Name')
          # print(prot_1)
          # print(prot_2)
          # print('###')

          prot_1 = torch.load(glob.glob(prot_1)[0])
          prot_2 = torch.load(glob.glob(prot_2)[0])
          prot_1 = bump(prot_1)
          prot_2 = bump(prot_2)        
          return prot_1, prot_2, torch.tensor(self.label[index])
      except:
         print(f'\n>>> Protein {self.protein_1[index], self.protein_2[index]} Not Found!')
      
# in pyg 2.*

def bump(g):
    return Data.from_dict(g.__dict__)



dataset = LabelledDataset(npy_file = npy_file ,processed_dir= processed_dir)
# print(f'dataset size{len(dataset)}')
final_pairs =  np.load(npy_file, allow_pickle=True)
size = final_pairs.shape[0]
# print(f"Num_Samples in total: {size}")
torch.manual_seed(seed)
#print(math.floor(0.8 * size))
#Make iterables using dataloader class  
trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size) ])
#print(trainset[0])
batch_size = 128
trainloader = DataLoader_n(dataset= trainset, batch_size= batch_size, num_workers = 0)
testloader = DataLoader_n(dataset= testset, batch_size= batch_size, num_workers = 0)
# print(f"Num_Batches in train/test: {len(trainloader)}/{len(testloader)}")
# print(f"Batch Size = {batch_size}")






### LONGSHORT TEST

class LabelledDataset_test(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = npy_file
      self.processed_dir = processed_dir
      self.protein_1 = self.npy_ar[:,2]
      self.protein_2 = self.npy_ar[:,5]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
      return(self.n_samples)

    def __getitem__(self, index):
      try:
        try:
          # print("\n",self.protein_1[index])
          prot_1 = os.path.join(self.processed_dir, "AF-"+self.protein_1[index]+"-F1-model_v4"+".pt")
          prot_2 = os.path.join(self.processed_dir, "AF-"+self.protein_2[index]+"-F1-model_v4"+".pt")

          # print('\n###')
          # print(prot_1)
          # print(prot_2)
          # print('###')

          #print(f'Second prot is {prot_2}')
          # print('fetched',prot_1)
          prot_1 = torch.load(glob.glob(prot_1)[0])
          # print(f'prot1:\n{prot_1}\nlabel:\n{torch.tensor(self.label[index])}')
          #print(f'Here lies {glob.glob(prot_2)}')
          prot_2 = torch.load(glob.glob(prot_2)[0])
          
          # prot_1 = bump(prot_1)
          # prot_2 = bump(prot_2)
          return prot_1, prot_2, torch.tensor(self.label[index])
        except:
          prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
          prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")

          # print('\n### Encountered Former Name')
          # print(prot_1)
          # print(prot_2)
          # print('###')

          prot_1 = torch.load(glob.glob(prot_1)[0])
          prot_2 = torch.load(glob.glob(prot_2)[0])
          prot_1 = bump(prot_1)
          prot_2 = bump(prot_2)        
          return prot_1, prot_2, torch.tensor(self.label[index])
      except:
         print(f'\n>>> Protein {self.protein_1[index], self.protein_2[index]} Not Found!')


# 除去LS_test中的重复数据
train_npy = np.load(npy_file, allow_pickle='True')
test_npy = np.load(LS_npy_file, allow_pickle='True')
print(f">>> Shape before filter: {test_npy.shape}")
set1 = set(map(tuple, train_npy))
set2 = set(map(tuple, test_npy))
LS_npy_data = np.array([line for line in set2 if line not in set1])
print(f">>> Shape after filter: {LS_npy_data.shape}")

LS_testset = LabelledDataset_test(npy_file = LS_npy_data ,processed_dir= processed_dir)
LS_testloader = DataLoader_n(dataset= LS_testset, batch_size= batch_size, num_workers = 0)
# print(f"Num_Batches in LS_test: {len(LS_testloader)}")

print(f'>>> Loaded Test  file: \033[46m{LS_npy_file}\033[0m')
npy_ar = np.load(LS_npy_file, allow_pickle=True)
ones = np.where((npy_ar[:, 6] == 1) | (npy_ar[:, 6] == '1'))[0]
num1 = len(ones)
# print(f'Ones = {num1}  |  1 at line = {ones[:15]}')
zeros = np.where((npy_ar[:, 6] == 0) | (npy_ar[:, 6] == '0'))[0]
num0 = len(zeros)
# print(f'Zeros = {num0}  |  0 at line = {zeros[:15]}')
# print(f'Ratios:\t\t1:  {(num1/(num1+num0))*100:2f}%       0:  {(num0/(num1+num0))*100:2f}%\n')


### BS ES NS:
# Load npy files
npy_data = np.load(npy_file, allow_pickle=True)
# LS_npy_data = np.load(LS_npy_file, allow_pickle=True)
# Extract columns 3 and 6 from npy_data and store in a set called seen_prot
col3_set = set(npy_data[:, 2])  # Column indices are 0-based
col6_set = set(npy_data[:, 5])
seen_prot = col3_set.union(col6_set)  # Combine both sets
# Initialize counters for BS, ES, and NS
BS, ES, NS = 0, 0, 0
bs,es,ns = [],[],[]
# Iterate through each row of LS_npy_data
for i, row in enumerate(LS_npy_data):
    prot1 = row[2]  # Column 3
    prot2 = row[5]  # Column 6
    # Count how many proteins are in seen_prot
    prot_count_in_seen = sum([prot1 in seen_prot, prot2 in seen_prot])
    if prot_count_in_seen == 2:
        BS += 1
        bs.append(i)
    elif prot_count_in_seen == 1:
        ES += 1
        es.append(i)
    else:
        NS += 1
        ns.append(i)

# Print the results
total = BS+ES+NS
print(f">>> Number of Seen Proteins: {len(seen_prot)}   >>> BS: {BS*100/total:.1f}  |  ES: {ES*100/total:.1f}  |  NS: {NS*100/total:.1f}")

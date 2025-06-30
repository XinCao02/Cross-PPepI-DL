import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from data_prepare import LS_testloader, LS_npy_file, args
from models import GCNN, AttGNN
from tqdm import tqdm


# Load the model
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
model = GCNN()
model2load = "/home/PPI_test1/PPI_GNN/Human_features/Different_data/GCN_S_20.0k.pth"
model.load_state_dict(torch.load(model2load)) #path to load the model
print(f"\n>>> Loaded Model: \033[32m{model2load.split('/')[-2:]}\033[0m")

model.to(device)
model.eval()
predictions = torch.Tensor()
labels = torch.Tensor()
with torch.no_grad():
    # try:
    pbar = tqdm((LS_testloader), total=len(LS_testloader))
    for prot_1, prot_2, label in pbar:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      #print("H")
      #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
      output = model(prot_1, prot_2)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
    # except Exception as e:
      #  print(e)
labels = labels.numpy().flatten()
predictions = predictions.numpy().flatten()

# Print the Predictions & Labels in pairs:
print('Now printing Labels/Prediction:')
for i in range(15):
    print(f'Label: {labels[i]} Prediction: {predictions[i]}')
print(len(labels), "more ...")

# Print Binary form of Predictions:
bin_classes = pred_to_classes(labels, predictions, 0.5)
print('\nBinary Classes:', bin_classes)

# Save Binary Classes to the 7th col of .npy file:
# print(LS_npy_file, LS_npy_file.split('/')[-1])
if args.new_pyg != 0:
  out_file = np.load(f"pred_out/{LS_npy_file.split('/')[-1]}")
  print(f'\n>>> Before Saving Predictions, the .npy file is as follows:\n{out_file}')
  out_file[:, 6] = predictions
  np.save(f"pred_out/{LS_npy_file.split('/')[-1]}", out_file)
  print(f'\n>>> Binary Class Labels saved to {LS_npy_file.split("/")[-1]}.npy !')
  print(f'  > Now, the .npy file is as follows:\n{out_file}')

### Only test the metrics if in new_pyg is in 0 mode
if args.new_pyg == 0:
  loss = get_mse(labels, predictions)
  acc = get_accuracy(labels, predictions, 0.5)
  prec = precision(labels, predictions, 0.5)
  sensitivity = sensitivity(labels, predictions,  0.5)
  specificity = specificity(labels, predictions, 0.5)
  f1 = f_score(labels, predictions, 0.5)
  mcc = mcc(labels, predictions,  0.5)
  auroc = auroc(labels, predictions)
  auprc = auprc(labels, predictions)

  print('\n======= The Metrics are as follows =======:')
  # print(f'loss : {loss}')
  print(f'Accuracy : {acc}')
  print(f'precision: {prec}')
  print(f'Sensititvity :{sensitivity}')
  print(f'specificity : {specificity}')
  print(f'f-score : {f1}')
  print(f'MCC : {mcc}')
  print(f'AUROC: {auroc}')
  print(f'AUPRC: {auprc}')





##############################################################################################################




# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from metrics import *
# from data_prepare import LS_testloader
# from models import GCNN, AttGNN
# from tqdm import tqdm


# # Load the model
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
# model = GCNN()
# model.load_state_dict(torch.load("/home/PPI_test1/PPI_GNN/Human_features/Different_data/GCN_LS_20.0k.pth")) #path to load the model
# model.to(device)
# model.eval()
# predictions = torch.Tensor()
# labels = torch.Tensor()
# with torch.no_grad():
#     # try:
#     # pbar = tqdm((LS_testloader), total=len(LS_testloader))
#     for prot_1, prot_2, label in LS_testloader:
#       prot_1 = prot_1.to(device)
#       prot_2 = prot_2.to(device)
#       print("prot_1 & prot_2:", prot_1, prot_2)
#       print("H:")
#       print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
#       output = model(prot_1, prot_2)
#       print("\n> output:", output)
#       predictions = torch.cat((predictions, output.cpu()), 0)
#       print("> predictions:", predictions)
#       labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
#       print("> labels:", labels)
#     # except Exception as e:
#     #   print('Error:', e)
# labels = labels.numpy().flatten()
# predictions = predictions.numpy().flatten()

# # Print the Predictions & Labels in pairs:
# print('\n> Now printing Labels/Prediction:')
# for i in range(len(labels)):
#     print(f'\tLabel: {labels[i]}\tPrediction: {predictions[i]}')

# loss = get_mse(labels, predictions)
# acc = get_accuracy(labels, predictions, 0.5)
# prec = precision(labels, predictions, 0.5)
# sensitivity = sensitivity(labels, predictions,  0.5)
# specificity = specificity(labels, predictions, 0.5)
# f1 = f_score(labels, predictions, 0.5)
# mcc = mcc(labels, predictions,  0.5)
# auroc = auroc(labels, predictions)
# auprc = auprc(labels, predictions)

# print('\n======= The Metrics are as follows =======:')
# # print(f'loss : {loss}')
# print(f'Accuracy : {acc}')
# print(f'precision: {prec}')
# print(f'Sensititvity :{sensitivity}')
# print(f'specificity : {specificity}')
# print(f'f-score : {f1}')
# print(f'MCC : {mcc}')
# print(f'AUROC: {auroc}')
# print(f'AUPRC: {auprc}')

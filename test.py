import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import numpy as np

from model import BiLSTMVariationalAutoencoder
from dataset import ECG5000

import matplotlib.pyplot as plt

# Hyperparameters
DATA_PATH = 'data/ECG5000'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_SIZE = 128
SEQ_LEN = 140
INPUT_SIZE = 1

# Load data
test_data = np.load(os.path.join(DATA_PATH, 'test.npy'))
test_dataset = ECG5000(test_data)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
saved_state_dict = torch.load('model.pth')
base_model = BiLSTMVariationalAutoencoder(seq_len=SEQ_LEN, n_features=INPUT_SIZE, embedding_dim=HIDDEN_SIZE, device=DEVICE).double().to(DEVICE)
base_model.load_state_dict(saved_state_dict['model_dict'])
C = torch.tensor(saved_state_dict['C']).to(DEVICE)
R = torch.tensor(saved_state_dict['R']).to(DEVICE)

# Test
print('Testing...')
base_model.eval()
with torch.no_grad():
    label_scores = []
    data_bar = tqdm(test_loader, desc=f'Test')
    for batch_idx, (data, label) in enumerate(data_bar):
        data = data.to(DEVICE)
        embedding = base_model.encode(data)
        dist = torch.sum((embedding - C) ** 2, dim=1)

        pred = torch.where(dist < R, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))

        label_scores += list(
            zip(
                label.cpu().data.numpy().tolist(),
                pred.cpu().data.numpy().tolist()
            )
        )

    labels, scores = zip(*label_scores)
    labels = np.array(labels)
    scores = np.array(scores)

    # Calculate AUC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
    print('Accuracy: ', accuracy_score(labels, scores))
    print('Precision: ', precision_score(labels, scores))
    print('Recall: ', recall_score(labels, scores))
    print('F1: ', f1_score(labels, scores))
    print('AUC: ', roc_auc_score(labels, scores))
    

    fpr, tpr, threshold = roc_curve(labels, scores)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()






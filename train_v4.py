import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import numpy as np

from models.tcn_vae import TCN_VAE
from models.svdd import SVDD_Plus
from dataset import data_preprocessing
from utils import noise_mask, performance_display, positive_negative_sampling

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hyperparameters
DATA_PATH = 'data/ECG5000'
BATCH_SIZE = 16
PRE_EPOCHS = 10
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_SAMPLES = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENCODER_PARAMS = {
    'in_channels': 1,
    'channels': 40,
    'depth': 7,
    'reduced_size': 128,
    'out_channels': 64,
    'kernel_size': 3,
    'softplus_eps': 1e-3
}

DECODER_PARAMS = {
    'k': 64,
    'width': 140,
    'in_channels': 128,
    'channels': 40,
    'depth': 7,
    'out_channels': 1,
    'kernel_size': 3
}



# Load data
train_dataset, val_dataset, test_dataset = data_preprocessing(DATA_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
base_model = TCN_VAE(ENCODER_PARAMS, DECODER_PARAMS).double().to(DEVICE)
base_model.load_state_dict(torch.load('output/base_model.pth'))

class TCN_VAE_AD(nn.Module):
    def __init__(self, base_model, svdd_model):
        super().__init__()
        self.base_model = base_model
        self.svdd_model = svdd_model

    def forward(self, x):
        recon_x, z, (mu, sigma) = self.base_model(x)
        return recon_x, z, (mu, sigma), self.svdd_model(z)

# SVDD model
model = TCN_VAE_AD(base_model, SVDD_Plus(64)).double().to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.7)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
mse_criterion = nn.MSELoss().to(DEVICE)
bce_criterion = nn.BCELoss().to(DEVICE)

# Train
train_losses = []
train_val_losses = []
train_accs = []
train_val_accs = []
train_precisions = []
train_val_precisions = []
train_recalls = []
train_val_recalls = []
train_f1s = []
train_val_f1s = []
print('DeepSVDD Training...')
for epoch in range(EPOCHS):
    train_loss = 0
    model.train()
    data_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
    preds = []
    labels = []
    for batch_idx, (data, label) in enumerate(data_bar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        recon_data, z, (mu, sigma), anomaly_score = model(data)

        MSE = mse_criterion(recon_data, data)
        KLD =- 0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        BCE = bce_criterion(anomaly_score, label)

        loss = MSE + KLD + BCE

        data_bar.set_postfix(loss=loss.item(), mse_loss=MSE.item(), kld_loss=KLD.item(), bce_loss=BCE.item())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds.append(anomaly_score)
        labels.append(label)

    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)

    preds = torch.cat(preds)
    preds = (preds > 0.5).float()
    print(preds[preds == 1].shape[0])
    print(preds[preds == 0].shape[0])
    labels = torch.cat(labels)
    print(labels[labels == 1].shape[0])
    print(labels[labels == 0].shape[0])


    train_acc = accuracy_score(labels.cpu().data.numpy(), preds.cpu().data.numpy())
    train_accs.append(train_acc)
    train_precision = precision_score(labels.cpu().data.numpy(), preds.cpu().data.numpy())
    train_precisions.append(train_precision)
    train_recall = recall_score(labels.cpu().data.numpy(), preds.cpu().data.numpy())
    train_recalls.append(train_recall)
    train_f1 = f1_score(labels.cpu().data.numpy(), preds.cpu().data.numpy())
    train_f1s.append(train_f1)

    lr_scheduler.step()

    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            recon_data, z, (mu, sigma), anomaly_score = model(data)

            MSE = mse_criterion(recon_data, data)
            KLD =- 0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            BCE = bce_criterion(anomaly_score, label)

            loss = MSE + KLD + BCE

            # loss = torch.mean((1 - label) * torch.sum(torch.pow(embedding - C, 2), dim=1))
            val_loss += loss.item()
            pred = (anomaly_score > 0.5).float()
            val_preds.append(pred)
            val_labels.append(label)

    val_loss = val_loss / len(val_loader)
    train_val_losses.append(val_loss)

    val_preds = torch.cat(val_preds)
    print(val_preds[val_preds == 1].shape[0])
    print(val_preds[val_preds == 0].shape[0])
    val_labels = torch.cat(val_labels)

    # normal_preds_max = val_preds[val_labels == 0].max()
    # val_preds = (val_preds > normal_preds_max).float()

    val_acc = accuracy_score(val_labels.cpu().data.numpy(), val_preds.cpu().data.numpy())
    train_val_accs.append(val_acc)
    val_precision = precision_score(val_labels.cpu().data.numpy(), val_preds.cpu().data.numpy())
    train_val_precisions.append(val_precision)
    val_recall = recall_score(val_labels.cpu().data.numpy(), val_preds.cpu().data.numpy())
    train_val_recalls.append(val_recall)
    val_f1 = f1_score(val_labels.cpu().data.numpy(), val_preds.cpu().data.numpy())
    train_val_f1s.append(val_f1)

    print(f'Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}')

train_loss_display = {}
train_loss_display['train_loss'] = train_losses
train_loss_display['val_loss'] = train_val_losses
performance_display(train_loss_display, 'Training-Loss', 'output')

train_acc_display = {}
train_acc_display['train_acc'] = train_accs
train_acc_display['val_acc'] = train_val_accs
performance_display(train_acc_display, 'Training-ACC', 'output')

train_precision_display = {}
train_precision_display['train_precision'] = train_precisions
train_precision_display['val_precision'] = train_val_precisions
performance_display(train_precision_display, 'Training-Precision', 'output')

train_recall_display = {}
train_recall_display['train_recall'] = train_recalls
train_recall_display['val_recall'] = train_val_recalls
performance_display(train_recall_display, 'Training-Recall', 'output')

train_f1_display = {}
train_f1_display['train_f1'] = train_f1s
train_f1_display['val_f1'] = train_val_f1s
performance_display(train_f1_display, 'Training-F1', 'output')


# Save Model
torch.save({'C': C, 'R': R, 'model_dict': model.state_dict()}, 'output/model.pth')







import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import numpy as np

from models.tcn_vae import TCN_VAE
from dataset import data_preprocessing
from utils import noise_mask, performance_display, positive_negative_sampling

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hyperparameters
DATA_PATH = 'data/ECG5000'
BATCH_SIZE = 16
PRE_EPOCHS = 10
EPOCHS = 50
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

# Optimizer
optimizer = optim.Adam(base_model.parameters(), lr=LEARNING_RATE)

criterion = nn.MSELoss().to(DEVICE)

pretrain_losses = []
pretrain_val_losses = []

# Pre-Train
print('Pre-Training...')
for epoch in range(PRE_EPOCHS):
    pretrain_loss = 0
    base_model.train()
    data_bar = tqdm(train_loader, desc=f'Pre-Epoch {epoch + 1}/{PRE_EPOCHS}')
    for batch_idx, (data, label) in enumerate(data_bar):    
        data = data.to(DEVICE)

        data_np = data.cpu().data.numpy()
        mask = noise_mask(data_np, 0.3)
        data_masked = torch.tensor(data_np * mask).to(DEVICE) 

        optimizer.zero_grad()
        recon_x, z, (mu, sigma) = base_model(data_masked)
        MSE = criterion(recon_x, data)
        KLD = - 0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

        loss = MSE + KLD
        loss.backward(retain_graph=True)

        positive_samples, negative_samples = positive_negative_sampling(data_np, label, train_dataset.dataset)
        positive_samples = torch.from_numpy(positive_samples).permute(1, 0, 2).to(DEVICE)
        negative_samples = torch.from_numpy(negative_samples).permute(1, 0, 2).to(DEVICE)

        multiplicative_ratio = 1 / NUM_SAMPLES
        contrastive_loss = 0

        for i in range(NUM_SAMPLES):
            _, pos_i_embedding, _ = base_model(positive_samples[i])
            _, neg_i_embedding, _ = base_model(negative_samples[i])
            contrastive_loss += multiplicative_ratio * -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                                    z.view(-1, 1, DECODER_PARAMS['k']),
                                    pos_i_embedding.view(-1, DECODER_PARAMS['k'], 1)
                                    )))
            contrastive_loss += multiplicative_ratio * -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                                    z.view(-1, 1, DECODER_PARAMS['k']),
                                    neg_i_embedding.view(-1, DECODER_PARAMS['k'], 1)
                                    )))

        contrastive_loss.backward(retain_graph=True)



        data_bar.set_postfix(loss=loss.item(), mse_loss=MSE.item(), kld_loss=KLD.item(), contrastive_loss=contrastive_loss.item())
        # loss = loss + contrastive_loss

        optimizer.step()

    base_model.eval()
    pretrain_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(DEVICE)
            recon_x, z, (mu, sigma) = base_model(data)
            MSE = criterion(recon_x, data)
            KLD = - 0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            loss = MSE + KLD
            pretrain_loss += loss.item()
    pretrain_loss /= len(val_loader)
    pretrain_val_losses.append(pretrain_loss)

    val_loss = 0
    val_mse = 0
    val_kld = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(DEVICE)
            recon_x, z, (mu, sigma) = base_model(data)
            MSE = criterion(recon_x, data)
            KLD = - 0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

            loss = MSE + KLD
            val_loss += loss.item()
            val_mse += MSE.item()
            val_kld += KLD.item()
    val_loss /= len(val_loader)
    val_mse /= len(val_loader)
    val_kld /= len(val_loader)
    pretrain_val_losses.append(val_loss)

    print(f'Pre-Epoch {epoch + 1}/{PRE_EPOCHS} - Loss: {pretrain_loss:.4f} - Val Loss: {val_loss:.4f} - Val MSE: {val_mse:.4f} - Val KLD: {val_kld:.4f}')

pretrain_display = {}
pretrain_display['train_loss'] = pretrain_losses
pretrain_display['val_loss'] = pretrain_val_losses

performance_display(pretrain_display, 'Pre-Training', 'output')

torch.save(base_model.state_dict(), 'output/base_model.pth')

# Hyperparameters
# EPS = 0.5

print('Initialize hypersphere center c as the mean from an initial forward pass on the data.')
base_model.eval()

n_samples = 0
C = torch.zeros(base_model.embedding_dim).to(DEVICE)
embeds = []
with torch.no_grad():
    for data, label in train_loader:
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        embedding  = base_model.encode(data)
        embedding = embedding[label == 0]
        C += torch.sum(embedding, dim=0)
        n_samples += embedding.shape[0]
        embeds.append(embedding)

C /= n_samples
C = C.detach().cpu().numpy()

print(n_samples)
# C[(abs(C) < EPS) & (C < 0)] = -EPS
# C[(abs(C) < EPS) & (C > 0)] = EPS

print('Initialize hypersphere radius r as the maximum distance from c to any point in the data.')
R = 0
embeds = torch.cat(embeds, dim=0).cpu().numpy()
dists = np.linalg.norm(embeds - C, axis=1)
C = torch.tensor(C).to(DEVICE)
C = C.repeat(BATCH_SIZE, 1).double()
R = np.max(dists)
R = torch.tensor(R).to(DEVICE)
print(R)

optimizer = optim.Adam(base_model.parameters(), lr=LEARNING_RATE)
# optimizer.add_param_group({'params': C})
# optimizer.add_param_group({'params': R})
criterion = nn.BCELoss().to(DEVICE)

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
    base_model.train()
    data_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
    preds = []
    labels = []
    for batch_idx, (data, label) in enumerate(data_bar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        embedding = base_model.encode(data)

        pred = torch.sigmoid(torch.sum(torch.pow(embedding - C, 2), dim=1) - R**2)

        loss = criterion(pred, label)

        data_bar.set_postfix(loss=loss.item())

        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        preds.append(pred)
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

    base_model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            embedding = base_model.encode(data)

            pred = torch.sigmoid(torch.sum(torch.pow(embedding - C, 2), dim=1) - R**2)
            loss = criterion(pred, label)

            # loss = torch.mean((1 - label) * torch.sum(torch.pow(embedding - C, 2), dim=1))
            val_loss += loss.item()
            pred = (pred > 0.5).float()
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
torch.save({'C': C, 'R': R, 'model_dict': base_model.state_dict()}, 'output/model.pth')







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
    'out_channels': 32,
    'kernel_size': 3,
    'softplus_eps': 1e-3
}

DECODER_PARAMS = {
    'k': 32,
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
import torch
import torch.nn.functional as F

import config
from model import *

def train_fn(model, optimizer, epochs, dataloader, T):
    """
    Trains the model.

    Inputs:
    - model: model to train
    - optimizer: optimizer to use
    - epochs: int, number of epochs to train
    - dataloader: dataloader to use
    - T: int, number of timesteps

    """
    model.train()
    model.to(config.DEVICE)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            t = torch.rand(0, T, (config.BATCH_SIZE,), device = config.DEVICE).long()
            x_t, noise = forward_diffusion_sample(batch, t, device=config.DEVICE)
            noise_pred = model(x_t, t)
            loss_ = F.l1_loss(noise, noise_pred)

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            print(f"Epoch: {epoch} | Step: {i} | Loss: {loss_}")
import torch
from torch.utils.data import DataLoader

from datasets import get_data
from model import SimpleUnet
from train import train_fn
import config

def main():
    dataset = get_data()
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    model = SimpleUnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_fn(model, optimizer, config.EPOCH, dataloader, config.T)

if __name__ == "__main__":
    main()
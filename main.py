from torch.utils.data import DataLoader

from datasets import get_data
import config

def main():
    dataset = get_data()
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

if __name__ == "__main__":
    main()
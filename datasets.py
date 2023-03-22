import torch
import torchvision
from torchvision import transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import config

def get_data():
    """
    Returns the data from the Stanford Cars dataset.
    """
    data_transforms = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])

def show_image(dataset, num_sample=6, cols=3):
    """
    Displays images.

    Inputs:
    - dataset: dataset to display
    - num_sample: int, number of images to display
    - cols: int, number of columns to display
    """
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(dataset):
        if i >= num_sample:
            break
        plt.subplot(num_sample // cols + 1, cols, i + 1)
        plt.imshow(image[0])

def show_tensor_image(image):
    """
    Displays a tensor image.

    Inputs:
    - image: tensor, image to display
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def unit_test1():
    data = torchvision.datasets.StanfordCars(root=".", download=True)
    show_image(data)

if __name__ == "__main__":
    unit_test1()
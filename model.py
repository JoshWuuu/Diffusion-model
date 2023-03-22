import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from datasets import *
import config

def alpha_schedule(timesteps, start = 0.0001, end = 0.02):
    """
    Returns a beta value for the beta for noise generation.

    Inputs:
    - timesteps: int, number of timesteps
    - start: float, start value of beta, default 0.0001
    - end: float, end value of beta, default 0.02

    Returns:
    - sqrt_recip_alphas: tensor, sqrt(1 / alpha)
    - sqrt_alphas_cumprod: tensor, sqrt(cumprod(alpha))
    - sqrt_one_minus_alphas_cumprod: tensor, sqrt(1 - cumprod(alpha))
    - posterior_variance: tensor, beta * (1 - cumprod(alpha)) / (1 - cumprod(alpha))
    """
    betas = torch.linspace(start, end, timesteps)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance

def get_index_from_list(alphas, t, x_shape):
    """
    Returns a index t of vals

    Inputs:
    - alphas: tensor, cumulative sum of alphas
    - t: tensor, random number
    - x_shape: tuple, shape of the input tensor

    Returns:
    - out: tensor, index t of alphas
    """
    batch_size = t.shape[0]
    # Get index from batch of list, alphas (B, T), -1 focus on T dim, and out[i][j] = alphas[i][t[i][j]]
    out = alphas.gather(-1, t.cpu())
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    return out

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Returns a sample from the forward diffusion process.

    Inputs:
    - x_0: tensor, input image
    - t: tensor, random number
    - device: string, device to use

    Returns:
    - x_t: tensor, sample from the forward diffusion process
    """
    # Calculate alphas
    alphas = alpha_schedule(timesteps=1000, start=0.0001, end=0.02)
    sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = alphas

    # Calculate variance
    noise = torch.randn_like(x_0)

    # weight of mean and variance
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    # Calculate x_t, mean + variance * noise
    x_t = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) +  sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

    return x_t, noise.to(device)

def unit_test_forward():
    T = 1000
    dataset = get_data()
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, (idx/stepsize) + 1)
        image, noise = forward_diffusion_sample(image, t)
        show_tensor_image(image)

class Block(nn.Moudle):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


if __name__ == "__main__":
    pass
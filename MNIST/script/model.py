__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/16 02:50:08"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MNIST_Dataset(Dataset):    
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return self.image[idx, :]

class VAE(nn.Module):
    def __init__(self, dim_latent_vars, dim_image_vars, num_hidden_units):
        super(VAE, self).__init__()
        
        self.dim_latent_vars = dim_latent_vars
        self.dim_image_vars = dim_image_vars
        self.num_hidden_units = num_hidden_units
        
        self.encoder_fc1 = nn.Linear(dim_image_vars, num_hidden_units)
        self.encoder_fc2_mu = nn.Linear(num_hidden_units, dim_latent_vars, bias = True)
        self.encoder_fc2_logsigma2 = nn.Linear(num_hidden_units, dim_latent_vars, bias = True)

        self.decoder_fc1 = nn.Linear(dim_latent_vars, num_hidden_units, bias = True)
        self.decoder_fc2 = nn.Linear(num_hidden_units, dim_image_vars, bias = True)

    def encoder(self, x):
        h = self.encoder_fc1(x)
        h = torch.tanh(h)
        mu = self.encoder_fc2_mu(h)
        logsigma2 = self.encoder_fc2_logsigma2(h)
        sigma = torch.sqrt(torch.exp(logsigma2))
        return mu, sigma

    def sample_latent_var(self, mu, sigma):
        eps = Variable(sigma.data.new(sigma.size()).normal_())
        z = mu + sigma * eps
        return z

    def decoder(self, z):
        h = self.decoder_fc1(z)
        h = torch.tanh(h)
        h = self.decoder_fc2(h)
        p = torch.sigmoid(h)
        return p
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sample_latent_var(mu, sigma)
        p = self.decoder(z)        
        return mu, sigma, p
    
def loss_function(x, mu, sigma, p):
    cross_entropy = F.binary_cross_entropy(p, x) * x.size()[1]
    KLD = - 0.5 * torch.sum((1.0 + torch.log(sigma**2) - mu**2 - sigma**2))
    return cross_entropy + KLD / x.size()[0]

__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/19 16:13:31"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.gridspec as gridspec
import pickle
import torch
from torch.autograd import Variable
from model import *
import scipy.stats as stats
import sys

# vae = VAE(2, 784, 500)
# vae.cuda()
# vae.load_state_dict(torch.load('./output/vae_0.01.model'))

# num = 6
# z = Variable(torch.Tensor(torch.Size((num**2, 2))).normal_())
# samples = vae.decoder(z)
# samples = samples.data.numpy()

# fig = plt.figure(0, figsize = (6,6))
# gs = gridspec.GridSpec(num, num)
# gs.update(wspace = 0.025, hspace = 0.025)
# fig.clf()
# for n in range(num**2):
#     axes = plt.subplot(gs[n])
#     axes.imshow(samples[n,:].reshape(28,28), cmap = 'binary', vmax = 1, vmin = 0)
#     axes.axis('off')
#     # axes.get_xaxis().set_visible(False)
#     # axes.get_yaxis().set_visible(False)
# fig.savefig("./output/samples.png")

# num_poins = 20
# cdf1 = np.linspace(0, 1, num_poins + 2)[1:-1]
# cdf2 = np.linspace(0, 1, num_poins + 2)[1:-1]
# z1 = stats.norm.ppf(cdf1)
# z2 = stats.norm.ppf(cdf2)
# z = np.zeros((num_poins*num_poins, 2))
# for i in range(num_poins):
#     for j in range(num_poins):
#         z[i*num_poins + j, :] = [z1[i], z2[j]]
# z = Variable(torch.Tensor(z))
# samples = vae.decoder(z)
# samples = samples.data.numpy()

# fig = plt.figure(1, figsize = (10,10))
# gs = gridspec.GridSpec(num_poins, num_poins)
# gs.update(wspace=0.025, hspace=0.025)
# fig.clf()
# for n in range(samples.shape[0]):
#     axes = plt.subplot(gs[n])
#     axes.imshow(samples[n,:].reshape(28,28), cmap = 'binary', vmax = 1, vmin = 0)
#     axes.axis('off')    
#     # axes.get_xaxis().set_visible(False)
#     # axes.get_yaxis().set_visible(False)
# fig.savefig("./output/2D_manifold.png")

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image'].astype('float32')
train_label = data['train_label']
num_train_image = train_image.shape[0]
# train_data = MNIST_Dataset(train_image)
# #train_data_loader = DataLoader(train_data, batch_size = num_train_image, shuffle = False)
# train_data_loader = DataLoader(train_data, batch_size = 20000, shuffle = False)
# mu_all = []
# for idx, data in enumerate(train_data_loader):
#     print(idx)
#     inputs = Variable(data, volatile = True).cuda()
#     mu, sigma, p = vae.forward(inputs)
#     mu_all.append(mu.data.cpu().numpy())    
# mu = np.vstack(mu_all)

# with open("./output/latent_space_mu.pkl", 'wb') as file_handle:
#     pickle.dump(mu, file_handle)
    
with open("./output/latent_space_mu.pkl", 'rb') as file_handle:
    mu = pickle.load(file_handle)

#fig = plt.figure(0, figsize = (10,10))
fig = plt.figure(0)
fig.clf()
marker_list = ['bo', 'r^', 'm+', 'cx', 'bD', 'rD', 'ms', 'cs', 'cD', 'b^']
for i in range(10):
    idx = train_label == i
#    plt.plot(mu[idx, 0], mu[idx, 1], marker_list[i], label = i, alpha = 0.2, markersize = 4)
    plt.plot(mu[idx, 0], mu[idx, 1], '.', label = i, markersize = 4)
plt.legend(markerscale = 5)
plt.savefig('./output/latent_space_embedding.png')
#plt.show()






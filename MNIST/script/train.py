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
from model import *

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image'].astype('float32')
train_label = data['train_label']
test_image = data['test_image'].astype('float32')
test_label = data['test_label']

batch_size = 1000
train_data = MNIST_Dataset(train_image)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

vae = VAE(2, 784, 500)
vae.cuda()
weight_decay = 0.01
optimizer = optim.Adam(vae.parameters(), weight_decay = 0.01)
num_epoches = 1000
train_loss_epoch = []
test_loss_epoch = []

for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        inputs = Variable(data).cuda()
        optimizer.zero_grad()    
        mu, sigma, p = vae.forward(inputs)
        loss = loss_function(inputs, mu, sigma, p)
        loss.backward()
        optimizer.step()    
        print("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}".format(epoch, idx, loss.data[0]))
        running_loss.append(loss.data[0])        
    train_loss_epoch.append(np.mean(running_loss))

    running_loss = []    
    for idx, data in enumerate(test_data_loader):
        inputs = Variable(data).cuda()
        mu, sigma, p = vae.forward(inputs)
        loss = loss_function(inputs, mu, sigma, p)
        running_loss.append(loss.data[0])
    test_loss_epoch.append(np.mean(running_loss))
        
torch.save(vae.state_dict(), "./output/vae_{:.2f}.model".format(weight_decay))

with open('./output/loss.pkl', 'wb') as file_handle:
    pickle.dump({'train_loss_epoch': train_loss_epoch, 'test_loss_epoch': test_loss_epoch,}, file_handle)
   
fig = plt.figure(0)
fig.clf()
plt.plot(train_loss_epoch, label = "train", color = 'r')
plt.plot(test_loss_epoch, label = "test", color = 'b')
plt.ylim((140, 180))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title("Loss")
fig.savefig("./output/loss.png")

import random
from datetime import datetime
# importing data science libraries
import pandas as pd
import numpy as np
# PyTorch libs
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print("======= running with CUDA ========") 
# visualization

# transformed TS data set
def get_dataSet(TSData, b, f, step):
	allMb = [] # Mb belongs to lookback data (in-sample)
	allMf = [] # Mf belongs to forward data (out-of-sample)
	t = 0 # current time index
	datasetIdx = 0
	allPrices = TSData.values
	while t <= TSData.shape[0] - (b + f): 
		Mb = allPrices[t : (t + b),:]
		#print("before: " + str(Mb))
		#print("after: " + str(np.transpose(Mb)))
		Mf = allPrices[t + b : (t + b + f),:]
		allMb.append(np.transpose(Mb))
		allMf.append(np.transpose(Mf))
		t += step
	
	return allMb, allMf

class Discriminator(nn.Module):
	def __init__(self, nAssets):
		'''
		:param nAssets: int, number of stocks
        :param T: int, number of look back points b associated with M_b
		'''
		# The super call delegates the function call to the parent class, which is nn.Module in your case.
		# This is needed to initialize the nn.Module properly. Have a look at the Python docs 175 for more information.
		super(Discriminator, self).__init__()

		self.crit_conv1 = nn.Conv1d(in_channels = nAssets, out_channels = 2*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu1 = nn.LeakyReLU()
		self.crit_conv2 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 4*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu2 = nn.LeakyReLU()
		self.crit_conv3 = nn.Conv1d(in_channels = 4*nAssets, out_channels = 8*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu3 = nn.LeakyReLU()
		self.crit_conv4 = nn.Conv1d(in_channels = 8*nAssets, out_channels = 16*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu4 = nn.LeakyReLU()
		self.crit_conv5 = nn.Conv1d(in_channels = 16*nAssets, out_channels = 32*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.leakyrelu5 = nn.LeakyReLU()
		self.crit_dense6 = nn.Linear(in_features =  32*nAssets, out_features = 1)
		self.leakyrelu6 = nn.LeakyReLU()

	def forward(self, x):
		#print("input: " + str(x.shape))
		# Layer 1
		out = self.leakyrelu1(self.crit_conv1(x))
		#print("ouput layer 1: " + str(out.shape))

		# Layer 2:
		out = self.leakyrelu2(self.crit_conv2(out))
		#print("ouput layer 2: " + str(out.shape))

		# Layer 3:
		out = self.leakyrelu3(self.crit_conv3(out))
		#print("ouput layer 3: " + str(out.shape))

		# Layer 4:
		out = self.leakyrelu4(self.crit_conv4(out))
		#print("ouput layer 4: " + str(out.shape))

		# Layer 5:
		out = self.leakyrelu5(self.crit_conv5(out))
		#print("ouput layer 5: " + str(out.shape))

		# Final layer - critic value
		out = torch.flatten(out, start_dim = 1, end_dim =2)
		out = self.leakyrelu6(self.crit_dense6(out))
		#print("ouput final layer: " + str(out.shape))

		return out

class Generator(nn.Module):
	def __init__(self, nAssets, f):
		'''
		:param nAssets: int, number of stocks
        :param T: int, number of look back points b associated with M_b
		'''
		# The super call delegates the function call to the parent class, which is nn.Module in your case.
		# This is needed to initialize the nn.Module properly. Have a look at the Python docs 175 for more information.
		super(Generator, self).__init__()

		'''
		Conditioning
		'''
		# first layer
		# input 
		self.nAssets = nAssets
		self.cond_conv1 = nn.Conv1d(in_channels = nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu1 = nn.ReLU()

		# second layer
		self.cond_conv2 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu2 = nn.ReLU()

		# third layer
		self.cond_conv3 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu3 = nn.ReLU()  

		# fourth layer
		self.cond_conv4 = nn.Conv1d(in_channels = 2*nAssets, out_channels = 2*nAssets, kernel_size = 5, stride=2, padding = 1)
		self.relu4 = nn.ReLU()

		# output layer fully connected / dense
		self.cond_dense5 = nn.Linear(in_features = 2*nAssets, out_features = nAssets)
		self.relu5 = nn.ReLU()

		'''
		Simulator
		'''
		# first layer - dense
		self.sim_dense1 = nn.Linear(in_features = 3*nAssets, out_features = f*nAssets)
		self.relu1 = nn.ReLU()

		# second layer
		self.sim_conv2 = nn.ConvTranspose1d(in_channels = 4*nAssets, out_channels = 2*nAssets, kernel_size = 4, stride=2, padding = 1)
		self.relu2 = nn.ReLU()

		# third layer
		self.sim_conv3 = nn.ConvTranspose1d(in_channels = 2*nAssets, out_channels = nAssets, kernel_size = 4, stride=2, padding = 1)
		self.out_Tanh = nn.Tanh()  

	def forward(self, x):
		'''
		conditioning forward pass
		'''
		# Layer 1
		out = self.relu1(self.cond_conv1(x))

		# Layer 2:
		out = self.relu2(self.cond_conv2(out))

		# Layer 3:
		out = self.relu3(self.cond_conv3(out))

		# Layer 4:
		out = self.relu4(self.cond_conv4(out))

		# Final layer
		out = torch.flatten(out, start_dim = 1, end_dim =2)
		cond_out = self.relu5(self.cond_dense5(out))

		'''
		simulator forward pass
		'''
		# sample a prior latent vector from a normal distribution
		lambd_prior = torch.from_numpy(np.array([np.random.normal(size = 2*self.nAssets) for train_example in range(cond_out.shape[0])])).float().to(device)
		#print("prior shape: " + str(lambd_prior.shape))
		# concat cond_out with the latent vector sampled from the prior
		sim_input = torch.cat((cond_out, lambd_prior), dim = 1)
		#print("concat shape: " + str(sim_input.shape))

		# First layer
		#print("first") 
		out = self.relu1(self.sim_dense1(sim_input))

		# Layer 2:
		# unflatten the out tensor
		out = out.view(out.shape[0], 4*self.nAssets, -1) #https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
		#print("unflatten shape: " + str(out.shape))
		#print("second_sim")
		out = self.relu2(self.sim_conv2(out))
		#print(out.shape)

		# Layer 3:
		#print("third_sim")
		out = self.out_Tanh(self.sim_conv3(out))
		#print(out.shape)

		return out

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
	#print("gradpnlty")
  """Calculates the gradient penalty loss for WGAN GP"""
  # Random weight term for interpolation between real and fake samples
  alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
  # Get random interpolation between real and fake samples
  interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
  d_interpolates = D(interpolates)
  fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
  # Get gradient w.r.t. interpolates
  gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
  return gradient_penalty
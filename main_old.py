# utilities
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
'''
# pytorch examples #
https://github.com/pytorch/examples/blob/master/dcgan/main.py
https://louisenaud.github.io/time_series_prediction.html
https://github.com/GitiHubi/deepAI/blob/master/GTC_2018_Lab.ipynb 
https://colab.research.google.com/github/GitiHubi/deepAD/blob/master/KDD_2019_Lab.ipynb--> [Schreyer et al., 2019]

Wasserstein GANs with Gradient Penalty: https://github.com/eriklindernoren/PyTorch-GAN
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

How to make GANs work?
https://github.com/soumith/ganhacks
https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2
'''

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
	def __init__(self, nAssets):
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
		self.relu3 = nn.ReLU()  

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
		lambd_prior = torch.from_numpy(np.array([np.random.normal(size = 2*nAssets) for train_example in range(cond_out.shape[0])])).float()
		#print("prior shape: " + str(lambd_prior.shape))
		# concat cond_out with the latent vector sampled from the prior
		sim_input = torch.cat((cond_out, lambd_prior), dim = 1)
		#print("concat shape: " + str(sim_input.shape))

		# First layer
		#print("first") 
		out = self.relu1(self.sim_dense1(sim_input))

		# Layer 2:
		# unflatten the out tensor
		out = out.view(out.shape[0], 4*nAssets, -1) #https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
		#print("unflatten shape: " + str(out.shape))
		#print("second_sim")
		out = self.relu2(self.sim_conv2(out))
		#print(out.shape)

		# Layer 3:
		#print("third_sim")
		out = self.relu3(self.sim_conv3(out))
		#print(out.shape)

		return out

Tensor = torch.FloatTensor

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

######################################## Main Script ################################################

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# get the dataset
ibovDB = pd.read_excel("data/IBOV_DB_useThis.xlsx") 
ibovDB = ibovDB.reindex(index=ibovDB.index[::-1]) # reverse the data set rows order
ibovDB = ibovDB.drop(ibovDB.columns[0], axis = 1)
numpy_data = ibovDB.to_numpy()
# get returns and the number of assets in the universe
priceDB = ibovDB.drop(ibovDB.columns[0], axis = 1)
returnDB = priceDB.pct_change()[1:]
nAssets = returnDB.shape[1]
# get asset symbols
symbols = ibovDB.columns[1:].to_list()
# get dates
dates = numpy_data[:,0]

# analysis window
w = 60
b = 40
f = w - b

# initialize a generator instance
generator = Generator(nAssets)

# initialize a generator instance
discriminator = Discriminator(nAssets)
n_critic = 5 # number of critic training steps

# Adam optimizers parameters
learning_rate = 2*(10**-5)
beta1 = 0.5
n_epochs = 500 #15000

# loss function parameters
lambda_gp = 10

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0
USE_CUDA = False

generator_optimizer = torch.optim.Adam(generator.parameters(), betas = (beta1, 0.999), lr=learning_rate)
discriminator_optimizer =  torch.optim.Adam(discriminator.parameters(), betas = (beta1, 0.999), lr=learning_rate)

[allMb, allMf] = get_dataSet(TSData = returnDB, b = b, f = f, step = 1)
torch_Mb = torch.from_numpy(np.array(allMb)).float()
torch_Mf = torch.from_numpy(np.array(allMf)).float()
M_raw = torch.cat((torch_Mb, torch_Mf), dim = 2)
M_train = M_raw[0:M_raw.shape[0]-120,:,:]
#print("Mb:" + str(torch_Mb.shape))
#print("Mf:" + str(torch_Mf.shape))
#print("M_:" + str(M_train.shape))
dataloader = DataLoader(M_train, batch_size = 100, shuffle=False, num_workers=ngpu)

# start timer
start_time = datetime.now()
criticBatchTime = start_time 
epoch_checkpoint = 10
epoch_checkpoint_delta = 10

# train GAN model
filepath = "models/final/"
for epoch in range(n_epochs):
	for i, M_real in enumerate(dataloader):
		currBatch = i + 1 # current batch idx
		Mb = M_real[:, :, 0 : b]

		# ---------------------
		#  Train Discriminator
		# ---------------------

		# reset gradients
		discriminator_optimizer.zero_grad()

		# run a simulation (generator forward pass)
		Mf_fake = generator(Mb) # encode mini-batch data
		M_fake = torch.cat((Mb, Mf_fake), dim = 2)
		#print("Mb:" + str(Mb.shape))
		#print("Mfake:" + str(Mf_fake.shape))
		#print("M_fake:" + str(M_fake.shape))

		# fake market scenario
		fake_validity = discriminator(M_fake)
	    # real market scenario
		real_validity = discriminator(M_real)
		# Gradient penalty
		gradient_penalty = compute_gradient_penalty(discriminator, M_real, M_fake)
	    # Adversarial loss
		d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

		d_loss.backward()
		discriminator_optimizer.step()

		generator_optimizer.zero_grad()

	    # Train the generator every n_critic steps
		if currBatch % n_critic == 0:
			# Generate a simulation
			Mf_fake = generator(Mb) # encode mini-batch data
			M_fake = torch.cat((Mb, Mf_fake), dim = 2)		
	    	# Loss measures generator's ability to fool the discriminator
	    	# Train on fake images
			fake_validity = discriminator(M_fake)
			g_loss = -torch.mean(fake_validity)

			g_loss.backward()
			generator_optimizer.step()

			now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
			total_time = datetime.now() - start_time
			end_time = datetime.now() - criticBatchTime
			print(
	           		"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Elapsed time: %fs] [Total Elapsed time: %fs]"
	                % (epoch, n_epochs, currBatch, len(dataloader), d_loss.item(), g_loss.item(), end_time.seconds, total_time.seconds)
	            )
			# restart timer
			criticBatchTime = datetime.now()

	if epoch + 1 == epoch_checkpoint:
		epoch_checkpoint += epoch_checkpoint_delta # increase checkpoint
		filename = filepath + "GAN_state_" + str(epoch+1) + ".pth"
		torch.save({
            'epoch': epoch + 1,
            'G_state_dict': generator.state_dict(),
            'D_state_dict': discriminator.state_dict(),
            'G_optimizer_state_dict': generator_optimizer.state_dict(),
            'D_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'G_loss': g_loss,
            'D_loss': d_loss,
            }, filename)



# =================== (3) backward pass ==================================

'''
# run backward pass
reconstruction_loss.backward()

# =================== (4) update model parameters ========================

# update network parameters
conditioning_optimizer.step()
#simulator_optimizer.step()
#discriminator_optimizer.step()

# =================== monitor training progress ==========================

# print training progress each 1'000 mini-batches
#if mini_batch_count % 1000 == 0:
    
# print the training mode: either on GPU or CPU
mode = 'GPU' if (torch.backends.cudnn.version() != None) and (USE_CUDA == True) else 'CPU'

# print mini batch reconstuction results
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
end_time = datetime.now() - start_time
#print('[LOG {}] training status, epoch: [{:04}/{:04}], batch: {:04}, loss: {}, mode: {}, time required: {}'.format(now, (epoch+1), num_epochs, mini_batch_count, np.round(reconstruction_loss.item(), 4), mode, end_time))
print('[LOG {}] training status, epoch: [{:04}/{:04}], loss: {}, mode: {}, time required: {}'.format(now, (epoch+1), num_epochs, np.round(reconstruction_loss.item(), 4), mode, end_time))

# reset timer
start_time = datetime.now()

# =================== evaluate model performance =============================

# set networks in evaluation mode (don't apply dropout)
conditioning_train.cpu().eval()
#simulator_train.cpu().eval()

# simulate the market for f days
#reconstruction = decoder_train(encoder_train(data))
simulation = simulator_train(conditioning_train(data))

# determine reconstruction loss - all transactions
reconstruction_loss_all = loss_function(reconstruction, data)
        
# collect reconstruction loss
losses.extend([reconstruction_loss_all.item()])

# print reconstuction loss results
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] training status, epoch: [{:04}/{:04}], loss: {:.10f}'.format(now, (epoch+1), num_epochs, reconstruction_loss_all.item()))

# =================== save model snapshot to disk ============================

# save trained encoder model file to disk
encoder_model_name = "ep_{}_encoder_model.pth".format((epoch+1))
torch.save(encoder_train.state_dict(), os.path.join("./models", encoder_model_name))

# save trained decoder model file to disk
decoder_model_name = "ep_{}_decoder_model.pth".format((epoch+1))
torch.save(decoder_train.state_dict(), os.path.join("./models", decoder_model_name))
'''


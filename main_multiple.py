# libraries
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
from utils import *
import torch.autograd as autograd

import os

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

######################################## Main Script ################################################

# Set random seed for reproducibility
manualSeed = 8899
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
#returnDB = priceDB
nAssets = returnDB.shape[1]
# get asset symbols
symbols = ibovDB.columns[1:].to_list()
# get dates
dates = numpy_data[:,0]

# analysis window
w = 60
b = 40
f = w - b

[allMb, allMf] = get_dataSet(TSData = returnDB, b = b, f = f, step = 1)
torch_Mb = torch.from_numpy(np.array(allMb)).float().to(device)
torch_Mf = torch.from_numpy(np.array(allMf)).float().to(device)
M_raw = torch.cat((torch_Mb, torch_Mf), dim = 2)
M_train = M_raw[0:M_raw.shape[0]-120,:,:]
#print("Mb:" + str(torch_Mb.shape))
#print("Mf:" + str(torch_Mf.shape))
#print("M_:" + str(M_train.shape))
ngpu = 0
dataloader = DataLoader(M_train, batch_size = 128, shuffle=False, num_workers=ngpu)

n_critic = 5 # number of critic training steps

# Adam optimizers parameters
learning_rate = 2*(10**-5)
beta1 = 0.5
n_epochs = 1000 #15000

# loss function parameters
lambda_gp = 10

# train multiple GAN models
for run in range(30):
  epoch_checkpoint = 20
  epoch_checkpoint_delta = 20
  # initialize a generator instance
  generator = Generator(nAssets,f)
  # initialize a discriminator instance
  discriminator = Discriminator(nAssets)

  if cuda:
    generator.cuda()
    discriminator.cuda()

  generator_optimizer = torch.optim.Adam(generator.parameters(), betas = (beta1, 0.999), lr=learning_rate)
  discriminator_optimizer =  torch.optim.Adam(discriminator.parameters(), betas = (beta1, 0.999), lr=learning_rate)

  print('training model ' + str(run + 1))
  filepath = "models/500_epochs/run_" + str(run+1) + '/'
  if not os.path.exists(filepath):
    os.makedirs(filepath)

  # start timer
  start_time = datetime.now()
  criticBatchTime = start_time 
  total_time = 0

  # start the training process
  for epoch in range(n_epochs):
    for i, M_real in enumerate(dataloader):
      currBatch = i + 1 # current batch idx
      Mb = M_real[:, :, 0 : b]
      Mf_real = M_real[:, :, b : ]

  		# ---------------------
  		#  Train Discriminator
  		# ---------------------

  		# reset gradients
      discriminator_optimizer.zero_grad()

  		# run a simulation (generator forward pass)
      Mf_fake = generator(Mb) # encode mini-batch data
      M_fake = torch.cat((Mb, Mf_fake), dim = 2) # concatenate Mb and Mf to form the fake data set
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

  	  # Train the generator every n_critic steps
      if currBatch % n_critic == 0:
        generator_optimizer.zero_grad()
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
      
    if epoch+1 == epoch_checkpoint:
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
  
  filehandle = open(filepath + 'training_time_seconds.txt', 'w') 
  filehandle.write(str(total_time.seconds))
  filehandle.close()
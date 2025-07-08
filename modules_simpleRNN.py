#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Aug  5 16:36:07 2022

@author: mbeiran
"""
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
from math import sqrt
import random
# %%
# =============================================================================
#   Define the RNN class in pytorch
# =============================================================================
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wi_init, wo_init, wrec_init, h0_init = None,
                 train_conn = True, train_wout = False, train_wi=False, noise_std=0.0005, alpha=0.1):
        
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.train_conn = train_conn
        self.non_linearity = torch.tanh #this is the non-linearity, you could use another one
        self.alpha = alpha
        
        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad= False
        else:
            self.wi.requites_grad = True
        
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_conn:
            self.wrec.requires_grad= False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        self.h0.requires_grad = False
        
        self.wout = nn.Parameter(torch.Tensor( hidden_size, output_size))
        if train_wout:
            self.wout.requires_grad= True
        else:
            self.wout.requires_grad= False
        
        # Initialize parameters
        with torch.no_grad():
            self.wi.copy_(wi_init)
            self.wrec.copy_(wrec_init)
            self.wout.copy_(wo_init)
            if h0_init is None:
                self.h0.zero_
                self.h0.fill_(0)
            else:
                self.h0.copy_(h0_init)
            
    def forward(self, input): #Here you define the dynamics
        batch_size = input.shape[0] 
        seq_len = input.shape[1]
        h = self.h0
        r = self.non_linearity(self.h0)
        
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output[:,0,:] = r.matmul(self.wout)
        # simulation loop
        for i in range(seq_len-1):
            h = h + self.noise_std*noise[:,i,:]+self.alpha * (-h + r.matmul(self.wrec.t())+ input[:,i,:].matmul(self.wi))
            r = self.non_linearity(h)
            output[:,i+1,:] = r.matmul(self.wout)
        return output
    
    
def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    
    # Compute loss for each (trial, timestep) (average accross output dimensions)    
    
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


# %%
# =============================================================================
# Define training function, and loss calculation, etc
# =============================================================================
def net_loss(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, cuda=False):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    
    # CUDA management
    if cuda: #this is to use GPUs if available
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
    return(initial_loss.item())

def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          clip_gradient=None, cuda=False, save_loss=False, save_params=True, verbose=True, adam=True):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :return: nothing
    """
    print("Training...")
    if adam: #this is the type of backprop implementation you want to use.
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)#
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    # graD = np.zeros((hidden_size, n_epochs))
    hidden_size = net.hidden_size
    wr = np.zeros((hidden_size, hidden_size, n_epochs))
    if plot_gradient:
        gradient_norms = []
    
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        # if keep_best:
        #     best = net.clone()
        #     best_loss = initial_loss.item()

    for epoch in range(n_epochs):
        begin = time.time()
        losses = []

        #for i in range(num_examples // batch_size):
        optimizer.zero_grad()
        
        random_batch_idx = random.sample(range(num_examples), batch_size)
        #random_batch_idx = random.sample(range(num_examples), num_examples)
        batch = input[random_batch_idx]
        output = net(batch)
        # if epoch==0:
        #     output0 = output
        loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
        
        losses.append(loss.item())
        all_losses.append(loss.item())
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        if plot_gradient:
            tot = 0
            for param in [p for p in net.parameters() if p.requires_grad]:
                tot += (param.grad ** 2).sum()
            gradient_norms.append(sqrt(tot))
        #This is for debugging
        # for param in [p for p in net.parameters() if p.requires_grad]:
        #     graD[:,epoch] = param.grad.detach().numpy()[:,0]
        optimizer.step()
        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()

        if np.mod(epoch, 10)==0 and verbose is True:
            # if keep_best and np.mean(losses) < best_loss:
            #     best = net.clone()
            #     best_loss = np.mean(losses)
            #     print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
            # else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if torch.cuda.is_available():
            wr[:,:,epoch] = net.cpu().wrec.detach().numpy()      
            net.to(device=device)
        else:
            wr[:,:,epoch] = net.wrec.detach().numpy()
        
            
    if plot_learning_curve:
        plt.figure()
        plt.plot(all_losses)
        plt.yscale('log')
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.figure()
        plt.plot(gradient_norms)
        plt.yscale('log')
        plt.title("Gradient norm")
        plt.show()

    return(all_losses, wr[:,:,-1])

# %%
def set_plot():
    plt.style.use('ggplot')
    
    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
     
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markeredgewidth'] = 0.003
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['font.size'] = 14#9
    plt.rcParams['legend.fontsize'] = 11#7.
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'
    
    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 14#9
    plt.rcParams['xtick.labelsize'] = 11#7
    plt.rcParams['ytick.labelsize'] = 11#7
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams["axes.grid"] = False
    return()

def remove_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return()

def plotFn(DWsaved, wrec_pre, doWhat=None):     
    #doWhat = 'stim' # 'nonstim' 'all'

    # choose which to do
    preallmean = np.mean(np.absolute(wrec_pre))
    if doWhat == 'stim':
        DW = DWsaved[:,:60]
    elif doWhat == 'all':
        DW = DWsaved
    elif doWhat == 'nostim':
        DW = DWsaved[:,60:]
    DW_norm = DW / preallmean * 100
    print(f'doing: {doWhat}')

    thresh = 0.1
    count = np.sum(np.abs(DW) > thresh)
    print (f'Abs weight changes > {thresh}: {count}, percent {count/DW.size*100:.3g}')

    thresh = 100 # percent
    count = np.sum(np.abs(DW_norm) > thresh)
    print (f'Norm weight changes > {thresh}: {count}, percent {count/DW.size*100:.3g}')

    thresh = 200 # percent
    ctPos = np.sum(DW_norm > thresh)
    ctNeg = np.sum(DW_norm < -thresh)
    print (f'Norm weight changes > {thresh}: {ctPos}, percent {ctPos/DW.size*100:.3g}')
    print (f'Norm weight changes < {thresh}: {ctNeg}, percent {ctNeg/DW.size*100:.3g}')

    plt.imshow(DW_norm)

    #Get mean, median and STDev of the normalized matrix
    mean_DW_norm = np.mean(DW_norm)
    print (f'DW mean = {mean_DW_norm:.3g}, std {np.std(DW_norm):.3g}')

    median_DW_norm = np.median(DW_norm)
    print ('DW median = ', median_DW_norm)

    #%% Normalized synaptic shift CDF
    xl0 = np.r_[-1,1]*3*100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(xl0[0], xl0[1], 1000).tolist()
    plt.hist(np.ravel(DW_norm), bins = bins, color='k', lw=1, cumulative=True, density=True, histtype='step', label='$\Delta W$')
    plt.xlabel('synaptic shift')
    plt.ylabel('Density')
    ax.set_xlim(xl0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.vlines(ymin = 0, ymax = 1, x=0, color='k', ls = '--')
    plt.title(r'$\mu=$'+str(np.mean(DW_norm))[0:5]+', $\sigma=$'+str(np.std(DW_norm))[0:5], fontsize=10)
    plt.savefig(f'cdf_norm_shift-{doWhat}.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(xl0[0], xl0[1], 200).tolist()
    plt.hist(np.ravel(DW_norm), bins = bins, alpha=0.5, label='$\Delta W$')
    plt.xlabel('norm. synaptic shift (%)')
    plt.ylabel('count')
    ax.set_xlim(xl0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.title(r'$\mu=$'+str(np.mean(DW_norm))[0:5]+', $\sigma=$'+str(np.std(DW_norm))[0:5], fontsize=10)

    plt.show()
    return()

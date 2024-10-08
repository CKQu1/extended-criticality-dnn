import torch
import os
import numpy as np
import scipy.io as sio
import torch.nn as nn
#import model_loader2 as ml

import train_DNN_code.model_loader as model_loader
from train_DNN_code.dataloader import get_data_loaders, get_synthetic_gaussian_data_loaders

# get network type from folder name, only works for fcN_tanh_mnist
def get_nettype(folder_name):
    return folder_name.split("_")[0] + "_mnist_tanh"

# extract weight parameters from epoch
def get_epoch_weights(path, folder_name, epoch):

    epoch_file = "model_" + str(epoch) + "_sub_loss_w.mat"
    epoch_weights = sio.loadmat(path + '/' + folder_name + '/' + epoch_file)

    return epoch_weights['sub_weights'][0]

def layer_names(net):
    return list(net.state_dict().keys())

# load weights into network
def load_weights(net, weights):

    weight_index_ls = layer_names(net)    
    for i in range(len(weight_index_ls)):
        layer_id = weight_index_ls[i]
        net.state_dict()[layer_id].data.copy_(torch.from_numpy(weights[i]))

    return net

# returns layer structures of network
def layer_struct(net):
    return list(net.modules())[1]

# returns all the preactivation input of the layers including the input and output (better method in Alexnet.py under class FullyConnected_tanh)
def get_hidden_layers(net, weights, x):

    wm_num = len(weights)
    net_weights = list(net.state_dict().values())
    correct = 0
    for i in range(wm_num):
        if weights[i].shape == net_weights[i].shape:
            correct += 1

    if correct != wm_num:
        return f"Correct number of weights is {wm_num}, but you have {correct}."
    else:
        net_updated = load_weights(net, weights)

        layer_outputs = []
        # include the input
        layer_outputs.append(x)
        
        layer_ls = layer_struct(net_updated)
        x = layer_ls[0](x)
        layer_outputs.append(x)

        for j in range(1,len(layer_ls)):
            layer_func = layer_ls[j]
            # might need to replace the below

            x = layer_func(x)

            if j % 2 == 0:
                layer_outputs.append(x)

    return layer_outputs

# example
"""
folder_name = "fc10_mnist_tanh_id_stable1.2_1.0_epoch650_algosgd_lr=0.001_bs=256_data_mnist_fgsm_ngpu=2"
# needs to be changed accordingly
path += "/trained_nets"
w = get_epoch_weights(path, folder_name, 650)
net = model_loader.load('fc10_mnist_tanh')
x = torch.ones(1,784)
hidden = get_hidden_layers(net, w, x)
"""



import argparse
import numpy as np
import os
from scipy.stats import ortho_group # for creating random orthonormal basis
import sys

import torch
import torchvision
import torchvision.transforms as transforms

from typing import Tuple, List
from torchvision.transforms import functional as F

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from train_DNN_code.dataloader import get_data_loaders

# 1. MNIST
def load_mnist():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    args = parser.parse_args()

    args.batch_size = 1
    args.raw_data = False
    args.ngpu = 1
    args.noaug = False
    args.dataset = 'mnist'
    args.trainloader = ''
    args.testloader = ''

    _ , _, trainset = get_data_loaders(args)
    input_all = trainset.data.numpy()

    labels = [trainset[_][1] for _ in range(len(trainset))]

    return input_all.reshape((60000,784)), labels

# return indices of class in trainset
def mnist_class(y,labels):

    assert y in list(range(10)), "Label must be between 0 to 9."

    return [i for i,x in enumerate(labels) if x == y]

# get normalized cifar10 or mnist
def get_data_normalized(dataset,bs):

    #bs = len(trainset)

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    # no data agumentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    #kwargs = {'num_workers': 1, 'pin_memory': True} if args.ngpu else {}
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transform_test)
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

    if bs == None:

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                  shuffle=False, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                  shuffle=False , **kwargs)
    else:
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=False, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                  shuffle=False , **kwargs)
        

    return trainloader, testloader, trainset

# 2. Great circle
def gcircle(N_0,q,angles):

    # same as input size
    orth_basis = ortho_group.rvs(dim=N_0)
    u0 = orth_basis[0]
    u1 = orth_basis[1]

    gcirc = np.sqrt(N_0*q)*(np.cos(angles[0])*u0 + np.sin(angles[0])*u1)
    for ii in range(1,len(angles)):
        angle = angles[ii]
        layer_temp = np.sqrt(N_0*q)*(np.cos(angle)*u0 + np.sin(angle)*u1)
        gcirc = np.vstack((gcirc, layer_temp))

    return gcirc



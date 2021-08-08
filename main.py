from __future__ import print_function
import torch
import os
import random
import numpy as np
import argparse
import scipy.io as sio
import math
import time

import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel

import model_loader as ml

import get_gradient_weight.net_plotter as net_plotter
from get_gradient_weight.gradient_noise import get_grads

import train_DNN_code
import train_DNN_code.model_loader as model_loader
from train_DNN_code.dataloader import get_data_loaders, get_synthetic_gaussian_data_loaders

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from scipy.stats import levy_stable

def init_params(net, w_std):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            #init.normal_(m.weight, std=1e-3) # this can be changed
            init.normal_(m.weight, std=w_std) # this can be changed
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
def init_stable_params(net, alpha, scale_multiplier):    
    #if isinstance(net,train_DNN_code.models.Alexnet.FullyConnected):
    weight_index_ls = list(net.state_dict().keys())
    for i in range(len(weight_index_ls)):
           layer_name = weight_index_ls[i]
           dim = net.state_dict()[layer_name].size()
           #alpha= 1.5
           beta = 0
           loc = 0
           scale = (1/(2*math.sqrt(dim[0]*dim[1])))**(1/alpha) # this is our standard unit of the scale for stable init
           new_weights = levy_stable.rvs(alpha, beta, loc, scale * scale_multiplier, size=(dim[0],dim[1]))
           new_weights = torch.tensor(new_weights)
           net.state_dict()[layer_name].data.copy_(new_weights)   

# Training with save all transient state in one epoch
def train_save(trainloader, net, criterion, optimizer, stable_epoch, stable_inter, wm_total, interepoch_step, use_cuda):
#def train_save(trainloader, net, criterion, optimizer, stable_inter, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    grads = []
    sub_loss = []
    sub_weights = []

    # get the total steps for stable fitting
    if stable_inter == 'y':
        w = net_plotter.get_weights(net)
        wm_total = len(w)              # total number of weight matrices
        
        alphas = np.empty([wm_total,interepoch_step]) # stability 
        betas = np.empty([wm_total,interepoch_step])  # skewness
        deltas = np.empty([wm_total,interepoch_step]) # location
        sigmas = np.empty([wm_total,interepoch_step]) # scale

        # function for fitting
        pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # get gradient
            #grad = get_grads(net).cpu()
            #grads.append(grad)
            optimizer.step()

            # record tiny steps in every epoch
            sub_loss.append(loss.item())

            if stable_inter == 'y':

                w = net_plotter.get_weights(net) # initial parameters
                for j in range(len(w)):
                    w[j] = w[j].cpu().numpy()   
                    # newly added stable fit during training for each step of epoch
                    params = pconv(*levy_stable._fitstart(w[j].flatten()))

                    alphas[j,batch_idx] = params[0]
                    betas[j,batch_idx] = params[1]
                    deltas[j,batch_idx] = params[2]
                    sigmas[j,batch_idx] = params[3]

            #sub_weights.append(w)

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            #outputs = F.softmax(net(inputs))
            outputs = F.softmax(net(inputs),-1)
            loss = criterion(outputs, one_hot_targets)
            loss.backward()

            # get gradient
            #grad = get_grads(net).cpu()
            #grads.append(grad)

            optimizer.step()

            # record tiny steps in every epoch
            sub_loss.append(loss.item())
            ################################################################################################
            #import pdb; pdb.set_trace() # removed this line

            if stable_inter == 'y':

                w = net_plotter.get_weights(net) # initial parameters
                for j in range(len(w)):
                    w[j] = w[j].cpu().numpy()   
                    # newly added stable fit during training for each step of epoch
                    params = pconv(*levy_stable._fitstart(w[j].flatten()))

                    alphas[j,batch_idx] = params[0]
                    betas[j,batch_idx] = params[1]
                    deltas[j,batch_idx] = params[2]
                    sigmas[j,batch_idx] = params[3]

            #sub_weights.append(w)

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()
    
    # append the gradient and sub_weights at the end of the epoch (changes made) (removed as well to save space)
    #grad = get_grads(net).cpu()
    #grads.append(grad)
    
    if stable_epoch == 'y' and stable_inter == 'n':
        
        alphas = np.empty([wm_total,1]) # stability 
        betas = np.empty([wm_total,1])  # skewness
        deltas = np.empty([wm_total,1]) # location
        sigmas = np.empty([wm_total,1]) # scale

        # function for fitting
        pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
    
    w = net_plotter.get_weights(net)
    
    for j in range(len(w)):
        w[j] = w[j].cpu().numpy()
        
        params = pconv(*levy_stable._fitstart(w[j].flatten()))        
        alphas[j,0] = params[0]
        betas[j,0] = params[1]
        deltas[j,0] = params[2]
        sigmas[j,0] = params[3]
        
    sub_weights.append(w)  

    #M = len(grads[0]) # total number of parameters
    #grads = torch.cat(grads).view(-1, M)
    #mean_grad = grads.sum(0) / (batch_idx + 1) # divided by # batchs
    #noise_norm = (grads - mean_grad).norm(dim=1)
  
    #return train_loss/total, 100 - 100.*correct/total, sub_weights, sub_loss, grads, mean_grad, noise_norm
    return train_loss/total, 100 - 100.*correct/total, sub_weights, sub_loss, np.r_[alphas,betas,deltas,sigmas]


def test_save(testloader, net, criterion, use_cuda):
#def test_save(testloader, net, criterion, stable_inter, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    sub_loss = []

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            sub_loss.append(loss.item())
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            #outputs = F.softmax(net(inputs))
            outputs = F.softmax(net(inputs),-1)
            loss = criterion(outputs, one_hot_targets)
            sub_loss.append(loss.item())
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total, sub_loss

# Training without save all transient state in one epoch
def train(trainloader, net, criterion, optimizer, stable_epoch, stable_inter, wm_total, interepoch_step, use_cuda):
#def train(trainloader, net, criterion, optimizer, stable_inter, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    #grads = []    

    # get the total steps for stable fitting
    if stable_inter == 'y':
        
        alphas = np.empty([wm_total,interepoch_step]) # stability 
        betas = np.empty([wm_total,interepoch_step])  # skewness
        deltas = np.empty([wm_total,interepoch_step]) # location
        sigmas = np.empty([wm_total,interepoch_step]) # scale

        # function for fitting
        pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # get gradient (removed)
            #grad = get_grads(net).cpu()
            #grads.append(grad)
            optimizer.step()

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

            if stable_inter == 'y':

                w = net_plotter.get_weights(net) # initial parameters
                for j in range(len(w)):
                    w[j] = w[j].cpu().numpy()   
                    # newly added stable fit during training for each step of epoch
                    params = pconv(*levy_stable._fitstart(w[j].flatten()))

                    alphas[j,batch_idx] = params[0]
                    betas[j,batch_idx] = params[1]
                    deltas[j,batch_idx] = params[2]
                    sigmas[j,batch_idx] = params[3]

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            #outputs = F.softmax(net(inputs))
            outputs = F.softmax(net(inputs),-1)
            loss = criterion(outputs, one_hot_targets)
            loss.backward()

            # get gradient (removed)
            #grad = get_grads(net).cpu()
            #grads.append(grad)

            optimizer.step()            

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

            if stable_inter == 'y':

                w = net_plotter.get_weights(net) # initial parameters
                for j in range(len(w)):
                    w[j] = w[j].cpu().numpy()   
                    # newly added stable fit during training for each step of epoch
                    params = pconv(*levy_stable._fitstart(w[j].flatten()))

                    alphas[j,batch_idx] = params[0]
                    betas[j,batch_idx] = params[1]
                    deltas[j,batch_idx] = params[2]
                    sigmas[j,batch_idx] = params[3]

    #M = len(grads[0]) # total number of parameters
    #grads = torch.cat(grads).view(-1, M)
    #mean_grad = grads.sum(0) / (batch_idx + 1) # divided by # batchs
    #noise_norm = (grads - mean_grad).norm(dim=1)
    
    if stable_epoch == 'y' and stable_inter == 'n':
    
        w = net_plotter.get_weights(net)
        
        alphas = np.empty([wm_total,1]) # stability 
        betas = np.empty([wm_total,1])  # skewness
        deltas = np.empty([wm_total,1]) # location
        sigmas = np.empty([wm_total,1]) # scale

        # function for fitting
        pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
        
    for j in range(len(w)):
        w[j] = w[j].cpu().numpy()
        
        params = pconv(*levy_stable._fitstart(w[j].flatten()))        
        alphas[j,0] = params[0]
        betas[j,0] = params[1]
        deltas[j,0] = params[2]
        sigmas[j,0] = params[3]
  
    #return train_loss/total, 100 - 100.*correct/total, grads, mean_grad, noise_norm
    return train_loss/total, 100 - 100.*correct/total, np.r_[alphas,betas,deltas,sigmas]


def test(testloader, net, criterion, use_cuda):
#def test(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            #outputs = F.softmax(net(inputs))
            outputs = F.softmax(net(inputs),-1)
            loss = criterion(outputs, one_hot_targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total

def name_save_folder(args):

    if args.init_dist == 'gaussian':   
        save_folder = args.model + f"_{args.w_std}" + '_epoch' + str(args.epochs) + '_lr=' + str(args.lr)
    #elif args.init_dist == 'gaussianchaos':
    #    save_folder = args.model + '_id_' + args.init_dist + '_epoch' + str(args.epochs) + '_algo' + args.optimizer + '_lr=' + str(args.lr)                  
    elif args.init_dist == 'stable':
        save_folder = args.model + '_id_' + args.init_dist + f"{args.init_alpha}" + f"_{args.init_scale_multiplier}" + '_epoch' + str(args.epochs) + '_algo' + args.optimizer + '_lr=' + str(args.lr)  
        
    # added learning rate decay back
    if args.lr_decay != 1:
        save_folder += '_lr_decay=' + str(args.lr_decay)

    save_folder += '_bs=' + str(args.batch_size) + '_data_' + str(args.dataset)
    
    if args.loss_name != 'crossentropy':
        save_folder += '_loss=' + str(args.loss_name)
    if args.noaug:
        save_folder += '_noaug'
    if args.raw_data:
        save_folder += '_rawdata'
    if args.label_corrupt_prob > 0:
        save_folder += '_randlabel=' + str(args.label_corrupt_prob)
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)

    return save_folder

if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--init_dist', default='gaussian', type=str, help='gaussian | stable')
    parser.add_argument('--dataset', default='cifar10', type=str, help='mnist | cifar10 | gauss')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--weight_decay', default=0, type=float)#0.0005
    parser.add_argument('--momentum', default=0, type=float)#0.9
    parser.add_argument('--epochs', default=5000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=1, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # weight initilization option (new)
    # stable
    parser.add_argument('--init_alpha', default=1.5, type=float)  
    parser.add_argument('--init_scale_multiplier', default=1, type=float)             
    # Gaussian
    parser.add_argument('--w_std', default=1e-3, type=float)

    # stable fitting options (new)
    parser.add_argument('--stable_epoch', default='y', type=str, help='stable fitting for epochs')
    parser.add_argument('--stable_inter', default='n', type=str, help='stable fitting for steps')

    # model parameters
    parser.add_argument('--model', '-m', default='resnet20')#vgg9
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')

    #parameters for gaussian data
    parser.add_argument('--gauss_scale', default=10.0, type=float) 

    args = parser.parse_args()
    args.batch_size = int(args.batch_size)
    
    t0 = time.time()

    print('\nLearning Rate: %f' % args.lr)
    print('\nDecay Rate: %f' % args.lr_decay)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Current devices: ' + str(torch.cuda.current_device()))
        print('Device count: ' + str(torch.cuda.device_count()))

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    lr = args.lr  # current learning rate
    lr_decay = args.lr_decay  # current learning decay rate
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    save_folder = name_save_folder(args)
    if not os.path.exists('trained_nets/' + save_folder):
        os.makedirs('trained_nets/' + save_folder)

    f = open('trained_nets/' + save_folder + '/log.out', 'a')

    if args.dataset == 'gauss':
        trainloader, testloader = get_synthetic_gaussian_data_loaders(args)
    else:
        trainloader, testloader, _ = get_data_loaders(args)
        

    if args.label_corrupt_prob and not args.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')
        
    # no. of steps in each epoch
    interepoch_step = len(trainloader)

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args.model)
        print(net)
        if args.init_dist == 'stable':
            # and isinstance(net,train_DNN_code.models.Alexnet.FullyConnected)
            init_stable_params(net, args.init_alpha, args.init_scale_multiplier)

            print(f"FC stable init Success: alpha = {args.init_alpha}, scale_multiplier = {args.init_scale_multiplier}!")
        elif args.init_dist == 'gaussian':
            init_params(net, args.w_std)

            print(f"FC Gaussian init Success: std is {args.w_std}!")
        #elif args.init_dist == 'gaussianchaos':
        #    init_gaussianchaos_params(net)
        #    print("FC Gaussian chaos init Success!")
            
    # getting the initial conditions of the NN before training
    initial_weights = []
    w = net_plotter.get_weights(net) # initial parameters

    wm_total = len(w)                # number of weight matrices in the network
    w_dim = []                       # number of weights in each matrix

    for j in range(len(w)):
        w[j] = w[j].cpu().numpy()
        w_dim.append(len(w[j].flatten()))
    initial_weights.append(w)
    # saving them as model_0
    sio.savemat('trained_nets/' + save_folder + '/model_0' + '_sub_loss_w.mat',
                                mdict={'sub_weights': initial_weights},
                                )   

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_opt:
        checkpoint_opt = torch.load(args.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])

    # record the performance of initial model
    if not args.resume_model:
        train_loss, train_err = test(trainloader, net, criterion, use_cuda)
        test_loss, test_err = test(testloader, net, criterion, use_cuda)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (0, train_loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        state = {
            'acc': 100 - test_err,
            'epoch': 0,
            'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        #torch.save(state, 'trained_nets/' + save_folder + '/model_0.t7')
        #torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_0.t7')

    # network hyperparameters
    net_params = {}
    net_params['wm_total'] = wm_total
    net_params['w_dim'] = w_dim

    # training hyperparameters
    train_params = {}
    train_params['lr'] = args.lr
    train_params['bs'] = args.batch_size
    train_params['epochs'] = args.epochs
    train_params['save_epoch'] = args.save_epoch
    train_params['loss_name'] = args.loss_name
    train_params['optimizer'] = args.optimizer
    train_params['momentum'] = args.momentum
    train_params['weight_decay'] = args.weight_decay
    train_params['dataset'] = args.dataset

    # network initialization hyperparameters
    net_init_params = {}
    net_init_params['init_dist'] = args.init_dist
    if args.init_dist == 'gaussian':
        net_init_params['w_std'] = args.w_std
    elif args.init_dist == 'stable':
        net_init_params['init_alpha'] = args.init_alpha
        net_init_params['init_scale_multiplier'] = args.init_scale_multiplier
    
    # training logs per iteration
    training_history = []
    testing_history = []

    print(f"Cuda: {use_cuda} \n") ######################################

    for epoch in range(start_epoch, args.epochs + 1):
        print(epoch)      

        #if lr_decay != 1 and (epoch == 150 or epoch == 250 or epoch == 350):
        #    lr *= lr_decay
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] *= args.lr_decay
  
        # Save checkpoint.
        if epoch == 1 or epoch == args.epochs or epoch % args.save_epoch == 0:
            
            # only saving the weights this time
            #loss, train_err, sub_weights, sub_loss, grads, estimated_full_batch_grad, gradient_noise_norm = train_save(trainloader, net, criterion, optimizer, use_cuda)
            # , stable_epoch, stable_inter, wm_total, interepoch_step, use_cuda
            loss, train_err, sub_weights, sub_loss, stable_params_epoch = train_save(trainloader, net, criterion, optimizer, args.stable_epoch, args.stable_inter, wm_total, interepoch_step, use_cuda)
            test_loss, test_err, test_sub_loss = test_save(testloader, net, criterion, use_cuda)
            
            """
            # save loss and weights in each tiny step in every epoch
            sio.savemat('trained_nets/' + save_folder + '/model_' + str(epoch) + '_sub_loss_w.mat',
                                mdict={'sub_weights': sub_weights,'sub_loss': sub_loss, 'test_sub_loss': test_sub_loss,
                                'grads': grads.numpy(), 'estimated_full_batch_grad': estimated_full_batch_grad.numpy(),
                                'gradient_noise_norm': gradient_noise_norm.numpy()},
                                )         
            """
                                
            # save loss and weights in each tiny step in every epoch (no grads)
            sio.savemat('trained_nets/' + save_folder + '/model_' + str(epoch) + '_sub_loss_w.mat',
                                mdict={'sub_weights': sub_weights,'sub_loss': sub_loss, 'test_sub_loss': test_sub_loss},
                                )    
                                   
        else:
            #loss, train_err, grads, estimated_full_batch_grad, gradient_noise_norm = train(trainloader, net, criterion, optimizer, use_cuda)
            loss, train_err, stable_params_epoch = train(trainloader, net, criterion, optimizer, args.stable_epoch, args.stable_inter, wm_total, interepoch_step, use_cuda)
            test_loss, test_err = test(testloader, net, criterion, use_cuda)

        # stable params
        if epoch == 1:
        
            if args.stable_epoch == 'y' and args.stable_inter == 'n':
                stable_params_last = stable_params_epoch
                
        
            elif args.stable_inter == 'y':
                # stable params for all steps
                stable_params = stable_params_epoch
                # stable params for last step of each epoch
                stable_params_last = np.array(stable_params_epoch[:,-1]).T
        else:
            
            if args.stable_epoch == 'y' and args.stable_inter == 'n':
                stable_params_last = np.c_[stable_params_last, stable_params_epoch]
        
            elif args.stable_inter == 'y':
                stable_params = np.c_[ stable_params, stable_params_epoch ]     
                stable_params_last = np.c_[ stable_params_last, np.array(stable_params_epoch[:,-1]).T ]          

        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        # validation acc
        acc = 100 - test_err

        # record training history (starts at initial point)
        training_history.append([loss, 100 - train_err])
        testing_history.append([test_loss, acc])


        # save state for landscape on every epoch
        # state = {
        #     'acc': acc,
        #     'epoch': epoch,
        #     'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict(),
        # }
        # opt_state = {
        #     'optimizer': optimizer.state_dict()
        # }
        # torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
        # torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')

   
    f.close()

    # training hyperparameters
    train_params['interepoch_step'] = interepoch_step

    time_total = time.time() - t0
    train_params['time_total'] = time_total
    
    print(f"Seconds taken: {time_total}")
    
    post_dim = {'epoch_start': 1, 'epoch_last': epoch, 'wm_total': wm_total, 'epoch_total': args.epochs}
    post_dim_inter = {'epoch_start': 1, 'epoch_last': epoch, 'interepoch_step': interepoch_step, 'wm_total': wm_total, 'epoch_total': args.epochs}
    model = save_folder

    # all network parameters
    sio.savemat('/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/trained_nets/' + model + '/net_params_all.mat', 
                        mdict={'net_params':net_params, 'train_params': train_params, 'net_init_params': net_init_params},
                        )  
                        
    # accuracy and loss
    sio.savemat('trained_nets/' + save_folder + '/' + args.model + '_loss_log.mat',
                        mdict={'training_history': training_history,'testing_history': testing_history},
                        )

    if args.stable_inter == 'y' or (args.stable_epoch == 'y' and args.stable_inter =='n'):
        # w_stable_params
        sio.savemat('/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/trained_nets/' + model + '/W_stable_params_' + str(1) + '-' + 
                            str(epoch) + '.mat', 
                            mdict={'alphas':stable_params_last[0:wm_total,:], 'betas': stable_params_last[wm_total:wm_total*2,:], 'sigmas': stable_params_last[wm_total*2:wm_total*3,:], 'deltas': stable_params_last[wm_total*3:wm_total*4,:],
                            'model': model, 'post_dim': post_dim},
                            )  
                            
        print("Stable parameter dimensions: \n")   
        print(stable_params_last.shape) 

                            
    elif args.stable_inter == 'y':     
        # w_stable_params_inter
        sio.savemat('/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/trained_nets/' + model + '/W_stable_params_inter_' + str(1) + '-' + 
                            str(epoch) + '.mat', 
                            mdict={'alphas':stable_params[0:wm_total,:], 'betas': stable_params[wm_total:wm_total*2,:], 'sigmas': stable_params[wm_total*2:wm_total*3,:], 'deltas': stable_params[wm_total*3:wm_total*4,:],
                            'model': model, 'post_dim': post_dim_inter},
                            )
                        
        print("Stable parameter dimensions: \n")
        print(stable_params.shape)      
        print(stable_params_last.shape) 

    #--------------------------------------------------------------------------
    # Load weights and save them in a mat file (temporal unit: epoch)
    #--------------------------------------------------------------------------
    # all_weights = []
    # for i in range(0,args.epochs+1,args.save_epoch):
    #     model_file = 'trained_nets/' + save_folder + '/' + 'model_' + str(i) + '.t7'
    #     net = ml.load('cifar10', args.model, model_file)
    #     w = net_plotter.get_weights(net) # initial parameters
    #     #s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    #     for j in range(len(w)):
    #         w[j] = w[j].cpu().numpy()

    #     all_weights.append(w)

    # sio.savemat('trained_nets/' + save_folder + '/' + args.model + 'all_weights.mat',
    #                         mdict={'weight': all_weights},
    #                         )

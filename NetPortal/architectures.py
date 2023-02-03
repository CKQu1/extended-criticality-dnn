import copy
import math
import numpy as np
import numpy.random as rand
import scipy.io as sio
import torch
import torch.nn.functional as F

from numpy.random import normal
from numpy import linalg as LA

# for CONVNET_1d
from functools import reduce
from operator import __add__

import collections
from itertools import repeat

from os.path import join as pjoin
from scipy.stats import levy_stable
from torch import nn
from torch.nn.modules.utils import _pair

class FullyConnected(nn.Module):

    def __init__(self, dims, alpha, g, init_path, init_epoch, init_type='ht', activation='tanh', with_bias=False, **pretrained):
        super(FullyConnected, self).__init__()
        self.activation = activation
        self.input_dim = dims[0]
        self.dims = dims
        self.depth = len(dims) - 1
        self.num_classes = dims[-1]
        self.alpha, self.g = alpha, g
        self.init_type = init_type

        self.init_supported = ['mfrac_orthogonal', 'mfrac', 'mfrac_sym', 'ht']
        assert init_type in self.init_supported, "Initialization type does not exist!"
        
        modules = []
        for idx in range(len(dims) - 2):
            modules.append(nn.Linear(dims[idx], dims[idx + 1], bias=with_bias))
            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NameError("activation does not exist in NetPortal.architectures")
        modules.append(nn.Linear(dims[-2], dims[-1], bias=with_bias))
    
        # train from pretrained networks
        if init_path != None and init_epoch != None:
        # load initial matrix from trained weights of MLPs (taken from eigs_diffusion.py)
        #def init_mat_load(self, path, init_epoch):
            with torch.no_grad():
                widx = 0
                if 'fcn_grid' in init_path:     # (by main_last_epoch_2.py)
                    print('fcn_grid')
                    w_all = sio.loadmat(f"{init_path}/model_{init_epoch}_sub_loss_w.mat")                
                    for l in modules:
                        if not isinstance(l, nn.Linear): continue
                        l.weight = nn.Parameter(torch.Tensor( w_all['sub_weights'][0][widx].reshape(l.weight.shape) ))

                        widx += 1
                else:                   # (by train_supervised.py)
                    w_all = np.load(f"{init_path}/epoch_{init_epoch}/weights.npy")
                    start = 0
                    for l in modules:
                        if not isinstance(l, nn.Linear): continue
                        w_delta = dims[widx] * dims[widx+1]
                        l.weight = nn.Parameter(torch.Tensor( w_all[start:start+w_delta].reshape(l.weight.shape) ))                                    
            
                        start += dims[widx] * dims[widx+1]
                        widx += 1   

    
        else:
            if alpha != None and g != None:
                with torch.no_grad():
                    for l in modules:
                        if not isinstance(l, nn.Linear): continue
                        size = l.weight.shape
                        if init_type == 'ht':
                            N_eff = (size[0]*size[1])**0.5
                            l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                        scale=g*(0.5/N_eff)**(1./alpha))))

                            # init biases (should be at fixed point?)
                            if with_bias == True:
                                pass

                        elif init_type == 'mfrac_orthogonal':
                            multifractal_orthogonal_(l.weight, 1, alpha, g)

                        elif init_type == 'mfrac':
                            multifractal_(l.weight, 1, alpha, g)

                        elif init_type == 'mfrac_sym':
                            pass

        self.sequential = nn.Sequential(*modules)
             

    """
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), self.input_dim)
        #print(x.shape)         
        if x.shape[-1] == self.input_dim:
            return self.sequential(x)
        else:
            og_shape = x.shape
            x = x.view(-1, self.input_dim)
            return self.sequential(x).view( list(og_shape)[:-2] + [self.input_dim] )
    """

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), self.input_dim)
        #print(x.shape)         
        x = x.view(-1, self.input_dim)
        return self.sequential(x)

    def dm(self, x):
        m = nn.ReLU() if self.activation == 'relu' else (nn.Tanh() if self.activation == 'tanh' else None)
        x = torch.autograd.Variable(x, requires_grad=True)
        y = m(x)
        y.backward( torch.ones_like(x) )

        return x.grad

    # preactivation outputs
    def preact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.sequential[idx](x)
            else:
                hidden[idx] = self.sequential[ell * idx - 1: ell * idx + 1](hidden[idx - 1])

        return hidden + [self.sequential[-2:](hidden[-1])]

    # postactivation outputs
    def postact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.sequential[ell * idx: ell * idx + ell](x)
            else:
                hidden[idx] = self.sequential[ell * idx: ell * idx + ell](hidden[idx - 1])

        return hidden, self.sequential[-1](hidden[-1])
    """
    The problem for this is that the last weight matrix in all these settings are not symmetrical, i.e. for mnist W_L is 784 by 10 (might need to adjust this in the future)
    final argument decides whether it is post-activation or not (pre-activation)
    """
    def layerwise_jacob_ls(self, x, post: bool):    # option to get Jacobian for pre-/post-activation layerwise Jacobian
        # check "The Emergence of Spectral Universality in Deep Networks"
        
        m = nn.ReLU() if self.activation == 'relu' else (nn.Tanh() if self.activation == 'tanh' else None)
        preact_h = self.preact_layer(x)     # do not include the input layer check eq (1) and (2) of "Emergence of Spectral Universality in Deep Networks"     
        # get weights
        weights = [p for p in self.parameters()]        

        if post:
            dphi_h = self.dm(preact_h[0][0])
            DW_l = torch.matmul(torch.diag( dphi_h ), weights[0])
            #DW_l = torch.matmul(torch.diag( m(preact_h[0][0]) ), weights[0])
            DW_ls = [DW_l]

            for i in range(1, len(preact_h)):   # include the last one even if it is not symmetrical
            #for i in range(1, len(preact_h) - 1):
                dphi_h = self.dm(preact_h[i][0])
                if i != len(preact_h) - 1:
                    DW_l = torch.matmul(torch.diag( dphi_h ), weights[i])
                else:
                    DW_l = torch.matmul(torch.diag( torch.ones_like(dphi_h) ), weights[i])

                #DW_l = torch.matmul(torch.diag( m(preact_h[i][0]) ), weights[i])
                DW_ls.append(DW_l)

        else:
            # essentially the first activation function is just the identity map
            DW_ls = [weights[0]]    # it's actually W^{l + 1} D^l

            for i in range(0, len(preact_h) - 1):
                dphi_h = self.dm(preact_h[i][0])
                DW_l = torch.matmul(weights[i + 1], torch.diag( dphi_h ))
                #print(DW_l.shape)
                DW_ls.append(DW_l)
            
        return DW_ls

# method 1
def multifractal_orthogonal_(tensor, gain, alpha, g):
    
    """
    reference: https://pytorch.org/docs/stable/_modules/torch/nn/init.html#orthogonal_
    directly generate a levy matrix and to a QR decomposition
    """
    
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    if tensor.numel() == 0:
        # no-op
        return tensor

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    #a = torch.zeros((dim, dim)).normal_(0, 1)
    a = torch.Tensor(levy_stable.rvs(alpha, 0, size=(rows,cols), scale=g))
    if rows < cols:
        a.t_()

    q, r = torch.qr(a)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)


# method 2
def multifractal_(tensor, gain, alpha, g):
    size = tensor.shape
    a = levy_stable.rvs(alpha, 0, size=size, scale=g)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    a = torch.tensor(1/s*u @ np.diag(s) @ vh)

    with torch.no_grad():
        tensor.view_as(a).copy_(a)
        tensor.mul_(gain)

####################################################################### ALL CNNS #######################################################################


"""
class AlexNet(nn.Module):

    def __init__(self, alpha, g, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        # HT initialization
        self.alpha = alpha
        self.g = g

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
                                           
        self.size = self.get_size()
        print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

        
    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

"""


class AlexNet(nn.Module):

    def __init__(self, alpha, g, dataset, activation, fc_init, dropout: float = 0.5, num_classes=1000, with_bias=False):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        # HT initialization
        self.alpha = alpha
        self.g = g
        self.dataset = dataset
        self.activation = activation
        self.fc_init = fc_init

        #if dataset == "mnist":
        #    N_final = 1024
        #elif dataset == "cifar10":
        #    N_final = 4096
        N_final = 1024

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=with_bias),
            a_func,
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=with_bias),
            a_func,
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=with_bias),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            #nn.MaxPool2d(kernel_size=3, stride=2),

        )

        # for conv2d
        if alpha != None and g != None:
            with torch.no_grad():
                for m in self.features:
                    if isinstance(m, nn.Conv2d):
                        size = m.weight.shape
                        N_eff = int(np.prod(m.kernel_size)) * m.in_channels     # most correct
                        m.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                    size=size)))                                        

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096,bias=with_bias),
            a_func,
            nn.Dropout(p=dropout),
            nn.Linear(4096, N_final, bias=with_bias),
            a_func,
            nn.Linear(N_final, num_classes, bias=with_bias),
        )

        # for linear
        if fc_init == "fc_ht":
            if alpha != None and g != None:
                with torch.no_grad():
                    for l in self.classifier:
                        if isinstance(l, nn.Linear):
                            size = l.weight.shape
                            N_eff = (size[0]*size[1])**0.5
                            l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                        scale=g*(0.5/N_eff)**(1./alpha))))
        
        # orthogonal matrix
        elif fc_init == "fc_orthogonal":
            with torch.no_grad():
                for l in self.classifier:
                    if isinstance(l, nn.Linear):
                        nn.init.orthogonal_(l.weight, gain=1)

        # otherwise it is the default (uniform) option
        
    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetold(nn.Module):

    def __init__(self, alpha, g, activation, fc_init, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000, with_bias=False):
        # ch is the scale factor for number of channels
        super(AlexNetold, self).__init__()
        
        # HT initialization
        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.fc_init = fc_init

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        if self.activation=='tanh':
            #a_func=F.tanh
            a_func = nn.Tanh()
        elif self.activation=='relu':
            #a_func=F.relu
            a_func = nn.ReLU(inplace=True)
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # for conv2d
        if alpha != None and g != None:
            with torch.no_grad():
                for m in self.features:
                    if isinstance(m, nn.Conv2d):
                        size = m.weight.shape
                        N_eff = int(np.prod(m.kernel_size)) * m.in_channels     # most correct
                        m.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                    size=size)))  
                                           
        self.size = self.get_size()
        print(self.size)
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Dropout(),
            nn.Linear(self.width, self.width, bias=with_bias),
            #nn.ReLU(inplace=True),
            a_func,
            nn.Linear(self.width, num_classes, bias=with_bias),
        )

        # for linear
        if fc_init == "fc_ht":
            if alpha != None and g != None:
                with torch.no_grad():
                    for l in self.classifier:
                        if isinstance(l, nn.Linear):
                            size = l.weight.shape
                            N_eff = (size[0]*size[1])**0.5
                            l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                        scale=g*(0.5/N_eff)**(1./alpha))))
        
            
        elif fc_init == "fc_orthogonal":
            with torch.no_grad():
                for l in self.classifier:
                    if isinstance(l, nn.Linear):
                        nn.init.orthogonal_(l.weight, gain=1)
        
    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



############### ResNet ###############

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out    


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# CIFAR-10 models
def ResNet14():
    depth = 14
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n])


############### ResNet-HT ###############

# HT-initialization for nn.Conv2d structures
def conv2d_ht(module: nn.Conv2d, alpha, g):
    with torch.no_grad():
        size = [module.out_channels, module.in_channels]
        for dim in module.kernel_size:
            size.append(dim)
        size = tuple(size)
        # old
        #N_eff = np.prod(size)**0.5
        # new
        N_eff = int(np.prod(module.kernel_size)) * module.in_channels     # most correct
        module.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                    size=size)))   

    return module  


class BasicBlock_ht(nn.Module):
    expansion = 1

    def __init__(self, alpha, g, activation, in_planes, planes, stride=1):
        super(BasicBlock_ht, self).__init__()
        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # heavy-tailed initialization  
        self.conv1 = conv2d_ht(self.conv1, alpha, g)
        self.bn1   = nn.BatchNorm2d(planes)
        # should something be done for the batchn normalization layer?
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # heavy-tailed initialization  
        self.conv2 = conv2d_ht(self.conv2, alpha, g)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            conv_shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            conv_shortcut = conv2d_ht(conv_shortcut, alpha, g)
            self.shortcut = nn.Sequential(
                conv_shortcut,
                nn.BatchNorm2d(self.expansion*planes)
            )         

    def forward(self, x):
        if self.activation=='tanh':
            a_func=F.tanh
        elif self.activation=='relu':
            a_func=F.relu
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        #out = F.relu(self.bn1(self.conv1(x)))
        out = a_func(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        #out = F.relu(out)
        out = a_func(out)
        return out

    def get_hidden_outputs(self, x):
        if self.activation=='tanh':
            a_func=F.tanh
        elif self.activation=='relu':
            a_func=F.relu
        else:
            raise NameError("activation does not exist in NetPortal.architectures")     

        hidden_ls = []
        out = self.conv1(x)
        hidden_ls.append(out)
        out = self.bn1(out)
        hidden_ls.append(out)
        out = a_func(out)
        hidden_ls.append(out)
        out = self.bn2(self.conv2(out))
        hidden_ls.append(out)
        out += self.shortcut(x)
        hidden_ls.append(out)
        out = a_func(out)
        hidden_ls.append(out)
        return hidden_ls


class ResNet_ht(nn.Module):
    def __init__(self, alpha, g, activation, block, num_blocks, num_classes=10):
        super(ResNet_ht, self).__init__()
        self.in_planes = 16

        self.alpha = alpha
        self.g = g
        self.activation = activation
        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv2d_ht(self.conv1, alpha, g)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.alpha, self.g, self.activation, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.activation=='tanh':
            a_func=F.tanh
        elif self.activation=='relu':
            a_func=F.relu
        else:
            raise NameError("activation does not exist in NetPortal.architectures")  

        #out = F.relu(self.bn1(self.conv1(x)))
        out = a_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



# CIFAR-10 models
def ResNet14_ht(alpha, g, activation):
    depth = 14
    n = (depth - 2) // 6
    return ResNet_ht(alpha, g, activation, BasicBlock_ht, [n,n,n])


################ CNN with circular padding ####################


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, alpha, g, N_eff, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.g = g
        self.N_eff = N_eff

        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))


        with torch.no_grad():
            size = self.weight.shape
            #for dim in module.kernel_size:
            #    size.append(dim)
            #size = tuple(size)
            #N_eff = np.prod(size)**0.5
            self.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                        size=size))) 

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)



def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0, 
    so we need to implement the internal torch logic manually. 

    Currently the ``RuntimeError`` is
    
    "PyTorch convert function for op '_convolution_mode' not implemented"
    """

    def __init__(
            self,
            alpha,
            g,
            N_eff,
            in_channels, 
            out_channels, 
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        self.alpha = alpha
        self.g = g
        self.N_eff = N_eff

        # initialize weights
        with torch.no_grad():
            size = self.conv.weight.shape
            #print(size)
            self.conv.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                        size=size))) 

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_, 
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)



# HT-initialization for nn.Conv2d structures
def conv2d_ht_new(module: nn.Conv2d, alpha, g, N_eff):
    with torch.no_grad():
        #size = [module.out_channels, module.in_channels]
        #for dim in module.kernel_size:
        #    size.append(dim)
        #size = tuple(size)
        size = module.weight.shape

        #N_eff = np.prod(size)**0.5
        module.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                    size=size)))   

    return module  


def get_weight(shape, alpha, g, N_eff):
    return nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                    size=shape)))


# tensorflow version
#def conv2d_new(x, w, strides=1, padding='SAME'):
#  return tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)

# need to manually pad
def conv2d_new(x, w, strides=1):
    return nn.Conv2d(x, w, strides=[1, strides, strides, 1])


"""
def conv2d_same_pad(module: nn.Conv2d):

    module.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in module.kernel_size[::-1]]))
"""


# taken from: https://github.com/yl-1993/ConvDeltaOrthogonal-Init/blob/master/_ext/nn/init.py
def conv_delta_orthogonal_(tensor, gain=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal_(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
      raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")
    
    if tensor.size(1) > tensor.size(0):
      raise ValueError("In_channels cannot be greater than out_channels.")
    
    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2)-1)//2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
        else:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor


# directly taken from: https://github.com/brain-research/mean-field-cnns/blob/master/Delta_Orthogonal_Convolution_Demo.ipynb
def circular_padding(input_, width, kernel_size):
    """Padding input_ for computing circular convolution."""
    begin = kernel_size // 2
    end = kernel_size - 1 - begin
    #tmp_up = input_[0::, width - begin:width + 1, 0:width + 1, 0::]
    tmp_up = input_[0::, 0::, width - begin:width + 1, 0:width + 1]
    #tmp_down = input_[0::, 0:end + 1, 0:width + 1, 0::]
    tmp_down = input_[0::, 0::, 0:end + 1, 0:width + 1]

    print(tmp_up.shape)
    print(input_.shape)
    print(tmp_down.shape)

    tmp = torch.cat([tmp_up, input_, tmp_down], 2)

    new_width = width + kernel_size - 1
    #tmp_left = tmp[0::, 0:new_width + 1, width - begin:width+1, 0::]
    tmp_left = tmp[0::, 0::, 0:new_width + 1, width - begin:width+1]
    #tmp_right = tmp[0::, 0:new_width + 1, 0:end + 1, 0::]
    tmp_right = tmp[0::, 0::, 0:new_width + 1, 0:end + 1]

    print(tmp_left.shape)
    print(tmp.shape)
    print(tmp_right.shape)

    return  torch.cat([tmp_left, tmp, tmp_right], 3)


# no bias, cifar10
class CONVNET_1D(nn.Module):

    def __init__(self, alpha, g, depth, c_size, k_size, activation, num_classes=10):
        super(CONVNET_1D, self).__init__()

        self.activation = activation
        self.alpha = alpha
        self.g = g
        self.depth = depth
        self.c_size = c_size
        self.k_size = k_size
        self.num_classes = num_classes

        # creating structure
        modules = []

        # activation function
        if self.activation=='tanh':
            a_func = nn.Tanh()
        elif self.activation=='relu':
            a_func = nn.ReLU()
        else:
            raise NameError("activation does not exist in NetPortal.architectures")  

        #shape = [k_size, k_size, 1, c_size]
        std = np.sqrt(self.g / (k_size**2 * 3))
        #kernel_0 = nn.Conv2d(3, c_size, kernel_size=k_size, stride=1, padding=0, bias=False)        # padding='SAME' in tf
        N_eff = np.sqrt(k_size**2 * 3)      # to match the input in_channel
        #kernel_0 = conv2d_ht_new(kernel_0, alpha, g, N_eff)

        #kernel_0 = Conv2dSamePadding(alpha=alpha, g=g, N_eff=N_eff,in_channels=3, out_channels=c_size, kernel_size=k_size, stride=1, bias=False)
        kernel_0 = Conv2dSame(alpha=alpha, g=g, N_eff=N_eff,in_channels=3, out_channels=c_size, kernel_size=k_size, stride=(1,1), padding=0, bias=False)
        
        modules.append(kernel_0)
        modules.append(a_func)

        # for conv2d where the in_channels are no longer 3
        for idx in range(2):
            #kernel = nn.Conv2d(c_size, c_size, kernel_size=k_size, stride=2, padding=0, bias=False)
            N_eff = np.sqrt(k_size**2 * c_size)
            #kernel = conv2d_ht_new(kernel, alpha, g, N_eff)
            #kernel = Conv2dSamePadding(alpha=alpha, g=g, N_eff=N_eff,in_channels=c_size, out_channels=c_size, kernel_size=k_size, stride=2, bias=False)
            kernel = Conv2dSame(alpha=alpha, g=g, N_eff=N_eff,in_channels=c_size, out_channels=c_size, kernel_size=k_size, stride=(2,2), padding=0, bias=False)
            modules.append(kernel)

            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NameError("activation does not exist in NetPortal.architectures")

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        # is this still the case for cifar10? 
        # Apparently this is width of the current image after dimension reduction: https://github.com/brain-research/mean-field-cnns/blob/master/Delta_Orthogonal_Convolution_Demo.ipynb
        new_width = 9


        # cifar10 
        #z = torch.reshape(x, [-1,32,32,3])
        z = torch.reshape(x, [-1,3,32,32])
        out = self.sequential(z)

        std = self.g**self.alpha
        for idx in range(self.depth):
            kernel = nn.Conv2d(self.c_size, self.c_size, kernel_size=self.k_size, stride=(1,1), padding=0, bias=False)
            conv_delta_orthogonal_(kernel.weight, gain=std)   # delta orthogonalization

            print(f"kernel: {kernel.weight.shape}, {kernel.kernel_size}")

            # circular padding
            z_pad = circular_padding(out, new_width, self.k_size)
            print(f"{idx} z_pad: {z_pad.shape}")        # delete

            #out = conv2d_new(z_pad, kernel)     # equivalent to a "VALID" padding in tensorflow, which is equivalent to no padding
            out = kernel(z_pad)
            print(f"{idx} pre pre: {out.shape}")        # delete

            if self.activation == 'relu':
                #out = nn.ReLU(out)
                #a_func = nn.ReLU
                a_func = F.relu
            elif self.activation == 'tanh':
                #out = nn.Tanh(out)
                #a_func = nn.Tanh
                a_func = F.tanh
            else:
                raise NameError("activation does not exist in NetPortal.architectures")

            print(f"{idx} pre: {out.shape}")        # delete
            out = a_func(out)
           
        print(f"end: {out.shape}")        # delete
        # adjust size and take mean
        out = torch.mean(out, [0,3])
        print(f"out: {out.shape}")        # delete

        # final layer
        weight_final = nn.Linear(self.c_size, self.num_classes, bias=False)
        size = weight_final.weight.shape
        N_eff = self.c_size**0.5     # note this is somehow not `N_eff = (size[0]*size[1])**0.5`
        weight_final.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(self.alpha, 0, size=size,
                                    scale=self.g*(0.5/N_eff)**(1./self.alpha))))

        out = weight_final(out)

        return out


################ CNN vanilla modified ###############

# taken from: https://github.com/yl-1993/ConvDeltaOrthogonal-Init/blob/master/models/vanilla.py
"""
class Vanilla(nn.Module):

    def __init__(self, base, c, alpha, g, num_classes=10, conv_init='ht_initialization'):
    #def __init__(self, base, c, num_classes=10, conv_init='conv_delta_orthogonal'):
        super(Vanilla, self).__init__()
        self.alpha = alpha
        self.g = g

        #self.init_supported = ['conv_delta_orthogonal', 'kaiming_normal']
        self.init_supported = ['conv_delta_orthogonal', 'ht_initialization']
        if conv_init in self.init_supported:
            self.conv_init = conv_init
        else:
            print('{} is not supported'.format(conv_init))
            self.conv_init = 'kaiming_normal'
        print('initialize conv by {}'.format(conv_init))
        self.base = base
        self.fc = nn.Linear(c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.conv_init == self.init_supported[0]:
                    conv_delta_orthogonal_(m.weight)
                elif self.conv_init == self.init_supported[1]:
                    size = m.weight.shape
                    N_eff = np.sqrt(m.kernel_size[0]**2 * m.in_channels)
                    m.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                size=size)))                     
                else:
                    raise NameError("This initialization method does not exist!")

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
"""

# method 1
# generate an orthogonal matrix with multifractal column/row vectors
def multifractal_orthogonal(dim, alpha, g):
    # method 1
    """
    directly generate a levy matrix and to a QR decomposition
    """
    #a = torch.zeros((dim, dim)).normal_(0, 1)
    a = torch.Tensor(levy_stable.rvs(alpha, 0, size=(dim,dim), scale=g))
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeMultifractalOrthogonal(weights, gain, alpha, g):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    #q = genOrthgonal(dim)
    q = multifractal_orthogonal(dim,alpha,g)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
    weights.mul_(gain)


# method 2
# generate matrix with eigenvalues on the complex unity circle
def multifractal(dim, alpha, g):
    a = levy_stable.rvs(alpha, 0, size=(dim,dim), scale=g)
    eigvals, eigvecs = LA.eig(a)
    return torch.Tensor(np.real(eigvecs @ np.diag(eigvals/np.abs(eigvals)) @  LA.inv(eigvecs)))


def makeMultifractal(weights, gain, alpha, g):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    #q = genOrthgonal(dim)
    q = multifractal(dim,alpha,g)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
    weights.mul_(gain)


# method 3
def multifractal_sym(dim, alpha, g):
    a = levy_stable.rvs(alpha, 0, size=(int(dim/2),dim), scale=g) + levy_stable.rvs(alpha, 0, size=(int(dim/2),dim), scale=g) * 1j
    for row in range(a.shape[0]):
        a[row,:] = a[row,:]/LA.norm(a[row,:])
    P = np.vstack((a,np.conjugate(a))).T
    b = np.exp(rand.uniform(0,2*np.pi,int(dim/2)) * 1j)
    D = np.hstack((b,np.conjugate(b)))
    return torch.Tensor(np.real(P @ np.diag(D) @ LA.inv(P)))


def makeMultifractalSym(weights, gain, alpha, g):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    #q = genOrthgonal(dim)
    q = multifractal_sym(dim,alpha,g)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
    weights.mul_(gain)


class Vanilla(nn.Module):

    def __init__(self, base, c, alpha, g, fc_init, mfrac='ht', with_bias=False, num_classes=10):
        super(Vanilla, self).__init__()
        self.alpha = alpha
        self.g = g
        self.fc_init = fc_init

        #self.init_supported = ['conv_delta_orthogonal', 'kaiming_normal']
        #self.init_supported = ['conv_delta_orthogonal', 'ht_initialization']
        self.init_supported = ['mfrac_orthogonal', 'mfrac', 'mfrac_sym', 'ht']

        self.base = base
        self.mfrac = mfrac
        self.with_bias = with_bias
        self.fc = nn.Linear(c, num_classes, bias=with_bias)

        assert mfrac in self.init_supported, "Initialization type not supported!"

        # initialize fc layer

            # set bias with variance 2e-5 as in paper (only for Gaussian)
            #bias_shape = self.fc.bias.shape
            #self.fc.bias = nn.Parameter(torch.Tensor( normal(0,np.sqrt(2e-5),bias_shape) ))


        # for linear
        if fc_init == "fc_ht":
            if alpha != None and g != None:
                with torch.no_grad():
                    for l in self.classifier:
                        if isinstance(l, nn.Linear):
                            size = l.weight.shape
                            # old fc
                            N_eff = (size[0]*size[1])**0.5
                            # new fc
                            #N_eff = c**0.5      # size[1] is c
                            # fc type 3
                            #N_eff = c
                            l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                        scale=g*(0.5/N_eff)**(1./alpha))))
                    
        elif fc_init == "fc_orthogonal":
            with torch.no_grad():
                for l in self.classifier:
                    if isinstance(l, nn.Linear):
                        nn.init.orthogonal_(l.weight, gain=1)

        # ht initialize first 3 conv2d layers, and delta orthogonal initialization for the remaining conv2d layers
        idx = 0
        for m in self.modules():
            with torch.no_grad():
                if isinstance(m, nn.Conv2d):
                    if mfrac == 'mfrac_orthogonal':
                        makeMultifractalOrthogonal(m.weight, 1, alpha, g)
        
                    elif mfrac == 'mfrac':
                        makeMultifractal(m.weight, 1, alpha, g)

                    elif mfrac == 'mfrac_sym':
                        makeMultifractalSym(m.weight, 1, alpha, g)                    

                    elif mfrac == 'ht':
                        """
                        if idx > 2:
                            conv_delta_orthogonal_(m.weight)
                        else:
                            size = m.weight.shape
                            N_eff = np.sqrt(m.kernel_size[0]**2 * m.in_channels)
                            m.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                        size=size)))  
                        """
                        size = m.weight.shape
                        # type 1
                        N_eff = int(np.prod(m.kernel_size)) * m.in_channels     # most correct
                        # type 2
                        #N_eff = np.sqrt(int(np.prod(m.kernel_size)) * m.in_channels)
                        # type 3
                        #N_eff = np.sqrt(int(np.prod(m.kernel_size)) * m.in_channels * m.out_channels)
                        m.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                    size=size)))  
                        
                        #m.weight = nn.Parameter(torch.Tensor( normal(0, g*(1/N_eff)**(1./2),size) ))      # Gaussian init
                        # set bias with variance 2e-5 as in paper (only for Gaussian)
                        #bias_shape = m.bias.shape
                        #m.bias = nn.Parameter(torch.Tensor( normal(0,np.sqrt(2e-5),bias_shape) ))
  
                    idx += 1                 

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def make_layers(depth, dataset, with_bias=False):
    assert isinstance(depth, int)
    c = 256 if depth <= 256 else 128
    layers = []
    if dataset == "cifar10":
        in_channels = 3  
    elif dataset == "mnist":
        in_channels = 1
    for stride in [1, 2, 2]:
        conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=stride, bias=with_bias)
        layers += [conv2d, nn.Tanh()]
        in_channels = c
    for _ in range(depth):
        conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=with_bias)
        layers += [conv2d, nn.Tanh()]
    if dataset == "cifar10":
        layers += [nn.AvgPool2d(8)] # For mnist is 7
    elif dataset == "mnist":
        layers += [nn.AvgPool2d(7)]
    return nn.Sequential(*layers), c


def make_layers_nobias(depth):
    assert isinstance(depth, int)
    c = 256 if depth <= 256 else 128
    layers = []
    #in_channels = 3
    in_channels = 1
    for stride in [1, 2, 2]:
        conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=stride, bias=False)
        layers += [conv2d, nn.Tanh()]
        in_channels = c
    for _ in range(depth):
        conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        layers += [conv2d, nn.Tanh()]
    #layers += [nn.AvgPool2d(8)] # For mnist is 7
    layers += [nn.AvgPool2d(7)]
    return nn.Sequential(*layers), c



def van5(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(5, dataset, with_bias=True), alpha, g, with_bias=True, **kwargs)
    return model


def van20(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(5, dataset, with_bias=True), alpha, g, fc_init="fc_default", with_bias=True, **kwargs)
    return model


def van32(alpha, g, dataset, **kwargs):
    """Constructs a 32 layers vanilla model.
    """
    model = Vanilla(*make_layers(32,dataset,with_bias=True), alpha, g, with_bias=True, **kwargs)
    return model


def van50(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(50,dataset,with_bias=True), alpha, g, with_bias=True, **kwargs)
    return model


def van100(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(100,dataset,with_bias=True), alpha, g, with_bias=True, **kwargs)
    return model


def van200(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(200,dataset,with_bias=True), alpha, g, with_bias=True, **kwargs)
    return model


def van300(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(300,dataset,with_bias=True), alpha, g, with_bias=True, **kwargs)
    return model

# --- no bias ----

#def van100nobias(alpha, g, **kwargs):
#    model = Vanilla_nobias(*make_layers_nobias(100), alpha, g, **kwargs)
#    return model

def van2nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(2,dataset), alpha, g, **kwargs)
    return model

def van5nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(5,dataset), alpha, g, **kwargs)
    return model

def van10nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(10,dataset), alpha, g, **kwargs)
    return model

def van20nobias(alpha, g, fc_init, dataset, **kwargs):
    model = Vanilla(*make_layers(20,dataset), alpha, g, fc_init, **kwargs)
    return model

def van50nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(50,dataset), alpha, g, **kwargs)
    return model

def van100nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(100,dataset), alpha, g, **kwargs)
    return model

def van150nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(150,dataset), alpha, g, **kwargs)
    return model

def van300nobias(alpha, g, dataset, **kwargs):
    model = Vanilla(*make_layers(300,dataset), alpha, g, **kwargs)
    return model

################ CNN vanilla original ###############

class Vanilla_og(nn.Module):

    def __init__(self, base, c, num_classes=10, conv_init='conv_delta_orthogonal'):
        super(Vanilla_og, self).__init__()
        self.init_supported = ['conv_delta_orthogonal', 'kaiming_normal']
        if conv_init in self.init_supported:
            self.conv_init = conv_init
        else:
            print('{} is not supported'.format(conv_init))
            self.conv_init = 'kaiming_normal'
        print('initialize conv by {}'.format(conv_init))
        self.base = base
        self.fc = nn.Linear(c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.conv_init == self.init_supported[0]:
                    conv_delta_orthogonal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def van5_og(**kwargs):
    model = Vanilla_og(*make_layers(5), **kwargs)
    return model


def van32_og(**kwargs):
    """Constructs a 32 layers vanilla model.
    """
    model = Vanilla_og(*make_layers(32), **kwargs)
    return model


################ CNN simple ###############

# directly taken from the torch tute: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CONVNET_simple(nn.Module):
    def __init__(self, alpha, g, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation = activation
        self.alpha = alpha
        self.g = g

        # ht initialize conv2d layers and fc layers
        for l_name in list(self._modules.keys()):
            l = self._modules[l_name]
            with torch.no_grad():
                if isinstance(l, nn.Linear): 
                    size = l.weight.shape
                    N_eff = (size[0]*size[1])**0.5
                    l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha, 0, size=size,
                                                scale=g*(0.5/N_eff)**(1./alpha))))
                elif isinstance(l, nn.Conv2d):
                    size = l.weight.shape
                    #print(size)
                    #print(conv.kernel_size)
                    N_eff = np.sqrt(l.kernel_size[0]**2 * l.in_channels)
                    l.weight = nn.Parameter(torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=g*(0.5/N_eff)**(1./alpha),
                                                size=size)))                 


    def forward(self, x):

        if self.activation == 'relu':
            a_func = F.relu
        elif self.activation == 'tanh':
            a_func = F.tanh
        else:
            raise NameError("activation does not exist in NetPortal.architectures")


        x = self.pool(a_func(self.conv1(x)))
        x = self.pool(a_func(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = a_func(self.fc1(x))
        x = a_func(self.fc2(x))
        x = self.fc3(x)
        return x




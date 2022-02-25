import math
import numpy as np
import numpy.random as random

import torch
import torch.nn as nn
import copy 
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import levy_stable

# This script is only for propagating randomly initializaed networks with square connectivity matrices (NOT FOR TRAINING!)

# derivative for tanh, only for scalar output
def dtanh(x):
    m = nn.Tanh()
    x = torch.autograd.Variable(x, requires_grad=True)
    y = m(x)
    y.backward( torch.ones_like(x) )

    return x.grad

def randomize_weight(w, w_alpha, w_mult):
    if mu is None:
        mu = w.mean()
    if sigma is None:
        sigma = w.std()
    return sigma * np.random.randn(*w.shape) + mu

# class randnet(object):
class randnet(nn.Module):

    # add input_layer=None
    def __init__(self, input_dim , width, depth, num_classes, w_alpha, w_mult, **kwargs):

        super(randnet, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.w_alpha = w_alpha
        self.w_mult = w_mult

        if num_classes is None:
            num_classes = width
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.Tanh(),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
            nn.Tanh(),
        )

        if 'w_seed' in kwargs:
            # set seed for weights
            w_seed = kwargs.get('w_seed')
            #random.seed(seed=w_seed)

        # load weights w.r.t. w_alpha and w_mult (method 1)
        if False:
            for i, layer_id in enumerate(list(self.state_dict().keys())):
                layer_dim = self.state_dict()[layer_id].shape
                # simulate weight
                alpha = w_alpha 
                beta = 0
                loc = 0
                scale_multiplier =  w_mult

                if 'w_seed' in kwargs:
                    random.seed(seed=w_seed + i * 100000)
                scale = (1/(2*np.sqrt(layer_dim[0] * layer_dim[1])))**(1/alpha) # this is our standard unit of the scale for stable init
                new_weights = levy_stable.rvs(alpha, beta, loc, scale * scale_multiplier, size=layer_dim)
                
                self.state_dict()[layer_id].data.copy_(torch.tensor(new_weights))
        

        # load weights w.r.t. w_alpha and w_mult (method 2)
        
        with torch.no_grad():
            ii = 0
            for param in self.parameters():
                param_dim = param.shape
                # simulate weight
                alpha = w_alpha 
                beta = 0
                loc = 0
                scale_multiplier =  w_mult

                if 'w_seed' in kwargs:
                    random.seed(seed=w_seed + ii * 100000)
                scale = (1/(2*np.sqrt( int(np.prod(param_dim)) )))**(1/alpha) # this is our standard unit of the scale for stable init
#                print('prenew_weights')
                new_weights = levy_stable.rvs(alpha, beta, loc, scale * scale_multiplier, size=param_dim)
#                print('postnew_weights')
                #new_weights = torch.ones(param_dim).to(torch.double)
                
                #param.copy_(new_weights)
                param.copy_(torch.from_numpy(new_weights))
                ii += 1
        

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.Tanh())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

    # preactivation outputs 
    """
    def preact_layer(self, x):
        # number of hidden layers
        hidden = [None] * (self.depth - 1)
        x = x.view(x.size(0), -1)
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden[idx] = self.fc[idx](x)
            else:
                hidden[idx] = self.fc[ell * idx - 1: ell * idx + 1](hidden[idx - 1])

        return hidden + [self.fc[-2:](hidden[-1])]
    """
    def preact_layer(self, x):
        x = x.view(x.size(0), self.input_dim)
        hidden_all = torch.empty((self.depth+1, x.shape[0], x.shape[1]))    # includes input layer
        #hidden_all = torch.empty((self.depth+1, x.shape[0], x.shape[1])).type(torch.DoubleTensor)
        hidden_all[0,:,:] = x   # input layers
        ell = 2

        for idx in range(self.depth):
            if idx == 0:
                hidden_all[idx + 1,:,:] = self.fc[idx](x)
            else:
                hidden_all[idx + 1,:,:] = self.fc[ell * idx - 1: ell * idx + ell - 1](hidden_all[idx,:,:].clone())

        #hidden_all[self.depth,:,:] = self.fc[-2:](hidden_all[self.depth-1,:,:].clone())

        return hidden_all
    
    # postactivation outputs (key is to return everything altogether as just one big tensor)
    def postact_layer(self, x):
        x = x.view(x.size(0), self.input_dim)
        hidden_all = torch.empty((self.depth+1, x.shape[0], x.shape[1]))    # includes input layer
        #hidden_all = torch.empty((self.depth+1, x.shape[0], x.shape[1])).to(torch.double)    # includes input layer
        hidden_all[0,:,:] = x   # input layers
        ell = 2

        for idx in range(self.depth - 1):
            if idx == 0:
                hidden_all[idx + 1,:,:] = self.fc[ell * idx: ell * idx + ell](x)
            else:
                hidden_all[idx + 1,:,:] = self.fc[ell * idx: ell * idx + ell](hidden_all[idx,:,:].clone())

        #hidden_all[self.depth,:,:] = self.sequential[-2:](hidden_all[self.depth-1,:,:])
        hidden_all[self.depth,:,:] = self.fc[-2:](hidden_all[self.depth-1,:,:].clone())

        #torch.autograd.set_detect_anomaly(True)

        return hidden_all

    # xs is the input
    #def get_acts_and_derivatives(self, xs, include_hessian=False):
            
        

    # the problem for this is that the last weight matrix in all these settings are not symmetrical, i.e. for mnist W_L is 784 by 10 (might need to adjust this in the future)
    """
    def jacob_ls(self, x):
        # check "The Emergence of Spectral Universality in Deep Networks"
        
        m = nn.Tanh()
        preact_h = self.preact_layer(x)     
        # get weights
        weights = [p for p in self.parameters()]
        
        dphi_h = dtanh(preact_h[0][0])
        DW_l = torch.matmul(torch.diag( dphi_h ), weights[0])
        #DW_l = torch.matmul(torch.diag( m(preact_h[0][0]) ), weights[0])
        DW_ls = [DW_l]

        # due to the last matrix being non-square, the case l = L is not included
        for i in range(1, len(preact_h) - 1):
            dphi_h = dtanh(preact_h[i][0])
            DW_l = torch.matmul(torch.diag( dphi_h ), weights[i])
            #DW_l = torch.matmul(torch.diag( m(preact_h[i][0]) ), weights[i])
            DW_ls.append(DW_l)

        return DW_ls    
    """

# great circle
class GreatCircle():

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        U, _, _ = np.linalg.svd(np.random.randn(output_dim, 2), full_matrices=False)
        self.U = K.variable(U.T)
        self.scale = K.variable(1.0)
        kwargs['input_shape'] = (1, )
        super(GreatCircle, self).__init__(**kwargs)

    


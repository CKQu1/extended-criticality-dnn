import numpy as np
import math
from scipy.stats import levy_stable
import torch

# 1. torch NN weights, bulk input data to get hidden (pre/post-act)

# following gets specific hidden layer
def preact_from_torch(w, inputs, n_l):

    assert n_l >= 1 and n_l <= len(w), "Incorrect depth for hidden layer specified."

    hidden = inputs
    for l in range(n_l - 1):
        hidden = np.matmul(w[l],hidden)
        hidden = np.tanh(hidden)        

    return np.matmul(w[n_l - 1],hidden)

def postact_from_torch(w, inputs, n_l):

    assert n_l >= 1 and n_l <= len(w), "Incorrect depth for hidden layer specified."

    hidden = inputs
    for l in range(n_l):
        hidden = np.matmul(w[l],hidden)
        hidden = np.tanh(hidden)        

    return hidden

# following gets all hidden layer
def preact_all(w, inputs):

    hidden = [inputs]
    temp = inputs
    for l in range(len(w)):
        temp = np.matmul(w[l],temp)
        hidden.append(temp)
        temp = np.tanh(temp)        

    return hidden

def postact_all(w, inputs):

    hidden = [inputs]
    temp = inputs
    for l in range(len(w)):
        temp = np.matmul(w[l],temp)
        temp = np.tanh(temp)    
        hidden.append(temp)    

    return hidden

# 2. FCNs at different initialization schemes

# no bias, square weight matrices
def pre_act(alpha_m,xLN_0):

    print(alpha_m)
    x,L,N_0 = xLN_0
    alpha, m = alpha_m
    # Levy alpha params
    beta = 0
    loc = 0
    scale = (1/(2*math.sqrt(N_0*N_0)))**(1/alpha) # this is our standard unit of the scale for stable init

    preact_layers = [x]
    for l in range(L):
        W_l = levy_stable.rvs(alpha, beta, loc, scale * m, size=(N_0,N_0))
        x_l = np.matmul(preact_layers[l],W_l)        
        preact_layers.append(x_l)
        x_l = np.tanh(x_l)

    return layer_pca(preact_layers, 3)

def post_act(alpha_m,xLN_0):

    print(alpha_m)
    x,L,N_0 = xLN_0
    alpha, m = alpha_m
    # Levy alpha params
    beta = 0
    loc = 0
    scale = (1/(2*math.sqrt(N_0*N_0)))**(1/alpha) # this is our standard unit of the scale for stable init

    postact_layers = [x]
    for l in range(L):
        W_l = levy_stable.rvs(alpha, beta, loc, scale * m, size=(N_0,N_0))
        #x_l = np.tanh(np.matmul(W_l,postact_layers[l]))
        x_l = np.tanh(np.matmul(postact_layers[l],W_l))                
        postact_layers.append(x_l)

    return layer_pca(postact_layers, 3)

def layer_pca(layer_ls,n_pc):

    hs_proj = []
    singular_values = []

    for j in range(len(layer_ls)):
        hidden = layer_ls[j]

        hidden = hidden - hidden.mean(0,keepdims=True)
        u, s, v = np.linalg.svd(hidden, full_matrices=False)
        norm_s = s[:n_pc] / s[:n_pc].max()
        u = (norm_s[None, :]) * u[:, :n_pc] 
        u = u/ np.abs(u).max()
        singular_values.append(s)
        #hs_proj.append(u[:, :3])
        hs_proj.append(u[:, :])

    return [hs_proj, singular_values]

# 3. Simulation of connectivity matrix for FCN with numpy

def wm_np_sim(N_0,L,alpha,m):
    
    wm_ls = []
    beta = 0
    loc = 0
    scale = (1/(2*math.sqrt(N_0*N_0)))**(1/alpha)

    for l in range(L):
        wm_ls.append(levy_stable.rvs(alpha, beta, loc, scale * m, size=(N_0,N_0)))

    return wm_ls


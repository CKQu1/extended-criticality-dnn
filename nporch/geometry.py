import math
import numpy as np
import scipy.io as sio
import time
import torch

from numpy import dot
from torch import tensor

# computing curvature
def kappa(v,a):
    return (dot(v,v))**(-3/2)*(dot(v,v)*dot(a,a) - dot(v,a)**2)

# returned as torch.tensor
def gbasis(u0,u1,theta_ls,N_0,q):
    
    assert len(u0) == N_0 and len(u1) == N_0, "u_0 or u_1 dimension doesn't match N_0"

    u0, u1 = u0.reshape((-1,1)), u1.reshape((-1,1))
    #U0,U1 = np.repeat(u0, len(theta_ls), axis=1), np.repeat(u1, len(theta_ls), axis=1)
    U0, U1 = u0.repeat((1,len(theta_ls))), u1.repeat((1,len(theta_ls)))  

    # just watch this part if torch diff doesn't work (dtype=torch.float32)       
    return torch.sqrt( tensor([float(N_0*q)]) )*(torch.cos(theta_ls)*U0 + torch.sin(theta_ls)*U1)
    #return torch.sqrt( tensor([N_0*q]) )*(torch.cos(theta_ls)*U0 + torch.sin(theta_ls)*U1, dtype=torch.float32)

# wm_ls,u0,u1 fixed
def hidden_fixed(N_0,q,theta,wm_ls,u0,u1,post=True):

    L = len(wm_ls)
    #gcirc = np.sqrt(N_0*q)*(np.cos(angles[0])*u0 + np.sin(angles[0])*u1)
    gcirc = np.sqrt(N_0*q)*(np.cos(theta)*u0 + np.sin(theta)*u1)
    hidden_ls = [gcirc - gcirc.mean(0,keepdims=True)] # includes input layer

    if post == True:
        for l in range(L):
            W_l = wm_ls[l]
            x_l = np.tanh(np.matmul(hidden_ls[l],W_l))
            hidden_ls.append(x_l)
    else:
        for l in range(L):
            W_l = wm_ls[l]
            x_l = np.matmul(hidden_ls[l],W_l)        
            preact_layers.append(x_l)
            x_l = np.tanh(x_l)

    return np.array(hidden_ls)

# velocity and acceleration (first and second order central)
def vel_acc(N_0,q,theta_ls,wm_ls,u0,u1,post=True):        

    h = 1e-15 # error
    vel_ls = []
    acc_ls = []

    for i in range(len(theta_ls)):
        theta = theta_ls[i]
        hidden_ls = hidden_fixed(N_0,q,theta,wm_ls,u0,u1,post)

        vel = (hidden_fixed(N_0,q,theta + h,wm_ls,u0,u1,post) - hidden_fixed(N_0,q,theta - h,wm_ls,u0,u1,post))/(2*h)
        acc = (hidden_fixed(N_0,q,theta + h,wm_ls,u0,u1,post) + hidden_fixed(N_0,q,theta,wm_ls,u0,u1,post) - hidden_fixed(N_0,q,theta - h,wm_ls,u0,u1,post))/(h**2)

        vel_ls.append(vel)
        acc_ls.append(acc)

    return np.array(vel_ls), np.array(acc_ls)

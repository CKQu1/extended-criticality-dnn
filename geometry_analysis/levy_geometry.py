import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy.io as sio
import sys
import time
import torch

from numpy import dot
from scipy.stats import levy_stable
from tqdm import tqdm

import torch.nn as nn
import copy 
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import ortho_group
from torch import tensor
from torch.autograd import Variable
from torch.autograd.functional import jacobian, hessian

#lib_path = os.getcwd()s
#sys.path.append(f'{lib_path}')

#lib_path = os.getcwd()
#sys.path.append(f'{lib_path}')

from nporch.np_forward import wm_np_sim
from nporch.geometry import kappa, gbasis

from nporch.Randnet import randnet
from nporch.theory import q_star


# tanh derivative
def tanh_der(x):

    #return torch.ones(x.shape) - nn.Tanh()(x)**2
    return 1 - nn.Tanh()(x)**2

def tanh_dd(x):

    #return torch.ones(x.shape) - nn.Tanh()(x)**2
    return -2 * nn.Tanh()(x) * (1 - nn.Tanh()(x)**2)

###############################

def init_to_geometry(N_theta, w_alpha, w_mult, n_coarse, *args):

    hess_include=False
    torch.set_default_dtype(torch.float64)

    data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact"
    
    w_alpha = float(w_alpha)
    w_mult = float(w_mult)
    #tidx = int(tidx)
    n_coarse = int(n_coarse)

    # depth as no. of Ws
    L = 40
    # input dimension/hidden units
    N_0 = 1000
    # input angles
    N_theta = int(N_theta)

    theta_ls = tensor(np.linspace(0, np.pi*2, N_theta))
    delta_theta = theta_ls[1] - theta_ls[0]
    # fix basis
    u0 = torch.zeros(N_0)
    u1 = torch.zeros(N_0)
    u0[0] = 1
    u1[1] = 1

    # create folder
    folder_name = f"fc{L}_{round(w_alpha,1)}_{round(w_mult,2)}"
    if not os.path.exists(f'{data_path}/{folder_name}'):
        os.makedirs(f'{data_path}/{folder_name}')

    # feed the input into network
    from time import time
    pretime = time()
    net = randnet(input_dim=N_0, width=N_0, depth=L, num_classes=None, w_alpha=w_alpha, w_mult=w_mult, w_seed=n_coarse)
    print(f'net loaded in {time()-pretime} s')

    gEs_n = torch.empty((L + 1, N_theta))
    if hess_include:
        kappas_sq_n = torch.empty((L + 1, N_theta))
        L_gs_n = torch.empty((L + 1, N_theta))

    # save hyperparams
    if not os.path.exists(f'{data_path}/{folder_name}/net_init.csv'):
        # operate at the fixed length
        q = q_star(w_alpha, w_mult)

        df_params = pd.DataFrame(np.zeros((1,6)))
        df_params.columns = ["L", "w_alpha", "w_mult", "N_0", "q", "N_theta"]
        df_params.iloc[0,:] = [L, w_alpha, w_mult, N_0, q, N_theta]
        df_params.to_csv(f'{data_path}/{folder_name}/net_init.csv', index=False)
    else:
        df = pd.read_csv(f'{data_path}/{folder_name}/net_init.csv')
        q = df.iloc[0,4]    
    
    for tidx in tqdm(range(0,N_theta), miniters=50):

        def dx0_dtheta(thetas):
            return gbasis(u0,u1,thetas,N_0,q).T

        # post-activation jacobians and hessians

        weights_all = net.state_dict()
        w_keys = list(weights_all.keys())
        hidden_tidx = net.preact_layer(gbasis(u0,u1,theta_ls[tidx:tidx+1],N_0,q).T)

        # jacobian init
        jacs_tidx = torch.empty((L + 1, N_0)) 
        x0_dash = jacobian(dx0_dtheta, theta_ls[tidx:tidx + 1])
        x0_dash = torch.squeeze(x0_dash)

        jacs_tidx[0,:] = x0_dash

        if hess_include == True:

            # hessian init
            hesses_tidx = torch.empty((L + 1, N_0))         

            x0_dd = -hidden_tidx[0,:,:]
            x0_dd = torch.squeeze(x0_dd)

            hesses_tidx[0,:] = x0_dd

        for l in range(0,L):

            dphi_h = tanh_der(hidden_tidx[l + 1,:,:])
            W_l = weights_all[w_keys[l]].T
            vec_temp = torch.matmul(jacs_tidx[l,:], W_l)

            xl_dash = dphi_h * vec_temp
            jacs_tidx[l + 1,:] = xl_dash

            if hess_include == True:

                ddphi_h = tanh_dd(hidden_tidx[l,:,:])
                hl_dd = ddphi_h * vec_temp**2 + dphi_h * torch.matmul(hesses_tidx[l,:], W_l)
                hesses_tidx[l + 1,:] = hl_dd

        #return jacs_tidx, hesses_tidx

        # ------ geometrical computation ------
        net_full_path = f'{data_path}/{folder_name}'

        v_sq = torch.empty((int(L) + 1, 1))
        v_sq[:,0] = (jacs_tidx * jacs_tidx).sum(axis=1)
        gEs_n[:,tidx] = v_sq.squeeze()

        if hess_include == True:
            a_sq = torch.empty((int(L) + 1, 1))
            vdota = torch.empty((int(L) + 1, 1))

            a_sq[:,0] = (hesses_tidx * hesses_tidx).sum(axis=1)
            vdota[:,0] = (jacs_tidx * hesses_tidx).sum(axis=1)

            # compute geometry metrics from above
            kappas_sq_no_mean = v_sq**(-3) * (v_sq * a_sq - vdota**2)
            #kappas_no_mean = kappas_sq_no_mean**0.5
            L_gs_no_mean = kappas_sq_no_mean * v_sq

            kappas_sq_n[:,tidx] = (v_sq**(-3) * (v_sq * a_sq - vdota**2)).squeeze()
            L_gs_n[:,tidx] = L_gs_no_mean.squeeze()

        #print(f"Time: {time.time() - t0} secs.")

        #torch.save(kappas_no_mean, f"{net_full_path}/kappa_{w_alpha}_{w_mult}_{tidx}_{n_coarse}")
        #torch.save(v_sq,f"{net_full_path}/gE_{w_alpha}_{w_mult}_{tidx}_{n_coarse}")
        #torch.save(L_gs_no_mean,f"{net_full_path}/Lg_{w_alpha}_{w_mult}_{tidx}_{n_coarse}")

    print(f"Saving data now for tidx: {tidx}, n_coarse: {n_coarse}.")

    torch.save(gEs_n,f"{net_full_path}/gE_{w_alpha}_{w_mult}_{n_coarse}")
    if hess_include == True:
        torch.save(kappas_sq_n, f"{net_full_path}/kappa_{w_alpha}_{w_mult}_{n_coarse}")
        torch.save(L_gs_n,f"{net_full_path}/Lg_{w_alpha}_{w_mult}_{n_coarse}")

"""
def init_to_geometry2(N_theta, w_alpha, w_mult, tidx, n_coarse, *args):

    data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact"

    w_alpha = float(w_alpha)
    w_mult = float(w_mult)
    tidx = int(tidx)
    n_coarse = int(n_coarse)

    # depth as no. of Ws
    L = 40
    # input dimension/hidden units
    N_0 = 10
    # input angles
    N_theta = int(N_theta)

    theta_ls = tensor(np.linspace(0, np.pi*2, N_theta))
    delta_theta = theta_ls[1] - theta_ls[0]
    # fix basis
    u0 = torch.zeros(N_0)
    u1 = torch.zeros(N_0)
    u0[0] = 1
    u0[1] = 1

    # create folder
    folder_name = f"fc{L}_{round(w_alpha,1)}_{round(w_mult,2)}"


    # ------ geometrical computation ------

    net_full_path = f'{data_path}/{folder_name}'

    a = torch.load(f"{net_full_path}/kappa_{w_alpha}_{w_mult}_{tidx}_{n_coarse}")
    b = torch.load(f"{net_full_path}/gE_{w_alpha}_{w_mult}_{tidx}_{n_coarse}")
    c = torch.load(f"{net_full_path}/Lg_{w_alpha}_{w_mult}_{tidx}_{n_coarse}")
    print(a)
    print(a.shape)
    print(b)
    print(b.shape)
    print(c)
    print(c.shape)
"""

# 1. run tidx: range(1), n_coarse: range(1)
# 2. run tidx: range(1, N_theta), n_coarse: range(1, 100)
# 3. run tidx: range(1), n_coarse: range(1, 100)

def submit(*args):
    from qsub import qsub
    pbs_array_data = [(f'{w_alpha:.1f}', str(w_mult), str(n_coarse))
                      for w_alpha in np.arange(1, 2.01, .1)
                      for w_mult in np.arange(0.25, 3.01, .25)
                      #for tidx in range(1, int(args[1]))
                      for n_coarse in range(0,100)
                      ]
    qsub(f'python levy_geometry.py {" ".join(args)}', pbs_array_data,
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact/',
         P='phys_DL',
         mem='2GB')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
    
    
    
    
    
    
    
    

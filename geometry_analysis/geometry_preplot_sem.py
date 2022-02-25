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

import torch.nn as nn
import copy 
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import ortho_group
from torch import tensor
from torch.autograd import Variable
from torch.autograd.functional import jacobian, hessian

#lib_path = os.getcwd()
#sys.path.append(f'{lib_path}')

#lib_path = os.getcwd()
#sys.path.append(f'{lib_path}')

from nporch.np_forward import wm_np_sim
from nporch.geometry import kappa, gbasis, hidden_fixed, vel_acc

from nporch.geometry import kappa, gbasis, hidden_fixed, vel_acc
from nporch.Randnet import randnet
from nporch.theory import q_star

from tqdm import tqdm

def geometry_to_mean(w_alpha, w_mult, *args):

#    global SEMs, gEs

    N_coarse = 100
    L = 40
    w_alpha = float(w_alpha)
    w_mult = float(w_mult)
#    torch.set_default_dtype(torch.float64)

    # data dirs
    data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact"
    folder_name = f"fc{L}_{round(w_alpha,1)}_{round(w_mult,2)}"
    net_full_path = f'{data_path}/{folder_name}'

    df = pd.read_csv(f'{data_path}/{folder_name}/net_init.csv')
    _, _, _, N_0, _, N_theta = df.iloc[0,:]    
    N_0, N_theta = int(N_0), int(N_theta)

    SEMs = np.zeros([N_coarse, L+1])
    nmiss = 0
    for n_coarse in tqdm(range(0,N_coarse), miniters=5):
        try:
            gEs = torch.load(f"{net_full_path}/gE_{w_alpha}_{w_mult}_{n_coarse}").detach().numpy().T
        except FileNotFoundError:
            gEs = np.nan*np.zeros([N_theta, L+1])
            nmiss += 1
        SEMs[n_coarse,:] = np.nanstd(gEs, axis=0)/np.nanmean(gEs, axis=0)

    # convert back to torch
    SEMs = torch.from_numpy(SEMs)

#    print(SEMs)
#    print(type(SEMs))
    print(f"Missing: {nmiss}")

    print(f"Saving data now for w_alpha: {w_alpha}, w_mult: {w_mult}.")
    torch.save(SEMs, f"{net_full_path}/SEM_{w_alpha}_{w_mult}")

def submit(*args):
    from qsub import qsub
    pbs_array_data = [(f'{w_alpha:.1f}', str(w_mult))
                      for w_alpha in np.arange(1, 2.01, .1)
                      for w_mult in np.arange(0.25, 3.01, .25)
                      #for w_alpha in [1.0, 1.1]
                      #for w_mult in [0.5, 0.75]
                      ]
    qsub(f'python geometry_preplot_sem.py {" ".join(args)}', pbs_array_data,
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact/',
         P='phys_DL',
         mem="2GB")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
    
    
    
    
    
    
    
    

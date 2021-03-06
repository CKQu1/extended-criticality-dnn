import numpy as np
import math
import os
import pandas as pd
import powerlaw as plaw
import random
import scipy.io as sio
import sys
import time
import torch

from numpy import dot
from scipy.stats import levy_stable
#from scipy.stats import anderson
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions, norm, entropy
#from scipy.stats import kstest

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
 
#import net_load as nl
#from net_load.get_layer import get_hidden_layers, load_weights, get_epoch_weights, layer_struct

#import train_DNN_code.model_loader as model_loader
#from train_DNN_code.dataloader import get_data_loaders, get_synthetic_gaussian_data_loaders


t0 = time.time()

# ----------------------------

import torchvision.models as models

####### SOME OF THE MODELS DON'T EXIST DUE TO TORCH VERSION 

def get_pretrained_names():
    model_ls = [] 
    # we know the type of alexnet
    class_type = type(models.__dict__['alexnet'])

    attr_ls = list(models.__dict__.keys()) 
    for attr in attr_ls:
        obj = models.__dict__[attr]
        if isinstance(obj, class_type):
            model_ls.append(attr)

    return model_ls

def pretrained_store(n_model, *args):

    t0 = time.time()

    model_ls = get_pretrained_names()
    model_name = model_ls[int(n_model)]
    model = models.__dict__[model_name](pretrained=True)

    t1 = time.time()
    print(f"Loaded {model_name} in {t1 - t0} s")

    # path for saving the weights
    main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
    weight_path = f"{main_path}/weights_all"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # create dataframe that stores the model_name
    if not os.path.exists(f'{main_path}/net_names_all.csv'):
        col_names = ["model_name"]
        df_names = pd.DataFrame(columns=col_names)
        df_names.loc[0] = model_name
    else:
        df_names = pd.read_csv(f'{main_path}/net_names_all.csv')
        if model_name not in df_names.values:
            print(df_names.shape[0])
            df_names.loc[df_names.shape[0]] = model_name
    # save renewed version
    df_names.to_csv(f'{main_path}/net_names_all.csv', index=False)

    i = 0
    #for i in range(len(wmat_name_ls)):
    wmat_idx = 0
    for name, param in model.named_parameters():
        
        if 'bias' not in name and param.dim() > 1:
            weights = param.flatten()
            torch.save(weights, f"{weight_path}/{model_name}_layer_{i}_{wmat_idx}")
            #weights = weights.detach().numpy()          

            print(rf"W{i}: {wmat_idx} done!")
            i += 1
        wmat_idx += 1

    # clear some space
    t_last = time.time()
    print(f"{model_name}: Ws of {i} stored in {t_last - t1} s!")        

def submit(*args):
    from qsub import qsub
    N = len(get_pretrained_names())  # number of models
    pbs_array_data = [(f'{n_model:.1f}')
                      for n_model in list(range(N))
                      #for n_model in list(range(12))
                      #for n_model in list(range(2))
                      #for dummy in [0]
                      ]
    #qsub(f'python geometry_preplot.py {" ".join(args)}', pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact/', P='phys_DL')
    qsub(f'python pretrained_workflow/pretrained_download.py {" ".join(args)}', pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow', P='phys_DL', mem="4GB")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])




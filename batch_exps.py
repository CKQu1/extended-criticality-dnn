import math
import os
import pandas as pd
import re
from itertools import product
from os import makedirs
from os.path import isfile, isdir

from constants import DROOT, CLUSTER, RESOURCE_CONFIGS
from UTILS.mutils import njoin
from qsub_parser import add_common_kwargs, str_to_time, time_to_str

# train FCNs according to the phase transition diagram
def train_mlps():
    pass


# train vanilla cnns according to the phase transition diagram
def train_cnns():

    script_name = 'tf_train_v3.py'
    script_func = 'run_model'

    # c_size = 100  # channel size
    c_size = 250  # channel size
    k_size = 3    # kernel size
    LEARNING_RATE = 5e-3
    MOMENTUM = 0
    BATCH_SIZE = 1024 

    # alpha100s = list(range(100,201,5))
    # g100s = list(range(20, 301, 20))
    alpha100s = [100,200]
    g100s = [20,100,300]
    seeds = [0]
    
    epochs = 50
    DEPTH = 10
    net_type = 'cnn_cpad'  # cnn with circular pad
    fc_init = "fc_default"
    dataset = 'mnist'
    optimizer = 'sgd'

    # resources
    is_use_gpu = True
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]   
    nstack = 3                              
    # single_walltime = '02:10:59'  # cpu  20 epochs
    single_walltime = '00:30:59'  # gpu  50 epochs
    walltime = time_to_str(str_to_time(single_walltime) * nstack)   
    mem = '8GB'   
    select = 1

    # set up dir
    models_path = njoin(DROOT, 'trained_cnns', f'cnn{DEPTH}-v2')
    job_path = njoin(models_path, 'jobs_all')
    if not isdir(models_path): makedirs(models_path)
    # save shared training settings
    df_settings = pd.DataFrame(columns=['net_type', 'depth', 'fc_init','c_size', 'k_size', 
                                        'lr', 'momentum', 'batch_size', 'optimizer','dataset'])
    df_settings.loc[0,:] = [net_type, DEPTH, fc_init, c_size, k_size, 
                            LEARNING_RATE, MOMENTUM, BATCH_SIZE, optimizer, dataset]
    df_settings.to_csv(njoin(models_path, 'settings.csv'))

    kwargss_all = []    
    common_kwargs = {                                  
        'net_type':          net_type,
        'fc_init':           fc_init,
        'dataset':           dataset,    
        'optimizer':         optimizer                    
    }                 

    kwargss = []      
    for alpha100, g100, seed in product(alpha100s, g100s, seeds):
        kwargss.append({'alpha100': alpha100, 'g100': g100, 'seed': seed,
                        'depth': DEPTH, 'c_size': c_size, 'k_size': k_size,
                        'epochs': epochs, 'root_path': njoin(models_path,f'cnn{DEPTH}_seed={seed}')})

    kwargss = add_common_kwargs(kwargss, common_kwargs)
    kwargss_all += kwargss

    return kwargss_all, script_name, script_func, q, ncpus, ngpus, select, walltime, mem, job_path, nstack


# multifractal analysis for Jacobian eigenvectors of pretrained MLPs
def mlp_jac_analysis():
    from UTILS.mutils import point_to_path

    # script and function
    script_name = 'pretrained_dq.py'
    script_func = 'jac_dq'

    # arguments
    seeds_root = njoin(DROOT, 'fc10_sgd_mnist')
    job_path = njoin(seeds_root, 'jac_jobs')

    seeds = list(range(5))
    # seeds = list(range(1))
    alpha100s, g100s = [120, 200], [20, 100, 300]
    navg = 10
    post, reig = 0, 1
    net_paths = [point_to_path(seeds_root, alpha100, g100, seed)\
                  for alpha100, g100, seed in product(alpha100s, g100s, seeds)]
    epochs = [0, 100]
    
    # resources
    is_use_gpu = False
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]   
    nstack = 10                              
    single_walltime = '00:15:59'  # gpu  50 epochs
    walltime = time_to_str(str_to_time(single_walltime) * nstack)   
    mem = '8GB'   
    select = 1

    kwargss_all = []                  
    kwargss = []      
    for net_path, epoch in product(net_paths, epochs):
        kwargss.append({'net_path': net_path, 'navg': navg, 'epoch': epoch,
                        'post': post, 'reig': reig})

    # kwargss = add_common_kwargs(kwargss, common_kwargs)
    kwargss_all += kwargss

    return kwargss_all, script_name, script_func, q, ncpus, ngpus, select, walltime, mem, job_path, nstack
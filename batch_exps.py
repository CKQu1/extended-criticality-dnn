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

def train_mlps():
    pass


# train vanilla cnns
def train_cnns():

    script_name = 'tf_train_v3.py run_model'

    C_SIZE = 100  # channel size
    K_SIZE = 3    # kernel size
    LEARNING_RATE = 1e-2
    MOMENTUM = 0
    BATCH_SIZE = 1024 

    # alpha100s = list(range(100,201,10))
    # g100s = list(range(25, 301, 25))
    alpha100s = [100,200]
    g100s = [25,100,300]
    epochs = 50
    seeds = [0]
    
    DEPTH = 4
    net_type = 'cnn_cpad'  # cnn with circular pad
    fc_init = "fc_default"
    dataset = 'mnist'
    optimizer = 'sgd'

    # resources
    is_use_gpu = True
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]   
    nstack = 1                              
    single_walltime = '00:20:59'
    walltime = time_to_str(str_to_time(single_walltime) * nstack)   
    mem = '4GB'   
    select = 1

    # set up dir
    models_path = njoin(DROOT, 'trained_cnns', f'cnn{DEPTH}_{fc_init}_{dataset}_{optimizer}_epochs={epochs}')
    job_path = njoin(models_path, 'jobs_all')
    if not isdir(models_path): makedirs(models_path)
    # save shared training settings
    df_settings = pd.DataFrame(columns=['c_size', 'k_size', 'lr', 'momentum', 'batch_size'])
    df_settings.loc[0,:] = [C_SIZE, K_SIZE, LEARNING_RATE, MOMENTUM, BATCH_SIZE]
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
        kwargss.append({'alpha100': alpha100, 'g100': g100, 'seed': seed})

    kwargss = add_common_kwargs(kwargss, common_kwargs)
    kwargss_all += kwargss

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack
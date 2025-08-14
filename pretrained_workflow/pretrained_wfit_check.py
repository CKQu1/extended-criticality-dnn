import numpy as np
import os
import pandas as pd
import time
import sys
from ast import literal_eval
from os.path import join, isfile, isdir
from tqdm import tqdm

# lib_path = os.getcwd()
# sys.path.append(f'{lib_path}')
# from path_names import root_data
root_data = '/project/PDLAI/project2_data'

def get_name(weight_name):    
    return '_'.join(weight_name.split("_")[:-3])

def replace_name(weight_name,other):
    assert isinstance(other,str)
    ls = weight_name.split("_")
    ls[-3] = other
    #ls += other
    return '_'.join(ls)

"""
python -i pretrained_workflow/pretrained_wfit_check.py plfit_method_check\
 /project/PDLAI/project2_data/pretrained_workflow/weights_all\
 /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_v2/ 6060
"""
def plfit_method_check(weight_path, save_dir, n_weight):
                        #**kwargs):  # ,remove_weights=False
    """
    Fit tails of each weight matrix to powerlaw (adopted from WeightWatcher):
        - weigh_path (str): path of stored weights (/project/PDLAI/project2_data/pretrained_workflow/weights_all)
        - save_dir (str): path of saved fitted parameters 
            1. /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_all
            2. /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_v2
        - n_model (int): the index of the model from get_pretrained_names()
    """
    import powerlaw
    import sys         

    lib_path = os.getcwd()
    sys.path.append(f'{lib_path}')

    from weightwatcher.constants import TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL, EXPONENTIAL
    from weightwatcher.WW_powerlaw import fit_powerlaw  

    # global model_path, df_pl, plfit, df_path, col_names, col_names_all, fit_types
    # global alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, Rs_all, ps_all
    # global bestfit, unfitted_dists, weights
    # global dist_compare, fit_compare, fit_type, ps_all, FIT
    # global alphas, Ds, xmins, lower, upper

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    #fit_type = kwargs.get('fit_type', 'power_law')
    #fit_type = POWER_LAW

    fit_types = [POWER_LAW]
    #fit_types = [POWER_LAW, TRUNCATED_POWER_LAW]

# Loading weight matrix ----------------------
    
    all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL, EXPONENTIAL]    

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    layer_idx, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)

    if not os.path.exists(model_path): os.makedirs(model_path)
    print(f"{model_path} directory set up!")
    print("\n")

# Setting up dir ----------------------

    data_name = replace_name(weight_name,'plfit')
    df_path = f'{model_path}/{data_name}.csv'

# Powerlaw fitting ----------------------

    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")
    w_size = len(weights)

    # values significantly smaller than zero are filtered out 
    #weights = np.abs(weights)
    """
    if remove_weights:
        weights = weights[np.abs(weights) >0.00001]    

    # 1. split into upper and lower tail
        #weights_upper = weights[weights > weights.mean()]
        weights_upper = np.abs(weights[weights > weights.mean()])
        weights_lower = np.abs(weights[weights < weights.mean()])

    else:
        weights_upper = np.abs(weights[weights >= 0])
        weights_lower = np.abs(weights[weights <= 0])
    """

    upper = np.abs(weights[weights>=0])
    lower = np.abs(weights[weights<=0])    

    # added for speed up and PL fit does not deal well with small values
    upper = upper[upper > 0.00001]
    lower = lower[lower > 0.00001]        

    # for entries exceeding certain threshold, start from the 50 percentile
    pctage = 75
    size_threshold = 5e6
    if len(upper) >= size_threshold:
        upper = upper[upper > np.percentile(upper, pctage)]
        upper_triggered = True
    else:
        upper_triggered = False
    if len(lower) >= size_threshold:
        lower = lower[lower > np.percentile(lower, pctage)]  
        lower_triggered = True     
    else:
        lower_triggered = False

    if isfile(df_path):
        df_pl = pd.read_csv(df_path)

    # --------------------
    tail_types = ['lower', 'upper']
    for idx, data in enumerate([lower, upper]):

        N = len(data)

        xmins = []
        alphas = []
        Ds = []
        #for i in tqdm(range(0, len(lower), 10000)):
        #for i in tqdm(range(0, len(data), 1000000)):
        for i in tqdm(range(0, len(data), 10000)):

            xmin = data[i]
            xmins.append(xmin)
            n = float(N - i)
            alpha = 1 + n / (np.sum(np.log(data)[i:]) - n * np.log(data)[i])
            alphas.append(alpha)
            #if alpha > 1:
            D = np.max(np.abs( 1 - (data[i:] / xmin) ** (-alpha + 1) - np.arange(n) / n ))     
            Ds.append(D)      

        Ds = np.array(Ds)
        alphas = np.array(alphas)
        xmins = np.array(xmins)

        if isfile(df_path):
            print(f'Tail {tail_types[idx]}')
            alpha, xmin, xmax, D = df_pl.loc[idx,['alpha', 'xmin', 'xmax', 'D']]
            print(f'alpha = {alpha}, xmin = {xmin}, xmax = {xmax}, D = {D}, fitted data = {len(data[data>=xmin])}, entries = {len(data)}')
        print('Refitted')
        i_min = Ds.argmin()
        print(f'alpha = {alphas[i_min]}, xmin = {xmins[i_min]}, xmax = {data.max()} D = {Ds[i_min]}, fitted data = {len(data[data>=xmins[i_min]])}, entries = {len(data)}')

        print('\n')
     

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
from path_names import root_data
from qsub import qsub, job_divider, project_ls, command_setup
from path_names import singularity_path, bind_path    

# funcs: plfit_method_check
def submit(*args):
    global df, pbs_array_data, n_weights, pbss, df_tail

    pytorch = True
    main_path = join(root_data, "pretrained_workflow")
    root_path = join(main_path, 'weights_all')  
    if pytorch:
        save_dir = join(main_path, "ww_plfit_v2")
    else:
        save_dir = join(main_path, "ww_plfit_all_tf")

    pbs_array_data = []
    #n_weights = [6060, 6071, 6084, 6097, 6113, 6129, 6148, 6167]
    df_tail_file = "tail_dist_grouped.csv"
    grouped_stats_path = join(main_path, "grouped_stats_v2")
    df_tail = pd.read_csv(join(grouped_stats_path, df_tail_file), index_col=0)
    cond = (df_tail.loc[:,'warning_lower'].str.contains('pctage',na=False)==True)
    cond = cond | (df_tail.loc[:,'warning_upper'].str.contains('pctage',na=False)==True)    
    n_weights = list(df_tail[cond].index)

    #quit()
    for n_weight in n_weights:
        pbs_array_data.append( (root_path, save_dir, n_weight) )
        n_weights.append(n_weight)
 
    print(len(pbs_array_data))

    ncpus, ngpus = 1, 0
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)                
        
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    #quit()  # delete
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',
             pbs_array_true, 
             path=join(main_path,'jobs_all', args[0]),
             P=project_ls[pidx], 
             walltime='23:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             mem="6GB")          


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])

import numpy as np
import os
import pandas as pd
import sys
from os.path import join
from tqdm import tqdm

from constants import *
from WW_powerlaw import fit_powerlaw
from WW_powerlaw import pl_fit

#root_data = "/project/PDLAI/project2_data"

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from path_names import root_data

tail_names = ["upper", "lower"]
upper_fits = []
lower_fits = []
#pl_fits = []

#n_weights = [0]
n_weights = list(range(0,50))

#for n_weight in n_weights:
for n_weight in tqdm(n_weights):

    # ---------------------- Load weights ----------------------
    weight_path = "/project/PDLAI/project2_data/pretrained_workflow/np_weights_all"
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]    

    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")

    print(f"n_weight = {n_weight}: {weight_name}, size = {len(weights)}")

    # separate into lower and upper tail
    upper = np.abs(weights[weights>=0])
    lower = np.abs(weights[weights<=0])
    #del weights

    # ---------------------- Powerlaw fitting ----------------------

    # function
    """
    def fit_powerlaw(evals, xmin=None, xmax=None, total_is=1000, 
                    plot=True, layer_name="", layer_id=0, plot_id=0, \
                    sample=False, sample_size=None,  savedir=DEF_SAVE_DIR, savefig=True, \
                    thresh=EVALS_THRESH,\
                    fix_fingers=False, finger_thresh=DEFAULT_FINGER_THRESH, xmin_max=None, max_fingers=DEFAULT_MAX_FINGERS, \
                    fit_type=POWER_LAW, pl_package=WW_POWERLAW_PACKAGE):
    """

    #fit_type = TPL  # this doesn't even to even be an option (check pl_fit function in WW_powerlaw.py)
    #fit_type = TRUNCATED_POWER_LAW

    total_is = 1000    

    #savedir = join(DEF_SAVE_DIR, "model_name")

    for tail_name in tail_names:
    #for tail_name in tail_names[:1]:
        plot_id = f"{model_name}_{i}_{wmat_idx}_{tail_name}"

        # 1. pl_fit direct
        #plfit = pl_fit(locals()[tail_name])
        #pl_fits.append( plfit )          

        # 2. save plot
        if 'fit_type' not in locals().keys():
            plfit = fit_powerlaw(locals()[tail_name], total_is=total_is, plot_id=plot_id)  #,savedir=savedir
        else:
            plfit = fit_powerlaw(locals()[tail_name], total_is=total_is, plot_id=plot_id,
                                fit_type=fit_type)  #,savedir=savedir

        # 3. don't save plot
        #plfit = fit_powerlaw(locals()[tail_name], total_is=total_is, plot=False)
        
        locals()[f"{tail_name}_fits"].append( plfit )
        #print(f"{plot_id} tail best fit: {plfit[-1]} \n")
        print(f"{plot_id} tail best fit: {plfit[-3]} \n")
    

    # ---------------------- Fit compare ----------------------    

all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL, EXPONENTIAL]
# including exponential
best_fits_1 = np.zeros(2, len(n_weights), len(all_dists))
# excluding exponential
best_fits_2 = np.zeros(2, len(n_weights), len(all_dists[:-1]))

for n_weight in tqdm(n_weights):
    best_fits_1[0, n_weight, np.argmax(lower_fits[n_weight][-2])] += 1
    best_fits_1[1, n_weight, np.argmax(upper_fits[n_weight][-2])] += 1

    best_fits_2[0, n_weight, np.argmax(lower_fits[n_weight][-2][:-1])] += 1
    best_fits_2[1, n_weight, np.argmax(upper_fits[n_weight][-2][:-1])] += 1
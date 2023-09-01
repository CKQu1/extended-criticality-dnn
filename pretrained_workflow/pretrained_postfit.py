import numpy as np
import math
import os
import pandas as pd
import sys
from os.path import join
from tqdm import tqdm

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
from path_names import root_data

def compute_aic(N, k, LL):
    return -2 * LL + 2 * k

def compute_bic(N, k, LL):
    return -2 * LL + math.log(N) * k

# main_path:
# /project/PDLAI/project2_data/pretrained_workflow/allfit_all
# /project/PDLAI/project2_data/pretrained_workflow/allfit_all_tf
def postfit_stats(main_path, pytorch=True):
    global df, fitfiles, subdirs, df_summary, total_nn, total_wm
    global ll_summary, aic_summary, bic_summary
    global data_dict, ll_summary, aic_summary, bic_summary
    global LL_maxs, AIC_mins, BIC_mins, fit_sizes

    subdirs = [join(main_path, x) for x in next(os.walk(main_path))[1]]
    #subdirs = [join(main_path, x) for x in next(os.walk(main_path))[1] if "alexnet" in x]
    total_nn = len(subdirs)
    print(f"Total networks: {total_nn}")
    total_wm = 0

    LL_colname_map = {'stable': 'logl_stable', 'normal': 'logl_norm',
                      'tstudent': 'logl_t', 'lognorm': 'logl_lognorm'}
    param_shape_map = {'stable': 4, 'normal': 2,
                       'tstudent': 3, 'lognorm': 3}
    fit_sizes = pd.DataFrame({"fit_size": []})
    data_types = ['ll', "aic", "bic"]
    dist_types = list(LL_colname_map.keys())    

    data_dict = {}  # for storing all dataframes
    for data_type in data_types:
        content_dict = {}
        for dist_name in dist_types:
            content_dict[dist_name] = []
        data_dict[data_type] = pd.DataFrame(content_dict)
        del content_dict

    error_files = []
    # separating different networks
    for sidx, subdir in tqdm(enumerate(subdirs)):
        fitfiles = [join(subdir,fitfile) for fitfile in os.listdir(subdir) if fitfile[-4:]==".csv" ]    # and "_1_2" in fitfile 
        for fitfile in fitfiles:
            df = pd.read_csv(fitfile)
            fit_size = df.loc[0,'fit_size']
            # get LL first
            LLs = [ df.loc[0, LL_colname_map[dist_type] ] for dist_type in dist_types ]
            isnotnan = True
            for LL in LLs:
                isnotnan = isnotnan and (not (np.isnan(LL or np.isposinf(LL) or np.isneginf(LL) )))
            
            if isnotnan:
                data_dict['ll'].loc[total_wm] = LLs
                # compute AIC/BIC
                AICs = []; BICs = []
                for kidx, dist_type in enumerate(dist_types):
                    AICs.append( compute_aic(fit_size, param_shape_map[dist_types[kidx]], LLs[kidx]) )
                    BICs.append( compute_bic(fit_size, param_shape_map[dist_types[kidx]], LLs[kidx]) )

                data_dict['aic'].loc[total_wm] = AICs
                data_dict['bic'].loc[total_wm] = BICs

                # get fit_size
                fit_sizes.loc[total_wm] = fit_size

                total_wm += 1

            else:
                error_files.append( fitfile )

    print(f"Total weight matrices: {total_wm}")
    # summarize stats
    selected_dists = [data_dict['ll'].idxmax(axis=1), data_dict['aic'].idxmin(axis=1), data_dict['bic'].idxmin(axis=1)]
    ll_summary = {}; aic_summary = {}; bic_summary = {}
    summaries = [[] for i in range(len(selected_dists))]
    
    for didx, data_type in enumerate(data_types):
        summaries[didx] = {}
        for colname in data_dict['ll'].columns:
            idxs = selected_dists[didx] == colname
            summaries[didx][colname] = [idxs.sum()/total_wm, fit_sizes[idxs].mean().item()]

        # print results
        print(f"data_types[didx]")
        print(summaries[didx])
        print('\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from os.path import join
from tqdm import tqdm

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
from path_names import root_data
from pretrained_wfit import is_num_defined

def compute_aic(N, k, LL):
    return -2 * LL + 2 * k

def compute_bic(N, k, LL):
    return -2 * LL + math.log(N) * k

# convert to scientific notation in string
def num_to_scifi_str(num):
    return "{:.2e}".format(num)

# return (row, col) for plot
def get_plot_rc(num: int, ncols: int):
    return num // ncols, num % ncols

def postfit_stats(pytorch, with_outliers=True):
    global df, fitfiles, subdirs, df_summary, total_nn, total_wm
    global ll_summary, aic_summary, bic_summary
    global data_dict, ll_summary, aic_summary, bic_summary
    global LL_maxs, AIC_mins, BIC_mins, fit_sizes
    global error_files, dist_fit_sizes, fit_sizes, idxs

    """

    pytorch (variable)
    - 0: tensorflow
    - 1: pytorch
    - 2: both

    """

    pytorch = int(pytorch)
    assert pytorch in [0,1,2], "pytorch can only take values 0, 1 or 2"
    main_paths = ["/project/PDLAI/project2_data/pretrained_workflow/allfit_all", "/project/PDLAI/project2_data/pretrained_workflow/allfit_all_tf"]
    if pytorch <= 1:
        main_paths = [main_paths[pytorch]]
    subdirs = []
    for main_path in main_paths:
        for x in next(os.walk(main_path))[1]:
            subdirs.append(join(main_path, x))

    total_nn = len(subdirs)
    print(f"Total networks: {total_nn}")
    total_wm = 0

    LL_colname_map = {'stable': 'logl_stable', 'normal': 'logl_norm',
                      'tstudent': 'logl_t', 'lognorm': 'logl_lognorm'}
    ad_colname_map = {'stable': 'ad sig level stable', 'normal': 'ad sig level normal',
                      'tstudent': 'ad sig level tstudent', 'lognorm': 'ad sig level lognorm'}
    ks_colname_map = {'stable': 'ks pvalue stable', 'normal': 'ks pvalue normal',
                      'tstudent': 'ks pvalue tstudent', 'lognorm': 'ks pvalue lognorm'}                      

    param_shape_map = {'stable': 4, 'normal': 2,
                       'tstudent': 3, 'lognorm': 3}
    fit_sizes = pd.DataFrame({"fit_size": []})
    data_types = ['ll', "aic", "bic", "ks"]
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
                isnotnan = isnotnan and is_num_defined(LL)
            
            if isnotnan:
                # get AD and KS test pvalues
                #data_dict['ad'].loc[total_wm] = [ df.loc[0, ad_colname_map[dist_type] ] for dist_type in dist_types ]
                data_dict['ks'].loc[total_wm] = [ df.loc[0, ks_colname_map[dist_type] ] for dist_type in dist_types ]

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

    # box plots
    fig, axs = plt.subplots(2, 2, figsize=(17/3*2, 5.67*2))    

    # summarize stats
    selected_dists = [ data_dict['ll'].idxmax(axis=1), data_dict['aic'].idxmin(axis=1), data_dict['bic'].idxmin(axis=1),
                       data_dict['ks'].idxmax(axis=1) ]
    ll_summary = {}; aic_summary = {}; bic_summary = {}; ad_summary = {}; ks_summary = {}   
    summaries1 = [[] for i in range(len(selected_dists))]        
    for didx, data_type in enumerate(data_types):
        dist_fit_sizes = {}
        fit_sizes_legends = []
        summaries1[didx] = {}   
        row, col = get_plot_rc(didx, axs.shape[1])     
        for colname in data_dict['ll'].columns:
            idxs = selected_dists[didx] == colname
            percentage = idxs.sum()/total_wm
            summaries1[didx][colname] = [percentage]
            # distribution for fit_sizes            
            dist_fit_sizes[colname] = list(fit_sizes[idxs].iloc[:,0])
            fit_sizes_legends.append(str(round(percentage*100, 2)) + "%")

        if with_outliers:
            axs[row, col].boxplot(dist_fit_sizes.values())
        else:
            axs[row, col].boxplot(dist_fit_sizes.values(), showfliers=False) 
        axs[row, col].set_xticklabels(dist_fit_sizes.keys())        
        axs[row, col].set_title(data_type.upper() + ": " + ',  '.join(fit_sizes_legends))
        #axs[didx].set_ylim([0,1e8])

        # print results
        print(f"{data_types[didx]}")
        print(summaries1[didx])
        print('\n')

    axs[0,0].set_ylabel(f"Total weights: {total_wm}")

    save_path = join(root_data, "pretrained_workflow", "weight_summary_stats")
    if not os.path.isdir(save_path): os.makedirs(save_path)
    # save summary
    # ----- code here -----

    # save plot    
    plot_name = "fit_size_dist"
    plt.savefig(f"{save_path}/{plot_name}.pdf", bbox_inches='tight', format='pdf')  


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])

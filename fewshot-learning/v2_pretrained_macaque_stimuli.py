import matplotlib.pyplot as plt # delete

import numpy as np
import pandas as pd
import sys
import torch
from tqdm import tqdm
import os

from ast import literal_eval
from time import time
from os.path import join
lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
import path_names
from path_names import root_data

from pretrained_macaque_stimuli import load_shared_info, get_CI, transform_model_name

# for saving all the relevant analysis on pretrained nets
global pretrained_path, untrained_path
pretrained_path = join(root_data, "pretrained_workflow", "pretrained_dnns")
untrained_path = join(root_data, "pretrained_workflow", "untrained_dnns")

t0 = time()
dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")


# ---------------------- Extracting metrics version 2 ----------------------

def load_dict_v2():
    metric_dict = {"SNR"        : "SNRs_layerwise",
                   "error"      : "SNRs_layerwise",
                   "errstrue"   : "errstrue_layerwise",
                   "D"          : "Ds_layerwise",
                   "dist_norm"  : "dist_norm_layerwise",
                   "css"        : "css_layerwise",         
                   "bias"       : "biases_layerwise",
                   "R_100"      : "Rsmb_n_top=100_layerwise",
                   "R_50"       : "Rsmb_n_top=50_layerwise",
                   "hidden_d2"  : "hidden_d2_layerwise",
                   #"singvec_mean": "singvec_mean",
                   #"singvec_std": "singvec_std",
                   'r_singvec_mean': 'r_singvec_dq',
                   'l_singvec_mean': 'l_singvec_dq',
                   "r_singvec_principal": "r_singvec_dq",
                   "l_singvec_principal": "l_singvec_dq",
                   'r_fftsingvec_mean': 'r_fftsingvec_dq',
                   'l_fftsingvec_mean': 'l_fftsingvec_dq',
                   "r_fftsingvec_principal": "r_fftsingvec_dq",
                   "l_fftsingvec_principal": "l_fftsingvec_dq"                   
                   }

    name_dict = {"SNR"        : "SNR",
                 "error"      : "Error",
                 "errstrue"   : "Error (true)",
                 "D"          : 'Dimension',
                 "dist_norm"  : "Signal",
                 "css"        : "Signal-noise overlap",
                 "bias"       : "Bias",      
                 "R_100"      : 'Cumulative variance',
                 "R_50"       : 'Cumulative variance',
                 "d2_avg"     : r'Weighted $D_2$',
                 "hidden_d2"  : r'Hidden $D_2$',
                 #"singvec_mean": r'Singvec $D_2$ (mean)',        
                 #"singvec_principal": r'Singvec $D_2$ (max)',
                 'r_singvec_mean': 'Right singvec (mean)',
                 'l_singvec_mean': 'Left singvec (mean)',
                 'r_singvec_principal': 'Right singvec (max)',
                 'l_singvec_principal': 'Left singvec (max)',
                 'r_fftsingvec_mean': 'Right fft singvec (mean)',
                 'l_fftsingvec_mean': 'Left fft singvec (mean)',
                 'r_fftsingvec_principal': 'Right fft singvec (max)',
                 'l_fftsingvec_principal': 'Left fft singvec (max)'                                    
                 }   

    return metric_dict, name_dict

# load raw metrics from network
def load_raw_metric_v2(model_name, pretrained:bool, metric_name, **kwargs):
    #print(f'metric_name: {metric_name}')
    global metric_lidx, metric_data, df, weight_shape, has_weight

    from scipy.stats import norm

    if "m" in kwargs.keys():
        m = kwargs.get("m")

    metric_dict, _ = load_dict_v2()
    # check if metric_name includes "d2_"
    if "d2" in metric_name and '_d2' not in metric_name:
        dq_ls = metric_name.split("_") if "d2" in metric_name else []
        dq_filename = f"d2smb_n_top=100_layerwise" if len(dq_ls) > 0 else None
        #dq_filename = f"d2smb_n_top=50_layerwise" if len(dq_ls) > 0 else None
        metric_dict[metric_name] = dq_filename

    # model path
    init_path = join(root_data, "macaque_stimuli")
    pretrained = pretrained if isinstance(pretrained, bool) else literal_eval(pretrained)
    pretrained_str = "pretrained" if pretrained else "untrained"
    manifold_path = join(init_path, model_name, pretrained_str)    
    emb_path = manifold_path

    # load shared_info
    df = load_shared_info(model_name)    
    
    metric_lidx_shape = None
    metric_data = []
    n_singvec = 0; n_fftsingvec = 0
    for lidx in range(df.shape[0]):
        has_weight = df.loc[lidx,'has_weight']
        has_weight = bool(has_weight == np.bool_(True))
        if has_weight:
            weight_shape = literal_eval(df.loc[lidx,'weight_shape'])
        else:
            weight_shape = None

        #if lidx == 0:                                               # delete
        #    print(f"{metric_dict[metric_name]}_lidx={lidx}.npy")    # delete
        
        if "d2_" in metric_name:
            metric_file = join(emb_path, f"{metric_dict[metric_name]}_lidx={lidx}.npy")
        #if "d2_" in metric_name or metric_name == "d2":
            # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are                            
        #elif metric_name == "SNR" or metric_name == "error" or metric_name == "D":
        elif metric_name in ['SNR', 'error', 'errstrue']:
            if m == np.inf:
                metric_file = join(emb_path, f"{metric_dict[metric_name]}_lidx={lidx}_m=inf.npy")
            else:
                metric_file = join(emb_path, f"{metric_dict[metric_name]}_lidx={lidx}_m={m}.npy")    
                metric_file = join(emb_path, f'{metric_dict[metric_name]}_lidx={lidx}_m={m}.npy')  
        elif "_singvec_" in metric_name:
            if has_weight and len(weight_shape) > 1:                
                metric_file = join(emb_path, f"{metric_dict[metric_name]}_lidx={lidx}.npy")            
            else:
                metric_file = None
        elif "_fftsingvec_" in metric_name or "_rshpsingvec_" in metric_name:
            if has_weight and len(weight_shape) == 4:
                metric_file = join(emb_path, f"{metric_dict[metric_name]}_lidx={lidx}.npy") 
                #n_fftsingvec += 1
            else:  # continue to use _singvec_ for fully-connected layers
                metric_name_split = metric_name.split('_')
                metric_name_temp = metric_name_split[0] + '_' + 'singvec' + '_' + metric_name_split[-1]

                metric_file = join(emb_path, f"{metric_dict[metric_name_temp]}_lidx={lidx}.npy") 
        else:
            metric_file = join(emb_path, f"{metric_dict[metric_name]}_lidx={lidx}.npy") 


        if os.path.isfile( metric_file ):
            metric_lidx = np.load( metric_file )
            if metric_name == "error":
                metric_lidx = 1 - norm.cdf(metric_lidx)                    
        else:
            metric_lidx = None

        if metric_lidx_shape is None:
            if metric_lidx is not None:
                metric_lidx_shape = metric_lidx.shape
    
        metric_data.append(metric_lidx)


    if 'singvec' not in metric_name:
        dim_ls = [metric_data[lidx].shape if metric_data[lidx] is not None else metric_lidx_shape for lidx in range(len(metric_data))]
        is_same = all(dim == dim_ls[0] for dim in dim_ls)
        if not is_same:
            print(f'{model_name} ({pretrained}): {metric_name}')
            print(dim_ls)

        return np.stack( [metric_data[lidx] if metric_data[lidx] is not None else np.full(metric_lidx_shape, np.nan) for lidx in range(len(metric_data))] ).squeeze()
    else:
        return metric_data   


# load processed metrics from network
def load_processed_metric_v2(model_name, pretrained:bool, metric_name, **kwargs):
    #print(f'metric_name = {metric_name}')  # delete
    global metric_R, metric_D, metric_data, metric_og, batch_idx, l, Rs
    
    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained

    #if "n_top" in kwargs.keys():
    #    n_top = kwargs.get("n_top")
    if "m" in kwargs.keys():
        m = kwargs.get("m")
    if "avg" in kwargs.keys():
        avg = kwargs.get("avg")

    #if "d2_" not in metric_name and metric_name != "R":
    if metric_name == "SNR" or metric_name == "error" or metric_name == 'errstrue':
        metric_data = load_raw_metric_v2(model_name, pretrained, metric_name, m=m)
    # signal is dist_norm squared
    elif metric_name == "dist_norm":
        metric_data = load_raw_metric_v2(model_name, pretrained, metric_name)**2
    #cumulative variance explained by n_top PCs
    elif metric_name == "R_100":
        metric_R = load_raw_metric_v2(model_name, pretrained, metric_name)
        if avg:
            # (17, 32, 100)
            metric_data = metric_R[:,:,:n_top].cumsum(-1)/metric_R.sum(-1)[:,:,None]
        else:
            metric_data = metric_R[:,:,pc_idx]/metric_R.sum(-1)

    # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are     
    elif metric_name == "d2_avg":
        # D_2's
        metric_og = load_raw_metric_v2(model_name, pretrained, metric_name)
        # R's (eigenvalues/variance explained)     
        metric_R = load_raw_metric_v2(model_name, pretrained, "R_100")  
        metric_D = load_raw_metric_v2(model_name, pretrained, "D")  
        # storing weighted averaged of D_2's based on the eigenvalues/variance explained
        metric_data = np.zeros(metric_og[:,:,0].shape) 

        #print(f"metric_R: {metric_R.shape}")
        #print(f"metric_og: {metric_og.shape}")
        #print(f"metric_data: {metric_data.shape}")

        #print(f"Total layers {metric_data.shape[0]}")
        for l in range(metric_data.shape[0]):              
            #n_top = round(np.nanmean(metric_D[l,:])) 
            # round to closest integer to ED of the layer

            # method 1 (original in manuscript, renormalized by fixed n_top as below)
            
            # n_top = round(np.ma.masked_invalid(metric_D[l:]).mean())    # average across pre-evaluated D
            # var_percentage = metric_R[l,:,:n_top]/metric_R[l,:,:n_top].sum(-1)[:,None]  # renormalized by n_top
            # metric_data[l,:] = ( metric_og[l,:,:n_top] * var_percentage ).sum(-1)
            # #print(f"n_top = {n_top}, total PCs = {metric_R[l].shape[-1]}, avg var explained = {metric_R[l,:,:n_top].sum(-1).mean(-1)}")
            

            # method 2 (original in manuscript, renormalized by all)     
                   
            # n_top = round(np.ma.masked_invalid(metric_D[l:]).mean())    # average across pre-evaluated D
            # var_percentage = metric_R[l,:,:]/metric_R[l,:,:].sum(-1)[:,None]  # renormalized by all
            # metric_data[l,:] = ( metric_og[l,:,:n_top] * var_percentage[:,:n_top] ).sum(-1)
            # #print(f"n_top = {n_top}, total PCs = {metric_R[l].shape[-1]}, avg var explained = {metric_R[l,:,:n_top].sum(-1).mean(-1)}")      


            # method 3 (no renormalization) 
            
            #var_percentage = metric_R[l,:,:]/metric_R[l,:,:].sum(-1)[:,None]  # renormalized by all
            #metric_data[l,:] = ( metric_og[l,:,:] * var_percentage ).sum(-1) 
            
            
            # method 4 (evaluate n_top for each batch)
            
            #print((metric_og.shape[1], metric_R.shape[1], metric_data.shape[1]))
            assert metric_og.shape[1] ==  metric_R.shape[1] and metric_data.shape[1] ==  metric_R.shape[1], "dimension inconsistent"
            for batch_idx in range(metric_og.shape[1]):                

                # renormalized by n_top
                Rs = metric_R[l,batch_idx]
                n_top = np.sum(Rs**2,axis=-1)**2 / np.sum(Rs**4, axis=-1)   # participation ratio for each batch
                n_top = round(n_top)
                if n_top == 0:
                    n_top = 1
                #print(f"n_top: {n_top}")      
                
                # type 1 (renormalized by fixed n_top)    
                     
                var_percentage = metric_R[l,batch_idx,:n_top]/metric_R[l,batch_idx,:n_top].sum(-1)  
                metric_data[l,batch_idx] += ( metric_og[l,batch_idx,:n_top] * var_percentage ).sum(-1)  
                

                # type 2 (renormalized by all)
                
                #var_percentage = metric_R[l,batch_idx,:]/metric_R[l,batch_idx,:].sum(-1)  
                #metric_data[l,batch_idx] += ( metric_og[l,batch_idx,:n_top] * var_percentage[:n_top] ).sum(-1)         
                

                # type (no renormalization)
                #var_percentage = metric_R[l,:,:]/metric_R[l,:,:].sum(-1)[:,None]  # renormalized by all
                #metric_data[l,:] = ( metric_og[l,:,:] * var_percentage ).sum(-1)   

    elif "singvec" in metric_name:        
        metric_data = load_raw_metric_v2(model_name, pretrained, metric_name)

    elif metric_name == "d2":
        metric_data = load_raw_metric(model_name, pretrained, metric_name)
    elif "d2_" in metric_name:
        pc_idx = int(metric_name.split("_")[-1])
        metric_og = load_raw_metric(model_name, pretrained, metric_name)   
        metric_data = metric_og[:,:,pc_idx]
    else:
        metric_data = load_raw_metric_v2(model_name, pretrained, metric_name)

    if model_name == "alexnet":
        conv_idxs = [0,3,6,8,10,15,16]
        if "singvec" not in metric_name:
            return metric_data[conv_idxs]
        else:
            return [metric_data[conv_idx] for conv_idx in conv_idxs]
    else:
        return metric_data   

    #return metric_data

# --------------------------------------------------  Plot ------------------------------------------------------

# general plot settings
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import pubplot as ppt
from scipy.stats import norm

tick_size = 18.5 * 0.8
label_size = 18.5 * 0.8
title_size = 18.5
axis_size = 18.5 * 0.8
legend_size = 14.1 * 0.8
# arrow head
prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",lw=2.0,
            shrinkA=0,shrinkB=0)
markers = ["o", "s", "v", "p", "*", "P", "H", "D", "d", "+", "x"]
lstyle_ls = ["-", "--"]
transparency_grid = 0.3
lstyle_grid = "--"

def get_model_names_v2(main):
    if not main:
        model_names = ["squeezenet1_1","wide_resnet101_2"]
    else:
        model_names = ["alexnet", "resnet101"]
    return model_names

# Main text plot (minibatch d2 version)
def snr_metric_plot_v2(main=True, n_top=100, display=False):
    global metric_data_all, metric_data, d2_data, metric_data_y, singvec_data, total_layers, d2_data_mean, selected_layers 

    # option 1
    #metric_names="d2_avg,D,SNR,error"
    #metric_names = "hidden_d2,D,SNR,error"
    # option 2
    #metric_names = "r_singvec_mean,D,SNR,error"
    # option 3
    #metric_names = "r_singvec_principal,D,SNR,error"
    # option 4
    #metric_names = "r_fftsingvec_mean,D,SNR,error"
    #metric_names = "l_fftsingvec_mean,D,SNR,error"
    # option 5
    #metric_names = "r_fftsingvec_principal,D,SNR,error"
    #metric_names = "l_fftsingvec_principal,D,SNR,error"    

    # color
    c_ls = list(mcl.TABLEAU_COLORS.keys())

    """
    Fig 1:
    Plots the a selected metric1 (based on metric_dict) vs layer and SNR vs layer,
    for dq, the pc_idx PC needs to be selected

    Fig 2:
    Plots the scatter plot between error and D_2
    """

    display = literal_eval(display) if isinstance(display, str) else display 
    n_top = int(n_top)    

    if main in [True, False, 'True', 'False']:
        main = literal_eval(main) if isinstance(main, str) else main
        # get available networks
        #all_models = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv")).loc[:,"model_name"]

        model_names = get_model_names_v2(main)
    else:
        if ',' in main:
            model_names = main.split(',')
        else:
            model_names = [main]

    # Plot settings
    fig_size = (11/2*3,7.142+2) # 2 by 3

    # load dict
    metric_dict, name_dict = load_dict_v2()

    #metric_names_options = ["d2_avg,D,SNR,error", "hidden_d2,D,SNR,error"]
    metric_names_options = ["hidden_d2,D,SNR,error"]
 
    for metric_names in metric_names_options: 
        if ',' in metric_names:  # publication figure
            metric_names = metric_names.split(',')
        else:
            metric_names = [metric_names]        
        print(f"metric list: {metric_names}")
        assert len(metric_names) == 4, f"There can only be 4 metrics, you have {len(metric_names)}!"    
        for metric_name in metric_names:
            if "d2_" in metric_name:
                if metric_name != "d2_avg":
                    pc_idx = int(metric_name.split("_")[1])
                name_dict[metric_name] = r'Weighted $D_2$' 

        # transparency list
        trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]

        # need to demonstrate for pretrained and random DNNs
        pretrained_ls = [True, False]
        #pretrained_ls = [True]

        # m-shot learning 
        #ms = np.arange(1,11)
        m_featured = 5
        ms = [m_featured,np.inf]

        d2_centers = np.zeros([len(model_names), len(pretrained_ls)])
        error_centers = np.zeros([len(ms), len(model_names), len(pretrained_ls)])

        plt.rc('font', **ppt.pub_font)
        plt.rcParams.update(ppt.plot_sizes(False))
        fig, axs = plt.subplots(2, 3,sharex = False,sharey=False,figsize=fig_size)
        axs = axs.flat
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            for nidx in range(len(model_names)):
                
                metric_data_all = {}
                model_name = model_names[nidx]

                # fractional layer
                #total_layers = load_raw_metric(model_name, pretrained, "d2").shape[0]            
                total_layers = load_processed_metric_v2(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the error/SNR and D_2 centers
                deep_layers = np.where(frac_layers >= 0.8)

                # true errors
                errstrue = load_processed_metric_v2(model_name, pretrained, "errstrue", m=m_featured)

                # --------------- Plot 1 (upper) ---------------

                # load all data            
                print("Load all data!")  # delete
                for metric_idx, metric_name in enumerate(metric_names):
                    print(f'metric_name = {metric_name}')  # delete
                    metric_data = load_processed_metric_v2(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
                    metric_data_all[metric_name] = metric_data
                    if "d2_" in metric_name:
                        d2_data = metric_data
                        # centers of D_2
                        d2_centers[nidx,pretrained_idx] = np.nanmean(d2_data[deep_layers,:].flatten())
                    if metric_name == "error":
                        name_dict[metric_name] = rf"{m_featured}-shot error"
                    if metric_name == "SNR":
                        name_dict[metric_name] = rf"SNR ({m_featured}-shot)"   

                for metric_idx, metric_name in enumerate(metric_names):
                    color = c_ls[metric_idx] if pretrained else "gray"

                    metric_data = metric_data_all[metric_name]
                    # get mean and std of metric
                    if 'singvec' not in metric_name:
                        if metric_data.ndim == 2:
                            metric_data_mean = np.ma.masked_invalid(metric_data).mean(-1)
                            metric_data_std = np.ma.masked_invalid(metric_data).std(-1)
                            lower, upper = get_CI(metric_data)
                        elif metric_data.ndim == 3:
                            metric_data_mean = np.ma.masked_invalid(metric_data).mean((1,2))
                            metric_data_std = np.ma.masked_invalid(metric_data).std((1,2))
                            lower, upper = get_CI(metric_data.reshape(-1, np.prod(metric_data.shape[1:])))
                    else:
                        singvec_data = metric_data
                        metric_data_mean = []   
                        metric_data_std = []                 
                        for metric_lidx in metric_data:
                            if metric_lidx is not None:
                                if 'mean' in metric_name:
                                    metric_data_mean.append(metric_lidx.mean())
                                    metric_data_std.append(metric_lidx.std())
                                elif 'principal' in metric_name:
                                    if metric_lidx.ndim == 1:
                                        metric_data_mean.append(metric_lidx[0].mean())
                                        #metric_data_std.append(metric_lidx[0].std()) 
                                        metric_data_std.append(None)
                                    else:
                                        metric_data_mean.append(metric_lidx[:,:,0].mean())
                                        metric_data_std.append(metric_lidx[:,:,0].std())                                
                            else:
                                metric_data_mean.append(None)
                                metric_data_std.append(None)               
                        
                    #print(metric_data.shape)
                    axs[metric_idx].plot(frac_layers, metric_data_mean, 
                                        c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)

                    # true error
                    if metric_name == 'error':                        
                        errstrue_mean = np.ma.masked_invalid(errstrue).mean((1,2))
                        axs[metric_idx].plot(frac_layers, errstrue_mean, 
                                             c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle='dashed')         

                        lower, upper = get_CI(errstrue.reshape(-1, np.prod(errstrue.shape[1:])))
                        axs[metric_idx].fill_between(frac_layers, lower, upper, 
                                                     color=color, alpha=0.2)                                                        


                    # 95% CI
                    if 'singvec' not in metric_name:                
                        axs[metric_idx].fill_between(frac_layers, lower, upper, 
                                            color=color, alpha=0.2)

                # --------------- Plot 2 (lower) ---------------

                # scatter plot
                metric_name_y = "error"
                d2_name = list(metric_data_all.keys())[0]
                d2_data = metric_data_all[d2_name]                
                for midx, m in enumerate(ms):
                    axis = axs[midx+4]                

                    metric_data_y = load_processed_metric_v2(model_name, pretrained, metric_name_y, m=m, n_top=n_top)
                    # centers of error
                    #error_centers[midx,nidx,pretrained_idx] = np.nanmean(metric_data_y[deep_layers,:,:].flatten())
                    metric_data_y = np.nanmean(metric_data_y, (1,2)) 
                    color = c_ls[metric_names.index(metric_name_y)] if pretrained else "gray"                

                    # plots all layers
                    if '_singvec_' in d2_name or '_fftsingvec_' in d2_name or '_rshpsingvec_' in d2_name:
                        if 'mean' in d2_name:
                            d2_data_mean = [d2_data[selected_layer].mean() for selected_layer in selected_layers[0]]
                        elif 'principal' in d2_name:
                            d2_data_mean = []
                            for selected_layer in selected_layers[0]:
                                #print(f'd2 shape: {d2_data[selected_layer].shape}')  # delete
                                if len(d2_data[selected_layer].shape) == 3:
                                    d2_data_mean.append( d2_data[selected_layer][:,:,0].mean() )
                                elif len(d2_data[selected_layer].shape) == 1:
                                    d2_data_mean.append( d2_data[selected_layer][0] )

                        d2_data_mean = np.array(d2_data_mean)
                    elif d2_name == 'hidden_d2':
                        d2_data_mean = np.ma.masked_invalid(d2_data).mean((1,2))
                    else:
                        d2_data_mean = d2_data.mean(-1)[selected_layers]

                    axis.scatter(d2_data_mean, metric_data_y[selected_layers], 
                                 c=color, marker=markers[nidx], alpha=trans_ls[nidx])                        

                    # plots all layers
                    axis.scatter(d2_data_mean, metric_data_y[selected_layers], 
                                        c=color, marker=markers[nidx], alpha=trans_ls[nidx])
                    # plots deep layers
                    #axis.scatter(d2_data.mean(-1)[deep_layers], metric_data_y[deep_layers], 
                    #                    c='k', marker='x', alpha=trans_ls[nidx])

                    # if pretrained_idx == len(pretrained_ls) - 1:
                    #     # plot centered arrows
                    #     color_arrow = c_ls[metric_names.index(metric_name_y)]

                    #     error_center = error_centers[midx,nidx,0]
                    #     d2_center = d2_centers[nidx,0]
                    #     # arrow head
                    #     prop['color'] = color_arrow; prop['alpha'] = trans_ls[nidx]

                    #     error_y = error_centers[midx,nidx,1]
                    #     axis.annotate("", xy=(d2_centers[nidx,1],error_y), xytext=(d2_center,error_center), arrowprops=prop)

                    #     # axis labels
                    #     #axs[midx+2].set_title(name_dict[metric_name_y] + rf" ($m$ = {m})", fontsize=title_size)
                    #     axis.set_xlabel(r"Weighted $D_2$", fontsize=label_size)
                    #     if m == np.inf:
                    #         axis.set_ylabel(rf"$\infty$-shot error", fontsize=label_size)
                    #     else:
                    #         axis.set_ylabel(rf"{m}-shot error", fontsize=label_size)

                    #axis.set_xlabel(r"Weighted $D_2$", fontsize=label_size)
                    axis.set_xlabel(name_dict[d2_name], fontsize=label_size)
                    if m == np.inf:
                        axis.set_ylabel(rf"$\infty$-shot error", fontsize=label_size)
                    else:
                        axis.set_ylabel(rf"{m}-shot error", fontsize=label_size)                        

        print(f"{model_name} plotted!")

        # --------------- Plot settings ---------------
        for ax_idx, ax in enumerate(axs):        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=transparency_grid,linestyle=lstyle_grid)
            if ax_idx < len(metric_names):
                ax.set_xlabel("Fractional depth", fontsize=label_size)
                ax.set_ylabel(name_dict[metric_names[ax_idx]], fontsize=label_size)
                #ax.set_title(name_dict[metric_names[ax_idx]], fontsize=title_size)

            # scientific notation
            #if ax_idx >= 0:
            #    ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
            # ticklabel size
            ax.xaxis.set_tick_params(labelsize=label_size)
            ax.yaxis.set_tick_params(labelsize=label_size)

            #if ax_idx in [4,5]:
            #    ax.set_xscale('log')

        # legends
        for nidx, model_name in enumerate(model_names):
            label = transform_model_name(model_name)
            label = label.replace("n","N")
            axs[0].plot([], [], c=c_ls[0], alpha=trans_ls[nidx], 
                        marker=markers[nidx], linestyle = 'None', label=label)

        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            label = pretrained_str
            axs[0].plot([],[],c="gray", linestyle=lstyle, label=label)

        # if main:
        #     #axs[0].set_ylim((0.7,0.95))
        #     axs[0].set_ylim((0.1,0.95))
        # else:
        #     #axs[0].set_ylim((0.6,1))
        #     axs[0].set_ylim((0.1,1))
        #axs[0].set_ylim((0.5, 0.95))
        axs[0].legend(frameon=False, ncol=2, loc="upper left", 
                      bbox_to_anchor=(-0.05, 1.2), fontsize=legend_size)

        # adjust vertical space
        plt.subplots_adjust(hspace=0.4)

        # --------------- Save figure ---------------
        if not display:
            if main in [True, False]:
                fig_path = join(root_data,"figure_ms/pretrained-fewshot-main") if main else join(root_data,"figure_ms/pretrained-fewshot-appendix")
            else:
                models_cat = "_".join(model_names)
                fig_path = join(root_data,f"figure_ms/pretrained-fewshot-v2")
            if not os.path.isdir(fig_path): os.makedirs(fig_path)
            net_cat = "_".join(model_names)
            fig_name = f"pretrained_m={m_featured}_metric={metric_names}_{net_cat}.pdf"
            print(f"Saved as {join(fig_path, fig_name)} \n")
            plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')
        else:
            plt.show()


# Appendix plot for other metrics corresponding to SNR, i.e. signal (dist_norm), bias, signal-to-noise overlap (css)
def extra_metric_plot_v2(main=True, n_top=100, display=False):
    metric_names="dist_norm,bias,css,d2_avg"

    # plot settings
    c_ls = list(mcl.TABLEAU_COLORS.keys())[4:]

    """

    Plots the extra metrics including: signal, bias, signal-noise-overlap, 
    and plots D_2 against all 3 metrics aforementioned.

    """

    if ',' in metric_names:  # publication figure
        metric_names = metric_names.split(',')
    else:
        metric_names = [metric_names]
    assert len(metric_names) == 4, f"There can only be 3 metrics, you have {len(metric_names)}!"
    #fig_size = (11/2*len(metric_names),(7.142+2)/2) # 2 by 3
    fig_size = (11/2*3,7.142+2)

    main = literal_eval(main) if isinstance(main, str) else main
    display = literal_eval(display) if isinstance(display, str) else display
    
    # load dict
    metric_dict, name_dict = load_dict_v2()
 
    # model names
    model_names = get_model_names_v2(main)
    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]

    # need to demonstrate for pretrained and random DNNs
    pretrained_ls = [False, True]

    # m-shot learning 
    ms = [5]

    for midx, m_featured in enumerate(tqdm(ms)):
        plt.rc('font', **ppt.pub_font)
        plt.rcParams.update(ppt.plot_sizes(False))
        fig, axs = plt.subplots(2, 3,sharex=False,sharey=False,figsize=fig_size)
        axs = axs.flat

        d2_centers = np.zeros([len(model_names), len(pretrained_ls)])
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            for nidx in range(len(model_names)):
            
                metric_data_all = {}
                model_name = model_names[nidx]
                # fractional layer
                total_layers = load_processed_metric_v2(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the phase centers
                deep_layers = np.where(frac_layers >= 0.8)

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    metric_data = load_processed_metric_v2(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
                    metric_data_all[metric_name] = metric_data
                    if "d2_" in metric_name:
                        d2_data = metric_data
                        # centers of D_2
                        d2_centers[nidx,pretrained_idx] = np.nanmean(d2_data[deep_layers,:].flatten()) 

                for metric_idx, metric_name in enumerate(metric_names[:3]):

                    # --------------- Plot 1 ---------------
                    color = c_ls[metric_idx] if pretrained else "gray"

                    metric_data = metric_data_all[metric_name]
                    if metric_data.ndim == 2:
                        metric_data_mean = np.ma.masked_invalid(metric_data).mean(-1)
                        lower, upper = get_CI(metric_data)
                    else:
                        metric_data_mean = np.ma.masked_invalid(metric_data).mean((1,2))
                        lower, upper = get_CI(metric_data.reshape(-1, np.prod(metric_data.shape[1:])))
                    axs[metric_idx].plot(frac_layers, metric_data_mean, 
                                         c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)
                    # 95% CI
                    axs[metric_idx].fill_between(frac_layers, lower, upper, 
                                         color=color, alpha=0.2)

                    # --------------- Plot 2 ---------------
                    # all layers
                    axs[metric_idx+3].scatter(d2_data.mean(-1)[selected_layers], metric_data_mean[selected_layers], 
                                        c=color, marker=markers[nidx], alpha=trans_ls[nidx])
                    # deep layers
                    #axs[metric_idx+3].scatter(d2_data.mean(-1)[deep_layers], metric_data_mean[deep_layers], 
                    #                    c='k', marker='x', alpha=trans_ls[nidx])
        
        for ax_idx, ax in enumerate(axs):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if ax_idx < 3:        
                ax.set_xlabel("Fractional depth",fontsize=label_size)
                ax.set_ylabel(name_dict[metric_names[ax_idx]],fontsize=label_size)
                #ax.set_title(name_dict[metric_names[ax_idx]],fontsize=title_size)
            else:
                ax.set_xlabel(r"Weighted $D_2$")
                ax.set_ylabel(name_dict[metric_names[ax_idx-3]],fontsize=label_size)
            #if ax_idx > 0:
            #    ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
            # make space for legend
            #if ax_idx == 0:
            #    ax.set_ylim(0.17,0.242)
            # setting the bias and signal-noise-overlap to log scale
            #if ax_idx in [1,2,4,5]: 
            #    ax.set_yscale('log')
            #if ax_idx in [3,4,5]: 
            #    ax.set_xscale('log')
            ax.grid(alpha=transparency_grid,linestyle=lstyle_grid)

        # legends
        for nidx, model_name in enumerate(model_names):
            label = transform_model_name(model_name)
            label = label.replace("n","N")
            axs[0].plot([], [], c=c_ls[0], alpha=trans_ls[nidx], 
                        marker=markers[nidx], linestyle = 'None', label=label)

        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            label = pretrained_str
            axs[0].plot([],[],c="gray", linestyle=lstyle, label=label)

        axs[0].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.05, 1.2),
                      fontsize=legend_size)
        #ax2.set_yscale('log')
        #ax2.ticklabel_format(style="sci", scilimits=(0,1), axis="y" )

        #plt.show()

        # adjust vertical space
        plt.subplots_adjust(hspace=0.4)

        if display:
            plt.show()
        else:
            # --------------- Save figure ---------------
            fig_path = join(root_data,"figure_ms/pretrained-fewshot-main") if main else join(root_data,"figure_ms/pretrained-fewshot-appendix")
            if not os.path.isdir(fig_path): os.makedirs(fig_path)    
            net_cat = "_".join(model_names)
            fig_name = f"pretrained_m={m_featured}_extra_metrics_{net_cat}.pdf"
            print(f"Saved as {join(fig_path, fig_name)}")
            plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')


def snr_delta_plot_v2(main=True, n_top=100):
    metric_names="d2_avg,D,SNR,error"


    global metric_data_all
    # plot settings
    c_ls = list(mcl.TABLEAU_COLORS.keys())

    """
    Fig 1:
    Plots the selected metric1 (based on metric_dict) vs layer and SNR vs layer,
    for dq, the pc_idx PC needs to be selected

    Fig 2:
    Plots the scatter plot between error and D_2
    """

    main = literal_eval(main) if isinstance(main, str) else main
    n_top = int(n_top)

    # Plot settings
    fig_size = (11/2*3,7.142+2) # 2 by 3

    # load dict
    metric_dict, name_dict = load_dict_v2()

    if ',' in metric_names:  # publication figure
        metric_names = metric_names.split(',')
    else:
        metric_names = [metric_names]

    assert len(metric_names) == 4, f"There can only be 4 metrics, you have {len(metric_names)}!"
    print(f"metric list: {metric_names}")
    for metric_name in metric_names:
        if "d2_" in metric_name:
            if metric_name != "d2_avg":
                pc_idx = int(metric_name.split("_")[1])
            name_dict[metric_name] = r'$D_2$' 

    # model names
    model_names = get_model_names_v2(main)
    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]
    # need to demonstrate for pretrained and random DNNs
    pretrained_ls = [False, True]
    #pretrained_ls = [True]

    # m-shot learning 
    m_featured = 5
    ms = [m_featured,np.inf]

    d2_centers = np.zeros([len(model_names), len(pretrained_ls)])
    error_centers = np.zeros([len(ms), len(model_names), len(pretrained_ls)])

    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))
    fig, axs = plt.subplots(2, 3,sharex = False,sharey=False,figsize=fig_size)
    axs = axs.flat

    for nidx in range(len(model_names)):
        model_name = model_names[nidx]
        metric_data_all = [{}, {}]
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]

            # fractional layer
            total_layers = load_processed_metric_v2(model_name, pretrained, "dist_norm").shape[0]
            frac_layers = np.arange(0,total_layers)/(total_layers-1)
            # only scatter plot the selected layers (can modify)
            selected_layers = np.where(frac_layers >= 0)
            # for the error/SNR and D_2 centers
            deep_layers = np.where(frac_layers >= 0.8)

            # --------------- Plot 1 (upper) ---------------

            # load all data
            for metric_idx, metric_name in enumerate(metric_names):
                metric_data = load_processed_metric_v2(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
                metric_data_all[pretrained_idx][metric_name] = metric_data
                if "d2_" in metric_name:
                    d2_data = metric_data
                    # centers of D_2
                    d2_centers[nidx,pretrained_idx] = np.nanmean(d2_data[deep_layers,:].flatten())
                if metric_name == "error":
                    name_dict[metric_name] = rf"{m_featured}-shot error"
                if metric_name == "SNR":
                    name_dict[metric_name] = rf"SNR ({m_featured}-shot)"   

            for metric_idx, metric_name in enumerate(metric_names):
                color = c_ls[metric_idx] if pretrained else "gray"

                metric_data = metric_data_all[pretrained_idx][metric_name]
                # get mean and std of metric
                if metric_data.ndim == 2:
                    metric_data_mean = np.ma.masked_invalid(metric_data).mean(-1)
                    metric_data_std = np.ma.masked_invalid(metric_data).std(-1)
                    lower, upper = get_CI(metric_data)
                elif metric_data.ndim == 3:
                    metric_data_mean = np.ma.masked_invalid(metric_data).mean((1,2))
                    metric_data_std = np.ma.masked_invalid(metric_data).std((1,2))
                    lower, upper = get_CI(metric_data.reshape(-1, np.prod(metric_data.shape[1:])))
                    
                #print(metric_data.shape)
                axs[metric_idx].plot(frac_layers, metric_data_mean, 
                                     c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)
                # 95% CI
                axs[metric_idx].fill_between(frac_layers, lower, upper, 
                                     color=color, alpha=0.2)

            # --------------- Plot 2 (lower) ---------------

            # scatter plot
            metric_name_y = "error"
            for midx, m in enumerate(ms):
                if pretrained_idx == len(pretrained_ls) - 1:
                    axis = axs[midx+4]                

                    metric_data_y = load_processed_metric_v2(model_name, pretrained, metric_name_y, m=m, n_top=n_top)
                    # centers of error
                    error_centers[midx,nidx,pretrained_idx] = np.nanmean(metric_data_y[deep_layers,:,:].flatten())
                    metric_data_y = np.nanmean(metric_data_y, (1,2)) 
                    color = c_ls[metric_names.index(metric_name_y)] if pretrained else "gray"

                    d2_delta = metric_data_all[1]["d2_avg"].mean(-1) - metric_data_all[0]["d2_avg"].mean(-1)
                    # plots all layers
                    axis.scatter(d2_delta[selected_layers], metric_data_y[selected_layers], 
                                 c=color, marker=markers[nidx], alpha=trans_ls[nidx])

                    # axis labels
                    axis.set_xlabel(r"$\Delta D_2$", fontsize=label_size)
                    if m == np.inf:
                        axis.set_ylabel(rf"$\infty$-shot error", fontsize=label_size)
                    else:
                        axis.set_ylabel(rf"{m}-shot error", fontsize=label_size)

    print(f"{model_name} plotted!")

    # --------------- Plot settings ---------------
    for ax_idx, ax in enumerate(axs):        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=transparency_grid,linestyle=lstyle_grid)
        if ax_idx < len(metric_names):
            ax.set_xlabel("Fractional depth", fontsize=label_size)
            ax.set_ylabel(name_dict[metric_names[ax_idx]], fontsize=label_size)
            #ax.set_title(name_dict[metric_names[ax_idx]], fontsize=title_size)

        # scientific notation
        #if ax_idx >= 0:
        #    ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
        # ticklabel size
        ax.xaxis.set_tick_params(labelsize=label_size)
        ax.yaxis.set_tick_params(labelsize=label_size)

        #if ax_idx in [4,5]:
        #    ax.set_xscale('log')

    # legends
    for nidx, model_name in enumerate(model_names):
        label = transform_model_name(model_name)
        label = label.replace("n","N")
        axs[0].plot([], [], c=c_ls[0], alpha=trans_ls[nidx], 
                    marker=markers[nidx], linestyle = 'None', label=label)

    for pretrained_idx, pretrained in enumerate(pretrained_ls):
        pretrained_str = "pretrained" if pretrained else "untrained"
        lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
        label = pretrained_str
        axs[0].plot([],[],c="gray", linestyle=lstyle, label=label)

    if main:
        axs[0].set_ylim((0.7,0.95))
    else:
        axs[0].set_ylim((0.6,1))    
    axs[0].legend(frameon=False, ncol=2, loc="upper left", 
                  bbox_to_anchor=(-0.05, 1.2), fontsize=legend_size)

    # adjust vertical space
    plt.subplots_adjust(hspace=0.4)

    # --------------- Save figure ---------------
    fig_path = join(root_data,"figure_ms/pretrained-fewshot-main") if main else join(root_data,"figure_ms/pretrained-fewshot-appendix")
    if not os.path.isdir(fig_path): os.makedirs(fig_path)
    net_cat = "_".join(model_names)
    fig_name = f"pretrained_m={m_featured}_delta_{net_cat}.pdf"
    print(f"Saved as {join(fig_path, fig_name)}")
    #plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')
    plt.show()


# Appendix plot for other metrics corresponding to SNR, i.e. signal (dist_norm), bias, signal-to-noise overlap (css)
def final_layers_v2(metric_names="hidden_d2,D,SNR,error,dist_norm,bias,css", n_top=100):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcl
    import pubplot as ppt

    # plot settings
    c_ls = list(mcl.TABLEAU_COLORS.keys())

    if ',' in metric_names:  # publication figure
        metric_names = metric_names.split(',')
    else:
        metric_names = [metric_names]
    assert len(metric_names) == 7, f"There can only be 3 metrics, you have {len(metric_names)}!"
    #fig_size = (11/2*len(metric_names),(7.142+2)/2) # 2 by 3
    fig_size = (11/2*3,7.142+2)

    # Plot settings
    
    # load dict
    metric_dict, name_dict = load_dict_v2()
 
    # "resnet152"    
    
    model_names = ["alexnet",   # trained
                   "resnet18", "resnet34", "resnet50", "resnet101",
                   "resnext50_32x4d", "resnext101_32x8d",
                   "squeezenet1_0", "squeezenet1_1",  
                   "wide_resnet50_2", "wide_resnet101_2"]   
    
    
    """
    model_names = ["alexnet",   # untrained
                   "resnet18", "resnet34", "resnet50", "resnet101",
                   "resnext50_32x4d", "resnext101_32x8d",
                   "squeezenet1_1",
                   "wide_resnet101_2"]    
    """

    #model_names = ["alexnet"]
    #model_names = ["resnet101"]
    #model_names = ["resnet18", "resnet34", "resnet50", "resnet101"]
    #model_names = ["resnext50_32x4d", "resnext101_32x8d"]
    #model_names = ["resnext50_32x4d"]
    #model_names = ["resnext101_32x8d"]
    #model_names = ["squeezenet1_0", "squeezenet1_1"]
    #model_names = ["squeezenet1_0"]
    #model_names = ["squeezenet1_1"]
    #model_names = ["wide_resnet50_2"]
    #model_names = ["wide_resnet101_2"]
    #model_names = ["mobilenet_v3_small"]

    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]

    # need to demonstrate for pretrained and random DNNs
    #pretrained_ls = [True, False]
    pretrained_ls = [True]
    #pretrained_ls = [False]
    lstyle_ls = ["-", "--"]

    # m-shot learning 
    ms = [5]

    for midx, m_featured in enumerate(tqdm(ms)):
        plt.rc('font', **ppt.pub_font)
        plt.rcParams.update(ppt.plot_sizes(False))
        fig, axs = plt.subplots(2, 3,sharex=False,sharey=False,figsize=fig_size)
        axs = axs.flat

        d2_centers = np.zeros([len(model_names), len(pretrained_ls)])
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            for nidx in range(len(model_names)):
            
                metric_data_all = {}
                model_name = model_names[nidx]
                # fractional layer
                total_layers = load_processed_metric_v2(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the phase centers
                deep_layers = np.where(frac_layers >= 0.8)

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    metric_data = load_processed_metric_v2(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
                    metric_data_all[metric_name] = metric_data
                    if "d2_" in metric_name or "_d2" in metric_name:
                        d2_data = metric_data
                        # centers of D_2
                        d2_centers[nidx,pretrained_idx] = np.nanmean(d2_data[deep_layers,:].flatten()) 

                for metric_idx, metric_name in enumerate(metric_names[1:]):
                    color = c_ls[metric_idx+1] if pretrained else "gray"

                    metric_data = metric_data_all[metric_name]
                    if metric_data.ndim == 2:
                        metric_data_mean = np.ma.masked_invalid(metric_data).mean(-1)
                    else:
                        metric_data_mean = np.ma.masked_invalid(metric_data).mean((1,2))

                    # last layer
                    
                    axs[metric_idx].scatter(d2_data.mean(-1)[-1], metric_data_mean[-1], 
                                            marker=markers[nidx], 
                                            c=color)    # alpha=trans_ls[nidx]
                    

                    # selected layers
                    """
                    depth_idxs = [-2,-1]
                    axs[metric_idx].scatter(d2_data.mean(-1)[depth_idxs], metric_data_mean[depth_idxs], 
                                            marker=markers[nidx], 
                                            c=color)    # alpha=trans_ls[nidx]                          
                    """

                    # last n layers   
                                    
                    # depth_idx = 0
                    # axs[metric_idx].scatter(d2_data.mean(-1)[depth_idx:], metric_data_mean[depth_idx:], 
                    #                         marker=markers[nidx], 
                    #                         c=color)    # alpha=trans_ls[nidx]   
                    
        
        for ax_idx, ax in enumerate(axs):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xlabel(r"$D_2$")
            ax.set_ylabel(name_dict[metric_names[ax_idx+1]],fontsize=label_size)
            #if ax_idx > 0:
            #    ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
            # make space for legend
            #if ax_idx == 0:
            #    ax.set_ylim(0.17,0.242)
            # setting the bias and signal-noise-overlap to log scale

            #ax.set_xscale('log')
            ax.grid(alpha=transparency_grid,linestyle=lstyle_grid)

        # legends
        for nidx, model_name in enumerate(model_names):
            label = transform_model_name(model_name)
            label = label.replace("n","N")
            axs[0].plot([], [], c=c_ls[0],  # alpha=trans_ls[nidx]
                        marker=markers[nidx], linestyle = 'None', label=label)

        # pretrained label
        """
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            label = pretrained_str
            axs[0].plot([],[],c="gray", linestyle=lstyle, label=label)
        """

        axs[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2),
                      fontsize=legend_size)
        #ax2.ticklabel_format(style="sci", scilimits=(0,1), axis="y" )

        #plt.show()

        # adjust vertical space
        plt.subplots_adjust(hspace=0.4)

        # --------------- Save figure ---------------
        fig_path = join(root_data,"figure_ms/pretrained-fewshot-main")
        if not os.path.isdir(fig_path): os.makedirs(fig_path)
        if len(model_names) == 1:
            fig_name = f"{model_names[0]}-m={m_featured}-features-layers.pdf"
        else:
            fig_name = f"pretrained-m={m_featured}-features-layers.pdf"
        print(f"Saved as {join(fig_path, fig_name)}")
        plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')
        #plt.show()


def aggregate_layers_v2(metric_names="d2_avg,D,SNR,error,dist_norm,bias,css", n_top=100):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcl
    import pubplot as ppt

    # plot settings
    c_ls = list(mcl.TABLEAU_COLORS.keys())

    if ',' in metric_names:  # publication figure
        metric_names = metric_names.split(',')
    else:
        metric_names = [metric_names]
    assert len(metric_names) == 7, f"There can only be 3 metrics, you have {len(metric_names)}!"
    #fig_size = (11/2*len(metric_names),(7.142+2)/2) # 2 by 3
    fig_size = (11/2*3,7.142+2)

    # Plot settings
    
    # load dict
    metric_dict, name_dict = load_dict_v2()
 
    # "resnet152"    
    
    model_names = ["alexnet",   # trained
                   "resnet18", "resnet34", "resnet50", "resnet101",
                   "resnext50_32x4d", "resnext101_32x8d",
                   "squeezenet1_0", "squeezenet1_1",  
                   "wide_resnet50_2", "wide_resnet101_2"]   
    
    
    """
    model_names = ["alexnet",   # untrained
                   "resnet18", "resnet34", "resnet50", "resnet101",
                   "resnext50_32x4d", "resnext101_32x8d"]    
    """

    #model_names = ["alexnet"]
    #model_names = ["resnet50"]
    #model_names = ["resnet18", "resnet34", "resnet50", "resnet101"]
    #model_names = ["resnext50_32x4d", "resnext101_32x8d"]
    #model_names = ["squeezenet1_0", "squeezenet1_1"]
    #model_names = ["wide_resnet50_2"]
    #model_names = ["wide_resnet101_2"]

    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]

    # need to demonstrate for pretrained and random DNNs
    #pretrained_ls = [True, False]
    pretrained_ls = [True]
    #pretrained_ls = [False]
    lstyle_ls = ["-", "--"]

    # m-shot learning 
    ms = [5]

    for midx, m_featured in enumerate(tqdm(ms)):
        plt.rc('font', **ppt.pub_font)
        plt.rcParams.update(ppt.plot_sizes(False))
        fig, axs = plt.subplots(2, 3,sharex=False,sharey=False,figsize=fig_size)
        axs = axs.flat

        d2_centers = np.zeros([len(model_names), len(pretrained_ls)])
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            for nidx in range(len(model_names)):
            
                metric_data_all = {}
                model_name = model_names[nidx]
                # fractional layer
                total_layers = load_processed_metric_v2(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the phase centers
                deep_layers = np.where(frac_layers >= 0.8)

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    metric_data = load_processed_metric_v2(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
                    metric_data_all[metric_name] = metric_data
                    if "d2_" in metric_name:
                        d2_data = metric_data
                        # centers of D_2
                        d2_centers[nidx,pretrained_idx] = np.nanmean(d2_data[deep_layers,:].flatten()) 

                for metric_idx, metric_name in enumerate(metric_names[1:]):
                    color = c_ls[metric_idx+1] if pretrained else "gray"

                    metric_data = metric_data_all[metric_name]
                    if metric_data.ndim == 2:
                        metric_data_mean = np.ma.masked_invalid(metric_data).mean(-1)
                    else:
                        metric_data_mean = np.ma.masked_invalid(metric_data).mean((1,2))

                    # last n layers                                       
                    depth_idx = 1
                    axs[metric_idx].scatter(d2_data.mean(-1)[depth_idx:].mean(), metric_data_mean[-1], 
                                            marker=markers[nidx], 
                                            c=color)
                        
        for ax_idx, ax in enumerate(axs):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xlabel(r"$D_2$")
            ax.set_ylabel(name_dict[metric_names[ax_idx+1]],fontsize=label_size)
            #if ax_idx > 0:
            #    ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
            # make space for legend
            #if ax_idx == 0:
            #    ax.set_ylim(0.17,0.242)
            # setting the bias and signal-noise-overlap to log scale

            #ax.set_xscale('log')
            ax.grid(alpha=transparency_grid,linestyle=lstyle_grid)

        # legends
        for nidx, model_name in enumerate(model_names):
            label = transform_model_name(model_name)
            label = label.replace("n","N")
            axs[0].plot([], [], c=c_ls[0],  # alpha=trans_ls[nidx]
                        marker=markers[nidx], linestyle = 'None', label=label)

        axs[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2),
                      fontsize=legend_size)

        # adjust vertical space
        plt.subplots_adjust(hspace=0.4)

        # --------------- Save figure ---------------
        fig_path = join(root_data,"figure_ms/pretrained-fewshot-main")
        if not os.path.isdir(fig_path): os.makedirs(fig_path)    
        fig_name = f"pretrained-m={m_featured}-features-aggregate_layers.pdf"
        print(f"Saved as {join(fig_path, fig_name)}")
        #plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')
        plt.show()


# microscopic statistics of the neural representations (perhaps leave out)
def snr_microscopic_plot_v2(small=True, log=False, display=True):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcl
    import pubplot as ppt
    c_ls = list(mcl.TABLEAU_COLORS.keys())
    tick_size = 18.5 * 0.8
    label_size = 18.5 * 0.8
    title_size = 18.5
    axis_size = 18.5 * 0.8
    legend_size = 14.1 * 0.9

    global dq_data, dq_name, metric_dict, name_dict
    global metric_data_all, metric_og, metric_name, var_cum, metric_names, metric_R

    small = literal_eval(small) if isinstance(small, str) else small
    display = literal_eval(display) if isinstance(small, str) else display
    log = literal_eval(log) if isinstance(log, str) else log

    # Plot settings
    #fig_size = (9.5 + 1.5,7.142/2) # 1 by 2
    #fig_size = (9.5 + 1.5,7.142) # 2 by 2
    fig_size = (11/2*3,7.142+2) # 2 by 3
    #markers = ["o", "v", "s", "p", "*", "P", "H", "D", "d", "+", "x"]
    markers = [None]*11
    transparency, lstyle = 0.4, "--"

    metric_names = ["d2", "R_100"]
    metric_dict, name_dict = load_dict_v2()
    
    print(f"metric list: {metric_names}")
    for metric_name in metric_names:
        if "d2" in metric_name:
            #if metric_name != "d2_avg":
            #    pc_idx = int(metric_name.split("_")[1])
            name_dict[metric_name] = r'$D_2$' 

    if small:
        # small models
        """
        model_names = ["alexnet", 
                  "resnet18", "resnet34", "resnet50",
                  "resnext50_32x4d",
                  "wide_resnet50_2"]
        """
        #model_names = ["resnet50"]
        model_names = ["alexnet", "resnet101"]

    else:
        # large models
        #model_names = ["resnet101", "resnet152", 
        #               "resnext101_32x8d",
        #               "squeezenet1_0", "squeezenet1_1", 
        #               "wide_resnet101_2"]
        model_names = ["resnet101", "squeezenet1_1"]

    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+2)[::-1]

    # need to demonstrate for pretrained and random DNNs
    pretrained_ls = [True, False]
    #pretrained_ls = [False]
    lstyle_ls = ["-", "--"]

    # m-shot learning 
    #ms = np.arange(1,11)
    ms = [1]

    # fractional layer
    total_layers = load_raw_metric(model_names[0], True, "d2").shape[0]
    frac_layers = np.arange(0,total_layers)/(total_layers-1)
    # only scatter plot the selecged layers (can modify)
    selected_layers = np.where(frac_layers >= 0)
    # for the SNR and D_2 centers
    deep_layers = np.where(frac_layers >= 0.5)

    for midx, m in enumerate(tqdm(ms)):
        plt.rc('font', **ppt.pub_font)
        plt.rcParams.update(ppt.plot_sizes(False))
        fig, axs = plt.subplots(2, 3,sharex = False,sharey=False,figsize=fig_size)
        axs = axs.flat
        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            for nidx in range(len(model_names)):
                
                color = c_ls[nidx] if pretrained else "gray"
                metric_data_all = {}
                model_name = model_names[nidx]

                # --------------- Plot 1 (upper) ---------------

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    if "d2" not in metric_name and metric_name != "R_100":
                        metric_data_all[metric_name] = load_metric(model_name, pretrained, metric_name, m)
                    #cumulative variance explained by n_top PCs
                    elif metric_name == "R_100":
                        metric_R = load_raw_metric(model_name, pretrained, metric_name)
                        # cumulative variance
                        metric_data_all[metric_name] = metric_R                        

                    # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are     
                    else:
                        metric_og = load_raw_metric(model_name, pretrained, metric_name)   
                        d2_data = metric_og
                        d2_name = metric_name
                        metric_data_all[metric_name] = d2_data

                l = -1  # final depth
                # plot D_2 for each PC
                metric_data = metric_data_all["d2"]
                axs[0].plot(list(range(1,metric_data.shape[2]+1)), metric_data[l,:,:].mean(0), 
                                     c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)

                metric_D = load_raw_metric(model_name, pretrained, "D")
                print(f"ED = {metric_D[l,:].mean(-1)}, {pretrained} shape {metric_D.shape}")

                axs[0].axvline(x=metric_D[l,:].mean(-1), c=color, alpha=trans_ls[nidx])

                # plot cumulative var
                metric_data = metric_data_all["R_100"]
                var_cum = metric_data[:,:,:].cumsum(-1)/metric_data.sum(-1)[:,:,None]
                axs[1].plot(list(range(1,var_cum.shape[2]+1)), var_cum[l,:,:].mean(0), 
                                     c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)

                # plot of ordered eigenvalues (variance)
                axs[2].plot(list(range(1,metric_data.shape[2]+1)), metric_data[l,:,:].mean(0), 
                                     c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)            
                if log:
                    axs[2].set_xscale('log'); axs[2].set_yscale('log')

                # --------------- Plot 2 (lower) ---------------

                if midx == len(ms)-1 and pretrained_idx == 0:
                    print(f"{model_name} saved!")

        # --------------- Plot settings ---------------
        for ax_idx, ax in enumerate(metric_names):  
            ax = axs[ax_idx]      
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=transparency_grid,linestyle=lstyle_grid)
            if ax_idx < 5:
                ax.set_xlabel("PC dimension", fontsize=label_size)
                #ax.set_ylabel(name_dict[metric_names[ax_idx]], fontsize=label_size)
                ax.set_title(name_dict[metric_names[ax_idx]], fontsize=title_size)
            if ax_idx >= 0:
                ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
            axs[2].set_title("Ordered eigenvalues", fontsize=title_size)
            # ticklabel size
            ax.xaxis.set_tick_params(labelsize=label_size)
            ax.yaxis.set_tick_params(labelsize=label_size)

        # legends
        for nidx, model_name in enumerate(model_names):
            label = model_name[0].upper() + model_name[1:]
            label = label.replace("n","N")
            axs[0].plot([], [], c=c_ls[nidx], alpha=trans_ls[nidx], marker=markers[nidx], label=label)

        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            label = pretrained_str
            axs[0].plot([],[],c="gray", linestyle=lstyle, label=label)

        axs[0].set_xlim((-5,105))
        axs[0].set_ylim((0.5,1.))

        axs[1].set_xlim((-5,105))
        axs[1].set_ylim((-0.1,1.1))
        axs[0].legend(frameon=False, ncol=2, loc="upper center", fontsize=legend_size)
        axs[1].ticklabel_format(style="sci", scilimits=(0,1), axis="y" )

        #axs[2].set_ylim(1e-1,1e3)

        # adjust vertical space
        plt.subplots_adjust(hspace=0.4)
        plt.show()

        if display:
            plt.show()
        else:
            # --------------- Save figure ---------------
            fig_path = join(root_data,"figure_ms/pretrained-fewshot-old-small") if small else join(root_data,"figure_ms/pretrained-fewshot-old-large")
            if not os.path.isdir(fig_path): os.makedirs(fig_path)    
            #plt.savefig(join(fig_path, f"pretrained_m={m}_microscopic.pdf") , bbox_inches='tight')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
import numpy as np
import math
import os
import pandas as pd
#import powerlaw as plaw
import random
import scipy.io as sio
import sys
import time
import torch

from ast import literal_eval
from numpy import dot
from os.path import join
from scipy.stats import levy_stable
#from scipy.stats import anderson
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions, norm, entropy
#from scipy.stats import kstest

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from path_names import root_data
 
#import net_load as nl
#from net_load.get_layer import get_hidden_layers, load_weights, get_epoch_weights, layer_struct

#import train_DNN_code.model_loader as model_loader
#from train_DNN_code.dataloader import get_data_loaders, get_synthetic_gaussian_data_loaders

global main_path, project_ls
#main_path = "/project/dnn_maths/project2_data/pretrained_workflow"
main_path = "/project/PDLAI/project2_data/pretrained_workflow"
project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]

t0 = time.time()

# ----------------------------

def log(save_path, file_name, **kwargs):
    fi = join(save_path, f"{file_name}.csv")
    df = pd.DataFrame(columns = kwargs)
    df.loc[0,:] = list(kwargs.values())
    if os.path.isfile(fi):
        df_og = pd.read_csv(fi)
        # outer join
        if "net_names" in file_name:
            if kwargs.get('model_name') not in df_og['model_name'].unique():
                df = pd.concat([df_og,df], axis=0, ignore_index=True)
            else:
                df = df_og
        elif "weight_info" in file_name:
            if kwargs.get('name') not in df_og[df_og['model_name'] == kwargs.get('model_name')]['name'].unique():
                df = pd.concat([df_og,df], axis=0, ignore_index=True)
            else:
                df = df_og            
    else:
        if not os.path.isdir(f"{save_path}"): os.makedirs(save_path)
    df.to_csv(fi, index=False)
    print(f'Log {fi} saved!')

####### SOME OF THE MODELS DON'T EXIST DUE TO TORCH VERSION 

def get_pretrained_names():
    import torchvision.models as models

    """
    Getting string names for all pretrained CNNs on Pytorch
    """

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
    import torchvision.models as models

    """
    Downloading all fully-connected weight matrices and 
    convolution tensors from Pytorch pretrained CNNs in get_pretrained_names()
    """

    t0 = time.time()

    model_ls = get_pretrained_names()
    model_name = model_ls[int(n_model)]
    if "vit" in model_name.lower() or "swin" in model_name.lower():
        print("Transformers are not considered!")
        quit()
    try:
        model = models.__dict__[model_name](pretrained=True)

        t1 = time.time()
        print(f"Loaded {model_name} in {t1 - t0} s")

        # path for saving the weights
        #main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
        weight_path = f"{main_path}/weights_all"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

        """
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
        """

        i = 0
        #for i in range(len(wmat_name_ls)):
        wmat_idx = 0
        for name, param in model.named_parameters():

            if 'bias' not in name and param.dim() > 1:
                weights = param.flatten()
                weight_file = f"{model_name}_layer_{i}_{wmat_idx}"

                if not os.path.isfile(f"{weight_path}/{model_name}_layer_{i}_{wmat_idx}"):
                    torch.save(weights, f"{weight_path}/{model_name}_layer_{i}_{wmat_idx}")
                    #weights = weights.detach().numpy()     

                    # create dataframe that stores the weight info for each NN
                    log(main_path, "weight_info", 
                        model_name=model_name, name=name, param_shape=list(param.shape),
                        weight_num=len(weights), wmat_idx=wmat_idx, idx=i,
                        weight_path=weight_path, weight_file=weight_file)     

                print(rf"W{i}: {wmat_idx} done!")
                i += 1
            wmat_idx += 1

        # create dataframe that stores the model_name
        log(main_path, "net_names_all", 
            model_name=model_name, total_wmat=wmat_idx, saved_wmat=i)

        # clear some space
        t_last = time.time()
        print(f"{model_name}: Ws of {i} stored in {t_last - t1} s!")      
    except (NotImplementedError,ValueError):    # versions of networks which either don't exist in current lib version or don't have pretrained version
        print(f"{model_name} not implemented!")

def pretrained_store_dnn(n_model, pretrained=True, *args):
    import torchvision.models as models

    """
    Downloading pretrained DNN from Pytorch in get_pretrained_names()
    """

    pretrained = pretrained if isinstance(pretrained,bool) else literal_eval(pretrained)

    t0 = time.time()

    model_ls = get_pretrained_names()
    model_name = model_ls[int(n_model)]
    try:
        model = models.__dict__[model_name](pretrained=pretrained)

        t1 = time.time()
        print(f"Loaded {model_name} in {t1 - t0} s")

        # path for saving the weights
        #main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
        net_path = join(main_path, "pretrained_dnns", model_name) if pretrained else join(main_path, "untrained_dnns", model_name)
        print(net_path)
        if not os.path.exists(net_path):
            os.makedirs(net_path)

        if not os.path.isfile(join(net_path, "model_pt")):
            torch.save(model, join(net_path, "model_pt"))

            # create dataframe that stores the model_name
            #log(main_path, "net_names_all", 
            #    model_name=model_name, total_wmat=wmat_idx, saved_wmat=i)

            # clear some space
            t_last = time.time()
            print(f"{model_name} saved under {net_path}: stored in {t_last - t1} s!")   

        else:
            print(f"Model already downloaded to {net_path}!")
   
    except (NotImplementedError,ValueError):    # versions of networks which either don't exist in current lib version or don't have pretrained version
        print(f"{model_name} not implemented!")

def get_pretrained_names_tf():
    import tensorflow.keras.applications as kapp
    #import tensorflow as tf

    """
    Getting string names for all pretrained CNNs on TensorFlow (non-overlapping ones with Pytorch)    
    """

    model_ls = [] 
    # we know the type of vgg
    class_type = type(kapp.VGG16)

    # use dir(kapp) to pre-explore the types of architecture
    architecture_ls = ['efficientnet', 'efficientnet_v2', 
                       'inception_resnet_v2', 'inception_v3',
                       'nasnet', 'xception']

    for arch_name in architecture_ls:
        arch = kapp.__dict__[arch_name]
        for model_name in dir(arch):
            arch_name_new = ''.join(arch_name.lower().split("_")).lower()   # specifically to deal with efficientnet_v2
            if arch_name_new in model_name.lower() and kapp.__dict__[model_name]:
                model_ls.append(model_name)

    return model_ls
      
def pretrained_store_tf(n_model, *args):
    import tensorflow.keras.applications as kapp

    """
    Downloading all fully-connected weight matrices and 
    convolution tensors from TensorFlow pretrained CNNs in get_pretrained_names_tf()
    """

    t0 = time.time()

    model_ls = get_pretrained_names()
    model_name = model_ls[int(n_model)]

    try:
        model_precursor = kapp.__dict__[model_name]
        #model = locals()["model_precursor"](pretrained=True)
        # random weights?
        model = locals()["model_precursor"]()
        # pretrained weights on Imagenet
        model = locals()["model_precursor"](weights='imagenet', include_top=True)

        t1 = time.time()
        print(f"Loaded {model_name} in {t1 - t0} s")

        # path for saving the weights
        #main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
        #main_path = os.getcwd()
        weight_path = f"{main_path}/weights_all_tf"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

        # create dataframe that stores the model_name
        """
        if not os.path.exists(f'{main_path}/net_names_all_tf.csv'):
            col_names = ["model_name"]
            df_names = pd.DataFrame(columns=col_names)
            df_names.loc[0] = model_name
        else:
            df_names = pd.read_csv(f'{main_path}/net_names_all_tf.csv')
            if model_name not in df_names.values:
                print(df_names.shape[0])
                df_names.loc[df_names.shape[0]] = model_name
        # save renewed version
        df_names.to_csv(f'{main_path}/net_names_all_tf.csv', index=False)
        """

        i = 0
        #for i in range(len(wmat_name_ls)):
        wmat_idx = 0
        names = []
        for widx in range(len(model.trainable_variables)):
            weight_name = model.trainable_variables[widx].name
            names.append(weight_name)
            if len(model.trainable_variables[widx]._shape) > 1:  
                weight_file = f"{model_name}_layer_{i}_{wmat_idx}"   
                weights = model.trainable_variables[widx]
                param_shape = weights.shape
                weights = torch.from_numpy(weights.numpy().flatten())
                if not os.path.isfile(f"{weight_path}/{model_name}_layer_{i}_{wmat_idx}":)
                    torch.save(weights, f"{weight_path}/{model_name}_layer_{i}_{wmat_idx}")

                    # create dataframe that stores the weight info for each NN
                    log(main_path, "weight_info_tf", 
                        model_name=model_name, name=weight_name, param_shape=list(param_shape),
                        weight_num=len(weights), wmat_idx=wmat_idx, idx=i,
                        weight_path=weight_path, weight_file=weight_file)   

                print(rf"W{i}: {wmat_idx}, dim: {model.trainable_variables[widx]._shape} done!")
                i += 1
            wmat_idx += 1

        # create dataframe that stores the model_name
        log(main_path, "net_names_all_tf", 
            model_name=model_name, total_wmat=wmat_idx, saved_wmat=i)

        # clear some space
        t_last = time.time()
        print(f"{model_name}: Ws of {i} stored in {t_last - t1} s!")  
        print("All weight names") 
        print(names)       
    except (NotImplementedError,ValueError):
        print(f"({model_name},n_model) not implemented!")
      
def pretrained_store_dnn_tf(n_model, pretrained=True, *args):

    """
    Downloading all full pretrained networks from TensorFlow pretrained CNNs in get_pretrained_names_tf()
    """

    t0 = time.time()

    model_ls = get_pretrained_names()
    model_name = model_ls[int(n_model)]

    try:
        model_precursor = kapp.__dict__[model_name]
        #model = locals()["model_precursor"](pretrained=True)
        # random weights?
        model = locals()["model_precursor"]()
        if pretrained:
            # pretrained weights on Imagenet
            model = locals()["model_precursor"](weights='imagenet', include_top=True)
        else:
            # random DNN
            model = locals()["model_precursor"](weights=None, include_top=True)

        t1 = time.time()
        print(f"Loaded {model_name} in {t1 - t0} s")

        # path for saving the network
        net_path = join(main_path, "pretrained_dnns_tf", model_name)
        if not os.path.exists(net_path):
            os.makedirs(net_path)

        if not os.path.isfile(join(net_path, "model_pt")):
            torch.save(model, join(net_path, "model_pt"))

            # create dataframe that stores the model_name
            #log(main_path, "net_names_all_tf", 
            #    model_name=model_name, total_wmat=wmat_idx, saved_wmat=i)

            # clear some space
            t_last = time.time()
            print(f"{model_name} saved under {net_path}: stored in {t_last - t1} s!")  
            #print("All weight names") 
            #print(names)             

        else:
            print(f"Model already downloaded to {net_path}!")            
      
    except (NotImplementedError,ValueError):
        print(f"({model_name},n_model) not implemented!")

def submit(*args):
    from qsub import qsub, job_divider, command_setup
    #N = len(get_pretrained_names())  # number of models
    N = 85
    pretrained_ls = [True, False]
    #pretrained_ls = [False]
    #pbs_array_data = [(f'{n_model:.1f}', pretrained)
    pbs_array_data = [(n_model, pretrained)
                      for n_model in range(N)
                      for pretrained in pretrained_ls
                      #for n_model in list(range(12))
                      #for n_model in list(range(2))
                      #for dummy in [0]
                      ]
    #qsub(f'python geometry_preplot.py {" ".join(args)}', pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact/', P='phys_DL')
    """
    qsub(f'python pretrained_workflow/pretrained_download.py {" ".join(args)}', 
         pbs_array_data, 
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow', 
         P='phys_DL', 
         mem="4GB")
    """

    ncpus, ngpus = 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    #pbs_array_data = pbs_array_data[:2]  # delete
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    #perm, pbss = job_divider(pbs_array_data, 1)
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path='/project/PDLAI/project2_data/pretrained_workflow/untrained_dnns', 
             P=project_ls[pidx], 
             ncpus=ncpus,
             ngpus=ngpus,
             mem="2GB")

def submit_tf(*args):
    from qsub import qsub
    N = len(get_pretrained_names_tf())  # number of models
    pbs_array_data = [(f'{n_model:.1f}')
                      for n_model in list(range(N))
                      #for n_model in list(range(12))
                      #for n_model in list(range(2))
                      #for dummy in [0]
                      ]
    #qsub(f'python geometry_preplot.py {" ".join(args)}', pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact/', P='phys_DL')
    """
    qsub(f'python pretrained_workflow/pretrained_download.py {" ".join(args)}', 
         pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow', 
         P='phys_DL', 
         mem="4GB")
    """

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path='/project/PDLAI/project2_data', 
             P=project_ls[pidx], 
             mem="4GB")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])




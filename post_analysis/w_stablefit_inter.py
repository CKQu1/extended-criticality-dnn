import numpy as np
from math import ceil
import scipy.io as sio
import scipy.stats as st
from scipy.stats import levy_stable
#import levy # https://pylevy.readthedocs.io/en/latest/levy.html#functions

import pandas as pd

import multiprocessing
from functools import partial

import time

# autonomous script for obtaining the stable parameters of weight matrix entries

#def w_stablefit(model_file, wm_total, epoch):
def w_stablefit_inter(step, model_file, interepoch_info):

    epoch_start, epoch_last, interepoch_steps = interepoch_info
    
    epoch_cur = epoch_start - 1 + ceil(step/interepoch_steps)
    
    file_step = step % interepoch_steps + (step % interepoch_steps == 0)*interepoch_steps;
    
    data= sio.loadmat("/project/cortical/Anomalous-diffusion-dynamics-of-SGD/trained_nets/" + model_file + f"/Epoch_{epoch_cur}/weights_{epoch_cur}_{file_step}.mat") ############################################### needs to be changed back
    #data= sio.loadmat("/project/cortical/Anomalous-diffusion-dynamics-of-SGD/trained_nets/" + model_file + f"/Epoch_{epoch_cur}/weights_{epoch_cur}_{file_step-1}.mat")
    w = data['sub_weights']
    
    wm_total = len(w[0])
    
    alphas = np.empty([wm_total,1]) # stability 
    betas = np.empty([wm_total,1])  # skewness
    deltas = np.empty([wm_total,1]) # location
    sigmas = np.empty([wm_total,1]) # scale
    #logls = np.empty([wm_total,1])  # negative log likelihood
    
    print('Step ' + str(step) + ' started')
    
    for w_num in range(wm_total):
        
        pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
        params = pconv(*levy_stable._fitstart(w[0][w_num].flatten()))
        
        alphas[w_num,0], betas[w_num,0], deltas[w_num,0], sigmas[w_num,0] = params
        
    print('Step ' + str(step) + ' done')

    return alphas, betas, sigmas, deltas

# function to extract the number of layers and epochs for fc networks only
def get_layer_epoch(model_file):
    # stripping the relevant information about the network
    model_file_str = model_file.split('_')
    for string in model_file_str:
        if 'epoch' in string:
            break

    return int(model_file_str[0][len('fc')::]), int(string[len('epoch')::])

def get_interepoch(model_file):
    
    epoch_dim = pd.read_csv("/project/cortical/Anomalous-diffusion-dynamics-of-SGD/trained_nets/" + model_file + "/epoch_dim.csv")
    epoch_start, epoch_last, interepoch_step = epoch_dim
    epoch_start, epoch_last, interepoch_step = int(epoch_start), int(epoch_last), int(interepoch_step)
    epoch_dim = [epoch_start, epoch_last, interepoch_step]

    return epoch_dim, (epoch_last - epoch_start + 1)*interepoch_step

def parallel_runs(model_ls):

    for model in model_ls: 

        t1 = time.time()

        wm_total, epoch_total = get_layer_epoch(model)
        epoch_dim, total_steps = get_interepoch(model)
        epoch_start, epoch_last, interepoch_step = epoch_dim
        
        step_list = list(range(1, total_steps + 1))  ############################################# this needs to be changed for some networks rip 
        #step_list = list(range(total_steps))
        
        print("Fitting the entries.")
        
        # need to the change the number of processors accordingly
        pool = multiprocessing.Pool(processes=4)
        w_fit = partial(w_stablefit_inter, model_file=model, interepoch_info=epoch_dim) ### might need to change this
        stable_params = pool.map(w_fit, step_list)
        
        print("Unpacking the parameters.")

        # unzip the output
        alphas, betas, sigmas, deltas= zip(*stable_params)
        alphas = np.concatenate(alphas, axis = 1)
        betas = np.concatenate(betas, axis = 1)
        sigmas = np.concatenate(sigmas, axis = 1)
        deltas = np.concatenate(deltas, axis = 1)
        
        # distribution fitting time
        t2 = time.time()
        elapsed = t2 - t1
        
        message = f"Epoch {epoch_start} to {epoch_last} for {model} done: {elapsed} seconds"
        print(message)

        post_dim = {'epoch_start': epoch_start, 'epoch_last': epoch_last, 'interepoch_step': interepoch_step, 'wm_total': wm_total, 'epoch_total': epoch_total}
        # save as .mat file 
        #mdict = {'alphas':alphas, 'betas': betas, 'sigmas': sigmas, 'deltas': deltas, 'logls': logls, 'model': model, 'post_dim': post_dim}
        #mdict = {'stable_params':stable_params, 'label': model_file}
        sio.savemat('/project/cortical/Anomalous-diffusion-dynamics-of-SGD/trained_nets/' + model + '/W_stable_params_inter_' + str(epoch_start) + '-' + 
                            str(epoch_last) + '.mat', 
                            mdict={'alphas':alphas, 'betas': betas, 'sigmas': sigmas, 'deltas': deltas,
                            'model': model, 'post_dim': post_dim},
                            )
        
        #print(model + epoch + "done in: " + str(elapsed) + " seconds")
        
        # saving time
        save_time = time.time() - t2
        print(f"Saved in {save_time}.")
    
# list for different models
# already done: 
# "fc5_mnist_tanh_epoch550_lr=0.1_bs=128_data_mnist", 
# "fc4_mnist_tanh_epoch650_lr=0.1_bs=1024_data_mnist",

#data_list = ["fc4_mnist_tanh_epoch650_lr=0.1_bs=1024_data_mnist", "fc3_mnist_tanh_epoch550_lr=0.1_bs=1024_data_mnist" 
#                , "fc10_mnist_tanh_idstable_epoch650_lr=0.001_bs=64_data_mnist"]

data_list = ["fc3_mnist_tanh_epoch550_lr=0.1_bs=1024_data_mnist"]

        
if __name__ == '__main__':

    parallel_runs(data_list)
    
    

import numpy as np
import scipy.io as sio
import scipy.stats as st
from scipy.stats import levy_stable
#import levy # https://pylevy.readthedocs.io/en/latest/levy.html#functions

import multiprocessing
from functools import partial

import time

# autonomous script for obtaining the stable parameters of weight matrix entries

#def w_stablefit(model_file, wm_total, epoch):
def w_stablefit(epoch, model_file):

    data= sio.loadmat("/project/cortical/Anomalous-diffusion-dynamics-of-SGD/trained_nets/" + model_file + "/model_" + str(epoch) + "_sub_loss_w.mat")
    w = data['sub_weights']
    
    wm_total = len(w[0])
    
    alphas = np.empty([wm_total,1]) # stability 
    betas = np.empty([wm_total,1])  # skewness
    deltas = np.empty([wm_total,1]) # location
    sigmas = np.empty([wm_total,1]) # scale
    #logls = np.empty([wm_total,1])  # negative log likelihood
    
    print('Epoch ' + str(epoch) + ' started')
    
    for w_num in range(wm_total):
        """
        dist_fit = levy.fit_levy(w[0][w_num].flatten())
        (params,logl) = dist_fit
        params_unpacked = params.get('B')
        alphas[w_num,0] = params_unpacked[0]
        betas[w_num,0] = params_unpacked[1]
        sigmas[w_num,0] = params_unpacked[2]
        deltas[w_num,0] = params_unpacked[3]
        logls[w_num,0] = logl
        """
        
        pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
        params = pconv(*levy_stable._fitstart(w[0][w_num].flatten()))
        
        alphas[w_num,0] = params[0]
        betas[w_num,0] = params[1]
        deltas[w_num,0] = params[2]
        sigmas[w_num,0] = params[3]
        
        #print(w_num)
        
    print('Epoch ' + str(epoch) + ' done')

    return alphas, betas, sigmas, deltas

# function to extract the number of layers and epochs for fc networks only
def get_layer_epoch(model_file):
    # stripping the relevant information about the network
    model_file_str = model_file.split('_')
    for string in model_file_str:
        if 'epoch' in string:
            break

    return int(model_file_str[0][len('fc')::]), int(string[len('epoch')::])

def parallel_runs(model_ls):

    for model in model_ls: 

        t1 = time.time()

        wm_total, epoch_total = get_layer_epoch(model)

        # uncomment for standard method
        """
        for epoch in range(epoch_total): # need to multiprocess this part
            # entry distribution analysis
            alphas, betas, sigmas, deltas, logls = w_stablefit(model_file, wm_total, epoch)
        """

        # multi-processing
        #data_list = list(range(0,epoch_total + 1))
        
        # epoch for starting the post analysis
        epoch_start = 1
        # epoch for ending the post analysis ######################################################## needs to be changed
        #epoch_last = 24 # just for testing
        epoch_last = epoch_total
        epoch_list = list(range(epoch_start,epoch_last+1)) 
        
        print("Fitting the entries.")
        
        # need to the change the number of processors accordingly
        pool = multiprocessing.Pool(processes=18)
        w_fit = partial(w_stablefit, model_file=model)
        stable_params = pool.map(w_fit, epoch_list)
        
        print("Unpacking the parameters.")

        # unzip the output
        alphas, betas, sigmas, deltas= zip(*stable_params)
        alphas = np.concatenate(alphas, axis = 1)
        betas = np.concatenate(betas, axis = 1)
        sigmas = np.concatenate(sigmas, axis = 1)
        deltas = np.concatenate(deltas, axis = 1)
        #logls = np.concatenate(logls, axis = 1)
        
        # distribution fitting time
        t2 = time.time()
        elapsed = t2 - t1
        
        message = f"Epoch {epoch_start} to {epoch_last} for {model} done: {elapsed} seconds"
        print(message)

        post_dim = {'epoch_start': epoch_start, 'epoch_last': epoch_last, 'wm_total': wm_total, 'epoch_total': epoch_total}
        # save as .mat file 
        #mdict = {'alphas':alphas, 'betas': betas, 'sigmas': sigmas, 'deltas': deltas, 'logls': logls, 'model': model, 'post_dim': post_dim}
        #mdict = {'stable_params':stable_params, 'label': model_file}
        sio.savemat('/project/cortical/Anomalous-diffusion-dynamics-of-SGD/trained_nets/' + model + '/W_stable_params_' + str(epoch_start) + '-' + 
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
# "fc10_mnist_tanh_epoch650_lr=0.001_bs=64_data_mnist"
# "fc4_mnist_tanh_epoch650_lr=0.1_bs=1024_data_mnist"
#data_list = ["fc4_mnist_tanh_epoch650_lr=0.1_bs=1024_data_mnist", "fc3_mnist_tanh_epoch550_lr=0.1_bs=1024_data_mnist" 
#                , "fc10_mnist_tanh_idstable_epoch650_lr=0.001_bs=64_data_mnist"]

data_list = ["fc7_mnist_tanh_epoch650_lr=0.1_bs=128_data_mnist"]
        
if __name__ == '__main__':

    parallel_runs(data_list)
    
    

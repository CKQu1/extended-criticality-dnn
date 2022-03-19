import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from tqdm import tqdm

plt.switch_backend('agg')

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

def IPR(vec, q):
    return sum(abs(vec)**(2*q)) / sum(abs(vec)**2)**q

# computes all the D^l W^l's matrix multiplication
def jac_layerwise(post, alpha100, g100, input_idx, epoch, *args):
    import scipy.io as sio
    import torch
    import train_DNN_code.model_loader as model_loader
    from net_load.get_layer import get_hidden_layers, load_weights, get_epoch_weights, layer_struct
    from nporch.input_loader import get_data_normalized    

    assert post == 1 or post == 0, "No such option!"

    # storage of trained nets
    post = int(post)
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
    data_path = f"{main_path}/fcn_grid/{fcn}_grid"

    # Extract numeric arguments.
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    input_idx = int(input_idx)
    # load MNIST (check trainednet_grad.py)
    image_type = 'mnist'
    trainloader , _, _ = get_data_normalized(image_type)      

    # load nets and weights
    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"
    w = get_epoch_weights(data_path, net_folder, epoch)
    net = model_loader.load(f'{net_type}')
    net = load_weights(net, w)    
    # push the image through
    image = trainloader.dataset[input_idx][0]
    num = trainloader.dataset[input_idx][1]
    print(torch.sum(image))
    print(num)

    if post == 1:
        print("Computing post-activation layerwise Jacobian")
    elif post == 0:
        print("Computing pre-activation layerwise Jacobian")

    return net.layerwise_jacob_ls(image, post)

def jac_save(post, alpha100, g100, input_idx, epoch, *args):
    import torch
    post = int(post)
    assert post == 1 or post == 0, "No such option!"
    if post == 1:
        path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/postjac_layerwise"
    elif post == 0:
        path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/prejac_layerwise"
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    DW_ls = jac_layerwise(post, alpha100, g100, input_idx, epoch, *args)
    print("Computation complete, saving data now!")
    DW_all = torch.ones((len(DW_ls), DW_ls[0].shape[0], DW_ls[0].shape[1]))
    for idx in range(0, len(DW_ls)):
        DW = DW_ls[idx]
        DW_all[idx,:DW.shape[0],:DW.shape[1]] = DW.detach().clone()

    print(DW_all.shape)
    torch.save(DW_all, f"{path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}")
    print(f"{path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}")

def submit(*args):
    from qsub import qsub
    pbs_array_data = [(alpha100, g100, input_idx, epoch)
                      for alpha100 in range(100, 201, 10)
                      for g100 in range(25, 301, 25)
                      #for alpha100 in [100, 200]
                      #for g100 in [100]
                      for input_idx in range(1)
                      for epoch in [0,1] + list(range(50,651,50))
                      #for epoch in [0, 100, 650]
                      ]
    #qsub(f'python geometry_analysis/greatcircle_proj_trial.py {sys.argv[0]} {" ".join(args)}',
    #qsub(f'python greatcircle_proj_trial.py {sys.argv[0]} {" ".join(args)}',
    qsub(f'python {sys.argv[0]} {" ".join(args)}',    
         pbs_array_data, 
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
         P='phys_DL')

# ----- preplot -----

def jac_to_dq(alpha100, g100, input_idx, epoch, *args):
    import numpy as np    
    import torch
    from numpy import linalg as la
    from tqdm import tqdm

    post, reig = 0, 1
    post = int(post)
    assert post == 1 or post == 0, "No such option!"
    reig = int(reig)    # whether to compute for right eigenvector or not
    assert reig == 1 or reig == 0, "No such option!"
    post_dict = {0:'pre', 1:'post'}
    reig_dict = {0:'l', 1:'r'}

    main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data"
    data_path = f"{main_path}/{post_dict[post]}jac_layerwise"
    save_path = f"{main_path}/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}"
    fig_path = f"{main_path}/dq_layerplot_{post_dict[post]}_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   
    if not os.path.exists(f'{fig_path}'):
        os.makedirs(f'{fig_path}')  

    # load DW's
    DW_all = torch.load(f"{data_path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}")
    DWs_shape = DW_all.shape
    # D_q init
    qs = np.linspace(0,2,50)

    for l in tqdm(range(DWs_shape[0])):
        D_qss = []

        DW = DW_all[l].numpy()
        if l == DWs_shape[0]:
            DW = DW[:10,:]
            DW = DW.T @ DW  # final DW is 10 x 784

        # left eigenvector
        if reig == 0:
            DW = DW.T

        print(f"layer {l}: {DW.shape}")
        eigvals, eigvecs = la.eig(DW)
        nan_count = 0
        for i in range(len(eigvals)):  
            eigvec = eigvecs[:,i]
            IPRs = np.array([IPR(eigvec ,q) for q in qs])
            D_qs = np.log(IPRs) / (1-qs) / np.log(len(eigvec))
            
            if not any(np.isnan(D_qs)): D_qss.append(D_qs)
            else: nan_count += 1  

        # save D_q values individually for layers
        """
        result = np.transpose([qs, np.mean(D_qss, axis=0), np.std(D_qss, axis=0)])
        outfname = f'dq_alpha{alpha100}_g{g100}_ep{epoch}_l{l}.txt'
        outpath = f"{save_path}/{outfname}"
        print(f'Saving {outpath}')
        np.savetxt(outpath, result)
        """
        if l == 0:
            dq_means = np.transpose([qs, np.mean(D_qss, axis=0)])
            dq_stds = np.transpose([qs, np.std(D_qss, axis=0)])
        else:
            dq_mean = np.mean(D_qss, axis=0)
            dq_std = np.std(D_qss, axis=0)
            dq_mean, dq_std = np.expand_dims(dq_mean,axis=1), np.expand_dims(dq_std,axis=1)
            dq_means = np.concatenate((dq_means, dq_mean),axis=1)
            dq_stds = np.concatenate((dq_stds, dq_std),axis=1)

        # plot D_q image
        plt.errorbar(qs, np.mean(D_qss, axis=0), yerr=np.std(D_qss, axis=0))
        plt.ylim([0,1.1])
        plt.xlim([0, 2])
        plt.ylabel('$D_q$')
        plt.xlabel('$q$')
        #plt.title(path.split('/')[-2:]+[str(nan_count)])
        outfname = f'dq_alpha{alpha100}_g{g100}_ipidx{input_idx}_ep{epoch}_l{l}.txt'
        plt.title([outfname]+[str(nan_count)])
        #plt.show()

        outfname = f'dq_alpha{alpha100}_g{g100}_ep{epoch}_l{l}.png'
        outpath = f"{fig_path}/{outfname}"
        print(f'Saving {outpath}')
        plt.savefig(outpath)
        plt.clf()

        print("\n")
        print(dq_means.shape)
        print(dq_stds.shape)

    # save D_q values
    # mean
    outfname = f'dqmean_alpha{alpha100}_g{g100}_ipidx{input_idx}_ep{epoch}.txt'
    outpath = f"{save_path}/{outfname}"
    print(f'Saving means: {outpath}')
    np.savetxt(outpath, dq_means)
    outfname = f'dqstd_alpha{alpha100}_g{g100}_ipidx{input_idx}_ep{epoch}.txt'
    outpath = f"{save_path}/{outfname}"
    print(f'Saving stds: {outpath}')
    np.savetxt(outpath, dq_stds)


def submit_preplot(*args):
#def submit_preplot(path):

    data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/jac_layerwise"
    # find the `alpha100`s and `g100`s of the files in the folder
    # dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}
    pbs_array_data = set([tuple(re.findall('\d+', fname)[:4]) for fname in os.listdir(data_path)
                      if all(s in fname for s in ('dw', 'alpha', 'g', 'ipidx', 'epoch'))])

    pbs_array_data = set([tuple(re.findall('\d+', fname)[:4]) for fname in os.listdir(data_path) if all(s in fname for s in ('dw', 'alpha', 'g', 'ipidx', 'epoch'))])

    # test
    #pbs_array_data = { ('100', '100', '0', '650'), ('200', '100', '0', '650') }
    print(list(pbs_array_data)[0])
    print(f"Total subjobs: {len(pbs_array_data)}")

    # rerunning missing data
    """
    dq_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data"
    missing_data = np.loadtxt(f"{dq_path}/missing_data.txt")
    pbs_array_data = []
    for m in missing_data:
        pbs_array_data.append(tuple(m.astype('int64')))
    pbs_array_data = set(pbs_array_data)
    """

    from qsub import qsub
    qsub(f'python {sys.argv[0]} jac_to_dq', pbs_array_data, 
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
         P='phys_DL')

# ----- plot -----

# phase transition plot
def dq_plot(*args):

    pass

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

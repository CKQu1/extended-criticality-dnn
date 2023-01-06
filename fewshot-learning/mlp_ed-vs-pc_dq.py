import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
import sys
import torch
import os
from ast import literal_eval
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from time import time
from os.path import join
lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
import path_names
#from mlp_fshot import quick_dataload
from NetPortal.models import ModelFactory
from path_names import root_data

from train_supervised import set_data, get_data, IPR, compute_dq

#t0 = time()

def mlp_ed_dq(init_alpha, init_g, init_epoch, root_path):
    #global postact_eds, preact_eds, postact_dqs, preact_dqs, postact_dq_dims, preact_dq_dims
    dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                       if torch.cuda.is_available() else "cpu")
    
    # load pretrained weights (FC10)
    print("Load phase transition initialized pretrained weights.")

    hidden_N = [784]*10 + [10]
    alpha, g = None, None
    #init_path = join(root_path, f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_algosgd_lr=0.001_bs=1024_data_mnist")
    if "PDLAI" in root_path:
        #matches = [net[0] for net in os.walk(root_path) if f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_" in net[0]]
        matches = [ f.path for f in os.scandir(root_path) if f.is_dir() and f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_" in f.path ]
    init_path = matches[0]
    activation = "tanh"
    net_type = "fc"
    kwargs = {"dims": hidden_N, "alpha": alpha, "g": g,
              "init_path": init_path, "init_epoch": init_epoch,
              "activation": activation, "architecture": net_type}
    model = ModelFactory(**kwargs)

    # dataset
    dataset_name = "mnist"
    print(f"Load dataset {dataset_name}.")
    train_ds, valid_ds = set_data(dataset_name,True)
    # move entire dataset to GPU
    train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(dev),
                             torch.tensor(train_ds.targets).to(dev))
    valid_ds = TensorDataset(torch.stack([e[0] for e in valid_ds]).to(dev),
                             torch.tensor(valid_ds.targets).to(dev))

    # get hidden layers
    pc_type = "train"
    if pc_type == "train":
        # make batches
        dataset, _ = get_data(train_ds, valid_ds, len(train_ds))
    elif pc_type == "test":
        _, dataset = get_data(train_ds, valid_ds, len(train_ds))

    for batch_idx, (xb, yb) in enumerate(dataset):
        preact_layers = model.preact_layer(xb)
        postact_layers, outputs = model.postact_layer(xb)

    # compute D_2 (up to pidx PC) and ED
    pidx_max = 50
    preact_eds = np.zeros([len(preact_layers)])
    postact_eds = np.zeros([len(postact_layers)])
    preact_dqs = np.zeros([len(preact_layers), pidx_max])
    postact_dqs = np.zeros([len(postact_layers), pidx_max])
    preact_dq_dims = np.array([min(pidx_max, min(preact_layers[idx].shape)) for idx in range(len(preact_layers))], dtype=int)
    postact_dq_dims = np.array([min(pidx_max, min(postact_layers[idx].shape)) for idx in range(len(postact_layers))], dtype=int)

    q = 2
    for string in ["pre", "post"]:
        layer_name = f"{string}act_layers"
        ed_name = f"{string}act_eds"
        dq_name = f"{string}act_dqs"
        dim_name = f"{string}act_dq_dims"
        for depth in tqdm(range(len(locals()[layer_name]))):
            # compute covariance matrix
            hidden_layer = locals()[layer_name][depth]
            #hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
            #C = (1/hidden_layer.shape[0])*(hidden_layer.T @ hidden_layer) - (1/hidden_layer.shape[0]**2)*(hidden_layer_mean.T @ hidden_layer_mean)
            #C = np.cov(hidden_layer.detach().cpu().numpy())
            C = np.cov(hidden_layer.T.cpu().detach().numpy())        
            # ED
            #eigvals, eigvecs = torch.eig(C, eigenvectors=True)
            eigvals, eigvecs = LA.eig(C)
            ed = np.mean(eigvals**2)**2/np.sum(eigvals**4)
            locals()[ed_name][depth] = ed
            for pidx in range(locals()[dim_name][depth]):
                dq = compute_dq(eigvecs[:,pidx],q)
                locals()[dq_name][depth,pidx] = dq

        print(f"{string} done!")

    
    data_path = join(init_path, "ed-vs-dq")
    if not os.path.isdir(data_path): os.makedirs(f'{data_path}')
    for f1 in ["pre", "post"]:
        for f2 in ["eds", "dqs", "dq_dims"]:
            fname = f"{f1}act_{f2}"
            np.save(join(data_path, fname), locals()[fname])
    
    """
    print(preact_eds)
    print(postact_eds)
    print(preact_dq_dims)
    print(postact_dq_dims)
    print(preact_dqs[-1,:])
    print(postact_dqs[-1,:])
    """

    print("Data saved!")


def ed_dq_submit(*args):
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

    alpha_ls = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    g_ls = [str(g) for g in np.arange(0.25,3.01,0.25)]
    g_ls[-1] = '3.0'

    init_epoch = 650
    root_path = join(root_data,"trained_cnns","/fcn_grid/fc10_grid")
    pbs_array_data = [(alpha, g, init_epoch, root_path)
                      #for alpha100 in alpha100_ls
                      #for g100 in g100_ls
                      for alpha in alpha_ls
                      for g in g_ls
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=root_data,
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='8GB') 

#def ed_dq_plot(lidx, pidx, postact=True):
def ed_dq_plot(pidx, init_g, postact=True):
    #global metrics, matches, root_path

    import pubplot as ppt
    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))

    #lidx, pidx = int(lidx), int(pidx)
    pidx = int(pidx)    # the top pidx(th) principal component
    postact = literal_eval(postact) if isinstance(postact, str) else postact
    root_path = join(root_data,"trained_mlps","fcn_grid/fc10_grid")

    metrics = {}
    fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=False,figsize=(9.5,7.142/2 + 0.15))

    #init_g = 1.0
    #init_alphas = [1.0, 1.2, 1.5, 1.7, 2.0]
    init_alphas = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

    preact_dqs_layers = np.zeros([len(init_alphas), 10])
    postact_dqs_layers = np.zeros([len(init_alphas), 9])

    preact_eds_layers = np.zeros([len(init_alphas), 10])
    postact_eds_layers = np.zeros([len(init_alphas), 9])
    for aidx, init_alpha in enumerate(init_alphas):
        if "PDLAI" in root_path:
            #matches = [net[0] for net in os.walk(root_path) if f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_" in net[0]]
            matches = [ f.path for f in os.scandir(root_path) if f.is_dir() and f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_" in f.path ]
        init_path = matches[0]        
        data_path = join(init_path, "ed-vs-dq")
        for f1 in ["pre", "post"]:
            for f2 in ["eds", "dqs", "dq_dims"]:
                fname = f"{f1}act_{f2}"

                metrics[fname] = np.load(join(data_path, fname + ".npy"))

        #ax1.plot(metrics["postact_eds"][lidx], metrics["postact_dqs"][lidx,pidx], '.', label=init_alpha)
        #ax2.plot(metrics["preact_eds"][lidx], metrics["preact_dqs"][lidx,pidx], '.', label=init_alpha)

        for lidx in range(preact_dqs_layers.shape[1]):
            preact_eds_layers[aidx,lidx] = metrics['preact_eds'][lidx]
            preact_dqs_layers[aidx,lidx] = metrics['preact_dqs'][lidx, pidx]

        for lidx in range(postact_dqs_layers.shape[1]):
            postact_eds_layers[aidx,lidx] = metrics['postact_eds'][lidx]
            postact_dqs_layers[aidx,lidx] = metrics['postact_dqs'][lidx, pidx]

    if postact:
        for lidx in [0,3,5,7,9]:
        #for lidx in range(preact_dqs_layers.shape[1]):
            ax1.plot(init_alphas, preact_dqs_layers[:,lidx], label=lidx+1)
            ax2.plot(init_alphas, preact_eds_layers[:,lidx], label=lidx+1)
    else: 
        for lidx in [0,2,4,6,8]:
        #for lidx in range(postact_dqs_layers.shape[1]):
            ax1.plot(init_alphas, postact_dqs_layers[:,lidx], label=lidx+1)
            ax2.plot(init_alphas, postact_eds_layers[:,lidx], label=lidx+1)


    print(metrics["postact_dqs"].shape)
    print(metrics["preact_dqs"].shape)
    print(metrics.keys())

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        #ax.set_xlabel('Effective dimension')
        ax.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r"$D_2$")
    ax2.set_ylabel("ED")
    ax1.legend(frameon=False, loc='best') #, ncol=2)
    ax2.legend(frameon=False, loc='best')
    #plt.show()

    fig_path = join(root_data, "figure_ms")
    plt.savefig(f"{fig_path}/fc10_g={init_g}_postact={postact}_ed-vs-dq.pdf", bbox_inches='tight')
    print("Plot saved!")
    


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])



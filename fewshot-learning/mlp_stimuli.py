import numpy as np
import pandas as pd
import sys
import torch
from tqdm import tqdm
#from matplotlib import pyplot as plt
import os
from ast import literal_eval
from os.path import join
from time import time
from torch.utils.data import TensorDataset, DataLoader

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
import path_names
#from mlp_fshot import quick_dataload
from NetPortal.models import ModelFactory
from path_names import root_data

t0 = time()
dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")

"""

creating manifold.npy for MLPs

"""

def quick_dataload(dataset_name):
    from torchvision import transforms

    if dataset_name.lower() == "omniglot":
        from torchvision.datasets import Omniglot

        image_size = 28
        train_set = Omniglot(
            root=join(root_data, "data"),
            background=True,
            transform=transforms.Compose(
                [
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )
        test_set = Omniglot(
            root=join(root_data, "data"),
            background=False,
            transform=transforms.Compose(
                [
                    # Omniglot images have 1 channel, but our model will expect 3-channel images
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    return train_set, test_set

def get_batch(batch_size, dataset):
    images = []
    #dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #images, labels = next(iter(dataset))
        
    #return images.reshape((images.shape[0],-1))
    return dataset

# ---------------------- MODEL ----------------------
print("Setting up models.")

def mlp_stimuli(init_alpha, init_g, init_epoch, root_path=join(root_data, "trained_mlps", "fcn_grid/fc10_grid"), 
                dataset_name = "omniglot", dataset_type="test", projection=False):

    #global manifolds, emb_path

    # ---------------------- DATA ----------------------    
    print(f"Loading dataset {dataset_name}.")

    train_set, test_set = quick_dataload(dataset_name)

    # pretrained network (MLP)
    hidden_N = [784]*10 + [10]
    #init_alpha, init_g = 1.0, 1.0
    alpha, g = None, None
    #init_path = join(root_path, f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_algosgd_lr=0.001_bs=1024_data_mnist")
    if "PDLAI" in root_path:
        matches = [net[0] for net in os.walk(root_path) if f"fc10_mnist_tanh_id_stable{init_alpha}_{init_g}_epoch650_" in net[0]]
    init_path = matches[0]
    #init_epoch = 650
    activation = "tanh"
    net_type = "fc"
    kwargs = {"dims": hidden_N, "alpha": alpha, "g": g,
              "init_path": init_path, "init_epoch": init_epoch,
              "activation": activation, "architecture": net_type}
    model = ModelFactory(**kwargs)

    class MlpBackbone(torch.nn.Module):
        def __init__(self, model):
            super(MlpBackbone, self).__init__()
            self.features = model.sequential[:-1]
            self.flatten = torch.nn.Flatten()
            
        def forward(self, x):
            x = self.features(x)
            x = self.flatten(x)
            return x

    backbone = MlpBackbone(model)
    backbone.to(dev).eval()

    print(f"{time() - t0}s")
    # ---------------------- COMPUTATION ----------------------
    print("Computing.")

    batch_size = 10
    #dataset_type = "train"
    if dataset_type == "train":
        dataset = get_batch(batch_size, train_set)
    elif dataset_type == "test":
        dataset = get_batch(batch_size, test_set)

    input_tensor, _ = next(iter(dataset))   # dummy variable
    input_tensor = input_tensor.reshape((input_tensor.shape[0],-1))
    with torch.no_grad():
        output = backbone(input_tensor.to(dev)).cpu()
    #N = input_tensor.shape[0]
    N = 784    # this can really be arbitrary (dimension you want to project the output on)
    M = output.shape[1]
    # idxs = torch.randint(0,100352,(N,))
    if projection:
        O = torch.randn((M,N))/np.sqrt(N)
        O = O.to(dev)

    # N = 2048
    #K = 64
    K = 128
    P = 50
    #K = 100
    #P = 500
    batch_size = 10
    #n_classes = len(imnames)
    #n_classes = 10

    manifolds = []
    for batch_idx, (images, labels) in enumerate(tqdm(dataset)):    
        if batch_idx < K*P//batch_size: 
            images = images.reshape((images.shape[0],-1))
            with torch.no_grad():
                output = backbone(images.to(dev))
            if projection:
                manifolds.append( (output@O).cpu().numpy() )
            else:
                manifolds.append( output.cpu().numpy() )
        else:
            break
    print(len(manifolds))
    manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,M)
    print(f"{time() - t0}s")

    emb_path = join(init_path, "manifold", f"{dataset_name}={dataset_type}_ep={init_epoch}_bs={batch_size}_K={K}_P={P}_N={N}_proj={projection}")

    if not os.path.isdir(emb_path):
        os.makedirs(emb_path)
    np.save(os.path.join(emb_path,f'manifolds.npy'), manifolds)
    print(f"Manifold saved in {emb_path}!")

def stimuli_submit(*args):
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

    #alpha_ls = [str(alpha) for alpha in [1.0,1.5,2.0]]
    #g_ls = [str(g) for g in [0.25,1.0,3.0]]
    #alpha_ls = [str(alpha) for alpha in np.arange(1.0,2.01,0.1)]
    #alpha_ls = ['2.0']
    alpha_ls = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    g_ls = [str(g) for g in np.arange(0.25,3.01,0.25)]
    g_ls[-1] = '3.0'
    
    #init_epoch = 100
    #init_epoch = 650
    init_epoch = 650
    root_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD/fcn_grid/fc10_grid"
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
             path='/project/PDLAI/project2_data',
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='1GB') 

# ---------------------- SNR Theory ----------------------

from train_supervised import compute_dq

def snr_components(model_name, init_alpha100, init_g100, fname, 
                   root_path=join(root_data, "trained_mlps", "fcn_grid/fc10_grid"), projection=False, force_compute=True):

    import numpy.linalg as LA
    from NetPortal.models import ModelFactory

    global model
    global Us, manifolds_all, Rs_all, dists_all, PRs_all, css_all, SNRs_all
    global EDs_all, dqs_all
    global layers
    global manifolds
    global init_epoch, emb_path, matches
    global depth, verify_ls

    print("Setting up model path.")
    if "PDLAI" in root_path:
        matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epoch650" in f.path and f"stable{init_alpha100}_{init_g100}_" in f.path]        
    init_path = matches[0]
    print(init_path)

    emb_path = join(init_path, "manifold", fname)
    #manifolds = np.load(join(emb_path, f'manifolds.npy'))
    #print(f"Loaded manifold from {emb_path}!")

    print("Loading model weights!")
    epoch_str = [s for s in fname.split("_") if "ep=" in s][0]
    init_epoch = int( epoch_str[epoch_str.find("=")+1:] )
    dataset_name, dataset_type = fname.split("_")[0].split("=")
    activation = "tanh"
    hidden_N = [784]*10 + [10]
    net_type = "fc"
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": init_path, "init_epoch": init_epoch,
              "activation": activation, "architecture": net_type}
    model = ModelFactory(**kwargs)

    """
    Rs = []
    centers = []
    Us = []
    for manifold in manifolds:
        centers.append(manifold.mean(0))
        U,R,V = np.linalg.svd(manifold - manifold.mean(0),full_matrices=False)
        Rs.append(R)
        Us.append(V)
    Rs = np.stack(Rs)
    centers = np.stack(centers)
    Us = np.stack(Us)
    #plt.plot(Rs.T)
    """

    class Backbone(torch.nn.Module):
        def __init__(self, model, module_idx, layer_idx=None):
            super(Backbone, self).__init__()
            self.layer_idx = layer_idx
            self.pre_features = torch.nn.Sequential(*list(model.children())[:module_idx])
            if layer_idx:
                self.features = torch.nn.Sequential(*list(model.children())[module_idx][:layer_idx])
            self.flatten = torch.nn.Flatten()
            
        def forward(self, x):
            x = self.pre_features(x)
            if self.layer_idx:
                x = self.features(x)
            x = self.flatten(x)
            return x
    # backbone = Backbone(model)
    # backbone.to('cuda').eval()

    N = 2048
    #N_all = 169000
    #idxs = np.random.randint(0,N_all,N)
    batch_size = 64
    i=0

    batch_size = 10
    #K = 64
    K = 128
    P = 50

    # loading dataset
    print(f"Loading dataset {dataset_name}")
    if dataset_type == "train":
        dataset, _ = quick_dataload(dataset_name)
    elif dataset_type == "test":
        _, dataset = quick_dataload(dataset_name)
    dataset = get_batch(batch_size, dataset)

    layers = []
    for module_idx in tqdm(range(len(list(model.children())))):
        try:
            len_module = len(list(model.children())[module_idx])
            for layer_idx in range(len_module):
                
                backbone = Backbone(model, module_idx, layer_idx)
                backbone.to(dev).eval()

                # Get test batch
                #input_tensor = get_batch(0,batch_size)
                input_tensor, _ = next(iter(dataset))
                input_tensor = input_tensor.reshape((input_tensor.shape[0],-1))
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                layers.append(list(model.children())[module_idx][layer_idx])
        except:
            layer_idx = 0
            layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
            
            backbone = Backbone(model, module_idx)
            backbone.to(dev).eval()

            # Get test batch
            #input_tensor = get_batch(0,batch_size)
            input_tensor, _ = next(iter(dataset))
            input_tensor = input_tensor.reshape((input_tensor.shape[0],-1))
            with torch.no_grad():
                output = backbone(input_tensor.to(dev))
            try:
                layers.append(list(model.children())[module_idx][layer_idx])
            except:
                layers.append('avgpool')

    print(f"Total number of layers: {len(layers)}")

    #conv_idxs = [0,3,6,8,10,15]     # useless
    #np.array(layers)[conv_idxs]     # useless

    # -------------------

    N = 2048
    #N_all = 169000
    #idxs = np.random.randint(0,N_all,N)
    batch_size = 64
    i=0

    batch_size = 10
    #K = 64
    K = 128
    P = 50

    depth = len(list(model.children()))
    verify_ls = []
    layerwise_file = "manifolds_layerwise.npy"
    print("Computation for manifolds_layerwise.npy")
    for module_idx in range(depth):
        if os.path.isfile( join(emb_path, f"manifolds_{module_idx}.npy") ): 
            verify_ls.append(1)

    #if os.path.isfile( join(emb_path, layerwise_file) ):
    if sum(verify_ls) == depth and (not force_compute):
        print("manifolds_layerwise.npy computed already, loading now.")
        #manifolds_all = np.load(join(emb_path, layerwise_file))
        manifolds_all = []
        for module_idx in range(depth):
            manifolds_all.append( np.load(join(emb_path, f"manifolds_{module_idx}.npy")) )
    else:
        print(f"Total number of modules: { len(list(model.children())) }")
        counter = 0
        manifolds_all = []
        for module_idx in tqdm(range(len(list(model.children())))):

            try:
                len_module = len(list(model.children())[module_idx])
                for layer_idx in range(len_module):

                    print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                    
                    backbone = Backbone(model, module_idx, layer_idx)
                    backbone.to(dev).eval()

                    # Get test batch
                    #input_tensor = get_batch(0,batch_size)
                    input_tensor, _ = next(iter(dataset))
                    input_tensor = input_tensor.reshape((input_tensor.shape[0],-1))
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    N_all = output.shape[-1]
                    if projection:
                        O = torch.randn(N_all,N) / np.sqrt(N)
                        O = O.to(dev)
        #             idxs = np.random.randint(0,N_all,N)
                    
                    manifolds = []

                    #for i in tqdm(range(K*P//batch_size)):
                    for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
                        #input_tensor = get_batch(i,batch_size)
                        if batch_idx < K*P//batch_size:
                            input_tensor = input_tensor.reshape((images.shape[0],-1))                        
                            with torch.no_grad():
                                output = backbone(input_tensor.to(dev))
                            if projection:
                                manifolds.append((output@O).cpu().numpy())
                            else:
                                manifolds.append(output.cpu().numpy())
                        else:
                            break
                    manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,N_all) 
                    manifolds_all.append(manifolds)
                    # for MLPs save things layer by layer
                    np.save(os.path.join(emb_path, f'manifolds_{counter}.npy'), manifolds)       
                    counter += 1

            except:
                layer_idx = 0
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                backbone = Backbone(model, module_idx)
                backbone.to(dev).eval()

                # Get test batch
                #input_tensor = get_batch(0,batch_size)
                input_tensor, _ = next(iter(dataset))
                input_tensor = input_tensor.reshape((input_tensor.shape[0],-1))
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                N_all = output.shape[-1]
                if projection:
                    O = torch.randn(N_all,N) / np.sqrt(N)
                    O = O.to(dev)
        #         idxs = np.random.randint(0,N_all,N)        

                manifolds = []
                #for i in tqdm(range(K*P//batch_size)):
                for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
                    #input_tensor = get_batch(i,batch_size)
                    if batch_idx < K*P//batch_size:
                        input_tensor = input_tensor.reshape((images.shape[0],-1))                        
                        with torch.no_grad():
                            output = backbone(input_tensor.to(dev))
                        if projection:
                            manifolds.append((output@O).cpu().numpy())
                        else:
                            manifolds.append(output.cpu().numpy())
                    else:
                        break


                manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,N_all)
                manifolds_all.append(manifolds)
                # for MLPs save things layer by layer
                np.save(os.path.join(emb_path, f'manifolds_{counter}.npy'), manifolds)       
                counter += 1

        #np.save(os.path.join(emb_path, f'manifolds_layerwise.npy'), manifolds_all)
        print("manifolds_layerwise.npy saved!")

    """
    # random projection
    N = 2048
    M = 88
    A = np.random.randn(N,M)/np.sqrt(M)

    manifolds_all = []
    for manifolds in manifolds_all_load:
        manifolds_all.append(manifolds@A)
    """

    # ------------------

    def geometry(centers,Rs,Us,m):
        K = len(centers)
        P = Rs.shape[1]
        dists = np.sqrt(((centers[:,None] - centers[None])**2).sum(-1))
        dist_norm = dists / np.sqrt((Rs**2).sum(-1)[:,None] / P)

        Ds = np.sum(Rs**2,axis=-1)**2 / np.sum(Rs**4, axis=-1)

        # Center-subspace
        csa = []
        csb = []
        for a in range(K):
            for b in range(K):
                if a!=b:
                    dx0 = centers[a] - centers[b]
                    dx0hat = dx0 / np.linalg.norm(dx0)
                    costheta_a = Us[a]@dx0hat
                    csa.append((costheta_a**2 * Rs[a]**2).sum() / (Rs[a]**2).sum())
                    costheta_b = Us[b]@dx0hat
                    csb.append((costheta_b**2 * Rs[b]**2).sum() / (Rs[a]**2).sum())
                else:
                    csa.append(np.nan)
                    csb.append(np.nan)
        csa = np.stack(csa).reshape(K,K)
        csb = np.stack(csb).reshape(K,K)

        # Subspace-subspace
        ss = []
        for a in range(K):
            for b in range(K):
                if a!=b:
                    cosphi = Us[a]@Us[b].T
                    ss_overlap = (cosphi**2*Rs[a][:,None]**2*Rs[b]**2).sum() / (Rs[a]**2).sum()**2
                    ss.append(ss_overlap)
                else:
                    ss.append(np.nan)
        ss = np.stack(ss).reshape(K,K)

        css = (csa + csb/m) * dist_norm**2

        bias = (Rs**2).sum(-1) / (Rs**2).sum(-1)[:,None] - 1
        SNR = 1/2*(dist_norm**2 + bias/m)/ np.sqrt(1/Ds[:,None]/m + css + ss/m)
        
        return dist_norm, Ds, csa, ss, SNR


    print("Computing SNR metrics!")
    from scipy.spatial.distance import pdist, squareform
    m = 5

    #K = len(manifolds)
    K = len(manifolds_all[-1])
    PRs_all = []
    Rs_all = []
    dists_all = []
    css_all = []
    SNRs_all = []

    # added analysis for correlation matrix (same as MLP)
    EDs_all = []

    pc_idxs = list(range(25))   # top 25 PCs
    dqs_all = {}
    for pc_idx in pc_idxs:
        dqs_all[pc_idx] = []
    qs = np.linspace(0,2,100)   # for computing D_q

    for manifolds in tqdm(manifolds_all):
        manifolds = np.stack(manifolds)

        # get a general version of cov mat      
        eigvals, eigvecs = LA.eig( np.cov( manifolds.reshape( manifolds.shape[0] * manifolds.shape[1], manifolds.shape[2] ).T ) )
        EDs_all.append( np.mean(np.abs(eigvals)**2)**2 / np.sum(np.abs(eigvals)**4) )
        
        for pc_idx in pc_idxs:
            dqs_all[pc_idx].append( [compute_dq(eigvecs[:,pc_idx],q) for q in qs ] )
        print(eigvecs.shape)

        Rs = []
        centers = []
        Us = []
        for manifold in manifolds:
            centers.append(manifold.mean(0))
            U,R,V = np.linalg.svd(manifold - manifold.mean(0),full_matrices=False)
            Rs.append(R)
            Us.append(V)
        Rs = np.stack(Rs)
        Rs_all.append(Rs)
        centers = np.stack(centers)
        Us = np.stack(Us)
        
        dist_norm, Ds, csa, ss, SNR = geometry(centers,Rs,Us,m)
        dists_all.append(dist_norm)
        PRs_all.append(Ds)
        css_all.append(csa)
        SNRs_all.append(SNR)
        
    Rs_all = np.stack(Rs_all)
    dists_all = np.stack(dists_all)
    PRs_all = np.stack(PRs_all)
    css_all = np.stack(css_all)
    SNRs_all = np.stack(SNRs_all)

    EDs_all = np.stack(EDs_all)
    #dqs_all = np.stack(dqs_all)

    #data_path = f"{fname}_epoch={init_epoch}"
    #if not os.path.isdir(f"{data_path}"): os.makedirs(data_path)

    np.save(os.path.join(emb_path,'SNRs_layerwise.npy'),SNRs_all)
    np.save(os.path.join(emb_path,'Ds_layerwise.npy'),PRs_all)
    np.save(os.path.join(emb_path,'dist_norm_layerwise.npy'),dists_all)
    np.save(os.path.join(emb_path,'css_layerwise.npy'),css_all)

    np.save(join(emb_path,'EDs_layerwise.npy'), EDs_all)    
    for pc_idx in pc_idxs:
        dqs_all[pc_idx] = np.stack(dqs_all[pc_idx])
        np.save(join(emb_path,f'dqs_layerwise_{pc_idx}.npy'), dqs_all[pc_idx])
    print(f"SNR metrics saved in: {emb_path}!")

# def snr_components(model_name, init_alpha100, init_g100, init_epoch, fname, root_path):
def snr_submit(*args):
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

    #alpha100_ls = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    #g100_ls = [str(g) for g in np.arange(0.25,3.01,0.25)]
    #g100_ls[-1] = '3.0'
    alpha100_ls = [1.0,2.0]
    g100_ls = ['1.0']

    models = ["fc10"]    
    fname = "omniglot=test_ep=650_bs=10_K=128_P=50_N=784_proj=False"

    pbs_array_data = [(model_name, alpha100, g100, fname)
                      for model_name in models
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      #if (alpha100, g100) not in no_pair
                      ]

    #print(len(pbs_array_data))
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path='/project/PDLAI/project2_data',
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='2GB') 

# ---------------------- Plotting ----------------------

# creates two plots
def snr_metric_plot(alpha100_ls, g100, metric1, metric2, 
                    fname, root_path=join(root_data,"trained_mlps/fcn_grid/fc10_grid")):
    alpha100_ls = literal_eval(alpha100_ls) if isinstance(alpha100_ls, str) else alpha100_ls
    #g100_ls = literal_eval(g100_ls) if isinstance(g100_ls, str) else g100_ls

    global metric_data, dq_data, SNRs_all, metric_dict, SNRs_all, net_ls, matches, init_path, emb_path

    """
    Fig 1:
    Plots the a selected metric1 (based on metric_dict) vs layer and SNR vs layer,
    for dq, the pc_idx PC needs to be selected

    Fig 2:
    Plots the a selected metric2 (based on metric_dict) vs layer and a scatter plot between SNR and D_2
    """

    # Plot settings
 
    markers = ["o", "v", "s", "p", "*", "P", "H", "D", "d", "+", "x"]

    metric0 = "SNR"     # always plot SNR
    dq_ls = metric1.split("_") if "dq" in metric1 else []
    dq_filename = f"dqs_layerwise_{dq_ls[1]}" if len(dq_ls) > 0 else None
    metric_dict = {"SNR"        : "SNRs_layerwise",
                   "D"          : "Ds_layerwise",
                   "dist_norm"  : "dist_norm_layerwise",
                   "css"        : "css_layerwise"               
                   }

    name_dict = {"D"          : r'$D$',
                 "dist_norm"  : "Distance norm",
                 "css"        : "Centre subsplace"               
                 }   

    if "dq_" in metric1:
        metric_dict[metric1] = dq_filename
        name_dict[metric1] = r'$D_2$' 

    assert metric1 in metric_dict.keys(), "metric1 not in dictionary!"
    assert metric2 in metric_dict.keys(), "metric2 not in dictionary!" 

    import matplotlib.pyplot as plt
    import pubplot as ppt

    # get available networks
    net_ls = []
    for aidx, alpha100 in enumerate(alpha100_ls):
        if "PDLAI" in root_path:
            matches = [join(root_path,net_path) for net_path in os.listdir(root_path) if f"fc10_mnist_tanh_id_stable{alpha100}_{g100}_epoch650_" in net_path]
            net_ls.append(matches[0])
    
    # transparency list
    trans_ls = np.linspace(0,1,len(net_ls)+1)[::-1]

    # --------------- Plot 1 ---------------
    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))
    fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=False,figsize=(9.5,7.142/2 + 0.15))
    for nidx, init_path in enumerate(net_ls):

        alpha = int(alpha100_ls[nidx])
        # set paths
        manifold_path = join("manifold", fname)

        # load data     
        emb_path = join(init_path, manifold_path)
        metric_data = np.load( join(emb_path, f"{metric_dict[metric1]}.npy") )
        # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
        metric_data = metric_data[:,-1] if "dq_" in metric1 else metric_data
        SNRs_all = np.load(join(emb_path, f"{metric_dict[metric0]}.npy"))
        # fractional layer
        total_layers = SNRs_all.shape[0]
        frac_layers = np.arange(0,total_layers)/(total_layers-1)
            
        if "dq" not in metric1:
            ax1.plot(frac_layers, metric_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=alpha)
        else:
            ax1.plot(frac_layers, metric_data, alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=alpha)
        ax2.plot(frac_layers, np.nanmean(SNRs_all,(1,2)), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=alpha)
        print(f"{alpha} done!")

    ax1.set_ylim((0,1))
    ax2.set_ylim((0.05,0.4))

    ax1.legend(frameon=False, ncol=2, fontsize=10)
    #ax2.set_yscale('log')
    #ax2.ticklabel_format(style="sci", axis="y" )

    ax1.set_ylabel(name_dict[metric1])
    ax2.set_ylabel("SNR")
    ax1.set_xlabel("Fractional depth")
    ax2.set_xlabel("Fractional depth")
    plt.show()

    fig_path = "/project/PDLAI/project2_data/figure_ms"
    #plt.savefig(join(fig_path, f"mlp_snr_{metric1}-vs-layer.pdf") , bbox_inches='tight')
    print(f"Plot 1 saved!")

    # --------------- Plot 2 ---------------
    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))
    fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = False,sharey=False,figsize=(9.5,7.142/2 + 0.15))
    for nidx, init_path in enumerate(net_ls):
    
        alpha = int(alpha100_ls[nidx])
        # set paths
        manifold_path = join("manifold", fname)

        # load data     
        emb_path = join(init_path, manifold_path)
        #PRs_all = np.load(join(emb_path, "Ds_layerwise.npy"))
        metric_data = np.load( join(emb_path, f"{metric_dict[metric2]}.npy") )
        # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
        metric_data = metric_data[:,-1] if "dq_" in metric2 else metric_data
        dq_name = "dq_0"
        dq_data = np.load( join(emb_path, f"{metric_dict[dq_name]}.npy") )
        dq_data = dq_data[:,-1]
        SNRs_all = np.load(join(emb_path, f"{metric_dict[metric0]}.npy"))
        # fractional layer
        total_layers = SNRs_all.shape[0]
        frac_layers = np.arange(0,total_layers)/(total_layers-1)
            
        if "dq" not in metric2:
            ax1.plot(frac_layers, metric_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=alpha100)
        else:
            ax1.plot(frac_layers, metric_data, alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=alpha100)

        # only scatter plot the latter layers
        deep_layers = np.where(frac_layers >= 0)
        ax2.scatter(dq_data[deep_layers], np.nanmean(SNRs_all,(1,2))[deep_layers], alpha=0.6)
        print(f"{alpha100} done!")
    
    #ax2.set_xlim((0,1))
    #ax2.set_ylim((0.05,0.4))

    ax1.legend(frameon=False, ncol=2, fontsize=10)
    ax2.set_yscale('log')
    #ax2.ticklabel_format(style="sci", axis="y" )

    ax1.set_ylabel(name_dict[metric2])
    ax2.set_ylabel("SNR")
    ax1.set_xlabel("Fractional depth")
    ax2.set_xlabel(r"$D_2$")
    plt.show()

    #fig_path = "/project/dnn_maths/project_qu3/fig_path"
    fig_path = "/project/PDLAI/project2_data/figure_ms"
    #plt.savefig(join(fig_path, f"mlp_snr_{metric2}-dq_scatter.pdf") , bbox_inches='tight')
    print(f"Plot 2 saved!")



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])


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

# for saving all the relevant analysis on pretrained nets
global pretrained_path, untrained_path
pretrained_path = join(root_data, "pretrained_workflow", "pretrained_dnns")
untrained_path = join(root_data, "pretrained_workflow", "untrained_dnns")

t0 = time()
dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")

"""

creating manifold.npy for pretrained DNNs (mostly CNNs)

"""

# ---------------------- DATA ----------------------
print("Loading data.")

#imnames = pd.read_csv(join(os.getcwd(),"brain-score/image_dicarlo_hvm-public.csv"))
log_path = join(root_data, "fewshot-data")
imdir = join(log_path, "image_dicarlo_hvm-public")
imnames = np.load(join(imdir,"majaj_2015_imnames_2.npy"),allow_pickle=True)

# Image preprocessing
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_batch(i, batch_size):
    from PIL import Image
#     ims = imnames[class_id]
    images = []
    for im in imnames[i*batch_size:(i+1)*batch_size]:
        #impath = os.path.join(imdir, im+ '.png')
        impath = os.path.join(imdir, im)
        images.append(preprocess(Image.open(impath)))
        
    images = torch.stack(images)
        
    return images

def get_full_batch():
    from PIL import Image
#     ims = imnames[class_id]
    images = []
    for im in imnames:
        #impath = os.path.join(imdir, im+ '.png')
        impath = os.path.join(imdir, im)
        images.append(preprocess(Image.open(impath)))
        
    images = torch.stack(images)
        
    return images

# ---------------------- Backbone for models ----------------------

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

# ---------------------- SNR Theory ----------------------

from train_supervised import compute_dq

# projection must equal True lol otherwise computation gets too expensive for computing the covariance matrix
def snr_components(model_name, pretrained, projection=True):

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable

    #global Us, manifolds_all, Rs_all, dists_all, PRs_all, css_all, SNRs_all
    #global U, R, V, Rs, Us
    #global d2s_all
    #global layers
    #global manifolds, manifolds_all
    #global init_epoch, emb_path
    #global model, module_idx, layer_idx, Backbone
    global input_fbatch, output_fbatch, Un, Rn, Vn, d2, backbone

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained

    print(f"Setting up path for {model_name}.")        
    # where the desired manifolds.npy/manifolds_N={N}.npy, etc is located
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model weights!")
    if pretrained:
        model = torch.load(join(pretrained_path, model_name, "model_pt"))
    else:
        model = torch.load(join(untrained_path, model_name, "model_pt"))        

    """
    manifolds = np.load(join(emb_path, f'manifolds.npy'))
    print(f"Loaded manifold from {emb_path}!")

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

    N = 2048
    batch_size = 64

    batch_size = 10
    K = 64
    P = 50

    # finds out the total amount of layers needed for evaluation (this is not rlly needed as well)
    """
    layers = []
    for module_idx in tqdm(range(len(list(model.children())))):
        try:
            len_module = len(list(model.children())[module_idx])
            for layer_idx in range(len_module):
                
                backbone = Backbone(model, module_idx, layer_idx)
                backbone.to(dev).eval()

                # Get test batch
                input_tensor = get_batch(0,batch_size)
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                layers.append(list(model.children())[module_idx][layer_idx])
        except:
            layer_idx = 0
            layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
            
            backbone = Backbone(model, module_idx)
            backbone.to(dev).eval()

            # Get test batch
            input_tensor = get_batch(0,batch_size)
            with torch.no_grad():
                output = backbone(input_tensor.to(dev))
            try:
                layers.append(list(model.children())[module_idx][layer_idx])
            except:
                layers.append('avgpool')

    print(f"Total number of layers: {len(layers)}")
    """

    # -------------------

    # same variables as in cnn_macaque_stimuli.py
    N = 2048
    N_dq = 500
    navg_dq = 10

    batch_size = 10
    K = 64
    P = 50

    # analysis for top 200 PCs
    n_top = 50 # not used
    
    layerwise_file = "manifolds_layerwise.npy"
    print(f"Computation for {layerwise_file}!")
    if os.path.isfile( join(emb_path, 'manifolds_layerwise.npy') ):
        #manifolds_all_load = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        #manifolds_all = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        print("manifolds_layerwise.npy computed already, loading now.")
        manifolds_all = np.load(join(emb_path, layerwise_file))

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

                    print('Get test batch')
                    # Get test batch
                    input_tensor = get_batch(0,batch_size)
                    print('Input loaded')
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    print('Output computed')
                    N_all = output.shape[-1]
                    print(N_all)
                    
                    if projection:
                        O = torch.randn(N_all,N) / np.sqrt(N)   # Gaussian projection
                        #O = torch.Tensor(levy_stable.rvs(alpha, 0, size=(N_all,N), scale=(0.5/N)**(1./alpha)))  # Levy projection (very expensive)
                        O = O.to(dev)

                    print('Starting batch manifolds')                    
                    manifolds = []
                    for i in tqdm(range(K*P//batch_size)):
                        input_tensor = get_batch(i,batch_size)
                        with torch.no_grad():
                            output = backbone(input_tensor.to(dev))                            

                        if projection:
                            manifolds.append((output@O).cpu().numpy())
                        else:
                            manifolds.append(output.cpu().numpy())

                    manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,N_all) 
                    manifolds_all.append(manifolds)
                    counter += 1
                    print('Batch manifolds complete!')   
  

            except:
                layer_idx = 0
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                backbone = Backbone(model, module_idx)
                backbone.to(dev).eval()

                print('Get test batch')
                # Get test batch
                input_tensor = get_batch(0,batch_size)
                print('Input loaded')
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                print('Output computed')
                N_all = output.shape[-1]
                print(N_all)
                if projection:
                    O = torch.randn(N_all,N) / np.sqrt(N)   # Gaussin projection
                    #O = torch.Tensor(levy_stable.rvs(alpha, 0, size=(N_all,N), scale=(0.5/N)**(1./alpha)))
                    O = O.to(dev)       

                print('Starting batch manifolds')
                manifolds = []
                for i in tqdm(range(K*P//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))

                    if projection:
                        manifolds.append((output@O).cpu().numpy())
                    else:
                        manifolds.append(O.cpu().numpy())

                manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,N_all)
                manifolds_all.append(manifolds)
                counter += 1
                print('Batch manifolds complete!')

        np.save(os.path.join(emb_path, f'manifolds_layerwise.npy'), manifolds_all)
        #manifolds_all_load = manifolds_all
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
        
        # additionally returning bias
        return dist_norm, Ds, csa, ss, SNR, bias

    print("Computing SNR metrics!")
    from scipy.spatial.distance import pdist, squareform

    ms = list(range(1,11))

    #K = len(manifolds)
    K = len(manifolds_all[0])
    PRs_all = []
    Rs_all = []
    dists_all = []
    css_all = []    
    biases_all = []
    SNRs_all = {}
    for m in ms:
        SNRs_all[m] = []

    for mf_idx, manifolds in tqdm(enumerate(manifolds_all)):
        manifolds = np.stack(manifolds)

        # get a general version of cov mat (# Method 1: full batch)     
        """
        eigvals, eigvecs = LA.eigh( np.cov( manifolds.reshape( manifolds.shape[0] * manifolds.shape[1], manifolds.shape[2] ).T ) )
        ii = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[ii]
        eigvals = eigvecs[:,ii]
        #EDs_all.append( eigvals )
        d2s = []
        for eidx in range(len(eigvals)):
            d2s.append( compute_dq(eigvecs[:,eidx],2) )
        """

        # save correlation dimension D_2 layer by layer
        #d2s = []
        Rs = []
        centers = []
        Us = []
        for manifold in manifolds:
            centers.append(manifold.mean(0))
            # this is equivalent to PCA!!!
            U,R,V = np.linalg.svd(manifold - manifold.mean(0),full_matrices=False)
            Rs.append(R)
            Us.append(V)
            # PCA of the neural representations (# Method 2: minibatches)
            #d2 = []
            #for eidx in range(len(R)):
            #    d2.append( compute_dq(V[:,eidx],2) ) 
            #d2 = np.array(d2)
            #d2s.append(d2)
        Rs = np.stack(Rs)
        Rs_all.append(Rs)
        centers = np.stack(centers)
        Us = np.stack(Us)
        #d2s = np.stack(d2s)
        #d2s_all.append(d2s)
        
        #dist_norm, Ds, csa, ss, SNR = geometry(centers,Rs,Us,m)
        dist_norm, Ds, csa, ss, SNR, bias = geometry(centers,Rs,Us,m)
        dists_all.append(dist_norm)
        PRs_all.append(Ds)
        css_all.append(csa)
        biases_all.append(bias)

        for midx, m in enumerate(ms):
            # only the SNR is affected by the No. of learning examples m
            _, _, _, _, SNR, _ = geometry(centers,Rs,Us,m)
            SNRs_all[m].append(SNR)
        
    Rs_all = np.stack(Rs_all)
    dists_all = np.stack(dists_all)
    PRs_all = np.stack(PRs_all)
    css_all = np.stack(css_all)
    biases_all = np.stack(biases_all)

    #d2s_all = np.stack(d2s_all)

    for m in ms:
        SNRs_all[m] = np.stack(SNRs_all[m])
        np.save(os.path.join(emb_path,f'SNRs_layerwise_m={m}.npy'),SNRs_all[m])

    np.save(os.path.join(emb_path,'Rs_layerwise.npy'),Rs_all)    
    np.save(os.path.join(emb_path,'Ds_layerwise.npy'),PRs_all)
    np.save(os.path.join(emb_path,'dist_norm_layerwise.npy'),dists_all)
    np.save(os.path.join(emb_path,'css_layerwise.npy'),css_all)
    np.save(os.path.join(emb_path,f'biases_layerwise.npy'),biases_all)

    # save manifolds_all dimension
    np.save( join(emb_path, "manifold_dim.npy"), np.array([len(manifolds_all), len(manifolds), manifold.shape[0], manifold.shape[1]]) )

# get d2 from minibatch
def snr_d2_mbatch(model_name, pretrained, n_top=100):

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable
    from sklearn.decomposition import PCA

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained
    n_top = int(n_top)

    print(f"Setting up path for {model_name}.")        
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model weights!")
    if pretrained:
        model = torch.load(join(pretrained_path, model_name, "model_pt"))
    else:
        model = torch.load(join(untrained_path, model_name, "model_pt"))        

    # -------------------
    
    d2_file = f"d2smb_n_top={n_top}_layerwise.npy"
    Rsmb_file = f"Rsmb_n_top={n_top}_layerwise.npy"
    d2_exist = os.path.isfile( join(emb_path, d2_file) )
    Rsmb_exist = os.path.isfile( join(emb_path, Rsmb_file) ) 
    print(f"Computation for {d2_file}!")
    if d2_exist and Rsmb_exist:
        print(f"{d2_file} and {Rsmb_file} computed already, no computation required.")
        return
    else:
        print(f"Total number of modules: { len(list(model.children())) }")

        batch_size = 100
        N_images = len(imnames)

        counter = 0
        d2smb_all = []    # only compute D_2's
        Rsmb_all = []   # the corresponding singularvalues (from large to small)
        # load full-batch input data
        input_fbatch = get_full_batch()
        for module_idx in tqdm(range(len(list(model.children())))):
            try:
                len_module = len(list(model.children())[module_idx])
                for layer_idx in range(len_module):

                    print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )

                    backbone = Backbone(model, module_idx, layer_idx).to(dev)
                    backbone.eval()

                    d2s = []
                    Rs = []
                    for i in tqdm(range(N_images//batch_size)):
                        input_tensor = get_batch(i,batch_size)
                        with torch.no_grad():
                            output = backbone(input_tensor.to(dev))      
                        output -= output.mean(0)                    
                        output = output.detach().numpy()

                        # scklearn PCA
                        pca = PCA(n_top)
                        pca.fit(output)  # fit to data
                        Rn = pca.explained_variance_
                        Vn = pca.components_

                        d2 = [compute_dq(Vn[eidx,:], 2) for eidx in range(len(Rn))]
                        d2s.append(d2)
                        Rs.append(Rn)

                    d2s = np.stack(d2s)
                    Rs = np.stack(Rs)
                    d2smb_all.append(d2s)
                    Rsmb_all.append(Rs)

                    del backbone
                    counter += 1    
                    #print(f"try {counter}")
            
            except:
                layer_idx = 0
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                if "backbone" in locals():
                    del backbone
                backbone = Backbone(model, module_idx).to(dev)
                backbone.eval()

                d2s = []
                Rs = []
                for i in tqdm(range(N_images//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))      
                    output -= output.mean(0)                    
                    output = output.detach().numpy()

                    # scklearn PCA
                    pca = PCA(n_top)
                    pca.fit(output)  # fit to data
                    Rn = pca.explained_variance_
                    Vn = pca.components_

                    d2 = [compute_dq(Vn[eidx,:], 2) for eidx in range(len(Rn))]
                    d2s.append(d2)
                    Rs.append(Rn)
    
                d2s = np.stack(d2s)
                Rs = np.stack(Rs)
                d2smb_all.append(d2s)
                Rsmb_all.append(Rs)

                del backbone
                counter += 1
                #print(f"except {counter}")
                
        print(f"d2 computation complete for {counter} layers!")

        d2smb_all = np.stack(d2smb_all)
        Rsmb_all = np.stack(Rsmb_all)
        # save correlation dimension D_2
        #np.save(join(emb_path,f'd2s_layerwise.npy'), d2s_all)   # minibatch
        np.save(join(emb_path,f'd2smb_n_top={n_top}_layerwise.npy'), d2smb_all)   # fullbatch
        np.save(join(emb_path,f'Rsmb_n_top={n_top}_layerwise.npy'), Rsmb_all)   # fullbatch

# delete later, this is because I messed up
def snr_rsmb_mbatch(model_name, pretrained, n_top=100):

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable
    from sklearn.decomposition import PCA

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained
    n_top = int(n_top)

    print(f"Setting up path for {model_name}.")        
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model weights!")
    if pretrained:
        model = torch.load(join(pretrained_path, model_name, "model_pt"))
    else:
        model = torch.load(join(untrained_path, model_name, "model_pt"))        

    # -------------------

    print(f"Total number of modules: { len(list(model.children())) }")

    batch_size = 100
    N_images = len(imnames)

    counter = 0
    d2smb_all = []    # only compute D_2's
    Rsmb_all = []   # the corresponding singularvalues (from large to small)
    # load full-batch input data
    input_fbatch = get_full_batch()
    for module_idx in tqdm(range(len(list(model.children())))):
        try:
            len_module = len(list(model.children())[module_idx])
            for layer_idx in range(len_module):

                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )

                backbone = Backbone(model, module_idx, layer_idx).to(dev)
                backbone.eval()

                Rs = []
                for i in tqdm(range(N_images//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))      
                    output -= output.mean(0)                    
                    output = output.detach().numpy()

                    # scklearn PCA
                    pca = PCA(n_top)
                    pca.fit(output)  # fit to data
                    Rn = pca.explained_variance_

                    Rs.append(Rn)

                Rs = np.stack(Rs)
                Rsmb_all.append(Rs)

                del backbone
                counter += 1    
                #print(f"try {counter}")
        
        except:
            layer_idx = 0
            print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
            layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
            
            if "backbone" in locals():
                del backbone
            backbone = Backbone(model, module_idx).to(dev)
            backbone.eval()

            Rs = []
            for i in tqdm(range(N_images//batch_size)):
                input_tensor = get_batch(i,batch_size)
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))      
                output -= output.mean(0)                    
                output = output.detach().numpy()

                # scklearn PCA
                pca = PCA(n_top)
                pca.fit(output)  # fit to data
                Rn = pca.explained_variance_

                Rs.append(Rn)

            Rs = np.stack(Rs)
            Rsmb_all.append(Rs)

            del backbone
            counter += 1
            #print(f"except {counter}")
            
    print(f"Rsmb computation complete for {counter} layers!")

    Rsmb_all = np.stack(Rsmb_all)
    np.save(join(emb_path,f'Rsmb_n_top={n_top}_layerwise.npy'), Rsmb_all)   # minibatch

# get d2 from full batch, analysis for top n_top PCs
"""
def snr_d2_fbatch(model_name, pretrained, n_top=100):

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable
    from sklearn.decomposition import PCA

    global output_fbatch, Un, Rn, Vn, d2, backbone, model, Backbone
    global input_fbatch
    global d2s_all, Rsfb_all

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained
    n_top = int(n_top)

    print(f"Setting up path for {model_name}.")        
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model weights!")
    if pretrained:
        model = torch.load(join(pretrained_path, model_name, "model_pt"))
    else:
        model = torch.load(join(untrained_path, model_name, "model_pt"))        

    # -------------------
    
    d2_file = f"d2sfb_n_top={n_top}_layerwise.npy"
    print(f"Computation for {d2_file}!")
    if os.path.isfile( join(emb_path, d2_file) ):
        print(f"{d2_file} computed already, no computation required.")
        return

    else:
        print(f"Total number of modules: { len(list(model.children())) }")

        counter = 0
        d2s_all = []    # only compute D_2's
        Rsfb_all = []   # the corresponding singularvalues (from large to small)
        # load full-batch input data
        input_fbatch = get_full_batch()
        for module_idx in tqdm(range(len(list(model.children())))):
        #for module_idx in tqdm([2]):
            try:
                len_module = len(list(model.children())[module_idx])
                for layer_idx in range(len_module):
                #for layer_idx in tqdm([2]):

                    print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                    
                    backbone = Backbone(model, module_idx, layer_idx).to(dev)
                    backbone.eval()
                    
                    with torch.no_grad():
                        output_fbatch = backbone(input_fbatch.to(dev))
                    del backbone
                    output_fbatch -= output_fbatch.mean(0)  # center manifold
                    output_fbatch = output_fbatch.detach().numpy()
                    # sparse SVD from scipy (but need to pass in full batch image)
                    print("Starting SVD")
                    # principal directions
                    #_,Rn,Vn = scipy.sparse.linalg.svds(output_fbatch.detach().numpy(), k=n_top, which='LM', return_singular_vectors=True)
                    # principal components
                    #Un,Rn, _ = scipy.sparse.linalg.svds(output_fbatch.detach().numpy(), k=n_top, which='LM', return_singular_vectors=True)
                    #Un = Un @ np.diag(Rn)
                    # arrange singularvalues from large to small
                    #iis = np.argsort(Rn)[::-1]

                    # scklearn PCA
                    pca = PCA(n_top)
                    pca.fit(output_fbatch)  # fit to data
                    del output_fbatch
                    Rn = pca.explained_variance_
                    Vn = pca.components_

                    d2s = []
                    for eidx in range(len(Rn)):
                        d2s.append(compute_dq(Vn[eidx,:], 2))
                        #d2s.append(compute_dq(Un[:,eidx], 2))
                    d2s_all.append(d2s)
                    #Rsfb_all.append(Rn[iis])
                    Rsfb_all.append(Rn)

                    counter += 1    
                    print(f"try {counter}")
            
            except:
                layer_idx = 0
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                if "backbone" in locals():
                    del backbone
                backbone = Backbone(model, module_idx).to(dev)
                backbone.eval()

                with torch.no_grad():
                    output_fbatch = backbone(input_fbatch.to(dev))
                del backbone
                output_fbatch -= output_fbatch.mean(0)
                output_fbatch = output_fbatch.detach().numpy()
                print("Starting SVD except")
                # principal component
                #_,Rn,Vn = scipy.sparse.linalg.svds(output_fbatch.detach().numpy(), k=n_top, which='LM', return_singular_vectors=True)
                # principal direction
                #Un,Rn, _ = scipy.sparse.linalg.svds(output_fbatch.detach().numpy(), k=n_top, which='LM', return_singular_vectors=True)
                #Un = Un @ np.diag(Rn)
                #iis = np.argsort(Rn)[::-1]

                # scklearn PCA
                pca = PCA(n_top)
                pca.fit(output_fbatch)  # fit to data
                del output_fbatch
                Rn = pca.explained_variance_
                Vn = pca.components_

                d2s = []
                for eidx in range(len(Rn)):
                    d2s.append(compute_dq(Vn[eidx,:], 2))
                    #d2s.append(compute_dq(Un[:,eidx], 2))
                d2s_all.append(d2s)
                #Rsfb_all.append(Rn[iis])
                Rsfb_all.append(Rn)

                counter += 1
                print(f"except {counter}")
                
        print(f"d2 computation complete for {counter} layers!")

        d2s_all = np.stack(d2s_all)
        Rsfb_all = np.stack(Rsfb_all)
        # save correlation dimension D_2
        #np.save(join(emb_path,f'd2s_layerwise.npy'), d2s_all)   # minibatch
        np.save(join(emb_path,f'd2sfb_n_top={n_top}_layerwise.npy'), d2s_all)   # fullbatch
        np.save(join(emb_path,f'Rsfb_n_top={n_top}_layerwise.npy'), Rsfb_all)   # fullbatch
"""


# layer sparsity
def layer_sparsity(model_name, pretrained, threshold=1e-4):

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable

    #global input_fbatch, output_fbatch, Un, Rn, Vn, d2, backbone

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained

    print(f"Setting up path for {model_name}.")        
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model weights!")
    if pretrained:
        model = torch.load(join(pretrained_path, model_name, "model_pt"))
    else:
        model = torch.load(join(untrained_path, model_name, "model_pt"))        

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

    # test batch (this is not rlly needed as well)
    batch_size = 10
    layers = []
    for module_idx in tqdm(range(len(list(model.children())))):
        try:
            len_module = len(list(model.children())[module_idx])
            for layer_idx in range(len_module):
                
                backbone = Backbone(model, module_idx, layer_idx)
                backbone.to(dev).eval()

                # Get test batch
                input_tensor = get_batch(0,batch_size)
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                layers.append(list(model.children())[module_idx][layer_idx])

                print(output.shape[1])
        except:
            layer_idx = 0
            layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
            
            backbone = Backbone(model, module_idx)
            backbone.to(dev).eval()

            # Get test batch
            input_tensor = get_batch(0,batch_size)
            with torch.no_grad():
                output = backbone(input_tensor.to(dev))
                print(output.shape[1])
            try:
                layers.append(list(model.children())[module_idx][layer_idx])
            except:
                layers.append('avgpool')

    print(f"Total number of layers: {len(layers)}")

    # -------------------
    
    fname = f"sparsity_{threshold}_layerwise.npy"
    print(f"Computation for {fname}!")
    if os.path.isfile( join(emb_path, fname) ):
        print(f"{fname} computed already, no computation required.")
        return

    else:
        print(f"Total number of modules: { len(list(model.children())) }")

        counter = 0
        sparsity_all = []
        # load full-batch input data
        input_fbatch = get_full_batch()
        for module_idx in tqdm(range(len(list(model.children())))):
            try:
                len_module = len(list(model.children())[module_idx])
                for layer_idx in range(len_module):

                    print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                    
                    backbone = Backbone(model, module_idx, layer_idx)
                    backbone.to(dev).eval()
                
                    output_fbatch = backbone(input_fbatch.to(dev))
                    sparsity = np.sum(np.abs(output_fbatch.detach().numpy()) < threshold)/np.prod(output_fbatch.shape)
                    sparsity_all.append(sparsity)
                    print(sparsity)

                    del output_fbatch
                    counter += 1    

            except:
                layer_idx = 0
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                backbone = Backbone(model, module_idx)
                backbone.to(dev).eval()

                output_fbatch = backbone(input_fbatch.to(dev))
                sparsity = np.sum(np.abs(output_fbatch.detach().numpy()) < threshold)/np.prod(output_fbatch.shape)
                sparsity_all.append(sparsity)
                print(sparsity)

                del output_fbatch
                counter += 1

        print(f"d2 computation complete for {counter} layers!")

    # save correlation dimension D_2
    print(sparsity_all)
    np.save(join(emb_path,f'sparsity_{threshold}_layerwise.npy'), sparsity_all)   # fullbatch


def snr_submit(*args):
    from qsub import qsub, job_divider, project_ls

    # get all appropriate networks
    """
    #nets_with_backbone = ["resnet", "resnext", "vgg", "alexnet", "squeeze", "efficient"]
    nets_with_backbone = ["resnet", "resnext", "alexnet", "squeeze", "efficient"]
    net_names = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv"))    # locally available pretrained nets
    models = []
    for model_name in net_names.loc[:, "model_name"]:
        matches = 0
        for net_type in nets_with_backbone:
            if net_type in model_name.lower():
                matches += 1
        if matches > 0:
            models.append(model_name)
    """
    # small models
    #models = ["alexnet", "resnet18"]
    #models = ["resnet18"]

    # medium models
    """
    models = ["alexnet", "resnet18", "resnet34", "resnet50", 
              "resnext50_32x4d",
              "squeezenet1_0", "squeezenet1_1",
              "wide_resnet50_2"]
    """

    # large models
    # 
    models = ["resnet152"]
    """
    models = ["resnet101", 
              "resnext101_32x8d", 
              "wide_resnet101_2"]
    """
    
    pretrained_ls = [True, False]
    pbs_array_data = [(model_name, pretrained)
                      for model_name in models
                      for pretrained in pretrained_ls
                      #if not os.path.isfile(join(root_data,"pretrained_workflow/pretrained_dnns",model_name,"manifold",fname,"css_layerwise.npy"))
                      ]

    print(len(pbs_array_data))
    print(pbs_array_data)
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             #path=join(root_data, "macaque_stimuli"),
             path=join(root_data, "macaque_stimuli", "pure_rsmb"),
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='47:59:59',
             #walltime='23:59:59',   # small/medium
             mem='32GB')
             #mem='24GB')
             #mem='20GB')
             #mem='16GB')
             #mem='8GB') 

# ---------------------- Plotting ----------------------

# creates two plots
def snr_metric_plot(metric0, metric1, metric2, metric3, small=True, log=True):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcl
    import pubplot as ppt
    c_ls = list(mcl.TABLEAU_COLORS.keys())

    assert "d2_" in metric0
    #global metric_dict
    global metric0_data, metric1_data, metric2_data, metric3_data, all_models
    #global dq_data
    #global manifold_dim, Rs_all

    """
    Fig 1:
    Plots the a selected metric1 (based on metric_dict) vs layer and SNR vs layer,
    for dq, the pc_idx PC needs to be selected

    Fig 2:
    Plots the a selected metric2 (based on metric_dict) vs layer and a scatter plot between SNR and D_2
    """
    small = literal_eval(small) if isinstance(small, str) else small
    log = literal_eval(log) if isinstance(log, str) else log

    # Plot settings
    fig_size = (9.5 + 1.5,7.142/2)
    markers = ["o", "v", "s", "p", "*", "P", "H", "D", "d", "+", "x"]
    transparency, lstyle = 0.4, "--"
    
    # always include D_2 in metric_0
    dq_ls = metric0.split("_") if "d2" in metric0 else []
    dq_filename = f"d2s_layerwise" if len(dq_ls) > 0 else None
    metric_dict = {"SNR"        : "SNRs_layerwise",
                   "D"          : "Ds_layerwise",
                   "dist_norm"  : "dist_norm_layerwise",
                   "css"        : "css_layerwise",        
                   "ED"         : "EDs_layerwise_all",   
                   "bias"       : "biases_layerwise_all"
                   }

    name_dict = {"SNR"        : "SNR",
                 "D"          : r'$D$',
                 "dist_norm"  : "Distance norm",
                 "css"        : "Centre subsplace",
                 "ED"         : "ED",
                 "bias"       : "Bias"              
                 }   

    if "d2_" in metric0 and metric0 != "d2_avg":
        metric_dict[metric0] = dq_filename
        name_dict[metric0] = r'$D_2$' 
        pc_idx = int(metric0.split("_")[1])

    # average dq's over the top n_top PCs
    n_top = 25
    if metric0 == "d2_avg":
        name_dict[metric0] = r'$D_2$'

    assert metric1 in metric_dict.keys(), "metric1 not in dictionary!"
    assert metric2 in metric_dict.keys(), "metric2 not in dictionary!" 
    assert metric3 in metric_dict.keys(), "metric3 not in dictionary!" 

    # get available networks
    all_models = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv")).loc[:,"model_name"]

    """
    model_names = ["alexnet", "resnet18", "resnet34", "resnet50", "resnet101", 
                   "resnext50_32x4d", "resnext101_32x8d", 
                   "wide_resnet50_2", "wide_resnet101_2", 
                   "squeezenet1_1"]
    """
 
    if small:
        # small models
        """
        model_names = ["alexnet", 
                  "resnet18", "resnet34", "resnet50",
                  "resnext50_32x4d",
                  "wide_resnet50_2"]
        """
        model_names = ["alexnet"]

    else:
        # large models
        model_names = ["resnet101", "resnet152", 
                       "resnext101_32x8d",
                       "squeezenet1_0", "squeezenet1_1", 
                       "wide_resnet101_2"]

    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+2)[::-1]

    # need to demonstrate for pretrained and random DNNs
    pretrained_ls = [True, False]
    lstyle_ls = ["-", "--"]

    # m-shot learning 
    #ms = np.arange(1,11)
    ms = [1]

    # --------------- Plot 1 ---------------
    for midx, m in enumerate(tqdm(ms)):
        plt.rc('font', **ppt.pub_font)
        plt.rcParams.update(ppt.plot_sizes(False))
        fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=False,figsize=fig_size)
        for pretrained in pretrained_ls:
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            for nidx in range(len(model_names)):
            
                model_name = model_names[nidx]
                # set paths
                init_path = join(root_data, "macaque_stimuli")
                manifold_path = join(init_path, model_name, pretrained_str)

                # manifolds_all dimension from snr_components() above
                manifold_dim = np.load(join(manifold_path, "manifold_dim.npy"))

                # load data     
                emb_path = manifold_path
                if metric0 != "d2_avg":
                    metric0_data = np.load( join(emb_path, f"{metric_dict[metric0]}.npy") ) 
                    # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
                    # load singularvalues
                    #Rs_all = np.load( join(emb_path, "Rs_layerwise.npy") )
                    # rank D_2 from eigenvectors based on the eigenvalues (linalg.svd seems to do this automatically)           
                    metric0_data = metric0_data[:,:,pc_idx]          
                                  
                else:
                    metric0_sample = np.load( join(emb_path, f"d2s_layerwise.npy") ) 
                    metric0_data = metric0_sample[:,:,:n_top].mean(-1)
                        
                if metric1 == "SNR":
                    metric1_data = np.load( join(emb_path, f"{metric_dict[metric1]}_m={m}.npy") )
                else:
                    metric1_data = np.load( join(emb_path, f"{metric_dict[metric1]}.npy") )

                # fractional layer
                total_layers = metric1_data.shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                    
                label = model_name if pretrained else "__nolegend__"
                ax1.plot(frac_layers, metric0_data.mean(-1), c=c_ls[nidx], alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle, label=label)
                ax2.plot(frac_layers, np.nanmean(metric1_data,(1,2)), c=c_ls[nidx], alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle, label=label)

                if midx == 0 and not pretrained:
                    print(f"{model_name} done!")

            for ax in [ax1,ax2]:        
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            ax1.grid(alpha=transparency,linestyle=lstyle); ax2.grid(alpha=transparency,linestyle=lstyle) 

            ax1.set_ylim((0,1))
            #ax2.set_ylim((0.05,0.45))

            ax1.legend(frameon=False, ncol=2, fontsize=9.5)
            #ax2.set_yscale('log')
            ax2.ticklabel_format(style="sci", scilimits=(0,1), axis="y" )

            ax1.set_ylabel(name_dict[metric0])
            ax2.set_ylabel(name_dict[metric1])
            ax1.set_xlabel("Fractional depth")
            ax2.set_xlabel("Fractional depth")
            #plt.show()

            #fig_path = "/project/dnn_maths/project_qu3/fig_path"
            fig_path = join(root_data,"figure_ms/pretrained-fewshot-old-small") if small else join(root_data,"figure_ms/pretrained-fewshot-old-large")
            if not os.path.isdir(fig_path): os.makedirs(fig_path)    
            plt.savefig(join(fig_path, f"pretrained_m={m}_{metric0}_{metric1}-vs-layer.pdf") , bbox_inches='tight')
            if midx == 0:
                print(f"Plot 1 saved!")

        # --------------- Plot 2 ---------------
        for midx, m in enumerate(tqdm(ms)):
            plt.rc('font', **ppt.pub_font)
            plt.rcParams.update(ppt.plot_sizes(False))
            fig, ((ax3,ax4)) = plt.subplots(1, 2,sharex = True,sharey=False,figsize=fig_size)
            for pretrained in pretrained_ls:
                pretrained_str = "pretrained" if pretrained else "untrained"
                lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]

                for nidx in range(len(model_names)):
                    model_name = model_names[nidx]
                    # set paths
                    init_path = join(root_data, "macaque_stimuli")
                    manifold_path = join(init_path, model_name, pretrained_str)

                    # load data     
                    emb_path = join(init_path, manifold_path)
                    for metric_name in ["metric2", "metric3"]:                        
                        fname = metric_dict[locals()[metric_name]] + f"_m={m}.npy" if locals()[metric_name] == "SNR" else metric_dict[locals()[metric_name]] + ".npy"
                        locals()[f"{metric_name}_data"] = np.load( join(emb_path, fname) )

                    dq_name = metric0
                    if metric0 != "d2_avg":
                        dq_data = np.load( join(emb_path, f"{metric_dict[dq_name]}.npy") ) 
                        dq_data = dq_data[:,:,pc_idx]                        
                    else:
                        dq_sample = np.load( join(emb_path, f"d2s_layerwise.npy") ) 
                        dq_data = dq_sample[:,:,:n_top].mean(-1)
                    # fractional layer
                    total_layers = locals()['metric3_data'].shape[0]
                    frac_layers = np.arange(0,total_layers)/(total_layers-1)
                        
                    if "d2" not in metric2 and metric2 != 'D':
                        ax3.plot(frac_layers, np.nanmean(locals()['metric2_data'],(1,2)), 
                                 c=c_ls[nidx], alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle, label=model_name)
                    else:
                        #ax3.plot(frac_layers, metric2_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=model_name)
                        ax3.plot(frac_layers, locals()['metric2_data'].mean(-1), 
                                 c=c_ls[nidx], alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle, label=model_name)

                    # only scatter plot the deeper layers (can modify)
                    deep_layers = np.where(frac_layers >= 0)
                    if log:
                        ax4.scatter(dq_data.mean(-1)[deep_layers], np.log(np.nanmean(locals()['metric3_data'],(1,2))[deep_layers]), 
                                    c=c_ls[nidx], marker=markers[nidx], alpha=0.6)
                    else:
                        ax4.scatter(dq_data.mean(-1)[deep_layers], np.nanmean(locals()['metric3_data'],(1,2))[deep_layers], 
                                    c=c_ls[nidx], marker=markers[nidx], alpha=0.6)
                    if midx == 0:
                        print(f"{model_name} done!")

                for ax in [ax3,ax4]:        
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                ax3.grid(alpha=transparency,linestyle=lstyle); ax4.grid(alpha=transparency,linestyle=lstyle) 
                #ax4.set_xlim((0,1))
                #ax4.set_ylim((0.05,0.4))

                #ax3.legend(frameon=False, ncol=2, fontsize=10)
                #ax4.set_xscale('log')
                ax4.ticklabel_format(style="sci" , scilimits=(-3,0),  axis="y" )

                ax3.set_ylabel(name_dict[metric2])
                ax4_ylabel = "log({})".format(name_dict[metric3]) if log else name_dict[metric3]
                ax4.set_ylabel(ax4_ylabel)
                ax3.set_xlabel("Fractional depth")
                ax4.set_xlabel(r"$D_2$")
                #plt.show()

                plt.savefig(join(fig_path, f"pretrained_m={m}_{metric2}_{metric3}-dq_scatter.pdf") , bbox_inches='tight')
                if midx == 0:
                    print(f"Plot 2 saved!")

# load metrics from network
def load_metric(model_name, pretrained:bool, metric_name, m):
    metric_dict = {"SNR"        : "SNRs_layerwise",
                   "D"          : "Ds_layerwise",
                   "dist_norm"  : "dist_norm_layerwise",
                   "css"        : "css_layerwise",         
                   "bias"       : "biases_layerwise",
                   "R"          : "Rsmb_n_top=100_layerwise"
                   }

    # check if metric_name includes "d2_"
    if "d2_" in metric_name:
        dq_ls = metric_name.split("_") if "d2" in metric_name else []
        dq_filename = f"d2smb_n_top=100_layerwise" if len(dq_ls) > 0 else None
        metric_dict[metric_name] = dq_filename

    # model path
    init_path = join(root_data, "macaque_stimuli")
    pretrained_str = "pretrained" if pretrained else "untrained"
    manifold_path = join(init_path, model_name, pretrained_str)    
    emb_path = manifold_path

    if "d2_" in metric_name:
        # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
        metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}.npy") )            
    elif metric_name == "SNR":
        metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}_m={m}.npy") )
    else:
        metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}.npy") )

    return metric_data

# minibatch d2 version
def snr_metric_plot_2(*args, small=True, log=False, n_top = 5):
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
    global metric_data_all, metric_og, metric_name, var_percentage, metric_names, metric_R

    """
    Fig 1:
    Plots the a selected metric1 (based on metric_dict) vs layer and SNR vs layer,
    for dq, the pc_idx PC needs to be selected

    Fig 2:
    Plots the a selected metric2 (based on metric_dict) vs layer and a scatter plot between SNR and D_2
    """

    small = literal_eval(small) if isinstance(small, str) else small
    log = literal_eval(log) if isinstance(log, str) else log

    # Plot settings
    #fig_size = (9.5 + 1.5,7.142/2) # 1 by 2
    #fig_size = (9.5 + 1.5,7.142) # 2 by 2
    fig_size = (11/2*3,7.142+2) # 2 by 3
    markers = ["o", "v", "s", "p", "*", "P", "H", "D", "d", "+", "x"]
    transparency, lstyle = 0.4, "--"

    name_dict = {"SNR"        : "SNR",
                 "D"          : 'Dimension',
                 "dist_norm"  : "Signal",
                 "css"        : "Signal-noise overlap",
                 "bias"       : "Bias",      
                 "R"          : 'Cumulative variance'        
                 }   

    metric_names = [metric_name for metric_name in args]
    print(f"metric list: {metric_names}")
    for metric_name in metric_names:
        if "d2_" in metric_name:
            if metric_name != "d2_avg":
                pc_idx = int(metric_name.split("_")[1])
            name_dict[metric_name] = r'$D_2$' 

    # get available networks
    #all_models = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv")).loc[:,"model_name"]

    if small:
        # small models
        """
        model_names = ["alexnet", 
                  "resnet18", "resnet34", "resnet50",
                  "resnext50_32x4d",
                  "wide_resnet50_2"]
        """
        model_names = ["alexnet","resnet101"]
        #model_names = ["resnet18"]

    else:
        # large models
        model_names = ["resnet101", "resnet152", 
                       "resnext101_32x8d",
                       "squeezenet1_0", "squeezenet1_1", 
                       "wide_resnet101_2"]

    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+2)[::-1]

    # need to demonstrate for pretrained and random DNNs
    pretrained_ls = [True, False]
    #pretrained_ls = [False]
    lstyle_ls = ["-", "--"]

    # m-shot learning 
    #ms = np.arange(1,11)
    ms = [1]

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
                    if "d2_" not in metric_name and metric_name != "R":
                        metric_data_all[metric_name] = load_metric(model_name, pretrained, metric_name, m)
                    #cumulative variance explained by n_top PCs
                    elif metric_name == "R":
                        metric_R = load_metric(model_name, pretrained, metric_name, m)
                        if "d2_avg" in metric_names:
                            # (17, 32, 100)
                            var_percentage = metric_R[:,:,:n_top].cumsum(-1)/metric_R.sum(-1)[:,:,None]
                        else:
                            var_percentage = metric_R[:,:,pc_idx]/metric_R.sum(-1)
                        metric_data_all[metric_name] = var_percentage                        

                    # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are     
                    elif metric_name == "d2_avg":
                        metric_og = load_metric(model_name, pretrained, metric_name, m)
                        d2_data = metric_og[:,:,:n_top].mean(-1)
                        d2_name = metric_name
                        metric_data_all[metric_name] = d2_data
                    else:
                        metric_og = load_metric(model_name, pretrained, metric_name, m)   
                        d2_data = metric_og[:,:,pc_idx]
                        d2_name = metric_name
                        metric_data_all[metric_name] = d2_data

                # fractional layer
                total_layers = list(metric_data_all.items())[0][1].shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                #frac_layers = np.arange(0,total_layers)
                    
                for metric_idx, metric_name in enumerate(metric_names[:5]):
                    metric_data = metric_data_all[metric_name]
                    if metric_data.ndim == 2:
                        metric_data_mean = metric_data.mean(-1)
                    else:
                        metric_data_mean = np.nanmean(metric_data,(1,2))
                    axs[metric_idx].plot(frac_layers, metric_data_mean, 
                                         c=color, alpha=trans_ls[nidx], marker=markers[nidx], linestyle=lstyle)

                # --------------- Plot 2 (lower) ---------------

                # scatter plot
                metric_name_y = metric_names[-1]
                metric_data_y = np.nanmean(metric_data_all[metric_name_y],(1,2))

                # only scatter plot the deeper layers (can modify)
                deep_layers = np.where(frac_layers >= 0)
                #print(np.nanmean(locals()['metric3_data'],(1,2))[deep_layers].shape)    # delete
                if log:
                    axs[-1].scatter(np.log(d2_data[deep_layers].mean(-1)), np.log(metric_data_y[deep_layers]), 
                                c=color, marker=markers[nidx], alpha=0.6)
                else:
                    axs[-1].scatter(d2_data[deep_layers].mean(-1), metric_data_y[deep_layers], 
                                c=color, marker=markers[nidx], alpha=0.6)

                if midx == len(ms)-1 and pretrained_idx == 0:
                    print(f"{model_name} saved!")

        # --------------- Plot settings ---------------
        for ax_idx, ax in enumerate(axs):        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=transparency,linestyle=lstyle)
            if ax_idx < 5:
                ax.set_xlabel("Fractional depth", fontsize=label_size)
                #ax.set_ylabel(name_dict[metric_names[ax_idx]], fontsize=label_size)
                ax.set_title(name_dict[metric_names[ax_idx]], fontsize=title_size)
            if ax_idx >= 0:
                ax.ticklabel_format(style="sci" , scilimits=(0,100),  axis="y" )
            # ticklabel size
            ax.xaxis.set_tick_params(labelsize=label_size)
            ax.yaxis.set_tick_params(labelsize=label_size)

        # scatter plot
        axs[-1].arrow(0.825, 0.05, -0.05, 0.075, 
                      alpha=0.7, color="red", 
                      linestyle="--", head_width=0.006, width=0.001)
        axs[-1].set_xlabel(name_dict[d2_name], fontsize=label_size)
        axs[-1].set_ylabel(name_dict[metric_name_y], fontsize=label_size)

        # legends
        for nidx, model_name in enumerate(model_names):
            label = model_name
            axs[0].plot([], [], c=c_ls[nidx], alpha=trans_ls[nidx], marker=markers[nidx], label=label)

        for pretrained_idx, pretrained in enumerate(pretrained_ls):
            pretrained_str = "pretrained" if pretrained else "untrained"
            lstyle = lstyle_ls[0] if pretrained else lstyle_ls[1]
            label = pretrained_str
            axs[0].plot([],[],c="gray", linestyle=lstyle, label=label)

        axs[0].set_ylim((0.6,1))
        axs[0].legend(frameon=False, ncol=2, loc="upper center", fontsize=legend_size)
        #ax2.set_yscale('log')
        axs[1].ticklabel_format(style="sci", scilimits=(0,1), axis="y" )

        # adjust vertical space
        plt.subplots_adjust(hspace=0.4)

        """
        ax3.set_ylabel(name_dict[metric2])
        ax3_ylabel = "log({})".format(name_dict[dq_name]) if log else name_dict[dq_name]            
        ax4_ylabel = "log({})".format(name_dict[metric3]) if log else name_dict[metric3]
        ax4.set_xlabel(ax3_ylabel)
        ax4.set_ylabel(ax4_ylabel)
        """
        #plt.show()

        # --------------- Save figure ---------------
        fig_path = join(root_data,"figure_ms/pretrained-fewshot-old-small") if small else join(root_data,"figure_ms/pretrained-fewshot-old-large")
        if not os.path.isdir(fig_path): os.makedirs(fig_path)    
        plt.savefig(join(fig_path, f"pretrained_m={m}_metric_all.pdf") , bbox_inches='tight')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])


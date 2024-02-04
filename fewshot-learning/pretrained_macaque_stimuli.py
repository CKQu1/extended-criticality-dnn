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
imnames = sorted(imnames)   # a total of 64 concepts with 50 examples each, this groups them into the respective input classes

def get_batch(i, batch_size):
    from PIL import Image
#     ims = imnames[class_id]

    # Image preprocessing
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

    # Image preprocessing
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

# preserves shape
class Backbone_og(torch.nn.Module):
    def __init__(self, model, module_idx, layer_idx=None):
        super(Backbone_og, self).__init__()
        self.layer_idx = layer_idx
        self.pre_features = torch.nn.Sequential(*list(model.children())[:module_idx])
        if layer_idx:
            self.features = torch.nn.Sequential(*list(model.children())[module_idx][:layer_idx])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        if self.layer_idx:
            x = self.features(x)
        return x        

# extracts all features
class Backbone_full(torch.nn.Module):
    def __init__(self, model, module_idx, layer_idx=None):
        super(Backbone_og, self).__init__()
        self.layer_idx = layer_idx
        self.pre_features = torch.nn.Sequential(*list(model.children())[:module_idx])
        if layer_idx:
            self.features = torch.nn.Sequential(*list(model.children())[module_idx][:layer_idx])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        if self.layer_idx:
            x = self.features(x)
        return x         

# ---------------------- SNR Theory ----------------------

from utils_dnn import compute_dq

def load_model(model_name, pretrained):
    global model    # delete
    if pretrained:
        model = torch.load(join(pretrained_path, model_name, "model_pt"))
    else:
        model = torch.load(join(untrained_path, model_name, "model_pt")) 
    return model


def load_shared_info(model_name):
    assert os.path.isfile(join(root_data, "macaque_stimuli", model_name, "shared_info")), f'shared_info does not exist for {model_name}'
    return pd.read_csv(join(root_data, "macaque_stimuli", model_name, "shared_info"))    


# get the direct error from few-shot learning
def get_error(model_name, pretrained):

    if 'resnet' in model_name or 'resnext' in model_name:
        backbone = WideResNetBackbone(model, 1)
        random_projection=False
    elif 'vgg' in model_name:
        backbone = VGGBackbone(model)
    elif 'alexnet' in model_name:
        backbone = AlexnetBackbone(model)
    elif 'squeeze' in model_name:
        backbone = SqueezeBackbone(model)
    elif 'efficient' in model_name:
        backbone = model
        backbone.N = 1000
        random_projection=False
    #     backbone = EfficientNetBackbone(model)
    backbone.to(dev).eval()

    N = 2048
    K = 64
    P = 50
    batch_size = 10
    n_classes = len(imnames)

    manifolds = []
    # for class_id in tqdm(range(n_classes)):
    for i in tqdm(range(K*P//batch_size)):
        input_tensor = get_batch(i,batch_size)
        with torch.no_grad():
            output = backbone(input_tensor.cuda())
        manifolds.append((output@O).cpu().numpy())
    manifolds = np.stack(manifolds).reshape(K,P,N)

    # Compute error
    m = 5
    n_avg = 50

    err_all = np.zeros((len(manifolds),len(manifolds)))
    for a in tqdm(range(len(manifolds))):
        Xa = manifolds[a]
        for b in range(len(manifolds)):
            Xb = manifolds[b]

            errs = []
            for _ in range(n_avg):
                perma = np.random.permutation(len(Xa))
                permb = np.random.permutation(len(Xb))

                xa,ya = np.split(Xa[perma],(m,))
                xb,yb = np.split(Xb[permb],(m,))
                w = (xa-xb).mean(0)
                mu = (xa+xb).mean(0)/2

                h = ya@w - w@mu
                err = (h<0).mean()
                errs.append(err)
            err_all[a,b] = np.mean(errs)
    np.fill_diagonal(err_all,np.nan)

    # save error data
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    np.save(join(emb_path,"err_all"), err_all)


def get_num_layers(model_name, *args):
    """
    Get the total number of analyzable layers of the network, knowing this will allow parallelization over the layers
    and not just over the networks.
    """

    global df, backbone, module_idx, layer_idx, has_child

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable

    emb_path = join(root_data, "macaque_stimuli", model_name)
    if not os.path.isfile(join(emb_path, "shared_info")):

        print(f"Setting up path for {model_name}.")            

        print("Loading model!")
        pretrained = True
        model = load_model(model_name, pretrained)      

        # finds out the total amount of layers needed for evaluation (this is not rlly needed as well)    
        batch_size = 2

        # create df
        dict_ls = []
        df = pd.DataFrame(columns=['module_idx', 'layer_idx', 'has_child'])    

        layers = []
        for module_idx in tqdm(range(len(list(model.children())))):
            ########## Case 1: when the child has submodules ##########
            try:
                has_child = True
                len_module = len(list(model.children())[module_idx])
                for layer_idx in range(len_module):
                    
                    backbone = Backbone(model, module_idx, layer_idx)
                    backbone.to(dev).eval()

                    # Get test batch
                    input_tensor = get_batch(0,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    layers.append(list(model.children())[module_idx][layer_idx])

                    ##### self info #####
                    print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                    has_weight = ('weight' in dir(list(model.children())[module_idx][layer_idx]) )
                    if has_weight:
                        weight_shape = tuple(list(model.children())[module_idx][layer_idx].weight.shape)
                        print(f'which has weight with shape {weight_shape}')
                    else:
                        weight_shape = None
                        print(f'has no weight')

                    # save to df
                    row_data = {'module_idx': module_idx, 'layer_idx': layer_idx, 'has_child': has_child,
                                'has_weight': has_weight, 'weight_shape': weight_shape}
                    dict_ls.append(row_data)                   

            ########## Case 2: when the child does not have submodules ##########
            except:
                has_child = False
                layer_idx = 0
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                backbone = Backbone(model, module_idx)
                backbone.to(dev).eval()

                # Get test batch
                input_tensor = get_batch(0,batch_size)
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                try:
                    layers.append(list(model.children())[module_idx])
                except:
                    layers.append('avgpool')

                ##### self info #####
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )   
                has_weight = ('weight' in dir(list(model.children())[module_idx]) )
                if has_weight:
                    weight_shape = tuple(list(model.children())[module_idx].weight.shape)
                    print(f'which has weight with shape {weight_shape}')
                else:
                    weight_shape = None
                    print(f'has no weight')         

                # save to df
                row_data = {'module_idx': module_idx, 'layer_idx': layer_idx, 'has_child': has_child,
                            'has_weight': has_weight, 'weight_shape': weight_shape}
                dict_ls.append(row_data)                      
                

        print(f"Total number of layers: {len(layers)}")    

        num_layers = len(layers)

        # save df
        df = pd.DataFrame.from_dict(dict_ls)    
        if not os.path.isdir(emb_path): os.makedirs(emb_path)
        df.to_csv(join(emb_path, "shared_info"), index=False)

        print(f"{model_name} has total analyzable layers {len(layers)}")

    else:
        print(f"{model_name} done")


# only for pytorch, make sure the input wmat has correct dimensions
def tensor_svd_dq(wmat, q):
    import numpy.linalg as LA
    from utils_dnn import D_q_all
    assert wmat.ndim > 1, "wmat must have ndim greater than 1"

    """
    if wmat.ndim == 2:  # from dq_analysis/wmat_fcn.py
                                
        r_nan_count = 0; l_nan_count = 0
        ##### right singvec #####   
        D_2ss = D_q_all(Vh.T, q)        
        
        r_nan_count = np.count_nonzero(~np.isnan(data))  
        r_d2_principal = D_2ss[0]      
        # case 1: average d2
        r_d2_means = D_2ss.mean()
        r_d2_stds = D_2ss.std()              

        ##### left singvec #####
        D_2ss = D_q_all(U, q)
        
        
        l_nan_count = np.count_nonzero(~np.isnan(data))  
        l_d2_principal = D_2ss[0]      
        # case 1: average d2
        l_d2_means = D_2ss.mean()
        l_d2_stds = D_2ss.std()    

        # case 1: average d2
        l_d2_means = D_2ss.mean()
        l_d2_stds = D_2ss.std()                         

    elif wmat.ndim == 3:
        pass
    elif wmat.ndim == 4:  # from dq_analysis/wmat_cnn.py

        d2_shape = [wshape[-2], wshape[-1]]

        d2_means = np.zeros(d2_shape)
        d2_stds = np.zeros(d2_shape)         

        nan_count = 0
        for csize1 in range(wshape[-2]):
            for csize2 in range(wshape[-1]):
                D_2ss = []

                # channel       
                wmat_ch = wmat[:,:,csize1,csize2] 
                # left eigenvector
                #if reig == 0:
                #    wmat_ch = wmat_ch.T                 
                _, singvals, Vh = LA.svd(wmat_ch)

                
                for i in range(len(singvals)):  
                    singvec = Vh[i,:]
                    IPR_2 = IPR(singvec ,2)
                    D_2s = np.log(IPR_2) / (1-q) / np.log(len(singvec))
                    
                    # option 1: do not include nan
                    #if not any(np.isnan(D_2s)): D_2ss.append(D_2s)
                    #else: nan_count += 1  

                    # option 2: include nan
                    D_2ss.append(D_2s)
                    if np.isnan(D_2s): 
                        nan_count += 1   

                    # get principal D_2
                    if i == 0:
                        d2_principal = D_2s                                     

                D_2ss = np.array(D_2ss)

                # case 1: average d2
                d2_means[csize1, csize2] = D_2ss.mean()
                d2_stds[csize1, csize2] = D_2ss.std()
    """

    wmat_shape = list(wmat.shape)
    if wmat.ndim == 2:
        U, singvals, Vh = LA.svd(wmat)

        r_D_2ss = D_q_all(Vh.T, q)
        l_D_2ss = D_q_all(U, q)   
    else:
        # notes
        """
        U, singvals, Vh = LA.svd(wmat.transpose(2,0,1))
        U, singvals, Vh = LA.svd(wmat.transpose(2,3,0,1))
        """

        wmat_shape_idxs = list(range(wmat.ndim))
        wmat_reshape_idxs = list(wmat_shape_idxs[2:]) + list(wmat_shape_idxs[:2])
        wmat_reshape = [wmat_shape[idx] for idx in wmat_reshape_idxs]
        U, singvals, Vh = LA.svd(wmat.transpose(wmat_reshape_idxs))
        l_shape = wmat_reshape[:wmat.ndim - 2] + [U.shape[-1]]
        r_shape = wmat_reshape[:wmat.ndim - 2] + [Vh.shape[-2]]
        r_D_2ss = np.zeros(r_shape); l_D_2ss = np.zeros(l_shape)
        if wmat.ndim == 3:
            for k1_idx in range(U.shape[0]):
                l_D_2ss[k1_idx] = D_q_all(U[k1_idx],q)
            for k1_idx in range(Vh.shape[1]):
                r_D_2ss[k1_idx] = D_q_all(Vh[k1_idx].T,q)
        elif wmat.ndim == 4:
            for k1_idx in range(U.shape[0]):
                for k2_idx in range(U.shape[1]):
                    l_D_2ss[k1_idx,k2_idx] = D_q_all(U[k1_idx,k2_idx],q)
            for k1_idx in range(Vh.shape[0]):
                for k2_idx in range(Vh.shape[1]):
                    r_D_2ss[k1_idx,k2_idx] = D_q_all(Vh[k1_idx,k2_idx].T,q)

        else:
            print("ndim greater than 4")
            return

    """    
    return r_d2_means, r_d2_stds, r_d2_principal, r_nan_count,\
           l_d2_means, l_d2_stds, l_d2_principal, l_nan_count
    """

    return r_D_2ss, l_D_2ss

#only for pytorch, make sure the input wmat has correct dimensions
def tensor_fft_svd_dq(wmat, input_shape, q):
    import numpy.linalg as LA
    from utils_dnn import D_q_all
    assert wmat.ndim > 2, "wmat must have ndim greater than 1"

    wmat_shape = list(wmat.shape)
    if wmat.ndim <= 4:

        wmat_shape_idxs = list(range(wmat.ndim))
        wmat_reshape_idxs = list(wmat_shape_idxs[2:]) + list(wmat_shape_idxs[:2])
        wmat_reshape = [wmat_shape[idx] for idx in wmat_reshape_idxs]

        transforms = np.fft.fft2(wmat.transpose(wmat_reshape_idxs), input_shape, axes=[0,1])
        U, singvals, Vh = LA.svd(transforms)

        tshape = list(transforms.shape)
        l_shape = tshape[:wmat.ndim - 2] + [U.shape[-1]]
        r_shape = tshape[:wmat.ndim - 2] + [Vh.shape[-2]]
        r_D_2ss = np.zeros(r_shape); l_D_2ss = np.zeros(l_shape)
        if wmat.ndim == 3:
            for k1_idx in range(U.shape[0]):
                l_D_2ss[k1_idx] = D_q_all(U[k1_idx],q)
            for k1_idx in range(Vh.shape[1]):
                r_D_2ss[k1_idx] = D_q_all(Vh[k1_idx].T,q)
        elif wmat.ndim == 4:
            for k1_idx in range(U.shape[0]):
                for k2_idx in range(U.shape[1]):
                    l_D_2ss[k1_idx,k2_idx] = D_q_all(U[k1_idx,k2_idx],q)
            for k1_idx in range(Vh.shape[0]):
                for k2_idx in range(Vh.shape[1]):
                    r_D_2ss[k1_idx,k2_idx] = D_q_all(Vh[k1_idx,k2_idx].T,q)

    else:
        print("ndim greater than 4")
        return

    return r_D_2ss, l_D_2ss    

# new version
def layerwise_snr_components(model_name, lidx, pretrained, n_top=100, replace=False):
    """
    Obtain the SNR and its associated geometrical metrics for one model at a time:
        - lidx (int): the lidx th layer in the model   

    Note that D_2 computed for the singular vector is for both the left and right singular vector
    """

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable
    from sklearn.decomposition import PCA    

    """
    global Us, manifolds_all, Rs_all, dists_all, PRs_all, css_all, SNRs_all
    global U, R, V, Rs, Us
    global d2s_all
    global layers
    global manifolds, manifolds_all
    global init_epoch, emb_path
    global model, module_idx, layer_idx, Backbone
    global input_fbatch, output_fbatch, Un, Rn, Vn, d2, backbone
    """

    global wmat, model, r_D_2ss, l_D_2ss, input_shape, df, backbone

    lidx = int(lidx)
    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained

    print(f"Setting up path for {model_name}.")        
    # where the desired manifolds.npy/manifolds_N={N}.npy, etc is located
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model!")
    model = load_model(model_name, pretrained)     

    # -------------------

    # same variables as in cnn_macaque_stimuli.py
    N = 2048
    N_dq = 500
    navg_dq = 10

    batch_size = 50
    #batch_size = 10
    K = 64
    P = 50

    # analysis for top 200 PCs
    #n_top = 50 # not used

    # loading file
    df = pd.read_csv(join(root_data, "macaque_stimuli", model_name, "shared_info")) 
    module_idx,layer_idx,has_child,has_weight,weight_shape = df.loc[lidx,:]         

    #################### 1. Compute singular vector multifractality ####################

    """
    multifrac_file1 = f"r_singvec_mean_lidx={lidx}.npy"
    multifrac_file2 = f"r_singvec_std_lidx={lidx}.npy"
    multifrac_file3 = f"r_singvec_principal_lidx={lidx}.npy"
    # left
    multifrac_file4 = f"l_singvec_mean_lidx={lidx}.npy"
    multifrac_file5 = f"l_singvec_std_lidx={lidx}.npy"
    multifrac_file6 = f"l_singvec_principal_lidx={lidx}.npy"  
    """

    multifrac_file1 = f"r_singvec_dq_lidx={lidx}.npy"  
    multifrac_file2 = f"l_singvec_dq_lidx={lidx}.npy"

    multifrac_file3 = f"r_fftsingvec_dq_lidx={lidx}.npy"  
    multifrac_file4 = f"l_fftsingvec_dq_lidx={lidx}.npy" 
    do_singvec_files = not os.path.isfile( join(emb_path, multifrac_file1) ) or not os.path.isfile( join(emb_path, multifrac_file2) ) or replace
    do_singvec_files = do_singvec_files or not os.path.isfile( join(emb_path, multifrac_file3) )  
    do_singvec_files = do_singvec_files or not os.path.isfile( join(emb_path, multifrac_file4) )
    if do_singvec_files:
        if has_weight:     
            if has_child:       
                wmat = list(model.children())[module_idx][layer_idx].weight  # could be any dimension
            else:
                wmat = list(model.children())[module_idx].weight
            wmat = wmat.detach().numpy()
            wshape = wmat.shape
            print(f"wmat shape: {wshape}, wmat ndim: {wmat.ndim}")

            if wmat.ndim > 1:              

                q = 2
                #r_d2_means, r_d2_stds, r_d2_principal, r_nan_count,\
                #l_d2_means, l_d2_stds, l_d2_principal, l_nan_count = tensor_svd(wmat, q)                

                #np.save(join(emb_path, multifrac_file1), d2_means) 
                #np.save(join(emb_path, multifrac_file2), d2_stds)                       
                #np.save(join(emb_path, multifrac_file3), d2_principal)  

                #if nan_count > 0:
                #    print(f"nan_count: {nan_count}")                  

                r_D_2ss, l_D_2ss = tensor_svd_dq(wmat, q)
                np.save(join(emb_path, multifrac_file1), r_D_2ss) 
                np.save(join(emb_path, multifrac_file2), l_D_2ss)   

                if wmat.ndim == 4:
                    if lidx == 0:
                        input_shape = [224,224]
                    else:
                        previous_module_idx,previous_layer_idx,_,_,_ = df.loc[lidx-1,:]  
                        backbone = Backbone_og(model, previous_module_idx, previous_layer_idx)
                        backbone.to(dev).eval()

                        with torch.no_grad():
                            input_shape = backbone(torch.ones([1,3,224,224]).to(dev)).shape

                        input_shape = list(input_shape[2:])                        
                    print(f"input_shape: {input_shape}")  # delete                        

                    r_D_2ss, l_D_2ss = tensor_fft_svd_dq(wmat, input_shape, q)
                    np.save(join(emb_path, multifrac_file3), r_D_2ss) 
                    np.save(join(emb_path, multifrac_file4), l_D_2ss)  

                    print(f'FFT SVD can be computed and data is saved!')                   

        print("Computation related to singular vectors complete! \n")

    else:
        print(f'Computation related to singular vectors complete and not to be replaced! \n')

    #################### 2. Compute PD D_2 ####################

    d2_file = f"d2smb_n_top={n_top}_layerwise_lidx={lidx}.npy"
    Rsmb_file = f"Rsmb_n_top={n_top}_layerwise_lidx={lidx}.npy"
    d2_exist = os.path.isfile( join(emb_path, d2_file) )
    Rsmb_exist = os.path.isfile( join(emb_path, Rsmb_file) ) 
    print(f"Computation for {d2_file}!")
    if d2_exist and Rsmb_exist and not replace:
        print(f"{d2_file} and {Rsmb_file} computed already, no computation required. \n")        
    else:

        N_images = len(imnames)

        # d2smb_all = []    # only compute D_2's
        # Rsmb_all = []   # the corresponding singularvalues (from large to small)

        #len_module = len(list(model.children())[module_idx])
        #for layer_idx in range(len_module):

        print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )

        if has_child:
            backbone = Backbone(model, module_idx, layer_idx).to(dev)
        else:
            backbone = Backbone(model, module_idx).to(dev)
        backbone.eval()

        d2s = []
        Rs = []
        # batch_size set to n_top
        for i in tqdm(range(N_images//n_top)):
            input_tensor = get_batch(i,n_top)
            with torch.no_grad():
                output = backbone(input_tensor.to(dev))      
            output -= output.mean(0)
            output = output.detach().numpy()
            # scklearn PCA
            pca = PCA(n_top)
            #pca.fit(output)
            pca.fit_transform(output)
            Rn = pca.explained_variance_
            Vn = pca.components_

            d2 = [compute_dq(Vn[eidx,:], 2) for eidx in range(len(Rn))]
            d2s.append(d2)
            Rs.append(Rn)

        d2s = np.stack(d2s)
        Rs = np.stack(Rs)
        # d2smb_all.append(d2s)
        # Rsmb_all.append(Rs)
                
        print(f"PD d2 computation complete for lidx = {lidx} layers! \n")

        # d2smb_all = np.stack(d2smb_all)
        # Rsmb_all = np.stack(Rsmb_all)
        np.save(join(emb_path, d2_file), d2s) 
        np.save(join(emb_path, Rsmb_file), Rs)   


    #################### 3. Compute SNR quantities ####################

    # check if relevant files exist even from the older versions    
    layerwise_file1 = "manifolds_layerwise.npy"    
    layerwise_file2 = f'manifolds_layerwise_lidx={lidx}.npy'    
    if os.path.isfile( join(emb_path, layerwise_file1) ) and not replace:
        #manifolds_all_load = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        #manifolds_all = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        print(f"{layerwise_file1} computed already, loading now.")
        manifolds_all = np.load(join(emb_path, layerwise_file1))
        manifolds = manifolds_all[lidx]

    elif os.path.isfile( join(emb_path, layerwise_file2) ) and not replace:
        print(f"{layerwise_file2} computed already, loading now.")
        manifolds = np.load(join(emb_path, layerwise_file2))

    else:
        print(f"Computation for {layerwise_file2}!")
        #print(f"Total number of modules: { len(list(model.children())) }")   

        if has_child:
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

        else:
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
        print(f'N_all = {N_all}')     

        if N_all > N or "resnet" in model_name:
            O = torch.randn(N_all,N) / np.sqrt(N)   # Gaussin projection
            #O = torch.Tensor(levy_stable.rvs(alpha, 0, size=(N_all,N), scale=(0.5/N)**(1./alpha)))
            O = O.to(dev)                

        print('Starting batch manifolds')
        manifolds = []
        for i in tqdm(range(K*P//batch_size)):
            input_tensor = get_batch(i,batch_size)
            with torch.no_grad():
                output = backbone(input_tensor.to(dev))

            if N_all > N or "resnet" in model_name:
                manifolds.append((output@O).cpu().numpy())
            else:
                manifolds.append(output.cpu().numpy())

        #manifolds = np.stack(manifolds).reshape(K,P,N)
        manifolds = np.stack(manifolds).reshape(K,P,N) if (N_all > N or "resnet" in model_name) else np.stack(manifolds).reshape(K,P,N_all)         

        np.save(join(emb_path, layerwise_file2), manifolds)
        #manifolds_all_load = manifolds_all
        print(f"{layerwise_file2} saved!")

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

        # ----- Overlaps -----
        # Subspace-subspace
        ss = []
        # Center-subspace
        csa = []
        csb = []

        for a in range(K):
            for b in range(K):
                if a!=b:
                    # Center-subspace
                    dx0 = centers[a] - centers[b]
                    dx0hat = dx0 / np.linalg.norm(dx0)
                    costheta_a = Us[a]@dx0hat
                    csa.append((costheta_a**2 * Rs[a]**2).sum() / (Rs[a]**2).sum())
                    costheta_b = Us[b]@dx0hat
                    csb.append((costheta_b**2 * Rs[b]**2).sum() / (Rs[a]**2).sum())

                    # Subspace-subspace
                    cosphi = Us[a]@Us[b].T
                    ss_overlap = (cosphi**2*Rs[a][:,None]**2*Rs[b]**2).sum() / (Rs[a]**2).sum()**2
                    ss.append(ss_overlap)
                else:
                    csa.append(np.nan)
                    csb.append(np.nan)
                    ss.append(np.nan)
        csa = np.stack(csa).reshape(K,K)
        csb = np.stack(csb).reshape(K,K)
        ss = np.stack(ss).reshape(K,K)

        css = (csa + csb/m) * dist_norm**2

        bias = (Rs**2).sum(-1) / (Rs**2).sum(-1)[:,None] - 1
        if m == np.inf:
            SNR = 1/2*(dist_norm**2)/ np.sqrt(css)
        else:
            SNR = 1/2*(dist_norm**2 + bias/m)/ np.sqrt(1/Ds[:,None]/m + css + ss/m)
        
        # additionally returning bias
        return dist_norm, Ds, csa, ss, SNR, bias

    print("Computing SNR metrics!")
    from scipy.spatial.distance import pdist, squareform

    ms = [1,5,10,100,1000,10000,np.inf]

    K = len(manifolds)
    #K = len(manifolds_all[0])
    # PRs_all = []
    # Rs_all = []
    # dists_all = []
    # css_all = []    
    # biases_all = []
    SNRs_all = {}
    for midx, m in enumerate(ms):
        SNRs_all[midx] = []

    #for mf_idx, manifolds in tqdm(enumerate(manifolds_all)):    
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

    Rs = np.stack(Rs)
    # Rs_all.append(Rs)
    centers = np.stack(centers)
    Us = np.stack(Us)

    #dist_norm, Ds, csa, ss, SNR = geometry(centers,Rs,Us,m)
    dist_norm, Ds, csa, ss, SNR, bias = geometry(centers,Rs,Us,m)
    # dists_all.append(dist_norm)
    # PRs_all.append(Ds)
    # css_all.append(csa)
    # biases_all.append(bias)

    for midx, m in enumerate(ms):
        # only the SNR is affected by the No. of learning examples m
        _, _, _, _, SNR, _ = geometry(centers,Rs,Us,m)
        SNRs_all[midx].append(SNR)
        
    # Rs_all = np.stack(Rs_all)
    # dists_all = np.stack(dists_all)
    # PRs_all = np.stack(PRs_all)
    # css_all = np.stack(css_all)
    # biases_all = np.stack(biases_all)

    #d2s_all = np.stack(d2s_all)

    for midx, m in enumerate(ms):
        SNRs_all[midx] = np.stack(SNRs_all[midx])
        if m == np.inf:
            np.save(os.path.join(emb_path,f'SNRs_layerwise_lidx={lidx}_m=inf.npy'),SNRs_all[midx])
        else:
            np.save(os.path.join(emb_path,f'SNRs_layerwise_lidx={lidx}_m={m}.npy'),SNRs_all[midx])

    np.save(os.path.join(emb_path,f'Rs_layerwise_lidx={lidx}.npy'),Rs)    
    np.save(os.path.join(emb_path,f'Ds_layerwise_lidx={lidx}.npy'),Ds)
    np.save(os.path.join(emb_path,f'dist_norm_layerwise_lidx={lidx}.npy'),dist_norm)
    np.save(os.path.join(emb_path,f'css_layerwise_lidx={lidx}.npy'),csa)
    np.save(os.path.join(emb_path,f'biases_layerwise_lidx={lidx}.npy'),bias)

    print("All SNR quantities saved! \n")

    # save manifolds_all dimension
    #np.save( join(emb_path, "manifold_dim.npy"), np.array([len(manifolds_all), len(manifolds), manifold.shape[0], manifold.shape[1]]) )

    # np.load(os.path.join(emb_path,f'Rs_layerwise_lidx={lidx}.npy'))    
    # np.load(os.path.join(emb_path,f'Ds_layerwise_lidx={lidx}.npy'))
    # np.load(os.path.join(emb_path,f'dist_norm_layerwise_lidx={lidx}.npy'))
    # np.load(os.path.join(emb_path,f'css_layerwise_lidx={lidx}.npy'))
    # np.load(os.path.join(emb_path,f'biases_layerwise_lidx={lidx}.npy'))
    
    # print("All SNR quantities can be loaded! \n")


def snr_components(model_name, pretrained, replace=True):
    """
    Obtain the SNR and its associated geometrical metrics for one model at a time.
    """

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable

    global Us, manifolds_all, Rs_all, dists_all, PRs_all, css_all, SNRs_all
    global U, R, V, Rs, Us
    global d2s_all
    global layers
    global manifolds, manifolds_all
    global init_epoch, emb_path
    global model, module_idx, layer_idx, Backbone
    global input_fbatch, output_fbatch, Un, Rn, Vn, d2, backbone

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained

    print(f"Setting up path for {model_name}.")        
    # where the desired manifolds.npy/manifolds_N={N}.npy, etc is located
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model!")
    model = load_model(model_name, pretrained)      

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


    # finds out the total amount of layers needed for evaluation (this is not rlly needed as well)
    """
    N = 2048
    batch_size = 64

    batch_size = 10
    K = 64
    P = 50

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

    batch_size = 50
    #batch_size = 10
    K = 64
    P = 50

    # analysis for top 200 PCs
    #n_top = 50 # not used
    
    layerwise_file = "manifolds_layerwise.npy"    
    if os.path.isfile( join(emb_path, 'manifolds_layerwise.npy') ) and not replace:
        #manifolds_all_load = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        #manifolds_all = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        print("manifolds_layerwise.npy computed already, loading now.")
        manifolds_all = np.load(join(emb_path, layerwise_file))

    else:
        print(f"Computation for {layerwise_file}!")
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
                    
                    if N_all > N or "resnet" in model_name:
                        O = torch.randn(N_all,N) / np.sqrt(N)   # Gaussian projection
                        #O = torch.Tensor(levy_stable.rvs(alpha, 0, size=(N_all,N), scale=(0.5/N)**(1./alpha)))  # Levy projection (very expensive)
                        O = O.to(dev)

                    print('Starting batch manifolds')                    
                    manifolds = []
                    for i in tqdm(range(K*P//batch_size)):
                        input_tensor = get_batch(i,batch_size)
                        with torch.no_grad():
                            output = backbone(input_tensor.to(dev))                            

                        if N_all > N or "resnet" in model_name:
                            manifolds.append((output@O).cpu().numpy())
                        else:
                            manifolds.append(output.cpu().numpy())

                    #manifolds = np.stack(manifolds).reshape(K,P,N)
                    manifolds = np.stack(manifolds).reshape(K,P,N) if (N_all > N or "resnet" in model_name) else np.stack(manifolds).reshape(K,P,N_all) 
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
                if N_all > N or "resnet" in model_name:
                    O = torch.randn(N_all,N) / np.sqrt(N)   # Gaussin projection
                    #O = torch.Tensor(levy_stable.rvs(alpha, 0, size=(N_all,N), scale=(0.5/N)**(1./alpha)))
                    O = O.to(dev)       

                print('Starting batch manifolds')
                manifolds = []
                for i in tqdm(range(K*P//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))

                    if N_all > N or "resnet" in model_name:
                        manifolds.append((output@O).cpu().numpy())
                    else:
                        manifolds.append(output.cpu().numpy())

                #manifolds = np.stack(manifolds).reshape(K,P,N)
                manifolds = np.stack(manifolds).reshape(K,P,N) if (N_all > N or "resnet" in model_name) else np.stack(manifolds).reshape(K,P,N_all)
                manifolds_all.append(manifolds)
                counter += 1
                print('Batch manifolds complete!')

        np.save(join(emb_path, f'manifolds_layerwise.npy'), manifolds_all)
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

        # ----- Overlaps -----
        # Subspace-subspace
        ss = []
        # Center-subspace
        csa = []
        csb = []

        for a in range(K):
            for b in range(K):
                if a!=b:
                    # Center-subspace
                    dx0 = centers[a] - centers[b]
                    dx0hat = dx0 / np.linalg.norm(dx0)
                    costheta_a = Us[a]@dx0hat
                    csa.append((costheta_a**2 * Rs[a]**2).sum() / (Rs[a]**2).sum())
                    costheta_b = Us[b]@dx0hat
                    csb.append((costheta_b**2 * Rs[b]**2).sum() / (Rs[a]**2).sum())

                    # Subspace-subspace
                    cosphi = Us[a]@Us[b].T
                    ss_overlap = (cosphi**2*Rs[a][:,None]**2*Rs[b]**2).sum() / (Rs[a]**2).sum()**2
                    ss.append(ss_overlap)
                else:
                    csa.append(np.nan)
                    csb.append(np.nan)
                    ss.append(np.nan)
        csa = np.stack(csa).reshape(K,K)
        csb = np.stack(csb).reshape(K,K)
        ss = np.stack(ss).reshape(K,K)

        css = (csa + csb/m) * dist_norm**2

        bias = (Rs**2).sum(-1) / (Rs**2).sum(-1)[:,None] - 1
        if m == np.inf:
            SNR = 1/2*(dist_norm**2)/ np.sqrt(css)
        else:
            SNR = 1/2*(dist_norm**2 + bias/m)/ np.sqrt(1/Ds[:,None]/m + css + ss/m)
        
        # additionally returning bias
        return dist_norm, Ds, csa, ss, SNR, bias

    print("Computing SNR metrics!")
    from scipy.spatial.distance import pdist, squareform

    ms = [1,5,10,100,1000,10000,np.inf]

    #K = len(manifolds)
    K = len(manifolds_all[0])
    PRs_all = []
    Rs_all = []
    dists_all = []
    css_all = []    
    biases_all = []
    SNRs_all = {}
    for midx, m in enumerate(ms):
        SNRs_all[midx] = []

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

        Rs = np.stack(Rs)
        Rs_all.append(Rs)
        centers = np.stack(centers)
        Us = np.stack(Us)

        #dist_norm, Ds, csa, ss, SNR = geometry(centers,Rs,Us,m)
        dist_norm, Ds, csa, ss, SNR, bias = geometry(centers,Rs,Us,m)
        dists_all.append(dist_norm)
        PRs_all.append(Ds)
        css_all.append(csa)
        biases_all.append(bias)

        for midx, m in enumerate(ms):
            # only the SNR is affected by the No. of learning examples m
            _, _, _, _, SNR, _ = geometry(centers,Rs,Us,m)
            SNRs_all[midx].append(SNR)
        
    Rs_all = np.stack(Rs_all)
    dists_all = np.stack(dists_all)
    PRs_all = np.stack(PRs_all)
    css_all = np.stack(css_all)
    biases_all = np.stack(biases_all)

    #d2s_all = np.stack(d2s_all)

    for midx, m in enumerate(ms):
        SNRs_all[midx] = np.stack(SNRs_all[midx])
        if m == np.inf:
            np.save(os.path.join(emb_path,f'SNRs_layerwise_m=inf.npy'),SNRs_all[midx])
        else:
            np.save(os.path.join(emb_path,f'SNRs_layerwise_m={m}.npy'),SNRs_all[midx])

    np.save(os.path.join(emb_path,'Rs_layerwise.npy'),Rs_all)    
    np.save(os.path.join(emb_path,'Ds_layerwise.npy'),PRs_all)
    np.save(os.path.join(emb_path,'dist_norm_layerwise.npy'),dists_all)
    np.save(os.path.join(emb_path,'css_layerwise.npy'),css_all)
    np.save(os.path.join(emb_path,f'biases_layerwise.npy'),biases_all)

    # save manifolds_all dimension
    np.save( join(emb_path, "manifold_dim.npy"), np.array([len(manifolds_all), len(manifolds), manifold.shape[0], manifold.shape[1]]) )


# get d2 from minibatch
#def snr_d2_mbatch(model_name, pretrained, n_top=50):
def snr_d2_mbatch(model_name, pretrained, n_top=100):

    # only the top n_top eigenvectors (of the covariance matrix) are explored, the feature space is large

    import numpy.linalg as LA
    import scipy
    from scipy.stats import levy_stable
    from sklearn.decomposition import PCA

    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained
    n_top = int(n_top) if n_top != None else None

    print(f"Setting up path for {model_name}.")        
    emb_path = join(root_data, "macaque_stimuli", model_name, "pretrained") if pretrained else join(root_data, "macaque_stimuli", model_name, "untrained")
    if not os.path.isdir(emb_path): os.makedirs(emb_path)

    print("Loading model weights!")
    model = load_model(model_name, pretrained)         

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

        #batch_size = 50
        #batch_size = 100
        batch_size = n_top
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
                        #pca.fit(output)
                        pca.fit_transform(output)
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
                    #pca.fit(output)
                    pca.fit_transform(output)
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
        np.save(join(emb_path, d2_file), d2smb_all)   # fullbatch
        np.save(join(emb_path, Rsmb_file), Rsmb_all)   # fullbatch


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
    from qsub import qsub, job_divider, project_ls, command_setup

    # get all appropriate networks
    """
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

    # simulation guide
    """
    - SNR
        - level 1: "alexnet" (12GB)
        - level 2: "resnet18", "resnet34", "resnet50",  "resnext50_32x4d", "squeezenet1_1", "wide_resnet50_2" (24GB)
        - level 3: "squeezenet1_0", "resnet101", "resnet152",  "resnext101_32x8d",  "wide_resnet101_2" (32GB)
    - d2
        - level 1: "alexnet", "resnet18" (8GB)
        - level 2: "resnet34", "resnet50",  "resnext50_32x4d", "squeezenet1_0", "squeezenet1_1", "wide_resnet50_2" (16GB)
        - level 3: "resnet101", "resnet152",  "resnext101_32x8d",  "wide_resnet101_2" (32GB)
    """

    # more image classification models from https://pytorch.org/vision/stable/models.html 
    # (does not include transformers, i.e. SwinTransformer, VisionTransformer)
    """   

    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', (done)

    'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'

    'mobilenet_v2',
    'mobilenet_v3_small', 'mobilenet_v3_large',

    'googlenet',

    'densenet121', 'densenet169', 'densenet201',  

    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',                 

    # -----------------------------------------------

    'inception_v3',

    'densenet161',

    'convnext_tiny', 'convnext_small', 

    'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',

    # -----------------------------------------------

    'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',     
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 
    
    'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf', 
    'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 

    'regnet_y_128gf', (cannot extract)    

    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l/',    

    'resnext101_64x4d',

    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',

    'convnext_base', 'convnext_large',

    """

    # SNR
    #models = ["resnet152"]
    #models = ["squeezenet1_0"]

    # d2
    #models = ["resnet34", "resnet50",  "resnext50_32x4d", "squeezenet1_0", "squeezenet1_1", "wide_resnet50_2"]
    #models = ["alexnet", "resnet18"]

    # --------------- new models ---------------
    
    # 24GB (20 networks)
    # 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', (done)
    """
    models = ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
              'mobilenet_v2',
              'mobilenet_v3_small', 'mobilenet_v3_large',
              'googlenet',
              'densenet121', 'densenet169', 'densenet201',  
              'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4'
              ]     
    """

    
    # 32GB (7 networks)
    """
    models = ['inception_v3',
              'densenet161',
              'convnext_tiny', 'convnext_small', 
              'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'
              ]
    """
    
    
    # 48GB (28 networks)
    
    # 'regnet_y_128gf', (cannot extract)
    models = ['regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',     
              'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',     
              'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf', 
              'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf',                   
              'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l/',    
              'resnext101_64x4d',
              'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
              'convnext_base', 'convnext_large'
              ]
    

    pretrained_ls = [True, False]
    #pretrained_ls = [True]
    pbs_array_data = [(model_name, pretrained)
                      for model_name in models
                      for pretrained in pretrained_ls
                      #if not os.path.isfile(join(root_data,"pretrained_workflow/pretrained_dnns",model_name,"manifold",fname,"css_layerwise.npy"))
                      ]

    #pbs_array_data = pbs_array_data[2:]  # delete
    print(f"Total jobs: {len(pbs_array_data)} \n")
    print(pbs_array_data)
    
    #quit()

    perm, pbss = job_divider(pbs_array_data, len(project_ls))

    select = 1
    ncpus, ngpus = 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)
    #print(command)  # delete

    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             #path=join(root_data, "macaque_stimuli/new_snr"),
             #path=join(root_data, "macaque_stimuli"),
             path=join(root_data, "macaque_stimuli/new_nets_job"),
             #path=join(root_data, "macaque_stimuli", "pure_rsmb"),
             #path=join(root_data, "macaque_stimuli/new_d2"),
             P=project_ls[pidx],
             #ngpus=ngpus,
             ncpus=ncpus,
             select=select,
             walltime='47:59:59',
             #walltime='23:59:59',   # small/medium
             mem='48GB')
             #mem='32GB')
             #mem='24GB')
             #mem='20GB')
             #mem='16GB')
             #mem='12GB')
             #mem='8GB') 

# get_num_layers()
def snr_submit_v2(*args):
    global pbs_array_data

    from qsub import qsub, job_divider, project_ls, command_setup

    # get all appropriate networks
    net_names = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv"))    # locally available pretrained nets
    models = []
    for model_name in net_names.loc[:, "model_name"]:
        models.append(model_name)
    
    pbs_array_data = []
    for model_name in models:
        if not os.path.isfile(join(root_data, "macaque_stimuli", model_name, "shared_info")):
            pbs_array_data.append((model_name, True))

    #pbs_array_data = pbs_array_data[2:]  # delete
    print(f"Total jobs: {len(pbs_array_data)} \n")
    print(pbs_array_data)
    
    #quit()

    perm, pbss = job_divider(pbs_array_data, len(project_ls))

    select = 1
    ncpus, ngpus = 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             #path=join(root_data, "macaque_stimuli/new_snr"),
             #path=join(root_data, "macaque_stimuli"),
             path=join(root_data, "macaque_stimuli/get_layers_job"),
             #path=join(root_data, "macaque_stimuli", "pure_rsmb"),
             #path=join(root_data, "macaque_stimuli/new_d2"),
             P=project_ls[pidx],
             #ngpus=ngpus,
             ncpus=ncpus,
             select=select,             
             walltime='23:59:59',   # small/medium
             mem='1GB')           


# layerwise_snr_components
def snr_submit_v3(*args):
    global pbs_array_data, net_names, models, file3_check, df_info, do_fft, has_weight

    from qsub import qsub, job_divider, project_ls, command_setup
        
    pretrained_ls = [True, False]
    #pretrained_ls = [True]

    # specified models
    # models = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121',
    #           'squeezenet1_0', 'squeezenet1_1',
    #           'vgg11', 'vgg13', 'vgg16',
    #           'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
    #           'shufflenet_v2_x0_5','shufflenet_v2_x1_0','shufflenet_v2_x1_5','shufflenet_v2_x2_0']

    # models = ['resnet18', 'resnet34', 'resnet50', 
    #           'densenet121',
    #           'squeezenet1_0']

    models = ['alexnet']

    # all models
    # models = []    
    # net_names = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv"))    # locally available pretrained nets
    # for model_name in net_names.loc[:, "model_name"]:
    #     if os.path.isfile(join(root_data, "macaque_stimuli", model_name, "shared_info")):
    #         models.append(model_name)   
        
    
    pbs_array_data = []
    for model_name in models:
        df = load_shared_info(model_name)
        for lidx in range(df.shape[0]):
        #for lidx in range(1,6):
        #for lidx in [1,2]:
            for pretrained_type in pretrained_ls:
                pretrained_name = 'pretrained' if pretrained_type else 'untrained'
                file1_check = os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                                                   f'd2smb_n_top=100_layerwise_lidx={lidx}.npy') )
                file2_check = os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                                                   f'dist_norm_layerwise_lidx={lidx}.npy') )    

                df_info = pd.read_csv(join(root_data, "macaque_stimuli", model_name, "shared_info"))
                module_idx,layer_idx,has_child,has_weight,weight_shape = df_info.loc[lidx,:]  
                file3_check = has_weight                
                #print(f'file3_check ={type(file3_check)}')
                if file3_check:
                    do_fft = (len(literal_eval(weight_shape)) == 4)
                    #file3_check = os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                    #                                f'singvec_mean_lidx={lidx}.npy') )   
                    file3_check = os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                                                    f'r_singvec_dq_lidx={lidx}.npy') )  
                    file3_check = file3_check and os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                                                                       f'l_singvec_dq_lidx={lidx}.npy') )  
                    if do_fft:
                        file3_check = file3_check and os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                                                                        f'r_fftsingvec_dq_lidx={lidx}.npy') )  
                        file3_check = file3_check and os.path.isfile( join(root_data, "macaque_stimuli", model_name, pretrained_name,
                                                                        f'l_fftsingvec_dq_lidx={lidx}.npy') )                                                                                                                                                                                                                         
                else:
                    file3_check = True
                if not (file1_check and file2_check and file3_check):
                    pbs_array_data.append( (model_name, lidx, pretrained_type) )


    #pbs_array_data = pbs_array_data[2:]  # delete
    print(f"Total jobs: {len(pbs_array_data)} \n")

    quit()

    perm, pbss = job_divider(pbs_array_data, len(project_ls))

    select = 1
    ncpus, ngpus = 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "macaque_stimuli/jobs_all/snr_v2"),
             P=project_ls[pidx],
             #ngpus=ngpus,
             ncpus=ncpus,
             select=select,             
             walltime='23:59:59',   # small/medium
             mem='16GB')                  

# ---------------------- Plotting ----------------------

def load_dict():
    metric_dict = {"SNR"        : "SNRs_layerwise",
                   "error"      : "SNRs_layerwise",
                   "D"          : "Ds_layerwise",
                   "dist_norm"  : "dist_norm_layerwise",
                   "css"        : "css_layerwise",         
                   "bias"       : "biases_layerwise",
                   "R_100"      : "Rsmb_n_top=100_layerwise",
                   "R_50"       : "Rsmb_n_top=50_layerwise"
                   }

    name_dict = {"SNR"        : "SNR",
                 "error"      : "error",
                 "D"          : 'Dimension',
                 "dist_norm"  : "Signal",
                 "css"        : "Signal-noise overlap",
                 "bias"       : "Bias",      
                 "R_100"      : 'Cumulative variance',
                 "R_50"       : 'Cumulative variance'        
                 }   

    return metric_dict, name_dict


# ---------------------- Extracting metrics ----------------------

# load raw metrics from network
def load_raw_metric(model_name, pretrained:bool, metric_name, **kwargs):
    from scipy.stats import norm

    if "m" in kwargs.keys():
        m = kwargs.get("m")

    metric_dict, _ = load_dict()
    # check if metric_name includes "d2_"
    if "d2" in metric_name:
        dq_ls = metric_name.split("_") if "d2" in metric_name else []
        dq_filename = f"d2smb_n_top=100_layerwise" if len(dq_ls) > 0 else None
        #dq_filename = f"d2smb_n_top=50_layerwise" if len(dq_ls) > 0 else None
        metric_dict[metric_name] = dq_filename

    # model path
    init_path = join(root_data, "macaque_stimuli")
    pretrained_str = "pretrained" if pretrained else "untrained"
    manifold_path = join(init_path, model_name, pretrained_str)    
    emb_path = manifold_path

    if "d2_" in metric_name or metric_name == "d2":
        # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
        metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}.npy") )            
    elif metric_name == "SNR" or metric_name == "error":
        if m == np.inf:
            metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}_m=inf.npy") )
        else:
            metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}_m={m}.npy") )
        if metric_name == "error":
            metric_data = 1 - norm.cdf(metric_data)     
    else:
        metric_data = np.load( join(emb_path, f"{metric_dict[metric_name]}.npy") )

    return metric_data

# load processed metrics from network
def load_processed_metric(model_name, pretrained:bool, metric_name,**kwargs):
    #global metric_R, metric_D, metric_data, metric_og, batch_idx, l, Rs
    
    pretrained = literal_eval(pretrained) if isinstance(pretrained,str) else pretrained

    #if "n_top" in kwargs.keys():
    #    n_top = kwargs.get("n_top")
    if "m" in kwargs.keys():
        m = kwargs.get("m")
    if "avg" in kwargs.keys():
        avg = kwargs.get("avg")

    #if "d2_" not in metric_name and metric_name != "R":
    if metric_name == "SNR" or metric_name == "error":
        metric_data = load_raw_metric(model_name, pretrained, metric_name, m=m)
    # signal is dist_norm squared
    elif metric_name == "dist_norm":
        metric_data = load_raw_metric(model_name, pretrained, metric_name)**2
    #cumulative variance explained by n_top PCs
    elif metric_name == "R_100":
        metric_R = load_raw_metric(model_name, pretrained, metric_name)
        if avg:
            # (17, 32, 100)
            metric_data = metric_R[:,:,:n_top].cumsum(-1)/metric_R.sum(-1)[:,:,None]
        else:
            metric_data = metric_R[:,:,pc_idx]/metric_R.sum(-1)

    # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are     
    elif metric_name == "d2_avg":
        # D_2's
        metric_og = load_raw_metric(model_name, pretrained, metric_name)
        # R's (eigenvalues/variance explained)     
        metric_R = load_raw_metric(model_name, pretrained, "R_100")  
        metric_D = load_raw_metric(model_name, pretrained, "D")  
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
                        

    elif metric_name == "d2":
        metric_data = load_raw_metric(model_name, pretrained, metric_name)
    elif "d2_" in metric_name:
        pc_idx = int(metric_name.split("_")[-1])
        metric_og = load_raw_metric(model_name, pretrained, metric_name)   
        metric_data = metric_og[:,:,pc_idx]
    else:
        metric_data = load_raw_metric(model_name, pretrained, metric_name)

    if model_name == "alexnet":
        conv_idxs = [0,3,6,8,10,15,16]
        return metric_data[conv_idxs]
    else:
        return metric_data


# transform model string name into formal name with capitals
def transform_model_name(s):
    result = ""
    for i, letter in enumerate(s):
        if i == 0 or letter == "n":
            result += letter.upper()
        elif s[i-1] == "_" and letter.isalpha():
            result += letter.upper()
        else: 
            result += letter
    return result

# get CI for the SNR and associated geometrical metrics
from scipy import stats
def get_CI(y):
    CImin,CImax = stats.t.interval(0.95, len(y),
                     loc=y.mean(-1), scale=stats.sem(y,axis=-1))
    return CImin, CImax


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

def get_model_names(main):
    if not main:
        model_names = ["squeezenet1_1","wide_resnet101_2"]
    else:
        model_names = ["alexnet", "resnet101"]
    return model_names

# Main text plot (minibatch d2 version)
def snr_metric_plot(main=True, n_top=100, display=False):
    metric_names="d2_avg,D,SNR,error"

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

        model_names = get_model_names(main)
    else:
        if ',' in main:
            model_names = main.split(',')
        else:
            model_names = [main]

    # Plot settings
    fig_size = (11/2*3,7.142+2) # 2 by 3

    # load dict
    metric_dict, name_dict = load_dict()

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
            name_dict[metric_name] = r'Weighted $D_2$' 

    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]

    # need to demonstrate for pretrained and random DNNs
    pretrained_ls = [False, True]
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
            total_layers = load_processed_metric(model_name, pretrained, "dist_norm").shape[0]
            frac_layers = np.arange(0,total_layers)/(total_layers-1)
            # only scatter plot the selected layers (can modify)
            selected_layers = np.where(frac_layers >= 0)
            # for the error/SNR and D_2 centers
            deep_layers = np.where(frac_layers >= 0.8)

            # --------------- Plot 1 (upper) ---------------

            # load all data
            for metric_idx, metric_name in enumerate(metric_names):
                metric_data = load_processed_metric(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
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
                axis = axs[midx+4]                

                metric_data_y = load_processed_metric(model_name, pretrained, metric_name_y, m=m, n_top=n_top)
                # centers of error
                error_centers[midx,nidx,pretrained_idx] = np.nanmean(metric_data_y[deep_layers,:,:].flatten())
                metric_data_y = np.nanmean(metric_data_y, (1,2)) 
                color = c_ls[metric_names.index(metric_name_y)] if pretrained else "gray"

                # plots all layers
                axis.scatter(d2_data.mean(-1)[selected_layers], metric_data_y[selected_layers], 
                                    c=color, marker=markers[nidx], alpha=trans_ls[nidx])
                # plots deep layers
                #axis.scatter(d2_data.mean(-1)[deep_layers], metric_data_y[deep_layers], 
                #                    c='k', marker='x', alpha=trans_ls[nidx])

                if pretrained_idx == len(pretrained_ls) - 1:
                    # plot centered arrows
                    color_arrow = c_ls[metric_names.index(metric_name_y)]

                    error_center = error_centers[midx,nidx,0]
                    d2_center = d2_centers[nidx,0]
                    # arrow head
                    prop['color'] = color_arrow; prop['alpha'] = trans_ls[nidx]

                    error_y = error_centers[midx,nidx,1]
                    axis.annotate("", xy=(d2_centers[nidx,1],error_y), xytext=(d2_center,error_center), arrowprops=prop)

                    # axis labels
                    #axs[midx+2].set_title(name_dict[metric_name_y] + rf" ($m$ = {m})", fontsize=title_size)
                    axis.set_xlabel(r"Weighted $D_2$", fontsize=label_size)
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
        #axs[0].set_ylim((0.7,0.95))
        axs[0].set_ylim((0.1,0.95))
    else:
        #axs[0].set_ylim((0.6,1))
        axs[0].set_ylim((0.1,1))
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
            fig_path = join(root_data,f"figure_ms/pretrained-fewshot-all")
        if not os.path.isdir(fig_path): os.makedirs(fig_path)
        net_cat = "_".join(model_names)
        fig_name = f"pretrained_m={m_featured}_metric_{net_cat}.pdf"
        print(f"Saved as {join(fig_path, fig_name)}")
        plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')
    else:
        plt.show()


# Appendix plot for other metrics corresponding to SNR, i.e. signal (dist_norm), bias, signal-to-noise overlap (css)
def extra_metric_plot(main=True, n_top=100, display=False):
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
    metric_dict, name_dict = load_dict()
 
    # model names
    model_names = get_model_names(main)
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
                total_layers = load_processed_metric(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the phase centers
                deep_layers = np.where(frac_layers >= 0.8)

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    metric_data = load_processed_metric(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
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


def snr_delta_plot(main=True, n_top=100):
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
    metric_dict, name_dict = load_dict()

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
    model_names = get_model_names(main)
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
            total_layers = load_processed_metric(model_name, pretrained, "dist_norm").shape[0]
            frac_layers = np.arange(0,total_layers)/(total_layers-1)
            # only scatter plot the selected layers (can modify)
            selected_layers = np.where(frac_layers >= 0)
            # for the error/SNR and D_2 centers
            deep_layers = np.where(frac_layers >= 0.8)

            # --------------- Plot 1 (upper) ---------------

            # load all data
            for metric_idx, metric_name in enumerate(metric_names):
                metric_data = load_processed_metric(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
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

                    metric_data_y = load_processed_metric(model_name, pretrained, metric_name_y, m=m, n_top=n_top)
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
def final_layers(metric_names="d2_avg,D,SNR,error,dist_norm,bias,css", n_top=100):
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
    metric_dict, name_dict = load_dict()
 
    # "resnet152"    
    """
    model_names = ["alexnet",   # trained
                   "resnet18", "resnet34", "resnet50", "resnet101",
                   "resnext50_32x4d", "resnext101_32x8d",
                   "squeezenet1_0", "squeezenet1_1",  
                   "wide_resnet50_2", "wide_resnet101_2"]   
    """
    
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
    model_names = ["mobilenet_v3_small"]

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
                total_layers = load_processed_metric(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the phase centers
                deep_layers = np.where(frac_layers >= 0.8)

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    metric_data = load_processed_metric(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
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

                    # last layer
                    """
                    axs[metric_idx].scatter(d2_data.mean(-1)[-1], metric_data_mean[-1], 
                                            marker=markers[nidx], 
                                            c=color)    # alpha=trans_ls[nidx]
                    """

                    # selected layers
                    """
                    depth_idxs = [-2,-1]
                    axs[metric_idx].scatter(d2_data.mean(-1)[depth_idxs], metric_data_mean[depth_idxs], 
                                            marker=markers[nidx], 
                                            c=color)    # alpha=trans_ls[nidx]                          
                    """

                    # last n layers   
                                    
                    depth_idx = 0
                    axs[metric_idx].scatter(d2_data.mean(-1)[depth_idx:], metric_data_mean[depth_idx:], 
                                            marker=markers[nidx], 
                                            c=color)    # alpha=trans_ls[nidx]   
                    
        
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
        #plt.savefig(join(fig_path, fig_name) , bbox_inches='tight')
        plt.show()


def aggregate_layers(metric_names="d2_avg,D,SNR,error,dist_norm,bias,css", n_top=100):
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
    metric_dict, name_dict = load_dict()
 
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
                total_layers = load_processed_metric(model_name, pretrained, "dist_norm").shape[0]
                frac_layers = np.arange(0,total_layers)/(total_layers-1)
                # only scatter plot the selected layers (can modify)
                selected_layers = np.where(frac_layers >= 0)
                # for the phase centers
                deep_layers = np.where(frac_layers >= 0.8)

                # load all data
                for metric_idx, metric_name in enumerate(metric_names):
                    metric_data = load_processed_metric(model_name, pretrained, metric_name, m=m_featured, n_top=n_top)
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
def snr_microscopic_plot(small=True, log=False, display=True):
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
    metric_dict, name_dict = load_dict()
    
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


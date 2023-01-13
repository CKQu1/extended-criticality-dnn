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
global pretrained_path
pretrained_path = join(root_data, "pretrained_workflow", "pretrained_dnns")

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

def pretrained_stimuli(model_name, pretrained=True, projection=False):
    from NetPortal.backbone_architectures import AlexnetBackbone, SqueezeBackbone, WideResNetBackbone, VGGBackbone, EfficientNetBackbone

    pretrained = pretrained if isinstance(pretrained, bool) else literal_eval(pretrained)

    #global net_names, net_path, model, emb_path

    # make sure model has backbone
    nets_with_backbone = ["resnet", "resnext", "vgg", "alexnet", "squeeze", "efficient"]
    matches = 0
    for net_type in nets_with_backbone:
        if net_type in model_name.lower():
            matches += 1
    assert matches > 0, "backbone model does not exists, try to create one!"

    # ---------------------- MODEL ----------------------
    print("Setting up models.")
    # other methods to load pretrained network
    #model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    #model = torch.hub.load("pytorch/vision", model_name, pretrained=pretrained)

    # list of allowable networks (file created from pretrained_workflow/pretrained_download.py)
    net_names = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv"))
    net_names_tf = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all_tf.csv"))
    assert model_name in list(net_names.loc[:,"model_name"]) or model_name in list(net_names_tf.loc[:,"model_name"]), "model_name does not exist!"

    # downloaded the pretrained models in bulk locally and load them
    net_path = join(pretrained_path, model_name)
    model = torch.load(join(net_path, "model_pt"))

    print("Model loaded.")
    if pretrained:
        emb_path = join(net_path, f'manifold/embeddings_new/macaque/trained/')
    else:
        emb_path = join(net_path, f'manifold/embeddings_new/macaque/untrained/')
    if not os.path.isdir(emb_path):
        os.makedirs(emb_path)

    if 'resnet' in model_name or 'resnext' in model_name:
        backbone = WideResNetBackbone(model, 1)
        #random_projection=False
    elif 'vgg' in model_name:
        backbone = VGGBackbone(model)
    elif 'alexnet' in model_name:
        backbone = AlexnetBackbone(model)
    elif 'squeeze' in model_name:
        backbone = SqueezeBackbone(model)
    elif 'efficient' in model_name:
        backbone = model
        backbone.N = 1000
    #     backbone = EfficientNetBackbone(model)
    backbone.to(dev).eval()

    print(f"{time() - t0}s")
    # ---------------------- COMPUTATION ----------------------
    print("Computing manifold.")

    # example to get M
    batch_size = 10
    input_tensor = get_batch(0,batch_size)
    with torch.no_grad():
        output = backbone(input_tensor.to(dev)).cpu()
    M = output.shape[1]

    if projection:
        N = 2048    # projection dimension (set up arbitrarily)
        # idxs = torch.randint(0,100352,(N,))
        #O = torch.randn((M,2048))/np.sqrt(2048)
        O = torch.randn((M,N))/np.sqrt(N)
        O = O.to(dev)

    # K and P can be adjusted
    K = 64
    P = 50
    batch_size = 10
    n_classes = len(imnames)

    manifolds = []
    # for class_id in tqdm(range(n_classes)):
    for i in tqdm(range(K*P//batch_size)):
        input_tensor = get_batch(i,batch_size)
        with torch.no_grad():
            output = backbone(input_tensor.to(dev))
        if projection:
            manifolds.append((output@O).cpu().numpy())
        else:
            manifolds.append(output.cpu().numpy())

    manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,M)
    
    """
    if projection:
        emb_path = join(pretrained_path, model_name, "manifold", 
                        f"bs={batch_size}_K={K}_P={P}_epoch={init_epoch}_pretrained={pretrained}_projection={projection}")
    else:
        emb_path = join(pretrained_path, model_name, "manifold", 
                        f"bs={batch_size}_K={K}_P={P}_N={N}_epoch={init_epoch}_pretrained={pretrained}_projection={projection}")
    """

    manifold_fname = 'manifolds.npy' if not projection else f'manifolds_N={N}.npy'
    np.save(os.path.join(emb_path, manifold_fname), manifolds)
    print("Manifold saved!")

def stimuli_submit(*args):

    global pbs_array_data, perm, pbss
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]    

    # ----- Pretrained Nets -----
    nets_with_backbone = ["resnet", "resnext", "vgg", "alexnet", "squeeze", "efficient"]
    net_names = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv"))    # locally available pretrained nets
    models = []
    for model_name in list(net_names.loc[:, "model_name"]):
        matches = 0
        for net_type in nets_with_backbone:
            if net_type in model_name.lower():
                matches += 1
        if matches > 0:
            models.append(model_name)

    pretrained = True
    pbs_array_data = [(model_name, pretrained)
                      for model_name in models
                      ]

    #pbs_array_data = pbs_array_data[0:2]
    print(pbs_array_data)

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
             mem='8GB') 
        

# ---------------------- SNR Theory ----------------------

from train_supervised import compute_dq

# projection must equal True lol otherwise computation gets too expensive for computing the covariance matrix
def snr_components(model_name, fname, projection=True):

    import numpy.linalg as LA

    """
    global Us, manifolds_all, Rs_all, dists_all, PRs_all, css_all, SNRs_all
    global EDs_all, dqs_all
    global layers
    global manifolds
    global init_epoch, emb_path
    """
    global model, module_idx, layer_idx, Backbone

    print("Setting up model path.")        
    # where the desired manifolds.npy/manifolds_N={N}.npy, etc is located
    emb_path = join(pretrained_path, model_name, "manifold", fname)


    print("Loading model weights!")
    model = torch.load(join(pretrained_path, model_name, "model_pt"))

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

    # -------------------

    # same variables as in cnn_macaque_stimuli.py
    N = 2048
    N_dq = 500
    navg_dq = 10

    batch_size = 10
    K = 64
    P = 50

    layerwise_file = "manifolds_layerwise.npy"
    print("Computation for manifolds_layerwise.npy")
    if os.path.isfile( join(emb_path, 'manifolds_layerwise.npy') ):
        #manifolds_all_load = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        #manifolds_all = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        print("manifolds_layerwise.npy computed already, loading now.")
        manifolds_all = np.load(join(emb_path, layerwise_file))

    else:
        print(f"Total number of modules: { len(list(model.children())) }")

        pc_idxs = list(range(25))
        EDs_all = []
        dqs_all = {}
        for pc_idx in pc_idxs:
            dqs_all[pc_idx] = []
        qs = np.linspace(0,2,100)   # for computing D_q

        counter = 0
        manifolds_all = []
        #for module_idx in tqdm(range(len(list(model.children())))):
        for module_idx in tqdm(range(len(list(model.children())))):

            try:
                len_module = len(list(model.children())[module_idx])
                for layer_idx in range(len_module):

                    print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                    
                    backbone = Backbone(model, module_idx, layer_idx)
                    backbone.to(dev).eval()

                    # Get test batch
                    input_tensor = get_batch(0,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    N_all = output.shape[-1]
                    if projection:
                        O = torch.randn(N_all,N) / np.sqrt(N)
                        O = O.to(dev)
        #             idxs = np.random.randint(0,N_all,N)
                    
                    dqs = {}
                    for pc_idx in pc_idxs:
                        dqs[pc_idx] = []
                    EDs = []
                    manifolds = []
                    for i in tqdm(range(K*P//batch_size)):
                        input_tensor = get_batch(i,batch_size)
                        with torch.no_grad():
                            output = backbone(input_tensor.to(dev))

                        # D_q must be computed before projection to lower dim
                        dqs_layer = {}
                        for pc_idx in pc_idxs:
                            dqs_layer[pc_idx] = []                        
                        EDs_layer = []
                        # averaging the dqs
                        for nsamp in range(navg_dq):
                            # arbitrarily sample neurons
                            subsampled_idxs = np.random.choice(output.shape[-1], N_dq)
                            output_sub = output[:,subsampled_idxs]
                            # get cov matrix
                            #eigvals, eigvecs = LA.eig( np.cov( output_sub.T ) )
                            eigvals, eigvecs = LA.eig( np.cov( output_sub.numpy().T ) )
                            EDs_layer.append( np.mean(eigvals**2)**2 / np.sum(eigvals**4) )
                            for pc_idx in pc_idxs:
                                dqs_layer[pc_idx].append( [compute_dq(eigvecs[:,pc_idx],q) for q in qs ] )
                        for pc_idx in pc_idxs:
                            dqs[pc_idx].append(np.stack(dqs_layer[pc_idx]).mean(0)) 
                        EDs.append(np.mean(EDs_layer))

                        if projection:
                            manifolds.append((output@O).cpu().numpy())
                        else:
                            manifolds.append(output.cpu().numpy())

                    EDs_all.append(EDs)
                    for pc_idx in pc_idxs:
                        dqs_all[pc_idx].append(np.stack(dqs[pc_idx])) 
                    print(eigvecs.shape) # for double-checking

                    manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,N_all) 
                    manifolds_all.append(manifolds)
                    counter += 1

            except:
                layer_idx = 0
                print('Computing embeddings module '+str(module_idx)+', layer '+str(layer_idx) )
                layer = 'layer_' + str(module_idx) + '_' + str(layer_idx)
                
                backbone = Backbone(model, module_idx)
                backbone.to(dev).eval()

                # Get test batch
                input_tensor = get_batch(0,batch_size)
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                N_all = output.shape[-1]
                if projection:
                    O = torch.randn(N_all,N) / np.sqrt(N)
                    O = O.to(dev)
        #         idxs = np.random.randint(0,N_all,N)        

                dqs = {}
                for pc_idx in pc_idxs:
                    dqs[pc_idx] = []                
                EDs = []
                manifolds = []
                for i in tqdm(range(K*P//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))

                    # dq must be computed before projection to lower dim
                    dqs_layer = {}
                    for pc_idx in pc_idxs:
                        dqs_layer[pc_idx] = []
                    EDs_layer = []
                    # get a general version of cov mat      
                    for nsamp in range(navg_dq):
                        # arbitrarily sample neurons
                        subsampled_idxs = np.random.choice(output.shape[-1], N_dq)
                        output_sub = output[:,subsampled_idxs]
                        # get cov matrix
                        eigvals, eigvecs = LA.eig( np.cov( output_sub.numpy().T ) )
                        EDs_layer.append( np.mean(eigvals**2)**2 / np.sum(eigvals**4) )
                        for pc_idx in pc_idxs:
                            dqs_layer[pc_idx].append( [compute_dq(eigvecs[:,pc_idx],q) for q in qs ] )
                    for pc_idx in pc_idxs:
                        dqs[pc_idx].append(np.stack(dqs_layer[pc_idx]).mean(0)) 
                    EDs.append(np.mean(EDs_layer))

                    if projection:
                        manifolds.append((output@O).cpu().numpy())
                    else:
                        manifolds.append(O.cpu().numpy())

                EDs_all.append(EDs)
                for pc_idx in pc_idxs:
                    dqs_all[pc_idx].append(np.stack(dqs[pc_idx])) 

                print(eigvecs.shape) # for double-checking

                manifolds = np.stack(manifolds).reshape(K,P,N) if projection else np.stack(manifolds).reshape(K,P,N_all)
                manifolds_all.append(manifolds)
                counter += 1

        np.save(os.path.join(emb_path, f'manifolds_layerwise.npy'), manifolds_all)
        #manifolds_all_load = manifolds_all
        print("manifolds_layerwise.npy saved!")

        # save dqs_all and EDs_all
        np.save(join(emb_path, "EDs_layerwise_all.npy"), EDs_all)
        for pc_idx in pc_idxs:
            np.save(join(emb_path, f"dqs_layerwise_{pc_idx}"), dqs_all[pc_idx])

        del dqs, dqs_layer, dqs_all, EDs, EDs_layer, EDs_all

    """
    # random projection
    N = 2048
    M = 88
    A = np.random.randn(N,M)/np.sqrt(M)

    manifolds_all = []
    for manifolds in manifolds_all_load:
        manifolds_all.append(manifolds@A)
    """
    #quit()

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

    K = len(manifolds)
    PRs_all = []
    Rs_all = []
    dists_all = []
    css_all = []
    SNRs_all = []

    # added analysis for correlation matrix (same as MLP)
    #pc_idx = 0  # select first pc of cov mat
    #EDs_all = []
    #dqs_all = []

    for manifolds in tqdm(manifolds_all):
        manifolds = np.stack(manifolds)

        # get a general version of cov mat      
        #eigvals, eigvecs = LA.eig( np.cov( manifolds.reshape( manifolds.shape[0] * manifolds.shape[1], manifolds.shape[2] ).T ) )
        #EDs_all.append( eigvals )
        #dqs_all.append( [compute_dq(eigvecs[:,pc_idx],q) for q in qs ] )
        #print(eigvecs.shape)

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

    #EDs_all = np.stack(EDs_all)
    #dqs_all = np.stack(dqs_all)

    #data_path = f"{fname}_epoch={init_epoch}"
    #if not os.path.isdir(f"{data_path}"): os.makedirs(data_path)

    np.save(os.path.join(emb_path,'SNRs_layerwise.npy'),SNRs_all)
    np.save(os.path.join(emb_path,'Ds_layerwise.npy'),PRs_all)
    np.save(os.path.join(emb_path,'dist_norm_layerwise.npy'),dists_all)
    np.save(os.path.join(emb_path,'css_layerwise.npy'),css_all)

    #np.save(join(emb_path,'EDs_layerwise.npy'), EDs_all)    
    #np.save(join(emb_path,f'dqs_layerwise_{pc_idx}.npy'), dqs_all)

    print(f"SNR metrics saved in: {emb_path}!")

def snr_submit(*args):
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

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
    #models = ["alexnet", "resnet18", "resnet34", "resnet50"]
    # large models
    models = ["resnet101", "resnet152", 
              "resnext50_32x4d", "resnext101_32x8d",
              "squeezenet1_0", "squeezenet1_1", 
              "wide_resnet50_2", "wide_resnet101_2"]
    
    fname = "embeddings_new/macaque/trained"
    pbs_array_data = [(model_name , fname)
                      for model_name in models
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
             path='/project/PDLAI/project2_data/pretrained_workflow/',
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='47:59:59',
             #walltime='23:59:59',
             mem='32GB') 

# ---------------------- Plotting ----------------------

# creates two plots
def snr_metric_plot(metric1, metric2):

    global metric_data, SNRs_all, all_models

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
                   "css"        : "css_layerwise",        
                   "ED"         : "EDs_layerwise_all"     
                   }

    name_dict = {"D"          : r'$D$',
                 "dist_norm"  : "Distance norm",
                 "css"        : "Centre subsplace",
                 "ED"         : "ED"              
                 }   

    if "dq_" in metric1:
        metric_dict[metric1] = dq_filename
        name_dict[metric1] = r'$D_2$' 

    assert metric1 in metric_dict.keys(), "metric1 not in dictionary!"
    assert metric2 in metric_dict.keys(), "metric2 not in dictionary!" 

    import matplotlib.pyplot as plt
    import pubplot as ppt

    # get available networks
    all_models = pd.read_csv(join(root_data, "pretrained_workflow", "net_names_all.csv")).loc[:,"model_name"]
    #model_names = [model_name for model_name in all_models if os.path.isfile(join(root_data,"pretrained_workflow/pretrained_dnns",model_name,"manifold/embeddings_new/macaque/trained","css_layerwise.npy"))]
    """
    model_names = ["alexnet", "resnet18", "resnet34", "resnet50", "resnet101", 
                   "resnext50_32x4d", "resnext101_32x8d", 
                   "wide_resnet50_2", "wide_resnet101_2", 
                   "squeezenet1_1"]
    """
    #model_names = ["alexnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnext101_32x8d", "squeezenet1_1"]
    model_names = ["alexnet", "resnet50", "resnet101", 
                   "resnext50_32x4d", "resnext101_32x8d", 
                   "wide_resnet50_2", "wide_resnet101_2", 
                   "squeezenet1_1"]
    
    # transparency list
    trans_ls = np.linspace(0,1,len(model_names)+1)[::-1]

    # --------------- Plot 1 ---------------
    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))
    fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=False,figsize=(9.5,7.142/2 + 0.15))
    for nidx in range(len(model_names)):
    
        model_name = model_names[nidx]
        # set paths
        init_path = join(root_data, "pretrained_workflow", "pretrained_dnns", model_name)
        manifold_path = join("manifold", "embeddings_new/macaque/trained")

        # load data     
        emb_path = join(init_path, manifold_path)
        #PRs_all = np.load(join(emb_path, "Ds_layerwise.npy"))
        metric_data = np.load( join(emb_path, f"{metric_dict[metric1]}.npy") )
        # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
        metric_data = metric_data[:,:,-1] if "dq_" in metric1 else metric_data
        SNRs_all = np.load(join(emb_path, f"{metric_dict[metric0]}.npy"))
        # fractional layer
        total_layers = SNRs_all.shape[0]
        frac_layers = np.arange(0,total_layers)/(total_layers-1)
            
        if "dq" not in metric1:
            ax1.plot(frac_layers, metric_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=model_name)
        else:
            ax1.plot(frac_layers, metric_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=model_name)
        ax2.plot(frac_layers, np.nanmean(SNRs_all,(1,2)), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=model_name)
        print(f"{model_name} done!")

    ax1.set_ylim((0,1))
    ax2.set_ylim((0.05,0.4))

    ax1.legend(frameon=False, ncol=2, fontsize=10)
    #ax2.set_yscale('log')
    #ax2.ticklabel_format(style="sci", axis="y" )

    ax1.set_ylabel(name_dict[metric1])
    ax2.set_ylabel("SNR")
    ax1.set_xlabel("Fractional depth")
    ax2.set_xlabel("Fractional depth")
    #plt.show()

    #fig_path = "/project/dnn_maths/project_qu3/fig_path"
    fig_path = "/project/PDLAI/project2_data/figure_ms"
    plt.savefig(join(fig_path, f"pretrained_snr_{metric1}-vs-layer.pdf") , bbox_inches='tight')
    print(f"Plot 1 saved!")

    # --------------- Plot 2 ---------------
    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))
    fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = False,sharey=False,figsize=(9.5,7.142/2 + 0.15))
    for nidx in range(len(model_names)):
    
        model_name = model_names[nidx]
        # set paths
        init_path = join(root_data, "pretrained_workflow", "pretrained_dnns", model_name)
        manifold_path = join("manifold", "embeddings_new/macaque/trained")

        # load data     
        emb_path = join(init_path, manifold_path)
        #PRs_all = np.load(join(emb_path, "Ds_layerwise.npy"))
        metric_data = np.load( join(emb_path, f"{metric_dict[metric2]}.npy") )
        # we are only interested in the correlation D_2 for the dqs as it reveals how localized the eigenvectors are
        metric_data = metric_data[:,:,-1] if "dq_" in metric2 else metric_data
        dq_name = "dq_0"
        dq_data = np.load( join(emb_path, f"{metric_dict[dq_name]}.npy") )
        dq_data = dq_data[:,:,-1]
        SNRs_all = np.load(join(emb_path, f"{metric_dict[metric0]}.npy"))
        # fractional layer
        total_layers = SNRs_all.shape[0]
        frac_layers = np.arange(0,total_layers)/(total_layers-1)
            
        if "dq" not in metric2:
            ax1.plot(frac_layers, metric_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=model_name)
        else:
            ax1.plot(frac_layers, metric_data.mean(-1), alpha=trans_ls[nidx], marker=markers[nidx], linestyle="-", label=model_name)

        # only scatter plot the latter layers
        deep_layers = np.where(frac_layers >= 0)
        ax2.scatter(dq_data.mean(-1)[deep_layers], np.nanmean(SNRs_all,(1,2))[deep_layers], alpha=0.6)
        print(f"{model_name} done!")
    
    #ax2.set_xlim((0,1))
    #ax2.set_ylim((0.05,0.4))

    #ax1.legend(frameon=False, ncol=2, fontsize=10)
    ax2.set_yscale('log')
    #ax2.ticklabel_format(style="sci", axis="y" )

    ax1.set_ylabel(name_dict[metric2])
    ax2.set_ylabel("SNR")
    ax1.set_xlabel("Fractional depth")
    ax2.set_xlabel(r"$D_2$")
    #plt.show()

    #fig_path = "/project/dnn_maths/project_qu3/fig_path"
    fig_path = "/project/PDLAI/project2_data/figure_ms"
    plt.savefig(join(fig_path, f"pretrained_snr_{metric2}-dq_scatter.pdf") , bbox_inches='tight')
    print(f"Plot 2 saved!")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])


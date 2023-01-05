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

t0 = time()
dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")

"""

creating manifold.npy for different types of NNs based on the phase transition diagram

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

from PIL import Image

def get_batch(i, batch_size):
#     ims = imnames[class_id]
    images = []
    for im in imnames[i*batch_size:(i+1)*batch_size]:
        #impath = os.path.join(imdir, im+ '.png')
        impath = os.path.join(imdir, im)
        images.append(preprocess(Image.open(impath)))
        
    images = torch.stack(images)
        
    return images

def cnn_stimuli(model_name, init_alpha100, init_g100, init_epoch, root_path, projection=False):

    from NetPortal.backbone_architectures import AlexnetBackbone, SqueezeBackbone, WideResNetBackbone, VGGBackbone, EfficientNetBackbone

    #global manifolds, O, output, input_tensor

    # ---------------------- MODEL ----------------------
    print("Setting up models.")

    if "alexnet" in model_name:
        matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epochs=100" in f.path and f"_{init_alpha100}_{init_g100}_" in f.path]        
    elif "resnet" in model_name:        
        matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epochs=650" in f.path and f"_{init_alpha100}_{init_g100}_" in f.path]        
    init_path = matches[0]
    print(init_path)

    #model_name = 'alexnet'
    pretrained = True
    model = torch.load(join(init_path, f"epoch_{init_epoch}/weights"), map_location=dev)

    print("Model loaded.")
    if pretrained:
        #emb_path = '/mnt/fs2/bsorsch/manifold/embeddings_new/macaque/{}/'.format(model_name)
        #emb_path = '/project/dyson/dyson_dl/embeddings_new/{}/'.format(model_name)
        emb_path = join(log_path, '/manifold/embeddings_new/macaque/{}/'.format(model_name))
    else:
        #emb_path = '/mnt/fs2/bsorsch/manifold/embeddings_new/macaque/{}_untrained/'.format(model_name)
        #emb_path = '/project/dyson/dyson_dl/embeddings_new/{}_untrained/'.format(model_name)
        emb_path = join(log_path, '/manifold/embeddings_new/macaque/{}_untrained/'.format(model_name))

    random_projection=True
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
    #backbone.to('cuda').eval()
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
    N = 2048    # dimension (set up arbitrarily)

    if projection:
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

    
    #emb_path = f'{log_path}/manifold/embeddings_new/macaque/{model_name}/'
    emb_path = join(init_path, "manifold", f"bs={batch_size}_K={K}_P={P}_N={N}_epoch={init_epoch}_projection={projection}")

    if not os.path.isdir(emb_path):
        os.makedirs(emb_path)
    np.save(os.path.join(emb_path,f'manifolds.npy'), manifolds)
    
    print("Manifold saved!")

def stimuli_submit(*args):
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

    #g100_ls = list(range(25, 301, 25))
    #alpha100_ls = list(range(100,201,10))
    #alpha100_ls = [100,150,200]
    #g100_ls = [25,100,300]
    #alpha100_ls = [100,200]
    #g100_ls = [100]        

    # ----- AlexNets -----
    #models = ["alexnet"]
    #init_epoch = 100
    #root_path = join(log_path, "alexnets_nomomentum")

    # ----- ResNets -----
    alpha100_ls = [100,200]
    g100_ls = [100, 300]
    models = ["resnet34_HT", "resnet50_HT"]    
    init_epoch = 650    
    root_path = join(log_path, "resnets")

    pbs_array_data = [(model_name, alpha100, g100, init_epoch, root_path)
                      for model_name in models
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path='/project/dyson/dyson_dl',
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='1GB') 

# ---------------------- SNR Theory ----------------------

from train_supervised import compute_dq

def snr_components(model_name, init_alpha100, init_g100, fname, root_path, projection=True):

    import numpy.linalg as LA

    """
    global Us, manifolds_all, Rs_all, dists_all, PRs_all, css_all, SNRs_all
    global EDs_all, dqs_all
    global layers
    global manifolds
    global init_epoch, emb_path
    """

    print("Setting up model path.")
    if "alexnet" in model_name:
        matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epochs=100" in f.path and f"_{init_alpha100}_{init_g100}_" in f.path]        
    elif "resnet" in model_name:        
        matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epochs=650" in f.path and f"_{init_alpha100}_{init_g100}_" in f.path]        
    init_path = matches[0]
    print(init_path)

    emb_path = join(init_path, "manifold", fname)
    manifolds = np.load(join(emb_path, f'manifolds.npy'))
    print(f"Loaded manifold from {emb_path}!")

    print("Loading model weights!")
    epoch_str = [s for s in fname.split("_") if "epoch=" in s][0]
    init_epoch = int( epoch_str[epoch_str.find("=")+1:] )
    model = torch.load(join(init_path, f"epoch_{init_epoch}/weights"), map_location=dev)

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

    #N = 2048
    #N_all = 169000
    #idxs = np.random.randint(0,N_all,N)
    batch_size = 64
    i=0

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

    #conv_idxs = [0,3,6,8,10,15]     # useless
    #np.array(layers)[conv_idxs]     # useless

    # -------------------

    N = 5000
    #N = 2048
    #N_all = 169000
    #idxs = np.random.randint(0,N_all,N)
    batch_size = 64
    i=0

    batch_size = 10
    K = 64
    P = 50

    layerwise_file = "manifolds_layerwise.npy"
    print(f"Computation for {layerwise_file}")
    if os.path.isfile( join(emb_path, layerwise_file) ):
        #manifolds_all_load = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        #manifolds_all = np.load(join(emb_path, 'manifolds_layerwise.npy'))
        print(f"{layerwise_file} computed already, loading now.")
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

                    # Get test batch
                    input_tensor = get_batch(0,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    N_all = output.shape[-1]
                    if projection:
                        O = torch.randn(N_all,N) / np.sqrt(N)
                        O = O.to(dev)
        #             idxs = np.random.randint(0,N_all,N)
                    
                    manifolds = []
                    for i in tqdm(range(K*P//batch_size)):
                        input_tensor = get_batch(i,batch_size)
                        with torch.no_grad():
                            output = backbone(input_tensor.to(dev))
                        if projection:
                            manifolds.append((output@O).cpu().numpy())
                        else:
                            manifolds.append(output.cpu().numpy())
                    if projection:  # not sure about the non-projection version
                        manifolds = np.stack(manifolds).reshape(K,P,N)
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

                manifolds = []
                for i in tqdm(range(K*P//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    if projection:
                        manifolds.append((output@O).cpu().numpy())
                    else:
                        manifolds.append(output.cpu().numpy())
                manifolds = np.stack(manifolds).reshape(K,P,N)
                manifolds_all.append(manifolds)
                counter += 1

        np.save(os.path.join(emb_path, layerwise_file), manifolds_all)
        #manifolds_all_load = manifolds_all
        print(f"{layerwise_file} saved!")

    """
    # random projection
    N = 2048
    M = 88
    A = np.random.randn(N,M)/np.sqrt(M)

    manifolds_all = []
    for manifolds in manifolds_all_load:
        manifolds_all.append(manifolds@A)
    """
    #quit()    # delete

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
    pc_idxs = list(range(25))   # top 25 PCs for each neural representation
    EDs_all = []
    dqs_all = {}
    for pc_idx in pc_idxs:
        dqs_all[pc_idx] = []

    qs = np.linspace(0,2,100)
    for manifolds in tqdm(manifolds_all):
        manifolds = np.stack(manifolds)

        # get a general version of cov mat      
        eigvals, eigvecs = LA.eig( np.cov( manifolds.reshape( manifolds.shape[0] * manifolds.shape[1], manifolds.shape[2] ).T ) )
        EDs_all.append( eigvals )
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
    for pc_idx in pc_idxs:
        dqs_all[pc_idx] = np.stack(dqs_all[pc_idx])

    #data_path = f"{fname}_epoch={init_epoch}"
    #if not os.path.isdir(f"{data_path}"): os.makedirs(data_path)

    if projection:
        np.save(os.path.join(emb_path,f'SNRs_layerwise_N={N}.npy'),SNRs_all)
        np.save(os.path.join(emb_path,f'Ds_layerwise_N={N}.npy'),PRs_all)
        np.save(os.path.join(emb_path,f'dist_norm_layerwise_N={N}.npy'),dists_all)
        np.save(os.path.join(emb_path,f'css_layerwise_N={N}.npy'),css_all)

        np.save(join(emb_path,f'EDs_N={N}_layerwise.npy'), EDs_all)    

        for pc_idx in pc_idxs:
            np.save(join(emb_path,f'dqs_layerwise_N={N}_{pc_idx}.npy'), dqs_all[pc_idx])

    else:
        np.save(os.path.join(emb_path,'SNRs_layerwise.npy'),SNRs_all)
        np.save(os.path.join(emb_path,'Ds_layerwise.npy'),PRs_all)
        np.save(os.path.join(emb_path,'dist_norm_layerwise.npy'),dists_all)
        np.save(os.path.join(emb_path,'css_layerwise.npy'),css_all)

        np.save(join(emb_path,'EDs_layerwise.npy'), EDs_all) 

        for pc_idx in pc_idxs:
            np.save(join(emb_path,f'dqs_layerwise_{pc_idx}.npy'), dqs_all[pc_idx])

    print(f"SNR metrics saved in: {emb_path}!")


# def snr_components(model_name, init_alpha100, init_g100, init_epoch, fname, root_path):
def snr_submit(*args):
    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

    # ----- AlexNets -----
    
    alpha100_ls = [100,150,200]
    g100_ls = [25,100,300]
    #no_pair = [(alpha,g) for alpha in alpha100_ls for g in g100_ls]
    #g100_ls = list(range(25, 301, 25))
    #alpha100_ls = list(range(100,201,10))

    models = ["alexnet"]
    init_epoch = 100
    root_path = join(log_path, "alexnets_nomomentum")
    #fname = "bs=10_K=64_P=50_N=2048"
    fname = "bs=10_K=64_P=50_N=2048_epoch=100_projection=False"

    # ----- ResNets -----
    """
    alpha100_ls = [100,200]
    g100_ls = [100]
    models = ["resnet34_HT", "resnet50_HT"]    
    init_epoch = 650    
    root_path = join(log_path, "resnets")
    fname = "bs=10_K=64_P=50_N=2048"
    """

    pbs_array_data = [(model_name, alpha100, g100, fname, root_path)
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
             path='/project/dyson/dyson_dl',
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='12GB') 
    

# ---------------------- Plotting ----------------------


def snr_metric_plot(model_name, init_alpha100s, init_g100s, init_epoch, root_path):

    global PRs_all, SNRs_all

    import matplotlib.pyplot as plt
    import pubplot.pubplot as ppt
    plt.rc('font', **ppt.pub_font)
    plt.rcParams.update(ppt.plot_sizes(False))
    
    init_alpha100s, init_g100s = literal_eval(init_alpha100s), literal_eval(init_g100s)
    # plot dimension for now
    for init_g100 in init_g100s:
        fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=False,figsize=(9.5,7.142/2 + 0.15))
        for init_alpha100 in init_alpha100s:
            # load data
            if "alexnet" in model_name:
                matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epochs=100" in f.path and f"_{init_alpha100}_{init_g100}_" in f.path]        
            elif "resnet" in model_name:        
                matches = [f.path for f in os.scandir(root_path) if f.is_dir() and "epochs=650" in f.path and f"_{init_alpha100}_{init_g100}_" in f.path]        
            init_path = matches[0]
            #print(init_path)
            emb_path = join(init_path,"manifold")
            PRs_all = np.load(join(emb_path, "Ds_layerwise.npy"))
            SNRs_all = np.load(join(emb_path, "SNRs_layerwise.npy"))
                
            #break
            #ax1.plot(1/PRs_all.mean(-1), label=init_alpha100)
            ax1.plot(PRs_all.mean(-1), label=int(init_alpha100)/100)
            ax2.plot(np.nanmean(SNRs_all,(1,2)), label=int(init_alpha100)/100)

        ax1.legend(frameon=False)
        ax1.set_yscale('log')
        #ax1.set_ylabel(r'$1/D$')
        ax1.set_ylabel(r'$D$')
        ax2.set_ylabel("SNR")
        ax1.set_xlabel("Layer")
        ax2.set_xlabel("Layer")
        #plt.show()

        fig_path = "/project/dnn_maths/project_qu3/fig_path"
        plt.savefig(f"{fig_path}/{model_name}_g={init_g100s[0]/100}_snr_metrics.pdf", bbox_inches='tight')
        print(f"Plot saved for {init_g100}!")



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])


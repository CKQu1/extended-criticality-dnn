import numpy as np
import pandas as pd
import sys
import torch
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
#import seaborn as sns
import os
#sns.set_style('darkgrid')
#sns.set(font_scale=1.3, rc={"lines.linewidth": 2.5})
#from scipy.io import loadmat

from time import time
from os.path import join
lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
import path_names
from path_names import log_path, cnn_path

t0 = time()
dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")

"""

creating manifold.npy for different types of NNs
based on https://github.com/bsorsch/geometry-fewshot-learning/blob/master/ResNet_macaque_stimuli.ipynb

"""

# ---------------------- DATA ----------------------
print("Loading data.")

#imnames = pd.read_csv(join(os.getcwd(),"brain-score/image_dicarlo_hvm-public.csv"))
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

# ---------------------- MODEL ----------------------

model_name = 'resnet50'
#model_name = 'alexnet'
print(f"Setting up model {model_name}.")
repo = 'pytorch/vision:v0.6.0'
# repo = 'rwightman/gen-efficientnet-pytorch'
pretrained = True
#model = torch.hub.load(repo, model_name, pretrained=pretrained, map_location=dev)
#model = torch.hub.load(repo, model_name, pretrained=pretrained, force_reload=True)
model = torch.hub.load(repo, model_name, pretrained=pretrained)
#model = torch.load(model_name, pretrained=pretrained,map_location=dev)

#from torchvision.models import resnet50
#model = resnet50().to(dev)

print("Model loaded.")
if pretrained:
    #emb_path = '/mnt/fs2/bsorsch/manifold/embeddings_new/macaque/{}/'.format(model_name)
    #emb_path = '/project/dyson/dyson_dl/embeddings_new/{}/'.format(model_name)
    emb_path = join(log_path, '/manifold/embeddings_new/macaque/{}/'.format(model_name))
else:
    #emb_path = '/mnt/fs2/bsorsch/manifold/embeddings_new/macaque/{}_untrained/'.format(model_name)
    #emb_path = '/project/dyson/dyson_dl/embeddings_new/{}_untrained/'.format(model_name)
    emb_path = join(log_path, '/manifold/embeddings_new/macaque/{}_untrained/'.format(model_name))

class AlexnetBackbone(torch.nn.Module):
    def __init__(self, model):
        super(AlexnetBackbone, self).__init__()
        self.N = list(model.children())[-1][-1].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.features = torch.nn.Sequential(*list(model.children())[-1][:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.flatten(x)
        x = self.features(x)
        return x


class SqueezeBackbone(torch.nn.Module):
    def __init__(self, model):
        super(SqueezeBackbone, self).__init__()
        self.N = list(model.children())[-1][-3].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.features = torch.nn.Sequential(*list(model.children())[-1][:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.features(x)
        x = self.flatten(x)
        return x


class WideResNetBackbone(torch.nn.Module):
    def __init__(self, model, module_idx):
        super(WideResNetBackbone, self).__init__()
        self.N = list(model.children())[-1].weight.shape[-1]
        self.features = torch.nn.Sequential(*list(model.children())[:-module_idx])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

    
class VGGBackbone(torch.nn.Module):
    def __init__(self, model):
        super(VGGBackbone, self).__init__()
        self.N = list(model.children())[-1][3].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.features = torch.nn.Sequential(*list(model.children())[-1][:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.flatten(x)
        x = self.features(x)
        x = self.flatten(x)
        return x
    
    
class EfficientNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super(EfficientNetBackbone, self).__init__()
        self.N = list(model.children())[-1].weight.shape[-1]
        self.pre_features = torch.nn.Sequential(*list(model.children())[:-1])
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = self.pre_features(x)
        x = self.flatten(x)
        return x

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
backbone.to(dev).eval()


print(f"{time() - t0}s")
# ---------------------- MANIFOLD COMPUTATION ----------------------

print("Computing manifold.")

batch_size = 10
input_tensor = get_batch(0,batch_size)
with torch.no_grad():
    output = backbone(input_tensor.to(dev)).cpu()

N = 2048
M = output.shape[1]
# idxs = torch.randint(0,100352,(N,))
O = torch.randn((M,2048))/np.sqrt(2048)
O = O.to(dev)


# N = 2048
K = 64
P = 50
#K = 100
#P = 500
batch_size = 10
n_classes = len(imnames)

emb_path = f'{log_path}/manifold/embeddings_new/macaque/{model_name}/'

if os.path.isfile( join(emb_path,f'manifolds_bs={batch_size}_K={K}_P={P}_N={N}.npy') ):
    manifolds = np.load(join(emb_path,f'manifolds_bs={batch_size}_K={K}_P={P}_N={N}.npy'))
else:
    manifolds = []
    # for class_id in tqdm(range(n_classes)):
    for i in tqdm(range(K*P//batch_size)):
        input_tensor = get_batch(i,batch_size)
        with torch.no_grad():
            output = backbone(input_tensor.to(dev))
        manifolds.append((output@O).cpu().numpy())
    manifolds = np.stack(manifolds).reshape(K,P,N)


    # save manifold
    if not os.path.isdir(emb_path):
        os.makedirs(emb_path)
    np.save(join(emb_path,f'manifolds_bs={batch_size}_K={K}_P={P}_N={N}.npy'), manifolds)

# ---------------------- SNR Theory ----------------------

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

N = 2048
N_all = 169000
idxs = np.random.randint(0,N_all,N)
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

conv_idxs = [0,3,6,8,10,15]

np.array(layers)[conv_idxs]

# -------------------

N = 2048
N_all = 169000
idxs = np.random.randint(0,N_all,N)
batch_size = 64
i=0

batch_size = 10
K = 64
P = 50

if os.path.isfile( join(emb_path, 'manifolds_layerwise.npy') ):
    #manifolds_all_load = np.load(join(emb_path, 'manifolds_layerwise.npy'))
    manifolds_all = np.load(join(emb_path, 'manifolds_layerwise.npy'))

else:
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
                O = torch.randn(N_all,N) / np.sqrt(N)
                O = O.to(dev)
    #             idxs = np.random.randint(0,N_all,N)
                
                manifolds = []
                for i in tqdm(range(K*P//batch_size)):
                    input_tensor = get_batch(i,batch_size)
                    with torch.no_grad():
                        output = backbone(input_tensor.to(dev))
                    manifolds.append((output@O).cpu().numpy())
                manifolds = np.stack(manifolds).reshape(K,P,N)
                manifolds_all.append(manifolds)

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
            O = torch.randn(N_all,N) / np.sqrt(N)
            O = O.to(dev)
    #         idxs = np.random.randint(0,N_all,N)        

            manifolds = []
            for i in tqdm(range(K*P//batch_size)):
                input_tensor = get_batch(i,batch_size)
                with torch.no_grad():
                    output = backbone(input_tensor.to(dev))
                manifolds.append((output@O).cpu().numpy())
            manifolds = np.stack(manifolds).reshape(K,P,N)
            manifolds_all.append(manifolds)

    np.save(os.path.join(emb_path, 'manifolds_layerwise.npy'), manifolds_all)
    #manifolds_all_load = manifolds_all

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


from scipy.spatial.distance import pdist, squareform
m = 5

K = len(manifolds)
PRs_all = []
Rs_all = []
dists_all = []
css_all = []
SNRs_all = []
for manifolds in tqdm(manifolds_all):
    manifolds = np.stack(manifolds)
    
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

plt.plot(1/PRs_all.mean(-1), 'o-', c='C0', label='')
plt.ylabel(r'$1/D_{SVD}$')
plt.show()

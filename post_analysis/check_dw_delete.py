import os
import sys
from os.path import join

sys.path.append(os.getcwd())
from NetPortal.models import ModelFactory
from path_names import root_data

alpha100, g100 = 100, 100
epoch = 650

# storage of trained nets
L = 10
total_epoch = 650
fcn = f"fc{L}"
net_type = f"{fcn}_mnist_tanh"
data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")

# Extract numeric arguments.
alpha, g = int(alpha100)/100., int(g100)/100.  

# load nets and weights
net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
hidden_N = [784]*L + [10]
kwargs = {"dims": hidden_N, "alpha": None, "g": None,
            "init_path": join(data_path, net_folder), "init_epoch": epoch,
            "activation": 'tanh', "with_bias": False,
            "architecture": 'fc'}
net = ModelFactory(**kwargs)    

from nporch.input_loader import get_data_normalized
image_type = 'mnist'
trainloader , _, _ = get_data_normalized(image_type,1)

input_idx = 0
image = trainloader.dataset[input_idx][0]
postj = net.layerwise_jacob_ls(image, True)
prej = net.layerwise_jacob_ls(image, False)
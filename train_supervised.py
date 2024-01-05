# original version: https://gist.github.com/asemptote/3c9f901f1346dffb29d21742cb83c933
import numpy as np
import os
import pandas as pd  
import path_names
import torch
from os.path import join
from time import time
from torch.autograd import Variable
from tqdm import tqdm
from scipy.stats import levy_stable

from ast import literal_eval
from path_names import root_data, log_model, read_log
from utils_dnn import save_weights, get_weights, compute_dq, compute_IPR, layer_ipr, effective_dimension, store_model

dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")
print(dev)

# Dataset management -------------------------------------------------------------------------------------------

import torchvision
from torchvision import datasets

def transform_cifar10(image):   # flattening cifar10
    return (torch.Tensor(image.getdata()).T.reshape(-1)/255)*2 - 1

def set_data(name, rshape: bool, **kwargs):
    import torchvision.transforms as transforms

    data_path = "data"
    if rshape:        
        if name.lower() == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),
                                            transforms.Lambda(lambda x: torch.flatten(x))]
                                            )
            #transform = transform_cifar10
            train_ds = datasets.MNIST(root=data_path, download=True, transform=transform)
            valid_ds = datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
        elif name.lower() == "gaussian":
            from generate_gaussian_data import delayed_mixed_gaussian
            #num_train, num_test, X_dim, Y_classes, X_clusters, n_hold, final_time_point, noise_sigma=0,
                #cluster_seed=None, assignment_and_noise_seed=None, avg_magn=0.3, min_sep=None,
                #freeze_input=False
            
            # same setting as MNIST
            num_train, num_test = kwargs.get("num_train"), kwargs.get("num_test")
            X_dim = kwargs.get("X_dim")
            Y_classes, X_clusters = kwargs.get("Y_classes"), kwargs.get("X_clusters")
            n_hold, final_time_point = kwargs.get("n_hold"), kwargs.get("final_time_point")    # not needed for feedforwards nets, always set to 0 for MLPs
            noise_sigma = kwargs.get("noise_sigma")
            cluster_seed, assignment_and_noise_seed = kwargs.get("cluster_seed"), kwargs.get("assignment_and_noise_seed")
            avg_magn = kwargs.get("avg_magn")
            min_sep = kwargs.get("min_sep")
            freeze_input = kwargs.get("freeze_input")

            class_datasets, centers, cluster_class_label = delayed_mixed_gaussian(num_train, num_test, X_dim, Y_classes, X_clusters, n_hold,
                                                                                  final_time_point, noise_sigma,
                                                                                  cluster_seed, assignment_and_noise_seed, avg_magn,                                        
                                                                                  min_sep, freeze_input)
            train_ds = class_datasets['train']
            valid_ds = class_datasets['val']        

        elif name.lower() == "fashionmnist":
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)), 
                                            transforms.Lambda(lambda x: torch.flatten(x))]
                                            )
            train_ds = datasets.FashionMNIST(root=data_path, download=True, train=True, transform=transform)
            valid_ds = datasets.FashionMNIST(root=data_path, download=True, train=False, transform=transform)
        elif name.lower() == 'cifar10':
            transform=transform_cifar10
            train_ds = datasets.CIFAR10(root=data_path, download=True, transform=transform)
            valid_ds = datasets.CIFAR10(root=data_path, download=True, transform=transform, train=False)
        else:
            raise NameError("name is not defined in function set_data")

    else:
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        """
        transform_train = transforms.Compose([transforms.Resize((70, 70)),
                                               transforms.RandomCrop((64, 64)),
                                               transforms.ToTensor(),
                                               normalize,])

        transform_test = transforms.Compose([transforms.Resize((70, 70)),
                                              transforms.CenterCrop((64, 64)),
                                              transforms.ToTensor(),
                                              normalize,])
        """

        if name == 'mnist':
            """
            train_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform_train)
            valid_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform_test, train=False)
            """
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            train_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize,]))
            valid_ds = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize,]))

        elif name == 'cifar100':
            #mean and std of cifar100 dataset
            CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

            #CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
            #CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

            transform_train = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
            #cifar100_training = CIFAR100Train(path, transform=transform_train)
            train_ds = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
            #cifar100_test = CIFAR100Test(path, transform=transform_test)
            valid_ds = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)


        elif name == 'cifar10':
            if "cifar_upsize" in kwargs and kwargs.get("cifar_upsize") == True:     # for AlexNet (torch version)
                upsize_cifar10 = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                train_ds = torchvision.datasets.CIFAR10(root=join(data_path, "cifar10_upsize"), download=True, transform=upsize_cifar10)
                valid_ds = torchvision.datasets.CIFAR10(root=join(data_path, "cifar10_upsize"), download=True, transform=upsize_cifar10, train=False)
            else:
                train_ds = torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform_train)
                valid_ds = torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform_test, train=False)

        elif name == 'cifar10circ':
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1994, 0.2010])
            
            train_ds = torchvision.datasets.CIFAR10(root=join(data_path,"cifar10_circ"), train=True, download=True,
                                                    transform=transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]))
            
            valid_ds = torchvision.datasets.CIFAR10(root=join(data_path,"cifar10_circ"), train=False, download=True,
                                                    transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]))      

        else:
            raise NameError("name is not defined in function set_data")
    
    if name.lower() == "gaussian":
        return train_ds, valid_ds, centers, cluster_class_label
    else:
        return train_ds, valid_ds

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def get_data(train_ds, valid_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )
    
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func  # func(data, targets)

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))       

from torch import nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# Training methods ----------------------------------------------------------------------------------------

def loss_batch(model, loss_func, xb, yb, opt=None, stats=True):
    xb = xb.to(dev)
    yb = yb.to(dev)
    model_xb = model(xb)
    loss = loss_func(model_xb, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    # cannot use count_nonzero for older verions of pytorch, i.e. before 1.6.0
    #if stats: return loss.item(), len(xb), torch.count_nonzero(model_xb.argmax(dim=1)==yb).item()
    if stats:
        batch = model_xb.argmax(dim=1)==yb 
        return loss.item(), len(xb), torch.numel(batch.nonzero())

def get_eigs(W):
    if W.shape[0] == W.shape[1]:
        eigvals, _ = torch.eig(W)
    else:   
        # return as complex pair        
        _, eigvals, _ = torch.svd(W)
        eigvals = torch.stack( (eigvals, torch.zeros_like(eigvals)), 1)

    return eigvals

# save eigvals/singvals of weight matrices (should perhaps merge with stablefit_model)
def store_model_eigvals(model, path):
    Ws = get_weights(model)
    wm_total = len(Ws)              # total number of weight matrices
    for widx in range(wm_total):
        if widx == 0:
            eigvals_all = get_eigs(Ws[widx])
        else:
            eigvals_all = torch.cat((eigvals_all, get_eigs(Ws[widx])), 0)

    # synchronize the GPU and ram to speed up cpu().numpy() conversion
    # https://discuss.pytorch.org/t/convert-tensor-to-numpy-is-very-slow/122220
    if dev.type != 'cpu':
        torch.cuda.synchronize()

    eigvals_all = eigvals_all.detach().cpu().numpy()
    eigvals_all = eigvals_all[:,0] + eigvals_all[:,1] * 1j
    np.save(f"{path}", eigvals_all)    

# fit weight distribution for fully-connected individually layers
pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
def stablefit_model(model):
    Ws = get_weights(model)
    wm_total = len(Ws)              # total number of weight matrices
    stable_params = np.empty([wm_total, 4])

    if dev.type != 'cpu':
        torch.cuda.synchronize()
    for widx in range(wm_total):
        stable_params[widx,:] = pconv(*levy_stable._fitstart(Ws[widx].detach().cpu().numpy().flatten()))    
    return stable_params  

def train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=0, 
                save_type='numpy', **kwargs):

    """
    Both `weight_save`, `stablefit_save`, `eigvals_save` have 3 mods, 
    - 0: save nothing
    - 1: save end of epoch
    - 2 save all steps
    """
    assert save_type in ['numpy', 'torch'], "save_type does not exist!"

    start = time()
    model.train()
    save_epoch, save_step = kwargs.get('weight_save') == 1, kwargs.get('weight_save') == 2
    stablefit_epoch, stablefit_step = kwargs.get('stablefit_save') == 1, kwargs.get('stablefit_save') == 2
    eigvals_epoch, eigvals_step = kwargs.get('eigvals_save') == 1, kwargs.get('eigvals_save') == 2

    if opt is not None:
        # initialize
        params_step = np.zeros([len(train_dl),len(list(model.parameters())),4]) if stablefit_step else None

        for hidden_epoch in range(hidden_epochs):
            for batch_idx, (xb, yb) in enumerate(train_dl):
                # each time step of SGD
                loss_batch(model, loss_func, xb, yb, opt, stats=False)
                if save_step:    # needed if want to save all steps
                    store_model(model, f"{kwargs.get('epoch_path')}/weights_{batch_idx}", save_type)
                if eigvals_step:
                    store_model_eigvals(model, f"{kwargs.get('epoch_path')}/eigvals_{batch_idx}")
                if stablefit_step:
                    params_step[batch_idx,:,:] = stablefit_model(model)
    else:
        # initialize
        params_step = np.zeros([1,len(list(model.parameters())),4]) if stablefit_step else None

        if save_step:    # needed if want to save all steps
            store_model(model, f"{kwargs.get('epoch_path')}/weights_0", save_type)
        if eigvals_step:
            store_model_eigvals(model, f"{kwargs.get('epoch_path')}/eigvals_0")
        if stablefit_step:
            params_step[0,:,:] = stablefit_model(model)

    if save_epoch and (not save_step):    # save only the last step of each epoch
        store_model(model, f"{kwargs.get('epoch_path')}/weights", save_type)  
    if eigvals_epoch and (not eigvals_step):
        store_model_eigvals(model, f"{kwargs.get('epoch_path')}/eigvals")
    if stablefit_epoch and (not stablefit_step):
        params_epoch = stablefit_model(model)

    # pure evaluation of train and test acc/loss
    losses, nums, corrects = zip(
        *[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
    )
    train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    train_acc = np.sum(corrects) / np.sum(nums)
    model.eval()
    with torch.no_grad():
        losses, nums, corrects = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        )
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    val_acc = np.sum(corrects) / np.sum(nums)

    # save levy alpha stable distribution (for MLP)
    if not (stablefit_epoch or stablefit_step):
        return (train_loss, train_acc, val_loss, val_acc, f'{time()-start:.2f}s')
    elif stablefit_epoch:
        return (train_loss, train_acc, val_loss, val_acc, f'{time()-start:.2f}s'), params_epoch
    elif stablefit_step:
        return (train_loss, train_acc, val_loss, val_acc, f'{time()-start:.2f}s'), params_step

def get_optimizer(optimizer, model, **kwargs):
    if optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), **kwargs)
    # for the following use the default setting
    elif optimizer == 'adam':
        opt = optim.Adam(model.parameters(), **kwargs)
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(), **kwargs)        
    else:
        raise ValueError("THIS OPTIMIZER DOES NOT EXIST!")
    return opt

# --------------------------------------- Training functions ---------------------------------------

from torch import optim
import torch.nn.functional as F

# original setting
"""
(name, alpha100, g100, optimizer, bs, net_type = "fc", activation='tanh', hidden_structure=2, depth=10,  # network structure
 loss_func_name="cross_entropy", lr=0.001, momentum=0.9, num_workers=1, epochs=650,                      # training setting
 save_epoch=50, weight_save=1, stablefit_save=0,                                                        # save data options
 with_ed=False, with_pc=False, with_ipr=False):
"""

# maybe write a network Wrapper to include differenet kinds of attributes like ED, IPR, etc

# FC10: lr=0.001, activation='tanh'
def train_ht_dnn(name, Y_classes, X_clusters, cluster_seed, assignment_and_noise_seed, num_train, num_test,
                 alpha100, g100, optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                 epochs=25, save_epoch=50, net_type="fc", activation='tanh', hidden_structure=2, with_bias=False,                                    # network structure
                 loss_func_name="cross_entropy", momentum=0, weight_decay=0, num_workers=1,                      # training setting
                 weight_save=1, stablefit_save=0, eigvals_save=0,                                       # save data options
                 with_ed=False, with_pc=False, with_ipr=False):
                 #with_ed=True, with_pc=True, with_ipr=False):                                                            # save extra data options

    #global stablefit_params_all, stablefit_params, model

    """
    Trains MLPs and saves weights along the steps of training, can add to save gradients, etc.
    """   

    from time import time; t0 = time()
    from NetPortal.models import ModelFactory

    Y_classes, X_clusters = int(Y_classes), int(X_clusters)
    init_path = None if init_path.lower() == "none" else init_path
    init_epoch = None if init_epoch.lower() == "none" else init_epoch
    if isinstance(alpha100,str)==True or isinstance(g100,str)==True:
        if alpha100 != "None" or g100 != "None": 
            alpha100, g100 = int(alpha100), int(g100)
            alpha, g = int(alpha100)/100., int(g100)/100.
        else:
            alpha100, g100 = None, None
            alpha, g = None, None
    else:
        alpha100, g100 = int(alpha100), int(g100)
    bs, lr = int(bs), float(lr)
    epochs = int(epochs)
    save_epoch = int(save_epoch)
    depth = int(depth)
    num_workers = int(num_workers)
    with_bias = literal_eval(with_bias) if isinstance(with_bias, str) else with_bias

    if name.lower() != "gaussian":
        train_ds, valid_ds = set_data(name,True)
        Y_classes, X_clusters= None, None
        cluster_seed, assignment_and_noise_seed = None, None
    else:
        # similar setting to MNIST, can adjust
        X_dim = 784
        noise_sigma = 0.2
        num_train, num_test = int(num_train), int(num_test)
        Y_classes, X_clusters = int(Y_classes), int(X_clusters)
        cluster_seed, assignment_and_noise_seed = int(cluster_seed), int(assignment_and_noise_seed)
        gaussian_data_kwargs = {"num_train": num_train, "num_test": num_test,
                                "X_dim": 784,
                                "Y_classes": Y_classes, "X_clusters": X_clusters,
                                "n_hold": 0, "final_time_point": 0,     # not needed for feedforwards nets
                                "noise_sigma": noise_sigma,
                                "cluster_seed": cluster_seed, "assignment_and_noise_seed": assignment_and_noise_seed,
                                "avg_magn": 1,
                                "min_sep":None,
                                "freeze_input": True
                                }

        train_ds, valid_ds, centers, cluster_class_label = set_data(name,True,**gaussian_data_kwargs)    
    #N = len(train_ds[0][0])
    N = train_ds[0][0].numel()
    if name != "gaussian":
        C = len(train_ds.classes)
        #root_path += f"_{activation}" + f"_{name}"
    else:
        C = len(np.unique(cluster_class_label))
        #root_path += f"_{activation}" + "_gaussian_data" + f"_{num_train}_{num_test}_{X_dim}_{Y_classes}_{X_clusters}" +  f"_{noise_sigma}_{cluster_seed}_{assignment_and_noise_seed}"

    # generate random id and date
    import uuid
    from datetime import datetime
    model_id = str(uuid.uuid1())
    train_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")  

    # create path
    model_path = f"{root_path}/{net_type}{depth}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}"
    if not os.path.isdir(model_path): os.makedirs(f'{model_path}')
    # log marker
    np.savetxt(f"{model_path}/lr={lr}_momentum={momentum}_bs={bs}", np.array([0]))
    np.savetxt(f"{model_path}/epochs={epochs}_save_epoch={save_epoch}_weight_save={weight_save}_stablefit_save={stablefit_save},eigvals_save={eigvals_save}", np.array([0]))
    np.savetxt(f"{model_path}/with_ed={with_ed}_with_ipr={with_ipr}_with_pc={with_pc}", np.array([0]))

    if name.lower() == "gaussian":
        # only need to save the associated parameters, i.e. seeds which generated the dataset
        if not os.path.isfile(join(root_path, "gaussian_data_setting.csv")):
            gaussian_data_setting = pd.DataFrame(gaussian_data_kwargs, index=[0], dtype=object)
            gaussian_data_setting.to_csv(join(root_path, "gaussian_data_setting.csv"), index=False)
        # save centers and labels for gaussian data (no longer needed)
        """
        if not os.path.isfile(join(root_path, f"centers.npy")):
            np.save(join(root_path, "centers"), centers)
        if not os.path.isfile(join(root_path, f"cluster_class_label.npy")):
            np.save(join(root_path, "cluster_class_label"), cluster_class_label)
        """

    # move entire dataset to GPU
    if name.lower() != "gaussian":
        train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(dev),
                                 torch.tensor(train_ds.targets).to(dev))
        valid_ds = TensorDataset(torch.stack([e[0] for e in valid_ds]).to(dev),
                                 torch.tensor(valid_ds.targets).to(dev))
    else:
        train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(dev),
                                 torch.stack([e[1] for e in train_ds]).to(dev))
        valid_ds = TensorDataset(torch.stack([e[0] for e in valid_ds]).to(dev),
                                 torch.stack([e[1] for e in valid_ds]).to(dev))

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)#, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    steps = len(train_dl)
    #print(f"Training data shape: {len(train_dl.dataset)}")

    # new method
    hidden_N = [N]
    if hidden_structure == 1:   # network setup 1 (power-law decreasing layer size)
        a = (C/N)**(1/depth)
        hidden_N = hidden_N + [int(N*a**l) for l in range(1, depth)]
    elif hidden_structure == 2:   # network setup 2 (square weight matrix)
        hidden_N = hidden_N + [N]*(depth - 1)
    else: 
        assert isinstance(hidden_structure,list), 'hidden_N must be a list!'
        assert len(hidden_structure) == depth - 1, 'hidden_N length and depth inconsistent!'
        hidden_N += hidden_structure

    hidden_N.append(C)
    # pretrained/saved weights    
    """
    if "init_path" in pretrained and "init_epoch" in pretrained:
        kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                  "init_path": pretrained.get("init_path"), "init_epoch": pretrained.get("init_epoch"),
                  "activation": activation, "architecture": net_type}    
    """
    # randomly initialized weights
    #else:
    kwargs = {"dims": hidden_N, "alpha": alpha, "g": g,
              "init_path": None, "init_epoch": None,
              "activation": activation, "with_bias": with_bias,
              "architecture": net_type}
    model = ModelFactory(**kwargs)
    # info
    print(f"Total layers = {depth+1}", end=' ')
    print(f"Network hidden layer structure: type {hidden_structure}" + "\n")
    print(f"Training method: {optimizer}, lr={lr}, batch_size={bs}, epochs={epochs}" + "\n")
    model.to(dev)   # move to CPU/GPU
    print(model)

    print("Saving information into local model_path before training.")
    # save information to model_path locally
    print((model_id,train_date,name))
    log_model(model_path, model_path, file_name="net_log", local_log=False, net_type=net_type, model_dims=model.dims, model_id=model_id, train_date=train_date, name=name, 
              alpha100=alpha100, g100=g100, activation=activation, loss_func_name=loss_func_name, optimizer=optimizer, hidden_structure=hidden_structure, depth=depth, bs=bs, lr=lr, 
              num_workers=num_workers, epochs=epochs, steps=steps,
              Y_classes=Y_classes, X_clusters=X_clusters)

    # for effective dimension (ED) analysis
    if with_pc or with_ipr:
        q_ls = torch.tensor(np.linspace(0,2,50))
    if with_ed:
        train_ed, valid_ed = get_data(train_ds, valid_ds, len(train_ds))
        images_train, labels_train = next(iter(train_ed))
        images_valid, labels_valid = next(iter(valid_ed))
        del train_ed, valid_ed

        ed_train = np.zeros([len(model.sequential), epochs + 1])
        ed_test = np.zeros([len(model.sequential), epochs + 1])
    if with_ipr:
        train_ed, valid_ed = get_data(train_ds, valid_ds, len(train_ds))
        images_train, labels_train = next(iter(train_ed))
        images_valid, labels_valid = next(iter(valid_ed))
        del train_ed, valid_ed   

    opt_kwargs = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay} if optimizer == "sgd" else {"lr": lr, "weight_decay": weight_decay}
    opt = get_optimizer(optimizer, model, **opt_kwargs)

    # initial state of network
    print(f'Epoch 0:', end=' ')
    loss_func = F.__dict__[loss_func_name]
    epoch_path = f"{model_path}/epoch_0"
    if not os.path.isdir(epoch_path): os.makedirs(epoch_path)
    
    if stablefit_save > 0:
        epoch0_data, stablefit_params = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, 
                                                    weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=eigvals_save)
    else:
        epoch0_data = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, 
                                                    weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=eigvals_save)

    print(epoch0_data)
    #if not os.path.isdir(f'{model_path}/epoch_0'): os.makedirs(f'{model_path}/epoch_0')
    acc_loss = pd.DataFrame(columns=['train loss', 'train acc', 'test loss', 'test acc'], dtype=object)
    acc_loss.loc[0,:] = epoch0_data[:-1]

    if stablefit_save > 0:
        stablefit_params_all = []
        for widx in range(len(list(model.parameters()))):
            stablefit_params_all.append( pd.DataFrame(columns=['alpha', 'beta', 'mu', 'sigma'], dtype=object) )
        if stablefit_save == 1:
            for widx in range(len(stablefit_params_all)):
                stablefit_params_all[widx].loc[0,:] = stablefit_params[widx,:]
        elif stablefit_save == 1:
            for widx in range(len(stablefit_params_all)):
                stablefit_params_all[widx].loc[0,:] = stablefit_params[0,widx,:]

    if with_ed:
        with torch.no_grad():
            ed_train[:,0], C_dims, pc_dqss_train = effective_dimension(model,images_train,with_pc,q_ls)
            ed_test[:,0], _, pc_dqss_test = effective_dimension(model,images_valid,with_pc,q_ls)   
            if with_pc:
                np.savetxt(f"{model_path}/C_dims", C_dims)
                np.savetxt(f"{model_path}/pc_dqss_train_0", pc_dqss_train)
                np.savetxt(f"{model_path}/pc_dqss_test_0", pc_dqss_test)                                             
 
    if with_ipr:
        with torch.no_grad():
            epoch0_ipr_train = layer_ipr(model,images_train,q_ls)
            epoch0_ipr_test = layer_ipr(model,images_valid,q_ls)
            #print(epoch0_ipr_test)
            np.savetxt(f"{model_path}/ipr_train_0", epoch0_ipr_train)
            np.savetxt(f"{model_path}/ipr_test_0", epoch0_ipr_test)
        del epoch0_ipr_train, epoch0_ipr_test

    # training
    for epoch in tqdm(range(epochs)):
        #print(f'Epoch {(1+epoch)*(1+num_workers)}:', end=' ')
        print(f'Epoch {(1+epoch)*(num_workers)}:', end=' ')
        # record selected accuracies
        epoch_path = f"{model_path}/epoch_{epoch + 1}"
        if ((epoch+1) % save_epoch == 0 or epoch == epochs - 1) and weight_save != 0:
            # train_loss, train_acc, val_loss, val_acc, epoch_time
            if not os.path.isdir(epoch_path): os.makedirs(epoch_path)
            if stablefit_save > 0:
                final_acc, stablefit_params = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
                                        epoch_path=epoch_path,
                                        weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=eigvals_save)            
            else:
                final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
                                        epoch_path=epoch_path,
                                        weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=eigvals_save)
            
            print(final_acc)
        else:           
            if (not os.path.isdir(epoch_path)) and eigvals_save > 0: 
                os.makedirs(epoch_path)
                if stablefit_save > 0:
                    final_acc, stablefit_params = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
                                            epoch_path=epoch_path,
                                            weight_save=0, stablefit_save=stablefit_save, eigvals_save=eigvals_save)
                else:
                    final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
                                            epoch_path=epoch_path,
                                            weight_save=0, stablefit_save=stablefit_save, eigvals_save=eigvals_save)

            else:
                final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers)                

            print(final_acc)

        #if stablefit_save > 0:
        #    print(stablefit_params)     # delete
        if stablefit_save == 1:
            for widx in range(len(stablefit_params_all)):
                stablefit_params_all[widx].loc[epoch+1,:] = stablefit_params[widx,:]
        if stablefit_save == 2:
            for widx in range(len(stablefit_params_all)):
                row_dict = {}
                for param_idx, col_name in enumerate(stablefit_params_all[widx].columns):
                    row_dict[col_name] = stablefit_params[:,widx,param_idx]
                    #stablefit_params_all[widx].loc[ epoch*steps + 1 : (epoch+1)*steps + 1, :] = stablefit_params[:,widx,:]   

                df_epoch = pd.DataFrame(row_dict, dtype=object)
                stablefit_params_all[widx] = stablefit_params_all[widx].append(df_epoch,ignore_index=True)     

        if with_ed:
            with torch.no_grad():
                if epoch <= 19 or (epoch + 1) % 5 == 0:
                    ed_train[:,epoch+1], _, pc_dqss_train = effective_dimension(model,images_train,with_pc,q_ls)
                    ed_test[:,epoch+1], _, pc_dqss_test = effective_dimension(model,images_valid,with_pc,q_ls)  
                    if with_pc:
                        np.savetxt(f"{model_path}/pc_dqss_train_{epoch+1}", pc_dqss_train)
                        np.savetxt(f"{model_path}/pc_dqss_test_{epoch+1}", pc_dqss_test)          
                else:
                    ed_train[:,epoch+1], _, _ = effective_dimension(model,images_train,False,q_ls)
                    ed_test[:,epoch+1], _, _ = effective_dimension(model,images_valid,False,q_ls)     
        
        if with_ipr:
            with torch.no_grad():
                ipr_train = layer_ipr(model,images_train,q_ls)
                ipr_test = layer_ipr(model,images_valid,q_ls)
            np.savetxt(f"{model_path}/ipr_train_{epoch+1}", ipr_train)
            np.savetxt(f"{model_path}/ipr_test_{epoch+1}", ipr_test)

        acc_loss.loc[epoch + 1,:] = final_acc[:-1]

    # stablefit params
    if stablefit_save == 1:
        for widx in range(len(stablefit_params_all)):
            stablefit_params_all[widx].to_csv(f"{model_path}/stablefit_epoch_widx={widx}")
    elif stablefit_save == 2:
        for widx in range(len(stablefit_params_all)):
            stablefit_params_all[widx].to_csv(f"{model_path}/stablefit_step_widx={widx}")
        
    # at least store accuracy at the end of each epoch
    acc_loss.to_csv(f"{model_path}/acc_loss", index=False)

    total_time = time() - t0
    # log dataframe
    log_path = root_data
    log_model(log_path, model_path, file_name="net_log", net_type=net_type, model_dims=model.dims, model_id=model_id, train_date=train_date, name=name, alpha100=alpha100, g100=g100, 
              activation=activation, loss_func_name=loss_func_name, optimizer=optimizer, hidden_structure=hidden_structure, depth=depth, bs=bs, lr=lr, num_workers=num_workers, 
              epochs=epochs, steps=steps, final_acc=final_acc, total_time=total_time,
              Y_classes=Y_classes, X_clusters=X_clusters)

    if with_ed:
        np.savetxt(f"{model_path}/ed_train",ed_train)
        np.savetxt(f"{model_path}/ed_test",ed_test)

    print(model_path)


# python torch_dnn.py train_submit_cnn train_ht_cnn
def train_ht_cnn(name, alpha100, g100, optimizer, net_type, fc_init, lr=0.001, activation="tanh",                                             # network structure
                 loss_func_name="cross_entropy", bs=128, weight_decay=0, momentum=0, num_workers=1, epochs=500,                             # training setting (weight_decay=0.0001)
                 save_epoch=500, weight_save=0, stablefit_save=0):                                                                           # save data options

    #global model, train_dl, valid_dl

    """
    For training CNNs and saving accuracies and losses.
    """
    
    from time import time; t0 = time()
    from NetPortal.models import ModelFactory

    alpha100, g100 = int(alpha100), int(g100)
    alpha, g = alpha100/100., g100/100.

    bs, lr = int(bs), float(lr)
    lr_ls = [(0,lr)]
    epochs = int(epochs)
    num_workers = int(num_workers)

    if 'alexnet' in net_type and 'cifar' in name:
        train_ds, valid_ds = set_data(name,False,cifar_upsize=True)
        print("cifar upsized")
    else:
        train_ds, valid_ds = set_data(name,False)    
    N, C = len(train_ds[0][0]), len(train_ds.classes)

    # generate random id and date
    import uuid
    from datetime import datetime
    model_id = str(uuid.uuid1())
    train_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # create path
    model_path = join(root_data, "trained_cnns", net_type + "_new", f"{net_type}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}")
    if not os.path.isdir(model_path): os.makedirs(f'{model_path}')

    # log marker
    np.savetxt(f"{model_path}/{fc_init}", np.array([0]))

    # move entire dataset to GPU
    """
    train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(dev),
                             torch.tensor(train_ds.targets).to(dev))
    valid_ds = TensorDataset(torch.stack([e[0] for e in valid_ds]).to(dev),
                             torch.tensor(valid_ds.targets).to(dev))    
    """

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)#, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    steps = len(train_dl)
    print("Data loaded.")

    if net_type != 'resnext':
        if net_type == "alexnet":
            kwargs = {"alpha" :alpha, "g": g, "num_classes": C, "architecture": net_type, "activation": activation, "dataset": name, "fc_init": fc_init}
        elif "van" in net_type:
            kwargs = {"alpha" :alpha, "g": g , "fc_init": fc_init, "num_classes": C, "architecture": net_type, "dataset": name}
        else:
            kwargs = {"alpha" :alpha, "g": g, "num_classes": C, "architecture": net_type, "activation": activation, "dataset": name}            
        model = ModelFactory(**kwargs)
    #else:        
    #    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    # printed info
    print(model)
    print(f"Training method: {optimizer}, lr={lr}, batch_size={bs}, epochs={epochs}" + "\n")
    print(rf"Initialization: alpha={alpha}, g={g}" + "\n")
    model.to(dev)

    print("Saving information into local model_path before training.")
    # save information to model_path locally
    log_model(model_path, model_path, file_name="net_log", local_log=False, net_type=net_type, model_id=model_id, train_date=train_date, name=name, activation=activation, 
              alpha100=alpha100, g100=g100, loss_func_name=loss_func_name, optimizer=optimizer, bs=bs, lr=lr, lr_ls=lr_ls, weight_decay=weight_decay,
              num_workers=num_workers, epochs=epochs, steps=steps)
    
    opt_kwargs = {"lr": lr, "momentum": momentum} if optimizer == "sgd" else {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
    opt = get_optimizer(optimizer, model, **opt_kwargs)
    print(opt)

    # initial state of network
    print(f'Epoch 0:', end=' ')
    loss_func = F.__dict__[loss_func_name]

    epoch_path = f"{model_path}/epoch_0"
    epoch0_data = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, save=False)
    print(epoch0_data)
    #if not os.path.isdir(f'{model_path}/epoch_0'): os.makedirs(f'{model_path}/epoch_0')
    acc_loss = pd.DataFrame(columns=['train loss', 'train acc', 'test loss', 'test acc'])
    acc_loss.loc[0,:] = epoch0_data[:-1]
    
    # training
    for epoch in tqdm(range(epochs)):
        #print(f'Epoch {(1+epoch)*(1+num_workers)}:', end=' ')
        print(f'Epoch {(1+epoch)*(num_workers)}:', end=' ')
        print(f'Epoch {(1+epoch)}:', end=' ')        

        # record selected accuracies
        if (epoch+1) % save_epoch == 0 or epoch == epochs - 1 or epoch == 9:
            # train_loss, train_acc, val_loss, val_acc, epoch_time
            epoch_path = f"{model_path}/epoch_{epoch + 1}"
            if not os.path.isdir(epoch_path): os.makedirs(epoch_path)
            final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers, save_type='torch',
                                    epoch_path=epoch_path,
                                    weight_save=weight_save, stablefit_save=stablefit_save)

            print(final_acc)
        else:
            final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers, 
                                    epoch_path = epoch_path,
                                    weight_save=0, stablefit_save=stablefit_save)
            print(final_acc)

        acc_loss.loc[epoch + 1,:] = final_acc[:-1]

    if not isinstance(lr_ls, list):     # for learning rate scheme
        lr_ls = None

    # at least store accuracy at the end of each epoch
    acc_loss.to_csv(f"{model_path}/acc_loss", index=False)

    total_time = time() - t0
    # log dataframe
    log_path = root_data
    log_model(log_path, model_path, file_name="net_log", net_type=net_type, model_id=model_id, train_date=train_date, name=name, activation=activation, alpha100=alpha100, g100=g100,
              loss_func_name=loss_func_name, optimizer=optimizer, bs=bs, lr=lr, lr_ls=lr_ls, weight_decay=weight_decay, num_workers=num_workers, 
              epochs=epochs, steps=steps, final_acc=final_acc, total_time=total_time)

    print(model_path)

# ---------------------------------------

# wrapper for training MLPs on MNIST dataset
def mnist_train_ht_dnn(alpha100_ls, g100,
                       optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                       epochs=25, save_epoch=50, net_type="fc", activation='tanh', hidden_structure=2, with_bias=False,                                    
                       loss_func_name="cross_entropy", momentum=0, weight_decay=0, num_workers=1,                      
                       weight_save=1, stablefit_save=0, eigvals_save=0,                                       
                       with_ed=False, with_pc=False, with_ipr=False):                                                           

    alpha100_ls = literal_eval(alpha100_ls) if not isinstance(alpha100_ls,list) else alpha100_ls
    for alpha100 in alpha100_ls:
        train_ht_dnn("mnist", 0, 0, 0, 0, 0, 0,
                     alpha100, g100, optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                     epochs, save_epoch, net_type, activation, hidden_structure, with_bias,                                   
                     loss_func_name, momentum, weight_decay, num_workers,                      
                     weight_save, stablefit_save, eigvals_save,                                       
                     with_ed, with_pc, with_ipr)


# ---------------------------------------

# wrapper for training MLPs on Gaussain generated data
def gaussian_train_ht_dnn(seed_ls, name, Y_classes, X_clusters, num_train, num_test,
                          alpha100, g100, optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                          epochs=25, save_epoch=50, net_type="fc", activation='tanh', hidden_structure=2, with_bias=False,                                    
                          loss_func_name="cross_entropy", momentum=0, weight_decay=0, num_workers=1,                      
                          weight_save=1, stablefit_save=0, eigvals_save=0,                                       
                          with_ed=False, with_pc=False, with_ipr=False):                                                           

    # we make cluster_seed, assignment_and_noise_seed the same for tractability
    seed_ls = literal_eval(seed_ls) if not isinstance(seed_ls,list) else seed_ls
    for seed in seed_ls:
        train_ht_dnn(name, Y_classes, X_clusters, seed, seed, num_train, num_test,
                     alpha100, g100, optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                     epochs, save_epoch, net_type, activation, hidden_structure, with_bias,                                   
                     loss_func_name, momentum, weight_decay, num_workers,                      
                     weight_save, stablefit_save, eigvals_save,                                       
                     with_ed, with_pc, with_ipr)

        print(f"fc{depth} init at ({alpha100}, {g100}) trained with {optimizer} for {epochs} epochs done!")


# --------------------------------------- Submit functions ---------------------------------------

# train networks in array jobs

# batch training mnist with MLPs
def batch_mnist_submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    from path_names import singularity_path, bind_path

    dataset_name = "mnist"
    net_type = "fc"
    alpha100_ls = list(range(100,201,10))
    alpha100_ls = str(alpha100_ls).replace(" ", "")
    g100_ls = list(range(25,301,25))
    #alpha100_ls = '[100,200]'
    #g100_ls = [100,200]
    optimizer = "rmsprop"
    bs_ls = [1024]  
    lr_ls = [0.001]
    depth = 10
    epochs = 650
    save_epoch = 50
    #depth=2
    #epochs=2    
    #save_epoch=1
    init_path, init_epoch = None, None
    root_folder = join("trained_mlps", f"fc{depth}_{optimizer}_{dataset_name}")
    root_path = join(root_data, root_folder)    

    # raw submittions    
    pbs_array_data = [(alpha100_ls, g100, 
                       optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                       epochs, save_epoch
                      )
                      for g100 in g100_ls
                      for bs in bs_ls
                      for lr in lr_ls
                      ]   

    print(len(pbs_array_data))                 

    ncpus, ngpus = 1, 1
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)   

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, root_folder),
             P=project_ls[pidx],
             source="virt-test-qu/bin/activate",
             ngpus=ngpus,
             ncpus=ncpus,
             #walltime='0:59:59',
             walltime='23:59:59',
             mem='10GB')      


# for figure 1
def fig1_submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    from path_names import singularity_path, bind_path

    dataset_name = "mnist"
    net_type = "fc"
    alpha100_lss = ['[100]', '[200]']
    g100_ls = [100]
    optimizer = "sgd"
    bs_ls = [1024]  
    #lr_ls = [0.001, 0.003]  # too slow
    lr_ls = [0.01]
    depth = 3
    epochs = 200
    save_epoch = 101
    init_path, init_epoch = None, None
    root_folder = join("trained_mlps", f"fc{depth}_{optimizer}_fig1")
    root_path = join(root_data, root_folder)    

    # raw submittions    
    pbs_array_data = [(alpha100_ls, g100, 
                       optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                       epochs, save_epoch
                      )
                      for alpha100_ls in alpha100_lss
                      for g100 in g100_ls
                      for bs in bs_ls
                      for lr in lr_ls
                      ]   

    print(len(pbs_array_data))         

    ncpus, ngpus = 1, 1
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)      

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, root_folder),
             P=project_ls[pidx],
             #source="virt-test-qu/bin/activate",
             ncpus=ncpus,
             ngpus=ngpus,
             #ncpus=1,
             walltime='0:09:59',
             #walltime='23:59:59',
             #mem='10GB'
             #mem='2GB'  # fc3 cpu
             mem='4GB'  # fc3 gpu
             )                  


# for training mnist with MLPs
def mnist_submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    from path_names import singularity_path, bind_path

    dummy_ls = [0] * 6
    Y_classes, X_clusters, cluster_seed, assignment_and_noise_seed, num_train, num_test = dummy_ls

    dataset_name = "mnist"
    net_type = "fc"
    #alpha100_ls = [120, 200]
    #g100_ls = [25,100,300]
    alpha100_ls = list(range(100,201,10))
    g100_ls = list(range(25,301,25))
    #optimizer_ls = ["sgd"]
    optimizer = "adam"
    bs_ls = [1024]  
    lr_ls = [0.001]
    depth = 10
    epochs = 650
    save_epoch = 50
    init_path, init_epoch = None, None
    root_folder = join("trained_mlps", f"fc{depth}_{optimizer}_{dataset_name}")
    root_path = join("/project/PDLAI/project2_data", root_folder)    

    # raw submittions    
    pbs_array_data = [(dataset_name, 
                       Y_classes, X_clusters, cluster_seed, assignment_and_noise_seed, num_train, num_test,
                       alpha100, g100, optimizer,
                       bs, init_path, init_epoch, root_path, depth, lr,
                       epochs, save_epoch
                      )
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for bs in bs_ls
                      for lr in lr_ls
                      ]   

    print(len(pbs_array_data))         

    # resubmissions
    """
    alpha_g_pair = []
    for alpha100 in alpha100_ls:
        for g100 in g100_ls:
            if (alpha100,g100) not in [(100,25), (100,300), (200,25), (200,100), (200,300)]:
                alpha_g_pair.append( (alpha100,g100) )
    
    pbs_array_data = [(name, pair[0], pair[1], optimizer, bs, init_path, init_epoch, root_path, depth, lr, epochs)
                      for name in dataset_ls
                      for pair in alpha_g_pair
                      for optimizer in optimizer_ls
                      for bs in bs_ls
                      for lr in lr_ls
                      ]
    """

    """
    pbs_array_true = []
    pbs_array_nosubmit = []
    net_ls = [net[0] for net in os.walk(path_names.fc_path)]
    for nidx in range(1,len(net_ls)):
        ag_str = net_ls[nidx].split("/")[-1].split("_")
        pbs_array_nosubmit.append( (int(ag_str[1]), int(ag_str[2])) )

    for sub in pbs_array_data:
        if (sub[1],sub[2]) not in pbs_array_nosubmit:
            pbs_array_true.append(sub)

        #print((sub[1],sub[2]))
        #break
    print(len(pbs_array_true))
    """
    
    """
    ncpus, ngpus = 1, 0
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)   

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, root_folder),
             P=project_ls[pidx],
             #ngpus=1,
             ngpus=ngpus,
             ncpus=ncpus,
             #walltime='0:59:59',
             walltime='23:59:59',
             mem='6GB') 
    """

# for training gaussian datasets
def gaussian_submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    from path_names import singularity_path, bind_path

    #dataset_ls = ["mnist"]
    dataset_ls = ["gaussian"]
    #Y_classes = 10
    Y_classes = 2
    #X_clusters_ls = [60,120]
    #X_clusters_ls = list(range(20,121,20))
    X_clusters_ls = [2,3]
    #X_clusters_ls = list(range(4,11)) + list(range(20,121,20))
    #num_train_ls = [5,10,20,30,40,50,100,200,300,400,500,750,1000,1250,1500,2000]   # 16
    #num_train_ls = [5,10,20,30,40,50,100,200,300,400]
    #num_train_ls = [10,400]
    num_train_ls = [2000]   # binary classification
    num_test = 1000

    alpha100_ls = [120, 200]
    g100_ls = [25,100,300]
    #alpha100_ls = [None]
    #g100_ls = [None]
    #alpha100_ls = list(range(100,201,10))
    #g100_ls = list(range(25,301,25))
    optimizer_ls = ["sgd"]
    bs_ls = [1024]  
    lr_ls = [0.001]
    depth = 10
    epochs = 650
    save_epoch = 50
    init_path, init_epoch = None, None
    #root_path = join(root_data, "trained_mlps/bs_analysis")
    #root_path = join("/project/phys_DL/project2_data", "trained_nets")
    #root_path = join("/project/PDLAI/project2_data", "trained_mlps/debug")
    #root_folder = "trained_mlps_gaussian"
    root_folder = "gaussian_binary_classification"
    root_path = join("/project/PDLAI/project2_data", f"{root_folder}/fc{depth}")
    
    total_ensembles = 1
    seed_pairs = []
    for i in range(total_ensembles):
        seed_pairs.append( (0,i) )

    # submissions
    pbs_array_data = [(name, Y_classes, X_clusters, seed_pair[0], seed_pair[1],
                       num_train, num_test,
                       alpha100, g100, optimizer, bs, 
                       init_path, init_epoch, root_path, depth, lr, epochs, save_epoch)
                      for name in dataset_ls
                      for X_clusters in X_clusters_ls
                      for seed_pair in seed_pairs
                      for num_train in num_train_ls
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for optimizer in optimizer_ls
                      for bs in bs_ls
                      for lr in lr_ls
                      ]

    # resubmissions
    """
    alpha_g_pair = []
    for alpha100 in alpha100_ls:
        for g100 in g100_ls:
            if (alpha100,g100) not in [(100,25), (100,300), (200,25), (200,100), (200,300)]:
                alpha_g_pair.append( (alpha100,g100) )
    
    pbs_array_data = [(name, pair[0], pair[1], optimizer, bs, init_path, init_epoch, root_path, depth, lr, epochs)
                      for name in dataset_ls
                      for pair in alpha_g_pair
                      for optimizer in optimizer_ls
                      for bs in bs_ls
                      for lr in lr_ls
                      ]
    """

    """
    pbs_array_true = []
    pbs_array_nosubmit = []
    net_ls = [net[0] for net in os.walk(path_names.fc_path)]
    for nidx in range(1,len(net_ls)):
        ag_str = net_ls[nidx].split("/")[-1].split("_")
        pbs_array_nosubmit.append( (int(ag_str[1]), int(ag_str[2])) )

    for sub in pbs_array_data:
        if (sub[1],sub[2]) not in pbs_array_nosubmit:
            pbs_array_true.append(sub)

        #print((sub[1],sub[2]))
        #break
    print(len(pbs_array_true))
    """
    
    ncpus, ngpus = 1, 0
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)   

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, root_folder),
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             #walltime='0:59:59',
             walltime='23:59:59',
             mem='4GB') 

    """
    qsub(f'python {sys.argv[0]} {" ".join(args)}',    
         pbs_array_true, 
         path='/project/dyson/dyson_dl',
         #P='dnn_maths',
         P='phys_DL',
         #P='PDLAI',
         #P='ddl',
         #P='dyson',
         ngpus=1,
         ncpus=1,
         walltime='1:59:59',
         #walltime='23:59:59',
         mem='6GB') 
    """

# ---------------------------------------

# train networks in array jobs
def train_ensemble_submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    from path_names import singularity_path, bind_path

    # dataset     
    name = "gaussian"
    Y_classes = 10

    # ---------- X_clusters as a control ----------
    #X_clusters_ls = [60,120]
    #X_clusters_ls = list(range(20,121,20))
    #X_clusters_ls = list(range(140,241,20))

    # ---------- num_train as a control ----------
    #num_train_ls = [5,10,20,30,40,50,100,200,300,400,500,750,1000,1250,1500,2000]   # 16
    #num_train_ls = [5,10,20,30,40,50,100,200,300,400]
    #num_train_ls = [500,600,700,800]
    #num_train_ls = [10,400]
    #num_test = 1000

    #alpha100_ls = [100, 200]
    #g100_ls = [25,100,300]

    # ---------- phase transition ----------
    alpha100_ls = list(range(100,201,10))
    g100_ls = list(range(25,301,25))

    X_clusters_ls = [60, 120]
    num_train_ls = [60000]
    num_test = 10000
    # training setting
    optimizer_ls = ["sgd"]
    bs, lr = 1024, 0.001
    depth = 10
    epochs = 650
    init_path, init_epoch = None, None
    root_folder = "trained_mlps_gaussian_phase"
    root_path = join("/project/PDLAI/project2_data", f"{root_folder}/fc{depth}")   
    #seed_ls = list(range(10))
    seed_ls = list(range(10,20))
    # convert seed_ls to string list
    seed_ls = str(seed_ls).replace(" ", "")

    # submissions
    pbs_array_data = [(seed_ls, name, Y_classes, X_clusters, num_train, num_test,
                       alpha100, g100, optimizer, bs, 
                       init_path, init_epoch, root_path, depth, lr, epochs, epochs)
                      for X_clusters in X_clusters_ls
                      for num_train in num_train_ls
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for optimizer in optimizer_ls
                      ]

    #resubmissions

    ncpus, ngpus = 1, 0
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)       
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, root_folder),
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             #walltime='0:59:59',
             walltime='23:59:59',
             mem='4GB') 



# version corresponding train_dnn
def train_submit_cnn(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    from path_names import singularity_path, bind_path

    #net_type_ls = ["alexnet", "resnet14"]
    alpha100_ls = list(range(100,201,10))
    g100_ls = list(range(25, 301, 25))
    #g100_ls = list(range(10, 201, 10))

    #alpha100_ls = [100, 200]
    #g100_ls = [25,300]

    #alpha100_ls = [100] 
    #g100_ls = [25,100,300]

    #net_type_ls = ["convnet_old"]
    #net_type_ls = ["van20nobias"]
    #net_type_ls = ["van50","van100","van200"]
    #net_type_ls = ["van100"]
    #net_type_ls = ["van100nobias"]
    net_type_ls = ["resnet14_ht"]
    #net_type_ls = ["alexnetold"]
    #net_type_ls = ["alexnet"]
    #net_type_ls = ["resnet34_HT", "resnet50_HT"]
    #fc_init = "fc_ht"
    #fc_init = "fc_orthogonal"
    fc_init = "fc_default"
    #dataset_ls = ["mnist"]
    dataset_ls = ["cifar100"]
    optimizer_ls = ["sgd"]

    pbs_array_data = [(name, alpha100, g100, optimizer, net_type, fc_init)
                      for name in dataset_ls
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for optimizer in optimizer_ls
                      for net_type in net_type_ls
                      ]

    """
    pbs_array_no = [(100,25), (100,100), (100,300), (200,25), (200,100), (200,300)]
    for sub in pbs_array_data:
        if (sub[1],sub[2]) not in pbs_array_no:
            pbs_array_true.append(sub)

        #print((sub[1],sub[2]))
        #break
    """

    ncpus, ngpus = 1, 1
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)   

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data,"trained_cnns"),
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='8GB')
     

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])


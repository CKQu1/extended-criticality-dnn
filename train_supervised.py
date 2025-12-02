import json
import numpy as np
import os
import pandas as pd  
import torch, torchvision
import torch.nn.functional as F
from ast import literal_eval
from datetime import datetime
from os.path import join
from time import time
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy.stats import levy_stable

from constants import DROOT, log_model, read_log, njoin
from utils_dnn import save_weights, get_weights, compute_dq, compute_IPR, layer_ipr, effective_dimension, store_model

dev = torch.device(f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu")
print(dev)

def transform_cifar10(image):   # flattening cifar10
    return (torch.Tensor(image.getdata()).T.reshape(-1)/255)*2 - 1

def set_data(name, rshape: bool, **kwargs):
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
    save_epoch = (kwargs.get('weight_save') == 1)
    save_step = (kwargs.get('weight_save') == 2)
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

    print("EVAL STARTS!")  # delete

    # pure evaluation of train and test acc/loss
    losses, nums, corrects = zip(
        *[loss_batch(model, loss_func, xb, yb) for xb, yb in tqdm(train_dl)]
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

    print("EVAL ENDS!")  # delete

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
from NetPortal.models import ModelFactory

# Trains MLPs and saves weights along the steps of training, can add to save gradients, etc.
def train_ht_dnn(name, Y_classes, X_clusters, cluster_seed, assignment_and_noise_seed, num_train, num_test,
                 train_seed, alpha100, g100, optimizer, bs, init_path, init_epoch, root_path, depth, lr,
                 epochs=25, save_epoch=50, net_type="fc", activation='tanh', hidden_structure=2, 
                 with_bias=False, is_weight_share=False,       # network structure
                 loss_func_name="cross_entropy", momentum=0, weight_decay=0, num_workers=1,  # training setting
                 weight_save=1, stablefit_save=0, eigvals_save=0,  # save data options
                 with_ed=False, with_pc=False, with_ipr=False):   # save extra data options

    t0 = time()
    init_path = None if init_path.lower() == "none" else init_path
    init_epoch = None if init_epoch.lower() == "none" else init_epoch

    train_seed = int(train_seed)
    if isinstance(alpha100,str)==True or isinstance(g100,str)==True:
        if alpha100 != "None" or g100 != "None": 
            alpha100, g100 = int(alpha100), int(g100)
            alpha, g = int(alpha100)/100., int(g100)/100.
        else:
            alpha100, g100 = None, None
            alpha, g = None, None
    else:
        alpha100, g100 = int(alpha100), int(g100)
        alpha, g = int(alpha100)/100., int(g100)/100.
    bs, lr = int(bs), float(lr)
    epochs = int(epochs)
    save_epoch = int(save_epoch)
    depth = int(depth)
    num_workers = int(num_workers)
    with_bias = literal_eval(with_bias) if isinstance(with_bias, str) else with_bias
    is_weight_share = literal_eval(is_weight_share) if isinstance(is_weight_share, str) else is_weight_share

    if name.lower() != "gaussian":
        train_ds, valid_ds = set_data(name,True)
        Y_classes, X_clusters= None, None
        cluster_seed, assignment_and_noise_seed = None, None
    else:
        X_dim = 784
        noise_sigma = 0.2
        Y_classes, X_clusters = int(Y_classes), int(X_clusters)
        num_train, num_test = int(num_train), int(num_test)
        Y_classes, X_clusters = int(Y_classes), int(X_clusters)
        cluster_seed, assignment_and_noise_seed = int(cluster_seed), int(assignment_and_noise_seed)
        gaussian_data_kwargs = {"num_train": num_train, "num_test": num_test,
                                "X_dim": X_dim,
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
    else:
        C = len(np.unique(cluster_class_label))

    torch.manual_seed(train_seed) 

    model_id = str(train_seed)
    train_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")  

    # create path
    model_path = njoin(root_path, f'{net_type}{depth}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}')
    if not os.path.isdir(model_path): os.makedirs(f'{model_path}')
    # log marker
    if name.lower() == "gaussian":
        with open(njoin(model_path,"gaussian_data_kwargs.json"), "w") as ofile: 
            json.dump(gaussian_data_kwargs, ofile)

    save_config = {'lr': lr, 'momentum': momentum, 'bs':bs, 'epochs': epochs, 'save_epoch': save_epoch, 
                   'weight_save': weight_save, 'stablefit_save': stablefit_save, 'eigvals_save': eigvals_save,
                   'with_ed': with_ed, 'with_ipr': with_ipr, 'with_pc': with_pc}
    with open(njoin(model_path,"save_config.json"), "w") as ofile: 
        json.dump(save_config, ofile)    

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
              "is_weight_share": is_weight_share,
              "architecture": net_type}

    with open(njoin(model_path,"model_config.json"), "w") as ofile: 
        json.dump(kwargs, ofile)                 
    
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
              alpha100=alpha100, g100=g100, activation=activation, loss_func_name=loss_func_name, optimizer=optimizer, hidden_structure=hidden_structure, depth=depth, 
              with_bias=with_bias, is_weight_share=is_weight_share,
              bs=bs, lr=lr, 
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
    with open(njoin(model_path,"opt_kwargs.json"), "w") as ofile: 
        json.dump(opt_kwargs, ofile)        
    opt = get_optimizer(optimizer, model, **opt_kwargs)

    # initial state of network
    print(f'Epoch 0:', end=' ')
    loss_func = F.__dict__[loss_func_name]
    epoch_path = njoin(model_path, 'epoch_0')
    if not os.path.isdir(epoch_path): os.makedirs(epoch_path)
    
    if stablefit_save > 0:
        epoch0_data, stablefit_params = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, 
                                                    weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=eigvals_save)
    else:
        epoch0_data = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, 
                                                    weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=eigvals_save)

    print(epoch0_data)
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
    log_path = DROOT
    log_model(log_path, model_path, file_name="net_log", net_type=net_type, model_dims=model.dims, model_id=model_id, train_date=train_date, name=name, alpha100=alpha100, g100=g100, 
              activation=activation, loss_func_name=loss_func_name, optimizer=optimizer, hidden_structure=hidden_structure, depth=depth, 
              with_bias=with_bias, is_weight_share=is_weight_share,
              bs=bs, lr=lr, num_workers=num_workers, 
              epochs=epochs, steps=steps, final_acc=final_acc, total_time=total_time,
              Y_classes=Y_classes, X_clusters=X_clusters)

    if with_ed:
        np.savetxt(f"{model_path}/ed_train",ed_train)
        np.savetxt(f"{model_path}/ed_test",ed_test)

    print(model_path)


# For training CNNs and saving accuracies and losses.
def train_ht_cnn(root_path, name, alpha100, g100, optimizer, net_type, fc_init, lr=0.001, activation="tanh",                                             # network structure
                 loss_func_name="cross_entropy", bs=256, weight_decay=0, momentum=0, num_workers=1, epochs=5,                             # training setting (weight_decay=0.0001)
                 save_epoch=5, weight_save=0, stablefit_save=0):                                                                           # save data options

    t0 = time()
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
        #train_ds, valid_ds = set_data(name,False,cifar_upsize=True)
        #print("cifar upsized")        
    N, C = len(train_ds[0][0]), len(train_ds.classes)

    # generate random id and date
    import uuid
    from datetime import datetime
    model_id = str(uuid.uuid1())
    train_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # create path
    #model_path = join(DROOT, "trained_cnns", net_type + "_new", f"{net_type}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}")
    model_path = join(root_path, net_type, f"{net_type}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}")
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
        elif "van" in net_type.lower():
            depth = int(net_type[3:])
            assert isinstance(depth, int)
            kwargs = {"architecture": 'van', "depth": depth, "dataset": name,
                      "alpha" :alpha, "g": g , "fc_init": fc_init, 
                      "num_classes": C}
        else:
            kwargs = {"alpha" :alpha, "g": g, "num_classes": C, "architecture": net_type, "activation": activation, "dataset": name}            
        model = ModelFactory(**kwargs)
    #else:        
    #    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

    print(model)
    print(f"Training method: {optimizer}, lr={lr}, batch_size={bs}, epochs={epochs}" + "\n")
    print(rf"Initialization: alpha={alpha}, g={g}" + "\n")
    model.to(dev)

    print("Saving information into local model_path before training.")
    # save information to model_path locally
    log_model(model_path, model_path, file_name="net_log", local_log=False, net_type=net_type, model_id=model_id, train_date=train_date, 
              name=name, activation=activation, 
              alpha100=alpha100, g100=g100, 
              loss_func_name=loss_func_name, optimizer=optimizer, bs=bs, lr=lr, lr_ls=lr_ls, weight_decay=weight_decay,
              num_workers=num_workers, epochs=epochs, steps=steps)
    
    opt_kwargs = {"lr": lr, "momentum": momentum} if optimizer == "sgd" else {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
    opt = get_optimizer(optimizer, model, **opt_kwargs)
    print(opt)

    # initial state of network
    print(f'Epoch 0:', end=' ')
    loss_func = F.__dict__[loss_func_name]

    epoch_path = f"{model_path}/epoch_0"
    # if stablefit_save > 0:
    #     epoch0_data, stablefit_params = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, save_type='torch',
    #                                                 weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=0)
    # else:
    #     epoch0_data = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, save_type='torch',
    #                               weight_save=weight_save, stablefit_save=stablefit_save, eigvals_save=0)

    acc_loss = pd.DataFrame(columns=['train loss', 'train acc', 'test loss', 'test acc'])
    
    # training
    print("Starting training! \n")
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

        else:
            final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers, 
                                    epoch_path = epoch_path,
                                    weight_save=0, stablefit_save=stablefit_save)
        print(final_acc)        
        acc_loss.loc[epoch + 1,:] = final_acc[:-1]

    if not isinstance(lr_ls, list):     # for learning rate scheme
        lr_ls = None

    acc_loss.to_csv(f"{model_path}/acc_loss", index=False)

    total_time = time() - t0
    # log dataframe
    log_path = DROOT
    log_model(log_path, model_path, file_name="net_log", net_type=net_type, model_id=model_id, train_date=train_date, name=name, activation=activation, alpha100=alpha100, g100=g100,
              loss_func_name=loss_func_name, optimizer=optimizer, bs=bs, lr=lr, lr_ls=lr_ls, weight_decay=weight_decay, num_workers=num_workers, 
              epochs=epochs, steps=steps, final_acc=final_acc, total_time=total_time)

    print(f'All data saved in {model_path}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
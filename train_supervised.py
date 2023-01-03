# original version: https://gist.github.com/asemptote/3c9f901f1346dffb29d21742cb83c933
import numpy as np
import os
import pandas as pd  
import path_names
import torch
from time import time
from torch.autograd import Variable
from tqdm import tqdm
from scipy.stats import levy_stable

dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")
print(dev)

# Record relevants attributes for trained neural networks -------------------------------------------------------

def log_model(log_path, model_path, **kwargs):    
    fi = f"{log_path}/net_log.csv"
    df = pd.DataFrame(columns = kwargs)
    df.loc[0,:] = list(kwargs.values())
    df.to_csv(f"{model_path}/log", index=False)
    if os.path.isfile(fi):
        df_og = pd.read_csv(fi)
        # outer join
        df = pd.concat([df_og,df], axis=0, ignore_index=True)
    else:
        if not os.path.isdir(f"{log_path}"): os.makedirs(log_path)
    df.to_csv(fi, index=False)
    print('Log saved!')

def read_log():    
    fi = f"{path_names.log_path}/net_log.csv"
    if os.path.isfile(fi):
        df_og = pd.read_csv(fi)
        print(df_og)
    else:
        raise ValueError("Network logbook has not been created yet, please train a network.")

# Dataset management -------------------------------------------------------------------------------------------

# save current state of model
def save_weights(model, path):
    weights_all = np.array([]) # save all weights vectorized 
    for p in model.parameters():
        weights_all = np.concatenate((weights_all, p.flatten().data.numpy()), axis=0)

    #np.savetxt(f"{path}", weights_all, fmt='%f')
    # saveas .npy instead
    np.save(f"{path}", weights_all)

# get weights flatten
def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

import torchvision

def transform_cifar10(image):   # flattening cifar10
    return (torch.Tensor(image.getdata()).T.reshape(-1)/255)*2 - 1

def set_data(name, rshape, **kwargs):
    import torchvision.transforms as transforms

    if rshape:
        transform=transform_cifar10
        if name == 'mnist':
            train_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform)
            valid_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform, train=False)
        elif name == 'cifar10':
            train_ds = torchvision.datasets.CIFAR10('cifar10', download=True, transform=transform)
            valid_ds = torchvision.datasets.CIFAR10('cifar10', download=True, transform=transform, train=False)
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

        if name == 'mnist':
            """
            train_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform_train)
            valid_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform_test, train=False)
            """
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            train_ds = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize,]))
            valid_ds = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize,]))

        elif name == 'cifar10':
            if "cifar_upsize" in kwargs and kwargs.get["cifar_upsize"] == True:     # for AlexNet (torch version)
                upsize_cifar10 = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                train_ds = torchvision.datasets.CIFAR10('cifar10', download=True, transform=upsize_cifar10)
                valid_ds = torchvision.datasets.CIFAR10('cifar10', download=True, transform=upsize_cifar10, train=False)
            else:
                train_ds = torchvision.datasets.CIFAR10('cifar10', download=True, transform=transform_train)
                valid_ds = torchvision.datasets.CIFAR10('cifar10', download=True, transform=transform_test, train=False)

        elif name == 'cifar10circ':
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1994, 0.2010])
            
            train_ds = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=True,
                                                    transform=transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]))
            
            valid_ds = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=True,
                                                    transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]))      

        else:
            raise NameError("name is not defined in function set_data")
    
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

# Extra quantities

def IPR(vec,q):
    vec = torch.abs(vec.detach().cpu())
    ipr = torch.sum(vec**(2*q))/torch.sum(vec**2)**q
    return ipr

def compute_dq(vec,q):
    return  (torch.log(IPR(vec,q))/np.log(len(vec)))/(1 - q)

def layer_ipr(model,hidden_layer,lq_ls):
    arr = np.zeros([len(model.sequential),len(q_ls)])

    for lidx in range(len(model.sequential)):
        """
        hidden_layer = model.sequential[lidx](torch.Tensor(hidden_layer).to(dev))
        hidden_layer = hidden_layer.detach().numpy()
        hidden_layer_mean = np.mean(hidden_layer,axis=0)    # center the hidden representations
        C = (1/hidden_layer.shape[0])*np.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*np.matmul(hidden_layer_mean.T, hidden_layer_mean)
        arr[lidx] = np.trace(C)**2/np.trace(np.matmul(C,C))
        """
        with torch.no_grad():
            hidden_layer = model.sequential[lidx](hidden_layer)
        hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
        C = (1/hidden_layer.shape[0])*torch.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*torch.matmul(hidden_layer_mean.T, hidden_layer_mean)
        eigvals, _ = torch.eig(C)
        arr[lidx,:] = [ IPR(eigvals[:,0],q) for q in q_ls ]
                          
    return arr

def effective_dimension(model,hidden_layer,with_pc:bool,q_ls):    # hidden_layer is image    
    ed_arr = np.zeros([len(model.sequential)])
    C_dims = np.zeros([len(model.sequential)])   # record dimension of correlation matrix
    with torch.no_grad():
        for lidx in range(len(model.sequential)):
            """
            hidden_layer = model.sequential[lidx](torch.Tensor(hidden_layer).to(dev))
            hidden_layer = hidden_layer.detach().numpy()
            hidden_layer_mean = np.mean(hidden_layer,axis=0)    # center the hidden representations
            C = (1/hidden_layer.shape[0])*np.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*np.matmul(hidden_layer_mean.T, hidden_layer_mean)
            ed_arr[lidx] = np.trace(C)**2/np.trace(np.matmul(C,C))
            """
            hidden_layer = model.sequential[lidx](hidden_layer)
            hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
            C = (1/hidden_layer.shape[0])*torch.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*torch.matmul(hidden_layer_mean.T, hidden_layer_mean)
            ed_arr[lidx] = torch.trace(C)**2/torch.trace(torch.matmul(C,C))
            C_dims[lidx] = C.shape[0]
            # get PCs/eigenvectors for C (correlation matrix)
            if with_pc:
                _, eigvecs = torch.eig(C,eigenvectors=with_pc)           
                pc_dqs = np.zeros([eigvecs.shape[1],len(q_ls)])

                for eidx in range(C.shape[0]):
                    pc_dqs[eidx,:] = [compute_dq(eigvecs[:,eidx], q) for q in q_ls]
                if lidx == 0:
                    pc_dqss = pc_dqs
                else:
                    pc_dqss = np.vstack([pc_dqss, pc_dqs])
    
    if not with_pc:
        pc_dqss = None                
                     
    return ed_arr, C_dims, pc_dqss

# Training methods ----------------------------------------------------------------------------------------

# loss
def loss_batch(model, loss_func, xb, yb, opt=None, stats=True):
    #xb = xb.to(dev)
    #yb = yb.to(dev)
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

# save weights
def store_model(model, path ,save_type):       
    if save_type == 'torch':
        torch.save(model, path)        # save as torch
    elif save_type == 'numpy':
        save_weights(model, path)      # save as .npy (flattened)
    else:
        raise TypeError("save_type does not exist!")     

# fit weight distribution for individually layers
pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
def stablefit_model(model):
    Ws = get_weights(model)
    wm_total = len(Ws)              # total number of weight matrices
    stable_params = np.empty([wm_total, 4])
    for widx in range(wm_total):
        stable_params[widx,:] = pconv(*levy_stable._fitstart(Ws[widx].flatten()))    
    return stable_params  

def train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=0, 
                save_type='torch', **kwargs):

    """
    Both `weight_save` and `stablefit_save` have 3 mods, 
    - 0: save nothing
    - 1: save end of epoch
    - 2 save all steps
    """

    start = time()
    model.train()
    save_epoch, save_step = kwargs.get('weight_save') == 1, kwargs.get('weight_save') == 2
    stablefit_epoch, stablefit_step = kwargs.get('stablefit_save') == 1, kwargs.get('stablefit_save') == 2
    params_step = np.zeros([len(train_dl),len(list(model.parameters())),4]) if stablefit_epoch else None
    for hidden_epoch in range(hidden_epochs):
        for batch_idx, (xb, yb) in enumerate(train_dl):
            loss_batch(model, loss_func, xb, yb, opt, stats=False)
            if save_step:    # needed if want to save all steps
                store_model(model, f"{kwargs.get('epoch_path')}/weights_{batch_idx}", save_type)
            if stablefit_step:
                params_step[batch_idx,:,:] = stablefit_model(model)
    if save_epoch:    # save only the last step of each epoch
        store_model(model, f"{kwargs.get('epoch_path')}", save_type)         
    if stablefit_epoch:
        params_epoch = stablefit_model(model)

    losses, nums, corrects = zip(
        *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
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
    if not (stablefit_epoch or stablefit_step):
        return (train_loss, train_acc, val_loss, val_acc, f'{time()-start:.2f}s')
    elif stablefit_epoch:
        return (train_loss, train_acc, val_loss, val_acc, f'{time()-start:.2f}s'), stablefit_epoch
    elif stablefit_step:
        return (train_loss, train_acc, val_loss, val_acc, f'{time()-start:.2f}s'), stablefit_step

from torch import optim
import torch.nn.functional as F

def train_ht_dnn(name, alpha100, g100, optimizer, bs, net_type = "fc", activation='tanh', hidden_structure=2, depth=10,  # network structure
                 loss_func_name="cross_entropy", lr=0.001, num_workers=1, epochs=650,                                    # training setting
                 save_epoch=50, weight_save=1, stablefit_save=0,                                                        # save data options
                 with_ed=True, with_pc=True, with_ipr=False):                                                            # save extra data options

    """
    Trains MLPs and saves weights along the steps of training, can add to save gradients, etc.
    """
    
    from time import time; t0 = time()
    from NetPortal.models import ModelFactory

    alpha100, g100 = int(alpha100), int(g100)
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    bs = int(bs)
    lr = float(lr)
    epochs = int(epochs)
    depth = int(depth)
    num_workers = int(num_workers)

    train_ds, valid_ds = set_data(name,True)
    N = len(train_ds[0][0])
    C = len(train_ds.classes)

    # generate random id and date
    import uuid
    from datetime import datetime
    model_id = str(uuid.uuid1())
    train_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")  

    # create path
    model_path = f"{path_names.fc_path}/{net_type}{depth}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}"
    if not os.path.isdir(model_path): os.makedirs(f'{model_path}')
    # log marker
    np.savetxt(f"{model_path}/epochs={epochs}_save_epoch={save_epoch}_weight_save={weight_save}_stablefit_save={stablefit_save}", np.array([0]))
    np.savetxt(f"{model_path}/with_ed={with_ed}_with_ipr={with_ipr}_with_pc={with_pc}", np.array([0]))

    # move entire dataset to GPU
    train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(dev),
                             torch.tensor(train_ds.targets).to(dev))
    valid_ds = TensorDataset(torch.stack([e[0] for e in valid_ds]).to(dev),
                             torch.tensor(valid_ds.targets).to(dev))

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)#, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    steps = len(train_dl)
    #print(f"Training data shape: {len(train_dl.dataset)}")

    # new method
    hidden_N = [N]
    if hidden_structure == 1:   # network setup 1 (power-law decreasing layer size)
        a = (C/N)**(1/depth)
        hidden_N = hidden_N + [int(N*a**l) for l in range(1, depth)]
    elif hidden_structure == 2:   # network setup 2 (square weight matrix)
        hidden_N = hidden_N + [N for l in range(1, depth)]
    else: 
        assert isinstance(hidden_structure,list), 'hidden_N must be a list!'
        assert len(hidden_structure) == depth - 1, 'hidden_N length and depth inconsistent!'
        hidden_N += hidden_structure

    hidden_N.append(C)
    kwargs = {"dims": hidden_N, "alpha": alpha, "g": g,
              "activation": activation, "architecture": net_type}
    model = ModelFactory(**kwargs)
    # info
    print(f"Total layers = {depth+1}", end=' ')
    print(f"Network hidden layer structure: type {hidden_structure}" + "\n")
    print(f"Training method: {optimizer}, lr={lr}, batch_size={bs}, epochs={epochs}" + "\n")
    model.to(dev)   # move to CPU/GPU

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
    
    if optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0)
    elif optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        print("THIS OPTIMIZER DOES NOT EXIST!")

    # initial state of network
    print(f'Epoch 0:', end=' ')
    loss_func = F.__dict__[loss_func_name]
    epoch_path = f"{model_path}/epoch_0"
    epoch0_data = train_epoch(model, loss_func, None, train_dl, valid_dl, epoch_path=epoch_path, weight_save=0, stablefit_save=0)
    print(epoch0_data)
    #if not os.path.isdir(f'{model_path}/epoch_0'): os.makedirs(f'{model_path}/epoch_0')
    acc_loss = pd.DataFrame(columns=['train loss', 'train acc', 'test loss', 'test acc'])
    acc_loss.loc[0,:] = epoch0_data[:-1]

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
        #if epoch == 0.1:
        if (epoch+1) % save_epoch == 0 or epoch == epochs - 1:
            # train_loss, train_acc, val_loss, val_acc, epoch_time
            epoch_path = f"{model_path}/epoch_{epoch + 1}"
            #if not os.path.isdir(epoch_path): os.makedirs(epoch_path)
            final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
                                    epoch_path=epoch_path,
                                    weight_save=weight_save, stablefit_save=stablefit_save)
            
            print(final_acc)
        else:
            final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
                                    epoch_path=epoch_path,
                                    weight_save=0, stablefit_save=stablefit_save)
            print(final_acc)

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

    # at least store accuracy at the end of each epoch
    acc_loss.to_csv(f"{model_path}/acc_loss", index=False)

    total_time = time() - t0
    # log dataframe
    log_path = path_names.log_path
    log_model(log_path, model_path, net_type=net_type, model_dims=model.dims, model_id=model_id, train_date=train_date, name=name, alpha100=alpha100, g100=g100, 
              activation=activation, loss_func_name=loss_func_name, optimizer=optimizer, hidden_structure=hidden_structure, depth=depth, bs=bs, lr=lr, num_workers=num_workers, 
              epochs=epochs, steps=steps, final_acc=final_acc, total_time=total_time)

    if with_ed:
        np.savetxt(f"{model_path}/ed_train",ed_train)
        np.savetxt(f"{model_path}/ed_test",ed_test)

    print(model_path)


# python torch_dnn.py train_submit_cnn train_ht_cnn
def train_ht_cnn(name, alpha100, g100, optimizer, net_type, fc_init, lr=0.001, activation="tanh",                                             # network structure
                 loss_func_name="cross_entropy", bs=1024, weight_decay=0, momentum=0, num_workers=1, epochs=1000,                             # training setting (weight_decay=0.0001)
                 save_epoch=1000, weight_save=0, stablefit_save=0):                                                                           # save data options

    """
    For training CNNs and saving accuracies and losses.
    """
    
    from time import time; t0 = time()
    from NetPortal.models import ModelFactory

    alpha100, g100 = int(alpha100), int(g100)
    alpha = alpha100/100.
    g = g100/100.

    bs = int(bs)
    lr = float(lr)
    lr_ls = [(0,lr)]
    epochs = int(epochs)
    num_workers = int(num_workers)

    train_ds, valid_ds = set_data(name,False)
    N = len(train_ds[0][0])
    C = len(train_ds.classes)

    # generate random id and date
    import uuid
    from datetime import datetime
    model_id = str(uuid.uuid1())
    train_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # create path
    model_path = f"{path_names.trained_path}/{net_type}_{alpha100}_{g100}_{model_id}_{name}_{optimizer}_lr={lr}_bs={bs}_epochs={epochs}"
    if not os.path.isdir(model_path): os.makedirs(f'{model_path}')

    # log marker
    np.savetxt(f"{model_path}/{fc_init}", np.array([0]))

    # move entire dataset to GPU
    train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(dev),
                             torch.tensor(train_ds.targets).to(dev))
    valid_ds = TensorDataset(torch.stack([e[0] for e in valid_ds]).to(dev),
                             torch.tensor(valid_ds.targets).to(dev))

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)#, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    steps = len(train_dl)
    print("Data loaded.")

    kwargs = {"alpha" :alpha, "g": g, "num_classes": C, "architecture": net_type, "activation": activation, "dataset": name, "fc_init": fc_init}
    model = ModelFactory(**kwargs)
    # printed info
    print(model)
    print(f"Training method: {optimizer}, lr={lr}, batch_size={bs}, epochs={epochs}" + "\n")
    print(rf"Initialization: alpha={alpha}, g={g}" + "\n")
    model.to(dev)
    
    if optimizer == 'sgd':
        opt_func = optim.SGD
        opt = opt_func(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        opt_func = optim.Adam
        opt = opt_func(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print("THIS OPTIMIZER DOES NOT EXIST!")

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
        if epoch == 0.1:
            # train_loss, train_acc, val_loss, val_acc, epoch_time
            epoch_path = f"{model_path}/epoch_{epoch + 1}"
            #if not os.path.isdir(epoch_path): os.makedirs(epoch_path)
            final_acc = train_epoch(model, loss_func, opt, train_dl, valid_dl, hidden_epochs=num_workers,
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
    log_path = path_names.log_path
    log_model(log_path, model_path, net_type=net_type, model_id=model_id, train_date=train_date, name=name, activation=activation, alpha100=alpha100, g100=g100,
              loss_func_name=loss_func_name, optimizer=optimizer, bs=bs, lr=lr, lr_ls=lr_ls, weight_decay=weight_decay, num_workers=num_workers, 
              epochs=epochs, steps=steps, final_acc=final_acc, total_time=total_time)
    print(model_path)


# train networks in array jobs
def train_submit(*args):
    from qsub import qsub

    dataset_ls = ["mnist"]
    #dataset_ls = ["cifar10"]

    #alpha100_ls = list(range(100,111,10))
    #alpha100_ls = list(range(120,141,10))
    #alpha100_ls = list(range(150,171,10))
    #alpha100_ls = list(range(180,201,10))
    #g100_ls = list(range(25, 301, 25))

    alpha100_ls = [100,200]
    g100_ls = [100]
    optimizer_ls = ["sgd"]
    #bs_ls = [int(2**p) for p in range(1,12)]
    bs_ls = [1024]

    pbs_array_data = [(name, alpha100, g100, optimizer, bs)
                      for name in dataset_ls
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for optimizer in optimizer_ls
                      for bs in bs_ls
                      ]

    #print(pbs_array_data)
    #pbs_array_data = pbs_array_data[0:1]

    pbs_array_true = pbs_array_data
    #pbs_array_true = [("mnist", 140, 275, "sgd"), ("mnist", 180, 175, "sgd")]
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

    qsub(f'python {sys.argv[0]} {" ".join(args)}',    
         pbs_array_true, 
         path='/project/dnn_maths/project_qu3/',
         P='dnn_maths',
         #P='phys_DL',
         #P='PDLAI',
         #P='ddl',
         ngpus=1,
         ncpus=1,
         #walltime='1:59:59',
         walltime='12:59:59',
         mem='4GB') 


# version corresponding train_dnn
def train_submit_cnn(*args):
    from qsub import qsub

    #net_type_ls = ["alexnet", "resnet14"]
    #alpha100_ls = list(range(100,201,10))

    #alpha100_ls = list(range(100,111,10))
    #alpha100_ls = list(range(120,141,10))
    #alpha100_ls = list(range(150,171,10))
    #alpha100_ls = list(range(180,201,10))
    #g100_ls = list(range(25, 301, 25))
    #g100_ls = list(range(10, 201, 10))

    alpha100_ls = [200] 
    g100_ls = [25,100,300]

    #net_type_ls = ["convnet_old"]
    #net_type_ls = ["van20nobias"]
    #net_type_ls = ["van50","van100","van200"]
    #net_type_ls = ["van100"]
    #net_type_ls = ["van100nobias"]
    #net_type_ls = ["resnet14_ht"]
    net_type_ls = ["alexnetold"]
    #fc_init = "fc_ht"
    #fc_init = "fc_orthogonal"
    fc_init = "fc_default"
    #dataset_ls = ["mnist"]
    dataset_ls = ["cifar10"]
    optimizer_ls = ["sgd"]

    pbs_array_data = [(name, alpha100, g100, optimizer, net_type, fc_init)
                      for name in dataset_ls
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for optimizer in optimizer_ls
                      for net_type in net_type_ls
                      ]

    #print(len(pbs_array_data))
    #pbs_array_data = pbs_array_data[0:1]

    pbs_array_true = pbs_array_data
    """
    pbs_array_no = [(100,25), (100,100), (100,300), (200,25), (200,100), (200,300)]
    for sub in pbs_array_data:
        if (sub[1],sub[2]) not in pbs_array_no:
            pbs_array_true.append(sub)
        #print((sub[1],sub[2]))
        #break
    """

    print(len(pbs_array_true))

    qsub(f'python {sys.argv[0]} {" ".join(args)}',    
         pbs_array_true, 
         path='/project/dnn_maths/project_qu3/',
         #P='dnn_maths',
         #P='phys_DL',
         P='PDLAI',
         #P='ddl',
         ngpus=1,
         ncpus=1,
         #walltime='5:59:59',
         walltime='0:59:59',
         mem='6GB') 
     

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

import numpy as np
import pandas as pd
import torch
from os.path import join
from path_names import root_data

# ---------- Computation related to D_q ----------

# numpy version
def IPR(vec, q):
    return sum(abs(vec)**(2*q)) / sum(abs(vec)**2)**q

# pytorch version
def compute_IPR(vec,q):
    if isinstance(vec, torch.Tensor):
        vec = torch.abs(vec.detach().cpu())
        ipr = torch.sum(vec**(2*q))/torch.sum(vec**2)**q
    else:
        vec = np.abs(vec)
        ipr = np.sum(vec**(2*q))/np.sum(vec**2)**q

    return ipr

# numpy version
def D_q(vec, q):
    return np.log(IPR(vec, q)) / (1-q) / np.log(len(vec))

# pytorch version
def compute_dq(vec,q):
    if isinstance(vec, torch.Tensor):
        return  (torch.log(IPR(vec,q))/np.log(len(vec)))/(1 - q)
    else:
        return  (np.log(IPR(vec,q))/np.log(len(vec)))/(1 - q)

# ---------- Load phase transition lines ----------

def load_transition_lines():

    # phase transition lines
    
    # old version
    """
    boundaries = []
    boundaries.append(pd.read_csv(f"{root_data}/phase_bound/phasediagram_pow_1_line_1.csv", header=None))
    for i in list(range(1,10,2)):
        boundaries.append(pd.read_csv(f"{root_data}/phase_bound/phasediagram_pow_{i}_line_2.csv", header=None))
    bound1 = boundaries[0]
    """

    # new version
    bound1 = pd.read_csv(join(root_data, "phase_bound/phasediagram_pow_1_line_1.csv"), header=None)
    boundaries = []
    bd_path = join(root_data, "phasediagram")
    for i in range(1,92,10):
    #for i in range(1,102,10):
        boundaries.append(pd.read_csv(f"{bd_path}/pow_{i}.csv"))
    
    return bound1, boundaries

# ---------- Training functions ----------

# save current state of model
def save_weights(model, path):
    weights_all = np.array([]) # save all weights vectorized 
    for p in model.parameters():
        weights_all = np.concatenate((weights_all, p.detach().cpu().numpy().flatten()), axis=0)
    #np.savetxt(f"{path}", weights_all, fmt='%f')
    # saveas .npy instead
    np.save(f"{path}", weights_all)

# get weights flatten
def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

# save weights
def store_model(model, path ,save_type):       
    if save_type == 'torch':
        torch.save(model, path)        # save as torch
    elif save_type == 'numpy':
        save_weights(model, path)      # save as .npy (flattened)
    else:
        raise TypeError("save_type does not exist!")  

# ---------- Extra quantities ----------

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
        arr[lidx,:] = [ compute_IPR(eigvals[:,0],q) for q in q_ls ]
                          
    return arr

def effective_dimension(model, hidden_layer, with_pc:bool, q_ls):    # hidden_layer is input image    
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
                C = C.numpy()
                #_, eigvecs = torch.eig(C,eigenvectors=with_pc)  # best use torch.eigh for this 
                _, eigvecs = np.linalg.eigh(C)         
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

# ---------- Getting relevant folder names ----------

# input data_path is folder which contains the batch of networks trained for a phase transition diagram
def setting_from_path(data_path, alpha100, g100):
    import os
    all_net_folder = data_path.split("/")[-1] if len(data_path.split("/")[-1]) > 0 else data_path.split("/")[-2]
    fcn = all_net_folder.split("_")[0]

    if "fcn_grid" in data_path:
        alpha, g = int(alpha100)/100., int(g100)/100.
        # storage of trained nets
        net_type = f"{fcn}_mnist_tanh"
        net_ls = [net for net in next(os.walk(data_path))[1] if f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch" in net]
        assert len(net_ls) > 0
        net_folder = net_ls[0]
        epoch_last = int([ s for s in net_folder.split("_") if "epoch" in s ][0][5:])
        #net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
        activation = "tanh"
    else:
        # find folder which matches alpha100 and g100
        net_ls = [net for net in next(os.walk(data_path))[1] if fcn in net and "epochs=" and f"_{alpha100}_{g100}_" in net]
        assert len(net_ls) > 0
        net_folder = net_ls[0]
        net_setup = pd.read_csv(join(data_path,net_folder,"log"))
        activation = net_setup.loc[0,"activation"]
        epoch_last = int(net_setup.loc[0,"epochs"])   

    return fcn, activation, epoch_last, net_folder      



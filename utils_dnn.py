import numpy as np
import pandas as pd
from os.path import join
from path_names import root_data

# ---------- Computation ----------

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


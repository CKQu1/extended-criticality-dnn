import matplotlib.colors as mcl
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from os.path import join, isdir, isfile

import sys
import os
import re

from mpl_toolkits.mplot3d import axes3d

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from path_names import root_data

def layer_pca(hidden_stacked, **kwargs):
    """
    Performs PCA for hidden_stacked.
    """
    #import torch

    h_dim = hidden_stacked.shape    
    if 'n_pc' in kwargs:
        n_pc = kwargs.get('n_pc')
        #hs_proj = torch.zeros((h_dim[0], n_pc, h_dim[1]))
        #hs_proj = torch.zeros((h_dim[0], h_dim[1], n_pc))
        hs_proj = np.zeros((h_dim[0], h_dim[1], n_pc))
    else:
        #hs_proj = torch.zeros((h_dim[0], h_dim[2], h_dim[1]))
        #hs_proj = torch.zeros_like(hidden_stacked)        
        hs_proj = np.zeros_like(hidden_stacked)
    #singular_values = torch.zeros((h_dim[0], min(h_dim[1], h_dim[2])))
    singular_values = np.zeros((h_dim[0], min(h_dim[1], h_dim[2])))

    for l in tqdm(range(h_dim[0])):
        #hidden = hidden_stacked[l,:].detach().numpy()
        hidden = hidden_stacked[l,:]
        hidden = hidden - hidden.mean(0, keepdims=True)
        u, s, v = np.linalg.svd(hidden, full_matrices=False)
        if l == h_dim[0] - 1:
            print(hs_proj.shape)
            print(singular_values.shape)
            print('\n')
            print(hidden.shape)
            print(s.shape)
            print(u.shape)

        if 'n_pc' in kwargs:
            #norm_s = s[:n_pc] / s[:n_pc].max()
            #u = (norm_s[None, :]) * u[:, :n_pc] 
            #u = u / np.abs(u).max()
            #hs_proj[l,:] = torch.tensor(u[:, :3])
            hs_proj[l,:] = u[:, :3]
        else:
            #norm_s = s[:] / s[:].max()
            #u = (norm_s[None, :]) * u[:, :] 
            #u = u / np.abs(u).max()
            #hs_proj[l,:] = torch.tensor(u)
            hs_proj[l,:] = u

        #singular_values[l,:] = torch.tensor(s)
        singular_values[l,:] = s

    return hs_proj, singular_values


def gcircle_prop(N, L, N_thetas, alpha100, g100, *args):
    """
    Gets activations for a 2D circular manifold propagated through MLP with tanh activation:
        - N (int): size of the hidden layers
        - L (int): depth of network
        - N_thetas (int): the number of intervals that divides [0,2\pi]
    """
    global hs_all, hs

    #import torch
    #from nporch.theory import 
    
# get the SEM of the manifold distances propagated through the DNN layers
    # Extract numeric arguments.
    N = int(N)
    L = int(L)
    N_thetas = int(N_thetas)
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    # operate at fixed point
    #q_fixed = q_star(alpha,g)
    q_fixed = 1
    #print(q_fixed)
    # Generate circular manifold.
    hs = np.zeros([N, N_thetas])
    thetas = np.linspace(0, 2*np.pi, N_thetas)
    hs[0,:] = q_fixed * np.cos(thetas)
    hs[1,:] = q_fixed * np.sin(thetas)
    hs_all = np.zeros([L + 1, N_thetas, N])  # distance SEMs, angular SEMs
    hs_all[0,:,:] = hs.T
    from scipy.stats import levy_stable
    for l in tqdm(range(L)):
        # preactivations (same case as in random_dnn/random_dnn.py)
        hs = np.dot(levy_stable.rvs(alpha, 0, size=[N,N], scale=g*(0.5/N)**(1./alpha)),
                    np.tanh(hs))    
        hs_all[l + 1,:,:] = hs.T
    #return torch.tensor(hs_all)   # convert to torch
    return hs_all


def gcircle_save(N, L, N_thetas,
                 alpha100, g100, *args):
    """
    Saves data from gcircle_prop().
    """

    #import torch    
    #path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    if len(args) == 0:
        #path = join(root_data, 'geometry_data', 'gcircle3d_data')
        path = join(root_data, 'geometry_data', 'gcircle3d_data_v2')
    else:
        path = args[0]
        if len(args) == 2:
            ensemble = int(args[1])
    
    if not os.path.exists(path): os.makedirs(path)

    if len(args) < 2:
        # torch.save(gcircle_prop(N, L, N_thetas, alpha100, g100),
        #         f"{path}/gcircles_alpha_{alpha100}_g_{g100}")
        np.save(f"{path}/gcircles_alpha100={alpha100}_g100={g100}", gcircle_prop(N, L, N_thetas, alpha100, g100))                
    else:
        # torch.save(gcircle_prop(N, L, N_thetas, alpha100, g100),
        #         f"{path}/gcircles_alpha_{alpha100}_g_{g100}_ensemble={ensemble}")        

        np.save(f"{path}/gcircles_alpha100={alpha100}_g100={g100}_ensemble={ensemble}", gcircle_prop(N, L, N_thetas, alpha100, g100))           


# check https://github.com/ganguli-lab/deepchaos/blob/master/cornet/curvature.py
def corr_prop(N, L, N_thetas, alpha100, g100, *args):
    """
    Get angle correlation averaged over one network ensemble
    """    
    global hs_all, hs, R, r_, r_std, thetas
    
    from scipy.stats import levy_stable

    # Extract numeric arguments.
    N = int(N)
    L = int(L)
    N_thetas = int(N_thetas)
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    q_fixed = 1
    # Generate circular manifold.
    hs = np.zeros([N, N_thetas])
    #thetas = np.linspace(0, 2*np.pi, N_thetas)
    thetas = np.linspace(-np.pi, np.pi, N_thetas, endpoint=False)
    hs[0,:] = q_fixed * np.cos(thetas)
    hs[1,:] = q_fixed * np.sin(thetas)
    hs_all = np.zeros([L + 1, N_thetas, N])  # distance SEMs, angular SEMs
    hs_all[0,:,:] = hs.T    

    r_ = np.zeros([L+1, N_thetas])
    r_std = np.zeros([L+1, N_thetas])
    for l in tqdm(range(L)):
        # preactivations (same case as in random_dnn/random_dnn.py)
        hs = np.dot(levy_stable.rvs(alpha, 0, size=[N,N], scale=g*(0.5/N)**(1./alpha)),
                    np.tanh(hs))    
        hs_all[l + 1,:,:] = hs.T
        V = hs/np.linalg.norm(hs, axis=0)
        R = np.dot(V.T, V)
        #quit()  # delete
        for tidx in range(len(thetas)):
            d = np.diag(R, tidx-int(N_thetas/2))
            r_[l+1,tidx] = d.mean()
            r_std[l+1,tidx] = d.std()

    return r_, r_std


def corr_save(N, L, N_thetas,
              alpha100, g100, *args):
    """
    Saves data from corr_prop().
    """
    if len(args) == 0:
        path = join(root_data, 'geometry_data', 'corr_data')
    else:
        path = args[0]
        if len(args) == 2:
            ensemble = int(args[1])
    
    if not os.path.exists(path): os.makedirs(path)

    if len(args) < 2:
        r_, r_std = corr_prop(N, L, N_thetas, alpha100, g100)
        np.save(f"{path}/corr_alpha100={alpha100}_g100={g100}", r_)       
        np.save(f"{path}/corrstd_alpha100={alpha100}_g100={g100}", r_std)           
    else:
        r_, r_std = corr_prop(N, L, N_thetas, alpha100, g100)
        np.save(f"{path}/corr_alpha100={alpha100}_g100={g100}_ensemble={ensemble}", r_)    
        np.save(f"{path}/corrstd_alpha100={alpha100}_g100={g100}_ensemble={ensemble}", r_std)  


def corr_plot(N, alpha100s=[100,150,200], g100s=[10,100,300], *args):
    '''
    Plots results from corr_save()
    '''

    import matplotlib.pyplot as plt
    global thetas, r_, r_std, V, R, hs, thetas_mean, thetas_std

    #c_ls = ['grey', 'r', 'b', 'g']
    c0 = 'grey'
    c_ls = list(mcl.TABLEAU_COLORS.keys())
    path = join(root_data, 'geometry_data', 'corr_data')

    selected_ls = [5, 10, 20, 30]
    N = int(N)
    #alpha100s = literal_eval(alpha100s)
    #g100s = literal_eval(g100s)
    ensembles = list(range(5))

    nrows, ncols = len(alpha100s), len(g100s)
    fig_size = (11/2*ncols,9.142/2*nrows)
    fig, axs = plt.subplots(nrows, ncols,sharex=True,sharey=True,figsize=fig_size)
    total_seeds = 10
    path = join(root_data, 'geometry_data', f'corr_data_N={N}')
    
    for aidx, alpha100 in enumerate(alpha100s):
        for gidx, g100 in tqdm(enumerate(g100s)):
            alpha, g = int(alpha100)/100, int(g100)/100    
            for ii, ensemble in enumerate(ensembles):        
                if ii == 0:
                    r_ = np.load(f"{path}/corr_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy")    
                    r_std = np.load(f"{path}/corrstd_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy") 
                else:
                    r_ += np.load(f"{path}/corr_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy")  
                    r_std += np.load(f"{path}/corrstd_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy")

            r_ /= len(ensembles)
            r_std /= len(ensembles)

            axis = axs[aidx,gidx]
            #axis.set_xlim([-0.5,0.5])    

            q_fixed = 1
            if aidx==0 and gidx==0:
                N_thetas = r_.shape[1]
                thetas = np.linspace(-np.pi, np.pi, N_thetas, endpoint=False)         
                thetas_mean = np.zeros([len(thetas)]); thetas_std = np.zeros([len(thetas)])
                hs = np.zeros([N, N_thetas])
                hs[0,:] = q_fixed * np.cos(thetas)
                hs[1,:] = q_fixed * np.sin(thetas)   
                V = hs/np.linalg.norm(hs, axis=0)
                R = np.dot(V.T, V)             
                for tidx in range(len(thetas)):
                    d = np.diag(R, tidx-int(N_thetas/2))             
                    thetas_mean[tidx] = d.mean()
                    thetas_std[tidx] = d.std()   

            axis.plot(thetas, thetas_mean, linestyle='--', c=c0, label='Input')
            axis.fill_between(thetas, thetas_mean-thetas_std, thetas_mean+thetas_std, 
                              color=c0, alpha=0.2, label='_nolegend_')                

            for lidx, l in enumerate(selected_ls):
                axis.plot(thetas, r_[l], c=c_ls[lidx], label=rf'$l$ = {l}')
                axis.fill_between(thetas, r_[l]-r_std[l], r_[l]+r_std[l], color=c_ls[lidx], alpha=0.2, label='_nolegend_')


            axs[0,gidx].set_title(rf'$\sigma_w$ = {g}')
            axs[1,gidx].set_xlabel(r'$\Delta \theta$ (offset)')
        axs[aidx,0].set_ylabel(rf'$\alpha$ = {alpha}')

    axs[0,0].legend(frameon=False)

    plot_path = join(root_data, 'figure_ms', 'gcircle')
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    #plt.legend()
    fname = f'corr_N={N}_alpha100s={alpha100s}_g100s={g100s}_total={total_seeds}.pdf'
    plt.savefig(join(plot_path, fname))                
    print(f'Saved as {join(plot_path, fname)}')


def avalanche_prop(N, alpha100, g100, seed, *args):

    global hs_all, hs, S_acts, S_actss, S_actss_arr, Ls

    from scipy.stats import levy_stable

    threshold = 0
    #threshold = 1e-10

    N = int(N)
    alpha, g = int(alpha100)/100., int(g100)/100.  
    seed = int(seed)  
    wmat = levy_stable.rvs(alpha, 0, size=[N,N], scale=g*(0.5/N)**(1./alpha), random_state=seed)  # quenched

    S_actss = []
    Ls = np.zeros([N])   
    for idx in tqdm(range(N)):
    #for idx in [0]:  # test

        hs = np.zeros([N])
        hs[idx] = 1
        S_act = 1  # dummy
        S_acts = []  # number of activated neurons
        L = 0
        while S_act > 0:
            # preactivations (same case as in random_dnn/random_dnn.py)
            hs = np.dot(wmat, np.tanh(hs))   
            S_act = (np.abs(hs) > threshold).sum()
            S_acts.append(S_act)
            L += 1
            #if L % 10 == 0:
            #    print(f'L = {L}, min: {hs.min()}, max: {hs.max()}')

        S_actss.append(S_acts)   
        Ls[idx] = L     
        #print(S_acts)

    assert len(S_actss) == len(Ls)

    L_max = int(Ls.max())
    S_actss_arr = np.full((N+1,L_max), np.nan)
    for idx in range(len(S_actss)):
        S_acts = S_actss[idx]
        S_actss_arr[idx,:len(S_acts)] = S_acts

    return S_actss_arr, Ls   


def avalanche_save(N, alpha100, g100, seed, path, *args):
    """
    Saves data from avalanche_prop().
    """
      
    if not os.path.exists(path): os.makedirs(path)

    S_actss_arr, Ls = avalanche_prop(N, alpha100, g100, seed)
    np.save(f"{path}/size_alpha100={alpha100}_g100={g100}_seed={seed}", S_actss_arr)
    np.save(f"{path}/lifetime_alpha100={alpha100}_g100={g100}_seed={seed}", Ls)


def avalanche_plot(N, alpha100s, g100s, *args):
    '''
    Plots the avalanche size and lifetime, data obtained from avalanche_save()
    '''
    global S_actss, Ls, S_ag, L_ag, L, alpha, g

    import matplotlib.pyplot as plt

    N = int(N)
    alpha100s = literal_eval(alpha100s)
    g100s = literal_eval(g100s)

    nrows, ncols = 2, len(alpha100s)
    fig_size = (11/2*ncols,9.142/2*nrows)
    fig, axs = plt.subplots(nrows, len(alpha100s),sharex = False,sharey=False,figsize=fig_size)
    total_seeds = 10
    path = join(root_data, 'geometry_data', f'avalanche_data_N={N}')

    for aidx, alpha100 in enumerate(alpha100s):
        for gidx, g100 in tqdm(enumerate(g100s)):
            alpha, g = int(alpha100)/100, int(g100)/100
            S_ag = np.array([])
            L_ag = np.array([])
            for seed in range(total_seeds):
                S_actss = np.load(f"{path}/size_alpha100={alpha100}_g100={g100}_seed={seed}.npy")
                Ls = np.load(f"{path}/lifetime_alpha100={alpha100}_g100={g100}_seed={seed}.npy")     
                for idx in range(N):
                    L = int(Ls[idx])
                    #S_ag = np.concatenate((S_ag, S_actss[idx,:L]))
                    S_ag = np.hstack((S_ag, S_actss[idx,:L]))
                #L_ag = np.concatenate((L, L_ag))               
                L_ag = np.hstack((L, L_ag))

            # avalanche size distribution
            axs[0,aidx].hist(S_ag, bins=50, density=True, label=rf'$g$ = {g}')
            
            # avalanche lifetime distribution         
            axs[1,aidx].hist(L_ag, bins=50, density=True, label=rf'$g$ = {g}')

            # log-log scale
            axs[0,aidx].set_xscale('log'); axs[0,aidx].set_yscale('log')
            axs[1,aidx].set_xscale('log'); axs[1,aidx].set_yscale('log')

        axs[0,aidx].set_title(rf'$\alpha$ = {alpha}')

    axs[0,0].set_ylabel('Avalanche size')
    axs[1,0].set_ylabel('Avalanche lifetime')

    plot_path = join(root_data, 'figure_ms', 'gcircle')
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    plt.legend()
    plt.savefig(join(plot_path, f'avalanche_N={N}_alpha100s={alpha100s}_g100s={g100s}_total={total_seeds}.pdf'))    


# functions: gcircle_save()
def submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup   

    save_path = join(root_data, 'geometry_data', 'gcircle3d_data_v2')
    N, L, N_thetas = 784, 50, 1000
    total_ensembles = 10
    #pbs_array_data = [(alpha100, g100, save_path, ensemble)
    pbs_array_data = [(N, L, N_thetas,alpha100, g100, save_path, ensemble)
                      #for alpha100 in [100, 120, 200]
                      for alpha100 in [200]
                      for g100 in [10, 100, 300]
                      #for alpha100 in [120]
                      #for g100 in [10, 100, 300]                      
                      for ensemble in range(total_ensembles)                      
                      #for ensemble in [0]
                      ]
    #qsub(f'python geometry_analysis/greatcircle_proj_trial.py {sys.argv[0]} {" ".join(args)}',
    #qsub(f'python greatcircle_proj_trial.py {sys.argv[0]} {" ".join(args)}', 

    select, ncpus, ngpus = 1, 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    #singularity_path = ''
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             #path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
             #path=join(root_data,'geometry_data', 'jobs_all', 'gcircle_save'),
             path=join(root_data,'geometry_data', 'jobs_all', 'gcircle3d'),             
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             select=select,             
             walltime='23:59:59',   # small/medium
             mem='2GB')         


# functions: gcircle_save()
# def submit_v2(*args):
#     from qsub import qsub, job_divider, project_ls, command_setup
#     N, L, N_thetas = 784, 50, 1000
#     path = join(root_data, 'geometry_data', 'gcircle_hidden_layers')
#     pbs_array_data = [(N, L, N_thetas, alpha100, g100, path, ensemble)
#                       for alpha100 in [120,150,200]
#                       for g100 in [10, 100, 300]
#                       for ensemble in range(50)
#                       ]

#     select, ncpus, ngpus = 1, 1, 0
#     singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
#     bind_path = "/project"
#     command = command_setup(singularity_path, bind_path=bind_path)

#     perm, pbss = job_divider(pbs_array_data, len(project_ls))
#     for idx, pidx in enumerate(perm):
#         pbs_array_true = pbss[idx]
#         print(project_ls[pidx])
#         qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
#              pbs_array_true, 
#              ##path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
#              path=join(root_data,'geometry_data', 'jobs_all', 'gcircle_hidden_layers'),
#              P=project_ls[pidx],
#              ngpus=ngpus,
#              ncpus=ncpus,
#              select=select,             
#              walltime='23:59:59',   # small/medium
#              mem='2GB')           


# functions: avalanche_save()
def submit_v2(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    N = 1000
    #N = 10000
    total_seeds = 10
    path = join(root_data, 'geometry_data', f'avalanche_data_N={N}')
    # test
    #pbs_array_data = [(N, 120, 5, 0, path), (N, 120, 10, 0, path)]

    pbs_array_data = [(N, alpha100, g100, seed, path)
                      for alpha100 in [120,150,180,200]
                      for g100 in [5,10,15,20,25,30]
                      for seed in range(total_seeds)
                      ]

    select, ncpus, ngpus = 1, 1, 0
    #singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    singularity_path = ''
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             ##path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
             path=join(root_data,'geometry_data', 'jobs_all', 'corr_job'),
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             select=select,             
             walltime='23:59:59',   # small/medium
             #mem='2GB'
             #mem='6GB'  # N = 1000
             #mem='26GB'  # N = 10000
             )         


# functions: corr_save()
def submit_v3(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    N, N_thetas = 1000, 1000    
    total_seeds = 5
    path = join(root_data, 'geometry_data', f'corr_data_N={N}')

    L = 50
    #alpha100s = [100,150,200]
    alpha100s = [150,200]
    g100s = [10,50,100,200,300]
    pbs_array_data = [(N, L, N_thetas, alpha100, g100, path, seed)
                      for alpha100 in alpha100s
                      for g100 in g100s
                      #for seed in range(total_seeds)
                      for seed in [4]
                      ]

    #pbs_array_data = pbs_array_data[:2]

    select, ncpus, ngpus = 1, 1, 0
    #singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    singularity_path = ''
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             ##path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
             path=join(root_data,'geometry_data', 'jobs_all', 'corr_job'),
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             select=select,             
             walltime='23:59:59', 
             mem='2GB'
             )                           


# ----- preplot -----
def gcircle_preplot(alpha100, g100, ensemble, *args):
    """
    Saves the PCs and eigenvalues (correlation matrix) to be plotted in gcircle_plot().
    """

    #import torch    

    #path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    #path = join(root_data, 'geometry_data', 'gcircle3d_data')
    path = join(root_data, 'geometry_data', 'gcircle3d_data_v2')
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}') 

    #hs_all = torch.load(f"{path}/gcircles_alpha_{alpha100}_g_{g100}")
    #hs_all = np.load(f"{path}/gcircles_alpha_{alpha100}_g_{g100}")
    hs_all = np.load(f"{path}/gcircles_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy")
    hs_proj, singular_values = layer_pca(hs_all, n_pc=3)

    # torch.save(hs_proj, f"{path}/eigvecs_alpha_{alpha100}_g_{g100}")
    # torch.save(singular_values, f"{path}/singvals_alpha_{alpha100}_g_{g100}")
    np.save(f"{path}/eigvecs_alpha100={alpha100}_g100={g100}_ensemble={ensemble}", hs_proj)
    np.save(f"{path}/singvals_alpha100={alpha100}_g100={g100}_ensemble={ensemble}", singular_values)    

    print("Preplot done!")

# functions: gcircle_preplot()
def submit_preplot(*args):
#def submit_preplot(path):
    from qsub import qsub, job_divider, project_ls, command_setup

    #data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    path = join(root_data, 'geometry_data', 'gcircle3d_data')
    # find the `alpha100`s and `g100`s of the files in the folder
    # pbs_array_data = set([tuple(re.findall('\d+', fname)[:2]) for fname in os.listdir(data_path)
    #                   if all(s in fname for s in ('alpha', 'g', 'gcircles'))])

    total_ensembles = 10
    pbs_array_data = [(alpha100, g100, ensemble)
                      #for alpha100 in [100,120,200]
                      for alpha100 in [200]
                      for g100 in [10, 100, 300]
                      for ensemble in range(total_ensembles)
                      ]

    select, ncpus, ngpus = 1, 1, 0
    #singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    singularity_path = ''
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)    


    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        #qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
        qsub(f'{command} {sys.argv[0]} gcircle_preplot',
             pbs_array_true, 
             #path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
             #path=join(root_data,'geometry_data', 'jobs_all', 'avalanche_jobs'),
             path=join(root_data,'geometry_data', 'jobs_all', 'layer_pca'),
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             select=select,             
             walltime='23:59:59',   # small/medium
             mem='6GB'  # N = 1000
             #mem='26GB'  # N = 10000
             )   

# ----- plot -----

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old

def gcircle_plot(ensemble, cbar_separate=True):
    global alpha_mult_pair, folder_ii, singular_values, ls, f_ii

    #alpha100_ls, g100_ls = [200], [1]
    #alpha100_ls, g100_ls = [100,150,200], [1,100,200]
    #alpha100_ls, g100_ls = [100,150,200], [25,100,200]
    #alpha100_ls, g100_ls = [100,150,200], [25,100,300]
    alpha100_ls, g100_ls = [100,150,200], [10,100,300]
    #alpha100_ls, g100_ls = [100,120,200], [10,100,300]
    #alpha100_ls, g100_ls = [100], [10,100,300]
    if ensemble != 'None':
        ensemble = int(ensemble)    

    from time import time
    t0 = time()

    import torch

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import figure
    from matplotlib.gridspec import GridSpec

    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib import cm
    # colorbar
    cm_type = 'twilight'
    plt.rcParams["font.family"] = 'sans-serif'     # set plot font globally
    #plt.rcParams["font.family"] = "Helvetica"

    tick_size = 18.5
    label_size = 18.5
    axis_size = 18.5
    legend_size = 14
    linewidth = 0.8
    text_size = 14

    #data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    #plot_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
    if ensemble == 'None':
        data_path = join(root_data, 'geometry_data', 'gcircle3d_data')
    else:
        data_path = join(root_data, 'geometry_data', 'gcircle3d_data_v2')
    plot_path = join(root_data, 'figure_ms', 'gcircle')
    if not os.path.isdir(plot_path): os.makedirs(plot_path)
   
    alpha_mult_pair = []
    # Projection plot
    #ls_all = [list(range(3)), list(range(3,6)), list(range(6,9))]
    ls_all = []    
    gidx = 0
    for alpha100 in alpha100_ls:
        ls = []
        for g100 in g100_ls:
            alpha_mult_pair.append((alpha100,g100))  
            ls.append(gidx)
            gidx += 1
        ls_all.append(ls)
 
    # Font size
    tick_size = 16.5
    label_size = 16.5
    axis_size = 16.5
    legend_size = 14
    linewidth = 0.8

    # Set up figure
    #rows = 3
    rows = 1
    cols = 3

    #layer_ii = [5,15,35]
    #layer_ii = [35]
    #layer_ii = [45]
    layer_ii = [30]
    #layer_ii = [5]
    #layer_ii = [25]

    #assert max(layer_ii) <= L, "layer_ii incorrect"

    # thetas
    N_thetas = 1000
    thetas = np.linspace(0, 2*np.pi, N_thetas)

    vert_dists = [0.78, 0.5, 0.22]
    for ls in ls_all:
        #fig = plt.figure(figsize=(9.5,7.142))
        fig = plt.figure(figsize=(9.5,7.142/3 + 0.75))

        """
        mins = []
        maxs = []
        for f_ii in range(len(ls)):
            folder_ii = ls[f_ii]
            alpha100, g100 = alpha_mult_pair[folder_ii]
            print((alpha100,g100))
            alpha, m = alpha100/100, g100/100
            hs_proj = torch.load(f"{data_path}/eigvecs_alpha_{alpha100}_g_{g100}")
            singular_values = torch.load(f"{data_path}/singvals_alpha_{alpha100}_g_{g100}")

            for k in range(len(layer_ii)):
                ii = layer_ii[k]
                z = hs_proj[ii, :, 2]
                mins.append(min(z))
                maxs.append(max(z))
        """
        #cmap_bd = [min(mins), max(maxs)]
        #cmap_bd = [-0.05,0.05]
        cmap_bd = [0, 2*np.pi]

        for f_ii in range(len(ls)):
            folder_ii = ls[f_ii]
            alpha100, g100 = alpha_mult_pair[folder_ii]
            print(f'(alpha100, g100) = {(alpha100,g100)}')
            alpha, m = alpha100/100, g100/100
            if ensemble == 'None':
                hs_proj = torch.load(f"{data_path}/eigvecs_alpha_{alpha100}_g_{g100}")
                singular_values = torch.load(f"{data_path}/singvals_alpha_{alpha100}_g_{g100}")
            else:
                hs_proj = np.load(f"{data_path}/eigvecs_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy")
                singular_values = np.load(f"{data_path}/singvals_alpha100={alpha100}_g100={g100}_ensemble={ensemble}.npy") 

            #quit()  # delete                                   

            for k in range(len(layer_ii)):
                # figure index
                #fig_ii = 3*f_ii + k + 1
                fig_ii = rows*f_ii + k + 1
                print(fig_ii)

                ax = fig.add_subplot(rows,cols,fig_ii,projection='3d')
                #ax = fig.add_subplot(3,3,fig_ii)
                ii = layer_ii[k]
                total_var = sum(singular_values[ii,:])
                print(f"Total variance {total_var}")
                print(f"Top 3 singular value: {singular_values[ii, :3]}")

                # singular_values
                #hs_proj[ii] = hs_proj[ii] * singular_values[ii, 0:3]
                x, y ,z = hs_proj[ii, :, 0], hs_proj[ii, :, 1],hs_proj[ii, :, 2]

                im = ax.scatter(x , y , z, c=thetas, vmin=cmap_bd[0], vmax=cmap_bd[1], marker='.', s=4, alpha=1, cmap=cm.get_cmap(cm_type))
                #im = ax.scatter(x , y , z, c=z, vmin=cmap_bd[0], vmax=cmap_bd[1], marker='.', s=4, cmap=cm.get_cmap(cm_type))
                # lines
                #if alpha != 2:
                ax.plot(x , y , z,color='k', zorder=0, linewidth=0.25, alpha=.35)
                ax.zaxis.set_rotate_label(False) 

                #ax.set_title(r"$D_w^{1/\alpha}$ = " + f"{m}", loc='left', fontsize=label_size - 3)

                if fig_ii == 2:
                    #fig.text(0.07, vert_dists[f_ii], r"$D_w^{1/\alpha}$" + f" = {m}", rotation=90, va='center', fontsize=label_size - 2)
                    #fig.text(0.07, vert_dists[f_ii], "Layer {fname}".format(fname=layer_ii[k]), rotation=90, va='center', fontsize=label_size - 3)
                    pass

                #if fig_ii % 3 == 2:
                #    #ax.set_zlabel(rf"$\alpha, m$ = {alpha}, {m}", rotation=90)
                #    ax.set_zlabel("")

                # keep zaxis on the right (https://www.py4u.net/discuss/13794)
                ax = fig.add_axes(MyAxes3D(ax, 'l'))

                # set lims
                pmax = 0.05
                #pmax = 0.06
                ax.set_xlim(-pmax, pmax); ax.set_ylim(-pmax, pmax)
                ax.set_zlim(-pmax, pmax)

                #ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')  # delete

                # tick labels
                #if fig_ii != 1:
                #    ax.set_zticklabels([]);
                ax.set_xticklabels([]); ax.set_yticklabels([])
                ax.set_zticklabels([])

                # inset plot
                borderpad = -.5
                ins = inset_axes(ax, width="30%", height="30%", borderpad=borderpad)
                ins.bar(list(range(1,6)),singular_values[ii, 0:5]/total_var, color='k')
                print(singular_values[ii, 0:5]/total_var)
                ins.axhline(y=0.5, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
                ins.set_yticks([0, 0.5, 1.0])

                if fig_ii != 1:
                    ins.set_yticklabels([])
                    #ins.set_xticklabels([])
                else:
                    ins.set_yticklabels([0, 0.5, 1.0])
                    #ins.set_xticklabels(list(range(1,6)))
                ins.set_xticklabels([])
                ins.set_xticks(list(range(1,6)))
                ins.set_ylim(0,1)

        # colorbar (in all plots)
        if not cbar_separate:       
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.20, 0.015, 0.75])
            cbar_ticks = [0, np.pi, 2*np.pi]
            cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
            cbar.ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'],labelsize=tick_size)
        

        # suptitle as alpha
        #fig.suptitle(rf"$\alpha$ = {alpha}", fontsize=label_size)
        if ensemble == 'None':
            plt.savefig(f"{plot_path}/proj3d_single_alpha={alpha}_layer={ii}.pdf", bbox_inches='tight')
        else:
            plt.savefig(f"{plot_path}/proj3d_single_alpha={alpha}_layer={ii}.pdf", bbox_inches='tight')
        print(f"alpha={alpha} done!")
        #plt.savefig(f"{plot_path}/proj3d_{alpha}.pdf")
        #plt.clf()
        plt.close()
        #quit()

    # separate colorbar
    if cbar_separate:    
        #fig.subplots_adjust(right=0.8)
        fig = plt.figure()
        cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])
        cbar_ticks = [0, np.pi, 2*np.pi]
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
        cbar.ax.tick_params(axis='y', labelsize=tick_size)
        
        plt.savefig(f"{plot_path}/proj3d_colorbar.pdf", bbox_inches='tight')   
        

    print(f"{time() - t0} s!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

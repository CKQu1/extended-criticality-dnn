import numpy as np
import os
import re
import sys
from os.path import join
from tqdm import tqdm

sys.path.append(os.getcwd())
from path_names import root_data
from utils_dnn import load_transition_lines

def SEM(N, L, N_thetas, alpha100, g100):
# get the SEM of the manifold distances propagated through the DNN layers
    # Extract numeric arguments.
    N = int(N)
    L = int(L)
    N_thetas = int(N_thetas)
    alpha = int(alpha100)/100.
    g = int(g100)/100.
    # Generate circular manifold.
    hs = np.zeros([N, N_thetas])
    thetas = np.linspace(0, 2*np.pi, N_thetas)
    hs[0,:] = np.cos(thetas)
    hs[1,:] = np.sin(thetas)
#    print(f"angles: {thetas}")
    SEMs = np.zeros([L, 2])  # distance SEMs, angular SEMs
    from scipy.stats import levy_stable
    for l in tqdm(range(L)):
        hs = np.dot(levy_stable.rvs(alpha, 0, size=[N,N], scale=g*(0.5/N)**(1./alpha)),
                    np.tanh(hs))
        dhs = np.diff(hs, axis=1)
        distances = np.linalg.norm(dhs, axis=0)
        SEMs[l, 0] = np.std(distances) / np.mean(distances)
        # note: the distances may become too large for angular dists to be meaningful
        angular_distances = distances / np.linalg.norm((hs[:,1:] + hs[:,:-1])/2, axis=0)
        SEMs[l, 1] = np.std(angular_distances) / np.mean(angular_distances)        
#        ds = angular_distances
#        print(np.mean(ds), np.std(ds), np.std(ds)/np.mean(ds))
    return SEMs

def SEM_save(N, L, N_thetas,
             alpha100, g100, rep,
             path=''):
    np.savetxt(path+f'SEMs_alpha_{alpha100}_g_{g100}_rep_{rep}.txt',
               SEM(N, L, N_thetas, alpha100, g100))

def submit(*args):
    from qsub import qsub, job_divider, project_ls
    pbs_array_data = [(alpha100, g100, rep)
                      for alpha100 in range(100, 201, 5)
                      for g100 in range(5, 301, 5)
                      for rep in range(50)
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "geometry_data"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

# ----- preplot -----

def SEM_preplot(alpha100, g100,
                path=''):
    # find number of reps in folder
    fnames = [fname for fname in os.listdir(path)
              if f'SEMs_alpha_{alpha100}_g_{g100}_rep_' in fname]
    L = np.loadtxt(path+fnames[0]).shape[0]
    N_reps = len(fnames)
    SEMs = np.zeros([N_reps, L, 2])
    for i, fname in tqdm(enumerate(fnames), total=len(fnames)):
        SEMs[i] = np.loadtxt(path+fname)
    mean_SEMs = np.mean(SEMs, axis=0)
    np.savetxt(path+f'SEMs_alpha_{alpha100}_g_{g100}.txt', mean_SEMs)
    return mean_SEMs

def submit_preplot(path):
    # find the `alpha100`s and `g100`s of the files in the folder
    pbs_array_data = set([tuple(re.findall('\d+', fname)[:2]) for fname in os.listdir(path)
                      if all(s in fname for s in ('alpha', 'g', 'rep', 'txt'))])
    from qsub import qsub, job_divider, project_ls

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        # SEM_preplot as arg
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "geometry_data"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

# ----- plot -----

# plot settings
global tick_size, label_size, axis_size, legend_size, linewidth
tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8
# colorbar
interp = "quadric"
#cm_type = 'RdGy'
#cm_type = 'coolwarm'
#cm_type = 'RdBu'
cm_type = 'RdYlBu'

# /project/phys_DL/dnn_project/python_random_dnn.py_SEM_save_1000_50_1000
# path = join(root_data, "/project/phys_DL/dnn_project", "python_random_dnn.py_SEM_save_1000_50_1000")
def SEM_plot(path='', layer=None, one_row=True):
    from ast import literal_eval

    #phase_path = f"/project/phys_DL/dnn_project"
    # plot settings
    one_row = literal_eval(one_row) if isinstance(one_row,str) else one_row
    fnames = [fname for fname in os.listdir(path)
              if 'rep' not in fname and fname.endswith('txt')]
    xs = []
    ys = []
    cs = []
    for fname in tqdm(fnames):
        alpha100, g100 = re.findall('\d+', fname)
        SEMs = np.loadtxt(path+fname)
        xs.append(int(alpha100)/100.)
        ys.append(int(g100)/100.)
        cs.append(SEMs[:,0])  # linear distances
    alphas100 = sorted(set(xs))
    gs100 = sorted(set(ys))
    cs = np.array(cs)
    L = cs.shape[1]
    import pandas as pd
    import matplotlib.pyplot as plt
    import string
    plt.rcParams["font.family"] = "serif"     # set plot font globally
    if layer and ',' in layer:  # publication figure
        # phase transition lines
        bound1, boundaries = load_transition_lines()
        
        layers = list(map(int, layer.split(',')))
        # 2 by 2
        if not one_row:
            ncols = int(len(layers)**0.5)
            nrows = int(np.ceil(len(layers)/ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     constrained_layout=True,
                                     sharex=True, sharey=True,
                                     figsize=(9.5,7.142))  
        # 1 by n
        else:
            if len(layers) == 4:
                nrows, ncols = 1, len(layers)  
                fig, axes = plt.subplots(nrows, ncols,
                                         constrained_layout=True,
                                         sharex=True, sharey=True,
                                         figsize=(9.5*2 - 2,7.142/2)) 
            elif len(layers) == 3:
                nrows, ncols = 1, len(layers)  
                fig, axes = plt.subplots(nrows, ncols,
                                         constrained_layout=True,
                                         sharex=True, sharey=True,
                                         figsize=(9.52/2*3 - 2.6,7.142/2 - 0.1)) 
                fig.tight_layout(pad=3.4)
  
        for i, l in enumerate(layers):
            ax = axes.flat[i]
            if one_row:
                #ax.tick_params(axis='both',labelsize=tick_size + 4)
                xticks = [1.0,1.2,1.4,1.6,1.8,2.0]
                yticks = np.arange(0,3.01,0.5)
                ax.set_xticks(xticks); ax.set_yticks(yticks)
                ax.set_xticklabels(xticks, fontsize=tick_size-1)
                ax.set_yticklabels(yticks, fontsize=tick_size-1)
            else:
                ax.tick_params(axis='both',labelsize=tick_size)     # tick label size

            # plot phase transition lines for each axs
            ax.plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')
            for j in range(0,len(boundaries)):
                bd = boundaries[j]
                ax.plot(bd.iloc[:,0], bd.iloc[:,1], 'k--')#'k-.')
            # plot x, y labels
            #if not i%ncols: ax.set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)
            #if i >= (nrows-1)*ncols: ax.set_xlabel(r'$\alpha$', fontsize=axis_size)
            #ax.text(-0.1 if not i%ncols else 0.1, 1, f'({string.ascii_lowercase[i]})', transform=ax.transAxes, fontsize=label_size, va='top', ha='right')    # fontweight='bold'
            #ax.text(0.1, 1.1, f'({string.ascii_lowercase[i]})', transform=ax.transAxes, fontsize=label_size, va='top', ha='right')
            # convert cs to a grid, assuming alphas and gs are evenly spaced
            mesh = np.zeros([len(alphas100), len(gs100)])
            for j in range(len(xs)):
                mesh[alphas100.index(xs[j]), gs100.index(ys[j])] = cs[j,l]
#            im = ax.scatter(xs, ys, c=cs[:,l],
            im = ax.imshow(mesh.T, interpolation=interp, aspect='auto', origin='lower', extent=(min(alphas100), max(alphas100), min(gs100), max(gs100)),
                        cmap=plt.cm.get_cmap(cm_type),
                        vmin=0, vmax=1
                        )
            if not one_row:
                ax.set_title(f'Layer {l}', fontsize=label_size)
        if not one_row:
            cbar = fig.colorbar(im, ax=axes, shrink=.6) 
        else:
            cbar = fig.colorbar(im, ax=axes, shrink=1., aspect=17, pad = 0.028)
        cbar.ax.tick_params(labelsize=tick_size-2)
        #plt.show()
        fig1_path = join(root_data, "figure_ms")
        if one_row:
            plt.savefig(f"{fig1_path}/random_dnn_1by{len(layers)}.pdf", bbox_inches='tight')
        else:        
            plt.savefig(f"{fig1_path}/random_dnn.pdf", bbox_inches='tight')

        return
    # debugging
    # phase boundaries
    bound1, boundaries = load_transition_lines()
    # fig
    plt.figure()
    plt.show(block=False)
    for l in tqdm([int(layer)] if layer else range(L)):
        plt.clf()
                # plot boundaries for each axs
        plt.plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')
        for j in range(len(boundaries)):
            bd = boundaries[j]
            plt.plot(bd.iloc[:,0], bd.iloc[:,1], 'k-.')
        # convert cs to a grid, assuming alphas and gs are evenly spaced
        mesh = np.zeros([len(alphas100), len(gs100)])
        for i in range(len(xs)):
            mesh[alphas100.index(xs[i]), gs100.index(ys[i])] = cs[i,l]
#        plt.scatter(xs, ys, c=cs[:,l],
        plt.imshow(mesh.T, interpolation=interp, aspect='auto', origin='lower', extent = [0.25, max(alphas100), min(gs100), max(gs100)],
                    #extent = [min(alphas100), max(alphas100), min(gs100), max(gs100)],
                    cmap=plt.cm.get_cmap(cm_type),
                    vmin=0, vmax=1
                    )
        plt.colorbar()
        plt.title(l)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(1./30)
    plt.show()
    return cs


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

import numpy as np
from tqdm import tqdm

import sys
import os
import re

from mpl_toolkits.mplot3d import axes3d

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

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

def layer_pca(hidden_stacked, **kwargs):
    import torch

    h_dim = hidden_stacked.shape    
    if 'n_pc' in kwargs:
        n_pc = kwargs.get('n_pc')
        #hs_proj = torch.zeros((h_dim[0], n_pc, h_dim[1]))
        hs_proj = torch.zeros((h_dim[0], h_dim[1], n_pc))
    else:
        hs_proj = torch.zeros_like(hidden_stacked)
        #hs_proj = torch.zeros((h_dim[0], h_dim[2], h_dim[1]))
    singular_values = torch.zeros((h_dim[0], min(h_dim[1], h_dim[2])))

    for l in tqdm(range(h_dim[0])):
        hidden = hidden_stacked[l,:].detach().numpy()
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
            hs_proj[l,:] = torch.tensor(u[:, :3])
        else:
            #norm_s = s[:] / s[:].max()
            #u = (norm_s[None, :]) * u[:, :] 
            #u = u / np.abs(u).max()
            hs_proj[l,:] = torch.tensor(u)

        singular_values[l,:] = torch.tensor(s)

    return hs_proj, singular_values

def gcircle_prop(N, L, N_thetas, alpha100, g100, *args):
    import torch
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
    hs_all = np.zeros([L + 1, N_thetas, N])  # distance SEMs, angular SEMs
    hs_all[0,:,:] = hs.T
    from scipy.stats import levy_stable
    for l in tqdm(range(L)):
        hs = np.dot(levy_stable.rvs(alpha, 0, size=[N,N], scale=g*(0.5/N)**(1./alpha)),
                    np.tanh(hs))
        hs_all[l + 1,:,:] = hs.T
    return torch.tensor(hs_all)   # convert to torch

def gcircle_save(N, L, N_thetas,
                 alpha100, g100, *args):
    import torch    
    path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    torch.save(gcircle_prop(N, L, N_thetas, alpha100, g100),
               f"{path}/gcircles_alpha_{alpha100}_g_{g100}")

def submit(*args):
    from qsub import qsub
    pbs_array_data = [(alpha100, g100)
                      #for alpha100 in range(100, 201, 5)
                      #for g100 in range(5, 301, 5)
                      for alpha100 in range(100, 201, 25)
                      for g100 in [1, 100, 200]
                      #for alpha100 in [100]
                      #for g100 in [1, 100]
                      ]
    #qsub(f'python geometry_analysis/greatcircle_proj_trial.py {sys.argv[0]} {" ".join(args)}',
    #qsub(f'python greatcircle_proj_trial.py {sys.argv[0]} {" ".join(args)}',
    qsub(f'python {sys.argv[0]} {" ".join(args)}',    
         pbs_array_data, 
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
         P='phys_DL')

# ----- preplot -----

def gcircle_preplot(alpha100, g100, *args):
    import torch    

    path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}') 

    hs_all = torch.load(f"{path}/gcircles_alpha_{alpha100}_g_{g100}")
    hs_proj, singular_values = layer_pca(hs_all, n_pc=3)

    torch.save(hs_proj, f"{path}/eigvecs_alpha_{alpha100}_g_{g100}")
    torch.save(singular_values, f"{path}/singvals_alpha_{alpha100}_g_{g100}")

    print("Preplot done!")

def submit_preplot(*args):
#def submit_preplot(path):
    data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    # find the `alpha100`s and `g100`s of the files in the folder
    pbs_array_data = set([tuple(re.findall('\d+', fname)[:2]) for fname in os.listdir(data_path)
                      if all(s in fname for s in ('alpha', 'g', 'gcircles'))])

    print(pbs_array_data)

    from qsub import qsub
    qsub(f'python {sys.argv[0]} gcircle_preplot', pbs_array_data, 
         path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/',
         P='phys_DL')

# ----- plot -----

def gcircle_plot(*args):

    #alpha100_ls, g100_ls = [200], [1]
    alpha100_ls, g100_ls = [100,150,200], [1,100,200]

    from time import time
    t0 = time()

    import torch

    from matplotlib import pyplot as plt
    from matplotlib.pyplot import figure
    from matplotlib.gridspec import GridSpec
    from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar

    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.cm import coolwarm
    # colorbar
    cm = cm.get_cmap('plasma')

    tick_size = 18.5
    label_size = 18.5
    axis_size = 18.5
    legend_size = 14
    linewidth = 0.8
    text_size = 14

    data_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/gcircle3d_data"
    plot_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
   
    alpha_mult_pair = []
    for alpha100 in alpha100_ls:
        for g100 in g100_ls:
            alpha_mult_pair.append((alpha100,g100))  
 
    # Font size
    tick_size = 16.5
    label_size = 16.5
    axis_size = 16.5
    legend_size = 14
    linewidth = 0.8

    # Set up figure
    rows = 3
    cols = 3

    layer_ii = [5,15,35]
    #layer_ii = [1,10,30]
    #layer_ii = [5,10,20]
    #layer_ii = [0,20,40]
    #assert max(layer_ii) <= L, "layer_ii incorrect"

    # Projection plot
    ls_all = [list(range(3)), list(range(3,6)), list(range(6,9))]
    #ls_all = [list(range(3))]
    #ls_all = [list(range(1))]
    #ls_all = [list(range(3,6))]

    vert_dists = [0.78, 0.5, 0.22]
    for ls in ls_all:
        fig = plt.figure(figsize=(9.5,7.142))

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
        cmap_bd = [-0.05,0.05]

        for f_ii in range(len(ls)):
            folder_ii = ls[f_ii]
            alpha100, g100 = alpha_mult_pair[folder_ii]
            print((alpha100,g100))
            alpha, m = alpha100/100, g100/100
            hs_proj = torch.load(f"{data_path}/eigvecs_alpha_{alpha100}_g_{g100}")
            singular_values = torch.load(f"{data_path}/singvals_alpha_{alpha100}_g_{g100}")

            for k in range(len(layer_ii)):
                # figure index
                fig_ii = 3*f_ii + k + 1
                print(fig_ii)

                ax = fig.add_subplot(rows,cols,fig_ii,projection='3d')
                #ax = fig.add_subplot(3,3,fig_ii)
                ii = layer_ii[k]
                total_var = sum(singular_values[ii,:])

                x, y ,z = hs_proj[ii, :, 0], hs_proj[ii, :, 1],hs_proj[ii, :, 2]

                im = ax.scatter(x , y , z, c=z, vmin=cmap_bd[0], vmax=cmap_bd[1], marker='.', s=4, cmap=cm)
                ax.plot(x , y , z,color='k', zorder=0, linewidth=0.5, alpha=.4)
                ax.zaxis.set_rotate_label(False) 

                if f_ii == 0:
                    ax.set_title(f"Layer {ii}", loc='left', fontsize=label_size - 2)

                if fig_ii % 3 == 1:
                    #ax.set_zlabel(rf"$\alpha, m$ = {alpha}, {m}", rotation=90)
                    #ax.set_zlabel(r"$D_w^{1/\alpha}$" + f" = {m/100}", rotation=90, fontsize=label_size)
                    fig.text(0.07, vert_dists[f_ii], r"$D_w^{1/\alpha}$" + f" = {m}", rotation=90, va='center', fontsize=label_size - 2)

                #if fig_ii % 3 == 2:
                #    #ax.set_zlabel(rf"$\alpha, m$ = {alpha}, {m}", rotation=90)
                #    ax.set_zlabel("")

                # keep zaxis on the right (https://www.py4u.net/discuss/13794)
                ax = fig.add_axes(MyAxes3D(ax, 'l'))

                # set lims
                pmax = 0.05
                ax.set_xlim(-pmax, pmax); ax.set_ylim(-pmax, pmax);
                ax.set_zlim(-pmax, pmax);

                if fig_ii != 1:
                    ax.set_zticklabels([]);
                ax.set_xticklabels([]); ax.set_yticklabels([])

                # inset plot
                borderpad = -.5
                ins = inset_axes(ax, width="30%", height="30%", borderpad=borderpad)
                ins.bar(list(range(1,6)),singular_values[ii, 0:5]/total_var, color='k')
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

        # colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        # suptitle as alpha
        fig.suptitle(rf"$\alpha$ = {alpha}", fontsize=label_size)

        plt.savefig(f"{plot_path}/proj3d_alpha={alpha}.pdf", bbox_inches='tight')
        print(f"alpha={alpha} done!")
        #plt.savefig(f"{plot_path}/proj3d_{alpha}.pdf")
        plt.clf()
        #quit()

    print(f"{time() - t0} s!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"     # set plot font globally
sns.set(font_scale=1.3, rc={"lines.linewidth": 2.5})
sns.set_style('ticks')

import os
import sys
from time import time
from os.path import join
lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
import path_names
from path_names import log_path, cnn_path

# load manifold
"""
net_type = "resnet50"
manifolds_load = np.load(join(log_path, f"manifold/embeddings_new/macaque/{net_type}/manifolds.npy"),
                   allow_pickle=True)
"""

def pve_plot(net_type, init_alpha100, init_g100, root_path, fname, plot_type="snr", force_replot=True):

    

    assert plot_type == "snr" or plot_type == "err", "plot_type does not exist!"

    if "PDLAI" in root_path:
        matches = [net[0] for net in os.walk(root_path) if f"fc10_mnist_tanh_id_stable{init_alpha100}_{init_g100}_epoch650_" in net[0]]
    else:
        if "alexnets_nomomentum" in root_path:
            matches = [net[0] for net in os.walk(root_path) if f"alexnet_{init_alpha100}_{init_g100}_" in net[0] and "epochs=100" in net[0]]
        elif "resnets" in root_path:
            matches = [net[0] for net in os.walk(root_path) if f"{net_type}_{init_alpha100}_{init_g100}_" in net[0] and "epochs=650" in net[0]]
    init_path = matches[0]

    # ----- Universal variables -----
    #navg = 100
    navg = 500
    #navg = 10
    data_path = join(init_path, 'proto_vs_NN_experiments', f'navg={navg}_{fname}')

    #ms = np.arange(2,100,4)
    #ms = np.arange(2,50,1)
    #ks = np.arange(1,100,2)

    ms = np.arange(2,50,2)
    #ks = np.arange(1,100,4)
    ks = np.arange(1,100,2)

    #P = 500
    P = 50
    m = 5

    if plot_type == "snr":
        isExist = True
        for file_name in ["NNmeans.npy", "NNstds.npy", "proto_means.npy", "proto_stds.npy"]:
            if not os.path.exists( join(data_path, file_name) ):
                isExist = False
                break    

    elif plot_type == "err":
        isExist = True
        for file_name in ["NNerrs.npy", "proto_errs.npy"]:
            if not os.path.exists( join(data_path, file_name) ):
                isExist = False
                break   

    if isExist and not force_replot:
        if plot_type == "snr":
            means = np.load(join(data_path, 'NNmeans.npy'))
            stds = np.load(join(data_path, 'NNstds.npy'))
            means_proto = np.load(join(data_path, "proto_means.npy"))
            stds_proto = np.load(join(data_path, "proto_stds.npy"))

        elif plot_type == "err":
            errs_NN = np.load(join(data_path, "NNerrs.npy"))
            errs_proto = np.load(join(data_path, "proto_errs.npy"))

    else:
        #fname = "bs=10_K=64_P=50_N=2048"
        manifolds_load = np.load(join(init_path, f"manifold/manifolds_{fname}.npy"),
                           allow_pickle=True)


        manifolds = []
        for manifold in manifolds_load:
            manifolds.append(manifold[:P])
        manifolds = np.stack(manifolds)

        #manifolds = manifolds[:100]

        manifolds = manifolds[:20]

        means = []
        stds = []
        means_proto = []
        stds_proto = []
        errs_proto = []
        errs_NN = []
        for _ in tqdm(range(navg)):
            #a,b = np.random.choice(100,2,replace=False)
            a,b = np.random.choice(len(manifolds),2,replace=False)
            Xa = manifolds[a]
            Xb = manifolds[b]
            
            Ua,Sa,Va = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
            Ub,Sb,Vb = np.linalg.svd(Xb - Xb.mean(0), full_matrices=False)
            for k in ks:
                Xak = Ua[:,:k]*Sa[:k]@Va[:k] * (Sa**2).sum()/(Sa[:k]**2).sum() + Xa.mean(0)
                Xbk = Ub[:,:k]*Sb[:k]@Vb[:k] * (Sb**2).sum()/(Sb[:k]**2).sum() + Xb.mean(0)
                for m in ms:
                    perma = np.random.permutation(P)
                    permb = np.random.permutation(P)
                    Xatrain,Xatest = np.split(Xak[perma],(m,))
                    Xbtrain,_ = np.split(Xbk[permb],(m,))

                    da = ((Xatrain[:,None] - Xatest[None])**2).sum(-1).min(0)
                    db = ((Xbtrain[:,None] - Xatest[None])**2).sum(-1).min(0)
                    h = -da + db
                    errs_NN.append((h<0).mean())

                    means.append(h.mean())
                    stds.append(h.std())
                    
                    da = ((Xatrain.mean(0) - Xatest)**2).sum(-1)
                    db = ((Xbtrain.mean(0) - Xatest)**2).sum(-1)
                    hproto = -da + db
                    errs_proto.append((hproto<0).mean())
                    means_proto.append(hproto.mean())
                    stds_proto.append(hproto.std())
        means = np.stack(means).reshape(navg,len(ks),len(ms)).mean(0)
        stds = np.stack(stds).reshape(navg,len(ks),len(ms)).mean(0)
        means_proto = np.stack(means_proto).reshape(navg,len(ks),len(ms)).mean(0)
        stds_proto = np.stack(stds_proto).reshape(navg,len(ks),len(ms)).mean(0)
        errs_NN = np.stack(errs_NN).reshape(navg,len(ks),len(ms)).mean(0)
        errs_proto = np.stack(errs_proto).reshape(navg,len(ks),len(ms)).mean(0)

        #data_path = join('proto_vs_NN_experiments', net_type)
        data_path = join(init_path, 'proto_vs_NN_experiments', f'navg={navg}_{fname}')
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        np.save(join(data_path, 'NNmeans.npy'), means)
        np.save(join(data_path, 'NNstds.npy'), stds)
        np.save(join(data_path, "proto_means.npy"), means_proto)
        np.save(join(data_path, "proto_stds.npy"), stds_proto)
        np.save(join(data_path, "NNerrs.npy"), errs_NN)
        np.save(join(data_path, "proto_errs.npy"), errs_proto)

    SNR = means/stds
    SNR_proto = means_proto/stds_proto

    sns.set_style('white')
    
    if plot_type == "err":
        diff = errs_proto - errs_NN
    elif plot_type == "snr":
        diff = SNR_proto - SNR
    difference = diff.flatten()
    print((np.min(difference), np.max(difference)))

    """
    if len(difference[difference < 0]) == 0:
        bd_l = 0
    else:
        bd_l = np.abs(np.max(difference[difference < 0]))
    if len(difference[difference > 0]) == 0:
        bd_u = 0
    else:
        bd_u = np.min(difference[difference > 0])

    print((bd_l,bd_u))
    bd = min( np.abs([bd_l, bd_u]) )
    cmap_bd = [-bd,bd]
    """
    bd = min( np.abs([np.min(difference), np.max(difference)]) )
    cmap_bd = [-bd,bd]
    print(cmap_bd)

    plt.imshow(diff,
               origin='lower', extent=(np.min(ms),np.max(ms),np.min(ks),np.max(ks)),
               cmap='coolwarm', aspect='auto', vmin=cmap_bd[0],vmax=cmap_bd[1])
    plt.plot(ms, 15*np.log(ms), c='black', linestyle='dashed')
    plt.colorbar()

    plt.tight_layout()
    fig_name = f"{net_type}_{init_alpha100}_{init_g100}_navg={navg}_{fname}_{plot_type}_pro-vs-ex.pdf"
    plt.savefig(join(data_path,fig_name), bbox_inches='tight')
    #plt.show()
    print("Figure saved!")


# N is the total number of projects
def job_divider(pbs_array: list, N: int):
    total_jobs = len(pbs_array)
    ncores = min(total_jobs//2, N)
    pbss = []
    delta = int(np.floor(total_jobs/ncores))
    for idx in range(ncores):
        if idx != ncores - 1:
            pbss.append( pbs_array[idx*delta:(idx+1)*delta] )
        else:
            pbss.append( pbs_array[idx*delta::] )    
    perm = list(np.random.choice(N,ncores,replace=False))
    assert len(perm) == len(pbss), "perm length and pbss length not equal!"

    return perm, pbss

def plot_submit(*args):
    from qsub import qsub
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]
    
    # MLPs
    #alpha100_ls = [1.0,1.5,2.0]
    #g100_ls = [0.25,1.0,3.0]
    #alpha100_ls = np.arange(1.0,2.01,0.1)
    
    alpha100_ls = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    g100_ls = np.arange(0.25,3.01,0.25)
    g100_ls[-1] = 3.0
    net_type = "fc10"
    root_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD/fcn_grid/fc10_grid"
    
    #fname = "name=omniglot_bs=10_K=64_P=50_N=1024"    
    #fname = "name=omniglot_ep=650_bs=10_K=64_P=50_N=1024"
    #fname = "name=omniglot=test_ep=650_bs=10_K=128_P=50_N=784"
    #fname = "omniglot=test_ep=650_bs=10_K=128_P=50_N=784"
    fname = "omniglot=test_ep=0_bs=10_K=128_P=50_N=784"

    # -----------
    #alpha100_ls = [100,200]
    #g100_ls = [100]
    #alpha100_ls = list(range(100,201,10))
    #g100_ls = list(range(25, 301, 25))

    # AlexNets
    """
    net_type = "alexnet"    
    root_path = "/project/dyson/dyson_dl/alexnets_nomomentum"
    fname = "bs=10_K=64_P=50_N=2048"
    """

    # ResNets
    """
    net_type = "resnet50_HT"
    net_type = "resnet34_HT"
    root_path = "/project/dyson/dyson_dl/resnets"
    fname = "bs=10_K=64_P=50_N=2048"
    """

    pbs_array_data = [(net_type, alpha100, g100, root_path, fname)
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path='/project/dyson/dyson_dl',
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='1GB') 


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])



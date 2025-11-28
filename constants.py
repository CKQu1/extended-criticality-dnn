import matplotlib as mpl
import os
import pandas as pd
from UTILS.mutils import njoin
from matplotlib.cm import get_cmap

# ----- GENERAL -----
RT = os.path.abspath(os.getcwd())
DROOT = njoin(RT, '.droot')
# root_data = "/project/PDLAI/project2_data"
# root_data = "/project/phys_DL/extended-criticality-dnn/.droot"
root_data = DROOT
FIGS_DIR = njoin(DROOT, 'figs_dir')
SCRIPT_DIR = njoin(DROOT, 'submitted_scripts')

#CLUSTER = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS' if 'taiji1' in DROOT else 'FUDAN_BRAIN'
if 'uu69' in DROOT:
    CLUSTER = 'GADI' 
elif 'taiji1' in DROOT:
    CLUSTER = 'PHYSICS'
else:
    CLUSTER = None
# -------------------

# ----- RESOURCES -----
RESOURCE_CONFIGS = {
    "GADI": {
        True:  {"q": "gpuvolta", "ngpus": 1, "ncpus": 12},
        False: {"q": "normal",   "ngpus": 0, "ncpus": 1},
    },
    "PHYSICS": {
        True:  {"q": "l40s", "ngpus": 1, "ncpus": 1},
        False: {"q": "taiji", "ngpus": 0, "ncpus": 1},
    },
}
# -------------------

# ----- GADI -----
GADI_PROJECTS = ['uu69']
GADI_SOURCE = '/scratch/uu69/cq5024/myenvs/fsa/bin/activate'
# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
# PHYSICS_CONDA = 'frac_attn' if 'chqu7424' in RT else '~/conda'
PHYSICS_CONDA = '/taiji1/chqu7424/myenvs/pydl'
# -------------------

# ----- FUDAN-BRAIN -----
FUDAN_CONDA = 'frac_attn'
# -------------------

# ----- ARTEMIS -----
ARTEMIS_PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']
BPATH = njoin('/project')  # path for binding to singularity container
#SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path
SPATH = njoin('/project/frac_attn/built_containers/pydl.img')
# -------------------


# color for model hyperparameters
#HYP_CM = 'gist_ncar'
HYP_CM = 'turbo'
HYP_CMAP = get_cmap(HYP_CM)
HYP_CNORM = mpl.colors.Normalize(vmin=1, vmax=2)

# ---------- create logs for networks during training ----------

# Record relevants attributes for trained neural networks -------------------------------------------------------

def log_model(log_path, model_path, file_name="net_log", local_log=True, **kwargs):    
    fi = f"{log_path}/{file_name}.csv"
    #df = pd.DataFrame(columns = kwargs)
    df = pd.DataFrame(columns = kwargs, dtype=object)
    df.loc[0,:] = list(kwargs.values())
    if local_log:
        df.to_csv(f"{model_path}/log", index=False)
    if os.path.isfile(fi):
        df_og = pd.read_csv(fi)
        # outer join
        df = pd.concat([df_og,df], axis=0, ignore_index=True)
    else:
        if not os.path.isdir(f"{log_path}"): os.makedirs(log_path)
    #df.to_csv(fi, index=False)
    print('Log saved!')

def read_log():    
    fi = njoin(root_data, "net_log.csv")
    if os.path.isfile(fi):
        df_og = pd.read_csv(fi)
        print(df_og)
    else:
        raise ValueError("Network logbook has not been created yet, please train a network.")

# ---------- model_id extraction and conversion ----------

# return network info with the model_id
def model_log(model_id):    # tick
    log_book = pd.read_csv(f"{root_data}/net_log.csv")
    #if model_id in log_book['model_id'].item():
    if model_id in list(log_book['model_id']):
        model_info = log_book.loc[log_book['model_id']==model_id]        
    else:
        #if f"{id_to_path(model_id)}"
        print("Update model_log() function in path_names.py!")
        model_info = None
    #model_info = log_book.iloc[0,log_book['model_id']==model_id]
    return model_info

# return network path from model_id
def id_to_path(model_id, path):   # tick
    for subfolder in os.walk(path):
        if model_id in subfolder[0]:
            model_path = subfolder[0]
            break
    assert 'model_path' in locals(), "model_id does not exist!"

    return model_path

# ---------- For plotting the accuracy phase transitions of CNNs ----------

# transfer full path to model_id
def get_model_id(full_path):
    str_ls = full_path.split('/')[-1].split('_')
    for s in str_ls:
        if len(s) == 36:
            return s    

# get (alpha,g) when the pair is not saved
def get_alpha_g(full_path):
    str_ls = full_path.split('/')
    str_alpha_g = str_ls[-1].split("_")
    if str_alpha_g[2].isnumeric() and str_alpha_g[3].isnumeric():
        return (int(str_alpha_g[2]), int(str_alpha_g[3]))
    else:
        return (int(str_alpha_g[1]), int(str_alpha_g[2]))

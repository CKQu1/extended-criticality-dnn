import pandas as pd

root_data = "/project/PDLAI/project2_data"

# ---------- create logs for networks during training ----------

# Record relevants attributes for trained neural networks -------------------------------------------------------

def log_model(log_path, model_path, file_name="net_log", local_log=True, **kwargs):    
    fi = f"{log_path}/{file_name}.csv"
    df = pd.DataFrame(columns = kwargs)
    df.loc[0,:] = list(kwargs.values())
    if local_log:
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
    fi = join(root_data, "net_log.csv")
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

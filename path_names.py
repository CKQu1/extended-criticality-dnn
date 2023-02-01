import pandas as pd

root_data = "/project/PDLAI/project2_data"

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

def id_to_path(model_id, path):   # tick
    for subfolder in os.walk(path):
        if model_id in subfolder[0]:
            model_path = subfolder[0]
            break
    assert 'model_path' in locals(), "model_id does not exist!"

    return model_path

# ------------

# For plotting the accuracy phase transitions of CNNs

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

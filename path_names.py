root_data = "/project/PDLAI/project2_data"

def model_log(model_id):    # tick
    log_book = pd.read_csv(f"{root_data}/net_log.csv")
    #if model_id in log_book['model_id'].item():
    if model_id in list(log_book['model_id']):
        model_info = log_book.loc[log_book['model_id']==model_id]        
    else:
        #if f"{id_to_path(model_id)}"
        print("Update model_log() function in tamsd_analysis/tamsd.py!")
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

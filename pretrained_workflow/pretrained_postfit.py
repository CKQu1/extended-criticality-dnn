import numpy as np
import math
import os
import pandas as pd
import sys
from os.path import join
from tqdm import tqdm

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
from path_names import root_data

def compute_aic(N, k, LL):
    return -2 * LL + 2 * k

def compute_bic(N, k, LL):
    return -2 * LL + math.log(N) * k

def postfit_stats(main_path, pytorch=True):
    global df, fitfiles, subdirs, df_summary, total_nn, total_wm, df_error
    global IC_mins, AIC_dict, BIC_dict

    subdirs = [join(main_path, x) for x in next(os.walk(main_path))[1]]
    total_nn = len(subdirs)
    print(f"Total networks: {total_nn}")
    total_wm = 0
    stable_aics = []
    stable_bics = []
    normal_aics = []
    normal_bics = []
    student_aics = []
    student_bics = []
    lognorm_aics = []
    lognorm_bics = []
    df_error = None
    for sidx, subdir in tqdm(enumerate(subdirs)):
        fitfiles = [join(subdir,fitfile) for fitfile in os.listdir(subdir) if fitfile[-4:]==".csv" ]        
        for fitfile in fitfiles:
            df = pd.read_csv(fitfile)
            fit_size = df.loc[0,'fit_size']
            # stable
            stable_aic =  compute_aic(fit_size, 4, df.loc[0,'logl_stable']) 
            stable_bic = compute_bic(fit_size, 4, df.loc[0,'logl_stable']) 
            # normal
            normal_aic = compute_aic(fit_size, 2, df.loc[0,'logl_norm']) 
            normal_bic = compute_bic(fit_size, 2, df.loc[0,'logl_norm']) 
            # student-t
            student_aic = compute_aic(fit_size, 3, df.loc[0,'logl_t']) 
            student_bic = compute_bic(fit_size, 3, df.loc[0,'logl_t']) 
            # log-normal
            lognorm_aic = compute_aic(fit_size, 3, df.loc[0,'logl_lognorm']) 
            lognorm_bic = compute_bic(fit_size, 3, df.loc[0,'logl_lognorm'])
            is_not_nan = (np.isnan(stable_aic) or np.isnan(stable_bic) or np.isnan(normal_aic) or np.isnan(normal_bic) or np.isnan(student_aic) or np.isnan(student_bic) or np.isnan(lognorm_aic) or np.isnan(lognorm_bic))
            is_not_nan = not is_not_nan
            if is_not_nan:
                stable_aics.append(stable_aic)
                stable_bics.append(stable_bic)
                normal_aics.append(normal_aic)
                normal_bics.append(normal_bic)                
                student_aics.append(student_aic)
                student_bics.append(student_bic)
                lognorm_aics.append(lognorm_aic)
                lognorm_bics.append(lognorm_bic)  

                total_wm += 1            
            else:
                if (not isinstance(df_error, pd.DataFrame)) and "alexnet" in  subdir:
                    df_error = df
                    print(fitfile)



    print(f"Total weight matrices: {total_wm}")
    data = {'stable_aic':stable_aics,
            'stable_bic':stable_bics, 
            'normal_aic':normal_aics,
            'normal_bic':normal_bics, 
            'student_aic':student_aics, 
            'student_bic':student_bics, 
            'lognorm_aic':lognorm_aics, 
            'lognorm_bic':lognorm_bics
            }
    df_summary = pd.DataFrame(data)
    #df_summary = df_summary.astype('object')

    # summarize stats
    #IC_mins = [0]*len(df_summary.columns)
    AICs = df_summary.iloc[:,[0,2,4,6]]
    BICs = df_summary.iloc[:,[1,3,5,7]]
    AIC_mins = AICs.idxmin(axis=1)
    BIC_mins = BICs.idxmin(axis=1)
    AIC_dict =  {}
    BIC_dict = {}
    for colname in AICs.columns:
        AIC_dict[colname] = (AIC_mins==colname).sum()/total_wm
    for colname in BICs.columns:
        BIC_dict[colname] = (BIC_mins==colname).sum()/total_wm

    print("AIC")
    print(AIC_dict)
    print('\n')
    print("BIC")
    print(BIC_dict)

    # save the summary


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])

import argparse
from constants import DROOT, CLUSTER
from UTILS.mutils import njoin, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs

from batch_exps import *  # experiments


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='batch_submit_main.py args')   
    parser.add_argument('--is_qsub', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--exp', default='exp1', type=str) 
    args = parser.parse_args()

    IS_ARGS_PARSER = False

    batch_script_name = 'batch_main.py'

    is_qsub = args.is_qsub
    exp_type = args.exp
    if exp_type == 'exp1':                           # train full-sized models (R^d)
        EXPS_TO_RUN = train_mlps(); EXP_NAME = 'Training MLPs on the phase transition diagram.'
    elif exp_type == 'exp2':
        EXPS_TO_RUN = train_cnns(); EXP_NAME = 'Training Vanilla CNNs on the phase transition diagram.'
        ARGS_ORDER = ['alpha100', 'g100', 'seed', 'depth', 'c_size', 'k_size',
                      'epochs', 'root_path']

    print('-----------------------')
    print(f'{exp_type}: {EXP_NAME}')
    print('----------------------- \n')
    
    kwargss_all, script_name, script_func, q, ncpus, ngpus, select, walltime, mem, job_path, nstack = EXPS_TO_RUN

    # ----- submit jobs -----
    print(f'Total jobs: {len(kwargss_all)} \n')      

    batch_kwargss_all = []
    kwargsss = [kwargss_all[i:i+nstack] for i in range(0, len(kwargss_all), nstack)]
    for kwargss in kwargsss:
        arg_strss = ''
        if IS_ARGS_PARSER:
            for kwargs in kwargss:
                arg_strss += ",".join("=".join((str(k),str(v))) for k,v in kwargs.items()) + ';'
        else:
            for kwargs in kwargss:
                arg_strss += ",".join([str(kwargs[kwarg_name]) for kwarg_name in ARGS_ORDER]) + ';'
        batch_kwargss_all.append({'arg_strss': arg_strss[:-1], 'script': script_name, 'script_func': script_func})

    print(f'Batched Total jobs: {len(batch_kwargss_all)} \n')

    commands, batch_script_names, pbs_array_trues, kwargs_qsubs =\
            job_setup(batch_script_name, batch_kwargss_all,
                    is_args_parser=IS_ARGS_PARSER,  
                    q=q,
                    ncpus=ncpus,
                    ngpus=ngpus,
                    select=select, 
                    walltime=walltime,
                    mem=mem,                    
                    job_path=job_path,
                    nstack=nstack,
                    cluster=CLUSTER)
    
    if is_qsub:
        print(f'----- SUBMITTING ----- \n')
        for i in range(len(commands)):
            if 'tf_train' in script_name:
                kwargs_qsubs[i]['conda'] = '/taiji1/chqu7424/myenvs/tf_v2' 
            else:
                kwargs_qsubs[i]['conda'] = '/taiji1/chqu7424/myenvs/pydl'
            qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])  


# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) < 2:
#         print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
#         quit()
#     result = globals()[sys.argv[1]](*sys.argv[2:])
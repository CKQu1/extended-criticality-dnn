import numpy as np
import sys
import os
import random
from datetime import datetime, timedelta
from os.path import isdir, isfile
from constants import *
from UTILS.mutils import njoin

def qsub(command, pbs_array_data, **kwargs):
    global pbs_array_data_chunks

    cluster = kwargs.get('cluster')
    nstack = kwargs.get('nstack', 1)

    if 'path' in kwargs:
        path = kwargs['path']
        if path and path[-1] != os.sep: path += os.sep
    else:
        path = command.replace(' ', '_') + os.sep
    if 'additional_command' in kwargs:
        additional_command = kwargs['additional_command']
    else:
        additional_command = ''
    if kwargs.get('pass_path', False):
        post_command = path if kwargs['pass_path'] == True else kwargs['pass_path']
    else:
        post_command = ''
    # Create output folder.
    date_str = datetime.today().strftime('%Y-%m-%d')
    job_dir = f"{date_str}_out"
    if not isdir(njoin(path,job_dir)): os.makedirs(njoin(path,job_dir))
    # source virtualenv
    if 'source' in kwargs:
        #assert os.path.isfile(kwargs.get('source')), "source for virtualenv incorrect"
        source_exists = 'true'
        source_activate = f"source {kwargs.get('source')}"
    else:
        source_activate = ''
    # conda activate  
    if 'conda' in kwargs:  
        conda_exists = 'true' if 'conda' in kwargs else 'false'
        conda_activate = f"conda activate {kwargs.get('conda')}"
    else:
        conda_activate = ''
    if kwargs.get('local', False):  # Run the subjobs in the current process.
        for pbs_array_args in pbs_array_data:
            str_pbs_array_args = ' '.njoin(map(str, pbs_array_args))
            os.system(f"""bash <<'END'
                cd {kwargs.get('cd', '.')}
                echo "pbs_array_args = {str_pbs_array_args}"
                {command} {str_pbs_array_args} {post_command}
END""")
        return    

    # Distribute subjobs evenly across array chunks.
    pbs_array_data = random.sample(pbs_array_data, len(pbs_array_data))
    # Submit array job.
    print(f"Submitting {len(pbs_array_data)} subjobs")
    
    # ---------- begin{GADI} ----------
    if cluster == 'GADI':
        #MAX_SUBJOBS = 1000
        MAX_SUBJOBS = 1
        # PBS array jobs are limited to 1000 subjobs by default
        pbs_array_data_chunks = [pbs_array_data[x:x+MAX_SUBJOBS]
                                for x in range(0, len(pbs_array_data), MAX_SUBJOBS)]
        # if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
        #     pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
        #args=($(python3 -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{MAX_SUBJOBS*i}])))" $PBS_ARRAY_INDEX))
        for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):

            #print(f'i = {i}')  # delete
            PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
#PBS -P {kwargs.get('P',"''")}
#PBS -q {kwargs.get('q','normal')}
#PBS -o {path}{job_dir} -e {path}{job_dir}
#PBS -l ncpus={kwargs.get('ncpus',1)}
#PBS -l mem={kwargs.get('mem','1GB')}
#PBS -l jobfs={kwargs.get('mem','1GB')}
{'#PBS -l ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
#PBS -l storage=gdata/{kwargs.get('P',"''")}+scratch/{kwargs.get('P',"''")}
#PBS -l walltime={kwargs.get('walltime','23:59:00')}
#PBS -l wd
##PBS -J {MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_chunk)-1}
# module purge
# module load git/2.39.2 intel-mkl/2023.2.0 pbs python3/3.9.2
# module list
args=($(python3 -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[0])))"))
cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
echo "pbs_array_args = ${{args[*]}}"
{source_activate}
{conda_activate}
{command} ${{args[*]}} {additional_command} {post_command}
END"""        

            os.system(f'qsub {PBS_SCRIPT}')
            # print(PBS_SCRIPT)

    # ---------- end{GADI} ----------

    # ---------- begin{PHYSICSX} ---------- for bash shell
    elif cluster == 'PHYSICS':
        MAX_SUBJOBS = 1000
        # PBS array jobs are limited to 1000 subjobs by default
        pbs_array_data_chunks = [pbs_array_data[x:x+MAX_SUBJOBS]
                                for x in range(0, len(pbs_array_data), MAX_SUBJOBS)]
        if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
            pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
        for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):

            # https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
            PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
##PBS -q {kwargs.get('q','defaultQ')}
#PBS -q {kwargs.get('q','taiji')}
#PBS -V
#PBS -m n
##PBS -o {path} -e {path}
#PBS -o {path}/{job_dir} -e {path}/{job_dir}
#PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
#PBS -l walltime={kwargs.get('walltime','23:59:00')}                       
#PBS -J {MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_chunk)-1}
args=($(python3 -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{MAX_SUBJOBS*i}])))" $PBS_ARRAY_INDEX))
cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
echo "pbs_array_args = ${{args[*]}}"    
#export CONDA_PKGS_DIRS=~/.conda/pkgs
{source_activate}
{conda_activate}                                                     
{command} ${{args[*]}} {additional_command} {post_command}
exit
END"""  

            os.system(f'qsub {PBS_SCRIPT}')
            #print(PBS_SCRIPT)

    # ---------- end{PHYSICSX} ----------

    # ---------- begin{FUDAN_BRAIN} ---------- for bash shell
    elif cluster == 'FUDAN_BRAIN':
        MAX_SUBJOBS = 1000
        # PBS array jobs are limited to 1000 subjobs by default
        pbs_array_data_chunks = [pbs_array_data[x:x+MAX_SUBJOBS]
                                for x in range(0, len(pbs_array_data), MAX_SUBJOBS)]
        if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
            pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
        for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):

            # https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
            SLURM_SCRIPT = f"""<<'END'
#!/bin/bash
#SBATCH --job-name={kwargs.get('N', sys.argv[0] or 'job')}
#SBATCH -p {kwargs.get('q','DCU')}
##SBATCH -o {path}/%j.out -e {path}/%j.err
#SBATCH -o {path}/{job_dir}/%j.out -e {path}/{job_dir}/%j.err
#SBATCH -N {kwargs.get('select',1)}
#SBATCH -n {kwargs.get('ncpus',1)}
#SBATCH --mem={kwargs.get('mem','1GB')}
#SBATCH --gres={'gpu:'+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
#SBATCH -t {kwargs.get('walltime','23:59:00')}                       
#SBATCH --array={MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_chunk)-1}

#SBATCH
args=($(python3 -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{MAX_SUBJOBS*i}])))" $SLURM_ARRAY_TASK_ID))

cd {kwargs.get('cd', '$SLURM_SUBMIT_DIR')}
echo "pbs_array_args = ${{args[*]}}"
       
{source_activate}
{conda_activate}                                                       
{command} ${{args[*]}} {additional_command} {post_command}
exit
END"""

            os.system(f'sbatch {SLURM_SCRIPT}')
            #print(SLURM_SCRIPT)

    # ---------- end{FUDAN_BRAIN} ----------

    # ---------- begin{ARTEMIS} ----------
    if cluster == 'ARTEMIS':
        MAX_SUBJOBS = 1000
        # PBS array jobs are limited to 1000 subjobs by default
        pbs_array_data_chunks = [pbs_array_data[x:x+MAX_SUBJOBS]
                                for x in range(0, len(pbs_array_data), MAX_SUBJOBS)]
        if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
            pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
        for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):

            PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
#PBS -P {kwargs.get('P',"''")}
#PBS -q {kwargs.get('q','defaultQ')}
#PBS -V
#PBS -m n
##PBS -o {path}job -e {path}job
#PBS -o {path}/{job_dir} -e {path}/{job_dir}
#PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
#PBS -l walltime={kwargs.get('walltime','23:59:00')}
#PBS -J {MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_chunk)-1}
args=($(python3 -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{MAX_SUBJOBS*i}])))" $PBS_ARRAY_INDEX))
cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
echo "pbs_array_args = ${{args[*]}}"
{source_activate}
{conda_activate}
{command} ${{args[*]}} {additional_command} {post_command}
END"""        

            os.system(f'qsub {PBS_SCRIPT}')
            #print(PBS_SCRIPT)

    # ---------- end{ARTEMIS} ----------

    # ---------- begin{PHYSICS0} ---------- perhaps can use for SBATCH
    # elif cluster == 'PHYSICS':
    #     MAX_SUBJOBS = 1000

    #     if not isdir(SCRIPT_DIR): makedirs(SCRIPT_DIR)                
    #     sh_path = kwargs.get('sh_path', njoin(SCRIPT_DIR, kwargs.get('sh_name', f'script_'+datetime.today().strftime('%Y%m%d'))+'.sh'))
    #     for i, pbs_array_data_point in enumerate(pbs_array_data):

    #         # old args
    #         # args=($(python3 -c "import sys;print(' '.join(map(str, {pbs_array_data_point})))"))
    #         # echo "pbs_array_args = ${{args[*]}}"
    #         # {command} ${{args}} {additional_command} {post_command}

    #         args = ' '.join(map(str, pbs_array_data_point))
    #         full_command = f'{command} {args} {additional_command} {post_command}'

    #     write_script_kwargs = {'sh_path': kwargs.get('sh_path',sh_path), 'scheduler': 'PBS'}        
    #     write_script(full_command, **write_script_kwargs)
    #     #os.system('sbatch %s.sh' % expe_name)
    #     #time.sleep(.1)

    # ---------- end{PHYSICS0} ----------

def job_setup(script_name, kwargss, **kwargs):

    assert isfile(script_name), f"{script_name} does not exist!"

    # cluster
    cluster = kwargs.get('cluster')
    nstack = kwargs.get('nstack', 1)

    # computing resource settings
    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)  # number of nodes    
    walltime = kwargs.get('walltime', '23:59:59')
    mem = kwargs.get('mem', '8GB')            
    
    pbs_array_data = get_pbs_array_data(kwargss)     
    if cluster == 'GADI':   
        perm, pbss = job_divider(pbs_array_data, len(GADI_PROJECTS))    
    elif cluster == 'ARTEMIS':   
        perm, pbss = job_divider(pbs_array_data, len(ARTEMIS_PROJECTS))
    else:
        perm, pbss = job_divider(pbs_array_data, 1)  # projects not needed

    #master_port = 0
    HOST_NODE_ADDR = 0
    commands, script_names, pbs_array_trues, kwargs_qsubs = [], [], [], []
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]        
        kwargs_qsub = {"path":        kwargs.get("job_path"),  # acts as PBSout                       
                       "ngpus":       ngpus, 
                       "ncpus":       ncpus, 
                       "select":      select,
                       "walltime":    walltime,
                       "mem":         mem,
                       "nstack":      nstack                       
                       }        

        kwargs_command = kwargs_qsub; del kwargs_command["path"]
        kwargs_command["cluster"] = cluster

        # ----- GADI -----
        if cluster == 'GADI':            
            # project names
            kwargs_qsub["P"] = GADI_PROJECTS[pidx]
            print(GADI_PROJECTS[pidx])

            if select * max(ncpus, ngpus) > 1:
                # master_port += 1            
                HOST_NODE_ADDR += 1

            if ngpus >= 1:
                kwargs_qsub["q"] = 'gpuvolta'
            else:
                #kwargs_qsub["q"] = 'yossarian'                
                kwargs_qsub["q"] = 'normal'

            # kwargs_command["HOST_NODE_ADDR"] = HOST_NODE_ADDR
            # kwargs_command["singularity_path"] = SPATH
            kwargs_qsub["source"] = GADI_SOURCE
        # -------------------

        # ----- PHYSICS -----
        elif cluster == 'PHYSICS':
            if ngpus >= 1:
                kwargs_qsub["q"] = 'l40s'
            else:
                #kwargs_qsub["q"] = 'yossarian'                
                kwargs_qsub["q"] = 'taiji'

            kwargs_qsub["source"] = PHYSICS_SOURCE 
            kwargs_qsub["conda"] = PHYSICS_CONDA
        # -------------------

        # ----- FUDAN -----
        elif cluster == 'FUDAN_BRAIN':
            if ngpus >= 1:
                kwargs_qsub["q"] = 'gpu'
            else:
                kwargs_qsub["q"] = 'defaultQ'                

            kwargs_qsub["conda"] = FUDAN_CONDA
        # -------------------        

        # ----- ARTEMIS -----
        elif cluster == 'ARTEMIS':            
            # project names
            kwargs_qsub["P"] = ARTEMIS_PROJECTS[pidx]
            print(ARTEMIS_PROJECTS[pidx])

            if select * max(ncpus, ngpus) > 1:
                # master_port += 1            
                HOST_NODE_ADDR += 1

            kwargs_command["HOST_NODE_ADDR"] = HOST_NODE_ADDR
            kwargs_command["singularity_path"] = SPATH
        # -------------------

        command, additional_command = command_setup_ddp(**kwargs_command)                

        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        #qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        #print("\n")
        commands.append(command)
        script_names.append(script_name)
        pbs_array_trues.append(pbs_array_true)
        kwargs_qsubs.append(kwargs_qsub)

    assert len(list(set([len(commands), len(script_names), len(pbs_array_trues), len(kwargs_qsubs)]))) == 1, 'len not same'
    return commands, script_names, pbs_array_trues, kwargs_qsubs

def add_common_kwargs(kwargss, common_kwargs):
    for i in range(len(kwargss)):
        for key, value in common_kwargs.items():
            kwargss[i][key] = value
    return kwargss

def get_pbs_array_data(kwargss):
    pbs_array_data = []
    for kwargs in kwargss:
        args_ls = []
        for key, value in kwargs.items():
            args_ls.append(f'--{key}={value}')
        pbs_array_data.append(tuple(args_ls))
    return pbs_array_data

# N is the total number of projects
def job_divider(pbs_array: list, N: int):
    total_jobs = len(pbs_array)
    assert total_jobs >= 2, 'total_jobs does not exceed 1'
    ncores = min(int(np.floor(total_jobs/2)), N)
    pbss = []
    delta = int(round(total_jobs/ncores))
    for idx in range(ncores):
        if idx != ncores - 1:
            pbss.append( pbs_array[idx*delta:(idx+1)*delta] )
        else:
            if len(pbs_array[idx*delta:]) < 2:
                pbss[-1] = pbss[-1] + pbs_array[idx*delta:]
            else:    
                pbss.append( pbs_array[idx*delta:] )   
    ncores = len(pbss)
    perm = list(np.random.choice(N,ncores,replace=False))
    assert len(perm) == len(pbss), "perm length and pbss length not equal!"

    return perm, pbss

def command_setup_ddp(**kwargs):

    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)
    
    if 'singularity_path' in kwargs:
        singularity_path = kwargs.get('singularity_path')
        assert isfile(singularity_path) or isdir(singularity_path), "singularity_path does not exist!"
    else:
        singularity_path = ''

    if len(singularity_path) > 0:
        bind_path = kwargs.get('bind_path', BPATH)
        home_path = kwargs.get('home_path', os.getcwd())    
        if ngpus > 0:
            if 'pydl.img' not in singularity_path:
                command = f"singularity exec --nv --bind {bind_path} --home {home_path} {singularity_path}"
            else:
                command = f"singularity exec --nv {singularity_path}"
        else:
            if 'pydl.img' not in singularity_path:
                command = f"singularity exec --bind {bind_path} --home {home_path} {singularity_path}"
            else:
                command = f"singularity exec {singularity_path}"
    else:
        command = ""

    additional_command = ''
    #if max(ngpus, ncpus) <= 1:
    if ngpus == 1 or ncpus == 1:
        command += " python3"
    elif ngpus > 1:
        #command += f" CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node={ngpus}"
        #additional_command = 'run --backend=nccl'
        if select == 1:
            # command += f" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0"
            # command += f" --nnodes=1 --nproc_per_node={ngpus} --max-restarts=3"
            command += f" torchrun --standalone --nnodes=1 --nproc_per_node={ncpus}"
        else:
            # tolerates 3 failures
            # command += f" torchrun --nnodes={select} --nproc_per_node={ngpus}"
            # command += f" --max-restarts=3 --rdzv-id=$JOB_ID"
            # command += f" --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"
            command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"
    elif ngpus == 0 and ncpus > 1:
        #python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train_classification_imdb.py run --backend=gloo
        #additional_command = 'run --backend=gloo'
        if select == 1:
            # 1.
            command += f" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0"            
            # command += f" --nnodes=1 --nproc_per_node={ncpus} --max-restarts=3"  # (optional)
            command += f" --nnodes=1 --nproc_per_node={ncpus}"
            # 2.
            #command += f" torchrun --standalone --nnodes=1 --nproc_per_node={ncpus}"
            # 3.
            # command += f" torchrun --nnodes=1 --nproc_per_node={ncpus}"
        else:
            command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"
            command += f" --max-restarts=3 --rdzv-id=$JOB_ID"
            command += f" --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"    
            #command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"          

    if len(singularity_path) == 0:
        command = command[1:]

    return command, additional_command    


def str_to_time(s):
    t = datetime.strptime(s, "%H:%M:%S")
    time_delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)    
    return time_delta

def time_to_str(delta):
    total_seconds = int(delta.total_seconds())
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" 


# ---------------------------------------- UNUNSED FUNCTIONS ----------------------------------------


def write_script(full_command, **kwargs):
    '''
    Writes the bash script to launch expe
    '''

    scheduler = kwargs.get('scheduler', 'PBS')
    assert scheduler in ['PBS', 'SBATCH']

    path = kwargs.get('path', njoin(DROOT, 'jobs_all'))

    # -------------------- 1. SBATCH --------------------
#     if scheduler == 'SBATCH':
#         with open('%s.sh' % sh_name, 'w') as rsh:
#             rsh.write(f'''\
# #!/bin/bash
# #SBATCH -A ynt@gpu
# #SBATCH --job-name=%s%%j     # job name
# #SBATCH --ntasks=1                   # number of MP tasks
# #SBATCH --ntasks-per-node=1          # number of MPI tasks per node
# #SBATCH --ntasks-per-node=1          # number of MPI tasks per node
# #SBATCH --gres=gpu:1               # number of GPUs per node
# #SBATCH --cpus-per-task=10           # number of cores per tasks
# #SBATCH --hint=nomultithread         # we get physical cores not logical
# #SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
# #SBATCH --time=16:00:00              # maximum execution time (HH:MM:SS)
# #SBATCH --output=job_outputs/%s%%j.out # output file name
# #SBATCH --partition=gpu_p13
# #SBATCH --qos=qos_gpu-t3
# #SBATCH --error=job_outputs/%s%%j.err  # error file name

# set -x
# cd ${SLURM_SUBMIT_DIR}    

# module purge
# module load pytorch-gpu/py3/1.8.1
# module load cmake
# module load cuda

# python3 ./one_expe.py %s
# ''' % (name, name, name, args_string))
    # ---------------------------------------------------

    # -------------------- 2. PBS --------------------
    if scheduler == 'PBS':                   
        with open(kwargs.get('sh_path'), 'w') as rsh:
            rsh.write(f'''\
#PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
#PBS -q {kwargs.get('q','defaultQ')}
#PBS -V
#PBS -m n
#PBS -o {path} -e {path}
#PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
#PBS -l walltime={kwargs.get('walltime','23:59:59')}     

cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
#cd fractional-attn/vit-pytorch

source /usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh
conda activate frac_attn                                                        
{full_command}

''')        
    # ---------------------------------------------------

# originally in submit_main.py

"""
def train_submit(script_name, kwargss, **kwargs):

    assert isfile(script_name), f"{script_name} does not exist!"

    # cluster
    cluster = kwargs.get('cluster')
    nstack = kwargs.get('nstack', 1)

    # computing resource settings
    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)  # number of nodes    
    walltime = kwargs.get('walltime', '23:59:59')
    mem = kwargs.get('mem', '8GB')            
    
    pbs_array_data =  (kwargss)     
    if cluster == 'ARTEMIS':   
        perm, pbss = job_divider(pbs_array_data, len(PROJECTS))
    elif cluster == 'PHYSICS':
        perm, pbss = job_divider(pbs_array_data, 1)  # not needed for projects

    #master_port = 0
    HOST_NODE_ADDR = 0
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]        
        kwargs_qsub = {"path":        kwargs.get("job_path"),  # acts as PBSout                       
                       "ngpus":       ngpus, 
                       "ncpus":       ncpus, 
                       "select":      select,
                       "walltime":    walltime,
                       "mem":         mem,
                       "nstack":      nstack                       
                       }        

        kwargs_command = kwargs_qsub; del kwargs_command["path"]
        kwargs_command["cluster"] = cluster

        # ----- ARTEMIS -----
        if cluster == 'ARTEMIS':            
            # project names
            kwargs_qsub["P"] = PROJECTS[pidx]
            print(PROJECTS[pidx])

            if select * max(ncpus, ngpus) > 1:
                # master_port += 1            
                HOST_NODE_ADDR += 1

            kwargs_command["HOST_NODE_ADDR"] = HOST_NODE_ADDR
            kwargs_command["singularity_path"] = SPATH

        # ----- PHYSICS -----
        elif cluster == 'PHYSICS':
            if ngpus >= 1:
                kwargs_qsub["q"] = 'l40s'
            else:
                #kwargs_qsub["q"] = 'yossarian'
                pass

            kwargs_qsub["source"] = PHYSICS_SOURCE 
            kwargs_qsub["conda"] = PHYSICS_CONDA

        command, additional_command = command_setup_ddp(**kwargs_command)                

        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        qsub(f'{command} {script_name}', pbs_array_true, path=kwargs.get('job_path'), **kwargs_qsub)   
        print("\n")
"""  
import numpy as np
import sys
import os
import random
from datetime import datetime
from os.path import join, normpath, isdir, isfile
from constants import njoin

#project_ls = ["PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]
project_ls = ["phys_DL", "PDLAI", "dnn_maths", "dyson", "vortex_dl", "frac_attn"]
#project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

def qsub(command, pbs_array_data, **kwargs):
    """A general PBS array job submitting function.
    
    Submits a PBS array job, each subjob calling `command` followed by the
    arguments of an element of `pbs_array_data`, ending with the path of the
    output folder (if `pass_path`=True):
    
        command *(pbs_array_data[i]) path
    
    The subjob argument ordering follows the input-processing-output paradigm.
    
    Args:
        `command` is a string which may have spaces
        `pbs_array_data` is an array of argument tuples
    Keyword args (optional):
        `path` is passed in at the end and determines the PBS job output
               location.
        `N`, `P`, `q`, `select`, `ncpus`, `mem`, `walltime`, `gpu`: as in PBS qsub
        `local`: if True, runs the array job locally (defaults to False).
                 Intended for debugging.
        `cd`: Set the subjob working directory, defaults to cwd
        `pass_path`: Pass `path` at the end of the call, defaults to False.
                     Expects bool. If string, passes custom path to command.
    """
    if 'path' in kwargs:
        path = kwargs['path']
        if path and path[-1] != os.sep: path += os.sep
    else:
        path = command.replace(' ', '_') + os.sep
    if kwargs.get('pass_path', False):
        post_command = path if kwargs['pass_path'] == True else kwargs['pass_path']
    else:
        post_command = ''
    # Create output folder.
    #if not isdir(join(path,"job")): os.makedirs(join(path,"job"))
    # Create output folder.
    date_str = datetime.today().strftime('%Y-%m-%d')
    job_dir = f"{date_str}_out"
    if not isdir(njoin(path,job_dir)): os.makedirs(njoin(path,job_dir))
    # source virtualenv
    if 'source' in kwargs:
        #assert isfile(kwargs.get('source')), "source for virtualenv incorrect"
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
    # additional command
    if 'add_command' in kwargs:  
        add_command_exists = 'true' if 'add_command' in kwargs else 'false'
        add_command = kwargs.get('add_command')
    else:
        add_command = ''    
    if kwargs.get('local', False):  # Run the subjobs in the current process.
        for pbs_array_args in pbs_array_data:
            str_pbs_array_args = ' '.join(map(str, pbs_array_args))
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
    # PBS array jobs are limited to 1000 subjobs by default
    pbs_array_data_chunks = [pbs_array_data[x:x+1000]
                             for x in range(0, len(pbs_array_data), 1000)]
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
#PBS -J {1000*i}-{1000*i + len(pbs_array_data_chunk)-1}
args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{1000*i}])))" $PBS_ARRAY_INDEX))
cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
echo "pbs_array_args = ${{args[*]}}"
# <<remove1>>
{add_command}
{source_activate}
{conda_activate}            
{command} ${{args[*]}} {post_command}
END"""
        os.system(f'qsub {PBS_SCRIPT}')
        #print(PBS_SCRIPT)

# <<remove1>>
"""
if [ {source_exists} ]; then
   source {SOURCE}
fi
if [ {conda_exists} ]; then
   conda activate {CONDA}
fi
"""

# ----- added by CKQu1 -----

# N is the total number of projects
def job_divider(pbs_array: list, N: int):
    total_jobs = len(pbs_array)
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

# singularity exec usage
def command_setup(singularity_path, **kwargs):   

    repo_dir = os.getcwd()
    bind_path = kwargs.get('bind_path', '')
    ncpus = kwargs.get('ncpus', 1)
    ngpus = kwargs.get('ngpus', 0)

    if len(singularity_path) > 0:
        assert isfile(singularity_path) or isdir(singularity_path), "singularity_path does not exist!"
        if ngpus == 0:
            command = f"singularity exec"
        else:
            command = f"singularity exec --nv"
        if len(bind_path) > 0:
            command += f" --bind {bind_path}"
        command += f" --home {repo_dir} {singularity_path}"
    else:
        command = ""

    command += f" python"
    
    if len(singularity_path) == 0:
        command = command[1:]

    return command


def list_str_divider(ls, chunks):
    """
    Divide list into size of no more than chunks.
    """
    start = 0
    n = len(ls)
    lss = []
    while start + chunks < n:
        lss.append(str( ls[start:start+chunks] ).replace(" ",""))
        #lss.append('\'' + str( ls[start:start+chunks] ).replace(' ','') + '\'')
        start += chunks
    if start <= n - 1:
        lss.append(str( ls[start:] ).replace(" ",""))
        #lss.append('\'' + str( ls[start:] ).replace(' ','') + '\'')    
    return lss
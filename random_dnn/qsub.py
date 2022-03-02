"""Submits a PBS array job using Python array data.

qsub.py: A convenient Python wrapper for qsub.

Author: Asem Wardak, University of Sydney

Since the -J option to the PBS command qsub takes a single-dimensional range,
passing through multidimensional ranges requires some kind of conversion either
at the PBS script level, or at the level of the computational script being
called, which can be cumbersome.

This script allows the generation of the multidimensional PBS array data to
occur prior to the submission of the script, allowing the subjob to run
completely independently of the PBS array, and consequently for it to be called
using PBS-agnostic, regular arguments. In particular, the subjob does not know
if it is being called directly, or if it is part of an asynchronous parallel
computation on a cluster. This greatly simplifies the development, testing and
running of computations on a massively parallel cluster. It also encourages the
clean management and distinction of input, processing, and output data, by
encapsulating the concept of a PBS array range in the `processing data`.

A typical use case would be to prepare the PBS array data for submission to
qsub(..), which will call a computational function which stores its result in a
unique file:
    input -> submit_array_job(..) -> qsub(..) -> run_subjob(..) -> output

This script may be used to run subjobs which can call any program which accepts
positional arguments.

Notes:
- os.system uses the working directory of the terminal rather than that of the
  imported file, so the shell qsub command sees the same directory as the
  currently running script
- the cluster compute nodes must have access to `path`, even if they do not
  have access to this file
- if the PBS_SCRIPT string is modified, the shell calling qsub(..) may need to
  be restarted.
- PBS arguments with string quotes may have problems
- `path` must be absolute for #PBS -o and -e
"""

import sys
import os
import random


def qsub(command, pbs_array_data, **kwargs):
    """A general PBS array job submitting function.
    
    Submits a PBS array job, each subjob calling `command` followed by the
    arguments of an element of `pbs_array_data`, ending with the path of the
    output folder:
    
        command *(pbs_array_data[i]) path
    
    The subjob argument ordering follows the input-processing-output paradigm.
    
    Args:
        `command` is a string which may have spaces
        `pbs_array_data` is an array of argument tuples
    Keyword args (optional):
        `path` is passed in at the end and determines the PBS job output
               location.
        `N`, `P`, `q`, `select`, `ncpus`, `mem`, `walltime`: as in PBS qsub
        `local`: if True, runs the array job locally (defaults to False).
                 Intended for debugging.
        `cd`: Set the subjob working directory, defaults to cwd
        `pass_path`: Pass `path` at the end of the call, defaults to True
    """
    if 'path' in kwargs:
        path = kwargs['path']
        if path and path[-1] != os.sep: path += os.sep
    else:
        path = command.replace(' ', '_') + os.sep
    # Create output folder.
    if not os.path.isdir(path+'job'): os.makedirs(path+'job')
    if kwargs.get('local', False):  # Run the subjobs in the current process.
        for pbs_array_args in pbs_array_data:
            str_pbs_array_args = ' '.join(map(str, pbs_array_args))
            os.system(f"""bash <<'END'
                cd {kwargs.get('cd', '.')}
                echo "pbs_array_args = {str_pbs_array_args}"
                {command} {str_pbs_array_args} {path if kwargs.get('pass_path', True) else ''}
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
            #PBS -o {path}job -e {path}job
            #PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}
            #PBS -l walltime={kwargs.get('walltime','23:59:00')}
            #PBS -J {1000*i}-{1000*i + len(pbs_array_data_chunk)-1}
            args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{1000*i}])))" $PBS_ARRAY_INDEX))
            cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
            echo "pbs_array_args = ${{args[*]}}"
            {command} ${{args[*]}} {path if kwargs.get('pass_path', True) else ''}
END"""
        os.system(f'qsub {PBS_SCRIPT}')
        #print(PBS_SCRIPT)

"""Submit the RMT code to the cluster."""

import sys
import subprocess
from datetime import datetime
import random
from pathlib import Path

import concurrent.futures as cf

scriptpath = "~/wolfram/13.3/Executables/wolframscript"


def consolidate_arrays(path: Path):
    import numpy as np
    from tqdm import tqdm
    path = Path(path)
    pattern = "alpha*_g*_seed*.txt"
    files = path.glob(pattern)
    arrays_dict = {}
    with cf.ThreadPoolExecutor() as executor:
        future_to_file_stem = {
            executor.submit(np.loadtxt, file): file.stem for file in files
        }
        for future in tqdm(cf.as_completed(future_to_file_stem)):
            file_stem = future_to_file_stem[future]
            arrays_dict[file_stem] = future.result()
    # for file in tqdm(files):
    #     arrays_dict[file.stem] = np.loadtxt(file)
    if arrays_dict:
        np.savez(path.with_suffix(".npz"), **arrays_dict)
    del arrays_dict


# at width 1000, depth 50: about 5 seconds on a CPU core
def submit_RMT(host, q):
    width = 1000
    depth = 50
    num_func_calls_per_subjob = 210
    func_calls = [
        f"savetxt('{fname}', MLP_log_svdvals, {alpha100/100}, {g100/100}, 0, torch.tanh, {width}, {depth}, seed={seed}, device='cpu')"
        for alpha100 in range(100, 201, 5)
        for g100 in range(1, 301, 5)
        for seed in range(50)
        if not Path(
            fname := f"fig/MLP_log_svdvals/width{width}_depth{depth}/alpha{alpha100}_g{g100}_seed{seed}.txt"
        ).exists()
    ]
    if host == "headnode":
        init = "source /import/silo3/wardak/.venv/bin/activate"
        pythonpath = "python"
    elif host == "gadi":
        init = "\n".join(
            ["module load python3 cuda", "source /g/data/au05/venv/bin/activate"]
        )
        pythonpath = "python3"
    subjobs = [
        "\n".join(
            [init, f'{pythonpath} -c "from RMT import *']
            + func_calls[i : i + num_func_calls_per_subjob]
            + ['"']
        )
        for i in range(0, len(func_calls), num_func_calls_per_subjob)
    ]
    # print(subjobs[0])
    # return
    qsub_gadi(
        subjobs,
        "/scratch/au05/aw9402" if host == "gadi" else Path(".") / "fig",
        q=q,
        storage="gdata/au05+scratch/au05" if host == "gadi" else None,
        P="au05",
        ncpus=1,
        # ngpus=1,
        # ncpus=12,
        walltime="0:59:00",
    )


def submit_qmap():
    cmd_list = [
        f"""
            {scriptpath} -f theory.wl "PutLogQMap[{alpha100}, {g100}, 0]"
        """
        for alpha100 in range(100, 201, 5)
        for g100 in range(0, 301, 5)[1:]
    ]
    random.shuffle(cmd_list)
    jobids = qsub(cmd_list, Path(".") / "fig")
    # submit_reducer(jobids[-1])


def submit_reducer(depend_after=False):
    cmd_list = [
        rf"""{scriptpath} -f theory.wl "SavePuts[\"empiricalMLP\", None]" """,
        # rf"""{scriptpath} -f theory.wl "SavePuts[\"loginvCDF_1000\", \"Table\", ToExpression]" """,
        # rf"""{scriptpath} -f theory.wl SavePuts[\"empiricalLogSingVals_1000_50\"]""",
        # rf"""{scriptpath} -f theory.wl SavePuts[\"empiricalLogAbsEigs_1000_50\"]""",
    ]
    qsub(
        cmd_list,
        Path(".") / "fig",
        "consolidator",
        mem="16GB",
        depend_after=depend_after,
    )


def qsub(
    cmd_list: list[str],
    path: Path = Path("."),
    N=sys.argv[0] or "job",
    P="''",
    q="defaultQ",
    select=1,
    ncpus=1,
    mem="1GB",
    ngpus: int = None,
    walltime="23:59:00",
    max_array_size=1000,
    max_run_subjobs: int = None,
    depend_after=False,  # False, True or the name of a job
    print_script=False,
):
    lastjobid = None if depend_after in [False, True] else depend_after
    if max_run_subjobs is None:
        max_run_subjobs = max_array_size
    if len(cmd_list) == 1:
        cmd_list.append("")
    path = Path(path)
    label = datetime.now().strftime(rf"{N}_%Y%m%d_%H%M%S_%f")
    jobpath = (path / "job" / label).resolve()
    jobpath.mkdir(parents=True, exist_ok=True)
    # Create the input files.
    for idx, cmd in enumerate(cmd_list):
        with open(jobpath / f"cmd_{idx}.sh", "w") as f:
            f.write(cmd)
    cmd_list_chunks = [
        cmd_list[i : i + max_array_size]
        for i in range(0, len(cmd_list), max_array_size)
    ]
    # Make sure no array job has length 1.
    if len(cmd_list_chunks[-1]) == 1:
        cmd_list_chunks[-1].insert(0, cmd_list_chunks[-2].pop())
    jobids = []
    for chunk_idx, cmd_list_chunk in enumerate(cmd_list_chunks):
        PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -k oed
#PBS -N {N}
#PBS -P {P}
#PBS -q {q}
#PBS -V
#PBS -m n
#PBS -o {jobpath}/ -e {jobpath}/
#PBS -l select={select}:ncpus={ncpus}:mem={mem}{f':ngpus={ngpus}' if ngpus else ''}
#PBS -l walltime={walltime}
#PBS -J {max_array_size*chunk_idx}-{max_array_size*chunk_idx + len(cmd_list_chunk) - 1}{f'%{max_run_subjobs}' if max_run_subjobs else ''}
{('#PBS -W depend=afterany:'+lastjobid) if depend_after and (lastjobid is not None) else ''}
cd $PBS_O_WORKDIR
bash {jobpath}/cmd_$PBS_ARRAY_INDEX.sh
END"""
        lastjobid = subprocess.check_output(
            f"qsub {PBS_SCRIPT}", shell=True, text=True
        ).strip()
        print(lastjobid)
        jobids.append(lastjobid)
        if print_script:
            print(PBS_SCRIPT)
    return jobids


def qsub_gadi(
    cmd_list: list[str],
    path: Path = Path("."),
    N=sys.argv[0] or "job",
    P="''",
    q="defaultQ",
    ncpus=1,
    mem="1GB",
    ngpus: int = None,
    walltime="23:59:00",
    storage=None,  # this appears to be specific to NCI Gadi
    max_array_size=1,
    max_run_subjobs: int = None,
    depend_after=False,  # False, True or the name of a job
    print_script=False,
):
    """A `qsub` variant for NCI Gadi because it doesn't accept array jobs.
    
    TODO:
    - Submit to multiple queues simultaneously
    """
    if "/" in N:
        N = N.split("/")[-1]
    lastjobid = None if depend_after in [False, True] else depend_after
    if max_run_subjobs is None:
        max_run_subjobs = max_array_size
    if len(cmd_list) == 1:
        cmd_list.append("")
    path = Path(path)
    label = datetime.now().strftime(rf"{N}_%Y%m%d_%H%M%S_%f")
    jobpath = (path / "job" / label).resolve()
    jobpath.mkdir(parents=True, exist_ok=True)
    # Create the input files.
    for idx, cmd in enumerate(cmd_list):
        with open(jobpath / f"cmd_{idx}.sh", "w") as f:
            f.write(cmd)
    cmd_list_chunks = [
        cmd_list[i : i + max_array_size]
        for i in range(0, len(cmd_list), max_array_size)
    ]
    # Make sure no array job has length 1.
    # if len(cmd_list_chunks[-1]) == 1:
    #     cmd_list_chunks[-1].insert(0, cmd_list_chunks[-2].pop())
    jobids = []
    for chunk_idx, cmd_list_chunk in enumerate(cmd_list_chunks):
        PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -k oed
#PBS -N {N}
#PBS -P {P}
#PBS -q {q}
#PBS -V
#PBS -m n
#PBS -o {jobpath}/ -e {jobpath}/
#PBS -l ncpus={ncpus}
#PBS -l mem={mem}
{f'#PBS -l ngpus={ngpus}' if ngpus else ''}
#PBS -l walltime={walltime}
{f'#PBS -l storage={storage}' if storage else ''}
{('#PBS -W depend=afterany:'+lastjobid) if depend_after and (lastjobid is not None) else ''}
cd $PBS_O_WORKDIR
bash {jobpath}/cmd_{chunk_idx}.sh
END"""
        lastjobid = subprocess.check_output(
            f"qsub {PBS_SCRIPT}", shell=True, text=True
        ).strip()
        print(lastjobid)
        jobids.append(lastjobid)
        if print_script:
            print(PBS_SCRIPT)
    return jobids


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python %s FUNCTION_NAME ARG1 ... ARGN" % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

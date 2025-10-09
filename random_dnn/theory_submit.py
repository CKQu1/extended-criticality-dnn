"""Submit the RMT code to the cluster."""

import sys
import subprocess
import random
import concurrent.futures as cf
from datetime import datetime
from pathlib import Path


scriptpath = "~/wolfram/13.3/Executables/wolframscript"


def updatez(file, *args, **kwds):
    """Like `np.savez` but appends.

    Follows the logic in the underlying function
    [`numpy.lib.npyio._savez()`](https://github.com/numpy/numpy/blob/v2.3.0/numpy/lib/_npyio_impl.py#L769).

    Appending with an existing filename in the archive will cause duplicate names,
    so we only add new names.

    Returns the set of new keys added.

    Refs:
    - [demonstration on npz file](https://stackoverflow.com/a/66618141)
    - [append demonstration on compressed zip](https://stackoverflow.com/a/25154589)
    """
    import zipfile
    import numpy as np

    namedict = kwds
    for i, val in enumerate(args):
        key = "arr_%d" % i
        if key in namedict.keys():
            raise ValueError("Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val

    with zipfile.ZipFile(file, mode="a") as zipf:
        new_keys = kwds.keys() - set(zipf.namelist())  # only append new names
        for key in new_keys:
            val = np.asanyarray(namedict[key])
            fname = key + ".npy"
            with zipf.open(fname, "w", force_zip64=True) as fid:
                np.lib.format.write_array(fid, val)
    return new_keys


def consolidate_arrays(path: Path, pattern="*.txt", file_delay_minutes=0):
    import numpy as np
    from tqdm import tqdm

    # Find all files matching the pattern that are older than file_delay_minutes
    current_time_sec = int(datetime.now().timestamp())
    files = set(
        f
        for f in path.glob(pattern)
        if f.stat().st_mtime < current_time_sec - file_delay_minutes * 60
    )
    # If the npz file already exists, skip files that are already in it
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path) as npz:
            files = files - npz.keys()
    # Load the files in parallel
    arrays_dict = {}
    with cf.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(np.loadtxt, file): str(file.relative_to(path))
            for file in files
        }
        for future in tqdm(cf.as_completed(future_to_file)):
            file = future_to_file[future]
            arrays_dict[file] = future.result()
    # for file in tqdm(files):
    #     arrays_dict[file.stem] = np.loadtxt(file)
    if arrays_dict:
        # Save to npz, appending if it already exists
        if npz_path.exists():
            saved_files = updatez(npz_path, **arrays_dict)
            print(f"Appended {len(saved_files)} files to {npz_path}")
        else:
            np.savez(npz_path, **arrays_dict)
            saved_files = arrays_dict.keys()
            print(f"Saved {len(saved_files)} files to {npz_path}")
        # Delete the saved files
        for file in saved_files:
            (path / file).unlink()
    del arrays_dict


def submit_jac_cavity_svd_log_pdf(
    num_doublings="6",
    logspace_params="-3,3,1000",
    walltime="0:59:00",
    mem="1GB",
    dry=False,
):
    num_func_calls_per_subjob = 1
    path = (
        Path("fig")
        / "jac_cavity_svd_log_pdf"
        / f"doublings{num_doublings}_logspace_{logspace_params}"
    )
    func_calls_dict = {
        fname: f"savetxt('{path/fname}', "
        "jac_cavity_svd_log_pdf, "
        f"np.logspace({logspace_params}), "
        f"{alpha100/100}, {sigma_W/100}, "
        f"num_doublings={num_doublings})"
        for alpha100 in range(100, 201, 5)
        for sigma_W in range(1, 301, 5)
        if not (path / (fname := f"alpha{alpha100}_sigmaW{sigma_W}.txt")).exists()
    }
    if path.with_suffix(".npz").exists():
        import zipfile

        with zipfile.ZipFile(path.with_suffix(".npz"), "r") as zipf:
            remaining_keys = func_calls_dict.keys() - set(
                name.removesuffix(".npy") for name in zipf.namelist()
            )
            func_calls_dict = {k: func_calls_dict[k] for k in remaining_keys}
    func_calls = list(func_calls_dict.values())
    if not func_calls:
        print("No new files.")
        return
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
    if dry:
        print(f"Would submit {len(subjobs)} jobs.")
        return
    queue_rotation = (  # max walltime 48 hours
        ["normal" for _ in range(300)]
        + ["normalsr" for _ in range(300)]
        + ["normalbw" for _ in range(300)]
        + ["normalsl" for _ in range(300)]
    )
    qsub_single(
        subjobs,
        "/scratch/au05/aw9402/job",
        q=queue_rotation[: len(subjobs)],
        storage="gdata/au05+scratch/au05",
        P="au05",
        ncpus=1,
        mem=mem,
        # ngpus=1,
        # ncpus=12,
        walltime=walltime,
    )


def submit_MLP_log_svdvals(host, q):
    """We are chaining function calls to save on the pytorch loading time

    at width 1000, depth 50: about 5 seconds on a CPU core

    TODO: 
    - combine this and the other submit function into a generalised python function call submission
        - and provide two wrapper functions for the MLP and RMT evaluations
    """
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
    qsub_single(
        subjobs,
        "/scratch/au05/aw9402/job" if host == "gadi" else Path(".") / "fig",
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
    jobpath = (path / label).resolve()
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


def qsub_single(
    cmd_list: list[str],
    path: Path = Path("."),
    N=sys.argv[0] or "job",
    P="''",
    q: str | list[str] = "defaultQ",
    ncpus=1,
    mem="1GB",
    ngpus: int = None,
    walltime="23:59:00",
    storage=None,  # this appears to be specific to NCI Gadi
    print_script=False,
):
    """A `qsub` variant that does not use array jobs.

    Needed for NCI Gadi.

    If `q` is a list it will rotate through the queues for each job.
    """
    if "/" in N:
        N = N.split("/")[-1]
    path = Path(path)
    label = datetime.now().strftime(rf"{N}_%Y%m%d_%H%M%S_%f")
    jobpath = (path / label).resolve()
    jobpath.mkdir(parents=True, exist_ok=True)
    # Create the input files.
    for cmd_idx, cmd in enumerate(cmd_list):
        with open(jobpath / f"cmd_{cmd_idx}.sh", "w") as f:
            f.write(cmd)
    jobids = []
    for cmd_idx, cmd in enumerate(cmd_list):
        PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -k oed
#PBS -N {N}
#PBS -P {P}
#PBS -q {q if isinstance(q, str) else q[cmd_idx % len(q)]}
#PBS -V
#PBS -m n
#PBS -o {jobpath}/ -e {jobpath}/
#PBS -l ncpus={ncpus}
#PBS -l mem={mem}
{f'#PBS -l ngpus={ngpus}' if ngpus else ''}
#PBS -l walltime={walltime}
{f'#PBS -l storage={storage}' if storage else ''}
cd $PBS_O_WORKDIR
bash {jobpath}/cmd_{cmd_idx}.sh
END"""
        jobid = subprocess.check_output(
            f"qsub {PBS_SCRIPT}", shell=True, text=True
        ).strip()
        jobids.append(jobid)
        if print_script:
            print(PBS_SCRIPT)
    print(f"Submitted {len(cmd_list)} jobs: {jobids[0]} ... {jobids[-1]}")
    return jobids


if __name__ == "__main__":
    # Example usage:
    # python filename.py func_name arg1_name arg1 arg1_type arg2_name arg2 arg2_type ...
    # the arguments will be passed in as keyword args
    import sys
    from time import time

    tic = time()
    func = eval(sys.argv[1])
    args = [sys.argv[i : i + 3] for i in range(2, len(sys.argv), 3)]
    arg_dict = {arg[0]: eval(arg[2])(arg[1]) for arg in args}
    result = func(**arg_dict)
    print(result)
    toc = time()
    print(f"Time: {toc - tic:.2f} sec")

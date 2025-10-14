"""Submit the RMT code to the cluster."""

import platform
import sys
import subprocess
import random
import concurrent.futures as cf
from datetime import datetime
from pathlib import Path
from typing import Union


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
    num_doublings="8",
    logspace_params="-10,10,1000",
    alpha100_step=5,
    sigmaW100_step=5,
    **submit_python_kwargs,
):
    # 8 GB for 10 doublings
    submit_python_kwargs = {
        "mem": "8GB",
        "walltime": "0:59:00",
        **submit_python_kwargs,
    }
    dir = (
        Path("fig")
        / "jac_cavity_svd_log_pdf"
        / f"num_doublings={num_doublings};logspace_params={logspace_params}"
    )
    func_calls_dict = {
        (
            fname := f"alpha100={alpha100};sigmaW100={sigmaW100}.txt"
        ): f"call_save('{dir/fname}', jac_cavity_svd_log_pdf, np.logspace({logspace_params}), {alpha100/100}, {sigmaW100/100}, num_doublings={num_doublings})"
        for alpha100 in range(100, 201, alpha100_step)
        for sigmaW100 in range(1, 301, sigmaW100_step)
    }
    submit_python_funcs(func_calls_dict, dir=dir, **submit_python_kwargs)


def submit_MLP_agg(
    width=1000,
    depth=50,
    alpha100_step=5,
    sigmaW100_step=5,
    num_realisations=50,
    seeds=range(1),
    **submit_python_kwargs,
):
    submit_python_kwargs = {
        # "num_func_calls_per_job": 210,
        "mem": "8GB",
        "walltime": "5:59:00",
        **submit_python_kwargs,
    }
    dir = (
        Path("fig")
        / "MLP_agg"
        / f"width={width};depth={depth};num_realisations={num_realisations}"
    )
    func_calls_dict = {
        (fname := Path(f"alpha100={alpha100};sigmaW100={sigmaW100};seed={seed}.txt"))
        .with_stem(fname.stem + f";log_svdvals_mean")
        .name: f"call_save('{dir/fname}', MLP_agg, torch.linspace(-1, 1, {width}), {depth}, {num_realisations}, {alpha100/100}, {sigmaW100/100}, seed={seed}, device='cpu')"
        for alpha100 in range(100, 201, alpha100_step)
        for sigmaW100 in range(1, 301, sigmaW100_step)
        for seed in seeds
    }
    submit_python_funcs(func_calls_dict, dir=dir, **submit_python_kwargs)


# General submission functions


def submit_python_funcs(
    func_calls_dict: dict[str, str],
    dir: Path,
    num_func_calls_per_job=1,
    init: str = None,  # modules and venvs
    pythonpath: str = "python3",
    init_call: str = "from RMT import *",
    dry=False,
    qsub_func=None,
    **qsub_kwargs,
):
    """A cluster workflow for Python function calls.

    Chains python function calls in a single instance to save on the library loading time.
    Useful for libraries like pytorch which load slowly.

    Each func call is associated with a file to be checked in either the directory or the corresponding npz archives.
    File paths are relative and correspond to the file name in the folder or npz archive.
    If the file name exists, the corresponding function call is not run.

    Note: this can be generalised for function calls in other languages, and to arbitrary archive formats.

    Defaults for USYD Physics and NCI Gadi clusters are provided.
    """
    if platform.node().endswith("physics.usyd.edu.au"):
        if init is None:
            init = "source /import/silo3/wardak/.venv/bin/activate"
        if qsub_func is None:
            qsub_func = qsub
        default_qsub_kwargs = dict(
            path="/taiji1/wardak/job",
            q="defaultQ",  # see email for rules
        )
    elif platform.node().endswith("gadi.nci.org.au"):
        if init is None:
            init = "\n".join(
                ["module load python3 cuda", "source /g/data/au05/venv/bin/activate"]
            )
        if qsub_func is None:
            qsub_func = qsub_single
        default_qsub_kwargs = dict(
            path="/scratch/au05/aw9402/job",
            # normal queues: max walltime 48 hours, 300 jobs per queue
            q=[
                q
                for q in [
                    "normal",
                    "normalsr",
                    "normalbw",
                    "normalsl",
                ]
                for _ in range(300)
            ],
            # gpu queues require 12 cpus per gpu
            storage="gdata/au05+scratch/au05",
            P="au05",
        )
    else:
        if init is None:
            init = ""
        if qsub_func is None:
            qsub_func = qsub
        default_qsub_kwargs = dict()
    qsub_kwargs = {
        "mem": "4GB",
        "walltime": "0:59:00",
        **default_qsub_kwargs,
        **qsub_kwargs,
    }
    # === End of defaults ===
    remaining_func_calls_dict = {
        fname: fcall
        for fname, fcall in func_calls_dict.items()
        if not (dir / fname).exists()
    }
    if dir.with_suffix(".npz").exists():
        import zipfile

        with zipfile.ZipFile(dir.with_suffix(".npz"), "r") as zipf:
            remaining_keys = remaining_func_calls_dict.keys() - set(
                name.removesuffix(".npy") for name in zipf.namelist()
            )
            remaining_func_calls_dict = {
                k: remaining_func_calls_dict[k] for k in remaining_keys
            }
    func_calls = list(remaining_func_calls_dict.values())
    print(f"{len(func_calls)}/{len(func_calls_dict)} function calls to run.")
    if not func_calls:
        return
    newline = (
        "\n"  # avoids SyntaxError: f-string expression part cannot include a backslash
    )
    cmd_list = [
        rf"""
{init}
{pythonpath} - <<'HEREEND'
{init_call}
{newline.join(func_calls[i : i + num_func_calls_per_job])}
HEREEND"""
        for i in range(0, len(func_calls), num_func_calls_per_job)
    ]
    if dry:
        print(f"Would submit {len(cmd_list)} jobs.")
        return
    qsub_func(cmd_list, **qsub_kwargs)


def qsub(
    cmd_list: list[str],
    path: Path = Path("."),  # where to save the job info
    N=sys.argv[0] or "job",
    P="''",
    q: Union[str, list[str]] = "defaultQ",
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
    if "/" in N:
        N = N.split("/")[-1]
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
#PBS -q {q if isinstance(q, str) else q[chunk_idx % len(q)]}
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
    q: Union[str, list[str]] = "defaultQ",
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
    if result is not None:
        print(result)
    toc = time()
    print(f"Script time: {toc - tic:.2f} sec")

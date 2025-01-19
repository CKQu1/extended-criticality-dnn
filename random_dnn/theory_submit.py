"""Submit the RMT code to the cluster."""

import sys
import subprocess
from datetime import datetime
import random
from pathlib import Path

scriptpath = "~/wolfram/13.3/Executables/wolframscript"


def submit(num_samples=1000):
    (Path(".") / "fig" / "data").mkdir(parents=True, exist_ok=True)
    cmd_list = [
        f"""
            # {scriptpath} -f theory.wl "PutJacobianLogInvCDF[{alpha100}, {g100}, 0, {num_samples}, {seed}]"
            {scriptpath} -f theory.wl "PutEmpiricalMLP[{alpha100}, {g100}, 0, Tanh, {num_samples}, 50, {seed}]"
        """
        for alpha100 in range(100, 201, 5)
        for g100 in range(0, 301, 5)[1:]
        for seed in [42] #range(10)
        # if alpha100 == 150 and g100 in [150, 200, 300]
    ]
    random.shuffle(cmd_list)
    jobids = qsub(
        cmd_list,
        Path(".") / "fig",
        mem="4GB",
        # max_run_subjobs=200,
        # depend_after=True,
    )
    submit_reducer(jobids[-1])


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
    label = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")
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
#PBS -l select={select}:ncpus={ncpus}:mem={mem}{':ngpus='+str(ngpus) if ngpus else ''}
#PBS -l walltime={walltime}
#PBS -J {max_array_size*chunk_idx}-{max_array_size*chunk_idx + len(cmd_list_chunk) - 1}%{max_run_subjobs}
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python %s FUNCTION_NAME ARG1 ... ARGN" % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

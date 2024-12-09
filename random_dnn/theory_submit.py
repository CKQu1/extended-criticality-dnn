"""Submit the RMT code to the cluster."""

import sys
import subprocess
import time
import random
from pathlib import Path

scriptpath = "~/wolfram/13.3/Executables/wolframscript"


def submit():
    (Path(".") / "fig" / "data").mkdir(parents=True, exist_ok=True)
    cmd_list = [
        f"""
            {scriptpath} -f theory.wl "PutJacobianLogAvg[{alpha100}, {g100}, 0, 1000, 1000]"
        """
        for alpha100 in range(100, 201, 1)
        for g100 in range(1, 301, 1)
        # if alpha100 == 150 and g100 in [150, 200, 300]
    ]
    random.shuffle(cmd_list)
    qsub(cmd_list, Path(".") / "fig", mem="4GB", max_run_subjobs=100, afterok=True)


def qsub(
    cmd_list: list[str],
    path: Path = Path("."),
    N=sys.argv[0] or "job",
    P="''",
    q="defaultQ",
    select=1,
    ncpus=1,
    mem="1GB",
    ngpus=None,
    walltime="23:59:00",
    max_array_size=1000,
    max_run_subjobs=1000,
    afterok=False,
    print_script=False,
):
    path = Path(path)
    (path / "job").mkdir(parents=True, exist_ok=True)
    # Create the input files.
    label = time.strftime(r"%Y%m%d%H%M%S")
    for idx, cmd in enumerate(cmd_list):
        with open(path / "job" / f"cmd_{idx}_{label}.sh", "w") as f:
            f.write(cmd)
    cmd_list_chunks = [
        cmd_list[i : i + max_array_size]
        for i in range(0, len(cmd_list), max_array_size)
    ]
    # Make sure no array job has length 1.
    if len(cmd_list_chunks[-1]) == 1:
        cmd_list_chunks[-1].insert(0, cmd_list_chunks[-2].pop())
    jobpath = (path / "job").resolve()
    lastjobid = None
    for chunk_idx, cmd_list_chunk in enumerate(cmd_list_chunks):
        PBS_SCRIPT = f"""<<'END'
#!/bin/bash
#PBS -N {N}
#PBS -P {P}
#PBS -q {q}
#PBS -V
#PBS -m n
#PBS -o {jobpath}/ -e {jobpath}/
#PBS -l select={select}:ncpus={ncpus}:mem={mem}{':ngpus='+str(ngpus) if ngpus else ''}
#PBS -l walltime={walltime}
#PBS -J {max_array_size*chunk_idx}-{max_array_size*chunk_idx + len(cmd_list_chunk) - 1}%{max_run_subjobs}
{('#PBS -W depend=afterok:'+lastjobid) if afterok and (lastjobid is not None) else ''}
cd $PBS_O_WORKDIR
bash {jobpath}/cmd_$PBS_ARRAY_INDEX\_{label}.sh
END"""
        lastjobid = subprocess.check_output(f"qsub {PBS_SCRIPT}", shell=True, text=True).strip()
        print(lastjobid)
        if print_script:
            print(PBS_SCRIPT)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python %s FUNCTION_NAME ARG1 ... ARGN" % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

"""Submit the RMT code to the cluster."""

import sys
import os
import random
from pathlib import Path

scriptpath = "~/math13.1/Executables/wolframscript"


def submit():
    cmd_list = [
        f"""
            {scriptpath} -f theory.wl "PutJacobianLogAvg[{alpha100}, {g100}, 0, 100, 1]"
        """
        for alpha100 in range(100, 201, 5)
        for g100 in range(5, 301, 5)
        if alpha100 == 200 or g100 in [150, 200, 300]
    ]
    qsub(cmd_list, Path('.') / 'fig')


QSUB_MAX_ARRAY_SIZE = 1000


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
):
    path = Path(path)
    (path / "job").mkdir(parents=True, exist_ok=True)
    # Create the input files.
    for idx, cmd in enumerate(cmd_list):
        with open(path / "job" / f"cmd_{idx}.sh", "w") as f:
            f.write(cmd)
    cmd_list_chunks = [
        cmd_list[i : i + QSUB_MAX_ARRAY_SIZE]
        for i in range(0, len(cmd_list), QSUB_MAX_ARRAY_SIZE)
    ]
    # Make sure no array job has length 1.
    if len(cmd_list_chunks[-1]) == 1:
        cmd_list_chunks[-1].insert(0, cmd_list_chunks[-2].pop())
    jobpath = (path / 'job').resolve()
    for chunk_idx, cmd_list_chunk in enumerate(cmd_list_chunks):
        PBS_SCRIPT = f"""<<'END'
            #!/bin/bash
            #PBS -N {N}
            #PBS -P {P}
            #PBS -q {q}
            #PBS -V
            #PBS -m n
            #PBS -o {jobpath} -e {jobpath}
            #PBS -l {select}:ncpus={ncpus}:mem={mem}{':ngpus'+str(ngpus) if ngpus else ''}
            #PBS -l walltime={walltime}
            #PBS -J {1000*chunk_idx}-{1000*chunk_idx + len(cmd_list_chunk) - 1}
            cd #PBS_O_WORKDIR
            bash {jobpath}/cmd_$PBS_ARRAY_INDEX.sh
END"""
        # os.system(f"qsub {PBS_SCRIPT}")
        print(PBS_SCRIPT)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python %s FUNCTION_NAME ARG1 ... ARGN" % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

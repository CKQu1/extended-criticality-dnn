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
            {scriptpath} -f theory.wl "PutJacobianLogInvCDF[{alpha100}, {g100}, 0, 1000, 100]"
        """
        for alpha100 in range(100, 201, 5)
        for g100 in range(0, 301, 5)[1:]
        # if alpha100 == 150 and g100 in [150, 200, 300]
    ]
    random.shuffle(cmd_list)
    qsub(cmd_list, Path(".") / "fig", mem="4GB", max_run_subjobs=100, depend_after=True)


def submit_consolidator():
    cmd_list = [
        f"{scriptpath} -f theory.wl SaveJacobianLogAvg[]",
        f"{scriptpath} -f theory.wl SaveLogFPStable[]",
        f"{scriptpath} -f theory.wl SaveNeuralNorms[]",
    ]
    qsub(cmd_list, Path(".") / "fig", "consolidator")


def submit_remaining(min_avg_samples: int = 1000):
    min_avg_samples = int(min_avg_samples)
    cmd_list = []
    data_path = Path(".") / "fig" / "data"
    for alpha100 in range(100, 201, 5):
        for g100 in range(0, 301, 5)[1:]:
            num_avg_samples = 0
            for fname in data_path.glob(f"jaclogavg_{alpha100}_{g100}_0_1000_*.txt"):
                with fname.open() as f:
                    num_avg_samples += int(f.readlines()[-1])
            remaining_samples = min_avg_samples - num_avg_samples
            if remaining_samples > 0:
                cmd_list.append(
                    f"""
                        {scriptpath} -f theory.wl "PutJacobianLogAvg[{alpha100}, {g100}, 0, 1000, {remaining_samples}]"
                    """
                )
                # print(f"{alpha100}, {g100}, {remaining_samples}")
    random.shuffle(cmd_list)
    qsub(cmd_list, Path(".") / "fig", mem="4GB", max_run_subjobs=200, depend_after=True)


def submit_neural_norms():
    cmd_list = []
    data_path = Path(".") / "fig" / "data"
    cmd_list = [
        f"""
            {scriptpath} -f theory.wl "PutNeuralNorm[{alpha100}, {g100}, 0]"
        """
        for alpha100 in range(100, 201, 5)
        for g100 in range(0, 301, 5)[1:]
        if not len(list(data_path.glob(f"logneuralnorm_{alpha100}_{g100}_0_*.txt")))
    ]
    random.shuffle(cmd_list)
    qsub(cmd_list, Path(".") / "fig", mem="4GB", max_run_subjobs=100, depend_after=True)


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
    depend_after=True,
    print_script=False,
):
    path = Path(path)
    label = time.strftime(r"%Y%m%d_%H%M%S")
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
    lastjobid = None
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
        if print_script:
            print(PBS_SCRIPT)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python %s FUNCTION_NAME ARG1 ... ARGN" % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

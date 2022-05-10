import os
import subprocess
import numpy as np

DO_SUBMIT = True
BLOCK = """#!/bin/bash
#PJM -L rscgrp=batch
#PJM -L vnode=1
#PJM -L vnode-core=6
#PJM -L elapse=71:00:00
#PJM -g Q22567
#PJM -N MEET_EXP
#PJM -o output.log
#PJM -j
source $HOME/.bashrc
module load gcc/8.3.0
conda activate mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/erinaldi/.mujoco/mujoco210/bin
export CPATH=/home/erinaldi/OSS/miniconda3/envs/mujoco/include
export PYTHONPATH==$PYTHONPATH:/home/erinaldi/Code/uncertainty-buffer
export OMP_NUM_THREADS=6
export EXECUTABLE=/home/erinaldi/Code/uncertainty-buffer/playground/reinforcement_learning_multi_head.py
echo ${PJM_O_WORKDIR}
"""


def main():
    # loop over parameters for algorithm, buffer and environment
    strategy = ["uniform", "uncertainty", "prioritized"]
    algorithm = ["SAC", "DDPG", "TD3"]
    environment = [
        "Humanoid-v3",
        "Ant-v3",
        "HalfCheetah-v3",
        "Walker2d-v3",
        "InvertedPendulum-v2",
    ]
    # how many different experiments to run
    n_experiments = 2
    n_start = 8

    for buf in strategy:
        for alg in algorithm:
            for env in environment:
                for i in range(n_start,n_start+n_experiments):
                    # create run folder
                    folder_name = f"{alg}_{env}_{buf}_{i}"
                    os.makedirs(folder_name, exist_ok=False)
                    # move into it
                    os.chdir(folder_name)
                    print(f"Moving into folder {os.getcwd()} ...:")
                    # create bash submit script
                    script_name = f"pjrun.sh"
                    with open(script_name, "w") as f:
                        f.write(BLOCK)
                        # pass arguments as strings
                        f.write(
                            f"python $EXECUTABLE --alg {alg} --buffer {buf} --env {env}\n"
                        )

                    if DO_SUBMIT:
                        # submit bash submit script
                        print(
                            subprocess.run(["pjsub", script_name], capture_output=True)
                        )

                    # move back out
                    os.chdir("../")
                    print(f"... moving back to {os.path.basename(os.getcwd())}")


if __name__ == "__main__":
    main()

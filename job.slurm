#!/bin/bash

# If your job will need to use the gpu ressource, add this two lines
#SBATCH -p longq                           # partition name
##SBATCH --gres=gpu:1                      # Necessary to activate the gpu card (The number of GPUs allowed by node is 1)
#SBATCH -N 1                               # number of nodes (for gpu partition you can use 2 nodes max)
#SBATCH -n 40                              # number of cores ( max 44 per node)
#SBATCH --time=48:00:00                    # wall time to finish the job
#SBATCH --job-name PROTEINE_Classification # job name
#SBATCH --output job-%j.log   # output file

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="simlab-cluster"
port=$(( 8004 ))

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.um6p.ma
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# If your job will need to use the gpu ressource, add this line also module purge
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA
module load Anaconda3
source activate pygenomics

# module load OpenMPI/4.0.5-GCC-10.2.0
# conda activate rapids-24.06
# Run Jupyter lab
# module load rapids-24.06
# conda activate rapids-24.06
# Activate the Conda environment
# source activate rapids-24.06

# Run Cmd
python notebook/pyrunner.py > job-output-%j.txt 2>&1

# Run App
#jupyter-lab --no-browser --port=${port} --ip=${node}
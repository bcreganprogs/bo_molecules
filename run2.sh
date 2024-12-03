#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=vv121 # required to send email notifcations - please replace <your_username> with your college log>export PATH=/vol/bitbucket/${USER}/molopt/bin/:$PATH
export PATH=/vol/project/2023/70079/g237007905/bo_molecules/venv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source /vol/project/2023/70079/g237007905/bo_molecules/venv/bin/activate
source /vol/cuda/11.8.0/setup.sh
/usr/bin/nvidia-smi
 
# List of oracle names
declare -a oracles=("troglitazone_rediscovery")
 
# Loop over the list of oracle names and submit jobs
for oracle_name in "${oracles[@]}"
do
    # Modify the config file
    sed -i "s/oracle_name: .*/oracle_name: $oracle_name/" experiments/configs/experiment.yaml
    python main.py
done
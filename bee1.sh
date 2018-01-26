#!/bin/bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:p100:1
#SBATCH -t 1:00:00
#SBATCH --job-name=experiment
#SBATCH --workdir=/pylon5/ci5616p/piperod/
#SBATCH --output=out_2l.out
#SBATCH --error=err_2l.out
# echo commands to stdout
set -x

# move to working directory
cd /pylon5/ci5616p/piperod/pollenlab
#activate workshop enviroment 
source activate workshop
# run script
python /pylon5/ci5616p/piperod/pollelab/shallow_experiment.py

#!/bin/bash
#SBATCH --job-name=run
#SBATCH --time=10-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --output=./cpu_jupyter_log.log
#SBATCH -p rudin

source /home/users/zg78/miniconda3/bin/activate
conda activate tools
jupyter-notebook --ip=0.0.0.0 --port=8808 --no-browser
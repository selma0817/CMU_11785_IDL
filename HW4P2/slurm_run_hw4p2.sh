#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=YIP33@pitt.edu

#SBATCH --output=/ihome/hkarim/yip33/11785/HW4P2/run_logs/%x_%j.out
#SBATCH --error=/ihome/hkarim/yip33/11785/HW4P2/run_logs/%x_%j.err



echo "hw4p2 run"
module load gcc/9.2.0
eval "$(conda shell.bash hook)"
source /ix1/hkarim/yip33/custom_miniconda/bin/activate transformer_env

nvidia-smi
echo "srun"
srun --cpu-bind=none python /ihome/hkarim/yip33/11785/HW4P2/hw4p2_c.py 
echo "finish hw4p2 run high-cutoff"

#export RUN_LOG_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" 
#--run_log_name $RUN_LOG_NAME
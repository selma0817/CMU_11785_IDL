#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=36G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=YIP33@pitt.edu

#SBATCH --output=/ihome/hkarim/yip33/11785/HW4P2/run_logs/%x%j.out
#SBATCH --error=/ihome/hkarim/yip33/11785/HW4P2/run_logs/%x%j.err
export RUN_LOG_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" 

echo "predict run"
module load gcc/9.2.0
# Source conda setup
eval "$(conda shell.bash hook)"
#conda activate ctc_test
source /ix1/hkarim/yip33/custom_miniconda/bin/activate transformer_env

nvidia-smi
echo "srun"
srun python /ihome/hkarim/yip33/11785/HW4P2/prediction.py --run_log_name $RUN_LOG_NAME
echo "finish predict"

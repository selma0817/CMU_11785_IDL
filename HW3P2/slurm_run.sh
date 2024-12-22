#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=36G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --mail-user=YIP33@pitt.edu

#SBATCH --output=/ihome/hkarim/yip33/11785/HW3P2/run_log/%x%j.out
#SBATCH --error=/ihome/hkarim/yip33/11785/HW3P2/run_log/%x%j.err

echo "hw3p2 run"
module load gcc/9.2.0
# Source conda setup
eval "$(conda shell.bash hook)"
conda activate ctc_test

nvidia-smi
echo "srun"
srun python3 /ihome/hkarim/yip33/11785/HW3P2/hw3p2_c.py
echo "finish hw3p2 run early submission"

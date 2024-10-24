#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --mail-user=YIP33@pitt.edu

#SBATCH --output=/ihome/hkarim/yip33/HW2P2/run_log/%x%j.out
#SBATCH --error=/ihome/hkarim/yip33/HW2P2/run_log/%x%j.err


echo "hw2p2 run"
module load cuda/11.8
module load pytorch/2.0.1_cuda11.8
nvidia-smi


#eval "$(conda shell.bash hook)"
python3 hw2p2_c.py
#python3 upload.py
echo "finish hw2p2 run high cut-off submission"

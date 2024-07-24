#!/bin/bash
#SBATCH -t 0-72:4800:00                    # time limit set to 1 minute
#SBATCH --mem=32G                         # reserve 1GB of memory
#SBATCH -J Tutorial                      # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1                             # use 5 tasks
#SBATCH --cpus-per-task=8                # use 1 thread per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --gres=gpu:RTX8000:1 --partition=dcv
#SBATCH --output=/home/dmittal/Desktop/Job_Outputs/test_%j_out.txt         # capture output
#SBATCH --error=/home/dmittal/Desktop/Job_Outputs/test_%j_err.txt          # and error streams


module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate RUL_Prediction
python /home/dmittal/Desktop/CGCD-main/src/CGCD-HAR_GBASELINE_WANDB_ALL.py  --model='tinyhar' --processes $SLURM_NTASKS --threads $SLURM_CPUS_PER_TASK "$@"
# python /home/dmittal/Desktop/CGCD-main/src/edison_paper_implemetation_online.py --processes $SLURM_NTASKS --threads $SLURM_CPUS_PER_TASK "$@"

# edison_paper_implemetation , CGCD-HAR-GBaseline_WANDB, edison_paper_implemetation_online, offline_model, edison_paper_implemetation_online_DeepLSTM, CGCD-HAR-GBaseline_Contrastive_WANDB



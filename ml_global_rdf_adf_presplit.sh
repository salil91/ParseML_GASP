#!/bin/bash
#SBATCH --job-name=ml_global_rdf_adf_presplit
#SBATCH --mail-type=ALL
#SBATCH --mail-use=salil.bavdekar@ufl.edu
#SBATCH --output=ml_global_rdf_adf/job_%j.out
#SBATCH --error=ml_global_rdf_adf/job_%j.err
#SBATCH --account=subhash
#SBATCH --qos=subhash
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00

pwd;
echo 'Job Started: ml_global_rdf_adf_presplit'
date;
nvidia-smi;

echo 'Working directory:' $1
echo 'Percentage for Training Set:' $2
echo 'ML Method:' $3
echo 'Target:' $4

export PATH=/home/salil.bavdekar/.conda/envs/ai_gasp/bin:$PATH
python ml_global_rdf_adf_presplit.py $1 $2 $3 $4

echo 'Done.'
date;

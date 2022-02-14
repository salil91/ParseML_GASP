#!/bin/bash
#SBATCH --job-name=prep_ml_files
#SBATCH --mail-type=ALL
#SBATCH --mail-use=salil.bavdekar@ufl.edu
#SBATCH --output=prep_ml_files/job_%j.out
#SBATCH --error=prep_ml_files/job_%j.err
#SBATCH --account=subhash
#SBATCH --qos=subhash
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --mem=2GB
#SBATCH --time=24:00:00

pwd;
echo 'Job Started: prep_ml_files'
date;

echo 'Working directory:' $1

export PATH=/home/salil.bavdekar/.conda/envs/ai_gasp/bin:$PATH
python prep_ml_files.py $1

echo 'Done.'
date;

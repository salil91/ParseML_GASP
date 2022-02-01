#!/bin/bash
#SBATCH --job-name=intrinsic_hardness_prepped
#SBATCH --mail-type=ALL
#SBATCH --mail-use=salil.bavdekar@ufl.edu
#SBATCH --output=intrinsic_hardness_prepped/job_%j.out
#SBATCH --error=intrinsic_hardness_prepped/job_%j.err
#SBATCH --account=subhash
#SBATCH --qos=subhash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=2GB
#SBATCH --time=24:00:00

pwd;
echo 'Job Started: intrinsic_hardness_prepped'
date;

echo 'Working directory:' $1

# export PATH=/home/salil.bavdekar/.conda/envs/ai_gasp/bin:$PATH
python intrinsic_hardness_prepped.py $1

echo 'Done.'
date;

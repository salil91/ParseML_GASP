#!/bin/bash
#SBATCH --job-name=parse_global_rdf_adf
#SBATCH --mail-type=ALL
#SBATCH --mail-use=salil.bavdekar@ufl.edu
#SBATCH --output=parse_global_rdf_adf/job_%j.out
#SBATCH --error=parse_global_rdf_adf/job_%j.err
#SBATCH --account=subhash
#SBATCH --qos=subhash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

pwd;
echo 'Job Started: parse_global_rdf_adf'
date;

echo 'Working directory:' $1
echo 'Elements:' $2 $3

# export PATH=/home/salil.bavdekar/.conda/envs/ai_gasp/bin:$PATH
python parse_global_rdf_adf.py $1 $2 $3

echo 'Done.'
date;

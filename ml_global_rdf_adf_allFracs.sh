#!/bin/bash

export PATH=/home/salil.bavdekar/.conda/envs/ai_gasp/bin:$PATH

sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 90
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 80
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 70
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 60
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 50
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 40
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 30
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 20
sbatch ml_global_rdf_adf.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ energy KRR 10









#!/bin/bash

bash ml_train_test_split.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70
sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70 SVR energy
sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70 SVR hardness
sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70 KRR energy
sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70 KRR hardness

# bash ml_train_test_split.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_N/garun_phase_diag/relaxations/ 70 SVR energy
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_N/garun_phase_diag/relaxations/ 70 SVR hardness
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_N/garun_phase_diag/relaxations/ 70 KRR energy
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_N/garun_phase_diag/relaxations/ 70 SVR hardness

# bash ml_train_test_split.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/C_N/garun_phase_diag/relaxations/ 70 SVR energy
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/C_N/garun_phase_diag/relaxations/ 70 SVR hardness
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/C_N/garun_phase_diag/relaxations/ 70 KRR energy
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/C_N/garun_phase_diag/relaxations/ 70 KRR hardness

# bash ml_train_test_split.sh /blue/subhash/salil.bavdekar/test/CdTe_Archive/ 70
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ 70 SVR energy
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ 70 SVR hardness
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ 70 KRR energy
# sbatch ml_global_rdf_adf_presplit.sh /blue/subhash/salil.bavdekar/GASP/Si_C/garun_phase_diag/relaxations/ 70 KRR hardness



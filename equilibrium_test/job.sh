#!/bin/sh
#PBS -S /bin/sh
#PBS -N eq_test
#PBS -j oe
#PBS -l nodes=1:ppn=128,walltime=24:00:00
#PBS -m abe
#PBS -M currodri@gmail.com

module purge
module load inteloneapi/2022.1.2

cd /home/currodri/Codes/DustRAMSES/equilibrium_test
mpirun -genv FI_PROVIDER=tcp -np 128 ./ramses3d params_eq0_Zm0.nml
mpirun -genv FI_PROVIDER=tcp -np 128 ./ramses3d params_eq1_Zm0.nml >& run_00001.log 
exit 0

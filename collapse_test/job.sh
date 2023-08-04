#!/bin/sh
#PBS -S /bin/sh
#PBS -N nocou_noratd
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=01:00:00
#PBS -m abe
#PBS -M currodri@gmail.com

module purge
module load inteloneapi/2022.1.2

cd /home/currodri/Codes/DustRAMSES/collapse_test 
mpirun -genv FI_PROVIDER=tcp -np 1 ./ramses3d nml_zm1.nml >& zm1_test.log 
exit 0

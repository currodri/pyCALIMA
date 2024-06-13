#!/bin/sh
#PBS -S /bin/sh
#PBS -N rates
#PBS -j oe
#PBS -l nodes=1:ppn=32,walltime=01:00:00
#PBS -m abe
#PBS -M currodri@gmail.com

source /home/currodri/.bashrc

cd /home/currodri/Codes/DustRAMSES 
PYTHON="/home/currodri/miniconda3/bin/python"
export OPENMP_NUM_THREADS=32
python compute_peh_rates.py
exit 0

#!/bin/bash
#SBATCH --job-name=rates
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:02:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=currodri@gmail.com

source /home/currodri/.bashrc

cd /home/currodri/Codes/DustRAMSES 
PYTHON="/home/currodri/miniconda3/bin/python"
export OPENMP_NUM_THREADS=32
python export_dust_sputtering_yields.py
# python test_dust_sputtering.py
exit 0
#python test-sputtering.py
#exit 0
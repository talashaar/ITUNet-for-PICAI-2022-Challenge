#!/bin/bash
#SBATCH --job-name=picai_build
#SBATCH --output=/HOME/alshaart/logs/build_ADCavg_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/build_ADCavg_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --mail-user=tala.alshaar@unifr.ch

bash /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/build.sh ;
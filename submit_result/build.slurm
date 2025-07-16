#!/bin/bash
#SBATCH --job-name=picai_build_submition_ADCavg
#SBATCH --output=/HOME/alshaart/logs/build_submition_ADCavg_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/build_submition_ADCavg_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --mail-user=tala.alshaar@unifr.ch

bash /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/submit_result/build.sh  ;
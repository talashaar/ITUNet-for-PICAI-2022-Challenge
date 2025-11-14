#!/bin/bash
#SBATCH --job-name=picai_build_submition_HBVzero
#SBATCH --output=/HOME/alshaart/logs/build_submition_HBVzero_4folds_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/build_submition_HBVzero_4folds_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --mail-user=tala.alshaar@unifr.ch

bash /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/submit_result/build.sh  ;
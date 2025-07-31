#!/bin/bash
#SBATCH --job-name=picai_test_submition_T2Wavg
#SBATCH --output=/HOME/alshaart/logs/test_submition_T2Wavg_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/test_submition_T2Wavg_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --mail-user=tala.alshaar@unifr.ch

docker load -i /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/submit_result/picai_baseline_unet_processor.tar ;
docker load -i /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/picai_trainer_latest.tar ;

bash /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/submit_result/test.sh  ;
#!/bin/bash
#SBATCH --job-name=picai_supervisedclass
#SBATCH --output=/HOME/alshaart/logs/supervisedclass_new_full_4folds_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/supervisedclass_new_full_4folds_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tala.alshaar@unifr.ch

docker load -i /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/picai_trainer_latest_new_full.tar ;

docker login nvcr.io -u '$oauthtoken' -p 'ZmMzMmlkcjN1bm92Nm5xbGRva3VhOTJjMW46MGUxZjFmMzgtNjE4Mi00ZTgyLTg3NTctODk3ZTk0NGQ0MzBl' && 
docker run --rm --gpus all -v /HOME/alshaart/Data:/Data  -v /HOME/alshaart/output:/output --shm-size=16g \
  yukiyaumimi/picai_trainer:latest \
  python3 /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/step_1_supervised_classification.py \
   --workdir /Workdirnewfull4folds/ \
   --preprocesseddir /output/new_full_4folds/ \
   --outputdir /output/step1_classification_new_full_4folds \
   --checkpointsdir /Data/checkpoint/
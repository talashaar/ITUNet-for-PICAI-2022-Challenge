#!/bin/bash
#SBATCH --job-name=picai_preprocess
#SBATCH --output=/HOME/alshaart/logs/picai_preprocess_new_ADCzero_4folds_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/picai_preprocess_new_ADCzero_4folds_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --mail-user=tala.alshaar@unifr.ch

docker info | grep -i runtime

docker load -i /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/picai_trainer_latest__new_ADCzero.tar ;

docker login nvcr.io -u '$oauthtoken' -p 'ZmMzMmlkcjN1bm92Nm5xbGRva3VhOTJjMW46MGUxZjFmMzgtNjE4Mi00ZTgyLTg3NTctODk3ZTk0NGQ0MzBl' && 
docker run --rm -v /HOME/alshaart/Data:/Data  -v /HOME/alshaart/output:/output --shm-size=16g \
  yukiyaumimi/picai_trainer:latest \
  python3 /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/preprocess/preprocess.py \
  --workdir /WorkdirnewADCzero4folds/ \
  --imagesdir /Data/picai_data/images_path_4folds \
  --labelsdir /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/picai_labels \
  --outputdir /output/ADC_Zero_new_4folds
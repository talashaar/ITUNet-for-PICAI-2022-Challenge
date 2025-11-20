#!/bin/bash
#SBATCH --job-name=picai_semisupervisedseg
#SBATCH --output=/HOME/alshaart/logs/semisupervisedseg_T2Wzero_4folds_logstdout.txt
#SBATCH --error=/HOME/alshaart/logs/semisupervisedseg_T2Wzero_4folds_logstderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --nodelist=diufrd204
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tala.alshaar@unifr.ch

docker load -i /HOME/alshaart/ITUNet-for-PICAI-2022-Challenge/picai_trainer_latest_t2wzero.tar ;

docker login nvcr.io -u '$oauthtoken' -p 'ZmMzMmlkcjN1bm92Nm5xbGRva3VhOTJjMW46MGUxZjFmMzgtNjE4Mi00ZTgyLTg3NTctODk3ZTk0NGQ0MzBl' && 
docker run --rm --gpus all -v /HOME/alshaart/Data:/Data  -v /HOME/alshaart/output:/output --shm-size=16g \
  yukiyaumimi/picai_trainer:latest \
  python3 /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/step_4_semi_supervised_segmentation.py \
   --workdir /WorkdirT2Wzero4folds/ \
   --preprocesseddir /output/step3_predict_T2Wzero_4folds/ \
   --outputdir /output/step4_semi_T2Wzero_4folds/ \
   --checkpointsdir /output/step4_semi_T2Wzero_4folds/checkpoints
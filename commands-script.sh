# preprocessing
python3 preprocess.py --workdir /WorkdirHBVzero/ --imagesdir /Data/picai_data/images_path --labelsdir /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/picai_labels --outputdir /output/HBV_Zero

# supervised classifications
python3 step_1_supervised_classification.py --workdir /WorkdirHBVzero/ --preprocesseddir /output/HBV_Zero/ --outputdir /output/step1_classification_HBVZero --checkpointsdir /Data/checkpoint/

# supervised segmentation
python3 step_2_supervised_segmentation.py --workdir /WorkdirHBVzero/ --preprocesseddir /output/HBV_Zero/ --outputdir /output/step2_segmentation_HBVZero --checkpointsdir /Data/checkpoint/

# predict pseudo labels
python3 step_3_prepare_detection_data.py --workdir /WorkdirHBVzero/ --preprocesseddir /output/HBV_Zero/ --supervisedweightsdir /output/ --outputdir /output/step3_predict_HBVZero

# semi supervised segmentation
python3 step_4_semi_supervised_segmentation.py --workdir /WorkdirHBVzero/ --preprocesseddir /output/step3_predict_HBVZero/ --outputdir /output/step4_semi_HBVZero/ --checkpointsdir /output/step4_semi_HBVZero/checkpoints
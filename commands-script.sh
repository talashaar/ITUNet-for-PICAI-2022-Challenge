# preprocessing
python3 preprocess.py --workdir /WorkdirT2Wzero/ --imagesdir /Data/picai_data/images_path --labelsdir /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/picai_labels --outputdir /output/T2W_Zero

# supervised classifications
python3 step_1_supervised_classification.py --workdir /WorkdirT2Wzero/ --preprocesseddir /output/T2W_Zero/ --outputdir /output/step1_classification_T2WZero --checkpointsdir /Data/checkpoint/

# supervised segmentation
python3 step_2_supervised_segmentation.py --workdir /WorkdirT2Wzero/ --preprocesseddir /output/T2W_Zero/ --outputdir /output/step2_segmentation_T2WZero --checkpointsdir /Data/checkpoint/

# predict pseudo labels
python3 step_3_prepare_detection_data.py --workdir /WorkdirT2Wzero/ --preprocesseddir /output/T2W_Zero/ --supervisedweightsdir /output/ --outputdir /output/step3_predict_T2WZero

# semi supervised segmentation
python3 step_4_semi_supervised_segmentation.py --workdir /WorkdirT2Wzero/ --preprocesseddir /output/step3_predict_T2WZero/ --outputdir /output/step4_semi_T2WZero/ --checkpointsdir /output/step4_semi_T2WZero/checkpoints
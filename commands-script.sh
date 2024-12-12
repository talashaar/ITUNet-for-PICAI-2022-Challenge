# preprocessing
python3 preprocess.py --workdir /WorkdirADCzero/ --imagesdir /Data/picai_data/images_path --labelsdir /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/picai_labels --outputdir /output/ADC_Zero

# supervised classifications
python3 step_1_supervised_classification.py --workdir /WorkdirADCzero/ --preprocesseddir /output/ADC_Zero/ --outputdir /output/step1_classification_ADCZero --checkpointsdir /Data/checkpoint/

# supervised segmentaion
python3 step_2_supervised_segmentation.py --workdir /WorkdirADCzero/ --preprocesseddir /output/ADC_Zero/ --outputdir /output/step2_segmentation_ADCZero --checkpointsdir /Data/checkpoint/

# predict pseudo labels
python3 step_3_prepare_detection_data.py --workdir /WorkdirADCzero/ --preprocesseddir /output/ADC_Zero/ --supervisedweightsdir /output/ --outputdir /output/step3_predict_ADCZero

# semi supervised segmentation
python3 step_4_semi_supervised_segmentation.py --workdir /WorkdirADCzero/ --preprocesseddir /output/step3_predict_ADCZero/ --outputdir /output/step4_semi_ADCZero/ --checkpointsdir /output/step4_semi_ADCZero/checkpoints
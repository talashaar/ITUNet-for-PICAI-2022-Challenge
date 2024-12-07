# preprocessing
python3 preprocess.py --workdir /ResamplingWorkdir/ --imagesdir /Data/picai_data/images_path --labelsdir /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/picai_labels --outputdir /output/preprocess_Resample_Final

# supervised classifications
python3 step_1_supervised_classification.py --workdir /ResamplingWorkdir/ --preprocesseddir /output/preprocess_Resample_Final/ --outputdir /output/step1_classification_resampling --checkpointsdir /Data/checkpoint/

# supervised segmentation
python3 step_2_supervised_segmentation.py --workdir /ResamplingWorkdir/ --preprocesseddir /output/preprocess_Resample_Final/ --outputdir /output/step2_segmentation_resampling --checkpointsdir /Data/checkpoint/

# predict pseudo labels
python3 step_3_prepare_detection_data.py --workdir /ResamplingWorkdir/ --preprocesseddir /output/preprocess_Resample_Final/ --supervisedweightsdir /output/ --outputdir /output/step3_predict_resample

# semi supervised segmentation
python3 step_4_semi_supervised_segmentation.py --workdir /ResamplingWorkdir/ --preprocesseddir /output/step3_predict_resampling/ --outputdir /output/step4_semi_resampling/ --checkpointsdir /output/step4_semi_resampling/checkpoints
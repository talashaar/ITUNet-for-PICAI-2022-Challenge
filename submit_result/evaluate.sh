#!/usr/bin/env bash

#./build.sh
# change /HOME/alshaart/testdata/ to the directory where weights are 
# change/HOME/alshaart/testoutput to the directory where the outputs are to be written


for i in $(find "/HOME/alshaart/Data/picai_data/images_path/fold0/" -type d -maxdepth 1 -printf '%f '); do 
    docker run  --gpus='"device=2"' --rm \
        -v /HOME/alshaart/testdata_T2Wzero/:/model/ \
        -v /HOME/alshaart/testoutput_T2Wzero:/output/ \
        -v /HOME/alshaart/Data/picai_data/images_path/fold0/:/input/ \
        picai_baseline_unet_processor_t2wzero python3 process.py --caseid $i;
done
#!/usr/bin/env bash

#./build.sh
# change /HOME/alshaart/testdata/ to the directory where weights are 
# change/HOME/alshaart/testoutput to the directory where the outputs are to be written


for i in $(find "/HOME/alshaart/Data/picai_data/images_path/fold4/" -type d -maxdepth 1 -printf '%f '); do 
    docker run  --gpus='"device=2"' --rm \
        -v /HOME/alshaart/testdata_newfull/:/model/ \
        -v /HOME/alshaart/testoutput_newfull:/output/ \
        -v /HOME/alshaart/Data/picai_data/images_path/fold4/:/input/ \
        picai_baseline_unet_processor_newfull python3 process.py --caseid $i;
done
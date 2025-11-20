#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t picai_baseline_unet_processor_t2wzero

docker save -o picai_baseline_unet_processor_t2wzero.tar picai_baseline_unet_processor_t2wzero
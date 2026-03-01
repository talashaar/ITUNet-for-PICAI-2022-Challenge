#!/usr/bin/env bash

docker build . \
  --tag yukiyaumimi/picai_trainer:latest

docker save -o picai_trainer_latest_new_full.tar yukiyaumimi/picai_trainer:latest
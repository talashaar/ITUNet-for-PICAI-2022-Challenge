#!/usr/bin/env bash

docker build . \
  --tag yukiyaumimi/picai_trainer:latest

docker save -o picai_trainer_latest_t2wzero.tar yukiyaumimi/picai_trainer:latest
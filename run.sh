#!/bin/bash

accelerate launch \
    --num_processes 0 \
    --num_machines 1 \
    --mixed_precision 'no' \
    --dynamo_backend 'no' \
    run.py \
    --config_path 'config/bart-base/v11/asr-vocab-noise-test.yaml' 
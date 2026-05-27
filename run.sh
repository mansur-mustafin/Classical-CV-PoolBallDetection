#!/bin/bash

python3 src/task2.py \
    --image_dir_train data/dataset/train/ \
    --image_dir_val data/dataset/valid/ \
    --image_dir_test data/dataset/test/ \
    --output_json data/json/output.json \
    --model_name resnet18

#!/bin/bash

python3 src/main.py \
    --input_json data/json/input.json \
    --output-json data/json/output.json \
    --data-dir data/ \
    --topdown-dir data/result_set/

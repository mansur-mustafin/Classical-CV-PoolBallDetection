#!/bin/bash

python3 src/main.py \
    --input_json data/json/input.json \
    --output-json data/json/output.json \
    --data-dir data/ \
    --topdown-dir data/result_set/ \
    --n_jobs 8

python3 src/main.py \
    --input_json data/json/input_v3_our_table.json \
    --output-json data/json/output_v3_our_table.json \
    --data-dir data/ \
    --topdown-dir data/result_set_v3_our_table/ \
    --n_jobs 8

# python3 src/main.py \
#     --input_json data/json/input_v3_processed.json \
#     --output-json data/json/output_v3_processed.json \
#     --data-dir data/ \
#     --topdown-dir data/result_set_v3_processed/
#     --n_jobs 8

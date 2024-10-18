#!/bin/bash

export MODEL_PATH= #model path
export SOURCE_DIR= #cg_image dir
export OUTPUT_DIR= #output_dir
export NEGATIVE_EMBEDDING_DIR= #Negative domain embedding

CUDA_VISIBLE_DEVICES=0 python ./Realistic_translation.py \
    --model_path=$MODEL_PATH \
    --source_image_path=$SOURCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --negative_embedding_dir=$NEGATIVE_EMBEDDING_DIR \
    --source_prompt='' \
    --target_prompt='' \
    --replace_steps_ratio=0.9 \
    --denoising_strength=0.3 \
    --cfg_scale=7.5 \
    --attn_replace_layers=256 \
    --inversion_as_start \
    --use_negEmbedding
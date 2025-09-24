#!/bin/bash

CHECKPOINT="/data/pr/results_multimodal/004-DiTMultimodal-XL-2/checkpoints/0100000.pt"
NUM_SAMPLES=500                            
BATCH_SIZE=2   
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=12350 /data/pr/DiT_AIVCdiff_lzr/DiT_syn/sample_ddp.py \
    --model "DiTMultimodal-XL/2" \
    --vae ema \
    --sample-dir drug_samples \
    --per-proc-batch-size $BATCH_SIZE \
    --num-fid-samples $NUM_SAMPLES \
    --num-sampling-steps 1000 \
    --global-seed 42 \
    --ckpt $CHECKPOINT \
    --h5ad_path /data/pr/DiT_AIVCdiff/pr_tutorial/DiT_input_512_image_rna.h5ad \
    --num-drug-classes 98 \
    --num-rna-features 977 \
    --fp_size 1024 \
    --drug-split 1
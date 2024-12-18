#!/bin/bash

# 激活conda环境
source /root/.bashrc
conda activate meta_portrait_base

case "$1" in
    "inference")
        cd base_model
        python inference.py --save_dir result --config config/meta_portrait_256_eval.yaml --ckpt checkpoint/ckpt_base.pth.tar
        ;;
    "train_warp")
        cd base_model
        python main.py --config config/meta_portrait_256_pretrain_warp.yaml --fp16 --stage Warp --task Pretrain
        ;;
    "train_full")
        cd base_model
        python main.py --config config/meta_portrait_256_pretrain_full.yaml --fp16 --stage Full --task Pretrain
        ;;
    "sr")
        cd sr_model
        python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/test.py -opt options/test/same_id_demo.yml --launcher pytorch
        ;;
    *)
        exec "$@"
        ;;
esac
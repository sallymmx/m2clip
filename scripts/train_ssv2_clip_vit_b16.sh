#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
dir=output_dir/ssv2/clip_vit_base_patch16_multimodal_adapter24x384_cross_adapter_4gpu_no_motion
mkdir -p ${dir}
python -m torch.distributed.launch --nproc_per_node 4 --master_port 47700 main.py \
    --model clip_vit_base_patch16_multimodal_adapter24x384 \
    --save_dir ${dir} \
    --auto_resume --auto_remove \
    --dataset ssv2 \
    --num_frames 8 \
    --sampling_rate 0 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --batch_size 32 \
    --label_csv 'lists/something_v2_labels.csv' \
    --epochs 50 \
    --warmup_epochs 2 \
    --eval_freq 2 \
    2>&1|tee ${dir}/$now.log
    # &CUDA_VISIBLE_DEVICES="0,1"

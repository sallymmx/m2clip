#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
dir=output_dir/k400/clip_vit_base_patch16_multimodal_adapter12x384_best_new
mkdir -p ${dir}
CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node 2 --master_port 47708 main.py \
    --model clip_vit_base_patch16_multimodal_adapter12x384 \
    --save_dir ${dir} \
    --auto_resume --auto_remove \
    --dataset k400 \
    --num_frames 8 \
    --sampling_rate 16 \
    --resize_type random_short_side_scale_jitter \
    --scale_range 1.0 1.15 \
    --num_spatial_views 4 \
    --num_temporal_views 3 \
    --label_csv 'lists/kinetics_400_labels.csv' \
    --mlm_label 'lists/k400_mlm_lables.txt' \
    --mirror \
    --batch_size 36 \
    --epochs 12 \
    --warmup_epochs 2 \
    --eval_freq 12 \
     2>&1|tee ${dir}/$now.log

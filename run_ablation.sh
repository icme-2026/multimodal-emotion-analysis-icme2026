#!/usr/bin/env bash
set -e

COMMON_ARGS="
  --data_dir datasets/MVSA_Single
  --train_data_dir datasets/MVSA_Single
  --test_data_dir datasets/MVSA_Single
  --gpu 0
  --save_name 'adamw_lr5e-5_T1.0_p095_lu05_rp01'
  --optim AdamW --lr 5e-5 --weight_decay 1e-4
  --epoch 150 --num_train_iter 512
  --batch_size 2 --uratio 4 --num_labels 600
  --T 1.0
  --p_cutoff 0.95
  --ulb_loss_ratio 0.5
  --warmup_ratio 0.10
  --clip 1.0
  --patience 40 --seed 1 --deterministic True --num_workers 0 --overwrite
  --ulb_rampup_ratio 0.1
  --enable_motivation_log True
  --motivation_samples_per_batch 32
  --motivation_flush_size 2048
"

# 两两消融：一次关闭两个模块，其余保持开启
declare -A EXTRA_ARGS=(
  ["woCPL_MCO"]="--enable_cpl False --enable_mco False --enable_drr True"
  ["woCPL_DRR"]="--enable_cpl False --enable_mco True --enable_drr False"
  ["woMCO_DRR"]="--enable_cpl True --enable_mco False --enable_drr False"
)

for mode in "${!EXTRA_ARGS[@]}"; do
  echo ">>> Running mode: ${mode}"
  python main.py \
    $COMMON_ARGS \
    --save_dir "./SGDR.11_22_seed1/n600_${mode}" \
    ${EXTRA_ARGS[$mode]}
done

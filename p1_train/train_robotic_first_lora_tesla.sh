
# bash train_robotic_first_lora_tesla.sh


# 训练测试使用的机器人外观lora（真实世界视频数据）
#!/usr/bin/env bash
set -e

# ---- 1. 硬件与分布设置 ----
# 如果你想用单卡，比如 2 号卡：
GPU_IDS="5"
NUM_PROCESSES=1
DIST_TYPE="no" # 单卡填 no，多卡填 multi_gpu

# 如果你想用多卡，比如 1,2,4,6 号卡：
# GPU_IDS="1,2,4,6"
# NUM_PROCESSES=4
# DIST_TYPE="multi_gpu"

# ---- 2. 训练超参 ----
# ROBOT_NAME="passive-marker-man" 
ROBOT_NAME="tesla-optimus" 
DATA_ROOT=""
META_CSV="/opt/liblibai-models/user-workspace2/users/dxy/robotic/data_test/${ROBOT_NAME}.csv"
OUT="/opt/liblibai-models/user-workspace2/users/dxy/robotic/DiffSynth-Studio/first_phrase_output/test_real_world/${ROBOT_NAME}/continue"

LR=1e-4
EPOCHS=200
RANK=64

# 环境修复：防止显存碎片化和残留
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 强制指定可见显卡，这比 accelerate 内部指定更彻底
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# ---- 3. 构建命令 (处理 Flag 冲突) ----
CMD="accelerate launch --num_processes $NUM_PROCESSES --gpu_ids all"

# 如果是多卡训练，才加上 --multi_gpu 开关
if [ "$DIST_TYPE" == "multi_gpu" ]; then
    CMD="$CMD --multi_gpu"
fi

# ---- 3. 启动命令 ----
# 注意：我们用参数覆盖了 accelerate 的默认 config       480  832
$CMD examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$META_CSV" \
  --height 736 --width 1280 --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate $LR \
  --num_epochs $EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank $RANK \
  --use_gradient_checkpointing_offload \
  --save_steps 1000 \
  --lora_checkpoint "/opt/liblibai-models/user-workspace2/users/dxy/robotic/DiffSynth-Studio/first_phrase_output/test_real_world/tesla-optimus/step-6000.safetensors"
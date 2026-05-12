# #!/usr/bin/env bash
# set -euo pipefail

# # === 1. 基础设置 ===
# export PYTHONPATH="$(pwd)"
# BASE_CONFIG="configs/train_p2_h2r_all.yaml"
# SEED=1342
# CUDA_DEVICES="2,3"

# # P1 LoRA 库位置
# P1_LIBRARY_ROOT="/opt/liblibai-models/user-workspace2/users/dxy/robotic/InteractionVideo/p1_lora_library/DS"

# # 数据集根目录 (父级)
# DATASET_BASE="/opt/liblibai-models/user-workspace2/users/dxy/fyp/WorldWander/struc-data" # <-- 修改为你截图里的父目录路径

# # Human 视频永远是这个，不随 category 变
# HUMAN_ROOT="${DATASET_BASE}/Remy-mixamo"

# # 定义参考图的根目录
# REF_IMG_BASE="/opt/liblibai-models/user-workspace2/users/dxy/robotic/InteractionVideo/robot_ref_img"

# # === 2. 循环训练策略 ===
# TOTAL_TARGET_STEPS=10000  # 总目标步数：10000步停止
# STEP_PER_ROUND=50        # 每次换 LoRA 跑 50 步

# # === 3. 训练类别列表 ===
# CATEGORIES=(
#   unitree-h1
#   unitree-g1
#   fourier-gr2
#   fourier-gr3
#   tesla-optimus
#   X-Bot
#   Y-Bot
# )
# NUM_CATS=${#CATEGORIES[@]} # 类别总数

# # === 工具函数 ===

# base_dir_for_category () {
#   local cat="$1"
#   # 对应 YAML output_root
#   echo "/opt/liblibai-models/user-workspace2/users/dxy/robotic/second_phrase_output/train_ec_01/${cat}"
# }

# get_p1_lora_path () {
#   local cat="$1"
#   echo "${P1_LIBRARY_ROOT}/${cat}.ckpt"
# }

# # 查找指定类别目录下，步数最大的 ckpt
# find_best_ckpt_for_category () {
#   local cat="$1"
#   local base; base="$(base_dir_for_category "${cat}")"
#   local -a search_dirs=("${base}/checkpoints_cil" "${base}/checkpoints")
#   local best=""
#   local best_step=-1

#   for d in "${search_dirs[@]}"; do
#     [[ -d "$d" ]] || continue
#     shopt -s nullglob
#     for f in "$d"/*.ckpt; do
#       [[ -e "$f" ]] || continue
#       local fname; fname="$(basename "$f")"
#       local step=-1
#       if [[ "$fname" =~ step=([0-9]+)\.ckpt$ ]]; then
#         step="${BASH_REMATCH[1]}"
#       elif [[ "$fname" == "last.ckpt" ]]; then
#         step=2147483647
#       fi
#       if (( step > best_step )); then
#         best_step="$step"
#         best="$f"
#       fi
#     done
#     shopt -u nullglob
#   done

#   [[ -n "$best" ]] && { echo "$best"; return 0; }
#   return 1
# }

# make_overridden_config () {
#   local in_cfg="$1"
#   local out_cfg="$2"
#   local new_steps="$3"
#   mkdir -p "$(dirname "$out_cfg")"
#   sed -E "s/^([[:space:]]*max_steps:[[:space:]]*)[0-9]+/\\1${new_steps}/" "$in_cfg" > "$out_cfg"
# }

# # === 4. 主训练循环 ===

# echo "Plan: Train cyclically until ${TOTAL_TARGET_STEPS} steps."
# echo "Step size: ${STEP_PER_ROUND} steps per task."
# echo "Total Categories: ${NUM_CATS}"
# echo

# TMP_CFG_DIR=".run_configs_cyclic"
# mkdir -p "${TMP_CFG_DIR}"

# # ==========================================
# # [新增] 自动断点恢复逻辑
# # ==========================================
# echo "Checking for existing progress..."

# max_found_step=-1
# found_category=""
# found_ckpt=""

# # 遍历所有类别，寻找步数最大的 Checkpoint
# for cat in "${CATEGORIES[@]}"; do
#   # 利用现有的查找函数
#   ckpt=$(find_best_ckpt_for_category "${cat}") || true
  
#   if [[ -n "$ckpt" ]]; then
#     # 从文件名提取步数 (假设文件名格式为 step=1234.ckpt)
#     fname=$(basename "$ckpt")
#     if [[ "$fname" =~ step=([0-9]+)\.ckpt$ ]]; then
#       step="${BASH_REMATCH[1]}"
      
#       if (( step > max_found_step )); then
#         max_found_step=$step
#         found_category=$cat
#         found_ckpt=$ckpt
#       fi
#     fi
#   fi
# done

# if (( max_found_step > 0 )); then
#   echo "Found previous checkpoint:"
#   echo "  Category: ${found_category}"
#   echo "  Step:     ${max_found_step}"
#   echo "  Path:     ${found_ckpt}"
#   echo "Resuming training..."
  
#   # 恢复 loop_idx
#   # 公式：current_step = (loop_idx + 1) * 50
#   # 所以：loop_idx = (current_step / 50)
#   # 注意：这里恢复出的 loop_idx 是“已经完成的轮次”，所以下一轮就是 loop_idx
#   # 例如：找到了 step=50，说明第0轮(loop_idx=0)已完成。
#   #       我们应该从 loop_idx=1 开始。
#   #       50 / 50 = 1，正好对应下一轮的 idx。
  
#   loop_idx=$(( max_found_step / STEP_PER_ROUND ))
  
#   # 恢复 prev_category (为了下一轮能找到接力棒)
#   prev_category="${found_category}"
  
# else
#   echo "No valid checkpoint found. Starting from scratch."
#   loop_idx=0
#   prev_category=""
# fi

# echo "Starting loop from index: ${loop_idx}"
# echo "---------------------------------------------------"
# # ==========================================

# while true; do
#   # 1. 计算当前这一轮的目标步数
#   # loop_idx 从 0 开始
#   # current_target_steps = (0+1)*50 = 50, (1+1)*50=100 ...
#   current_target_steps=$(( (loop_idx + 1) * STEP_PER_ROUND ))

#   # 2. 检查是否达到总目标 10000 步
#   if (( current_target_steps > TOTAL_TARGET_STEPS )); then
#     echo "=== Reached target steps (${TOTAL_TARGET_STEPS}). Stopping. ==="
#     break
#   fi

#   # === [新增] 计算动态随机种子 ===
#   # 基础种子 + 当前循环次数，确保每一轮的数据 shuffle 顺序都不一样
#   CURRENT_SEED=$(( SEED + loop_idx ))   # <--- 修改：计算新种子

#   # 3. 确定当前类别 (取模循环)
#   # 比如有9个类别，idx=9时，9%9=0 (回到第一个)
#   cat_idx=$(( loop_idx % NUM_CATS ))
#   category="${CATEGORIES[$cat_idx]}"
#   export CATEGORY="${category}" # 供 YAML 读取

#   # 4. 获取 P1 路径
#   p1_path=$(get_p1_lora_path "${category}")
#   if [[ ! -f "${p1_path}" ]]; then
#     echo "ERROR: P1 LoRA not found for ${category} at: ${p1_path}"
#     # exit 1 
#   fi

#    # === [修改] 动态构建参考图路径 ===
#   # 假设图片名就是 category.jpg (如 fourier-gr2.jpg)
#   # 如果有的格式是 png，有的是 jpg，你可以写个简单的查找逻辑，或者统一转成 jpg
#   REF_IMG_PATH="${REF_IMG_BASE}/${category}.jpg"
  
#   # 检查一下图片是否存在
#   if [[ ! -f "${REF_IMG_PATH}" ]]; then
#       echo "[WARN] Reference image not found: ${REF_IMG_PATH}. Training will proceed with ZERO first frame."
#       REF_IMG_PATH="" # 设为空，代码内部会处理为 None
#   fi

#   # 5. 生成临时 Config (设置 max_steps)
#   ov_cfg="${TMP_CFG_DIR}/step${current_target_steps}_${category}.yaml"
#   make_overridden_config "${BASE_CONFIG}" "${ov_cfg}" "${current_target_steps}"

#   # 6. 确定接力 Checkpoint
#   if [[ ${loop_idx} -eq 0 ]]; then
#     # 第一轮第一次：从头开始
#     ckpt_cli=(--ckpt_path "")
#     echo "[$(date '+%T')] Round ${loop_idx}: ${category} (FRESH START)"
#     echo "  > Target Steps: ${current_target_steps}"
#   else
#     # 后续轮次：去“上一个类别”的目录下找最新的 ckpt
#     # 注意：prev_category 变量在循环底部更新
#     prev_ckpt="$(find_best_ckpt_for_category "${prev_category}")" || {
#       echo "ERROR: Could not find checkpoint from previous round (${prev_category})"
#       exit 1
#     }
#     ckpt_cli=(--ckpt_path "${prev_ckpt}")
#     echo "[$(date '+%T')] Round ${loop_idx}: ${category} (CONTINUE)"
#     echo "  > Target Steps: ${current_target_steps}"
#     echo "  > Resuming from: ${prev_ckpt}"
#   fi

#   # 动态拼接 Robot 路径
#   # 对应你截图结构: robot-videos/fourier-gr2
#   ROBOT_ROOT="${DATASET_BASE}/robot-videos/${category}"
  
#   # 首帧路径 (如果不使用首帧，或者首帧路径不同，请在这里修改)
#   # 假设首帧跟机器人视频放在一起，或者是另一个文件夹？
#   # 这里暂时设为空，如果需要请填写
#   # FIRST_ROOT="" 
#   # 或者: FIRST_ROOT="${DATASET_BASE}/robot-videos/${category}/first_frames"

#   # 7. 执行训练
#   CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
#   python src/wan2_trainer_p2_with_first_frame_ec.py \
#     --config="${ov_cfg}" \
#     --seed="${CURRENT_SEED}" \
#     --p1_model="${p1_path}" \
#     dataset.video_root="${HUMAN_ROOT}" \
#     dataset.video_root2="${ROBOT_ROOT}" \
#     dataset.first_root="${REF_IMG_PATH}" \
#     "${ckpt_cli[@]}"

#   # 8. 更新状态，准备下一轮
#   prev_category="${category}"
#   loop_idx=$(( loop_idx + 1 ))

#   echo "---------------------------------------------------"
# done

# echo "All Done."



#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Train P2 Human-to-Robot (JSONL Version)
# ==============================================================================

# === 1. 基础设置 ===
export PYTHONPATH="$(pwd)"
BASE_CONFIG="configs/train_p2_h2r_all.yaml"
SEED=1342
CUDA_DEVICES="0,1,2,3"  # 根据你的实际显卡情况修改

# P1 LoRA 库位置 (保持不变)
P1_LIBRARY_ROOT="./p1_lora_library/DS_new"

# [修改] JSONL 文件所在的文件夹
# 假设你的目录结构是项目根目录下有一个 json_outputs 文件夹
# 建议写绝对路径以防万一，或者保持相对路径
JSONL_BASE_DIR="./data/second_phrase/json_outputs"

# === 2. 循环训练策略 ===
TOTAL_TARGET_STEPS=20000  # 总目标步数
STEP_PER_ROUND=50         # 每次换 LoRA 跑 50 步

# === 3. 训练类别列表 ===
# 这里的名字必须与你的文件名 data_{NAME}.jsonl 严格对应
# 根据你的截图，我整理了以下列表：
CATEGORIES=(
  unitree-h1
  fourier-gr2
  fourier-gr3
  tesla-optimus
  X-Bot
  Y-Bot
  passive-marker-man
  Remy-mixamo
  dummy02
  # 如果需要训练 dummy02 或其他角色，请在这里添加，例如:
  # unitree-g1
)
NUM_CATS=${#CATEGORIES[@]}

# === 工具函数 ===

base_dir_for_category () {
  local cat="$1"
  # 对应 YAML output_root，确保这里和 yaml 里的路径一致
  echo "./second_phrase_output/train_ec_new_01/${cat}"
}

get_p1_lora_path () {
  local cat="$1"
  echo "${P1_LIBRARY_ROOT}/${cat}.ckpt"
}

# 查找指定类别目录下，步数最大的 ckpt
find_best_ckpt_for_category () {
  local cat="$1"
  local base; base="$(base_dir_for_category "${cat}")"
  local -a search_dirs=("${base}/checkpoints_cil" "${base}/checkpoints")
  local best=""
  local best_step=-1

  for d in "${search_dirs[@]}"; do
    [[ -d "$d" ]] || continue
    shopt -s nullglob
    for f in "$d"/*.ckpt; do
      [[ -e "$f" ]] || continue
      local fname; fname="$(basename "$f")"
      local step=-1
      if [[ "$fname" =~ step=([0-9]+)\.ckpt$ ]]; then
        step="${BASH_REMATCH[1]}"
      elif [[ "$fname" == "last.ckpt" ]]; then
        step=2147483647
      fi
      if (( step > best_step )); then
        best_step="$step"
        best="$f"
      fi
    done
    shopt -u nullglob
  done

  [[ -n "$best" ]] && { echo "$best"; return 0; }
  return 1
}

make_overridden_config () {
  local in_cfg="$1"
  local out_cfg="$2"
  local new_steps="$3"
  mkdir -p "$(dirname "$out_cfg")"
  sed -E "s/^([[:space:]]*max_steps:[[:space:]]*)[0-9]+/\\1${new_steps}/" "$in_cfg" > "$out_cfg"
}

# === 4. 主训练循环 ===

echo "Plan: Train cyclically until ${TOTAL_TARGET_STEPS} steps."
echo "Step size: ${STEP_PER_ROUND} steps per task."
echo "Total Categories: ${NUM_CATS}"
echo

TMP_CFG_DIR=".run_configs_cyclic"
mkdir -p "${TMP_CFG_DIR}"

# ==========================================
# 自动断点恢复逻辑
# ==========================================
echo "Checking for existing progress..."

max_found_step=-1
found_category=""
found_ckpt=""

for cat in "${CATEGORIES[@]}"; do
  ckpt=$(find_best_ckpt_for_category "${cat}") || true
  if [[ -n "$ckpt" ]]; then
    fname=$(basename "$ckpt")
    if [[ "$fname" =~ step=([0-9]+)\.ckpt$ ]]; then
      step="${BASH_REMATCH[1]}"
      if (( step > max_found_step )); then
        max_found_step=$step
        found_category=$cat
        found_ckpt=$ckpt
      fi
    fi
  fi
done

if (( max_found_step > 0 )); then
  echo "Found previous checkpoint:"
  echo "  Category: ${found_category}"
  echo "  Step:     ${max_found_step}"
  echo "  Path:     ${found_ckpt}"
  echo "Resuming training..."
  
  # 恢复 loop_idx
  loop_idx=$(( max_found_step / STEP_PER_ROUND ))
  prev_category="${found_category}"
else
  echo "No valid checkpoint found. Starting from scratch."
  loop_idx=0
  prev_category=""
fi

echo "Starting loop from index: ${loop_idx}"
echo "---------------------------------------------------"

# ==========================================
# 循环开始
# ==========================================

while true; do
  # 1. 计算当前目标步数
  current_target_steps=$(( (loop_idx + 1) * STEP_PER_ROUND ))

  # 2. 检查是否结束
  if (( current_target_steps > TOTAL_TARGET_STEPS )); then
    echo "=== Reached target steps (${TOTAL_TARGET_STEPS}). Stopping. ==="
    break
  fi

  # 3. 计算动态种子 (保证每轮数据 shuffle 不同)
  CURRENT_SEED=$(( SEED + loop_idx ))

  # 4. 确定当前类别
  cat_idx=$(( loop_idx % NUM_CATS ))
  category="${CATEGORIES[$cat_idx]}"
  export CATEGORY="${category}" # 供 YAML 读取

  # 5. [修改] 构造 JSONL 路径
  JSONL_PATH="${JSONL_BASE_DIR}/data_${category}.jsonl"

  if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "ERROR: JSONL file not found: ${JSONL_PATH}"
    echo "Please check if filename 'data_${category}.jsonl' exists in ${JSONL_BASE_DIR}"
    exit 1
  fi

  # 6. 获取 P1 LoRA 路径
  p1_path=$(get_p1_lora_path "${category}")
  # P1 文件如果不强制存在，可以只是警告
  if [[ ! -f "${p1_path}" ]]; then
     echo "[WARN] P1 LoRA not found for ${category} at: ${p1_path}. Training will proceed without P1 adapter."
  fi

  # 7. 生成临时 Config
  ov_cfg="${TMP_CFG_DIR}/step${current_target_steps}_${category}.yaml"
  make_overridden_config "${BASE_CONFIG}" "${ov_cfg}" "${current_target_steps}"

  # 8. 确定接力 Checkpoint
  if [[ ${loop_idx} -eq 0 ]]; then
    ckpt_cli=(--ckpt_path "")
    echo "[$(date '+%T')] Round ${loop_idx}: ${category} (FRESH START)"
  else
    prev_ckpt="$(find_best_ckpt_for_category "${prev_category}")" || {
      echo "ERROR: Could not find checkpoint from previous round (${prev_category})"
      exit 1
    }
    ckpt_cli=(--ckpt_path "${prev_ckpt}")
    echo "[$(date '+%T')] Round ${loop_idx}: ${category} (CONTINUE)"
    echo "  > Resuming from: ${prev_ckpt}"
  fi

  echo "  > Dataset: ${JSONL_PATH}"
  echo "  > Target Steps: ${current_target_steps}"

  # 9. 执行训练
  # [修改] 这里移除了旧的 dataset.video_root 等参数，改为传入 dataset.jsonl_path
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python src/wan2_trainer_p2_with_first_frame_ec.py \
    --config="${ov_cfg}" \
    --seed="${CURRENT_SEED}" \
    --p1_model="${p1_path}" \
    dataset.jsonl_path="${JSONL_PATH}" \
    "${ckpt_cli[@]}"

  # 10. 准备下一轮
  prev_category="${category}"
  loop_idx=$(( loop_idx + 1 ))

  echo "---------------------------------------------------"
done

echo "All Done."
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=5
# python src/wan2_inference_p2_with_first_frame_ec.py \
#     --model_id ./my-model/Wan2.2-TI2V-5B-Diffusers \
#     --p1_path ./p1_lora_library/DS/passive-marker-man.ckpt \
#     --p2_path ./second_phrase_output/train_ec_01/Y-Bot/checkpoints_cil/step=7000.ckpt \
#     --cond_video_path ./datasets/test_data/family/dispensing_soap_on_left_hand_and_washing-517.mp4 \
#     --ref_image_path ./robot_ref_img/passive-marker-man.jpg \
#     --prompt "The video features a 3D animated humanoid robot standing in a modern, minimalist room with wooden flooring and large windows allowing natural light to flood the space. The robot, dressed in a black suit with white dots marking its joints, is positioned near a rectangular table. It extends its right arm forward, reaching towards a small object on the table, which appears to be a bottle or container. As it moves, the robot's fingers close around the object, lifting it slightly before bringing it closer to its body. The robot's movements are smooth and deliberate, showcasing its articulated limbs and precise control. In the background, there is a white sofa, a round coffee table, and a potted plant, contributing to the clean and contemporary aesthetic of the room. The camera remains stationary throughout the sequence, providing a clear view of the robot's actions and the surrounding environment from a slightly elevated angle." \
#     --output_dir ./test/ec_p2/passive-marker-man \
#     --output_name 03.mp4 \
#     --num_frames 49 \
#     --height 224 \
#     --width 416 \
#     --fps 24

# python src/wan2_inference_p2_with_first_frame_ec.py \
#     --model_id ./my-model/Wan2.2-TI2V-5B-Diffusers \
#     --p1_path ./p1_lora_library/DS_real/atlas_5500.ckpt \
#     --p2_path ./second_phrase_output/train_ec_new_01/fourier-gr2/checkpoints_cil/step=10900.ckpt \
#     --cond_video_path ./data/test_data/family/mop_the_floor.mp4 \
#     --ref_image_path ./robot_ref_img/atlas.jpg \
#     --prompt "A humanoid robot is engaged in cleaning the floor of a living room using a handheld vacuum cleaner. The robot stands upright, holding the vacuum with both hands, and moves it back and forth across the wooden floor. Its arms extend forward as it maneuvers the vacuum head, ensuring thorough coverage of the area. The robot has a compact build with a light-colored torso and dark limbs, and it wears red shoes. In the background, there is a large flat-screen television on a white stand, a tall white cabinet, and various household items including a guitar case and a small refrigerator. The room is well-lit, with a neutral color palette and a clean, organized appearance. The camera remains stationary throughout the sequence, providing a clear side view of the robot's actions, allowing for an unobstructed observation of its movements and the surrounding environment." \
#     --output_dir ./test/ec_new_01/atlas \
#     --output_name mop_the_floor_1280.mp4 \
#     --num_frames 49 \
#     --height 736 \
#     --width 1280 \
#     --fps 24 \
#     --steps 30 


#!/bin/bash

# 1. 定义数据所在的目录
DATA_DIR="./input_for_drawing/real/tesla"

# 2. 提取出公共参数，让代码更整洁
MODEL_ID="./my-model/Wan2.2-TI2V-5B-Diffusers"
P1_PATH="./p1_lora_library/DS_new/tesla-optimus.ckpt"
P2_PATH="./second_phrase_output/train_ec_new_01/fourier-gr2/checkpoints_cil/step=10900.ckpt"
REF_IMAGE="./robot_ref_img/tesla-optimus.jpg"
OUTPUT_BASE_DIR="./test"

# 3. 遍历目标目录下的所有 .mp4 文件
for VIDEO_PATH in "$DATA_DIR"/*.mp4; do
    
    # 提取不带后缀的文件名 (例如: mop_the_floor)
    BASENAME=$(basename "$VIDEO_PATH" .mp4)
    
    # 拼接对应的 .txt 文件路径
    TXT_PATH="$DATA_DIR/${BASENAME}.txt"
    
    # 检查同名的 txt 文件是否存在
    if [ ! -f "$TXT_PATH" ]; then
        echo "⚠️ 警告: 找不到对应的文本文件 $TXT_PATH，跳过 $BASENAME..."
        continue
    fi
    
    # 读取 txt 文件的内容作为 prompt
    PROMPT=$(cat "$TXT_PATH")
    
    # 动态生成输出文件名，加上原视频的名字防止覆盖
    OUTPUT_NAME="${BASENAME}.mp4"
    
    echo "=================================================="
    echo "🚀 开始处理: $BASENAME"
    echo "📜 使用 Prompt: $PROMPT"
    echo "=================================================="
    
    # 执行原来的 Python 脚本，传入动态变量
    python src/wan2_inference_p2_with_first_frame_ec.py \
        --model_id "$MODEL_ID" \
        --p1_path "$P1_PATH" \
        --p2_path "$P2_PATH" \
        --cond_video_path "$VIDEO_PATH" \
        --ref_image_path "$REF_IMAGE" \
        --prompt "$PROMPT" \
        --output_dir "$OUTPUT_BASE_DIR" \
        --output_name "$OUTPUT_NAME" \
        --num_frames 49 \
        --height 736 \
        --width 1280 \
        --fps 24 \
        --steps 50 
        
    echo "✅ 完成: $BASENAME -> 输出为 $OUTPUT_NAME"
    
done

echo "🎉 所有视频批量推理完成！"
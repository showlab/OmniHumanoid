export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python src/wan2_inference_p1.py \
  --config configs/train_p1_droid.yaml \
  --lora_ckpt ./p1_lora_library/DS_real/atlas_5500.ckpt \
  --prompts "The video shows a humanoid robot walking across a spacious indoor area. The robot moves forward with a steady gait, its legs bending at the knees and hips in a coordinated manner to propel it forward. Its arms swing slightly in opposition to its legs, mimicking natural human walking motion. The robot has a sleek, white and black design with a rounded head and a torso that appears to house internal mechanisms. It stands on two feet, each equipped with a flat sole, allowing it to maintain balance as it walks. The environment around the robot includes large windows that let in natural light, revealing an outdoor scene with water and trees. Orange safety barriers line one side of the room, while red and black mats are stacked against the wall. A blue table with various items is visible in the background. The camera follows the robot from a side angle, maintaining a consistent distance as it moves from left to right across the frame. The camera remains steady, capturing the robot's movement in a smooth, continuous shot." \
  --output_dir ./test/p1/DS_real/atlas \
  --num_frames 81 \
  --steps 30 \
  --fps 24

#gr3要再测试一下

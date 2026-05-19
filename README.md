# OmniHumanoid

[![arXiv](https://img.shields.io/badge/arXiv-2605.12038-b31b1b.svg)](https://arxiv.org/abs/2605.12038)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/dddb0513/Human-Humanoid-4D)

OmniHumanoid is a cross-embodiment motion transfer system built on the Wan2.2-TI2V-5B video diffusion model. It enables generating videos of a **target robot** reproducing the same actions from a **human (or other robot) action video**.

We also release **[Human-Humanoid-4D](https://huggingface.co/datasets/dddb0513/Human-Humanoid-4D)**, a large-scale human/robot-to-robot paired video dataset rendered in Unity. The paired videos are strictly temporally aligned, covering diverse scenes and multiple robot characters. This dataset is used for training the Action Transfer Module (P2).

---

## 🧠 Method Overview

The system consists of two complementary modules:

| Module | Name | Purpose |
|--------|------|---------|
| **P1** | Embodiment Video LoRA | Fine-tunes the base model to generate videos of a specific robot from text descriptions |
| **P2** | Action Transfer Module  | Preserves motion fidelity when transferring actions across embodiments |
| **P1 + P2** | Cross-embodiment video generation | Joint inference: input a human/robot action video + target robot appearance → output robot reproducing the same motion |

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/showlab/OmniHumanoid.git
cd OmniHumanoid

# 2. Create the Conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate interactionvideo
```

---

## 📁 Project Structure

```
OmniHumanoid/
├── configs/              # Training YAML configs
├── datasets/             # Dataset loading code
├── models/               # Core model (Transformer, Pipeline, Attention, CIL)
│   └── wan2/             # Wan2.2 customized modules
├── src/                  # Main training & inference scripts
├── tools/                # Scheduler, checkpoint utilities
├── p1_train/             # P1 LoRA training (DiffSynth-Studio based)
│   ├── diffsynth/        # DiffSynth training library
│   └── examples/         # P1 training entry scripts
├── p1_lora_library/      # Trained P1 LoRA weights
├── robot_ref_img/        # Robot reference images
├── inference_p1.sh       # P1 text-to-video inference
├── inference_p2_h2r.sh   # P2 H2R video-to-video inference
└── train_p2_h2r_multi.sh # P2 multi-robot round-robin training
```

---

## 📥 Model Preparation

### Base Model

Download the **Wan2.2-TI2V-5B** diffusers-format model and place it in `./my-model/Wan2.2-TI2V-5B-Diffusers/`:

```
my-model/Wan2.2-TI2V-5B-Diffusers/
├── tokenizer/
├── text_encoder/
├── vae/
├── transformer/
└── scheduler/
```

You can download it from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) on HuggingFace.

### P2 Checkpoint (for V2V inference)

Download `step=10800.ckpt` from [ModelScope: dddb0513/p2_ckpt](https://modelscope.cn/datasets/dddb0513/p2_ckpt/files) and place it at a path you'll reference in the inference script (e.g., `./second_phrase_output/train_ec_new_01/.../step=10900.ckpt`).

---

## 🎯 Adapt to Your Own Robot (Full Pipeline)

This section walks you through how to **train a P1 appearance LoRA for your own robot** and then use it for video-to-video motion transfer.

---

### Step 1: Prepare Training Data

You need **40-60 short video clips** (3–5 seconds each) of your target robot performing various motions.

#### Data Format

Create a **CSV metadata file** with the following columns:

```csv
video_path,text
/path/to/robot_video_001.mp4,"A humanoid robot with a white torso and black limbs walks forward in an indoor space."
/path/to/robot_video_002.mp4,"A humanoid robot waves its right hand while standing in a bright room."
/path/to/robot_video_003.mp4,"A humanoid robot bends down to pick up an object from the floor."
```

#### Data Requirements

- **Resolution**: 736×1280 (H×W) recommended. Videos will be resized during training.
- **Frame count**: At least 81 frames (≈3.4s at 24fps). Longer videos will be randomly cropped.
- **Content**: Show the robot from various angles and performing diverse motions.
- **Prompts**: Describe **both the robot's appearance and its action** in detail. Be consistent in how you describe the robot's appearance across all samples.

#### Reference Image

Take a clear, front-facing photo of your robot and save it to `robot_ref_img/your-robot-name.jpg`. This will be used during P2 inference.

---

### Step 2: Train P1 Robot Appearance LoRA

The P1 training uses the [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) framework included in `p1_train/`.

#### 2.1 Configure the training script

Copy and edit the training script:

```bash
cp p1_train/train_robotic_first_lora_tesla.sh p1_train/train_my_robot.sh
```

Edit `p1_train/train_my_robot.sh`:

```bash
# ---- Hardware Setup ----
GPU_IDS="0"          # GPU ID(s) to use
NUM_PROCESSES=1      # Number of GPUs
DIST_TYPE="no"       # "no" for single GPU, "multi_gpu" for multi-GPU

# ---- Training Parameters ----
ROBOT_NAME="my-robot"
META_CSV="/path/to/your/robot_data.csv"   # Your CSV metadata file
OUT="./p1_output/${ROBOT_NAME}"           # Output directory

LR=1e-4              # Learning rate
EPOCHS=200           # Number of epochs (typically converges in 3000-6000 steps)
RANK=64              # LoRA rank
```

#### 2.2 Run training

```bash
cd p1_train
bash train_my_robot.sh
```

**Training tips:**
- Monitor loss convergence. Typically 3000–6000 steps are sufficient.
- Checkpoints are saved every 1000 steps as `.safetensors` files in the output directory.
- Use `--lora_checkpoint` to resume from a previous checkpoint if needed.

#### 2.3 Convert weights to inference format

The training outputs `.safetensors` format, but the inference scripts expect `.ckpt` format. Convert using:

```bash
python p1_train/safetensor2ckpt.py \
  --input ./p1_output/my-robot/step-5000.safetensors \
  --output ./p1_lora_library/my-robot.ckpt
```

This converts and remaps the weight keys (DiffSynth → diffusers naming convention) and packages them into the expected structure:

```python
{"state_dict": {"transformer_lora": {...}, "text_encoder_lora": {...}, "patch_embedding_extra": {...}}}
```

---

### Step 3: Test P1 LoRA (Text-to-Video)

Verify your LoRA works by generating a video from text:

```bash
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python src/wan2_inference_p1.py \
  --config configs/train_p2_h2r_all.yaml \
  --lora_ckpt ./p1_lora_library/my-robot.ckpt \
  --prompts "A humanoid robot walks forward in a bright indoor space. The robot has [describe your robot's appearance]." \
  --output_dir ./test/p1/my-robot \
  --num_frames 81 \
  --steps 30 \
  --fps 24
```

If the generated video clearly shows your robot's appearance, the LoRA is ready.

---

### Step 4: V2V Inference (Human-to-Robot Motion Transfer)

Now use P1 + P2 together to transfer human actions to your robot.

#### 4.1 Prepare input data

Create a directory with your source videos and corresponding text prompts:

```
input_videos/
├── action_001.mp4       # Human action video
├── action_001.txt       # Text prompt describing the robot doing this action
├── action_002.mp4
├── action_002.txt
└── ...
```

**Prompt writing guide** for `.txt` files:
- Describe **your target robot** (not the human) performing the action
- Include the robot's appearance details and the environment
- Example: *"A humanoid robot with a white torso and black limbs is mopping the floor in a living room. The robot holds the mop with both hands and moves it back and forth across the wooden floor..."*

#### 4.2 Run batch V2V inference

Edit `inference_p2_h2r.sh`:

```bash
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Input directory containing .mp4 + .txt pairs
DATA_DIR="./input_videos"

# Model paths
MODEL_ID="./my-model/Wan2.2-TI2V-5B-Diffusers"
P1_PATH="./p1_lora_library/my-robot.ckpt"          # Your P1 LoRA
P2_PATH="./second_phrase_output/.../step=10800.ckpt" # P2 checkpoint
REF_IMAGE="./robot_ref_img/my-robot.jpg"            # Your robot's reference image
OUTPUT_BASE_DIR="./output_videos"

for VIDEO_PATH in "$DATA_DIR"/*.mp4; do
    BASENAME=$(basename "$VIDEO_PATH" .mp4)
    TXT_PATH="$DATA_DIR/${BASENAME}.txt"

    if [ ! -f "$TXT_PATH" ]; then
        echo "⚠️ Warning: $TXT_PATH not found, skipping..."
        continue
    fi

    PROMPT=$(cat "$TXT_PATH")

    python src/wan2_inference_p2_with_first_frame_ec.py \
        --model_id "$MODEL_ID" \
        --p1_path "$P1_PATH" \
        --p2_path "$P2_PATH" \
        --cond_video_path "$VIDEO_PATH" \
        --ref_image_path "$REF_IMAGE" \
        --prompt "$PROMPT" \
        --output_dir "$OUTPUT_BASE_DIR" \
        --output_name "${BASENAME}.mp4" \
        --num_frames 49 \
        --height 736 \
        --width 1280 \
        --fps 24 \
        --steps 50
done
```

Then run:

```bash
bash inference_p2_h2r.sh
```

---

## 🔄 Quick Reference: End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Collect 10-30 robot videos + write text descriptions    │
│                          ↓                                  │
│  2. Train P1 LoRA (3000-6000 steps)                        │
│     p1_train/train_my_robot.sh                             │
│                          ↓                                  │
│  3. Convert weights: safetensors → ckpt                    │
│     python p1_train/safetensor2ckpt.py                     │
│                          ↓                                  │
│  4. Verify P1: text-to-video generation                    │
│     bash inference_p1.sh                                   │
│                          ↓                                  │
│  5. V2V transfer: human video → robot video                │
│     bash inference_p2_h2r.sh                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Pre-trained P1 LoRAs

We provide pre-trained P1 LoRAs for the following robots in `p1_lora_library/`:

- unitree-h1, unitree-g1
- fourier-gr2, fourier-gr3
- tesla-optimus
- atlas
- X-Bot, Y-Bot
- passive-marker-man
- Remy-mixamo, dummy02

---

## 🏋️ Advanced: P2 Training

If you need to train/fine-tune the P2 motion consistency module with custom paired data:

```bash
bash train_p2_h2r_multi.sh
```

The P2 training requires **paired data**: human action videos + corresponding robot action videos performing the same motion. See `configs/train_p2_h2r_all.yaml` for configuration details.

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{song2026omnihumanoidstreamingcrossembodimentvideo,
      title={OmniHumanoid: Streaming Cross-Embodiment Video Generation with Paired-Free Adaptation}, 
      author={Yiren Song and Xiyao Deng and Pei Yang and Yihan Wang and Mike Zheng Shou},
      year={2026},
      eprint={2605.12038},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.12038}, 
}
```

## 📜 License

This project is released under the [Apache 2.0 License](LICENSE).

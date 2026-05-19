# OmniHumanoid

## 🛠️ Installation

We recommend using Conda to manage the runtime environment. Please follow the steps below to configure the dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/showlab/OmniHumanoid.git
cd OmniHumanoid

# 2. Create the Conda environment from the yaml file
conda env create -f environment.yml

# 3. Activate the environment (please replace [your_env_name] with the actual environment name defined in the yaml file)
conda activate [your_env_name]
```

## 🧠 Method Overview

OmniHumanoid is built on top of two complementary modules, **Phase 1 (P1)** and **Phase 2 (P2)**, which can be used independently or jointly:

- **Phase 1 (P1) — Robot Appearance LoRA**: The P1 weights are essentially appearance LoRAs of specific robots. When loaded on top of the base model at inference time, P1 enables **text-to-video** generation of a particular robot model purely from a textual description (e.g., generating a video of a specified robot performing actions described in the prompt).
- **Phase 2 (P2) — Motion Consistency Module**: The P2 weights provide a motion consistency module that transfers and preserves motion patterns across subjects. P2 alone is responsible for ensuring that the generated motion remains faithful to a reference action sequence.
- **P1 + P2 Joint Inference — Human/Robot-to-Robot (H2R) Video Generation**: When P1 and P2 are loaded **simultaneously**, the system can take a human (or robot) action video as input and generate a video of a **specific target robot** (determined by the loaded P1 LoRA) reproducing the same actions. This unlocks cross-embodiment motion transfer with controllable robot appearance.

## 📁 Project Structure

The core directory structure of this repository is outlined below:

- `configs/`: Contains YAML configuration files for model training and inference (e.g., `train_p2_h2r_all.yaml`).
- `datasets/`: Contains data processing code.
- `dataset/`: Put the training data here.
- `models/`: Contains the core model files.
- `src/`: Contains the main Python scripts for underlying training and inference (including `wan2_inference_p2_with_first_frame_ec.py` based on first-frame conditioning).
- `tools/`: Contains auxiliary utility scripts.
- `p1_lora_library/`: Contains the trained appearance LoRA weights from Phase 1.
- `robot_ref_img/`: Contains reference images of various target robots used to guide the generation.

## 📥 Model and Data Preparation

Before running inference or training, please ensure:

1. **Base Model**: Place the pre-trained Wan2.2-TI2V-5B weights in the `model/` directory or the path specified in the script.
2. **LoRA Weights**: Place the corresponding LoRA weights in the `p1_lora_library/` directory.
3. **Reference Images**: Ensure that `robot_ref_img/` contains the robot reference images you need.

## 🚀 Inference

```bash
# Run Phase 1 inference
bash inference_p1.sh
```

⚠️ **Prerequisite for Phase 2 Inference:**

Before running the Phase 2 (H2R) inference, you must download the Phase 2 checkpoint (`step=10800.ckpt`) from ModelScope.

- **Download Link:** [dddb0513/p2_ckpt](https://modelscope.cn/datasets/dddb0513/p2_ckpt/files)

- **Placement:** Once downloaded, please place `step=10800.ckpt` into the path specified in your inference script.

```bash
# Run Phase 2 (H2R) inference
bash inference_p2_h2r.sh
```

## 🏋️ Training

If you need to fine-tune the model using custom data, you can modify the configuration files in the `configs/` directory and run the corresponding training scripts.

```bash
# Run Phase 2 H2R multi-GPU/multi-modal training
bash train_p2_h2r_multi.sh
```

For more training parameters (such as `batch_size`, `learning_rate`), please directly edit the corresponding `.sh` scripts or the `configs/train_p2_h2r_all.yaml` file.

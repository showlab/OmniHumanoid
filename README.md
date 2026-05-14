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

1. **Base Model**: Place the pre-trained Wan2 weights in the `model/` directory or the path specified in the script.
2. **LoRA Weights**: Place the corresponding LoRA weights in the `p1_lora_library/` directory.
3. **Reference Images**: Ensure that `robot_ref_img/` contains the robot reference images you need.

## 🚀 Inference

```bash
# Run Phase 1 inference
bash inference_p1.sh

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

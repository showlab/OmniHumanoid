# OmniHumanoid

## 🛠️ 环境安装 (Installation)

推荐使用 Conda 管理运行环境。请按照以下步骤配置依赖：

```bash
# 1. 克隆仓库
git clone https://github.com/showlab/OmniHumanoid.git
cd OmniHumanoid

# 2. 根据 yaml 文件创建 Conda 环境
conda env create -f environment.yml

# 3. 激活环境 (请将 myenv 替换为 yaml 文件中定义的实际环境名称)
conda activate [你的环境名称]
```



## 📁 项目结构 (Project Structure)

本仓库的核心目录结构说明如下：

- `configs/`: 存放模型训练和推理的 YAML 配置文件 (例如 `train_p2_h2r_all.yaml`)。
- `datasets/`: 存放数据处理的代码。
- `models/`: 存放核心模型文件。
- `src/`: 存放底层训练和推理的主 Python 脚本 (包含基于首帧条件的 `wan2_inference_p2_with_first_frame_ec.py`)。
- `tools/`: 存放辅助工具脚本。
- `p1_lora_library/`: 存放第一阶段 (Phase 1) 训练好的外观 LoRA 权重。
- `robot_ref_img/`: 存放各类目标机器人的参考图像 (Reference Images)，用于指导生成。



## 📥 模型与数据准备 (Preparation)

在运行推理或训练之前，请确保：

1. **基础模型**: 将 Wan2 的预训练权重放置在 `model/` 或脚本指定的路径下。
2. **LoRA 权重**: 将对应的 LoRA 权重放置在 `p1_lora_library/` 目录下。
3. **参考图**: 确保 `robot_ref_img/` 中包含你需要的机器人参考图片。

## 🚀 推理 (Inference)

Bash

```
# 运行 P1 阶段推理
bash inference_p1.sh

# 运行 P2 阶段 (H2R) 推理
bash inference_p2_h2r.sh
```

## 🏋️ 训练 (Training)

如果你需要使用自定义数据进行微调，可以修改 `configs/` 中的配置文件，并运行对应的训练脚本。

Bash

```
# 运行 P2 阶段 H2R 多卡/多模态训练
bash train_p2_h2r_multi.sh
```

更多训练参数（如 `batch_size`, `learning_rate`），请直接编辑对应的 `.sh` 脚本或 `configs/train_p2_h2r_all.yaml` 文件。

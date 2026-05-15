# Phase 1 LoRA Library

本目录用于存放第一阶段 (Phase 1) 训练好的外观 LoRA 权重文件。

由于模型权重文件较大，我们将其托管在了 [ModelScope (魔搭社区)](https://modelscope.cn)。在运行推理或训练代码之前，请务必下载所需的权重文件，并将其放置在当前目录 (`p1_lora_library/`) 下。

## 🔗 模型下载链接

**ModelScope 数据集主页:** [dddb0513/p1_lora](https://modelscope.cn/datasets/dddb0513/p1_lora/files)

（包含 `dummy02.ckpt`, `fourier-gr2.ckpt`, `fourier-gr3.ckpt`, `passive-marker-man.ckpt` 等文件）

---

## 📥 如何下载

您可以选择以下任意一种方式获取权重：

### 方法一：网页手动下载
直接点击上方的数据集链接，进入 **“数据文件”** 页面，找到您需要的 `.ckpt` 文件，点击右侧的“下载”按钮，保存到您本地的 `OmniHumanoid/p1_lora_library/` 文件夹中。

### 方法二：使用 Git 命令行下载（推荐）
如果您需要下载所有权重，可以使用 Git 命令一键克隆（请确保已安装 [Git LFS](https://git-lfs.com/)）：

```bash
# 下载完整的权重仓库
git clone [https://www.modelscope.cn/datasets/dddb0513/p1_lora.git](https://www.modelscope.cn/datasets/dddb0513/p1_lora.git)

# 将 .ckpt 文件移动到当前目录
mv p1_lora/*.ckpt ./

# （可选）删除多余的文件夹
rm -rf p1_lora

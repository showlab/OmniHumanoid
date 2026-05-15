# 📁 Dataset Directory

由于数据集文件体积较大，我们已将其分卷压缩并托管在 [ModelScope (魔搭社区)](https://modelscope.cn)。

在运行训练代码之前，请务必下载完整的数据集，并将其合并解压到当前目录 (`data/`) 下。

## 🔗 数据集链接
**ModelScope 主页:** [dddb0513/OmniHumanoid](https://modelscope.cn/datasets/dddb0513/OmniHumanoid/files)

---

## 📥 下载与解压指南

数据文件被分割成了 7 个部分（`part00` 到 `part06`）。请严格按照以下步骤操作：

### 1. 下载分卷文件

**推荐使用 Git LFS 直接克隆到当前 `data/` 目录：**
```bash
# 请确保您的系统已安装 Git LFS
git clone [https://www.modelscope.cn/datasets/dddb0513/OmniHumanoid.git](https://www.modelscope.cn/datasets/dddb0513/OmniHumanoid.git)

### 2. 合并与解压数据

下载完成后，进入存放了 `part00~06` 文件的文件夹，然后根据您的操作系统执行合并与解压命令。

**🐧 Linux / 🍎 macOS 用户:**
```bash
# 1. 进入下载好的文件夹 (请根据实际情况修改路径)
cd OmniHumanoid

# 2. 将所有分卷合并为一个完整的 tar.gz 文件
cat OmniHumanoid.tar.gz.part* > OmniHumanoid.tar.gz

# 3. 解压合并后的文件
tar -zxvf OmniHumanoid.tar.gz
```

**🪟 Windows 用户 (需在 CMD 命令行中执行):**
```cmd
# 1. 进入下载好的文件夹
cd OmniHumanoid

# 2. 使用二进制模式合并分卷文件
copy /B OmniHumanoid.tar.gz.part* OmniHumanoid.tar.gz
```
*Windows 用户在合并完成后，直接使用 7-Zip 或 WinRAR 等常规解压软件解压生成的 `OmniHumanoid.tar.gz` 即可。*

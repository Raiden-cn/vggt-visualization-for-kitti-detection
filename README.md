# VGGT + KITTI 3D 目标检测与 3D 点云重建可视化系统

本项目基于 Meta **VGGT (Visual Geometry Grounded Transformer)** 单图 3D 重建大模型，针对 **KITTI 数据集** 实现了**锚点物理深度对齐（Anchor-Based Physical Scale Alignment）**、**Ground Truth (红色) 与 预测 (绿色) 3D Bounding Box 联合渲染**、**类别动态筛选** 与 **交互式 Web 3D 浏览**。

---

## 🌟 核心功能亮点

1. **锚点物理深度对齐 (Anchor-Based Physical Scale Alignment)**：
   * 自动将 VGGT 单图预测的无量纲相对深度图，通过 3D 检测框锚点的真实物理深度 $Z_{\text{kitti}}$ 计算精确的缩放因子 $\text{scale}_{\text{global}}$。
   * 利用 KITTI 相机真实内参矩阵 $P_2$ 进行 3D 反投影，生成**具备真实物理单位（米）**的绝对 3D 点云，使 3D 框完美包裹车辆点云，彻底解决点云与框漂移、悬空的问题。
2. **Ground Truth (红色) 与 预测 (绿色) 联合对比**：
   * **🔴 真实框 (GT)**：显示为高亮纯红色，提取自 KITTI 标准标签 `label_2/`。
   * **🟢 预测框 (Pred)**：显示为高亮绿色，提取自检测模型输出 `./kitti_pred/data/`。
3. **Web 端交互式 3D 控件**：
   * 🏷️ **3D 文本标签开关**：自由切换是否显示 `GT: Car` 或 `Pred: Car 0.98` 标签。
   * 🔍 **类别动态筛选 (Category Filter)**：按 `Car`、`Pedestrian`、`Cyclist`、`Truck`、`Van` 等类别单独或组合筛选展示。
   * 📷 相机视锥体（Frustum）展示、点云置信度过滤与点大小调节。

---

## 🛠️ 1. 环境安装指南 (Environment Setup)

建议在 Linux 环境下使用 Conda 创建独立的 Python 3.10 环境。**注意：为避免 Conda 重新覆盖系统 CUDA，推荐优先使用 pip 安装 PyTorch。**

```bash
# 1. 创建并激活 Conda 环境
conda create -n vggt python=3.10 -y
conda activate vggt

# 2. 安装 PyTorch (以 CUDA 12.1 为例)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目核心依赖
pip install viser trimesh scipy matplotlib einops safetensors huggingface_hub

# 4. 以可编辑模式安装 vggt 源码包
pip install -e .
```

---

## 📂 2. 数据集与预测结果路径配置 (Dataset & Path Setup)

打开核心启动文件 **`output_kitti_val_3d/kitti_val_3d_app.py`**，在 `main()` 函数顶部可以灵活修改数据集与预测结果路径：

```python
# 位于 output_kitti_val_3d/kitti_val_3d_app.py 中的配置部分：

val_txt   = "/root/dataset/kitti/ImageSets/val.txt"       # KITTI 验证集清单文件
img_dir   = "/root/dataset/kitti/training/image_2"        # KITTI 图像存放目录 (.png)
calib_dir = "/root/dataset/kitti/training/calib"          # KITTI 相机标定文件目录 (.txt)
gt_dir    = "/root/dataset/kitti/training/label_2"         # KITTI GT 标签目录 (.txt)
pred_dir  = "./kitti_pred/data"                            # 3D 检测模型预测结果目录 (.txt)
port      = 12342                                          # Web 可视化服务端口
```

### 📁 推荐的目录摆放结构：

```text
your_project_root/
├── output_kitti_val_3d/
│   └── kitti_val_3d_app.py          # 🚀 主程序
├── kitti_pred/
│   └── data/                        # 📂 存放您的 3D 检测预测结果文件
│       ├── 000001.txt
│       ├── 000002.txt
│       └── ...
└── vggt/                            # VGGT 核心模型代码
```

### 📄 预测结果文本格式说明：
`./kitti_pred/data/{val_id}.txt` 需遵循 KITTI 官方标准 16 列格式：
```text
# 类型  truncation  occlusion  alpha  u1      v1      u2      v2      h     w     l     x     y     z     ry    score
Car     0.00        0          0.00   192.37  402.31  374.00  1.60    1.57  3.23  -2.70 1.74  3.68  -1.29 0.95
```

---

## 🚀 3. 快速运行与远程浏览 (Quick Start)

### 步骤 1：启动 3D 可视化服务

```bash
conda activate vggt
python output_kitti_val_3d/kitti_val_3d_app.py
```
*程序会自动载入 `val.txt` 中前 10 帧图像与对应的 3D 预测/GT 标签，完成后启动 Viser Web 服务。*

### 步骤 2：SSH 端口转发 (SSH Tunnel)

在**您自己的本地电脑终端**（非服务器终端）运行以下命令将端口转发至本地：

```bash
ssh -L 12342:localhost:12342 user@your_server_ip
```

### 步骤 3：在本地浏览器中查看

在本地浏览器中打开：👉 **[http://localhost:12342](http://localhost:12342)**

---

## 🎛️ 4. Web UI 界面控制指南

进入网页端后，左侧面板包含以下核心控制选项：

| 控制项 | 功能说明 |
| :--- | :--- |
| **`Select Val Scene`** | 下拉框自由选择切换 10 个独立场景 |
| **`Show GT 3D Boxes (Red)`** | ☑ 勾选/取消勾选显示红色 Ground Truth 真实框 |
| **`Show Prediction 3D Boxes (Green)`** | ☑ 勾选/取消勾选显示高亮绿色模型预测框 |
| **`Show 3D Text Labels`** | ☑ 勾选/取消勾选显示 `GT: Car` 或 `Pred: Car 0.98` 标签 |
| **`Category Filter`** | 折叠菜单中可自由勾选显示/隐藏 `Car`、`Pedestrian`、`Cyclist`、`Truck` 等类别 |
| **`Show Camera Frustum`** | ☑ 显示相机视锥体与拍摄视角 |
| **`Confidence Filter (%)`** | 调节点云置信度阈值过滤背景噪点 |

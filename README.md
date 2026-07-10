# 🚗 端到端自动驾驶算法

基于 **NVIDIA PilotNet** 架构的端到端自动驾驶模型。输入摄像头画面，直接输出转向角。

## 项目特点

- **纯合成数据**：自动生成带标注的道路图像，无需真实驾驶数据集
- **PilotNet 架构**：经典的端到端卷积神经网络
- **完整流水线**：数据生成 → 模型训练 → 评估 → 驾驶模拟
- **Colab 友好**：免费 GPU 即可运行，10-20 分钟出结果
- **可解释评估**：回归指标 + 驾驶模拟双重验证

## 快速开始

```bash
pip install -r requirements.txt

# 完整流水线(推荐)
python main.py full

# 或分步执行:
python main.py generate    # 1. 生成合成数据
python main.py train       # 2. 训练模型
python main.py evaluate --weights runs/smallnet_xxx/best.weights.h5  # 3. 评估
python main.py simulate --weights runs/smallnet_xxx/best.weights.h5  # 4. 驾驶模拟
```

---

## Google Colab 运行指南

### 方式 A：上传压缩包（推荐）

**第 1 步：在本地压缩项目**

```bash
cd D:\编程\个人项目
tar -czf autonomous_driving.tar.gz autonomous_driving/
```

**第 2 步：上传到 Colab**
- 打开 https://colab.research.google.com/ 新建笔记本
- 左侧 📁 文件图标 → 上传 → 选择 `autonomous_driving.tar.gz`

**第 3 步：运行以下单元格**

---

### 单元格 1：解压 + 安装依赖

```python
!tar -xzf autonomous_driving.tar.gz
!pip install -q opencv-python matplotlib tensorflow

# 验证依赖
import tensorflow as tf
import cv2
import matplotlib
print("✅ 所有依赖安装成功")
print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
```

---

### 单元格 2：生成合成道路数据

```python
%cd autonomous_driving
!python main.py generate
```

输出示例：
```
生成训练数据...
  样本数: 30000
  生成进度: 3000/30000
  ...
  已保存 30000 个样本到 data/train

生成验证数据...
生成测试数据...
全部数据生成完成!
```

---

### 单元格 3：训练模型

```python
%cd /content/autonomous_driving
!python main.py train --model smallnet
```

训练过程：
- `smallnet` 轻量模型：Colab GPU 约 5-8 分钟
- `pilotnet` 完整模型：Colab GPU 约 10-15 分钟
- 训练完成后自动运行评估和驾驶模拟

训练中途可能会看到 `ReduceLROnPlateau` 降低学习率，这是正常的。

---

### 单元格 4（可选）：单独评估 + 驾驶模拟

如果训练被中断或想重新评估，指定权重路径：

```python
%cd /content/autonomous_driving

# 先查看可用的权重文件
import os
checkpoints = [d for d in os.listdir("models/checkpoints") if d.startswith("smallnet")]
print("可用的训练结果:", checkpoints)

if checkpoints:
    latest = sorted(checkpoints)[-1]
    weights = f"models/checkpoints/{latest}/best.weights.h5"
    print(f"使用: {weights}")

    !python main.py evaluate --weights "$weights"
    !python main.py simulate --weights "$weights"
```

---

### 单元格 5：查看生成的图表

```python
%cd /content/autonomous_driving
from IPython.display import Image, display
import os

output_dir = "output"
images = [f for f in os.listdir(output_dir) if f.endswith(".png")]
images.sort()

for img_name in images:
    print(f"\n### {img_name}")
    display(Image(filename=f"{output_dir}/{img_name}"))
```

---

### 单元格 6（可选）：效果展示图

```python
%cd /content/autonomous_driving
!python main.py demo
```

会显示合成道路样本和增强效果的对比图。

---

### 方式 B：直接在 Colab 里克隆/下载

```python
# 如果你把项目上传到了 GitHub
!git clone https://github.com/你的用户名/autonomous_driving.git
%cd autonomous_driving
```

---

## 项目结构

```
autonomous_driving/
├── main.py                # 主入口 (generate/train/evaluate/simulate/demo)
├── config.py              # 全局配置参数
├── data_generator.py      # 合成道路数据生成器
├── train.py               # 训练脚本
├── evaluate.py            # 评估模块 (回归指标 + 可视化)
├── simulator.py           # 驾驶模拟器 (闭环测试)
├── models/
│   ├── pilotnet.py        # PilotNet / SmallNet 模型定义
│   └── __init__.py
├── data/                  # 生成的数据
│   ├── train/             # 训练集 (30000样本)
│   ├── val/               # 验证集 (5000样本)
│   └── test/              # 测试集 (5000样本)
├── models/checkpoints/    # 训练好的模型权重
├── output/                # 评估图表输出
├── runs/                  # TensorBoard日志
├── requirements.txt
└── README.md
```

## 命令行速查

```bash
# 完整流水线
python main.py full --model smallnet

# 分步执行
python main.py generate                                    # 生成数据
python main.py train --model smallnet                      # 训练
python main.py train --model smallnet --resume PATH        # 恢复训练
python main.py evaluate --weights PATH --model_type smallnet  # 评估
python main.py simulate --weights PATH                     # 驾驶模拟
```

## 技术原理

### PilotNet 架构 (NVIDIA, 2016)

```
输入: 80×160×3 道路图像
  ↓ Conv 5×5, 24, stride 2 + ELU
  ↓ Conv 5×5, 36, stride 2 + ELU
  ↓ Conv 5×5, 48, stride 2 + ELU
  ↓ Conv 3×3, 64, stride 1 + ELU
  ↓ Conv 3×3, 64, stride 1 + ELU
  ↓ Flatten
  ↓ Dense 100 + Dropout
  ↓ Dense 50  + Dropout
  ↓ Dense 10
  ↓ Dense 1 (tanh)
输出: 转向角 [-1, 1]
```

### 合成数据生成

每张道路图像通过以下方式生成：
1. 根据曲率计算每行的道路中心位置：`center_x = W/2 + curvature × t² × scale`
2. 考虑透视效果：近处道路宽、远处窄
3. 绘制道路表面、车道线、路肩
4. 数据增强：亮度/对比度/噪声/平移

### 评估体系

| 指标 | 类别 | 说明 |
|------|------|------|
| MSE/MAE | 回归 | 预测 vs ground truth 误差 |
| R² Score | 回归 | 模型解释方差的比例 |
| 误差分布 | 回归 | 直/缓/急弯分段评估 |
| 车道保持率 | 模拟 | 车辆在道路内的比例 |
| RMS 横向误差 | 模拟 | 偏离道路中心的程度 |

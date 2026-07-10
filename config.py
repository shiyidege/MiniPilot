"""全局配置"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"
RUNS_DIR = BASE_DIR / "runs"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== 图像参数 ==========
IMG_HEIGHT = 80
IMG_WIDTH = 160
IMG_CHANNELS = 3

# ========== 数据生成参数 ==========
NUM_TRAIN_SAMPLES = 30000
NUM_VAL_SAMPLES = 5000
NUM_TEST_SAMPLES = 5000
CURVATURE_RANGE = (-0.8, 0.8)       # 曲率范围(-1~1),负=左弯,正=右弯
ROAD_WIDTH_RANGE = (0.35, 0.50)      # 道路宽度占图像宽度的比例

# ========== 数据增强参数 ==========
AUGMENT_PROB = 0.8                    # 应用增强的概率
BRIGHTNESS_RANGE = (0.6, 1.4)         # 亮度变化范围
CONTRAST_RANGE = (0.7, 1.3)           # 对比度变化范围
NOISE_STD = 0.02                      # 高斯噪声标准差
SHIFT_PIXELS = 5                      # 随机平移像素数

# ========== 模型参数 ==========
# NVIDIA PilotNet 架构
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ========== 训练参数 ==========
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-5
LR_PATIENCE = 3
LR_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 8
VALIDATION_SPLIT = 0.15

# ========== 评估参数 ==========
MAX_SPEED_PIXELS_PER_STEP = 4         # 模拟器中每步前进像素数
STEERING_SMOOTHING = 0.3              # 模拟器转向平滑系数
OFFROAD_PENALTY_MULTIPLIER = 2.0      # 偏离道路的权重

# ========== Colab配置 ==========
USE_TPU = False                        # Colab上使用TPU加速
DATA_CACHE_IN_MEMORY = True            # 小数据集时缓存到内存加速

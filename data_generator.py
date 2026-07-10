"""合成道路数据生成器

在没有真实驾驶数据集的情况下, 生成带有标注转向角度的合成道路图像。
通过调整道路曲率、宽度、光照等参数, 产生多样化的训练数据。

生成原理:
  道路在图像中从底部的"近处"延伸到顶部的"地平线处"。
  每一行(水平扫描线)上,道路中心的位置由曲率函数决定:
    center_x = W/2 + curvature * t² * scale
  其中 t 是从地平线到图像底部的归一化位置。
  curvature > 0 → 道路右弯 → 转向角为正
  curvature < 0 → 道路左弯 → 转向角为负
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

from config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    CURVATURE_RANGE, ROAD_WIDTH_RANGE,
    AUGMENT_PROB, BRIGHTNESS_RANGE, CONTRAST_RANGE,
    NOISE_STD, SHIFT_PIXELS,
    NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES, NUM_TEST_SAMPLES,
    DATA_DIR,
)


@dataclass
class RoadSample:
    """一条道路样本"""
    image: np.ndarray        # (H, W, 3), float32, [0, 1]
    steering: float          # 目标转向角, [-1, 1]
    curvature: float         # 生成时使用的曲率
    road_width: float        # 生成时使用的道路宽度


def _rgb(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """将0-255的RGB值转为0-1的float"""
    return (r / 255, g / 255, b / 255)


# ========== 配色方案 ==========
SKY_COLORS = [
    _rgb(135, 206, 235),   # 晴空蓝
    _rgb(173, 216, 230),   # 浅蓝
    _rgb(100, 149, 237),   # 矢车菊蓝
    _rgb(119, 136, 153),   # 灰蓝(阴天)
    _rgb(255, 200, 100),   # 黄昏橙
]

GROUND_COLORS = [
    _rgb(80, 140, 60),     # 草绿
    _rgb(90, 130, 50),     # 深草绿
    _rgb(120, 100, 60),    # 土黄
    _rgb(150, 140, 120),   # 沙色
]

ROAD_COLORS = [
    _rgb(80, 80, 80),      # 沥青灰
    _rgb(90, 90, 90),      # 浅沥青
    _rgb(70, 70, 75),      # 深沥青
]

LANE_COLOR = _rgb(255, 255, 255)    # 白线
SHOULDER_COLOR = _rgb(255, 200, 0)  # 黄线(路肩)


def generate_road_image(curvature: float, road_width: float,
                        rng: np.random.Generator = None) -> np.ndarray:
    """生成一张合成道路图像

    Args:
        curvature: 曲率 [-1, 1], 负=左弯, 正=右弯, 0=直道
        road_width: 道路宽度占图像宽度的比例
        rng: 随机数生成器

    Returns:
        image: (IMG_HEIGHT, IMG_WIDTH, 3), float32, [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = IMG_HEIGHT, IMG_WIDTH

    # 1. 创建背景
    sky_color = SKY_COLORS[rng.integers(len(SKY_COLORS))]
    ground_color = GROUND_COLORS[rng.integers(len(GROUND_COLORS))]
    road_color = ROAD_COLORS[rng.integers(len(ROAD_COLORS))]

    image = np.zeros((H, W, 3), dtype=np.float32)

    # 天空区域(顶部40%)
    horizon = int(H * 0.4)
    image[:horizon] = sky_color

    # 地面区域(底部60%)
    image[horizon:] = ground_color

    # 2. 计算道路边界
    # 从地平线到图像底部,每一行计算道路左右边界
    left_bdry, right_bdry = [], []
    road_half_max = int(W * road_width / 2)

    for y in range(horizon, H):
        # t: 0=地平线处, 1=图像底部
        t = (y - horizon) / (H - horizon)

        # 道路中心偏移: curvature * t² 产生抛物线型弯道
        center_offset = curvature * (t ** 2) * (W * 0.25)
        center_x = W // 2 + int(center_offset)

        # 道路宽度随距离变化(透视效果:远处变窄)
        half_w = int(road_half_max * (0.15 + 0.85 * t))
        left = max(0, center_x - half_w)
        right = min(W - 1, center_x + half_w)

        left_bdry.append((left, y))
        right_bdry.append((right, y))

    # 3. 绘制道路(填充多边形)
    road_poly = np.array(left_bdry + right_bdry[::-1], dtype=np.int32)
    cv2.fillPoly(image, [road_poly], road_color)

    # 4. 绘制路肩(道路两侧的彩色边缘)
    shoulder_width = max(2, int(W * 0.01))
    shoulder_color = SHOULDER_COLOR

    for i in range(len(left_bdry)):
        lx, ly = left_bdry[i]
        rx, ry = right_bdry[i]
        if lx > 0:
            image[ly, max(0, lx - shoulder_width):lx] = shoulder_color
        if rx < W - 1:
            image[ry, rx:min(W, rx + shoulder_width)] = shoulder_color

    # 5. 绘制车道线
    lane_color = LANE_COLOR
    # 在道路中心画虚线
    for y in range(horizon, H):
        t = (y - horizon) / (H - horizon)
        center_offset = curvature * (t ** 2) * (W * 0.25)
        center_x = W // 2 + int(center_offset)
        half_w = int(road_half_max * (0.15 + 0.85 * t))

        # 在道路左右1/3处画车道线(左右各一条)
        for lane_pos in [-0.5, 0.5]:
            lx = int(center_x + lane_pos * half_w * 0.8)
            if 0 <= lx < W:
                # 虚线效果:每隔一段画线
                dash_period = 8
                if (y // dash_period) % 2 == 0:
                    image[y, max(0, lx-1):min(W, lx+1)] = lane_color

    # 6. 增加地面纹理变化
    texture = rng.uniform(-0.03, 0.03, (H - horizon, W, 3))
    image[horizon:] = np.clip(image[horizon:] + texture, 0, 1)

    return image


def apply_augmentation(image: np.ndarray, steering: float,
                       rng: np.random.Generator = None) -> Tuple[np.ndarray, float]:
    """数据增强

    模拟不同的光照条件、天气变化和拍摄角度变化。

    Args:
        image: 输入图像 (H, W, 3), float32
        steering: 原始转向角
        rng: 随机数生成器

    Returns:
        (增强后的图像, 调整后的转向角)
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.uniform() > AUGMENT_PROB:
        return image, steering

    img = image.copy()

    # 1. 亮度调整
    brightness = rng.uniform(*BRIGHTNESS_RANGE)
    img = np.clip(img * brightness, 0, 1)

    # 2. 对比度调整
    contrast = rng.uniform(*CONTRAST_RANGE)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = np.clip((img - mean) * contrast + mean, 0, 1)

    # 3. 添加高斯噪声
    noise = rng.normal(0, NOISE_STD, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)

    # 4. 水平平移(模拟车辆在车道内横向偏移)
    if rng.uniform() < 0.5:
        shift = rng.integers(-SHIFT_PIXELS, SHIFT_PIXELS + 1)
        if shift > 0:
            img[:, :-shift] = img[:, shift:]
            img[:, -shift:] = 0
        elif shift < 0:
            img[:, -shift:] = img[:, :shift]
            img[:, :-shift] = 0
        # 平移时转向角需要微调(偏移越大转向修正越大)
        steering_adjust = shift / IMG_WIDTH * 0.5
        steering = np.clip(steering + steering_adjust, -1, 1)

    # 5. 随机颜色偏移
    color_shift = rng.uniform(-0.05, 0.05, 3).astype(np.float32)
    img = np.clip(img + color_shift, 0, 1)

    return img, steering


def generate_dataset(num_samples: int, augment: bool = True,
                     seed: int = None) -> List[RoadSample]:
    """生成完整数据集

    Args:
        num_samples: 样本数
        augment: 是否应用数据增强
        seed: 随机种子

    Returns:
        RoadSample列表
    """
    rng = np.random.default_rng(seed)
    samples = []
    progress_interval = max(1, num_samples // 10)

    for i in range(num_samples):
        # 随机采样曲率和道路宽度
        curvature = rng.uniform(*CURVATURE_RANGE)
        road_width = rng.uniform(*ROAD_WIDTH_RANGE)

        # 让直道(曲率接近0)占比多一些, 更接近真实驾驶
        if rng.uniform() < 0.2:
            curvature = 0.0

        # 生成图像
        image = generate_road_image(curvature, road_width, rng)

        # 转向角 = 曲率的线性映射到 [-1, 1]
        steering = curvature

        # 数据增强
        if augment:
            image, steering = apply_augmentation(image, steering, rng)

        samples.append(RoadSample(
            image=image,
            steering=steering,
            curvature=curvature,
            road_width=road_width,
        ))

        if (i + 1) % progress_interval == 0:
            print(f"  生成进度: {i+1}/{num_samples}")

    return samples


def save_dataset(samples: List[RoadSample], save_dir: Path):
    """将数据集保存到磁盘"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num = len(samples)
    images = np.stack([s.image for s in samples], axis=0)  # (N, H, W, 3)
    steerings = np.array([s.steering for s in samples], dtype=np.float32)

    np.save(str(save_dir / "images.npy"), images)
    np.save(str(save_dir / "steerings.npy"), steerings)

    # 保存元数据
    with open(str(save_dir / "metadata.txt"), "w") as f:
        f.write(f"num_samples: {num}\n")
        f.write(f"image_shape: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}\n")
        f.write(f"steering_range: [{steerings.min():.3f}, {steerings.max():.3f}]\n")

    print(f"  已保存 {num} 个样本到 {save_dir}")


def load_dataset(save_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从磁盘加载数据集"""
    save_dir = Path(save_dir)
    images = np.load(str(save_dir / "images.npy"))
    steerings = np.load(str(save_dir / "steerings.npy"))
    return images, steerings


def generate_all_data():
    """生成训练/验证/测试全部数据集"""
    print("=" * 50)
    print("生成训练数据...")
    print(f"  样本数: {NUM_TRAIN_SAMPLES}")
    train_data = generate_dataset(NUM_TRAIN_SAMPLES, augment=True, seed=42)
    save_dataset(train_data, DATA_DIR / "train")

    print("\n生成验证数据...")
    print(f"  样本数: {NUM_VAL_SAMPLES}")
    val_data = generate_dataset(NUM_VAL_SAMPLES, augment=False, seed=100)
    save_dataset(val_data, DATA_DIR / "val")

    print("\n生成测试数据...")
    print(f"  样本数: {NUM_TEST_SAMPLES}")
    test_data = generate_dataset(NUM_TEST_SAMPLES, augment=False, seed=200)
    save_dataset(test_data, DATA_DIR / "test")

    print("\n全部数据生成完成!")
    print(f"  训练集: {DATA_DIR / 'train'}")
    print(f"  验证集: {DATA_DIR / 'val'}")
    print(f"  测试集: {DATA_DIR / 'test'}")

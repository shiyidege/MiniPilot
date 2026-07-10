"""驾驶决策预览：场景 → 决策 → 结果 可视化"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from pathlib import Path

from config import IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIR
from data_generator import generate_road_image
from simulator import Track, Simulator


def _draw_steering_gauge(ax, value, label="转向角", max_abs=1.0):
    """在轴上画一个转向角仪表盘"""
    ax.clear()
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # 半圆刻度
    angles = np.linspace(np.pi, 0, 7)
    for a in angles:
        x, y = np.cos(a), np.sin(a)
        ax.plot([0, x], [0, y], color="lightgray", linewidth=0.5)
        tick_val = (a / np.pi - 1) * 2  # -1 到 1
        label_x, label_y = x * 1.15, y * 1.15
        ax.text(label_x, label_y, f"{tick_val:.0f}", ha="center", va="center",
                fontsize=8, color="gray")

    # 方向盘图标
    circle = plt.Circle((0, 0), 0.15, fill=False, color="gray", linewidth=1.5)
    ax.add_patch(circle)
    ax.plot([0, 0], [-0.15, 0.15], color="gray", linewidth=1)
    ax.plot([-0.15, 0.15], [0, 0], color="gray", linewidth=1)

    # 指针
    angle = -value * np.pi / 2  # 左负右正
    ptr_len = 0.9
    ax.arrow(0, 0, np.sin(angle) * ptr_len, np.cos(angle) * ptr_len,
             head_width=0.08, head_length=0.08, fc="red", ec="red", linewidth=2)

    ax.set_title(f"{label}\n{value:+.4f}", fontsize=10)


def _make_car_icon(ax, lateral_error, max_lane=16):
    """在道路横截面上标注车辆位置"""
    ax.clear()
    ax.set_xlim(-max_lane * 1.5, max_lane * 1.5)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # 道路区域
    ax.axvspan(-max_lane, max_lane, alpha=0.2, color="green")
    ax.axvline(-max_lane, color="orange", linestyle="--", linewidth=1)
    ax.axvline(max_lane, color="orange", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.5)

    # 车辆(三角形)
    car_x = lateral_error
    ax.plot(car_x, 0, marker="^", markersize=15, color="blue")
    ax.annotate(f"偏离: {lateral_error:+.1f}px",
                xy=(car_x, 0), xytext=(car_x, 0.5),
                ha="center", fontsize=9, color="blue")

    ax.set_title("车辆横向位置（俯视截面）", fontsize=10)


def preview_driving(weights_path: str, model_type: str = "smallnet",
                    num_frames: int = 8, run_name: str = None):
    """生成驾驶决策预览图

    选几个有代表性的弯道场景，展示每帧的：
    - 道路图像（模型看到的）
    - 模型预测的转向角（仪表盘）
    - 理想转向角（仪表盘）
    - 车辆在道路中的横向位置

    Args:
        weights_path: 模型权重路径
        model_type: 模型类型
        num_frames: 预览帧数
        run_name: 运行名称(用于保存)
    """
    print("=" * 60)
    print("驾驶决策预览")
    print("=" * 60)

    # 加载模型
    from models.pilotnet import build_model
    model = build_model(model_type)
    model.load_weights(weights_path)
    print(f"已加载权重: {weights_path}")

    # 手动挑选几个典型弯道: 直道→左缓弯→左急弯→直道→右缓弯→右急弯→直道
    curvature_profile = [0.0, -0.1, -0.3, -0.5, 0.0, 0.1, 0.3, 0.5, 0.0,
                         -0.2, 0.4, -0.4, 0.2, 0.0]

    # 采样 num_frames 个点
    indices = np.linspace(0, len(curvature_profile) - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(num_frames, 4, figsize=(16, 3 * num_frames))

    for i, idx in enumerate(indices):
        curvature = curvature_profile[idx]
        road_width = 0.4
        gt_steering = curvature

        # 生成道路图像
        rng = np.random.default_rng(idx + 100)
        image = generate_road_image(curvature, road_width, rng)

        # 模型预测
        pred = float(model.predict(image[np.newaxis, ...], verbose=0)[0, 0])

        # 列1: 道路图像
        ax = axes[i, 0] if num_frames > 1 else axes[0]
        ax.imshow(image)
        ax.set_title(f"场景 {i+1}: 曲率={curvature:+.1f}", fontsize=9)
        ax.axis("off")

        # 列2: 模型预测转向
        ax = axes[i, 1] if num_frames > 1 else axes[1]
        _draw_steering_gauge(ax, pred, f"模型决策", max_abs=1.0)
        if abs(pred - gt_steering) < 0.05:
            ax.set_facecolor((0.9, 1.0, 0.9))  # 绿色背景=准确
        else:
            ax.set_facecolor((1.0, 0.9, 0.9))  # 红色背景=偏差大

        # 列3: 理想转向(ground truth)
        ax = axes[i, 2] if num_frames > 1 else axes[2]
        _draw_steering_gauge(ax, gt_steering, f"理想转向(GT)", max_abs=1.0)

        # 列4: 车辆横向位置模拟
        # 简单模拟: 如果预测偏差大, 车辆会偏离中心
        pred_error = pred - gt_steering
        lateral_error = pred_error * 30  # 累积偏差放大效果
        ax = axes[i, 3] if num_frames > 1 else axes[3]
        _make_car_icon(ax, lateral_error)

    plt.tight_layout()
    save_path = str(OUTPUT_DIR / f"{run_name}_preview.png") if run_name else str(OUTPUT_DIR / "driving_preview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"预览图已保存: {save_path}")
    plt.show()


def preview_simulation(weights_path: str, model_type: str = "smallnet",
                       num_frames: int = 12, run_name: str = None):
    """从实际模拟中抽取关键帧展示

    在 Track 上跑完整模拟，然后均匀采样 num_frames 帧展示。
    """
    print("=" * 60)
    print("模拟过程关键帧预览")
    print("=" * 60)

    from models.pilotnet import build_model
    model = build_model(model_type)
    model.load_weights(weights_path)
    print(f"已加载权重: {weights_path}")

    # 运行模拟
    track = Track(length=500)
    sim = Simulator(model, track)
    history = sim.run()
    results = sim.evaluate(history)
    print(f"模拟完成: lane_keep_rate={results['lane_keep_rate']:.1%}")

    # 采样帧
    total_frames = len(history["steerings"])
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(num_frames, 4, figsize=(16, 3 * num_frames))

    for row, idx in enumerate(indices):
        image = history["pred_images"][idx]
        pred = history["steerings"][idx]
        gt = history["gt_steerings"][idx]
        lat_err = history["lateral_errors"][idx]

        n = num_frames
        # 列1: 场景
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"步骤 {idx}", fontsize=9)
        axes[row, 0].axis("off")

        # 列2: 预测转向
        _draw_steering_gauge(axes[row, 1], pred, f"模型决策")
        match = abs(pred - gt) < 0.05
        axes[row, 1].set_facecolor((0.9, 1.0, 0.9) if match else (1.0, 0.9, 0.9))

        # 列3: 理想转向
        _draw_steering_gauge(axes[row, 2], gt, "理想转向")

        # 列4: 横向位置
        _make_car_icon(axes[row, 3], lat_err)

    plt.tight_layout()
    save_path = str(OUTPUT_DIR / f"{run_name}_sim_preview.png") if run_name else str(OUTPUT_DIR / "sim_preview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"模拟预览图已保存: {save_path}")
    plt.show()

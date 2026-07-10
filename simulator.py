"""驾驶模拟器

在合成道路上模拟自动驾驶, 验证模型在实际闭环中的表现。

模拟原理:
  1. 生成一条测试道路(由一系列连续变化的曲率定义)
  2. 在每个位置, 渲染前视图像 → 模型预测转向角
  3. 根据转向角更新车辆位置
  4. 测量车辆是否保持在道路内、偏离了多少

评估指标(有ground truth道路位置可参考):
  - 平均车道偏移: 车辆偏离道路中心的程度
  - 车道保持率: 车辆在道路内的帧数比例
  - 路径跟随误差: 车辆轨迹与理想轨迹的偏差
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import List, Tuple, Optional

from config import (
    IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIR,
    MAX_SPEED_PIXELS_PER_STEP, STEERING_SMOOTHING,
)
from data_generator import generate_road_image


class Track:
    """测试道路"""

    def __init__(self, length: int = 500, seed: int = 42):
        """
        Args:
            length: 道路长度(步数)
            seed: 随机种子
        """
        rng = np.random.default_rng(seed)
        self.length = length

        # 生成平滑变化的曲率序列(使用低通滤波)
        raw_curvatures = rng.uniform(-0.6, 0.6, length)
        # 滑动平均平滑
        window = 20
        kernel = np.ones(window) / window
        self.curvatures = np.convolve(raw_curvatures, kernel, mode="same")
        # 裁剪到合法范围
        self.curvatures = np.clip(self.curvatures, -0.8, 0.8)
        # 让开头和结尾都是直道
        self.curvatures[:10] = 0
        self.curvatures[-10:] = 0

        self.road_width = 0.4

    def get_curvature_at(self, position: int) -> float:
        """获取指定位置的曲率"""
        pos = min(position, self.length - 1)
        return float(self.curvatures[pos])

    def render_view(self, position: int) -> np.ndarray:
        """渲染该位置的前视图像"""
        curvature = self.get_curvature_at(position)
        return generate_road_image(curvature, self.road_width)


class Simulator:
    """驾驶模拟器"""

    def __init__(self, model, track: Track):
        self.model = model
        self.track = track

    def run(self) -> dict:
        """模拟驾驶全过程

        Returns:
            history: {
                "positions": [(x, y), ...],     # 车辆路径
                "steerings": [float, ...],       # 预测转向角
                "gt_steerings": [float, ...],    # 理想转向角
                "offroad": [bool, ...],          # 是否偏离道路
            }
        """
        # 车辆状态
        car_x = 0.0  # 横向偏移(正=右)
        car_y = 0.0  # 纵向位置
        car_heading = 0.0  # 朝向角
        speed = MAX_SPEED_PIXELS_PER_STEP

        steering_smooth = 0.0  # 平滑后的转向角

        history = {
            "positions": [(car_x, car_y)],
            "steerings": [],
            "gt_steerings": [],
            "lateral_errors": [],
            "pred_images": [],
        }

        for step in range(self.track.length):
            # 1. 获取当前位置的ground truth
            gt_curvature = self.track.get_curvature_at(step)
            gt_steering = gt_curvature  # 曲率直接映射为转向角

            # 2. 渲染前视图像
            image = self.track.render_view(step)

            # 3. 模型预测转向角
            img_input = image[np.newaxis, ...]
            pred = float(self.model.predict(img_input, verbose=0)[0, 0])

            # 4. 平滑转向(防止抖动)
            steering_smooth = (
                STEERING_SMOOTHING * steering_smooth +
                (1 - STEERING_SMOOTHING) * pred
            )

            # 5. 更新车辆状态
            #   假设车辆以恒定速度前进
            #   转向使车辆横向移动: dx = sin(heading) * speed
            car_heading += steering_smooth * 0.08  # 转向系数
            car_heading = np.clip(car_heading, -0.5, 0.5)
            dx = np.sin(car_heading) * speed
            car_x += dx
            car_y += speed

            # 6. 计算横向误差(相对于理想路径)
            #   理想路径在曲率为0时保持在x=0
            lateral_error = car_x  # 简化: 假设理想路径在x=0

            # 7. 判断是否偏离道路
            max_lane_width = IMG_WIDTH * 0.2 / 2  # 道路半宽(像素)
            offroad = abs(lateral_error) > max_lane_width

            # 记录
            history["positions"].append((car_x, car_y))
            history["steerings"].append(steering_smooth)
            history["gt_steerings"].append(gt_steering)
            history["lateral_errors"].append(lateral_error)
            history["pred_images"].append(image)

        return history

    def evaluate(self, history: dict) -> dict:
        """计算模拟驾驶评估指标

        有ground truth道路位置可参考:
          - 车辆应该保持在道路内
          - 路径应该平滑
        """
        lateral_errors = np.array(history["lateral_errors"])
        steerings = np.array(history["steerings"])
        gt_steerings = np.array(history["gt_steerings"])

        # RMS横向误差
        rms_lateral = float(np.sqrt(np.mean(lateral_errors ** 2)))

        # 车道保持率: 横向误差在合理范围内的比例
        max_lane_width = IMG_WIDTH * 0.2 / 2
        on_road_rate = float(np.mean(np.abs(lateral_errors) < max_lane_width))

        # 路径跟随误差(转向角MSE)
        steering_error = float(np.mean((steerings - gt_steerings) ** 2))

        # 最大偏差
        max_deviation = float(np.max(np.abs(lateral_errors)))

        # 转向平滑度
        steering_jerk = float(np.std(np.diff(steerings)))

        return {
            "rms_lateral_error": rms_lateral,
            "lane_keep_rate": on_road_rate,
            "steering_mse_vs_gt": steering_error,
            "max_deviation_pixels": max_deviation,
            "steering_jerk_std": steering_jerk,
        }

    def visualize(self, history: dict, save_path: str = None):
        """可视化模拟结果"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        positions = np.array(history["positions"])
        steerings = history["steerings"]
        gt_steerings = history["gt_steerings"]
        lateral_errors = history["lateral_errors"]

        steps = np.arange(len(steerings))

        # 1. 车辆路径(俯视图)
        ax = axes[0]
        ax.plot(positions[:, 0], positions[:, 1], "b-", label="车辆路径", linewidth=1)
        ax.set_xlabel("横向位置")
        ax.set_ylabel("纵向位置")
        ax.set_title("车辆行驶路径")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        # 2. 转向角对比
        ax = axes[1]
        ax.plot(steps, gt_steerings, "g--", alpha=0.7, label="理想转向角")
        ax.plot(steps, steerings, "b-", alpha=0.7, label="预测转向角")
        ax.set_xlabel("时间步")
        ax.set_ylabel("转向角")
        ax.set_title("转向角跟踪")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 横向误差
        ax = axes[2]
        ax.plot(steps, lateral_errors, "r-", alpha=0.7)
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        half_width = IMG_WIDTH * 0.2 / 2
        ax.axhline(half_width, color="orange", linestyle="--", alpha=0.7, label="道路边界")
        ax.axhline(-half_width, color="orange", linestyle="--", alpha=0.7)
        ax.fill_between(steps, -half_width, half_width, alpha=0.1, color="green")
        ax.set_xlabel("时间步")
        ax.set_ylabel("横向误差(像素)")
        ax.set_title("车道偏离监测")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  模拟结果图已保存: {save_path}")
        plt.show()


def run_simulation(model, run_name: str = None):
    """运行完整驾驶模拟"""
    print("=" * 60)
    print("驾驶模拟")
    print("=" * 60)

    track = Track(length=500)
    simulator = Simulator(model, track)

    print("\n开始模拟...")
    history = simulator.run()
    print(f"模拟完成: {track.length} 步")

    results = simulator.evaluate(history)
    print("\n模拟评估结果:")
    print(f"  RMS横向误差:     {results['rms_lateral_error']:.2f} 像素")
    print(f"  车道保持率:       {results['lane_keep_rate']:.1%}")
    print(f"  转向MSE(对GT):   {results['steering_mse_vs_gt']:.4f}")
    print(f"  最大偏差:         {results['max_deviation_pixels']:.2f} 像素")
    print(f"  转向平滑度:       {results['steering_jerk_std']:.4f}")

    save_path = str(OUTPUT_DIR / f"{run_name}_simulation.png") if run_name else None
    simulator.visualize(history, save_path)

    return results

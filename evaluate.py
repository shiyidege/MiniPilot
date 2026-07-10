"""模型评估模块

评估自动驾驶模型的多项指标:
  - MSE / MAE: 转向角预测误差
  - 误差分布分析: 不同曲率下的表现差异
  - 车道偏离率: 模拟器中是否跑出道路
  - 转向平滑度: 连续预测的抖动程度

评估思路:
  回归任务的评估相对直接——因为有ground truth转向角。
  但仅看MSE不够, 还需要在模拟器中验证"实际驾驶表现"。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    IMG_HEIGHT, IMG_WIDTH,
    OUTPUT_DIR,
)
from data_generator import load_dataset


class DrivingEvaluator:
    """驾驶模型评估器"""

    def __init__(self, model):
        self.model = model

    def evaluate_regression(self, images: np.ndarray,
                            steerings: np.ndarray) -> Dict[str, float]:
        """回归指标评估

        直接比较模型预测值和ground truth之间的误差。
        有明确的理想值(ground truth), 可以算差值。

        Args:
            images: (N, H, W, 3) 测试图像
            steerings: (N,) ground truth转向角

        Returns:
            metrics字典
        """
        preds = self.model.predict(images, verbose=0).flatten()

        mse = float(np.mean((preds - steerings) ** 2))
        mae = float(np.mean(np.abs(preds - steerings)))
        r2 = float(1 - np.sum((preds - steerings) ** 2) / np.sum(
            (steerings - np.mean(steerings)) ** 2))

        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mae,
            "r2_score": r2,
        }

    def evaluate_by_curvature(self, images: np.ndarray,
                              steerings: np.ndarray) -> Dict[str, Dict]:
        """按曲率分段评估

        考察模型在直道/缓弯/急弯下的表现差异。

        这里根据ground truth转向角分段:
          - 直道: |steering| < 0.1
          - 缓弯: 0.1 <= |steering| < 0.4
          - 急弯: |steering| >= 0.4
        """
        preds = self.model.predict(images, verbose=0).flatten()
        abs_gt = np.abs(steerings)

        masks = {
            "直道 (|s|<0.1)": abs_gt < 0.1,
            "缓弯 (0.1≤|s|<0.4)": (abs_gt >= 0.1) & (abs_gt < 0.4),
            "急弯 (|s|≥0.4)": abs_gt >= 0.4,
        }

        results = {}
        for name, mask in masks.items():
            if mask.sum() == 0:
                continue
            mse = float(np.mean((preds[mask] - steerings[mask]) ** 2))
            mae = float(np.mean(np.abs(preds[mask] - steerings[mask])))
            results[name] = {
                "count": int(mask.sum()),
                "mse": mse,
                "mae": mae,
            }

        return results

    def evaluate_smoothness(self, images: np.ndarray,
                            steerings: np.ndarray) -> Dict[str, float]:
        """转向平滑度评估

        连续帧之间的转向角变化应该平滑, 不能剧烈抖动。
        用相邻预测差值的标准差来衡量。
        """
        preds = self.model.predict(images, verbose=0).flatten()

        # 预测的帧间变化
        diff_pred = np.diff(preds)
        diff_gt = np.diff(steerings)

        return {
            "smoothness_jerk_std": float(np.std(diff_pred)),
            "smoothness_jerk_mean": float(np.mean(np.abs(diff_pred))),
            "gt_jerk_std": float(np.std(diff_gt)),
        }

    def full_evaluation(self, images: np.ndarray,
                        steerings: np.ndarray) -> Dict:
        """完整评估"""
        results = {}
        results["regression"] = self.evaluate_regression(images, steerings)
        results["by_curvature"] = self.evaluate_by_curvature(images, steerings)
        results["smoothness"] = self.evaluate_smoothness(images, steerings)
        return results


def plot_predictions(images: np.ndarray, steerings: np.ndarray,
                     preds: np.ndarray, save_path: str = None):
    """绘制预测 vs ground truth对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 预测 vs 真实散点图
    ax = axes[0, 0]
    ax.scatter(steerings, preds, alpha=0.3, s=2)
    ax.plot([-1, 1], [-1, 1], "r--", lw=2, label="理想线")
    ax.set_xlabel("Ground Truth 转向角")
    ax.set_ylabel("预测转向角")
    ax.set_title("预测 vs 真实")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 误差直方图
    ax = axes[0, 1]
    errors = preds - steerings
    ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("预测误差")
    ax.set_ylabel("频数")
    ax.set_title(f"误差分布  (MSE={np.mean(errors**2):.4f})")
    ax.grid(True, alpha=0.3)

    # 3. 误差 vs 曲率
    ax = axes[1, 0]
    abs_steering = np.abs(steerings)
    ax.scatter(abs_steering, np.abs(errors), alpha=0.3, s=2)
    ax.set_xlabel("|转向角| (弯道急缓)")
    ax.set_ylabel("|误差|")
    ax.set_title("误差 vs 弯道急缓")
    ax.grid(True, alpha=0.3)

    # 4. 连续序列误差(前200个样本)
    ax = axes[1, 1]
    n_show = min(200, len(steerings))
    x = np.arange(n_show)
    ax.plot(x, steerings[:n_show], "b-", alpha=0.7, label="真实")
    ax.plot(x, preds[:n_show], "r-", alpha=0.7, label="预测")
    ax.set_xlabel("样本序号")
    ax.set_ylabel("转向角")
    ax.set_title("前200个样本的预测对比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图表已保存: {save_path}")
    plt.show()


def run_evaluation(model, test_images, test_steerings, run_name: str):
    """运行完整评估并生成报告"""
    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    evaluator = DrivingEvaluator(model)
    results = evaluator.full_evaluation(test_images, test_steerings)

    print("\n回归指标:")
    for k, v in results["regression"].items():
        print(f"  {k}: {v:.4f}")

    print("\n曲率分段评估:")
    for name, metrics in results["by_curvature"].items():
        print(f"  {name}: 样本数={metrics['count']}, "
              f"MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    print("\n转向平滑度:")
    for k, v in results["smoothness"].items():
        print(f"  {k}: {v:.4f}")

    # 绘制图表
    preds = model.predict(test_images, verbose=0).flatten()
    save_path = str(OUTPUT_DIR / f"{run_name}_evaluation.png") if run_name else None
    plot_predictions(test_images, test_steerings, preds, save_path)

    return results

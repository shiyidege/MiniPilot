"""主入口"""

import sys
import argparse
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR
from data_generator import generate_all_data
from train import train_model


def cmd_generate(args):
    """生成合成数据"""
    generate_all_data()


def cmd_train(args):
    """训练模型"""
    model, history, run_name = train_model(
        model_type=args.model,
        resume_path=args.resume,
        max_samples=args.max_samples,
    )

    # 自动评估
    print("\n" + "=" * 60)
    print("自动评估...")
    print("=" * 60)

    from data_generator import load_dataset
    from evaluate import run_evaluation

    test_images, test_steerings = load_dataset(DATA_DIR / "test")
    run_evaluation(model, test_images, test_steerings, run_name)

    # 模拟驾驶
    from simulator import run_simulation
    run_simulation(model, run_name)


def cmd_evaluate(args):
    """评估已有模型"""
    from data_generator import load_dataset
    from models.pilotnet import build_model
    from evaluate import run_evaluation

    # 加载模型权重
    model = build_model(args.model_type)
    model.load_weights(args.weights)
    print(f"已加载权重: {args.weights}")

    # 评估
    test_images, test_steerings = load_dataset(DATA_DIR / "test")
    run_evaluation(model, test_images, test_steerings, args.run_name or "eval")


def cmd_simulate(args):
    """运行模拟驾驶"""
    from models.pilotnet import build_model
    from simulator import run_simulation

    model = build_model(args.model_type)
    model.load_weights(args.weights)
    print(f"已加载权重: {args.weights}")

    run_simulation(model, args.run_name or "sim")


def cmd_preview(args):
    """驾驶决策预览"""
    if args.mode == "scene":
        from preview import preview_driving
        preview_driving(args.weights, args.model_type, args.frames, args.run_name)
    else:
        from preview import preview_simulation
        preview_simulation(args.weights, args.model_type, args.frames, args.run_name)


def cmd_demo(args):
    """生成效果展示图片"""
    import matplotlib.pyplot as plt
    from data_generator import generate_road_image, apply_augmentation
    import numpy as np

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))

    curvatures = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.0, 0.0, 0.0]
    titles = ["左急弯", "左缓弯", "直道", "右缓弯",
              "右急弯", "原始", "增强后", "增强后"]

    for i in range(4):
        img = generate_road_image(curvatures[i], 0.4, rng)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{titles[i]}  (c={curvatures[i]:.1f})")
        axes[0, i].axis("off")

    # 展示数据增强效果
    orig = generate_road_image(0.0, 0.4, rng)
    axes[1, 0].imshow(orig)
    axes[1, 0].set_title("原始直道")
    axes[1, 0].axis("off")

    for i in range(1, 4):
        aug, _ = apply_augmentation(orig.copy(), 0.0, rng)
        axes[1, i].imshow(aug)
        axes[1, i].set_title(f"增强 {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    save_path = str(OUTPUT_DIR / "data_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"样本图已保存: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="自动驾驶算法项目")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # generate
    p = subparsers.add_parser("generate", help="生成合成道路数据")

    # train
    p = subparsers.add_parser("train", help="训练模型")
    p.add_argument("--model", default="smallnet", choices=["pilotnet", "smallnet"],
                   help="模型类型")
    p.add_argument("--resume", help="从已有权重恢复训练")
    p.add_argument("--max_samples", type=int, default=None,
                   help="限制训练样本数(内存不足时使用, 如8000)")

    # evaluate
    p = subparsers.add_parser("evaluate", help="评估模型")
    p.add_argument("--weights", required=True, help="模型权重路径")
    p.add_argument("--model_type", default="smallnet", help="模型类型")
    p.add_argument("--run_name", help="运行名称")

    # simulate
    p = subparsers.add_parser("simulate", help="驾驶模拟")
    p.add_argument("--weights", required=True, help="模型权重路径")
    p.add_argument("--model_type", default="smallnet", help="模型类型")
    p.add_argument("--run_name", help="运行名称")

    # demo
    subparsers.add_parser("demo", help="生成效果展示图")

    # preview
    p = subparsers.add_parser("preview", help="驾驶决策预览: 场景→决策→结果")
    p.add_argument("--weights", required=True, help="模型权重路径")
    p.add_argument("--model_type", default="smallnet", help="模型类型")
    p.add_argument("--frames", type=int, default=8, help="预览帧数")
    p.add_argument("--mode", default="scene", choices=["scene", "sim"],
                   help="scene=独立场景, sim=模拟中采样")
    p.add_argument("--run_name", help="运行名称(用于保存文件名)")

    # full pipeline
    p = subparsers.add_parser("full", help="完整流水线: 生成→训练→评估→模拟")
    p.add_argument("--model", default="smallnet", choices=["pilotnet", "smallnet"])
    p.add_argument("--max_samples", type=int, default=None,
                   help="限制训练样本数(内存不足时使用, 如8000)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "preview":
        cmd_preview(args)
    elif args.command == "full":
        cmd_generate(args)
        train_args = argparse.Namespace(model=args.model, resume=None, max_samples=args.max_samples)
        cmd_train(train_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

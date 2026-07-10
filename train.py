"""训练脚本"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

from config import (
    DATA_DIR, MODEL_DIR, RUNS_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, LEARNING_RATE_MIN,
    LR_PATIENCE, LR_FACTOR, EARLY_STOPPING_PATIENCE,
    USE_TPU, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
)


def create_tf_dataset(images, steerings,
                      batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """将numpy数组转换为 tf.data.Dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((images, steerings))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(images), 5000))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_memmap_dataset(data_dir: Path, batch_size: int,
                           shuffle: bool = True) -> tf.data.Dataset:
    """内存映射方式加载 .npy, 避免一次性加载到 RAM"""
    images = np.load(str(data_dir / "images.npy"), mmap_mode='r')
    steerings = np.load(str(data_dir / "steerings.npy"), mmap_mode='r')
    return create_tf_dataset(images, steerings, batch_size, shuffle)


def count_samples(data_dir: Path) -> int:
    """获取 .npy 文件中的样本数(只读头部, 不加载数据)"""
    steerings = np.load(str(data_dir / "steerings.npy"), mmap_mode='r')
    n = len(steerings)
    del steerings
    return n


def train_model(model_type: str = "pilotnet", resume_path: str = None,
                max_samples: int = None):
    """训练驾驶模型

    Args:
        model_type: "pilotnet" 或 "smallnet"
        resume_path: 可选, 从已有权重恢复训练
        max_samples: 可选, 限制训练样本数(内存不足时使用)

    Returns:
        (训练后的模型, 历史记录)
    """
    print("=" * 60)
    print("自动驾驶模型训练")
    print("=" * 60)

    # 1. 设置TPU(Colab上可用)
    if USE_TPU:
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print(f"TPU可用, 设备数: {strategy.num_replicas_in_sync}")
        except:
            print("TPU不可用, 回退到GPU/CPU")
            strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()
        print(f"使用策略: {strategy.__class__.__name__}")

    # 2. 加载数据
    print("\n[1/5] 加载数据...")
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    num_train = count_samples(train_dir)
    num_val = count_samples(val_dir)

    if max_samples and max_samples < num_train:
        print(f"  限制训练样本数: {num_train} → {max_samples} (内存优化)")
        num_train = max_samples

    print(f"  训练集: {num_train} 样本")
    print(f"  验证集: {num_val} 样本")
    print(f"  图像尺寸: ({IMG_HEIGHT}, {IMG_WIDTH}, {IMG_CHANNELS})")

    # 3. 构建模型
    print("\n[2/5] 构建模型...")
    from models.pilotnet import build_model

    with strategy.scope():
        model = build_model(model_type)

        if resume_path:
            model.load_weights(resume_path)
            print(f"  已加载权重: {resume_path}")

        model.summary()

    # 4. 配置回调
    print("\n[3/5] 配置回调...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"
    log_dir = RUNS_DIR / run_name
    checkpoint_dir = MODEL_DIR / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # 学习率衰减: 验证loss不下降时就减小lr
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LEARNING_RATE_MIN,
            verbose=1,
        ),
        # 早停: 防止过拟合
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # 保存最佳模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # TensorBoard日志
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
        ),
    ]

    # 5. 训练
    print("\n[4/5] 开始训练...")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  最大轮数: {EPOCHS}")
    print(f"  日志目录: {log_dir}")

    # 使用内存映射数据集, 避免全部加载到内存
    train_ds = create_memmap_dataset(train_dir, BATCH_SIZE)
    val_ds = create_memmap_dataset(val_dir, BATCH_SIZE, shuffle=False)

    # 如果限制了样本数, 取前N个
    if max_samples and max_samples < count_samples(train_dir):
        train_ds = train_ds.take(max_samples // BATCH_SIZE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. 保存最终模型
    print("\n[5/5] 保存模型...")
    model_path = str(checkpoint_dir / "final.weights.h5")
    model.save_weights(model_path)
    print(f"  模型权重: {model_path}")

    # 保存为SavedModel格式(便于部署)
    export_dir = str(checkpoint_dir / "saved_model")
    model.export(export_dir)
    print(f"  SavedModel: {export_dir}")

    print(f"\n训练完成! 运行名称: {run_name}")
    print(f"  最佳验证loss: {min(history.history['val_loss']):.6f}")
    print(f"  最佳验证MAE:  {min(history.history['val_mae']):.6f}")

    return model, history, run_name


def predict_steering(model: tf.keras.Model, image: np.ndarray) -> float:
    """单张图像预测转向角

    Args:
        model: 训练好的模型
        image: (H, W, 3) 或 (1, H, W, 3), float32, [0, 1]

    Returns:
        steering: [-1, 1], 负=左转, 正=右转
    """
    if image.ndim == 3:
        image = image[np.newaxis, ...]
    pred = model.predict(image, verbose=0)
    return float(pred[0, 0])


if __name__ == "__main__":
    train_model()

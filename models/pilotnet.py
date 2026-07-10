"""NVIDIA PilotNet 端到端自动驾驶模型

PilotNet 是 NVIDIA 在 2016 年提出的经典端到端驾驶模型(论文: End to End
Learning for Self-Driving Cars)。输入摄像头图像,直接输出转向角。

架构:
  Input: (H, W, 3)  →  Normalization
    → Conv1: 5×5, 24, stride 2  →  ELU
    → Conv2: 5×5, 36, stride 2  →  ELU
    → Conv3: 5×5, 48, stride 2  →  ELU
    → Conv4: 3×3, 64, stride 1  →  ELU
    → Conv5: 3×3, 64, stride 1  →  ELU
    → Flatten
    → FC1: 100  →  ELU  →  Dropout(0.3)
    → FC2: 50   →  ELU  →  Dropout(0.3)
    → FC3: 10   →  ELU
    → Output: 1 (steering angle)

训练后使用:
  加载模型 → 输入图像 → 前向传播 → 得到[-1, 1]的转向角
  负值=左转, 正值=右转, 0=直行
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from config import INPUT_SHAPE, LEARNING_RATE, LEARNING_RATE_MIN


def build_pilotnet(input_shape=INPUT_SHAPE, l2_reg=1e-5) -> Model:
    """构建 NVIDIA PilotNet

    Args:
        input_shape: (H, W, C)
        l2_reg: L2正则化系数

    Returns:
        Keras Model (未编译)
    """
    reg = regularizers.l2(l2_reg)

    inputs = Input(shape=input_shape, name="camera_input")

    # 归一化层(在训练时学习均值和方差)
    x = layers.Normalization(name="normalize")(inputs)

    # === 卷积特征提取 ===
    x = layers.Conv2D(24, (5, 5), strides=(2, 2),
                      kernel_regularizer=reg, name="conv1")(x)
    x = layers.ELU()(x)

    x = layers.Conv2D(36, (5, 5), strides=(2, 2),
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.ELU()(x)

    x = layers.Conv2D(48, (5, 5), strides=(2, 2),
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.ELU()(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      kernel_regularizer=reg, name="conv4")(x)
    x = layers.ELU()(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1),
                      kernel_regularizer=reg, name="conv5")(x)
    x = layers.ELU()(x)

    # === 全连接回归 ===
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(100, kernel_regularizer=reg, name="fc1")(x)
    x = layers.ELU()(x)
    x = layers.Dropout(0.3, name="drop1")(x)

    x = layers.Dense(50, kernel_regularizer=reg, name="fc2")(x)
    x = layers.ELU()(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    x = layers.Dense(10, kernel_regularizer=reg, name="fc3")(x)
    x = layers.ELU()(x)

    # 输出: 转向角 [-1, 1], tanh激活确保输出范围
    outputs = layers.Dense(1, activation="tanh", name="steering")(x)

    model = Model(inputs=inputs, outputs=outputs, name="PilotNet")
    return model


def build_smallnet(input_shape=INPUT_SHAPE) -> Model:
    """轻量版PilotNet, Colab上训练更快"""
    inputs = Input(shape=input_shape, name="camera_input")

    x = layers.Normalization(name="normalize")(inputs)

    x = layers.Conv2D(16, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(48, (3, 3), strides=(1, 1), activation="relu")(x)
    x = layers.Conv2D(48, (3, 3), strides=(1, 1), activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="tanh")(x)

    return Model(inputs=inputs, outputs=outputs, name="SmallNet")


def build_model(model_type: str = "pilotnet") -> Model:
    """工厂方法: 根据类型构建模型"""
    if model_type == "pilotnet":
        model = build_pilotnet()
    elif model_type == "smallnet":
        model = build_smallnet()
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


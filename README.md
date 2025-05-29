# learn_pytorch
This repository is for practicing the basic usage of pytorch by learning from tudui.

# PyTorch学习项目
## 项目简介
这是一个基于PyTorch框架的深度学习学习项目，记录了我在学习PyTorch过程中的各种实践和探索。项目涵盖了从基础的张量操作到复杂的CNN模型训练的完整学习路径，是我学习深度学习的成长记录。

## 项目内容
### 基础组件学习
- 张量操作 ：学习PyTorch的基本数据结构和操作
- 神经网络模块 ：实现了各种神经网络层的基本操作
  - 卷积层（Conv2d）
  - 池化层（MaxPool2d）
  - 激活函数（Sigmoid, ReLU等）
  - 全连接层（Linear）
- 损失函数 ：实现和使用交叉熵损失函数
- 优化器 ：使用SGD优化器进行模型训练
### 数据处理
- 数据加载 ：使用DataLoader加载数据集
- 数据转换 ：使用transforms进行数据预处理和增强
- 自定义数据集 ：实现了自定义Dataset类
### 模型构建与训练
- 模型定义 ：构建了用于图像分类的CNN模型
- 模型训练 ：实现了完整的训练循环
- 模型评估 ：计算测试集上的准确率和损失
- 模型保存 ：保存训练好的模型参数
### 高级特性
- GPU加速 ：使用CUDA加速模型训练
- TensorBoard可视化 ：使用TensorBoard记录和可视化训练过程
- Sequential模型 ：使用Sequential容器简化模型构建
## 数据集
项目使用了以下数据集：

- CIFAR-10 ：包含10个类别的彩色图像数据集
- 蚂蚁和蜜蜂图像 ：用于二分类任务的自定义数据集
## 项目结构
```
├── dataloader.py          # 数据加载器示例
├── dataset_transform.py   # 数据集转换示例
├── model.py               # 模型定义
├── nn_conv.py             # 卷积层学习
├── nn_conv2d.py           # 二维卷积层学习
├── nn_loss.py             # 损失函数学习
├── nn_maxpool.py          # 最大池化层学习
├── nn_module.py           # 神经网络模块学习
├── nn_optim.py            # 优化器学习
├── nn_seq.py              # Sequential容器学习
├── nn_sigmoid.py          # Sigmoid激活函数学习
├── read_data.py           # 数据读取示例
├── test.py                # 测试脚本
├── test_tb.py             # TensorBoard测试
├── train.py               # 模型训练
├── train_gpu1.py          # GPU加速训练示例1
├── train_gpu2.py          # GPU加速训练示例2
└── transform.py           # 数据转换示例
```
## 环境要求
- Python 3.6+
- PyTorch 1.0+
- torchvision
- tensorboard
- PIL
- numpy
## 安装与使用
1. 克隆仓库
```
git clone https://github.com/Ethan-GY/
learn_pytorch.git
cd learn_pytorch
```
2. 安装依赖
```
pip install torch torchvision tensorboard 
pillow numpy
```
3. 运行示例
```
# 基础模型训练
python train.py

# GPU加速训练
python train_gpu1.py

# TensorBoard可视化
python test_tb.py
```
## TensorBoard可视化
本项目使用TensorBoard进行训练过程的可视化，可以通过以下命令启动TensorBoard：

```
tensorboard --logdir=logs
```
然后在浏览器中访问 http://localhost:6006 查看可视化结果。

## 学习心得
通过这个项目，我系统地学习了PyTorch的基本用法和深度学习的核心概念。从最基础的张量操作，到构建复杂的CNN模型，再到使用GPU加速训练，每一步都加深了我对深度学习的理解。

特别是在实现CNN模型对CIFAR-10数据集的分类任务中，我学会了如何处理数据、构建模型、训练模型以及评估模型性能。这些经验对我未来的深度学习研究和应用都有很大帮助。

## 未来计划
- 实现更复杂的网络架构（如ResNet, VGG等）
- 探索更多的数据增强技术
- 尝试迁移学习和微调预训练模型
- 应用所学知识解决实际问题

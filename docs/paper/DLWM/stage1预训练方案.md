# 智驾端到端模型预训练方案

该方案以图像 backbone 为中心，通过 **LiDAR 深度与几何监督构建 3D 感知能力**、**DinoV3特征蒸馏监督单帧图像特征**，使用**隐式空间(BEV)世界模型训练**进一步提升对未来的推演能力。

## 一、整体框架概述

本框架的核心目标是：输入多视角环视视频序列，经过预训练后得到一个具备3D空间理解能力、语义丰富性和时序稳定性的图像 backbone，可直接用于下游 BEV感知、3D检测、端到端规划等任务。

整体架构由以下两个核心模块构成：

1. **3D几何和特征预训练**：利用LiDAR点云提供的深度与几何信息作为监督，迫使图像 backbone 学会从2D图像中推理3D结构；以冻结的DinoV3作为语义teacher，指导图像backbone学习语义丰富的视觉特征。
2. **隐式世界模型训练**：在隐式特征空间(e.g. BEV)进行未来场景的推演监督训练，迫使图像backbone提升对未来场景的推演能力。

两个模块在预训练阶段联合优化，使 image backbone / BEV 同时获得**几何、语义、未来推演**等能力，从而形成一个真正适用于端到端自动驾驶的**视觉特征提取模型**。

## 二、部署目标与芯片适配

> 地平线J6P：560 TOPS，4核BPU® Nash，支持混合精度int8/int16/fp32。

我们选择HENet作为模型的image backbone，HENet-TinyM为J6系列专门设计，纯CNN架构，无Deformable Conv等算子，在BPU上效率极高。

## 三、模块一：几何与语义预训练（阶段一）

![image-20260610214322108](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610214323175.png)

![image-20260610214346542](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610214346810.png)

### 3.1 深度与几何监督

#### 3.1.1 几何监督GT

Lidar metric pointcloud

#### 3.1.2 像素级Token重采样

`HEnet` 直接输入多视角原始图像序列 $I\in R^{B×T×V×3×H×W}$,原位执行 1/16 尺度的密集空间下采样，随后，特征图在通道维度重构并沿空间维度平铺拉扁，转换为密集视觉 Token 序列$X_{patch}\in R^{(B⋅T⋅V)×N_{patches}×C}$，作为时空骨干网的几何输入基底；

#### 3.1.3 Ray token embedding

在线计算出每一个 Patch 中心绝对对齐的 6D 物理射线场$r=[o,d]$（3D光心原点与3D单位方向矢量），经由 `RayEmbedding` 模块映射为空间几何偏置，叠加至视觉 Token 上；

#### 3.1.4 Looped Block Vit trunk

为了最大化释放分布式集群的硬件吞吐率与车端推理效率，打破了vggt/dvgt trunk 使用24 层不同参数 Transformer 纵向堆叠的范式，引入 **Déjà View Style ，Single Looped Block，Trunk 在物理结构和独特参数量上仅包含一层独特的时空注意力特征块，特征序列在进入 Trunk 后，在原地进行 K 次自回归循环空转演进；**

#### 3.1.5 Single Dense Head

 替换DPT 3D conv 为 mlp + pixel shuffle 输出当前自身坐标系下的 ego pts

#### 3.1.6 Future Point Head

经 1x1 卷积直接外推外推下一帧场景 3D ego pts 的真实物理位移 $dX,dY,dZ$，通过与相邻时刻在 GPU 端在线动态生成真实相邻帧 ego pts（`gt_future_displacement`）计算 L1 连续性损失 $L_{temp}$，在完全无任何人工 3D 框标注的情况下，以极低开销让模型自发剥离出运动前景与静态背景的时空物理边界。

#### 3.1.7 ego pose Head

Trunk 输出的 ego token decode 成 pose，使用自车pose 进行监督；

### 3.2 语义监督：DINOv3蒸馏

与单帧方案相同，在每个时刻t，将backbone特征与冻结的DINOv3教师特征对齐：

两种监督方式（倾向于后者）：

- 2D L1 直连

语义学习（DINO 特征对齐）和几何重建学习（DenseHead 的深度 Huber Loss）是**完全解耦、两条分立的后向导数通路, 它们之间会在卷积核内部为了参数空间的分配而发生很大梯度分歧。**

- 3DGS render

$$
\frac{∂L_{dino}}{∂Backbone}=\frac{∂L_{dino}}{∂\hat{F}_{render}}⋅\frac{∂\hat{F}_{render}}{∂(μ,α,Σ)}⋅\frac{∂(μ,α,Σ)}{∂Backbone}
$$

**自监督语义与几何学习是自洽的。**

## 四、模块二：隐式世界模型预训练（阶段二）

### 4.1 BEV特征生成


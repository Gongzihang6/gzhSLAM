# 基于 3D Gaussian 的感知模型预训练

## 1. 核心理念与阶段目标

Stage1 预训练的目标是：把 3D Gaussian Lifter 和 DINOv3 当作高纬度的“训练工具”，通过 3D 几何约束和开放世界语义约束，倒逼 HENet 同时学会以下能力：

- **3D 空间与几何感知能力：** 能从单调的 2D 纹理中推断出遮挡关系、绝对深度和空间结构；
- **泛化语义提取能力：** 能像视觉大模型（DINOv3）一样，深刻理解复杂、长尾的交通场景语义；

## 2. 训练期的模型架构

在训练阶段搭建了如下“学生-代理-教师”结构：

- **学生网络 (Student)**：`HENet-TinyM + FPN`。负责提取多视角的 2D 特征图。
- **3D 高斯生成器 (Gaussian Lifter)**：直接接在 FPN 之后。直接在 2D 平面上高效预测每个像素的深度期望值以及对应的 3D 高斯基元参数，随后结合相机内外参，将其“反投影”提升到 3D 空间。
- **教师网络 (Teacher)**：`DINOv3 (ViT-H+/16, 840M参数量)`。参数完全冻结，仅用作提供带有极强表征能力的 2D 多尺度语义特征作为“软标签”。

### 2.1 损失函数 (Loss Functions)

HENet 能力的跃升（即 3D 几何结构与大模型语义的成功注入），主要依赖以下三大核心损失函数的联合反向传播：

**A. 深度监督：稠密深度 SILog 损失（`L_silog`，权重 1.0）**

- **监督目标**：为纯 2D 主干网络注入 3D 空间结构先验。
- **输入真值**：采用由 Metric3D 离线生成的稠密深度伪真值（`dense_depth_gt`）。
- **计算机制**：Gaussian Lifter 模块首先通过预测离散的 depth bins 计算出像素级绝对深度的期望值（`depth_pred`）。随后，将其与稠密深度真值代入 **尺度不变对数损失 (SILog Loss)** 中进行比对。SILog 函数能有效惩罚深度结构的相对拓扑误差，而不会对绝对尺度的全局偏移过度敏感。
- **核心收益**：这一强监督信号迫使底层的 HENet 在提取特征时，必须“理解”画面的遮挡关系和几何纵深，补足了纯视觉网络天生缺乏的测距能力。

**B. 语义蒸馏：GS 渲染特征对齐损失（`L_gs_distill`，权重 0.1）**

- **监督目标**：向网络注入大模型的开放世界高维泛化语义。
- **输入真值**：被冻结的教师网络 DINOv3 从原始 2D 图像中提取的高质量目标特征图（`dino_feats`）。
- **计算机制 (3D 到 2D 的降维对齐)**：
    a.  **3D 特征生成**：Gaussian Lifter 为每一个生成的 3D 高斯基元分配并预测一个 256 维的蒸馏特征向量。
    b.  **可微渲染降维**：利用高斯可微渲染技术，将散布在 3D 空间中带有 256 维特征的高斯体重新“拍扁”渲染回 2D 相机透视平面，得到预测的 2D 特征图（`distill_feat_pred`）。
    c.  **余弦相似度计算**：将渲染出的特征图与 DINOv3 的真值特征图进行 **余弦相似度 (Cosine Similarity)** 比对，计算误差。
- **核心收益**：大模型的知识精华主要体现在高维特征向量的方向上（而非模长）。渲染加余弦对齐的组合，能最有效地迫使网络吸收 DINOv3 对复杂交通场景的深层理解。

**C. 重建约束：RGB 渲染外观损失（`L_rgb`，权重 1.0）**

- **监督目标**：维持高斯体物理外观与几何形态的基础合理性。
- **输入真值**：真实的相机多视角原始 RGB 图像（`rgb_gt`）。
- **计算机制**：通过渲染 3D 高斯的 RGB 颜色和不透明度 (Opacity) 属性，在 2D 平面生成重建图像（`rgb_pred`），并与真实原图计算 MSE (均方误差) 损失。
- **核心收益**：作为兜底的基础监督信号，它能确保生成的 3D 高斯体在宏观形状、位置分布和透明度上符合真实的物理世界光学规律，防止高斯点云的空间分布发生彻底发散。

## 3. 中间产物可视化

**gt 图像 vs 渲染图像**

![image-20260610213317471](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213413349.png)

| DINOv3                                                       | HENet                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20260610213510654](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213510924.png) | ![image-20260610213621492](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213621702.png) |
| ![image-20260610213643845](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213644109.png) | ![image-20260610213709596](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213709798.png) |
| ![image-20260610213823328](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213823449.png) | ![image-20260610213840335](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213840424.png) |
| ![image-20260610213927256](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213927367.png) | ![image-20260610213949852](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610213949962.png) |

![image-20260610214200729](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610214200920.png)



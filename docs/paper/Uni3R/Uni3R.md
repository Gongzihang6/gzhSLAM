# Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images

## 引言

从稀疏的二维视角中重建三维场景，并进行语义解析，仍然是计算机视觉领域的一项根本性挑战。传统方法通常将语义理解与三维重建过程解耦分开进行，或者需要代价高昂的逐场景优化，从而限制了其可扩展性和泛化能力。本文提出了 Uni3R，一种新颖的前馈式框架，能够直接从无姿态标注的多视角图像中联合重建富含开放词汇语义的统一三维场景表示。我们的方法利用 Cross-View Transformer 对任意多视角输入进行鲁棒的信息融合，并回归出一组带有语义特征场的三维高斯图元。该统一表示支持高保真的新视角合成、开放词汇的三维语义分割以及深度预测，全部过程均通过单次前馈完成。大量实验表明，Uni3R 在多个基准上实现了新的最先进性能，包括在 RE10K 上达到 25.07 的 PSNR，在 ScanNet 上达到 55.84 的 mIoU。本工作标志着迈向可泛化的统一三维场景重建与理解的新范式。代码发布于 <https://github.com/HorizonRobotics/Uni3R>

从稀疏图像中感知和理解三维世界是计算机视觉的核心能力，对机器人、自动驾驶和增强现实等领域具有深远影响。尽管近年来有很多非常逼真的三维场景重建方法，比如 NeRF、3DGS，然而它们非常耗时，且需要逐场景优化。因此，大量可泛化性的三维重建方法陆续出现，通过在大量不同场景中学习几何先验，预训练之后，以单次前馈的方式实现前馈式三维重建。但是这些研究通常仅关注几何和外观，忽视了对整体场景理解至关重要的语义丰富性。

虽然近期的研究，如 LangSplat 和 Feature-3DGS 已将语义场融合到 3D 高斯泼溅中，但是仍受限于场景特定的优化，且在现实世界中的零样本应用中可扩展性不足。最近，LSM 和 UniForward 等方法尝试统一语义场与辐射场，以实现几何、外观、语义的联合预测。然而，这些方法基于 DUSt3R，本质上是为双试图输入设计的，因此，将其扩展至多视图场景需要在视图间进行昂贵的逐对特征匹配，导致效率降低，并因缺乏全局 3D 上下文而产生不一致的重建结果。

为解决这些局限性，我们提出了 Uni3R，一种新颖且具有泛化能力的框架，能够从任意多视角图像中合成统一的 3D 表示，以同时实现高保真渲染和密集的开放词汇语义理解。Uni3R 利用 Cross-View Transformer 有效融合跨视角信息并生成全局一致的表示，预测出富含开放词汇语义特征的统一 3D 高斯基元。这些高斯表示可通过源图像进行端到端监督，在无需逐场景优化的情况下实现实时无缝渲染以合成新视角。同时，嵌入的语义特征可通过任意文本提示对场景进行查询，实现零样本 3D 语义分割。

为进一步提升几何保真度和训练稳定性，我们引入了一种点图引导的几何损失（point-map-guided geometric loss），该损失具有两个关键作用。首先，它增强了结构一致性并提高了几何精度，表现为更低的深度误差（例如 AbsRel）。其次，它通过防止模型在预测自由度较高的 3D 点分布时陷入局部极小值，从而稳定了训练过程。具体而言，我们采用一个冻结的 VGGT [36] 来生成带有置信度分数的稠密点图，这些点图作为软几何先验，用于指导 3D 高斯分布的空间布局。

我们的贡献总结如下：

- 我们提出了 Uni3R，一种新颖的前馈架构，统一了 3D 重建与语义理解。该方法在单次前向传播中预测一组融合了几何、外观和开放词汇语义的高斯基元（Gaussian primitives），**无需针对每个场景进行单独优化**。
- 我们展示了强大的几何基础模型可有效扩展至几何估计之外的任务，支持光度重建与 3D 场景理解。其跨帧注意力机制实现了鲁棒的特征融合，能够从任意数量的输入视图中生成全局一致的场景表示，而其预测的点图提供了强有力的几何引导。
- Uni3R 在多个任务上取得了最先进的性能，包括在具有挑战性的 RE10K [46] 和 ScanNet [6] 数据集上的 **新视角合成、开放词汇 3D 语义分割以及深度预测**，充分体现了其卓越的泛化能力与多功能性。

## 方法

![image-20260521105842044](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260521105842185.png)

### Feed-Forward Gaussian Splatting

#### Intrinsic Embedding

为了解决单目重建中因焦距未知导致的固有尺度模糊问题，我们引入了内参嵌入来提供关键的几何线索。参考 NoPoSplat [40]，我们通过线性投影对每台相机的焦距和主点进行编码。在进行 Patch Tokenization（图像块特征化）之前，将得到的内参嵌入在通道维度上与对应图像进行拼接，从而赋予网络推理几何感知信息的能力。

#### Cross-View Transformer Encoder

借鉴 VGGT 的设计，Uni3R 采用跨视角 Transformer 编码器，将从各视角图像中提取的特征融合到一个一致且视角无关的隐空间（Latent Space）中。 每个输入视角 $I^{(i)}$ 连同其内参嵌入，首先通过预训练的 Vision Transformer 模型（DINOv2）来提取图像块级别的特征 Token（Patch-level Feature Tokens）。为了支持任意数量的多视图输入并保持排列等变性（Permutation Equivariance），每个视角的 Token 序列末尾都会拼接一个可学习的相机 Token（Camera Token）。

跨视角 Transformer 编码器由一系列 Transformer 模块组成，这些模块交替进行**帧内自注意力（Intra-frame Self-attention）**和**跨帧全局注意力（Cross-frame Global Attention）**：

- **帧内自注意力**：在单视图的 Token 集合内部进行，利用局部上下文优化单视图特征。
- **跨帧全局注意力**：聚合来自所有视图的 Token，用于构建视图间的对应关系并推断全局 3D 几何结构。 最终，编码器输出的隐式 Token 封装了对 3D 场景整体且全局一致的理解。

#### Decoding Gaussian Parameters

融合后的潜在表示通过密集预测Transformer（DPT）[30]被解码为一组密集的3D高斯基元，并由针对不同高斯参数的专用预测头进行进一步处理。DPT 逐步将中间层的细粒度局部细节融入粗糙的patch-level特征中，从而生成密集的逐像素特征图。随后，我们通过独立的 MLP 头部预测一组像素对齐的 3D 高斯分布的属性。每个基元由以下参数表示：

$$
 G_j = \{\mu_j, \alpha_j, c_j, s_j, r_j, f^{\text{sem}}_j\}, \tag{1}
$$

其中 $\mu_j \in \mathbb{R}^3$ 表示 3D 中心点，$s_j \in \mathbb{R}^3$ 是尺度，$r_j \in \mathbb{R}^4$ 为旋转四元数，$\alpha_j \in [0, 1]$ 为不透明度，$c_j \in \mathbb{R}^3$ 是颜色，而 $f^{\text{sem}}_j \in \mathbb{R}^d$ 是一个高维语义特征向量。

Point Head从预训练的 VGGT 权重初始化，并通过基于渲染的监督进一步微调，以对齐真实世界的度量尺度。对预测参数应用不同的激活函数以将其约束在有效范围内：

$$
 \alpha_j = \sigma(f^\alpha_j)
$$

$$
s_j = \exp(f^s_j) \cdot d_{\text{median}}
$$

$$
r_j = \text{normalize}(f^r_j)
$$

其中 $\sigma(\cdot)$ 表示 sigmoid 激活函数，$f^\alpha_j$、$f^s_j$ 和 $f^r_j$ 分别是不透明度、尺度和旋转的隐变量。$d_{\text{median}}$ 是从预测的 3D 位置计算得到的中位深度值，用于归一化尺度。

#### Rendering with Open-Vocabulary Semantics

在预测出 3D 高斯基元集合后，Uni3R 利用扩展了语义特征场的**可微分 3D 高斯光栅化器（Differentiable 3D Gaussian Rasterizer）**将其渲染为新视角图像 。为了缓解渲染高维语义特征带来的巨额显存开销，Uni3R 引入了一个自编码器（Autoencoder），在**三维空间体渲染前**对高斯基元的特征进行降维压缩。

**3D高斯基元特征压缩 (3D Primitive Compression)** 对于每个高斯基元，其预测的高维原始语义特征为 $f_j^{\mathrm{sem}}$ 。首先通过自编码器的编码器 $\mathcal{F}_{\mathrm{enc}}$ 将其压缩为低维语义特征 $\hat{f}_j^{\mathrm{sem}}$ ：

$$
\hat{f}_j^{\mathrm{sem}} = \mathcal{F}_{\mathrm{enc}}(f_j^{\mathrm{sem}})
$$

**低维语义体渲染 (Low-Dimensional Volume Rendering)** 每个像素处渲染的颜色 $\hat{I}$ 和**低维像素特征 $\hat{F}$**，是通过对所有重叠且已排序的高斯基元属性进行 Alpha 混合（Alpha-blending）计算得到的。以低维语义特征的渲染为例：

$$
\hat{F} = \sum_i \hat{f}_i^{\mathrm{sem}} \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

二维像素特征解压 (2D Pixel Feature Decompression) 在通过光栅化得到低维二维像素特征 $\hat{F}$ 后，利用自编码器的解码器 $\mathcal{F}_{\mathrm{dec}}$ 在二维图像平面将其恢复（解压）至高维的 CLIP 语义空间，得到最终的高维像素特征 $F'$ ：

$$
F' = \mathcal{F}_{\mathrm{dec}}(\hat{F})
$$

该自编码器采用端到端（End-to-end）的方式进行联合训练，强制使最终解压后的像素特征 $F'$ 与基于 CLIP 的 2D 图像特征保持对齐，从而实现了高效且无需 3D 标注的开放词汇（Open-Vocabulary）零样本语义推理。

在推理过程中，语义分割通过计算逐像素语义特征与一组文本衍生的原型之间的余弦相似度来实现。给定期望类别的文本提示集合（例如，“wall”，“chair”，“sofa”），CLIP文本编码器生成相应的特征原型 $f_{\text{txt}} \in \mathbb{R}^{N_C \times C}$，其中 $N_C$ 为类别数量。随后通过余弦相似度计算语义logits $S$：  

$$
S = \frac{F_{\text{sem}} \cdot f_{\text{txt}}^\top}{\|F_{\text{sem}}\| \cdot \|f_{\text{txt}}\|}
$$

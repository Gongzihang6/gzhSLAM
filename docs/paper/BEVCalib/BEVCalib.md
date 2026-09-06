# BEVCalib: Lidar-Camera Calibration via Geometry-Guided Bird’s-Eye View Representations

## 1. Abstract

准确的激光雷达-相机标定对于融合自动驾驶和机器人系统中的多模态感知至关重要。传统的标定方法需要在受控环境中进行大量数据采集，并且无法补偿车辆/机器人运动过程中的位姿变化。本文提出了首个利用鸟瞰图（BEV）特征直接从原始数据执行激光雷达-相机标定的模型，称为 BEVCALIB。为实现这一目标，我们分别提取相机 BEV 特征和激光雷达 BEV 特征，并将它们融合到共享的 BEV 特征空间中。为了充分利 用 BEV 特征中的几何信息，我们引入了一种新颖的特征选择器，用于在变换解码器中筛选最重要的特征，从而降低内存消耗并实现高效的训练。在 KITTI、NuScenes 和我们自建数据集上的大量实验表明，BEVCALIB 建立了新的最先进水平。在不同噪声条件下，BEVCALIB 在 KITTI 数据集上以（平移误差、旋转误差）衡量，平均优于文献中最佳基线模型达 (47.08%, 82.32%)，在 NuScenes 数据集上达 (78.17%, 68.29%)。在开源领域，其性能较最佳可复现基线提升了一个数量级。我们的代码和演示结果可在 <https://cisl.ucr.edu/BEVCalib> 获取。

## 3. Methodology

### 3.1 Architecture Overview

BEVCALIB 被设计为一种无需标定靶标的激光雷达-相机校准模型，其输入为包含单张图像和全场景激光雷达数据的场景，并从激光雷达到相机的变换中预测校准参数。图 1 展示了 BEVCALIB 的整体架构。它首先使用独立的骨干网络从相机图像和 LiDAR 中提取特定模态的 3D 特征（§3.2）。随后，这些特征被投影并融合到统一的 BEV 表示中，以捕捉语义上下文及几何信息。为增强 BEV 的空间表征能力，我们通过特征金字塔网络（Feature Pyramid Network, FPN）BEV 编码器聚合多尺度特征。接着，我们提出一种新颖的几何引导 BEV 特征解码器（Geometry-Guided BEV Decoder,  GGBD, §3.3）。该解码器首先利用由 3D  图像特征推导出的坐标信息引导几何引导特征选择器，使模型能够聚焦于空间上有意义的区域。最后，它引入一个精炼模块，从所选特征中解码标定参数，以实现高效且有效的训练。遵循基于学习的标定方法 [45, 9] 的惯例，表 1 总结了描述我们方法的符号。

![image-20260616095832043](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260616095832231.png)


| Symbol | Dimension | Description |
| :---: | :---: | :--- |
| $I$ | $\mathbb{R}^{H \times W \times 3}$ | RGB image captured by camera |
| $P$ | $\mathbb{R}^{N \times 3}$ | Point clouds captured by lidar, where $P_i = [X_i, Y_i, Z_i]$ |
| $K$ | $\mathbb{R}^{4 \times 4}$ | Intrinsic matrix of camera |
| $T_{gt}$ | $\mathbb{R}^{4 \times 4}$ | Ground truth transformation from lidar to camera |
| $T_{\Delta}$ | $\mathbb{R}^{4 \times 4}$ | Random noise input superimposed on $T_{init}$ |
| $T_{init}$ | $\mathbb{R}^{4 \times 4}$ | Initial guess extrinsic matrix input (including $T_{\Delta}$) |
| $T_{pred}$ | $\mathbb{R}^{4 \times 4}$ | Prediction extrinsic matrix as a correction to $T_{init}$ |

具体而言，BEVCALIB 的图像分支接收图像输入 $I$，并利用 $T_{\text{init}}$ 和 $K$ 生成三维锥体特征 $F^{\text{3D}}_C$（详见 §3.2）。同时，LiDAR 分支将 LiDAR 输入 $P$ 编码为体素特征 $F^{\text{3D}}_L$。这些特征随后被融合为 BEV 特征 $F_B$，并由 GGBD 组件解码以获得预测结果 $T_{\text{pred}}$。在训练与评估过程中，初始外参矩阵通过 **在真实值 $T_{\text{gt}}$ 上叠加随机噪声 $T_{\Delta}$ 构建，即 $T_{\text{init}} = T_{\Delta} \cdot T_{\text{gt}}$**（详见 §3.2）。由于 $T_{\Delta}$ 表示随机噪声，$T_{\Delta}$ 越大，$T_{\text{init}}$ 的错位程度越高，问题也越具挑战性。

在本设置中，我们考虑了最大范围为 $\{\pm1.5\,\text{m}, \pm20^\circ\}$ 的多种扰动幅值，构成一个真实且具有挑战性的校准场景。在评估阶段，BEVCALIB 以 $I$、$P$、$K$ 和 $T_{\text{init}}$ 为输入，输出预测 $T_{\text{pred}}$ 以补偿注入的噪声。最终的 LiDAR 到相机外参预测为 $\hat{T}_{\text{gt}} = T_{\text{pred}}^{-1} \cdot T_{\text{init}}$。该策略有助于在不引入标签泄露的情况下控制校准问题的难度。

### 3.2 BEV Feature Extraction

BEV 特征具有固有的几何意义，因为 BEV 空间中的每个特征对应于真实世界中的特定区域。在我们的设置中，使用 LiDAR 的坐标系作为世界坐标系，同时也作为 BEV 坐标系。受先前跨模态方法 [18] 的启发，我们采用类似的范式，即分别处理每种模态，然后将其融合到统一的 BEV 特征空间中。具体而言，LiDAR 分支使用稀疏卷积主干网络处理输入点云 $P$，生成体素特征 $F^L_{3D} \in \mathbb{R}^{N_L \times X \times Y \times Z}$，随后将其展平为 BEV 特征 $B^L_{2D} \in \mathbb{R}^{(N_L \times Z) \times X \times Y}$，其中 $X、Y$ 为 BEV 平面的空间尺寸，$Z$ 为沿高度轴的垂直体素数量。

图像分支采用一个二维主干网络和一个 LSS [42] 模块。该模型首先从相机输入 $I$ 中提取图像特征 $F_{\text{2D}}^C \in \mathbb{R}^{f_H \times f_W \times N_C}$，其中 $f_H$、$f_W$ 为图像特征的尺寸。LSS 模块为每个像素 $(u, v)$ 定义一组离散深度值，记作 $D = \left\{d_{\text{min}} + \frac{d_{\text{max}} - d_{\text{min}}}{D-1} \times i\right\}_{i=0}^{D-1}$，其中 $D$ 为离散深度区间的数量。对于每个像素 $(u, v)$，LSS 生成 $D$ 个点，总共构成一个包含 $f_H \times f_W \times D$ 个点的锥体。对应的三维特征表示为 $F_{\text{3D}}^C \in \mathbb{R}^{D \times f_H \times f_W \times N_C}$，相机坐标系下的三维位置定义为 $P_C \in \mathbb{R}^{D \times f_H \times f_W \times 3}$。为了给模型提供初始的位置猜测，通过 $P_{W}^{C} = [T_{\text{init}}^{-1} \cdot \tilde{P}^{C}]_{1:3}$ 将视锥坐标变换为世界坐标。最后，我们可以通过 BEV 池化 [2] 获得相机的 BEV 特征 $B_{C}^{2D} \in \mathbb{R}^{N_C \times X \times Y}$。（这里的 $T_{init}$ 是 lidar 到 cam 的变换矩阵的初值）

为了获得统一的 BEV 表示，我们使用 1×1 卷积来融合来自不同模态的特征，即 $F_B = \text{Conv}_{1D}([B_{2D}^C, B_{2D}^L]) \in \mathbb{R}^{N_B\times X\times Y}$。随后，我们采用一个 FPN BEV 编码器来增强 BEV 表示中的多尺度几何信息。

### 3.3 Geometry-Guided BEV Decoder(GGBD)

基于场景的几何 BEV 表示，我们进一步提出了一种几何引导的 BEV 特征解码器（Geometry-Guided BEV feature Decoder），以学习相机与 LiDAR 之间有意义的几何关系。如图 2 所示，该解码器包含两个阶段：特征选择器和优化模块。BEV 特征选择器引导模型关注具有显著空间信息的 BEV 特征，而优化模块则聚合高层特征，有助于预测最终的外参参数。

![image-20260616104053389](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260616104053639.png)

**几何引导的 BEV 特征选择器。** 具体而言，在特征选择器中，遵循 BEV 特征提取的图像分支，我们将 3D 特征位置 $P_C^W$ 作为锚点，通过将其投影到 BEV 空间以实现跨模态交互。 具体而言，对于一个三维位置 $p_c = (x, y, z) \in P^C_W$，其对应的 BEV 空间坐标计算为   $x_B = \frac{X}{2} + \left\lfloor \frac{x}{s} \right\rfloor, \quad y_B = \frac{Y}{2} + \left\lfloor \frac{y}{s} \right\rfloor$ 其中 $s$ 是 BEV 网格的分辨率大小。我们定义投影操作为 $\text{Proj}(p) = (x_B, y_B)$，所选 BEV 特征位置的集合可表示为：

$$
P_B = \text{Set}\left(\{\text{Proj}(p) \mid p \in P^C_W\}\right)
$$

由于 BEV 空间是不同模态共享的统一融合空间，此类投影位置 $(x_B, y_B) \in P_B$ 自然地为不同模态提供了强大的空间先验。该策略本质上聚焦于相机与 LiDAR 之间的重叠区域，充当一种隐式的几何匹配器，同时消除了冗余特征。

 **refine 模块。** 为了说明我们几何选择器的优势和通用性，我们仅使用 vanilla self-attention [46] 作为 refine 模块。GeometryGuided BEV 解码器（GGBD）的整个过程可表示为

$$
\text{GGBD}(P_W^C, F_B) = \text{Self-Attention} (\varphi_Q(F_\delta), \varphi_K (F_\delta), \varphi_V (F_\delta)) \tag{2}
$$

$$
 F_\delta = \{F_B [:, x_B, y_B] \mid (x_B, y_B) \in P_B\} \tag{3}
$$

**在 GGBD 之后，我们应用平均池化操作来聚合特征。随后，使用两个独立的多层感知机（MLPs）分别预测平移和旋转。最后，将预测的分量组装成最终的预测结果 $T_{pred}$。**

### 3.4 Calibration Optimization

BEVCALIB 输出一个平移向量 $t \in \mathbb{R}^3$ 和一个旋转四元数 $r \in \mathbb{R}^4$，其监督信号 $\hat{r}$ 和 $\hat{t}$ 由 $T_{\text{pred}}^\text{init} = T_{\text{init}} \cdot T^{-1}_{\text{gt}} = \begin{bmatrix} Q2M(\hat{r}) & \hat{t} \\ 0 & 1 \end{bmatrix}$ 推导得到，其中 $Q2M(\hat{r})$ 表示从四元数 $\hat{r}$ 转换得到的旋转矩阵。为了有效优化外参标定，我们设计了一组损失函数，分别专注于仅旋转、仅平移以及联合标定。

**旋转损失**。对于旋转监督，我们采用基于四元数距离的测地线损失 [47]： $\mathcal{L}_{\text{ang}} = 2\arctan2\left( \|q_\Delta^{(1:3)}\|_2, |q_\Delta^{(0)}| \right),$ 其中 $q_\Delta = r \cdot \hat{r}^{-1}$ 是 $r$ 与 $\hat{r}$ 之间的相对四元数，$\|\cdot\|_2$ 表示 l2 范数，$|\cdot|$ 表示绝对值。我们还使用归一化损失以约束预测的四元数 $r$ 成为有效的旋转，即 $\mathcal{L}_{\text{norm}} = (\|r\|_2 - 1)^2$。最终，旋转损失为： $\mathcal{L}_R = \mathcal{L}_{\text{ang}} + \lambda_{\text{norm}} \mathcal{L}_{\text{norm}}.$

**平移损失**。对于平移优化，我们使用 Smooth-L1 损失进行优化。我们发现该损失单独即可有效优化平移，因此未引入额外目标。平移损失定义为： $\mathcal{L}_T = \text{Smooth-L1}(t, \hat{t}).$

**重投影损失**。我们采用 LCCNet [9] 提出的点云重投影损失。具体而言，它能够利用预测的平移和旋转联合监督变换后点云的对齐情况，可表示为： $\mathcal{L}_{PC} = \frac{1}{N} \sum_{i=1}^{N} \|T^{-1}_{\text{gt}} \cdot T^{-1}_{\text{pred}} \cdot T_{\text{init}} \cdot \tilde{P}_i - \tilde{P}_i\|^2,$ 其中 $N$ 是给定点云 $P$ 中的点数。

 **总损失函数**。综上所述，联合损失函数为： $\mathcal{L} = \lambda_R \mathcal{L}_R + \lambda_T \mathcal{L}_T + \lambda_{PC} \mathcal{L}_{PC}.$

**实现细节**。我们使用稀疏卷积 [43] 作为 LiDAR 的主干网络，并采用 Swin-Transformer [48] 结合 LSS [42] 作为相机的主干网络。对于室内数据集，我们将环境范围限制在半径 9 米内；而对于室外数据集，则将范围扩展至 90 米。在整个训练过程中，我们对 $(\mathcal{L}_R, \mathcal{L}_T, \mathcal{L}_{PC})$ 损失使用权重向量 $(1.0, 0.5, 0.5)$。BEVCALIB 的训练仅使用单块 NVIDIA RTX 6000 Ada GPU，在每个数据集上以 batch size 16 训练 500 个 epoch（§4）。我们采用 AdamW 优化器，权重衰减为 $1\times10^{-4}$，初始学习率为 $5\times10^{-5}$，并通过 StepLR 调度器按 0.5 倍衰减学习率。



# BEVWorld: A Multimodal World Simulator for Autonomous Driving via Scene-Level BEV Latents

## Abstract

世界模型因其能够预测潜在未来场景而在自动驾驶领域受到越来越多的关注。本文提出BEVWorld，这是一种新颖的框架，可将多模态传感器输入转换为统一且紧凑的鸟瞰图（Bird's Eye View, BEV）隐空间，用于整体环境建模。所提出的该世界模型包含两个主要组成部分：一个多模态分词器和一个潜BEV序列扩散模型。多模态分词器首先对异构的感知数据进行编码，其**解码器则以自监督方式通过光线投射渲染将潜BEV分词重建为LiDAR和环视图像观测**。这实现了在共享空间表示中对全景图像与点云数据的联合建模以及双向编码-解码。在此基础上，潜BEV序列扩散模型基于高层动作分词对未来的场景进行时间一致性的预测，实现跨时间的场景级推理。大量实验表明了BEVWorld在自动驾驶基准上的有效性，展示了其在生成逼真未来场景方面的能力及其在感知和运动预测等下游任务中的优势。

## 1.Introduction

自动驾驶世界模型（DWMs）已成为自动驾驶中日益关键的组成部分，使车辆能够基于当前或历史观测结果预测未来场景。除了增强训练数据外，DWMs还提供支持端到端强化学习的真实模拟环境。这些模型赋能自动驾驶系统模拟多样化场景并做出高质量决策。

近年来，通用图像和视频生成技术的进展显著加速了自动驾驶领域生成模型的发展。该领域的大多数研究通常利用在大规模2D图像或视频数据集上预训练的开源模型 Rombach et al. (2022)。Blattmann 等人 (2023) 将其强大的生成能力应用于驾驶领域。这些工作 Wang 等人 (2023a)；Li 等人  (2024)；Gao 等人 (2023)  要么扩展了图像生成模型的时间维度，要么直接微调视频基础模型，在仅有有限驾驶数据的情况下实现了令人印象深刻的 2D 生成结果。

然而，驾驶场景的真实感模拟需要三维空间建模，而仅靠二维表示无法充分捕捉这一特性。一些驾驶世界模型（DWMs）采用三维表示作为中间状态 Zheng et al. (2024)，或直接预测三维结构，例如点云 Zhang et al. (2024)。然而，这些方法通常聚焦于单一的三维模态，难以适应现代自动驾驶系统中多传感器、多模态的特点。**由于多模态数据本身存在的异构性，将二维图像或视频与三维结构整合到一个统一的生成模型中仍然是一个开放性挑战。**

此外，许多现有模型采用端到端架构来建模从过去到未来的状态转移（Yang et al., 2024b；Zhou et al., 2025）。然而，高质量的图像和点云生成依赖于对低级像素或体素细节以及车辆、行人等场景元素的高级行为动态的联合建模。在未显式解耦这两个层次的情况下直接预测未来状态，往往限制了模型性能。

为应对这些挑战，我们提出 BEVWorld——一种多模态世界模型，可将异构传感器数据转换为统一的鸟瞰图（BEV）表示，从而在共享的空间中实现基于动作条件的未来预测。BEVWorld 包含两个解耦的组件：一个多模态 tokenizer 网络和一个潜 Bird's Eye View 序列扩散模型。Tokenizer 专注于低层次信息压缩与高保真重建，而扩散模型则以时间结构化的方式预测高层级行为。

**我们的多模态 tokenizer 的核心在于将原始传感器输入投影到统一的潜在 BEV 空间中。这通过将视觉特征转换到三维空间，并借助自监督自动编码器将其与基于 LiDAR 的几何结构对齐来实现。为了重建原始的多模态数据，我们将 BEV 潜在表示提升回 3D 体素表示，并应用基于射线的渲染方法 Yang et al. (2023) 来合成高分辨率图像和点云。**

BEV扩散模型专注于预测未来的BEV帧。得益于tokenizer提供的抽象表示，该任务被极大地简化。具体而言，我们采用基于扩散的生成方法结合时空Transformer，对潜在BEV序列进行去噪，以**在给定规划动作的条件下生成精确的未来预测。**

我们的主要贡献如下：   • 我们提出了一种新颖的多模态分词器，将视觉语义和3D几何统一到BEV表示中。通过利用基于渲染的重建方法，我们确保了高精度的BEV质量，并通过消融实验、可视化结果以及下游任务验证了其有效性。   • 我们设计了一种基于潜在扩散的世界模型，能够同步生成多视角图像和点云数据。在nuScenes和Carla数据集上的大量实验表明，该模型在多模态未来预测方面具有卓越的性能。

## 2.Related Work

## 3.Method

在本节中，我们阐述 BEVWorld 的模型结构。整体架构如图 1 所示。给定一个多视角图像和 LiDAR 观测序列 $\{o_{t−P} , · · · , o_{t−1}, o_t, o_{t+1}, · · · , o_{t+N}\}$，其中 $o_t$ 表示当前观测，+/− 分别表示未来/过去观测，P/N 表示过去/未来观测的数量，我们的目标是在给定条件 $\{o_{t-P}, \dots, o_{t-1}, o_t\}$ 的情况下，预测 $\{o_{t+1}, \dots, o_{t+N}\}$。鉴于在原始观测空间中学习世界模型的计算成本高昂，我们提出了一种多模态分词器（multi-modal tokenizer），以逐帧将多视角图像和激光雷达（LiDAR）信息压缩到一个统一的 BEV（鸟瞰图）空间中。编码器-解码器（encoder-decoder）结构和自监督重构损失保证了适当的几何和语义信息被良好地存储在 BEV 表示中。这一设计恰好为世界模型和其他下游任务提供了一种足够简洁的表示。我们的世界模型被设计为一个基于扩散（diffusion-based）的网络，以避免像自回归（auto-regressive）方式那样出现误差累积的问题。在训练过程中，它以自车运动（ego motion）和 $\{x_{t-P}, \dots, x_{t-1}, x_t\}$（即 $\{o_{t-P}, \dots, o_{t-1}, o_t\}$ 的 BEV 表示）作为条件，来学习添加到 $\{x_{t+1}, \dots, x_{t+N}\}$ 中的噪声 $\{\epsilon_{t+1}, \dots, \epsilon_{t+N}\}$。

![image-20260703114526773](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260703114526977.png)

### 3.1 Multi-Modal Tokenizer

我们设计的多模态分词器包含三个部分：BEV编码网络、BEV解码网络和多模态渲染网络。BEV编码网络的结构如图2所示。为了使多模态网络尽可能同质化，我们采用Swin-Transformer Liu et al. (2021) 网络作为图像主干网络以提取多视角图像特征。对于LiDAR特征提取，我们首先将点云在BEV空间中划分为柱体（pillars）Lang et al. (2019)。然后使用Swin-Transformer网络作为LiDAR主干网络来提取LiDAR BEV特征。我们通过基于可变形的Transformer（deformable-based transformer）Zhu et al. (2020) 融合LiDAR BEV特征和多视角图像特征。 具体地，我们在柱体的高度维度上采样 $K$（$K =  4$）个点，并将这些点投影到图像上以采样对应的图像特征。在可变形注意力计算中，采样的图像特征被视为值（values），而LiDAR  BEV特征则作为查询（queries）。考虑到未来预测任务需要低维输入，我们进一步将融合后的BEV特征压缩为低维（$C' = 4$）BEV特征。

![image-20260703143137603](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260703143138013.png)

对于BEV解码器，由于融合后的BEV特征缺乏高度信息，直接使用解码器恢复图像和LiDAR时存在歧义问题。为解决该问题，我们首先通过上采样层和Swin-blocks堆叠层将BEV令牌转换为3D体素特征，然后采用基于体素化NeRF的光线渲染来恢复多视角图像和LiDAR点云。

多模态渲染网络可以优雅地划分为两个独立组件：图像重建网络和LiDAR重建网络。对于图像重建网络，我们首先获得光线 $r(t) = o + td$，该光线从相机中心 $o$ 沿方向 $d$ 投射至像素中心。然后沿该光线均匀采样一组点 $\{(x_i, y_i, z_i)\}_{i=1}^{N_r}$，其中 $N_r( N_r = 150)$为沿一条光线采样的总点数。给定一个采样点 $(x_i, y_i, z_i)$，其对应特征 $v_i$ 根据其位置从体素特征中获取。随后，一条光线中所有采样的特征被聚合为逐像素的特征描述符（公式 1）

$$
v(r) = \sum_{i=1}^{N_r} w_i v_i, \quad w_i = \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j), \quad \alpha_i = \sigma(\mathrm{MLP}(v_i))
$$

![image-20260703143159106](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260703143159366.png)

我们遍历所有像素，得到图像的二维特征图 $V \in \mathbb{R}^{H_f \times W_f \times C_f}$。该二维特征通过CNN解码器转换为RGB图像 $I_g \in \mathbb{R}^{H \times W \times 3}$。为了提升生成图像的质量，引入了三种常见的损失：感知损失 Johnson et al. (2016)、GAN损失 Goodfellow et al. (2020) 和L1损失。我们的图像重建完整目标函数为：

$$
\mathcal{L}_{\mathrm{rgb}} = \|I_g - I_t\|_1 + \lambda_{\mathrm{perc}}\|\sum_{j=1}^{N_\phi} \phi^j(I_g) - \phi^j(I_t)\| + \lambda_{\mathrm{gan}}\mathcal{L}_{\mathrm{gan}}(I_g, I_t)
$$

其中 $I_t$ 是 $I_g$ 的真实值，$\varphi_j$ 表示预训练 VGG 模型 Simonyan & Zisserman (2014) 的第 $j$ 层，$\mathcal{L}_{\text{gan}}(I_g, I_t)$ 的定义可参考 Goodfellow et al. (2020)。

对于激光雷达（LiDAR）重建网络，射线在球坐标系中由倾角 $\theta$ 和方位角 $\phi$ 定义。$\theta$ 和 $\phi$ 是通过从 LiDAR 中心向当前帧的 LiDAR 点发射射线而获得的。我们采用与图像重建相同的方法对点进行采样并获取对应的特征。由于 LiDAR 编码了深度信息，因此需要计算采样点的期望深度 $D_g(\mathbf{r})$ 以用于 LiDAR 模拟。深度模拟过程和损失函数如公式 3 所示：

$$
D_g(\mathbf{r}) = \sum_{i=1}^{N_r} w_i t_i, \quad \mathcal{L}_{\text{Lidar}} = \|D_g(\mathbf{r}) - D_t(\mathbf{r})\|_1
$$

其中 $t_i$ 表示采样点到 LiDAR 中心的深度，$D_t(\mathbf{r})$ 是根据 LiDAR 观测计算得出的深度真值。

点云的笛卡尔坐标可通过下式计算：

$$
(x, y, z) = (D_g(\mathbf{r}) \sin \theta \cos \phi, D_g(\mathbf{r}) \sin \theta \sin \phi, D_g(\mathbf{r}) \cos \theta)
$$

总体而言，多模态分词器（multi-modal tokenizer）利用公式 5 中的总损失进行端到端训练：

$$
\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Lidar}} + \mathcal{L}_{\text{rgb}}
$$

### 3.2 Latent BEV Sequence Diffusion

大多数现有的世界模型 Zhang et al. (2024); Hu et al. (2023) 采用自回归策略以获得更长的未来预测，但该方法容易受到累积误差的影响。相反，我们提出了一种潜在序列扩散框架，该框架输入多帧带噪声的 BEV 令牌，并同时获得所有未来的 BEV 令牌。

隐式序列扩散的结构如图1所示。在训练过程中，首先从传感器数据中获得低维的BEV tokens（$x_{t-P}, \cdots, x_{t-1}, x_t, x_{t+1}, \cdots, x_{t+N}$）。此过程仅涉及多模态标记器中的BEV编码器，且多模态标记器的参数保持冻结。为了便于世界模型模块学习BEV token特征，我们沿通道维度对输入的BEV特征进行归一化处理（$x_{t-P}, \cdots, x_{t-1}, x_t, x_{t+1}, \cdots, x_{t+N}$）。最新的历史BEV token与当前帧的BEV token$(x_{t-P}, \dots, x_{t-1}, x_t)$ 作为条件令牌，而 $(x_{t+1}, \dots, x_{t+N})$ 则通过噪声 $\{\varepsilon^i_{\hat{t}}\}_{i=t+1}^{t+N}$ 被扩散为噪声化的 BEV 令牌 $(x_\varepsilon^{t+1}, \dots, x_\varepsilon^{t+N})$，其中 $\hat{t}$ 表示扩散过程的时间戳。

去噪过程由一个包含一系列 Transformer 块的时空 Transformer（spatial-temporal transformer）来执行，其架构如图 4 所示。时空 Transformer 的输入是条件 BEV 令牌和加噪 BEV 令牌 $(\bar{x}_{t-P}, \cdots, \bar{x}_{t-1}, \bar{x}_t, \bar{x}_{t+1}^\epsilon, \cdots, \bar{x}_{t+N}^\epsilon)$ 的拼接。这些令牌通过**车辆运动和转向的动作令牌** $\{a_i\}_{i=T-P}^{T+N}$ 进行调制，它们共同构成了时空 Transformer 的输入。更具体地说，输入令牌首先被传递到时间注意力（temporal attention）块，以增强时间上的平滑性。为了避免时间混淆问题，我们在时间注意力中加入了因果掩码（causal mask）。然后，时间注意力块的输出被发送到空间注意力（spatial attention）块以获取精确的细节。空间注意力块的设计遵循标准 Transformer 块的准则 (Lu et al., 2023a)。动作令牌和扩散时间戳 $\{\hat{t}_i^d\}_{i=T-P}^{T+N}$ 被拼接作为扩散模型的条件 $\{c_i\}_{i=T-P}^{T+N}$，然后被送入 AdaLN (Peebles & Xie, 2023) 中以调制令牌特征。

$$
\mathbf{c} = \text{concat}(\mathbf{a}, \mathbf{\hat{t}}); \quad \gamma, \beta = \text{Linear}(\mathbf{c}); \quad \text{AdaLN}(\mathbf{\hat{x}}, \gamma, \beta) = \text{LayerNorm}(\mathbf{\hat{x}}) \cdot (1 + \gamma) + \beta
$$

其中 $\mathbf{\hat{x}}$ 是一个 Transformer 块的输入特征，$\gamma, \beta$ 是 $\mathbf{c}$ 的缩放和平移参数。

时空 Transformer 的输出是噪声预测 $\{\epsilon_{\hat{t}}^i(\mathbf{x})\}_{i=1}^N$，其损失如下式所示：

$$
\mathcal{L}_{\text{diff}} = \|\epsilon_{\hat{t}}(\mathbf{x}) - \epsilon_{\hat{t}}\|_1
$$

在测试过程中，归一化的历史帧和当前帧 BEV 令牌 $(\bar{x}_{t-P}, \cdots, \bar{x}_{t-1}, \bar{x}_t)$ 与纯噪声令牌 $(\epsilon_{t+1}, \epsilon_{t+2}, \cdots, \epsilon_{t+N})$ 被拼接作为世界模型的输入。从时刻 $T-P$ 到 $T+N$ 的自车运动令牌 $\{a_i\}_{i=T-P}^{T+N}$ 作为条件输入。我们采用 DDIM (Song et al., 2020) 调度来预测后续的 BEV 令牌。随后，对预测的 BEV 令牌应用反归一化操作，然后将它们送入 BEV 解码器和渲染网络，从而生成一组全面的预测多传感器数据。

![image-20260703143340025](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260703143340267.png)



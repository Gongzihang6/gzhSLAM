# Diffusion Policy（扩散策略）超详细解析

## 目录

1. 一句话概括
2. 问题背景：机器人模仿学习为什么难
3. 扩散模型（Diffusion Model）基础知识
4. Diffusion Policy 的核心思想
5. 网络架构与工程实现细节
6. 训练流程与推理流程（含伪代码）
7. 数学推导：从 DDPM 到策略学习
8. 为什么扩散策略有效？——作者的深度分析
9. 实验结果与 Benchmark
10. 关键超参数与调参经验
11. 局限性与批评
12. 后续改进与变体家族（DP3、Consistency Policy、π0、RDT 等）
13. 与其他方法的系统对比
14. 代码实践指南
15. 总结与展望

---

## 1. 一句话概括

**Diffusion Policy 是哥伦比亚大学 TRI（丰田研究院）团队（Cheng Chi、Zhenjia Xu 等）在 2023 年 RSS 上发表、2024 年发表于 IJRR 的机器人模仿学习方法，其核心思想是：不再把策略（policy）看作一个"输入观测、输出动作"的回归函数，而是把动作序列的生成过程建模为一个条件去噪扩散过程（Conditional Denoising Diffusion Process），从而能够表达多模态、高精度的动作分布，在 12 个任务、4 个 Benchmark 上相比此前最优方法平均提升 46.9%。**

原论文：**《Diffusion Policy: Visuomotor Policy Learning via Action Diffusion》**（Chi et al., RSS 2023 / IJRR 2024）。

---

## 2. 问题背景：机器人模仿学习为什么难

### 2.1 模仿学习的基本设定

模仿学习（Imitation Learning）/ 行为克隆（Behavior Cloning, BC）的目标是：给定专家演示数据集 $D = \{(O_i, A_i)\}_{i=1}^{N}$，其中 $O$ 是观测（observation，通常是相机图像、机器人本体状态），$A$ 是动作（action，通常是末端位姿、关节角度），学习一个策略 $\pi$，使得 $\pi(O) \approx A$。

最简单的做法是把这看成一个**监督学习回归问题**：

$$\pi_\theta = \arg\min_\theta \sum_{i} \|\pi_\theta(O_i) - A_i\|^2$$

### 2.2 朴素行为克隆的三大痛点

**痛点一：多模态动作分布（Multimodal Action Distribution）**

人类演示数据天然是多模态的。比如推一个 T 形方块到目标位置，演示者有时从左边推、有时从右边推；抓一个杯子，可以从不同角度抓。如果用 MSE 回归训练，网络会学到这些模式的**平均值**——比如"从中间推"——而这种平均动作在物理上往往是**无效的**（既不左也不右，直接撞上方块）。这就是著名的"均值坍缩"（mode averaging / regression to the mean）问题。

**痛点二：时序上的抖动与不一致**

逐帧独立预测动作（single-step prediction）会导致相邻时间步的动作抖动（jitter），机械臂运动不平滑。人类动作是有时间相关性的，单步回归无法建模这种相关性。

**痛点三：对高频、高精度动作的表达能力不足**

现有改进方案各有缺陷：

| 方法                                                    | 思路                                 | 缺陷                                                         |
| ------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| 离散化动作（tokenization，如 RT-1 早期的离散 bin、BET） | 把连续动作切成离散 token，用分类损失 | 精度损失；高频动作需要极细的 bin，组合爆炸                   |
| GMM / 混合密度网络（LSTM-GMM）                          | 输出高斯混合分布参数                 | 分量数需要预设；训练容易塌缩到单分量；表达力有限             |
| 能量模型 IBC（Implicit BC）                             | 用对比学习（InfoNCE）学能量函数      | 需要大量负样本，训练不稳定，推理时要采样大量候选动作再 argmin，延迟高 |
| GAN 式方法                                              | 生成对抗                             | 训练不稳定，模式塌缩（mode collapse）臭名昭著                |

### 2.3 扩散模型恰好能解决这些痛点

扩散模型在图像生成领域已经证明了：
- 能表达**极其复杂的多模态分布**（ImageNet 级别的多样图像）；
- 训练**稳定**（固定的回归损失，不需要对抗、不需要负采样）；
- 采样质量高。

那么自然的问题就是：**能不能把"动作生成"也当作一个条件扩散生成问题？** 这就是 Diffusion Policy 的出发点。

---

## 3. 扩散模型（Diffusion Model）基础知识

要理解 Diffusion Policy，必须先理解扩散模型本身。

### 3.1 前向过程：逐步加噪

DDPM（Denoising Diffusion Probabilistic Models，Ho et al., 2020）定义一个固定的前向马尔可夫过程：从干净数据 $x^0$ 出发，逐步加入高斯噪声，经过 $K$ 步后变成近似纯噪声 $x^K \sim \mathcal{N}(0, I)$：

$$q(x^k | x^{k-1}) = \mathcal{N}(x^k; \sqrt{1-\beta_k}\, x^{k-1}, \beta_k I)$$

其中 $\beta_k$ 是第 $k$ 步的噪声方差（noise schedule）。令 $\alpha_k = 1 - \beta_k$，$\bar{\alpha}_k = \prod_{s=1}^{k} \alpha_s$，可以得到**任意步的闭式表达**：

$$x^k = \sqrt{\bar{\alpha}_k}\, x^0 + \sqrt{1 - \bar{\alpha}_k}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

这个性质非常重要：训练时我们**不需要真的逐步加噪**，可以一次性采样任意 $k$ 步的带噪样本。

### 3.2 反向过程：学习去噪

生成数据就是前向过程的逆过程：从纯噪声 $x^K$ 出发，逐步去噪回到 $x^0$。由于真实的反向条件分布 $q(x^{k-1}|x^k)$ 不可解，用神经网络 $p_\theta$ 近似：

$$p_\theta(x^{k-1} | x^k) = \mathcal{N}(x^{k-1}; \mu_\theta(x^k, k), \Sigma_k)$$

关键发现是：与其直接预测均值 $\mu$，不如让网络**预测被加入的噪声 $\varepsilon$**。Ho et al. 证明了从 ELBO（证据下界）可以推导出一个极其简单的训练目标：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x^0, \varepsilon, k}\left[ \|\varepsilon - \varepsilon_\theta(x^k, k)\|^2 \right]$$

即：**随机取一条干净数据 $x^0$，随机采一个扩散步 $k$，随机加噪得到 $x^k$，训练网络 $\varepsilon_\theta$ 从 $x^k$ 和 $k$ 中预测出噪声 $\varepsilon$。** 就这么简单。

### 3.3 Score Matching 视角（重要！）

扩散模型与得分匹配（Score Matching, Song & Ermon, 2019）有深刻的联系。预测噪声 $\varepsilon_\theta$ 等价于学习数据分布对数的梯度（score function）：

$$\varepsilon_\theta(x^k, k) \propto -\nabla_{x^k} \log q(x^k)$$

也就是说，去噪网络实际上学的是"**往数据分布高密度区域移动的方向**"。反向采样过程就是在做朗之万动力学（Langevin dynamics）式的梯度上升。这个视角对理解 Diffusion Policy 为什么能表达多模态分布至关重要——后面会详细展开。

### 3.4 DDIM：加速采样

DDPM 的反向过程通常需要几十到几百步，太慢。DDIM（Denoising Diffusion Implicit Models, Song et al., 2020）发现可以构造一族**非马尔可夫**的采样过程，与 DDPM 共享同一个训练好的网络，但可以**跳步采样**（比如只用 10 步完成原本 100 步的去噪），几乎不损失质量。Diffusion Policy 在实际部署中就大量使用 DDIM 来降低推理延迟。

---

## 4. Diffusion Policy 的核心思想

### 4.1 从"图像生成"到"动作生成"

图像扩散模型学的是 $p(x)$（无条件）或 $p(x | \text{text})$（文本条件）。Diffusion Policy 做了一个直接的类比替换：

$$\text{生成对象：图像 } x \quad \Longrightarrow \quad \text{生成对象：动作序列 } A_t$$
$$\text{条件：文本提示} \quad \Longrightarrow \quad \text{条件：观测 } O_t$$

即学习条件分布：

$$p_\theta(A_t | O_t)$$

其中：
- $A_t = (a_t, a_{t+1}, \dots, a_{t+T_p-1})$：从当前时刻开始的**一段未来动作序列**（长度为预测时域 $T_p$，prediction horizon）；
- $O_t = (o_{t-T_o+1}, \dots, o_t)$：过去 $T_o$ 步的观测历史（observation horizon，通常 $T_o = 2$，即当前帧+上一帧）。

### 4.2 为什么是"动作序列"而不是单个动作？——Action Chunking（动作分块）

这是 Diffusion Policy 的关键设计决策之一。生成一整段动作序列（比如 16 步）而不是单个动作，有三大好处：

1. **时序一致性**：序列内的动作由同一个扩散过程联合生成，天然平滑、连贯，消除抖动；
2. **更好的多模态处理**：单步动作的多模态往往"看起来不像多模态"（因为历史不同导致分歧），整段序列的多模态模式更清晰，扩散模型更容易学；
3. **对停顿/怠速动作的鲁棒性**：演示中常有停顿（idle）片段，单步预测无法区分"停顿后向左"和"停顿后向右"，序列预测可以看到模式的整体走向。

### 4.3 Receding Horizon：滚动时域执行

虽然一次预测 $T_p$ 步动作（比如 16 步），但并不全部执行完，而是只执行前 $T_a$ 步（action horizon，比如 8 步），然后**重新观测环境、重新扩散采样、重新预测**。这就是滚动时域控制（receding horizon control，类似 MPC 的思想）：

```
观测 O_t → 扩散去噪生成 A_t (16步) → 执行前 8 步 → 重新观测 O_{t+8} → 循环
```

好处：
- **闭环纠错**：每一步都基于最新观测，能应对环境扰动、物体滑动等意外；
- **兼顾平滑与反应性**：$T_p$ 大保证平滑与预见性，$T_a < T_p$ 保证反应速度。

### 4.4 完整的扩散策略公式

把上述要素组合起来，Diffusion Policy 学习的反向去噪过程为：

$$p_\theta(A_t^{0:K} | O_t) = p(A_t^K) \prod_{k=K}^{1} p_\theta(A_t^{k-1} | A_t^k, O_t, k)$$

训练目标（预测噪声版本）：

$$\mathcal{L} = \mathbb{E}_{(O_t, A_t^0) \sim D,\; k,\; \varepsilon}\left[ \left\| \varepsilon - \varepsilon_\theta\big(\sqrt{\bar{\alpha}_k} A_t^0 + \sqrt{1-\bar{\alpha}_k}\,\varepsilon,\; O_t,\; k\big) \right\|^2 \right]$$

推理时：
$$A_t^K \sim \mathcal{N}(0, I), \qquad A_t^{k-1} = \frac{1}{\sqrt{\alpha_k}}\Big(A_t^k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}}\,\varepsilon_\theta(A_t^k, O_t, k)\Big) + \sigma_k z$$

（DDPM 采样器形式；实际部署常用 DDIM 去掉随机项并跳步。）

**一个绝妙的直觉解读**：整个去噪过程可以看作"动作轨迹的逐步精化"——从一团随机噪声动作开始，网络反复"擦拭"它，每一步都让轨迹更像专家会做的动作，同时被观测 $O_t$ 条件约束着走向当前情境下正确的模式。

---

## 5. 网络架构与工程实现细节

Diffusion Policy 论文给出了**两种骨干网络**，都非常实用：

### 5.1 CNN-based Diffusion Policy（主推，真机首选）

采用**一维时间卷积 U-Net（1D Temporal U-Net）**，与图像 U-Net 同构，但卷积是在"时间维 × 动作维"上做的：

- 输入：带噪动作序列 $A_t^k$，形状 $[T_p \times d_{\text{action}}]$（如 16×10）；
- 结构：多层 1D 卷积下采样 → 中间层 → 上采样 + 跳跃连接（skip connection），输出与输入同形状的噪声预测；
- **条件注入方式：FiLM（Feature-wise Linear Modulation）**。观测特征 $O_t$ 经编码后，为 U-Net 每一层的特征图生成逐通道的缩放 $\gamma$ 和偏移 $\beta$：$\text{FiLM}(h) = \gamma(O_t) \odot h + \beta(O_t)$。扩散步 $k$ 的 embedding（正弦位置编码 + MLP）也通过 FiLM 注入；
- 特点：**对超参数极其鲁棒**（论文强调这点），但有个小坑：卷积的局部性导致远处动作对当前条件的感知弱，作者发现必须固定第一帧动作或用较长 $T_o$ 补偿（实际上他们用 $T_o=2$ 配合 FiLM 已经够好）。

**为什么不用普通 MLP？** 因为 U-Net 的多尺度结构天然适合"轨迹精化"：深层捕捉全局模式（往哪走），浅层捕捉局部细节（怎么动）。

### 5.2 Transformer-based Diffusion Policy

- 动作序列作为 token，加位置编码，过若干层 Transformer block；
- 观测条件通过 **cross-attention** 或作为额外 token 拼接（in-context conditioning）注入；
- 扩散步 $k$ 用 AdaLN（Adaptive LayerNorm，DiT 风格）或 FiLM 注入；
- 优点：表达力更强、对复杂条件（多视角图像、语言指令）扩展性好；缺点：**对超参数和学习率更敏感**，需要仔细调参（论文报告了一定的训练不稳定）。

### 5.3 视觉编码器

- 默认 **ResNet-18**（不用预训练权重，从头训练），替换最后全局池化为 **soft spatial argmax** 或保持 spatial feature map；
- 关键工程细节：
  - **每个相机视角独立编码**，特征拼接后作为条件；
  - **GroupNorm 代替 BatchNorm**（避免评估时 batch 统计不一致导致控制抖动）；
  - 观测图像随机裁剪/颜色抖动做数据增强（防止过拟合背景）；
- 低维观测（如关节角度）则直接用 MLP 编码。

### 5.4 动作空间与归一化

- 动作通常是 **末端执行器位姿**（位置 xyz + 旋转 6D 表示 + 夹爪开合），或关节位置；
- 强烈建议将每个维度**归一化到 [-1, 1]**（用数据集 min-max），这对扩散模型稳定性影响很大；
- 旋转推荐用 **6D 连续旋转表示**（前两个列向量），避免四元数/欧拉角的奇异性干扰扩散过程。

### 5.5 EMA（指数滑动平均）

维护一份网络参数的滑动平均副本 $\theta_{\text{ema}} \leftarrow \lambda \theta_{\text{ema}} + (1-\lambda)\theta$（$\lambda$ 如 0.9999），**评估和部署永远用 EMA 权重**。这是从图像扩散模型继承的关键技巧，能让采样轨迹平滑、稳定，论文 ablation 显示去掉 EMA 成功率显著下降。

---

## 6. 训练流程与推理流程

### 6.1 训练伪代码

```python
# 每个训练 step
for (O, A) in dataloader:                    # O: 观测历史 [To, ...], A: 动作序列 [Tp, Da]
    A = normalize(A)                          # 归一化到 [-1, 1]
    k = randint(1, K)                         # 随机扩散步，K 通常 100
    eps = randn_like(A)                       # 采样噪声
    A_noisy = sqrt(alpha_bar[k]) * A + sqrt(1 - alpha_bar[k]) * eps
    eps_pred = denoise_net(A_noisy, k, cond=encode(O))   # FiLM/cross-attn 条件注入
    loss = mse(eps_pred, eps)                 # 噪声预测损失
    loss.backward(); optimizer.step()
    ema_update(denoise_net)                   # EMA 更新
```

### 6.2 推理（部署）伪代码

```python
# receding horizon 控制循环
while not done:
    O = get_observations(last To steps)       # 收集观测历史
    A = randn(Tp, Da)                          # 从纯噪声开始
    for k in scheduler_steps:                 # DDIM 跳步，如 10 步
        eps = denoise_net_ema(A, k, cond=encode(O))
        A = ddim_update(A, eps, k)            # 去噪一步
    A = denormalize(A)
    execute(A[:Ta])                            # 只执行前 Ta 步（如 8 步）
```

### 6.3 延迟分析

一次决策需要 $K_{\text{DDIM}}$ 次前向网络推理。以 $K_{\text{DDIM}}=10$、U-Net 单次约 5ms 计算，约 50ms 得到 16 步动作、执行 8 步——对 10Hz 控制完全够用。论文还指出：扩散去噪可以和上一步动作执行**并行**（提前开始算），进一步掩盖延迟。

---

## 7. 数学推导：从 DDPM 到策略学习（深入）

### 7.1 为什么是"策略"？

传统 BC：$\pi_\theta(O) = \arg\min_A \mathcal{L}(A, A^*)$，输出是**点估计**。

Diffusion Policy：学的是整个条件分布 $p_\theta(A|O)$，采样过程即"从分布中抽取一条动作序列"。这使得同一个观测 $O$ 可以采样出不同模式的动作（左推/右推），完整保留了数据的多样性。

### 7.2 与能量模型（EBM）的联系——作者的核心理论贡献之一

IBC（Implicit Behavioral Cloning）用对比学习学能量函数 $E_\theta(O, A)$，动作分布为 $p(A|O) \propto e^{-E_\theta(O,A)}$，推理时需采样大量候选动作选能量最低的。

Diffusion Policy 可以被解读为：**不显式学能量函数，而是直接学能量函数的梯度（score）**：

$$\varepsilon_\theta(A^k, O, k) \approx -\sigma_k \nabla_{A^k} \log p(A^k | O)$$

反向扩散就是在不同噪声尺度下做**退火朗之万采样**（annealed Langevin sampling）。好处：

1. **无需负样本**：IBC 的 InfoNCE 对负样本数量/质量极敏感；score matching 只回归噪声，完全绕开；
2. **无需归一化配分函数**：score 是梯度的商，配分函数自动消掉；
3. **多尺度细化**：噪声从大到小，相当于从粗到精地搜索动作空间——先决定"去哪个模式"，再精修"具体怎么动"。这解释了扩散策略在多模态任务上的碾压级表现。

### 7.3 损失函数性质

噪声预测损失是一个普通的 MSE，这意味着：
- 训练目标**光滑、无对抗项、无采样方差**；
- 梯度方差小，可以用大 batch、高学习率；
- 与图像扩散共享所有成熟工程经验（noise schedule、EMA、DDIM 等）。

---

## 8. 实验结果

### 8.1 设置

- **12 个任务、4 个 Benchmark**：Robomimic（Can/Lift/Square/Transport）、Push-T（仿真+真机）、Franka Kitchen、Block Pushing、Mug Flipping（真机）、Tool Hang、Bimanual 系列等；
- **对比基线**：LSTM-GMM、IBC、BET、BC-RNN、BCCo 等当时最强的模仿学习方法。

### 8.2 核心结论

- **平均成功率提升 46.9%**（相对此前最优基线）——这在机器人学习领域是极其罕见的提升幅度；
- 在**多模态性强**的任务（Push-T、Block Pushing）上优势最大，验证了多模态表达能力的理论分析；
- **真机实验**：Push-T 真机、Mug Flipping 等任务上表现出很强的鲁棒性，能处理物体滑动、初始位姿变化；
- Ablation 关键发现：
  - 预测时域 $T_p$：太短（≤2）失去时序一致性，太长浪费算力且反应慢，**$T_p \approx 16$ 最优**；
  - DDIM 步数：**10 步左右**即可，再少质量下降；
  - EMA：**必须开**；
  - $T_o = 2$ 优于 1（提供速度信息），再长收益递减；
  - CNN 版比 Transformer 版更稳定易调，Transformer 版上限略高但更挑超参。

---

## 9. 关键超参数速查表（实践经验）

| 超参数           | 推荐值                                           | 说明                        |
| ---------------- | ------------------------------------------------ | --------------------------- |
| 训练扩散步数 $K$ | 100                                              | DDPM 标准                   |
| 推理 DDIM 步数   | 10（真机）/ 16                                   | 越少越快，质量略降          |
| 噪声 schedule    | squaredcos（iDDPM）或 linear                     | squaredcos 在低维动作上更好 |
| 预测时域 $T_p$   | 16                                               | 平衡平滑与反应              |
| 执行时域 $T_a$   | 8                                                | 通常为 $T_p/2$              |
| 观测时域 $T_o$   | 2                                                | 提供历史/速度信息           |
| EMA 衰减         | 0.999 ~ 0.9999                                   | 必开                        |
| 学习率           | 1e-4（AdamW）                                    | cosine 衰减，warmup 500 步  |
| Batch size       | 64~256                                           | 越大越稳                    |
| 动作归一化       | min-max 到 [-1, 1]                               | 必做                        |
| 网络             | 1D U-Net（256/512/1024 通道）或 8 层 Transformer | 首选 U-Net                  |
| 视觉编码器       | ResNet-18 + GroupNorm，从头训练                  | 每相机独立                  |

---

## 10. 局限性与批评

1. **推理延迟**：10~100 次网络前向才能出一个动作块，高频控制（>50Hz）场景吃力。这是扩散策略被诟病最多的一点；
2. **动作块边界处的不连续**：虽然块内平滑，但相邻两个采样块之间可能跳变（尤其是两次采样落到不同模式时），需要 warm-start 或时序集成缓解；
3. **对演示数据质量敏感**：和一切 BC 方法一样，协变量偏移（covariate shift）问题依旧存在，演示之外的分布会失效；
4. **无法处理语言指令**（原版）：条件是观测，没有语言；后续工作才补上；
5. **理论保证缺失**：为什么 10 步 DDIM 就够、为什么 $T_p=16$ 最优，主要靠经验；
6. **采样随机性的双刃剑**：多样性在评测时可能导致偶发的模式选择错误（比如选错抓取侧）。

---

## 11. 后续改进与变体家族（2023–2025 的扩散策略宇宙）

Diffusion Policy 发表后迅速成为机器人学习的"默认范式"，衍生出庞大的改进家族：

### 11.1 加速推理
- **Consistency Policy**（Prasad et al., 2024）：用一致性蒸馏把多步去噪压缩到 **1 步采样**，速度提升一个数量级，性能略降但可用；
- **Streaming Diffusion Policy (SDPP)**：把去噪过程做成流式、随时间滚动精化，消除块边界问题；
- **iDP3 / DP 改进版**中的推理优化、以及用 flow matching rectified flow 减少采样步数。

### 11.2 3D 扩展
- **3D Diffusion Policy (DP3)**（Ze et al., RSS 2024）：用**稀疏点云**代替图像作为观测条件，只用 10 条演示就能学会多种任务，真机表现极强，证明了 3D 表征 + 扩散动作头的威力。

### 11.3 与大规模预训练/VLA 结合（最重要的方向）
- **RDT-1B（Robotics Diffusion Transformer）**：双臂机器人基础模型，DiT 架构 + 扩散动作头，10 亿参数，语言条件；
- **π0（pi-zero，Physical Intelligence, 2024）**：VLM（PaliGemma）+ flow matching 动作专家（action expert），flow matching 是扩散的连续化推广，50Hz 高频控制双臂，折叠衣服等精细任务，是目前影响力最大的扩散式 VLA；
- **Octo、OpenVLA-OFT** 等也提供 diffusion/flow matching 动作头选项；
- **DexVLA、CogACT、TinyVLA**：扩散/流匹配头被广泛用于提升 VLA 的动作精度。

### 11.4 灵巧操作与其他
- **Diffusion-EDF**：SE(3) 等变扩散策略，样本效率极高；
- **Crossway Diffusion**：引入中间状态条件；
- 用于灵巧手（dexterous hand）、触觉条件扩散策略、Mobile ALOHA 风格双臂任务的动作头等。

### 11.5 Flow Matching / Rectified Flow 分支
π0、GR00T N1 等使用的 flow matching 可以看作扩散模型的简化与推广：直接学习从噪声到数据的**直线流**的速度场 $v_\theta(x_t, t)$，ODE 采样步数更少、更稳定。可以说"扩散策略"正在演化为更广义的"**生成式动作头**（generative action head）"范式。

---

## 12. 与其他方法的系统对比

| 维度          | BC 回归    | LSTM-GMM   | IBC（能量模型）  | ACT（CVAE）     | **Diffusion Policy**                                         |
| ------------- | ---------- | ---------- | ---------------- | --------------- | ------------------------------------------------------------ |
| 多模态表达    | ❌ 均值坍缩 | ⚠️ 有限分量 | ✅                | ⚠️ 依赖潜变量    | ✅✅ 最强                                                      |
| 训练稳定性    | ✅          | ⚠️ 易塌缩   | ❌ 挑负样本       | ✅               | ✅                                                            |
| 推理速度      | ✅ 最快     | ✅          | ❌ 需大量候选采样 | ✅ 单次前向      | ⚠️ 需多步去噪（可 DDIM/蒸馏）                                 |
| 高频动作精度  | ✅          | ⚠️          | ⚠️                | ⚠️ 有时抖动      | ✅                                                            |
| 时序一致性    | ❌ 单步     | ⚠️          | ❌                | ✅ chunk         | ✅ chunk + receding horizon                                   |
| 与 ACT 的关系 | —          | —          | —                | CVAE 生成动作块 | 扩散通常被认为表达力强于 CVAE，且 π0 等已用 flow matching 取代 ACT 头 |

值得一提：**ACT（Action Chunking with Transformers，ALOHA 系列）** 与 Diffusion Policy 几乎同时期出现，思想都是"生成动作块"，区别在生成器用 CVAE 还是扩散。社区共识是：扩散在表达力和精度上更强，ACT 在推理速度上占优；而 2024 年后的趋势是用 flow matching/扩散头统一两者优点。

---

## 13. 代码实践指南

### 13.1 官方资源
- 官方仓库：`real-stanford/diffusion_policy`（PyTorch），包含 Push-T、Robomimic 全部训练脚本和预训练权重；
- 数据格式：zarr 存储的 episode 序列（obs + action），与 Robomimic 兼容；
- LeRobot（HuggingFace）也内置了 Diffusion Policy 实现，API 更现代，适合快速上手。

### 13.2 上手建议（踩坑清单）
1. **先跑通低维观测版本**（state-based Push-T），确认管线正确再接图像；
2. 动作**必须归一化**，旋转用 6D 表示；
3. **EMA 必开**，推理用 EMA 权重；
4. 视觉输入用 GroupNorm、开数据增强；
5. 推理用 DDIM 10 步起步，再按延迟预算调整；
6. 真机部署注意：推理与执行并行化；若两次采样块之间抖动，可对重叠区域做时序加权平均（temporal ensemble，ALOHA 的技巧）；
7. 演示数据量参考：Push-T 用 200 条；DP3 用点云时 10 条即可；任务越难数据需求越大；
8. 如果任务需要语言条件，直接上 RDT/π0 风格的 DiT + cross-attention，或给原版 U-Net 的 FiLM 条件里加入语言 embedding。

### 13.3 最小实现核心代码（PyTorch 伪代码）

```python
class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim, cond_dim, ...):
        # down_blocks / mid_block / up_blocks，每层 FiLM(cond)
        ...

    def forward(self, a_noisy, k, obs_cond):
        t_emb = sinusoidal_emb(k)
        cond = torch.cat([obs_cond, t_emb], -1)
        return self.unet(a_noisy, cond)   # 输出预测的 eps

# 训练
loss = F.mse_loss(net(add_noise(A, k, eps), k, obs_feat), eps)

# 推理（DDIM）
A = torch.randn(B, Tp, Da)
for k in ddim_timesteps:
    eps = net_ema(A, k, obs_feat)
    A = ddim_step(A, eps, k)
```

---

## 14. 总结与展望

**Diffusion Policy 的本质贡献**可以浓缩为三点：

1. **范式转换**：把机器人策略从"确定性回归"重新表述为"条件生成建模"，用扩散模型学习 $p(A|O)$，第一次让模仿学习策略能原生、稳定地表达多模态动作分布；
2. **工程配方**：动作分块（action chunking）+ 滚动时域（receding horizon）+ FiLM 条件 U-Net + DDIM 加速 + EMA，这套配方简单、鲁棒、即插即用，成为社区事实标准；
3. **性能飞跃**：平均 46.9% 的提升奠定了它在 2023 年后机器人学习中的统治地位，并通过 DP3、RDT、π0 等后裔，把"扩散/流匹配动作头"写入了当代 VLA（视觉-语言-动作）大模型的基因。

**展望**：扩散策略的两个前沿方向是——(a) 采样效率：1~4 步的蒸馏/一致性/rectified flow 方法正在抹平推理延迟的最后差距；(b) 规模化：作为大模型动作头，与预训练 VLM 结合（π0、GR00T、RDT 路线），走向通用机器人策略。可以预见，"生成式策略"（generative policy）将在相当长的时间内是机器人操作学习的主流范式，而 Diffusion Policy 正是这一切的起点。

---

如果你想深入某个具体方向（比如完整数学推导、DP3 的点云细节、π0 的 flow matching、或者动手复现 Push-T），告诉我，我可以继续展开。
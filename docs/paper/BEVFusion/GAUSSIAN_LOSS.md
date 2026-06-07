\*\*GaussianLoss 详细说明

文件位置：eggy/models/map_pretrain/utils/gaussian_loss.py

概述
----
`GaussianLoss` 是用于 Map-Pretrain / BEV 重建类任务的一套复合损失函数实现，融合了图像像素级重建损失、感知损失（LPIPS）、深度监督损失以及特征层的一致性损失。它同时使用像素级的有效掩码（由 GT mask、BEV->Image coverage 与可选的 coarse road mask 相交得到）来只在有效像素上计算损失，从而避免对 sky、车厢外或未被 BEV 覆盖的像素计算误导梯度。

输入与前置处理
----------------
- preds: 来自模型渲染/重建模块的字典，期望包含的项：
  - `rendered_images`: [B\*N, 3, H, W]，模型渲染/重建出的图像，范围为 0..1。
  - `rendered_feats`: [B\*N, C, H_f, W_f]，模型渲染得到的特征图（用于与 GT 特征比较）。
  - `rendered_depths` （可选）: [B\*N, 1, H, W]，模型渲染出的深度图。
  - `coverage_mask`（可选）: [B\*N, 1, H, W]，从 BEV 点投影得到的每帧每视角像素覆盖掩码。
  - `coarse_road_mask`（可选）: [B\*N, 1, H, W]，粗道路掩码（若提供，用于进一步限制损失区域）。

- gt_imgs: [B\*N, 3, H, W]，训练集中的目标真实图（通常按 ImageNet 均值/方差做过标准化）。
- gt_feats: [B\*N, C, H_f, W_f]，从真实图像或预计算网络（如 backbone）得到的目标特征。
- gt_depth: [B\*N, H, W]（可选），真实深度图（若有）。
- gt_masks: [B\*N, 1, H, W] 或 [B\*N, H, W]，语义/道路/有效像素掩码，用于指定哪些像素参与损失计算。

注意：代码会把 `gt_masks` 维度标准化为 `[B\*N,1,H,W]`。

掩码合成逻辑
------------
1. `coverage_mask` 与 `coarse_road_mask`（若存在）通过 `_ensure_mask_shape` 被缩放 / 扩维 / 插值到与 `gt_masks` 相同形状。
2. 最终生效掩码 `final_mask = gt_masks \* coverage_mask \* coarse_road_mask`（按需相交）。
3. `final_mask` 被裁剪到 0..1 浮点范围并用于后续各项损失的加权。

去归一化（Ground-truth 图像）
-----------------
函数 `_denormalize_gt_imgs` 的实现：
- 假设 `gt_imgs` 已经用 ImageNet 的 mean/std 标准化（即 (img - mean)/std），该函数执行：
  gt_imgs_denorm = gt_imgs \* img_std + img_mean   # 回到 0..255 范围
  gt_imgs_01 = gt_imgs_denorm / 255.0              # 归一化到 0..1
- 结果 `gt_imgs_01` 用来与 `rendered_images`（0..1）做像素与 LPIPS 计算。

各项损失与公式
----------------
以下符号与维度说明：
- R(x): 模型渲染结果（rendered_images），R ∈ ℝ^{B\*N×3×H×W}
- G(x): 去归一化后的 ground-truth 图像（denorm_gt_imgs），G ∈ ℝ^{B\*N×3×H×W}
- M(x): 生效掩码（final_mask），M ∈ {0,1}^{B\*N×1×H×W}
- LpipsMap(x): LPIPS 的逐像素映射（当 spatial=True）输出，形状 [B\*N,1,H,W]
- D_r(x): 渲染深度（rendered_depths），形状 [B\*N,1,H,W]
- D_gt(x): ground-truth depth（gt_depth），形状 [B\*N,H,W]
- F_r: 渲染特征（rendered_feats），F_r ∈ ℝ^{B\*N×C×H_f×W_f}
- F_gt: ground-truth 特征（gt_feats），F_gt ∈ ℝ^{B\*N×C×H_f×W_f}

1) 图像像素重建损失（L1）：
公式： L_img = rgb_weight \* mean( |R - G| \* M )
实现细节：
- 取逐元素绝对差后与掩码相乘，再对所有元素做算术平均（代码中使用 `.mean()`）。
- 注意：如果掩码中 0 的比例很高，直接 `.mean()` 会把 0 也计入分母，因此在某些场景下会考虑按权重求平均；当前实现直接取 `.mean()`。

2) 感知损失（LPIPS）：
公式： LPIPS_loss = lpips_weight \* ( sum( LpipsMap \* M ) / sum(M) )
实现细节：
- `self.lpips_vgg` 使用 `lpips.LPIPS(net=lpips_net, spatial=lpips_spatial)`。
  - 当 `spatial=True` 时，LPIPS 返回一个空间化的相似度图（[B,1,H,W]），否则返回单个标量。
- 代码中调用 `lpips_map = self.lpips_vgg(render_imgs, denorm_gt_imgs, normalize=True)`。
- 使用 `_weighted_mean(lpips_map, gt_masks)` 计算加权平均：该函数做 sum(value\*weight)/sum(weight)，因此真正按掩码有效像素归一化。

说明：LPIPS 提供的度量通常在感知相似度上比像素 L1 更鲁棒（更符合人眼感知），因此常用于重建任务中作为辅助感知损失。

3) 深度监督（Depth L1）：
公式： depth_loss = depth_weight \* ( sum( |D_r - D_gt| \* valid_mask \* M ) / sum(valid_mask \* M) )
实现细节：
- 首先判断 `gt_depth` 是否存在且 `rendered_depths` 在 preds 中存在。
- 构建 `gt_depth_valid = (gt_depth > 1e-3) & (gt_depth < 240.0)`，只在合理深度范围内视为有效。
- 若渲染深度与 gt 深度分辨率不一致，先用 `F.interpolate(..., mode='nearest')` 缩放到一致分辨率。
- 差值计算为绝对差 `|render_depth - gt_depth_resized|`，并用 `valid_mask = gt_depth_valid_resized.float() \* gt_masks.squeeze(1)` 作为权重（确保同时满足深度有效与像素掩码约束）。
- 使用 `_weighted_mean(diff, valid_mask)` 得到加权平均并乘以 `depth_weight`。

4) 特征一致性损失（Feature cosine loss）：
公式： feature_loss = feat_weight \* mean( (1 - cosine_similarity(F_r, F_gt)) \* M_f )
实现细节：
- 先将 `gt_masks` 插值（双线性）到 `rendered_feats` 的空间尺寸 `H_f×W_f`，得到 `feat_mask`。
- 使用 `torch.nn.functional.cosine_similarity(rendered_feats, gt_feats, dim=1)` 计算通道维度上的余弦相似度，结果形状为 [B\*N, H_f, W_f]。
- 取 `1 - cos_sim` 作为距离度量，乘以 `feat_mask` 后对全部元素取 `.mean()`，再乘以 `feat_weight`。

最终损失
---------
代码将四项损失相加并返回：

Loss_total = L_img + LPIPS_loss + feature_loss + depth_loss

同时返回分项字典：
- `loss`: 总和
- `img_loss`: L_img
- `lpips_loss`: LPIPS_loss
- `feat_loss`: feature_loss
- `depth_loss`: depth_loss

实现与数值稳定性细节
-----------------
- LPIPS 模型在成员变量中被设置为 `requires_grad_(False)`，避免更新 LPIPS 权重。
- 对 depth 的有效性检查用阈值 1e-3 与 240.0，防止无穷大或标注错误值影响训练。
- 对可能的尺寸不匹配（image / depth / features）使用 `F.interpolate` 做调整。
- `_weighted_mean` 使用 `weight_map.sum().clamp_min(1e-6)` 避免除以 0 情况。

与 mask 的交互（为什么要用 coverage & coarse mask）
-------------------------------------------------
- `coverage_mask`：由 BEV -> image 投影产生，表示 BEV 网格采样点在哪些像素上有对应投影；若某像素未被任何 BEV 点投影到它上面，那么不应对其计算 BEV 相关的重建损失（避免对模型惩罚那些 BEV 没有覆盖的图像区域）。
- `coarse_road_mask`：用于进一步只在道路类或大致目标区域内计算损失（减少对不可见或非道路区域的惩罚）。
- 合并后的 `final_mask` 确保损失只在那些同时满足（GT 有意义）且（BEV 覆盖）且（可选道路掩码）三者条件的像素上计算。

为何同时使用 L1 + LPIPS + Feature + Depth？
------------------------------------------
- L1（像素级）能约束低频与整体亮度、颜色关系，但对感知质量（纹理、结构）不足。
- LPIPS 捕捉高阶感知差异，更贴合人眼感知，对纹理/结构比 L1 更敏感。
- Feature cosine loss 约束渲染特征与真实图像特征在语义/语境层次的相似性（对检测/语义任务有更直接的辅助效果）。
- 深度监督显式约束几何一致性（尤其在 BEV 任务中，深度准确性直接影响 BEV 投影质量）。

使用建议
--------
- 在数据稀疏（覆盖非常少）场景，LPIPS 与像素 L1 在有效像素上权重的选取需谨慎，避免噪声主导。使用 `coverage_mask` 是关键。
- 若 gt_feats 是从一个固定 backbone（例如解码器或预训练网络）提取的，确保训练时该 backbone 与渲染特征的尺度/归一化一致。

参考/实现细节位置
-----------------
- 主要实现文件：`eggy/models/map_pretrain/utils/gaussian_loss.py`
- 掩码生成：见 `eggy/models/map_pretrain/road_bev_pretrain.py` 中 `coverage_mask` 的实现（BEV->Image 投影与膨胀）

结束语
-----
本说明尽可能详细地覆盖了 `GaussianLoss` 的构成、输入预处理、掩码逻辑和各项数学表达式。如果你希望我把该文档转换为带公式的 LaTeX 版、添加示例数值或在 README 中引用此文件，请告诉我。

LaTeX 公式
------------
下文使用 $L_{img}, L_{lpips}, L_{feat}, L_{depth}$ 表示各项损失，则总损失为：
$$
L = L_{img} + L_{lpips} + L_{feat} + L_{depth}
$$

每项定义：
$$
L_{img} = \alpha \cdot \frac{1}{|\Omega|} \sum_{p \in \Omega} \lvert R(p) - G(p) \rvert
$$
其中 $\Omega$ 表示生效掩码（GT mask 与 coverage mask 的交集），$\alpha$ 为 `rgb_weight`。

$$
L_{lpips} = \beta \cdot \frac{\sum_{p \in \Omega} LPIPS(R(p), G(p))}{\sum_{p \in \Omega} 1}
$$
其中 $\beta$ 为 `lpips_weight`。

$$
L_{depth} = \gamma \cdot \frac{\sum_{p \in \Omega_d} \lvert D_r(p) - D_{gt}(p) \rvert}{\sum_{p \in \Omega_d} 1}
$$
其中 $\Omega_d$ 为同时满足深度有效性与像素掩码的像素集合，$\gamma$ 为 `depth_weight`。

特征相似性损失：
$$
L_{feat} = \delta \cdot \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W \bigl(1 - \cos( F_r^{ij}, F_{gt}^{ij})\bigr) \cdot M^{ij}
$$
其中 $\delta$ 为 `feat_weight`，$M^{ij}$ 是插值到特征分辨率后的掩码。

示意图
--------
已生成示意图：`docs/gaussian_loss_flow.png`，展示输入、预测与各项损失的流程关系。
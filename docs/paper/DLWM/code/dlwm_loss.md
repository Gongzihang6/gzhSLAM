# dlwm_loss

这段代码的核心任务是将来自自动驾驶车端的 2D 图像语义、稀疏 LiDAR 点云，以及预测的稠密深度，与 3D 高斯基元进行物理与语义上的对齐与约束。

整个损失函数 $L_{total}$ 由 5 个子项组成，各自负责压制不同类型的网络作弊行为：

1、$L_d$（稀疏深度 L1 loss），控制深度尺度。这个是深度的绝对真值，利用 LiDAR 投影到图像上得到的极其精准但是稀疏的深度点，锚定整个场景的绝对物理尺度；

2、$L_{pd}$（稠密伪深度 loss），控制场景相对结构。由于 LiDAR 点云太稀疏，网络难以学到完整的物体表面（如一整面墙），因此引入 Metric3D 生成的稠密伪深度图，为了消除伪深度的尺度误差，采用了尺度不变（Scale-Invariant）损失；

3、$L_{ent}$（透明度熵 loss），3D 高斯专属正则化，逼迫高斯球变得纯粹，要么完全不透明变成固体，要么完全透明消失，消除场景中的雾霾和漂浮物；

4、$L_{sem}$（语义交叉熵 loss），理解场景语义结构，不仅要重建几何，还要直到哪个高斯球是车，哪个是路；

5、$L_{scale}$（尺度惩罚 loss），3D 高斯专属正则化，防止高斯球为了弥补多视角下的深度误差，而变成极度拉长的针或面条形状；

```python
"""
DLWMLoss: DLWM 一阶段范式渲染监督损失函数。

总损失公式：
    L_rec = w_d * L_d + w_pd * L_pd + w_sem * L_sem

- L_d  (稀疏深度损失):   L1 Loss，仅在 valid_lidar_mask 有效像素上计算，权重 1.0
- L_pd (稠密伪深度损失):  L1 Loss，全像素计算，权重 0.05
- L_sem (语义交叉熵损失): Cross-Entropy Loss，权重 1.0，ignore_index=empty_label

参考：DLWM 一阶段范式 Loss 设计
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class DLWMLoss(nn.Module):
    """DLWM 总损失：L_rec = w_d * L_d + w_pd * L_pd + w_sem * L_sem。

    动态类别过滤机制：
      - train_classes=None  → 所有像素均参与（深度有效 & 非背景）
      - train_classes=[1,2] → 仅 semantic_gt ∈ {1, 2} 的像素参与 depth/sem loss

    Args:
        weight_sparse_depth: L_d 权重，默认 1.0
        weight_dense_depth:  L_pd 权重，默认 0.05
        weight_semantic:     L_sem 权重，默认 1.0
        add_distance:        是否启用距离分段加权；False 时使用原损失策略
        num_classes:         总类别数（含背景 0），用于 CE Loss ignore_index
        train_classes:       参与训练的类别 id 列表；None 表示全部非背景类别
        depth_ignore_value:  深度图中代表"无效"的值（通常为 0.0）
        distance_bin_edges:  距离分段边界（米），如 [0, 20, 40, 60, 80]
        distance_bin_weights:与 distance_bin_edges 对应的各段权重，如 [1.0, 1.2, 1.5, 2.0]
    """

    def __init__(
        self,
        weight_sparse_depth: float = 1.0,
        weight_dense_depth:  float = 0.05,
        weight_semantic:     float = 1.0,
        weight_entropy:      float = 0.01,  # 透明度熵权重
        weight_scale:        float = 0.1,   # 尺度惩罚权重
        max_scale_thresh:    float = 3.0,   # 允许的最大半轴长度 (米)
        weight_semantic_sparse: float = 0.5,
        add_distance: bool = False,
        num_classes:         int   = 16,
        train_classes:       Optional[List[int]] = None,	# 只对有监督的语义类别进行损失计算
        depth_ignore_value:  float = 0.0,
        # 距离分段加权，允许对不同距离的深度误差赋予不同权重，比如给 0-20m 的近处赋予更高权重，保障自车周围的安全距离感知
        distance_bin_edges: Optional[List[float]] = [0.0, 20.0, 40.0, 60.0, 80.0],
        distance_bin_weights: Optional[List[float]] = [1.0, 1.0, 1.0, 1.0],
        camera_weights: Optional[List[float]] = None,	# 不同视角相机赋予不同权重
    ) -> None:
        super().__init__()
        self.w_sparse  = weight_sparse_depth	# 激光雷达稀疏点深度损失权重
        self.w_dense   = weight_dense_depth		# Metric3D稠密伪深度损失权重
        self.w_sem     = weight_semantic	# SMA3的伪语义图交叉熵损失权重
        self.w_ent     = weight_entropy	   # 不透明度交叉熵损失权重
        self.w_scale   = weight_scale		# 高斯球scale损失权重
        self.max_scale = max_scale_thresh
        self.w_sem_sparse = weight_semantic_sparse  # 稀疏雷达稀疏点语义损失权重（可选项）
        self.add_distance = add_distance	# 是否启用深度误差分段加权
        self.num_classes = num_classes
        self.train_classes = train_classes
        self.depth_ignore = depth_ignore_value

        # 距离分段加权配置
        if distance_bin_edges is None and distance_bin_weights is not None:
            raise ValueError("distance_bin_weights is set but distance_bin_edges is None")

        self.distance_bin_edges: Optional[List[float]] = None
        self.distance_bin_weights: Optional[List[float]] = None
        if distance_bin_edges is not None:
            if len(distance_bin_edges) < 2:
                raise ValueError("distance_bin_edges must contain at least 2 values")
            edges = [float(x) for x in distance_bin_edges]
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    raise ValueError("distance_bin_edges must be strictly increasing")

            if distance_bin_weights is None:
                weights = [1.0] * (len(edges) - 1)
            else:
                weights = [float(x) for x in distance_bin_weights]
                if len(weights) != len(edges) - 1:
                    raise ValueError(
                        "distance_bin_weights length must equal len(distance_bin_edges)-1"
                    )

            self.distance_bin_edges = edges
            self.distance_bin_weights = weights

        # CE Loss（不设 ignore_index，通过手动 masking 实现动态过滤）
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        # per-camera weights (optional): register as buffer for device-correctness
        if camera_weights is not None:
            self.register_buffer('camera_weights', torch.tensor(camera_weights, dtype=torch.float32))
        else:
            self.camera_weights = None

    def _weighted_mean(self, values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_tensor(0.0)
        if weights is None:
            return values.mean()
        w = torch.clamp(weights, min=0.0)
        denom = w.sum()
        if denom <= 0:
            return values.new_tensor(0.0)
        return (values * w).sum() / denom

    def _build_distance_weight_map(
        self,
        depth_gt: torch.Tensor,  # [B, N, 1, H, W]
    ) -> Optional[torch.Tensor]:  # [B, N, H, W] float
        if self.distance_bin_edges is None or self.distance_bin_weights is None:
            return None

        depth = depth_gt.squeeze(2)
        weight_map = torch.ones_like(depth, dtype=depth.dtype)
        distance_bin_edges = self.distance_bin_edges
        distance_bin_weights = self.distance_bin_weights
        assert distance_bin_edges is not None and distance_bin_weights is not None
        for i in range(len(distance_bin_edges) - 1):
            lo = distance_bin_edges[i]
            hi = distance_bin_edges[i + 1]
            w = distance_bin_weights[i]
            mask = (depth >= lo) & (depth < hi)
            weight_map = torch.where(mask, weight_map.new_full((), w), weight_map)
        return weight_map

    # ------------------------------------------------------------------
    # 内部：生成类别 mask
    # 提取参与当前 Loss 计算的有效像素
    # ------------------------------------------------------------------
    def _build_class_mask(
        self,
        semantic_gt: torch.Tensor,   # [B, N, H, W]  int64
    ) -> torch.Tensor:               # [B, N, H, W]  bool
        """根据 train_classes 生成像素级类别过滤 mask。

        Returns:
            mask: True 表示该像素属于 train_classes 中的类别，参与损失计算。
                  若 train_classes 为 None，则所有非背景（非 0）像素均为 True。
        """
        if self.train_classes is None:
            # 所有非背景像素
            return semantic_gt > 0    # [B, N, H, W]

        mask = torch.zeros_like(semantic_gt, dtype=torch.bool)
        for cls_id in self.train_classes:
            mask = mask | (semantic_gt == cls_id)
        return mask  # [B, N, H, W]

    # ------------------------------------------------------------------
    # L_d：稀疏深度 L1
    # ------------------------------------------------------------------

    def _loss_sparse_depth(
        self,
        depth_pred:      torch.Tensor,   # [B, N, 1, H, W]
        sparse_depth_gt: torch.Tensor,   # [B, N, 1, H, W]
        class_mask:      torch.Tensor,   # [B, N, H, W]  bool
        pixel_weight: Optional[torch.Tensor] = None,  # [B, N, H, W] float
    ) -> torch.Tensor:
        """L_d：仅在 sparse_depth_gt > 0 且 class_mask == True 的像素计算 L1。

        Returns:
            scalar loss
        """
        valid_depth_mask = (sparse_depth_gt > self.depth_ignore).squeeze(2)  # [B, N, H, W]
        combined_mask = valid_depth_mask & class_mask                         # [B, N, H, W]

        if not combined_mask.any():
            return depth_pred.new_tensor(0.0)

        pred_masked = depth_pred.squeeze(2)[combined_mask]    # [M]
        gt_masked   = sparse_depth_gt.squeeze(2)[combined_mask]  # [M]
        abs_err = torch.abs(pred_masked - gt_masked)

        # combine per-pixel distance weight and per-camera weight (if provided)
        cam_weights = getattr(self, 'camera_weights', None)
        cam_w_map = None
        if cam_weights is not None:
            cam_w_map = cam_weights.view(1, -1, 1, 1).to(depth_pred.device)

        if pixel_weight is not None and cam_w_map is not None:
            full_w = (pixel_weight * cam_w_map.expand_as(pixel_weight))[combined_mask]
        elif pixel_weight is not None:
            full_w = pixel_weight[combined_mask]
        elif cam_w_map is not None:
            full_w = cam_w_map.expand_as(combined_mask.float())[combined_mask]
        else:
            full_w = None

        return self._weighted_mean(abs_err, full_w)

    # ------------------------------------------------------------------
    # L_pd：稠密伪深度 L1
    # ------------------------------------------------------------------

    def _loss_dense_depth(
        self,
        depth_pred:     torch.Tensor,   # [B, N, 1, H, W]
        dense_depth_gt: torch.Tensor,   # [B, N, 1, H, W]
        class_mask:     torch.Tensor,   # [B, N, H, W]  bool
        pixel_weight: Optional[torch.Tensor] = None,  # [B, N, H, W] float
        eps: float = 1e-6               # 安全极小值
    ) -> torch.Tensor:
        """L_pd：仅在 dense_depth_gt > ignore 且 class_mask == True 的像素计算尺度不变 L1。

        Returns:
            scalar loss
        """
        valid_mask = (dense_depth_gt > self.depth_ignore).squeeze(2)  # [B, N, H, W]
        combined_mask = valid_mask & class_mask                       # [B, N, H, W]

        if not combined_mask.any():
            return depth_pred.sum() * 0.0

        # 提取一维的纯有效像素 [M]
        pred_masked = depth_pred.squeeze(2)[combined_mask]       
        gt_masked   = dense_depth_gt.squeeze(2)[combined_mask]   

        cam_weights = getattr(self, 'camera_weights', None)
        cam_w_map = None
        if cam_weights is not None:
            cam_w_map = cam_weights.view(1, -1, 1, 1).to(depth_pred.device)

        if pixel_weight is not None and cam_w_map is not None:
            w = (pixel_weight * cam_w_map.expand_as(pixel_weight))[combined_mask]
        elif pixel_weight is not None:
            w = pixel_weight[combined_mask]
        elif cam_w_map is not None:
            w = cam_w_map.expand_as(combined_mask.float())[combined_mask]
        else:
            w = None

        # ================= Scale-Invariant L1 计算逻辑 =================
        # 1. 极值截断：防止网络在训练初期吐出负数深度或 0 导致 log(0) 爆炸产生 NaN
        pred_safe = torch.clamp(pred_masked, min=eps)
        gt_safe = torch.clamp(gt_masked, min=eps)

        # 2. 转换到对数空间计算差值 (d_i)
        log_pred = torch.log(pred_safe)
        log_gt = torch.log(gt_safe)
        diff = log_pred - log_gt

        # 3. 计算全局尺度偏移量 alpha (对数空间内的平均漂移)
        if w is None:
            alpha = diff.mean()
        else:
            w_safe = torch.clamp(w, min=0.0)
            denom = w_safe.sum()
            if denom <= 0:
                return depth_pred.sum() * 0.0
            alpha = (w_safe * diff).sum() / denom

        # 4. 消除全局尺度后，计算残差的绝对值平均 (L1)
        loss = self._weighted_mean(torch.abs(diff - alpha), w)
        # ===============================================================

        return loss

    # ------------------------------------------------------------------
    # L_sem：语义交叉熵（动态 mask）
    # ------------------------------------------------------------------

    def _loss_semantic(
        self,
        semantic_pred: torch.Tensor,   # [B, N, C, H, W]  logits（Softmax 前）
        semantic_mask_gt: torch.Tensor,  # [B, N, H, W]   int64
        class_mask: torch.Tensor,        # [B, N, H, W]   bool
        pixel_weight: Optional[torch.Tensor] = None,  # [B, N, H, W] float
    ) -> torch.Tensor:
        """L_sem：按已给定的 class_mask 筛选像素，再计算 CrossEntropyLoss。

        class_mask 由外部预先计算，避免重复构建类别过滤掩码。

        Returns:
            scalar loss
        """
        B, N, C, H, W = semantic_pred.shape

        logits_all = semantic_pred.permute(0, 1, 3, 4, 2).reshape(-1, C)   # [B*N*H*W, C]
        gt_all = semantic_mask_gt.reshape(-1).long()                        # [B*N*H*W]
        valid_mask = class_mask.reshape(-1)                                  # [B*N*H*W]

        if not valid_mask.any():
            return semantic_pred.new_tensor(0.0)

        logits_sel = logits_all[valid_mask]   # [M, C]
        gt_sel = gt_all[valid_mask]           # [M]
        ce = F.cross_entropy(logits_sel, gt_sel, reduction='none')
        w = pixel_weight.reshape(-1)[valid_mask] if pixel_weight is not None else None
        return self._weighted_mean(ce, w)


    # ==================================================================
    # 🌟 新增：透明度熵损失 (Opacity Entropy Loss)
    # ==================================================================
    def _loss_opacity_entropy(self, opacities: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        惩罚半透明的高斯点，逼迫 alpha 趋向于 0 (完全透明) 或 1 (绝对物理表面)。
        """
        # 安全截断，防止 log(0) 爆炸
        alpha = opacities.clamp(eps, 1.0 - eps)
        # 二值交叉熵公式构造的“倒U型”惩罚山峰
        entropy = - (alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
        return entropy.mean()

    # ==================================================================
    # 🌟 新增：尺度惩罚损失 (Scale Penalty Loss - 平替失真损失)
    # ==================================================================
    def _loss_scale_penalty(self, scales: torch.Tensor) -> torch.Tensor:
        """
        平替深度失真损失。网络为了在光线上作弊，常常会拉出极长的高斯椭球。
        惩罚所有半轴长度超过 max_scale_thresh 的高斯基元，逼迫它们紧密聚集。
        """
        # 只惩罚超过阈值的部分，正常大小的高斯不产生 Loss
        excess_scale = F.relu(scales - self.max_scale)
        return excess_scale.mean()
    
    
    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(self, inputs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失。

        兼容 OpenOcc 训练入口：forward(inputs_dict)
        必需字段：
            depth_pred, semantic_pred, sparse_depth_gt,
            dense_depth_gt, semantic_gt
        可选字段：
            semantic_mask_gt（缺失时回退为 semantic_gt）

        Returns:
            total_loss: 标量总损失（可直接 .backward()）
            loss_dict:  各分项损失值字典（float，已 detach）
        """
        depth_pred = inputs['depth_pred']
        semantic_pred = inputs['semantic_pred']
        sparse_depth_gt = inputs['sparse_depth_gt']
        dense_depth_gt = inputs['dense_depth_gt']
        semantic_gt = inputs['semantic_gt']
        semantic_mask_gt = inputs.get('semantic_mask_gt', semantic_gt)

        # 提取 3D 空间的物理属性
        gaussians = inputs.get('gaussians', None)
        
        # 1. 生成类别 mask  [B, N, H, W]
        # 稀疏深度损失使用点云投影语义 semantic_gt 做类别过滤。
        # class_mask_sparse = (semantic_gt != 0) & (semantic_gt != 255)
                
        class_mask_sparse = self._build_class_mask(semantic_gt)
        class_mask = self._build_class_mask(semantic_mask_gt)

        if self.add_distance:
            sparse_dist_weight = self._build_distance_weight_map(sparse_depth_gt)
            dense_dist_weight = self._build_distance_weight_map(dense_depth_gt)
        else:
            sparse_dist_weight = None
            dense_dist_weight = None

        # 2. 各分项损失
        l_d   = self._loss_sparse_depth(
            depth_pred, sparse_depth_gt, class_mask_sparse, pixel_weight=sparse_dist_weight
        )
        l_pd  = self._loss_dense_depth(
            depth_pred, dense_depth_gt, class_mask, pixel_weight=dense_dist_weight
        )
        l_sem = self._loss_semantic(
            semantic_pred, semantic_mask_gt, class_mask, pixel_weight=dense_dist_weight
        )
        l_sem_sparse = self._loss_semantic(
            semantic_pred, semantic_gt, class_mask_sparse, pixel_weight=sparse_dist_weight
        )

        # 初始化正则化损失为 0
        l_ent = depth_pred.new_tensor(0.0)
        l_scale = depth_pred.new_tensor(0.0)
        
        # 挂载 3D 几何物理损失
        if gaussians is not None:
            l_ent = self._loss_opacity_entropy(gaussians.opacities)
            l_scale = self._loss_scale_penalty(gaussians.scales)
            
        total = (self.w_sparse * l_d) + \
                (self.w_dense * l_pd) + \
                (self.w_sem * l_sem) + \
                (self.w_ent * l_ent) + \
                (self.w_scale * l_scale)+\
                (self.w_sem_sparse * l_sem_sparse)
                
        loss_dict: Dict[str, float] = {
            'L_d':     l_d.detach().mean().item(),
            'L_pd':    l_pd.detach().mean().item(),
            'L_sem':   l_sem.detach().mean().item(),
            'L_ent':   l_ent.detach().mean().item(),
            'L_scale': l_scale.detach().mean().item(),
            'L_sem_sparse': l_sem_sparse.detach().mean().item(),
            'L_total': total.detach().mean().item(),
        }
        return total, loss_dict

    
```

## 稠密伪深度损失的计算原理

伪深度图能很好地表达物体的相对结构（比如车头比车尾离我近），但它的绝对距离可能是不准的（把 10 米预测成了 20 米，整体放大了 2 倍）。

为了消除这种全局尺度的缩放误差，我们将深度转换到对数空间：

$$
\log(D_{pred}) - \log(D_{gt}) = \log\left(\frac{D_{pred}}{D_{gt}}\right)
$$

接着计算这张图上所有像素对数差值的平均值 $\alpha$，即当前预测结果偏离真值的全局尺度漂移量，也就是数量级：

$$
\alpha = Mean(\log(D_{pred}) - \log(D_{gt}))
$$

最后，用每个像素的差值减去这个全局漂移量，再求绝对值：

$$
L_{pd} = \frac{1}{M} \sum \left| (\log D_{pred} - \log D_{gt}) - \alpha \right|
$$

这样即使伪深度图整体放大了一倍，，损失值依然是 0，只要整体结构没问题就是对的，网络可以方向的学习局部结构的高频细节，而把绝对尺度的任务交给 $L_d$。

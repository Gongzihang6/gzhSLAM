# semantic_depth_rasterizer.py

这个文件实现了专门为语义和深度定制的3D Gaussian Splatting 可微渲染器。在以高斯为中心的3D场景表示中，网络在3D空间中预测出一团团的高斯球来表示整个世界。但是我们的真值通常是2D的，比如车上7个相机拍到的图像，以及由图像分割出的2D语义图、LiDAR投影到图像上的稀疏深度图。

这个渲染器的作用就是**一座连接 3D 预测和 2D 真值的桥梁**。它把 3D 空间中的高斯球，结合相机的内外参，投影（Splatting）到各个 2D 相机平面上，生成 2D 的深度图和语义图，从而让网络能够计算 Loss 并进行反向传播。

```python
@dataclass
class GaussianParams:
    """3D Gaussian 参数容器，扩展支持高维语义特征。

    Args:
        means:             [B, N, 3]  Gaussian 中心（ego/world 坐标，单位：米）
        scales:            [B, N, 3]  正尺度值（已经过 exp/sigmoid 映射）
        rotations:         [B, N, 4]  归一化四元数 (w, x, y, z)
        opacities:         [B, N, 1]  不透明度，值域 [0, 1]
        semantic_features: [B, N, C]  语义特征，C=16，无激活（raw logits）
    """
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    semantic_features: torch.Tensor
```

标准的 3DGS 存储的是球谐函数（SH）来表示 RGB 颜色。但在这里，为了自动驾驶的语义理解，作者把颜色替换成了 `semantic_features`。在 Uni3R 范式中，这些特征保留为原始的 logits（不经过激活函数），直到渲染到 2D 图像平面后，才统一经过 Softmax 变为概率。

```python
def compute_3d_covariance(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """计算 3D 协方差矩阵：Sigma = R @ diag(s)^2 @ R^T。

    Args:
        scales:    [..., N, 3]  正尺度值
        rotations: [..., N, 4]  四元数
    Returns:
        Sigma: [..., N, 3, 3]
    """
    R = quat_to_rotmat(rotations)           # [..., N, 3, 3]
    RS = R * scales.unsqueeze(-2)           # [..., N, 3, 3]  R @ diag(s)
    Sigma = torch.matmul(RS, RS.transpose(-1, -2))  # [..., N, 3, 3]
    return Sigma
```

计算3D协方差矩阵$\Sigma_{3D}$。利用尺度矩阵 $S$ 和旋转矩阵 $R$，将一个标准球体拉伸旋转成 3D 椭球：

$$
\Sigma_{3D} = R S S^T R^T
$$

```python
def compute_2d_covariance_ewa(
    means_cam: torch.Tensor,  # [N, 3]  相机坐标系下的 Gaussian 中心
    scales: torch.Tensor,     # [N, 3]
    rotations: torch.Tensor,  # [N, 4]
    W_cam: torch.Tensor,      # [3, 3]  相机旋转矩阵（extrinsic 的 [:3, :3]）
    K3: torch.Tensor,         # [3, 3]  相机内参
) -> Tuple[torch.Tensor, torch.Tensor]:
    """EWA 投影：计算屏幕空间 2D 协方差矩阵。

    Reference: "EWA Splatting" (Zwicker et al. 2002)
               "3D Gaussian Splatting" (Kerbl et al. 2023)

    Returns:
        cov2d: [N, 2, 2]  2D 协方差矩阵
        valid: [N]        bool，True 表示 Gaussian 在相机前方
    """
    N = means_cam.shape[0]
    z = means_cam[:, 2]
    valid = z > 0.01
    z_safe = z.clamp(min=1e-4)

    x_c, y_c = means_cam[:, 0], means_cam[:, 1]
    fx, fy = K3[0, 0], K3[1, 1]

    # 透视投影的 Jacobian：[N, 2, 3]
    J = torch.zeros(N, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * x_c / z_safe.pow(2)
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * y_c / z_safe.pow(2)

    # 3D 协方差在相机空间：Sigma_cam = W @ Sigma_world @ W^T
    Sigma_world = compute_3d_covariance(scales, rotations)  # [N, 3, 3]
    W = W_cam.unsqueeze(0)                                   # [1, 3, 3]
    Sigma_cam = W @ Sigma_world @ W.transpose(-1, -2)        # [N, 3, 3]

    # 投影到 2D：Sigma_2D = J @ Sigma_cam @ J^T
    J_Sigma = torch.bmm(J, Sigma_cam)                       # [N, 2, 3]
    cov2d = torch.bmm(J_Sigma, J.transpose(-1, -2))         # [N, 2, 2]

    # 保证数值稳定性（最小方差）
    # 这是为了防止投影后的高斯球比一个像素还小（引发强烈的锯齿化或奇异矩阵）
    cov2d[:, 0, 0] = cov2d[:, 0, 0].clamp(min=0.3)
    cov2d[:, 1, 1] = cov2d[:, 1, 1].clamp(min=0.3)

    return cov2d, valid
```

透视投影是一个非线性过程，为了把3D协方差矩阵投影到2D平面，需要在一阶泰勒展开下求投影函数的雅可比矩阵$J$。

将世界坐标系下的协方差，先通过外参的旋转矩阵 $W$ 变换到相机坐标系，再通过雅可比矩阵 $J$ 投影到 2D：

$$
\Sigma_{2D} = J W \Sigma_{3D} W^T J^T
$$

```python
@MODELS.register_module()
class SemanticDepthRasterizer(nn.Module):
    """基于 Generalizable Gaussian Splatting 的深度与语义渲染器。

    渲染流程（参考 Uni3R arXiv:2508.03643）：
      1. 将 Gaussian 中心从 ego 系变换到目标相机坐标系
      2. 按 Z_c 深度排序（从近到远）
      3. 对每像素进行 alpha-blending 累积：
           alpha_i = opacity_i * gaussian_weight_2D_i(pixel)
           T_i = Π_{j<i}(1 - alpha_j)
           depth_pixel   += T_i * alpha_i * Z_c_i
           feature_pixel += T_i * alpha_i * semantic_feat_i
      4. 语义特征渲染后接 Softmax（用于 CE Loss）

    Args:
        num_semantic_classes (int):   语义类别数，默认 16（含 empty_label=0）
        near (float):                 近平面（米），默认 0.1
        far  (float):                 远平面（米），默认 100.0
        backend (str):                渲染后端，'gsplat' / 'pytorch' / 'auto'
        max_radius_px (int):          PyTorch 后端 Gaussian 最大半径（像素）
    """

    def __init__(
        self,
        num_semantic_classes: int = 16,
        near: float = 0.1,
        far: float = 100.0,
        backend: str = 'auto',
        max_radius_px: int = 32,
        depth_alpha_threshold: float = 1e-2,
    ):
        super().__init__()
        self.num_classes = num_semantic_classes
        self.near = near
        self.far = far
        self.max_radius_px = max_radius_px
        self.depth_alpha_threshold = depth_alpha_threshold

        if backend == 'auto':
            try:
                import gsplat
                del gsplat  # 只用于检测是否安装，避免占用命名空间
                self._backend = 'gsplat'
            except ImportError:
                self._backend = 'pytorch'
        else:
            self._backend = backend

    def _dump_means_xyz_for_cloudcompare(self, means_xyz: torch.Tensor, save_path: str) -> None:
        """Save Gaussian centers [N,3] to an ASCII .xyz file for CloudCompare."""
        if not save_path:
            raise ValueError("save_path must be a non-empty path")

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        xyz = means_xyz.detach().float().cpu().numpy()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"means_xyz must have shape [N,3], got {tuple(xyz.shape)}")
        np.savetxt(save_path, xyz, fmt="%.6f")

    def forward(
        self,
        gaussians: GaussianParams,
        target_intrinsics: torch.Tensor,  # [B, M, 4, 4]
        target_cam2ego: torch.Tensor,     # [B, M, 4, 4]
        image_size: Tuple[int, int],      # (H, W)
    ) -> dict:
        """渲染深度图和语义特征图。

        Args:
            gaussians:          GaussianParams，含 means/scales/rotations/opacities/semantic_features
            target_intrinsics:  [B, M, 4, 4]  目标视角相机内参（齐次）
            target_cam2ego:     [B, M, 4, 4]  目标视角 cam2ego 变换
            image_size:         (H, W)         渲染分辨率

        Returns:
            dict:
                'depth_map':    [B, M, 1, H, W]  渲染深度图（米）
                'semantic_map': [B, M, C, H, W]  语义 logits（Softmax 前）
                'alpha_map':    [B, M, 1, H, W]  累积不透明度
        """
        B, M = target_cam2ego.shape[:2]
        H, W = image_size

        # ego2cam: [B, M, 4, 4]，用于把Ego坐标系下生成的高斯球转化到具体的某一个相机视角下
        ego2cam = torch.inverse(target_cam2ego)

        depth_maps, semantic_maps, alpha_maps = [], [], []
        for m in range(M):	# 遍历所有相机视角
            E = ego2cam[:, m]              # [B, 4, 4]  ego→cam
            K = target_intrinsics[:, m]    # [B, 4, 4]
            K3 = K[:, :3, :3]             # [B, 3, 3]

            # 根据具体后端实现，使用高斯属性、相机内参、外参、图像大小实现深度图、语义图渲染
            if self._backend == 'gsplat':
                dep, sem, alp = self._rasterize_gsplat(gaussians, K3, E, H, W)
            else:
                dep, sem, alp = self._rasterize_pytorch(gaussians, K3, E, H, W)

            # Hard filter: invalidate expected depth where accumulated alpha is too low.
            if self.depth_alpha_threshold > 0:
                dep = dep * (alp > self.depth_alpha_threshold).to(dep.dtype)

            depth_maps.append(dep)
            semantic_maps.append(sem)
            alpha_maps.append(alp)

        return {
            'depth_map':    torch.stack(depth_maps, dim=1),    # [B, M, 1, H, W]
            'semantic_map': torch.stack(semantic_maps, dim=1), # [B, M, C, H, W]
            'alpha_map':    torch.stack(alpha_maps, dim=1),    # [B, M, 1, H, W]
        }

    # ------------------------------------------------------------------
    # gsplat 后端（推荐）
    # ------------------------------------------------------------------

    def _rasterize_gsplat(
        self,
        gaussians: GaussianParams,
        K3: torch.Tensor,   # [B, 3, 3]
        E: torch.Tensor,    # [B, 4, 4]  ego2cam
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用 gsplat 库渲染（推荐，高效 CUDA 实现）。

        安装命令：pip install gsplat
        项目地址：https://github.com/nerfstudio-project/gsplat

        显式使用 gsplat 的深度渲染模式：render_mode='RGB+ED'。
        其中颜色通道仅传入语义特征（C 维），深度由 gsplat 直接输出 Expected Depth (ED)。
        """
        try:
            from gsplat import rasterization
        except ImportError as exc:
            raise RuntimeError(
                "gsplat 未安装，请执行：pip install gsplat\n"
                "或在配置中设置 backend='pytorch' 使用纯 PyTorch 实现。"
            ) from exc
		  # 批量、视角、特征维度
        B, N, C = gaussians.semantic_features.shape
        device = gaussians.means.device

        depth_list, sem_list, alpha_list = [], [], []

        for b in range(B):
            means_b = gaussians.means[b]                          # [N, 3]
            quats_b = F.normalize(gaussians.rotations[b], dim=-1) # [N, 4]
            scales_b = gaussians.scales[b]                        # [N, 3]
            opacities_b = gaussians.opacities[b, :, 0]           # [N]

            # 颜色通道仅传语义特征；深度由 gsplat 的 ED 模式输出。
            colors = gaussians.semantic_features[b]                # [N, C]

            # gsplat.rasterization:
            #   viewmats: [num_cameras, 4, 4] world-to-cam (即 ego2cam)
            #   Ks:       [num_cameras, 3, 3] 相机内参
            renders, alpha, _ = rasterization(
                means=means_b,          # [N, 3]
                quats=quats_b,          # [N, 4]  (w, x, y, z)
                scales=scales_b,        # [N, 3]
                opacities=opacities_b,  # [N]
                colors=colors,          # [N, C]
                viewmats=E[b:b+1],      # [1, 4, 4]
                Ks=K3[b:b+1],           # [1, 3, 3]
                width=W,
                height=H,
                near_plane=self.near,
                far_plane=self.far,
                packed=False,
                render_mode='RGB+ED',
            )
            # renders: [1, H, W, C+1]（前 C 为语义特征，最后 1 维为 Expected Depth）
            # alpha:   [1, H, W, 1]
            renders = renders[0].permute(2, 0, 1)   # [C+1, H, W]
            alpha_out = alpha[0].permute(2, 0, 1)   # [1, H, W]

            sem_list.append(renders[:C])             # [C, H, W]  语义 logits
            depth_list.append(renders[C:C + 1])      # [1, H, W]  深度
            alpha_list.append(alpha_out)             # [1, H, W]

        return (
            torch.stack(depth_list),   # [B, 1, H, W]
            torch.stack(sem_list),     # [B, C, H, W]
            torch.stack(alpha_list),   # [B, 1, H, W]
        )

    # ------------------------------------------------------------------
    # PyTorch EWA 后端（调试/小规模测试）
    # ------------------------------------------------------------------

    def _rasterize_pytorch(
        self,
        gaussians: GaussianParams,
        K3: torch.Tensor,   # [B, 3, 3]
        E: torch.Tensor,    # [B, 4, 4]  ego2cam
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """纯 PyTorch EWA Gaussian Rasterizer（慢但正确）。

        适用于无 GPU CUDA 环境的调试和小规模单元测试。
        生产训练请使用 gsplat 后端（backend='gsplat'）。
        """
        B, N, C = gaussians.semantic_features.shape
        device = gaussians.means.device

        depth_list, sem_list, alpha_list = [], [], []
        for b in range(B):
            ones = torch.ones(N, 1, device=device)
            means_h = torch.cat([gaussians.means[b], ones], dim=-1)  # [N, 4]
            means_cam = (E[b] @ means_h.T).T[:, :3]                  # [N, 3]
            z_c = means_cam[:, 2].clamp(self.near, self.far)          # [N]

            # 拼接语义 + 深度为 (C+1) 维特征
            features_ext = torch.cat([
                gaussians.semantic_features[b],   # [N, C]
                z_c.unsqueeze(-1),                # [N, 1]
            ], dim=-1)                            # [N, C+1]

            W_cam = E[b, :3, :3]   # [3, 3]  相机旋转
            K3_b = K3[b]           # [3, 3]

            rendered, alpha = self._rasterize_single_ewa(
                means_cam=means_cam,
                scales=gaussians.scales[b],
                rotations=gaussians.rotations[b],
                opacities=gaussians.opacities[b, :, 0],
                features=features_ext,
                W_cam=W_cam,
                K3=K3_b,
                H=H, W=W,
            )
            # rendered: [C+1, H, W]  alpha: [1, H, W]
            sem_list.append(rendered[:C])
            depth_list.append(rendered[C:C + 1])
            alpha_list.append(alpha)

        return (
            torch.stack(depth_list),
            torch.stack(sem_list),
            torch.stack(alpha_list),
        )

    def _rasterize_single_ewa(
        self,
        means_cam: torch.Tensor,   # [N, 3]  相机坐标系下的 Gaussian 中心
        scales: torch.Tensor,       # [N, 3]
        rotations: torch.Tensor,    # [N, 4]
        opacities: torch.Tensor,    # [N]
        features: torch.Tensor,     # [N, D]
        W_cam: torch.Tensor,        # [3, 3]
        K3: torch.Tensor,           # [3, 3]
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单视角 EWA 光栅化（Python 循环实现）。

        算法流程：
          1. 计算各 Gaussian 在屏幕上的投影位置 (u, v)
          2. 利用 EWA 计算 2D 协方差 Sigma_2D
          3. 按深度排序，从近到远执行 alpha-blending

        注意：此实现使用 Python 循环，性能较低。
              推荐对 N > 1000 或图像 > 256×256 的场景使用 gsplat 后端。
        """
        N, D = features.shape
        device = features.device

        z = means_cam[:, 2]
        fx, fy = K3[0, 0], K3[1, 1]
        cx, cy = K3[0, 2], K3[1, 2]

        # 投影到图像坐标
        z_safe = z.clamp(min=1e-4)
        u_proj = means_cam[:, 0] / z_safe * fx + cx  # [N]
        v_proj = means_cam[:, 1] / z_safe * fy + cy  # [N]

        # 筛选可见 Gaussian（在相机前方且在屏幕范围内）
        visible = (
            (z > self.near)
            & (u_proj >= 0) & (u_proj < W)
            & (v_proj >= 0) & (v_proj < H)
        )

        # 初始化输出
        rendered = torch.zeros(D, H, W, device=device, dtype=features.dtype)
        T_map = torch.ones(H, W, device=device, dtype=features.dtype)
        alpha_acc = torch.zeros(1, H, W, device=device, dtype=features.dtype)

        if not visible.any():
            return rendered, alpha_acc

        # 只处理可见 Gaussians
        idx = visible.nonzero(as_tuple=True)[0]
        mc = means_cam[idx]
        sc = scales[idx]
        ro = rotations[idx]
        op = opacities[idx]
        fe = features[idx]
        uv = torch.stack([u_proj[idx], v_proj[idx]], dim=-1)  # [M, 2]
        z_v = z[idx]

        # 按深度排序（从近到远）
        sort_idx = z_v.argsort()
        mc, sc, ro, op, fe, uv, z_v = (
            mc[sort_idx], sc[sort_idx], ro[sort_idx],
            op[sort_idx], fe[sort_idx], uv[sort_idx], z_v[sort_idx],
        )

        # 计算 2D 协方差
        cov2d, _ = compute_2d_covariance_ewa(mc, sc, ro, W_cam, K3)  # [M, 2, 2]

        # 2D 协方差逆矩阵
        det = (cov2d[:, 0, 0] * cov2d[:, 1, 1]
               - cov2d[:, 0, 1] * cov2d[:, 1, 0]).clamp(min=1e-6)
        cov_inv = torch.stack([
            cov2d[:, 1, 1] / det, -cov2d[:, 0, 1] / det,
            -cov2d[:, 1, 0] / det, cov2d[:, 0, 0] / det,
        ], dim=-1).reshape(-1, 2, 2)  # [M, 2, 2]

        # 计算每个 Gaussian 的屏幕空间影响半径（3σ）
        trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
        disc = ((cov2d[:, 0, 0] - cov2d[:, 1, 1]).pow(2) / 4
                + cov2d[:, 0, 1].pow(2)).clamp(min=0)
        lambda_max = trace / 2 + disc.sqrt()
        radii = (3 * lambda_max.sqrt()).ceil().long().clamp(max=self.max_radius_px)

        M = uv.shape[0]
        for n in range(M):
            u0, v0 = uv[n, 0].item(), uv[n, 1].item()
            r = int(radii[n].item())

            u_lo = max(0, int(u0) - r)
            u_hi = min(W, int(u0) + r + 1)
            v_lo = max(0, int(v0) - r)
            v_hi = min(H, int(v0) + r + 1)

            if u_lo >= u_hi or v_lo >= v_hi:
                continue

            # 构建补丁像素坐标偏移
            us = torch.arange(u_lo, u_hi, device=device, dtype=features.dtype) - uv[n, 0]
            vs = torch.arange(v_lo, v_hi, device=device, dtype=features.dtype) - uv[n, 1]
            vv, uu = torch.meshgrid(vs, us, indexing='ij')          # [pH, pW]

            # 计算 Mahalanobis 距离：delta^T @ Sigma_2D^{-1} @ delta
            d = torch.stack([uu, vv], dim=-1).reshape(-1, 2)        # [pH*pW, 2]
            Ci = cov_inv[n]                                          # [2, 2]
            mahal = (d @ Ci * d).sum(-1).reshape(vv.shape)          # [pH, pW]

            # 2D Gaussian 权重与不透明度
            w = torch.exp(-0.5 * mahal)
            alpha_n = (op[n] * w).clamp(max=0.9999)                 # [pH, pW]

            # 使用当前 T_map（透射率）进行 alpha-blending
            T_patch = T_map[v_lo:v_hi, u_lo:u_hi]                   # [pH, pW]
            contrib = T_patch * alpha_n                               # [pH, pW]

            # 累积渲染特征和不透明度
            rendered[:, v_lo:v_hi, u_lo:u_hi] += (
                contrib[None] * fe[n, :, None, None]
            )
            alpha_acc[0, v_lo:v_hi, u_lo:u_hi] += contrib

            # 更新透射率
            T_map[v_lo:v_hi, u_lo:u_hi] = T_patch * (1 - alpha_n)

        return rendered, alpha_acc
```


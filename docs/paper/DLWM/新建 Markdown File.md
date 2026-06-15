# 3D Camera Ray Rmbeddding
下文为一份完整方案图文（3D Camera Ray Rmbeddding）的核心文档，包含原理公式推导与代码实现、Raw Ray Token。

任何一个相机像素格子 $(u,v)$，其本质都是三维空间中一条从相机光心出发、朝向无穷远处的物理射线；使用两个特征做三维射线完整无损方式表征射线：
1.  射线原点 Ray Origin：相机局部坐标系中心坐标
2.  射线方向 Ray Direction：该像素在 3D 空间中射出的单位方向矢量

假定当前相机 $w$ 的内参矩阵为 $K_w$，从当前相机视角任一车体 Ego 姿态外参矩阵为 $R_w$，平移向量为 $t_w$。对于图像上的任意一个像素点 $\boldsymbol{p} = [u,v,1]^T$：

## 1. 输出特征方向 Ray Origin

由于同一幅图像内全部像素共享同一个相机光心，因此对于图像上的所有像素，其对应射线 Ego 系下是一个严格相等的常数，直接等于外参的平移向量：
$$
o_w = t_w \in \mathbb{R}^3
$$

## 2. 输出特征方向 Ray Direction
$$
d_{w, p} = \frac{[R_w \cdot K_w^{-1} \cdot [u, v,1]^T]}{||R_w \cdot K_w^{-1} \cdot [u, v,1]^T||_2} \in \mathbb{R}^3
$$

## 3. Raw Ray Token
$$
R_{w, p} = [O_w, d_{w, p}] \in \mathbb{R}^6
$$

## 4. 6 维向量通过 MLP 升维作为多尺度图像特征输入；进行 cross-attention 时，自带 ray token 信息，隐式空间 3D 位置分辨。

---

# 代码实现
```python
# ray_token 三维射线表征位置编码特征模块
class RayEmbedding(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        # 4层 MLP 升维，输入6维ray raw特征，输出embed_dim维度特征(如1024)
        self.ray_mlp = nn.Sequential(
            nn.Linear(6, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def compute_raw_rays(self, intrinsics, extrinsics, height, width, patch_size=16):
        """
        逐图计算整张图所有patch的O-D射线原始特征
        intrinsics : [B, V, 3, 3]
        extrinsics : [B, V, 4, 4] (Cam to Ego)
        """
        B, V, _, _ = intrinsics.shape
        device = intrinsics.device

        # 1. 采样patch大小网格坐标
        oh, pw = width // patch_size, width // patch_size
        y_coords, x_coords = torch.meshgrid(
            torch.arange(oh, device=device),
            torch.arange(pw, device=device),
            indexing='ij'
        )
        # 转换到像素中心坐标
        x_centers = (x_coords * patch_size + patch_size / 2).float()
        y_centers = (y_coords * patch_size + patch_size / 2).float()
        ones = torch.ones_like(x_centers)
        pixel_points = torch.stack([x_centers, y_centers, ones], dim=-1) # [oh, pw, 3]

        # 扩展多相机batch维度 [B, V, oh, pw, 3]
        pixel_points = pixel_points.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1, 3)

        # 2. 相机内参求逆 K_inv @ pixel_points
        K_inv = torch.inverse(intrinsics) # [B, V, 3, 3]
        cam_dir = torch.matmul(K_inv, pixel_points[..., None]) # [B, V, oh, pw, 3, 1]

        # 3. 外参旋转部分分离
        R_mat = extrinsics[:, :, :3, :3] # [B, V, 3, 3] rotation
        t_ori = extrinsics[:, :, :3, 3]  # [B, V, 3] translation / origin

        # 4. 变换Ego系下射线方向 ray_dir
        ray_dir = torch.matmul(R_mat, cam_dir)
        ray_dir = F.normalize(ray_dir, p=2, dim=-2) # L2归一化为单位方向矢量

        # 5. 扩展原点到所有patch尺寸
        ray_ori = t_ori.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [B, V, 3, oh, pw]
        ray_ori = ray_ori.permute(0,1,3,4,2) # [B, V, oh, pw, 3]

        # 6. 拼接原点+方向6维raw射线
        raw_rays = torch.cat([ray_ori, ray_dir], dim=-1)

        # 调整维度输出 [B, V, oh, pw, 6]
        return raw_rays.reshape(B, V, -1, 6).contiguous()

    def forward(self, patch_tokens, intrinsics, extrinsics, height, width, patch_size=16):
        """
        Args:
            patch_tokens: [B, V, N_patches, C] 图像下采样得到的patch token
        """
        B, V, N_patches, C = patch_tokens.shape
        N = intrinsics.shape[0]
        assert N == B

        # 1. 求解raw rays: [B, V, N_patches, 6]
        raw_rays = self.compute_raw_rays(intrinsics, extrinsics, height, width, patch_size)
        raw_rays = raw_rays.view(B, V, N_patches, 6)

        # 2. MLP升维射线特征: [B, V, N_patches, embed_dim]
        ray_features = self.ray_mlp(raw_rays)

        # 3. 几何特征融合：将3D射线特征作为位置编码加到图像patch token上
        # 此处patch token 内部不仅包含了图像纹理，每一个token 都额外编码了对应2D窗口(uv)的3D射线几何信息
        output_tokens = patch_tokens + ray_features

        return output_tokens
```

代码调用示例

```python
# 1. Patch Token 提取
patch_tokens = self.patch_embed(image) # [B*V, N_patches, C]

# 2. 射线嵌入层实例化，输入6D raw射线升维叠加
ray_embedding_layer = RayEmbedding(embed_dim=1024)
patch_tokens = self.ray_embedding_layer(patch_tokens, intrinsics, extrinsics, H, W)

# 3. 拼接3种token：Ego Pose Token / Register Token / Patch Token
# 全局位姿、可学习register、图像patch三维token拼接
tokens = torch.cat([ego_pose_token, register_token, patch_tokens], dim=1) # [N, L, embed_dim]
```

---

## 方案核心总结
1.  用 **射线原点(3 维)** + **归一化射线方向(3 维)** 共 6 维向量唯一表征图像任意像素窗口对应的 3D 空间射线；
2.  6 维 Raw Ray Token 通过单层 MLP 升维至图像 patch token 相同维度，作为 **3D 位置编码** 叠加至图像视觉特征；
3.  无需额外深度真值，仅依赖相机内外参即可得到精准像素级 3D 几何位置先验，在跨视角 attention、3D 感知任务中天然引入空间几何约束。



----

### 整体车端框架

![image-20260610215345744](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610215345904.png)

### 模型训练框架



![image-20260610215404327](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610215404655.png)

## LiDAR 融合方法：

1.  ### 3DGS 方案: 

初始化 3DGS 基元：两类 3DGS query，一部分用 voxel 降采样后的 lidar 点初始化 3DGS 基元的位置，另一部分完全 learnable；

lidar 编码：两类编码方法

-   PointPillar 直接生成 BEV 空间特征：
-   PointTransformer， KPConv,  直接生成每个点对应的特征

3DGS 与 lidar 编码后特征的交互方式;

-   PointPillar: 3DGS query 与 BEV 空间 deform attn
-   PTv3: 3DGS query 与 K 近邻点 feature，CA

然后 3DGS 再与图像 feature 交互，按照 DLWM 里的方式

1.  ### DVGT 类方案：

GGPT 论文中，用 SFM 方法身成了稀疏的相对精确的几何点，我们可以把前 lidar 看等价为 SFM 点，这些点主要用来提升几何精度；

具体做法:

![image-20260610215509767](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610215509974.png)

## 压缩模块：

![image-20260610215538889](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610215539033.png)

## VLM 语义 align：

VGG-omega 论文提出，在后期 Fine-tuning，language align 作为一个辅助任务进行的，用于 **验证几何表征的语义互通性**，提出一个静态可学习的 language token 参数，但是只与 register token concat ，完全摒弃 patch token ，这样文本就不会影响 patch token 了；通过 4 层 self attention 得到 $$E_{vggt}$$  与  VLM 输出的 $$E_{text}$$ 进行 infoNCE 对比学习；

![image-20260610215618151](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260610215618314.png)

设计了高度结构化的 Prompt，促使 VLM（如 CLIP 的文本端或更大的多模态模型）将多视图下的图像视为一个 **整体相干场景** 进行描述，分为 `Scene`（场景类别）、`Content`（主要物体及空间布局）、`Appearance`（材质颜色风格）三个字段。

**特征提取与处理**:

1.  VLM 观察所有输入视图后，生成一段 Fact、简练的场景描述文本。
2.  提取 VLM 生成该文本时的 Token 隐状态（Hidden States）。
3.  排除 Prompt 本身，仅对新生成的文本 Token 隐状态进行 **平均池化（Mean-pooling）**。
4.  进行 **l2 归一化**，得到最终的 **目标语言嵌入向量** $$E_{text}$$





---

# Pretrain

## Geometry 表征与预训练 Feature 交互

>   预训练里的 feature 到底在哪个物理空间里，最终如何进入 BEV 工作空间，并被 Planning 做 KV 查询？

> [!Note]
> 最终物理工作空间是 `ego_t` 坐标系下的 BEV grid。Planning 在 BEV cell 上查询和打分，但是每个 BEV cell 里存的不是裸 BEV Feature，也不是 yes/no 占据，而是 Stage1/Stage2 预训练出来的 geometry/temporal/future latent 潜在表示。

workspace和特征的来源是分开的。

| 问题                                        | 回答                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| Planning最后在哪里工作？                    | 在当前帧ego_t的BEV x-y网格上工作                             |
| 候选轨迹在哪里定义？                        | 也是ego_t的BEV坐标，形如（x_k,y_k,t_k）                      |
| Planning查的是什么？                        | 查同一个BEV cell里的三层预训练latent：当前Geometry、历史temporal gemetroy、未来geometry rollout |
| 这些latent是不是普通BEV？                   | 形状是BEV-shaped，但内容来说ray/depth/point/visibility/temporal/future预训练 |
| Planning需要先等OD/Occ/FutureHead的输出吗？ | 不需要，OD/Occ/FutureHead是可选probe，Planning直接查latent； |


















































































































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

```
Final workspace = ego_t BEV grid
Cell feature    = Stage1 geometry latent
                 + Stage2 temporal geometry latent
                 + Stage2 future geometry latent
Planning        = candidate trajectory Q queries these BEV latent K/V
```

### 🧩 BEV Dense or Sparse？

最准确的定义不是“纯 sparse BEV”，也不是“传统裸 dense BEV”

**Visibility-aware dense BEV latent with sparse planning queries,可见性约束的稠密 BEV latent + 稀疏轨迹查询。**

| **模块 / 维度**       | **状态 / 属性**        | **详细说明**                                                 |
| --------------------- | ---------------------- | ------------------------------------------------------------ |
| **BEV 工作空间**      | **Dense**              | geometry_bev_seed, geometry_world_feature, future_geometry_feature。默认是固定 `Hb x Wb` 网格，便于 ego warp, temporal fusion, future rollout 和 probe 解码。 |
| **原始几何 evidence** | **Sparse / ray-based** | camera ray, LiDAR point。有效 dense-depth mask 都不是全空间均匀观测；只有被 ray / LiDAR / teacher 确认过的区域才有强证据。 |
| **BEV cell 可信度**   | **Visibility-aware**   | 每个 cell 可以有 latent, 但必须带 observed / occluded / unknown / uncertainty; 不能把所有 dense cell 都当成确定 free。 |
| **Planning 读取方式** | **Sparse query**       | Planning 不是读取 `Hb x Wb` dense BEV 直接喂给 MLP，而是只出 `N x K` 个候选轨迹点做 bilinear sample / KV query。 |
| **上车优化空间**      | **Hybrid**             | 工程上可把远端、低置信、非关键区域 token 化或稀疏化，但接口口径仍然与 BEV grid 对齐。 |

```
Representation: dense BEV latent
observation:   sparse/ray-based/visibility-aware
Planning read: sparse trajectory query
```

不是“sparse BEV”，因为sparse bev一般只有 occupied cells 或只有 sparse tokens；也不是 "dense BEV"，因为 dense bev 没有 ray evidence、unknown、uncertainty 的相关概念约束。 空间上是 **dense BEV grid**，但其实是 **sparse/visibility-aware**，Planning 访问上是 **sparse query**。

### 🧭 坐标系约定

全义默认用当前帧 ego 坐标系 `ego_t`：

```
       z up
       ▲
       |       
       |       camera ray / point map 有 z
       |
     ego_t ----▶ x forward
       |
       ▼ y lateral


  BEV working plane = x-y grid at ego_t.
  Height z is compressed into latent channels, confidence, visibility, and uncertainty.

```

物理坐标链路

```
                                                    Plain
  image pixel (u,v)
     -> camera ray r(u,v)
     -> metric ray depth d(u,v)
     -> ego 3D point p_ego=(x,y,z)
     -> BEV cell cell(ix, iy)
     -> temporal BEV latent at ego_t
     -> future BEV latent[k] at ego_t
     -> planning query at (x_k, y_k, t_k)
```

**future slice 也表达在当前 `ego_t` 坐标系下**，不是表达在未来 ego 坐标系下，这样 Planning 的候选轨迹可以在同一个 BEV grid 里查询 `t+1..t+K`

### 🏢 BEV Cell 工作空间

BEV 空间是一个二维的网格，每个 cell 里面存放着关于这一块物理空间的特征。

```
  cell(ix, iy)
   - covers x ∈ [x_min+ix*dx, x_min+(ix+1)*dx)
   -        y ∈ [y_min+iy*dy, y_min+(iy+1)*dy)
   - stores a D-dimensional latent vector
```

这里的 “特征” 不是简单的 “有/无障碍物 (occupancy)” 或 “语义类别 (category)”，而是高维的隐式特征向量 (Latent)。它不仅“浓缩”了跨模态特征（图像、点云等），还包含了对过去（时间上下文）的记忆和对未来的预测（这部分在后续会详细讲解）。

### 🗺️ BEV 网格俯视图

从上帝视角看 BEV cell，x 向前，y 向左（以 ego 为中心）。

```
                           x forward
                             ▲
               y             |             right
               left   +------+------+      + = trajectory point t+k
                      |      |      |      . = tau
                      |   .  |   +  |      . = tau
                      |      |      |      . = tau
                 ego ----+------+------+---
                      |      |      |      . = tau
                     |   +  |      |      . = tau
                     |      |   .  |
                     +------+------+
  
              ix = 0,1...       internal cells         ... Nx, Ny...
 
              ego candidate trajectory point (x_k, y_k) is up to t+K

```

🧩 单个 Cell 的内部卡片

```
   one bev cell: cell(ix, iy)
         +------------+
         |            |
         | cell(ix,iy)|
         |            |
         +------------+
               |
               v
   Physical Meaning:
    x in [x_min, x_max_dx)
    y in [y_min, y_max_dy)
    z is not explicitly partitioned, map/z-confidence in Latent.
 
  Stored Latent at the same cell:
    geometry_bev_seed[t, ix, iy]         # stage 1
    geometry_world_feature[t, ix, iy]    # stage 2
    visibility / unknown / uncertainty_to_...[ix, iy]
    future_geometry_feature[t_k, ix, iy] # if rollout

```

### 🎯 连续轨迹点如何采样 Cell

轨迹点是一个连续坐标，它并不一定刚好落在一个 cell 中心，所以工程上通常用双线性插值 (bilinear sample)。

```
代码块
1   continuous 2D point (x, y) --> fractional index (ix_f, iy_f)
2 
3             ix          ix+1
4         +-------+-------+
5         |       |       |
6     iy  |  00   |  10   |
7         |       |       |
8         +-------x-------+   <-- (x,y)
9         |       |       |
10   iy+1 |  01   |  11   |
11        |       |       |
12        +-------+-------+
13  
14  Value at (x,y) = w_00 * v_00
15                 + w_10 * v_10
16                 + w_01 * v_01
17                 + w_11 * v_11

```

### 💼 一个 Cell 里的三层 Latent

结合前文的架构总览，在一个 BEV cell 中存放着三层不同阶段、不同含义的特征 Latent。

```
代码块
1              One BEV cell: cell(ix, iy) at ego_t
2   
3   [ Layer 1: Stage 1 Observation Latent ]
4     geometry_bev_seed[t, ix, iy]
5       : ray / depth / point / confidence evidence pooled into this cell
6 
7     cell's meaning: "此时此刻，这个 cell 内 看到 了什么"
8   ---------------------------------------------------------
9   [ Layer 2: Stage 2 Temporal fused Latent ]
10    geometry_world_feature[t, ix, iy]
11      : visibility-aware fusion over time
12    uncertainty_feature[t, ix, iy]
13      : + ego_align_query / dynamic_geometry_tokens as global kv
14 
15    cell's meaning: "结合历史上下文，这个 cell 内 确定存在 什么"
16  ---------------------------------------------------------
17  [ Layer 3: Future Geometry Latent ]
18    future_geometry_feature[t_k, ix, iy] (k = 1...K)
19 
20    cell's meaning: "未来第 k 帧，这个 cell 内 预计会有 什么存在"
21 
22  optional decoded probas:
23    future_occupancy_prob = decode_occ( future_geometry_feature )
24    flow/semantic/update_prob = decode_other( future_geometry_feature )
25 
26  These probas are for loss/debug/visualization, Plan uses latent vector directly.
```

核心总结/特点：

```
代码块
1   Planning reads an M*D vector,
2   not the BEV cell index or a pre-defined label directly.
3   not a raw LSS feature and not a decoded head output.
```

### 🔀 Geometry 如何进入 BEV

`GeometryFeaturePack`的物理核心不是BEV，而是camera ray上的metric surface evidence。stage 1把图像evidence lift成ego 3D点，再把这些点聚合到BEV cell

![image-20260623202807629](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260623204016703.png)

| 层 | Feature | 物理意义 | 是否 BEV |
| :--- | :--- | :--- | :--- |
| ① Image evidence | `image_pyramid` | 像素纹理、边缘、语义线索 | 否 |
| ② Ray evidence | `ray_tokens`, `ray_direction`, `ray_depth` | 哪条 ray、沿 ray 多远命中 surface | 否 |
| ③ Ego 3D evidence | `point_map_curr_ego`, `point_confidence` | 真实 $(x,y,z)$ surface 点和可信度 | 否 |
| ④ BEV interface | `geometry_bev_seed` | 把 ray/point evidence 聚合到 x-y 网格 | 是，接口层 |

> ❗ **重点：** `geometry_bev_seed` 是 Stage 1 geometry 的 BEV 接口，不是完整世界状态。

## 🔄 预训练阶段如何交互

下面这张图只保留主链路，避免把 head/probe 和主输入混在一起。

![image-20260623204010785](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260623204011063.png)

| 阶段 | 吃什么 | 吐什么 | 谁消费 |
| :--- | :--- | :--- | :--- |
| Stage 1 | 7V 图像、pose/calib、离线 dense/sparse depth label | `GeometryFeaturePack` | Stage 2、DepthHead、Planning |
| Stage 2-A | `GeometryFeaturePack_t-N..t`、ego motion | `TemporalGeometryWorldLatent` | Future rollout、Planning、可选 Occ/Object |
| Stage 2-B | `TemporalGeometryWorldLatent`、future query | `FutureGeometryLatent` | Planning、可选 FuturePredictionProbe |
| Planning | 三层 latent + candidate trajectories + ego/route | trajectory scores | 规划输出 |

## 🔍 Planning KV 查询

PlanningHead 的 query 是候选轨迹点，K/V 是三层 latent bank。

```python
# For each candidate trajectory n and future step k:
Q[n,k] = encode(x_k, y_k, t_k, ego_state, route_command)

# K/V bank A = Stage 1 geometry BEV cells
K_A = BEV position encoding cell(ix,iy)
V_A = geometry_bev_seed[:, ix, iy]

# K/V bank B = Stage 2 temporal BEV cells + memory tokens
K_B = BEV position encoding + memory position
V_B = geometry_world_feature / visibility / uncertainty / ego_aligned_memory

# K/V bank C = Future BEV cells at step k
K_C = BEV position encoding + time encoding k
V_C = future_geometry_feature[k, :, ix, iv]

trajectory_token[n,k] = fuse(
    query(Q[n,k], K_A, V_A),
    query(Q[n,k], K_B, V_B),
    query(Q[n,k], K_C, V_C)
)

score[n] = score_head(pool_k(trajectory_token[n, 1:K]))
```

接口定义：

```python
PlanningQueryPack = 
    GeometryFeaturePack
    + TemporalGeometryWorldLatent
    + FutureGeometryLatent
    + candidate_trajectories
    + ego_history / route_command
```

## 📋 Feature 字段表

| Feature | 物理空间 | 典型 shape | 物理意义 | 是否 BEV |
| :--- | :--- | :--- | :--- | :--- |
| `image_pyramid` | image plane | $B, T, V, C, H, W$ | 图像纹理、边缘、语义线索 | 否 |
| `ray_tokens` | camera ray | $B, T, V, H, D$ | 每条 ray 上的图像/几何 token | 否 |
| `ray_direction` | camera/ego ray | $B, T, V, H, W, 3$ | ray 在 ego 系的方向 | 否 |
| `ray_depth` | camera ray | $B, T, V, H, W$ | ray 命中 surface 的米制距离 | 否 |
| `point_map_curr_ego` | ego 3D | $B, T, V, H, W, 3$ | 每个像素 lift 到 ego_t 的 $(x,y,z)$ | 否 |
| `point_confidence` | ray/point evidence | $B, T, V, H, W$ | depth/point 可信度 | 否 |
| `geometry_bev_seed` | ego BEV | $B, T, C, Hb, Wb$ | Stage 1 geometry 的 BEV 接口 | 是 |
| `visibility_quality_feature` | ego BEV / derived | $B, T, Cv, Hb, Wb$ | observed / occluded / unknown / quality | 是 |
| `uncertainty_feature` | ego BEV | $B, T, Cu, Hb, Wb$ | 远距、遮挡、label noise 的不确定性 | 是 |
| `geometry_world_feature` | ego BEV + time | $B, T, C, Hb, Wb$ | 当前+历史融合后的 temporal world latent | 是，带 temporal |
| `ego_aligned_memory` | token bank | $B, M, D$ | 历史几何 warp 到当前 ego 后的 memory | 不是固定 BEV |
| `dynamic_geometry_tokens` | token bank | $B, Q, D$ | 动态几何簇 / motion latent，不是 OD box | 不是固定 BEV |
| `future_geometry_feature` | ego BEV + future time | $B, K, C, Hb, Wb$ | K-step future geometry latent | 是，带 future |
| `FuturePredictionProbe` | decoded BEV output | $B, K, *, Hb, Wb$ | future_occ / flow / risk，训练和 debug 用 | 是，probe |
| `trajectory_token` | trajectory query | $B, N, K, D$ | 候选轨迹点查 latent 后的聚合结果 | 否，轨迹条件化 |

## ⚠️ 混淆点

| 容易误解 | 更准确说法 |
| :--- | :--- |
| 最终不是 BEV | 最终工作空间是 BEV；但 BEV cell 里的 feature 来自预训练 latent。 |
| Geometry feature 就是 BEV feature | Geometry 首先是 ray-depth-point evidence；`geometry_bev_seed` 是它的 BEV 接口。 |
| Planning 读 BEV | Planning 读候选轨迹在三层 BEV latent 上的查询结果。 |
| Planning 读 Occ/OD/FutureHead 输出 | Planning 直接读 latent；这些 head 是可选 probe。 |
| Future latent 就是 future_occ | `FutureGeometryLatent` 是 hidden latent；`future_occ` 是可选解码。 |
| Dynamic token 就是 OD box | dynamic token 是运动几何 latent；OD box 是可选解码结果。 |
| unknown 是一个 head 输出 | unknown 首先是 visibility / observation state，head 只是显式 probe。 |
| 未来图像进入 encoder | 未来帧只能作为 label，encoder 输入严格 $\le t$ 。 |



![image-20260623204756043](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260623204756469.png)

![image-20260623204823826](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260623204824348.png)

![](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260623204824348.png)

![image-20260623212020218](C:/Users/gzh/AppData/Roaming/Typora/typora-user-images/image-20260623212020218.png)




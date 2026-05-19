# Map Pretrain 完整训练链路 Mermaid 流程图

> 本文件包含多个 Mermaid 流程图，可在支持 Mermaid 的 Markdown 渲染器中查看（如 GitHub, VS Code 等）。

---

## 1. 整体训练链路总览

```mermaid
flowchart TB
    subgraph Data["数据流水线 (Data Pipeline)"]
        direction TB
        DB[("LanceDB Storage<br/>pooh_occ table")] 
        
        subgraph Load["数据加载"]
            L1["LoadMultiViewFromPooh<br/>- JPEG decode images<br/>- Load point clouds<br/>- Load camera params"]
        end

        subgraph CamPre["相机预处理"]
            C1["UnifyVirtualCamera<br/>- 统一虚拟相机参数<br/>- warpAffine 投影"]
            C2["MultiViewImageTransform<br/>- Resize + Crop<br/>- Normalize (mean/std)"]
        end

        subgraph PtPre["点云预处理"]
            P1["PointRandomSample<br/>- 随机采样点云"]
            P2["LidarToEgo<br/>- LiDAR→自车坐标系"]
            P3["GlobalBEVTransform<br/>- BEV 增强 (rot/scale/flip)<br/>- 生成 bda_mat"]
            P4["PointToMultiViewDepth<br/>- 点云→多视角深度图"]
        end

        subgraph Render["渲染 GT 准备"]
            R1["RenderPrepare<br/>- JPEG decode render images<br/>- Undistort<br/>- Select up to 4 cams"]
            R2["UnifyVirtualRenderCamera<br/>+ MultiViewImageTransform<br/>(with render_ prefix)"]
        end

        subgraph Coll["批处理打包"]
            CL1["Collect<br/>batch_type=MapBEVPretrainTrainBatch"]
            CL2["collate_fn<br/>stack/batch 组装"]
        end

        DB --> L1
        L1 --> C1
        C1 --> C2
        L1 --> P1
        P1 --> P2
        P2 --> P3
        P3 --> P4
        C2 --> CL1
        P4 --> CL1
        L1 --> R1
        R1 --> R2
        R2 --> CL1
        CL1 --> CL2
    end

    subgraph Model["模型 (EggyRoadBEVPretrain)"]
        direction TB
        
        subgraph RouteB["Route B: Camera Branch"]
            B1["Image Backbone<br/>HENetBackbone<br/>stride 4, 8, 16"]
            B2["CustomFPN<br/>out_ids=[2] @stride 8<br/>→ 256 channels"]
            B3["DepthNet (in LSS)<br/>softmax depth<br/>D=406 bins"]
            B4["LSS View Transform<br/>bev_pool_v2 CUDA<br/>→ BEV feature"]
        end

        subgraph RouteA["Route A: LiDAR Branch"]
            A1["SPConvVoxelization<br/>[N,4]→[M,12,4]"]
            A2["HardSimpleVFE<br/>[M,12,4]→[M,4]"]
            A3["SparseEncoder<br/>spconv 3D Conv<br/>→ [B,192,3,H,W]"]
        end

        subgraph FusionBEV["Fusion & BEV Encoder"]
            F1["BEVFusionConvFuser<br/>Concat + Conv2d 3×3<br/>448→384 ch"]
            F2["SECOND2D<br/>BEV Backbone<br/>3 stages"]
            F3["SECONDFPN2D<br/>BEV Neck + CBAM<br/>Upsample → 576ch"]
        end

        subgraph Head["Gaussian Head & Render"]
            H1["gs_attr_proj<br/>Conv2d 576→256→122"]
            H2["GaussianHead<br/>split+activate 122 dims<br/>→ Gaussian attributes"]
            H3["gsplat rendering<br/>diff. rasterization<br/>→ images, depth, feats"]
        end

        RouteB --> FusionBEV
        RouteA --> FusionBEV
        FusionBEV --> Head
    end

    subgraph Loss["损失计算 (DefaultMapPretrain)"]
        direction TB
        LS1["GaussianLoss<br/>- L1 RGB Loss (w=0.0)<br/>- LPIPS Loss (w=1.0)<br/>- Feature Cosine Loss (w=1.0)"]
        LS2["Depth Loss (aux)<br/>- CrossEntropy<br/>- ExpectedDepthL1<br/>weight=3.0"]
        LS3["Total Loss<br/>加权求和"]
    end

    Data -->|"MapBEVPretrainTrainBatch"| Model
    Model --> Head
    Head -->|"rendered_results + gt_feats"| Loss
```

---

## 2. Camera Route B 详细流程图

```mermaid
flowchart TB
    subgraph CamBranch["Camera Branch (Route B) 详细张量流"]
        direction TB
        
        IN["imgs<br/>[B, N_cam, 3, 480, 960]"]
        
        subgraph Backbone["Image Backbone (HENetBackbone)"]
            direction LR
            B_S["Stem<br/>stride 4"] --> B0["Stage 0<br/>64ch, s4"]
            B0 --> B1["Stage 1<br/>64ch, s4"]
            B1 --> B2["Stage 2<br/>128ch, s8"]
            B2 --> B3["Stage 3<br/>192ch, s16"]
        end
        
        subgraph FPN["CustomFPN (Neck)"]
            direction TB
            FPN1["1×1 Conv 投影到 256ch<br/>[192→256, 128→256, 64→256]"]
            FPN2["Top-down 上采样相加"]
            FPN3["3×3 Conv (out_ids=[2])<br/>→ [B*N, 256, 60, 120]"]
        end

        subgraph LSS["EggyLSSViewTransformer"]
            direction TB
            L0["(可选) LiDAR Depth<br/>→ lidar_input_net<br/>→ [B*N, 64, 60, 120]"]
            L1["DepthNet<br/>Conv2d × 3<br/>[B*N, 320, 60, 120]<br/>→ [B*N, 470, 60, 120]"]
            L1 --> L1A["split @dim=1"]
            L1A --> L1B["depth: [B*N, 406, 60, 120]<br/>features: [B*N, 64, 60, 120]"]
            L1B --> L2["softmax(depth, dim=1)<br/>depth distribution"]
            L2 --> L3["get_lidar_coor<br/>frustum → BEV coords<br/>[B, N, 406, 60, 120, 3]"]
            L3 --> L4["voxel_pooling_prepare_v2<br/>计算 ranks + intervals"]
            L4 --> L5["bev_pool_v2 CUDA<br/>scatter-add pooling"]
            L5 --> L6["post_net (Identity)<br/>permute to CHW"]
        end

        IN -->|"flatten(0,1)<br/>[B*N, 3, 480, 960]"| Backbone
        Backbone -->|"List[[B*N,64,120,240],<br/>[B*N,128,60,120],<br/>[B*N,192,30,60]]"| FPN
        FPN -->|"[B*N, 256, 60, 120]"| LSS
        LSS -->|"cam_bev_feat<br/>[B, 64, 384, 448]"| OUT
    end

    subgraph Config["关键配置"]
        CFG1["backbone_type: henet<br/>out_indices: [1,2,3]<br/>neck_out_ids: [2]<br/>downsample: 8"]
        CFG2["depth bins: D=406<br/>range: [1.0, 204.0], step=0.5"]
        CFG3["BEV grid: x=[-25.6,153.6]/0.4=448<br/>y=[-38.4,38.4]/0.2=384<br/>z=[-3.0,5.0]/8.0=1"]
        CFG4["out_channels: 64<br/>lidar_as_input: true"]
    end

    OUT["[B, 64, 384, 448]"]
```

---

## 3. LiDAR Route A 详细流程图

```mermaid
flowchart TB
    subgraph LidarBranch["LiDAR Branch (Route A) 详细张量流"]
        direction TB
        
        IN["points<br/>[N_i, 4+C]"]
        
        VOX["SPConvVoxelization<br/>spconv PointToVoxel<br/>voxel_size=[0.4, 0.2, 0.5]<br/>point_cloud_range=[-25.6,-38.4,-3.0, 153.6,38.4,5.0]"]
        
        VFE["HardSimpleVFE<br/>max_num_points=12<br/>mean pooling per voxel<br/>[M, 12, 4] → [M, 4]"]
        
        subgraph SE["SparseEncoder (spconv 3D)"]
            direction TB
            SE0["conv_input<br/>SubMConv3d 3×3×3<br/>4→24ch"]
            SE1["block1<br/>BasicBlock + SparseConv<br/>24→48ch, stride 2<br/>[41,3072,3584]→[21,1536,1792]"]
            SE2["block2<br/>BasicBlock + SparseConv<br/>48→96ch, stride 2<br/>[21,1536,1792]→[11,768,896]"]
            SE3["block3<br/>BasicBlock + SparseConv<br/>96→192ch, stride 2<br/>[11,768,896]→[6,384,448]"]
            SE4["block4<br/>BasicBlock ×2<br/>192→192ch<br/>[6,384,448]"]
            SE5["conv_out<br/>SparseConv3d (3,1,1)<br/>stride (2,1,1)<br/>192→192ch<br/>[6,384,448]→[3,384,448]"]
        end
        
        DENSE["to_dense()<br/>sparse → dense<br/>[B, 192, 3, 384, 448]"]
        
        FLATTEN["permute + reshape<br/>合并 z 和 channel 维度<br/>[B, 192×3=576, 384, 448]"]

        IN --> VOX
        VOX -->|"voxels[M,12,4],<br/>coords[M,3]"| VFE
        VFE -->|"[M,4]"| SE
        SE0 --> SE1 --> SE2 --> SE3 --> SE4 --> SE5
        SE5 -->|"sparse tensor"| DENSE
        DENSE --> FLATTEN
    end

    subgraph Config["关键配置"]
        CFG1["voxel_size: [0.4, 0.2, 0.5]"]
        CFG2["sparse_shape: [41, 3072, 3584]<br/>(z, y, x)"]
        CFG3["base_channels: 24<br/>output_channels: 192<br/>block_type: basicblock"]
        CFG4["spconv spatial shape:<br/>3072/8=384, 3584/8=448"]
        CFG5["在 fuser 中 z:3 × ch:192 = 576"]
    end

    FLATTEN -->|"pts_bev_feat<br/>[B, 384, 384, 448]"| FUSE["→ BEVFusionConvFuser"]
```

---

## 4. Fusion, BEV Encoder & Gaussian Head 流程图

```mermaid
flowchart TB
    subgraph FusionEnc["Fusion & BEV Encoder"]
        direction TB
        C["cam_bev_feat<br/>[B, 64, 384, 448]"]
        L["pts_bev_feat<br/>[B, 384, 384, 448]"]
        
        C --> CONCAT
        L --> CONCAT
        
        CONCAT["Concat @dim=1<br/>[B, 448, 384, 448]"]
        
        FUSER["BEVFusionConvFuser<br/>Conv2d(448,384,3,p=1)+BN+ReLU<br/>[B, 384, 384, 448]"]
        
        subgraph SECOND["SECOND2D Backbone"]
            S1["Block0: Conv2d 384→192<br/>kernel=7, stride=1<br/>6 layers<br/>→ [B,192,384,448]"]
            S2["Block1: Conv2d 192→384<br/>kernel=7, stride=2<br/>10 layers<br/>→ [B,384,192,224]"]
            S3["Block2: Conv2d 384→384<br/>kernel=7, stride=2<br/>10 layers<br/>→ [B,384,96,112]"]
        end
        
        subgraph FPN2D["SECONDFPN2D Neck"]
            F1["deblock0: Identity Conv<br/>192→192ch, ×1<br/>+ CBAM"]
            F2["deblock1: ConvTranspose2d<br/>384→192ch, ×2<br/>+ CBAM"]
            F3["deblock2: ConvTranspose2d<br/>384→192ch, ×4<br/>+ CBAM"]
            CAT3["Concat @dim=1<br/>192×3=576ch<br/>→ [B, 576, 384, 448]"]
        end

        CONCAT --> FUSER
        FUSER --> S1
        S1 --> S2
        S2 --> S3
        S1 --> F1
        S2 --> F2
        S3 --> F3
        F1 --> CAT3
        F2 --> CAT3
        F3 --> CAT3
    end

    subgraph GHead["Gaussian Head"]
        direction TB
        
        FEAT["dense_task_feat<br/>[B, 576, 384, 448]"]
        
        PROJ["gs_attr_proj<br/>Conv2d(576,256,3,p=1)+ReLU<br/>Conv2d(256,122,1)<br/>→ [B, 122, 384, 448]"]
        
        BEV["bev_points<br/>grid 中心点<br/>[384, 448, 2]"]
        
        GH["GaussianHead.forward()<br/>1. flatten: [B,1,122,384,448]<br/>   → [B*384*448, 122]<br/>2. split → 6/7 个部分<br/>3. 激活函数:<br/>   - scales = sigmoid × dist<br/>   - opacities = sigmoid<br/>   - colors = tanh + RGB2SH<br/>   - z = tanh × 2.0<br/>4. assemble → dict"]
        
        SPLIT1["scales: [B*HW, 2]<br/>→ sigmoid × NN_dist<br/>→ + z_scale → [B*HW, 3]"]
        SPLIT2["opacities: [B*HW, 1]<br/>→ sigmoid"]
        SPLIT3["sh_coeffs: [B*HW, 48]<br/>= [B*HW, 16, 3]<br/>× sh_mask"]
        SPLIT4["colors: [B*HW, 3]<br/>→ tanh → RGB2SH<br/>→ + sh_dc (rgb residual)"]
        SPLIT5["gs_feats: [B*HW, 64]"]
        SPLIT6["means_z: [B*HW, 1]<br/>→ tanh × 2.0"]
        MEANS["means_xy: [B, 384, 448, 1, 2]<br/>→ concat with means_z<br/>→ [B, 172032, 3]"]

        FEAT --> PROJ
        PROJ --> GH
        BEV --> GH
        GH --> SPLIT1 & SPLIT2 & SPLIT3 & SPLIT4 & SPLIT5 & SPLIT6
        SPLIT6 --> MEANS
        
        GH --> REARR["rearrange views<br/>→ List[dict] per view"]
    end

    subgraph Render["gsplat Rendering"]
        RD["gs_render()<br/>1. merge_and_split: 合并 view<br/>2. GaussianModel.from_predictions<br/>3. render_bev_view (BEV 俯视图)<br/>4. for each render camera:<br/>   render(camera, gaussians, pipeline)<br/>5. stack results"]
        
        IMG_OUT["rendered_images<br/>[B*N_render, 3, H, W]"]
        DEPTH_OUT["rendered_depths<br/>[B*N_render, 1, H, W]"]
        FEAT_OUT["rendered_feats<br/>[B*N_render, 96, H/8, W/8]"]
        BEV_OUT["rendered_bevs<br/>[B*N_cam, H_bev, W_bev, 3]"]
    end

    CAT3 -->|"dense_task_feat<br/>[B, 576, 384, 448]"| PROJ
    REARR --> RD
    RD --> IMG_OUT & DEPTH_OUT & FEAT_OUT & BEV_OUT
```

---

## 5. 损失计算流程图

```mermaid
flowchart TB
    subgraph LossFlow["损失计算流程 (DefaultMapPretrain.forward)"]
        direction TB
        
        subgraph Inputs["输入"]
            I1["preds (from backbone)<br/>- rendered_images<br/>- rendered_feats<br/>- rendered_depths"]
            I2["gt_imgs: render_imgs<br/>[B, N_render, 3, H, W]"]
            I3["gt_feats (from backbone)<br/>extract_render_img_features"]
            I4["gt_depth (aux)<br/>[B, N_cam, H, W]"]
        end

        subgraph LossFunc["GaussianLoss"]
            L1["gt_imgs denormalize<br/>× std + mean → /255 → [0,1]"]
            L2["L1 RGB Loss<br/>|render - denorm_gt| × mask<br/>rgb_weight=0.0 (disabled)"]
            L3["LPIPS Loss<br/>LPIPS(render, denorm_gt)<br/>lpips_net=alex<br/>lpips_weight=1.0"]
            L4["Feature Cosine Loss<br/>1 - cos_sim(rendered_feats, gt_feats)<br/>feat_weight=1.0"]
        end

        subgraph AuxLoss["Auxiliary Loss (from LSS)"]
            A1["depth_logits: [B*N, 406, 60, 120]"]
            A2["gt_depth: [B, N, H_gt, W_gt]"]
            A3["MinPool downsampling<br/>→ [B*N, H_f, W_f]"]
            A4["CrossEntropy Loss<br/>(深度 bin 分类)"]
            A5["ExpectedDepth L1<br/>连续深度回归<br/>weight=0.05"]
            A6["Distance-aware Weight<br/>1 + d/100"]
        end

        I1 --> L2
        I2 --> L1 --> L2
        I1 --> L3
        I2 --> L1 --> L3
        I1 --> L4
        I3 --> L4
        
        L2 & L3 & L4 --> L_SUM["loss_dict['loss']<br/>加权求和"]
        
        I4 --> A2
        A2 --> A3 --> A4
        A4 --> A5 --> A6
        A6 --> A_TOTAL["aux_losses['loss_depth']<br/>× loss_depth_weight=3.0"]
        A_TOTAL --> L_TOTAL["total_loss = loss + aux_losses"]
        L_SUM --> L_TOTAL
    end

    subgraph Config["损失权重"]
        W["rgb_weight: 0.0<br/>lpips_weight: 1.0<br/>feat_weight: 1.0<br/>depth_weight: 0.2 (未用)<br/>loss_depth_weight: 3.0<br/>expected_depth_l1_weight: 0.05"]
    end
```

---

## 6. 训练循环时序图

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as DefaultMapPretrain
    participant B as EggyRoadBEVPretrain<br/>(Backbone)
    participant C as GaussianLoss<br/>(Criteria)
    # 将别名从 Opt 改为 O，防止和关键字 opt 冲突
    participant O as Optimizer

    loop over epochs
        rect rgb(240, 240, 240)
            Note over D,O: [Loop] over batches
            D->>M: MapBEVPretrainTrainBatch
            
            Note over M: forward()
            M->>B: forward(batch)
            
            Note over B: Route A (LiDAR)
            Note over B: Route B (Camera)
            Note over B: Fusion + BEV Encoder
            Note over B: Gaussian Head
            Note over B: gsplat Render
            
            B-->>M: rendered_results + gt_feats
            
            M->>C: criteria(preds, gt_imgs, gt_features, gt_mask)
            Note over C: L1 + LPIPS + Feature Cosine Loss
            C-->>M: loss_dict
            
            Note over M: Merge aux_losses (depth)
            Note over M: loss = sum(all losses)
            
            M-->>D: {"loss": loss, "loss_dict": ..., "preds_dict": ...}
            
            O->>M: loss.backward()
            O->>O: clip_grad_norm(25.0)
            O->>O: optimizer.step()
            O->>O: scheduler.step()
        end
    end

```
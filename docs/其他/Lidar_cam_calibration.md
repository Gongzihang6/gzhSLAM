# Lidar和Camera外参标定算法研究

对于无目标的lidar和cam的外参标定，主要有几何显式法、时空联合/SLAM法、对象/语义对应法、统一深度表示法、

1.  几何显式法：先找 ground plane / edge / line，再优化投影一致性。代表是 Galibr](https://arxiv.org/abs/2406.11599)（2024-06）和 [MFCalib](https://arxiv.org/abs/2409.00992)（IROS 2024）。Galibr 先用地面做初始化，再用边缘做精配准，优点是工程上直接、可解释，适合车载和有明显地面的移动机器人；缺点是对“地面可见且稳定”有依赖。MFCalib 更强，做的是单次采集、无板标定，核心是同时利用 深度连续边、深度不连续边、平面上的强度边，并显式建模 LiDAR 光束导致的边缘膨胀，适合“停车采一帧/几帧就标”。
2.  时空联合/SLAM式方法：不只看单帧，而是把多帧轨迹一起优化。代表是 [SOAC](https://openaccess.thecvf.com/content/CVPR2024/papers/Herau_SOAC_Spatio-Temporal_Overlap-Aware_Multi-Sensor_Calibration_using_Neural_Radiance_Fields_CVPR_2024_paper.pdf)（CVPR 2024）和 Continuous-Time Estimation](https://arxiv.org/abs/2501.02821)（2025-01，preprint）。SOAC 用 NeRF 做公共场景表示，同时估空间外参与时间偏移，适合“已经有 IMU/SLAM/SfM 参考轨迹”的系统；优点是对多传感器、多时序统一处理很自然，缺点是算力重，而且需要参考传感器轨迹。Continuous-Time 那篇更偏系统工程：相机侧先做 SFM，自标定内参；LiDAR 侧建 voxel map；最后再做联合 BA。它的亮点是“可以不要求相机和 LiDAR 直接 FoV 重叠”，对多相机多雷达 rig 很有价值。
3.  对象/语义对应法：不直接对齐点和像素，而是先找“车、人、杆、结构”等共同目标。代表是[EdO-LCEC](https://arxiv.org/abs/2502.00801)（2025-02，preprint）和[CalibRefine](https://arxiv.org/abs/2502.17648)（2025-02，preprint）。EdO-LCEC 的思路是先判断当前环境特征密度，再自适应选强度/深度特征，并用结构+纹理双路径去做 3D-2D correspondence；CalibRefine 则更明确地走对象级路线：用自动检测目标、相对位置、外观 embedding、语义类别形成匹配，再做粗标定、迭代细化和 attention refinement。它们对真实城市场景很贴合，但通常更依赖检测器/训练分布。
4.  统一深度表示法：把跨模态问题改写成“深度对深度”或“BEV 对 BEV”的同模态对齐。代表是 [DF-Calib / UniCalib](https://openaccess.thecvf.com/content/WACV2026/html/Han_UniCalib_Targetless_LiDAR-camera_Calibration_via_Probabilistic_Flow_on_Unified_Depth_WACV_2026_paper.html)（arXiv 2025-04，WACV 2026 正式版改名 UniCalib）和 [CalibBEV](https://openaccess.thecvf.com/content/WACV2026/html/DAddeo_CalibBEV_LiDAR-Camera_Calibration_via_BEV_Alignment_WACV_2026_paper.html)（WACV 2026）。UniCalib 先把相机图像估成稠密深度，把稀疏 LiDAR 深度补全成统一深度，再学一个带不确定性的 flow 和 reliability map 去抑制动态物体、遮挡、坏匹配。CalibBEV 则把两模态都拉到共享 BEV 空间，先粗回归再显式对齐。对自动驾驶特别自然，因为道路场景本来就适合 BEV 表达。
5.  隐式场/3D Gaussian 联合优化：把外参和场景几何一起估。代表是 [Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting](https://arxiv.org/abs/2504.04597)（2025-04，RA-L 2026）。这类方法的核心不是“找几个边缘点”，而是直接优化“在当前外参下，LiDAR 和图像能不能共同解释同一个 3D 场景”。这篇用锚定 LiDAR 点作为 anchor Gaussians，再联合优化位姿和高斯参数。优点是对复杂自然场景适应性强，和 SLAM/重建/具身场景很契合；缺点是训练/优化代价高，不像几何法那样轻量。

| 论文                                                         | 时间/类型                     | 核心思路                                                     | 输入/前提                                  | 是否依赖初值                     | 应用场景                 | 优点                                                 | 局限                                             |
| :----------------------------------------------------------- | :---------------------------- | :----------------------------------------------------------- | :----------------------------------------- | :------------------------------- | :----------------------- | :--------------------------------------------------- | :----------------------------------------------- |
| Galibr: Targetless LiDAR-Camera Extrinsic Calibration Method via Ground Plane Initialization | 2024-06, arXiv                | 先用 ground plane 自动初始化，再用 edge alignment 做精配准   | 车载自然场景；需要能稳定看到地面和结构边缘 | 不需人工初值，但强依赖地面初始化 | 自动驾驶 / UGV           | 工程上直接，解释性强，适合道路场景                   | 地面弱、坡道大、越野/室内杂乱时会变弱            |
| Single-shot and Automatic Extrinsic Calibration for LiDAR and Camera in Targetless Environments Based on Multi-Feature Edge | 2024-09, IROS 2024 / arXiv    | 单次采集；联合 深度连续边 + 深度不连续边 + 平面强度边，并补偿 LiDAR beam divergence 带来的边界膨胀 | 单帧或少帧自然场景；边缘足够丰富           | 基本不需要初值                   | 自动驾驶 / 机器人        | 很贴近“停车采一组数据就标定”；对无板离线标定很实用   | 本质还是边缘法，低纹理/弱结构场景会掉性能        |
| SOAC: Spatio-Temporal Overlap-Aware Multi-Sensor Calibration using Neural Radiance Fields] | CVPR 2024                     | 用 NeRF 作为公共场景表示，联合估计 空间外参 + 时间偏移，并显式建模传感器时空重叠 | 多帧序列；通常需要参考轨迹/同步信息较好    | 弱初值，更依赖时序与轨迹质量     | SLAM / 多传感器系统      | 能同时处理空间和时间标定，适合系统级校准             | 算力重，部署复杂，不是最轻量的车端方案           |
| Targetless Intrinsics and Extrinsic Calibration of Multiple LiDARs and Cameras with IMU using Continuous-Time Estimation | 2025-01, arXiv                | 相机侧 SfM，LiDAR 侧建图，再用 continuous-time batch estimation 联合优化内外参与时序；可处理 非重叠 FoV | 多相机/多 LiDAR + IMU，多帧时序            | 通常需要弱初值，但不要求精确     | SLAM / 具身 / 复杂 rig   | 很适合多传感器平台，理论完整，和后端优化兼容         | 系统要求高；实现复杂；更像“全系统标定框架”       |
| CalibRefine: Deep Learning-Based Online Automatic Targetless LiDAR-Camera Calibration with Iterative and Attention-Driven Post-Refinement | 2025-02, arXiv                | 先做 粗标定，再基于 对象级匹配 + relative pose + appearance embedding + semantic class 迭代细化，并用 attention 做后处理 | 自然道路场景；依赖检测/语义先验            | 可弱化初值要求                   | 自动驾驶 / 在线重标定    | 对动态城市场景友好，适合在线漂移修正                 | 依赖训练分布与检测器质量，跨域风险比几何法大     |
| UniCalib: Targetless LiDAR-camera Calibration via Probabilistic Flow on Unified Depth Representations | WACV 2026，论文页近两年内公开 | 先把两模态变到 统一深度表示：图像估深度，LiDAR 补全深度，再学 probabilistic flow + reliability map 做对齐 | 多帧自然场景；不依赖标定板                 | 弱初值，对较大失配更鲁棒         | 自动驾驶                 | 是最近很强的一条线，天然适合道路场景和跨模态深度对齐 | 依赖深度估计/补全质量；训练和泛化是关键          |
| Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting | 2025-04, arXiv                | 用 anchored 3D Gaussians 联合优化 场景几何 + LiDAR-Camera 外参，不再只对边缘，而是让两模态共同解释同一场景 | 多帧自然场景；轨迹稳定性更重要             | 通常更需要粗初值或稳定多帧       | SLAM / 具身 / 场景重建   | 和 3DGS / NeRF / 具身场景建图路线高度一致            | 优化重，工程复杂，不是最快上线的标定方案         |
| CalibBEV: LiDAR-Camera Calibration via BEV Alignment         | WACV 2026                     | 把 LiDAR 和图像都拉到 共享 BEV 空间，先粗回归再显式对齐      | 车载道路场景，BEV 表达明显受益             | 弱初值                           | 自动驾驶                 | 很符合自动驾驶表征，和 BEV 感知链路一致              | 对 BEV 语义/结构可见性有依赖，非道路泛化可能一般 |
| Robust LiDAR-Camera Calibration with 2D Gaussian Splatting   | 2025-04, RA-L 2025 / arXiv    | 先用 LiDAR 点云重建 colorless 2DGS，再通过 photometric + reprojection + triangulation 联合优化外参 | 多帧图像序列 + LiDAR                       | 弱初值                           | SLAM / 机器人 / 自动驾驶 | 兼顾几何和光度，比单纯 photometric loss 更稳         | 对序列质量和视角覆盖有要求                       |



相关开源代码：

| 优先级 | 工具                                                         | 为什么适合你们的数据                                         |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1      | [direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration) | 最适合先跑 baseline。它是 targetless、single-shot、多帧都支持，不需要标定板和初值，支持 pinhole/fisheye/omnidirectional camera，也支持 spinning / non-repetitive LiDAR。你们有 JPG + undistorted point cloud，转换成它需要的 image-cloud pair 或 ROS bag 后就能试。 |
| 2      | [OpenCalib / SensorsCalibration](https://github.com/PJLab-ADG/SensorsCalibration) | 更贴近这种自动驾驶数据组织。它本身就是 multi-sensor calibration toolbox，包含 lidar2camera、camera_intrinsic、SensorX2car 等模块。你们目录里已经有 calibration、PCO_Calibration，很可能能较顺地接入。 |
| 3      | [CalibAnything](https://github.com/OpenCalib/CalibAnything)  | 你们已经有 *_mask，这点很关键。CalibAnything 本来就利用 SAM/mask 辅助 LiDAR-camera 标定；如果你们的 mask 是车道线、路沿、动态物体过滤或语义区域，它比普通 edge-based 方法更有潜力。缺点是通常需要一个还不错的初值。 |
| 4      | [MFCalib](https://github.com/Es1erda/MFCalib)                | 适合你们这种 road scene + enhanced image + mask 的数据。它利用多种边缘/结构特征做 single-shot targetless 标定，适合拿 JPG_*_enhanced 或 mask 后图像试。工程成熟度不如前两个，但可以作为验证方案。 |
| 5      | [TIER IV CalibrationTools](https://github.com/tier4/CalibrationTools) | 如果你们后面要接 Autoware/ROS2 体系，它值得用。它更像车端工程标定流程，适合多相机、多 LiDAR、已有车辆坐标系的场景。 |
| 6      | [hku-mars/joint-lidar-camera-calib](https://github.com/hku-mars/joint-lidar-camera-calib) | 如果你们怀疑相机内参也有问题，可以考虑。它能联合优化 camera intrinsic 和 LiDAR-camera extrinsic，但对数据准备、初值、场景结构要求更高，不是第一优先级。 |

## Galibr

Galibr: Targetless LiDAR-Camera Extrinsic Calibration Method  via Ground Plane Initialization。**该标定方法需要的数据是一段连续时间序列的激光雷达点云和相机图像数据。**需要连续数据的原因在于：该方法需要利用相机的运动恢复结构（SfM）和激光雷达里程计（Odometry）来估算传感器自身的运动轨迹（ego-motion），并且需要在一个滑动时间窗口（sliding window）内连续观测，以确保传感器的运动方向与提取到的地平面保持对齐。

![image-20260529145244324](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260529145244531.png)

Galibr 的完整标定流程分为两个核心阶段：基于地平面的初始位姿估计（GP-init）和基于边缘匹配的外参优化细调。以下是详细的完整流程：

**<span style="color:#d59bf6;">阶段一：基于地平面的初始位姿估计 (GP-init)</span>**

由于传统无目标标定方法往往需要一个较好的初始外参猜测值，Galibr 利用地平面的稳定性，分别计算相机和激光雷达相对于地面的位姿，从而桥接出两者之间的初始相对位姿。

1、相机端（Camera）的相对位姿估计：

运动与特征提取：利用运动恢复结构（SfM）技术提取 3D 地面特征，并借此估算相机的自我运动方向向量。

地平面拟合：使用 RANSAC 算法对 SfM 提取出的 3D 地面特征进行平面拟合，求出相机坐标系下的地平面法向量。

平面验证：在一个时间滑动窗口内，计算相机运动向量与地平面法向量的点积。如果点积低于设定的阈值，则证明相机的运动与一个平坦稳定的地平面高度对齐。

位姿计算：经过验证后，计算出从地面到相机的相对变换矩阵（包含旋转矩阵和平移向量）。

激光雷达端（LiDAR）的相对位姿估计：

里程计估计：使用带有匀速模型的迭代误差状态卡尔曼滤波（IESKF）来估算激光雷达的里程计信息和运动方向向量。

点云分割：利用 TRAVEL 算法（一种先进的地面/非地面分割方法）将激光雷达点云严格划分为“地面点”和“非地面物体点”。

地平面拟合与验证：同样使用 RANSAC 算法对分离出的地面点云进行平面拟合。并在滑动窗口内验证 LiDAR 的运动方向向量与地面法向量的点积是否低于阈值，以确保对齐。

位姿计算：得出从地面到激光雷达的相对变换矩阵。

3、计算初始外参（Initial Guess）：

获得了“地面到相机”以及“地面到激光雷达”的相对位姿后，通过矩阵求逆和相乘（即雷达先转换到地面，地面再转换到相机），直接推导出激光雷达相对于相机的初始相对位姿变换矩阵。

**<span style="color:#d59bf6;">阶段二：基于边缘匹配的外参细调 (Extrinsic Calibration)</span>**

有了初始外参后，Galibr 利用环境中提取的自然边缘（Edge）进行特征对齐和优化，完成高精度标定。

1、多模态边缘特征提取：

图像端：采用目前最快的边缘提取算法 ELSED，从相机图像中检测出 2D 图像边缘。

激光雷达端：利用前面 TRAVEL 算法分割出的“非地面点云”作为 LiDAR 的 3D 物体边缘特征。

2、初始重投影与视角误差消除：

由于在 LiDAR 视角下检测到的 3D 边缘，并不总是能与相机视角下的真实 2D 边缘完美对应（因为视角不同）。为了解决这个偏差，Galibr 利用第一阶段算出的初始位姿，将 LiDAR 提取的非地面物体直接投影到相机的 2D 图像平面上。这一步使得跨模态的边缘对比在同一个视角下进行，大大提升了特征的保真度。

3、非线性优化 (PnP 求解)：

将标定问题转化为一个 Perspective-n-Point (PnP) 优化问题。

优化目标：最小化 3D 激光雷达遮挡边缘点重投影到相机 2D 边缘线上的垂直距离误差（Reprojection error）。

鲁棒性处理：在优化函数中引入了 Huber 鲁棒核函数（Huber kernel），以降低误匹配点或异常值对结果的影响。

迭代求解：使用 Levenberg-Marquardt (LM) 算法对目标函数进行迭代优化，直到满足收敛条件。最终输出的矩阵即为高精度的 LiDAR-Camera 外参标定结果

---

## MFCalib

使用 MFCalib 方法进行激光雷达（LiDAR）和相机的外参标定，最大的优势在于它**只需要“单次数据采集”（single-shot）**即可在无目标的自然环境中自动完成。

所需的数据:**单次采集的同步数据**：

* **相机数据**：单张 RGB 图像。
* **高密度激光雷达点云**：根据雷达类型的不同，获取方式有所区别。对于非重复扫描雷达（如 Livox Avia），采用静态累积的方式来收集稠密点云；对于旋转式扫描雷达（如 Ouster），则需要应用激光雷达惯性里程计（LiDAR-inertial odometry）系统来累积点云。

* **先验参数**：需要提前标定好的相机内参，以及基于 CAD 模型推导出的雷达与相机间的粗略外参初始值。

![image-20260529134703126](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260529134741586.png)

完整标定流程：MFCalib 的核心思想是<span style="color:#d59bf6;">充分利用自然场景中丰富的多特征边缘信息，并通过考虑激光雷达的物理光束模型来消除误差。</span>其完整流程详细分为以下几个步骤：

1. 多特征边缘提取 (Multi-Feature Edge Extraction)

为了确保在单一场景中提取尽可能多的边缘特征并提高鲁棒性，算法会并行提取三种类型的边缘：

* **图像端边缘**：直接在相机图像上应用 Canny 边缘检测算法提取 2D 图像边缘。
* **雷达端 - 深度连续边缘 (Depth-continuous edges)**：基于构建的体素（voxel）从点云中提取。
* **雷达端 - 深度不连续与强度不连续边缘 (Depth/Intensity-discontinuous edges)**：
  * 首先将点云数据映射到球面坐标系，随后转换为笛卡尔坐标系生成 2D 强度图像。在此图像中，像素由 3D 空间点的平均强度着色，并保留深度信息。
  * 对生成的强度图应用 Canny 算法提取边缘。然后使用基于 KD-Tree 的过滤机制，根据局部性和深度变化，将这些边缘区分为“深度不连续边缘”和“平面上的强度不连续边缘”。
  * 提取完成后，这两种边缘会被重新映射回原始的三维激光雷达点云中。
* **边缘融合**：将上述从雷达中提取的所有 3D 边缘在 LiDAR 坐标系下进行合并。这样做可以有效避免因遮挡导致的多值或零值映射问题。

2、激光雷达光束建模与误差补偿 (Beam Model & Uncertainty)

在无目标标定中，提取“深度不连续边缘”通常是不准确的（会产生膨胀或流血点）。为了解决这个问题，MFCalib 深入分析了激光雷达的物理测量原理：

* 算法针对激光雷达的光束发散角（beam divergence angle）进行了建模。
* 它为深度不连续边缘引入了测量不确定性。当光束中心线或整体光束与前景物体边缘相切时，会根据深度和发散角计算出相应的误差和高斯噪声分布，从而极大地缓解了由激光束引起的边缘膨胀问题。

3、迭代优化与配准 (Iterative Optimization)

在这个阶段，算法在相机的 2D 图像空间内进行匹配和误差最小化：

* **特征投影**：利用当前的外参矩阵，将 LiDAR 坐标系下的 3D 边缘点投影到相机的 2D 图像平面上。
* **残差构建**：将投影后的雷达边缘点与图像中提取的 2D 边缘（由法向量及线上的点定义）进行比对。同时，将上一步计算出的 LiDAR 测量噪声（光束误差）以及相机的测量噪声纳入考量，构建最优观测模型和残差方程。
* **高斯-牛顿优化**：通过聚合所有的边缘对应关系，公式化为一个最大似然问题。算法采用高斯-牛顿法（Gauss-Newton method），并借助 Ceres Solver 求解器进行多次迭代优化，不断更新和微调外参。

最终，当优化收敛时，系统就会输出高精度的六自由度（6-DoF，包含旋转和平移）外参标定结果。

---

`OpenCalib` 和 `direct_visual_lidar_calibration` 是两个目前非常流行且功能强大的多传感器标定开源工具箱。虽然它们都可以实现 LiDAR 和相机的外参标定，但两者的侧重点、标定原理、流程以及对数据的要求有显著的不同。

以下是对这两个工具箱在 LiDAR-Camera 外参标定上的详细对比：

## OpenCalib 标定工具箱

OpenCalib 是一个面向自动驾驶的综合性多传感器标定工具箱，提供了从手动标定、基于目标的自动标定到无目标的场景/在线标定等全套解决方案。

### 需要什么样的数据？

取决于你选择的标定模式：

* **基于目标（Target-based）的自动标定**：需要同时包含**特定定制标定板（中间是棋盘格，四周有4个圆孔）**的单帧或多帧相机图像和高密度 LiDAR 点云数据。
* **无目标（Target-less）的自动标定**：需要车辆在**自然道路场景**下采集的数据，场景中最好包含明显的几何特征（如车道线、路灯杆、交通标志）；如果使用内置的 Calib-Anything 方法，则需要任意包含丰富语义的自然场景数据。
* **手动标定**：需要包含明显特征（树木、标志等）的道路场景图像和点云。

### 流程与原理

**A. 基于目标的自动标定（工厂/离线标定）：**

* **原理**：结合了传统棋盘格标定和点云几何特征对齐。利用相机的内参、标定板的物理尺寸和投影模型，构建多约束优化的目标函数。
* **流程**：
    1. 首先使用张氏标定法（Zhang's method）通过棋盘格角点校准相机的内参，并得出标定板到相机的初始外参。
    2. 根据上述参数和标定板尺寸，计算出4个圆孔中心在 2D 图像上的理论坐标。
    3. 在 LiDAR 点云中提取出这4个圆孔的 3D 中心点，并利用当前的 LiDAR-Camera 外参将它们投影到 2D 图像上。
    4. 构建联合损失函数：赋予圆孔中心投影误差和棋盘格角点投影误差不同的权重，使用非线性优化（如 Ceres）不断微调外参，直到两者的重投影误差最小化。

**B. 无目标（Target-less）的自动标定（路测/场景标定）：**

* **原理**：利用自然道路场景中的结构特征（线、面）或深度学习语义掩码（如 SAM）进行跨模态特征对齐。
* **流程（以路灯杆和车道线为例）**：
    1. 分别从相机图像和 LiDAR 点云中提取线性特征（如使用 BiSeNet-V2 提取图像中的车道线和路灯杆，通过反射强度和几何特征提取点云中的对应物）。
    2. 生成像素级掩码（Mask）。
    3. 设计代价函数，将点云特征投影至图像，并最小化其到图像特征掩码的距离以优化外参。
    *注：其更新版的 Calib-Anything 方法更是利用 Segment Anything (SAM) 模型生成图像的语义 Mask，将 LiDAR 点云投影进去，通过最大化 Mask 内部的反射率、法向量等几何/光度一致性来求解外参。*

---

## direct_visual_lidar_calibration (Koide et al.)

这是一个**通用、单次采集（Single-shot）、无目标、全自动**的标定工具箱，它最大的特点是不依赖特定的标定板，且能够兼容各种不同的相机（针孔、鱼眼、全景）和 LiDAR（机械旋转式、固态非重复扫描式）。

需要什么样的数据？

* **单次采集（Single-shot）的自然场景数据**：原则上只需要**一对无需任何标定目标的 LiDAR 点云和相机图像**即可完成标定。
* **数据预处理要求**：为了保证标定成功，LiDAR 点云需要足够“稠密”。对于 Livox 等固态雷达，直接累积一小段时间的点云即可；对于传统机械旋转 LiDAR，系统需要一段几秒钟的连续扫描，并通过内置算法补偿运动畸变来拼合成一帧稠密点云。

### 流程与原理

该工具箱的原理基于**跨模态深度学习特征匹配**与**互信息光度对齐（信息论）**，摒弃了传统的边缘或点面几何提取，从而在缺乏几何特征的环境中依然稳健。

**流程**：

**数据稠密化与预处理**：对采集到的 LiDAR 数据进行积分稠密化，并进行直方图均衡化（以利于后续的互信息计算）。

**自动获取初始位姿（Initial Guess）**：

* 系统首先估算 LiDAR 的视场角，并利用虚拟相机（根据视场角选择针孔或等距圆柱投影模型）将 3D 点云渲染成一张包含反射强度（Intensity）的 **LiDAR 强度 2D 图像**。
* 将这张 LiDAR 强度图与真实的相机图像输入到预训练的图神经网络 **SuperGlue** 中，在两张图像间寻找对应的 2D-2D 匹配点。
* 由于 LiDAR 强度图中的像素能够直接索引回 3D 坐标，这些匹配点就自动转换为了 **2D-3D 对应关系**。随后利用 RANSAC 和最小化重投影误差（PnP）计算出一个粗略的外参初始值。

**高精度直接配准微调（Refinement）**：

* **视点遮挡剔除（Hidden point removal）**：利用初始外参，运用深度缓冲测试剔除掉在相机视角下不可见（被遮挡）的 LiDAR 点，防止它们产生误导。
* **NID 最小化**：利用**归一化信息距离（Normalized Information Distance, NID）**作为损失函数。NID 是一种基于互信息的指标，它直接最大化“投影后的 LiDAR 反射强度”与“图像像素颜色/灰度”之间的统计相关性。系统通过 Nelder-Mead 优化器不断微调外参矩阵，直至 NID 最小化（即两种模态的信息对齐度最高）。

总结与对比建议

* **如果你的场景在量产工厂或有专门的标定场地**，需要极高的工业级精度：推荐使用 **OpenCalib 的基于目标的标定**。它的定制标定板和联合优化能够提供极高精度的约束。
* **如果你在研发阶段，设备常变，或者需要在野外、自然街道中快速完成标定**：推荐使用 **direct_visual_lidar_calibration**。它只需要你停下来或者开着车录几秒钟的数据，全自动运行，不挑传感器型号（甚至全景相机配固态雷达也能跑），而且对环境中是否存在明显的“线、面”特征要求极低（依靠互信息对齐）。

---

## CailBEV

**主要原理**
CalibBEV 将传统的激光雷达-相机（3D-2D）跨模态配准问题，创新性地转化为了**鸟瞰图（BEV）特征对齐问题**。由于 RGB 图像和 LiDAR 点云在 BEV 视角下能够描述相同的空间场景结构，该算法将两种模态的数据投影到统一的 3D 空间表示中，通过对齐这两种 BEV 特征图来求解传感器之间的刚体变换矩阵（旋转和平移）。
为了降低模态差异带来的对齐难度，该方法还引入了类似 CLIP 的对比损失（Contrastive Loss），利用已知的点-像素对应关系，强制让图像的像素特征和 LiDAR 的点特征在特征空间中保持一致性与语义兼容性。

**输入数据**
CalibBEV 需要的输入数据仅为两种：

1. **RGB 图像**：可以使用单目相机图像，也支持多相机（Multi-camera）系统输入的图像（通过融合相邻相机的视锥体特征来提供更加丰富密集的 360° 场景 RGB 信息）。
2. **稀疏的 LiDAR 点云**：独立输入的 3D 激光雷达点云数据。

![image-20260529153050968](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260529153051097.png)

**完整流程**
CalibBEV 的标定流程采用了一种由粗到细（Coarse-to-fine）的“两步走”对齐策略：

1. **独立特征提取与 BEV 投影**
   使用针对特定模态的骨干网络（Backbone），分别对 RGB 图像和 LiDAR 点云进行独立特征提取。在提取出 2D 像素特征和 3D 点云特征后，利用相机内参和空间坐标，将它们分别提升（Lift）并投影到两个预先定义好形状的 3D BEV 空间网格中，生成 `RGB BEV` 和 `LiDAR BEV` 两个特征体积。

2. **第一步：隐式对齐（Implicit Alignment）—— 粗标定**
   * **拼接与融合**：将上述两个未对齐的 BEV 特征沿通道维度进行拼接，通过卷积层进行初步融合。
   * **全局隐式回归**：将融合后的特征输入到一个基于 CNN 的解码器（如 ResNet-18）。此时网络并不进行显式的空间位置变换，而是利用 CNN 的大感受野全局理解这两个 BEV 视图的几何关系，通过后接的旋转和平移预测头（MLP），直接回归出一个**粗略的初始外参标定矩阵（$T_{coarse}$）**。
   * **特征空间约束**：在此步骤的训练中，会引入 CLIP 对比损失，引导网络将对应的点和像素映射到统一的特征空间，极大简化了解码器的隐式对齐难度。

3. **第二步：显式对齐（Explicit Alignment）—— 细标定**
   * **3D 特征扭曲（Warping）**：利用第一步得到的粗标定矩阵 $T_{coarse}$，在特征层面上对 LiDAR BEV 特征进行显式的 3D 空间扭曲，使其在空间上初步对齐到 RGB BEV 特征的参考系下，从而减少两者之间的位置错位。
   * **残差微调**：将初步对齐后的特征送入第二个解码器和预测头，网络基于对齐后的特征进一步微调，输出最终的高精度标定矩阵 $T$。

是否需要单独训练？是的，CalibBEV 不仅是一个需要通过真实外参进行有监督训练的深度学习方法，而且其两个核心对齐模块在训练时是需要“分阶段单独训练”的。

具体训练策略如下：

1. **先训练隐式对齐模块**：首先，在真实标定参数（Ground Truth）和 CLIP 损失的共同监督下，训练整个系统的图像/点云骨干网络、隐式对齐解码器以及其对应的预测头。此阶段大约需要训练 260k 次迭代。
2. **单独微调显式对齐模块**：隐式模块训练完成后，**冻结**前面所有的骨干网络、隐式对齐解码器和预测头的参数（即不让它们再发生改变）。随后，用隐式模块的权重初始化显式对齐模块的解码器和预测头，并以较小的学习率对其进行额外的 120k 次迭代的**单独微调训练**。这样做可以让显式对齐模块专注于从已扭曲的特征中学习如何细化残差，从而输出最精准的标定矩阵。

---

## M-LIC

Targetless Intrinsics and Extrinsic Calibration of Multiple LiDARs and Cameras with IMU using Continuous-Time Estimation 提出了一种非常系统且工程化的高精度标定方法（M-LIC）。

它采用**连续时间轨迹（Continuous-time Trajectory）建模与全局地图跨模态投影**的思路。具体来说，它利用 IMU 数据通过 B 样条（B-spline）构建了一条统一的系统连续时间运动轨迹，分别在此轨迹上对齐相机的 SfM（运动恢复结构）和激光雷达的体素地图（Voxel Map），最后将**全局 LiDAR 点云的反射强度投影成 2D 图像，利用深度学习网络与真实相机图像直接进行密集特征匹配**，以此建立 LiDAR 和 Camera 之间的严格空间约束。

以下是具体的原理、所需数据以及详细流程：

### 进行标定需要什么数据？

**输入数据类型**：多路激光雷达的原始点云、多路相机的图像序列、以及 IMU 的高频角速度和线加速度数据。

**场景与采集要求**：

* **无需标定板（Targetless）**：不需要人工布置棋盘格或反射板。
* **自然结构化场景**：要求车辆/机器人在**纹理丰富且具有几何结构（如建筑物平面、道路）**的环境中采集一段数据。
* **充分的运动激励**：车辆需要有充分的运动（如走“8”字形轨迹），以激发 IMU 的各个自由度，确保外参可观测。
* **无需重叠视场**：极大的优势在于，不同相机之间、以及相机和 LiDAR 之间**不需要有直接的重叠视场（FOV）**也能完成标定。因为它是在全局统一的三维地图里去建立约束的。

### 核心标定原理

* **B样条连续时间轨迹建模**：传统离散时间状态在处理异步多传感器时会产生巨大的计算量。该方法用 B 样条曲线来参数化 IMU 在世界坐标系下的 6 自由度轨迹。这种连续的函数表达使得系统能够以解析的方式插值出任意时间戳下的位姿，不仅能自然融合未同步的异步传感器数据，还能将传感器之间的**时间偏移量（Time offset）**作为参数放进因子图中一并优化求解。
* **LiDAR-Camera 跨模态匹配**：为了解决 LiDAR 和 Camera 特征难对齐的问题，算法放弃了不稳定的边缘或线特征匹配，而是把构建好的 3D LiDAR 地图投影到相机成像平面上，将点云的**反射强度（Intensity）作为伪图像的灰度值**。然后使用先进的深度学习匹配算法（SuperPoint + SuperGlue），在“LiDAR 强度图”和“相机灰度图”之间寻找 2D 像素到 3D 地图点的精确对应关系。

### 详细的标定流程

![image-20260529175250482](https://fastly.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/20260529175250720.png)

整个流程分为四个由粗到细的阶段：

#### 阶段一：各传感器解耦初始化 (Initialization)

由于联合优化高度非线性，需要先给各个参数找到一个合理的初值：

* **Camera 与 IMU 初始化**：利用开源软件 COLMAP 运行 SfM，利用 SuperPoint/SuperGlue 进行特征提取和匹配，恢复出所有相机的位姿、3D 视觉路标点及内参初值。接着，利用手眼标定法结合 IMU 轨迹，计算出 Camera-IMU 的外参初值。
* **LiDAR 与 IMU 初始化**：利用鲁棒的 KISS-ICP 算法跑出 LiDAR 的离散里程计位姿，然后与 IMU 积分出的相对位姿进行对齐，解算出 LiDAR-IMU 的外参初值，并利用重力对齐初始化 IMU 的连续时间轨迹。

#### 阶段二：位姿细化与多 LiDAR 联合地图构建 (Refine Pose and Extrinsic Params)

为了消除初始化误差导致的点云分层或不一致现象：

* **单 LiDAR 平面 BA 优化**：将地图分割为不同尺寸的体素（Adaptive Voxel Map），提取 LiDAR 点云的平面特征。以单个 LiDAR 为基准进行平面 Bundle Adjustment (BA) 优化，将点云强行对齐到 IMU 的连续轨迹上。
* **多 LiDAR 联合构图**：单 LiDAR 对齐后，不同 LiDAR 之间也就通过 IMU 轨迹被拉到了同一个坐标系下，从而构建出一个无明显分层、具备全局几何一致性的“多 LiDAR 联合体素地图”。

#### 阶段三：LiDAR-Camera 跨模态匹配 (Cross-modal Match)

此步骤是 LiDAR 与 Camera 建立直接外参联系的关键：

* **强度投影**：利用当前粗略的内外参，将上述优化好的**高质量全局 LiDAR 点云地图**投影到各个相机的图像平面上，渲染出一张 2D 的 LiDAR 强度图像。
* **深度学习匹配**：对雷达强度图进行直方图均衡化处理后，使用 SuperPoint 和 SuperGlue 算法将它与真实的相机照片进行匹配。这一步直接将 2D 的图像像素特征点关联到了 3D 的 LiDAR 世界坐标点上。

#### 阶段四：全系统因子图联合优化 (Joint Calibration)

在获取了所有跨模态和单模态约束后，系统构建一个庞大的非线性最小二乘（NLS）优化问题（因子图）：

* **构建残差**：通过 PnP RANSAC 剔除上一步匹配中的异常离群点。然后构建四种残差：IMU 运动学残差、LiDAR 平面特征残差、相机 SfM 重投影残差、以及**最关键的 LiDAR-Camera 2D-3D 重投影残差**。
* **联合求解**：在这个统一的因子图中，同时对 IMU 的 B 样条控制点、所有 LiDAR 和相机的外参（旋转平移）、相机的内参（焦距、畸变）、以及它们与 IMU 之间的**时间偏移量（Time offset）**进行联合求解。

通过这种方式，算法不仅能实现像素级的 LiDAR-Camera 外参标定，还顺带把多相机的内参、多激光雷达的联合标定以及硬件时间戳误差一并解决，极大消除了传感器单独标定带来的累积误差。

## CRLF

CRLF: Automatic Calibration and Refinement based on Line Feature  for LiDAR and Camera in Road Scenes。OpenCalib工具箱中关于lidar2cam的无目标外参标定的原始方法论文。

CRLF是一种完全自动化、无目标（Targetless）的单帧 LiDAR-Camera 外参标定算法。它巧妙地利用了自然道路场景中极其常见的**静态线状物体（如车道线、路灯杆、电线杆）**作为特征来进行标定。

**图像端依赖深度学习语义分割**：直接输入 RGB 图像，算法内置了 **BiSeNet-V2** 实时语义分割网络来提取“车道线”和“电线杆”的初始像素，随后使用全连接条件随机场（Dense CRF）来平滑和细化这些极细物体的轮廓，最终生成精确的 2D 语义掩码（Mask）。

**点云端完全依赖几何与物理特性（不需要深度学习）**：点云没有使用庞大的 3D 语义分割网络。对于**车道线**，算法先用 RANSAC 提取地面，然后利用车道线涂料具有**高反射强度（Intensity）****电线杆**，算法在非地面点云中构建 2D 俯视网格，通过设置高度阈值（如筛选高度大于 3 米的垂直柱状物）来利用几何先验提取电线杆 3D 点云

利用这些对应语义提取出的线特征，CRLF 就可以自动完成后续的标定。

以下是该算法的具体使用条件、输入数据以及完整原理流程：

1.   输入数据与使用条件

-   算法的完整流程与原理

CRLF 采用了一种“由粗到精（Coarse-to-Fine）”的策略，其核心流程分为三个阶段：

**<span style="color:#d59bf6;">第一阶段：线特征提取 (Line Feature Extractor)</span>**

如前文所述，算法分别在图像和点云中独立工作。在点云中通过反射率和高度几何特征提取出 3D 车道线和电线杆点云；在图像中通过 BiSeNet-V2 网络提取出 2D 的车道线和电线杆像素掩码。

<span style="color:#d59bf6;">**第二阶段：基于 P3L 的粗标定 (Coarse Calibration)**</span>

传统方法寻找初始外参往往很难，且动态物体（如行驶的汽车）会带来运动畸变误差。CRLF 放弃了点特征，将问题转化为一个 **Perspective-3-Lines (P3L)** 问题。

**特征拟合**：在提取出的点云和图像特征上，利用霍夫变换（Hough transform）和 RANSAC 拟合出数学意义上的直线方程。**构建约束**：算法在图像和点云中分别选取 **3 条线对应关系（2 条车道线，1 根电线杆）**。**简化与求解**：在一般情况下求解 P3L 涉及复杂的八次方程。为了极大地简化计算，CRLF 引入了一个“车道线彼此平行”的假设，并利用一个与地面平行的中间坐标系，快速解算出相机的粗略旋转和位移矩阵。**穷举与打分**：由于不知道图像里的哪条线对应点云里的哪条线，算法会穷举所有可能的线对应组合，计算出多个候选的外参矩阵，并利用一个代价函数对它们进行打分，得分最高的那组被选为**粗标定结果（Coarse calibration）**。

**<span style="color:#d59bf6;">第三阶段：基于语义代价函数的细标定 (Calibration Refinement)</span>**

由于粗标定阶段只用了 3 条线，且现实中车道线并非绝对平行，因此需要进行全局非线性优化来细化参数。

**构建代价函数 (Cost Function)**：在这个阶段，算法抛弃了“平行”假设，把第一步提取出的所有 3D 点云线特征（包含所有的车道线点和电线杆点）利用粗标定外参投影到 2D 图像平面上。算法设计了一个代价函数，用于衡量投影过来的 3D 语义点与图像自身提取出的 2D 语义掩码的重合度。**随机搜索优化**：由于这个代价函数是非凸的，很难用常规方法求解，CRLF 采用了一种带有衰减步长的随机搜索（Random search）算法。它在当前的粗标定参数附近不断产生微小的平移和旋转扰动（例如平移范围限制在 ±1 米内，旋转限制在 ±0.1°内），寻找能使代价函数（即语义重合度）最大的外参组合。**输出结果**：经过多次迭代直至步长收敛，输出最终的高精度 LiDAR-Camera 外参矩阵。

**总结来说：** CRLF 的精妙之处在于它避免了复杂的特征点匹配，而是利用了道路场景中极其稳定、天然带方向约束的“线”和“杆”。它利用深度学习在图像端提供语义 Mask 作为“靶靶”，利用 LiDAR 的物理几何特性提供 3D 射点，最后通过最大化两者的投影重叠度，实现了一键式、无需人工布置场景的自动化精准标定

## 目前的尝试

MFCalib

代码里面貌似写死了针孔相机模型，且畸变参数只能是5位，要适配我们14位的畸变参数以及鱼眼相机可能要单独适配；强行只取前5位畸变参数，效果比较差。

OpenCalib

需要先单独对图片做语义分割，得到多张mask图片；具体原理如M-LIC中描述。

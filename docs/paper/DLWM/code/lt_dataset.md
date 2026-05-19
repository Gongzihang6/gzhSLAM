# gaussian_perception/dataset/lt_dataset.py

**负责数据集加载**

```python
"""
dlwm/dataset.py
===============
DLWM (arXiv:2604.00969) 一阶段复现 —— 数据加载器

目录结构约定（data_root 下）：
    undistorted_imgs/
        JPG_2K_CAM_FN/              ← 多视角去畸变图像
            frame-XXXXXX_<13位ms时间戳>.jpg
            JPG_2K_CAM_FN_mask/     ← 每帧对应的二值 mask（可选）
                frame-XXXXXX_<ts>_road.png
                frame-XXXXXX_<ts>_vehicle.png
                ...
            cam0_params.yml         ← OpenCV 风格标定文件
        JPG_2K_CAM_FW/ ...          ← 其余相机同理
    DEPTH_GT_PNG/                   ← ★ 主时间轴：LiDAR 投影稀疏深度（16-bit PNG）
        frame-XXXXXX_<ts>_<cam>.png
    SEMANTIC_GT_PNG/                ← LiDAR 投影语义标签（8-bit PNG）
        frame-XXXXXX_<ts>_<cam>.png
    metric3d_output/
        dense_depth/                ← Metric3D 稠密深度（16-bit PNG）
            frame-XXXXXX_<ts>_<cam>.png
    calib_param/                    ← 相机标定 YAML（备选位置，若 img 目录内无 yaml）

命名约定（DEPTH_GT_PNG 文件名基准）：
    frame-XXXXXX_<13位ms时间戳>_<cam_key>.png
    例：frame-000001_1701234567890_JPG_2K_CAM_FN.png

设计说明：
  - 以 DEPTH_GT_PNG 中的文件名为主时间轴（LiDAR 对齐后的图像时间戳）
  - start_timestep / end_timestep 支持整数（帧序号）或字符串（时间戳）范围切片
  - 16-bit 深度 PNG 读取后除以 1000.0 还原为米
  - 支持将多个二值 mask 合并为单张类别索引图
  - 输出字典与 GaussianFormer 训练链路兼容
"""
```

这 7 个相机构成一个典型的 7V（7 个主感知相机）视觉感知方案。

```python
SENSOR_TO_PCO_NAME: Dict[str, str] = {
    'JPG_2K_CAM_FN': 'cam0',
    'JPG_2K_CAM_FW': 'cam1',
    'JPG_CAM_FL':    'cam4',
    'JPG_CAM_RL':    'cam5',
    'JPG_CAM_RN':    'cam6',
    'JPG_CAM_RR':    'cam7',
    'JPG_CAM_FR':    'cam8',
}
```

从外参平移向量（cam2ego_t）可以看出，车身坐标系（Ego Frame）标准是：X 正方向为车头向前，Y 正方向为车身向左，Z 正方向为垂直向上。

1、前向主视双目：FN 与 FW（一长一宽）。这两颗相机并排安装在挡风玻璃后方（内后视镜处），其中 cam0(FN，Front Narrow 前向窄角/长焦)，焦距越大，视场角越小，类似于望远镜，它的朝向完全正前，专职负责看极远处的红绿灯和车辆；cam1（FW-Front Wide 前向广角），和 cam0 的高度和靠前程度几乎一致，说明它们是紧挨着的前视感知模块。

2、侧前视相机：FL 与 FR（看斜前方）。cam4 (FL - 左前) / cam8 (FR - 右前)安装位置通常在车辆侧后视镜下方，FL 的光轴指向（X: 0.35, Y: 0.93），说明它有 35%指向正前方，93%指向侧左方；FR 同理指向侧右方。它们主要负责车辆斜前方的盲区。

3、侧后视相机：RL 与 RR（“向后看盲区”）。cam5 (RL - 左后) / cam7 (RR - 右后)安装位置： X≈2.45m，Y≈±0.99m，Z≈0.85m。比后视镜相机（X = 2.16m）更靠前且更低。这说明它们大概率安装在前翼子板（前轮眉附近）。 RL 的光轴指向（X: -0.77, Y: 0.62），说明它有 77%是向车后方看，62%向侧边看。安装在车头却向后看，完美覆盖了车身侧面直到车尾的巨大盲区。

4、正后视主感知：RN（“后向长焦”），用于看清正后方的远距离来车。cam6 (RN - Rear Normal 后视常规)，安装位置： X≈-0.26m，Y≈0m，Z≈1.57m。安装在车尾（X 为负数），位于正中央（Y 为 0），且位置较高（Z 为 1.57m，说明可能安装在后挡风玻璃顶部或鲨鱼鳍附近）。光轴指向（X: -0.999），极其精准地直指正后方。

---

下面这段代码主要负责从复杂的文件目录中，将不同传感器（如多个摄像头、深度传感器、占据栅格网络 Occ 等）在同一时间戳下采集的数据严格对齐，并打包乘训练或推理所需的字典格式。

作用是构建数据集的索引列表，在给定的时间范围内，扫描并匹配各个传感器的数据文件路径，将杂乱分散的 RGB 图像、稀疏深度图、语义标签图、稠密深度图以及 3D 占据栅格数据，严格按照时间戳和相机机位进行对齐缝合，输出一个包含每一帧完整数据的列表 `List[dict]`。

具体做法是以深度图为基准，扫描所有稀疏深度图，解析文件名提取时间戳和相机标号，然后剔除那些存在传感器数据缺失（比如某一帧丢了cam0的数据）的时间戳，保证每一帧都是“全套”的。根据传入的start和end截取需要的时间段，对于图像、语义等高频或同步触发的传感器，通过同名文件直接匹配，对于低频或异步触发的Occ数据，通过二分查找寻找时间戳最接近的文件。

```python
    def _build_data_infos(
        self,
        start_timestep: Union[int, str],
        end_timestep: Union[int, str],
    ) -> List[dict]:
        """以 DEPTH_GT_PNG 目录为主时间轴扫描所有可用帧，并按 [start, end) 切片。

        DEPTH_GT_PNG 文件命名约定：
            frame-XXXXXX_<13位ms时间戳>_<cam_key>.png

        一帧 = 一个时间戳（同一时间戳可能对应多个相机，但我们按时间戳聚合）。
        """
        assert os.path.isdir(self.depth_gt_dir), (
            f"DEPTH_GT_PNG 目录不存在：{self.depth_gt_dir}"
        )
        # 适配 DEPTH_GT_PNG/<cam_key>/*.png
        depth_files = sorted(glob.glob(os.path.join(self.depth_gt_dir, '*', '*.png')))
        assert depth_files, f"DEPTH_GT_PNG 目录下未找到 PNG 文件：{self.depth_gt_dir}"

        # 以 (时间戳, cam_key) 为 key，按时间戳聚合
        # 每个时间戳对应一个完整的"帧"（包含所有相机）
        ts_to_cam_depth: Dict[int, Dict[str, str]] = {}
        for fp in depth_files:	# 遍历所有深度图，获取时间戳，相机标号
            stem = os.path.splitext(os.path.basename(fp))[0]
            ts = _parse_timestamp_from_stem(stem)
            cam = _parse_cam_key_from_depth_stem(stem, self.cam_keys)
            if cam is None:
                cam = _parse_cam_key_from_path(fp, self.cam_keys)
            if cam is None:
                continue   # 不属于任何配置相机，跳过
            if ts not in ts_to_cam_depth:
                ts_to_cam_depth[ts] = {}
            ts_to_cam_depth[ts][cam] = fp

        # 只保留所有相机均有深度 GT 的时间戳
        full_timestamps = sorted([
            ts for ts, cams in ts_to_cam_depth.items()
            if all(c in cams for c in self.cam_keys)
        ])
        assert full_timestamps, (
            "DEPTH_GT_PNG 中没有包含所有相机的完整帧，请检查文件命名约定。"
        )

        # ── 时间戳切片 ────────────────────────────────────────
        total = len(full_timestamps)
        start_idx, end_idx = self._resolve_slice(
            full_timestamps, start_timestep, end_timestep, total
        )
        sliced_ts = full_timestamps[start_idx:end_idx]	# 最终需要处理的、切片好的时间戳列表
        assert sliced_ts, (
            f"切片后没有可用帧：start={start_timestep}, end={end_timestep}"
        )

        # ── 构建 data_infos ────────────────────────────────────────
        infos: List[dict] = []

        occ_ts_list: List[int] = []
        occ_path_list: List[str] = []
        if os.path.isdir(self.occ_gt_dir):
            occ_files = sorted(glob.glob(os.path.join(self.occ_gt_dir, '*.npz')))
            occ_entries: List[Tuple[int, str]] = []
            for occ_file in occ_files:
                occ_stem = os.path.splitext(os.path.basename(occ_file))[0]
                m = re.search(r'(\d+)', occ_stem)
                if m is None:
                    continue
                raw_ts = int(m.group(1))
                # 兼容 19 位纳秒时间戳命名（如 xxxxxxxxxxxxxxxxxxx_occ.npz）
                occ_ts_ms = raw_ts // 1_000_000 if raw_ts >= 10**15 else raw_ts
                occ_entries.append((occ_ts_ms, occ_file))
            occ_entries.sort(key=lambda x: x[0])
            occ_ts_list = [x[0] for x in occ_entries]
            occ_path_list = [x[1] for x in occ_entries]

         # 内部闭包函数，用于异步数据匹配，使用标准二分查找算法，在有序的 occ_ts_list 中快速找到目标时间戳 ts_ms 应该插入的位置
        def _match_occ_path(ts_ms: int, tol_ms: int = 100) -> Optional[str]:
            if not occ_ts_list:
                return None
            idx = bisect.bisect_left(occ_ts_list, ts_ms)
            cands = []
            if idx < len(occ_ts_list):
                cands.append(idx)
            if idx > 0:
                cands.append(idx - 1)
            if not cands:
                return None
            best_idx = min(cands, key=lambda i: abs(occ_ts_list[i] - ts_ms))
            if abs(occ_ts_list[best_idx] - ts_ms) <= tol_ms:
                return occ_path_list[best_idx]
            return None
		  # 遍历之前切好的“有效时间戳”
        for ts in sliced_ts:
            # 对每个相机按深度文件名直接定位 RGB 图像
            cam_img_paths: Dict[str, str] = {}
            for cam in self.cam_keys:
                depth_path = ts_to_cam_depth[ts][cam]
                depth_stem = os.path.splitext(os.path.basename(depth_path))[0]

                img_dir = os.path.join(self.data_root, 'undistorted_imgs', cam)
                candidates = [os.path.join(img_dir, f'{depth_stem}{self.img_ext}')]

                rgb_path = next((p for p in candidates if os.path.isfile(p)), None)
                assert rgb_path is not None, (
                    f"未找到与深度图匹配的 RGB 图像：cam={cam}, depth={depth_path}, "
                    f"尝试路径={candidates}"
                )
                cam_img_paths[cam] = rgb_path

            # 语义 GT 路径（与深度同命名规则）
            cam_sem_paths: Dict[str, str] = {}
            if os.path.isdir(self.semantic_gt_dir):
                for cam in self.cam_keys:
                    # 从深度路径推导语义路径（替换目录）
                    depth_basename = os.path.basename(ts_to_cam_depth[ts][cam])
                    sem_path = os.path.join(self.semantic_gt_dir, cam)
                    sem_path = os.path.join(sem_path, depth_basename)
                    if os.path.isfile(sem_path):
                        cam_sem_paths[cam] = sem_path

            # 稠密深度路径
            cam_dense_paths: Dict[str, str] = {}
            if os.path.isdir(self.dense_depth_dir):
                for cam in self.cam_keys:
                    depth_basename = os.path.basename(ts_to_cam_depth[ts][cam])
                    dense_path = os.path.join(self.dense_depth_dir, cam)
                    dense_path = os.path.join(dense_path, depth_basename)
                    if os.path.isfile(dense_path):
                        cam_dense_paths[cam] = dense_path

            occ_path = _match_occ_path(ts)

            infos.append({
                'timestamp_ms':       ts,
                'cam_img_paths':      cam_img_paths,
                'cam_depth_paths':    ts_to_cam_depth[ts],   # 稀疏深度
                'cam_sem_paths':      cam_sem_paths,
                'cam_dense_paths':    cam_dense_paths,
                'occ_path':           occ_path,
            })
        return infos

    @staticmethod
    def _resolve_slice(
        timestamps: List[int],
        start: Union[int, str],
        end: Union[int, str],
        total: int,
    ) -> Tuple[int, int]:
        """将 start/end 解析为 [start_idx, end_idx) 帧序号区间。

        若 start/end 为字符串且为纯数字，则视为毫秒时间戳，做最近邻匹配；
        否则视为整数帧序号（0-based，-1 = 末尾）。
        """
        def _to_idx(v: Union[int, str], default_end: int) -> int:
            if isinstance(v, str) and v.isdigit() and len(v) == 13:
                # 13 位时间戳字符串
                ts_arr = np.asarray(timestamps, dtype=np.int64)
                return int(np.argmin(np.abs(ts_arr - int(v))))
            iv = int(v)
            if iv < 0:
                return default_end
            return iv

        start_idx = _to_idx(start, 0)
        end_idx   = _to_idx(end, total)
        assert 0 <= start_idx <= end_idx <= total, (
            f"切片范围非法：start_idx={start_idx}, end_idx={end_idx}, total={total}"
        )
        return start_idx, end_idx
```


# 第七章 3D SLAM

## 1、推导式（7.4）中关于 R 的右乘导数

![image-20260304154721401](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2026%5C03%5Cimage-20260304154721401.png)

---

## 2、使用 srd:: reduce 对 Hessian 矩阵的累加部分进行并发处理

在本书的 `src/ch7/icp_3d.cc` 中，在手动使用高斯牛顿法求解点到点、点到线、点到面的 ICP 配准问题时，我们需要遍历源点云中的每一个点，对于源点云中的每一个点，我们去目标点云中寻找距离最近的点（这个操作用 K-d 树加速完成），找到之后，计算它们的距离，如果小于距离阈值，则认为找到一对正确的匹配点对，认为这个点对是有效的；对于每一个有效的点对，我们要根据初始位姿，来计算这个有效点对的误差、雅可比矩阵，然后存储在 `errors` 和 `jacobians` 中，然后在 ICP 的每一轮迭代中，我们需要将所有有效点对的误差和雅可比矩阵累加起来，然后求解增量方程，得到当前轮迭代中，位姿的更新量。源代码中累加误差和雅可比矩阵使用的是 `std::accumulate` 串行执行，在有效点对数量比较多的时候，串行执行就比较慢了，因此可以用 `std::transform_reduce` 实现多线程并发处理，每个线程分配一些点，来独立计算误差和雅可比矩阵之和，最后将所有线程的误差和雅可比矩阵之和加起来就可以了。

不过，直接把代码里的 `std::accumulate` 替换成 `std::transform_reduce` 并加上 `std::execution::par_unseq` 是会报错或产生严重 bug 的。原因在于原代码的 Lambda 表达式中包含了副作用：它通过引用捕获修改了外部变量 `total_res` 和 `effective_num`，如果直接在多线程下并发执行，会导致严重的数据竞争。

|   **特性**   | **std:: accumulate**                                   | **std:: reduce / std:: transform_reduce (C++17)**              |
| :----------: | ----------------------------------------------------- | ------------------------------------------------------------ |
| **执行方式** | 严格串行（单线程）。                                  | 可指定执行策略，支持并行 (`std::execution::par`) 和向量化。  |
| **计算顺序** | 严格从左到右依次累加：$(((a_1 + a_2) + a_3) + a_4)$。 | 顺序不确定，通常是树状规约（Tree Reduction）：$(a_1 + a_2) + (a_3 + a_4)$。 |
| **数学要求** | 无特殊要求。                                          | 操作符必须满足 **交换律 (Commutativity)** 和 **结合律 (Associativity)**。 |
|  **副作用**  | 允许修改外部变量（因为是串行执行的）。                | **严禁修改外部变量**，否则会引发线程间的数据竞争。           |

为了实现并发，我们必须消除对 `total_res` 和 `effective_num` 的外部引用修改。正确的做法是将这四个需要累加的量（$H$ 矩阵、残差向量 $err$、总误差平方 `total_res`、有效点数 `effective_num`）打包成一个新的数据结构，然后使用 **`std::transform_reduce`**。

`std::transform_reduce` 分为两步：

1.  **Transform（映射）**：把当前点的索引 `idx` 映射（转换）为包含 $H_i, err_i, res_i, num_i$ 的局部状态。
2.  **Reduce（规约）**：将所有线程产生的局部状态，按照加法规则合并起来。

以下是改写后的并发版本代码：

```c++
bool Icp3d::AlignP2P(SE3& init_pose) {
    LOG(INFO) << "aligning with point to point";
    assert(target_ != nullptr && source_ != nullptr);

    SE3 pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    // 对点的索引，预先生成
    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // 我们来写一些并发代码
    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Vec3d> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {    // ICP迭代次数
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 1);   // 遍历源点云中的每一个点，在目标点云中找到最近的一个点，nn

            if (!nn.empty()) {      // 如果找到最近邻
                Vec3d p = ToVec3d(target_->points[nn[0]]);
                double dis2 = (p - qs).squaredNorm();
                if (dis2 > options_.max_nn_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Vec3d e = p - qs;
                Eigen::Matrix<double, 3, 6> J;      // 这里的J是分子布局的雅可比矩阵，不是导数/梯度
                J.block<3, 3>(0, 0) = pose.so3().matrix() * SO3::hat(q);    // 参考公式（7.4）
                J.block<3, 3>(0, 3) = -Mat3d::Identity();

                jacobians[idx] = J;
                errors[idx] = e;
            } else {
                effect_pts[idx] = false;
            }
        });

        // // 累加Hessian和error,计算dx
        // // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        // double total_res = 0;
        // int effective_num = 0;
        // auto H_and_err = std::accumulate(
        //     index.begin(), index.end(), std::pair<Mat6d, Vec6d>(Mat6d::Zero(), Vec6d::Zero()),
        //     [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Mat6d, Vec6d>& pre,
        //                                                                    int idx) -> std::pair<Mat6d, Vec6d> {
        //         if (!effect_pts[idx]) {
        //             return pre;
        //         } 
        //         else {
        //             total_res += errors[idx].dot(errors[idx]);
        //             effective_num++;
        //             return std::pair<Mat6d, Vec6d>(pre.first + jacobians[idx].transpose() * jacobians[idx],
        //                                            pre.second - jacobians[idx].transpose() * errors[idx]);
        //         }
        //     });

        /* * 作用：利用 C++17 的 std::transform_reduce 并发计算高斯-牛顿法的正规方程
        * 功能：多线程并行遍历所有点，计算有效点的雅可比矩阵、误差，并最终安全地归约合并出总的海森矩阵 H、负梯度向量 err、总误差和有效点数量。
        * 实现了什么：消除了原 accumulate 代码中 Lambda 表达式的外部状态依赖（副作用），实现了线程安全的并行加速。
        * 怎么实现的：
        * 1. 定义局部结构体 `ReduceState` 将所有需要累加的状态打包，并重载 `operator+` 以满足 reduce 的结合律。
        * 2. 使用 `std::transform_reduce` 和 `std::execution::par_unseq` 策略。
        * 3. Transform 阶段：根据索引 idx 计算当前单个点的 ReduceState。
        * 4. Reduce 阶段：自动调用重载的 `+` 运算符，以树状结构并行合并所有状态。
        */

        // 1. 定义一个用于归约的复合状态结构体
        struct ReduceState {
            Mat6d H = Mat6d::Zero();
            Vec6d err = Vec6d::Zero();
            double total_res = 0.0;
            int effective_num = 0;

            // 必须重载加法运算符，告诉 reduce 如何合并两个状态
            ReduceState operator+(const ReduceState& other) const {
                ReduceState res;
                res.H = this->H + other.H;
                res.err = this->err + other.err;
                res.total_res = this->total_res + other.total_res;
                res.effective_num = this->effective_num + other.effective_num;
                return res;
            }
        };

        // 初始零状态
        ReduceState init_state;

        // 2. 执行并发变换与归约
        auto final_state = std::transform_reduce(
            std::execution::par_unseq, // 并发且允许向量化执行策略
            index.begin(), index.end(),
            init_state,
            std::plus<ReduceState>(),  // 归约操作：使用上面重载的 operator+
            [&jacobians, &errors, &effect_pts](int idx) -> ReduceState {
                ReduceState local_state;
                if (effect_pts[idx]) {
                    // Transform 阶段：只处理当前点的数据，绝对不碰任何外部的可变变量
                    local_state.H = jacobians[idx].transpose() * jacobians[idx];
                    // 注意：正规方程右侧是 -J^T * e，原代码是 pre.second - ...
                    // 在这里我们直接将其表示为局部项的符号
                    local_state.err = -jacobians[idx].transpose() * errors[idx];
                    local_state.total_res = errors[idx].dot(errors[idx]);
                    local_state.effective_num = 1; // 当前这是一个有效点
                }
                return local_state;
            }
        );

        // 3. 结果提取
        double total_res = final_state.total_res;
        int effective_num = final_state.effective_num;
        Mat6d H = final_state.H;
        Vec6d err = final_state.err; // 这里已经是 -J^T * e 的累加结果了    




        if (effective_num < options_.min_effective_pts_) {
            LOG(WARNING) << "effective num too small: " << effective_num;
            return false;
        }

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());       // 李代数的指数映射
        pose.translation() += dx.tail<3>();

        // 更新
        LOG(INFO) << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                  << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm();

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            LOG(INFO) << "iter " << iter << " pose error: " << pose_error;
        }

        if (dx.norm() < options_.eps_) {
            LOG(INFO) << "converged, dx = " << dx.transpose();
            break;
        }
    }

    init_pose = pose;
    return true;
}
```

---

## 3、推导点面残差和点线残差对 $R$ 的导数

![bf9c23c86f086384e877ec771f0ee4f](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2026%5C03%5Cbf9c23c86f086384e877ec771f0ee4f.png)

![1ffc4b2754f8dd733b5231dc07a3dfc](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2026%5C03%5C1ffc4b2754f8dd733b5231dc07a3dfc.png)

---

## 4、解释NDT中加权最小二乘问题和最大似然估计之间的关系

![f5b44596fa02bf3bfeaff7fca36f94b](https://cdn.jsdelivr.net/gh/Gongzihang6/Pictures@main/Medias/2026%5C03%5Cf5b44596fa02bf3bfeaff7fca36f94b.png)

---

5、为激光线束曲率的计算设计一个并发计算流程并实现

6、为本书的Loam Like Odometry设计一个地面提取的流程，并为地面点云单独使用点到面ICP

7、阅读文献，理解LOAM或LeGO-LOAM等算法在特征提取方法上的差异

8、尝试使用本书介绍的其他方法（点到面ICP、特征法等）实现一个松耦合LIO系统

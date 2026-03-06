# 误差状态卡尔曼滤波器的推导和理解

这里书中介绍定义了名义状态变量，真值，误差状态变量，其中 **真值 = 名义状态变量+误差状态变量**，名义状态变量就是根据 IMU 读数积分直接算出来的状态（这里不用考虑噪声误差等，因为会在误差状态变量中体现），然后推导计算误差状态变量的形式以及导数等，然后将误差作为卡尔曼滤波器的更新变量

RTK 设备为我们提供了一个不太稳定的位姿观测源。我们可以将它视为定位滤波器的一种“观测”（也对应普通卡尔曼滤波器中的观测方程），而 IMU 则提供了运动方程（通过加速度和角速度推算车辆的位置和状态），所以这里我们把 IMU 视为运动模型，把 GNSS 观测视为观测模型，推导误差状态卡尔曼滤波器。

## ESKF 的数学推导

设状态变量为

$$
x = [p, v, R, b_g, b_a, g]^T
$$

所有变量都默认取下标 $()_{WB}$，也就是车辆在世界坐标系下的状态。其中 $p$ 为平移，$v$ 为速度，$R$ 为旋转，$b_g$ 为陀螺仪零偏，$b_a$ 为加速度计零偏，$g$ 为重力加速度。

将 IMU 测量值代入到 IMU 运动学方程中，可以得到状态变量在连续时间下的运动方程为：

$$
\begin{align*}
\dot{p} &= v, \tag{2.a} \\
\dot{v} &= R(\tilde{a} - b_a - \eta_a) + g, \tag{2.b} \\
\dot{R} &= R(\tilde{\omega} - b_g - \eta_g)^{\wedge}, \tag{2.c} \\
\dot{b}_g &= \eta_{bg}, \tag{2.d} \\
\dot{b}_a &= \eta_{ba}, \tag{2.e} \\
\dot{g} &= 0. \tag{2.f}
\end{align*}
$$

式（2.b）中，右侧是根据 IMU 的加速度计读数，推算出的车辆在世界坐标系下的真实加速度值，首先对 IMU 的原始读数（包含真实受力、零偏、白噪声），所以要减去零偏和白噪声，然后乘 $R_{WB}$ 变换到世界坐标系下，但是由于 IMU 本身会多测一个重力加速度（静止时读出反向的 $g$），所以还需要再加上 $g$（$g$ 是负值，加上就可以抵消这个分量）。

为了在 EKF 的预测过程中对协方差进行预测，需要对该方程进行线性化。理论上，线性化的形式为：

$$
x_{k+1}= f(x_k)+Fdx+w
$$

其中 $F = \frac{\partial f}{\partial x(t)} \Big|_{x(t)}$ 为雅可比系数矩阵，该矩阵由运动方程和各项状态变量的导数构成。

这里会遇到一个非常现实的问题：

一方面，F 中需要计算旋转矩阵 R 相对于某个扰动($\delta x$)的导数，但是在不引入张量的情况下，是无法表达矩阵对向量的导数的形式的，于是，传统算法往往会退一步，用欧拉角或者四元数的四个标量作为状态变量，但是这样就无法优雅的使用流形上的方法了；

另一方面，如果考虑将惯性导航系统与卫星导航系统进行融合，那么 $x$ 中的平移变量就应该使用全局坐标系，这会使得 x 中的数值变得很大，在有些场合超出浮点数的有效数字范围，这可能导致一些场景的运算失效，例如数值计算中的“大数吃小数”现象。

于是，我们开始考虑，能否避免直接使用 $x$ 和 $P$ 来表达状态的均值和协方差，推导运动和观测方程呢？能否使用原先卡尔曼滤波器中的更新量来推导这两个方程？回忆卡尔曼滤波器中的观测部分：

$$
x_k = x_{k, pred}+K_k \underbrace{(z_k-H_kx_{k, pred})}_{\text{更新量}}
$$

在流形意义下，右侧的更新量应该是位于切空间中的矢量，中间的加法应为流形与切空间指数映射的广义加法。为了避免前面提到的两个问题，我们选择将更新量（或者称为误差状态）视为滤波器的状态变量，来推导运动和观测模型，这就引出了误差状态卡尔曼滤波器。我们不直接预测和更新状态，而是对状态误差进行预测和更新，具体来说，IMU 运动方程（积分）给出粗略估计，误差状态卡尔曼滤波器预测这个粗略估计的偏差，然后粗略估计+偏差 = 更新后的状态。

ESKF（Error State Kalman Filter）是许多传统的、现代的系统里都广泛使用的状态估计方法，既可以作为组合导航的滤波器，也可以用来实现 **LIO（Lidar-Inertial Odometry）、VIO（Visual-Inertial Odometry）** 等复杂系统。相比于传统的 KF，ESKF 的优点可以总结如下：

1、在旋转的处理上，ESKF 的状态变量可以采用最小化的参数表达，也就是 **使用三维变量（旋转矢量、李代数）来表达旋转的增量**，该变量位于旋转矩阵流形的切空间中，而切空间是一个矢量空间；

2、ESKF 总是在原点附近，离奇异点较远，数值方面更稳定，不会产生离工作点太远而导致线性化近似不够的问题；

3、ESKF 的状态量（误差状态）是小量，其二阶变量相对来说可以忽略，同时，大多数雅可比矩阵在小量情况下变得非常简单，甚至可以用单位阵替代；

4、误差状态的运动学相比原状态变量更小（小量的运动学），因此可以把更新部分归入原状态变量中；

---

> [!IMPORTANT]
>  在 ESKF 中，通常把原状态变量称为名义状态变量（Nominal State）（这里指的是 IMU 运动学积分结果），把 ESKF 里的状态变量称为误差状态变量（Error State）。名义状态变量和误差状态变量之和称为真值。把噪声的处理放到误差状态变量中，可以认为名义状态变量的方程是不含噪声的。（也就是 IMU 运动学方程积分过程中不考虑 IMU 的噪声，直接算就可以）

ESKF 的整体流程如下：当 IMU 测量数据到达时，把它积分后，放入名义状态变量中（得到粗略的状态变量 $[p, v, R, b_g, b_a, g]^T$）。由于这种做法没有考虑噪声，其结果自然会快速漂移，于是把误差部分作为误差变量。在运动过程中，名义状态随着 IMU 数据进行递推，误差状态则受到高斯噪声影响而变大。此时 ESKF 的误差状态的均值和协方差，会描述误差状态扩大的具体数值（视为高斯分布）。

此外，ESKF 的更新过程需要依赖 IMU 以外的传感器观测。更新过程中，利用传感器数据，更新误差状态的后验均值与协方差；随后可以把这部分误差合入到名义状态变量中，并把 ESKF 置零，这样就完成了一次预测——更新的循环。

下面我们来推导 ESKF 的两个方差：运动方程和状态方程。设 ESKF 的真值状态为 $x_t = \begin{bmatrix} p_t, v_t, R_t, b_{gt}, b_{at}, g_t \end{bmatrix}^T$，下标 $t$ 表示 true，即真值状态，这个状态随时间改变，可以记作 $x_t(t)$。

在连续时间上，记 IMU 的读数为 $\tilde{\omega},\tilde{a}$，那么根据 IMU 运动学积分，可以写出状态变量导数相对于观测量之间的关系式（**真值的运动方程**）：

$$
\begin{align*}
\dot{p}_t &= v_t \tag{4.a} \\
\dot{v}_t &= R_t(\tilde{a} - b_{at} - \eta_a) + g_t,\tag{4.b} \\
\dot{R}_t &= R_t(\tilde{\omega} - b_{gt} - \eta_g)^{\wedge} \tag{4.c} \\
\dot{b}_{gt} &= \eta_{bg} \tag{4.d} \\
\dot{b}_{at} &= \eta_{ba} \tag{4.e} \\
\dot{g}_t &= 0 \tag{4.f}
\end{align*}
$$

其中 4.d 和 4.e 是我们前面介绍的 IMU 噪声模型中的零偏，零偏本身并不是高斯白噪声，但是我们认为零偏的变化是白噪声，也就是每个时刻零偏是变大还是变小、变多少是高斯分布的。这一系列公式描述了 IMU 运动模型的基本条件，方便后面推导误差状态变量关于时间的导数。

下面推导误差状态方程，先定义误差状态变量为：

$$
\begin{align*}
p_t &= p + \delta p \tag{5.a} \\
v_t &= v + \delta v \tag{5.b} \\
R_t &= R\delta R \text{ 或 } q_t = q\delta q \tag{5.c} \\
b_{gt} &= b_g + \delta b_g \tag{5.d} \\
b_{at} &= b_a + \delta b_a \tag{5.e} \\
g_t &= g + \delta g \tag{5.f}
\end{align*}
$$

其中，不带下标的就是名义状态变量，带 $\delta$ 的就是误差状态变量，带下标 $t$ 的就是真值，这个方程描述了真值 = 名义状态变量+误差状态变量。**名义状态变量的运动方程与真值相同（如公式（4）），只是不必考虑噪声，因为噪声我们在误差状态方程中考虑。**

在误差状态方程，也就是式（5）中，我们在等式两侧同时对时间求导，得到对应的时间导数表达式为：

$$
\begin{align*}
\delta\dot{p} &= \delta v \tag{6.a} \\
\delta\dot{b}_g &= \eta_g \tag{6.b} \\
\delta\dot{b}_a &= \eta_a \tag{6.c} \\
\delta\dot{g} &= 0 \tag{6.d}
\end{align*}
$$

式（5.b）和式（5.c）由于和 $\delta R$ 有关系，形式稍微复杂一些，下面给出单独的推导过程。

### 误差状态的旋转项

对式（5.c）两侧求时间导数，可得：

$$
\dot{R}_t = \dot{R}\mathrm{Exp}(\delta\theta) + R\dot{\mathrm{Exp}(\delta\theta)}, \\
\underline{\underline{\text{4.c}}} \quad R_t (\tilde{\omega} - b_{gt} - \eta_g)^{\wedge}
$$

其中 $\delta \theta$ 为误差状态变量 $\delta R$ 对应的李代数。

注意，式（4）右侧的 $\dot{\text{Exp}(\delta \theta)}$ 满足：

$$
\dot{\mathrm{Exp}(\delta\theta)} = \mathrm{Exp}(\delta\theta)\delta\dot{\theta}^{\wedge}
$$

因此式（4）的 **第一个式子** 可以写成：

$$
\dot{R}\mathrm{Exp}(\delta\theta) + R\dot{\mathrm{Exp}(\delta\theta)} = R(\tilde{\omega} - b_g)^{\wedge}\mathrm{Exp}(\delta\theta) + R\mathrm{Exp}(\delta\theta)\delta\dot{\theta}^{\wedge}
$$

这里 $\dot{R}= R(\tilde{\omega} - b_g)^{\wedge}$，因为 $R$ 是 **名义状态变量（只考虑零偏，不考虑噪声）**，和真值 $\dot{R_t}$ 有所区别。

式（4）的 **第二个式子** 可以写成：

$$
R_t (\tilde{\omega} - b_{gt} - \eta_g)^{\wedge} = R\mathrm{Exp}(\delta\theta) (\tilde{\omega} - b_{gt} - \eta_g)^{\wedge}
$$

这是因为 $R_t = R\text{Exp}(\delta \theta)$，也就是真值 = 名义状态变量+误差状态变量

比较式（6）和式（7）的右侧，它们相等，将 $\delta \dot{\theta}^{\wedge}$ 移到一侧，约掉两侧左边的 $R$，整理类似项，得到：

$$
\mathrm{Exp}(\delta\theta)\delta\dot{\theta}^{\wedge} = \mathrm{Exp}(\delta\theta) (\tilde{\omega} - b_{gt} - \eta_g)^{\wedge} - (\tilde{\omega} - b_g)^{\wedge} \mathrm{Exp}(\delta\theta)
$$

注意，$\text{Exp}(\delta \theta)$ 本身是一个 $SO(3)$ 矩阵，利用 $SO(3)$ 上的伴随性质：

$$
\phi^{\wedge}R = R(R^T\phi)^{\wedge}
$$

所以：$(\tilde{\omega}-b_g)^{\wedge}\text{Exp}(\delta \theta)=\text{Exp}(\delta \theta)(\text{Exp}(-\delta \theta)(\tilde{\omega}-b_g))^{\wedge}$，也就是将 $(\tilde{\omega}-b_g)$ 看作 $\phi$，将 $\text{Exp}(\delta \theta)$ 看作 $R$。

所以：

$$
\begin{aligned}
\mathrm{Exp}(\delta\theta)\delta\dot{\theta}^{\wedge} &= \mathrm{Exp}(\delta\theta) (\tilde{\omega} - b_{gt} - \eta_g)^{\wedge} - \mathrm{Exp}(\delta\theta) (\mathrm{Exp}(-\delta\theta) (\tilde{\omega} - b_g))^{\wedge} \\
&= \mathrm{Exp}(\delta\theta) \left [(\tilde{\omega} - b_{gt} - \eta_g)^{\wedge} - (\mathrm{Exp}(-\delta\theta) (\tilde{\omega} - b_g))^{\wedge}\right] \\
&\approx \mathrm{Exp}(\delta\theta) \left [(\tilde{\omega} - b_{gt} - \eta_g)^{\wedge} - ((\mathrm{I} - \delta\theta^{\wedge})(\tilde{\omega} - b_g))^{\wedge}\right] \\
&= \mathrm{Exp}(\delta\theta) \left [b_g - b_{gt} - \eta_g + \delta\theta^{\wedge}\tilde{\omega} - \delta\theta^{\wedge}b_g\right]^{\wedge} \\
&= \mathrm{Exp}(\delta\theta) \left [(-\tilde{\omega} + b_g)^{\wedge}\delta\theta - \delta b_g - \eta_g\right]^{\wedge}
\end{aligned}
$$

第一个等式就是利用了 $SO(3)$ 上的伴随性质，第三个约等于是用来泰勒展开的一阶近似，第四个等号是直接乘法展开，第五个等式是用来外积交换要添加负号的性质。

将式（10）两侧消除 $\text{Exp}(\delta \theta)$，得到：

$$
\delta\dot{\theta} \approx -(\tilde{\omega} - b_g)^{\wedge}\delta\theta - \delta b_g - \eta_g
$$

### 误差状态的速度项

对式（5.b）（$v_t = v+\delta v$）两侧求时间的导数，就可以得到 $\delta \dot{v}$ 的表达式。

等式左侧为：

$$
\begin{aligned}
\dot{v}_t &= R_t(\tilde{a} - b_{at} - \eta_a) + g_t \\
&= R\mathrm{Exp}(\delta\theta)(\tilde{a} - b_a - \delta b_a - \eta_a) + g + \delta g \\
&\approx R(I + \delta\theta^{\wedge})(\tilde{a} - b_a - \delta b_a - \eta_a) + g + \delta g \\
&\approx R\tilde{a} - Rb_a - R\delta b_a - R\eta_a + R\delta\theta^{\wedge}\tilde{a} - R\delta\theta^{\wedge}b_a + g + \delta g \\
&= R\tilde{a} - Rb_a - R\delta b_a - R\eta_a - R\tilde{a}^{\wedge}\delta\theta + Rb_a^{\wedge}\delta\theta + g + \delta g
\end{aligned}
$$

将带下标 $t$ 的真值，全部用名义状态变量+误差状态变量来代替。第二行到第三行，是指数的一阶近似；**第三行到第四行，忽略了 $\delta \theta^{\wedge}$ 与 $\delta b_a,\eta_a$ 相乘的二阶小量。**

等式右侧为：

$$
\dot{v} + \delta \dot{v} = R(\tilde{a} - b_a) + g + \delta \dot{v}
$$

因为式（12）和式（13）相等，所以可以得到：

$$
\delta \dot{v}=-R(\tilde{a}-b_a)^{\wedge}\delta \theta-R\delta b_a-R\eta_a+\delta g
$$

由于 $\eta_a$ 是加速度计测量的零均值白噪声，它乘以任意旋转矩阵之后，仍然是一个零均值白噪声，而且 $R^TR = I$，**容易证明其协方差矩阵也不变**。所以，式（14）可以简化为：

$$
\delta \dot{v}=-R(\tilde{a}-b_a)^{\wedge}\delta \theta-R\delta b_a-\eta_a+\delta g
$$

至此，总结起来，可以将 **误差变量的运动方程** 整理如下：

$$
\begin{aligned}
\delta\dot{p} &= \delta v, \\
\delta\dot{v} &= -R(\tilde{a} - b_a)^{\wedge}\delta\theta - R\delta b_a - \eta_a + \delta g, \\
\delta\dot{\theta} &= -(\tilde{\omega} - b_g)^{\wedge}\delta\theta - \delta b_g - \eta_g, \\
\delta\dot{b}_g &= \eta_{bg}, \\
\delta\dot{b}_a &= \eta_{ba}, \\
\delta\dot{g} &= 0.
\end{aligned}
$$

## 离散时间的 ESKF 运动方程

根据 IMU 积分公式，从连续时间状态方程推出离散时间的状态方程，只需要设定时间间隔 $\Delta t$ 即可，名义状态变量的离散时间运动方程可以写为（依旧是只考虑 IMU 测量的零偏，不考虑噪声）：

$$
\begin{aligned}
p(t + \Delta t) &= p(t) + v\Delta t+ \frac{1}{2}(R(\tilde{a} - b_a)) \Delta t^2 + \frac{1}{2}g\Delta t^2 \\
v(t + \Delta t) &= v(t) + R(\tilde{a} - b_a)\Delta t + g\Delta t \\
R(t + \Delta t) &= R(t) \mathrm{Exp} ((\tilde{\omega} - b_g)\Delta t) \\
b_g(t + \Delta t) &= b_g(t) \\
b_a(t + \Delta t) &= b_a(t) \\
g(t + \Delta t) &= g(t)
\end{aligned}
$$

误差变量的状态方程的离散形式与名义状态十分相似

$$
\begin{aligned}
\delta p(t + \Delta t) &= \delta p + \delta v \Delta t \\
\delta v(t + \Delta t) &= \delta v + (-R(\tilde{a} - b_a)^{\wedge} \delta\theta - R\delta b_a + \delta g) \Delta t - \eta_v \\
\delta\theta(t + \Delta t) &= \mathrm{Exp} (-(\tilde{\omega} - b_g) \Delta t) \delta\theta - \delta b_g \Delta t - \eta_{\theta} \\
\delta b_g(t + \Delta t) &= \delta b_g + \eta_g \\
\delta b_a(t + \Delta t) &= \delta b_a + \eta_a \\
\delta g(t + \Delta t) &= \delta g
\end{aligned}
$$

具体来说就是，$t+\Delta t$ 时刻的误差，等于 $t$ 时刻的误差 + $\Delta t$ 时间段内误差的累积，**累积误差** 等于误差变量的导数（也就是变化率）乘以时间 $\Delta t$。参考式（16）的误差变量的运动方程应该比较好理解。

需要注意的是：

1、式（18）中右侧部分省略了括号里的(t)以简化公式，比如应该是 $\delta p(t + \Delta t) = \delta p(t) + \delta v \Delta t$；

2、关于旋转部分的积分，可以将式（16）中第三个方程，看成关于 $\delta \theta$ 的微分方程然后求解，求解过程类似对角速度进行积分；

3、噪声项并不参与递推，需要把它们单独归入噪声部分。**连续时间的噪声项可以视为随机过程的能量谱密度，而离散时间下的噪声变量就是我们日常看到的随机变量**。这些噪声随机变量的标准差可以列写为：

$$
\sigma(\eta_v) = \Delta t \sigma_a (k), \quad \sigma(\eta_\theta) = \Delta t \sigma_g (k), \quad \sigma(\eta_g) = \sqrt{\Delta t} \sigma_{bg}, \quad \sigma(\eta_a) = \sqrt{\Delta t} \sigma_{ba}
$$

至此，给出了误差状态变量在 ESKF 中进行 IMU 递推的过程，对应卡尔曼滤波器的状态方程，描述了误差状态变量如何随时间变化。

为了让滤波器收敛，还需要外部的观测对卡尔曼滤波器进行修正，也就是所谓的组合导航。下面，以融合 GNSS 观测为例，介绍如何在 ESKF 中荣光和这些观测数据，形成一个收敛的卡尔曼滤波器。

## ESKF 的运动过程

根据上述讨论，可以写出 ESKF 的运动过程（也就是式（18）的误差状态方程，它也描述了误差状态的变化规律，所以也可称为运动方程）。可以将式（18）整体抽象的记为：

$$
\delta x_{k+1} = f(\delta x_k) + w, \quad w \sim \mathcal{N}(0, Q)
$$

其中 $w$ 为噪声，按照前面的定义，$Q$ 为：

$$
Q = \mathrm{diag}(0_3, \mathrm{Cov} (\eta_v), \mathrm{Cov} (\eta_\theta), \mathrm{Cov} (\eta_g), \mathrm{Cov} (\eta_a), 0_3)
$$

两侧为 0 是因为第一个和最后一个方程本身没有噪声导致的。因为平移的噪声本质上是由速度引起的，噪声在速度方程里；而重力加速度一般认为是不变的。

为了保持与 EKF 的负号统一，同样将运动方程线性化近似（一阶近似）：

$$
\delta x(t + \Delta t) = \underbrace{f(\delta x(t))}_\text{= 0} + F\delta x(t) + w
$$

这里给 $f(\delta x(t))$ 下面加上“= 0”的标注，是因为 ESKF 每次更新之后，会把误差归零（Reset），所以我们默认上一时刻的误差估计值的均值为 0。

所以，我们发现，如果 $f(\delta x(t))$ 这一项为 0 的话，式（22）就可以写成 $\delta x(t + \Delta t) = F\delta x(t)$（不考虑噪声的情况下），其中 $F$ 为线性化后的雅可比矩阵：

$$
\mathbf{F} =
\begin{bmatrix}
\mathbf{I} & \mathbf{I}\Delta t & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I} & -R(\tilde{a} - b_a)^\wedge \Delta t & \mathbf{0} & -R \Delta t & \mathbf{I} \Delta t \\
\mathbf{0} & \mathbf{0} & \text{Exp}(-(\tilde{\omega} - b_g)\Delta t) & -\mathbf{I}\Delta t & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I}
\end{bmatrix}
$$

而我们观察原先的式（18），发现方程（18）本身就可以写成如下的矩阵形式，**在重置前一时刻的误差为 0 的情况下，这个误差状态方程本身就是线性的。**

$$
\begin{bmatrix}\delta p(t+\Delta t)\\ \delta v(t+\Delta t)\\ \delta \theta(t+\Delta t)\\ \delta bg(t+\Delta t)\\ \delta b_a(t+\Delta t)\\ \delta g(t+\Delta t) \end{bmatrix}=
\begin{bmatrix}
\mathbf{I} & \mathbf{I}\Delta t & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I} & -R(\tilde{a} - b_a)^\wedge \Delta t & \mathbf{0} & -R \Delta t & \mathbf{I} \Delta t \\
\mathbf{0} & \mathbf{0} & \text{Exp}(-(\tilde{\omega} - b_g)\Delta t) & -\mathbf{I}\Delta t & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I}
\end{bmatrix}
\begin{bmatrix} \delta p(t)\\ \delta v(t)\\ \delta \theta(t)\\ \delta bg(t)\\ \delta b_a(t)\\ \delta g(t) \end{bmatrix}
$$

在此基础上，执行 ESKF 的预测过程。预测过程包括对名义状态的预测（IMU 积分）以及对误差状态的预测：

$$
\delta x_{\text{pred}} = F\delta x, \\
P_{\text{pred}} = FPF^{\text{T}} + Q
$$

由于 ESKF 的误差状态在每次更新以后，都会被重置为 $\delta x = 0$，因此式（25）中均值的预测没有太大意义；协方差部分描述了整个误差估计的分布情况。从直观意义上来看，运动方程的噪声协方差中增加了 $Q$ 项，可以看作增大的过程。

这里的 $P$ 是卡尔曼滤波迭代过程中产生的，是一个 $18×18$ 的协方差矩阵，用来描述 18 个误差变量 $\delta x = [\delta p, \delta v, \delta \theta, \delta b_g, \delta b_a, \delta g]^T$ 的不确定性的。

1、初始时，我们需要给 $P$ 一个初始值 $P_0$，代码中是 $10^{-4} \cdot I$，表示初始时刻我们认为误差接近于 0，且非常自信（或者通过静态初始化计算出来的方差来赋值）。

2、但是如果仅使用 IMU 积分，也就是式（24）来预测更新我们的误差状态，积分会累积误差，所以我们对误差变量的不确定性是增加的，$P$ 会变大

$$
P_{\text{pred}} = \underbrace{F P F^\top}_{\text{第一部分：旧误差的传递}} + \underbrace{Q}_{\text{第二部分：新噪声的注入}}
$$

3、更新修正，当 GNSS 数据来了，我们获得了一个观测值，这可以帮助我们减少不确定性，$P$ 会变小

$$
P =(I-KH)P_{pred}
$$

## ESKF 的更新过程

前面介绍的是 ESKF 的运动过程（仅通过 IMU 数据来更新误差变量，误差会不断累积），现在考虑更新过程。假设一个抽象的传感器能够对状态变量 $x$ 产生观测，其观测方程为抽象的 $h$，那么可以写为：

$$
z = h(x) + v, \quad v \sim \mathcal{N}(0, V)
$$

在传统 EKF 中，可以直观的对观测方程线性化，求出观测方程相对于状态变量的雅可比矩阵，进而更新卡尔曼滤波器。但是在 ESKF 中，当前拥有名义状态 $x$ 的估计以及误差状态 $\delta x$ 的估计，且希望更新的是误差状态，因此要计算观测方程相对于误差状态的雅可比矩阵：

$$
H =\frac{\partial {h}}{\partial {\delta x}} \bigg|_{x_{pred}}
$$

然后计算卡尔曼增益，进而计算 **误差状态的更新过程**：

$$
\begin{aligned}
K &= P_{\text{pred}}H^{\text{T}}(HP_{\text{pred}}H^{\text{T}} + V)^{-1} \\
\delta x &= K(z - h(x_{\text{pred}})) \\
x &= x_{\text{pred}} + \delta x \\
P &= (I - KH)P_{\text{pred}}
\end{aligned}
$$

其中 $z$ 是观测方程结果，是 GNSS 的读数，GNSS 会给当前位置一个观测；$h(x_{pred})$ 表示如果车辆在 $x_{pred}$ 处，则 GNSS 读数应该是多少，这个 $x_{pred}$ 就是根据 IMU 运动学方程推算出来的，如果 $x_{pred}$ 很准的话，$z$ 和 $h(x_{pred})$ 应该很接近，那么预测的误差 $\delta x$ 应该也就很小（具体多小还取决于 $x_{pred}$ 的协方差矩阵 $P_{pred}$，$P_{pred}$ 和 $V$ 决定了多大程度上相信 $z$ 的观测结果，因为用 $z-h(x_{pred})$ 来衡量误差，本质上是假设 $z$ 是准确的），也就说明了 $x_{pred}$ 和真值 $x$ 比较接近。

大部分的观测数据是对名义状态的观测（这里我们是用名义状态来估计真实状态，因为 GNSS 一定是基于真实状态来观测的，只是我们不知道真实状态，所以用名义状态来估计，如果对名义状态的观测结果 $h(x_{pred})$ 和 GNSS 读数 $z$ 接近，说明名义状态和真实状态接近）。此时 $H$ 可以通过链式法则来生成：

$$
H = \frac{\partial h}{\partial x} \frac{\partial x}{\partial \delta x}
$$

第一项可以通过将观测方程线性化得到；第二项，根据式（5.a-5.f）对状态变量的定义，可以得到：

$$
\frac{\partial x}{\partial \delta x} = \text{diag}\left(I_3, I_3, \frac{\partial \text{Log}\left(R\left(\text{Exp}\left(\delta\theta\right)\right)\right)}{\partial \delta\theta}, I_3, I_3, I_3\right)
$$

其他几项都是平凡的，求导为单位矩阵。只有旋转部分，因为 $\delta \theta$ 定义为 $R$ 的右乘，用右乘的 BCH 即可：

$$
\frac{\partial \text{Log}(R(\text{Exp}(\delta\theta)))}{\partial \delta\theta} = J_r^{-1}(R)
$$

## ESKF 的误差状态后续处理

经过预测和更新过程之后，我们修正了误差状态的估计。接下来，只需要把误差状态加到名义状态中，更新名义状态对真实状态的估计，然后重置 ESKF 即可。误差状态加到名义状态可以写为：

$$
p_{k+1} = p_{k} + \delta p_{k} \\
v_{k+1} = v_{k} + \delta v_{k} \\
R_{k+1} = R_{k}\text{Exp}(\delta\theta_{k}) \\
b_{g, k+1} = b_{g, k} + \delta b_{g, k} \\
b_{a, k+1} = b_{a, k} + \delta b_{a, k} \\
g_{k+1} = g_{k} + \delta g_{k}  
$$

然后是 ESKF 的重置，重置分为均值部分和协方差部分，均值部分可以简单的实现为：

$$
\delta x = 0
$$

表示经过一次 GNSS 观测的修正，我们认为将两次 GNSS 观测时间间隔内 IMU 的累积误差都消除干净了。由于均值被重置了，所以之前描述的是关于 $x_k$ 切空间中的协方差，而现在描述的是 $x_{k+1}$ 中的协方差。重置会带来一些微小的差异，主要影响旋转部分。

事实上，在重置前，卡尔曼滤波器刻画了 $x_{pred}$ 切空间处的一个高斯分布 $\mathcal{N}(\delta x, P)$，而重置之后，应该刻画 $x_{pred}+\delta x$ 处的一个 $\mathcal{N}(0, P_{reset})$。这对本身就是矢量的状态是没有差别的，但对于旋转变量来说，它们的切空间零点发生了变化，对应的切平面也发生了变化（**因为旋转空间（流形）是弯曲的，你把“坐标原点”搬家了，描述“不确定性范围”的椭球形状也得跟着变一下。**），所以在数学上，需要对此区分。

设重置前的名义旋转估计为 $R_k$，误差状态为 $\delta \theta$（这是一个随机变量），卡尔曼滤波器的增量计算结果为 $\delta \theta_k$（这是卡尔曼滤波器对随机变量 $\delta \theta$ 的最优估计），则重置之后的名义旋转部分为 $R_k\text{Exp}(\delta \theta_k)= R^{+}$，误差状态为 $\delta \theta^{+}$，由于误差状态被重置了，这里的 $\delta \theta^{+}$ 其实等于 0。但是我们关系的并不是它们的直接取值，而是 $\delta \theta^{+}$ 和 $\delta \theta$ 的线性化关系（**也就是重置这一过程中误差状态分布是怎么变换的**），把实际的重置过程写出来：

$$
R_t = R^{+}\text{Exp}(\delta\theta^{+}) = R_{k}\text{Exp}(\delta\theta_{k})\text{Exp}(\delta\theta^{+}) = R_{k}\text{Exp}(\delta\theta).
$$

不难得到：

$$
\text{Exp}(\delta\theta^{+}) = \text{Exp}(-\delta\theta_{k})\text{Exp}(\delta\theta)
$$

这里 $\delta \theta$ 为小量，利用线性化后的 BCH 公式，可以得到

$$
\delta\theta^{+} = -\delta\theta_{k} + \delta\theta - \frac{1}{2}\hat{\delta\theta}_{k}\delta\theta + o((\delta\theta)^{2})
$$

这是因为在 $\mathfrak{so}(3)$ 李代数中，当 $\phi_1$ 和 $\phi_2$ 都是 **小量** 时，根据 BCH 公式的二阶近似（保留到二次项）：

$$
\ln(\text{Exp}(\phi_1)\text{Exp}(\phi_2))^\vee \approx \phi_1 + \phi_2 + \frac{1}{2}[\phi_1, \phi_2]
$$

其中 $[\phi_1, \phi_2]$ 是李括号 (Lie Bracket)。在旋转向量空间中，李括号等于叉积（或者写成反对称矩阵乘法）：

$$
[\phi_1, \phi_2] = \phi_1^\wedge \phi_2
$$

所以，BCH 近似公式可以写成：$\phi_{new} \approx \phi_1 + \phi_2 + \frac{1}{2} \phi_1^\wedge \phi_2$

所以：

$$
\frac{\partial \delta\theta^{+}}{\partial \delta\theta} \approx I - \frac{1}{2}\delta\theta_{k}^{\wedge}
$$

式（41）表面重置前后的误差状态（随机变量）相差一个旋转方面的小雅可比矩阵，记作 $J_{\theta}= I-\frac{1}{2}\delta \theta_k^{\wedge}$，把这个小雅可比矩阵放到整个状态变量维度下，并保持其他部分为单位阵，可以得到一个完整的雅可比矩阵：

$$
J_k = \text{diag}(I_3, I_3, J_\theta, I_3, I_3, I_3)
$$

因此，在把误差状态的均值归零的同时，它们的协方差矩阵也应该进行线性变换（旧中心到新中心的变换）：

$$
P_{reset}= J_kPJ_k^T
$$

不过由于 $\delta \theta_k$ 并不大，这里的 $J_k$ 依然十分接近单位阵，所以很多资料里面并不处理这一项，而是直接把前面估计的 $P$ 矩阵作为下一时刻的起点。

该问题的实际意义是做了切空间投影，即把一个切空间中的高斯分布投影到另一个切空间中。在 ESKF 中，两者没有明显差异，但后文的迭代卡尔曼滤波器（IEKF）还会牵扯到在观测过程中多次变换切空间，所以这里可以先熟悉一下这个过程和原理。


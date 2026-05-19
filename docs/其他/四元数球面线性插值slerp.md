# Spherical Linear Interpolation

球面线性插值（Spherical Linear Interpolation，简称 Slerp）是计算机图形学、动画和机器人运动学（SLAM）中极其核心的算法。它由 Ken Shoemake 在 1985 年提出，主要用于在两个单位向量（或单位四元数）之间进行平滑过渡。

要理解 Slerp，我们首先要明白为什么普通的线性插值（Lerp）不够用。

- **Lerp 的问题：** 如果直接对两个向量进行线性插值 $v(t) = (1-t)v_1 + t v_2$，插值出来的点会走直线穿过球体内部，导致向量长度缩短（不再是单位向量）。即使将其强行归一化（NLerp），插值的**角速度也是不均匀的**（两头慢，中间快）。
- **Slerp 的优势：** Slerp 保证了插值路径严格贴合球面（最短的大圆弧），并且**角速度是恒定的**，非常适合处理旋转。

下面我们来详细推导一下Slerp的原理：

## 一、 几何定义与已知条件

假设我们有两个单位向量 $v_1,v_2 \in \mathbb{R}^4$ （即 $\|v_1\| = 1, \|v_2\| = 1$）。它们之间的夹角为 $\Omega$（通过点乘求得：$\cos\Omega = v_1 \cdot v_2$）。我们引入插值参数 $t \in [0, 1]$。当 $t=0$ 时处于 $v_1$，当 $t=1$ 时处于 $v_2$。

我们的**目标**是求出插值向量 $v(t)$，它必须满足以下三个物理/几何条件：

1. **共面性：** $v(t)$ 必须在 $v_1$ 和 $v_2$ 构成的平面上。
2. **单位长度：** $\|v(t)\| = 1$（始终在球面上）。
3. **恒定角速度：** $v(t)$ 与 $v_1$ 的夹角必须严格等于 $t\Omega$；与 $v_2$ 的夹角必须严格等于 $(1-t)\Omega$。

## 二、 数学推导步骤

### 第一步：建立线性组合方程

因为 $v(t)$ 与 $v_1, v_2$ 共面，所以它可以表示为这两个向量的线性组合。我们设未知系数为 $c_1$ 和 $c_2$：

$$
v(t) = c_1 v_1 + c_2 v_2
$$

我们的任务就是解出 $c_1$ 和 $c_2$ 到底是多少。

### 第二步：利用点乘（夹角）列出方程组

根据条件 3，我们已知 $v(t)$ 与基准向量的夹角。由于它们都是单位向量，点乘等于夹角的余弦值：

1. 与 $v_1$ 的点乘：$v_1 \cdot v(t) = \cos(t\Omega)$
2. 与 $v_2$ 的点乘：$v_2 \cdot v(t) = \cos((1-t)\Omega)$

将第一步的线性组合代入上述两个等式中：

$$
v_1 \cdot (c_1 v_1 + c_2 v_2) = c_1(v_1 \cdot v_1) + c_2(v_1 \cdot v_2) = \cos(t\Omega)
$$

$$
v_2 \cdot (c_1 v_1 + c_2 v_2) = c_1(v_2 \cdot v_1) + c_2(v_2 \cdot v_2) = \cos((1-t)\Omega)
$$

因为 $v_1 \cdot v_1 = 1$，$v_2 \cdot v_2 = 1$，且 $v_1 \cdot v_2 = \cos\Omega$，方程组化简为：

**(方程 1):** $c_1 + c_2 \cos\Omega = \cos(t\Omega)$

**(方程 2):** $c_1 \cos\Omega + c_2 = \cos((1-t)\Omega)$

### 第三步：解方程组求 $c_2$

由 (方程 1) 得到 $c_1$ 的表达式：

$$
c_1 = \cos(t\Omega) - c_2 \cos\Omega
$$

将其代入 (方程 2) 中消去 $c_1$：

$$
(\cos(t\Omega) - c_2 \cos\Omega)\cos\Omega + c_2 = \cos((1-t)\Omega)
$$

$$
\cos(t\Omega)\cos\Omega - c_2 \cos^2\Omega + c_2 = \cos((1-t)\Omega)
$$

提取 $c_2$：

$$
c_2 (1 - \cos^2\Omega) = \cos((1-t)\Omega) - \cos(t\Omega)\cos\Omega
$$

根据三角恒等式 $1 - \cos^2\Omega = \sin^2\Omega$，以及余弦的差角公式 $\cos(A-B) = \cos A \cos B + \sin A \sin B$ 展开右侧的第一项：

$$
c_2 \sin^2\Omega = (\cos\Omega \cos(t\Omega) + \sin\Omega \sin(t\Omega)) - \cos(t\Omega)\cos\Omega
$$

神奇的事情发生了，$\cos\Omega \cos(t\Omega)$ 相互抵消了：

$$
c_2 \sin^2\Omega = \sin\Omega \sin(t\Omega)
$$

只要 $\sin\Omega \neq 0$（即两向量不重合或反向），等式两边同除以 $\sin\Omega$：

$$
c_2 = \frac{\sin(t\Omega)}{\sin\Omega}
$$

### 第四步：解方程组求 $c_1$

把求出的 $c_2$ 代回刚才的 $c_1$ 表达式：

$$
c_1 = \cos(t\Omega) - \frac{\sin(t\Omega)}{\sin\Omega} \cos\Omega
$$

通分：

$$
c_1 = \frac{\sin\Omega \cos(t\Omega) - \cos\Omega \sin(t\Omega)}{\sin\Omega}
$$

根据正弦的差角公式 $\sin(A-B) = \sin A \cos B - \cos A \sin B$，分子可以完美闭合：

$$
c_1 = \frac{\sin(\Omega - t\Omega)}{\sin\Omega} = \frac{\sin((1-t)\Omega)}{\sin\Omega}
$$

### 第五步：得出最终的 Slerp 公式

将 $c_1$ 和 $c_2$ 代回第一步的线性组合中，我们得到了教科书上最经典的 Slerp 公式：

$$
v(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} v_1 + \frac{\sin(t\Omega)}{\sin\Omega} v_2
$$

------

## 三、 工程实现中的关键防坑点 (Edge Cases)

在代码实现当中（尤其是针对四元数旋转时），直接套用上述公式会引发一些灾难性的 Bug，必须加入以下处理：

**1. 除零危险（夹角极小）：**

当 $v_1$ 和 $v_2$ 非常接近时，$\Omega \approx 0$，此时 $\sin\Omega \approx 0$，公式会导致除以零或产生极大的浮点数精度误差。

- **解决方案：** 在代码中判断 $v_1 \cdot v_2$（即 $\cos\Omega$）是否非常接近 1（例如 `> 0.9995`）。如果是，则退化为普通的线性插值并归一化（NLerp），因为在极小角度下，NLerp 和 Slerp 的效果几乎无异，且计算更快。

**2. 最短路径法则（针对四元数）：**

在 3D 旋转中，四元数 $q$ 和 $-q$ 表示的是同一个三维空间旋转。但如果两个四元数的夹角 $\Omega > 90^\circ$（即 $q_1 \cdot q_2 < 0$），Slerp 会选择绕远路（扫过大于 $180^\circ$ 的圆弧）去插值。

- **解决方案：** 如果发现点乘结果为负数，必须将其中一个四元数反向（比如令 $q_2 = -q_2$），这保证了点乘为正，从而使得插值始终走最短的圆弧路径。

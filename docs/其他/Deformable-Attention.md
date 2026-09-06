# Deformable Attention

Deformable Attention可以理解为：普通attention是“一个query和所有空间位置的key计算相关性，再和对应的value根据相关性加权求和”，Deformable Attention是“一个query先预测少量采样点的位置和权重，只从这些点插值取value再加权求和”。原始定义来自 Deformable DETR: Deformable Transformers for End-to-End Object Detection。

1、先和普通Attention对比：普通的多头attention对某个query q的计算大致是：
$$
\mathrm{Attn}(q)=\sum_{k\in \Omega_k} A_{qk}V_k,\quad A_{qk}=\mathrm{softmax}(Q_qK_k^T/\sqrt d)
$$
如果输入是一张特征图（必须先展平为特征序列），$\Omega_k$就是所有$H\times W$个像素/patch/token。假设特征图是$200 \times 200$，一个query就要看40000个位置；如果encoder里每个像素都当query，复杂度接近$O((HW)^2d)$，也就是token数量的平方，再乘以注意力层的通道数量d。

Deformable Attention把这个过程改成：每个query不再和所有key做dense attention，而是预测K个偏移点，只在这些位置采样，例如每个head每个尺度采K=4个点，M=8个heads，L=4个尺度，那么每个query总共只看$4\times 8 \times 4=128$个采样点，而不是几万甚至几十万个token。

2、单尺度Deformable Attention的公式：论文中单尺度版本写成：
$$
\mathrm{DeformAttn}(z_q,p_q,x)
=\sum_{m=1}^{M} W_m
\left[
\sum_{k=1}^{K}
A_{mqk}\cdot W'_m x(p_q+\Delta p_{mqk})
\right]
$$
这里z_q是第q个query的内容特征，p_q是它的reference point，x是输入特征图，m是attention head，k是采样点编号，$\Delta p_{mqk}$是这个head的第$k$个采样偏移，$A_{mqk}$是这个采样点的权重。论文说明$K << HW$，采样点坐标通常是小数，所以用双线性插值取$x(p_q+\Delta p)$






































































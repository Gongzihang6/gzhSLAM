<p class="theme-switcher-title">
  🎨 换个颜色，换个心情
</p>

<div class="color-picker-container">
  <button class="color-btn" data-color="red" style="background-color: #ef5350;">red</button>
  <button class="color-btn" data-color="pink" style="background-color: #ec407a;">pink</button>
  <button class="color-btn" data-color="purple" style="background-color: #ab47bc;">purple</button>
  <button class="color-btn" data-color="indigo" style="background-color: #5c6bc0;">indigo</button>
  <button class="color-btn" data-color="blue" style="background-color: #42a5f5;">blue</button>
  <button class="color-btn" data-color="cyan" style="background-color: #26c6da;">cyan</button>
  <button class="color-btn" data-color="teal" style="background-color: #26a69a;">teal</button>
  <button class="color-btn" data-color="green" style="background-color: #66bb6a;">green</button>
  <button class="color-btn" data-color="orange" style="background-color: #ffa726;">orange</button>
  <button class="color-btn" data-color="brown" style="background-color: #8d6e63;">brown</button>
  <button class="color-btn" data-color="grey" style="background-color: #bdbdbd;">grey</button>
</div>

<script>
  var buttons = document.querySelectorAll('.color-btn');
  var body = document.querySelector('body');
  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var color = this.getAttribute('data-color');
      body.setAttribute('data-md-color-primary', color);
      localStorage.setItem('user-color-preference', color);
    });
  });
  var savedColor = localStorage.getItem('user-color-preference');
  if (savedColor) { body.setAttribute('data-md-color-primary', savedColor); }
</script>

# SLAM学习之旅

> **从理论到代码，构建完整的 SLAM 知识体系。**
>
> *JourneyToSLAM: From Theory to Practice - A Learning Journey*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/gaoxiang12/slambook2)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-14%2F17-blue.svg)](https://isocpp.org/)
[![SLAM](https://img.shields.io/badge/SLAM-Visual_Odometry-orange)](https://github.com/gongzihang6/slambook2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 目录结构

```bash
gzhSLAM
├── README.md
├── docs
│   ├── index.md
│   ├── javascripts
│   │   └── mathjax.js
│   ├── stylesheets
│   │   └── extra.css
│   ├── 其他
│   │   ├── Cholesky分解.md
│   │   ├── cvparallel_for_并行.md
│   │   ├── 矩阵分解.md
│   │   └── 矩阵相似变换的意义.md
│   ├── 自动驾驶与机器人中的SLAM技术
│   │   ├── ch2基础数学知识回顾
│   │   │   ├── 卡尔曼滤波器的各种推导.md
│   │   │   └── 旋转的表示.md
│   │   ├── ch3惯性导航与组合导航
│   │   │   ├── IMU的静止初始化.md
│   │   │   ├── IMU运动学.md
│   │   │   └── 误差状态卡尔曼滤波器ESKF.md
│   │   ├── ch5基础点云处理
│   │   │   └── 最近邻问题.md
│   │   ├── ch7-3DSLAM
│   │   │   └── 松耦合LIO系统.md
│   │   ├── ch8紧耦合LIO系统
│   │   │   └── 基于IEKF的LIO系统.md
│   │   └── 课后习题
│   │       └── ch073DSLAM.md
│   └── 视觉SLAM十四讲
│       ├── ch10后端2
│       │   └── 位姿图.md
│       ├── ch13实践SLAM系统
│       │   ├── C++智能指针内存管理机制.md
│       │   └── cvcalcOpticalFlowPyrLK函数详解.md
│       ├── ch2初始slam
│       │   └── 初识SLAM.md
│       ├── ch4李群与李代数
│       │   ├── 左扰动和右扰动的区别.md
│       │   └── 李群与李代数.md
│       ├── ch6非线性优化
│       │   ├── LM方法.md
│       │   ├── 非线性最小二乘.md
│       │   └── 高斯牛顿法.md
│       ├── ch7视觉里程计1
│       │   ├── ICP.md
│       │   ├── 点云关键点与特征描述子.md
│       │   └── 视觉里程计1.md
│       ├── ch8视觉里程计2
│       │   └── 直接法.md
│       ├── ch9后端1
│       │   ├── BA与图优化.md
│       │   └── 概述.md
│       └── 课后习题
│           ├── ch03-三维空间刚体运动.md
│           ├── ch04-李群与李代数.md
│           ├── ch05-相机与图像.md
│           ├── ch06-非线性优化.md
│           └── ch07-视觉里程计1.md
├── mkdocs.yml
├── overrides
│   ├── main.html
│   └── partials
│       └── comments.html
└── requirements.txt
```
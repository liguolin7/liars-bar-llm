# SciPy API文档

## 简介

SciPy是基于NumPy的科学计算库，提供了优化、统计、积分、线性代数、傅里叶变换、信号和图像处理等多种功能模块。SciPy在数据分析、科学研究和工程应用中都非常有用。

官方文档: [scipy.org](https://docs.scipy.org/doc/scipy/reference/)

## 安装

```bash
pip install scipy
```

## 常用模块

SciPy包含多个子模块，每个子模块专注于不同的功能：

- `scipy.stats`: 统计函数
- `scipy.optimize`: 优化和拟合算法
- `scipy.cluster`: 聚类算法
- `scipy.constants`: 物理和数学常数
- `scipy.fftpack`: 快速傅里叶变换
- `scipy.integrate`: 积分例程
- `scipy.interpolate`: 插值
- `scipy.io`: 数据输入和输出
- `scipy.linalg`: 线性代数例程
- `scipy.ndimage`: n维图像处理
- `scipy.signal`: 信号处理
- `scipy.sparse`: 稀疏矩阵
- `scipy.spatial`: 空间数据结构和算法
- `scipy.special`: 特殊函数

## 统计函数 (scipy.stats)

```python
from scipy import stats
import numpy as np

# 创建一些示例数据
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1, 100)
```

### 描述性统计

```python
# 计算均值、标准差等
print(stats.describe(data1))

# 计算中位数
print(np.median(data1))

# 计算众数
print(stats.mode(data1))

# 计算变异系数
print(stats.variation(data1))

# 计算偏度和峰度
print(stats.skew(data1))
print(stats.kurtosis(data1))
```

### 概率分布

```python
# 正态分布
x = np.linspace(-5, 5, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)  # PDF，正态分布，均值=0，标准差=1
cdf = stats.norm.cdf(x, loc=0, scale=1)  # CDF，正态分布，均值=0，标准差=1

# 从分布中抽样
samples = stats.norm.rvs(loc=0, scale=1, size=100)

# t分布
t_pdf = stats.t.pdf(x, df=10)  # 自由度为10的t分布的PDF

# 二项分布
binom_pmf = stats.binom.pmf(k=3, n=10, p=0.5)  # 10次试验，成功概率0.5，恰好3次成功的概率

# 卡方分布
chi2_pdf = stats.chi2.pdf(x, df=5)  # 自由度为5的卡方分布的PDF
```

### 假设检验

```python
# 单样本t检验
t_stat, p_value = stats.ttest_1samp(data1, popmean=0)

# 两个独立样本t检验
t_stat, p_value = stats.ttest_ind(data1, data2)

# 配对样本t检验
t_stat, p_value = stats.ttest_rel(data1, data2)

# 方差分析（ANOVA）
f_stat, p_value = stats.f_oneway(data1, data2, np.random.normal(2, 1, 100))

# 卡方检验
observed = np.array([[10, 20], [30, 40]])  # 观察值
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

# KS检验（检验样本是否来自特定分布）
ks_stat, p_value = stats.kstest(data1, 'norm')

# Shapiro-Wilk正态性检验
w_stat, p_value = stats.shapiro(data1)

# Mann-Whitney U检验（非参数检验）
u_stat, p_value = stats.mannwhitneyu(data1, data2)
```

### 相关性分析

```python
# Pearson相关系数
r, p_value = stats.pearsonr(data1, data2)

# Spearman等级相关系数
rho, p_value = stats.spearmanr(data1, data2)

# Kendall等级相关系数
tau, p_value = stats.kendalltau(data1, data2)

# 相关矩阵
import pandas as pd
df = pd.DataFrame({'A': data1, 'B': data2, 'C': np.random.normal(0, 1, 100)})
correlation_matrix = df.corr()
```

### 非参数方法

```python
# 核密度估计
kde = stats.gaussian_kde(data1)
density = kde(x)

# Bootstrap重采样
from scipy.stats import bootstrap
data = np.column_stack([data1, data2])
result = bootstrap((data,), np.mean, n_resamples=1000)
```

## 优化 (scipy.optimize)

```python
from scipy import optimize

# 函数最小化
def f(x):
    return x**2 + 10*np.sin(x)
result = optimize.minimize(f, x0=0)  # 初始猜测值为0

# 曲线拟合
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
xdata = np.linspace(0, 4, 50)
ydata = func(xdata, 2.5, 1.3, 0.5) + np.random.normal(0, 0.2, len(xdata))
popt, pcov = optimize.curve_fit(func, xdata, ydata)

# 求解非线性方程组
def system(x):
    return [x[0] + x[1] - 10, x[0] * x[1] - 16]
sol = optimize.fsolve(system, [1, 1])
```

## 线性代数 (scipy.linalg)

```python
from scipy import linalg

# 创建一个矩阵
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# 求解线性方程组 Ax = b
x = linalg.solve(A, b)

# 求特征值和特征向量
eigenvalues, eigenvectors = linalg.eig(A)

# 奇异值分解
U, s, Vh = linalg.svd(A)

# 计算矩阵的行列式
det = linalg.det(A)

# 计算矩阵的逆
inv_A = linalg.inv(A)

# LU分解
P, L, U = linalg.lu(A)

# QR分解
Q, R = linalg.qr(A)

# Cholesky分解（对称正定矩阵）
L = linalg.cholesky(A.T @ A)
```

## 积分 (scipy.integrate)

```python
from scipy import integrate

# 定义一个要积分的函数
def f(x):
    return np.sin(x)

# 定积分
result, error = integrate.quad(f, 0, np.pi)

# 二重积分
def g(x, y):
    return np.sin(x) * np.cos(y)
result, error = integrate.dblquad(g, 0, np.pi, lambda x: 0, lambda x: np.pi)
```

## 数值微分

```python
# 计算一维梯度
gradient = np.gradient(np.sin(np.linspace(0, 10, 100)))
```

## 信号处理 (scipy.signal)

```python
from scipy import signal

# 创建信号
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)

# 设计滤波器
b, a = signal.butter(4, 0.15)  # 4阶巴特沃斯低通滤波器，截止频率0.15

# 应用滤波器
y = signal.filtfilt(b, a, x)

# 计算频谱
f, Pxx = signal.welch(x, fs=1000)
```

## 项目中的应用

在骗子酒馆项目中，SciPy可用于：

1. 使用scipy.stats进行统计假设检验，如比较不同模型的表现差异
2. 计算不同行为指标之间的相关性
3. 使用KDE估计行为分布
4. 应用非参数检验比较模型性能
5. 进行Bootstrap分析以获得置信区间 
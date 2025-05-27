# NumPy API文档

## 简介

NumPy是Python中用于科学计算的基础库，提供了多维数组对象、各种派生对象（如掩码数组和矩阵）以及用于数组快速操作的各种API，包括数学、逻辑、形状操作、排序、选择、I/O、离散傅立叶变换、基本线性代数、基本统计操作等。

官方文档: [numpy.org](https://numpy.org/doc/stable/)

## 安装

```bash
pip install numpy
```

## 基本操作

### 创建数组

```python
import numpy as np

# 从列表创建数组
a = np.array([1, 2, 3, 4, 5])

# 创建特殊数组
zeros = np.zeros((3, 4))  # 3x4全零数组
ones = np.ones((2, 3, 4))  # 2x3x4全一数组
empty = np.empty((2, 3))  # 2x3未初始化数组
arange = np.arange(10, 30, 5)  # [10, 15, 20, 25]
linspace = np.linspace(0, 1, 5)  # 5个等间距点，从0到1
random = np.random.random((3, 3))  # 3x3随机数组
```

### 数组属性

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape)      # 数组维度 (2, 3)
print(a.ndim)       # 数组维数 2
print(a.dtype)      # 数组元素类型 int64
print(a.itemsize)   # 数组元素大小（字节） 8
print(a.size)       # 数组元素总数 6
```

### 数组索引与切片

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 索引
print(a[0, 0])    # 1
print(a[2, 3])    # 12

# 切片
print(a[0, :])    # [1, 2, 3, 4]
print(a[:, 1])    # [2, 6, 10]
print(a[1:3, 2:4])  # [[7, 8], [11, 12]]

# 布尔索引
mask = a > 5
print(a[mask])    # [6, 7, 8, 9, 10, 11, 12]
```

### 数组操作

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 基本运算
print(a + b)      # 元素加
print(a - b)      # 元素减
print(a * b)      # 元素乘
print(a / b)      # 元素除
print(a ** 2)     # 元素幂运算

# 矩阵运算
print(a.dot(b))   # 矩阵乘法
print(np.dot(a, b))  # 矩阵乘法
print(a @ b)      # Python 3.5+中的矩阵乘法

# 统计函数
print(a.sum())    # 所有元素求和
print(a.sum(axis=0))  # 沿列求和
print(a.sum(axis=1))  # 沿行求和
print(a.min())    # 最小值
print(a.max())    # 最大值
print(a.mean())   # 平均值
print(np.median(a))  # 中位数
print(a.std())    # 标准差
```

### 数组变形

```python
a = np.arange(12)

# 改变形状
print(a.reshape(3, 4))  # 3x4数组
print(a.reshape(3, -1))  # 行数为3，列数自动计算

# 转置
print(a.reshape(3, 4).T)  # 转置

# 拉平
print(a.reshape(3, 4).flatten())  # 返回拷贝
print(a.reshape(3, 4).ravel())    # 返回视图
```

### 数组拼接

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 垂直拼接
print(np.vstack((a, b)))

# 水平拼接
print(np.hstack((a, b)))

# 沿新轴拼接
print(np.concatenate((a, b), axis=0))  # 垂直
print(np.concatenate((a, b), axis=1))  # 水平
```

### 数组分割

```python
a = np.arange(12).reshape(3, 4)

# 水平分割
print(np.hsplit(a, 2))  # 分成2部分
print(np.hsplit(a, [1, 3]))  # 在第1和第3列处分割

# 垂直分割
print(np.vsplit(a, 3))  # 分成3部分
```

## 高级功能

### 广播

NumPy的"广播"特性允许对不同形状的数组进行算术运算。

```python
a = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3数组
b = np.array([10, 20, 30])  # 1D数组

print(a + b)  # b会被广播到a的形状
```

### 随机数生成

```python
# 随机整数
print(np.random.randint(0, 10, size=(3, 3)))

# 正态分布
print(np.random.normal(0, 1, size=(3, 3)))

# 均匀分布
print(np.random.uniform(0, 1, size=(3, 3)))

# 设置随机种子
np.random.seed(42)
```

### 线性代数

```python
a = np.array([[1, 2], [3, 4]])

# 求解线性方程组 Ax = b
b = np.array([5, 6])
x = np.linalg.solve(a, b)

# 特征值和特征向量
eigenvals, eigenvecs = np.linalg.eig(a)

# 矩阵的奇异值分解
u, s, vh = np.linalg.svd(a)

# 计算行列式
det = np.linalg.det(a)

# 计算逆矩阵
inv = np.linalg.inv(a)
```

### 傅里叶变换

```python
# 一维傅里叶变换
x = np.array([1, 2, 1, 0, 1, 2, 1, 0])
y = np.fft.fft(x)

# 二维傅里叶变换
image = np.random.random((64, 64))
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
```

## 项目中的应用

在骗子酒馆项目中，NumPy可用于：

1. 创建和处理模型表现的数值数组
2. 执行统计分析，如计算平均值、标准差和相关性
3. 生成随机数据进行模拟或测试
4. 实现数组操作以支持数据处理管道
5. 使用线性代数函数进行高级数据分析 
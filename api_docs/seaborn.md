# Seaborn API文档

## 简介

Seaborn是基于Matplotlib的Python数据可视化库，提供了更高级的统计图形绘制功能。它与Pandas数据结构紧密集成，内置了多种主题和调色板，使得生成更美观的统计图形更加简单。

官方文档: [seaborn.pydata.org](https://seaborn.pydata.org/)

## 安装

```bash
pip install seaborn
```

## 基本使用

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置主题
sns.set_theme(style="whitegrid")

# 创建示例数据
data = pd.DataFrame({
    "x": np.random.normal(size=100),
    "y": np.random.normal(size=100),
    "category": np.random.choice(["A", "B", "C"], 100)
})

# 创建散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x="x", y="y", hue="category", data=data)
plt.title("Seaborn散点图示例")
plt.show()
```

## 常用图表类型

### 1. 关系图 (Relational Plots)

```python
# 散点图
sns.relplot(x="x", y="y", hue="category", data=data, kind="scatter")

# 线图
sns.relplot(x="x", y="y", hue="category", data=data, kind="line")
```

### 2. 分布图 (Distribution Plots)

```python
# 直方图
sns.histplot(data=data, x="x", kde=True)

# 密度图
sns.kdeplot(data=data, x="x", hue="category", fill=True)

# 联合分布图
sns.jointplot(data=data, x="x", y="y", hue="category")

# 成对关系图
sns.pairplot(data=data, hue="category")
```

### 3. 分类图 (Categorical Plots)

```python
# 箱线图
sns.boxplot(x="category", y="y", data=data)

# 小提琴图
sns.violinplot(x="category", y="y", data=data)

# 条形图
sns.barplot(x="category", y="y", data=data)

# 计数图
sns.countplot(x="category", data=data)

# 点图
sns.pointplot(x="category", y="y", data=data)

# 条形图（条形不重叠）
sns.stripplot(x="category", y="y", data=data, jitter=True)

# 蜂群图
sns.swarmplot(x="category", y="y", data=data)
```

### 4. 回归图 (Regression Plots)

```python
# 线性回归
sns.regplot(x="x", y="y", data=data)

# 线性回归（带更多选项）
sns.lmplot(x="x", y="y", hue="category", data=data)
```

### 5. 矩阵图 (Matrix Plots)

```python
# 热力图
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

# 聚类图
sns.clustermap(corr_matrix, cmap="coolwarm", standard_scale=1)
```

## 样式定制

```python
# 设置主题风格
sns.set_style("whitegrid")  # 其他选项: darkgrid, white, dark, ticks

# 设置上下文
sns.set_context("notebook")  # 其他选项: paper, talk, poster

# 自定义调色板
sns.set_palette("pastel")  # 内置调色板
custom_palette = sns.color_palette("husl", 8)  # 自定义颜色数量

# 显示所有调色板
sns.palplot(sns.color_palette())
```

## 图形参数

```python
# FacetGrid 用于在数据子集上创建多个图表
g = sns.FacetGrid(data, col="category", height=5)
g.map(sns.histplot, "x")

# 保存图表
plt.savefig("seaborn_plot.png", dpi=300, bbox_inches="tight")
```

## 项目中的应用

在骗子酒馆项目中，Seaborn可用于：

1. 创建行为策略热力图，显示不同模型的行为偏好
2. 生成质疑成功率的统计分析图表
3. 可视化模型表现的箱线图，比较不同模型的分布差异
4. 创建相关性矩阵，分析不同指标之间的关系
5. 使用成对关系图分析多个性能指标 
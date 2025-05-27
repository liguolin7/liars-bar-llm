# Matplotlib在骗子酒馆项目中的应用

## 简介

Matplotlib是一个用于创建静态、动态和交互式可视化的Python库。在骗子酒馆LLM项目中，我们主要使用matplotlib来生成各种分析图表，展示不同LLM模型在策略博弈中的表现差异。

## 基本配置

项目中的matplotlib配置示例：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS适用
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
```

## 常用图表类型及用法

### 1. 条形图

用于比较不同模型在各项指标上的表现。

```python
def plot_bar_chart(metrics_df, metric, output_dir):
    """生成指标条形图"""
    # 排序以便展示
    sorted_df = metrics_df.sort_values(by=metric, ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_df['player'], sorted_df[metric], 
                  color=sns.color_palette("viridis", len(sorted_df)))
    
    # 设置标题和标签
    plt.title(f'玩家{metric}比较', fontsize=16)
    plt.xlabel('玩家', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    
    # 在条形图上添加数值标签
    for bar, value in zip(bars, sorted_df[metric]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 保存图表
    output_file = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

### 2. 雷达图

用于多维度展示模型的特征分布，如欺骗策略、综合能力等。

```python
def plot_radar_chart(metrics_df, metrics, categories, title, output_file):
    """生成雷达图"""
    plt.figure(figsize=(10, 8))
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 设置图表
    ax = plt.subplot(111, polar=True)
    
    # 为每个玩家绘制雷达图
    for i, player in enumerate(metrics_df['player']):
        values = metrics_df.loc[metrics_df['player'] == player, metrics].values.flatten().tolist()
        values += values[:1]  # 闭合多边形
        
        # 绘制线条
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=player)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置雷达图标签
    plt.xticks(angles[:-1], categories, size=12)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

### 3. 散点图

用于展示两个指标间的关系，如质疑决策质量（精确度vs.召回率）。

```python
def plot_scatter(x_data, y_data, labels, x_label, y_label, title, output_file):
    """生成散点图"""
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    scatter = plt.scatter(
        x_data, y_data, 
        c=np.arange(len(x_data)), 
        cmap='viridis', 
        s=200, 
        alpha=0.7
    )
    
    # 添加标签
    for i, label in enumerate(labels):
        plt.annotate(
            label, 
            (x_data[i], y_data[i]),
            xytext=(7, 7),
            textcoords='offset points',
            fontsize=12
        )
    
    # 设置坐标轴标签和标题
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=18)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

### 4. 热力图

用于展示多维度指标的综合分析，如玩家综合能力评分。

```python
def plot_heatmap(data_df, output_file, title, cmap="YlGnBu"):
    """生成热力图"""
    plt.figure(figsize=(12, len(data_df) * 0.8 + 2))
    
    # 绘制热力图
    ax = sns.heatmap(
        data_df, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        cbar_kws={'label': '分数'},
        linewidths=0.5
    )
    
    # 设置标题
    plt.title(title, fontsize=18)
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

### 5. 时序图

用于展示指标随时间/回合的变化趋势。

```python
def plot_trend(x_data, y_data_dict, x_label, y_label, title, output_file):
    """生成趋势线图"""
    plt.figure(figsize=(12, 8))
    
    # 为每个系列绘制线条
    for label, y_data in y_data_dict.items():
        plt.plot(x_data, y_data, marker='o', linewidth=2, label=label)
    
    # 设置坐标轴标签和标题
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=18)
    
    # 添加图例
    plt.legend()
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

## 图表存储路径

在骗子酒馆项目中，图表按照分析类型存储在不同目录：

- **指标可视化**：保存在 `metrics_output` 目录(单独运行)或 `analysis_results/metrics` 目录(综合分析)
- **决策分析**：保存在 `analysis_results/decision_analysis` 目录
- **行为分析**：保存在 `analysis_results/behavior_analysis` 目录
- **统计分析**：保存在 `analysis_results/statistical_analysis` 目录

## 常见问题与解决方案

### 中文显示问题

在macOS环境中，使用 `Arial Unicode MS` 字体可以正确显示中文：

```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

在Linux环境中，可以尝试：

```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
```

### 图表大小与分辨率

为了确保图表在报告中显示清晰，建议设置合适的图表大小和DPI：

```python
plt.figure(figsize=(10, 6))  # 宽10英寸，高6英寸
plt.savefig('output.png', dpi=300, bbox_inches='tight')  # 300 DPI，自动调整边界
```

### 内存管理

当生成大量图表时，确保在每个图表处理完成后关闭图形，释放内存：

```python
plt.close()  # 关闭当前图形
```

### 颜色方案

使用seaborn的调色板可以生成协调的颜色方案：

```python
# 使用viridis调色板
colors = sns.color_palette("viridis", n_colors)

# 或使用自定义颜色
custom_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
```

## 最佳实践

1. **保持一致性**：在项目中使用统一的颜色方案、字体大小和图表样式
2. **添加标签和标题**：每个图表都应有清晰的标题、坐标轴标签和数据标签
3. **图例位置**：合理放置图例，避免遮挡重要数据
4. **数据排序**：对条形图等数据进行排序，使图表更具可读性
5. **适当的空间**：给图表元素留出足够的空间，避免拥挤
6. **配色考虑**：选择色盲友好的配色方案，如viridis, plasma等 
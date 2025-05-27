# Pandas API文档

## 简介

Pandas是Python中用于数据操作和分析的核心库，提供了高效的DataFrame和Series数据结构，使数据清洗、转换、聚合和可视化变得简单高效。

官方文档: [pandas.pydata.org](https://pandas.pydata.org/docs/)

## 安装

```bash
pip install pandas
```

## 基本数据结构

### Series
一维标记数组，可以包含任何数据类型

```python
import pandas as pd
import numpy as np

# 创建Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
```

### DataFrame
二维表格型数据结构，带有标记的行和列

```python
# 创建DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': pd.date_range('20230101', periods=4),
    'C': pd.Series(np.random.randn(4)),
    'D': pd.Categorical(["test", "train", "test", "train"]),
    'E': 'foo'
})
print(df)
```

## 数据导入与导出

```python
# 读取CSV文件
df = pd.read_csv('data.csv')

# 读取Excel文件
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 读取JSON文件
df = pd.read_json('data.json')

# 保存为CSV
df.to_csv('output.csv', index=False)

# 保存为Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# 保存为JSON
df.to_json('output.json', orient='records')
```

## 数据查看与选择

```python
# 查看数据前几行
df.head()

# 查看数据后几行
df.tail()

# 查看数据基本信息
df.info()

# 查看数据统计摘要
df.describe()

# 选择列
df['A']
df.A

# 选择行（按位置）
df.iloc[0]  # 第一行
df.iloc[0:3]  # 前三行
df.iloc[0, 1]  # 第一行第二列

# 选择行（按标签）
df.loc['index_label']
df.loc['index_label1':'index_label2']

# 条件选择
df[df.A > 0]
df[df['D'] == 'test']
```

## 数据清洗

```python
# 检查缺失值
df.isna().sum()

# 填充缺失值
df.fillna(0)  # 用0填充
df['A'].fillna(df['A'].mean())  # 用平均值填充

# 删除缺失值
df.dropna()  # 删除包含缺失值的行
df.dropna(axis=1)  # 删除包含缺失值的列

# 重复值处理
df.duplicated().sum()  # 检查重复行
df.drop_duplicates()  # 删除重复行

# 数据类型转换
df['A'] = df['A'].astype('int64')

# 重命名列
df.rename(columns={'A': 'Alpha', 'B': 'Beta'})
```

## 数据操作

```python
# 排序
df.sort_values(by='A', ascending=False)  # 按列A降序排列
df.sort_index()  # 按索引排序

# 分组运算
df.groupby('D').mean()  # 按D列分组并计算均值
df.groupby(['D', 'E']).sum()  # 多列分组

# 数据聚合
df.groupby('D').agg({'A': 'mean', 'C': ['min', 'max']})

# 合并数据
pd.concat([df1, df2])  # 纵向合并
pd.concat([df1, df2], axis=1)  # 横向合并
pd.merge(df1, df2, on='key')  # 按键合并

# 数据透视表
pd.pivot_table(df, values='A', index=['D'], columns=['E'])
```

## 时间序列处理

```python
# 创建日期范围
dates = pd.date_range('20230101', periods=6)

# 将字符串转换为日期时间
df['date'] = pd.to_datetime(df['date_str'])

# 提取日期组件
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# 重采样
df.set_index('date').resample('M').mean()  # 按月重采样并计算均值
```

## 数据可视化

```python
# 折线图
df.plot(kind='line')

# 柱状图
df.plot(kind='bar')

# 直方图
df['A'].plot(kind='hist', bins=20)

# 散点图
df.plot.scatter(x='A', y='C')

# 箱线图
df.plot.box()
```

## 项目中的应用

在骗子酒馆项目中，Pandas可用于：

1. 加载和处理游戏记录数据
2. 计算和分析各种性能指标
3. 聚合数据以生成可视化输入
4. 创建模型表现的对比数据框
5. 处理时间序列数据，分析游戏进程中的趋势 
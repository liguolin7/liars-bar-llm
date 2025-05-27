# Plotly API文档

## 简介

Plotly是一个用于创建交互式可视化的Python库，允许用户创建高质量的图表、图形和仪表板，这些可视化可以在浏览器中交互，并可以导出为静态图片或嵌入到Web应用程序中。Plotly支持超过40种图表类型，包括科学图表、3D图表、统计图表、地图等。

官方文档: [plotly.com/python](https://plotly.com/python/)

## 安装

```bash
pip install plotly
```

## 基本使用

Plotly提供两种主要的图表创建API：
- `plotly.graph_objects`: 更低级别的API，提供更多自定义选项
- `plotly.express`: 更高级别的API，简化了许多常见图表的创建

### 使用plotly.express创建图表

```python
import plotly.express as px
import pandas as pd

# 创建示例数据
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [1, 3, 2, 5, 3],
    'category': ['A', 'B', 'A', 'B', 'A']
})

# 创建散点图
fig = px.scatter(df, x='x', y='y', color='category')
fig.show()

# 创建柱状图
fig = px.bar(df, x='category', y='y')
fig.show()

# 创建线图
fig = px.line(df, x='x', y='y', color='category')
fig.show()
```

### 使用plotly.graph_objects创建图表

```python
import plotly.graph_objects as go

# 创建散点图
fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4, 5], y=[1, 3, 2, 5, 3], mode='markers'))
fig.show()

# 创建带多个数据系列的图表
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers', name='Series A'))
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[7, 8, 9], mode='lines', name='Series B'))
fig.show()
```

## 常用图表类型

### 1. 基础图表

```python
# 散点图
fig = px.scatter(df, x='x', y='y', color='category')

# 线图
fig = px.line(df, x='x', y='y', color='category')

# 柱状图
fig = px.bar(df, x='category', y='y')

# 饼图
fig = px.pie(df, values='y', names='category')

# 直方图
fig = px.histogram(df, x='y')

# 箱线图
fig = px.box(df, x='category', y='y')

# 小提琴图
fig = px.violin(df, x='category', y='y')
```

### 2. 统计图表

```python
# 散点图矩阵
fig = px.scatter_matrix(df, dimensions=['x', 'y'])

# 平行坐标图
fig = px.parallel_coordinates(df, color='category')

# 相关性热力图
fig = px.imshow(df.corr())

# 树状图
fig = px.treemap(df, path=['category', 'x'], values='y')

# 旭日图
fig = px.sunburst(df, path=['category', 'x'], values='y')
```

### 3. 地理地图

```python
# 地理散点图
geo_df = px.data.gapminder().query("year == 2007")
fig = px.scatter_geo(geo_df, locations="iso_alpha", 
                     color="continent", 
                     hover_name="country", 
                     size="pop",
                     projection="natural earth")

# 等值线地图
fig = px.choropleth(geo_df, locations="iso_alpha",
                    color="lifeExp", 
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma)
```

### 4. 3D图表

```python
# 3D散点图
df_3d = px.data.election()
fig = px.scatter_3d(df_3d, x="Joly", y="Coderre", z="Bergeron",
                    color="winner", size="total", hover_name="district")

# 3D曲面图
z_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
fig = px.imshow(z_data, text_auto=True)
fig = go.Figure(data=[go.Surface(z=z_data)])
```

## 图表布局与样式

```python
# 设置图表标题和轴标签
fig.update_layout(
    title="图表标题",
    xaxis_title="X轴标题",
    yaxis_title="Y轴标题",
    legend_title="图例标题"
)

# 设置颜色和主题
fig.update_layout(
    template="plotly_dark",  # 使用暗色主题
    colorway=px.colors.qualitative.Plotly  # 使用Plotly默认的颜色序列
)

# 添加注释
fig.add_annotation(
    x=2, y=4,
    text="注释文本",
    showarrow=True,
    arrowhead=1
)

# 更新点的样式
fig.update_traces(
    marker_size=10,
    marker_color='red',
    marker_symbol='circle'
)
```

## 子图和多图表

```python
# 创建子图网格
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2, subplot_titles=("子图1", "子图2", "子图3", "子图4"))

fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
fig.add_trace(go.Bar(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[7, 8, 9]), row=2, col=1)
fig.add_trace(go.Bar(x=[1, 2, 3], y=[7, 8, 9]), row=2, col=2)

fig.update_layout(height=600, width=800, title_text="多个子图")
```

## 交互功能

```python
# 添加悬停信息
fig = px.scatter(df, x='x', y='y', color='category',
                hover_data=['x', 'y', 'category'])

# 添加点击事件
from dash import Dash, dcc, html, Input, Output, callback
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='graph', figure=fig),
    html.Div(id='click-data')
])

@callback(
    Output('click-data', 'children'),
    Input('graph', 'clickData')
)
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)
```

## 保存与导出

```python
# 保存为HTML文件（可交互）
fig.write_html("plot.html")

# 保存为图片
fig.write_image("plot.png")  # 需要安装kaleido包
fig.write_image("plot.pdf")
fig.write_image("plot.svg")
```

## 项目中的应用

在骗子酒馆项目中，Plotly可用于：

1. 创建交互式胜率图表，允许用户悬停查看详细信息
2. 生成多指标对比的散点图或平行坐标图
3. 创建时间序列动画，展示游戏进程中策略的变化
4. 制作交互式热力图，展示模型行为模式
5. 构建可视化仪表板，综合展示多个分析图表 
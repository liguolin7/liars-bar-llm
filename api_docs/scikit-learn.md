# Scikit-learn API文档

## 简介

Scikit-learn是Python中最流行的机器学习库，提供了大量用于数据分析和数据挖掘的工具，包括分类、回归、聚类、降维、模型选择和预处理等功能。它构建在NumPy、SciPy和matplotlib之上，提供了简洁一致的接口。

官方文档: [scikit-learn.org](https://scikit-learn.org/stable/index.html)

## 安装

```bash
pip install scikit-learn
```

## 基本模块

Scikit-learn包含多个子模块，每个子模块专注于不同的机器学习任务：

- `sklearn.cluster`: 聚类算法
- `sklearn.ensemble`: 集成方法
- `sklearn.linear_model`: 线性模型
- `sklearn.metrics`: 评估指标
- `sklearn.model_selection`: 模型选择和评估
- `sklearn.preprocessing`: 数据预处理
- `sklearn.decomposition`: 降维算法
- `sklearn.svm`: 支持向量机
- `sklearn.tree`: 决策树
- `sklearn.neighbors`: 最近邻算法

## 数据预处理

```python
import numpy as np
from sklearn import preprocessing
import pandas as pd

# 创建示例数据
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])

# 标准化（均值为0，方差为1）
X_scaled = preprocessing.StandardScaler().fit_transform(X)

# MinMax缩放（缩放到特定范围，默认为[0, 1]）
X_minmax = preprocessing.MinMaxScaler().fit_transform(X)

# 归一化（缩放单个样本使其具有单位范数）
X_normalized = preprocessing.Normalizer().fit_transform(X)

# 独热编码
df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
one_hot = pd.get_dummies(df['category'])

# 标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = ['A', 'B', 'A', 'C']
encoded_labels = le.fit_transform(labels)  # 转换为 [0, 1, 0, 2]

# 处理缺失值
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_with_nan = np.array([[1, 2], [np.nan, 3], [7, 6]])
X_imputed = imp.fit_transform(X_with_nan)
```

## 特征选择

```python
from sklearn.feature_selection import SelectKBest, chi2

# 创建示例数据
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y = np.array([0, 1, 0])

# 选择k个最佳特征
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)
```

## 模型训练和评估

### 分类

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 创建示例数据
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
```

### 回归

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 创建示例数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 评估
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE:", mse)
print("R^2:", r2)
```

## 聚类

### K-means聚类

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 创建示例数据
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# 使用K-means聚类
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# 获取聚类标签
labels = kmeans.labels_

# 获取聚类中心
centers = kmeans.cluster_centers_

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('K-means聚类')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

### 层次聚类

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 创建示例数据
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# 使用层次聚类
clustering = AgglomerativeClustering(n_clusters=2).fit(X)
labels = clustering.labels_

# 创建层次聚类树状图
linked = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()
```

### DBSCAN聚类

```python
from sklearn.cluster import DBSCAN

# 创建示例数据
X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])

# 使用DBSCAN聚类
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
labels = clustering.labels_

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title('DBSCAN聚类')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

## 降维

### 主成分分析 (PCA)

```python
from sklearn.decomposition import PCA

# 创建示例数据
X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3],
              [4, 4, 4], [5, 5, 5], [6, 6, 6]])

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 获取解释的方差比例
explained_variance_ratio = pca.explained_variance_ratio_

# 可视化
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title('PCA降维')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.show()
```

### t-SNE

```python
from sklearn.manifold import TSNE

# 创建示例高维数据
X = np.random.rand(100, 50)

# 使用t-SNE降维到2维
tsne = TSNE(n_components=2, random_state=0)
X_reduced = tsne.fit_transform(X)

# 可视化
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title('t-SNE降维')
plt.xlabel('成分1')
plt.ylabel('成分2')
plt.show()
```

## 模型选择和评估

### 交叉验证

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# 创建示例数据
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# 5折交叉验证
model = SVC(kernel='linear', C=1)
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

### 网格搜索（超参数优化）

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 创建示例数据
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly']
}

# 使用网格搜索找到最佳参数
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X, y)

# 最佳参数
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# 使用最佳参数的模型
best_model = grid_search.best_estimator_
```

### 评估指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt

# 假设已有测试集和预测结果
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])

# 基本指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 假设我们有预测概率
y_score = np.array([0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.7, 0.6, 0.9, 0.1])

# ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('接收者操作特性曲线')
plt.legend(loc="lower right")
plt.show()

# 精确率-召回率曲线
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.show()
```

## 项目中的应用

在骗子酒馆项目中，Scikit-learn可用于：

1. 使用K-means聚类分析不同模型的行为模式
2. 应用层次聚类识别模型策略的相似性
3. 使用PCA降维分析多维性能指标
4. 实现特征选择，识别最重要的性能指标
5. 应用分类算法预测模型行为 
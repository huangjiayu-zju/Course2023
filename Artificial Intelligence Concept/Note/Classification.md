[TOC]

## Function

### 0. Import Library

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import RFE
sns.set(style="whitegrid", color_codes=True)
from imblearn.over_sampling import SMOTE
```



### 1. Load the dataset

##### pd.read_csv

`pd.read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, dtype=None, and many more...)`:

- `filepath_or_buffer`: String or file handle representing the file path.
- `sep` or `delimiter`: String to use as delimiter, default is comma.
- `header`: Row number(s) to use as column names.
- `names`: List of column names to use.
- `index_col`: Column to set as index.
- `dtype`: Data type for data or columns.

```py
import pandas as pd

# 加载CSV文件
data = pd.read_csv('path_to_file.csv')
# 加载Excel文件
data = pd.read_excel('path_to_file.xlsx', sheet_name='Sheet1')
# 加载Scikit-learn的内置数据集: scikit-learn 提供了几个内置数据集，这些数据集主要用于演示和教学。
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 加载文本文件
with open('path_to_file.txt', 'r') as file:
    text_data = file.read()
# 使用opencv加载图片
import cv2
image = cv2.imread('path_to_image.jpg')
# 注意：默认情况下，opencv将图片加载为BGR格式，可能需要转为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```



### 2. Dataset inspection

1. 了解数据的结构和特征
1. 数值特征的统计摘要，如均值、中位数、最小值、最大值等，获取数值特征的分布情况。
1. 确定数据的类型（如数值、类别）和查找有缺失值的列。
1. 查看类别特征的唯一值与分布
1. 查看并计算缺失值，并计划相应的缺失值处理策略
1. 数据可视化

```py
import pandas as pd
data = pd.read_csv('path_to_file.csv')

# 查看前5行
print(data.head())
# 查看数值特征的统计摘要
print(data.describe())
# 查看每列的数据类型和非缺失值的数量
print(data.info())
# 查看某一列的唯一值及其计数
print(data['category_column'].value_counts())
# 计算每列的缺失值数量
missing_values = data.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
# 绘制某一特征的直方图
sns.histplot(data['numeric_column'])
plt.show()
# 绘制两个特征之间的散点图
sns.scatterplot(x='feature1', y='feature2', data=data)
plt.show()
```

#### Handling of missing values

1. 删除含有缺失值的行或列
   - 当只有少量的行或某个特定的列有缺失值时。
2. 使用统计方法（如平均值、中位数、众数）来填充缺失值。
   - 不希望删除有缺失值的行或列，而是希望保留尽可能多的数据。
3. 使用其他特征来训练一个模型，预测缺失值。
   - 当缺失值与其他特征有一定的相关性时。
4. 使用统计和数学方法进行插值。
   - 时间序列数据或其他序列数据。
5. 填充固定的常数值，例如0、-999等。
   - 当有理由相信缺失值可以由一个特定的常数值来表示。
6. `SimpleImputer`类
   - `scikit-learn`提供了一个`SimpleImputer`类，它可以方便地填充缺失值。

```py
import pandas as pd
data = pd.read_csv('path_to_file.csv')
# 删除包含缺失值的行
data.dropna(axis=0, inplace=True)
# 删除包含缺失值的列
data.dropna(axis=1, inplace=True)

# 使用平均值填充缺失值
data['column_name'].fillna(data['column_name'].mean(), inplace=True)
# 使用中位数填充缺失值
data['column_name'].fillna(data['column_name'].median(), inplace=True)
# 使用众数填充缺失值
mode_value = data['column_name'].mode()[0]
data['column_name'].fillna(mode_value, inplace=True)

from sklearn.ensemble import RandomForestRegressor
# 分割数据为有缺失值和没有缺失值的部分
train_data = data.dropna()
test_data = data[data['column_name'].isnull()]
# 使用随机森林预测缺失值
clf = RandomForestRegressor(n_estimators=100)
clf.fit(train_data.drop('column_name', axis=1), train_data['column_name'])
predicted_values = clf.predict(test_data.drop('column_name', axis=1))
# 填充预测的值
data.loc[data['column_name'].isnull(), 'column_name'] = predicted_values

# 使用线性插值
data['column_name'].interpolate(method='linear', inplace=True)

# 填充0
data['column_name'].fillna(0, inplace=True)

from sklearn.impute import SimpleImputer
# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
data['column_name'] = imputer.fit_transform(data[['column_name']])
```



#### Data Split

先分离训练集和测试集

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### mean and standard deviation

```py
# 计算每个特征的均值和标准差
mean_vals = X_train.mean()
std_vals = X_train.std()
print("Mean Values:\n", mean_vals)
print("\nStandard Deviation Values:\n", std_vals)

# 判断数据是否"centered"
is_centered = mean_vals.abs() < 1e-9
print("\nIs Data Centered:\n", is_centered)

# 检查标准差是否各不相同
distinct_std = len(std_vals.unique()) == len(std_vals)
print("\nAre Standard Deviations Distinct:", distinct_std)
```

1. **均值 (Mean Values)**:
   - 均值为0的特征意味着该特征已经被中心化。
2. **标准差 (Standard Deviation Values)**:
   - 标准差为1的特征意味着该特征已经被标准化。
   - 如果某个特征的标准差非常小，它可能不包含太多信息或它可能是一个常数。
   - 如果每个特征的标准差都是不同的，这说明每个特征都有其独特的信息和分布。
3. **数据是否"centered"**:
   - 如果特征的均值接近于0（我们使用一个非常小的阈值如1e-9来判断），则我们可以认为数据是中心化的。
4. **标准差是否各不相同**:
   - 如果所有特征的标准差都是独特的，我们可以得出结论，每个特征都有其独特的信息和分布。



### 3. Label distribution and class imbalance

Techniques to address class imbalance: 当数据集中的类别分布不均匀时，称为类别不平衡。这可能导致模型偏向于多数类，因此需要使用策略来处理这种不平衡。

注意：**在训练数据上应用您选择的处理不平衡策略，在未经处理的测试数据上评估模型:**

#### 3.1. Resampling

这涉及对训练数据（不能在测试数据）进行重采样，使各个类别的实例数量达到平衡。

- **SMOTE** 通过为少数类合成新样本来工作。这是通过在少数类的实例之间随机挑选差异来完成的。
  - **适用场景**：当您有足够的数据，并且可以生成类似于原始数据的合成样本时。
- **上采样（Oversampling）**: 通过复制少数类的实例或生成新的实例来增加少数类的数量。
  - **适用场景**：当您担心合成样本可能不代表真实数据时，或当您的数据集相对较小且不能生成可靠的合成样本时。
- **下采样（Undersampling）**: 通过从多数类中随机删除实例来减少多数类的数量。
  - **适用场景**：当您有大量数据并且担心对于少数类的过采样可能导致过拟合时。然而，需要注意的是，下采样会丢失数据。

##### SMOTE

```py
from imblearn.over_sampling import SMOTE
# 应用SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

##### Ramdom Oversampling

```py
from imblearn.over_sampling import RandomOverSampler
# 应用随机上采样
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
```

##### Random Undersampling

```
from imblearn.over_sampling import RandomUnderSampler
# 应用随机上采样
ros = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
```

#### 3.2. Class weights

```py
from sklearn.ensemble import RandomForestClassifier
# 使用类权重
clf = RandomForestClassifier(class_weight="balanced", random_state=42)#以随机森林模型为例，在模型训练的损失函数中为每个类别分配权重
clf.fit(X_train, y_train)
```

#### 3.3. Algorithm

##### ensemble methods like Random Forest

随机森林可以很好地处理不平衡数据，因为它构建多个决策树并结合它们的输出。

- **适用场景**：当您有大量特征，或数据中存在非线性关系和交互时。

```py
from sklearn.ensemble import RandomForestClassifier
# 使用随机森林
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
```

##### boosting algorithms like AdaBoost

AdaBoost的工作原理是通过在每次迭代时加大错分类样本的权重来增强分类器的性能。

- **适用场景**：当您的数据是高维的，或当其他方法表现不佳时。

```py
from sklearn.ensemble import AdaBoostClassifier
# 使用AdaBoost
clf = AdaBoostClassifier(random_state=42)
clf.fit(X_train, y_train)
```

### 4. Feature Normalization

For each method, the general usage pattern is the same:

1. Import the scaler.
2. Initialize the scaler.
3. Use the `fit_transform` method on the data.

Remember to always `fit` the scaler on the training data and then use this fitted scaler to `transform` both the training and testing datasets to ensure that information from the testing set doesn't leak into the training set.

- `fit`方法计算数据的相关统计信息（如均值、标准差等），这些信息随后被用于`transform`方法来实际缩放数据。`fit_transform`是一种方便的方式，它首先调用`fit`然后调用`transform`。
- 避免数据泄漏：应该只使用训练数据来`fit`缩放器或任何其他预处理步骤，然后使用这个fitted缩放器来`transform`你的训练和测试数据。这样，你确保测试数据的任何信息都没有影响到预处理步骤。

##### StandardScaler

Standardize features by removing the mean and scaling to unit variance.

如果你的数据大致服从正态分布（或高斯分布），或者数据的特征具有不同的量纲或单位时，我们通常使用StandardScaler来进行标准化。例如，数据集中的一个特征是以千米为单位，另一个特征是以克为单位。在大多数机器学习算法中，特别是线性和逻辑回归、支持向量机、神经网络等，都推荐使用此方法。

```py
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

##### MinMaxScaler

Transforms features by scaling each feature to a given range, typically [0, 1].

当你知道数据的分布有边界，或者你想将特征缩放到一个特定的范围（如[0,1]）时，在深度学习中很常见。

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

##### RobustScaler

Scale features using statistics that are robust to outliers, using the median and the interquartile range.

当数据包含许多异常值，你需要一个对这些异常值不敏感的缩放方法时。

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```



### 5. Feature Selection

只在训练集上进行特征选择，然后将这些选择的特征应用到测试集上。

1. **SelectKBest (基于统计测试的特征选择)**:
   - 当数据维度不是特别高时较为有效
   - 当你想迅速检查几个最显著的特征时。
2. **Recursive Feature Elimination (RFE)**:
   - 当数据的维度中等或相对较小时。因为RFE逐步评估特征的重要性，所以计算上可能比较昂贵。
   - 当你想要基于模型自身的特征重要性进行特征选择时，例如支持向量机或线性回归。
3. **Tree-based Selection (如决策树和随机森林)**:
   - 当数据维度很高时，因为树模型能够有效地处理大规模数据。
   - 当数据中有非线性关系和高阶交互时，树模型能够捕捉这些关系。
   - 适用于分类和回归任务。
4. **L1 Regularization (Lasso Regression)**:
   - 当你处理线性关系或使用线性模型时。
   - 当数据是高维的，且需要稀疏解时。
   - 适用于回归任务，或者逻辑回归中的分类任务。
   - 当你需要模型的参数具有稀疏性，即大部分特征的系数为0时。
5. **Correlation Matrix with Heatmap**:
   - 当你想要直观地理解特征之间的关系时。
   - 当数据维度不是很高，从而可以在热力图上直观地看到所有特征的相关性时。
   - 用于初步筛选与目标变量高度相关或与其他特征高度相关的特征。

##### 1. SelectKBest

selects the top k features based on a chosen statistical test.

```py
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)#使用训练数据拟合选择器并转换训练数据,X: data。y: label 
X_test_selected = selector.transform(X_test_scaled)使用已拟合的选择器转换测试数据
```

`chi2`: 这是一个统计测试方法，用于测试非负特征和类别目标的相关性。它返回两个数组，第一个是计算出的每个特征的 chi^2 统计量，第二个是每个特征的 p-values。特征与目标越相关，其统计值就越高。除了 `chi2`，`SelectKBest` 还支持其他的统计测试方法，如：

- `f_classif`: 用于分类任务的ANOVA F-value。
- `mutual_info_classif`: 计算分类任务的mutual information。
- `f_regression`: 用于回归任务的F-value。
- `mutual_info_regression`: 计算回归任务的mutual information。

##### 2. Recursive Feature Elimination(RFE)

This method recursively removes attributes and builds a model on those attributes that remain. It uses the model's accuracy to identify which attributes (and combinations of attributes) contribute most to predicting the target attribute. 反复创建模型并选择当前迭代中最差的特征或最好的特征来工作，然后删除该特征。下一步再使用这些剩下的特征进行模型拟合

```py
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()# 初始化RFE并使用LogisticRegression作为估计器
rfe = RFE(model, n_features_to_select=5, step=1)# 我们选择的特征数量由'n_features_to_select'决定
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
print("Num Features: %s" % (fit.n_features_)) 
print("Selected Features: %s" % (fit.support_)) #显示哪些特征被选中
print("Feature Ranking: %s" % (fit.ranking_)) #显示特征的排名
```

##### 3. Tree-based algorithms

Tree-based algorithms like decision trees and random forests offer an importance score for each feature, indicating its usefulness in making a decision.

```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier() # 初始化随机森林分类器
#clf = DecisionTreeClassifier() # 初始化决策树分类器
clf.fit(X_train_scaled, y_train)

# 获取特征重要性
feature_importances = clf.feature_importances_ 

# 基于特征重要性选择特征
sfm = SelectFromModel(clf)
X_train_selected = sfm.transform(X_train_scaled)
X_test_selected = sfm.transform(X_test_scaled)
# 注意: 你还可以使用SelectFromModel的threshold参数来选择特征。
# 例如，如果你只想保留重要性大于0.05的特征，可以这样做:
# sfm = SelectFromModel(clf, threshold=0.05)
```

##### 4. L1 Regularization (Lasso Regression)

```py
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# 基于系数选择特征
sfm = SelectFromModel(lasso)
X_train_selected = sfm.transform(X_train_scaled)
X_test_selected = sfm.transform(X_test_scaled)
```

##### 5. Correlation Matrix with Heatmap

```py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 计算训练数据的相关性矩阵
corrmat = pd.DataFrame(X_train_scaled).corr()
# 使用heatmap显示
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, annot=True, cmap="YlGnBu")
plt.show()
# 基于这些相关性值，你可以选择与目标变量最相关的特征或删除某些高度相关的特征。
# 例如, 假设你想保留与目标变量相关性大于0.5的特征:
selected_features = corrmat.nlargest(5, 'target_column')['target_column'].index  # 替换'target_column'为你的目标列名称
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]
```



### 6. Model Selection

根据问题类型（如分类、回归、聚类等）选择一个或多个机器学习模型

#### Supervised Model

##### 1. Logistic Regression

**正则化类型**:

1. `'l1'`: L1正则化，也被称为Lasso回归。它通过将一些特征的系数设为0来实现特征选择。
2. `'l2'`: L2正则化，也被称为Ridge回归。它会尽量使得特征的系数趋于小而不是0。
3. `'elasticnet'`: 是L1和L2正则化的组合，旨在结合Lasso和Ridge的优点。
4. `'none'`: 不使用正则化。

**优化算法**:

1. `'newton-cg'`: Newton-Conjugate Gradient，是一种利用牛顿法的二阶导数信息来找到目标函数的局部最小值的算法。
2. `'lbfgs'`: Limited-memory Broyden-Fletcher-Goldfarb-Shanno。它是一种迭代方法，用于解决大规模无约束非线性优化问题。
3. `'liblinear'`: 一个用于大规模线性分类的库，尤其适用于高维数据。
4. `'sag'`: Stochastic Average Gradient descent，是一种线性收敛的随机优化算法。
5. `'saga'`: 是SAG的变体，可以与L1正则化一起使用。

```py
from sklearn.linear_model import LogisticRegression
# 初始化模型
clf = LogisticRegression(penalty='l2',       # 正则化类型: 'l1', 'l2', 'elasticnet', 'none'
                         C=1.0,              # 正则化强度的逆: 值越小，正则化强度越大
                         solver='lbfgs',     # 优化算法: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
                         max_iter=100)       # 算法收敛的最大迭代次数
# 训练模型
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
```

##### 2. Support Vector Machines

**核函数类型**:

1. `'linear'`: 线性核，表达的是线性关系。
2. `'poly'`: 多项式核，用于捕捉多项式关系。
3. `'rbf'`: 径向基核（Radial Basis Function），也被称为高斯核，用于捕捉非线性关系。
4. `'sigmoid'`: Sigmoid核，可以看作是神经网络的激活函数。
5. `'precomputed'`: 核矩阵是预先计算的。

**核系数**:

1. `'scale'`: 通常等于1/(n_features * X.var())，其中X是训练数据。
2. `'auto'`: 如果gamma没有被设置，那么它将被设置为1/n_features。
3. 浮点数: 允许用户直接为gamma设置一个特定值。

```py
from sklearn.svm import SVC

# 初始化模型
clf = SVC(kernel='linear',     # 核函数类型: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
          C=1.0,               # 错误项的惩罚参数
          gamma='scale',       # 核系数: 'scale', 'auto' 或浮点数
          degree=3)            # 多项式核的阶数 (只有kernel='poly'时才有意义)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

```

##### 3. Decision Trees

```py
from sklearn.tree import DecisionTreeClassifier
# 初始化模型
clf = DecisionTreeClassifier(criterion='gini',   # 衡量分裂质量的函数: 'gini' 或 'entropy'
                             max_depth=None,     # 树的最大深度: None表示不设置最大深度
                             min_samples_split=2,# 节点分裂的最小样本数
                             min_samples_leaf=1) # 叶子节点所需的最小样本数
# 训练模型
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
```

##### 4. Random Forest

```py
from sklearn.ensemble import RandomForestClassifier
# 初始化模型
clf = RandomForestClassifier(n_estimators=100,   # 构建的树的数量
                             criterion='gini',   # 衡量分裂质量的函数: 'gini' 或 'entropy'
                             max_depth=None,     # 树的最大深度
                             min_samples_split=2,# 节点分裂的最小样本数
                             min_samples_leaf=1) # 叶子节点所需的最小样本数
# 训练模型
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
```

##### 5. K-Nearest Neighbors

**用于计算最近邻的算法**:

1. `'ball_tree'`: 一种为了加速最近邻搜索而构建的数据结构。
2. `'kd_tree'`: KD树，也是为了加速最近邻搜索的数据结构，但对于高维数据可能不是很高效。
3. `'brute'`: 使用蛮力方法，计算每一对点之间的距离，当数据维度较低或样本数较少时效果还不错。
4. `'auto'`: 根据输入数据自动选择最合适的方法。

```py
from sklearn.neighbors import KNeighborsClassifier
# 初始化模型
clf = KNeighborsClassifier(n_neighbors=3,      # 考虑的邻居数量
                           weights='uniform',  # 邻居权重函数: 'uniform' 或 'distance'
                           algorithm='auto',   # 用于计算最近邻的算法: 'ball_tree', 'kd_tree', 'brute', 'auto'
                           p=2)                # 距离度量: 1为曼哈顿距离，2为欧几里得距离
# 训练模型
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
```



#### Unsupervised Model

#### PCA

**应用场景**：

1. **数据可视化**：对于高维数据，我们可以使用PCA将其降至2或3维，然后可视化它。
2. **速度与存储**：在机器学习中，降低数据的维度可以加速模型的训练，特别是在算法的计算复杂度随数据维度增加而增加时。
3. **噪声滤波**：PCA也可以作为一种去除数据中噪声的方法。
4. **特征提取**：在某些应用中，原始特征可能不是最佳选择。PCA可以为数据集生成一组新的特征。

**注意事项**：

1. PCA是基于线性假设的，这意味着它只能捕捉到数据中的线性变化。对于非线性数据，其他降维技术（如t-SNE或UMAP）可能更为合适。
2. 在应用PCA之前，通常需要将数据标准化，使所有特征都有均值为0、标准差为1。
3. PCA的主成分是基于数据的方差来确定的，这意味着它可能对异常值敏感。

```py
from sklearn.decomposition import PCA
# 初始化PCA
pca = PCA(n_components=2) #n_components：想要保留的主成分数量，降到2维

# 仅在训练集上拟合PCA
pca.fit(X_train)

# 使用在训练集上拟合的PCA模型来转换训练集和测试集
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 可视化转换后的训练数据
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', s=150)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS training dataset')
plt.show()
```



#### Clustering

##### 1. K-Means Clustering

K均值是一种分区方法，其核心思想是最小化每个点到其分配的簇中心点的距离的总和，同时最大化每个簇中心之间的距离。它是一个迭代算法，经常使用“质心”概念。

步骤如下:

1. 初始化K个簇的质心。
2. 将每个点分配到最近的质心，形成K个簇。
3. 重新计算每个簇的质心。
4. 重复2和3步，直到质心不再发生变化或达到最大迭代次数。

```py
from sklearn.cluster import KMeans
# 创建模型实例
kmeans = KMeans(n_clusters=3,               # 需要形成的簇的数量
                init='k-means++',          # 初始化中心的方法 'k-means++', 'random' 或 ndarray
                max_iter=300,              # 单次运行的 k-means 算法的最大迭代次数
                n_init=10,                 # 重新计算初始中心的次数
                random_state=0)            # 随机生成器的种子
# 训练模型
kmeans.fit(data)
# 获取聚类标签
labels = kmeans.labels_
```

特点：需要预先指定聚类数目。常常假设数据是球状的并且大小相似的。

应用场景：当你有一个明确的簇的数量，并且簇的形状大致是球形的。

缺点

- 必须提前指定K值。
- 对初始质心敏感。
- 可能会陷入局部最小值。

##### 2. Mean Shift Clustering

均值漂移是一种基于数据点密度的聚类算法。算法工作的基本原理是通过查找数据空间中的模式来估计数据的概率密度函数。

步骤如下:

1. 初始化滑窗的位置。
2. 计算滑窗内的数据点的均值。
3. 移动滑窗到该均值的位置。
4. 重复步骤2和3，直到滑窗移动的距离小于某个阈值。
5. 所有收敛到同一位置的滑窗将形成一个簇。

```py
from sklearn.cluster import MeanShift

mean_shift = MeanShift(bandwidth=2)       # 用于估计数据的核密度的滑窗大小
                                          # 如果未给出，则使用 sklearn.cluster.estimate_bandwidth
labels = mean_shift.fit_predict(data)
```

特点: 可以找到比KMeans更自然的簇，但在大数据上可能不是很可扩展。

应用场景: 当簇的形状是非球形的，或当簇的数量未知时。

缺点：

- 计算密集型，特别是对于大数据集。
- 选择合适的带宽是一个挑战。

##### 3. DBSCAN

DBSCAN 是基于数据点的密度的聚类方法。它将密度达到某个阈值的区域划为一个簇，并在高密度区域中进行扩展，直到密度低于某个阈值为止。

步骤如下:

1. 如果一个点的邻域中有超过 `min_samples` 个点，则开始一个新的簇。
2. 继续为当前簇添加满足这一条件的所有密度可达的点。
3. 如果不能再添加新的点，则开始下一个簇。

```py
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5,                 # 邻居的最大距离，用于从核心点扩展簇
                min_samples=5)           # 用于判断核心点的邻居的最小样本数
labels = dbscan.fit_predict(data)
```

特点:

- 不需要预先指定簇的数量。
- 可以发现任意形状的簇。
- 可以识别噪音，对噪声有很好的鲁棒性。

应用场景：当簇的数量未知并且簇的形状各异时，或者当数据中有噪声时。

缺点:

- 不适用于具有不同密度的聚类。
- `eps` 的选择可能很难确定。

##### 4. GMM (Gaussian Mixture Model)

GMM 假设数据是从几个高斯分布生成的。最大期望算法用于估计这些分布的参数。GMM是软聚类，其中每个点被分配到每个簇的概率。

步骤如下:

1. 随机选择高斯分布的参数。
2. (E步) 基于当前参数为每个数据点分配到某个高斯分布的概率。
3. (M步) 更新高斯分布的参数，以最大化观察到的数据的似然。

```py
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3,    # 高斯混合模型的数量
                      covariance_type='full', # 协方差类型 'full', 'tied', 'diag', 'spherical'
                      max_iter=100)            # EM算法的最大迭代次数

gmm.fit(data)
labels = gmm.predict(data)
```

应用场景：软聚类、连续数据的模型拟合、识别数据中可能的异常值或离群值。

特点:

- 为数据点提供概率分布。
- 可以建模椭圆形簇。

缺点:

- 计算密集型，特别是当 `n_components` 很大时。
- 必须选择正确的组件数量。

##### 5. Agglomerative Clustering

层次聚类旨在构建数据的层次结构。此方法可以是自顶向下或自底向上。在凝聚（自底向上）方法中，每个数据点开始时都在自己的簇中，然后算法迭代地将最接近的一对簇合并到一个簇中。

```py
from sklearn.cluster import AgglomerativeClustering

agglomerative = AgglomerativeClustering(n_clusters=3,          # 最终的簇数量
                                        affinity='euclidean',  # 用于计算距离的度量
                                        linkage='ward')        # 链接标准: 'ward', 'complete', 'average', 'single'

labels = agglomerative.fit_predict(data)
```

特点：将结果呈现为一个树状图或树状图。

应用场景: 当你对数据的层次结构感兴趣或当你想看到不同数量的簇时。

缺点:

- 计算密集型。
- 不能进行迭代赋值（一旦合并，簇就不能再分开）。

##### 6. AffinityPropagation

基于数据点之间的“消息传递”的聚类方法。它通过发送消息，让数据点选择它们的代表，从而决定簇的数量和中心。这种方法不需要预先指定簇的数量。

```py
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs

# 创建样本数据
X, _ = make_blobs(n_samples=300, centers=3, random_state=0)

# 使用AffinityPropagation进行聚类
af = AffinityPropagation(preference=-50,     # 设置点的偏好值，更小的值意味着更多的簇
                         damping=0.5,        # 学习参数，避免消息值数值的震荡，值在 0.5 和 1 之间
                         max_iter=200).fit(X)# AffinityPropagation 的最大迭代次数

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

# 打印簇的数量和轮廓系数
n_clusters_ = len(cluster_centers_indices)
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
```

应用场景:

- 小到中等大小的数据集。
- 当你不知道要找的簇的数量时。

特点

- 可以自动确定簇的数量。

缺点

- 算法复杂度高，不是很可扩展，对于大数据集可能表现不佳。



##### 7. SpectralClustering

基于图论的方法。它使用数据的相似性矩阵来构造图，并找到图的正规化拉普拉斯算子的特征向量。然后在这些向量上应用K-means或其他方法来完成聚类。

```py
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.datasets import make_circles

# 创建非球形数据
X, y = make_circles(n_samples=150, factor=.5, noise=.05)

# 使用SpectralClustering进行聚类
sc = SpectralClustering(n_clusters=2,               # 簇的数量
                        affinity="nearest_neighbors", # 如何构造仿射矩阵。'nearest_neighbors', 'rbf', 'precomputed' 或者一个可调用的函数
                        n_neighbors=10)              # 用于 affinity="nearest_neighbors" 的邻居数量
labels = sc.fit_predict(X)

# 打印轮廓系数
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
```

特点

- 数据的结构可以通过图或相似性矩阵来表示。
- 非球形数据或嵌套的簇。

应用场景：

- 可以处理复杂的、非球形的数据集。
- 维度约简: 在进行其他聚类算法之前，可以使用 SpectralClustering 进行维度约简。
- 图像分割: 用于图像处理中的图像分割，以找到图像中的相似区域。

劣势:

- 算法的复杂度较高，可能不适用于非常大的数据集。



### 7.Model Evaluation

1. **对于分类任务**：
   - **准确率 (Accuracy)**：正确分类的样本数与总样本数的比例。
   - **混淆矩阵 (Confusion Matrix)**：一个表格，用于描述分类模型的实际类别与预测类别之间的关系。
   - **精确率 (Precision)**：真正例与（真正例 + 假正例）的比例。
   - **召回率 (Recall 或 Sensitivity)**：真正例与（真正例 + 假反例）的比例。
   - **F1 分数 (F1 Score)**：精确率和召回率的调和平均。
   - **ROC 曲线 (Receiver Operating Characteristic curve)**：以假阳性率为 x 轴，真阳性率为 y 轴绘制的曲线。
   - **AUC (Area Under the ROC Curve)**：ROC 曲线下的面积，用于量化模型对正例和负例的区分能力。
2. **对于回归任务**：
   - **均方误差 (Mean Squared Error, MSE)**：预测值与实际值之间的平均平方差。
   - **均方根误差 (Root Mean Squared Error, RMSE)**：MSE 的平方根。
   - **平均绝对误差 (Mean Absolute Error, MAE)**：预测值与实际值之间的平均绝对差。
   - **R-squared (决定系数)**：模型解释的方差与总方差的比例，用于衡量模型的拟合优度。
3. **对于聚类任务**：
   - **轮廓系数 (Silhouette Coefficient)**：用于衡量样本与相同聚类中的其他样本的相似度与其他聚类中样本的相似度之间的差异。
   - **Davies-Bouldin Index**：各个聚类的平均相似度，值越小越好。
   - **Calinski-Harabasz Index**：聚类间方差与聚类内方差的比例，值越大越好。
4. **对于时间序列分析**：
   - **ACF (AutoCorrelation Function) 和 PACF (Partial AutoCorrelation Function)**：用于评估时间序列模型的残差。
   - **MAPE (Mean Absolute Percentage Error)**：平均绝对百分比误差，常用于衡量时间序列预测的准确性。
5. **模型验证策略**：
   - **留出法 (Hold-out Validation)**：数据分为训练集和测试集。
   - **交叉验证 (Cross-Validation)**：如 k-折交叉验证，数据被分成 k 份，模型在 k-1 份上训练并在剩余的一份上进行测试。这一过程重复 k 次。



```py
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
import matplotlib.pyplot as plt

# 真实标签和预测结果示例
y_true = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
y_pred = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]

# 准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
# 表示模型正确分类的比例，即正确预测的样本数占总样本数的比例。

# 混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
# 描述真实类别与预测类别的关系。格式：
# [[真负例, 假正例],
#  [假负例, 真正例]]

# 混淆矩阵的可视化表示
cm = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()

# 主要分类指标的文本报告
report = classification_report(y_true, y_pred)
print(report)

# 精确率
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")
# 表示预测为正类中实际为正类的比例，即真正例 / (真正例 + 假正例)

# 召回率
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")
# 表示真实为正类被预测为正类的比例，即真正例 / (真正例 + 假负例)

# F1分数
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")
# 精确率和召回率的调和平均，给出了一个综合的评估分数。

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
# 描述了在不同阈值下的假阳性率和真阳性率。

# AUC
area = auc(fpr, tpr)
print(f"AUC: {area}")
# ROC曲线下的面积，范围为[0,1]。值越接近1，表示模型的分类效果越好。
```



#### Cross-validation

##### 1. k-Fold Cross-Validation

k-折交叉验证：将原始数据均匀分成k个子集。每次将其中一个子集作为验证数据，其余k-1个子集合并后作为训练数据，重复k次，得到k个模型和性能评估。

应用场景: 当数据量不太大且数据分布较均匀时。

优点：可以充分利用数据进行训练和验证。

```py
#简单的k-折交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()

clf = RandomForestClassifier(n_estimators=50)
scores = cross_val_score(clf, iris.data, iris.target, cv=5) # 使用5-折交叉验证
# 分层k-折交叉验证（Stratified k-Fold）保持原始数据集中各个类的比例。
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, iris.data, iris.target, cv=skf)
```

##### 2. Leave-One-Out, LOO

留一交叉验证：每次只留下一个样本作为验证数据，其余的N-1个样本作为训练数据。

```py
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(clf, iris.data, iris.target, cv=loo)
```

##### 3. Leave-P-Out, LPO

留P交叉验证：从N个样本中留下P个样本作为验证数据。不同于LOO，每次留下P个样本进行验证。

```py
from sklearn.model_selection import LeavePOut

lpo = LeavePOut(p=2)
scores = cross_val_score(clf, iris.data, iris.target, cv=lpo)
```

##### 4. Shuffle Split

随机拆分：多次随机划分数据集为训练集和验证集。不保证每个数据点都被用作测试数据。适用于数据量大。

```py
from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
scores = cross_val_score(clf, iris.data, iris.target, cv=ss)
```



### 8.Hyperparameter tuning

在选择超参数优化方法时，选择哪种方法取决于具体任务、数据规模、计算资源和所使用的模型。例如，深度学习模型可能会受益于基于梯度的方法或 PBT，而传统的机器学习模型可能更适合于 Grid Search、Random Search 或 Bayesian Optimization。

##### 1. Grid Search

网格搜索

- 为每个超参数选择一组值，生成超参数所有可能的组合。
- 对于每种组合，都使用这组超参数训练模型，并验证其性能。
- 选择性能最佳的超参数组合。

```py
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 定义超参数网格
param_grid = {
    'C': [0.1, 1, 10],               # C: 正则化参数，值越小，正则化越强
    'gamma': [1, 0.1, 0.01],        # gamma: RBF核的系数
    'kernel': ['linear', 'rbf']     # kernel: 使用的核函数类型
}

# 初始化网格搜索对象
# refit=True 表示返回最佳参数的模型
# 这里的SVC()可以替换为任何scikit-learn中的其他模型，如：
#LinearRegression(): 线性回归
#DecisionTreeClassifier(): 决策树分类器
#RandomForestClassifier(): 随机森林分类器
#KNeighborsClassifier(): K最近邻分类器
grid = GridSearchCV(SVC(), param_grid, refit=True)

# 在训练集上训练网格搜索对象
grid.fit(X_train, y_train)
# 打印最佳超参数
print(grid.best_params_)
```

优势:

- 确保遍历了每个参数组合，找到的是最佳组合。

劣势:

- 计算成本高，因为它会训练每种可能的参数组合。
- 随着参数的增加，搜索空间指数级增长。

应用场景:

- 当超参数空间相对较小且计算资源足够时。

##### 2. Random search

随机搜索：

- 与 Grid Search 不同，而不是尝试所有可能的组合，Random Search 为每个超参数随机选择一个值，并进行有限次数的迭代。

优势:

- 更快且计算效率更高。
- 在大多数情况下都能找到非常优质的超参数。

劣势:

- 不保证找到最佳超参数组合。

应用场景:

- 当超参数空间较大或计算资源有限时。

##### 3. Bayesian Optimization

贝叶斯优化：

- 利用贝叶斯模型预测函数的性能。
- 在每次迭代中，选择下一个超参数，从而最大化预期提升。

```py
!pip install bayesian-optimization

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# 定义评估函数，返回交叉验证的均值
def svm_eval(C, gamma):
    return cross_val_score(SVC(C=C, gamma=gamma, kernel='rbf'), X_train, y_train, cv=3).mean()

# 初始化贝叶斯优化对象，并定义超参数的边界
optimizer = BayesianOptimization(f=svm_eval, pbounds={'C': (0.1, 10), 'gamma': (0.1, 1)})

# 进行贝叶斯优化
optimizer.maximize()
# 打印最佳超参数组合
print(optimizer.max)
```

优势:

- 通常比 Grid 和 Random Search 更快且更准确。
- 适用于大型超参数空间。

劣势:

- 更复杂，需要额外的库。
- 可能对于某些问题不那么稳健。

应用场景:

- 当超参数空间巨大，计算资源有限，但希望获得高质量的超参数时。

##### 4. Gradient-based Optimization

基于梯度的优化

- 在可微分的情境下，超参数的梯度可以被计算或估计。这意味着我们可以利用梯度下降或其他优化技术更新超参数。
- 对于深度学习模型，其中一些超参数，如学习率，可能会受益于这种方法。

```py
# 使用 PyTorch 的 lr_scheduler
import torch.optim as optim

# 初始化 SGD 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 使用 StepLR 调度器，每30个步骤学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(epochs):
    scheduler.step()
    # training process...
```

优势:

- 更快的收敛，特别是对于深度学习模型。
- 可以动态调整学习率。

劣势:

- 仅适用于可微分的超参数。

应用场景:

- 深度学习模型中的学习率调整。

##### 5. Evolutionary Algorithms

进化算法：Google 使用一种称为 "Population Based Training (PBT)" 的方法，这是一种进化算法的变种。

- 模拟生物进化的过程，如选择、交叉和变异来搜索最佳超参数。
- 开始时，会有一个“种群”由随机超参数组成的模型，之后基于他们的性能进行选择。

```py
from deap import base, creator, tools, algorithms
import random

# 定义适应度函数和个体
creator.create("FitnessMax", base.Fitness, weights=(1.0,))#名为FitnessMax的适应度函数，其目标是最大化（因为权重为正）
creator.create("Individual", list, fitness=creator.FitnessMax)#名为Individual的个体类，该类基于列表，并使用先前定义的FitnessMax为其分配适应度属性。

toolbox = base.Toolbox() #一个工具箱，包含创建随机个体和种群所需的函数。
toolbox.register("attr_float", random.random)     # attr_float: 随机浮点数生成器
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3) #注册一个函数来创建个体，背后的逻辑是重复地使用attr_float来填充个体的3个属性。
toolbox.register("population", tools.initRepeat, list, toolbox.individual)#注册一个函数来创建种群。
```

优势:

- 能够在大范围内搜索超参数。
- 可以避免局部最小值。

劣势:

- 可能需要更长的时间和计算资源。
- 不保证找到全局最优解。

应用场景:

- 大型超参数空间。
- 当需要避免局部最小值时。

##### 6. TPE (Tree-structured Parzen Estimator)

原理

- TPE 是序列模型的一种，用于优化。
- 它对超参数空间进行建模，尝试找出可能导致好性能的区域。

```py
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# 定义目标函数
def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK}

# 定义超参数空间
space = hp.uniform('x', -10, 10)

# 使用TPE算法进行优化
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)

# 打印最佳超参数
print(best)
```

优势:

- 通常比随机搜索更有效。
- 能够考虑超参数之间的依赖关系。

劣势:

- 计算上可能比其他方法更昂贵。
- 对于某些问题，可能不比随机搜索更好。

应用场景:

- 当超参数具有复杂的结构或依赖关系时。
- 当需要更好的超参数搜索方法时。














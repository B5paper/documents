# scikit-learn Note

## cache

* 写一段 python 代码，使用 sklern 的 svm 训练 fnirs channel 数据，做正常人和中风病人的二分类。目前 normal_data 和 stroke_data 是 list[np.ndarray] 的类型，已经准备好，其中 ndarray 的 shape 为 (num_timestamps, num_channels * num_wavelengths)。你需要把他们按七三开拆分成训练集和验证集。

    以下是一段使用 scikit-learn 的 SVM 对 fNIRS 通道数据进行二分类的 Python 代码。假设 `normal_data` 和 `stroke_data` 已按照问题描述准备好，代码会将每个样本的时间维度取平均作为特征，然后按 70/30 拆分训练集和验证集，训练 SVM 模型并评估性能。

    ```python
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report

    # 假设 normal_data 和 stroke_data 已经定义好
    # normal_data: list of np.ndarray, each shape (num_timestamps, num_channels * num_wavelengths)
    # stroke_data: list of np.ndarray, same format

    def extract_mean_features(data_list):
        """
        对每个样本的时间维度取平均，生成特征向量。
        """
        features = []
        for sample in data_list:
            # sample shape: (num_timestamps, num_features)
            mean_feat = np.mean(sample, axis=0)  # 沿时间轴取平均
            features.append(mean_feat)
        return np.array(features)

    # 提取特征
    X_normal = extract_mean_features(normal_data)
    X_stroke = extract_mean_features(stroke_data)

    # 创建标签：正常人为 0，中风病人为 1
    y_normal = np.zeros(len(X_normal))
    y_stroke = np.ones(len(X_stroke))

    # 合并数据和标签
    X = np.vstack((X_normal, X_stroke))
    y = np.hstack((y_normal, y_stroke))

    # 按 70% 训练，30% 验证拆分，并保持类别比例
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 标准化特征（使用训练集拟合，再转换验证集）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 训练 SVM（此处使用 RBF 核，可根据需要调整超参数）
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    # 在验证集上预测
    y_pred = svm.predict(X_val_scaled)

    # 输出评估结果
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Stroke']))
    ```

    代码说明

    1. **特征提取**：由于每个样本是一个时间序列矩阵，代码通过 `np.mean(axis=0)` 计算每个通道-波长组合在整个时间上的平均值，作为该样本的特征向量。  
    2. **数据拆分**：使用 `train_test_split` 按 70% 训练、30% 验证的比例拆分，并设置 `stratify=y` 确保两个类别在训练/验证集中的比例与原数据一致。  
    3. **标准化**：使用 `StandardScaler` 对特征进行标准化，以消除量纲影响，提升 SVM 的收敛速度和性能。  
    4. **模型训练与评估**：采用 RBF 核的 SVM，训练后输出验证集准确率和详细的分类报告（精确率、召回率、F1 值等）。

    如果时间序列长度不一致，取平均的方法依然适用；若需要更复杂的特征（如方差、斜率等），可相应修改 `extract_mean_features` 函数。


## note

安装：

`pip install -U scikit-learn`

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1, 2, 3],
    [11, 12, 13]]  # 2 samples, 3 features
y = [0, 1]  # classes of each sample
clf.fit(X, y)
clf.predict(X)  # predict classes of the training data
clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
```

transformers and pre-processors:

这些模块可以用 pipeline 预处理数据。和 mmdet 挺像的，不知道 sklearn 和 mmdet 他俩谁抄的谁。

```python
from sklearn.preprocessing import StandardScaler
X = [[0, 15],
    [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)
```

输出：

```
array([-1., 1.],
    [1., -1.])
```

注：

1. pre-processors, transformers 和 extimators 都继承自`BaseEstimator`。感觉这个设计挺智障的。

pipelines:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticREgression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LOgisticRergression()
)

# Load the iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)
accuracy_score(pipe.predict(X_test), y_test)
```

Model evaluation:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)  # defaults to 5-fold CV
result['test_score']  # r_squared score is high because dataset is easy
```

Automatic parameter searches

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizeSearchCV, train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10)}

# now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegression(random_state=0), n_iter=5, param_distributions=param_distributions, random_state=0)
search.fit(X_train, y_train)
search.best_params_

# the search object now acts like a normal random forest estimator
# with max_depth=9 and n_estimators=4
search.score(X_test, y_test)
```

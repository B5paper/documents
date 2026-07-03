# scikit-learn Note

## cache

* sklearn iris dataset inspect

    sklearn 中的 iris 数据集是个 toy 数据集，即不需要联网下载，内置的数据集。， 

    简介：

    Classes: 3

    Samples per class: 50

    Samples total: 150

    Dimensionality: 4

    Features: real, positive
    
    * 普通模式，会返回一个类字典的数据结构（Bunch）

        这个数据结构不只有字符串形式的 key，还直接有成员名形式的字段。有点像 matlab 中的 struct

        example:

        ```py
        from pandas._libs.lib import is_range_indexer
        import sklearn.datasets as D

        iris_data = D.load_iris()

        keys = iris_data.keys()
        vals = iris_data.values()
        print('keys: {}'.format(keys))

        print()
        print('--------')
        print()
        print('data:')
        data = iris_data['data']
        print(type(data))
        print(data.shape)
        print()

        print('target:')
        target = iris_data.target
        print(type(target))
        print(target.shape)
        print()

        print('feature_names:')
        feature_names = iris_data['feature_names']
        print(type(feature_names))
        print('len: {}'.format((len(feature_names))))
        print(feature_names)
        print()

        print('target_names:')
        target_names = iris_data.target_names
        print(type(target_names))
        print(target_names.shape)
        print(target_names)
        print()

        print('frame:')
        frame = iris_data.frame
        print(frame)
        print()

        print('DESCR:')
        DESCR = iris_data.DESCR
        print('{}'.format(DESCR))
        print()

        print('filename:')
        filename = iris_data.filename
        print('filename: {}'.format(filename))
        print()

        print('End.')

        ```

        output:

        ```
        keys: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

        --------

        data:
        <class 'numpy.ndarray'>
        (150, 4)

        target:
        <class 'numpy.ndarray'>
        (150,)

        feature_names:
        <class 'list'>
        len: 4
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

        target_names:
        <class 'numpy.ndarray'>
        (3,)
        ['setosa' 'versicolor' 'virginica']

        frame:
        None

        DESCR:
        .. _iris_dataset:

        Iris plants dataset
        --------------------

        **Data Set Characteristics:**

        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica

        :Summary Statistics:

        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================

        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988

        The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
        from Fisher's paper. Note that it's the same as in R, but not as in the UCI
        Machine Learning Repository, which has two wrong data points.

        This is perhaps the best known database to be found in the
        pattern recognition literature.  Fisher's paper is a classic in the field and
        is referenced frequently to this day.  (See Duda & Hart, for example.)  The
        data set contains 3 classes of 50 instances each, where each class refers to a
        type of iris plant.  One class is linearly separable from the other 2; the
        latter are NOT linearly separable from each other.

        .. dropdown:: References

          - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
            Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
            Mathematical Statistics" (John Wiley, NY, 1950).
          - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
            (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
          - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
            Structure and Classification Rule for Recognition in Partially Exposed
            Environments".  IEEE Transactions on Pattern Analysis and Machine
            Intelligence, Vol. PAMI-2, No. 1, 67-71.
          - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
            on Information Theory, May 1972, 431-433.
          - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
            conceptual clustering system finds 3 classes in the data.
          - Many, many more ...


        filename:
        filename: iris.csv

        End.
        ```

    * `return_X_y`模式

        ```py
        import sklearn.datasets as D

        iris_data = D.load_iris(return_X_y=True)

        print(type(iris_data))

        X, y = iris_data
        print('X type: {}, shape: {}'.format(type(X), X.shape))
        print('y type: {}, shape: {}'.format(type(y), y.shape))
        ```

        output:

        ```
        <class 'tuple'>

        Press ENTER or type command to continue
        <class 'tuple'>
        X type: <class 'numpy.ndarray'>, shape: (150, 4)
        y type: <class 'numpy.ndarray'>, shape: (150,)
        ```

    其他的 toy dataset list: <https://scikit-learn.org/stable/datasets/toy_dataset.html#toy-datasets>

    除了 toy dataset 之外的 dataset: <https://scikit-learn.org/stable/datasets.html>

* sklearn pca

    <https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py>

* sklearn examples

    <https://scikit-learn.org/stable/auto_examples/index.html#general-examples>

* sklearn random forest

    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>
* sklearn linear regression

    ```py
    from sklearn import linear_model
    reg = linear_model.LinearRegression()

    X = [[0, 0],
         [1, 1],
         [2, 2]]
    y = [0, 1, 2]
    reg.fit(X, y)
    coef = reg.coef_
    inte = reg.intercept_

    print('coef: {}'.format(coef))
    print('inte: {}'.format(inte))
    ```

    output:

    ```
    coef: [0.5 0.5]
    inte: 1.1102230246251565e-16
    ```

    `X`第一列是$x_1$，第二列是$x_2$。$y = c_0 + c_1 \cdot x_1 + c_2 \cdot x_2$

    拟合得到的结果`coef`即$[c_1, c_2]$，而`inte`即$c_0$

    对于二个 x 变量的线性拟合，得到的是一个平面，使得 y 到平面沿 y 轴方向的距离的平方和最小。

    对于一个 x 变量的线性拟合，得到的是一条直线，使得 y 到直线沿 y 轴方向的距离的平方和最小。即最小二乘法 (Ordinary Least Squares)。(这里的 ordinary 是什么意思？)
* sklearn pipeline

    ```py
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # create a pipeline object
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    # load the iris dataset and split it into train and test sets
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # fit the whole pipeline
    pipe.fit(X_train, y_train)
    # we can now use it like any other estimator
    score = accuracy_score(pipe.predict(X_test), y_test)
    print(score)  # 0.9736842105263158
    ```

    preprocessin 中还有哪些函数？

    除了 linear model，还有什么 model?

    load_iris 返回的原始数据是什么？

    为什么划分数据切片的 train_test_split 会被放在 model_selection 里？
* sklearn 中的 transormers and pre-processors

    ```py
    from sklearn.preprocessing import StandardScaler
    X = [[0, 15],
         [1, -10]]
    # scale data according to computed scaling values
    out = StandardScaler().fit(X).transform(X)
    print(out)
    # array([[-1.,  1.],
    #        [ 1., -1.]])
    ```

    注：

    1. transforms 竟然都继承于 BaseEstimator，这样确实方便了链式调用，但是类的意义变不对了
* sklearn tutorial

    随机森林的 example:

    ```py
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def main():
        clf = RandomForestClassifier(random_state=0)
        X = [[1, 2, 3],
             [11, 12, 13]]
        y = [0, 1]
        clf.fit(X, y)
        pred = clf.predict(X)
        print(pred)  # array([0, 1])
        clf.predict([[4, 5, 6], [14, 15, 16]])
        print(pred)  # array([0, 1])
        return

    if __name__ == '__main__':
        main()
    ```

    如果这段代码可以跑通，说明 sklearn 环境没有问题。

    sklearn 中默认 X 的 shape 为`(num_samples, num_features)` 
    
    y 通常是个 vector，行或列无所谓，用行就行。第 i 个元素代表 X 中第 i 个 sample 的 target.
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

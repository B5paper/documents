# scikit-learn Note

## cache

* 详细讲讲 sklearn 中 accuracy_score() 的用法

    1. 函数作用

        `accuracy_score` 用于计算**分类准确率**，即：
        \[
        \text{准确率} = \frac{\text{预测正确的样本数}}{\text{总样本数}}
        \]
        它返回一个浮点数（或一个数组，取决于参数设置），表示模型预测的准确程度。

    2. 导入方式

        ```python
        from sklearn.metrics import accuracy_score
        ```

    3. 函数签名与参数

        ```python
        accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
        ```

        **参数说明：**

        | 参数 | 类型 | 说明 |
        |------|------|------|
        | `y_true` | 1d array-like | 真实标签 |
        | `y_pred` | 1d array-like | 预测标签 |
        | `normalize` | bool, 默认 True | 若为 True，返回准确率（0~1 之间的浮点数）；若为 False，返回预测正确的样本数（整数） |
        | `sample_weight` | array-like, 可选 | 每个样本的权重，用于计算加权准确率 |

    4. 基本用法示例

        1. 二分类

            ```python
            from sklearn.metrics import accuracy_score

            y_true = [0, 1, 1, 0, 1]
            y_pred = [0, 1, 0, 0, 1]

            acc = accuracy_score(y_true, y_pred)
            print(acc)          # 0.8 （4/5 正确）
            ```

        2. 多分类

            ```python
            y_true = [0, 1, 2, 2, 1]
            y_pred = [0, 2, 2, 1, 1]

            print(accuracy_score(y_true, y_pred))  # 0.6 （3/5 正确）
            ```

        3. 返回正确样本数（`normalize=False`）

            ```python
            acc_count = accuracy_score(y_true, y_pred, normalize=False)
            print(acc_count)   # 3（正确个数）
            ```

    5. 样本权重（sample_weight）

        当样本重要程度不同时，可以给每个样本赋权。

        ```python
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]  # 第3个样本预测错误

        # 默认权重都是1，准确率为 3/4 = 0.75
        print(accuracy_score(y_true, y_pred))  # 0.75

        # 给第3个样本（索引2）高权重，则错误影响更大
        weights = [1, 1, 10, 1]
        print(accuracy_score(y_true, y_pred, sample_weight=weights))  
        # 加权正确数 = 1+1+0+1=3，总权重=13，准确率=3/13≈0.2308
        ```

    6. 注意事项与常见坑

        1. 标签顺序不重要

            `accuracy_score` 只比较对应位置的 `y_true` 和 `y_pred` 是否相等，不关心类别顺序。

        2. 输入格式要求

            - 可以是列表、NumPy数组、Pandas Series等。
            - 两者长度必须相等，否则报错。

        3. 多标签（multi-label）问题不支持

            `accuracy_score` 不适用于多标签分类（每个样本有多个标签）。

            对于多标签，应使用 `jaccard_score` 或 `f1_score`（带 `average` 参数）。

        4. 类别不平衡时的局限

            准确率对类别不平衡非常敏感（例如 90% 负例，全预测负类也能得 90% 准确率）。
            
            这种情况建议搭配混淆矩阵、精确率/召回率或 F1-score 一起使用。

    7. 与其他指标的简单对比

        | 指标 | 适用场景 | 优点 | 缺点 |
        |------|----------|------|------|
        | accuracy_score | 平衡分类问题 | 直观、计算快 | 对不平衡敏感 |
        | precision_score | 关注假阳性代价高（如垃圾邮件检测） | 衡量“预测为正的有多准” | 忽视假阴性 |
        | recall_score | 关注假阴性代价高（如疾病筛查） | 衡量“正例被找出多少” | 忽视假阳性 |
        | f1_score | 需要平衡 P/R | 调和平均数 | 不直观 |

    8. 完整代码总结

        ```python
        from sklearn.metrics import accuracy_score
        import numpy as np

        y_true = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 2, 2, 1, 1])

        # 默认准确率
        print(accuracy_score(y_true, y_pred))          # 0.6

        # 返回正确个数
        print(accuracy_score(y_true, y_pred, normalize=False))  # 3

        # 加权版本
        weights = [1, 1, 0.5, 1, 1]
        print(accuracy_score(y_true, y_pred, sample_weight=weights))  
        # 正确加权和 = 1+1+0.5+0+1 = 3.5，总权重4.5，结果≈0.7778
        ```

    ---

    如果你还想了解它和 `metrics.classification_report` 的配合使用，或者如何处理多标签场景下的准确率，可以继续问我，我可以进一步展开。

* 详细讲讲 SVC() 的用法

    好的，我们来详细讲解一下 `sklearn.svm.SVC()`（Support Vector Classifier，支持向量分类器）。

    `SVC` 是 scikit-learn 中支持向量机（SVM）的分类实现，它基于 **libsvm** 库，是一个非常强大且经典的分类算法。

    1. 核心原理

        SVC 的目标是找到一个**超平面**，使得不同类别的样本之间的**间隔最大化**。

        - **线性可分**：找到硬间隔超平面
        - **线性不可分**：引入**软间隔**（允许部分样本误分类），通过惩罚参数 `C` 控制
        - **非线性**：通过**核函数**（Kernel）将数据映射到高维空间，使其线性可分

    2. 基本用法

        ```python
        from sklearn.svm import SVC
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # 加载数据
        iris = load_iris()
        X, y = iris.data, iris.target

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 标准化（SVM 对尺度敏感，必须做）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建 SVC 模型（默认参数）
        svc = SVC()

        # 训练
        svc.fit(X_train_scaled, y_train)

        # 预测
        y_pred = svc.predict(X_test_scaled)

        # 评估
        print("准确率:", svc.score(X_test_scaled, y_test))
        ```

    3. 核心参数详解

        1. C（惩罚参数, Penalty Parameter, or Regularization Parameter）

            - **默认值**: 1.0
            - **作用**: 控制对误分类样本的惩罚力度。
            - **C 越大**：对误分类惩罚越重，决策边界更复杂，可能过拟合（硬间隔）
            - **C 越小**：允许更多误分类，决策边界更平滑，可能欠拟合（软间隔）
            - **调参建议**: 通常使用网格搜索（GridSearchCV）在对数尺度上搜索，如 `[0.001, 0.01, 0.1, 1, 10, 100]`

            ```python
            # C 的作用示例
            svc_hard = SVC(C=100)   # 强硬分类
            svc_soft = SVC(C=0.01)  # 宽松分类
            ```

        2. kernel（核函数）

            - **默认值**: 'rbf'
            - **作用**: 定义如何将数据映射到高维空间，以处理非线性问题。
            - **可选值**:

            | 核函数 | 参数 | 适用场景 |
            |--------|------|----------|
            | `'linear'` | 无 | 线性可分数据，特征数多 |
            | `'poly'` | `degree`（多项式次数） | 多项式决策边界 |
            | `'rbf'` | `gamma`（高斯核宽度） | 最常用，适合大多数非线性问题 |
            | `'sigmoid'` | `gamma`, `coef0` | 类似神经网络的激活函数 |
            | 自定义核函数 | 传入函数 | 特殊需求 |

            ```python
            svc_linear = SVC(kernel='linear')
            svc_poly = SVC(kernel='poly', degree=3)
            svc_rbf = SVC(kernel='rbf')
            ```

        3. gamma（RBF/Poly/Sigmoid 核参数）**

            - **默认值**: `'scale'`（在 scikit-learn 0.22+ 中）
            - **作用**: 控制单个训练样本的影响范围。
            - **gamma 越大**：每个样本影响范围小，决策边界更复杂、更扭曲，容易过拟合
            - **gamma 越小**：每个样本影响范围大，决策边界更平滑，偏向欠拟合
            - **可选值**:
            - `'scale'`: `1 / (n_features * X.var())`（自动计算）
            - `'auto'`: `1 / n_features`
            - 正浮点数：手动指定

            ```python
            svc_gamma_large = SVC(gamma=10)   # 过拟合风险高
            svc_gamma_small = SVC(gamma=0.01) # 欠拟合风险高
            ```

        4. degree（多项式核次数）

            - **默认值**: 3
            - **作用**: 仅当 `kernel='poly'` 时有效，指定多项式的次数。

            ```python
            svc_poly_degree5 = SVC(kernel='poly', degree=5)
            ```

        5. coef0（独立项系数）

            - **默认值**: 0.0
            - **作用**: 仅对 `'poly'` 和 `'sigmoid'` 核有效，影响高维映射的偏移。

            ```python
            svc_poly = SVC(kernel='poly', coef0=1)
            ```

        6. shrinking（启发式收缩）

            - **默认值**: True
            - **作用**: 是否使用收缩启发式方法来加速训练。通常保持默认 True。

        7. probability（概率估计）

            - **默认值**: False
            - **作用**: 是否启用概率预测（`predict_proba` 方法）。
            - 启用后会进行**Platt 缩放**，增加训练时间，但可以输出类别概率。
            - 适用于需要概率输出的场景（如排序、风险评估）。

            ```python
            svc_prob = SVC(probability=True)
            svc_prob.fit(X_train, y_train)
            probs = svc_prob.predict_proba(X_test)  # 返回每个类别的概率
            ```

        8. class_weight（类别权重）
            
            - **默认值**: None
            - **作用**: 处理类别不平衡问题。
            - `None`: 所有类别权重相等
            - `'balanced'`: 自动根据样本数调整权重，`n_samples / (n_classes * np.bincount(y))`
            - 字典: 手动指定，如 `{0: 1, 1: 10}`

            ```python
            # 处理不平衡数据
            svc_balanced = SVC(class_weight='balanced')
            ```

        9. random_state（随机种子）

            - **默认值**: None
            - **作用**: 控制随机数生成（影响数据打乱等），用于结果可复现。

            ```python
            svc = SVC(random_state=42)
            ```

    4. 重要属性（Attributes）

        | 属性 | 说明 |
        |------|------|
        | `support_` | 支持向量的索引 |
        | `support_vectors_` | 支持向量的样本值 |
        | `n_support_` | 每个类别的支持向量数量 |
        | `dual_coef_` | 对偶系数（支持向量的权重） |
        | `coef_` | 仅在线性核时有效，特征权重（可用于特征重要性） |
        | `intercept_` | 决策函数的截距（偏置） |
        | `classes_` | 类别标签 |
        | `fit_status_` | 拟合状态（0 表示成功，1 表示失败） |

        ```python
        print("支持向量数量:", svc.n_support_)
        print("支持向量:", svc.support_vectors_)
        ```

    5. 重要方法（Methods）

        | 方法 | 说明 |
        |------|------|
        | `fit(X, y)` | 训练模型 |
        | `predict(X)` | 预测类别 |
        | `predict_proba(X)` | 预测类别概率（需 `probability=True`） |
        | `predict_log_proba(X)` | 对数概率 |
        | `decision_function(X)` | 决策函数值（超平面距离），可用于置信度 |
        | `score(X, y)` | 返回平均准确率 |
        | `set_params(**params)` | 设置参数 |
        | `get_params()` | 获取参数 |

        ```python
        # 决策函数（距离超平面的距离）
        decision_vals = svc.decision_function(X_test)
        print("决策值:", decision_vals[:5])
        ```

    6. 数据预处理（重要！）

        **SVM 对特征尺度极其敏感**，因为核函数依赖欧氏距离。必须进行标准化或归一化：

        ```python
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svc = SVC()
        svc.fit(X_train_scaled, y_train)
        ```

        **不标准化的后果**：特征值大的特征主导距离计算，模型性能大幅下降。

    7. 调参策略（GridSearchCV）

        SVC 有多个超参数，通常使用网格搜索寻找最优组合：

        ```python
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'linear']
        }

        svc = SVC(random_state=42)
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        print("最佳参数:", grid_search.best_params_)
        print("最佳交叉验证得分:", grid_search.best_score_)

        # 使用最佳模型预测
        best_svc = grid_search.best_estimator_
        y_pred = best_svc.predict(X_test_scaled)
        ```

    8. 核函数选择指南

        | 数据特点 | 推荐核函数 | 原因 |
        |---------|-----------|------|
        | 特征数 > 样本数 | `linear` | 避免过拟合，计算快 |
        | 特征数 < 样本数 | `rbf` | 处理非线性能力强 |
        | 特征数 ≈ 样本数 | `rbf` | 默认首选 |
        | 线性可分 | `linear` | 简单高效 |
        | 需要解释特征重要性 | `linear` | 可以看 `coef_` |
        | 文本分类（高维稀疏） | `linear` | 线性核效果通常最好 |
        | 图像、复杂模式 | `rbf` 或 `poly` | 高维映射 |

    9. 多分类策略

        SVC 原生支持多分类，使用 **一对一**（One-vs-One, OvO）策略：

        - 对于 K 个类别，训练 K*(K-1)/2 个二分类器
        - 每个分类器区分两个类别
        - 最终通过投票决定类别

        也可以通过 `decision_function_shape` 参数调整：

        - `'ovr'`（One-vs-Rest）：训练 K 个分类器，每个类别 vs 其余
        - `'ovo'`（默认）：一对一

        ```python
        svc_ovr = SVC(decision_function_shape='ovr')
        ```

    10. 优缺点

        ✅ 优点

        - 有效处理高维数据
        - 核技巧使其能处理非线性问题
        - 基于支持向量，内存效率高（只依赖支持向量）
        - 理论保证（VC 维理论）

        ❌ 缺点

        - 对大规模数据（> 10万样本）训练慢（O(n²) 或 O(n³)）
        - 对参数和核函数选择敏感
        - 不直接输出概率（需要 Platt 缩放）
        - 对缺失数据敏感

    11. 大规模数据替代方案

        | 方案 | 适用场景 |
        |------|----------|
        | `LinearSVC` | 线性核，大规模数据，基于 liblinear |
        | `SGDClassifier` | 超大规模数据，在线学习 |
        | `NuSVC` | 用 ν 参数代替 C，控制支持向量数量 |

        ```python
        from sklearn.svm import LinearSVC

        linear_svc = LinearSVC(C=1.0, dual=False)  # dual=False 当样本数 > 特征数
        linear_svc.fit(X_train, y_train)
        ```

    12. 完整实战示例（带调参）

        ```python
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix

        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                n_informative=15, random_state=42)

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 调参
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }

        svc = SVC(random_state=42, probability=True)
        grid = GridSearchCV(svc, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        # 最佳模型
        best = grid.best_estimator_
        y_pred = best.predict(X_test_scaled)
        y_proba = best.predict_proba(X_test_scaled)

        print("最佳参数:", grid.best_params_)
        print("测试集准确率:", best.score(X_test_scaled, y_test))
        print("\n分类报告:\n", classification_report(y_test, y_pred))
        print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
        ```

    13. 总结

        | 要点 | 描述 |
        |------|------|
        | 核心作用 | 通过最大化间隔进行分类 |
        | 必须预处理 | **标准化**（StandardScaler） |
        | 关键参数 | `C`（惩罚）、`kernel`（核函数）、`gamma`（核宽度） |
        | 调参方法 | GridSearchCV 或 RandomizedSearchCV |
        | 多分类 | 默认一对一（OvO），可改为 OvR |
        | 概率输出 | 设置 `probability=True` |
        | 优缺点 | 强大但慢，适合中小规模数据 |
        | 大规模替代 | LinearSVC、SGDClassifier |

    如果你有具体的使用场景（如文本分类、图像识别、不平衡数据等），我可以进一步给出针对性的调参建议！

* 详细讲讲 sklearn 中 StandardScaler() 的用法

    好的，我们来详细讲解一下 `sklearn.preprocessing.StandardScaler` 的用法。

    `StandardScaler` 是 scikit-learn 中最常用、最基础的数据预处理工具之一。它的作用是对特征数据进行**标准化**（也称为 Z-score 标准化）。

    1. 核心原理

        **标准化公式**：  
        对于每个特征（每一列），`StandardScaler` 会计算：
        - 均值（mean）μ
        - 标准差（standard deviation） σ

        然后对每个样本的该特征值 x 进行转换：
        \[
        x' = \frac{x - \mu}{\sigma}
        \]

        **转换后的结果**：
        - 均值变为 0
        - 方差变为 1（即标准差为 1）

        这使得不同量纲、不同数量级的特征具有可比性，且对许多机器学习算法的性能有正面影响。

    ---

    2. 基本用法（代码示例）

        ```python
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # 原始数据（2个样本，3个特征）
        X = np.array([[1, 2, 3],
                    [4, 5, 6]])

        # 1. 创建 StandardScaler 对象
        scaler = StandardScaler()

        # 2. 拟合数据并转换（一步完成）
        X_scaled = scaler.fit_transform(X)

        print(X_scaled)
        ```

        输出（约等于）：
        ```
        [[-1. -1. -1.]
        [ 1.  1.  1.]]
        ```

        每个特征均值为0，标准差为1。

    ---

    3. 分步操作（fit + transform）

        更规范的做法是分开两步，尤其在训练/测试集上：

        ```python
        # 训练集
        X_train = np.array([[1, 2, 3],
                            [4, 5, 6]])

        # 测试集（新数据）
        X_test = np.array([[7, 8, 9],
                        [10, 11, 12]])

        scaler = StandardScaler()

        # 1. 在训练集上拟合（计算均值和标准差）
        scaler.fit(X_train)

        # 2. 转换训练集
        X_train_scaled = scaler.transform(X_train)

        # 3. 使用同样的参数转换测试集（重要！）
        X_test_scaled = scaler.transform(X_test)

        print("训练集均值:", scaler.mean_)
        print("训练集标准差:", scaler.scale_)
        ```

        **关键点**：测试集必须使用训练集的均值和标准差进行转换，而不是重新 fit，否则会导致数据泄露和评估失真。

    ---

    4. 重要属性和方法

        | 属性/方法 | 说明 |
        |---------|------|
        | `fit(X)` | 计算 X 每个特征的均值（`mean_`）和标准差（`scale_`） |
        | `transform(X)` | 使用已计算的均值和标准差进行标准化 |
        | `fit_transform(X)` | 拟合 + 转换（一步完成） |
        | `inverse_transform(X)` | 将标准化后的数据还原为原始尺度 |
        | `mean_` | 每个特征的均值（numpy 数组） |
        | `scale_` | 每个特征的标准差（numpy 数组） |
        | `var_` | 每个特征的方差（`scale_` 的平方） |
        | `n_samples_seen_` | 拟合时使用的样本数 |

    5. 逆变换（还原数据）

        ```python
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 还原
        X_original = scaler.inverse_transform(X_scaled)

        # X_original 应等于原始 X（可能有微小浮点误差）
        ```

    6. 重要参数

        `StandardScaler` 的构造函数主要有两个参数：

        - **`copy`** (bool, 默认 True)  
        是否在转换时复制数据。如果为 False，可能会直接修改输入数据（谨慎使用）。

        - **`with_mean`** (bool, 默认 True)  
        是否将数据中心化（减去均值）。  
        设为 `False` 时，只除以标准差（相当于缩放，不中心化）。  
        适用于稀疏矩阵（因为减去均值会破坏稀疏性）。

        - **`with_std`** (bool, 默认 True)  
        是否将数据缩放到单位方差（除以标准差）。  
        设为 `False` 时，只减去均值（中心化），不缩放。

        **示例**：只缩放，不中心化（适合稀疏数据）
        ```python
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)
        ```

    7. 使用场景与注意事项

        ✅ **什么时候使用 StandardScaler？**

        - 线性模型（如线性回归、逻辑回归、SVM）
        - 基于距离的算法（KNN、K-Means）
        - 梯度下降优化的算法（神经网络、PCA）
        - 任何假设特征服从正态分布或需要统一量纲的场景

        ❌ **什么时候不用（或慎用）？**

        - 树模型（决策树、随机森林、XGBoost）通常不受特征尺度影响，可以不标准化
        - 特征本身具有明确物理意义且不希望改变时

        ⚠️ **注意事项**

        1. **异常值敏感**：均值和标准差受异常值影响大，如果数据有较多异常值，可考虑使用 `RobustScaler`（基于中位数和四分位数）。
        2. **只对数值特征**：类别特征需要其他编码方式。
        3. **训练/测试一致**：永远先 `fit` 训练集，再用同一个 scaler 去 `transform` 测试集。

    8. 与管道（Pipeline）结合使用

        在实际项目中，通常将 `StandardScaler` 放在 `Pipeline` 中，避免数据泄露：

        ```python
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ])

        pipe.fit(X_train, y_train)
        pipe.predict(X_test)
        ```

        这样在交叉验证时，每次训练折都会重新拟合 scaler，测试折使用该折的统计量，更严谨。

    9. 完整实战示例

        ```python
        import pandas as pd
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        # 加载数据
        data = load_iris()
        X, y = data.data, data.target

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 建模
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_scaled, y_train)

        # 预测与评估
        y_pred = knn.predict(X_test_scaled)
        print("准确率:", accuracy_score(y_test, y_pred))
        ```

    10. 总结

        | 要点 | 描述 |
        |------|------|
        | 作用 | 将每个特征转换为均值0，方差1 |
        | 公式 | (x - μ) / σ |
        | 核心方法 | `fit()`, `transform()`, `fit_transform()` |
        | 关键属性 | `mean_`, `scale_` |
        | 逆变换 | `inverse_transform()` |
        | 适用算法 | 线性模型、距离模型、神经网络等 |
        | 注意 | 只 fit 训练集，测试集用同一个 scaler 的 transform |

    如果你还有具体的使用场景（比如大数据集、稀疏数据、在线学习等），我可以进一步展开。

* sklearn pca

    <https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py>

* sklearn examples

    <https://scikit-learn.org/stable/auto_examples/index.html#general-examples>

* sklearn random forest

    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>

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

## Topics

### metrics

* 详细讲讲 `classification_report()` 的用法

    `classification_report()` 是分类任务中**最全面的单函数评估工具**，能一次性输出多个核心指标。

    1. 函数作用

        `classification_report` 用于构建一个**文本报告**，展示每个类别的：

        - **精确率（Precision）**
        - **召回率（Recall）**
        - **F1-score**
        - **支持数（Support）**

        它帮助您快速了解模型在每个类别上的表现，而不仅仅是总体准确率。

    2. 导入方式

        ```python
        from sklearn.metrics import classification_report
        ```

    3. 函数签名与参数

        ```python
        classification_report(
            y_true,
            y_pred,
            *,
            labels=None,
            target_names=None, 
            sample_weight=None,
            digits=2,
            output_dict=False, 
            zero_division='warn'
        )
        ```

        **核心参数详解：**

        * `y_true`: 1d array-like, 真实标签
        * `y_pred`: 1d array-like, 预测标签
        * `labels`: list, 可选, 指定要报告的类别列表（默认使用所有类别）
        * `target_names`, list, 可选, 类别的显示名称（用于美化输出）
        * `sample_weight`, array-like, 可选, 样本权重
        * `digits`: int, 默认 2, 保留小数位数
        * `output_dict`: bool, 默认 False, 若为 True，返回字典而非字符串
        * `zero_division`: str, 默认 'warn', 当分母为0时的处理方式：'warn'（警告并返回0）、'0'（返回0）、'1'（返回1）

    4. 基本用法示例

        1. 二分类

            ```python
            from sklearn.metrics import classification_report

            y_true = [0, 1, 0, 1, 0, 1, 0, 1]
            y_pred = [0, 1, 0, 0, 0, 1, 1, 1]

            print(classification_report(y_true, y_pred))
            ```

            **输出：**

            ```
                        precision    recall  f1-score   support

                    0       0.67      0.75      0.71         4
                    1       0.75      0.67      0.71         4

                accuracy                           0.71         8
            macro avg       0.71      0.71      0.71         8
            weighted avg       0.71      0.71      0.71         8
            ```

        2. 多分类

            ```python
            y_true = [0, 1, 2, 2, 1, 0, 2, 1]
            y_pred = [0, 2, 2, 1, 1, 0, 1, 1]

            print(classification_report(y_true, y_pred))
            ```

            **输出：**

            ```
                        precision    recall  f1-score   support

                    0       1.00      1.00      1.00         2
                    1       0.67      1.00      0.80         3
                    2       0.50      0.33      0.40         3

                accuracy                           0.75         8
            macro avg       0.72      0.78      0.73         8
            weighted avg       0.69      0.75      0.70         8
            ```

    5. 参数详解与高级用法

        1. 使用 `target_names` 自定义类别名称

            当类别是数字时，可以替换为有意义的名称：

            ```python
            y_true = [0, 1, 2, 2, 1]
            y_pred = [0, 2, 2, 1, 1]

            print(classification_report(
                y_true, 
                y_pred, 
                target_names=['苹果', '香蕉', '橙子']
            ))
            ```

            **输出：**

            ```
                        precision    recall  f1-score   support

                    苹果       1.00      1.00      1.00         1
                    香蕉       0.50      1.00      0.67         2
                    橙子       0.50      0.50      0.50         2

                accuracy                           0.60         5
            macro avg       0.67      0.83      0.72         5
            weighted avg       0.60      0.60      0.57         5
            ```

        2. 使用 `labels` 指定报告的类别顺序

            ```python
            # 强制按 [2, 0, 1] 顺序报告
            print(classification_report(
                y_true, 
                y_pred, 
                labels=[2, 0, 1],
                target_names=['橙子', '苹果', '香蕉']
            ))
            ```

        3. 使用 `digits` 控制精度

            ```python
            print(classification_report(y_true, y_pred, digits=4))
            ```

        4. 使用 `output_dict` 返回字典（便于程序处理）

            ```python
            report_dict = classification_report(
                y_true, 
                y_pred, 
                output_dict=True
            )

            print(report_dict['0'])          # 类别0的指标
            print(report_dict['accuracy'])   # 准确率
            print(report_dict['macro avg'])  # 宏平均
            ```

            **返回的字典结构：**

            ```python
            {
                '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
                '1': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.666..., 'support': 2},
                '2': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 2},
                'accuracy': 0.6,
                'macro avg': {'precision': 0.666..., 'recall': 0.833..., 'f1-score': 0.722..., 'support': 5},
                'weighted avg': {'precision': 0.6, 'recall': 0.6, 'f1-score': 0.566..., 'support': 5}
            }
            ```

        5. 使用 `sample_weight` 加权

            ```python
            weights = [1, 1, 0.5, 0.5, 1]  # 第3、4个样本权重减半
            print(classification_report(y_true, y_pred, sample_weight=weights))
            ```

        6. 处理 `zero_division`（防止除以零）

            当某个类别在真实标签或预测标签中完全不存在时：

            ```python
            y_true = [0, 0, 0]
            y_pred = [0, 0, 1]  # 类别1从未真实出现

            # 默认会警告，并给类别1的 precision 设为 0.0
            print(classification_report(y_true, y_pred))

            # 安静模式，返回 0
            print(classification_report(y_true, y_pred, zero_division=0))

            # 或者返回 1
            print(classification_report(y_true, y_pred, zero_division=1))
            ```

    6. 报告中的指标详解

        1. 每类指标

            假设二分类，关注类别 A：

            | 指标 | 公式 | 含义 |
            |------|------|------|
            | **Precision** | TP / (TP + FP) | 预测为 A 的样本中，真正是 A 的比例 |
            | **Recall** | TP / (TP + FN) | 真正的 A 中，被正确预测出来的比例 |
            | **F1-score** | 2 × (P × R) / (P + R) | Precision 和 Recall 的调和平均 |
            | **Support** | TP + FN | 该类在真实标签中的样本数 |

        2. 总体平均指标

            | 指标 | 计算方式 | 适用场景 |
            |------|----------|----------|
            | **accuracy** | (TP+TN) / 总数 | 总体正确率 |
            | **macro avg** | 对每个类别的指标**简单平均** | 每个类别同等重要（不考虑类别不平衡） |
            | **weighted avg** | 按每个类别的 support 加权平均 | 类别不平衡时更合理 |

        **举例说明区别：**

        ```python
        # 极度不平衡数据
        y_true = [0]*90 + [1]*10
        y_pred = [0]*90 + [0]*10  # 全预测为0

        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"类别0的 recall: {report['0']['recall']}")      # 1.0
        print(f"类别1的 recall: {report['1']['recall']}")      # 0.0
        print(f"macro avg recall: {report['macro avg']['recall']}")  # 0.5
        print(f"weighted avg recall: {report['weighted avg']['recall']}")  # 0.9
        print(f"accuracy: {report['accuracy']}")                # 0.9
        ```

    7. 注意事项与常见坑

        1. 输入必须是一维
            
            不支持多标签分类（multi-label），多标签应使用 `multilabel_confusion_matrix` 等。

        2. 类别顺序影响 `macro avg`
            
            `macro avg` 对每个类别同等对待，即使某个类别样本极少。

            如果类别不平衡严重且想体现多数类影响，看 `weighted avg`。

        3. 与 `accuracy_score` 的区别

            | | `accuracy_score` | `classification_report` |
            |---|---|---|
            | 输出 | 单一数值 | 完整报告（多个指标） |
            | 信息量 | 少 | 丰富 |
            | 适用 | 快速评估 | 深入分析 |
            | 不平衡敏感 | 严重 | 通过 macro/weighted 可缓解 |

    8. 完整实战示例

        ```python
        from sklearn.metrics import classification_report
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        # 生成数据
        X, y = make_classification(n_samples=1000, n_classes=3, weights=[0.7, 0.2, 0.1], random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 训练
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)

        # 生成报告
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=['多数类', '中等类', '少数类'],
            digits=3
        ))

        # 获取字典用于后续分析
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        print(f"\n少数类的 F1: {report_dict['少数类']['f1-score']:.3f}")
        print(f"加权平均精确率: {report_dict['weighted avg']['precision']:.3f}")
        ```

    9. 总结

        - `classification_report` 是**分类问题分析的标准起点**
        - 同时提供 P/R/F1，比 `accuracy_score` 信息丰富得多
        - 通过 `macro avg` 和 `weighted avg` 可以判断模型在不同类别上的平衡性
        - 使用 `output_dict=True` 可以将结果集成到自动化流程中

    ---

    如果你还想了解：
    
    - 如何处理多标签分类（multi-label）的评估？
    - 如何画出每个类别的混淆矩阵？
    - 如何自定义指标添加到报告中？

    欢迎继续问我，我可以再展开讲解！

### 常用模型

* linear regression

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

### 自定义数据集

* pipeline

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

* `train_test_split()`

    它是 scikit-learn 中最常用、最基础的数据划分工具。

    * 函数原型

        ```py
        sklearn.model_selection.train_test_split(
            *arrays,
            test_size=None,
            train_size=None,
            random_state=None,
            shuffle=True,
            stratify=None
        )
        ```

    * 参数详解

        * `*arrays`（位置参数）

            * 接受一个或多个数组/矩阵（如特征矩阵 `X` 和标签向量 `y`）。

            * 它们必须具有相同的**第一维度（样本数）**。

            * 常见用法：`train_test_split(X, y)` 或 `train_test_split(X, y, group_ids)`。

        * `test_size`（浮点数或整数，默认 `None`）：
            
            - 决定测试集的大小。
            - 如果为浮点数（0.0~1.0），表示测试集占总样本的比例。
            - 如果为整数，表示测试集的绝对样本数量。
            - 如果为 `None`，则使用 `train_size` 的补集（若 `train_size` 也为 `None`，则默认 `test_size=0.25`）。

        * `train_size`（浮点数或整数，默认 `None`）：

            - 决定训练集的大小，规则同 `test_size`。
            - 通常只设置 `test_size`，让 `train_size` 自动为 `1 - test_size`。

        * `random_state`（整数或 `None`，默认 `None`）：

            - 控制随机打乱过程的随机种子。

            - 设置固定整数（如 `42`）可保证每次运行得到相同的划分结果，便于复现实验。

        * `shuffle`（布尔值，默认 `True`）：

            - 是否在划分前对数据进行随机打乱。

            - 如果数据本身是时间序列或有序的，通常设为 `False`。

            - 若 `stratify` 有值，则 `shuffle` 必须为 `True`（内部强制）。

        * `stratify`（数组或 `None`，默认 `None`）：

            - 用于**分层抽样**的标签数组。

            - 传入 `y` 后，函数会保证训练集和测试集中各类别（分类问题）的比例与原始数据集一致。

            - 对不平衡数据集非常关键。

    * 返回值

        返回一个**元组（tuple）**，顺序为：

        ```py
        X_train, X_test, y_train, y_test
        ```

        或更一般地：

        ```py
        array1_train, array1_test, array2_train, array2_test, ...
        ```

        每个返回的数组都是原数组的**随机子集**，且训练集和测试集互不重叠。

    * 典型用法（代码示例）

        * 示例 1：基本用法（二分类）

            ```python
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_iris
            from sklearn.svm import SVC

            # 加载数据
            X, y = load_iris(return_X_y=True)

            # 划分（默认 75% 训练，25% 测试）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            print(f"训练集样本数: {X_train.shape[0]}")
            print(f"测试集样本数: {X_test.shape[0]}")

            # 训练并评估
            clf = SVC().fit(X_train, y_train)
            print(f"测试集准确率: {clf.score(X_test, y_test):.3f}")
            ```

        * 示例 2：分层抽样（处理不平衡数据）

            ```python
            from sklearn.datasets import make_classification

            # 生成一个不平衡数据集（类别 0: 900, 类别 1: 100）
            X, y = make_classification(
                n_samples=1000, weights=[0.9, 0.1], random_state=42
            )

            # 普通划分（可能使测试集中类别 1 更少）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print("普通划分 - 训练集类别1比例:", y_train.mean())
            print("普通划分 - 测试集类别1比例:", y_test.mean())

            # 分层划分（保持比例）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print("分层划分 - 训练集类别1比例:", y_train.mean())
            print("分层划分 - 测试集类别1比例:", y_test.mean())
            ```

            输出会显示分层划分后训练/测试集的类别比例几乎一致（约 0.1）。

        * 示例 3：同时划分多个数组（如特征、标签、样本权重）

            ```python
            import numpy as np

            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            sample_weight = np.random.rand(100)

            X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                X, y, sample_weight, test_size=0.2, random_state=7
            )

            print(len(X_train), len(y_train), len(sw_train))  # 均为 80
            ```

        * 示例 4：时间序列数据（不随机打乱）

            ```py
            # 假设数据是按时间排序的
            time_series_X = np.arange(1000).reshape(-1, 1)
            time_series_y = np.arange(1000)

            # 取前 80% 作为训练，后 20% 作为测试
            X_train, X_test, y_train, y_test = train_test_split(
                time_series_X, time_series_y, 
                test_size=0.2, shuffle=False, random_state=None
            )

            print("训练集最后一条:", X_train[-1], "测试集第一条:", X_test[0])
            # 输出: 训练集最后一条: [799] 测试集第一条: [800]
            # 保持顺序不交叉
            ```

    * 注意事项与常见陷阱

        * 数据泄露
        
            不要在划分前做任何基于全数据的预处理（如标准化），应先 `fit` 在训练集上，再 `transform` 测试集

        * 小样本数据
        
            如果样本极少，测试集会过小导致评估方差大，建议使用交叉验证。
            
        * 分类问题的分层
        
            若 `y` 是多分类，`stratify=y` 同样有效，但必须保证每个类别至少有一个样本。
        
        * 回归问题
        
            不能使用 `stratify`（除非手动将连续值离散化），可采用 `sklearn.model_selection.StratifiedShuffleSplit` 或自定义分箱。

        * 随机种子
        
            调参时固定 `random_state`，最终评估时可换多个种子取平均

            注：
            
            1. `train_test_split()` 即使不指定 `random_state`，也会每次都使用不同的随机状态种子

### 内置数据集

* iris

    sklearn 中的 iris 数据集是个 toy 数据集，即不需要联网下载，内置的数据集

    * 简介：

        Classes: 3

        Samples per class: 50

        Samples total: 150

        Dimensionality (num features): 4

        Features: real, positive
    
    * `load_iris()`函数的普通模式，会返回一个类字典的数据结构（Bunch）

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

    * `load_iris()`函数的 `return_X_y` 模式

        直接返回用于训练的 feature 和 label
        
        * feature (X) shape: `(150, 4)`
        
        * label (y) shape: `(150, )`

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

* toy dataset list

    其他的 toy dataset list: <https://scikit-learn.org/stable/datasets/toy_dataset.html#toy-datasets>

    除了 toy dataset 之外的 dataset: <https://scikit-learn.org/stable/datasets.html>

### 常用函数


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

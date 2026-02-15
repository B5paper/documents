# sklearn note

## cache

* 简述 python sklearn 中 svm 的使用方法

    我来简述 sklearn 中 SVM 的使用方法：

    1. 导入必要的库

        ```python
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score
        ```

    2. 数据准备与预处理

        ```python
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 特征缩放（SVM对特征尺度敏感）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        ```

    3. 创建 SVM 模型

        ```python
        # 分类任务
        # 线性核
        model = svm.SVC(kernel='linear', C=1.0)

        # RBF核（默认）
        model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

        # 多项式核
        model = svm.SVC(kernel='poly', degree=3, C=1.0)

        # 回归任务
        model = svm.SVR(kernel='rbf', C=1.0)
        ```

    4. 训练模型

        ```python
        model.fit(X_train, y_train)
        ```

    5. 预测与评估

        ```python
        # 预测
        y_pred = model.predict(X_test)

        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        print(f"准确率: {accuracy}")
        print(classification_report(y_test, y_pred))
        ```

    6. 完整示例

        ```python
        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # 加载数据
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 特征缩放
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 创建并训练SVM
        clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)
        print(f"准确率: {accuracy_score(y_test, y_pred)}")
        ```

    关键参数说明

    - **kernel**: 核函数类型（'linear', 'rbf', 'poly', 'sigmoid'）
    - **C**: 惩罚参数，控制错误样本的惩罚力度
    - **gamma**: 核函数系数，影响单个样本的影响范围
    - **degree**: 多项式核的度数

    SVM 使用前建议进行特征缩放，并根据数据特点选择合适的核函数。

* svm 对向量形式的特征进行分类时，向量的维度设置为多少以下 svm 的效果比较好？

    关于 SVM 对向量维度的要求，没有一个固定的阈值，但可以根据经验和理论给出一些参考：

    维度建议范围

    1. **低维到中维（< 1000 维）**

        - **效果最好**：特别是样本量适中时
        - 线性可分或使用 RBF 核都能表现良好

    2. **中维到高维（1000 - 10^5 维）**

        - **依然有效**：但需要注意：

            - 计算复杂度增加

            - 过拟合风险增加

            - 建议使用**线性核**

    3. **超高维（> 10^5 维）**

        - **挑战较大**：

            - 计算开销大

            - 内存消耗大

            - 可能不如其他算法（如线性模型）

    **维度与样本量的关系**

    更重要是考虑 **维度(d) 与 样本量(n) 的比例**：

    ```python
    # 经验法则
    if n/d > 10:  # 样本充足
        # 可以使用 RBF 核等复杂核函数
        model = svm.SVC(kernel='rbf')
        
    elif n/d < 1:  # 样本少于维度（高维小样本）
        # 建议使用线性核
        model = svm.LinearSVC()  # 或 svm.SVC(kernel='linear')
    ```

    实际建议

    ```python
    from sklearn import svm

    def suggest_svm(X, y):
        n_samples, n_features = X.shape
        
        if n_features < 100:
            # 低维：可以使用复杂核函数
            return svm.SVC(kernel='rbf', C=1.0)
        
        elif n_features < 10000:
            # 中维：根据样本量决定
            if n_samples > n_features * 10:
                return svm.SVC(kernel='rbf', gamma='scale')
            else:
                return svm.LinearSVC()
        
        else:
            # 高维：建议线性核或降维
            from sklearn.decomposition import PCA
            # 先降维再使用 SVM
            pca = PCA(n_components=min(100, n_samples))
            X_reduced = pca.fit_transform(X)
            return svm.SVC(kernel='rbf')
    ```

    特殊情况

    - **文本分类**：即使维度很高（如 TF-IDF 特征 > 10^5），线性 SVM 仍然表现很好
    - **图像识别**：原始像素维度高，通常需要 CNN 或降维后再用 SVM
    - **基因数据**：维度高样本少，使用线性核 + 特征选择

    总结

    没有绝对的"效果比较好"的维度限制，关键在于：

    1. 维度与样本量的比例
    2. 数据的分布特性
    3. 选择的核函数
    4. 计算资源限制

    **最佳实践**：先尝试线性 SVM，如果效果不理想且计算资源允许，再尝试 RBF 核并做特征选择或降维。

## topics

# cuml note

## cache

* cuml

    cuML 是 RAPIDS 项目中的一个小项目，利用 cuda 实现 sklearn 中的算法，加速计算。

    RAPIDS 是 nv 实现的开源数据科学套件。

    RAPIDS 官网：<https://rapids.ai>

    doc: <https://docs.rapids.ai/api/cuml/stable/>

    github repo: <https://github.com/rapidsai/cuml>

    **安装要求**

    ```bash
    # 需要NVIDIA GPU和CUDA环境
    conda create -n rapids python=3.9
    conda install -c rapidsai -c conda-forge cuml
    ```

    **基本使用示例**

    1. 分类算法
    python

    from cuml import LogisticRegression
    from cuml.datasets import make_classification
    import cupy as cp

    # 生成GPU上的数据
    X, y = make_classification(n_samples=10000, n_features=20)

    # 创建并训练模型
    clf = LogisticRegression()
    clf.fit(X, y)

    # 预测
    predictions = clf.predict(X)

    2. 回归算法
    python

    from cuml.ensemble import RandomForestRegressor
    from cuml.metrics import mean_squared_error

    # 训练随机森林
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    # 评估
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    3. 聚类
    python

    from cuml import KMeans

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    labels = kmeans.labels_

    4. 与scikit-learn对比
    python

    from sklearn.ensemble import RandomForestClassifier as skRF
    from cuml.ensemble import RandomForestClassifier as cumlRF
    import time

    # scikit-learn（CPU）
    sk_model = skRF(n_estimators=100)
    start = time.time()
    sk_model.fit(X_cpu, y_cpu)
    print(f"CPU时间: {time.time()-start:.2f}s")

    # cuML（GPU）
    cuml_model = cumlRF(n_estimators=100)
    start = time.time()
    cuml_model.fit(X_gpu, y_gpu)
    print(f"GPU时间: {time.time()-start:.2f}s")

    四、重要特性
    1. GPU DataFrame集成
    python

    import cudf
    from cuml import PCA

    # 使用cuDF加载数据
    df = cudf.read_csv('large_dataset.csv')
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(df[features])

    2. 多GPU支持
    python

    from cuml.dask.cluster import KMeans
    from dask_cuda import LocalCUDACluster

    # 创建多GPU集群
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # 分布式K-Means
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(distributed_data)

    3. 模型持久化
    python

    import pickle

    # 保存模型
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # 加载模型
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    五、使用场景

        推荐系统：处理用户-物品矩阵

        金融风控：实时欺诈检测

        基因组学：大规模生物信息分析

        计算机视觉：特征降维和聚类

    六、注意事项

        数据量：小数据集可能看不出优势

        GPU内存：数据需要能放入GPU显存

        算法覆盖：比scikit-learn算法少，但核心算法都有

    七、性能对比示例
    python

    # 当数据量达到百万级时加速效果显著
    # CPU sklearn: ~120秒
    # GPU cuml: ~3秒 (40倍加速)

    总结：cuml是处理大规模机器学习任务的利器，特别适合需要快速迭代或实时推理的场景，API设计与scikit-learn高度一致，迁移成本低。

* cuml 零代码更改加速

    启用方式通常只需在导入其他库前添加几行代码：

    ```
    # 在Jupyter Notebook中
    %load_ext cuml.accel
    import sklearn
    # ... 你的原有代码

    # 或在Python脚本中
    import cuml.accel
    cuml.accel.install()
    import sklearn
    # ... 你的原有代码
    ```

## note

# Scipy Note

## cache

* 详细讲讲 scipy.stats 模块

	`scipy.stats` 是 SciPy 生态系统中专门用于**统计学**的核心模块。它包含了连续和离散概率分布、描述性统计、假设检验、非参数统计、生存分析等极其丰富的工具。

	对于数据分析师、数据科学家和科研人员来说，这个模块几乎是必不可少的。下面为你系统、详细地拆解 `scipy.stats` 的核心功能与用法。

	---

	## 1. 核心基石：概率分布（Distributions）

	`scipy.stats` 包含了上百种概率分布，它们被统一划分为两大类：

	* **连续型分布**（继承自 `rv_continuous`）：如正态分布 (`norm`)、t 分布 (`t`)、指数分布 (`expon`)。
	* **离散型分布**（继承自 `rv_discrete`）：如二项分布 (`binom`)、泊松分布 (`poisson`)。

	### 核心方法（所有分布通用）

	无论是哪种分布，SciPy 都设计了统一的方法接口，这让它的学习曲线非常平缓：

	| 方法名称 | 全称 / 含义 | 作用 |
	| --- | --- | --- |
	| `pdf(x)` | Probability Density Function | **概率密度函数**（仅限连续型），返回 $x$ 处的密度值。 |
	| `pmf(k)` | Probability Mass Function | **概率质量函数**（仅限离散型），返回离散值 $k$ 的具体概率。 |
	| `cdf(x)` | Cumulative Distribution Function | **累积分布函数**，返回 $P(X \le x)$，即左侧面积。 |
	| `sf(x)` | Survival Function | **生存函数**，等于 $1 - \text{cdf}(x)$，即右侧面积（常用于 $p\text{-value}$ 计算）。 |
	| `ppf(q)` | Percent Point Function | **百分点函数**（CDF 的逆函数），已知概率 $q$，求对应的分位数 $x$。 |
	| `rvs(size)` | Random Variates | **生成随机数**，根据该分布的特征抽取样本。 |
	| `stats()` | Statistics | 返回分布的均值、方差、偏度（Skewness）和峰度（Kurtosis）。 |

	### 代码实战：正态分布的标准操作

	```python
	import numpy as np
	from scipy import stats

	# 定义一个正态分布，均值(loc)=10，标准差(scale)=2
	my_norm = stats.norm(loc=10, scale=2)

	# 1. 计算 x=10 处的概率密度 (PDF)
	print("PDF at 10:", my_norm.pdf(10))  # 输出最高点密度

	# 2. 计算 x <= 12 的累积概率 (CDF)
	print("CDF at 12:", my_norm.cdf(12))  # 约 0.8413 (即均值加一个标准差的左侧面积)

	# 3. 已知左侧面积为 0.95，求对应的 x 值 (PPF)
	print("95th percentile:", my_norm.ppf(0.95))

	# 4. 生成 5 个符合该分布的随机数 (RVS)
	print("Random samples:", my_norm.rvs(size=5))

	```

	---

	## 2. 统计推断与假设检验（Hypothesis Testing）

	这是科研和 A/B 测试中最常用的功能。`scipy.stats` 提供了几乎所有主流的假设检验方法。**注意：SciPy 的检验函数默认返回一个包含 `(statistic, pvalue)` 的命名元组。**

	### A. T 检验（T-test）—— 均值比较

	* **单样本 T 检验 (`ttest_1samp`)**：检验样本均值是否与某个已知总体均值有显著差异。
	* **独立双样本 T 检验 (`ttest_ind`)**：比较两组独立样本的均值（如 A/B 测试的两组用户）。
	* **配对样本 T 检验 (`ttest_rel`)**：比较同一组对象在前后的变化（如服药前后的血压）。

	```python
	# 独立双样本 T 检验示例
	group_A = [12, 15, 14, 11, 13, 17, 12]
	group_B = [18, 20, 19, 16, 22, 19, 21]

	stat, p_val = stats.ttest_ind(group_A, group_B)
	print(f"T-statistic: {stat:.4f}, p-value: {p_val:.4f}")
	# 如果 p-value < 0.05，则拒绝原假设，认为两组有显著差异。

	```

	### B. 方差分析（ANOVA）—— 多组均值比较

	* **单因素方差分析 (`f_oneway`)**：检验三组或更多组数据的均值是否存在显著差异。

	### C. 卡方检验（Chi-Square Test）—— 分类变量分类

	* **卡方独立性检验 (`chi2_contingency`)**：分析两个分类变量之间是否有相关性（如：性别与购买意愿是否相关）。

	### D. 正态性检验（Normality Tests）

	很多参数检验（如 T 检验）要求数据服从正态分布。

	* `shapiro` (Shapiro-Wilk 检验，小样本首选)
	* `normaltest` (D'Agostino and Pearson 检验)

	---

	## 3. 描述性统计（Descriptive Statistics）

	虽然 Pandas 的 `.describe()` 很好用，但 `scipy.stats` 提供了更深入的统计量计算：

	* `stats.describe(data)`：一次性返回样本大小、极值、均值、方差、偏度和峰度。
	* `stats.gmean(data)` / `stats.hmean(data)`：计算**几何均值**和**调和均值**。
	* `stats.iqr(data)`：计算**四分位距（IQR）**，常用于检测异常值。
	* `stats.zscore(data)`：计算 **Z-score**（将数据标准化为均值为 0、方差为 1 的分布）。

	---

	## 4. 相关性与回归（Correlation）

	用于研究变量之间的关系：

	* **皮尔逊相关系数 (`pearsonr`)**：衡量连续变量之间的**线性**相关性（要求正态分布）。
	* **斯皮尔曼等级相关系数 (`spearmanr`)**：非参数相关系数，衡量**单调**关系（对异常值更鲁棒）。
	* **线性回归 (`linregress`)**：快速拟合一条一元线性回归方程 $y = kx + b$，并返回斜率、截距、$R^2$、p 值等。

	```python
	x = [1, 2, 3, 4, 5]
	y = [2, 4, 5, 4, 5]
	res = stats.linregress(x, y)
	print(f"斜率: {res.slope}, 截距: {res.intercept}, R²: {res.rvalue**2:.4f}")

	```

	---

	## 5. 高级功能：数据转换与拟合

	### A. 分布拟合 (`fit`)

	如果你有一组未知规律的数据，想知道它最符合什么分布，可以使用 `fit()` 方法来**估计分布的参数**。

	```python
	data = stats.norm.rvs(loc=5, scale=1.5, size=1000) # 生成模拟数据
	# 假设我们猜它是正态分布，用 norm.fit 逆向求解参数
	le_loc, le_scale = stats.norm.fit(data)
	print(f"拟合出的均值: {le_loc:.2f}, 标准差: {le_scale:.2f}")

	```

	### B. Box-Cox 转换 (`boxcox`)

	在回归分析中，如果数据严重偏态（非正态），可以使用 Box-Cox 转换将数据**正态化**，以满足线性模型的假设。

	```python
	# 生成一个严重右偏的指数分布数据
	skewed_data = stats.expon.rvs(size=500)
	# 转换成接近正态分布的数据
	transformed_data, best_lam = stats.boxcox(skewed_data)

	```

	---

	## 总结：如何高效使用 `scipy.stats`？

	1. **查阅具体分布**：当你需要某种分布时，直接去查它的官方缩写（如 `gamma`, `beta`, `poisson`），然后直接套用 `pdf/pmf/cdf/rvs` 即可。
	2. **注重 p 值判断**：在假设检验中，几乎所有的函数输出都包含 `pvalue`。通常约定 $p < 0.05$ 时拒绝原假设（即结果具有统计学显著性）。
	3. **配合大生态**：通常由 `numpy` 提供或生成数据，用 `scipy.stats` 做核心数学计算，再用 `matplotlib`/`seaborn` 进行可视化，用 `pandas` 进行表格化管理。

	你目前是在处理具体的科研数据，还是在准备 A/B 测试相关的业务分析？如果有具体的应用场景，我可以为你提供针对性的代码模版。

* 皮尔逊相关系数（Pearson correlation coefficient）

    `scipy.stats.pearsonr` 是最标准、最常用的方法，不仅返回相关系数，还返回 **p-value**（检验相关系数是否显著不为 0）。

    ```python
    from scipy.stats import pearsonr

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    r, p_value = pearsonr(x, y)
    print(f"皮尔逊相关系数: {r:.4f}")   # 输出: 1.0000
    print(f"p-value: {p_value:.4e}")    # 输出: 0.0000e+00
    ```

    **注意**：

    - 输入可以是列表、元组或 NumPy 数组
    - 如果数据中有 `NaN`，会返回 `NaN`
    - p-value 是双尾检验，原假设是相关系数为 0

* scipy 中的稀疏矩阵的运算有类似 numpy 那样的多线程或 smid 加速机制吗

    这是一个非常扎实的问题。简单直接的回答是：**Scipy 的稀疏矩阵（`scipy.sparse`）在原生状态下对多线程和 SIMD 的支持非常有限，远不如 Numpy 那样成熟。**

    底层原因在于稀疏矩阵的存储特性（非连续内存）与并行计算需要的“数据规整性”天然冲突。以下是详细的拆解：

    ---

    1. 为什么它不像 Numpy 那样快？

        Numpy 的核心运算（如矩阵乘法）通常链接了 **BLAS/LAPACK** 库（如 OpenBLAS, Intel MKL, 或 Apple Accelerate）。这些库是高度优化的，会自动利用多核 CPU 和 SIMD 指令集。

        相比之下，Scipy 的稀疏矩阵：
        * **多线程：** 大多数稀疏运算（如 `dot` 或 `spsolve`）运行在底层的 C++ 或 Fortran 代码中，但受到 **GIL (Global Interpreter Lock)** 的限制，或者其底层实现本身就是单线程的递归/循环。
        * **SIMD：** 由于稀疏矩阵存储的是（值, 行索引, 列索引），内存访问是**非连续的（Strided/Random access）**。这使得 CPU 很难预取数据，也难以应用典型的 SIMD 向量化指令。

    ---

    2. 不同格式的“加速”现状

        虽然原生支持弱，但不同的存储格式在效率上有天壤之别：

        | 格式 | 优势 | 局限性 |
        | :--- | :--- | :--- |
        | **CSR / CSC** | **最接近硬件加速。** 矩阵乘法（SPMM）和向量乘法（SPMV）在 Scipy 中有部分优化，但基本是单核逻辑。 | 结构固定，难以动态修改。 |
        | **COO** | 适合构建矩阵。 | 几乎没有任何计算加速，计算前通常需转为 CSR。 |
        | **LIL / DOK** | 方便像字典一样读写。 | 纯 Python 逻辑，极其缓慢，不支持任何底层加速。 |

    ---

    3. 如果你需要加速，该怎么办？

        如果你发现 Scipy 的性能遇到了瓶颈，目前业界标准的替代方案如下：

        A. 切换到 Intel MKL 运行时

        如果你使用的是 Intel CPU，通过 Conda 安装 `mkl` 和 `mkl-service`。在某些版本的 Scipy 中，针对 CSR 格式的运算会调用 MKL 的稀疏子程序（MKL Sparse BLAS），这可以显著提升 **SIMD** 的利用率。

        B. 使用 PyTorch 或 TensorFlow (GPU 加速)

        这是目前最主流的方案。如果你有大量稀疏运算：
        * `torch.sparse` 支持 GPU 上的稀疏矩阵运算。
        * GPU 拥有成千上万个核心，处理 CSR 格式的并行化能力远超 CPU。

        C. Cupy (Numpy/Scipy 的 GPU 替代品)

        如果你不想改变代码风格，**CuPy** 是最佳选择。它几乎完整克隆了 `scipy.sparse` 的 API，但运行在 CUDA 上。
        ```python
        import cupyx.scipy.sparse as sparse # 几乎与 scipy 语法一致
        ```

        D. 使用专用库：PyData/Sparse 或 PyKeOps

        * **PyData/Sparse:** 提供了更现代的稀疏张量支持，且能更好地与 Dask 集成进行分布式计算。
        * **PyKeOps:** 针对特定的大规模核运算，绕过了传统的矩阵存储，利用在线映射实现极速计算。

    ---

    **总结**

    Scipy 稀疏矩阵目前主要服务于**中等规模**和**算法验证**。如果你追求极致的多线程并发现 CPU 占用率始终只有 100%（单核），那么你应该考虑将计算负载转移到 **GPU (CuPy/PyTorch)** 或者检查是否正确链接了 **MKL** 库。

    你目前的计算场景是在处理大规模的图算法，还是在求解大型线性方程组？

* scipy 中的稀疏矩阵如何转换成 cupy 矩阵

    将 SciPy 的稀疏矩阵转换为 CuPy 矩阵是加速大规模数值计算的常见需求。由于 SciPy 运行在 CPU 上，而 CuPy 运行在 NVIDIA GPU 上，这个过程涉及**内存到显存的传输**。

    最直接的方法是使用 CuPy 提供的 **`cupyx.scipy.sparse`** 模块。

    1. 基础转换方法

        CuPy 提供了与 SciPy 几乎一一对应的稀疏矩阵格式（如 CSR, CSC, COO）。你可以直接将 SciPy 对象传递给 CuPy 的构造函数。

        ```python
        import scipy.sparse as sp
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

        # 1. 创建一个 SciPy 稀疏矩阵 (CPU)
        scipy_sparse_matrix = sp.csr_matrix([[0, 1, 0], [2, 0, 3]])

        # 2. 转换为 CuPy 稀疏矩阵 (GPU)
        cupy_sparse_matrix = cp_csr_matrix(scipy_sparse_matrix)

        print(type(cupy_sparse_matrix)) 
        # 输出: <class 'cupyx.scipy.sparse.csr.csr_matrix'>
        ```

    ---

    2. 支持的格式对照表

        为了保持计算效率，建议转换前后的格式保持一致：

        | SciPy 格式 | CuPy 对应类 | 适用场景 |
        | :--- | :--- | :--- |
        | `sp.csr_matrix` | `cupyx.scipy.sparse.csr_matrix` | 算术运算、矩阵向量乘法 |
        | `sp.csc_matrix` | `cupyx.scipy.sparse.csc_matrix` | 快速列切片 |
        | `sp.coo_matrix` | `cupyx.scipy.sparse.coo_matrix` | 快速构建矩阵 |

    ---

    3. 高级用法：自动转换

        如果你不确定输入的格式，可以使用 `cupy.asarray()`。虽然它主要用于稠密数组，但对于某些稀疏操作，CuPy 的接口设计非常灵活：

        ```python
        import cupyx.scipy.sparse as cps

        # 强制转换并移动到 GPU
        device_matrix = cps.array(scipy_sparse_matrix)
        ```

    ---

    4. 性能优化建议

        * **避免频繁传输**：将数据移动到 GPU（从 SciPy 到 CuPy）是一个耗时操作。理想的做法是“一次搬运，多次计算”。
        * **内存对齐**：如果 SciPy 矩阵的索引类型是 `int64`，而在 GPU 上 `int32` 就足够了，可以在转换时指定 `dtype` 以节省显存。
        * **反向转换**：如果需要将结果转回 CPU，使用 `.get()` 方法：
            ```python
            cpu_sparse = cupy_sparse_matrix.get()
            ```

    5. 注意事项

        > [!CAUTION]
        > **库依赖**：确保你安装了 `cupy` 且版本与你的 CUDA 环境匹配。CuPy 的稀疏矩阵 API 位于 `cupyx` 命名空间下，而不是核心的 `cupy` 命名空间。

    你目前处理的稀疏矩阵规模大约有多大？针对极大规模的稀疏矩阵，不同的存储格式（CSR vs CSC）对计算性能的影响会非常显著。

* 简述 scipy.spatial.KDTree 的用法

    `scipy.spatial.KDTree` 是一种用于快速处理**多维空间数据**的数据结构（例如实现最近邻查找、范围查找）。它通过递归分割空间来组织数据点，将查询的时间复杂度从 O(N) 降低到 O(log N)，非常适合处理大量点集。

    以下是其核心用法和常见操作简介：

    ### 1. 创建 KDTree
    首先需要导入模块并准备数据。数据应为形状为 `(n_samples, n_features)` 的二维数组。

    ```python
    import numpy as np
    from scipy.spatial import KDTree

    # 准备一些样本点
    points = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

    # 创建 KDTree 实例
    tree = KDTree(points)
    ```

    ### 2. 常用查询方法

    #### **`query`：查找最近邻**
    用于查找距离目标点最近的 **k** 个点。

    ```python
    # 查询点
    target_point = [4, 5]

    # 查找最近的2个点
    distances, indices = tree.query(target_point, k=2)

    print("距离:", distances)  # 到最近点的距离
    print("索引:", indices)    # 在原始数组中的位置
    ```

    #### **`query_ball_point`：查找半径内的所有点**
    用于查找指定点周围半径 r 内的所有点。

    ```python
    # 查找以 [4,5] 为圆心，半径为 3 范围内的所有点
    indices = tree.query_ball_point([4, 5], r=3)

    print("半径内的点索引:", indices)
    ```

    #### **`query_pairs`：查找距离小于阈值的点对**
    用于查找树内部所有点之间距离小于给定阈值的点对，常用于碰撞检测或聚类。

    ```python
    # 查找所有距离小于 3 的点对
    pairs = tree.query_pairs(r=3)

    print("点对:", pairs)  # 返回的是集合，元素为 (i, j) 元组
    ```

    #### **`query_ball_tree`：查找两棵树之间的近邻**
    用于查找本棵树中所有与另一棵树中点距离小于 r 的点对。

    ```python
    # 假设有另一棵树 other_tree
    # pairs = tree.query_ball_tree(other_tree, r=2.0)
    ```

    ### 3. 完整示例
    下面是一个包含数据准备、查询和结果打印的完整示例：

    ```python
    import numpy as np
    from scipy.spatial import KDTree

    # 生成随机数据
    np.random.seed(42)
    data = np.random.randn(100, 2)  # 100个二维点

    # 构建树
    tree = KDTree(data)

    # 查询点
    query_point = [0, 0]

    # 1. 查找最近的3个点
    dist, idx = tree.query(query_point, k=3)
    print("最近的3个点索引:", idx)
    print("对应距离:", dist)

    # 2. 查找半径 1.5 内的所有点
    r_idx = tree.query_ball_point(query_point, r=1.5)
    print(f"半径1.5内的点数量: {len(r_idx)}")
    ```

    ### 注意事项
    - **维度灾难**：当数据维度非常高（例如超过 20 维）时，KD-Tree 的性能可能会下降，此时可以考虑使用 `scipy.spatial.cKDTree`（实现相同接口但速度更快）或其他近似方法。
    - **数据规模**：在小数据集（例如少于 100 个点）上，暴力搜索（直接计算所有距离）可能比 KDTree 更快，因为构建树本身也有开销。
    - **cKDTree**：SciPy 还提供了一个用 C 语言编写的 `cKDTree`，用法与 `KDTree` 完全相同，但速度更快。在日常使用中，可以直接导入 `from scipy.spatial import cKDTree` 来获得更好的性能。


# sparse matrix

## cache

* 在 py 中构建与稀疏矩阵等价的稠密矩阵，比较稀疏乘法和稠密乘法的时间

    这是一个非常经典的实验。我们可以通过控制矩阵的**规模（Size）**和**稀疏度（Density）**来观察性能的“交叉点”。

    在运行这段代码时，请注意：**Numpy** 通常会调用 MKL 或 OpenBLAS，它们会吃满你所有的 CPU 核心（多线程）；而 **Scipy** 的稀疏乘法通常是单线程运行的。

    ```python
    import numpy as np
    import time
    from scipy import sparse

    def benchmark_sparse_vs_dense(size=2000, density=0.01):
        """
        比较在特定规模和稀疏度下，稀疏矩阵和稠密矩阵乘法的耗时
        """
        print(f"--- 矩阵规模: {size}x{size}, 稀疏度: {density*100}% ---")

        # 1. 生成随机稀疏矩阵 (CSR 格式是计算最快的格式)
        sparse_A = sparse.random(size, size, density=density, format='csr')
        sparse_B = sparse.random(size, size, density=density, format='csr')

        # 2. 转换为等价的稠密矩阵
        dense_A = sparse_A.toarray()
        dense_B = sparse_B.toarray()

        # --- 测试稠密矩阵乘法 (Numpy + BLAS 多线程) ---
        start_dense = time.time()
        # Numpy 的 dot 或 @ 会调用底层优化库 (MKL/OpenBLAS)
        dense_res = np.dot(dense_A, dense_B)
        end_dense = time.time()
        dense_time = end_dense - start_dense
        print(f"Numpy 稠密乘法耗时: {dense_time:.4f} 秒")

        # --- 测试稀疏矩阵乘法 (Scipy CSR) ---
        start_sparse = time.time()
        # Scipy 的稀疏乘法
        sparse_res = sparse_A.dot(sparse_B)
        end_sparse = time.time()
        sparse_time = end_sparse - start_sparse
        print(f"Scipy 稀疏乘法耗时: {sparse_time:.4f} 秒")

        # 计算加速比
        sparse_time = max(sparse_time, 1e-6)  # 防止除数为 0
        dense_time = max(dense_time, 1e-6)
        if sparse_time < dense_time:
            print(f"结论: 【稀疏】更快，领先 {dense_time/sparse_time:.2f} 倍")
        else:
            print(f"结论: 【稠密】更快，领先 {sparse_time/dense_time:.2f} 倍")
        print("\n")

    # --- 实验 1: 固定稀疏度 (1%)，改变规模 ---
    # 观察随着矩阵变大，稀疏矩阵优势是否扩大
    for s in [500, 2000, 5000]:
        benchmark_sparse_vs_dense(size=s, density=0.01)

    # --- 实验 2: 固定规模 (2000)，改变稀疏度 ---
    # 观察稀疏度增加到多少时，稠密算法的 SIMD/多线程优势会反超
    for d in [0.001, 0.05, 0.1]:
        benchmark_sparse_vs_dense(size=2000, density=d)
    ```

    output:

    ```
    --- 矩阵规模: 500x500, 稀疏度: 1.0% ---
    Numpy 稠密乘法耗时: 0.0000 秒
    Scipy 稀疏乘法耗时: 0.0000 秒
    结论: 【稠密】更快，领先 1.00 倍


    --- 矩阵规模: 2000x2000, 稀疏度: 1.0% ---
    Numpy 稠密乘法耗时: 0.0391 秒
    Scipy 稀疏乘法耗时: 0.0000 秒
    结论: 【稀疏】更快，领先 39106.61 倍


    --- 矩阵规模: 5000x5000, 稀疏度: 1.0% ---
    Numpy 稠密乘法耗时: 0.6427 秒
    Scipy 稀疏乘法耗时: 0.0878 秒
    结论: 【稀疏】更快，领先 7.32 倍


    --- 矩阵规模: 2000x2000, 稀疏度: 0.1% ---
    Numpy 稠密乘法耗时: 0.0450 秒
    Scipy 稀疏乘法耗时: 0.0000 秒
    结论: 【稀疏】更快，领先 44976.47 倍


    --- 矩阵规模: 2000x2000, 稀疏度: 5.0% ---
    Numpy 稠密乘法耗时: 0.0418 秒
    Scipy 稀疏乘法耗时: 0.0656 秒
    结论: 【稠密】更快，领先 1.57 倍


    --- 矩阵规模: 2000x2000, 稀疏度: 10.0% ---
    Numpy 稠密乘法耗时: 0.0419 秒
    Scipy 稀疏乘法耗时: 0.1159 秒
    结论: 【稠密】更快，领先 2.77 倍
    ```

    ---

    **实验现象背后的逻辑**

    1.  **为什么稀疏矩阵在极度稀疏时胜出？**
        * **计算量差异：** 稠密矩阵乘法的复杂度是 $O(n^3)$。对于 $5000 \times 5000$ 的矩阵，计算量是 1250 亿次浮点运算。
        * **跳过零点：** 稀疏算法只计算非零元素。如果稀疏度是 $1\%$，计算量大约减少了几个数量级，抵消了多线程带来的红利。

    2.  **为什么稠密矩阵在稠密时（如 10% 稀疏度）反超？**
        * **缓存友好性：** Numpy 使用的 BLAS 库会对内存进行分块（Tiling），极大地利用了 CPU L1/L2 缓存。
        * **并行与 SIMD：** 稠密乘法可以完美利用 CPU 的 AVX-512 等指令集，一个指令周期处理多个数据；而稀疏格式需要频繁读取索引表（间接寻址），导致 CPU 流水线频繁停顿（Cache Miss）。

    3.  **内存临界点：**
        > **注意：** 当 `size` 达到 10000 以上时，稠密矩阵会占用极大的内存（$10000^2 \times 8$ 字节 $\approx 800$ MB）。如果你的内存不足导致触发 Swap（虚拟内存），稠密矩阵的速度会断崖式下跌，而稀疏矩阵依然能轻盈运行。

    你打算处理的数据集大概是什么级别的稀疏度？（例如社交网络通常 $< 0.1\%$，而有限元分析可能在 $1\% \sim 5\%$ 之间）

* Python稀疏矩阵创建与乘法计算

    在 Python 中，创建稀疏矩阵和计算矩阵乘法通常使用 `scipy.sparse` 模块。以下是简要说明：

    一、创建稀疏矩阵

    1. **主要存储格式**

        ```python
        import numpy as np
        from scipy import sparse

        # COO格式（坐标格式） - 创建时最常用
        data = [1, 2, 3, 4]
        row = [0, 1, 2, 3]
        col = [1, 2, 3, 0]
        coo_matrix = sparse.coo_matrix((data, (row, col)), shape=(4, 4))

        # CSR格式（压缩稀疏行） - 计算时最常用
        csr_matrix = sparse.csr_matrix((data, (row, col)), shape=(4, 4))

        # CSC格式（压缩稀疏列）
        csc_matrix = sparse.csc_matrix((data, (row, col)), shape=(4, 4))

        # 从密集矩阵转换
        dense_matrix = np.array([[0, 1, 0], [0, 0, 2], [3, 0, 0]])
        sparse_matrix = sparse.csr_matrix(dense_matrix)

        # 特殊矩阵创建
        # 单位矩阵
        eye_matrix = sparse.eye(5)  # 5x5单位矩阵
        # 对角线矩阵
        diag_matrix = sparse.diags([1, 2, 3], offsets=[0, -1, 2], shape=(4, 4))
        ```

    2. **从文件加载**

        ```python
        # 使用 mmread 读取 Matrix Market 格式
        from scipy.io import mmread
        matrix = mmread('matrix.mtx')
        ```

    二、计算矩阵乘法

    1. **稀疏矩阵 × 稀疏矩阵**

        ```python
        import scipy.sparse as sp

        # 创建两个稀疏矩阵
        A = sp.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        B = sp.csr_matrix([[0, 1], [2, 0], [0, 3]])

        # 矩阵乘法（自动选择高效算法）
        C = A.dot(B)  # 或使用 A @ B
        ```

    2. **稀疏矩阵 × 密集矩阵/向量**

        ```python
        # 稀疏矩阵 × 密集矩阵
        sparse_mat = sp.csr_matrix([[0, 1], [2, 0]])
        dense_mat = np.array([[1, 2], [3, 4]])
        result = sparse_mat.dot(dense_mat)

        # 稀疏矩阵 × 向量
        vector = np.array([1, 2])
        result_vec = sparse_mat.dot(vector)
        ```

    3. **高效计算技巧**

        ```python
        # 1. 确保使用正确的格式进行计算
        # CSR × CSR 或 CSC × CSC 效率最高
        A_csr = A.tocsr() if not A.format == 'csr' else A
        B_csr = B.tocsr() if not B.format == 'csr' else B

        # 2. 批量乘法
        result = A_csr @ B_csr

        # 3. 元素级乘法（不是矩阵乘法）
        elementwise = A.multiply(B)  # 对应元素相乘
        ```

    三、实用示例

    ```python
    import scipy.sparse as sp
    import numpy as np

    # 创建大型稀疏矩阵（1000x1000，稀疏度0.1%）
    n = 1000
    density = 0.001
    nnz = int(n * n * density)

    rows = np.random.randint(0, n, nnz)
    cols = np.random.randint(0, n, nnz)
    data = np.random.randn(nnz)

    # 创建COO格式并转换为CSR用于计算
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    B = sp.random(n, n, density=density, format='csr')

    # 计算乘法
    C = A @ B  # 自动使用稀疏矩阵优化算法

    print(f"A形状: {A.shape}, 非零元素: {A.nnz}")
    print(f"结果形状: {C.shape}")
    ```

    四、注意事项

    1. **格式选择**：

       - COO：容易创建，但不适合算术运算
       - CSR：适合行操作和矩阵乘法
       - CSC：适合列操作和矩阵乘法

    2. **性能优化**：

       - 在计算前将矩阵转换为 CSR 或 CSC 格式
       - 避免频繁的格式转换
       - 对于极大矩阵，考虑使用分块计算

    3. **内存管理**：

        - 使用 `eliminate_zeros()` 删除显式零
        - 使用 `sum_duplicates()` 合并重复项

    这种方法可以高效处理大型稀疏矩阵，显著节省内存和计算时间。

## topics

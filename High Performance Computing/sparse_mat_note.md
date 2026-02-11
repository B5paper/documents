# sparse matrix

## cache

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

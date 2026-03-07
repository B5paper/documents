# Numpy Note

## cache

* NumPy 中的随机数生成器（Random Number Generator）

    这是 NumPy 中的随机数生成器（Random Number Generator）的**新式用法**，是 NumPy 1.17 版本后推荐的方式：

    ## 1. 基本用法

    ```python
    import numpy as np

    # 创建随机数生成器，42是随机种子
    rng = np.random.default_rng(42)

    # 生成随机数
    print(rng.random())                    # 0.7739560485559633（标量）
    print(rng.random(3))                    # [0.43887844 0.85859792 0.69736803]（数组）
    print(rng.random((2, 3)))               # 2x3的随机数组
    ```

    ## 2. 与旧版对比

    ```python
    # 旧版（不推荐）
    np.random.seed(42)
    a = np.random.rand(3)
    b = np.random.randint(0, 10, 3)

    # 新版（推荐）
    rng = np.random.default_rng(42)
    a = rng.random(3)
    b = rng.integers(0, 10, 3)
    ```

    ## 3. 常用方法

    ```python
    rng = np.random.default_rng(42)

    # 整数随机数
    print(rng.integers(0, 10, 5))           # [6 3 7 4 6] 0-9之间的5个整数
    print(rng.integers(0, 10, (2, 3)))       # 2x3的整数数组

    # 浮点数随机数
    print(rng.random(5))                     # [0.77 0.43 0.85 0.59 0.69]
    print(rng.uniform(0, 1, 5))              # 均匀分布，等同于random()

    # 正态分布
    print(rng.normal(0, 1, 3))               # 均值0，标准差1的正态分布

    # 随机选择
    arr = np.array([1, 2, 3, 4, 5])
    print(rng.choice(arr, 3))                 # 随机选择3个元素
    print(rng.choice(arr, 3, replace=False))  # 无放回选择

    # 打乱数组
    rng.shuffle(arr)                          # 原地打乱
    print(arr)

    # 随机排列
    print(rng.permutation(10))                 # 0-9的随机排列
    ```

    ## 4. 多维数组操作

    ```python
    rng = np.random.default_rng(42)

    # 生成多维随机数组
    matrix = rng.random((3, 4))                # 3行4列
    print(matrix)

    # 沿指定轴打乱
    matrix2d = np.array([[1, 2], [3, 4], [5, 6]])
    rng.shuffle(matrix2d, axis=0)              # 打乱行顺序
    print(matrix2d)
    ```

    ## 5. 不同的分布

    ```python
    rng = np.random.default_rng(42)

    # 各种概率分布
    print(rng.normal(0, 1, 5))                 # 正态分布
    print(rng.poisson(3, 5))                   # 泊松分布
    print(rng.exponential(1, 5))                # 指数分布
    print(rng.binomial(10, 0.5, 5))             # 二项分布
    print(rng.gamma(2, 2, 5))                   # Gamma分布
    ```

    ## 6. 并行处理优势

    ```python
    # 可以创建独立的生成器，避免线程安全问题
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)

    # 各自独立
    print(rng1.random(3))  # [0.77 0.43 0.85]
    print(rng2.random(3))  # [0.95 0.15 0.10]
    ```

    ## 7. 实用示例

    ```python
    rng = np.random.default_rng(42)

    # 生成随机数据集
    def generate_dataset(n_samples=100, n_features=5):
        X = rng.random((n_samples, n_features))
        y = rng.integers(0, 2, n_samples)  # 二分类标签
        return X, y

    X, y = generate_dataset()
    print(f"X shape: {X.shape}")  # (100, 5)
    print(f"y shape: {y.shape}")  # (100,)

    # 随机采样数据
    data = rng.random(1000)
    sample = rng.choice(data, size=100, replace=False)

    # 蒙特卡洛模拟
    def monte_carlo_pi(n_points=10000):
        points = rng.random((n_points, 2))
        inside = np.sum(np.sum(points**2, axis=1) <= 1)
        return 4 * inside / n_points

    print(f"π ≈ {monte_carlo_pi()}")
    ```

    ## 优势总结

    1. **面向对象**：可以创建多个独立的生成器
    2. **线程安全**：不同生成器互不影响
    3. **性能更好**：特别是对于大规模随机数生成
    4. **一致性**：不同版本NumPy结果更稳定
    5. **灵活性**：支持更多的随机分布和操作

    这是目前NumPy推荐的随机数生成方式！

* 简述 numpy 中拼接 array 的常用方法，并给出 example

    在 NumPy 中，拼接 array 的常用方法主要有以下几种，我会为每个方法提供简单的示例。

    1. np.concatenate() - 最通用的拼接方法

        可以沿现有维度拼接多个数组。

        ```python
        import numpy as np

        # 创建示例数组
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        # 沿第一个轴拼接（垂直方向）
        c = np.concatenate((a, b), axis=0)
        print("沿axis=0拼接:\n", c)
        # 输出:
        # [[1 2]
        #  [3 4]
        #  [5 6]
        #  [7 8]]

        # 沿第二个轴拼接（水平方向）
        d = np.concatenate((a, b), axis=1)
        print("\n沿axis=1拼接:\n", d)
        # 输出:
        # [[1 2 5 6]
        #  [3 4 7 8]]
        ```

    2. np.vstack() - 垂直堆叠

        相当于 `concatenate(axis=0)`，将数组按垂直方向堆叠。

        ```python
        # 垂直堆叠
        e = np.vstack((a, b))
        print("垂直堆叠:\n", e)
        # 输出:
        # [[1 2]
        #  [3 4]
        #  [5 6]
        #  [7 8]]

        # 一维数组的垂直堆叠
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.vstack((x, y))
        print("\n一维数组垂直堆叠:\n", z)
        # 输出:
        # [[1 2 3]
        #  [4 5 6]]
        ```

    3. np.hstack() - 水平堆叠

        相当于 `concatenate(axis=1)`，将数组按水平方向堆叠。

        ```python
        # 水平堆叠
        f = np.hstack((a, b))
        print("水平堆叠:\n", f)
        # 输出:
        # [[1 2 5 6]
        #  [3 4 7 8]]

        # 一维数组的水平堆叠
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.hstack((x, y))
        print("\n一维数组水平堆叠:\n", z)
        # 输出: [1 2 3 4 5 6]
        ```

    4. np.dstack() - 深度堆叠

        沿第三个轴（深度方向）堆叠数组。

        ```python
        # 深度堆叠
        g = np.dstack((a, b))
        print("深度堆叠:\n", g)
        print("形状:", g.shape)
        # 输出:
        # [[[1 5]
        #   [2 6]]
        # 
        #  [[3 7]
        #   [4 8]]]
        # 形状: (2, 2, 2)
        ```

    5. np.stack() - 沿新轴堆叠

        沿新的维度堆叠数组，会增加数组的维度。

        ```python
        # 沿新轴堆叠
        h = np.stack((a, b), axis=0)
        print("沿新axis=0堆叠:\n", h)
        print("形状:", h.shape)
        # 输出形状: (2, 2, 2)

        i = np.stack((a, b), axis=2)
        print("\n沿新axis=2堆叠:\n", i)
        print("形状:", i.shape)
        # 输出形状: (2, 2, 2)
        ```

    6. np.append() - 追加元素

        类似于Python列表的append，但效率较低，不推荐循环中使用。

        ```python
        # 追加到一维数组
        j = np.array([1, 2, 3])
        k = np.append(j, [4, 5, 6])
        print("append到一维数组:\n", k)
        # 输出: [1 2 3 4 5 6]

        # 追加到二维数组（需要保持形状）
        l = np.array([[1, 2], [3, 4]])
        m = np.append(l, [[5, 6]], axis=0)
        print("\nappend到二维数组（行追加）:\n", m)
        # 输出:
        # [[1 2]
        #  [3 4]
        #  [5 6]]
        ```

    总结对比

    | 方法 | 说明 | 适用场景 |
    |------|------|----------|
    | `concatenate` | 最通用，可指定轴 | 任意维度数组拼接 |
    | `vstack` | 垂直堆叠 | 行方向拼接，处理一维数组方便 |
    | `hstack` | 水平堆叠 | 列方向拼接，处理一维数组方便 |
    | `dstack` | 深度堆叠 | 需要沿第三维拼接时 |
    | `stack` | 沿新轴堆叠 | 需要增加数组维度时 |
    | `append` | 追加元素 | 简单追加操作（不推荐循环使用） |

    **注意事项**：

    - 除了 `stack` 方法，其他方法要求数组在非拼接维度上形状一致
    - `append` 会重新分配内存，效率较低
    - 推荐使用 `concatenate` 及其专门化的变体进行数组拼接

* 简述 numpy 中 `np.loadtxt()` 的原型、作用及用法

    `np.loadtxt()` 详解

    函数原型

    ```python
    numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, 
                  converters=None, skiprows=0, usecols=None, unpack=False, 
                  ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None, 
                  like=None)
    ```

    主要作用

    从文本文件加载数据，返回 NumPy 数组。专门用于读取格式规整的数据文件（如 CSV、TSV 等表格数据）。

    常用参数说明

    | 参数 | 说明 |
    |------|------|
    | **fname** | 文件路径或文件名（必需） |
    | **dtype** | 返回数组的数据类型，默认 float |
    | **delimiter** | 分隔符，默认空白（空格或制表符） |
    | **skiprows** | 跳过的行数（从文件开头） |
    | **usecols** | 指定要读取的列（索引从0开始） |
    | **unpack** | 是否转置（True时返回每列作为独立数组） |
    | **comments** | 注释标记，默认 '#'（忽略该标记后的内容） |
    | **converters** | 对特定列进行转换的字典 |
    | **encoding** | 文件编码，默认 'bytes' |

    用法示例

    1. 基本用法

        ```python
        import numpy as np

        # data.txt 内容：
        # 1 2 3
        # 4 5 6
        # 7 8 9

        data = np.loadtxt('data.txt')
        print(data)
        # 输出：
        # [[1. 2. 3.]
        #  [4. 5. 6.]
        #  [7. 8. 9.]]
        ```

    2. 指定分隔符（如 CSV 文件）

        ```python
        # data.csv 内容：
        # 1,2,3
        # 4,5,6

        data = np.loadtxt('data.csv', delimiter=',', dtype=int)
        print(data)
        # 输出：
        # [[1 2 3]
        #  [4 5 6]]
        ```

    3. 跳过表头和选择列

        ```python
        # data_with_header.txt 内容：
        # col1 col2 col3
        # 1 2 3
        # 4 5 6

        # 跳过第一行，只读取第1列和第3列
        data = np.loadtxt('data_with_header.txt', skiprows=1, usecols=(0, 2))
        print(data)
        # 输出：
        # [[1. 3.]
        #  [4. 6.]]
        ```

    4. 使用 unpack 转置

        ```python
        data = np.loadtxt('data.txt', unpack=True)
        print(data)
        # 输出（每一列变成一行）：
        # [[1. 4. 7.]
        #  [2. 5. 8.]
        #  [3. 6. 9.]]

        # 方便分别获取各列
        x, y, z = np.loadtxt('data.txt', unpack=True)
        ```

    5. 处理混合数据类型

        ```python
        # mixed.txt 内容：
        # A 1 2.5
        # B 3 4.7

        # 使用 converters 转换特定列
        data = np.loadtxt('mixed.txt', dtype=str)
        # 或指定数据类型
        data = np.loadtxt('mixed.txt', dtype={'names': ('label', 'id', 'value'),
                                              'formats': ('U1', 'i4', 'f8')})
        ```

    注意事项

    1. **文件必须格式规整**：每行数据数量相同
    2. **默认跳过注释行**：以 '#' 开头的行会被忽略
    3. **对于复杂数据**：考虑使用 `np.genfromtxt()` 或 `pd.read_csv()`（pandas）

* python numpy 中是否有类似 sortrows 的函数？

    NumPy 中类似 MATLAB sortrows() 的函数

    是的，NumPy 有类似的函数，但需要组合使用不同的功能来实现。

    一、直接等价函数：np.lexsort()

    1. 基本用法（按多列排序）

        ```python
        import numpy as np

        # 创建示例数组
        A = np.array([[3, 2, 1],
                      [1, 3, 2],
                      [2, 1, 3],
                      [1, 2, 1]])

        # 类似 sortrows(A)：默认按第一列，然后第二列...
        # 注意：lexsort 是按从右到左的顺序排序的！
        sorted_A = A[np.lexsort((A[:, 1], A[:, 0]))]  # 先按第0列，再按第1列
        # 或更简洁的方式：
        sorted_A = A[np.lexsort(A.T[::-1])]  # 按所有列排序，最后一列优先级最高
        print("按所有列排序:")
        print(sorted_A)
        ```

    2. 指定列排序

        ```python
        # 类似 sortrows(A, [1, 0])：先按第二列，再按第一列
        sorted_A = A[np.lexsort((A[:, 0], A[:, 1]))]  # 先按第1列，再按第0列
        print("\n先按第1列，再按第0列:")
        print(sorted_A)

        # 更易读的方式：使用 argsort
        indices = np.lexsort((A[:, 0], A[:, 1]))  # 排序索引
        sorted_A = A[indices]
        ```

    二、使用 np.argsort() 实现

    1. 单列排序

        ```python
        # 类似 sortrows(A, 0)：按第一列排序
        A = np.array([[3, 2, 1],
                      [1, 3, 2],
                      [2, 1, 3]])

        # 方法1：直接使用 argsort
        sorted_indices = np.argsort(A[:, 0])
        sorted_A = A[sorted_indices]
        print("按第0列排序:")
        print(sorted_A)

        # 方法2：使用花式索引
        sorted_A = A[A[:, 0].argsort()]
        ```

    2. 多列排序

        ```python
        # 类似 sortrows(A, [1, 0])：先按第二列，再按第一列
        # 使用结构化数组方法
        sorted_indices = np.lexsort((A[:, 0], A[:, 1]))  # 注意顺序：最后的关键字优先
        sorted_A = A[sorted_indices]
        print("\n先按第1列，再按第0列:")
        print(sorted_A)
        ```

    三、使用结构化数组（更接近 MATLAB 体验）

    1. 创建结构化数组并排序

        ```python
        # 创建结构化数组（类似表格）
        dtype = [('col0', int), ('col1', int), ('col2', int)]
        structured_A = np.array([(3, 2, 1),
                                 (1, 3, 2),
                                 (2, 1, 3),
                                 (1, 2, 1)], dtype=dtype)

        # 按指定字段排序（类似 sortrows）
        sorted_A = np.sort(structured_A, order=['col0', 'col1'])  # 先按col0，再按col1
        print("结构化数组排序:")
        print(sorted_A)
        ```

    2. 获取排序后的原始数组

        ```python
        # 获取排序索引并应用到原始数组
        sort_order = np.argsort(structured_A, order=['col0', 'col1'])
        sorted_original = A[sort_order]
        print("\n应用排序索引到原始数组:")
        print(sorted_original)
        ```

    四、Pandas 的替代方案（推荐用于表格数据）

    ```python
    import pandas as pd

    # 创建 DataFrame
    df = pd.DataFrame({
        'col0': [3, 1, 2, 1],
        'col1': [2, 3, 1, 2],
        'col2': [1, 2, 3, 1]
    })

    # 类似 sortrows：按多列排序
    sorted_df = df.sort_values(by=['col0', 'col1'])  # 先按col0，再按col1
    print("Pandas DataFrame 排序:")
    print(sorted_df)

    # 获取 NumPy 数组
    sorted_array = sorted_df.values
    ```

    五、完整对比：MATLAB vs NumPy

    MATLAB 代码：

    ```matlab
    A = [3, 2, 1;
         1, 3, 2;
         2, 1, 3;
         1, 2, 1];

    % 默认：按所有列排序
    B1 = sortrows(A)

    % 指定列：先按第2列，再按第1列
    B2 = sortrows(A, [2, 1])

    % 降序排序
    B3 = sortrows(A, 1, 'descend')
    ```

    NumPy 等价实现：

    ```python
    import numpy as np

    A = np.array([[3, 2, 1],
                  [1, 3, 2],
                  [2, 1, 3],
                  [1, 2, 1]])

    # 1. 默认排序（按所有列）
    B1 = A[np.lexsort(A.T[::-1])]
    print("默认排序（按所有列）:")
    print(B1)

    # 2. 指定列排序（先按第2列[索引1]，再按第1列[索引0]）
    # 注意：lexsort 最后一个参数优先级最高
    B2 = A[np.lexsort((A[:, 0], A[:, 1]))]
    print("\n先按第2列，再按第1列:")
    print(B2)

    # 3. 降序排序
    indices_desc = np.argsort(-A[:, 0])  # 负号实现降序
    B3 = A[indices_desc]
    print("\n第1列降序排序:")
    print(B3)

    # 4. 多列不同方向排序（更复杂）
    # 先按第1列升序，再按第2列降序
    sorted_indices = np.lexsort((-A[:, 1], A[:, 0]))  # 第2列降序，第1列升序
    B4 = A[sorted_indices]
    print("\n第1列升序，第2列降序:")
    print(B4)
    ```

    六、封装为函数（方便使用）

    ```python
    def sortrows(A, columns=None, ascending=True):
        """
        类似 MATLAB 的 sortrows 函数
        
        参数:
        A: numpy 数组
        columns: 列索引列表或整数，指定排序列
        ascending: 排序方向，True为升序，False为降序
        """
        if A.ndim != 2:
            raise ValueError("输入必须是2维数组")
        
        if columns is None:
            columns = range(A.shape[1])
        
        if isinstance(columns, int):
            columns = [columns]
        
        # 处理排序方向
        sort_keys = []
        for i, col in enumerate(columns):
            if isinstance(ascending, bool):
                key = A[:, col] if ascending else -A[:, col]
            elif isinstance(ascending, (list, tuple)):
                key = A[:, col] if ascending[i] else -A[:, col]
            else:
                raise ValueError("ascending 必须是 bool 或列表")
            sort_keys.append(key)
        
        # 注意：lexsort 最后一个参数优先级最高，需要反转
        sorted_indices = np.lexsort(tuple(reversed(sort_keys)))
        
        return A[sorted_indices]

    # 使用示例
    A = np.array([[3, 2, 1],
                  [1, 3, 2],
                  [2, 1, 3],
                  [1, 2, 1]])

    print("默认排序（所有列升序）:")
    print(sortrows(A))

    print("\n按第2列排序（索引1）:")
    print(sortrows(A, columns=1))

    print("\n先按第1列，再按第2列:")
    print(sortrows(A, columns=[0, 1]))

    print("\n第1列降序排序:")
    print(sortrows(A, columns=0, ascending=False))
    ```

    七、性能比较

    ```python
    import numpy as np
    import time

    # 生成大数据
    np.random.seed(42)
    large_A = np.random.randint(0, 100, size=(100000, 5))

    # 方法1: 使用 lexsort
    start = time.time()
    sorted1 = large_A[np.lexsort(large_A.T[::-1])]
    print(f"lexsort 耗时: {time.time() - start:.4f} 秒")

    # 方法2: 使用 argsort 组合（对于多列排序）
    start = time.time()
    sorted_indices = np.lexsort((large_A[:, 0], large_A[:, 1]))
    sorted2 = large_A[sorted_indices]
    print(f"两列排序耗时: {time.time() - start:.4f} 秒")
    ```

    八、总结

    | 方法 | 优点 | 缺点 | 适用场景 |
    | - | - | - | - |
    | np.lexsort() | 最接近 MATLAB，支持多列排序 | 参数顺序与 MATLAB 相反 | 复杂多列排序 |
    | np.argsort() | 简单直接，性能好 | 单列排序简单，多列较复杂 | 单列或简单排序 |
    | 结构化数组 | 语义清晰，类似表格 | 需要转换数据类型 | 固定结构的数据 |
    | Pandas | 功能最丰富，支持复杂操作 | 需要额外依赖 | 表格数据处理 |
    | 封装函数 | 使用方便，接口友好 | 需要额外代码 | 频繁使用类似功能 |

    推荐方案：

    * 简单排序：直接使用 A[A[:, col].argsort()]

    * 多列排序：使用 np.lexsort()

    * 表格数据处理：使用 Pandas（df.sort_values()）

    * 频繁使用：封装为自定义函数

    NumPy 虽然没有直接的 sortrows() 函数，但通过组合现有函数可以轻松实现相同功能，且性能优异。

* numpy 支持`arr[1, 2]`这样的索引方式

    ```python
    def main():
        a = np.zeros((3, 4))
        print(type(a[1, 2]))
        return

    if __name__ == '__main__':
        main()
    ```

    一直以为只有 matlab 支持。

## topics

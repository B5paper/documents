# Pandas Note

`ipython --pylab`

`ipython notebook --pylab=inline`

unofficial windows binaries for python extension packages: <https://www.lfd.uci.edu/~gohlke/pythonlibs/>

一些 ide:

* eclipse with pydev plugin
* python tools for visual studio
* pycharm
* spyder
* komodo ide

一些 maling list，不知道干啥用的：

* pydata: a Google Group list for questions related to Python for data analysis and pandas
* pystatsmodels: for statsmodels or pandas-related questions
* numpy-discussion: for NumPy-related questions
* scipy-user: for general SciPy or scientific Python questions

conferences:

* PyCon
* EuroPython
* SciPy
* EuroSciPy

## cache

* 简述 pandas 的用法与学习资料

    好的，我们来简要概述一下 Pandas 的用法，并推荐一些高质量的学习资料。

    ### 一、Pandas 是什么？

    Pandas 是一个开源的、强大的 Python 数据分析库。它提供了一个快速、灵活且富有表现力的数据结构，旨在使“关系”或“标记”数据的工作变得既简单又直观。它的名字源于“Panel Data”（面板数据）。

    **核心数据结构：**

    1.  **Series**： 带标签的一维数组，可以看作是 Excel 中的一列。
    2.  **DataFrame**： 带标签的二维表格型数据结构，是 Pandas 中最常用、最重要的对象。可以看作是 Excel 中的一个工作表。

    ---

    ### 二、Pandas 核心用法简述

    以下是使用 Pandas 进行数据分析时最常见的操作流程：

    **1. 导入与数据读取**

    ```python
    import pandas as pd

    # 从 CSV 文件读取
    df = pd.read_csv('filename.csv')

    # 从 Excel 文件读取
    df = pd.read_excel('filename.xlsx')

    # 手动创建 DataFrame
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Tokyo']
    })
    ```

    **2. 数据查看与探索**

    ```python
    df.head()        # 查看前5行
    df.info()        # 查看数据概览（行数、列类型、内存等）
    df.describe()    # 生成描述性统计（计数、均值、标准差等）
    df.shape         # 查看数据形状（行数， 列数）
    df.columns       # 查看所有列名
    ```

    **3. 数据选择与过滤**

    ```python
    # 选择单列（返回 Series）
    df['Name']

    # 选择多列（返回 DataFrame）
    df[['Name', 'Age']]

    # 按标签选择行（使用 .loc）
    df.loc[0]          # 选择第一行
    df.loc[0:2, 'Name'] # 选择前3行的 'Name' 列

    # 按位置选择行（使用 .iloc）
    df.iloc[0]         # 选择第一行
    df.iloc[0:2, 0:2]  # 选择前2行，前2列

    # 布尔索引（条件过滤）
    df[df['Age'] > 25]              # 筛选年龄大于25的行
    df[(df['City'] == 'London') & (df['Age'] > 28)] # 多条件筛选
    ```

    **4. 数据处理与清洗**

    ```python
    # 处理缺失值
    df.dropna()       # 删除包含缺失值的行/列
    df.fillna(0)      # 用0填充缺失值

    # 处理重复值
    df.drop_duplicates()

    # 修改数据类型
    df['Age'] = df['Age'].astype(float)

    # 重命名列
    df.rename(columns={'old_name': 'new_name'}, inplace=True)

    # 应用函数
    df['Age_squared'] = df['Age'].apply(lambda x: x**2)
    ```

    **5. 数据分组与聚合**

    ```python
    # 按‘City’分组，并计算‘Age’的平均值
    df.groupby('City')['Age'].mean()

    # 更复杂的聚合（一次性计算多个统计量）
    df.groupby('City').agg({
        'Age': ['mean', 'min', 'max', 'count'],
        'Name': 'count'
    })
    ```

    **6. 数据合并**

    ```python
    # 连接两个 DataFrame（类似 SQL JOIN）
    pd.merge(df1, df2, on='key_column')

    # 沿行方向拼接（堆叠）
    pd.concat([df1, df2], axis=0)
    ```

    **7. 简单可视化**

    Pandas 集成了 Matplotlib，可以快速绘图。

    ```python
    df['Age'].plot.hist()  # 绘制年龄的直方图
    df.groupby('City').size().plot.bar() # 绘制每个城市人数的条形图
    ```

    ---

    ### 三、学习资料推荐

    **1. 官方文档（首选！）**
       - **链接**: [pandas.pydata.org](https://pandas.pydata.org/)
       - **优点**: 最权威、最全面、最新。包含了完整的 API 参考和用户指南（User Guide），后者像一本详细的教程。
       - **建议**: 遇到任何函数或方法不确定时，首先查阅官方文档。

    **2. 经典书籍**
       - **《利用Python进行数据分析》（Python for Data Analysis）**
       - **作者**: Wes McKinney（Pandas 的创始人）
       - **内容**: 从 Pandas 基础到实战，是公认的入门圣经。

    **3. 在线教程与课程**
       - **Kaggle Learn**: 提供免费的 [Pandas 微课程](https://www.kaggle.com/learn/pandas)，交互式环境，非常适合新手快速上手。
       - **DataCamp** / **Coursera** / **Udemy**: 提供系统性的数据科学课程，其中 Pandas 是核心组成部分。通常需要付费，但质量很高。

    **4. 实践平台**
       - **Kaggle**: 上面有成千上万的真实数据集和别人的代码（Kernels/Notebooks），是学习 Pandas 高级用法和数据分析思路的最佳场所。
       - **LeetCode 题库之 Pandas**: 专门用 Pandas 解决数据处理问题，非常适合面试准备和巩固基础。

    ### 学习建议

    1.  **边学边练**：不要只看不写。安装好 Jupyter Notebook 或 VS Code 等环境，跟着例子敲代码。
    2.  **从实际问题出发**：找一个你感兴趣的数据集（比如电影数据、股票数据、体育数据），尝试用 Pandas 去回答一些问题。
    3.  **善用搜索引擎**：遇到报错或不知道如何实现的功能，直接搜索“pandas how to ...”，通常能在 Stack Overflow 上找到答案。

    掌握了 Pandas 的核心用法，你就拥有了处理和分析数据的强大能力，这是迈向数据科学家、分析师等领域至关重要的一步。

* 我想使用 pandas 像 numpy np.arange() ，或者 np.random.randint(), np.zeros() 等生成一些数据，该怎么办？

    非常好！Pandas 提供了类似 NumPy 的数据生成功能，而且更加方便，因为生成的就是带有标签的 Series 或 DataFrame。

    ## 1. 类似 `np.arange()` - 创建序列数据

    ```python
    import pandas as pd
    import numpy as np

    # 创建类似 np.arange 的序列
    # 方法1：使用 pd.RangeIndex (最接近 np.arange)
    s1 = pd.Series(np.arange(10))  # [0, 1, 2, ..., 9]
    s2 = pd.Series(range(10))      # 同上

    # 方法2：直接创建带索引的Series
    s3 = pd.Series(index=range(10), data=np.random.randn(10))
    ```

    ## 2. 类似 `np.random.randint()` - 生成随机整数

    ```python
    # 生成随机整数的Series
    s_randint = pd.Series(np.random.randint(0, 100, size=10))
    print(s_randint)

    # 生成随机整数的DataFrame
    df_randint = pd.DataFrame({
        'A': np.random.randint(0, 50, 5),
        'B': np.random.randint(50, 100, 5),
        'C': np.random.randint(100, 150, 5)
    })
    print(df_randint)
    ```

    ## 3. 类似 `np.zeros()`, `np.ones()` - 生成全0或全1数据

    ```python
    # 全0的Series
    s_zeros = pd.Series(np.zeros(5))
    # 全1的DataFrame
    df_ones = pd.DataFrame(np.ones((3, 4)), columns=['A', 'B', 'C', 'D'])
    ```

    ## 4. Pandas 特有的数据生成方法

    ### 日期范围数据
    ```python
    # 生成日期序列（非常实用！）
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    df_dates = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(10)
    })
    print(df_dates)
    ```

    ### 使用 `pd.util.testing` (新版在 `pandas._testing`)
    ```python
    # 快速生成测试数据
    # 方法1：makeDataFrame (生成随机数据的DataFrame)
    df_test = pd.util.testing.makeDataFrame()  # 4行4列的随机数据

    # 方法2：makeTimeDataFrame (带时间索引)
    df_time = pd.util.testing.makeTimeDataFrame()  # 30行4列，带日期索引
    ```

    ## 5. 更实用的数据生成示例

    ```python
    # 示例1：生成学生成绩数据
    np.random.seed(42)  # 设置随机种子保证可重复性

    students_data = pd.DataFrame({
        'student_id': range(1, 101),
        'math_score': np.random.randint(60, 100, 100),
        'english_score': np.random.randint(50, 95, 100),
        'science_score': np.random.randint(70, 98, 100)
    })
    print(students_data.head())

    # 示例2：生成时间序列销售数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(1000, 5000, 100),
        'product': np.random.choice(['A', 'B', 'C'], 100)
    })
    print(sales_data.head())
    ```

    ## 6. 使用 `pd.DataFrame()` 构造函数

    ```python
    # 从NumPy数组创建
    arr = np.random.randn(5, 3)
    df_from_np = pd.DataFrame(arr, columns=['X', 'Y', 'Z'], index=['a', 'b', 'c', 'd', 'e'])

    # 从字典创建（最常用）
    data_dict = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'city': ['NY', 'London', 'Tokyo', 'Paris']
    }
    df_from_dict = pd.DataFrame(data_dict)
    ```

    ## 7. 专业的数据生成技巧

    ```python
    # 生成分类数据
    categories = ['Low', 'Medium', 'High']
    df_cat = pd.DataFrame({
        'category': np.random.choice(categories, 50),
        'value': np.random.normal(100, 15, 50)
    })

    # 生成有相关性的数据
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = x * 0.8 + np.random.normal(0, 0.2, 100)  # y与x相关
    df_correlated = pd.DataFrame({'X': x, 'Y': y})
    ```

    ## 主要区别总结

    | NumPy 方法 | Pandas 对应方法 | 主要区别 |
    |------------|-----------------|----------|
    | `np.arange()` | `pd.Series(np.arange())` | Pandas 带索引 |
    | `np.random.randint()` | `pd.Series/DataFrame(np.random.randint())` | 可直接生成带列名的数据 |
    | `np.zeros()/ones()` | `pd.Series/DataFrame(np.zeros())` | 可指定索引和列名 |

    **核心优势**：Pandas 在 NumPy 的基础上增加了**索引**和**列名**，让数据更有意义、更易于操作！

    这些方法应该能满足你生成测试数据的需求了！

* 那如果我现在有一个 numpy 的 ndarray，如何转换成 DataFrame

    非常好！将 NumPy ndarray 转换为 DataFrame 是 Pandas 中最常见和简单的操作之一。有几种方法可以实现：

    ## 1. 基本转换方法

    ### 最基本的方式
    ```python
    import pandas as pd
    import numpy as np

    # 创建一个示例 numpy array
    arr = np.array([[1, 2, 3], 
                    [4, 5, 6], 
                    [7, 8, 9]])

    # 直接转换为 DataFrame
    df = pd.DataFrame(arr)
    print(df)
    ```
    输出：
    ```
       0  1  2
    0  1  2  3
    1  4  5  6
    2  7  8  9
    ```

    ## 2. 自定义索引和列名

    ```python
    # 指定行索引和列名
    df_custom = pd.DataFrame(
        arr,
        index=['row1', 'row2', 'row3'],  # 行索引
        columns=['A', 'B', 'C']          # 列名
    )
    print(df_custom)
    ```
    输出：
    ```
          A  B  C
    row1  1  2  3
    row2  4  5  6
    row3  7  8  9
    ```

    ## 3. 不同维度的数组转换

    ### 一维数组 → Series
    ```python
    # 一维数组
    arr_1d = np.array([10, 20, 30, 40, 50])

    # 转换为 Series
    series = pd.Series(arr_1d, index=['a', 'b', 'c', 'd', 'e'])
    print(series)
    ```

    ### 一维数组 → DataFrame (单列)
    ```python
    # 一维数组转为单列 DataFrame
    df_1d = pd.DataFrame(arr_1d, columns=['values'])
    print(df_1d)
    ```

    ### 二维数组 → DataFrame
    ```python
    # 二维数组（最常用）
    arr_2d = np.random.randn(4, 3)  # 4行3列
    df_2d = pd.DataFrame(arr_2d, columns=['Feature1', 'Feature2', 'Feature3'])
    print(df_2d)
    ```

    ### 三维数组的处理
    ```python
    # 三维数组需要先reshape为二维
    arr_3d = np.random.randn(2, 3, 4)  # 2×3×4
    arr_2d = arr_3d.reshape(2, -1)     # 转换为 2×12
    df_3d = pd.DataFrame(arr_2d)
    print(df_3d.shape)  # (2, 12)
    ```

    ## 4. 实际应用示例

    ### 示例1：机器学习数据集
    ```python
    # 生成特征矩阵和标签
    X = np.random.randn(100, 5)  # 100个样本，5个特征
    y = np.random.randint(0, 2, 100)  # 二分类标签

    # 转换为 DataFrame
    features_df = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    target_df = pd.DataFrame(y, columns=['target'])

    # 合并
    ml_data = pd.concat([features_df, target_df], axis=1)
    print(ml_data.head())
    ```

    ### 示例2：时间序列数据
    ```python
    # 生成时间序列数组
    time_series_data = np.cumsum(np.random.randn(100))  # 随机游走

    # 转换为 DataFrame 并添加时间索引
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    ts_df = pd.DataFrame(time_series_data, index=dates, columns=['Price'])
    print(ts_df.head())
    ```

    ### 示例3：从多个数组创建
    ```python
    # 多个相关数组
    arr1 = np.random.randint(1, 100, 50)
    arr2 = np.random.normal(0, 1, 50)
    arr3 = np.random.choice(['A', 'B', 'C'], 50)

    # 合并为 DataFrame
    multi_df = pd.DataFrame({
        'integers': arr1,
        'floats': arr2,
        'categories': arr3
    })
    print(multi_df.head())
    ```

    ## 5. 高级技巧

    ### 指定数据类型
    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    df_dtype = pd.DataFrame(arr, dtype=np.float32)  # 指定数据类型
    print(df_dtype.dtypes)
    ```

    ### 从结构化数组转换
    ```python
    # 结构化数组（类似数据库表）
    structured_arr = np.array([
        ('Alice', 25, 65.5),
        ('Bob', 30, 70.2),
        ('Charlie', 35, 68.9)
    ], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

    df_structured = pd.DataFrame(structured_arr)
    print(df_structured)
    ```

    ## 6. 验证转换结果

    ```python
    # 创建测试数组
    test_arr = np.random.randn(5, 4)

    # 转换
    df_test = pd.DataFrame(test_arr, columns=['A', 'B', 'C', 'D'])

    # 验证
    print("原始数组形状:", test_arr.shape)
    print("DataFrame形状:", df_test.shape)
    print("数据类型:")
    print(df_test.dtypes)
    print("\n前几行数据:")
    print(df_test.head())
    ```

    ## 重要提示

    1. **数据共享**：默认情况下，DataFrame 与原始数组共享内存（视图），修改 DataFrame 可能会影响原数组
    2. **复制数据**：如果需要独立副本，使用 `pd.DataFrame(arr.copy())`
    3. **性能考虑**：对于大型数组，直接转换通常很快，因为 Pandas 底层使用 NumPy

    ```python
    # 内存共享示例
    arr = np.array([1, 2, 3, 4, 5])
    df = pd.DataFrame(arr)

    # 修改 DataFrame
    df.iloc[0, 0] = 999
    print("原数组:", arr)  # 也会被修改！
    print("DataFrame:", df.iloc[:, 0].values)

    # 避免内存共享
    arr = np.array([1, 2, 3, 4, 5])
    df = pd.DataFrame(arr.copy())  # 创建副本
    ```

    这样你就可以轻松地在 NumPy 和 Pandas 之间转换数据了！

## Introduction

### Series

`Series`可以看作是扩展了很多功能的一维数组。

* 创建

    ```python
    s1 = pd.Series([2, 4, 6, 8])  # 可以通过 list 直接创建
    s1 = pd.Series(np.arange(2, 10, 2))  # 也可以和 numpy 互相兼容
    s2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])  # 用参数 index 指定索引
    s2 = pd.Series({'d': 4, 'b': 7, 'a': -5, 'c': 3})  # 用字典的方式指定索引
    print(s1)
    print()
    print(s2)
    ```

    输出：

    ```
    0    2
    1    4
    2    6
    3    8
    dtype: int32

    d    4
    b    7
    a   -5
    c    3
    dtype: int64
    ```

* 索引

    ```python
    s2[0]  # 使用数字索引
    s2.d  # 使用属性索引
    s2['d'] = 6  # 使用字符串索引，并赋值
    s2[['c', 'a', 'd']]  # 索引多个值
    s2[s2 > 0]  # 布尔索引
    ```

* 运算

    ```python
    s2 * 2  # 可以直接使用运算符
    np.exp(s2)  # 可以和 numpy 兼容
    'b' in s2  # 检查键值是否存在

    # 带缺失数据的运算（不知道这个功能复现的充要条件是什么）
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    obj3 = pd.Series(sdata)
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    obj4 = Series(sdata, index=states)
    print(obj3 + obj4)
    ```

    输出：

    ```
    California     NaN
    Ohio         70000
    Oregon       32000
    Texas       142000
    Utah           NaN
    ```

* 常用属性（attribute）

    * `name`

    * `index`

### DataFrame

`DataFrame`和一个二维数组类似，不同于二维数组的是，它的每一列都是一个`Series`对象。因此它的每一行代表了一条数据，每一列代表了一个属性。

* 创建

    ```python
    df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])  # 可以通过 array-like 的类型创建
    df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6]],
                        index=['a', 'b'],
                        columns=['A', 'B', 'C'])  # 指定行索引和列索引
    df3 = pd.DataFrame({'A': [1, 4], 'B': [2, 5], 'C': [3, 6]})  # 按列创建，同时指定列索引
    df4 = pd.DataFrame({'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}, columns=['a', 'c'])  # 使用部分数据创建
    df5 = df5 = pd.DataFrame({
        'a': {'A': 1, 'B': 4}, 
        'b': {'A': 2, 'B': 5},
        'c': {'A': 3, 'B': 6}})  # 使用嵌套的字典创建，这时可同时指定 index 和 columns 从而使用部分数据创建
        # 另一个作用是可以有缺失值，这时会自动填充为 NaN
    print(df1)
    print()
    print(df2)
    ```

    输出：

    ```
    0  1  2
    0  1  2  3
    1  4  5  6

    A  B  C
    a  1  2  3
    b  4  5  6
    ```

    不提供属性名字的情况下，会从`0`开始编号。这个例子中有`0`，`1`，`2`三列属性。每行也都有一个名字，这里是`0`和`1`。

    如果`index`指定不存在的`key`，那么数值就会被填充为`NaN`。

* 常用属性

    * `index`

    * `columns`

* 索引

    在`frame`后使用方括号可以索引到一个属性，即一列：

    ```python
    df['a']  # 索引一列
    df.a  # 同上
    df[df > 2]  # 布尔索引
    df.ix['A']  # 索引一行（好像不行
    ```

    使用索引得到的列都是引用，如果想得到副本，需要`Series`的`copy`方法。

* 切片

    ```python
    df1[:2]  # 取前两行
    ```

* 赋值

    ```python
    df[0] = 0  # 将整列都赋为 0
    df[0] = np.arange(5)  # 能和 numpy 兼容

    # 对部分赋值，缺失的数值赋 NaN
    val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
    frame2['debt'] = val

    # 创建一个新列
    frame2['eastern'] = frame2.state == 'Ohio'

    # 删除一列
    del frame2['eastern']
    ```

一些常用的方法：

* `value_counts()`

* `fillna(val)`

* `plot(kind, rot)`

* `notnull()`

* `unstack()`

* `groupby()`

* `sum()`

* `argsort()`

* `take()`

* `div()`

* `isnull()`

    找出 missing or NA values

    也有全局函数：`pd.isnull()`

常用的属性：

* `values`

* `index`

* `columns`

* `T`

然后是一个看不懂的操作：

```python
pdata = {'Ohio': frame3['Ohio'][:-1],
'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)
```

输出：

```
     Nevada Ohio
2000 NaN    1.5
2001 2.4    1.7
```

`DataFrame`的`index`和`columns`也可以有`name`：

```python
frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3
```

输出：

```
state Nevada Ohio
year
2000 NaN 1.5
2001 2.4 1.7
2002 2.9 3.6
```

如果`columns`是不同的`dtype`，那么`frame.values`得到的就是`dtype=object`类型的 ndarray。

`DataFrame`的构造函数一览表：

| Type | Notes |
| - | - |
| 2d ndarray | A matrix of data, passing optional row and column labels. |
| dict of arrays, lists, or tuples | Each sequence becomes a column in the DataFrame. All sequences must be the same length. |
| NumPy structured/record array | Treated as the 'dict of arrays' case. |
| dict of Series | Each value becomes a column. Indexes from each Series are unioned together to form the result's row index if no explicit index is passed. |
| dict of dicts | Each inner dict becomes a column. Keys are unioned to form the row index as in the 'dict of Series' case. |
| list of dicts of Series | Each item becomes a row in the DataFrame. Union of dict key or Series indexes become the DataFrame's column labels. |
| List of lists or tuples | Treated as the '2D ndarray' case. |
| Another DataFrame | The DataFrame's indexes are used unless different ones are passed. |
| NumPy MaskedArray | Like the '2D ndarray' case except masked values become NA/missing in the DataFrame result. |

**Index Objects**

`Index`对象无法被修改，只能被替换。因此它可以被共享。

```python
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index  # True
```

pandas 中常见的 index 的类型：

| Class | Description |
| - | - |
| `Index` | The most general Index object, representing axis labels in a NumPy array of Python objects. |
| `Int64Index` | Specialized Index for integer values. |
| `MultiIndex` | "Hierarchical" index object representing multiple levels of indexing on a single axis. Can be thought of as similar to an array of tuples. |
| `DatatimeIndex` | Stores nanosecond timestamps (represented using NumPy's datetime64 dtype). |
| `PeriodIndex` | Specialized Index for Period data (timespans). |

`index`和`columns`也可以看作是一个 set：可以用`in`判断某个元素是否在其中，也有如下所示的方法和属性：

| Method | Description |
| - | - |
| `append` | Concatenate with additional Index objects, producing a new Index |
| `diff` | Compute set difference as an Index |
| `intersection` | Compute set intersection |
| `union` | Compute set union |
| `isin` | Compute boolean array indicating whether each value is contained in the passed collection |
| `delete` | Compute new Index with element at index i deleted |
| `drop` | Compute new index by deleting passed values |
| `insert` | Compute new Index by inserting element at index i |
| `is_monotonic` | Returns True if each element is greater than or euqal to the previous element |
| `is_unique` | Returns True if the Index has no duplicate values |
| `unique` | Compute the array of unique values in the Index |

一些核心功能：

* `reindex(index, method, fill_value, limit, level, copy)`

    重新赋值索引。（`method`参数好像没啥用，有机会了找文档再核实下）

* `drop(index)`

    删除指定的索引及对应的值。

`Series`的索引和 numpy 类似，有一点不同：当使用 label 作为索引时，end point 是包含在内的。

对一个`DataFrame`索引会得到`Series`，然而只有两种情况是例外，会得到`DataFrame`，一种是切片，一种是布尔索引。

Indexing options with DataFrame

| Type | Notes |
| - | - |
| `obj[val]` | Select single column or sequence of columns from the DataFrame. Special case conveniences: boolean array (filter rows), slice (slice rows), or boolean DataFrame (set values based on some criterion). |
| `obj.iloc[val]` | Selects single row of subset of rows from the DataFrame. |
| `obj.iloc[:, val]` | Selects single column of subset of columns. |
| `obj.iloc[val1, val2]` | Select both rows and columns. |
| reindex method | Conform one or more axes to new indexes. |
| `xs` method | Select single row or column as a Series by label. |
| `icol`, `irow` methods | Select single column or row, respecitively, as a Series by integer location. |
| `get_value`, `set_value` methods | Select single value by row and column label. |

如果让两个行索引不同的`Series`相加，那么会相加有相同行索引的项，行索引不同的项会被赋`NaN`。`DataFrame`同理。如果不想这样，可以`df1.add(df2, fill_value=0)`，这样就会把不存在的元素赋 0，再做相加。对于`reindex`也是同理，可以将不存在的列填充：`df1.reindex(columns=df2.columns, fill_value=0)`。

常用的算术操作有 4 个：`add`，`sub`，`div`，`mul`。

对于`DataFrame`和`Series`的算术操作，会进行 broadcasting. `Series`会被看作行。如果想将`Series`看作列，可以使用`add(ser, axis=0)`。如果出现索引不匹配的情况，会赋值`NaN`.

dimension reduction:

```python
f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis=1)

def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)

format = labmda x: '%.2f' % x
frame.applymap(format)

# applymap 应用于 DataFrame，map 应用于 Series
frame['e'].map(format)
```

# pandas note

* pandas 生成数据 examples

    * 类似 np.arange() - 创建序列数据

        ```py
        import pandas as pd
        import numpy as np

        # 创建类似 np.arange 的序列
        # 方法1：使用 pd.RangeIndex (最接近 np.arange)
        s1 = pd.Series(np.arange(10))  # [0, 1, 2, ..., 9]
        s2 = pd.Series(range(10))      # 同上

        # 方法2：直接创建带索引的Series
        s3 = pd.Series(index=range(10), data=np.random.randn(10))
        ```

    * 类似 np.random.randint() - 生成随机整数

        ```py
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

    * 类似 np.zeros(), np.ones() - 生成全0或全1数据

        ```py
        # 全0的Series
        s_zeros = pd.Series(np.zeros(5))
        # 全1的DataFrame
        df_ones = pd.DataFrame(np.ones((3, 4)), columns=['A', 'B', 'C', 'D'])
        ```

    * 日期范围数据

        ```py
        # 生成日期序列（非常实用！）
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df_dates = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(10)
        })
        print(df_dates)
        ```

    * 使用 pd.util.testing (新版在 pandas._testing)

        ```py
        # 快速生成测试数据
        # 方法1：makeDataFrame (生成随机数据的DataFrame)
        df_test = pd.util.testing.makeDataFrame()  # 4行4列的随机数据

        # 方法2：makeTimeDataFrame (带时间索引)
        df_time = pd.util.testing.makeTimeDataFrame()  # 30行4列，带日期索引
        ```

    * 综合 example

        ```py
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

    * 其他数据生成方法

        ```py
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

* 将 numpy 转换成 DataFrame

    * 基本转换法

        ```py
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

        output:

        ```
           0  1  2
        0  1  2  3
        1  4  5  6
        2  7  8  9
        ```

    * 自定义索引和列名

        ```py
        # 指定行索引和列名
        df_custom = pd.DataFrame(
            arr,
            index=['row1', 'row2', 'row3'],  # 行索引
            columns=['A', 'B', 'C']          # 列名
        )
        print(df_custom)
        ```

        output:

        ```
              A  B  C
        row1  1  2  3
        row2  4  5  6
        row3  7  8  9
        ```

    * 不同维度的数组转换

        * 一维数组 → Series

            ```py
            # 一维数组
            arr_1d = np.array([10, 20, 30, 40, 50])

            # 转换为 Series
            series = pd.Series(arr_1d, index=['a', 'b', 'c', 'd', 'e'])
            print(series)
            ```

        * 一维数组 → DataFrame (单列)

            ```py
            # 一维数组转为单列 DataFrame
            df_1d = pd.DataFrame(arr_1d, columns=['values'])
            print(df_1d)
            ```

        * 二维数组 → DataFrame

            ```py
            # 二维数组（最常用）
            arr_2d = np.random.randn(4, 3)  # 4行3列
            df_2d = pd.DataFrame(arr_2d, columns=['Feature1', 'Feature2', 'Feature3'])
            print(df_2d)
            ```

        * 三维数组的处理

            ```py
            # 三维数组需要先reshape为二维
            arr_3d = np.random.randn(2, 3, 4)  # 2×3×4
            arr_2d = arr_3d.reshape(2, -1)     # 转换为 2×12
            df_3d = pd.DataFrame(arr_2d)
            print(df_3d.shape)  # (2, 12)
            ```
    * 综合 example

        * 机器学习数据集

            ```py
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

        * 时间序列数据

            ```py
            # 生成时间序列数组
            time_series_data = np.cumsum(np.random.randn(100))  # 随机游走

            # 转换为 DataFrame 并添加时间索引
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            ts_df = pd.DataFrame(time_series_data, index=dates, columns=['Price'])
            print(ts_df.head())
            ```

        * 从多个数组创建

            ```py
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

* pandas user guide api

    <https://pandas.pydata.org>

* pandas

    * 手动创建 DataFrame

        ```py
        import pandas as pd

        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })

        print(df)
        ```

        output:

        ```
              Name  Age      City
        0    Alice   25  New York
        1      Bob   30    London
        2  Charlie   35     Tokyo
        ```

    * 导入与数据读取

        ```py
        import pandas as pd

        # 从 CSV 文件读取
        df = pd.read_csv('filename.csv')

        # 从 Excel 文件读取
        df = pd.read_excel('filename.xlsx')
        ```

    * 数据查看与探索

        ```py
        df.head()        # 查看前5行
        df.info()        # 查看数据概览（行数、列类型、内存等）
        df.describe()    # 生成描述性统计（计数、均值、标准差等）
        df.shape         # 查看数据形状（行数， 列数）
        df.columns       # 查看所有列名
        ```

        example:

        ```py
        import pandas as pd
        from pandas import DataFrame

        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })

        # 查看前 5 行
        print('head:')
        df_head: DataFrame = df.head()
        print(df_head)
        print()

        # 查看数据概览（行数、列类型、内存等）
        print('info:')
        df.info()
        print()

        # 生成描述性统计（计数、均值、标准差等）
        print('describe:')
        df_des: DataFrame = df.describe()
        print(df_des)
        print()

        # 查看数据形状（行数， 列数）
        print('shape:')
        print('type: {}, data: {}'.format(type(df.shape), df.shape))
        print()

        # 查看所有列名
        print('columns:')
        print(df.columns)
        ```

        output:

        ```
        head:
              Name  Age      City
        0    Alice   25  New York
        1      Bob   30    London
        2  Charlie   35     Tokyo

        info:
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
         #   Column  Non-Null Count  Dtype 
        ---  ------  --------------  ----- 
         0   Name    3 non-null      object
         1   Age     3 non-null      int64 
         2   City    3 non-null      object
        dtypes: int64(1), object(2)
        memory usage: 200.0+ bytes

        describe:
                Age
        count   3.0
        mean   30.0
        std     5.0
        min    25.0
        25%    27.5
        50%    30.0
        75%    32.5
        max    35.0

        shape:
        type: <class 'tuple'>, data: (3, 3)

        columns:
        Index(['Name', 'Age', 'City'], dtype='object')
        ```

    * 数据选择与过滤

        ```py
        # 选择单列（返回 Series）
        df['Name']

        # 选择多列（返回 DataFrame）
        df[['Name', 'Age']]

        # 按标签选择行（使用 .loc）
        df.loc[0]          # 选择第一行
        df.loc[0:2, 'Name'] # 选择前 2 行的 'Name' 列

        # 按位置选择行（使用 .iloc）
        df.iloc[0]         # 选择第一行
        df.iloc[0:2, 0:2]  # 选择前 2 行，前 2 列

        # 布尔索引（条件过滤）
        df[df['Age'] > 25]              # 筛选年龄大于25的行
        df[(df['City'] == 'London') & (df['Age'] > 28)] # 多条件筛选
        ```

        其中`0:2`也可以写成`:2`。

    * 数据处理与清洗

        ```py
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

    * 数据分组与聚合

        ```py
        # 按‘City’分组，并计算‘Age’的平均值
        df.groupby('City')['Age'].mean()

        # 更复杂的聚合（一次性计算多个统计量）
        df.groupby('City').agg({
            'Age': ['mean', 'min', 'max', 'count'],
            'Name': 'count'
        })
        ```

    * 数据合并

        ```py
        # 连接两个 DataFrame（类似 SQL JOIN）
        pd.merge(df1, df2, on='key_column')

        # 沿行方向拼接（堆叠）
        pd.concat([df1, df2], axis=0)
        ```

    * 简单可视化

        Pandas 集成了 Matplotlib，可以快速绘图。

        ```py
        df['Age'].plot.hist()  # 绘制年龄的直方图
        df.groupby('City').size().plot.bar() # 绘制每个城市人数的条形图
        ```
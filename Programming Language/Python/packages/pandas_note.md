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
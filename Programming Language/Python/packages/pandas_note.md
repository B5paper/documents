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

```python
from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
frame
frame['tz'][:10]
tz_counts = frame['tz'].value_counts()
```

The output shown for the `frame` is the *summary view*, shown for large DataFrame objects.

```python
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10]

tz_counts[:10].plot(kind='barh', rot=0)
```

使用`Series`:

```python
results = Series([x.split()[0] for x in frame.a.drapna()])
results[:5]
results.value_counts()[:8]

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
operating_system[:5]

by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

indexer = agg_counts.sum(1).argsort()
index[:10]

count_subset = agg_counts.take(indexer)[-10:]
count_subset

count_subset.plot(kind='barh', stacked=True)
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
```

**Series**

一维数组。

```python
obj = Series([4, 7, -5, 3])
obj
obj.values
obj.index

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index
obj2['a']  # -5
obj2['d'] = 6
obj2[['c', 'a', 'd']]  # 索引多个值

obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)

'b' in obj2  # True
'e' in obj2  # False

# Create a Series from a dict
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3  # index 会按字典序排序

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)  # 指定 keys，在字典中查找，若找到则放到 obj4 里，否则给出 NaN
obj4
```

可以用`isnull()`和`notnull()`找出缺失值（missing or NA values）：

```python
pd.isnull(obj4)
```

输出：

```
California  True
Ohio        False
Oregon      False
Texas       False
```

```python
pd.notnull(obj4)
```

输出：

```
California  False
Ohio        True
Oregon      True
Texas       True
```

也可以用内置方法调用：`obj4.isnull()`

pandas 可以在缺失数据的情况下做运算：

```python
obj3
```

输出：

```
Ohio    35000
Oregon  16000
Texas   71000
Utah     5000
```

输入：

```python
obj4
```

输出：

```
California     NaN
Ohio         35000
Oregon       16000
Texas        71000
```

输入：

```python
obj3 + obj4
```

输出：

```
California     NaN
Ohio         70000
Oregon       32000
Texas       142000
Utah           NaN
```

`Series`对象和它的`index`都有一个`name` attribute。

`Series`的`index`也可直接被赋值：`obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']`

**DataFrame**

```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame
```

输出：

```
  pop state  year
0 1.5 Ohio   2000
1 1.7 Ohio   2001
2 3.6 Ohio   2002
3 2.4 Nevada 2001
4 2.9 Nevada 2002
```

也可以指定顺序：

```python
DataFrame(data, columns=['year', 'state', 'pop'])
```

输出：

```
  year state  pop
0 2000 Ohio   1.5
1 2001 Ohio   1.7
2 2002 Ohio   3.6
3 2001 Nevada 2.4
4 2002 Nevada 2.9
```

如果在`column`里指定不存在的`Series`，那么会被赋值为`NA`：

```python
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2
```

输出：

```
      year state  pop debt
one   2000 Ohio   1.5 NaN
two   2001 Ohio   1.7 NaN
three 2002 Ohio   3.6 NaN
four  2001 Nevada 2.4 NaN
five  2002 Nevada 2.9 NaN
```

```python
frame2.columns
```

输出：

```
Index([year, state, pop, debt], dtype=object)
```

两种方式索引某一列：

```python
frame2['state']
frame2.state
```

使用名称索引某一行：`frame2.ix['three']`

对整列赋值：

```python
frame2['debt'] = 16.5
frame2['debt'] = np.arange(5.)

# 对部分赋值，缺失的数值赋 NaN
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val

# 创建一个新列
frame2['eastern'] = frame2.state == 'Ohio'

# 删除一列
del frame2['eastern']
```

使用索引得到的列都是引用，如果想得到副本，需要`Series`的`copy`方法。

另外一种创建 DataFrame 的方法：

```python
pop = {'Nevada': {2001: 2.4, 2003: 2.9},
'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
```

对 DataFrame 进行转置：

```python
frame3.T
```

`index`还可以指定不存在的`key`：

```python
DataFrame(pop, index=[2001, 2002, 2003])
```

输出：

```
     Nevada Ohio
2001 2.4    1.7
2002 2.9    3.6
2003 NaN    NaN
```

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

拿到 2D 的`ndarray`数据：

```python
frame3.values
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
| NumPy MaskedArray | Like the '2D ndarray' case except masked values become NA/missing in the DataFrame result.|
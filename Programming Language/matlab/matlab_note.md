# Matlab Note

`help func_name`

查找函数：`lookfor keyword`

`lookfor`搜索所有函数的第一行注释行，找到对应的文件或函数。

在脚本中，两个`%`可以标志一个 block：

```
a = 1

%% new block
b = 2
```

使用`Ctrl + Enter`可以运行 block 中的内容。

块注释：

```matlab
%{

    comments

%}
```

其他常用命令：

* `who`：列出当前工作空间中的变量
* `what`：列出当前文件夹或指定目录下的 M 文件、MAT 文件和 MEX 文件
* `which`：显示指定函数或文件的路径
* `whos`：列出当前工作空间中变量的更多信息
* `exist`：检查指定变量或文件的存在性
* `doc`：直接查询在线文档。通常更详细
* `echo`：直接运行时，切换是否显示 m 文件中的内容。也可以`echo on`，`echo off`来指定状态。

预置变量：

* `eps`：计算机的最小正数
* `pi`：圆周率 pi 的近似值 3.14159265358979
* `inf`或`Inf`：无穷大
* `NaN`：不定量
* `i, j`：虚数单位定义`i`
* `flops`：浮点运算次数，用于统计计算量

环境 -> 设置路径 可以为 Matlab 添加或删除搜索路径。

`clear`：删除工作区中的所有变量。

`clear var1, var2`：清除指定变量

`pack`：用于整理内存。将内存中的数据先存储到磁盘上，再从磁盘将数据读入到内存中。

数值的显示格式：

|格式命令|作用|
|-|-|
|`format short`|5 位有效数字|
|`format long`|15 位有效数字|
|`format short e`|5位有效数字 + 科学计数法|
|`format long e`|15 位有效数字 + 科学计数法|
|`format short g`|短紧缩格式|
|`format long g`|长紧缩格式|
|`format hex`|十六进制，浮点数|
|`format bank`|2 位小数|
|`format +`|正，负或 0|
|`format rat`|有理数近似|
|`format debug`|短紧缩格式的内部存储信息|

常用的控制命令：

* `cd`：显示或改变当前文件夹
* `dir`：显示当前文件夹或指定文件夹下的文件
* `clc`：清除工作窗口中的所有显示内容
* `home`：将光标移至命令行窗口的最左上角
* `clf`：清除图形窗口
* `type`：显示文件内容
* `clear`：清理内存变量
* `echo`：工作窗口信息显示开关
* `disp`：显示变量或文字内容
* `load`：加载指定文件的变量
* `diary`：日志文件命令
* `!`：调用 dos 命令
* `exit`，`quit`：退出 
* `pack`：收集内存碎片
* `hold`：图形保持开关
* `path`：显示搜索目录
* `save`：将内存变量保存到指定文件中

命令行中的快捷键：

* `Up`, `Ctrl + P`：调用上一行
* `Down`, `Ctrl + N`：调用下一行
* `Left`,  `Ctrl + B`：光标左移一个字符
* `Right`, `Ctrl + F`：光标右移一个字符
* `Ctrl + Left`, `Ctrl + L`：光标左移一个单词
* `Ctrl + Right`, `Ctrl + R`：光标右移一个单词
* `Home`, `Ctrl + A`：光标置于当前行开头
* `End`, `Ctrl + E`：光标置于当前行末尾
* `Esc`, `Ctrl + U`：清除当前输入行
* `Del`, `Ctrl + D`：删除光标处的字符
* `Backspace`, `Ctrl + H`：删除光标前的字符
* `Alt + Backspace`：恢复上一次删除

运算符：除法为`\`或`/`，乘方为`^`。

显示运算符优先顺序：`help precedence`

输入矩阵：

```matlab
a = [1, 2, 3;, 4, 5, 6; 7, 8, 9]
a = [1 2 3 4
    5 6 7 8
    0 1 2 3]
```

还可以直接输入矩阵元素：

```matlab
B(1,2) = 3;
B(4,4) = 6;
B(4,2) = 11;
```

matlab 会自动构建一个剩下元素都是 0 的矩阵：

```matlab
B = 
  0 3 0 0
  0 0 0 0
  0 0 0 0
  0 11 0 6
```

注释：使用`%`进行单行注释。

整数类型：

`int8()`, `int16()`, `int32()`, `int64()`, `uint8()`, `uint16()`, `uint32()`, `uint64()`

整数的溢出会变成最小值和最大值：

```matlab
k = cast('hellothere', 'uint8');  % k = 104 101 108 108 111 116 104 101 114 101

double(k) + 150;  % ans = 254 251 258 261 266 254 251 264 251

k + 150;  % ans = 254 251 255 255 255 255 254 251 255 251

k - 110;  % and = 0 0 0 0 1 6 0 0 4 0
```

浮点类型：

`single()`, `double()`

```matlab
a = zeros(1, 5, 'single')

d = cast(6:-1:0, 'single')  % 转换单精度与双精度
```

单精度与双精度浮点数之间的运算结果是单精度浮点数。

判断是否为`nan`或`inf`：`isnan()`, `isinf()`。在 matlab 中，不同的`NaN`互不相等。不可以使用`a == nan`判断。

找到为`1`或`true`的索引：`find()`

查看维度：`size()`，查看长度：`length()`（行数或列数的最大值），元素的总数：`numel()`

```matlab
i = find(isnan(a));
a(i) = zeros(size(i));  % changes NaN into zeros
```

测试一个数组是否为空数组：`isempty()`

复数：

```matlab
a = 1 + 2i;
complex(2, 4)
```

常见的与复数相关的函数：

```matlab
conj(c)  % 计算 c 的共轭复数
real(c)  % 返回复数 c 的实部
imag(c)  % 返回复数 c 的虚部
isreal(c)  % 如果数组中全是实数，则返回 1，否则返回 0
abs(c)  % 返回复数 c 的模
angle(c)  % 返回复数 c 的幅角
```

与数据类型相关的函数：

```matlab
double
single
int8, int16, int32, int64
uint8, uint16, uint32, uint64
isnumeric
isinteger
isfloat
isa(x, 'type')  % 其中 type 可以是 'numeric', `integer` 和 'float'，当 x 的类型为 type 时，返回 true
cast(x, 'type')  % 将 x 类型置为 type
intmin('type')  % type 类型的最小整数值
realmax('type')  % type 类型的最大浮点实数值
realmin('type')  % type 类型的最小浮点实数值
eps('type')  % type 数据类型的 eps 值（浮点值）
eps(x)  % x 的 eps 值
zeros(..., 'type')
ones(..., 'type')
eye(..., 'type')
```

常用的初等函数：

```matlab
sin
sind  % 正弦，输入以度为单位
sinh  % 双曲正弦
asin  % 反正弦
asind
asinh
cos
cosd
cosh
acos
acosd
acosh
tan
tand
tanh
atan
atand
atan2  % 四象限反正切
sec  % 正割
secd
asec  % 反正割
asecd
asech
csc  % 余割
cscd
csch  % 双曲余割
acsc  % 反余割
acscd
acsch
cot  % 余切
cotd
coth
acot  % 反余切
acotd
acoth

exp
expm1  % 准确计算 exp(x)-1 的值
log
log1p  % 准确计算 log(1+x) 的值
log10
log2
realpow  % 对数，若结果是复数则报错
reallog  % 自然对数，若输入不是正数则报错
realsqrt  % 开平方根，若输入不是正数则报错
sqrt
nthroot  % 求 x 的 n 次方根
nextpow2  % 返回满足 2^P >= abs(N) 的最小正整数 P，其中 N 为输入

fix  % 向零取整
floor  % 向负无穷方向取整
ceil  % 向正无穷方向取整
round  % 四舍五入
mod  % 除法求余（与除数同号）
rem  % 除法求余（与被除数同号）
sign  % 符号函数
```

关系运算符：

`<`, `<=`, `>`, `>=`, `==`, `~=`

逻辑运算符：`&`, `|`, `~`

关系函数和逻辑函数：

```matlab
xor(x, y)  % 异或
any(x)  % 若 x 时向量，有任意一个不为 0 时返回 true，否则返回 false。若 x 是数组，则对于 x 的任意一列，如果有一个元素不为 0，则在该列返回 true，否则返回 false。
all(x)

ismember  % 检测一个值是否是某个集合中的元素
isglobal  % 检测一个变量是否是全局变量
mislocked  % 检测一个 M 文件是否被锁定
isempty  % 判断数组是否为空
isequal  % 检测两个数组是否相等
isequalwithequalNaN  % 检测两个数组是否相等，若数组中含有 NaN，则认为所有的 NaN 是相等的
isfinte  % 检测数组中各元素是否为有限值
isfloatpt  % 检测数组中各元素是否为浮点值
isscalar  % 检测一个变量是否为标量
isinf  % 检测数组中各元素是否为无穷大
islogical  % 检测一个数组是否为逻辑数组
isnan
isnumeric
isreal
isprime  % 检测一个数是否为素数
issorted  % 检测一个数组是否接序排列
automesh  % 如果输入参数是不同方向的向量，则返回 true
inpolygon  % 检测一个点是否位于一个多边形区域内
isvarname  % 检测一个变量名是否是一个合法的变量名
iskeyword  % 检测一个变量名是否是 matlab 的关键词或保留字
issparse  % 检测一个矩阵是否为稀疏矩阵
isvector  % 检测一个数组是否是一个向量
isappdata  % 检测应用程序定义的数据是否存在
ishandle  % 检测是否为图形句柄
ishold  % 检测一个图形是否为 hold 状态
figflag  % 检测一个图形是否是当前屏幕上显示的图形
iscellstr % 检测一个数组是否为字符串单元数组
ischar  % 检测一个数组是否为字符串数组
isletter  % 检测一个字符是否是英文字母
isspace  % 检测一个字符是否是空格字符
isa  % 检测一个对象是否为指定类型
iscell  % 检测一个数组是否为单元数组
isfield  % 检测一个名称是否是结构体中的域
isjava  % 检测一个数组是否是 java 对象数组
isobject  % 检测一个名称是否是一个对象
isstrcut  % 检测一个名称是否是一个结构体
isvalid  % 检测一个对象是否可以连接到硬件的串行端口对象
```

## 矩阵运算

**向量生成**

1. 直接创建

    ```matlab
    a = [1 2 3]  % shape: (1, 3)
    a = [1, 2, 3]  % shape: (1, 3)
    a = [1; 2; 3]  % shape: (3, 1)
    ```

    matlab 里没有纯量和一维向量，只有维度从二起始的矩阵。为了方便，这里把 shape 为`(1, n)`或`(n, 1)`的矩阵称为向量，把 shape 为`(1, 1)`的矩阵称为数字或纯量。把 shape 为`(m, n)`的二维矩阵或三维以上的矩阵称为矩阵或张量。

    可以用`size(a)`查看矩阵`a`的 shape。

1. `linspace`：线性向量生成

    `linspace(x1, x2)`默认生成 100 个点，也可以用`linspace(x1, x2, n)`指定生成的采样点数量。

    `logspace`：等比数列生成

1. `start:step:end`按间隔生成向量

    `start:end`按步长为 1 生成向量，比如`1:5`生成向量`[1 2 3 4 5]`，注意`end`是包括在区间内的。

    `start:step:end`可以按指定步长生成向量。

转置：`a'`或`transpose(a)`。这两种方法只适用二维数组（包含向量），如果维度超过二维，那么会报错。

可以使用`a([2, 3, 4])`进行多元素索引，也可以使用`a(1:2:10)`这样的方式。

常用的二维数组生成函数：

```matlab
zeros(2, 4)
ones(2, 4)
randn('state', 0)  % 把正态随机发生器置 0
randn(2, 3)  % 产生正态随机矩阵
D = eye(3)  % 产生 3 x 3 的单位矩阵
diag(D)  % 取 D 矩阵的对角元
diag(diag(D))  % 外 diag() 利用一维数组生成对角矩阵
randsrc(3, 10, [-3, -1, 1, 3], 1)  % 在[-3, -1, 1, 3]中产生 3 x 10 的均匀分布随机数组，随机发生器的状态设置为 1
```

数组寻址：

```matlab
a = [1, 3, 4, 5, 6, 7]
a(5)
a([1,3,5])  % 访问多个元素
a(1:2:5)  % 访问多个元素
a(find(x>3))  % 按条件访问多个元素

a = zeros(2, 6)
a(:) = 1:12  % 按照一维方式访问
a(2, 4)
a(8)
a(:, [1,3])
a([1, 2, 5, 6]')
a(:, 4:end)
a(2, 1:2:5)
a([1, 2, 2, 2], [1, 3, 5])
```

使用`sub2ind()`计算二维索引在拉伸为一维的数组中的索引：

```matlab
b = sub2ind(size(a), [2, 3], [2, 1])
a(b)
```

排序：

```matlab
a = rand(1, 10)
b = sort(a)
[b, index] = sort(a)
```

排序二维数组时可以指定维度：

```matlab
[b, index] = sort(A, dim, mode)  % 其中 mode 可以取 'ascend', 'descend'
```

默认情况下，matlab 会对`dim=1`维度进行排序，而 numpy 中的`sort`则会对`axis=-1`维度进行排序。matlab 与 numpy 的相同点是，第二个维度指的都是行，第一个维度指的都是列。

数组检测：

* `isempty()`

    用于检测某个数组是否为空数组。

    ```matlab
    TF = isempty(A)
    ```

* `isscalar()`

    检测某个数组是否为单元素的标量数组。

    ```matlab
    TF = isscalar(A)
    ```

* `isvector()`

    检测某个数组是否为只有一行元素或一列元素。

* `issparse()`

    检测某个数组是否为稀疏数组。

数组结构：

* `length(A)`

    一个数组的行数和列数的最大值。

* `numel(A)`

    数组元素总数。

* `[a, b] = size(A)`：数组的行数和列数。

直接使用`a * b`做的是矩阵乘法，想做逐元素相乘可以用`a .* b`。

逻辑运算：

```matlab
A & B
and(A, B)  % 若两个数均非 0 值，则结果为 1

A | B
or(A, B)  % 若两个数有一个不为 0，则结果为 1

~A
not(A)  % 若待运算矩阵的元素为 0，则结果元素为 1

xor(A, B)  % 若一个为 0，一个不为 0，则结果为 1
```

数组常用的运算：

```matlab
cumsum(a) 
sum(a)
dot(a, b)
cross(a, b)  % 叉乘运算
prod(a)
cumprod(a)
triu(a, k)  % 提取上三角矩阵
tril(a, k)
flipud()  % 矩阵翻转
fliplr()
rot90()
```

矩阵扩展：

```matlab
% 直接使用索引扩展
a = reshape(1:9, 3, 3)
a(5, 5) = 111
a(:, 6) = 222
aa = a(:, [1:6, 1:6])

% 使用分号扩展行
b = ones(2, 6)
ab_r = [a; b]

% 使用逗号扩展列
ab_c = [a, b(:, 1:5)']
```

高维数组：

```matlab
% 直接创建
A = zeros(2, 3)
% 使用索引创建
A(:, :, 4) = zeros(2, 3)
```

* `reshape`

    syntax:

    * `reshape(A, sz)`
    * `reshape(A, sz1, ..., szN)`

* `repmat`

    对于一个矩阵`A`，`repmat`可以将其维度进行重复。对于一个向量`A`，`repmat`可以先为其增加一个维度，然后再按矩阵做重复。对于一个纯量`A`，`repmat`可以构建具有重复元素的数组。

    其实无论输入的是矩阵，向量还是纯量，`repmat`都是先把其变换成至少两个维度的矩阵，然后再处理。shape 为`(n, )`的一维向量，会先变成`(1, d)`，shape 为`(1, )`的纯量，会先变成`(1, 1)`。

    `repmat`的处理方式很奇怪，如果待处理矩阵`A`后只有一个参数`r1`，那么会把`A`的前两个维度翻`r1`倍。如果`A`后有大于 1 个参数`(r1, r2, ....)`，`repmat`则会按对应位置对维度进行翻倍。

    syntax:

    * `repmat(A, n)`
    * `repmat(A, r1, ..., rN)`
    * `repmat(A, r)`

    其中`r1`，...，`rN`表示在这些维度上重复几遍。

    Examples:

    ```matlab
    a = 1
    b = zeros(3)
    ```

* `cat`

    syntax:

    * `C = cat(dim, A, B)`
    * `C = cat(dim, A1, A2, ..., An)`

* `squeeze`

    syntax:

    * `squeeze(A)`

    删除矩阵中大小为 1 的维度。
    
    具体实现的话不同情况挺复杂的，看例子理解吧。

    Examples:

    ```matlab
    a = 1
    size(squeeze(a))  % (1, 1)

    b = zeros(1, 3)
    size(squeeze(b))  % (1, 3)

    c = zeros(1, 3, 1)
    size(squeeze(c))  % (1, 3)

    d = zeros(1, 3, 1, 2)
    size(squeeze(d))  % (3, 2)
    ```

* `sub2ind`

    syntax:

    * `ind = sub2ind(sz, row, col)`
    * `ind = sub2ind(sz, I1, I2, ..., In)`

    将多维索引拉伸成一维索引。

    `ind2sub()`

* `permute`

    syntax:

    * `B = permute(A, dimorder)`

    重排维度。相当于 numpy 中的`transpose`。

    `ipermute()`

* `size`

    syntax:

    * `sz = size(A)`
    * `szdim = size(A, dim)`
    * `szdim = size(A, dim1, dim2, ..., dimN)`
    * `[sz1, ..., szN] = size(___)`

* `N = ndims(A)`

    获取维度的数量

* 其他

    * `cat()`
    * `flipdim()`
    * `shiftdim()`

可以使用赋空操作删除某一行或列：

```matlab
A(:, 2:3, :) = []
```

## 字符串

字符串是一个`1 x n`的`char`型数组，每个字符占 2 字节。

```matlab
str = 'I am a great person ';  % 使用单引号直接创建

str = ['second'; 'string']  % 字符串数组，要求字符串长度必须一致

c = char('first', 'second')  % 使用 char 创建字符串数组时，如果字符串长度不同，char() 会自动在较短的字符串后加空格，使所有字符串长度相等
```

常用函数：

* `c = strcat(a, b)`

    删去字符串末尾的空格`' '`后进行拼接

* `c = [a b]`

    不删除空格进行字符串拼接

* `c = strvcat('name', 'string')`

    与`char()`作用类似。

* `celldata = cellstr(c)`

    删除字符串末尾的空格，然后将字符串数组转换为字符串单元数组。

* `chararray = char(celldata)`

    把一个字符串单元数组转换成一个字符数组。

* `new_str = deblank(str)`

    删除字符串末尾的空格并返回。

* 字符串比较

    `strcmp()`, `strcmpi()`, `strncmp()`, `strncmpi()`

    可以使用关系运算符对单个字符进行比较。

* `isletter()`

    判断字符串的每个字符是否为一个字母。

* `isspace()`

    判断字符串中的每个字符是否为空白字符（空格，制表符，换行符）

* 查找和替换

    * `str = strrep(str1, str2, str3)`，把`str1`中的`str2`替换成`str3`

    * `k = findstr(str1, str2)`，查找输入中较长字符串中较短字符串的位置

    * `k = strfind(str, pattern)`，查找`str`中`pattern`出现的位置

    * `k = strfind(cellstr, pattern)`，查找单元字符串`cellstr`中`pattern`出现的位置

    * `strtok`：获得第一个分隔符之前的字符串

        `token = strtok('str')`，以空格符作为分隔符

        `token = strtok('str', delimiter)`，指定分隔符

        `[token, rem] = strtok(...)`，返回值`rem`为第1个分隔符之后的字符串（包含分隔符）

    * `strmatch`：在字符串数组中匹配指定的字符串

        * `x = strmatch('str', STRS)`，在字符串数组`STRS`中匹配字符串`str`，返回匹配上的字符串所在行的指标

        * `x = strmatch('str', STRS, 'exact')`：精确匹配，要求完全一致才算匹配上。

* `upper()`，`lower()`，把整个字符串转换大小写

* `eval()`：把字符串转换成数字

    `value = sscanf(string, format)`

    example:

    ```matlab
    v1 = sscanf('3.141593', '%g')  % 浮点数
    v2 = sscanf('3.141593', '%d')  % 整数
    ```

* 数字转换成字符串

    `num2str()`, `int2str()`, `dex2hex()`，`hex2num()`，`hex2dec()`，`bin2dex()`，`dec2bin()`，`base2dec()`, `dec2base`

    `mat2str()`可以把一个数组转换成字符串。

    `sprintf()`，`fprintf()`，格式化字符串。

    `char()`可以按 ascii 码将数字转换成字符。

* `str2num`, `uintN`, `str2double`, `hex2num`, `hex2dec`, `bin2dec`, `base2dec`

* `double`

    把字符串转换成 ascii 形式

* `blanks(n)`, `evalc(s)`

* 其他的一些函数

    `isstrprop()`, `strtrim()`, `strjust()`

正则表达式：

* `regexp(str, pattern)`

    `regexpi()`

    `regexprep()`：使用正则表达式替换字符串

    查找单个字符的字符串表达式：

    ```matlab
    .  % 任意单个字符
    [abcd35]  % 查找方框中任意一个字符
    [a-zA-Z]  % 查找指定范围字母
    [^aeiou]  % 取反
    \s  % 任意空白符
    \S  % 任意非空白符
    \w  % 任意文字字符，字母，数字，下划线
    \W  % 任意非文字字符，相当于[^a-zA-Z_0-9]
    \d  % 任意数字，相当于[0-9]
    \D  % 任意非数字字符，[^0-9]
    \xN 或 \x{N}  % 查找十六进制为 N 的字符
    \oN 或 \o{N}  % 查找八进制为 N 的字符
    \a  % 查找警告
    \b  % 退格 
    \t  % 横向制表符
    \n  % 换行符
    \v  % 纵向制表符
    \f  % 换页符
    \r  % 回车符
    \e  % 退出符
    \., \*, \?, \\ 等  % 转义字符

    (p)  % 限制后面修饰符的作用范围
    (?:p)  % 不懂 
    (?>p)  % 不懂
    (?#A Comment)  % 插入注释
    \N  % 与该表达式中第 N 个标记相同，不懂
    $N  % 不懂
    (?<name>p)  % 
    \k<name>  %
    (?(T)p)  %  if then 结构
    (?(T)p|q)  % if then else 结构
    ```

## 结构体（structures）

创建：

```matlab
a.b = 1
a.c = [1, 2, 3]

s = struct('Name', 'John', 'Score', 85.5, 'Salary', [4500 4200])

repmat(struct('Name', 'John', 'Score', 85.5, 'Salary', [4500, 4200]), 1, 3)

struct('Name', {'klj', 'Dana', 'John'}, 'Score', {98, 92, 85.5}, 'Salary', {[4500 4200], [], []})
```

将属性作为数组返回：

```matlab
[obj.attr]
{obj.attr}
```

## 单元数组（cell array）

cell array 可看成 Python 中保持矩阵形状的 list，可以存储任何类型的值。

cell 通过直接赋值的方式创建：

```matlab
a = {1, [1, 2]; 3, 'str'};

a(1, 1) = {[1, 2, 3]};
a(1, 2) = {'abc'};
a(2, 1) = {1};
```

注意，此时`a`会变成 size 为`(4, 4)`的 cell，其中`a(2, 2)`是 size 为`(0, 0)`的矩阵。

正是因为对 cell 矩阵化，所以大部分对维度的操作对 cell 同样也适用。

如果使用花括号进行索引，那么等号右侧的值就不需要再加花括号了：

```matlab
a{1, 1} = [1, 2, 3]
```

可以使用`celldisp(A)`完整地显示一个 cell 中的内容。

cell array，使用`()`进行索引，只能得到一个 cell，而使用`{}`进行索引，可以得到 cell 中的内容。

可以使用`deal()`取多个单元元素的内容。

## 语句

`if`语句：

```matlab
if condition_1
    statement_1
elseif condition_2
    statement_2
else
    statement_3
end
```

对于`condition`，当其为空数组`[]`，空字符串`''`或全零矩阵（`0`，`[0, 0]`，`[0 0; 0 0]`等）时，`condition`为假，其余情况全为真。空 cell 无法判断真假，会报错。

`matlab`对缩进和换行没有要求，因此`if`必须以`end`结尾。

如果不想在`if`后空格，也可以使用`if(expr)`的形式。

`switch`语句：

```matlab
switch expr
    case {val1, val2, val3},
        statement_1
    case val4,
        statement_2
    otherwise,
        statement_3
end
```

`while`语句：

```matlab
while expr
    statements
end
```

`for`语句：

```matlab
for index = expr
    statements
end
```

`expr`会首先被 reshape 成`(d1, n)`的二维数组，然后按`expr(:, 1)`，`expr(:, 2)`，`...`的方式赋值给`index`。

```matlab
try
    % ...
catch
    % ...
end
```

**与脚本文件相关的函数**

* `beep`：蜂鸣器
* `disp(variable_name)`：只显示结果，不显示变量名
* `echo`：控制 M 脚本文件的内容是否在 command 窗口中显示
* `input`：提示用户输入数据
* `keyboard`：临时终止 M 脚本文件的执行，让键盘获得控制权。
* `pause`，`pause(n)`：暂停，等待用户按键。或暂停 n 秒后继续执行
* `waitforbuttonpress`：暂停，直到用户按下鼠标或其他按键

特殊变量：

* `ans`
* `i(j)`：虚数单位
* `pi`
* `eps`：最小浮点数精度
* `inf`
* `NaN`
* `nargin`：函数的输入变量数目
* `nargout`：函数的输出变量数目
* `realmin`：最小可用正实数
* `realmax`：最大可用正实数

将函数中的某个变量声明为全局变量：`global var1 var2;`

计时器：

```matlab
% 使用计时器延迟启动一个脚本文件
my_timer = timer('TimerFcn', 'script_name', 'StartDelay', 100)  % 延迟 100 秒
start(my_timer)  % 100 秒后执行脚本 script_name

% 设置启动函数，终止函数等
my_timer = timer('TimerFcn', 'Mfile1', ...
    'StartFcn', 'Mfile2', ...
    'StopFcn', 'Mfile3', ...
    'ErrorFcn', 'Mfile4');
start(my_timer)  % 执行 Mfile2，然后循环执行 Mfile1
stop(my_timer)  % 执行 Mfile3
```

matlab 在启动时，会默认执行`MATLABrc.m`和`startup.m`这两个文件。

输入`exit`或`quit`可退出 matlab，在退出之前，会执行`finish.m`脚本。

## 函数

可以用`error()`终止函数执行，并返回命令行窗口。也可以使用`warning()`函数打印警告信息。

在函数内部调用的 M 脚本文件，创建的变量都作为函数的内部变量。

在函数文件中可以创建子函数。可以调用`helpwin func/subfunc`查看子函数的帮助文档。

放在某个函数子目录下的函数为私有函数，只有父目录下的主函数才能调用。

查看已经被编译在内存中的函数：`inmem()`。可以用`mlock`将某个函数锁定在内存中，此时使用`clear`命令不会清除编译好的函数。可以使用`munlock()`解除锁定，使用`mislocked()`检查一个函数是否被锁定。

使用`pcode`可以将编译好后的伪码存储在硬盘中。

matlab 会在启动时对`toolbox`目录下的所有函数做一次缓存，后续不再读取这些文件。可以使用`rehash toolbox`强制刷新缓存。也可以使用`clear`清除掉旧的缓存。

`clear functions`：清除所有未锁定的函数。

`depfunc()`可以检查一个函数与其他文件之间的关联性。

检查参数：`nargchk()`，`nargoutchk()`

* `varargin`, `varargout`

* `nargin()`, `nargout()`

`persistent`变量称为永久变量。类似 C 语言中的静态变量。

`evalin()`，`eval()`。不知道这俩有啥区别。

其他有关工作区的函数：

`assignin()`, `inputname()`

当前正在被执行的函数文件名：`mfilename`

`mlint`，`mlint()`可以检查脚本文件的语法。

## 画图

直接用`plot(y)`画图，如果`y`的 shape 为`(1, d2)`，那么会画一条曲线。如果`y`的 shape 为`(d1, d2)`，那么会画`d2`条曲线，每一列作为一条曲线的数据。

不过通常都用`plot(x, y)`画吧，这样更清晰。如果`x`的 shape 为`(1, dx2)`，`y`的 shape 为`(dy1, dy2)`，那么会画`dy1`条曲线。

剩下的用法都不怎么直观。

```matlab
plot(x, y, s)
```

其中`s`可以为：

* 颜色

    `b`：蓝色，`g`：绿色，`r`：红色，`c`：青色，`m`：洋红，`y`：黄色，`k`：黑色，`w`：白色

* 标记

    `.`, `o`, `x`, `+`, `*`, `s`（方形）, `d`（菱形）, `^`

* 线型

    `-`, `:`（点线）, `-.`, `--`, `v`（向下三角形）, `p`（五角星）, `h`（六角星）, `<`（向左三角形）

```matlab
plot(x, y, 's', 'PropertyName', PropertyValue, ...)
```

常用的属性：

* `Color`：`[r, g, b]`，取值范围为`0 ~ 1`。

* `LineStyle`：4 种线型

* `LineWidth`：正实数。默认线宽为 0.5。

* `Marker`：14 种点型

* `MarkerSize`：正实数。默认大小为 6.0。

* `MarkerEdgeColor`：`[r, g, b]`

* `MarkerFaceColor`：`[r, g, b]`

常用函数：

* `axis([xmin xmax ymin ymax])`

    设置当前图形的坐标范围

* `V = axis`

    返回包含当前坐标范围的一个行向量

* `axis auto`

    将坐标轴刻度恢复为自动的默认设置

* `axis manual`

    冻结坐标轴刻度。如果`hold`被设置为`on`，那么后面的图形将使用与前面相同的坐标轴刻度范围。

* `axis tight`

    将坐标范围设置为被绘制的数据范围

* `axis fill`

    设置坐标范围和屏幕高宽比，使坐标轴可以包含整个绘制区域。

    该选项只在`PlotBoxAspectRatio`或`DataAspectRatioMode`被设置为`manual`模式时才有效。

* `axis ij`

    将坐标轴设置为矩阵模式。此时水平坐标轴从左到右取值，垂直坐标轴从上到下取值。

* `axis xy`

    将坐标轴设置为笛卡尔模式。此时水平坐标轴从左到右取值，垂直坐标轴从下到上取值。

* `axis equal`

    设置屏幕高宽比，使每个坐标轴具有均匀的刻度间隔。

* `axis image`

    设置坐标范围，使其与被显示的图形相适应

* `axis square`

    将坐标轴框设置为正方形

* `axis normal`

    将当前的坐标轴恢复为全尺寸，并将单位刻度的所有限制取消。

* `axis vis3d`

    冻结屏幕高宽比，使一个三维对象的旋转不会改变坐标轴的刻度显示

* `axis off`, `axis on`

    关闭/打开所有的坐标轴标签、刻度和背景

可以同时给出多个`axis`命令：`axis auto on xy`

图片中文本的特殊字符：

`\alpha`, `\beta`, `\gamma`, `\delta`, `\epsilon`, `\zeta`, `\eta`, `\theta`, `\vartheta`, `\iota`, `\kappa`, `\lambda`, `\mu`, `\nu`, `\xi`, `\pi`, `\rho`, `\rfloor`, `\lfloor`, `\perp`, `\wedge`, `\rceil`, `vee`, `\langle`, `\upsilon`, `\phi`, `\chi`, `\psi`, `\omega`, `\Gamma`, `\Delta`, `\Theta`, `\Lambda`, `\Xi`, `\cong`, `\approx`, `\Re`, `\oplus`, `\cup`, `\subseteq`, `\in`, `\lceil`, `\cdot`, `\neg`, `\times`, `\surd`, `\varpi`, `\rangle`, `\sim`, `\leq`, `\infty`, `\sigma`, `\varsigma`, `\tau`, `\equiv`, `\Im`, `\otimes`, `\cap`, `\supset`, `\int`, `\circ`, `\pm`, `\geq`, `\propto`, `\partial`, `\bullet`, `\div`, `\Pi`, `\Sigma`, `\Upsilon`, `\Phi`, `\Psi`, `\Omega`, `\forall`, `\exists`, `\ni`, `\neq`, `\aleph`, `\wp`, `\oslash`, `\supseteq`, `\subset`, `\o`, `\clubsuit`, `\diamondsuit`, `\heartsuit`, `\spadesuit`, `\leftrightarrow`, `\leftarrow`, `\uparrow`, `\rightarrow`, `\downarrow`, `\nabla`, `\ldots`, `\prime`, `\0`, `\mid`, `\copyright`

字体控制：

* `\bf`：黑体
* `\it`：斜体
* `\sl`：透视
* `\rm`：标准形式
* `\fontname{fontname}`
* `\fontsize{fontsize}`

标注：

```matlab
text(x, y, 'string')
text(x, y, z, 'string')
text(...'PropertyName', PropertyValue...)

gtext('string')
gtext({'string1', 'string2', 'string3', ...})
gtext({'string1'; 'string2'; 'string3'; ...})
```

图例：

```matlab
legend('string1', 'string2', ...)
legend(..., 'Location', location)
```

其中`location`可以是一个向量`[left bottom width height]`或任意一个字符串。

常用的字符串：

`'North'`, `'East'`, `'NorthEast'`, `'SouthEast'`, `'NorthOutside'`, `'EastOutside'`, `'NorthEastOutside'`, `'SouthEastOutside'`, `'South'`, `'West'`, `'NorthWest'`, `'SouthWest'`, `'SouthOutside'`, `'WestOutside'`, `'NorthWestOutside'`, `'SouthWestOutside'`, `'BestOutside'`（绘图区外占用最小面积）, `'Best'`（标注与图形的重叠最小处）

`hold on`可不刷新继续画图，`hold off`表示刷新后画图。

双纵坐标图：

```matlab
plotyy(X1, Y1, X2, Y2)
plotyy(X1, Y1, X2, Y2, 'FUN')
plotyy(X1, Y1, X2, Y2, 'FUN1', 'FUN2')
```

多子图：

```matlab
subplot(m, n, k)
```

```matlab
fplot(function, limits)
fplot(function, limits, LineSpec)
fplot(function, limits, tol)
fplot()
```
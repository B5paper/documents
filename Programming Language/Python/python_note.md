# Python Note

## cached

* python 中的`set()`, an example:

    ```python
    def main():
        s = set()
        s.add(1)
        s.add(2)
        if 1 in s:
            print('1 is in set')
        else:
            print('1 is not in set')

        s.add('hello')
        s.add('world')
        if 'hello' in s:
            print('hello is in set')
        else:
            print('hello is not in set')

        s.add([1, 2, 3])

        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    1 is in set
    hello is in set
    Traceback (most recent call last):
      File "/home/hlc/Documents/Projects/python_test/main.py", line 22, in <module>
        main()
      File "/home/hlc/Documents/Projects/python_test/main.py", line 17, in main
        s.add([1, 2, 3])
    TypeError: unhashable type: 'list'
    ```

    可以看出来，`set()`比较像哈希表，只有 hashable 的对象才可以添加到 set 里，其他的不行。

    想判断一个对象是否在 set 里，可以使用`in`关键字。

* python 中的`os.path.samefile()`可以判断两个 path 是否相同

    ```python
    import os

    def main():
        is_same = os.path.samefile('/home/hlc/Documents/Projects/python_test', '././../python_test')
        if is_same:
            print('is same')
        else:
            print("is not same")
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    is same
    ```

    说明：

    1. `samefile()`既可以处理文件夹，也可以处理文件。并且对绝对路径和相对路径不敏感。

    2. `samefile()`要求输入的路径必须是存在的。

    3. `ln -s`创建的软链接和原文件/目录被会`samefile()`判定为同一文件/目录。

* pip 更新一个包： `pip install <package> --upgrade`

* python 可以使用`os.path`处理和路径相关的字符串

    ```python
    import os

    def main():
        path_1 = './hello'
        path_2 = 'world'
        path = os.path.join(path_1, path_2)
        print(path)

        path_1 = './hello/'
        path_2 = './world'
        paht = os.path.join(path_1, path_2)
        print(path)

        path_1 = os.path.abspath('./hello')
        path_2 = 'world'
        path = os.path.join(path_1, path_2)
        print(path)

        path_1 = './hello'
        path_2 = '../hello/world'
        path = os.path.join(path_1, path_2)
        print(path)
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    ./hello/world
    ./hello/world
    /home/hlc/Documents/Projects/python_test/hello/world
    ./hello/../hello/world/
    ```

    可以看到，`os.path`可以妥善处理`path_1`结尾的`/`，也可以妥善处理`path_2`开头的`./`，但是不能处理`../`。

    `os.path.abspath()`可以将一个相对路径转换为绝对路径。`os.path.relpath()`可以将一个绝对路径转换为相对当前目录的相对路径。`relpath()`的第二个参数可以指定起始路径的前缀，这个前缀可以是相对路径（相对于当前目录），也可以是绝对路径。

    `os.path.join()`还支持可变参数：

    ```python
    import os

    def main():
        path_1 = 'path_1'
        path_2 = 'path_2'
        path_3 = 'path_3'
        path = os.path.join(path_1, path_2, path_3)
        print(path)
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    path_1/path_2/path_3
    ```

    看到有人提到`pathlib`这个 package，似乎是专门拿来解决这个问题的。回头调研一下。

* `os.walk()`会递归遍历指定目录下的每一个文件夹

    ```py
    for cur_dir, dirnames, filenames in os.walk(root_path):
        print('cur_dir: ', cur_dir)
        print('dirnames: ', dirnames)
        print('filenames: ', filenames)
    ```

* python 的`rstrip()`不会做 in-place 修改，需要赋值才能修改

* python 中的`re`似乎不认为`\A`是一个字符

    因为`re.compile(r'(?<=\n|\A)\[.*\](.|\n)*?(?=\[.*\]\n|\Z)')`会报错：

    `look-behind requires fixed-width pattern`

    这样只能把写法改成

    `re.compile(r'((?<=\n)|(?<=\A))\[.*\](.|\n)*?(?=\[.*\]\n|\Z)')`才能正常运行。

* python hash

	直接用`hash()`函数就可以计算出各个 python 内置对象的哈希值。

	example:

	```py
	a = 3
	s = 'hello, world'
	print(hash(a))
	print(hash(s))
	```

	output:

	```
	3
	1966604262258436456
	```

	每次运行程序，即使对相同的字符串，哈希值也不同。不清楚为什么。

* python 获取内核时间

	```python
	import time
	time.process_time()
	time.thread_time()
	```

	这两个函数可以返回浮点数作为时间。经过测试，这俩函数的返回值基本都是递增的。可以放心用。

* python 常用的 format 语法

	```python
	txt1 = "My name is {fname}, I'm {age}".format(fname = "John", age = 36)
	txt2 = "My name is {0}, I'm {1}".format("John",36)
	txt3 = "My name is {}, I'm {}".format("John",36) 
	```

* python use `shutil` to copy file

    ```cpp
    import shutil

    def main():
        shutil.copyfile('./test_1.txt', './test_2.txt')

    if __name__ == '__main__':
        main()
    ```

    * <https://stackoverflow.com/questions/123198/how-to-copy-files>

    * <https://www.freecodecamp.org/news/python-copy-file-copying-files-to-another-directory/>

* python delete a file

    <https://blog.enterprisedna.co/delete-files-from-python/>

## pypi mirror

在上海使用上交的镜像比较快：<https://mirrors.sjtug.sjtu.edu.cn/docs/pypi/web/simple>

临时使用：`pip install -i https://mirror.sjtu.edu.cn/pypi/web/simple numpy`

## regular expression

### cache

* 如果一个字符串后面有很多`\n`，但是想清除多余的换行，只保留一个，可以用下面的正则表达式：

    `.*?\n(?=\n*)`

    比如匹配字符串`aaabb\n\n\n\n`，它的匹配结果是`aaabb\n`。

    这个情形常用于匹配文件里有许多空行，比如

    ```
    [config_1]
    aaa
    bbb



    [config_2]
    ccc
    ```

    这两个 config 之间的空行太多，可以用正则表达式只匹配一个换行。

    （潜在问题：如果最后一行只有`\Z`，没有`\n`，没办法匹配到，该怎么办）

* python 的 lambda 表达式中不能有`return`，最后一行的表达式就是返回值

    比如`lambda x: True if x == 1 else False`，这个函数的返回值类型就是`bool`。

* python 中使用`re`模块时，为了避免在 python 字符串的规则处理，通常需要加一个`r`：

    `re_pats['pat_unit'] = re.compile(r'\[unit\](.|\n)*?(?=\[unit\]|\Z)')`

    如果不加`r`，会运行时报错：

    ```
    /home/hlc/Documents/Projects/stochastic_exam_py/main.py:22: SyntaxWarning: invalid escape sequence '\['
    re_pats['pat_unit'] = re.compile('\[unit\](.|\n)*?(?=\[unit\]|\Z)')
    ```

* python 正则表达式中有关汉字的处理

	一个匹配表达式是：

	```python
	patstr_hanzi = r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?'
	```

	其他的匹配方法可以参考这个回答：<https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex>

* python 正则表达式中，方括号`[]`里不能有点号`.`，只能有`a-z`，数字，标点符号之类的。

	点号`.`可以匹配除了`\n`之外的任意一个字符。如果想匹配包括`\n`在内的所有字符，可以使用`(.|\n)`，用括号和或运算将这两个结合起来。

* python 正则中，可以使用`\A`匹配字符串的开头，使用`\Z`匹配末尾。

* python 正则表达式中，空格不需要转义

	比如使用`(.+), (.+)`去匹配`hello, world`，得到的 group 1 为`hello`，group 2 为`world`，空格被正确匹配了。

### search and match

example:

```python
import re
txt = 'hello, world'
pat_1 = re.compile('world')
m = pat_1.search(txt)
start_pos = m.start()
end_pos = m.end()
selected_txt = txt[start_pos:end_pos]
print(selected_txt)  # world
```

python 中使用正则表达式可以使用`re`模块，其中`re.compile()`表示将正则表达式编译成一段小程序（应该是转换成有限状态机）。

`pat_1.search()`表示从指定位置开始匹配，返回一个`Match`对象，`Match`对象保存了匹配结果，包括开始和结尾位置，group 情况之类的。

`search()`区别于`match()`，`match()`表示从头开始匹配。

### finditer

example 1:

```python
txt = 'abcbacaccba'
pat_2 = re.compile('a.{2}')
for m in pat_2.finditer(txt):
	start_pos = m.start()
	end_pos = m.end()
	selected_txt = txt[start_pos:end_pos]
	print(selected_txt)  # [abc, aca]
```

这个例子中，使用`pat_2.finditer()`

example 2:

```python
txt = \
'''
[unit]
hello
world
[unit]
hehe
haha
'''
pat_3 = re.compile('\[unit\](.|\n)*?(?=\[unit\]|\Z)')
for m in pat_3.finditer(txt):
	start_pos = m.start()
	end_pos = m.end()
	selected_txt = txt[start_pos:end_pos]
	print(selected_txt)
```

output:

```
[unit]
hello
world

[unit]
hehe
haha
```

其中`(?=...)`表示匹配括号中的表达式，但是不选中。这个操作叫 forward lookahead。

`*?`表示最近匹配，在所有符合条件的表达式中，找到最短的。

可以使用这个网站对正则表达式 debug: <https://regex101.com/>

目前不清楚`findall()`怎么个用法。

### group

```python
import re

string = 'hello, world'
patstr = '(.+), (.+)'
pat = re.compile(patstr)
m = pat.search(string)

print('-------- test 1 --------')
g0 = m.group(0)
print(g0)
g1 = m.group(1)
print(g1)
g2 = m.group(2)
print(g2)

print('-------- test 2 --------')
g1, g2 = m.groups()
print(g1)
print(g2)

print('-------- test 3 --------')
m = re.match(r"(?P<first_name>\w+) (?P<last_name>\w+)", "Malcolm Reynolds")
g_first_name = m.group('first_name')
g_last_name = m.group('last_name')
print(g_first_name)
print(g_last_name)
d = m.groupdict()
print(d['first_name'])
print(d['last_name'])
```

每个使用`()`括起来的表达式可以被 group 捕捉。

`group(0)`是整个表达式，`group(1)`是第一个括号对应的字符串，`group(2)`是第二个括号对应的字符串。

`groups()`以 tuple 的形式给出`group()`的结果。注意这里索引是从 1 开始的。

使用`(?P<var_name>...)`可以为子匹配命名，然后使用`group('<name>')`获得。

`groupdict()`以字典的形式返回命名匹配。如果表达式中没有命名子匹配，那么字典为空。

## Miscellaneous

1. 播放 mp3 文件时，`playsound`库不好用，在 windows 下会出现无法解码 gb2312 的问题。可以用`vlc`库代替。但是`vlc`似乎不支持阻塞式播放。

1. 一个文件作为模块运行时，才能相对导入，比如`from ..package.module import some_class`。
    
    让一个文件作为模块运行有两种方法，一种是运行其他 python 文件，让其他 python 文件把这个文件作为模块或包导入；另一种是直接使用`python -m xxx.py`运行当前文件。

    相对导入也是有极限的，那就是它只能把主脚本所在的目录作为顶级包，无法再向上查找。或者说，它只能找到`__name__`中指向的顶级包。

    假如一个工程项目`proj`目录，里面有`subpack_1`和`subpack_2`两个子包，然后`subpack_1`中有一个模块文件`mod_1.py`，`subpack_2`中有一个模块文件`mod_2.py`。想直接从`mod_1`直接调用`mod_2`是不可能的。要想调用，只有一种办法，那就是在`proj`下创建一个新文件`script.py`，然后在这个文件中，使用

    ```py
    import sys
    sys.path.append('./')

    from subpack_1 import mod_1
    ```

    把当前目录加入到搜索目录中，然后再在这个文件中运行`mod_1`中的代码。

    不加`sys.path.append('./')`是不行的，因为我们直接运行的是`script.py`，所以`proj`目录被作为顶层目录。然而顶层目录并不会被作为一个包，因此`mod_1`向上找最多只能找到`subpack_1`这里，而无法看到`subpack_2`。为了让`mod_1`看到`subpack_2`，还需要将当前目录加入到搜索目录中。

1. 将 c++ 文件编译为`.pyd`文件，获取当前系统的后缀的方法：

    * linux: `python3-config --extension-suffix`

    * windows: `python -c "from distutils import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`

1. python 3.1 之后，不再有`unichr()`函数，直接使用`chr()`函数就可以了。把一个整数转换成一个 Unicode 字符。

1. `PYTHONPATH`的作用

    `PYTHONPATH`中的内容会被添加到`sys.path`中，使用冒号`:`分隔不同路径（如果是 windows 系统怎么办？比如`C:/path_1:D:/path_2`这样的）。

    我们使用`import <package_name>`时，会在`sys.path`路径下搜索。

    Ref: <https://www.simplilearn.com/tutorials/python-tutorial/python-path>

1. 有关`.pyd`文件

    <https://stackoverflow.com/questions/50278029/how-to-import-a-pyd-file-as-a-python-module>

    这个资料里说 pyd 其实就是 dll，没法直接 import。

    实际上，可以使用`importlib`这个库导入 pyd 文件。有时间了研究一下。

1. python 中 dict 类型对象的`.copy()`方法是 shallow copy

    ```python
    d = {}
    d['hello'] = 'world'
    d['1'] = '2'

    d_1 = d  # 只是创建了一个 d 的引用，并没有做值拷贝
    d_1['1'] = '3'  # 改变 d_1 的内容会影响 d 的内容
    print(d['1'])  # 3

    d['1'] = '2'
    d_2 = d.copy()  # shallow copy
    d_2['1'] = '3'  # 改变 d_2 的内容不再影响 d
    print(d['1'])  # 2

    from copy import deepcopy
    my_data = [1, 2, 3]
    d['1'] = my_data  # d['1'] 是 my_data 的一个引用，并不拥有自己的数据
    d_3 = d  # d_3 是 d 的一个引用，因此和 d 等价
    d_4 = d.copy()  # d_4 是 d 的一个浅拷贝，因此 d_4['1'] = xxx 不影响 d['1']，但是 d_1['1'][xx] = xxx 会影响 d
    d_5 = deepcopy(d)  # d_5 与 d 完全互不影响
    d_3['1'][0] = 2  # 影响 my_data 中的内容
    d_4['1'][1] = 3  # 影响 my_data 中的内容
    d_5['1'][2] = 4  # 不影响 my_data 中的内容
    print(d['1'])  # [2, 3, 3]
    ```
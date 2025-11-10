# Python Note

## cached

* py 中显示一个 obj 的所有静态 attr

    ```py
    class Obj:
        val_1: int = 123
        def __init__(self):
            self.val_2 = 456
            return

    obj = Obj()
    obj.val_3 = 789

    for attr in dir(obj):
        print('attr: {}'.format(attr))
    ```

    output:

    ```
    attr: __annotations__
    attr: __class__
    attr: __delattr__
    attr: __dict__
    attr: __dir__
    attr: __doc__
    attr: __eq__
    attr: __format__
    attr: __ge__
    attr: __getattribute__
    attr: __getstate__
    attr: __gt__
    attr: __hash__
    attr: __init__
    attr: __init_subclass__
    attr: __le__
    attr: __lt__
    attr: __module__
    attr: __ne__
    attr: __new__
    attr: __reduce__
    attr: __reduce_ex__
    attr: __repr__
    attr: __setattr__
    attr: __sizeof__
    attr: __str__
    attr: __subclasshook__
    attr: __weakref__
    attr: val_1
    attr: val_2
    attr: val_3
    ```

    这里显示的 attr 都是`str`类型。

* `os.walk()`

    递归地遍历指定目录及其所有子目录。

    syntax:

    ```py
    os.walk(top, topdown=True, onerror=None, followlinks=False)
    ```

    返回值

    生成一个三元组 (root, dirs, files)：

    * root: 当前正在遍历的目录路径

    * dirs: 当前目录下的子目录列表

    * files: 当前目录下的文件列表

    example:

    ```py
    import os

    # 基本遍历
    for root, dirs, files in os.walk('.'):
        print(f"当前目录: {root}")
        print(f"子目录: {dirs}")
        print(f"文件: {files}")
        print("-" * 50)
    ```

    参数说明

    * topdown=True: 从上往下遍历（先父目录后子目录）

    * topdown=False: 从下往上遍历（先子目录后父目录）

    * onerror: 错误处理函数

    * followlinks: 是否跟随符号链接

        默认不跟随符号链接，避免无限循环

* 使用 venv 创建 python 虚拟环境

    ```py
    python3 -m venv myenv      # 创建虚拟环境
    source myenv/bin/activate # 激活虚拟环境
    ```

* python datetime 格式化打印当前日期

    ```py
    import datetime
    cur_dt = datetime.datetime.now()
    print(cur_dt)
    formatted_str = cur_dt.strftime("%Y/%m/%d %H:%M:%S")
    print(formatted_str)
    ```

    output:

    ```
    2025-10-31 15:08:58.421751
    2025/10/31 15:08:58
    ```

* python 删除文件

    python 可以使用`os.remove()`删除文件，但是`os.remove()`如果删除成功，不会有提示，如果删除失败，会报 exception。因此我们使用 try 来判断文件是否删除成功。

    ```py
    import os

    def remove_file(file_path):
        try:
            os.remove(file_path)
            print(f"文件 {file_path} 删除成功")
            return True
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在")
            return False
        except PermissionError:
            print(f"没有权限删除文件 {file_path}")
            return False
        except OSError as e:
            print(f"删除文件时出错：{e}")
            return False

    # 使用示例
    success = remove_file("to_delete.txt")
    if success:
        print("删除操作成功完成")
    else:
        print("删除操作失败")
    ```

    output:

    ```
    文件 to_delete.txt 删除成功
    删除操作成功完成
    ```

* 对 python 中的 list 进行 unique

    1. 使用 set()（最常用）

        ```py
        my_list = [1, 2, 2, 3, 4, 4, 5]
        unique_list = list(set(my_list))
        print(unique_list)  # 输出：[1, 2, 3, 4, 5]
        ```

        注意：这种方法会打乱原列表的顺序。

    2. 使用 dict.fromkeys()（保持顺序）

        ```py
        my_list = [1, 2, 2, 3, 4, 4, 5]
        unique_list = list(dict.fromkeys(my_list))
        print(unique_list)  # 输出：[1, 2, 3, 4, 5]
        ```

    3. 使用循环（保持顺序）

        ```py
        my_list = [1, 2, 2, 3, 4, 4, 5]
        unique_list = []
        for item in my_list:
            if item not in unique_list:
                unique_list.append(item)
        print(unique_list)  # 输出：[1, 2, 3, 4, 5]
        ```

    4. 使用列表推导式（保持顺序）

        ```py
        my_list = [1, 2, 2, 3, 4, 4, 5]
        unique_list = []
        [unique_list.append(x) for x in my_list if x not in unique_list]
        print(unique_list)  # 输出：[1, 2, 3, 4, 5]
        ```

    5. 使用 collections.OrderedDict（保持顺序）

        ```py
        from collections import OrderedDict
        my_list = [1, 2, 2, 3, 4, 4, 5]
        unique_list = list(OrderedDict.fromkeys(my_list))
        print(unique_list)  # 输出：[1, 2, 3, 4, 5]
        ```

    6. 使用 pandas（适用于复杂数据结构）

        ```py
        import pandas as pd
        my_list = [1, 2, 2, 3, 4, 4, 5]
        unique_list = pd.Series(my_list).drop_duplicates().tolist()
        print(unique_list)  # 输出：[1, 2, 3, 4, 5]
        ```

    性能比较：

    * 最快：set()（但不保持顺序）

    * 保持顺序且较快：dict.fromkeys()

    * 最慢：循环方法

* py 中实现 enum

    ```py
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    # 使用
    print(Color.RED)        # Color.RED
    print(Color.RED.name)   # RED
    print(Color.RED.value)  # 1
    ```

    自动赋值:

    ```py
    from enum import Enum, auto

    class Color(Enum):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    print(Color.RED.value)   # 1
    print(Color.GREEN.value) # 2
    ```

    字符串枚举:

    ```py
    from enum import Enum

    class HttpStatus(Enum):
        OK = "200 OK"
        NOT_FOUND = "404 Not Found"
        SERVER_ERROR = "500 Internal Server Error"

    print(HttpStatus.OK.value)  # "200 OK"
    ```

    使用 IntEnum（整数枚举）:

    ```py
    from enum import IntEnum

    class Priority(IntEnum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    # IntEnum 可以与其他整数比较
    print(Priority.LOW == 1)  # True
    ```

    使用 Flag（标志枚举）:

    ```py
    from enum import Flag, auto

    class Permission(Flag):
        READ = auto()
        WRITE = auto()
        EXECUTE = auto()
        READ_WRITE = READ | WRITE

    # 使用
    user_permissions = Permission.READ | Permission.WRITE
    print(Permission.READ in user_permissions)  # True
    ```

    唯一值枚举:

    ```py
    from enum import Enum, unique

    @unique
    class Status(Enum):
        PENDING = 1
        PROCESSING = 2
        COMPLETED = 3
        # ERROR = 1  # 这会抛出 ValueError，因为值重复
    ```

    对枚举进行迭代：

    ```py
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    for color in Color:
        print(color.name, color.value)
    ```

* pip 可以直接使用环境变量`http_proxy`, `https_proxy`等进行代理

* 打开文件时`a+`的行为分析

    使用`a+`打开时，`seek()`只对读取有效，对写入无效，写入总是发生在文件末尾。

    example:

    ```py
    with open('msg.txt', 'a+') as f:
        f.write('hello\n')
        f.seek(0)
        f.write('world\n')

        print('first read:')
        content = f.read()
        print(content)
        print('')

        print('second read:')
        f.seek(0)
        content = f.read()
        print(content)
    ```

    output:

    ```
    first read:


    second read:
    hello
    world

    ```

    可以看到，虽然在`f.write('world\n')`之前调用了`f.seek(0)`，但是写入的`world`仍然在`hello`后面。

    另外，调用完`f.write()`后，当前 pos 位置又变到文件末尾，所以第一次`f.read()`没有读到内容。

    `a+`模式下，虽然`seek()`不影响`write()`的行为，但是影响`read()`的行为，可以看到第二次 read 读到了文件的内容。

* py 中，open file 时`a+`表示追加并且可读，只有`a`表示追加，但是读取文件时会报错

    example:

    * 只可追加，不可读

        ```py
        with open('test_doc.txt', "a") as f:
            content = f.read()
        print(content)
        ```

        output:

        ```
        Traceback (most recent call last):
          File "/home/hlc/Documents/Projects/python_test/main_2.py", line 2, in <module>
            content = f.read()
                      ^^^^^^^^
        io.UnsupportedOperation: not readable
        ```

    * 既可追加，又可读

        ```py
        with open('test_doc.txt', "a+") as f:
            content = f.read()
        print('first read:')
        print(content)

        with open('test_doc.txt', "a+") as f:
            f.seek(0)
            content = f.read()
        print('second read:')
        print(content)
        ```

        output:

        ```
        first read:

        second read:
        你好
        世界
        nihao
        zaijian
        ```

        可以看到，第一次读文件时，没有内容。因为`a+`模式，默认当前位置在文件末尾。

* python 中 is 关键字用于身份比较（identity comparison），它检查两个变量是否引用内存中的同一个对象。

    example:

    ```py
    # 比较两个变量是否指向同一个对象
    a = [1, 2, 3]
    b = a  # b 和 a 指向同一个列表对象
    c = [1, 2, 3]  # c 指向一个新的列表对象

    print(a is b)  # True - 同一个对象
    print(a is c)  # False - 值相同但不是同一个对象

    # 与 None 的比较（推荐用法）
    x = None
    print(x is None)  # True
    print(x is not None)  # False
    ```

    * is 与 == 的区别:

        ```py
        # is: 身份比较（是否同一个对象）
        # ==: 值比较（值是否相等）

        a = ''
        b = ''

        print(a == b)  # True - 值相等
        print(a is b)  # 可能为 True 或 False，取决于字符串驻留
        ```

    * 小整数和字符串驻留

        Python 会对小整数和某些字符串进行驻留优化：

        ```py
        # 小整数（-5 到 256）会被缓存
        a = 100
        b = 100
        print(a is b)  # True

        # 空字符串通常也会被驻留
        a = ''
        b = ''
        print(a is b)  # True（在大多数实现中）
        ```

    * 正确的 None 比较方式

        ```py
        # 推荐：使用 is 来比较 None
        if x is None:
            print("x is None")

        # 不推荐：使用 == 来比较 None
        if x == None:  # 能工作，但不推荐
            print("x == None")
        ```

* python 读文件

    `read([size])`: 一次性读取整个文件内容，并将其作为一个字符串返回。

    可选的 size 参数，指定要读取的字符数（文本模式）或字节数（二进制模式）。如果不提供，则读取整个文件。

    `test_doc.txt`:

    ```
    你好
    世界
    nihao
    zaijian
    ```

    ```py
    file = 'test_doc.txt'

    with open(file) as f:
        content = f.read()  # read all characters
    print('------ test 1: read all characters ------')
    print(content)

    # open as text file
    with open(file) as f:
        content = f.read(7)  # read 7 characters
    print('------ test 2: read 7 characters ------')
    print(content)

    # open as binary file
    with open(file, 'rb') as f:
        content = f.read(7)  # read 7 bytes
    print('------ test 3: read 7 bytes ------')
    print(content)
    ```

    output:

    ```
    ------ test 1: read all characters ------
    你好
    世界
    nihao
    zaijian
    ------ test 2: read 7 characters ------
    你好
    世界
    n
    ------ test 3: read 7 bytes ------
    b'\xe4\xbd\xa0\xe5\xa5\xbd\n'
    ```

    * `readline([size])`

        一次只读取文件的一行。

        返回值：一个字符串，包含一行的内容（包括换行符 \n）。如果到达文件末尾，则返回一个空字符串。

        ```py
        file = 'test_doc.txt'

        with open(file) as f:
            line = f.readline()
            while line != '':
                print(line)
                line = f.readline()
        ```

        output:

        ```
        你好

        世界

        nihao

        zaijian
        ```

    * `readlines([hint])`

        读取整个文件，并将其作为一个列表返回，列表中的每个元素是文件中的一行（字符串）。

        可选的 hint 参数。如果指定了 hint，则读取大约 hint 个字节的行，直到读完这些字节所在的行为止，可能不会读取整个文件。

        ```py
        file = 'test_doc.txt'

        with open(file) as f:
            lines = f.readlines()
        print(lines)
        ```

        output:

        ```
        ['你好\n', '世界\n', 'nihao\n', 'zaijian']
        ```

        可以看到`\n`仍然被保留。

    * 文件对象本身是可迭代的

        迭代文件对象本身，这相当于一个“惰性”的 readline()，内存效率最高。

        ```py
        # 这是读取大文件的最佳方式
        with open('example.txt', 'r') as file:
            for line in file: # 直接遍历文件对象
                print(line, end='')
        ```

        对于非常大的文件，read() 和 readlines() 会一次性将整个文件加载到内存中，可能导致内存不足。此时，应使用 readline() 或直接迭代文件对象。

* python 中判断空字符串，只能用`if '' == ''`

    不能用`if '' is None`, `if '' == None`, `if '' is ''`

* python 中没有很好支持 do while 的方法，只能用 while + if + break 来模拟

* python 中判断一个 key 是否在 dict 中

    * 使用`in`关键字

    * 使用 get() 方法

        ```py
        my_dict = {'a': 1, 'b': 2, 'c': 3}

        # 如果 key 不存在，返回 None 或默认值
        value = my_dict.get('a')  # 返回 1
        value = my_dict.get('d')  # 返回 None
        value = my_dict.get('d', 'default')  # 返回 'default'

        # 判断存在性
        if my_dict.get('a') is not None:
            print("Key 'a' exists")
        ```

    * 使用 keys() 方法

        ```py
        my_dict = {'a': 1, 'b': 2, 'c': 3}

        if 'a' in my_dict.keys():
            print("Key 'a' exists")
        ```

    * 使用 try-except 块

        ```py
        my_dict = {'a': 1, 'b': 2, 'c': 3}

        try:
            value = my_dict['d']
            print("Key 'd' exists")
        except KeyError:
            print("Key 'd' does not exist")
        ```

* python 中使用实例可以直接定义成员变量

    ```py
    class MyStruc:
        def __init__(self):
            self.val_1 = 123

    obj_1 = MyStruc()
    obj_1.val_2 = 456

    print(obj_1.val_1)
    print(obj_1.val_2)
    ```

    output:

    ```
    123
    456
    ```

    在 IDE 里，`obj_1.`没有关于`val_2`的自动补全和提示，但是运行程序是正常的。

* python 中的 f-string

    f"xxx" 是 f-string（格式化字符串字面值，Formatted string literals）的语法，它在 Python 3.6 中首次引入。它是一种在字符串中直接嵌入表达式的字符串格式化机制.

    基本用法:

    * 嵌入变量（最基本的功能）

        在字符串前加上前缀 f 或 F，然后在字符串内部用大括号 {} 包裹变量名或表达式。Python 会在运行时计算 {} 中的内容，并将其值转换为字符串插入到相应位置。

        example:

        ```py
        name = "Alice"
        age = 30

        # 传统的格式化方法
        greeting_old = "Hello, {}. You are {} years old.".format(name, age)
        # 使用 f-string
        greeting_new = f"Hello, {name}. You are {age} years old."

        print(greeting_new)
        # 输出: Hello, Alice. You are 30 years old.
        ```

    * 执行表达式

        {} 内不仅可以放变量，还可以放任何有效的 Python 表达式。

        example:

        ```py
        a = 5
        b = 10

        result = f"The sum of {a} and {b} is {a + b}, and their product is {a * b}."
        print(result)
        # 输出: The sum of 5 and 10 is 15, and their product is 50.
        ```

    * 调用函数和方法

        可以在 {} 中直接调用函数或对象的方法。

        example:

        ```py
        name = "bob"
        message = f"Your name in uppercase is {name.upper()} and its length is {len(name)}."
        print(message)
        # 输出: Your name in uppercase is BOB and its length is 3.
        ```

    * 格式化输出（类似 str.format() 的格式规范）

        可以在表达式后面跟上格式说明符（format specifier），用来控制输出的格式，比如小数点精度、数字的进制、对齐方式等。语法是 `{expression:format_spec}`。

        example:

        ```py
        import math

        price = 19.9876
        number = 42

        # 控制浮点数精度（保留两位小数）
        f_price = f"The price is ${price:.2f}" # 输出: The price is $19.99

        # 格式化为十六进制
        f_hex = f"The number {number} in hex is {number:#x}" # 输出: The number 42 in hex is 0x2a

        # 百分比显示
        f_percent = f"Completion: {0.756:.2%}" # 输出: Completion: 75.60%

        # 对齐文本（:>10 表示右对齐，宽度为10个字符）
        f_align = f"'{name:>10}'" # 输出: '       bob'

        print(f_price)
        print(f_hex)
        print(f_percent)
        print(f_align)
        ```

    * 转义大括号

        如果需要在 f-string 中显示字面意义的大括号，需要使用双重大括号进行转义。

        example:

        ```py
        value = "data"
        escaped = f"This is how you show braces: {{{value}}}" # 注意三层括号
        print(escaped)
        # 输出: This is how you show braces: {data}
        ```

    注意事项:

    * 引号问题：f-string 可以使用单引号 `'`、双引号 `"` 和三引号 `'''/"""`。

        ```py
        f'Hello, {name}.'
        f"Hello, {name}."
        f"""Hello,
        {name}."""
        ```

    * 表达式求值：f-string 中的表达式在运行时求值。这意味着它们使用的是当前作用域中的变量值。

    * 不能为空：{} 内部不能是空的，必须包含表达式。

    * Python 版本：确保你的运行环境是 Python 3.6 或更高版本，否则会引发 SyntaxError。

* $\infty$在 python 中的表示

    可以使用`float('inf')`表示无穷大。

    ```python
    # Python 示例（用 float('inf') 表示 ∞）
    adj_matrix = [
        [0, 2, float('inf')],
        [2, 0, 3],
        [float('inf'), 3, 0]
    ]
    ```

* python 字符串的`.rindex()`, `.rfind()`是从右边开始搜索，但是返回的索引仍然是从左边开始数的。

* python 中的定义提前

    ```python
    aaa = 'my_aaa'

    def main():
        aaa = aaa.rstrip()
    ```

* `re.finditer()`的使用时机

    当同一个模式（pattern）在一个字符串中轮番出现多次时，可以使用`re.finditer()`一个接一个地查找。

* python 中的`strip()`并不是删除指定字符串，而是删除在指定字符集中的字符

    ```python
    def main():
        txt = 'hello, world'
        bbb = txt.lstrip('leoh')
        print(bbb)
    ```

    output:

    ```
    , world
    ```

    可以使用`removeprefix()`移除指定字符串。

* `txt = 'hello, world'`匹配` world`（`world`前有个空格）

    我们先想到，用直接匹配法是否能匹配到？

    ```python
    import re

    def main():
        txt = 'hello, world'
        pat = r' world'
        m = re.search(pat, txt)
        if m is None:
            print('fail to match')
            return
        selected_txt = txt[m.start():m.end()]
        print(selected_txt)
        return
    ```

    output:

    ```
     world
    ```

    可以看到使用直接匹配法可以成功匹配。并且说明`pat`中的空格也是有意义的，

    尝试将`pat`中的空格替换为`\ `，依然可以正常匹配，说明空格的转义不影响其含义。

    尝试将`re.search()`替换为`re.match()`，输出如下：

    ```
    fail to match
    ```

    说明`match()`只能从头开始匹配，如果匹配失败则返回空。

    另外一个想法是使用`[ world]+`进行匹配，理论上所有包含的字母都在这里面了，是没有问题的，然而实际写出的程序是这样的：

    ```python
    import re

    def main():
        txt = 'hello, world'
        pat = r'[ world]+'
        fail_to_match = True
        for m in re.finditer(pat, txt):
            fail_to_match = False
            selected_txt = txt[m.start():m.end()]
            print(selected_txt)
        if fail_to_match:
            print('fail to match')   
        return
    ```

    output:

    ```
    llo
     world
    ```

    可以看到，`finditer()`会从头开始尝试匹配，先匹配到`llo`，然后才匹配到` world`。如果使用`search()`匹配，那么只返回`llo`。

    将`pat`改为`pat = r'[\ world]+'`，输出不变。说明在`[]`内，空格` `和转义空格`\ `的含义相同。

    `[]`中的逗号`,`直接代表逗号，并不是分隔，将`pat`改为`pat = r'[,\ world]+'`后，输出为`llo, world`。

    如果我们将空格放在外面，则可第一次就匹配成功：

    ```python
    import re

    def main():
        txt = 'hello, world'
        pat = r' [a-z]+'
        m = re.search(pat, txt)
        if m is None:
            print('fail to match')
            return
        selected_txt = txt[m.start():m.end()]
        print(selected_txt)
        return
    ```

    output:

    ```
     world
    ```

* python 里`print()`指定`end=None`仍然会打印换行符，只有指定`end=''`才会不打印换行

* python 里`re`正则表达式匹配的都是字符串，而`^`代表字符串的开头，并不代表一行的开始

    因此使用`^`去匹配每行的开始，其实是有问题的，只能匹配到一次。

* python 的`re`模块不支持非固定长度的 look behind 的匹配

    比如，`(?<+\[.*\]).*`，这个表达式本意是想向前匹配一个`[]`括号，括号中的内容任意，但不能有換行符。

    比如`[hello]this is the world`，想匹配到的内容是`this is the world`。

    但是上面的匹配是不允许的，因为 look behind 时，要匹配的内容是一个非固定长度字符串。

    具体来说可能是因为实现起来太复杂，具体可参考这里：<https://stackoverflow.com/questions/9030305/regular-expression-lookbehind-doesnt-work-with-quantifiers-or>

* python `pathlib` 列出指定目录下的所有子目录

    ```python
    from pathlib import Path

    def main():
        aaa = '.'
        cur_path = Path(aaa)
        child_dirs = [x for x in cur_path.iterdir() if x.is_dir()]
        print(child_dirs)
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    [PosixPath('test_dir_2'), PosixPath('test_dir_1'), PosixPath('.test_dir_3')]
    ```

    说明隐藏文件夹也可以列出来。

    `x`是`Path`类型的实例。

* python format 基础用法

    ```python
    def main():
        # 基础用法，{} 占位，参数按 position 顺序填
        s_1 = 'hello, {}, {}'.format('world', 42)
        print(s_1)  # hello, world, 42

        # 按 key-value 的形式填
        world = 'world'
        forty_two = 42
        s_2 = 'hello, {s_world}, {num_forty_two}'.format(s_world=world, num_forty_two=forty_two)
        print(s_2)  # hello, world, 42

        # {} 占位对应 position parameter，字符串点位对应 key-value prarmeter
        s_3 = 'hello, {s_world}, {}'.format(forty_two, s_world=world)
        print(s_3)  # hello, world, 42

        # 指定占位顺序
        s_4 = '{2}, {1}, {0}'.format(forty_two, world, 'hello')
        print(s_4)  # hello, world, 42

        # 格式化
        year = 2024
        s_5 = '{year:08d}'.format(year=year)
        print(s_5)  # 00002024
        return
    ```

* py 可以直接用`in`判断一个 key 是否在一个 dict 中

    ```py
    a = {}
    a[1] = 2
    a['3'] = 4
    if 1 in a:
        print('1 in a')
    if '3' in a:
        print("'3' in a")
    ```

    output:

    ```
    1 in a
    '3' in a
    ```

* py 中使用`with open('xxx', 'w') as f:`打开的文件无法使用`f.read()`，会报错，只有使用`'w+'`打开才可以

    有时间了找找更多的资料。

* py 中`aaa: str`不能定义一个变量，只能声明

* py 中的`os.listdir()`可以列出指定文件夹下的所有文件和文件夹的名称

    ```python
    import os

    def main():
        path = '.'
        dirs = os.listdir(path)
        print(dirs)
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    ['main.py', '你好.txt', '再见', 'test_dir']
    ```

    说明：

    1. `path`可以包含中文，python 可以正常处理。

    2. `listdir()`给出的是一个`list[str]`，无法区分列出的 name 是一个文件还是文件夹。

    3. 如果`path`是一个`.`，那么表示`main.py`所在的文件夹

    4. 如果 path 是一个无效路径，那么 python 会直接报错

* py 中可以使用`datetime`包拿到当前的日期和时间

    ```py
    cur_datetime = datetime.datetime.now()
    year_str = str(cur_datetime.year)
    ```

    datetime 最小可以拿到秒和微秒的数据（macrosecond）。

* py 中`hash()`得出的结果有时候为负值，可以使用`ctypes`包把转换成正值

    ```py
    hash_int = hash(datetime_str)
    if hash_int < 0:
        hash_int = ctypes.c_ulong(hash_int).value
    ```

* python path 判断一个文件夹是否包含另一个文件/文件夹

    没有什么特别好的方法，比较常见的办法是`os.walk()`遍历，然后判断文件/文件夹是否存在。想了想，这种方法比较适合只搜索一次就结束的。

    如果不知道绝对路径，并且需要多次搜索，一个想法是构建出一棵树，再构建一个哈希表映射文件/文件夹字符串到 node 指针，然后不断找这个 node 的 parent，看另一个 node 是否会成为这个 parent。

    如果已知两个文件（夹）的绝对路径，那么直接 compare 一下就可以了。如果前 n 个字符都相等，并且较长的字符串的下一个字符是`/`，则说明有包含关系。

    一个实现如下：

    ```py
    import os

    def main():
        path_1 = './mydir_1'
        path_2 = './mydir_1/mydir_2'
        node_1 = os.path.abspath(path_1)
        node_2 = os.path.abspath(path_2)
        min_len = min(len(node_1), len(node_2))
        max_len = max(len(node_1), len(node_2))
        for i in range(min_len):
            if node_1[i] != node_2[i]:
                print('not included')
                return
        if len(node_2) > len(node_1) and node_2[min_len] == '/':
            print('included')
        if len(node_1) > len(node_2) and node_1[min_len] == '/':
            print('included')
        
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    included
    ```

    边界条件还需要再测测。

* 考虑下面一个场景，在 py 里，给定`lst_A`, `lst_B`，如何在不使用 for 的情况下得到`lst_C`？

    ```py
    lst_A = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}]
    lst_B = [3, 4]
    lst_C = [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    ```

* python 里，如果想从 iterable 里抽取一些信息，可以使用列表推导式

    ```py
    weights = [qa_file_info['weight'] for qa_file_info in qa_file_infos]
    weight_sum = sum(weights)
    ```

    目前没有找到其他比较好的方法

* python re

    finditer 可以不 compile pattern 直接用

    ```python
    import re

    def main():
        txt = 'abcbacaccba'
        for m in re.finditer('a.{2}', txt):
            start_pos = m.start()
            end_pos = m.end()
            selected_txt = txt[start_pos:end_pos]
            print(selected_txt)
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    abc
    aca
    ```

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

## subprocess

### cache

* 使用`subprocess.run()`将子程序的 stdout 重定向到程序内部的内存

    example:

    ```py
    import subprocess

    def main():
        ret = subprocess.run(['ls', '-lh'], capture_output=True, text=True)
        print('stdout:')
        print('{}'.format(ret.stdout))
        print('stderr:')
        print('{}'.format(ret.stderr))
        print('ret code:')
        print('{}'.format(ret.returncode))
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    stdout:
    total 4.0K
    -rw-rw-r-- 1 hlc hlc 327  6月 16 22:52 main.py

    stderr:

    ret code:
    0
    ```

    这项功能只有`subprocess.run()`可以完成，无法使用`subprocess.call()`完成。

    说明：

    1. 如果不写`text=True`，那么`ret.stdout`等保存的内容是二进制内容`b'xxxx'`，中文等字符会被编码成 utf-8 格式的三个字节，比如`\xe6\x9c\x88`。

* python subprocess

    在一个进程中调用命令起另一个进程。

    example:

    ```py
    import subprocess

    def main():
        ret = subprocess.call(['ls'])
        print('ret: {}'.format(ret))
        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    main.py
    ret: 0
    ```

    如果需要加参数，那么可以在 list 里添加更多的元素：

    `main.py`:

    ```py
    import subprocess

    def main():
        ret = subprocess.call(['ls', '-lh'])
        print('ret: {}'.format(ret))
        return

    if __name__ == '__main__':
        main()
    ```

    run:

    `python main.py`

    output:

    ```
    total 4.0K
    -rw-rw-r-- 1 hlc hlc 155  6月 16 22:29 main.py
    ret: 0
    ```

    需要注意的是，`['my_cmd', '-my_arg val']`和`['my_cmd', '-my_arg', 'val']`在大部分情况下功能相同，但是对一小部分软件来说是有区别的，这两种形式可能只有一种可以正确执行。

### note

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

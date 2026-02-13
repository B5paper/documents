# Python Note

## cache

* 简述 python 中 random.choice() 和 random.choices() 的原型、作用及用法

    random.choice()

    原型

    ```python
    random.choice(seq)
    ```

    作用

    从非空序列 `seq` 中随机返回一个元素，**各元素被选中的概率相等**。

    用法

    ```python
    import random

    # 从列表中随机选择一个
    fruits = ['apple', 'banana', 'orange']
    print(random.choice(fruits))  # 输出：随机一个水果

    # 从字符串中随机选择一个字符
    text = "Hello"
    print(random.choice(text))    # 输出：随机一个字符（H/e/l/l/o）

    # 从元组中随机选择
    nums = (1, 2, 3, 4, 5)
    print(random.choice(nums))    # 输出：随机一个数字

    # 注意：空序列会引发 IndexError
    # random.choice([])  # 错误！
    ```

    random.choices()

    原型

    ```python
    random.choices(population, weights=None, *, cum_weights=None, k=1)
    ```

    作用

    从序列 `population` 中**有放回**地随机抽取 `k` 个元素，返回一个**列表**。支持**权重设置**。

    用法

    ```python
    import random

    # 1. 基本用法：随机抽取3个元素（可重复）
    fruits = ['apple', 'banana', 'orange']
    print(random.choices(fruits, k=3))  
    # 输出示例：['apple', 'banana', 'apple']（元素可重复）

    # 2. 设置权重（相对概率）
    # weights：各元素被选中的相对权重
    print(random.choices(
        ['red', 'black', 'green'],
        weights=[3, 2, 1],  # red被选中的概率是green的3倍
        k=10
    ))

    # 3. 累积权重（性能更好）
    # cum_weights：累积权重 [3, 5, 6] 相当于 weights [3, 2, 1]
    print(random.choices(
        ['A', 'B', 'C'],
        cum_weights=[3, 5, 6],
        k=5
    ))

    # 4. 不指定k时，默认k=1，返回只有一个元素的列表
    result = random.choices(['a', 'b', 'c'])
    print(result)  # 输出示例：['b']（注意是列表）
    ```

    核心区别

    | 特性 | random.choice() | random.choices() |
    |------|-----------------|------------------|
    | **返回值** | 单个元素 | 列表（即使k=1） |
    | **抽样方式** | 每次1个 | 可抽取多个 |
    | **是否可重复** | - | 有放回（可重复） |
    | **权重支持** | ❌ 不支持 | ✅ 支持 |
    | **空序列** | ❌ IndexError | ✅ 返回[] |

    快速记忆

    - **choice**：单数 → **1**个元素
    - **choices**：复数 → 多个元素（列表）
    - 需要概率/权重 → 用 **choices**
    - 只需随机取1个 → 用 **choice**

* 简述 random 中其他常用函数的原型、作用及用法

    random.random()

    原型

    ```python
    random.random()
    ```

    作用

    返回 [0.0, 1.0) 范围内的**随机浮点数**，左闭右开。

    用法

    ```python
    import random
    print(random.random())  # 输出：0.37444887175646646
    ```

    random.randint()

    原型

    ```python
    random.randint(a, b)
    ```

    作用

    返回 [a, b] 范围内的**随机整数**，**闭区间**（包含两端）。

    用法

    ```python
    print(random.randint(1, 10))   # 输出：1-10之间的整数（包含1和10）
    print(random.randint(0, 100))  # 输出：0-100之间的整数
    ```

    random.uniform()

    原型

    ```python
    random.uniform(a, b)
    ```

    作用

    返回 [a, b] 范围内的**随机浮点数**（可能包含两端，取决于浮点数舍入）。

    用法

    ```python
    print(random.uniform(1.5, 6.5))  # 输出：1.5-6.5之间的浮点数
    ```

    random.randrange()

    原型

    ```python
    random.randrange(stop)                      # [0, stop)
    random.randrange(start, stop[, step])       # [start, stop) 步长为step
    ```

    作用

    返回指定范围内的**随机整数**，**左闭右开**，支持步长。

    用法

    ```python
    print(random.randrange(10))      # 输出：0-9之间的整数
    print(random.randrange(1, 10))   # 输出：1-9之间的整数
    print(random.randrange(0, 20, 2)) # 输出：0-18之间的偶数
    ```

    random.shuffle()

     原型

    ```python
    random.shuffle(x)
    ```

    作用

    **原地打乱**序列顺序，返回 `None`（修改原序列）。

    用法

    ```python
    cards = ['A', '2', '3', '4', '5']
    random.shuffle(cards)
    print(cards)  # 输出示例：['3', 'A', '5', '2', '4']

    # 注意：只能用于可变序列（列表），不能用于元组、字符串
    ```

    random.sample()

     原型

    ```python
    random.sample(population, k)
    ```

    作用

    从序列中**无放回**地随机抽取 `k` 个元素，返回列表（**不重复**）。

    用法

    ```python
    fruits = ['apple', 'banana', 'orange', 'grape', 'kiwi']
    print(random.sample(fruits, 3))  # 输出：3个不重复的水果

    # 也可用于字符串、元组等
    text = "Python"
    print(random.sample(text, 4))    # 输出：4个不重复的字符列表
    ```

    random.seed()

    原型

    ```python
    random.seed(a=None, version=2)
    ```

    作用

    初始化随机数生成器，固定种子可**复现随机结果**。

    用法

    ```python
    random.seed(42)
    print(random.random())  # 固定输出：0.6394267984578837
    print(random.randint(1, 100))  # 固定输出：82

    random.seed(42)  # 重置种子
    print(random.random())  # 再次得到：0.6394267984578837
    ```

    random.getrandbits()

    原型

    ```python
    random.getrandbits(k)
    ```

    作用

    返回 `k` 位二进制随机整数（0 ~ 2^k - 1）。

    用法

    ```python
    print(random.getrandbits(8))   # 输出：0-255之间的整数
    print(random.getrandbits(16))  # 输出：0-65535之间的整数
    ```

    函数速查表

    | 函数 | 返回值类型 | 范围 | 特点 |
    |------|-----------|------|------|
    | `random()` | float | [0.0, 1.0) | 基础随机浮点数 |
    | `randint(a, b)` | int | [a, b] | 闭区间整数 |
    | `uniform(a, b)` | float | [a, b] | 浮点数版本 |
    | `randrange()` | int | [start, stop) | 支持步长 |
    | `shuffle()` | None | - | 原地打乱 |
    | `sample()` | list | - | 无放回抽样 |
    | `seed()` | None | - | 固定随机结果 |
    | `getrandbits()` | int | [0, 2^k-1] | 二进制位随机 |

    选择建议

    - **随机浮点数** → `random()`, `uniform()`
    - **随机整数** → `randint()`（闭区间）, `randrange()`（开区间）
    - **随机选1个** → `choice()`
    - **随机选多个（可重复）** → `choices()`
    - **随机选多个（不重复）** → `sample()`
    - **打乱顺序** → `shuffle()`
    - **复现结果** → `seed()`

* python 中，if type(str_list[0]) is not list and not tuple: 可以这样写逻辑表达式吗？

    可以的，但这种写法有**优先级问题**，容易导致逻辑错误。让我们分析一下：

    当前写法的问题

    ```python
    if type(str_list[0]) is not list and not tuple:
    ```

    实际上等价于：

    ```python
    if (type(str_list[0]) is not list) and (not tuple):
    ```

    因为 `not tuple` 会把 tuple 类对象当作布尔值判断，`not tuple` 永远为 False（因为类对象在布尔上下文中为 True），所以整个表达式可能不符合你的预期。

    推荐写法

    根据你的需求，应该使用以下任意一种：

    方法1：使用 `and` 连接两个 `is not`

    ```python
    if type(str_list[0]) is not list and type(str_list[0]) is not tuple:
    ```

    方法2：使用 `not isinstance()`（更推荐）

    ```python
    if not isinstance(str_list[0], (list, tuple)):
    ```

    方法3：使用 `type() in` 判断

    ```python
    if type(str_list[0]) not in (list, tuple):
    ```

    总结对比

    ```python
    # ❌ 有问题：not tuple 永远为 False
    if type(str_list[0]) is not list and not tuple:

    # ✅ 正确：明确检查两种类型
    if type(str_list[0]) is not list and type(str_list[0]) is not tuple:

    # ✅ 更简洁，推荐使用
    if not isinstance(str_list[0], (list, tuple)):

    # ✅ 另一种写法
    if type(str_list[0]) not in (list, tuple):
    ```

    推荐使用 `isinstance()`，因为它还能正确处理继承关系。

* python 中，对于字符串 'CH1'，我如何才能将字符串中的 1 提取出来？

    在Python中，提取字符串 'CH1' 中的数字 1 有几种方法：

    方法1：使用正则表达式（推荐）

    ```python
    import re

    s = 'CH1'
    # 查找所有数字
    numbers = re.findall(r'\d+', s)
    result = int(numbers[0]) if numbers else None
    print(result)  # 输出: 1
    ```

    方法2：使用isdigit()过滤

    ```python
    s = 'CH1'
    result = ''.join(filter(str.isdigit, s))
    if result:
        result = int(result)
        print(result)  # 输出: 1
    ```

    方法3：使用列表推导式

    ```python
    s = 'CH1'
    result = ''.join([char for char in s if char.isdigit()])
    if result:
        result = int(result)
        print(result)  # 输出: 1
    ```

    方法4：如果知道数字位置

    ```python
    s = 'CH1'
    # 如果知道数字在最后一位
    result = int(s[-1])  # 输出: 1
    ```

    方法5：使用split分割（如果格式固定）

    ```python
    s = 'CH1'
    # 如果知道数字前面是'CH'
    result = int(s.split('CH')[1])  # 输出: 1
    ```

    推荐方案：

    如果你的字符串总是类似 "字母+数字" 的格式：

    ```python
    def extract_number_from_string(s):
        import re
        numbers = re.findall(r'\d+', s)
        return int(numbers[0]) if numbers else None

    # 使用
    result = extract_number_from_string('CH1')  # 1
    result2 = extract_number_from_string('ABC123')  # 123
    result3 = extract_number_from_string('TEST45XYZ')  # 45
    ```

    **建议使用正则表达式**，因为它：

    1. 可以处理多位数字（如 'CH123' → 123）
    2. 更灵活，适用于不同格式
    3. 代码可读性好

* python 没有类似 scanf() 的方法吗？

    在Python中没有直接类似C语言`scanf()`的**内置函数**，但有几种类似的方法可以实现类似功能：

    1. 使用 `re.match()` 最接近 `scanf()`

        ```python
        import re

        s = 'CH1'
        match = re.match(r'CH(\d+)', s)  # 类似 scanf("CH%d", &num)
        if match:
            result = int(match.group(1))  # 输出: 1
            print(result)
        ```

    2. 使用第三方库 `scanf`（需要安装）

        这是最接近C语言`scanf()`的方式：

        ```bash
        # 先安装
        pip install scanf
        ```

        ```python
        from scanf import scanf

        s = 'CH1'
        result = scanf("CH%d", s)
        print(result)  # 输出: (1,)

        s = 'Temperature: 25.5°C'
        result = scanf("Temperature: %f°C", s)
        print(result)  # 输出: (25.5,)

        # 多个变量
        s = 'John,25,Engineer'
        name, age, job = scanf("%s,%d,%s", s)
        print(name, age, job)  # 输出: John 25 Engineer
        ```

    3. 使用 `parse` 库（更Pythonic的类似scanf）

        ```bash
        pip install parse
        ```

        ```python
        from parse import parse

        s = 'CH1'
        result = parse("CH{}", s)
        print(result[0])  # 输出: '1'

        # 带类型转换
        result = parse("CH{:d}", s)
        print(result[0])  # 输出: 1

        # 多个变量
        s = 'Result: 42 items at $19.99'
        result = parse("Result: {} items at ${}", s)
        print(result)  # 输出: ('42', '19.99')
        ```

    4. 使用字符串的 `partition()` 或 `split()`

        ```python
        s = 'CH1'
        # 如果格式固定
        _, _, num = s.partition('H')
        result = int(num)  # 输出: 1

        # 或者
        prefix, num = s.split('H')
        result = int(num)  # 输出: 1
        ```

    5. 自己实现简单的scanf

        ```python
        def simple_scanf(pattern, string):
            """简单的scanf实现，只支持%d和%s"""
            import re
            
            # 转换格式说明符为正则表达式
            pattern = pattern.replace('%d', r'(\d+)')
            pattern = pattern.replace('%s', r'(\w+)')
            pattern = pattern.replace('%f', r'([\d\.]+)')
            
            match = re.match(pattern, string)
            return match.groups() if match else None

        # 使用
        result = simple_scanf('CH%d', 'CH1')
        print(result)  # 输出: ('1',)
        ```

    推荐方案：

    - **如果只需要简单提取**：用 `re.findall()` 或字符串方法
    - **如果经常需要模式匹配**：用 `parse` 库（最Pythonic）
    - **如果想要C语言风格**：用 `scanf` 库

    对于你的具体需求（提取'CH1'中的1），最简单的还是：

    ```python
    import re
    s = 'CH1'
    result = int(re.search(r'\d+', s).group())
    ```

    Python的哲学是"明确优于隐晦"，所以通常用正则表达式或字符串方法比`scanf`更清晰。

* python 如何将字符串转换为数字？

    在Python中，将字符串转换为数字有几种主要方法，具体取决于数字类型：

    1. **转换为整数**

        使用 `int()`

        ```python
        # 十进制整数
        s1 = "123"
        num1 = int(s1)        # 123

        # 指定进制
        s2 = "1010"
        num2 = int(s2, 2)     # 二进制 → 10
        num3 = int("FF", 16)  # 十六进制 → 255
        num4 = int("77", 8)   # 八进制 → 63

        # 带正负号
        num5 = int("-42")     # -42
        num6 = int("+100")    # 100
        ```

    2. **转换为浮点数**

        使用 `float()`

        ```python
        s1 = "3.14"
        num1 = float(s1)          # 3.14

        s2 = "-2.5e3"            # 科学计数法
        num2 = float(s2)         # -2500.0

        s3 = "inf"              # 无穷大
        num3 = float(s3)        # inf

        s4 = "nan"              # 非数字
        num4 = float(s4)        # nan
        ```

    3. **安全转换（带错误处理）**

        ```python
        def safe_convert(s):
            try:
                # 先尝试整数
                return int(s)
            except ValueError:
                try:
                    # 再尝试浮点数
                    return float(s)
                except ValueError:
                    # 都不是数字
                    return None

        print(safe_convert("123"))    # 123
        print(safe_convert("12.3"))   # 12.3
        print(safe_convert("abc"))    # None
        ```

    4. **处理带特殊格式的字符串**

        ```python
        # 去除空格和特殊字符
        s1 = " 123 "
        num1 = int(s1.strip())      # 123

        s2 = "$1,234.56"
        num2 = float(s2.replace('$', '').replace(',', ''))  # 1234.56

        # 使用正则表达式提取数字
        import re
        s3 = "Price: $99.99"
        match = re.search(r'\d+\.?\d*', s3)
        if match:
            num3 = float(match.group())  # 99.99
        ```

    5. **字符串列表批量转换**

        ```python
        # 列表中的所有字符串转数字
        str_list = ["1", "2", "3", "4"]
        int_list = list(map(int, str_list))           # [1, 2, 3, 4]
        int_list2 = [int(x) for x in str_list]       # [1, 2, 3, 4]

        # 混合数字字符串列表
        mixed_list = ["1", "2.5", "3.14", "4"]
        num_list = []
        for s in mixed_list:
            try:
                if '.' in s:
                    num_list.append(float(s))
                else:
                    num_list.append(int(s))
            except ValueError:
                num_list.append(None)
        # 结果: [1, 2.5, 3.14, 4]
        ```

    6. **使用第三方库**

        使用 `numpy`

        ```python
        import numpy as np

        s = "123"
        num = np.float64(s)   # 123.0  (高精度浮点)

        # 批量转换
        arr = np.array(["1", "2", "3"], dtype=np.int32)
        ```

        使用 `pandas`

        ```python
        import pandas as pd

        # Series转换
        s = pd.Series(["1", "2", "3"])
        nums = s.astype(int)   # 转换为整数

        # 自动推断类型
        df = pd.DataFrame({"col": ["1", "2.5", "3"]})
        df["col"] = pd.to_numeric(df["col"], errors='coerce')
        ```

    7. **特殊转换需求**

        ```python
        # 罗马数字转整数（需要自定义函数或库）
        # 安装: pip install roman
        import roman
        s = "XIV"
        num = roman.fromRoman(s)  # 14

        # 中文数字转阿拉伯数字（需要自定义函数）
        def chinese_to_arabic(s):
            # 简单示例，实际需要更复杂的实现
            mapping = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}
            return mapping.get(s, None)

        num = chinese_to_arabic('三')  # 3
        ```

    8. **性能对比**

        ```python
        import timeit

        s = "123456"

        # int() 最快
        t1 = timeit.timeit('int(s)', setup='s="123456"', number=1000000)

        # float() 稍慢
        t2 = timeit.timeit('float(s)', setup='s="123456"', number=1000000)

        # eval() 最慢且不安全（不推荐）
        t3 = timeit.timeit('eval(s)', setup='s="123456"', number=1000000)

        print(f"int(): {t1:.4f}s, float(): {t2:.4f}s, eval(): {t3:.4f}s")
        ```

    注意事项：

    1. **不要使用 `eval()`**：`eval("123")` 虽然可以转换，但非常危险
    2. **处理异常**：总是用 `try-except` 包装转换操作
    3. **类型检查**：转换后检查类型是否符合预期
    4. **区域设置**：注意小数点可能是 `.` 或 `,`（欧洲格式）

    推荐的最佳实践：

    ```python
    def convert_string_to_number(s):
        """安全地将字符串转换为数字"""
        if not isinstance(s, str):
            raise TypeError("输入必须是字符串")
        
        s = s.strip()
        if not s:
            return None
        
        try:
            # 先尝试整数
            return int(s)
        except ValueError:
            try:
                # 再尝试浮点数
                return float(s)
            except ValueError:
                # 不是有效的数字字符串
                return None

    # 使用
    result = convert_string_to_number("  123  ")  # 123
    result2 = convert_string_to_number("12.34")   # 12.34
    result3 = convert_string_to_number("abc")     # None
    ```

    根据你的具体需求选择合适的转换方法。如果只是简单的数字转换，`int()` 和 `float()` 通常就足够了。

* python中，如何分解质因数？

    在Python中，分解质因数有多种方法。以下是几种常见的实现方式：

    方法1：基础实现

    ```python
    def prime_factors(n):
        """返回质因数列表"""
        factors = []
        divisor = 2
        
        while divisor * divisor <= n:
            while n % divisor == 0:
                factors.append(divisor)
                n //= divisor
            divisor += 1 if divisor == 2 else 2  # 检查2后，只检查奇数
        
        if n > 1:
            factors.append(n)
        
        return factors

    # 示例
    num = 360
    print(f"{num}的质因数分解为: {prime_factors(num)}")
    ```

    方法2：使用字典统计质因数个数

    ```python
    def prime_factors_dict(n):
        """返回质因数及其指数的字典"""
        factors = {}
        divisor = 2
        
        while divisor * divisor <= n:
            while n % divisor == 0:
                factors[divisor] = factors.get(divisor, 0) + 1
                n //= divisor
            divisor += 1 if divisor == 2 else 2
        
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        return factors

    def format_prime_factors(factors_dict):
        """格式化输出质因数分解结果"""
        if not factors_dict:
            return "1"
        
        result = []
        for prime in sorted(factors_dict.keys()):
            exp = factors_dict[prime]
            if exp == 1:
                result.append(str(prime))
            else:
                result.append(f"{prime}^{exp}")
        return " × ".join(result)

    # 示例
    num = 360
    factors_dict = prime_factors_dict(num)
    print(f"{num} = {format_prime_factors(factors_dict)}")
    ```

    方法3：递归实现

    ```python
    def prime_factors_recursive(n, divisor=2):
        """递归方式分解质因数"""
        if n == 1:
            return []
        if n % divisor == 0:
            return [divisor] + prime_factors_recursive(n // divisor, divisor)
        return prime_factors_recursive(n, divisor + 1)

    # 示例
    num = 84
    print(f"{num}的质因数: {prime_factors_recursive(num)}")
    ```

    方法4：优化的算法（预先生成质数）

    ```python
    def sieve_of_eratosthenes(limit):
        """生成小于等于limit的所有质数"""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def prime_factors_optimized(n):
        """使用预生成的质数进行分解"""
        if n < 2:
            return []
        
        factors = []
        primes = sieve_of_eratosthenes(int(n**0.5))
        
        for prime in primes:
            if prime * prime > n:
                break
            while n % prime == 0:
                factors.append(prime)
                n //= prime
        
        if n > 1:
            factors.append(n)
        
        return factors

    # 示例
    num = 1001
    print(f"{num}的质因数: {prime_factors_optimized(num)}")
    ```

    方法5：完整的实用函数

    ```python
    def factorize(n):
        """
        完整的质因数分解函数
        
        参数:
            n: 要分解的正整数
        
        返回:
            字典: {质因数: 指数}
            字符串: 格式化表达式
        """
        if n < 1:
            raise ValueError("输入必须是正整数")
        if n == 1:
            return {1: 1}, "1"
        
        original_n = n
        factors = {}
        divisor = 2
        
        while divisor * divisor <= n:
            count = 0
            while n % divisor == 0:
                count += 1
                n //= divisor
            if count > 0:
                factors[divisor] = count
            divisor += 1 if divisor == 2 else 2
        
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        # 格式化输出
        formatted = " × ".join(
            f"{prime}^{exp}" if exp > 1 else str(prime)
            for prime, exp in sorted(factors.items())
        )
        
        return factors, formatted

    # 使用示例
    numbers = [12, 56, 100, 360, 997]  # 997是质数

    for num in numbers:
        factors_dict, expression = factorize(num)
        print(f"{num:4d} = {expression}")
        print(f"      质因数及指数: {factors_dict}")
        print("-" * 40)
    ```

    性能考虑

    1. **小数字**：使用基础方法足够
    2. **大数字**：可能需要优化（如只检查奇数）
    3. **重复分解多个数**：可以预先生成质数表

    时间复杂度

    - 最坏情况：O(√n)（当n是质数时）
    - 平均情况：远快于O(√n)

    这些方法涵盖了从简单到复杂的各种场景，你可以根据具体需求选择合适的方法。

* python 中是否有现成的包实现分解质因数

    是的，Python中有一些现成的包可以实现质因数分解。以下是常用的几个：

    1. **SymPy**（符号数学库）

        ```python
        from sympy import factorint, primefactors

        # 方法1: 返回质因数及其指数
        num = 360
        factors_dict = factorint(num)
        print(f"质因数分解（字典）: {factors_dict}")
        # 输出: {2: 3, 3: 2, 5: 1}

        # 方法2: 只返回质因数列表
        primes = primefactors(num)
        print(f"质因数列表: {primes}")
        # 输出: [2, 3, 5]

        # 处理大数
        large_num = 12345678901234567890
        factors_large = factorint(large_num)
        print(f"大数分解: {factors_large}")
        ```

    2. **gmpy2**（高性能数学库）

        ```python
        import gmpy2
        from gmpy2 import mpz, is_prime, next_prime

        def factorize_gmpy2(n):
            """使用gmpy2进行质因数分解"""
            n = mpz(n)
            factors = {}
            
            # 处理2
            count = 0
            while n % 2 == 0:
                n //= 2
                count += 1
            if count > 0:
                factors[2] = count
            
            # 处理奇数
            p = mpz(3)
            while p * p <= n:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                if count > 0:
                    factors[int(p)] = count
                p = next_prime(p)
            
            if n > 1:
                factors[int(n)] = 1
            
            return factors

        # 示例
        num = mpz(2**31 - 1)  # 2147483647，梅森素数
        print(f"因子分解: {factorize_gmpy2(num)}")
        ```

    3. **primefac**（专门的质因数分解库）

        ```python
        # 安装: pip install primefac
        import primefac

        num = 123456789

        # 方法1: 返回生成器
        factors = list(primefac.primefac(num))
        print(f"所有质因数: {factors}")

        # 方法2: 返回多重集
        factors_multiset = primefac.primefac(num, multi=True)
        print(f"多重集表示: {list(factors_multiset)}")

        # 检查是否是质数
        is_prime = primefac.isprime(997)
        print(f"997是质数吗？ {is_prime}")
        ```

    4. **math**标准库（有限功能）

        ```python
        import math

        def factorize_math(n):
            """使用math.gcd进行简单分解"""
            factors = []
            
            # 检查小质数
            for p in [2, 3, 5]:
                while n % p == 0:
                    factors.append(p)
                    n //= p
            
            # 检查6k±1形式的数
            i = 7
            increment = [4, 2, 4, 2, 4, 6, 2, 6]
            while i * i <= n:
                for inc in increment:
                    while n % i == 0:
                        factors.append(i)
                        n //= i
                    i += inc
            
            if n > 1:
                factors.append(n)
            
            return factors

        # 示例
        print(f"360的质因数: {factorize_math(360)}")
        ```

    5. **pyprimes**（质数相关操作）

        ```python
        # 安装: pip install pyprimes
        try:
            import pyprimes
            
            num = 100
            factors = list(pyprimes.factors(num))
            print(f"100的质因数: {factors}")
            
            # 生成质数
            primes = list(pyprimes.primes_below(50))
            print(f"50以下的质数: {primes}")
            
        except ImportError:
            print("需要安装pyprimes: pip install pyprimes")
        ```

    6. **完整的示例比较**

        ```python
        import time
        from sympy import factorint
        import primefac

        def benchmark_factoring(n, method_name, method_func):
            """基准测试"""
            start = time.time()
            result = method_func(n)
            elapsed = time.time() - start
            print(f"{method_name:20s}: {result} (耗时: {elapsed:.6f}秒)")
            return elapsed

        # 测试不同的数字
        test_numbers = [
            360,
            123456789,
            2**31 - 1,  # 2147483647 (质数)
            9876543210,
        ]

        print("质因数分解性能比较")
        print("=" * 60)

        for num in test_numbers:
            print(f"\n分解数字: {num}")
            print("-" * 40)
            
            # SymPy
            benchmark_factoring(num, "SymPy", factorint)
            
            # primefac
            benchmark_factoring(num, "primefac", 
                                lambda x: list(primefac.primefac(x)))
            
            # 自定义函数
            def custom_factor(n):
                factors = {}
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        factors[d] = factors.get(d, 0) + 1
                        n //= d
                    d += 1 if d == 2 else 2
                if n > 1:
                    factors[n] = factors.get(n, 0) + 1
                return factors
            
            benchmark_factoring(num, "自定义函数", custom_factor)
        ```

    7. **安装方法**

        ```bash
        # 安装SymPy
        pip install sympy

        # 安装gmpy2（可能需要编译工具）
        pip install gmpy2

        # 安装primefac
        pip install primefac

        # 安装pyprimes
        pip install pyprimes
        ```

    8. **推荐选择**

        | 场景 | 推荐包 | 理由 |
        |------|--------|------|
        | 一般用途 | **SymPy** | 功能全面，安装简单，文档完善 |
        | 大数运算 | **gmpy2** | 性能优秀，支持大整数 |
        | 专门质数操作 | **primefac** | 专注于质因数分解，算法多样 |
        | 教学/学习 | 标准库/自定义 | 理解原理更重要 |

    9. **实际应用示例**

        ```python
        from sympy import factorint

        def analyze_number(n):
            """分析数字的质因数特性"""
            factors = factorint(n)
            
            print(f"数字: {n}")
            print(f"质因数分解: {factors}")
            print(f"不同质因数个数: {len(factors)}")
            print(f"总质因数个数: {sum(factors.values())}")
            
            # 判断是否是...
            if len(factors) == 1 and list(factors.values())[0] == 1:
                print("这是一个质数")
            elif len(factors) == 1:
                prime, exp = list(factors.items())[0]
                print(f"这是一个质数的幂: {prime}^{exp}")
            
            # 计算欧拉函数 φ(n)
            phi = n
            for p in factors:
                phi = phi * (p - 1) // p
            print(f"欧拉函数 φ({n}) = {phi}")
            
            print("-" * 40)

        # 分析一些有趣的数字
        numbers = [28, 60, 101, 1024, 1729]
        for num in numbers:
            analyze_number(num)
        ```

    总结

    - **SymPy** 是最通用和推荐的选择，因为它功能全面且稳定
    - 对于非常大的数字或需要高性能的场景，**gmpy2** 是不错的选择
    - **primefac** 提供了多种分解算法，适合研究使用
    - 对于学习目的，自己实现或使用简单的自定义函数更有帮助

    根据你的具体需求选择合适的工具包即可！

* Python 函数重载解决方案探讨

    在Python中，**没有传统的函数重载**（像C++/Java那样根据参数类型和数量自动选择不同实现）。但可以通过以下方式实现类似功能：

    1. 使用默认参数

        ```python
        def func(a, b=None, c=None):
            if b is None and c is None:
                return a * 2
            elif c is None:
                return a + b
            else:
                return a + b + c

        print(func(5))        # 10
        print(func(5, 3))     # 8
        print(func(5, 3, 2))  # 10
        ```

    2. 使用 `*args` 和 `**kwargs`

        ```python
        def process_data(*args, **kwargs):
            if len(args) == 1 and 'multiply' in kwargs:
                return args[0] * kwargs['multiply']
            elif len(args) == 2:
                return args[0] + args[1]
            else:
                return sum(args)
        ```

    3. 使用 `functools.singledispatch`（类型分派）

        ```python
        from functools import singledispatch

        @singledispatch
        def process(obj):
            return f"处理通用对象: {obj}"

        @process.register(str)
        def _(text):
            return f"处理字符串: {text.upper()}"

        @process.register(int)
        def _(num):
            return f"处理整数: {num * 2}"

        print(process("hello"))  # 处理字符串: HELLO
        print(process(10))       # 处理整数: 20
        print(process([1,2,3]))  # 处理通用对象: [1, 2, 3]
        ```

    4. 使用 `functools.singledispatchmethod`（类方法）

        ```python
        from functools import singledispatchmethod

        class Calculator:
            @singledispatchmethod
            def calculate(self, arg):
                raise NotImplementedError("不支持的类型")
            
            @calculate.register(int)
            def _(self, arg):
                return arg * 2
            
            @calculate.register(list)
            def _(self, arg):
                return sum(arg)
        ```

    5. 手动类型检查

        ```python
        def process_input(data):
            if isinstance(data, str):
                return data.upper()
            elif isinstance(data, (int, float)):
                return data * 2
            elif isinstance(data, list):
                return sum(data)
            else:
                raise TypeError("不支持的类型")
        ```

    最佳实践建议：

    1. **优先使用默认参数和可变参数** - 适合参数数量变化的情况

    2. **使用 `singledispatch`** - 适合需要根据参数类型执行不同逻辑的情况

    3. **保持简洁** - Python哲学强调"简单优于复杂"，避免过度设计

    4. **文档字符串很重要** - 明确说明函数支持的不同用法

        ```python
        def example(a, b=None, c=None):
            """
            支持多种调用方式：
            example(x)        -> 返回 x * 2
            example(x, y)     -> 返回 x + y
            example(x, y, z)  -> 返回 x + y + z
            """
            # 实现代码...
        ```

    Python通过动态类型和灵活的参数处理机制，提供了比传统函数重载更强大和灵活的解决方案。

* uv pip compile

    `uv pip compile pyproject.toml -o requirements.txt`

    这个过程很慢，可能需要几分钟。

    **创建 requirements.in 文件**

    ```bash
    # 创建基础依赖文件
    echo "torch" > requirements.in
    echo "numpy" >> requirements.in

    # 然后编译
    uv pip compile requirements.in -o requirements.txt
    ```

    查看帮助信息：`uv pip compile --help`

    **常见工作流程：**

    方式A：使用 pyproject.toml

    ```bash
    # 如果有 pyproject.toml 中的 dependencies 部分
    uv pip compile pyproject.toml -o requirements.txt

    # 或者针对特定组
    uv pip compile pyproject.toml --group dev -o requirements-dev.txt
    ```

    方式B：使用 requirements.in

    ```bash
    # 1. 创建 requirements.in 文件并添加依赖
    echo "torch>=2.0" > requirements.in
    echo "numpy" >> requirements.in

    # 2. 编译成 locked 版本
    uv pip compile requirements.in -o requirements.txt
    ```

    根据你的项目结构选择合适的命令即可。

* 简述 pip install -e . 的作用

    pip install -e . 是 Python 包管理中的一个常用命令，用于以"开发模式"安装当前目录的包。

    作用

    * 开发模式安装：将包以"可编辑"（editable）模式安装到 Python 环境中

    * 实时更新：对源代码的修改会立即生效，无需重新安装

    * 符号链接：创建一个指向源码目录的链接，而不是复制文件

    * 保持可导入：让包可以像普通安装的包一样被导入和使用

    工作原理

    当执行 pip install -e . 时：

    * 查找 setup.py 或 pyproject.toml：在当前目录查找包的配置文件

    * 创建链接：在 Python 的 site-packages 目录中创建一个 .egg-link 文件

    * 记录路径：该链接文件指向源代码的实际位置

    用法示例

    项目结构：

    ```text
    my_package/
    ├── setup.py          # 或 pyproject.toml
    ├── my_package/
    │   ├── __init__.py
    │   └── module.py
    └── README.md
    ```

    执行命令：

    ```bash
    # 进入项目根目录
    cd my_package

    # 以开发模式安装
    pip install -e .
    ```

    主要优势

    * 即时反馈：

        ```python
        # 修改源代码后立即生效
        # 修改 my_package/module.py 后
        import my_package
        # 无需重新安装，直接使用新代码
        ```

    * 保留开发结构：

        ```bash
        # 安装后，包文件仍在原位置
        # 可以继续使用版本控制（git）
        # 方便调试和测试
        ```

    * 支持依赖安装：

        ```python
        # setup.py 中的依赖会被安装
        # 例如：
        # install_requires=['numpy', 'pandas']
        ```

    setup.py 示例

    ```python
    from setuptools import setup, find_packages

    setup(
        name="my_package",
        version="0.1.0",
        packages=find_packages(),
        install_requires=[
            "numpy>=1.20.0",
            "pandas>=1.3.0",
        ],
    )
    ```

    pyproject.toml 示例

    ```toml
    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "my_package"
    version = "0.1.0"
    dependencies = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ]
    ```

    应用场景

    * 包开发：在开发 Python 包时使用

    * 团队协作：共享开发环境配置

    * 测试和调试：快速迭代和测试代码修改

    * 本地依赖：在多个相关项目间共享本地包

    检查安装状态

    ```bash
    # 查看已安装的包
    pip list

    # 开发模式包会显示路径
    # my-package      0.1.0    /path/to/my_package

    # 或使用 pip show
    pip show my-package
    # Location: /path/to/my_package
    ```

    注意事项

    * 需要 setup.py 或 pyproject.toml：没有这些文件会失败

    * 不同 pip 版本：新版 pip 优先使用 pyproject.toml

    * 虚拟环境：建议在虚拟环境中使用

    * 卸载：使用 pip uninstall package_name 卸载

    与普通安装的区别

    | 特性 | pip install . | pip install -e . |
    | - | - | - |
    | 安装方式 | 复制文件 | 创建链接 |
    | 修改源代码 | 需要重新安装 | 立即生效 |
    | 文件位置 | site-packages | 原目录 |
    | 开发便利性 | 低 | 高 |

    这个命令是 Python 包开发的标准工作流程，极大提高了开发效率。

* 为什么 python argparse 中，可以写成 type=int？类型也能赋值给一个变量吗？

    这是一个很好的问题！这涉及到 Python 中的一些核心概念：函数是一等公民（first-class functions）和 类型也是对象。

    1. 类型在 Python 中也是对象

        在 Python 中，类型（类）本身也是对象，可以赋值给变量、作为参数传递、作为返回值返回。

        ```python
        # int 本身就是一个类型对象
        print(type(int))      # <class 'type'>
        print(int)            # <class 'int'>

        # 可以赋值给变量
        MyInt = int
        x = MyInt("42")
        print(x, type(x))     # 42 <class 'int'>
        ```

    2. argparse 中 type 参数的工作原理

        argparse 的 add_argument() 方法的 type 参数期望一个可调用对象（callable），这个对象接收一个字符串参数并返回转换后的值。

        ```python
        import argparse

        parser = argparse.ArgumentParser()
        # int 在这里被当作一个函数使用
        parser.add_argument('--number', type=int)

        args = parser.parse_args(['--number', '42'])
        print(args.number)    # 42，已经是整数类型
        ```

        实际上，int("42") 会返回整数 42，所以 int 完美符合要求。

    3. 自定义类型转换函数

        你也可以使用自定义函数：

        ```python
        import argparse

        def percentage(string):
            """将 '50%' 转换为 0.5"""
            if string.endswith('%'):
                return float(string[:-1]) / 100
            return float(string)

        parser = argparse.ArgumentParser()
        parser.add_argument('--discount', type=percentage)

        args = parser.parse_args(['--discount', '30%'])
        print(args.discount)  # 0.3
        ```

    4. 实际使用中的等价形式

        以下三种写法是等价的：

        ```python
        # 1. 直接使用内置类型
        parser.add_argument('--num', type=int)

        # 2. 使用 lambda 函数
        parser.add_argument('--num', type=lambda x: int(x))

        # 3. 使用自定义函数
        def to_int(x):
            return int(x)

        parser.add_argument('--num', type=to_int)
        ```

    5. 关键点总结

        | 概念 | 解释 |
        | - | - |
        | 类型是对象 | int, str, float 等都是 type 类的实例 |
        | 类型是可调用的 | int("42") 会调用 int 的构造函数 |
        | type 参数要求 | 需要的是一个接收字符串并返回任意值的可调用对象 |
        | 灵活性 | 可以是内置类型、自定义函数、lambda 表达式、类等 |

    6. 更复杂的例子

        ```python
        import argparse

        # 使用类作为 type（因为类是可调用的）
        class Range:
            def __init__(self, string):
                start, end = map(int, string.split('-'))
                self.start = start
                self.end = end
            
            def __repr__(self):
                return f"Range({self.start}-{self.end})"

        parser = argparse.ArgumentParser()
        parser.add_argument('--range', type=Range)

        args = parser.parse_args(['--range', '10-20'])
        print(args.range)  # Range(10-20)
        ```

    所以，type=int 能工作是因为：

    * int 是一个类型对象

    * int 是可调用的（int("42") 返回整数 42）

    * argparse 只需要一个接收字符串并返回值的可调用对象

    这正是 Python 动态性和灵活性的体现！

* re 的其他常用模式

    1. re.IGNORECASE / re.I - 忽略大小写

        ```python
        re.compile('hello', re.I)  # 匹配 hello, Hello, HELLO, hElLo 等
        ```

    2. re.DOTALL / re.S - 点匹配所有字符

        ```python
        # 默认情况：. 不匹配换行符 \n
        pat1 = re.compile('a.b')
        print(pat1.search('a\nb'))  # None

        # DOTALL 模式：. 匹配包括 \n 在内的所有字符
        pat2 = re.compile('a.b', re.DOTALL)
        print(pat2.search('a\nb'))  # 匹配成功
        ```

    3. re.VERBOSE / re.X - 详细模式（方便注释）

        ```python
        pat = re.compile(r'''
            \d{3,4}   # 区号
            -         # 分隔符
            \d{7,8}   # 号码
        ''', re.VERBOSE)
        ```

    4. re.ASCII / re.A - 限制为 ASCII 字符

        ```python
        # 默认：\w, \W, \b, \B, \d, \D, \s, \S 匹配 Unicode
        # ASCII 模式：只匹配 ASCII 字符
        re.compile(r'\w+', re.ASCII)  # \w 只匹配 [a-zA-Z0-9_]
        ```

    5. re.LOCALE / re.L - 本地化模式（已弃用）

        * 使 \w, \W, \b, \B 依赖当前区域设置

        * Python 3.6+ 不推荐使用

    6. re.DEBUG - 调试模式

        ```python
        pat = re.compile(r'\d+', re.DEBUG)
        # 输出编译过程的调试信息
        ```

    模式组合使用

    多个模式可以用 | 组合：

    ```python
    # 同时启用多行模式和忽略大小写
    pat = re.compile('^hello', re.MULTILINE | re.IGNORECASE)

    # 同时启用 DOTALL 和 VERBOSE
    pat = re.compile(r'''
        .+      # 任意字符（包括换行）
        \d+     # 数字
    ''', re.DOTALL | re.VERBOSE)
    ```

    常用模式总结表

    | 模式 | 常量名 | 简写 | 作用 |
    | - | - | - | - |
    | 多行模式 | re.MULTILINE | re.M | 让 ^ 和 $ 匹配每行的开始/结束 |
    | 忽略大小写 | re.IGNORECASE | re.I | 大小写不敏感匹配 |
    | 点匹配所有 | re.DOTALL | re.S | 让 . 匹配包括换行符的所有字符 |
    | 详细模式 | re.VERBOSE | re.X | 允许在正则中添加空白和注释 |
    | ASCII 模式 | re.ASCII | re.A | 让 \w, \d 等只匹配 ASCII 字符 |

    实际应用示例：

    ```python
    # 提取多行日志中所有以 ERROR 开头的行
    log_text = """INFO: System started
    ERROR: Database connection failed
    WARNING: High memory usage
    ERROR: File not found"""

    pattern = re.compile(r'^ERROR:.*', re.MULTILINE)
    errors = pattern.findall(log_text)
    # 结果：['ERROR: Database connection failed', 'ERROR: File not found']
    ```

    这些模式可以根据需要灵活组合，解决不同的文本匹配问题。

* python `re`模块多行模式`re.MULTILINE`

    作用： 改变 ^ 和 $ 的行为

    * 默认情况：

        * ^ 只匹配整个字符串开头

        * $ 只匹配整个字符串结尾（或末尾的换行符前）

    * 启用 MULTILINE 后：

        * ^ 匹配每一行的开头（字符串开头 + \n 后的位置）

        * $ 匹配每一行的结尾（字符串结尾 + \n 前的位置）

    example 1：

    ```python
    import re

    txt = 'first line\nsecond line\nthird line'

    # 默认模式
    pat1 = re.compile('^second')
    print(pat1.search(txt))  # None，因为 ^ 只匹配整个字符串开头

    # 多行模式
    pat2 = re.compile('^second', re.MULTILINE)
    print(pat2.search(txt))  # 匹配成功，因为 ^ 能匹配第二行开头
    ```

    example 2:

    ```py
    pat = re.compile('^world', re.MULTILINE)  # 多行模式
    txt = 'hello, world\nworld again'
    m = pat.match(txt, pos=12)  # 可以匹配第二行的 world
    ```

    这几行代码使用Python正则表达式的多行模式来匹配以"world"开头的行：

    * re.compile('^world', re.MULTILINE)

        * 编译正则表达式，^通常匹配字符串开头

        * re.MULTILINE标志让^也能匹配每一行的开头

    * txt = 'hello, world\nworld again'

        * 创建一个包含两行的字符串

        * 第一行：hello, world

        * 第二行：world again

    * pat.match(txt, pos=12)

        * 从位置12开始匹配（即第二行的开头）

        * 由于是多行模式，^world可以匹配第二行开头的"world"

        * 结果会成功匹配到第二行的"world"

    关键理解：

    * 没有re.MULTILINE时，^只能匹配整个字符串的开头（位置0）

    * 使用re.MULTILINE后，^可以匹配每行的开头（\n后的位置）

    * pos=12直接跳到第二行开头，所以匹配到的是第二行的"world"

    如果去掉pos=12参数，match()会从字符串开头尝试匹配，但由于第一行不是以"world"开头，所以会返回None。

* python 如何加载/保存 matlab save() 保存的 binary 文件？

    Python 加载/保存 MATLAB 的 .mat 文件主要通过 scipy.io <https://scipy.io/> 模块实现，以下是详细方法：

    1. 基础方法：使用 scipy.io

        安装依赖

        ```bash
        pip install scipy numpy
        ```

        加载MATLAB文件

        ```python
        import scipy.io as sio
        import numpy as np

        # 加载.mat文件
        mat_data = sio.loadmat('data.mat')

        # mat_data是一个字典，MATLAB变量名为键
        print(mat_data.keys())  # 查看所有变量名

        # 访问具体变量
        matrix_data = mat_data['variable_name']  # 替换为实际变量名
        array_data = mat_data['array_name']

        # 查看变量信息
        print(f"数据类型: {type(matrix_data)}")
        print(f"数据形状: {matrix_data.shape}")
        print(f"数据类型: {matrix_data.dtype}")
        ```

        保存到MATLAB文件

        ```python
        import scipy.io as sio
        import numpy as np

        # 创建数据
        matrix1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        matrix2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float64)
        string_var = "Hello MATLAB"
        scalar_var = 42.5

        # 保存为.mat文件
        sio.savemat('output.mat', {
            'matrix1': matrix1,
            'matrix2': matrix2,
            'my_string': string_var,
            'scalar_value': scalar_var
        })
        ```

    2. 高级功能和注意事项

        处理不同MATLAB版本

        ```python
        # MATLAB v7.3格式（HDF5格式）需要h5py
        import h5py

        # 加载v7.3格式
        with h5py.File('data_v73.mat', 'r') as f:
            data = f['variable_name'][:]  # 注意索引方式不同

        # 保存时指定版本
        sio.savemat('output_v7.mat', {'data': my_data}, do_compression=True)  # v7
        sio.savemat('output_v73.mat', {'data': my_data}, do_compression=True, 
                    format='7.3')  # v7.3
        ```

        处理复杂数据类型

        ```python
        # 结构体（struct）处理
        mat_data = sio.loadmat('struct_data.mat', squeeze_me=True)
        struct_data = mat_data['my_struct']

        # 元胞数组（cell array）
        cell_array = mat_data['my_cell']

        # 对于嵌套结构体，可能需要递归处理
        def unpack_mat_struct(struct_data):
            if isinstance(struct_data, np.ndarray) and struct_data.dtype.names:
                # 这是一个结构体数组
                result = {}
                for name in struct_data.dtype.names:
                    result[name] = unpack_mat_struct(struct_data[name])
                return result
            return struct_data
        ```

        读取选项参数

        ```python
        # 常用参数
        mat_data = sio.loadmat('data.mat', 
                               mat_dtype=False,      # 不强制转换为matlab double
                               squeeze_me=True,      # 压缩单元素维度
                               struct_as_record=True, # 结构体作为记录数组
                               verify_compressed_data_nonempty=False,
                               variable_names=None)   # 指定读取的变量名
        ```

    3. 实用工具函数

        批量转换脚本

        ```python
        import os
        import scipy.io as sio
        import numpy as np

        def convert_mat_to_npy(mat_dir, output_dir):
            """将目录下所有.mat文件转换为.npy格式"""
            os.makedirs(output_dir, exist_ok=True)
            
            for filename in os.listdir(mat_dir):
                if filename.endswith('.mat'):
                    mat_path = os.path.join(mat_dir, filename)
                    data = sio.loadmat(mat_path)
                    
                    # 为每个变量单独保存
                    for key, value in data.items():
                        if not key.startswith('__'):  # 跳过内部变量
                            npy_filename = f"{filename[:-4]}_{key}.npy"
                            npy_path = os.path.join(output_dir, npy_filename)
                            np.save(npy_path, value)
                            print(f"Saved {npy_filename}")

        def save_to_mat_with_metadata(data_dict, filename, metadata=None):
            """保存数据并添加元数据"""
            if metadata:
                # 将元数据作为变量保存
                data_dict['_metadata'] = metadata
            
            sio.savemat(filename, data_dict, 
                        do_compression=True,
                        long_field_names=True)
        ```

    4. 常见问题解决

        内存优化

        ```python
        # 部分读取大文件
        import h5py

        def read_large_mat_partial(filename, variable_name, start_idx, end_idx):
            """分块读取大MAT文件"""
            with h5py.File(filename, 'r') as f:
                dataset = f[variable_name]
                return dataset[start_idx:end_idx]
        ```

        数据类型转换

        ```python
        def convert_matlab_types(mat_data):
            """转换MATLAB数据类型到Python友好格式"""
            converted = {}
            
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    # 处理复数
                    if np.iscomplexobj(value):
                        value = value.astype(np.complex128)
                    # 处理逻辑类型
                    elif value.dtype == np.bool_ or value.dtype == bool:
                        value = value.astype(bool)
                    # 处理字符串/cell中的字符串
                    elif isinstance(value, np.ndarray) and value.dtype.char == 'U':
                        value = str(value.item())
                    
                    converted[key] = value
            
            return converted
        ```

        处理特殊字符和编码

        ```python
        # 处理中文字符
        def load_mat_with_encoding(filename, encoding='utf-8'):
            """处理包含非ASCII字符的MAT文件"""
            # scipy.io.loadmat通常能正确处理编码
            data = sio.loadmat(filename)
            
            # 手动处理字符串字段
            for key in data:
                if isinstance(data[key], str):
                    try:
                        data[key] = data[key].encode('latin1').decode(encoding)
                    except:
                        pass
            
            return data
        ```

    5. 完整示例

        ```python
        import scipy.io as sio
        import numpy as np
        import matplotlib.pyplot as plt

        # 示例：加载、处理、保存MATLAB数据
        def process_matlab_data(input_file, output_file):
            # 1. 加载数据
            mat_data = sio.loadmat(input_file)
            
            # 2. 提取变量
            signals = mat_data.get('signals', None)
            time_vector = mat_data.get('time', None)
            sampling_rate = mat_data.get('fs', 1000)  # 默认采样率
            
            # 3. 数据处理
            if signals is not None:
                # 计算均值
                mean_signal = np.mean(signals, axis=0)
                
                # 滤波等处理
                # ...
                
                # 4. 保存结果
                sio.savemat(output_file, {
                    'original_signals': signals,
                    'processed_signals': mean_signal,
                    'time': time_vector,
                    'fs': sampling_rate,
                    'processing_info': f"Processed with Python on {np.datetime64('now')}"
                })
                
                print(f"Data saved to {output_file}")
                return True
            
            return False

        # 使用示例
        process_matlab_data('input_data.mat', 'processed_data.mat')
        ```

    重要提示

    * 版本兼容性：MATLAB v7.3及以上使用HDF5格式，需要h5py库

    * 数据类型：注意MATLAB和Python数据类型的差异

    * 内存管理：大文件使用format='7.3'和HDF5格式更高效

    * Unicode：MATLAB字符串可能使用不同编码，可能需要转换

    * 稀疏矩阵：使用scipy.sparse处理稀疏矩阵

    这些方法基本覆盖了Python与MATLAB数据交换的所有常见需求。

* 简述 python 处理 csv 文件的方法

    Python处理CSV文件的常用方法主要有以下几种：

    1. 使用csv模块（标准库）

        基本读写操作

        ```python
        import csv

        # 读取CSV文件
        with open('data.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)

        # 读取为字典格式
        with open('data.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print(row['column_name'])

        # 写入CSV文件
        data = [['Name', 'Age'], ['Alice', 25], ['Bob', 30]]
        with open('output.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        ```

    2. 使用pandas库（最常用）

        ```python
        import pandas as pd

        # 读取CSV
        df = pd.read_csv('data.csv')
        print(df.head())  # 查看前几行
        print(df.info())  # 查看数据信息

        # 数据处理
        df['new_column'] = df['old_column'] * 2  # 新增列
        filtered_df = df[df['age'] > 25]         # 过滤数据
        grouped = df.groupby('category').mean()   # 分组聚合

        # 写入CSV
        df.to_csv('output.csv', index=False, encoding='utf-8')
        ```

    3. 使用numpy库（数值数据处理）

        ```python
        import numpy as np

        # 读取数值数据
        data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
        # 或
        data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

        # 处理数值数据
        mean_values = np.mean(data, axis=0)
        ```

    4. 使用内置open函数（简单情况）

        ```python
        # 简单读取
        with open('data.csv', 'r', encoding='utf-8') as f:
            for line in f:
                columns = line.strip().split(',')
                print(columns)

        # 简单写入
        data = [['Name', 'Age'], ['Alice', '25']]
        with open('output.csv', 'w', encoding='utf-8') as f:
            for row in data:
                f.write(','.join(map(str, row)) + '\n')
        ```

    常用参数说明

    pandas读取参数：

    * sep / delimiter: 分隔符（默认','）

    * header: 指定表头行（默认0）

    * index_col: 指定索引列

    * usecols: 指定读取的列

    * dtype: 指定列数据类型

    * na_values: 指定缺失值表示

    * encoding: 文件编码（如'utf-8'、'gbk'）

    写入参数：

    * index: 是否写入索引（默认True）

    * columns: 指定写入的列

    * encoding: 输出编码

    选择建议

    * 简单操作 → 使用csv模块

    * 数据分析和复杂处理 → 使用pandas（推荐）

    * 纯数值数据处理 → 使用numpy

    * 大文件处理 → 使用csv模块或pandas的chunksize参数

    * 内存限制 → 使用csv模块逐行处理

    处理大文件示例

    ```python
    # pandas分块读取
    chunk_size = 10000
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        process(chunk)  # 逐块处理

    # csv模块逐行处理
    with open('large_file.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            process_row(row)  # 逐行处理
    ```

    这些方法覆盖了大多数CSV文件处理场景，pandas通常是最方便的选择。

* pip 使用指定的 cache path 和 tmp path

    ```bash
    TMPDIR=/share/hlc/pip_tmp pip3 install --cache-dir=/share/hlc/pip_cache cupy -i https://mirrors.aliyun.com/pypi/simple
    ```

    pip 默认会使用 home dir 和 /tmp 进行下载包、解压、编译安装（具体哪个目录对应哪个功能，目前分不太清楚）。如果 home 目录已满，或者没有 /tmp 权限，那么可以手动指定这两个目录，如上所示。

    注意，
    
    1. `TMPDIR`必须是绝对路径。
    
    1. 这两个目录的权限最好设置为 777.

* 简述 python 包管理器 uv 的用法

    uv 是一款用 Rust 开发的快速 Python 包管理器和项目工具，由 Astral 开发。以下是其主要用法：

    **🚀 安装 uv**

    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows (PowerShell)
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

    # pip 安装
    pip install uv
    ```

    **📦 核心功能**

    1. 创建虚拟环境

        ```bash
        # 创建并激活虚拟环境
        uv venv
        source .venv/bin/activate  # Linux/macOS
        .venv\Scripts\activate     # Windows

        # 指定 Python 版本
        uv venv --python 3.11
        ```

    2. 依赖管理

        ```bash
        # 初始化项目（类似 npm init）
        uv init

        # 安装依赖
        uv add requests           # 添加到 pyproject.toml 并安装
        uv add pytest --dev       # 安装开发依赖
        uv add "django>=4.0"      # 指定版本

        # 安装 pyproject.toml 中的所有依赖
        uv sync

        # 更新所有依赖
        uv sync --upgrade

        # 移除依赖
        uv remove requests
        ```

    3. 运行 Python 代码

        ```bash
        # 直接运行（自动创建临时环境）
        uv run python script.py
        uv run pytest tests/
        ```

    4. 项目管理工具

        ```bash
        # 锁定依赖版本
        uv lock

        # 生成 requirements.txt
        uv pip compile -o requirements.txt
        uv pip compile -o requirements-dev.txt --extra dev

        # 从 requirements.txt 安装
        uv pip install -r requirements.txt
        ```

    ⚡ 高级用法

    并行安装

    ```bash
    uv pip install package1 package2 package3  # 并行下载安装
    ```

    项目模板

    ```bash
    # 从模板创建项目
    uv init --template https://github.com/username/template
    ```

    集成现有项目

    ```bash
    # 在已有项目中初始化
    cd existing-project
    uv init --no-create-project
    ```

    **🔧 配置选项**

    配置文件 (uv.toml)

    ```toml
    [project]
    name = "my-project"
    version = "0.1.0"

    [tool.uv]
    python = "3.11"
    ```

    环境变量

    ```bash
    # 设置镜像源
    export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

    # 启用并行下载
    export UV_PARALLEL=8
    ```

    **🆚 与传统工具对比**

    | 功能 | uv | pip | pip+venv |
    | - | - | - | - |
    | 虚拟环境管理 | ✅ | ❌ | ✅ |
    | 并行下载 | ✅ | ❌ | ❌ |
    | 项目初始化 | ✅ | ❌ | ❌ |
    | 依赖锁定 | ✅ | ❌ | ❌ |
    | 运行命令 | ✅ | ❌ | ❌ |

    **📝 示例工作流**

    ```bash
    # 1. 创建新项目
    uv init my-project
    cd my-project

    # 2. 添加依赖
    uv add fastapi
    uv add pytest --dev

    # 3. 编写代码后运行
    uv run python main.py

    # 4. 运行测试
    uv run pytest

    # 5. 生成生产环境需求文件
    uv pip compile --no-dev -o requirements.txt
    ```

    **💡 小贴士**

    * 极速安装：uv 的缓存机制和并行下载使其比 pip 快很多

    * 一体化工具：结合了 pip、venv、virtualenv、pip-tools 的功能

    * 跨平台：Windows、macOS、Linux 全支持

    * 向后兼容：兼容现有的 requirements.txt 和 pyproject.toml

    uv 的目标是成为 Python 项目的“一站式”工具，简化从项目创建到部署的整个流程。

* python 去重方法总结

    Python 中有多种去重方法，针对你提到的数据类型，需要不同处理方式：

    1. 基础列表去重（可哈希元素）

        ```python
        arr_1 = ['a', 'b', 'c', 'a']

        # 方法1：使用set（无序）
        arr_unique = list(set(arr_1))  # ['a', 'b', 'c'] 顺序可能改变

        # 方法2：保持原始顺序
        arr_unique = []
        for item in arr_1:
            if item not in arr_unique:
                arr_unique.append(item)
        # 或使用字典（Python 3.7+ 保证顺序）
        arr_unique = list(dict.fromkeys(arr_1))  # ['a', 'b', 'c']

        # 方法3：使用sorted排序
        arr_unique = sorted(set(arr_1), key=arr_1.index)  # ['a', 'b', 'c']
        ```

    2. 字典列表去重（不可哈希元素）

        字典是不可哈希的，不能直接用 set()，需要特殊处理：

        ```python
        arr_2 = [{'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 2, 'b': 3}]

        # 方法1：使用JSON字符串作为中间形式（简单但有限制）
        import json
        def deduplicate_dicts(arr):
            seen = set()
            result = []
            for d in arr:
                # 先排序键保证一致性
                sorted_str = json.dumps(d, sort_keys=True)
                if sorted_str not in seen:
                    seen.add(sorted_str)
                    result.append(d)
            return result

        arr_unique = deduplicate_dicts(arr_2)
        # [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}]

        # 方法2：转换为可哈希的元组
        def deduplicate_dicts_v2(arr):
            seen = set()
            result = []
            for d in arr:
                # 转换为排序后的键值元组
                t = tuple(sorted(d.items()))
                if t not in seen:
                    seen.add(t)
                    result.append(d)
            return result

        # 方法3：使用Pandas（大数据集更高效）
        import pandas as pd
        df = pd.DataFrame(arr_2)
        df_unique = df.drop_duplicates().to_dict('records')
        ```

    3. 使用工具函数

        ```python
        from typing import List, Any, Hashable
        import json

        def unique(sequence, keep_order=True):
            """
            通用去重函数
            """
            if not sequence:
                return []
            
            # 如果元素可哈希
            try:
                if keep_order:
                    return list(dict.fromkeys(sequence))
                else:
                    return list(set(sequence))
            except TypeError:
                # 处理不可哈希元素（如字典）
                if keep_order:
                    seen = set()
                    result = []
                    for item in sequence:
                        if isinstance(item, dict):
                            # 字典转换为可哈希的字符串
                            key = json.dumps(item, sort_keys=True)
                        else:
                            key = str(item)  # 其他不可哈希类型
                        
                        if key not in seen:
                            seen.add(key)
                            result.append(item)
                    return result
                else:
                    # 不保持顺序
                    unique_dicts = {}
                    for item in sequence:
                        key = json.dumps(item, sort_keys=True)
                        unique_dicts[key] = item
                    return list(unique_dicts.values())

        # 使用示例
        print(unique(arr_1))  # ['a', 'b', 'c']
        print(unique(arr_2))  # [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}]
        ```

    4. 性能比较和建议

        | 方法 | 优点 | 缺点 | 适用场景 |
        | - | - | - | - |
        | set() | 速度快，简洁 | 不保持顺序 | 简单列表，不关心顺序 |
        | dict.fromkeys() | 保持顺序，较快 | 只适用于可哈希元素 | 需要保持顺序的简单列表 |
        | 手动循环检查 | 灵活可控 | 速度较慢 | 小数据集，复杂逻辑 |
        | JSON转换 | 处理复杂类型 | JSON限制（如不能序列化所有类型） | 字典列表去重 |

        推荐做法：

        * 简单列表：list(dict.fromkeys(arr))（保持顺序）

        * 字典列表：使用元组转换或JSON字符串方法

        * 大数据集：考虑使用Pandas

    注意事项：

    * 字典去重要注意键的顺序问题

    * 嵌套字典需要递归处理

    * 自定义对象需要实现 __hash__ 和 __eq__ 方法

* py 中的`f.write()`接受变参数吗，可以写入多个 str 吗

    f.write() 不接受变参数，它只能接受一个字符串参数。但你可以通过多种方式写入多个字符串：

    1. 基本用法（一次一个字符串）

        ```python
        with open('test.txt', 'w') as f:
            f.write("Hello ")      # 一次写入一个字符串
            f.write("World")       # 再次写入
            f.write("\n")          # 写入换行符
        # 文件内容: Hello World\n
        ```

    2. 实现写入多个字符串的方法

        方法1：使用字符串拼接

        ```python
        with open('test.txt', 'w') as f:
            f.write("Hello " + "World" + "!")  # 拼接后再写入
        # 或
        strings = ["Hello", " ", "World", "!"]
        with open('test.txt', 'w') as f:
            f.write(''.join(strings))  # 使用join拼接
        ```

        方法2：使用 writelines() 方法

        ```python
        with open('test.txt', 'w') as f:
            f.writelines(["Hello ", "World", "!\n"])
        # 注意：writelines不会自动添加换行符
        ```

        方法3：使用 print() 函数

        ```python
        with open('test.txt', 'w') as f:
            print("Hello", "World", sep=" ", file=f)  # sep控制分隔符
            # 相当于 f.write("Hello World")
        ```

        方法4：自定义包装函数

        ```python
        def write_many(f, *args, sep="", end="\n"):
            """模拟支持多个参数的write函数"""
            content = sep.join(str(arg) for arg in args) + end
            f.write(content)

        with open('test.txt', 'w') as f:
            write_many(f, "Hello", "World", "!")  # HelloWorld!\n
            write_many(f, "Line", 2, sep=" ")     # Line 2\n
        ```

    3. 性能考虑

        ```python
        import time

        # 测试不同方法的性能
        def test_performance():
            test_data = ["str" + str(i) for i in range(10000)]
            
            # 方法1：多次 write
            start = time.time()
            with open('test1.txt', 'w') as f:
                for s in test_data:
                    f.write(s)
            time1 = time.time() - start
            
            # 方法2：一次性写入
            start = time.time()
            with open('test2.txt', 'w') as f:
                f.write(''.join(test_data))
            time2 = time.time() - start
            
            print(f"多次write: {time1:.4f}秒")
            print(f"一次性写入: {time2:.4f}秒")
            # 通常一次性写入更快，因为减少了I/O操作次数
        ```

    4. 实际应用示例

        ```python
        # 日志写入
        def write_log(filename, *messages, level="INFO", timestamp=True):
            import datetime
            with open(filename, 'a') as f:
                if timestamp:
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp_str}] [{level}] ")
                
                # 拼接所有消息
                full_message = ' '.join(str(msg) for msg in messages)
                f.write(full_message + "\n")

        # 使用
        write_log("app.log", "User", "John", "logged in", "successfully")
        # 文件内容: [2024-01-22 10:30:00] [INFO] User John logged in successfully
        ```

    5. 类型安全建议

        ```python
        def safe_write(f, *args, sep="", end="\n"):
            """安全写入，自动处理类型转换"""
            try:
                # 将所有参数转换为字符串
                str_args = (str(arg) for arg in args)
                content = sep.join(str_args) + end
                f.write(content)
            except Exception as e:
                print(f"写入失败: {e}")

        with open('output.txt', 'w') as f:
            safe_write(f, "Number:", 42, "List:", [1, 2, 3], sep=" ")
            # 输出: Number: 42 List: [1, 2, 3]\n
        ```

    总结

    | 方法 | 优点 | 缺点 | 适用场景 |
    | - | - | - | - |
    | f.write(str) | 标准方法，直接 | 只接受一个参数 | 简单写入 |
    | 字符串拼接 | 灵活控制格式 | 需要手动拼接 | 格式复杂的写入 |
    | f.writelines() | 可传入列表 | 不自动加分隔符 | 批量写入字符串列表 |
    | print() | 自动加换行，支持多个参数 | 输出格式固定 | 类似控制台输出的格式 |
    | 自定义函数 | 完全自定义行为 | 需要额外代码 | 特定业务需求 |

    推荐做法：

    * 如果需要写入多个字符串，先使用 join() 拼接

    * 如果格式类似 print()，直接使用 print(..., file=f)

    * 如果需要频繁写入，考虑批量处理减少I/O操作

    * 对于特定需求，封装自己的写入函数

    记住：Python 的 write() 设计简单是为了保持函数单一职责原则，更复杂的写入逻辑应该由调用者处理。

* python re 模块中 match() 与 search() 的区别

    ```py
    import re

    txt = 'hello, world'
    pat = re.compile('world')
    print('text: {}'.format(txt))
    print('pattern: {}'.format(pat.pattern))
    print()

    m = pat.match(txt)
    print('-------- match test 1 --------')
    print('start pos: 0, match result:')
    print(m)  # None
    print()

    m = pat.match(txt, pos=7)
    print('-------- match test 2 --------')
    print('start pos: 7, match result:')
    print(m)
    start_pos = m.start()
    end_pos = m.end()
    selected_txt = txt[start_pos:end_pos]
    print('selected text: {}'.format(selected_txt))  # world
    ```

    output:

    ```
    text: hello, world
    pattern: world

    -------- match test 1 --------
    start pos: 0, match result:
    None

    -------- match test 2 --------
    start pos: 7, match result:
    <re.Match object; span=(7, 12), match='world'>
    selected text: world
    ```

    可以看出，`match()`是从指定 pos 开始匹配，并且要求 pos 的第一个字符就要和 pattern 匹配。而`search()`是从 pos 开始向后搜索，直到找到和 pattern 匹配的子串为止。

    `match()`从 pos 处开始匹配，不代表这里就是行首。即对于字符串`hello, world`，指定 match() 时 pos 为 7，pattern 设置为`^world`，那么会返回 None。`^`只匹配真正的行首，不认为 match() 设置的 pos 为行首。

* python 抽象基类与其他语言的抽象基类对比

    1. Python的抽象基类（ABC）

        ```python
        from abc import ABC, abstractmethod
        from typing import Protocol

        # 方法1：使用ABC基类
        class DataProcessor(ABC):
            @abstractmethod
            def load_data(self):
                """必须实现的方法"""
                pass
            
            @abstractmethod
            def process_data(self):
                """必须实现的方法"""
                pass
            
            # 基类提供默认实现的功能
            def run(self):
                """利用子类实现的方法提供完整功能"""
                data = self.load_data()
                result = self.process_data(data)
                return self.save_result(result)
            
            def save_result(self, result):
                """可选的钩子方法"""
                # 默认实现
                return f"Saved: {result}"

        # 用户实现
        class CSVProcessor(DataProcessor):
            def load_data(self):
                return "CSV data"
            
            def process_data(self, data):
                return f"Processed {data}"
        ```

    2. Python的Protocol（类型提示）

        ```python
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class ProcessorProtocol(Protocol):
            def preprocess(self) -> str: ...
            def transform(self, data: str) -> str: ...
            
            # 注意：Protocol本身不提供实现，只是定义接口

        class BaseProcessor:
            """基类可以基于Protocol实现通用逻辑"""
            def execute_pipeline(self, processor: ProcessorProtocol):
                """保证processor有preprocess和transform方法"""
                data = processor.preprocess()
                result = processor.transform(data)
                return self.finalize(result)
        ```

    3. 其他语言的类似机制

        Java - 接口和抽象类

        ```java
        // 接口
        public interface Processor {
            void load();
            void process();
        }

        // 抽象类
        abstract class AbstractProcessor implements Processor {
            public abstract void load();
            public abstract void process();
            
            // 模板方法
            public void run() {
                load();
                process();
                cleanup();
            }
            
            protected void cleanup() {
                // 默认实现
            }
        }
        ```

    Go - 接口

    ```go
    type Processor interface {
        Load() error
        Process() error
    }

    // 提供通用函数
    func RunProcessor(p Processor) error {
        if err := p.Load(); err != nil {
            return err
        }
        return p.Process()
    }
    ```

    TypeScript - 抽象类和接口

    ```typescript
    abstract class DataHandler {
        abstract fetch(): Promise<any>;
        abstract transform(data: any): any;
        
        async execute(): Promise<any> {
            const data = await this.fetch();
            return this.transform(data);
        }
    }
    ```

    4. 设计模式：模板方法模式

        ```python
        class TemplateProcessor:
            """模板方法模式的经典实现"""
            def process_pipeline(self):
                # 固定流程
                self.setup()           # 可选的钩子
                data = self.extract()  # 必须实现
                transformed = self.transform(data)  # 必须实现
                self.cleanup()         # 可选的钩子
                return transformed
            
            def setup(self):
                """钩子方法，子类可覆盖"""
                pass
            
            def extract(self):
                """必须实现的方法"""
                raise NotImplementedError
            
            def transform(self, data):
                """必须实现的方法"""
                raise NotImplementedError
            
            def cleanup(self):
                """钩子方法，子类可覆盖"""
                pass
        ```

    5. 实际应用示例

        ```python
        from abc import ABC, abstractmethod
        from dataclasses import dataclass

        class PluginBase(ABC):
            """插件系统的基础类"""
            
            @abstractmethod
            def validate_config(self, config: dict) -> bool:
                """验证配置"""
                pass
            
            @abstractmethod
            def execute(self, input_data) -> any:
                """执行核心逻辑"""
                pass
            
            def run(self, config: dict, input_data) -> any:
                """模板方法：完整的执行流程"""
                if not self.validate_config(config):
                    raise ValueError("Invalid configuration")
                
                self.on_start()
                result = self.execute(input_data)
                self.on_complete(result)
                return result
            
            def on_start(self):
                """钩子：开始执行时的回调"""
                print(f"Starting {self.__class__.__name__}")
            
            def on_complete(self, result):
                """钩子：完成时的回调"""
                print(f"Completed with result: {result}")

        # 用户实现
        class EmailPlugin(PluginBase):
            def validate_config(self, config: dict) -> bool:
                return "recipient" in config and "@" in config["recipient"]
            
            def execute(self, input_data):
                return f"Email sent: {input_data}"
        ```

    这种机制的核心思想是：

    * 定义契约：通过抽象方法定义必须实现的方法

    * 提供模板：在基类中实现通用的流程控制

    * 保证功能：只要用户实现了必要方法，就能获得完整功能

    * 允许扩展：通过钩子方法让用户定制特定步骤

    Python 的 ABC 是最常用的实现方式，它在运行时强制检查子类是否实现了所有抽象方法。

* python 如何删除一个文件？

    在 Python 中删除文件有多种方法，以下是几种常用的方式：

    1. 使用 os.remove() - 最常用的方法

        ```python
        import os

        # 删除文件
        file_path = "example.txt"
        os.remove(file_path)
        print(f"文件 {file_path} 已删除")
        ```

    2. 使用 os.unlink() - 与 os.remove() 功能相同

        ```python
        import os

        file_path = "example.txt"
        os.unlink(file_path)
        print(f"文件 {file_path} 已删除")
        ```

    3. 使用 pathlib.Path.unlink() (Python 3.4+)

        ```python
        from pathlib import Path

        # 删除文件
        file_path = Path("example.txt")
        file_path.unlink()
        print(f"文件 {file_path} 已删除")
        ```

    4. 完整的示例代码（包含错误处理）

        ```python
        import os
        from pathlib import Path

        def delete_file_safe(file_path):
            """安全删除文件，包含错误处理"""
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"文件 {file_path} 删除成功")
                else:
                    print(f"文件 {file_path} 不存在")
            except PermissionError:
                print(f"权限错误：无法删除 {file_path}")
            except OSError as e:
                print(f"系统错误：无法删除 {file_path} - {e}")

        # 使用示例
        delete_file_safe("example.txt")
        ```

    5. 删除前检查文件是否存在

        ```python
        import os

        file_path = "example.txt"

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"文件 {file_path} 已删除")
        else:
            print(f"文件 {file_path} 不存在")
            ```

    6. 使用 pathlib 的完整示例

        ```python
        from pathlib import Path

        file_path = Path("example.txt")

        if file_path.exists():
            file_path.unlink()
            print(f"文件 {file_path} 已删除")
        else:
            print(f"文件 {file_path} 不存在")
         ```

    注意事项：

    * 权限问题：确保程序有删除文件的权限

    * 文件占用：如果文件正在被其他程序使用，删除可能会失败

    * 路径问题：使用绝对路径或确保相对路径正确

    * 删除不可恢复：文件删除后通常无法恢复

    推荐做法：

    对于现代 Python 代码，推荐使用 pathlib 方法，因为它提供了更面向对象和可读性更好的 API。

    ```python
    from pathlib import Path

    def delete_file(file_path):
        path = Path(file_path)
        if path.is_file():  # 确保是文件而不是目录
            path.unlink()
            return True
        return False
        ```

* Python Type Hint

    Python Type Hint（类型提示）是在 Python 3.5+ 中引入的静态类型检查辅助工具，它允许开发者为变量、函数参数和返回值等标注期望的类型，以提高代码的可读性、可维护性，并借助工具（如 mypy）进行静态类型检查。

    **基本语法**

    ```python
    # 变量注解
    name: str = "Alice"
    count: int = 10

    # 函数参数和返回值注解
    def greet(name: str) -> str:
        return f"Hello, {name}"
    ```

    **常用类型**

    * 基础类型：int, str, float, bool, bytes

    * 复合类型：

        * List[int]：整数列表（需 from typing import List）

        * Dict[str, int]：键为字符串、值为整数的字典

        * Tuple[int, str]：固定长度的元组

        * Optional[int]：可能为 int 或 None（等价于 Union[int, None]）

        * Union[int, str]：可以是 int 或 str

        * Any：任意类型（禁用静态类型检查）

    **Python 3.9+ 简化写法**

    从 Python 3.9 开始，可以直接使用内置类型代替 typing 中的泛型：

    ```python
    # Python 3.9+
    list[int]          # 代替 List[int]
    dict[str, int]     # 代替 Dict[str, int]
    tuple[int, ...]    # 变长元组
    ```

    **示例代码**

    ```python
    from typing import Optional, Union

    def process_data(
        data: list[int],
        prefix: Optional[str] = None
    ) -> dict[str, Union[int, float]]:
        result = {"sum": sum(data)}
        if prefix:
            result["prefix"] = len(prefix)
        return result
    ```

    **静态检查工具**

    * mypy：最常用的类型检查器。

        ```bash
        pip install mypy
        mypy your_script.py
        ```

    注意

    * 运行时可以通过 __annotations__ 属性获取类型信息（用于反射或文档生成）。

    **循环中的类型注解**

    1. 在循环内部注解（Python 3.6+）

        ```python
        for i in range(123):
            i: int  # 在循环体内添加类型注解
            # 或者直接依赖 range() 的类型推断
            pass
        ```

    2. 使用类型注释（Type Comment，较旧的写法，Python 3.x 兼容）
    
        ```python
        for i in range(123):  # type: int
            pass
        ```

    3. 使用 typing.cast（当类型不明显时）

        ```python
        from typing import cast

        items = [1, 2, 3]  # 假设这里 items 类型不明确
        for item in items:
            item_int = cast(int, item)  # 明确告诉类型检查器这是 int
            # 但这不是运行时检查，只是提示类型检查器
        ```

* py 中，open file 的不同模式

    * a - 只追加模式

        ```python
        # 只能写入，不能读取
        with open('file.txt', 'a') as f:
            f.write('新内容\n')  # ok, 可以写入
            content = f.read()   # error, 会出错，不能读取
        ```

    * a+ - 追加和读取模式

        ```python
        # 可以读取和写入
        with open('file.txt', 'a+') as f:
            f.write('新内容\n')  # ✅ 可以写入
            
            # 读取前需要移动文件指针
            f.seek(0)  # 将指针移动到文件开头
            content = f.read()  # ✅ 可以读取
        ```

    * a 和 a+ 都可以在文件不存在时自动创建文件

    * 各种文件打开模式对比

        | 模式 | 描述 | 文件不存在时 | 可读 | 可写 | 指针位置 |
        | - | - | - | - | - | - |
        | r | 只读 | 报错 | ✅ | ❌ | 开头 |
        | r+ | 读写 | 报错 | ✅ | ✅ | 开头 |
        | w | 只写 | 创建 | ❌ | ✅ | 开头（清空内容） |
        | w+ | 读写 | 创建 | ✅ | ✅ | 开头（清空内容） |
        | a | 追加 | 创建 | ❌ | ✅ | 末尾 |
        | a+ | 追加读 | 创建 | ✅ | ✅ | 末尾（写），可移动（读） |
        | x | 创建 | 创建，存在则报错 | ❌ | ✅ | 开头 |

    * 显式创建文件可以使用 x 模式（独占创建）

        ```python
        try:
            with open('new_file.txt', 'x') as f:
                f.write('创建新文件\n')
        except FileExistsError:
            print("文件已存在")
        ```

    * w 模式

        ```python
        # 如果文件存在会清空内容，不存在则创建
        with open('file.txt', 'w') as f:
            f.write('新内容\n')
        ```

    * 使用 pathlib（推荐）

        ```python
        from pathlib import Path

        # 创建空文件
        Path('new_file.txt').touch()

        # 创建并写入内容
        Path('new_file.txt').write_text('文件内容')
        ```

* 使用 a+ 打开文件时，读取是发生在文件末尾，需要移动指针到开头才能读取，写入时自动回到末尾

    （如果把指针放到文件头后，再追加写入，此时指针是在头还是在尾？猜测仍在末尾）

    这种设计保证了追加模式的核心特性：不会意外覆盖现有内容

* python 访问全局变量

    使用 global 关键字：

    ```python
    aaa = "我是全局变量"  # 全局变量

    def my_function(aaa):
        print("形参 aaa:", aaa)           # 访问形参
        print("全局变量 aaa:", globals()['aaa'])  # 方法1：使用 globals()
        
        # 或者先声明 global
        global aaa
        print("全局变量 aaa:", aaa)        # 方法2：使用 global 关键字

    my_function("我是形参")
    ```

    注意：在 Python 中，如果函数内部有同名的形参或局部变量，直接使用 global aaa 会有冲突。推荐使用 globals()['aaa']。

* python 中比较 None 时应该使用 is 而不是 ==

* python class 中定义成员变量

    1. 在`__init__()`或其他成员函数中，使用`self.xxx = yyy`定义成员变量

        ```py
        class DynamicClass:
            def __init__(self):
                self.defined_in_init = "I'm from init" 

            def add_attribute_later(self):
                self.defined_later = "I was created later!"

        # 使用
        obj = DynamicClass()
        print(obj.defined_in_init) # 正常工作

        # print(obj.defined_later) # 这里会报错，因为还没有执行定义它的方法

        obj.add_attribute_later() # 调用方法，动态创建了成员
        print(obj.defined_later)  # 现在可以正常工作了
        ```

    2. 使用类属性

        ```py
        class MyClass:
            # 这是类属性
            class_attr = "I'm a class attribute"

            def __init__(self, instance_attr):
                # 这是实例属性
                self.instance_attr = instance_attr

        # 使用
        obj1 = MyClass("Obj1 value")
        obj2 = MyClass("Obj2 value")

        # 访问实例属性：每个对象独有
        print(obj1.instance_attr) # Obj1 value
        print(obj2.instance_attr) # Obj2 value

        # 访问类属性：所有对象共享，也可以通过类本身访问
        print(obj1.class_attr)    # I'm a class attribute
        print(obj2.class_attr)    # I'm a class attribute
        print(MyClass.class_attr) # I'm a class attribute
        ```

        共享性：所有实例对象共享同一个类属性。如果通过类名修改它（如 MyClass.class_attr = "new"），所有实例看到的都会改变。

        实例访问的陷阱：如果你通过实例对类属性进行赋值（如 obj1.class_attr = "new for obj1"），你实际上是在该实例的命名空间内创建了一个新的同名实例属性，它会遮蔽（shadow）掉类属性。此时，obj1.class_attr 是实例属性，而 obj2.class_attr 和 MyClass.class_attr 仍然是原来的类属性。

    3. 使用`@property`装饰器

        ```py
        class Circle:
            def __init__(self, radius):
                self.radius = radius # 这里只存储了半径

            @property
            def area(self):
                # 面积不需要存储，每次访问时根据半径计算
                return 3.14159 * self.radius ** 2

            @property
            def diameter(self):
                return self.radius * 2

        # 使用
        c = Circle(5)
        print(c.radius)   # 5 (实例属性)
        print(c.diameter) # 10 (看起来是属性，实则是方法计算的结果)
        print(c.area)     # 78.53975 (看起来是属性，实则是方法计算的结果)

        # c.area = 100 # 这会报错，因为@property默认是只读的
        ```

    在使用类成员时，如果不知道初始值，可以使用`Nonde`:

    ```py
    class User:
        # 使用 None 作为占位符，表示这些属性需要后续初始化
        name = None
        email = None
        age = None
    ```

    但是只有`None`无法提供类型信息，可以使用类型注解（Type Hints）配合 None:

    ```py
    class User:
        name: str | None = None
        email: str | None = None
        age: int | None = None
    ```

    不可以只写类型注解，不写初始化值：

    ```py
    class User:
        name: str          # 这只是类型注解
        age: int = 0       # 这是真正的属性定义 + 类型注解

    # 测试
    user = User()
    print(user.age)        # 正常工作，输出: 0
    print(user.name)       # 报错！AttributeError: 'User' object has no attribute 'name'
    ```

* python 中的 int

    在Python中，int 类型既不是固定的32位也不是64位，而是任意精度整数（arbitrary precision），可以表示任意大小的整数，只受限于可用内存。

    Python整数类型的特点

    * 自动扩展精度：当整数超出当前表示范围时，Python会自动分配更多内存

    * 不需要指定signed/unsigned：Python的int总是带符号的（signed）

    * 没有位数限制（理论上）

    如何获取整数位数信息:

    ```py
    import sys

    x = 42
    # 获取当前对象占用的字节数
    print(sys.getsizeof(x))  # 通常是28字节（包括Python对象开销）

    # 获取实际数值的位长度
    print(x.bit_length())  # 最少需要多少位表示这个数（不包括符号位）
    ```

    虽然Python本身没有unsigned int，但在与底层系统交互时可能需要处理：

    * 模拟unsigned行为

        ```py
        def to_unsigned(n, bits=32):
            """将有符号整数转换为无符号表示"""
            return n & ((1 << bits) - 1)

        def from_unsigned(n, bits=32):
            """将无符号整数转换为有符号表示"""
            if n >= (1 << (bits - 1)):
                n -= (1 << bits)
            return n

        # 示例
        x = -1
        unsigned = to_unsigned(x, 32)  # 4294967295
        signed = from_unsigned(unsigned, 32)  # -1
        ```

    * 使用ctypes模块

        ```py
        import ctypes

        # 转换为C语言的32位有符号/无符号整数
        x = 0xFFFFFFFF

        signed_32 = ctypes.c_int32(x).value  # -1
        unsigned_32 = ctypes.c_uint32(x).value  # 4294967295

        # 64位
        signed_64 = ctypes.c_int64(x).value  # 4294967295
        unsigned_64 = ctypes.c_uint64(x).value  # 4294967295
        ```

    * 使用struct模块处理二进制数据

        ```py
        import struct

        # 打包为32位无符号整数
        packed = struct.pack('I', 0xFFFFFFFF)  # 'I'表示unsigned int
        unpacked = struct.unpack('I', packed)[0]  # 4294967295

        # 打包为32位有符号整数
        packed = struct.pack('i', -1)  # 'i'表示signed int
        unpacked = struct.unpack('i', packed)[0]  # -1
        ```

    常见场景：

    * 处理网络协议数据

        ```py
        def parse_ip_header(data):
            # data是bytes类型
            import struct
            
            # 解析为无符号整数
            version_ihl, tos, total_length = struct.unpack('!BBH', data[:4])
            # '!'表示网络字节序，'H'表示unsigned short
            
            return total_length  # 返回的是无符号整数
        ```

    * 处理硬件寄存器

        ```py
        def read_register(address):
            # 从硬件读取32位寄存器值
            raw_value = 0xFFFFFFFF  # 假设读取的值
            
            # 作为无符号解释
            unsigned_value = raw_value & 0xFFFFFFFF
            
            # 如果需要作为有符号
            if unsigned_value & 0x80000000:
                signed_value = unsigned_value - 0x100000000
            else:
                signed_value = unsigned_value
                
            return signed_value
        ```

    * 数值范围检查

        ```py
        def check_32bit_range(value):
            """检查值是否在32位有符号/无符号范围内"""
            
            # 32位有符号范围
            signed_min = -2**31
            signed_max = 2**31 - 1
            
            # 32位无符号范围
            unsigned_min = 0
            unsigned_max = 2**32 - 1
            
            is_signed_ok = signed_min <= value <= signed_max
            is_unsigned_ok = unsigned_min <= value <= unsigned_max
            
            return is_signed_ok, is_unsigned_ok
        ```

* argparse 支持多个短参数的组合（传统Unix风格）

    ```py
    parser.add_argument('-a', action='store_true', help='选项A')
    parser.add_argument('-b', action='store_true', help='选项B')
    parser.add_argument('-c', action='store_true', help='选项C')
    ```

    `python script.py -abc`相当于`-a -b -c`。

* argparse 给参数赋值时，使用空格或等号都可以

* argparse 中的帮助信息 -h

    ```py
    # -h 是默认的，但你也可以自定义
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-?', '--help', action='help', help='显示帮助信息')
    ```

* argparse 中的互斥参数组

    ```py
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='store_true', help='详细模式')
    group.add_argument('-q', '--quiet', action='store_true', help='安静模式')
    ```

    不能同时使用 -v 和 -q

* arparse 中对参数值进行限制

    ```py
    parser.add_argument('-l', '--level', type=int, choices=[1, 2, 3], help='级别(1-3)')
    ```

* argparse 中指定多个值

    ```py
    parser.add_argument('-i', '--input', nargs='+', help='多个输入文件')
    ```

    run:

    `python script.py -i file1.txt file2.txt file3.txt`

* argparse 中的位置参数（positional arguments） 和 可选参数（optional arguments）

    * 位置参数（没有 --）

        ```py
        parser.add_argument('input_file', help='输入文件')
        ```

        必须提供，不提供会报错

        顺序敏感：在命令行中必须按照定义的顺序出现

        没有前缀：直接写参数值

    * 可选参数（有 - 或 --）

        ```py
        parser.add_argument('--output', help='输出文件')
        parser.add_argument('-v', '--verbose', action='store_true')
        ```

        可选提供，可以不写

        顺序无关：可以在命令行的任何位置

        有前缀：以 - 或 -- 开头

    examples:

    * exapmle 1

        ```py
        import argparse

        parser = argparse.ArgumentParser(description='文件处理工具')
        parser.add_argument('input_file', help='输入文件路径')
        parser.add_argument('output_file', help='输出文件路径')
        parser.add_argument('-v', '--verbose', action='store_true', help='详细模式')
        parser.add_argument('-f', '--format', choices=['json', 'xml'], help='输出格式')

        args = parser.parse_args()

        print(f"输入文件: {args.input_file}")
        print(f"输出文件: {args.output_file}")
        print(f"详细模式: {args.verbose}")
        print(f"输出格式: {args.format}")
        ```

        run:

        ```bash
        # 正确：位置参数必须按顺序提供
        python script.py input.txt output.json
        python script.py input.txt output.json -v --format json
        python script.py -v --format json input.txt output.json  # 顺序无关

        # 错误：缺少位置参数
        python script.py input.txt                    # 缺少 output_file
        python script.py --verbose                    # 缺少两个位置参数
        ```

    * example 2

        ```py
        import argparse

        parser = argparse.ArgumentParser(description='复制文件')
        parser.add_argument('source', help='源文件')
        parser.add_argument('destination', help='目标位置')
        parser.add_argument('-r', '--recursive', action='store_true', help='递归复制')
        parser.add_argument('-f', '--force', action='store_true', help='强制覆盖')

        args = parser.parse_args()

        print(f"从 {args.source} 复制到 {args.destination}")
        if args.recursive:
            print("递归模式")
        if args.force:
            print("强制覆盖模式")
        ```

        run:

        ```bash
        python script.py file.txt backup/ -r -f
        # 或者
        python script.py -r -f file.txt backup/
        ```

    * example 3

        ```py
        import argparse

        parser = argparse.ArgumentParser(description='数据处理工具')

        # 位置参数（必须的）
        parser.add_argument('input_file', help='输入数据文件')
        parser.add_argument('operation', choices=['process', 'validate', 'export'], 
                           help='要执行的操作')

        # 可选参数
        parser.add_argument('-o', '--output', help='输出文件')
        parser.add_argument('--format', default='csv', help='输出格式')
        parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

        args = parser.parse_args()
        ```

        run:

        ```
        python script.py data.csv process -o result.json --format json -v
        ```

* argparse 使用 prefix_chars 参数

    argparse 允许自定义前缀字符：

    ```py
    import argparse

    # 允许使用 - 和 / 作为前缀
    parser = argparse.ArgumentParser(prefix_chars='-/')

    parser.add_argument('-s', '/s', '--silent', action='store_true', help='静默模式')
    parser.add_argument('-v', '/v', '--verbose', action='store_true', help='详细模式')
    parser.add_argument('-f', '/f', '--file', help='输入文件')
    parser.add_argument('-?', '/?', action='help', help='显示帮助')

    args = parser.parse_args()

    print(f"静默模式: {args.silent}")
    print(f"详细模式: {args.verbose}")
    print(f"输入文件: {args.file}")
    ```

    run:

    ```bash
    # 都可以工作
    python script.py -s -v -f data.txt
    python script.py /s /v /f data.txt
    python script.py --silent --verbose --file data.txt
    ```

    这样可以用于适配 windows 环境。

* argparse 处理短参数

    example:

    ```py
    import argparse

    parser = argparse.ArgumentParser()
    # 短参数: -v, 长参数: --verbose
    parser.add_argument('-v', '--verbose', action='store_true', help='详细模式')
    # 短参数: -f, 长参数: --file
    parser.add_argument('-f', '--file', type=str, help='输入文件')
    # 短参数: -n, 长参数: --number
    parser.add_argument('-n', '--number', type=int, default=1, help='重复次数')

    args = parser.parse_args()

    if args.verbose:
        print("详细模式开启")
    if args.file:
        print(f"处理文件: {args.file}")
    print(f"重复次数: {args.number}")
    ```

    `add_argument()`的原型：

    ```py
    def add_argument(
        *name_or_flags: str,
        action: str | type[Action] = ...,
        nargs: int | str | None = None,
        const: Any = ...,
        default: Any = ...,
        type: _ActionType = ...,
        choices: Iterable[_T@add_argument] | None = ...,
        required: bool = ...,
        help: str | None = ...,
        metavar: str | tuple[str, ...] | None = ...,
        dest: str | None = ...,
        version: str = ...,
        **kwargs: Any
    ) -> Action
    ```

    可以看到，其实现短参数的原理是使用`*name_or_flags`这个位置参数，可以指定同一个参数的多个别名。
    
    这个很巧妙，无论是`-v`，`--verbose`，还是`-verbose`，`--v`，都由用户自由设定，如果使用`add_argument(long='verbose', short='v')`，那遇到`-verbose`该选择 long 还是 short？这样就不好处理了。

* python 函数中的 static 变量

    1. 使用函数属性（推荐）

        ```py
        def counter():
            if not hasattr(counter, "count"):
                counter.count = 0  # 初始化静态变量
            counter.count += 1
            return counter.count

        print(counter())  # 1
        print(counter())  # 2
        print(counter())  # 3
        print(f"静态变量值: {counter.count}")  # 可以直接访问
        ```

    2. 使用闭包

        ```py
        def make_counter():
            count = 0  # 闭包中的变量，类似于静态变量
            
            def counter():
                nonlocal count  # 声明为nonlocal以修改闭包变量
                count += 1
                return count
            
            return counter

        counter = make_counter()
        print(counter())  # 1
        print(counter())  # 2
        print(counter())  # 3
        ```

    3. 使用装饰器

        这个本质也是函数属性。

        ```py
        def static_vars(**kwargs):
            def decorate(func):
                for key, value in kwargs.items():
                    setattr(func, key, value)
                return func
            return decorate

        @static_vars(counter=0)
        def my_func():
            my_func.counter += 1
            return my_func.counter

        print(my_func())  # 1
        print(my_func())  # 2
        ```

        这个也可以写成：

        ```py
        def call_counter(func):
            def wrapper(*args, **kwargs):
                wrapper.calls += 1
                print(f"{func.__name__} 已被调用 {wrapper.calls} 次")
                return func(*args, **kwargs)
            
            wrapper.calls = 0  # 初始化计数器
            return wrapper

        @call_counter
        def greet(name):
            return f"Hello, {name}!"

        print(greet("Alice"))
        print(greet("Bob"))
        print(greet("Charlie"))
        # 输出:
        # greet 已被调用 1 次
        # Hello, Alice!
        # greet 已被调用 2 次
        # Hello, Bob!
        # greet 已被调用 3 次
        # Hello, Charlie!
        ```

    4. 使用类

        把函数看作一个 callable object。

        ```py
        class Counter:
            def __init__(self):
                self.count = 0
            
            def __call__(self):
                self.count += 1
                return self.count

        counter = Counter()
        print(counter())  # 1
        print(counter())  # 2
        ```

    注意事项

    * 线程安全：上述方法在单线程中工作良好，但在多线程环境下需要加锁

    * 可读性：使用函数属性是最直观的方式

    * 重置静态变量：可以直接访问并重置，如 func.static_var = new_value

* 生成从指定日期开始的 N 天

    ```py
    from datetime import datetime, timedelta

    # 指定起始日期
    start_date = datetime(2024, 1, 1)

    # 生成未来5天（包括起始日）
    for i in range(5):
        current_date = start_date + timedelta(days=i)
        print(current_date.strftime('%Y-%m-%d'))
    ```

    output:

    ```
    2024-01-01
    2024-01-02
    2024-01-03
    2024-01-04
    2024-01-05
    ```

    同理，如果向前推日期的话，只需要减去`timedelta`就可以了。

    还可以使用 `date` 对象（只处理日期，不含时间）:

    ```py
    from datetime import date, timedelta

    # 使用date对象
    start_date = date(2024, 1, 1)

    # 生成未来5天
    for i in range(5):
        current_date = start_date + timedelta(days=i)
        print(current_date)

    # 向前推3天
    for i in range(1, 4):
        past_date = start_date - timedelta(days=i)
        print(past_date)
    ```

* argparse 中的 action

    `action='store_true'`表示当命令行中出现这个选项时，将参数值设置为 True；如果不出现，则设置为 False。

    配置了这个后，只需要写`--verbose`，就相当于`--verbose True`了。否则需要自己手动指定参数的值。（如果不写 action，只指定`--verbose`会发生什么？）

    `action`可接收的值：

    * `store` (默认值)

        存储参数的值（默认行为）

        ```py
        parser.add_argument('--file', action='store', type=str)
        # 命令行: --file data.txt
        # 结果: args.file = 'data.txt'
        ```

    * `store_true` / `store_false`

        ```py
        parser.add_argument('--enable', action='store_true')
        # 命令行指定 --enable: args.enable = True
        # 不指定: args.enable = False

        parser.add_argument('--disable', action='store_false')
        # 命令行指定 --disable: args.disable = False
        ```

    * `store_const`

        参数出现时设置为固定值

        example:

        `parser.add_argument('--level', action='store_const', const=10, help='出现时设置为固定值')`

        ```py
        parser.add_argument('--mode', action='store_const', const='fast')
        # 命令行指定 --mode: args.mode = 'fast'
        ```

    * `append`

        将多个参数值收集到列表中

        ```py
        parser.add_argument('--tag', action='append')
        # 命令行: --tag python --tag argparse --tag tutorial
        # 结果: args.tag = ['python', 'argparse', 'tutorial']
        ```

    * `count`

        计算参数出现的次数

        ```py
        parser.add_argument('-v', '--verbose', action='count', default=0)
        # 命令行: -v -v -v
        # 结果: args.verbose = 3
        # 或者: -vvv 同样得到 args.verbose = 3
        ```

    * `append_const`

        ```py
        parser.add_argument('--add-python', action='append_const', const='python')
        parser.add_argument('--add-java', action='append_const', const='java')
        # 命令行: --add-python --add-java --add-python
        # 结果: args.const_list = ['python', 'java', 'python']
        ```

* typer

    `pip install typer`

    example:

    ```py
    import typer

    app = typer.Typer()

    @app.command()
    def hello(name: str, age: int = 18, verbose: bool = False):
        """向某人问好"""
        typer.echo(f"你好 {name}, 年龄 {age}")
        if verbose:
            typer.echo("详细模式已开启")

    @app.command()
    def goodbye(name: str):
        """向某人道别"""
        typer.echo(f"再见 {name}!")

    if __name__ == "__main__":
        app()
    ```

    run:

    * `python main.py --help`

        output:

        ```
                                                                                        
         Usage: main_2.py [OPTIONS] COMMAND [ARGS]...                                   
                                                                                        
        ╭─ Options ────────────────────────────────────────────────────────────────────╮
        │ --install-completion          Install completion for the current shell.      │
        │ --show-completion             Show completion for the current shell, to copy │
        │                               it or customize the installation.              │
        │ --help                        Show this message and exit.                    │
        ╰──────────────────────────────────────────────────────────────────────────────╯
        ╭─ Commands ───────────────────────────────────────────────────────────────────╮
        │ hello     向某人问好                                                         │
        │ goodbye   向某人道别                                                         │
        ╰──────────────────────────────────────────────────────────────────────────────╯

        ```

    * `python main.py hello zhangsan --age 16 --verbose`

        output:

        ```
        你好 zhangsan, 年龄 16
        详细模式已开启
        ```

* python fire

    `pip install fire`

    `main.py`:

    ```py
    import fire

    class Calculator:
        def add(self, a, b=2, msg='hello, world', verbose: bool = False):
            """相加两个数字"""
            print('a: {}, b: {}'.format(a, b))
            print('msg: {}'.format(msg))
            print('verbose: {}'.format(verbose))
            return a + b
        
        def multiply(self, a, b):
            """相乘两个数字"""
            return a * b

    if __name__ == '__main__':
        fire.Fire(Calculator)
    ```

    run and output:

    * `python main.py add 10`

        output:

        ```
        a: 10, b: 2
        msg: hello, world
        verbose: False
        12
        ```

    * `python main.py add 10 --a 1 -msg='hello' --verbose`

        output:

        ```
        a: 1, b: 10
        msg: hello
        verbose: True
        11
        ```

    * `python main.py multiply 2 3`

        output:

        ```
        6
        ```

    可以看到，将 class 传给 fire 时，每个成员函数都是一个 subcommand。成员函数的参数直接对应 cli 的参数。

* python 不允许对一个 tuple 进行类型标注

    比如：`a, b: (str, str) = 'hello', 'world'`

    或者：`a, b: str, str = 'hello', 'world'`

    如果确实需要标注，可以考虑下面几种办法：

    ```py
    # 方式1：最清晰
    input_data: list
    gt: list
    input_data, gt = data

    # 方式2：使用类型别名
    from typing import Tuple
    def process_data(data: Tuple[list, list]) -> None:
        input_data, gt = data
    ```

* python 中的`-m`运行

    在当前文件夹下的`mod_1.py`，可以使用`python -m mod_1`启动运行。在`pkg`文件夹下的`mod_2.py`，可以使用`python -m pkg.mod_2`运行。

    使用`-m`运行时，py 文件不能加`.py`。

* python 中的相对导入

    如果`pkg`文件夹下有两个文件：`mod_1.py`, `mod_2.py`，其内容分别如下：

    `mod_1.py`:

    ```py
    def print_hello():
        print('hello')
    ```

    `mod_2.py`:

    ```py
    from . import mod_1

    mod_1.print_hello()
    ```

    运行：

    * 在 pkg 的父目录中运行：`python -m pkg.mod_2`, OK

    * 在 pkg 的父目录中运行：`python pkg/mod_2.py`, Error

        相对导入需要 package 的信息，这里没有提供。

    * 在 pkg 目录中运行：`python -m mod_2`, Error

        同理，这里没有提供 package 信息。

    * 在 pkg 目录中运行：`python mod_2.py`, Error

        未提供 package 信息。

* argparse

    ```py
    import argparse

    # 创建解析器
    parser = argparse.ArgumentParser(description='这是一个示例程序')

    # 添加参数
    parser.add_argument('--name', type=str, required=True, help='你的名字')
    parser.add_argument('--age', type=int, default=18, help='你的年龄')
    parser.add_argument('--verbose', action='store_true', help='详细模式')
    parser.add_argument('input_file', help='输入文件')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    print(f"你好 {args.name}, 年龄 {args.age}")
    if args.verbose:
        print("详细模式已开启")
    print(f"处理文件: {args.input_file}")
    ```

    run:

    `python script.py data.txt --name 张三 --age 25 --verbose`

    output:

    ```
    你好 张三, 年龄 25
    详细模式已开启
    处理文件: data.txt
    ```

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

## topics

### re

* finditer

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

* search and match

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

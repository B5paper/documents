# Regular expressions

正则表达式的相关笔记。

## cache

* 正则表达式中`^`指的是字符串的开头还是`\n`的下一个字符？

    在正则表达式中，^ 的行为取决于具体的模式修饰符（flags）：

    1. 默认情况（没有多行模式）

        ^ 匹配整个字符串的开头位置

        ```regex
        /^abc/  # 只匹配字符串开头的"abc"
        ```

        示例：

        ```javascript
        "first line\nsecond line".match(/^second/)  // null，不匹配
        ```

    2. 多行模式（Multiline mode）

        使用 m 标志时，^ 会匹配：

        * 整个字符串的开头

        * 以及每一个换行符（\n）之后的位置

            ```regex
            /^abc/m  # 匹配字符串开头和每一行开头的"abc"
            ```

        示例：

        ```javascript
        "first line\nsecond line".match(/^second/m)  // 匹配成功
        ```

    3. 重要对比

        | 情况 | ^ 匹配的位置 |
        | - | - |
        | 无 m 标志 | 仅字符串开头 |
        | 有 m 标志 | 字符串开头 + 每个 \n 之后 |

    4. 实际示例

        ```javascript
        // 无多行模式
        "a\nb\nc".match(/^b/g)      // null

        // 有多行模式
        "a\nb\nc".match(/^b/gm)     // ["b"]

        // 匹配所有行首
        "line1\nline2\nline3".match(/^.*$/gm)
        // 匹配：["line1", "line2", "line3"]

        // 无 m 标志时，只能匹配整个字符串
        "line1\nline2\nline3".match(/^.*$/g)
        // 匹配：["line1\nline2\nline3"]
        ```

    5. 类似的行为

        * $：默认匹配字符串结尾，多行模式下匹配每个 \n 之前的位置

        * \A：在某些引擎（如PCRE）中总是匹配字符串开头（无视多行模式）

        * \Z, \z：匹配字符串结尾（细节因引擎而异）

    总结：默认情况下 ^ 只匹配整个字符串的开头，只有在启用多行模式（m 标志）时，它才会匹配每个换行符后的位置。

* 正则表达式

    正则表达式（Regular Expression）是一种用于匹配、查找和替换文本的强大工具，通过定义特定模式来处理字符串。以下是其核心用法简述：

    一、基本语法

    | 符号 | 说明 | 示例 |
    | - | - | - |
    | `.` | 匹配任意单个字符（除换行符） | a.c → "abc" |
    | `\d` | 匹配数字（0-9） | \d\d → "42" |
    | `\w` | 匹配字母、数字、下划线 | \w+ → "Hello_1" |
    | `\s` | 匹配空白字符（空格、制表符等） | a\sb → "a b" |
    | `[abc]` | 匹配括号内任意字符 | [aeiou] → "e" |
    | `[^abc]` | 匹配不在括号内的字符 | [^0-9] → "a" |

    二、量词（控制重复次数）

    | 符号 | 说明 | 示例 |
    | - | - | - |
    | `*` | 匹配前一个元素0次或多次 | ab*c → "ac", "abbc" |
    | `+` | 匹配前一个元素1次或多次 | ab+c → "abc"（不匹配"ac"） |
    | `?` | 匹配前一个元素0次或1次 | colou?r → "color" 或 "colour" |
    | `{n}` | 匹配前一个元素恰好n次 | \d{3} → "123" |
    | `{n,}` | 匹配至少n次 | \w{2,} → "abc" |

    三、锚点与边界

    | 符号 | 说明 | 示例 |
    | - | - | - |
    | `^` | 匹配字符串开头 | ^Hello → "Hello world" 中的"Hello" |
    | `$` | 匹配字符串结尾 | world$ → 匹配以"world"结尾的字符串 |
    | `\b` | 匹配单词边界 | \bcat\b → 匹配独立的"cat" |

    四、分组与捕获

    | 符号 | 说明 |
    | - | - |
    | `(abc)` | 捕获分组，可提取匹配内容 |
    | `(?:abc)` | 非捕获分组，仅分组不捕获 |
    | `|` | 逻辑"或" | `cat|dog` → 匹配"cat"或"dog" |

    注：

    1. markdown 表格中如何写 code fence 中的`|`符号？

    五、转义字符

    特殊字符如 .、*、? 需用反斜杠转义：\. 匹配真正的句点。

    六、应用场景

    验证格式（邮箱、电话号码）

    ```regex
    邮箱示例：^\w+([.-]?\w+)*@\w+([.-]?\w+)*\.\w{2,3}$
    ```

    查找与提取（从日志中提取IP地址）

    ```regex
    IP地址：\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b
    ```

    文本替换（删除多余空格）

    ```regex
    替换连续空格：\s+  →  " "
    ```

    字符串分割（按分隔符切分）

    ```regex
    按逗号或分号分割：[,;]
    ```

    七、常用技巧

    * 贪婪与非贪婪匹配

        `.*`（贪婪）匹配尽可能多的字符；`.*?`（非贪婪）匹配尽可能少的字符。

    * 向前/向后断言

        `(?=abc)` 匹配后面紧跟"abc"的位置；`(?<=abc)` 匹配前面是"abc"的位置。

    八、工具与支持

    * 在线测试：RegEx101、Regexr

    * 编程语言：Python（re模块）、JavaScript（/pattern/）、Java（java.util.regex）等均内置支持。

    简单示例

    ```regex
    匹配手机号（中国大陆）：^1[3-9]\d{9}$
    匹配日期（YYYY-MM-DD）：^\d{4}-\d{2}-\d{2}$
    提取HTML标签内容：<div>(.*?)</div>
    ```

* 正则表达式中的 shorthands

    预定义字符类（也叫元字符或转义序列）,它们分别匹配特定类型的字符。

    | Character class | Description | Equivalent |
    | - | - | - |
    | `\s` | 任意空白字符 | 空格、制表符（\t）、换行符（\n）、回车符（\r）、换页符（\f）等。 |
    | `\d` | Digits from 0 to 9 | `[0-9]` |
    | `\w` | Word characters, 任意单词字符（字母、数字、下划线） | `[0-9A-Za-z_]` |
    | `\D` | 任意非数字字符 | `[^0-9]`，即 \d 的反义。 |

    注：

    1. `\w`通常不包括非拉丁字母（如中文），但某些引擎（如 Python 的 re 模块）可通过 Unicode 模式支持更多字符。

    1. 大写形式的 \S、\W 分别是 \s、\w 的反义：

        * `\S`：匹配非空白字符

        * `\W`：匹配非单词字符（如标点、空格）

* `re.findall()`的作用

    example:

    ```py
    import re

    text = 'hello, world'
    pat = r'l+l'
    ret = re.findall(pat, text)
    print(ret)  # ['ll']

    text = 'name: zhangsan, age: 14; name: liming, age: 15; name: wangwu, age: 16;'
    pat = r'name: \w+, age: \d+;'
    ret = re.findall(pat, text)
    print(ret)  # ['name: zhangsan, age: 14;', 'name: liming, age: 15;', 'name: wangwu, age: 16;'

    # 使用 () 提取出想要的具体数值
    pat = r'name: (\w+), age: (\d+);'
    ret = re.findall(pat, text)
    print(ret)  # [('zhangsan', '14'), ('liming', '15'), ('wangwu', '16')]
    ```

    重要特性:

    1. 非重叠匹配

        ```py
        text = "aaa"
        matches = re.findall(r'aa', text)
        print(matches)  # ['aa'] 只匹配一次，不重叠
        ```

    1. 空匹配处理

        ```py
        text = "a1b2c"
        matches = re.findall(r'\d*', text)
        print(matches)  # ['', '1', '', '2', ''] 包含空匹配
        ```

    1. 使用 flags 参数

        ```py
        text = "Hello WORLD"
        # 忽略大小写
        matches = re.findall(r'hello', text, re.IGNORECASE)
        print(matches)  # ['Hello']
        ```

    与相似函数的区别:

    | 函数 | 返回值 | 用途 |
    | - | - | - |
    | `re.findall()` | 列表，所有匹配的字符串或元组 | 提取所有匹配内容 |
    | `re.finditer()` | 迭代器，返回匹配对象 | 需要匹配的详细信息 |
    | `re.search()` | 第一个匹配的匹配对象 | 只找第一个匹配 |
    | `re.match()` | 开头匹配的匹配对象 | 只匹配字符串开头 |

    example:

    ```py
    # 提取日志中的IP地址
    log = "192.168.1.1 - GET /home, 10.0.0.1 - POST /login"
    ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', log)
    print(ips)  # ['192.168.1.1', '10.0.0.1']

    # 提取HTML标签内容
    html = "<h1>标题</h1><p>段落内容</p>"
    contents = re.findall(r'<[^>]+>(.*?)</[^>]+>', html)
    print(contents)  # ['标题', '段落内容']
    ```

* ?： 在量词（如 *）之后使用，表示“非贪婪”或“最小”模式。它会匹配尽可能少的字符，直到满足后续条件（即遇到 phrase\.）。

* 贪婪匹配与非贪婪匹配

    非贪婪匹配（也称为惰性匹配或最小匹配）是正则表达式中一种匹配模式，它会尽可能少地匹配字符。通常使用`*?`

    贪婪匹配会尽可能多地匹配字符。通常使用`*`。

    example:

    ```py
    import re

    s = 'abbbabba'

    # 贪婪匹配
    for m in re.finditer(r'a.*a', s):
        start_pos = m.start()
        end_pos = m.end()
        selected_str = s[start_pos:end_pos]
        print(selected_str)  # abbbabba

    # 非贪婪匹配
    for m in re.finditer(r'a.*?a', s):
        start_pos = m.start()
        end_pos = m.end()
        selected_str = s[start_pos:end_pos]
        print(selected_str)  # abbba
    ```

## note

## 语法与规则

* 直接输入的字符串可以直接匹配

    比如`abc`可以匹配字符串`"abc"`。这里的匹配指的是找到匹配成功的字符串的起始位置。

* 一些特殊的转义字符

    * `\d`: 匹配  0 到 9 这 10 个数字

        the character `\d` can be used in place of any digit from 0 to 9. The preceding slash distinguishes it from the simple d character and indicates that it is a metacharacter.

* 点`.`（dot）

    点`.`是一种通匹配符（wildcard）,可以匹配任何单个字符

    > there is the concept of a wildcard, which is represented by the . (dot) metacharacter, and can match any single character (letter, digit, whitespace, everything).

    如果要指定匹配`.`字符，那么必须使用转义字符：`\.`

* 方括号`[]`

    方括号被看作单个字符，任何出现在方括号中出现的字符都算匹配成功。

    比如输入`[afp]`，那么如果一个字符是`a`，或者`f`，或者`p`，都算匹配成功。

    > There is a method for matching specific characters using regular expressions, by defining them inside square brackets. For example, the pattern [abc] will only match a single a, b, or c letter and nothing else.

* 排除某些字符`[^]`

    如果“不出现某些字符”算匹配成功，那么可以使用`[^]`。

    比如，`[^abc]`表示，只要一个字符不是`a`，`b`，`c`，就算匹配成功了。

    > Excluding specific characters uses the square brackets and the ^ (hat). For example, the pattern [^abc] will match any single character except for the letters a, b, or c.

## 常见场景分析


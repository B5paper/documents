# Regular expressions

正则表达式的相关笔记。

## cache

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


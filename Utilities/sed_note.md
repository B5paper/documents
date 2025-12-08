# Sed Note

sed（Stream Editor）是 Linux/Unix 下的流编辑器，主要用于对文本进行过滤和转换

## cache

* `sed -i <pattern> <file>`会直接修改原文件

* sed 实际应用场景

    ```bash
    # 提取IP地址
    ifconfig | sed -n '/inet /p' | sed 's/.*inet //' | sed 's/ netmask.*//'

    # 格式化日志文件
    sed -n '/ERROR/p' app.log | sed 's/ERROR/\\033[31mERROR\\033[0m/'

    # 批量重命名文件
    ls *.txt | sed 's/\(.*\)\.txt/mv "&" "\1.html"/' | sh
    ```

    常用选项

    * -n：安静模式，只显示处理后的行

    * -i：直接修改文件

    * -e：执行多个sed命令

    * -r：使用扩展正则表达式

* sed 中的正则表达式

    在 `s/old/new/` 中，old 会被 sed 解释为基本正则表达式（BRE，Basic Regular Expression）。

    这意味着：

    * 一些特殊字符需要转义才能使用其正则含义（如 `\+`、`\?`、`\{m,n\}`）

    * `^`、`$`、`.`、`*`、`[ ]` 等与正则表达式中含义相同

    * 如果要使用扩展正则表达式（ERE），需要加 `-E` 或 `-r` 选项

    new 部分不是正则表达式：new 是替换文本，但有特殊含义：

    * & 表示匹配到的整个内容

    * \1、\2 等表示捕获组（如果 old 中用了 \(\)）

    * \ 用于转义特殊字符或引入特殊序列

    example:

    ```bash
    # 基本正则表达式（默认）
    sed 's/^old/new/'      # ^ 是正则锚点
    sed 's/old\./new/'     # \. 匹配字面点号
    sed 's/\(old\)/\1/'    # \(\) 是捕获组

    # 扩展正则表达式（-E 选项）
    sed -E 's/old+/new/'   # + 直接表示"一个或多个"
    sed -E 's/(old)/\1/'   # () 直接是捕获组
    ```

    如果文本包含 /，可以换其他分隔符：`s#old#new#`，`sed 's|old|new|g' test.txt`

* sed 中，大部分命令需要使用单引号`''`包裹起来，防止 bash 预处理

    比如要替换点号`.`，需要写成`sed 's/\./new_char/ file.txt'`

* sed

    examples:

    * 文件处理

        ```bash
        # 替换文件内容并保存（-i选项）
        sed -i 's/foo/bar/g' file.txt

        # 备份原文件并替换
        sed -i.bak 's/foo/bar/g' file.txt

        # 删除HTML标签
        sed 's/<[^>]*>//g' file.html
        ```

    * 文本转换

        ```bash
        # 将DOS换行符(CRLF)转换为UNIX换行符(LF)
        sed 's/\r$//' file.txt

        # 在每行行首添加内容
        sed 's/^/# /' file.txt

        # 在每行行尾添加内容
        sed 's/$/ --- EOF/' file.txt
        ```

    * 高级用法

        ```bash
        # 使用正则表达式分组
        echo "abc123" | sed 's/\([a-z]*\)[0-9]*/\1/g'

        # 多点编辑（多个命令）
        sed -e 's/foo/bar/g' -e '/baz/d' file.txt

        # 使用其他分隔符（当路径包含/时）
        sed 's|/usr/local|/opt|g' file.txt
        ```

* sed 替换操作

    ```py
    # 基本替换（每行第一个匹配）
    sed 's/old/new/' file.txt

    # 全局替换（所有匹配）
    sed 's/old/new/g' file.txt

    # 替换每一行的第 N 次出现
    sed 's/old/new/2' file.txt

    # 只替换匹配的行
    sed '/pattern/s/old/new/g' file.txt
    # 如果不写 g，那么替换匹配行的第 1 个出现的 pattern
    ```

    example:

    `test.txt`:

    ```
    hello, hello
    world, hello

    ```

    run: `sed s/hello/nihao/ test.txt`

    output:

    ```
    nihao, hello
    world, nihao

    ```

    可以看到，sed 并没有改变原文件，只是把修改过后的内容输出到 stdout 上。

* 删除操作

    ```py
    # 删除第3行
    sed '3d' file.txt

    # 删除第2-5行
    sed '2,5d' file.txt

    # 删除匹配的行
    sed '/pattern/d' file.txt

    # 删除空行
    sed '/^$/d' file.txt
    ```

* 打印操作

    ```py
    # 打印第5行
    sed -n '5p' file.txt

    # 打印匹配的行
    sed -n '/pattern/p' file.txt

    # 打印第10-20行
    sed -n '10,20p' file.txt
    ```

* 插入和追加

    ```py
    # 在第3行前插入
    sed '3i\插入的内容' file.txt

    # 在第3行后追加
    sed '3a\追加的内容' file.txt
    ```

## topics

## note

* How to add to the end of lines containing a pattern with sed

    <https://stackoverflow.com/questions/9591744/how-to-add-to-the-end-of-lines-containing-a-pattern-with-sed-or-awk>

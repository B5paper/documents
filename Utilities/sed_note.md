# Sed Note

sed（Stream Editor）是 Linux/Unix 下的流编辑器，主要用于对文本进行过滤和转换

## cache

* sed 's#.*/##' <<< "/tmp/dir/target"
    详细讲讲这行 sed 命令的作用

    详解 `sed 's#.*/##' <<< "/tmp/dir/target"`

    这行命令的作用是 **从路径字符串中提取文件名（basename）**，即去掉所有目录部分，只保留最后一段。下面逐部分拆解。

    ---

    ## 一、整体结果

    ```bash
    $ sed 's#.*/##' <<< "/tmp/dir/target"
    target
    ```

    输入 `/tmp/dir/target`，输出 `target`。

    ---

    ## 二、拆解各部分

    ### 1. `<<<` —— Here-String（here 字符串）

    `<<<` 是 Bash（及 zsh、ksh）的 **here-string** 重定向语法，作用是把后面的字符串作为标准输入（stdin）传给命令。

    ```bash
    sed 's#.*/##' <<< "/tmp/dir/target"
    ```

    等价于：

    ```bash
    echo "/tmp/dir/target" | sed 's#.*/##'
    ```

    也等价于（用 here-doc）：

    ```bash
    sed 's#.*/##' <<EOF
    /tmp/dir/target
    EOF
    ```

    注意：
    - `<<<` 是 **Bash 扩展**，不是 POSIX `sh` 标准语法。在纯 `sh`/`dash` 下会报错。
    - `<<<` 会自动在字符串末尾补一个换行符，所以 `sed` 能按行处理。

    ### 2. `sed 's#.*/##'` —— 替换命令

    `sed` 的替换语法是 `s/正则/替换/`，默认只替换每行 **第一个** 匹配。

    这里用 `#` 作为分隔符，而不是常见的 `/`，原因是 **正则和替换内容里含有 `/`**，用 `#` 可以避免大量转义。写成 `/` 版本是：

    ```bash
    sed 's/.*\///'
    ```

    两者完全等价，但 `#` 版本可读性更好。

    #### 正则 `.*/` 的含义

    | 部分 | 含义 |
    |------|------|
    | `.`  | 任意单个字符（除换行） |
    | `*`  | 前面的字符重复 0 次或多次（**贪婪匹配**） |
    | `/`  | 字面斜杠 |

    所以 `.*/` 匹配「**任意字符 + 最后一个斜杠**」——因为 `*` 是贪婪的，它会一直吃到 **最后一个** `/` 为止。

    #### 替换部分为空

    `##` 中第二个 `#` 后面什么都没有，表示把匹配到的内容替换成 **空字符串**，即删除。

    #### 举例

    | 输入 | 匹配到的 `.*/` | 结果 |
    |------|---------------|------|
    | `/tmp/dir/target` | `/tmp/dir/` | `target` |
    | `/a/b/c.txt` | `/a/b/` | `c.txt` |
    | `foo/bar` | `foo/` | `bar` |
    | `nofile`（无斜杠） | 不匹配 | `nofile`（原样输出） |
    | `/` | `/` | 空字符串 |
    | `/a/b/`（尾部有斜杠） | `/a/b/` | 空字符串 |

    注意最后两行：如果路径以 `/` 结尾，结果会是空行——这正是「纯 sed 方案」的局限，`basename` 命令会做额外处理。

    ---

    ## 三、与 `basename` 的对比

    ```bash
    $ basename /tmp/dir/target
    target

    $ sed 's#.*/##' <<< "/tmp/dir/target"
    target
    ```

    对普通路径两者结果一致。但差异在于：

    | 场景 | `basename` | `sed 's#.*/##'` |
    |------|-----------|-----------------|
    | `/tmp/dir/target` | `target` | `target` |
    | `/a/b/`（尾斜杠） | `b` | 空 |
    | `//` | `/` | 空 |
    | 多行输入 | 只处理一个参数 | 每行都处理（天然适合批量） |
    | 是否依赖外部命令 | 是（coreutils） | 否（sed 内置） |
    | 性能（大文件/流） | 需逐次调用 | 一次流式处理，更快 |

    所以 sed 方案常见于 **批量处理** 或 **纯 shell 内建/流式** 场景，比如：

    ```bash
    # 列出所有文件的 basename
    find /tmp -type f | sed 's#.*/##'
    ```

    ---

    ## 四、几个容易踩的坑

    1. **`<<<` 不是 POSIX**
       在 `#!/bin/sh` 脚本里会失败，需改用 `echo ... | sed ...` 或 `printf '%s\n' ... | sed ...`。

    2. **贪婪匹配**
       `.*/` 会匹配到 **最后一个** `/`。若想匹配第一个 `/` 之前的部分，需用非贪婪写法（GNU sed 的 `-E` + `.*?` 并不支持非贪婪，需改用其他技巧）。

    3. **无斜杠时不替换**
       `nofile` 原样输出，这通常符合预期；但若你希望「无斜杠时报错」，sed 做不到，要用 `basename` 或加判断。

    4. **尾部斜杠**
       如 `/a/b/` 会得到空串。若需处理，可先去掉尾部斜杠：
       ```bash
       sed 's#/*$##; s#.*/##' <<< "/a/b/"
       # 输出 b
       ```

    5. **分隔符选择**
       这里用 `#` 是因为模式里有 `/`。你也可以用 `|`、`@` 等任意字符作分隔符，只要正则里不出现它。

    ---

    ## 五、等价写法汇总

    ```bash
    # 1. 用 / 作分隔符（需转义斜杠）
    sed 's/.*\///' <<< "/tmp/dir/target"

    # 2. 用管道代替 here-string
    echo "/tmp/dir/target" | sed 's#.*/##'

    # 3. 用 basename
    basename "/tmp/dir/target"

    # 4. 纯 Bash 参数扩展（无需外部命令，最快）
    p="/tmp/dir/target"; echo "${p##*/}"    # 输出 target

    # 5. awk
    awk -F/ '{print $NF}' <<< "/tmp/dir/target"
    ```

    其中 **`${p##*/}`** 是 Bash 内建的「删除最长前缀匹配」，性能最好，也常被用来替代这行 sed。

    ---

    ## 总结

    - `<<<` 把字符串喂给 sed 的 stdin；
    - `s#.*/##` 用贪婪匹配删掉「最后一个 `/` 及其之前的所有内容」；
    - 整体效果 = 取路径的 basename；
    - 优点是纯 sed、适合流式/批量；局限是尾斜杠、无斜杠等边界情况不如 `basename` 稳健，且 `<<<` 非 POSIX。

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

# Bash Note

Reference: <https://www.computerhope.com/unix.htm>

## cache

* 详细讲讲 mapfile 的用法

    `mapfile`（也叫 `readarray`）是 Bash 4.0 版本开始提供的一个内置命令，用于高效地将文件或标准输入中的**每一行**，完整地读取到一个索引数组（Array）中。相比 `while read` 循环，它的速度要快得多，尤其适合处理大文件。

    `mapfile` 和 `readarray` 是完全等价的，你可以根据自己的习惯选用。它的核心用法如下。

    **基本用法**

    最基础的用法是将一个文件的所有行读入数组，数组的每个元素对应文件中的一行。

    ```bash
    # 将 /path/to/file.txt 的每一行读入名为 my_array 的数组
    mapfile my_array < /path/to/file.txt

    # 如果不指定数组名，默认存入 MAPFILE 数组
    mapfile < /path/to/file.txt
    echo "${MAPFILE[0]}"  # 打印文件第一行
    ```

    **常用选项 (Options)**

    `mapfile` 提供了多个选项，让你能灵活地控制读取过程。

    | 选项 | 说明 | 示例 |
    | :--- | :--- | :--- |
    | **`-t`** | **移除行尾的定界符**（默认为换行符 `\n`）。这是最常用的选项，避免数组中每个元素都带着一个换行符。 | `mapfile -t my_array < file.txt` |
    | **`-n COUNT`** | **最多读取 `COUNT` 行**。如果 `COUNT` 是 0，则读取所有行。 | `mapfile -n 10 my_array < file.txt` # 只读前10行 |
    | **`-s COUNT`** | **跳过开头的 `COUNT` 行**，从第 `COUNT+1` 行开始读取。 | `mapfile -s 5 my_array < file.txt` # 跳过前5行 |
    | **`-O ORIGIN`** | **从数组的 `ORIGIN` 索引处开始赋值**。默认从 0 开始，这会**清空**数组原有内容。 | `mapfile -O 10 my_array < file.txt` # 从索引10开始存 |
    | **`-d DELIM`** | **使用自定义定界符 `DELIM`** 来分隔记录，而不是换行符。此选项需要 Bash 4.4 或更高版本。 | `mapfile -d ':' my_array < file.txt` # 用冒号分隔 |
    | **`-u FD`** | **从指定的文件描述符 `FD`** 读取，而不是标准输入。 | `mapfile -u 3 my_array` # 从文件描述符3读取 |

    **高级技巧与示例**

    1. 读取特定行或范围 (`-s` 与 `-n`)

        结合 `-s` 和 `-n`，可以轻松提取文件中的某一段内容。例如，要读取文件的第 42 行（不包括行尾换行符）：

        ```bash
        # -s 41 跳过前41行，-n 1 只读取1行
        mapfile -t -s 41 -n 1 line_array < file.txt
        printf '%s' "${line_array[0]}"
        ```

        要读取第 42 到 666 行（共 625 行），可以这样计算：

        ```bash
        start_line=42
        end_line=666
        # -s 41 跳过前41行，-n 625 读取625行
        mapfile -t -s $((start_line - 1)) -n $((end_line - start_line + 1)) range_array < file.txt
        printf '%s\n' "${range_array[@]}"
        ```

    2. 处理非换行符分隔的数据 (`-d`)

        `-d` 选项让你能处理用其他字符分隔的数据。一个非常强大的用法是处理以 **null 字符 (`\0`)** 结尾的数据流，这可以有效处理包含空格、换行符等特殊字符的文件名。

        ```bash
        # 使用 find 命令打印以 null 结尾的文件名，并读入数组
        # 注意：mapfile -d '' 表示使用 null 字符作为定界符
        mapfile -d '' file_array < <(find . -type f -print0)
        printf "[%s]\n" "${file_array[@]}"
        ```
        **重要提示**：使用管道 `|` 将数据传给 `mapfile` 会使其在子 shell 中执行，导致数组变量在父 shell 中不可用。推荐使用进程替换 `< <(command)`。

    3. 追加数据到已有数组 (`-O`)

        如果不想覆盖数组原有内容，可以用 `-O` 指定起始索引，实现追加效果。

        ```bash
        my_array=("existing_item")
        # 从当前数组长度处开始追加
        mapfile -t -O "${#my_array[@]}" my_array < additional_data.txt
        ```

    4. 使用回调函数显示进度 (`-C` 与 `-c`)

        对于处理大文件，可以用 `-C callback` 和 `-c quantum` 选项定义一个回调函数，每处理一定行数就执行一次，可以用来显示进度条。

        ```bash
        # 定义一个简单的进度显示函数
        show_progress() {
            echo "已处理第 $1 行: $2"
        }

        # -c 10 每10行调用一次 show_progress 函数
        mapfile -c 10 -C show_progress -t big_array < huge_file.txt
        ```

    **注意事项**

    *   **Bash 版本**：`mapfile` 命令在 Bash 4.0 及更高版本中可用。`-d` 选项需要 Bash 4.4+。如果你的脚本需要在旧版 Bash（如 macOS 默认的 Bash 3.2）中运行，需要使用 `while read` 循环等替代方案。

    *   **默认数组**：如果未指定数组名，`mapfile` 会将数据存入名为 `MAPFILE` 的数组。

    *   **性能**：`mapfile` 是 Bash 内置命令，比管道或循环读取要快得多，是处理大型文本文件的首选。

* 用中文详细讲讲 while IFS= read

    这是一个非常经典且强大的 Bash 用法，常用于**逐行处理文本文件**。下面我为你由浅入深地详细拆解。

    1. 完整命令结构

        最常见的模板是：
        
        ```bash
        while IFS= read -r line; do
            # 对 $line 进行处理
            echo "$line"
        done < "filename.txt"
        ```

    2. 逐个拆解核心零件

        1. `read`

            - **作用**：从标准输入（或文件描述符）中读取**一行**数据。
            - **默认行为**：读到换行符（`\n`）为止，并将该行内容赋值给变量（此处是 `line`）。
            - **注意**：`read` 默认会对反斜杠（`\`）进行转义处理（比如 `\n` 会被解释成换行，`\t` 会被解释成制表符），所以通常配合 `-r` 使用。

        2. `-r` 参数

            - **含义**：`raw`（原始模式）。
            - **作用**：禁止 `read` 对反斜杠进行转义。让反斜杠作为普通字符原样保留。
            - **建议**：**几乎总是加上 `-r`**，除非你明确需要转义功能。

        3. `IFS=` （重点）

            - **全称**：Internal Field Separator（内部字段分隔符）。
            - **默认值**：通常是空格、制表符、换行符（`' \t\n'`）。
            - **这里写成 `IFS=` 是什么意思？**
              - **把 IFS 设为空字符串**。
              - **效果**：**去除行首和行尾的空白字符（空格和 Tab）**。
              - **对比**：如果不加 `IFS=`，`read` 会自动去掉行首和行尾的空格/缩进。
              - **例子**：
                - 文件某行是 `"    Hello World    "`
                - 不加 `IFS=` → 读到的变量是 `"Hello World"`（空格被削掉了）
                - 加 `IFS=` → 读到的变量是 `"    Hello World    "`（保留了原始缩进和尾随空格）

            > **注意语法**：是 `IFS=` 紧挨着 `read`，**没有空格**（`IFS= read`）。这是临时环境变量赋值，只对后面的 `read` 命令生效。

        4. `while ...; do ... done`

            - 循环结构。`read` 每次读取一行，如果有数据则返回状态码 `0`（真），循环继续；读到文件末尾（EOF）返回非零，循环结束。

        5. 输入重定向 `< "filename.txt"`

            - 将文件内容作为整个 `while` 循环的标准输入，`read` 会逐行从该文件中读取。

    3. 为什么要用 `while IFS= read` 而不是 `for` 循环？

        | 方式 | 问题 |
        | :--- | :--- |
        | `for line in $(cat file)` | 会将文件按**空格/换行**拆分成单词，且会展开通配符（`*`），无法处理包含空格的行。 |
        | `while read` | 严格按**换行符**分割，保留每行的完整性。 |

        **所以 `while IFS= read` 是处理包含空格、缩进、特殊符号的文本文件最稳健的方法。**

    4. 完整实战例子

        假设有一个 `data.txt` 文件，内容如下（注意行首和行尾有空格）：

        ```
          Alice   30
         Bob  25  
        Charlie   28
        ```

        **脚本：**
        
        ```bash
        #!/bin/bash
        while IFS= read -r line; do
            echo "原始行: [$line]"
        done < "data.txt"
        ```

        **输出：**

        ```
        原始行: [  Alice   30]
        原始行: [ Bob  25  ]
        原始行: [Charlie   28]
        ```

        （保留了行首空格和行尾空格）

    5. 进阶：同时拆分成多个变量

        如果想按列拆分（比如用空格或冒号分隔），可以给 `read` 多个变量名：

        ```bash
        while IFS=: read -r name age; do
            echo "Name: $name, Age: $age"
        done < "passwd.txt"
        ```

        此时 `IFS=:` 表示以冒号作为分隔符，第一段给 `name`，第二段给 `age`。

    6. 处理管道输入（如 `find | while`）

        ```bash
        find . -name "*.txt" | while IFS= read -r file; do
            echo "找到文件: $file"
        done
        ```

        **注意**：管道会开启子 shell，导致循环内修改的变量在循环外失效。如果需要在循环外保留变量，建议使用重定向 `< <(command)`（进程替换）而不是管道。

    7. 常见陷阱与避坑指南

        | 陷阱 | 解决方案 |
        | :--- | :--- |
        | 忘记写 `-r`，导致 `\` 被吃掉 | 永远写成 `read -r` |
        | 忘记写 `IFS=`，行首空格被删除 | 永远写成 `IFS= read` |
        | 从管道读取时变量赋值失效 | 改用 `while ...; do ...; done < <(command)` |
        | 文件最后一行没有换行符 | `read` 依然能正常读取（只要不是空文件） |

    8. 终极记忆口诀

        > **"无空格，不转义，逐行读，最稳健"**
        > —— `IFS=`（去空格）+ `-r`（防转义）= 逐行原样读取。

    如果你需要我进一步讲解如何搭配 `数组` 或 `关联数组` 处理复杂文本，或者结合 `sed/awk` 对比，可以告诉我。

* `$()`命令替换会去除字符串中的换行符，但如果文件名包含换行符（极少数情况），会导致补全结果错乱

    example:

    ```bash
    echo -e "hello\nworld\n"
    ```

    output:

    ```
    hello
    world

    ```

    而

    ```bash
    echo $(echo -e "hello\nworld\n")
    ```

    输出为

    ```
    hello world
    ```

* echo -e（启用反斜杠转义）

    example:

    ```bash
    echo -e "hello\nworld"
    ```

* `printf`可以直接输出`\n`

    ```bash
    printf "hello\nworld\n"
    ```

* 在单引号或双引号内按 Enter 也可以换行：

    ```bash
    echo "hello
    world"
    ```

* 普通 echo（不加 -e）默认不会解析 \n，会直接输出 hello\nworld

* `find ... | sed ... | compgen -W "$(cat)" -- "$cur"`

    example:

    ```bash
    find . -name main* | sed 's|^./||' | compgen -W "$(cat)" -- "ma"
    ```

    output:

    ```
    main.sh
    ```

    这种写法比`$(compgen -W "$(find xxx | sed xxx)" -- "$cur")`稍微自然一点。

* `$(< <(find . -type f | sed 's|^\./||'))`详细解释

    将当前目录下（包含子目录）所有文件的文件名（去掉开头的 ./）作为一个字符串数组/列表，传递给外层的命令（由于外层没有具体命令，它会被当作参数列表，通常会导致报错或输出第一个文件名）。

    * `find . -type f`

        递归查找当前目录（.）下的所有普通文件（-type f）。

    * `<( ... )`: 进程替换

        这是 Bash 特有的语法（不是所有 Shell 都支持）。

        作用：将括号内命令的输出，伪装成一个临时文件（FIFO 或 /dev/fd/63）。即返回的是一个文件路径（比如 /dev/fd/63），这个文件的内容就是刚才的命令的输出字符串。

    * `< <( ... )`: 输入重定向

        这里把 `<` 和 `<( ... )` 结合，将进程替换生成的临时文件，作为标准输入（stdin）。

        相当于：`< /dev/fd/63`，把文件内容喂给外层命令。

    * `$(< <(...))`

        执行一个空命令（因为没有外层命令），但将它的标准输入重定向到临时文件。然后 $() 捕获这个命令的输出。

        在 Bash 中，如果 $() 内部只有一个输入重定向而没有实际命令，它会将这个重定向的文件的内容删掉所有換行符，然后作为命令行的参数，并执行。

        example:

        ```bash
        # hello.txt
        echo hello
        echo world
        ```

        run:

        `$(< hello.txt)`

        output:

        ```
        hello echo world
        ```

        可以看到，原文件内容被删除了`\n`，并变成了命令：`echo hello echo world`，所以就输出了 echo 的内容。

        如果使用变量来保存的话，那么效果是存入变量或数组：

        ```bash
        a=$(< hello.txt)
        echo $a
        a=($(< hello.txt))
        echo $a
        echo ${a[@]}
        ```

        output:

        ```
        echo hello echo world
        echo
        echo hello echo world
        ```

* `compgen` 或许可以被替换成 `mapfile`:

    `mapfile -t COMPREPLY < <(find . -type f -name "*${cur}*" | sed 's|^\./||')`

    compgen 会自动转义空格和特殊字符，但是这里是 mapfile 是按行读取的，所以没有问题。

    如果写成这样：`COMPREPLY=($(find ...))  # 没有 mapfile，直接用命令替换`，可能会变成`COMPREPLY=("aaa bbb cc dd")`，即变成了 4 个元素。如果是 compgen ，则会变成`COMPREPLY=("aaa bbb cc\ dd")`，对空格进行转义。

* bash 中检查某个函数是否存在：

    ```bash
    type <func_name>
    # 应该显示 "<func_name> is a function" 或 "<func_name> is aliased to ..."
    ```

* 详细讲讲 compgen 的用法

    `compgen` 是 Bash 的一个内置命令，用于**生成可能的自动补全匹配项**。它是 Bash 可编程补全功能的核心，当你在终端里按下 `Tab` 键时，背后就有它的身影。

    你可以把它理解成一个用于“查询”或“生成”补全候选列表的工具，通常有三种核心用法：

    ### 🚀 核心用法一：快速查询系统信息

    这是最直接的用法，可以迅速列出系统中各类可用的名称，非常适合用来探索和调试。

    | 选项 | 功能描述 | 示例 |
    | :--- | :--- | :--- |
    | `-c` | 列出所有可执行的命令（包括内置命令、函数和`$PATH`里的程序） | `compgen -c` |
    | `-a` | 列出所有已定义的别名 | `compgen -a` |
    | `-b` | 列出所有 Shell 内置命令 | `compgen -b` |
    | `-k` | 列出所有 Shell 保留关键字（如 `if`, `for`, `then`） | `compgen -k` |
    | `-A function` | 列出所有 Shell 函数 | `compgen -A function` |
    | `-v` / `-e` | 列出所有变量 / 环境变量 | `compgen -v` |
    | `-u` / `-g` | 列出系统用户 / 用户组 | `compgen -u` |

    > **注意**：`-A` 选项后面需要跟一个“动作（action）”，比如 `function`、`variable`、`service` 等，可以实现与单字母选项相同的功能，并且能覆盖更多类型，如 `-A hostname` 列出主机名。

    ### 🎯 核心用法二：带前缀过滤的查询

    你可以在命令后面加上一个“单词（word）”，这样 `compgen` 就只会**列出以该单词开头的匹配项**。

    ```bash
    # 列出所有以 'git' 开头的命令
    compgen -c git

    # 列出所有以 'HO' 开头的变量
    compgen -v HO
    ```

    ### ⚙️ 核心用法三：高级补全生成

    这部分是 `compgen` 真正强大的地方，也是编写自定义补全脚本时的关键。

    #### 1. 自定义单词列表 (`-W`)
    `-W` 选项允许你提供一个用空格分隔的单词列表作为补全候选。这在脚本中非常实用。
    ```bash
    # 从列表中匹配以 's' 开头的项
    compgen -W "start stop restart status" s
    # 输出: start, stop, status
    ```

    #### 2. 文件名匹配 (`-G`)
    `-G` 选项允许你使用通配符（Glob）模式来匹配文件名作为补全结果。
    ```bash
    # 列出当前目录下所有 .txt 文件
    compgen -G "*.txt"
    ```

    #### 3. 添加前缀和后缀 (`-P`, `-S`)
    这两个选项可以为所有生成的补全项统一添加前缀或后缀。
    ```bash
    # 为所有目录名添加前缀和后缀
    compgen -d -P "dir-" -S "/"
    ```

    #### 4. 过滤结果 (`-X`)
    `-X` 选项可以根据一个模式**过滤掉**匹配的项。如果模式以 `!` 开头，则效果反转，即只保留不匹配的项。
    ```bash
    # 列出所有命令，但排除包含 'grep' 的命令
    compgen -c -X "*grep*"

    # 只列出包含 'grep' 的命令
    compgen -c -X "!*grep*"
    ```

    ### 🛠️ 实战演练：自定义命令补全

    `compgen` 最常见的实战场景是与 `complete` 命令配合，为自定义脚本添加补全功能。

    ```bash
    # 1. 定义一个补全函数
    _my_script_completions() {
        # 获取当前正在输入的单词
        local cur="${COMP_WORDS[COMP_CWORD]}"
        
        # 使用 compgen 生成补全列表，并存入 COMPREPLY 数组
        COMPREPLY=( $(compgen -W "build test deploy --verbose --help" -- "$cur") )
    }

    # 2. 将补全函数关联到你的脚本
    complete -F _my_script_completions my_script
    ```

    这样，当你在终端输入 `my_script ` 后再按 `Tab` 键时，就会自动补全 `build`、`test` 等选项了。

    ### 💎 总结

    *   **`compgen` 是 `Tab` 补全背后的核心引擎**，用于**生成**候选列表。
    *   它的基本用法是**查询信息**（`-c`, `-a`, `-v` 等）。
    *   它的高级用法用于**生成自定义补全**（`-W`, `-G`, `-P`, `-S`, `-X`）。
    *   在编写补全脚本时，`compgen` 与 `complete` 和 `COMPREPLY` 数组是黄金搭档。

* `COMPREPLY=( ... )`：这是 Bash 补全机制的核心数组变量。当用户按下 Tab 键时，Bash 会读取这个数组的内容，并将其作为候选词显示在终端上。

* 详细讲讲 bash 中的 Ctrl + S 和 CTRL + R

    在 Bash 中，`Ctrl + S` 和 `Ctrl + R` 都是与**终端输入输出控制**和**历史命令管理**密切相关的快捷键。但它们的功能截然不同，而且 `Ctrl + S` 有一个非常著名的“坑”会让新手误以为终端卡死了。

    下面为你详细拆解这两个快捷键的底层原理和使用场景。

    ---

    ### 1. Ctrl + S：终端流量控制（XOFF）

    **功能**：**暂停终端输出**（Flow Control / Stop）。

    当你按下 `Ctrl + S` 时，你并不是在“保存”任何东西（那是图形软件里的快捷键），而是在向终端发送一个 **XOFF**（停止传输）信号。

    -   **现象**：终端屏幕会瞬间冻结。你继续打字或者程序正在输出内容，屏幕上都不会有任何变化，看起来像是“卡死”或“无响应”了。
    -   **底层原理**：这是 Unix/Linux 系统古老的软件流控制（Software Flow Control）机制。终端会阻塞程序的输出，防止屏幕被刷爆，但程序实际上**依然在后台正常运行**，只是输出被挂起了。

    **如何解除（救命技能）**：
    按下 **`Ctrl + Q`**（XON / 恢复传输）。
    按下后，之前被暂停的所有输出会瞬间全部喷涌而出，终端恢复响应。

    > **避坑指南**：
    > 很多 Vim 用户习惯按 `Ctrl + S` 保存文件，在终端下误按后会卡死。此时千万不要强行关闭终端或重启电脑，只需按 `Ctrl + Q` 即可恢复。

    **如何永久禁用这个功能**：
    如果你觉得这个快捷键很烦人，可以在 `~/.bashrc` 中添加 `stty -ixon` 来关闭 XON/XOFF 流量控制，之后 `Ctrl + S` 和 `Ctrl + Q` 将不再起任何作用。

    ---

    ### 2. Ctrl + R：反向历史搜索（Reverse-i-search）

    **功能**：**在命令历史中逆向增量搜索**（Reverse Search History）。

    这是 Bash 中最强大的“时光机”功能，用于快速找回之前输入过的长命令。

    -   **现象**：按下后，提示符会变成 `(reverse-i-search)` 并等待你输入关键词。
    -   **操作逻辑**：
        1. 按下 `Ctrl + R`。
        2. 开始输入命令的某一部分（哪怕是中间的几个字母）。
        3. Bash 会实时显示最近一条匹配该关键词的历史命令（高亮显示匹配部分）。
        4. **继续按 `Ctrl + R`**：会向上回溯，查找更早之前的匹配记录。
        5. **按 `Ctrl + S`（特殊用法）**：在某些配置下，按下 `Ctrl + S` 可以**正向搜索**（即从旧记录向新记录搜索），但这通常需要你先关闭流量控制功能（`stty -ixon`），否则它会触发上面的屏幕冻结。

    **找到命令后的操作**：

    -   按 **`Enter`**：直接执行该命令。
    -   按 **`Tab`** 或 **`Esc`** 或 **`左右方向键`**：将命令调取到当前命令行中，供你编辑修改后再执行。
    -   按 **`Ctrl + G`**：放弃搜索，退出搜索模式。

    ---

    ### 3. 两者的对比总结

    | 特性 | **Ctrl + S** | **Ctrl + R** |
    | :--- | :--- | :--- |
    | **核心作用** | 暂停/冻结屏幕输出（XOFF） | 反向搜索历史命令 |
    | **常见误解** | 以为是“保存”，导致终端卡死 | 以为是普通查找，需配合输入关键词 |
    | **解除/退出方式** | 按 **`Ctrl + Q`** 解除冻结 | 按 **`Ctrl + G`** 或 `Enter` 退出搜索 |
    | **操作对象** | 当前正在输出的程序（如日志、编译过程） | 存储在 `~/.bash_history` 中的历史指令 |
    | **配合使用** | 若想用 `Ctrl + S` 做正向搜索，需先执行 `stty -ixon` 取消流量控制 | 默认逆向搜索，反复按可跳转更早匹配项 |

    ---

    ### 4. 高级技巧：让 Ctrl + R 更好用

    原生的 `Ctrl + R` 只能按时间顺序一条条翻找，效率略低。如果你经常使用它，强烈建议安装 **`fzf`**（模糊查找器）。

    安装 `fzf` 后，按下 `Ctrl + R` 会弹出一个**可预览的模糊匹配列表**，你可以用方向键上下选择，或者直接输入空格分隔的关键词进行模糊过滤，查找命令的效率会提升数倍。

    ---

    ### 5. 扩展提醒（极易踩坑）

    如果你在 **SSH 远程连接** 的终端中按下 `Ctrl + S`，**冻结的是你的本地终端显示**，而不是远程服务器。此时：

    1.  远程服务器上的程序依然在跑。
    2.  你的键盘输入依然被 SSH 客户端接收，只是不显示在屏幕上。
    3.  千万不要断网或重启 SSH，只需要心平气和地按一下 **`Ctrl + Q`**，一切就会恢复原样。

    如果你还想了解如何自定义这些快捷键，或者想知道 `Ctrl + Z`（挂起进程）和 `Ctrl + C`（中断进程）的区别，我可以继续为你讲解。

* bash 自动补全文件名

    * 方案一：直接补全当前目录下的文件（最简单的情况）

        在 `~/.bashrc` 中添加：

        ```bash
        hvim() {
            if [ $# -eq 0 ]; then
                echo "Usage: ffind <filename>"
                return 1
            fi
            # 经过补全后，补全的内容直接就是应用的参数，可以直接使用
            vim "$1"
        }

        # 定义 find 风格的补全函数
        _hvim_complete() {
            local cur="${COMP_WORDS[COMP_CWORD]}"
            # 使用 find 搜索当前目录及子目录中的文件
            # 这里搜索的是以 cur 开头的文件，但是搜索出来的路径带有前缀目录
            # 2>/dev/null 剔除没有权限的目录的输出
            cands=$(find . -type f -name "${cur}*" 2>/dev/null | sed 's|^\./||')
            COMPREPLY=($(compgen -W "${cands}" -- "$cur"))
        }

        # complete 只能作用于已有的命令或函数，所以我们在开头额外创建了个 hvim 函数
        complete -F _hvim_complete hvim
        ```

        然后使用：

        ```bash
        hvim myfile<Tab>
        ```

        会列出所有包含 `myfile` 的文件名。

        **详细解释：**

        * `${COMP_WORDS[...]}`：这是一个数组元素引用。`COMP_WORDS` 是 Bash 自动补全系统维护的一个特殊数组，它保存了当前正在输入的命令行被空格分割后的所有单词。
        
            如果有引号，比如`"he llo`或者`"he llo"`，那么这算作一个单词，而不是两个。如果有引号，引号也算在单词内。

        * `COMP_CWORD`：这是一个特殊整数变量，表示当前光标（光标位置）所在的那个单词，在 COMP_WORDS 数组中的索引下标。索引从 0 开始计数。

        * `${COMP_WORDS[COMP_CWORD]}`: 从当前输入的命令行中，取出光标正在输入的那个单词。

        * `sed 's|^\./||`: 删除每一行开头出现的 ./ 字符。

            sed 标准格式：`sed 's|旧内容|新内容|'`

            * s：代表替换（substitute）操作。

            * |：是分隔符（通常习惯用 /，但这里用了 |，目的是为了避免与路径中的斜杠混淆，让代码更清晰）。

            * `^\./`: 精确匹配以 ./ 开头的这三个字符

            * `||`: 替换为空字符串。

        * `COMPREPLY=($(compgen -W "${cands}" -- "$cur"))`

            `COMPREPLY`是个特殊 bash 数组变量，用于输出所有的候选词。

            `compgen -W "${cands}" -- "$cur"`: 从候选词中筛选出以`$cur`开头的词语构成数组。`-W`表示 words，表示候选词。`--`表示选项的结束，防止`$cur`被当作 cands。`$cur`表示当前 bash 正在输入的词。

        **潜在问题：**

        * find 搜索的输出包含前缀路径，比如`subdir/myfile.txt`，但是`compgen -W "${cands}" -- "${cur}"`只筛选以`${cur}`开头的词，假如 cur 是 myf，那么就无法补全到`subdir/myfile.txt`。

            所以这个代码只适用于当前目录，无法补全子目录中的文件。而对于当前目录，bash 有默认补全行为，可以直接补全当前目录下的文件。我们写这个 hvim 纯粹是又造了个轮子，所以称其为“最简单的情况”。

    * 方案二：按路径逐级展开

        ```bash
        # 在 ~/.bashrc 中添加
        _find_like_vim() {
            local cur="${COMP_WORDS[COMP_CWORD]}"
            # 搜索当前目录及子目录（排除隐藏文件）
            COMPREPLY=($(compgen -W "$(find . -type f -not -path '*/.*' 2>/dev/null | sed 's|^\./||')" -- "$cur"))
        }
        complete -F _find_like_vim findfile
        ```

        使用：

        ```bash
        findfile sub<tab>  # 展开为 findfile subdir/he
        findfile subdir/hell<tab>  # 展开为 findfile subdir/hello.txt
        ```

        详细解释：

        * `find . -type f -not -path '*/.*'`: 在当前目录（包括所有子目录）中，查找所有普通文件，但排除那些路径中包含隐藏目录的文件。

            简单说就是：只找“非隐藏目录”里的普通文件。

            实际效果如下：

            ```
            ./script_2.sh
            ./subdir/hello.txt
            ./subdir/he_world.txt
            ./main.sh
            ./script.sh
            ```

            经过 sed 删除前缀`./`后，即得到纯净的路径。

        * `-type f`

            限定只查找普通文件（regular file）。

            排除目录、符号链接、设备文件等。

        * `-not -path '*/.*'`

            这是核心过滤条件，分两层理解：

            * `-path '*/.*'`

                * `-path` 会匹配整个路径字符串（不是仅文件名）。

                * `'*/.*'` 是一个通配符模式：

                    * `*/` 表示任意层级的目录（包括当前目录）

                    * `.*` 表示以点开头的文件名或目录名

                * 所以 `*/.*` 会匹配任何路径中包含隐藏元素的情况，例如：

                    * `./.git/config`

                    * `./src/.cache/temp`

                    * `./.env`

            * `-not`

                * 逻辑取反，即不匹配上述模式。

                * 最终效果：排除所有路径中包含隐藏目录或隐藏文件的条目。

        这个功能其实 bash 也是默认实现的，我们再次造了轮子。

        **潜在问题：**

        * 这个本质是 find 把所有非隐藏文件列出来，然后让 compgen 使用`-- $cur`前缀做选择。因为`$cur`只能用作前缀，所以我们只能递进地补全路径。无法像 vim find 那样，输入文件名，然后自动展开为完整路径。

    * 方案三：搜索子目录 + 匹配文件名（自定义写法）

        ```bash
        hvim() {
            if [ $# -eq 0 ]; then
                echo "Usage: ffind <filename>"
                return 1
            fi
            vim "$1"
        }

        _hvim_complete() {
            local cur="${COMP_WORDS[COMP_CWORD]}"
            cands=$(find . -type f -name "${cur}*" 2>/dev/null | sed 's|^\./||')
            COMPREPLY=($(compgen -W "${cands}" -X '!'"*$cur*"))
        }

        complete -F _hvim_complete hvim
        ```

        **详细解释：**

        * `-name "${cur}*"`: 只搜索`${cur}`开头的文件名。注意此时 find 输出的是完整路径。

        * `-X '!'"*$cur*"`: `-X`表示“排除这些通配符匹配到的 word”
        
            * `'!'`：单引号包裹的叹号。首先叹号表示取反，即“不排除这些 word”，即“只包含这些 word”。单引号表示不要把叹号转义成 bash 的“历史命令”功能。可以使用`set -o | grep histexpand`查看当前 bash 是否开启对 ! 历史扩展（History Expansion）的转义。

            * `"*$cur*"`：整个路径中只要包含`$cur`就可以，防止`-- "$cur"`无法匹配到`subdir/myfile.txt`的输出。这里用双引号包裹，防止`$cur`中有空格。

        * 为什么不直接用 grep 过滤？

            因为 compgen 额外提供了自动转义功能，这在补全包含空格的文件名时至关重要：

            ```bash
            # 假设文件名为 "my file.txt"
            # 直接用 grep 输出：my file.txt（未转义，会导致补全错误）
            # compgen 输出：my\ file.txt 或 "my file.txt"（已转义，安全可用）
            ```

        **潜在问题：**

        * `"*$cur*"`匹配的范围有点太大了，其实应该只匹配最后一段路径的内容，比如`../../myfile.txt`，应该只匹配`myfile.txt`。
* bash 快捷键

	```bash
	# 先输入 rm，然后按 Alt+. 或 Esc+. 
	# 这会自动插入上一条命令的最后一个参数
	rm 2026.07.23.md  # Alt+. 会自动补全文件路径
	```

* 详细讲讲 alias 的用法

    好的，我们来详细讲讲 Linux/macOS 中 `alias` 的用法。`alias` 是一个 shell 内置命令，用来为长命令或复杂命令创建**快捷键**（别名）。用好它，可以极大地提高你的命令行操作效率。

    下面我从基础到进阶，分几个部分详细讲解。

    ---

    ### 1. 基础用法

    **基本语法：**
    ```bash
    alias 别名='命令'
    ```

    **注意：**
    - 等号两边**不能有空格**。
    - 如果命令中包含空格或特殊字符，必须用引号（单引号或双引号）括起来。
    - 别名通常用小写字母，但也可以混合大小写。

    **基础示例：**
    ```bash
    # 简单缩写
    alias ll='ls -alF'
    alias gs='git status'
    alias c='clear'

    # 带常用选项
    alias rm='rm -i'        # 删除前询问确认
    alias cp='cp -i'        # 覆盖前询问
    alias df='df -h'        # 以人类可读格式显示磁盘空间
    alias du='du -h -c'     # 汇总目录大小
    ```

    ---

    ### 2. 查看与删除别名

    - **查看所有已定义的别名：**
      ```bash
      alias
      ```
      或者查看特定别名：
      ```bash
      alias 别名
      # 例如：alias ll
      ```

    - **删除（取消）别名：**
      ```bash
      unalias 别名
      # 例如：unalias ll
      ```

    - **临时绕过别名（执行原始命令）：**
      如果你定义了 `alias rm='rm -i'`，但某次想强制删除不想确认，可以在命令前加反斜杠：
      ```bash
      \rm -rf temp_folder
      ```
      或者使用完整路径 `/bin/rm -rf temp_folder`。

    ---

    ### 3. 进阶用法：组合与嵌套

    别名可以执行多条命令，也可以调用其他别名。

    **执行多条命令：**
    用分号 `;` 或逻辑运算符 `&&`、`||` 连接。
    ```bash
    # 进入目录并列出内容
    alias cdls='cd /var/log && ls -la'

    # 创建目录并进入
    alias mcd='mkdir -p $1 && cd $1'   # 注意：此处 $1 不生效，后面会讲函数
    ```

    **嵌套别名（别名里调用另一个别名）：**
    ```bash
    alias ll='ls -l'
    alias la='ll -A'   # 别名 la 调用了别名 ll，可以正常工作
    ```

    ---

    ### 4. 带参数怎么办？—— 使用 Shell 函数

    **核心要点：** `alias` **不支持**直接传递位置参数（如 `$1`, `$2`）。如果你需要参数，必须使用 **Shell 函数**，然后给函数起一个短名字。

    **错误示范（不会按预期工作）：**
    ```bash
    alias mcd='mkdir -p $1 && cd $1'   # $1 不会传给 alias
    ```

    **正确做法（用函数替代 alias）：**
    ```bash
    # 定义函数：创建目录并进入
    mcd() {
      mkdir -p "$1" && cd "$1"
    }

    # 或者写成一行（注意分号）
    mcd() { mkdir -p "$1" && cd "$1"; }
    ```

    **带多个参数的函数示例：**
    ```bash
    # 查找并 grep 内容
    findgrep() {
      find "$1" -type f -exec grep -l "$2" {} \;
    }
    # 使用：findgrep /home/user "error"
    ```

    ---

    ### 5. 如何让别名永久生效？

    你在终端直接输入 `alias ll='ls -alF'`，只对**当前终端会话**有效。关闭终端就没了。

    要让别名永久生效，需要把它写入 shell 的配置文件。

    | **Shell 类型** | **全局配置文件（所有用户）** | **用户级配置文件（推荐）** |
    | :--- | :--- | :--- |
    | **Bash** | `/etc/bash.bashrc` 或 `/etc/profile` | `~/.bashrc` （登录 shell 可能是 `~/.bash_profile` 或 `~/.profile`） |
    | **Zsh** (macOS 默认) | `/etc/zshrc` | `~/.zshrc` |

    **推荐做法：**
    1. 打开你的配置文件：`vim ~/.bashrc` 或 `vim ~/.zshrc`
    2. 在文件末尾添加别名定义：
       ```bash
       # 我的自定义别名
       alias ll='ls -alF'
       alias update='sudo apt update && sudo apt upgrade'   # Ubuntu/Debian
       alias ip='ip -c'      # 带颜色显示
       ```
    3. 保存退出后，让配置立即生效：
       ```bash
       source ~/.bashrc   # 或 source ~/.zshrc
       ```

    ---

    ### 6. 实用且高效的 alias 示例合集

    这里分享一些我最常用且能显著提效的别名：

    #### 系统与文件管理
    ```bash
    # 列出目录
    alias l='ls -CF'
    alias ll='ls -alF'
    alias la='ls -A'
    alias lr='ls -ltr'   # 按时间排序（最新在最后）

    # 安全操作
    alias rm='rm -i'
    alias mv='mv -i'
    alias cp='cp -i'

    # 磁盘与内存
    alias df='df -h'
    alias du='du -h -c'
    alias free='free -m'   # 以 MB 显示内存

    # 历史命令
    alias h='history'
    alias hg='history | grep'   # 搜索历史命令
    ```

    #### Git 快捷操作
    ```bash
    alias gs='git status'
    alias ga='git add'
    alias gc='git commit -m'
    alias gp='git push'
    alias gl='git log --oneline --graph --decorate'
    alias gd='git diff'
    alias gco='git checkout'
    alias gb='git branch'
    ```

    #### 网络相关
    ```bash
    # 查看端口占用
    alias ports='sudo netstat -tulpn | grep LISTEN'
    # 或者用 lsof
    alias port='lsof -i :'   # 后面加端口号，如 port 8080

    # 查看外网 IP
    alias myip='curl ifconfig.me'
    # 或
    alias myip='dig +short myip.opendns.com @resolver1.opendns.com'
    ```

    #### 程序启动
    ```bash
    # Python
    alias py='python3'
    alias pi='pip install'
    alias pir='pip install -r requirements.txt'

    # 编辑器
    alias v='vim'
    alias s='subl .'   # 用 Sublime 打开当前目录
    alias c='code .'   # 用 VSCode 打开当前目录
    ```

    #### 便捷导航
    ```bash
    # 快速到常用目录
    alias docs='cd ~/Documents'
    alias down='cd ~/Downloads'
    alias work='cd ~/workspace'
    alias ..='cd ..'
    alias ...='cd ../..'
    alias ....='cd ../../..'
    ```

    ---

    ### 7. 高级技巧

    #### 根据操作系统/主机区分别名
    有时你在不同机器上（如 Linux 服务器 vs macOS 本地），命令参数不同，可以加条件判断：
    ```bash
    # 在 ~/.bashrc 中
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        alias ls='ls --color=auto'
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        alias ls='ls -G'   # macOS 用 -G 开启颜色
    fi
    ```

    #### 安全提醒：警惕 `alias sudo='sudo '`
    如果你想给 `sudo` 后面的命令也自动应用别名（比如 `sudo ll` 能生效），可以定义：
    ```bash
    alias sudo='sudo '    # 注意末尾有一个空格！
    ```
    当别名末尾有空格时，shell 会解析空格后的下一个单词，并检查它是否也是别名。这样 `sudo ll` 就会先展开 `ll`。

    ---

    ### 8. 注意事项与陷阱

    1. **别名的优先级**：别名优先于函数和内置命令，但低于 `which` 找到的外部命令路径。
    2. **脚本中慎用**：在 shell 脚本（`.sh` 文件）里，默认**不加载** `~/.bashrc`，所以别名在脚本中不可用。如果需要，可以在脚本里 `source ~/.bashrc`，但更推荐在脚本中直接写完整命令或函数。
    3. **避免与系统命令重名**：例如 `alias ls='ls -l'`，如果你又需要原始 `ls`，就得用 `\ls`。这会增加记忆负担，所以尽量起新名字（如 `ll`）。
    4. **引号的使用**：如果别名值中有变量或命令替换，使用双引号会在定义时展开；使用单引号会保留为字面量，在调用时展开。通常推荐用**单引号**。
       ```bash
       # 双引号：定义时立刻展开，获取当前时间
       alias now="date +%T"   # 这里会把 date +%T 的执行结果存进去，错误！
       
       # 单引号：调用时才展开（正确）
       alias now='date +%T'   # 每次输入 now，都会执行 date 命令
       ```

    ---

    ### 总结

    | **功能** | **命令/方法** |
    | :--- | :--- |
    | 创建临时别名 | `alias 短名='长命令'` |
    | 永久生效 | 写入 `~/.bashrc` 或 `~/.zshrc` |
    | 查看所有别名 | `alias` |
    | 删除别名 | `unalias 短名` |
    | 需要参数时 | 用 Shell **函数**代替别名 |
    | 绕过别名 | 用 `\命令` 或绝对路径 |

    掌握 `alias` 是成为命令行高手的必经之路。花 10 分钟整理一套自己顺手的别名，长期来看能节省大量时间。建议从今天就开始累积你的 `.bashrc` 配置文件吧！如果还有特定场景（比如 Docker、Kubernetes 别名）需要进一步展开，随时告诉我。

* 使用 mv 代替 rm 的一个技巧

    ```bash
    hlc_rm() {
        if [ -z "$1" ]; then
            echo "错误：请指定要删除的文件" >&2
            return 1
        fi
        mv -- "$1" "/tmp/hlc_trash/"
    }
    alias rm='hlc_rm'
    ```

	解释：

	* -- 是"参数结束标记"，防止文件名以 - 开头被误认为选项（如 rm "-f"）

	* 检查参数是否为空，避免误操作

    这种定法主要是为了处理 rm 不带参数时的报错：

	`rm ""` 或 `rm` 不带参数:

	不加引号的版本：

	```bash
	mv  /tmp/hlc_trash/
	```

	这时 `$1` 为空，mv 接收到的参数变成：

	* 参数1：（空）

	* 参数2：`/tmp/hlc_trash/`

	mv 会报错 missing file operand

* 我在 .env 中定义了 PYTHONPATH=..，如何才能在 bash 里 export .env

    有几种方法可以在 bash 中导出 .env 文件中定义的环境变量：

    ## 方法1：使用 source 命令（推荐）
    ```bash
    source .env
    ```
    或者
    ```bash
    . .env
    ```

    ## 方法2：使用 export 命令
    ```bash
    export $(grep -v '^#' .env | xargs)
    ```

    ## 方法3：逐行处理（更安全）
    ```bash
    while IFS= read -r line; do
        # 跳过空行和注释
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        
        # 导出变量
        export "$line"
    done < .env
    ```

    ## 方法4：使用专门的工具
    如果你经常需要处理 .env 文件，可以安装 `dotenv` 工具：
    ```bash
    # 安装 dotenv-cli
    npm install -g dotenv-cli

    # 使用
    dotenv -e .env your-command
    ```

    ## 验证是否生效
    导出后，可以验证 PYTHONPATH 是否设置成功：
    ```bash
    echo $PYTHONPATH
    ```

    ## 注意事项
    - 确保 .env 文件中的变量定义格式正确：`VARIABLE_NAME=value`
    - 如果值包含空格，需要用引号括起来
    - 以 `#` 开头的行会被视为注释
    - 方法2和方法3会自动跳过空行和注释

    **推荐使用方法1**，因为它简单直接且兼容性最好。

    注：

    1. 这个之前好像研究过，就是 dotenv 工具

    * source .env 似乎不行，它只是在当前环境中定义了PYTHONPATH，并不是 export 属性

        你说得对！`source .env` 只是在当前 shell 进程中设置变量，但不会自动导出（export）给子进程。有几种解决方案：

        * 使用 awk 或 sed 处理

            ```bash
            # 方法2a: 使用 awk
            source <(awk -F= '{print "export "$1"="$2}' .env)

            # 方法2b: 使用 sed
            source <(sed 's/^/export /' .env)
            ```

        * 逐行处理并导出

            ```bash
            while IFS= read -r line; do
                # 跳过空行和注释
                [[ -z "$line" || "$line" =~ ^# ]] && continue
                
                # 获取变量名和值
                var_name="${line%%=*}"
                var_value="${line#*=}"
                
                # 导出变量
                export "$var_name"="$var_value"
            done < .env
            ```

        * 使用 envsubst 配合 source

            ```bash
            source <(envsubst < .env)
            ```

        * 如果只需要 PYTHONPATH

            ```bash
            export PYTHONPATH=..
            # 或者从 .env 读取
            export PYTHONPATH=$(grep PYTHONPATH .env | cut -d '=' -f2)
            ```

        * 如果要从 `.env` 文件中读取：

            ```bash
            export $(cat .env | xargs)
            ```

        * 使用 set -a

            ```bash
            set -a
            source .env
            set +a
            ```

            `set -a` 会自动标记之后所有定义的变量用于导出，`set +a` 关闭这个功能。

* 详细讲讲 bash 中的 LANG，LC_ALL 等环境变量

	在 Bash 中，`LANG` 和 `LC_ALL` 等环境变量是 **Locale（区域设置）** 系统的核心。它们决定了命令行程序如何显示文本、处理字符、排序、格式化时间和数字。

	要彻底讲清楚，我们需要从**机制**、**优先级**和**实战场景**三个维度来拆解。

	---

	### 1. 核心概念：什么是 Locale？

	Locale 是一组定义用户语言、国家/地区和文化习惯的环境变量。它由 `glibc`（GNU C 库）提供支持。

	一个完整的 Locale 名称通常格式为：`语言_地区.字符编码`，例如：

	- `zh_CN.UTF-8`：中国中文，UTF-8编码
	- `en_US.UTF-8`：美国英文，UTF-8编码
	- `C` 或 `POSIX`：这是特殊的回退模式，表示兼容 ASCII，不启用任何国际化支持（速度最快，无乱码风险）。

	---

	### 2. 分类详解（变量家族）

	Locale 设置被切分成了多个独立的类别，以便精细控制：

	| 环境变量 | 作用范围 |
	| :--- | :--- |
	| **`LANG`** | **默认值（兜底）**。如果下面某个具体变量未设置，则使用 `LANG` 的值。 |
	| **`LC_ALL`** | **终极覆盖（强制）**。它的优先级最高，一旦设置，会**覆盖**所有其他 `LC_*` 变量和 `LANG`。 |
	| **`LC_CTYPE`** | **字符分类与编码**（最重要）。决定终端能识别什么字符，影响中文输入法、正则表达式匹配（`[a-z]` 的范围）。 |
	| **`LC_COLLATE`** | **排序规则**。决定 `sort` 命令和 `ls` 列出的文件名顺序（按拼音还是按字母）。 |
	| **`LC_TIME`** | **时间格式**。决定 `date` 命令显示星期/月份的语言。 |
	| **`LC_NUMERIC`** | **数字格式**。决定小数点（`.` 还是 `,`）和千位分隔符。 |
	| **`LC_MONETARY`** | **货币格式**。决定货币符号和位置。 |
	| **`LC_MESSAGES`** | **程序提示语**。决定 `ls --help` 或 `bash` 报错信息显示为中文还是英文。 |
	| **`LC_PAPER`**、`LC_NAME` 等 | 纸张大小、姓名格式等次要设置。 |

	---

	### 3. 终极规则：优先级（从高到低）

	这是最容易混淆的地方，记住这个顺序即可：

	1. **最高级：`LC_ALL`** —— 一旦设定，绝无例外。
	2. **中级：`LC_*`**（如 `LC_TIME`, `LC_CTYPE`）—— 针对具体功能。
	3. **最低级（保底）：`LANG`** —— 当以上变量都未设置时，才使用此值。

	**另外还有一个特殊变量 `LANGUAGE`**（仅 GNU gettext 使用）：
	它用于程序提示语的**多语言回退**，优先级高于 `LC_MESSAGES` 但低于 `LC_ALL`。例如 `LANGUAGE="zh_CN:en"` 表示优先显示中文，没有中文则回退到英文。

	---

	### 4. 实战场景与常见误区

	#### 场景 A：SSH 连上服务器全是乱码
	服务端是中文，客户端是英文终端。
	```bash
	# 错误示范（强行覆盖）
	export LC_ALL=zh_CN.UTF-8
	# 如果服务器没安装中文字体或 locale，会报错 "locale: Cannot set LC_CTYPE"
	```

	**正确做法**：仅修改提示语言，保留字符编码为英文（兼容性最好）：
	```bash
	export LANG=en_US.UTF-8   # 使用英文提示，但支持UTF-8中文显示
	export LC_CTYPE=zh_CN.UTF-8 # (可选) 让正则支持中文
	```

	#### 场景 B：想要 `ls -l` 的时间显示为中文，但报错必须是英文（方便查 Stack Overflow）
	```bash
	export LC_TIME=zh_CN.UTF-8
	export LC_MESSAGES=en_US.UTF-8
	# 不设置 LC_ALL，因为它的存在会抹掉上面两个差异化设置
	```

	#### 场景 C：写脚本时保证绝对可复现（规避国际化坑）
	在脚本开头强制设定标准 C 环境，防止 `awk` 小数点识别错误，或 `sort` 排序因语言变化而出错：
	```bash
	#!/bin/bash
	export LC_ALL=C
	# 此时 date 显示英文，sort 按 ASCII 码排序，数字用 "." 做小数点，速度最快。
	```
	*注意：`LC_ALL=C` 等同于关闭所有国际化，这是 DevOps 运维脚本的黄金法则。*

	---

	### 5. 如何查看与生成 Locale？

	- **查看当前生效的设置**：
	  ```bash
	  locale
	  ```

	- **查看某个变量具体影响**：
	  ```bash
	  locale -k LC_TIME  # 查看时间格式的具体定义
	  ```

	- **如果修改无效（报错）**，说明系统未生成该 locale：
	  ```bash
	  # Debian/Ubuntu
	  sudo locale-gen zh_CN.UTF-8
	  # CentOS/RHEL
	  sudo localedef -i zh_CN -f UTF-8 zh_CN.UTF-8
	  ```

	---

	### 6. 记住 3 条黄金法则（面试/实战必背）

	1. **`LC_ALL` 是锤子**，用来暴力解决乱码，但**不要**在 `.bashrc` 中永久设置它，因为它会阻止你精细调整其他选项。
	2. **`LANG` 是默认值**，通常在 `/etc/locale.conf` 或 `~/.bashrc` 中设置它（如 `export LANG=en_US.UTF-8`）。
	3. **`LC_CTYPE` 是命脉**，它决定是否支持中文输入和正则。如果 `grep` 无法匹配中文，优先检查这个值。

	---

	### 总结图（优先级流向）

	```
	你不设置任何变量？ → 系统使用 "C" (默认)
	你设置了 LANG?    → 使用 LANG 的值
	你设置了 LC_TIME? → 时间用 LC_TIME，其他用 LANG
	你设置了 LC_ALL?  → 所有设置全部失效，强制使用 LC_ALL 的值（最终BOSS）
	```

* bash 设置忽略`\r`字符

    `set -o igncr`

    在运行脚本时临时显式指定：

    `bash -o igncr your_script.sh`

* 关于 bash 中 ssh 与 nc 执行顺序与同步的问题

    ```bash
    ssh -NL 1111:127.0.0.1:1111 user@some_host &
    nc 127.0.0.1 1111
    ```

    但是 ssh 还没建立好连接，nc 就开始执行了，导致 nc 经常失败。如何解决这个问题？

    你可以用以下几种方法解决这个问题：

    方法1：等待端口就绪（推荐）

    ```bash
    ssh -NL 1111:127.0.0.1:1111 user@some_host &
    # 等待端口可用
    while ! nc -z 127.0.0.1 1111 >/dev/null 2>&1; do
        sleep 0.1
    done
    nc 127.0.0.1 1111
    ```

    方法2：使用 sleep（简单但不精确）

    ```bash
    ssh -NL 1111:127.0.0.1:1111 user@some_host &
    sleep 2  # 根据网络情况调整等待时间
    nc 127.0.0.1 1111
    ```

* 是否有办法准确地知道后台ssh的执行状态？或者ssh连接建立成功后，能主动发一个信号通知，而前台的代码等到这个通知后再继续往下执行？或者是否有办法区别这两种情况：“我希望把某个程序A执行完，或者执行到某个阶段后再放到后台，而不是放到后台后才开始执行”？

    有几种方法可以准确知道 SSH 的状态和实现你需要的控制：

    1. 使用 SSH 的控制套接字 (ControlMaster)

        这是最推荐的方法，可以精确控制 SSH 连接状态：

        ```bash
        # 创建控制套接字，但不建立隧道（只是建立连接）
        ssh -M -S ~/.ssh/ctl_socket -fnNT user@some_host

        # 检查连接是否成功建立
        if ssh -S ~/.ssh/ctl_socket -O check user@some_host 2>/dev/null; then
            # 现在通过现有连接建立端口转发
            ssh -S ~/.ssh/ctl_socket -O forward -L 1111:127.0.0.1:1111 user@some_host
            nc 127.0.0.1 1111
            # 完成后清理
            ssh -S ~/.ssh/ctl_socket -O cancel -L 1111:127.0.0.1:1111 user@some_host
            ssh -S ~/.ssh/ctl_socket -O exit user@some_host
        fi
        ```

        更简洁的版本：
        ```bash
        # 建立连接（前台阻塞直到连接成功）
        if ssh -M -S /tmp/ssh_tunnel -fN user@some_host; then
            # 连接成功后，再建立隧道（这会在后台进行）
            ssh -S /tmp/ssh_tunnel -O forward -L 1111:127.0.0.1:1111 user@some_host
            nc 127.0.0.1 1111
        fi
        ```

    2. 使用命名管道 (Named Pipe) 进行进程间通信

        ```bash
        # 创建命名管道
        pipe=/tmp/ssh_ready.$$
        mkfifo $pipe

        # 启动 SSH，连接成功后写入管道
        (
            if ssh -NL 1111:127.0.0.1:1111 user@some_host; then
                echo "SSH connected successfully" > $pipe
            else
                echo "SSH failed" > $pipe
            fi
        ) &

        # 等待 SSH 状态
        read result < $pipe
        rm $pipe

        if [ "$result" = "SSH connected successfully" ]; then
            nc 127.0.0.1 1111
        fi
        ```

    3. 使用文件锁或状态文件

        ```bash
        status_file=$(mktemp)

        # SSH 进程在连接成功后创建状态文件
        (
            if ssh -NL 1111:127.0.0.1:1111 user@some_host; then
                touch "$status_file.connected"
            fi
        ) &
        ssh_pid=$!

        # 等待状态文件出现（最多10秒）
        for i in {1..100}; do
            if [ -f "$status_file.connected" ]; then
                break
            fi
            sleep 0.1
        done

        if [ -f "$status_file.connected" ]; then
            nc 127.0.0.1 1111
            # 清理
            kill $ssh_pid 2>/dev/null
        fi

        rm -f "$status_file" "$status_file.connected"
        ```

    4. 使用 coproc（协程） - Bash 4.0+

        ```bash
        # 使用协程，可以更容易控制
        coproc SSH_TUNNEL {
            ssh -NL 1111:127.0.0.1:1111 user@some_host
        }

        # 等待一段时间让 SSH 启动
        sleep 1

        # 检查协程是否还在运行
        if kill -0 $SSH_TUNNEL_PID 2>/dev/null; then
            # SSH 还在运行，尝试连接
            nc 127.0.0.1 1111
            
            # 完成后清理
            kill $SSH_TUNNEL_PID 2>/dev/null
        fi
        ```

    5. 使用 timeout 和检查 SSH 的 stdout/stderr

        ```bash
        # 创建一个管道来捕获 SSH 的输出
        (
            # 设置 SSH 为详细模式，可以看到连接过程
            if ssh -v -NL 1111:127.0.0.1:1111 user@some_host 2>&1 | 
            grep -q "Local forwarding listening"; then
                echo "READY" > /tmp/ssh_status
            fi
        ) &
        ssh_pid=$!

        # 等待状态指示
        timeout=10
        for ((i=0; i<timeout*10; i++)); do
            if [ -f /tmp/ssh_status ] && grep -q "READY" /tmp/ssh_status; then
                break
            fi
            sleep 0.1
        done

        nc 127.0.0.1 1111

        # 清理
        kill $ssh_pid 2>/dev/null
        rm -f /tmp/ssh_status
        ```

    6. 使用更专业的工具：autossh 或 sshuttle

        ```bash
        # 使用 autossh（自动重连的 SSH）
        autossh -M 0 -NL 1111:127.0.0.1:1111 user@some_host &
        # autossh 会确保连接稳定，但依然需要等待端口就绪

        # 或者使用 sshuttle（更高级的 SSH 隧道）
        # sshuttle -r user@some_host 0/0 --python python2
        ```

    **推荐方案**

    对于你的需求，**ControlMaster 方案是最佳选择**，因为它：

    1. **精确控制**：可以分别建立连接和隧道
    2. **状态可查**：可以检查 SSH 连接状态
    3. **复用连接**：多个隧道可以共享一个 SSH 连接
    4. **干净退出**：可以优雅地关闭连接

    **最简单实用的方案**还是结合端口检测：
    
    ```bash
    ssh -NL 1111:127.0.0.1:1111 user@some_host &
    until nc -z 127.0.0.1 1111 2>/dev/null; do sleep 0.1; done
    nc 127.0.0.1 1111
    ```

    这样可以确保在端口真正可用后才执行 nc 命令，避免竞态条件。

* 是否有办法区别这两种情况：“我希望把某个程序A执行完，或者执行到某个阶段后再放到后台，而不是放到后台后才开始执行”？
那这个问题呢？看你的回答似乎是做不到，只能先把放到后台，然后通过进程间同步的方式来控制流程？

    你说得很对，我之前的回答没有直接回答这个核心问题。**确实有办法区别这两种情况**，而且你的理解非常准确。

    ## 方法1：使用 subshell + 显式延迟后台化

    ```bash
    # 先在前台执行初始化，然后显式放到后台
    (
        echo "程序A开始初始化..."
        # 执行初始化阶段
        sleep 2  # 模拟初始化工作
        echo "初始化完成，即将进入后台"
        
        # 现在执行主要的后台工作
        while true; do
            echo "后台工作中..."
            sleep 1
        done
    ) &
    # 此时才真正进入后台
    echo "程序A已在后台运行"
    ```

    ## 方法2：使用 bash 的 `coproc`（协程）

    ```bash
    # coproc 会先启动进程，但保持连接，直到你显式让它运行
    coproc MYPROC {
        echo "这是协程的初始化阶段" >&2
        # 初始化代码
        sleep 2
        
        echo "初始化完成，开始主循环" >&2
        # 主循环
        while read line; do
            echo "处理: $line"
        done
    }

    # 此时协程已经开始执行初始化部分
    echo "等待初始化..."
    sleep 3  # 等待初始化完成

    # 现在可以向协程发送数据
    echo "发送数据给协程" >&${MYPROC[1]}
    ```

    ## 方法3：使用命名管道控制执行阶段

    ```bash
    # 创建控制管道
    control_pipe=$(mktemp -u)
    mkfifo $control_pipe

    # 启动程序，但它在等待信号
    (
        echo "程序A：等待开始信号..."
        # 阻塞直到收到信号
        read signal < $control_pipe
        
        if [ "$signal" = "START" ]; then
            echo "程序A：收到开始信号，开始后台工作"
            # 后台工作代码
            for i in {1..5}; do
                echo "后台工作 $i"
                sleep 1
            done
        fi
    ) &
    proc_pid=$!

    echo "主程序：程序A已启动但未开始工作"
    sleep 2

    echo "主程序：现在触发程序A开始工作"
    echo "START" > $control_pipe

    # 等待程序完成
    wait $proc_pid
    rm $control_pipe
    ```

    ## 方法4：使用进程替换 + 文件锁

    ```bash
    # 创建锁文件
    lock_file=$(mktemp)

    # 程序先执行初始化，然后获取锁表示已就绪
    (
        echo "阶段1：初始化（前台执行）"
        sleep 2
        
        echo "阶段1完成，获取锁"
        touch $lock_file  # 表示初始化完成
        
        echo "阶段2：开始后台工作"
        # 真正的后台工作
        sleep 5
        echo "后台工作完成"
    ) &
    pid=$!

    # 主程序等待初始化完成
    echo "等待程序初始化..."
    while [ ! -f $lock_file ]; do
        sleep 0.1
    done

    echo "程序初始化完成，现在它在后台运行"
    rm $lock_file
    ```

    ## 方法5：使用信号控制（最符合你的需求）

    ```bash
    #!/bin/bash

    # 信号处理函数，用于启动后台工作
    start_background_work() {
        echo "信号收到，开始后台工作"
        # 真正的后台工作
        while true; do
            echo "后台工作中..."
            sleep 1
        done
    }

    # 设置信号处理器
    trap start_background_work USR1

    # 前台初始化阶段
    echo "开始初始化..."
    sleep 3
    echo "初始化完成"

    # 发送信号给自己，启动后台工作
    kill -USR1 $$

    # 主进程继续执行（如果不需要可以退出）
    echo "主进程继续..."
    wait
    ```

    ## 方法6：针对 SSH 的具体方案

    对于 SSH 隧道，可以使用 `-f` 和 `-o ExitOnForwardFailure=yes`：

    ```bash
    # SSH 会先在前台建立连接和隧道，成功后自动进入后台
    if ssh -f -o ExitOnForwardFailure=yes -NL 1111:127.0.0.1:1111 user@some_host; then
        echo "SSH隧道已建立并进入后台"
        nc 127.0.0.1 1111
    else
        echo "SSH隧道建立失败"
    fi
    ```

    这里的关键参数：
    - `-f`：要求认证后进入后台
    - `-o ExitOnForwardFailure=yes`：端口转发失败时退出（不会进入后台）

    ## 总结对比

    | 方法 | 是否先执行再后台 | 控制粒度 | 适合场景 |
    |------|------------------|----------|----------|
    | Subshell | ✓ | 中 | 简单的两阶段任务 |
    | Coproc | ✓ | 高 | 需要双向通信的后台任务 |
    | 命名管道 | ✓ | 高 | 需要精确控制启动时机 |
    | 文件锁 | ✓ | 低 | 只需知道初始化完成 |
    | 信号 | ✓ | 高 | 最灵活，完全控制 |
    | SSH -f | ✓ | 低 | 专门用于SSH |

    **对于你的SSH场景，方法6是最佳选择**，因为它利用了SSH自身的特性。

    **对于通用场景，方法5（信号）是最灵活和强大的**，因为它：
    1. 完全控制执行阶段
    2. 可以精确知道何时切换到后台
    3. 支持复杂的初始化逻辑
    4. 代码清晰易读

* 其实 -f 比较贴合我的想法，但是 -f 的原理是在后台 fork 一个进程，然后退出前台进程吧？

    **如何获取后台进程的PID？**
    ```bash
    # 使用 $! 获取最近后台进程的PID
    ssh -f -NL 1111:127.0.0.1:1111 user@some_host &
    ssh_pid=$!
    echo "SSH后台进程PID: $ssh_pid"

    # 但实际上，这个PID是父进程的，真正的后台进程是它的子进程
    # 要获取真正的后台SSH进程：
    real_ssh_pid=$(ps -ef | grep "ssh.*1111:127.0.0.1:1111" | grep -v grep | awk '{print $2}')
    ```

    **`-f` 的替代方案：手动实现类似逻辑**

    如果你想自己控制这个流程，可以这样：

    ```bash
    #!/bin/bash

    # 阶段1：前台建立连接
    echo "阶段1：建立SSH连接和隧道"
    ssh_pid_file=$(mktemp)

    # 启动SSH，但不立即后台
    ssh -NL 1111:127.0.0.1:1111 user@some_host &
    ssh_pid=$!

    # 等待连接建立（检查端口）
    timeout=10
    for i in $(seq 1 $timeout); do
        if netstat -tln 2>/dev/null | grep -q ":1111 "; then
            echo "隧道已建立"
            break
        fi
        sleep 1
       
        # 检查SSH是否还活着
        if ! kill -0 $ssh_pid 2>/dev/null; then
            echo "SSH连接失败"
            exit 1
        fi
    done

    # 阶段2：SSH现在已经在后台运行了
    echo "阶段2：SSH已在后台运行，PID: $ssh_pid"
    echo $ssh_pid > $ssh_pid_file

    # 现在可以安全地使用隧道
    nc 127.0.0.1 1111

    # 完成后清理
    kill $ssh_pid
    rm $ssh_pid_file
    ```

    `-f` 的局限性

    `-f` 的一个问题是：它进入后台后就**失去控制**了。你不能：
    1. 轻松地获取它的PID（需要额外处理）
    2. 优雅地关闭它（需要找到并kill）
    3. 知道它何时异常退出

    改进方案：结合 `-f` 和进程管理

    ```bash
    #!/bin/bash

    # 使用命名管道来同步
    pipe=$(mktemp -u)
    mkfifo $pipe

    # 在一个子shell中启动SSH
    (
        # 执行SSH，成功后写入管道
        if ssh -f -o ExitOnForwardFailure=yes \
                -NL 1111:127.0.0.1:1111 user@some_host; then
            echo "READY:$$" > $pipe
        else
            echo "FAILED" > $pipe
        fi
    ) &

    # 读取管道获取状态
    read status < $pipe

    if [[ $status == READY:* ]]; then
        ssh_pid=${status#READY:}
        echo "SSH已就绪，进程PID: $ssh_pid"
        
        # 使用隧道
        nc 127.0.0.1 1111
        
        # 清理
        kill $ssh_pid
    else
        echo "SSH启动失败"
    fi

    rm $pipe
    ```

* bash 中，我想知道这个 bash 进程接收到了操作系统发过来的哪些 signal，怎么办

    在 Bash 中，想要知道当前进程（或指定 Bash 进程）接收、捕获或忽略了哪些信号（Signals），可以通过以下几种方法来实现。最直接、最底层的方法是读取 Linux 的 `/proc` 伪文件系统。

    ---

    ## 方法一：通过 `/proc` 文件系统查看（最精准）

    Linux 会将每个进程的信号处理状态记录在 `/proc/[PID]/status` 文件中。你可以直接查看当前 Bash 进程（PID 为 `$$`）的状态。

    运行以下命令：

    ```bash
    grep -E "Sig(Pnd|Blk|Ign|Cgt)" /proc/$$/status

    ```

    ### 1. 输出结果解析

    你会看到类似下面的一串十六进制掩码：

    * **`SigPnd` (Pending):** 进程排队等待接收的信号。
    * **`SigBlk` (Blocked):** 进程目前阻塞的信号（暂时不处理）。
    * **`SigIgn` (Ignored):** 进程**显式忽略**的信号。
    * **`SigCgt` (Caught):** 进程**捕获并自定义了处理函数**的信号（这也是你最需要关注的）。

    ### 2. 如何将十六进制转换为可读的信号名称？

    手算十六进制位掩码太痛苦了，我们可以用 Linux 自带的 `ps` 工具直接转换，或者用一个简单的命令行组合。

    **直接用 `ps` 查看当前 Shell 捕获的信号：**

    ```bash
    ps -o pid,comm,caught -p $$

    ```

    *输出示例：*

    ```text
      PID COMMAND          CAUGHT
    12345 bash     0000000180014003

    ```

    **终极解析脚本：**
    如果你想抓出具体是哪些信号（如 `SIGINT`, `SIGTERM`），可以直接运行下面这段单行命令，它会把 `SigCgt` 翻译成具体的信号名称：

    ```bash
    cat /proc/$$/status | grep SigCgt | awk '{print $2}' | xargs -I {} -s 256 sample=0x{} perl -e '
    $mask = oct($ENV{sample});
    for ($i=1; $i<=64; $i++) {
        print "Signal $i (" . `kill -l $i` . ") is caught\n" if ($mask & (1 << ($i-1)));
    }' | sed 's/\n//g'

    ```

    *(注：`kill -l` 会把数字转为标准的信号名，比如 2 代表 `INT`，15 代表 `TERM`。)*

    ---

    ## 方法二：使用 Bash 内置的 `trap` 命令（最常用）

    如果你只想知道 **Bash 脚本或当前交互式终端自己用 `trap` 显式捕获了哪些信号**，直接输入 `trap` 即可：

    ```bash
    trap

    ```

    * **如果没有任何输出：** 说明当前 Bash 没用自定义的 `trap` 钩子，它正在使用操作系统默认的信号处理机制。
    * **如果有输出：** 它会列出所有被捕获的信号和对应的触发动作。例如：
    ```text
    trap -- 'echo "Caught SIGINT!"' SIGINT
    trap -- '' SIGTERM  # 这代表 SIGTERM 被忽略了

    ```



    ---

    ## 方法三：使用 `ssignal` 或 `psig` 工具（如果系统支持）

    在某些 Linux 发行版或通过安装 `debian-goodies` / `procps` 增强包后，你可以使用更人性化的工具：

    ```bash
    # 如果系统中安装了 psig (通常在 Solaris 或部分高级工具集里)
    psig $$

    ```

    或者使用 `ps` 的扩展格式查看：

    ```bash
    ps -q $$ -o s वर्गीकृत信号相关字段 (需查阅具体man手册)

    ```

    ---

    ## 💡 总结建议

    * 想快速看当前 Shell 脚本里绑定了什么监控函数：直接输 **`trap`**。
    * 想看操作系统底层到底把哪些信号投递给了这个 Bash（包括动态捕获的）：用 **`grep -E "Sig(Cgt|Ign)" /proc/$$/status`**。
* bash 中，一个变量 my_env 无论是 export 的，还是非 export 的，都可以使用 unset 取消这个变量。如果只想取消这个 export 前缀，那么可以使用`export -n my_env`

* 简述 bash 中 exec 的用法

    `exec` 是 Bash 中的一个内置命令，用于**替换当前 shell 进程**而不创建新进程。主要有以下几种用法：

    1. **进程替换**

        用新程序替换当前 shell 进程：

        ```bash
        exec ls -l          # 执行 ls，之后当前 shell 结束
        exec /bin/bash      # 替换为新的 bash（相当于重新登录）
        ```

    2. **重定向（最常用）**

        永久性地修改当前 shell 的文件描述符：

        ```bash
        exec > output.log   # 之后所有输出都重定向到文件
        exec 2> error.log   # 所有错误输出重定向
        exec 1>&2           # 标准输出合并到标准错误
        exec 3<> file.txt   # 打开文件描述符 3 可读写
        ```

    3. **关闭文件描述符**

        ```bash
        exec 3>&-           # 关闭文件描述符 3
        ```

    4. **在脚本中使用**

        ```bash
        #!/bin/bash
        exec > log.txt 2>&1    # 脚本中所有输出都写入 log.txt
        echo "这条会写入文件"    
        exec 3< /etc/passwd    # 打开文件读取
        ```

    5. **无命令时**

        ```bash
        exec                 # 无效果，保持当前 shell
        ```

    执行区别示例

    ```bash
    # 普通执行 - 创建子进程
    ls -l                # 子进程执行，结束后返回父 shell

    # exec 执行 - 替换当前进程
    exec ls -l           # 替换当前 shell，执行完后直接退出
    ```

    典型应用场景

    - **日志重定向**：脚本中统一管理输出

    - **文件描述符操作**：高级 I/O 重定向

    - **包装脚本**：`exec python app.py` 替换脚本进程为应用进程

    - **清理资源**：关闭不需要的文件描述符

* 在 bash 中，分号 ; 的主要作用是分隔多个命令，类似于换行。它告诉 shell 在分号前一个命令结束后，立即执行分号后的下一个命令。

    比如`export LD_LIBRARY_PATH=/path/to/libs; bash run_main.sh`等价于：

    ```bash
    export LD_LIBRARY_PATH=/path/to/libs
    bash run_main.sh
    ```

    需要注意分号和 && 的区别：

    * 分号 ;：无论前一个命令是否成功，都会执行下一个命令

        ```bash
        false; echo "这个会执行"  # echo 命令一定会执行
        ```

    * 逻辑与 &&：只有前一个命令成功（返回 0），才执行下一个命令

        ```bash
        false && echo "这个不会执行"  # echo 不会执行
        true && echo "这个会执行"     # echo 会执行
        ```

* chatgpt 写的 bash 定时器

    `timer.sh`:

    ```bash
    #!/bin/bash

    if [ $# -ne 2 ]; then
      echo "用法: $0 <N分钟> <audio_file>"
      exit 1
    fi

    TOTAL_MIN=$1
    AUDIO_FILE=$2
    TOTAL_SEC=$((TOTAL_MIN * 60))

    # 设置终端：关闭回显和规范模式
    stty -echo -icanon time 0 min 0

    paused=0
    elapsed=0

    cleanup() {
      stty sane
      tput cnorm
    }
    trap cleanup EXIT

    tput civis  # 隐藏光标

    while [ $elapsed -lt $TOTAL_SEC ]; do
      # 捕获键盘输入
      key=$(dd bs=1 count=1 2>/dev/null)
      if [ "$key" = " " ]; then
        paused=$((1 - paused))  # 切换暂停/恢复
      fi

      if [ $paused -eq 0 ]; then
        elapsed=$((elapsed + 1))
      fi

      remain=$((TOTAL_SEC - elapsed))
      min=$((remain / 60))
      sec=$((remain % 60))

      # 清理并重写
      tput cup 0 0
      tput ed
      printf "总时间: %2d 分钟\n" "$TOTAL_MIN"
      printf "剩余:   %02d:%02d\n" "$min" "$sec"
      if [ $paused -eq 1 ]; then
        printf "[已暂停]\n"
      else
        printf "         \n"
      fi

      sleep 1
    done

    cleanup
    mpv --really-quiet "$AUDIO_FILE"
    ```

    用法：
    
    `bash timer.sh <N> <audio_file>`
    
    定时`N`分钟后播放音频`<audio_file>`，期间 terminal 上会显示倒计时，按空格可以暂时计时，再次按空格恢复。

* bash 中与目录相关的命令

    1. 基础路径切换

        | 命令 | 功能说明 |
        | - | - |
        | `cd -` | 切换回上一个工作目录（在两个目录间快速切换） |
        | `cd ~` 或 `cd` | 回到家目录 |
        | `cd ..` | 切换到上级目录 |
        | `cd ../..` | 向上移动两级目录 |
        | `pushd <目录>` | 将目录压入栈并切换 |
        | `popd` | 从栈中弹出并切换到该目录 |

    2. 特殊路径表示

        | 符号 | 含义 |
        | - | - |
        | `.` | 当前目录 |
        | `..` | 上级目录 |
        | `~` | 当前用户的家目录 |
        | `~username` | 指定用户的家目录 |
        | `/` | 根目录 |
        | `-` | 上一个工作目录 |

    example:

    ```bash
    # 切换到 /var/log 并记住当前位置
    pushd /var/log

    # 再切换到 /etc 并记住
    pushd /etc

    # 查看目录栈
    dirs

    # 返回上一个目录（/var/log）
    popd

    # 再返回最初的目录
    popd
    ```

    相关的环境变量:

    | 变量 | 用途 |
    | - | - |
    | `$HOME` | 用户家目录路径 |
    | `$PWD` | 当前工作目录路径 |
    | `$OLDPWD` | 上一个工作目录路径（`cd -` 就是用它） |

* bash 的快捷键

    这些快捷键主要基于 GNU Readline 库，因此在其他使用 Readline 的软件（如 mysql、python 等交互式环境）中同样适用。

    * 移动光标（Cursor Movement）

        * `Ctrl + A`: 移动到行首 (Ahead)

        * `Ctrl + E`: 移动到行尾 (End)

        * `Alt + B`: 向后（左）移动一个单词 (Back)

        * `Alt + F`: 向前（右）移动一个单词 (Forward)

        * `Ctrl + B`: 向后（左）移动一个字符 (等于左箭头)

        * `Ctrl + F`: 向前（右）移动一个字符 (等于右箭头)

    * 编辑文本（Editing）

        * `Ctrl + W`: 剪切光标前的一个单词（到空格处）

        * `Ctrl + U`: 剪切从光标到行首的所有内容

        * `Ctrl + K`: 剪切从光标到行尾的所有内容

        * `Ctrl + Y`: 粘贴刚才剪切的内容 (Yank)

        * `Ctrl + D`: 删除光标后的一个字符（如果行空，则退出 shell）

        * `Ctrl + H`: 删除光标前的一个字符（等同于 Backspace）

        * `Alt + D`: 删除光标后的一个单词

        * `Ctrl + _`或`Ctrl + X Ctrl + U`: 撤销上一步操作

        * `Ctrl + T`: 交换光标前的两个字符

        * `Alt + T`: 交换光标前的两个单词

    * 历史命令（Command History）

        * Ctrl + P: 显示上一条命令 (等于上箭头)

        * Ctrl + N: 显示下一条命令 (等于下箭头)

        * Ctrl + R: 反向搜索历史命令（强烈推荐！输入关键词即可搜索）

        * Ctrl + G: 从反向搜索模式中退出

        * Ctrl + O: 执行当前在历史中找到的命令，并自动显示下一条（用于重复执行一系列命令）

        * Ctrl + J: 结束当前在历史中搜索到的命令，相当于回车

        小技巧：在 Ctrl + R 搜索模式下，可以：

        * 再次按 Ctrl + R 向前循环搜索匹配项。

        * 按 Esc 或 Ctrl + J 将当前匹配项放到命令行上并编辑它。

        * 按 Ctrl + G 退出搜索而不执行任何操作。

    * 进程控制（Process Control）

        * Ctrl + C: 中断（终止）当前正在运行的程序

        * Ctrl + Z: 暂停（挂起）当前正在运行的程序，送到后台

        * Ctrl + L: 清屏（等同于 clear 命令）

        * Ctrl + S: 暂停屏幕输出（冻结终端，有些终端不适用）

        * Ctrl + Q: 恢复屏幕输出（解冻终端）
        
        注意：Ctrl+Z 后，程序会被挂起。你可以：

        * 用 fg 命令将其恢复到前台继续运行。

        * 用 bg 命令将其在后台继续运行。

        * 用 jobs 命令查看所有被暂停的作业。

    * 特殊功能（Special）

        * Ctrl + D: (EOF) 结束输入。如果命令行是空的，会退出当前 shell。

        * Tab: 自动补全文件、目录、命令名。按两次会列出所有可能选项。

        * `Alt + .`或`Esc + .`: 使用上一条命令的最后一个参数。
        
            例如：`mkdir my_long_folder_name`，然后`cd`再按`Alt + .`，会自动变成`cd my_long_folder_name`。
        
        * `Ctrl + X Ctrl + E`: 用默认编辑器（如 vim）编辑当前命令，退出编辑器后命令会自动执行。非常适合编辑长而复杂的命令。


* bash-completion

    `/etc/bash_completion.d/hlc_main`:

    ```bash
    # 补全脚本：main
    _main_completion() {
        local cur prev words cword
        _init_completion || return

        # 定义所有可用的子命令
        local subcommands="init build test deploy clean help version"

        # 如果当前是第一个参数，补全子命令
        if [[ $cword -eq 1 ]]; then
            COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
        fi
    }

    # 注册补全函数
    complete -F _main_completion hlc_main
    ```

    然后执行`source ~/.bashrc`即可生效。

    其中，local 变量的作用：

    | 变量 | 类型 | 描述 |
    | - | - | - |
    | cur | 字符串 | 当前正在输入的单词（光标前的部分） |
    | prev | 字符串 | 前一个单词 |
    | words | 数组 | 命令行中所有的单词 |
    | cword | 整数 | 当前单词在 words 数组中的索引 |


* bash 中输入密码显示为星号（`*`）的模板代码

    ```bash
    stty -echo  # 关闭回显
    unset password
    prompt="Enter password: "
    while IFS= read -p "$prompt" -r -s -n1 char; do
        if [[ $char == $'\0' ]]; then  # 回车键结束
            break
        fi
        prompt='*'
        password+="$char"
    done
    stty echo  # 恢复回显
    echo
    echo "Password: [hidden]"
    ```

    效果：

    ```
    Enter password: **********
    Password: [hidden]
    ```

* IFS 与 read

    `IFS`指的是 Internal Field Separator，可以决定`read`是否过滤用户输入的前缀、后缀空格。

    ```bash
    read -p "input something: " msg
    echo "$msg"

    IFS= read -p "input something: " msg
    echo "$msg"
    ```

    交互输入与输出：

    ```
    input something:     hello world
    hello world
    input something:     hello world
        hello world
    ```

    其中，`IFS= read -p "input something: " msg`等价于`IFS="" read -p "input something: " msg`，或`IFS='' read -p "input something: " msg`，即为空。

    `IFS`的默认值为`IFS=" \t\n"`（空格，制表，换行）。

    `echo "$msg"`是为了将`msg`内容解释为一个字符串。假如`msg`值为`    hello`，那么`echo $msg`就变成`echo     hello`，输出为`hello`。而`echo "$msg"`则为`echo "    hello"`，输出为`    hello`。

    `IFS= read`仅作用于当前命令`read`，当`read`执行完后，`IFS`又恢复默认值。

* `read -t`用于设置读取输入的超时时间

    `read -t <秒数> [变量名]`

    parameters:

    * `-t`：指定超时时间（单位：秒，支持小数如 0.5）。

    * 变量名（可选）：存储用户输入的内容，默认存入 REPLY 变量。

    若用户在 -t 指定的时间内未输入，read 会立即结束，并返回 非0 状态码（表示超时）。


## topics

### 字符串与逃逸

* bash escape single quote

    下面是 bash 中使用单引号组成的字符串的 example 和解释：

    ```bash
    echo 'hello there'
    echo 'hello
       there'

    echo 'hello 'there'
      aaa'
    echo 'hello \'there'
      bbb'

    echo $'hello 'there'
       ccc'
    echo $'hello \'there'
    ```

    output:

    ```
    hello there
    hello
      there
    hello there
      aaa
    hello \there
      bbb
    hello there
      ccc
    hello 'there
    ```

    说明：

    1. 使用单引号`'`括起的 bash 字符串，里面的大部分都按照原始字符解释

        比如前两个 echo，

        ```bash
        echo 'hello there'
        echo 'hello
           there'
        ```

        第二个 echo 有一个换行，echo 输出的字符串也照原样换行了。

    2. 如果想在单引号创建的字符串里加入额外的单引号，这样写是不行的：

        `echo 'hello 'there'`

        这样会被 bash 识别为三个部分：

        1. 第一个字符串`'hello '`

        2. 第二个字符串`there`

        3. 第三个不完整字符串`'`

            这个字符串只写了左单引号，没有写字符串内容和右单引号，因此 bash 会继续往后找另外一个单引号，作为字符串的结尾。

        为了验证这个猜想，上面第三个 echo 换行后把右单引号补全：

        ```bash
        echo 'hello 'there'
          aaa'
        ```

        而对应的输出为：

        ```
        hello there
          aaa
        ```

        与我们的预期相符。

    3. 第四个 echo 的字符串，想使用`\'`对单引号进行转义，但是却输出了`\`，该如何解释？

        ```bash
        echo 'hello \'there'
          bbb'
        ```

        由于单引号将字符串看作 raw string，所以 bash 将其解释为 3 个字符串：

        1. 第一个字符串`'hello \'`

        2. 第二个字符串`there`

        3. 第三个字符串，换行 + 两个空格 + bbb：

            ```
            '
              bbb'
            ```

        程序的输出也符合预期。

    4. 如果想在单引号括起的字符串中加入单引号，可以在字符串前加一个美元符号`$`，再在字符串中对单引号进行转义

        第 5 个 echo:

        ```bash
        echo $'hello 'there'
           ccc'
        ```

        虽然使用了`$`，但是并未对字符串中的单引号进行转义，因此 bash 仍认为它是三个字符串：

        1. `'hello '`

        2. `there`

        3. 换行 + 3 空格 + `ccc`

        第 6 个 echo:

        ```bash
        echo $'hello \'there'
        ```

        满足了在字符串前加`$`，并且在字符串中对单引号进行了`\'`转义，因此输出与预期一致。


* bash 中的字符串引号逃逸

    `echo It's my work.`会显示语法错误：

    ```
    main.sh: line 4: unexpected EOF while looking for matching `''
    main.sh: line 5: syntax error: unexpected end of file
    ```

    因为 bash 认为`'`是标记了一个字符串的开始，但是没有找到另一个`'`标记字符串的结束。

    如果我们写成

    `echo 'It's my work.'`

    输出同样同上，因为 bash 匹配最近的两个`'`，即`'It's`中的这两个。导致 work 后的`'`未被匹配。

    即使加上转义也不行：`echo 'It\'s my work.'`，因为 bash 认为单引号括住的字符默认不转义，即字符串被分割为`'It\'`和`s my work.`，以及`'`开头的未完成字符串。

    双引号也不转义：

    `echo "It\'s my work."`

    输出为：

    ```
    It\'s my work.
    ```

    要想正确输出，只有两种办法：

    * `echo $'It\'s my work.'`

    * `echo "It's my work."`

    这点和 C 语言不太一样。


### 数组 array

* bash array 使用小括号来定义：`arr=(elm_1 elm_2 elm_3)`

* bash array 使用`[]`作为下标，并从 0 开始索引，`${arr[0]}`, `${arr[1]}`

* 当使用`@`或`*`作为索引时，会索引数组中的所有元素：`${arr[@]}`, `${arr[*]}`

* 使用`${#arr[@]}`或`${#arr[*]}`可以获得数组的长度

* bash 中打印数组中的字符串，每个一行

    ```bash
    arr=(hello world nihao zaijian)
    arr_len=${#arr[@]}
    i=0
    while [ $i -lt $arr_len ] ; do
        echo ${arr[i]}
        i=$((i+1))
    done
    ```


### set 命令

* `set -e`: 任何命令返回非零（失败）状态时，立即退出脚本

    可以通过`set +e`关闭这一行为。

    example:

    ```bash
    set -e
    cd haha
    echo hello
    ```

    (`haha`文件夹不存在)

    output:

    ```
    main.sh: line 2: cd: haha: No such file or directory
    ```

    退出 bash 脚本后，`echo $?`的值为`1`。

    ```bash
    # set -e
    cd haha
    echo hello
    ```

    output:

    ```
    main.sh: line 2: cd: haha: No such file or directory
    hello
    ```

    `set -e`等价于`set -o errexit`。

* `set -u`: 遇到未定义的变量时，报错并退出（防止误用空变量）

    example:

    ```bash
    set -u
    echo "$my_var"
    echo "hello"
    ```

    output:

    ```
    main.sh: line 2: my_var: unbound variable
    ```

    ```bash
    # set -u
    echo "$my_var"
    echo "hello"
    ```

    output:

    ```

    hello
    ```

* `set -o pipefail`

    管道命令`|`中任意一个子命令失败时，整个管道返回非零状态。
    
    example:

    `main.sh`:

    ```bash
    set -o pipefail
    cd haha | echo "hello"
    echo $?
    ```

    run: `bash main.sh`

    output:

    ```
    main.sh: line 2: cd: haha: No such file or directory
    hello
    1
    ```

    如果不设置`pipefail`，则只返回最后一个命令的状态：

    ```bash
    # set -o pipefail
    cd haha | echo "hello"
    echo $?
    ```

    run: `bash main.sh`

    output:

    ```
    hello
    main.sh: line 2: cd: haha: No such file or directory
    0
    ```

    注意，`set -o pipefail`只改变了管道命令的返回值，并不会使 bash 脚本退出。

    `set +o pipefail`可以关闭这个参数。


### signal 相关

* trap

    用于在脚本执行过程中捕获和处理信号或事件。它允许你在脚本接收到特定信号时执行指定的命令或函数，常用于清理临时文件、优雅退出或调试。

    syntax:

    ```bash
    trap [COMMAND] [SIGNALS]
    ```

    * `COMMAND`：捕获到信号后要执行的命令或函数（用引号包裹）。

    * `SIGNALS`：要捕获的信号名称或编号（如`INT`、`TERM`、`EXIT`等）。

    examples:

    * 捕获中断信号（如 Ctrl+C）

        ```bash
        trap "echo '脚本被中断！'; exit 1" INT
        ```

        当用户按下 Ctrl+C（发送 SIGINT 信号）时，脚本会打印消息并退出。

    * 脚本退出时清理资源

        ```bash
        trap "rm -f /tmp/tempfile; echo '清理完成'" EXIT
        ```

        无论脚本正常结束还是因错误退出，都会执行清理操作（删除临时文件）。

    * 忽略信号

        ```bash
        trap "" TERM
        ```

        忽略 SIGTERM 信号（常用于防止脚本被意外终止）。

    * 捕获调试信号

        ```bash
        trap "echo '调试模式：变量 x=$x'" DEBUG
        ```

        每次命令执行后打印变量 x 的值（用于调试）。

    * 重置信号处理

        ```bash
        trap - INT
        ```

        恢复对 SIGINT 的默认行为（移除之前的 trap 设置）。

    常用信号列表:

    | 信号名称 | 编号 | 触发条件 |
    | - | - | - |
    | INT | 2 | Ctrl+C 中断 |
    | TERM | 15 | 默认的 kill 命令 |
    | EXIT | 0 | 脚本退出时（非真实信号） |
    | ERR | - | 命令执行失败时（非真实信号） |
    | DEBUG | - | 每条命令执行后（非真实信号） |

    example:

    ```bash
    #!/bin/bash

    cleanup() {
        echo "正在清理临时文件..."
        rm -f /tmp/temp_*
    }

    trap cleanup EXIT    # 脚本退出时调用 cleanup
    trap "echo '忽略中断信号'" INT  # 捕获 Ctrl+C

    echo "创建临时文件..."
    touch /tmp/temp_1234

    echo "按 Ctrl+C 测试中断信号，或等待脚本完成..."
    sleep 5
    ```

    注意事项:

    * trap 的作用范围是当前的 Shell 环境。

    * 在函数中定义的 trap 会覆盖全局设置（除非显式声明为全局）。


* bash 中常见信号与触发条件

    | 信号名称 | 编号 |	触发条件 |
    | - | - | - |
    | `INT` | 2 | Ctrl + C 中断 |
    | `TERM` | 15 | 默认的 kill 命令 |
    | `EXIT` | 0 | 脚本退出时（非真实信号） |
    | `ERR` | - | 命令执行失败时（非真实信号） |
    | `DEBUG` | - | 每条命令执行后（非真实信号） |

    example:

    `main.sh`：

    ```bash
    trap "echo INT triggered!; exit 1" INT
    trap "echo TERM triggered; exit 1" TERM
    trap "echo EXIT triggered; exit 1" EXIT
    trap "echo ERR triggered; exit 1" ERR
    trap "echo DEBUG triggered" DEBUG

    while true; do
        echo "current time: $(date)"
        sleep 1
    done
    ```

    run: `bash main.sh`

    使用不同的方式触发 signal 后，output 如下：

    * `Ctrl` + `C`

        ```
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:41:38 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:41:39 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:41:40 CST
        DEBUG triggered
        ^CDEBUG triggered
        INT triggered!
        DEBUG triggered
        DEBUG triggered
        EXIT triggered
        DEBUG triggered
        ```

    * kill

        ```
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:44:15 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:44:16 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:44:17 CST
        DEBUG triggered
        DEBUG triggered
        TERM triggered
        DEBUG triggered
        DEBUG triggered
        EXIT triggered
        DEBUG triggered
        ```

    所有的信号对大小写不敏感，即`INT`和`int`是等价的，其他的同理。

* bash 中的 trap 可以让用户指定函数或命令去处理 signal 信号

    example:

    `main.sh`:

    ```bash
    trap "echo manully terminated!; exit 1" INT

    while true; do
        echo "current time: $(date)"
        sleep 1
    done
    ```

    执行：`bash ./main.sh`，等待几秒后，按`Ctrl` + `C`。

    output:

    ```
    current time: 2025年 06月 30日 星期一 13:03:09 CST
    current time: 2025年 06月 30日 星期一 13:03:10 CST
    current time: 2025年 06月 30日 星期一 13:03:11 CST
    current time: 2025年 06月 30日 星期一 13:03:12 CST
    ^Cmanully terminated!
    ```

### test 命令

* `test`

    `test`命令没有输出，但是有一个 exit status，`0`代表`true`，`1`代表`false`。

    Example:

    ```bash
    num=4
    if (test $num -gt 5)
    then
        echo "yes"
    else
        echo "no"
    fi 
    ```

    `test`命令也可被方括号`[]`替代：

    ```bash
    file="/etc/passwd"
    if [ -e $file ]
    then
        echo "whew"
    else
        echo "uh-oh"
    fi
    ```

    Syntax:

    1. File tests:

        ```bash
        test [-a] [-b] [-c] [-d] [-e] [-f] [-g] [-h] [-L] [-k] [-p] [-r] [-s] [-S] [-u] [-w] [-x] [-O] [-G] [-N] [file]

        test -t fd

        test file1 {-nt | -ot | -ef} file2
        ```

    1. String tests:

        ```bash
        test [-n | -z] string

        test string1 {= | != | < | >} string2
        ```

    1. Shell options and variables:

        ```bash
        test -o option

        test {-v | -R} var
        ```

    1. Simple logic (test if values are `null`):

        ```bash
        test [!] expr

        test expr1 {-a | -o} expr2
        ```

    1. Numerical comparison (for integer values only; bash doesn't do floating point math):

        ```bash
        test arg1 {-eq | -ne | -lt | -le | -gt | -ge} arg2
        ```

    Options:

    1. `-a file`, `-e file`

        Returns true if *file* exists. Does the same thing as `-e`. Both are included for compatibility reasons with legacy versions of Unix.

    1. `-b file`

        如果 *file* 是块 (block) 文件则返回 true。

    1. `-c file`

        如果 *file* 是字符 (character) 文件则返回 true 
        
        For example, `/dev/urandom` is a character-special file.

    1. `-d file`

        Returns true if *file* is a directory.

    1. `-f file`

        Returns true if *file* exists, and is a regular file.

    1. `-g file`

        Returns true if *file* has the setgid bit set.

    1. `-h file`

    	Returns true if *file* is a symbolic link. Does the same thing as `-L`. Both are included for compatibility reasons with legacy versions of Unix.

    1. `-L file`

        Returns true if *file* is a symbolic link. Does the same thing as `-h`. Both are included for compatibility reasons with legacy versions of Unix.

    1. `-k file`

        Returns true if *file* has its sticky bit set.

    1. `-p file`

        Returns true if the file is a named pipe, e.g., as created with the command `mkfifo`.

    1. `-r file`

        Returns true if *file* is readable by the user running `test`.

    1. `-s file`

        Returns true if *file* exists, and is not empty.

    1. `-S file`

        Returns true if *file* is a socket.

    1. `-t fd`

        Returns true if file descriptor *fd* is opened on a terminal.

    1. `-u file`

        Returns true if *file* has the setuid bit set.

    1. `-w file`

        Returns true if the user running `test` has write permission to *file*, i.e., make changes to it.

    1. `-x file`

        Returns true if *file* is executable by the user running `test`.

    1. `-O file`

    	Returns true if *file* is owned by the user running `test`.

    1. `-G file`

        Returns true if *file* is owned by the group of the user running `test`.

    1. `-N file`

        Returns true if *file* was modified since the last time it was read.

    1. `file1 -nt file2`

        Returns true if *file1* is newer (has a newer modification date/time) than *file2*.

    1. `file1 -ot file2`

        Returns true if *file1* is older (has an older modification date/time) than *file2*.

    1. `file1 -ef file2`

        Returns true if *file1* is a hard link to *file2*.

    1. `test [-n] string`

        Returns true if *string* is not empty. Operates the same with or without `-n`.

        For example, if `mystr=""`, then `test "$mystr"` and `test -n "$mystr"` would both be false. If `mystr="Not empty"`, then `test "$mystr"` and `test -n "$mystr"` would both be true.

    1. `-z string`

        Returns true if string *string* is empty, i.e., `""`.

    1. `string1 = string2`

        Returns true if *string1* and *string2* are equal, i.e., contain the same characters.

    1. `string1 != string2`

        Returns true if *string1* and *string2* are not equal.

    1. `string1 < string2`

        Returns true if *string1* sorts before *string2* lexicographically, according to ASCII numbering, based on the first character of the string. For instance, `test "Apple" < "Banana"` is true, but `test "Apple" < "banana"` is false, because all lowercase letters have a lower ASCII number than their uppercase counterparts.

        **Tip**: Enclose any variable names in double quotes to protect whitespace. Also, escape the less than symbol with a backslash to prevent bash from interpreting as a redirection operator. For instance, use t`est "$str1" \< "$str2"` instead of `test $str1 < $str2`. The latter command will try to read from a file whose name is the value of variable *str2*. For more information, see redirection in bash.

    1. `string1 > string2`

        Returns true if *string1* sorts after *string2* lexicographically, according to the ASCII numbering. As noted above, use `test "$str1" \> "$str2"` instead of `test $str1 > $str2`. The latter command creates or overwrites a file whose name is the value of variable *str2*.

    1. `-o option`

        Returns true if the shell option *opt* is enabled.

    1. `-v var`

        Returns true if the shell variable *var* is set.

    1. `-R var`

        Returns true if the shell variable *var* is set, and is a name reference. (It's possible this refers to an *indirect reference*, as described in Parameter expansion in bash.)

    1. `! expr`

        Returns true if and only if the expression *expr* is null.

    1. `expr1 -a expr2`

        Returns true if expressions *expr1* and *expr2* are both not null.

    1. `expr1 -o expr2`

        Returns true if either of the expressions *expr1* or *expr2* are not null.

    1. `arg1 -eq arg2`

        True if argument *arg1* equals *arg2*.

    1. `arg1 -ne arg2`

        True if argument *arg1* is not equal to *arg2*.

    1. `arg1 -lt arg2`

        True if numeric value *arg1* is less than *arg2*.

    1. `arg1 -le arg2`

        True if numeric value *arg1* is less than or equal to *arg2*.

    1. `arg1 -gt arg2`

        True if numeric value *arg1* is greater than *arg2*.

    1. `arg1 -ge arg2`

        True if numeric value *arg1* is greater than or equal to *arg2*.

    Notes:

    1. All arguments to test must be separated by a space, including all operators.

    1. The `<` and `>` operators are lexicographical comparisons, based on ASCII numbering. They are not numerical operators (instead, use `-lt`, `-gt`, etc. for comparing numbers).

    1. The precise behavior of `test`, depending on the number of arguments provided, is as follows:

        | #<br>args | test behavior |
        | - | - |
        | 0 | Always return false. |
        | 1 | Return true, if and only if the expression is not null. |
        | 2 | If the first argument is `!`, return true if and only if the expression is null. <br> If the first argument if one of the other unary operators (`-a`, `-b`, etc.), return true if and only if the unary test of the second argument is true. <br> If the first argument is not an unary operator, return false. |
        | 3 | The following conditions are applied in the order listed. <br> If the second argument is one of the binary conditional operators listed above, the result is the binary test using the first and third arguments as operands. Binary conditional operators are those which take two operands, e.g., `-nt`, `-eq`, `<`, etc. <br> The `-a` and `-o` operators are considered binary operators when there are three arguments. <br> If the first argument is `!`, the value is the negation of the two-argument test using the second and third arguments. <br> If the first argument is exactly `(` and the third argument is exactly `)`, the result is the one-argument test of the second argument. In other words, `( expr )` returns the value of `expr`. This special case exists as a way to override the normal precedence of operations. <br> Otherwise, the expression is false. |
        | 4 | If the first argument is `!`, the result is the negation of the three-argument expression composed of the remaining arguments. <br> Otherwise, the expression is parsed and evaluated according to precedence using the rules listed above. |
        | 5+ | The expression is parsed and evaluated according to precedence using the rules listed above. |

    Exit status

    `0` for true, `1` for false. Anything greater than 1 indicates an error or malformed command.

    `$?` can be used to get the exit status of `test`.

* `[ -f file ]`

    检查文件是否存在且为普通文件，等价于`test -f file`

    如果 file 是符号链接，-f 会检查链接指向的目标文件是否为普通文件（跟随链接）

    其他常见文件测试选项
    选项	作用
    -e file	文件/目录是否存在（不检查类型）。
    -d file	是否为目录。
    -L file	是否为符号链接。
    -s file	文件是否存在且大小大于0字节。
    -r file	文件是否可读。
    -w file	文件是否可写。
    -x file	文件是否可执行。

### 重定向

* `command > filename`

    将输出重定向到文件，如果文件存在，则会被覆盖。

    其实这行命令应该这么理解：`[command] [> filename]`，把它分成两个部分。第二部分`> filename`的完整版应该是`1> filename`，`1`表示 stdout。

    例子：

    `echo "hello, world" > hello.txt`
    
* `[descriptor]> filename`

    把文件描述符重定向到某个文件。`descriptor`可以是`0`，`1`，`2`，其中`0`表示 stdin，`1`表示 stdout，`2`表示 stderr。

    注意，文件描述符和`>`中间不能有空格，而后面的 filename 与`>`之间可以有空格，也可以没有。

    Example:

    file name: `echo_stderr.sh`

    ```bash
    #!/bin/bash

    echo "hello, world!" >& 2
    ```

    `./echo_stderr.sh 2> hello.txt`

    在这个例子中，`echo_stderr.sh`在 stderr 进行输出。而下面的命令`./echo_stderr.sh 2> hello.txt`将 stderr 重定向到`hello.txt`文件。因此可以在文件中得到输出的内容。

    注意，如果将上述命令替换为`./echo_stderr.sh 1> hello.txt`或`./echo_stderr.sh > hello.txt`，则仍会在屏幕中进行输出，`hello.txt`文件中不会有任何内容。

* `[descriptor_1]>& <descriptor_2>`

    将一个文件描述符重定向到另一个文件描述符。

    例子：

    `echo "helo, world" >& 2`

    将 stdout 重定向到 stderr。

    注：

    * `[descriptor_1]>&`这 3 个符号之间不能有空格，而`<descriptor_2>`之前可以有空格，也可以没有。

    * 如果省略不写`[descriptor_1]`，那么`[descriptor_1]`默认为`1`。

* `command &> filename`

    将 stdout 和 stderr 都重定向到指定文件。注意`&>`前没有其他的参数。

    这个命令等价于`1>filename 2>&1`

    一些参考资料：<https://stackoverflow.com/questions/24793069/what-does-do-in-bash>

## Variables

Example:

```bash
#!/usr/bin/bash
my_var=value
echo $my_var
```

* 在定义变量时，等号前后不能有空格

* bash 对大小写敏感

如果`value`中有空格，可以使用单引号`'`或双引号`"`将它们括起来。

单引号不对字符串转义修改，双引号允许你对字符串中的内容替换。

我们可以用`$()`执行命令并将结果返回到字符串中：

`myvar=$( ls /etc | wc -l )`

或者`myvar=$(ls -lh)`（小括号两边的空格不是必须的）

如果结果是多行输出，那么换行符都会被删除，从而并成单行的结果。

* Check if a variable is set in Bash

    Ref: <https://stackoverflow.com/questions/3601515/how-to-check-if-a-variable-is-set-in-bash>

* 可以用`export`导出一个变量，使得一个新的脚本在执行时，以值传递到新脚本中：

    `export var1`

    因为是按值传递，所以原脚本中的`var1`与新脚本中的`var1`修改一个并不会影响另外一个。

## User input

```bash
read var1
```

`read`命令可以读取输入并将其存到变量`var1`中。输入字符串中的空格，单引号，双引号，不会被特殊处理。但是如果输入左方向键，右方向键等，则会有回显乱码。退格键可以正常使用。反斜杠`\` + 换行会被特殊处理。反斜杠`\`加`n`，`t`等字符，不会按转义字符处理。

`-p`参数可以给出输入提示，`-s`则是 silent 模式，输入不回显。

```bash
#!/bin/bash
read -p 'Username: ' uservar
read -sp 'Password: ' passvar
echo
```

这里加`echo`是因为`-s`会抑制回车换行。

读取多个变量：`read var1 var2 var3`。在输入变量时，变量之间使用空格分隔。如果输入的变量数大于指定的变量数，那么会把多余的输入都存储到最后一个输入中。如果输入的变量少于指定的变量，那么多余的变量会保持空白。

假如一个文件有多行内容，使用`cat text.txt | ./test.sh`的方式给脚本`read`，`read`会只处理第一行。换句话说，`read`只按空格对字符串分隔，而不按换行符、制表符等分隔。

在 bash 中，`STDIN`, `STDOUT`, `STDERR`分别对应 3 个 linux 文件：

* `STDIN`: `/proc/<processID>/fd/0`

* `STDOUT`: `/proc/<processID>/fd/1`

* `STDERR`: `/proc/<processID>/fd/2`

同时 linux 还给出了这些文件的快捷方式：

* `STDIN`: `/dev/stdin` or `/proc/self/fd/0`

* `STDOUT`: `/dev/stdout` or `/proc/self/fd/1`

* `STDERR`: `/dev/stderr` or `/proc/self/fd/2`

其中`fd`指的是 file descriptor。

Example:

```bash
#!/bin/bash
# A basic summary of my sales report
echo Here is a summary of the sales data:
echo ====================================
echo
cat /dev/stdin | cut -d' ' -f 2,3 | sort
```

输入与输出：

```
user@bash: cat salesdata.txt
Fred apples 20 August 4
Susy oranges 5 August 7
Mark watermelons 12 August 10
Terry peaches 7 August 15
user@bash:
user@bash: cat salesdata.txt | ./summary
Here is a summary of the sales data:
====================================
apples 20
oranges 5
peaches 7
watermelons 12
user@bash:
```

使用 flag 的例子：

```bash
while getopts u:a:f: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        a) arg=${OPTARG};;
        f) fullname=${OPTARG};;
    esac
done
echo "Username: $username";
echo "Age: $age";
echo "Full Name: $fullname";
```

我们还可以使用`$@`拿到所有的参数：

```bash
i=1
for user in "$@"
do
    echo "Username - $i: $user"
    i=$((i + 1))
done
```

还可以用`shift`改变`$1`这些参数的指向：

```bash
i=1;
j=$#;
while [ $i -le $j ] 
do
    echo "Username - $i: $1";
    i=$((i + 1));
    shift 1;
done
```

**Special variables**:

1. `$0`: The name of the Bash script.

    实际上，这个变量存储的是脚本被调用时的路径名。

    如果用`./my_script.sh`调用，那么`$0`就等于`./my_script.sh`；如果用`/path/to/my_script.sh`绝对路径调用，那么`$0`就等于`/path/to/my_script.sh`；如果使用`bash my_script.sh`调用，那么`$0`就等于`my_script.sh`。

1. `$1` - `$9`: The first 9 arguments to the Bash script.

1. `$#`: How many arguments supplied to the Bash script.

1. `$@`: All the arguments supplied to the Bash script.

1. `$?`: The exit status of the most recently run process.

1. `$$`: The process ID of the current script.

1. `$USER`: The username of the user running the script.

1. `$HOSTNAME`: The hostname of the machine the script is running on.

1. `$SECONDS`: The number of seconds since the script was started.

1. `$RANDOM`: Returns a different random number each time is it referred to.

1. `$LINENO`: Returns the current line number in the Bash script.

1. `env`命令可列出其它可用的环境变量。

## Arithmetic

`let`可以让我们做算术运算。

```bash
#!/bin/bash
# Basic arithmetic using let

let a=5+4
echo $a # 9

let "a = 5 + 4"
echo $a # 9

let a++
echo $a # 10

let "a = 4 * 5"
echo $a # 20

let "a = $1 + 30"
echo $a # 30 + first command line argument
```

可以使用的运算符：

* `+`, `-`, `\*`, `/`
* `var++`, `var--`
* `%`

`expr`的用法和`let`相似，只不过它直接输出表达式的内容，而且不需要将表达式使用双引号括起来。

`expr item1 operator item2`

Examples:

```bash
#!/bin/bash
# Basic arithmetic using expr
expr 5 + 4
expr "5 + 4"
expr 5+4
expr 5 \* $1
expr 11 % 2
a=$( expr 10 - 3 )
echo $a # 7
```

输出

```
user@bash: ./expr_example.sh 12
9
5 + 4
5+4
60
1
7
user@bash:
```

我们也可以用双括号做运算：

`$(( expression ))`

Examples:

```bash
#!/bin/bash
# Basic arithmetic using double parentheses

a=$(( 4 + 5 ))
echo $a # 9

a=$((3+5))
echo $a # 8

b=$(( a + 3 ))
echo $b # 11

b=$(( $a + 4 ))
echo $b # 12

(( b++ ))
echo $b # 13

(( b += 3 ))
echo $b # 16

a=$(( 4 * 5 ))
echo $a # 20
```

可以使用`${#variable}`得到一个变量的长度：

```bash
#!/bin/bash
# Show the length of a variable.

a='Hello World'
echo ${#a} # 11

b=4953
echo ${#b} # 4
```

## If statements

```bash
if [ <some test> ]
then
    <commands>
fi
```

Example:

```bash
#!/bin/bash
# Basic if statement

if [ $1 -gt 100 ]
then
    echo Hey that\'s a large number.
    pwd
fi

date
```

输出：

```
user@bash: ./if_example.sh 15
Sat 21 Aug 3:01:25 2021
user@bash: ./if_example.sh 150
Hey that's a large number.
/home/ryan/bin
Sat 21 Aug 3:01:25 2021
user@bash:
```

`[]`相当于命令`test`，常见的测试命令如下：

|Operator|Description|
|-|-|
|`! EXPRESSION`|The `EXPRESSION` is false.|
|`-n STRING`|The length of `STRING` is greater than zero.|
|`-z STRING`|The length of `STRING` is zero (ie is empty).|
|`STRING1 = STRING2`|`STRING1` is equal to `STRING2`|
|`STRING1 != STRING2`|`STRING1` is not equal to `STRING2`|
|`INTEGER1 -eq INTEGER2`|`INTEGER1` is numerically equal to `INTEGER2`|
|`INTEGER1 -gt INTEGER2`|`INTEGER1` is numerically greater than `INTEGER2`|
|`INTEGER1 -lt INTEGER2`|`INTEGER1` is numerically less than `INTEGER2`|
|`-d FILE`|`FILE` exists and is a directory.|
|`-e FILE`|`FILE` exists.|
|`-r FILE`|`FILE` exists and the read permisson is granted.|
|`-s FILE`|`FILE` exists and it's size is greater than zero (ie. it is not empty).|
|`-w FILE`|`FILE` exists and the write permission is granted.|
|`-x FILE`|`FILE` exists and the execute permission is granted.|

Examples:

```
user@bash: test 001 = 1
user@bash: echo $?
1
user@bash: test 001 -eq 1
user@bash: echo $?
0
user@bash: touch myfile
user@bash: test -s myfile
user@bash: echo $?
1
user@bash: ls /etc > myfile
user@bash: test -s myfile
user@bash: echo $?
0
user@bash: 
```

bash 对缩进没有要求，但是最好还是缩进。

嵌套的 if 语句：

```bash
#!/bin/bash
# Nested if statements
if [ $1 -gt 100 ]
then
    echo Hey that\'s a large number.

    if (( $1 % 2 == 0 ))
    then
        echo And is also an even number.
    fi
fi
```

if else 语句：

```bash
if [ <some test> ]
then
    <commands>
else
    <other commands>
fi
```

Example:

```bash
#!/bin/bash
# else example

if [ $# -eq 1 ]
then
    nl $1
else
    nl /dev/stdin
fi
```

if elif else:

```bash
if [ <some test> ]
then
    <commands>
elif [ <some test> ]
then
    <different commands>
else
    <other commands>
fi
```

Example:

```bash
#!/bin/bash
# elif statements

if [ $1 -ge 18 ]
then
    echo You may go to the party.
elif [ $2 == 'yes' ]
then
    echo You may go to the party but be back before midnight.
else
    echo You may not go to the party.
fi
```

You can have as many `elif` branches as you like. The final `else` is also optional.

Boolean Operations:

and: `&&`, or: `||`

Example:

```bash
#!/bin/bash
# and example

if [ -r $1 ] && [ -s $1 ]
then
    echo This file is useful.
fi
```

```bash
#!/bin/bash
# or example
if [ $USER == 'bob' ] || [ $USER == 'andy' ]
then
    ls -alh
else
    ls
fi
```

Case Statements:

```bash
case <variable> in
<pattern 1>)
    <commands>
    ;;
<pattern 2>)
    <other commands>
    ;;
esac
```

Example:

```bash
#!/bin/bash
# case example
case $1 in
    start)
        echo starting
        ;;
    stop)
        echo stoping
        ;;
    restart)
        echo restarting
        ;;
    *)
        echo don\'t know
        ;;
esac
```

Another example:

```bash
#!/bin/bash
# Print a message about disk useage.
space_free=$( df -h | awk '{ print $5 }' | sort -n | tail -n 1 | sed 's/%//' )
case $space_free in
    [1-5]*)
        echo Plenty of disk space available
        ;;
    [6-7]*)
        echo There could be a problem in the near future
        ;;
    8*)
        echo Maybe we should look at clearing out old files
        ;;
    9*)
        echo We could have a serious problem on our hands soon
        ;;
    *)
        echo Something is not quite right here
        ;;
esac
```

## Loops

while:

```bash
while [ <some test> ]
do
    <commands>
done
```

Example:

```bash
#!/bin/bash
# Basic while loop

counter=1
while [ $counter -le 10 ]
do
    echo $counter
    ((counter++))
done

echo All done
```

until:

```bash
until [ <some test> ]
do
    <commands>
done
```

Example:

```bash
#!/bin/bash
# Basic until loop

counter=1
until [ $counter -gt 10 ]
do
    echo $counter
    ((counter++))
done

echo All done
```

for:

```bash
for var in <list>
do
    <commands>
done
```

The list is defined as a series of strings, seperated by spaces.

Example:

```bash
#!/bin/bash
# Basic for loop

names='Stan Kyle Cartman'
for name in $names
do
    echo $name
done

echo All done
```

```bash
#!/bin/bash
# Basic range in for loop
for value in {1..5}
do
    echo $value
done
echo All done
```

```bash
#!/bin/bash
# Basic range with steps for loop

for value in {10..0..2}
do
    echo $value
done
echo All done
```

```bash
#!/bin/bash
# Make a php copy of any html files

for value in $1/*.html
do
    cp $value $1/$( basename -s .html $value ).php
done
```

```bash
x=5
for ((i=1; i<=x; i++))
do
    echo $i
done
```

break 与 continue:

```bash
#!/bin/bash
# Make a backup set of files

for value in $1/*
do
    used=$( df $1 | tail -1 | awk '{ print $5 }' | sed 's/%//' )
    if [ $used -gt 90 ]
    then
        echo Low disk space 1>&2
        break
    fi
    cp $value $1/backup/
done
```

```bash
#!/bin/bash
# Make a backup set of files

for value in $1/*
do
    if [ ! -r $value ]
    then
        echo $value not readable 1>&2
        continue
    fi
    cp $value $1/backup/
done
```

select:

```bash
select var in <list>
do
    <commands>
done
```

`select`可以创建一个菜单。当接收到`EOF` signal，或者执行`break`语句时退出菜单循环。

Example:

```bash
#!/bin/bash
# A simple menu system

names='Kyle Cartman Stan Quit'

PS3='Select character: '

select name in $names
do
    if [ $name == 'Quit' ]
    then
        break
    fi
    echo Hello $name
done
echo Bye
```

修改变量`PS3`可以改变提示语。

## Functions

```bash
# form 1
function_name () {
    <commands>
}

# form 2
function function_name {
    <commands>
}
```

bash 中的函数不能有形参，但给函数传递参数时，可以用`$1`，`$2`等：

```bash
#!/bin/bash
# Passing arguments to a function

print_something () {
    echo Hello $1
}

print_something Mars
print_something Jupiter
```

函数不能有返回值，但是可以返回一个状态（status）。

```bash
#!/bin/bash
# Setting a return status for a function

print_something () {
    echo Hello $1
    return 5
}

print_something Mars
print_something Jupiter
echo The previous function has a return value of $?
```

如果想返回字符串，我们可以把函数当作一个命令来执行：

```bash
#!/bin/bash
# Setting a return value to a function

lines_in_file () {
    cat $1 | wc -l
}

num_lines=$( lines_in_file $1 )
echo The file $1 has $num_lines lines in it.
```

如果我们创建一个变量，那么它默认是`global`属性的。如果我们在函数中使用`local`关键字创建一个变量，那么这个变量就只能在函数中可见：

`local var_name=<var_value>`

Example:

```bash
#!/bin/bash
# Experimenting with variable scope

var_change () {
    local var='local 1'
    echo Inside function: var1 is $var1 : var2 is $var2
    var1='changed again'
    var2='2 changed again'
}

var1='global 1'
var2='global 2'

echo Before function call: var1 is $var1 : var2 is $var2

var_change

echo After function call: var1 is $var1 : var2 is $var2
```

我们可以使用重名函数来覆盖 linux 命令：

```bash
#!/bin/bash
# Create a wrapper around the command ls

ls () {
    command ls -lh
}

ls
```

因为函数的优先级较高，所以想区别命令和函数时，需要在 linux 前加`command`。如果不加`command`，就会形成递归调用。

## User interface

## Common used command

* 简述 bash 中 alias 的用法

    在 Bash 中，`alias`（别名）是一个非常实用的功能。简单来说，它允许你为冗长的命令设定一个**简短的“绰号”**，从而提高敲代码的效率。

    1. 基本语法

        设置别名的标准格式如下：

        ```bash
        alias 别名='实际执行的命令'

        ```

        > **注意**：等号 `=` 两边**不能有空格**，否则 Bash 会报错。

    2. 常用操作

        | 操作类型 | 命令示例 | 说明 |
        | --- | --- | --- |
        | **创建别名** | `alias ll='ls -alF'` | 输入 `ll` 即可执行详细列出文件的命令 |
        | **查看单个别名** | `alias ll` | 查看 `ll` 具体指向什么命令 |
        | **查看所有别名** | `alias` | 直接输入 `alias` 不带参数，列出当前所有已定义的别名 |
        | **删除别名** | `unalias ll` | 取消 `ll` 这个别名 |
        | **临时跳过别名** | `\ls` | 在命令前加反斜杠，强制执行原始程序而非别名 |

    3. 让别名永久生效

        如果你直接在终端输入 `alias` 命令，它只在**当前会话**有效。一旦关闭终端或重启，别名就会消失。

        要让它永久生效，需要将其写入 Bash 的配置文件中：

        1. 打开文件：`nano ~/.bashrc` (或使用 `vim`)。

        2. 在文件末尾添加你的别名，例如：

            ```bash
            alias gs='git status'
            alias ..='cd ..'
            alias py='python3'
            ```


        3. 保存退出后，执行以下命令立即使其生效：

            `source ~/.bashrc`

    4. 经典用法示例

        这里有一些大家公认的“效率利器”：

        * **安全保护**：防止误删重要文件。

            `alias rm='rm -i'` (删除前确认)

        * **简化路径**：快速回退目录。

            `alias ..='cd ..'`

            `alias ...='cd ../..'`

        * **网络检查**：

            `alias myip="curl ifconfig.me"`

        * **组合命令**：

        `alias update='sudo apt update && sudo apt upgrade'`

    ⚠️ 注意事项

    1. **参数处理**：`alias` 只能简单替换命令头。如果你需要处理复杂的参数（例如：把参数插在命令中间），建议使用 **Bash 函数 (Function)**。

    2. **优先级**：别名的优先级高于系统内建命令和可执行文件。如果你定义了 `alias ls='echo hi'`，那么原本的 `ls` 就被“遮住”了。


### trap

Syntax:

`trap COMMAND SIGNALS...`

Examples:

1. `EXIT`

    ```bash
    tempfile=/tmp/tmpdata
    trap "rm -f $tempfile" EXIT
    ```

    ```bash
    function cleanup()
    {
        # ...
    }
    
    trap cleanup EXIT
    ```

    Note that if you send a `kill -9` to your script, it will not execute the `EXIT` trap before exiting.

1. `SIGINT`

    `SIGINT` can catch Ctrl-C.

    ```bash
    ctrlc_count=0

    function no_ctrlc()
    {
        let ctrlc_count++
        echo
        if [[ $ctrlc_count == 1 ]]; then
            echo "Stop that."
        elif [[ $ctrlc_count == 2 ]]; then
            echo "Once more and I quit."
        else
            echo "That's it. I quit."
            exit
        fi
    }

    trap no_ctrlc SIGINT

    while true
    do
        echo Sleeping
        sleep 10
    done
    ```

1. `-`

    ```bash
    # Run something important, no Ctrl-C allowed.
    trap "" SIGINT
    important_command

    # Less important stuff from here on out, Ctrl-C allowed.
    trap - SIGINT
    not_so_important_command
    ```

1. `USR1`

    ```bash
    nopens=0
    function show_opens()
    {
        echo "Seen $nopens sudo session opens"
    }

    sudo journalctl -f | while read line
    do
        if [[ -z "$trap_set" ]]; then
            trap_set=1
            echo "Trap set in $BASHPID"
            trap show_opens USR1
        fi
        if [[ $line =~ sudo.*session.*opened ]]; then
            let nopens++
        fi
    done
    ```

    使用：

    ```
    $ sudo -k  # reset the sudo timestamp
    $ bash bkgnd.sh &
    [1] 1000
    Trap set in 1002
    $ kill -SIGUSR1 1002
    ```

## String manipulation

1. string length

    `${#string}`

    `expr length $string`

    `expr "$string" : '.*'`

1. extract substring

    `${string:position}`

    `${string:position:length}`

    (0-based indexing)

    If the $string parameter is `*` or `@`, then this extracts the positional parameters, [1] starting at `$position`.

    `expr substr $string $position $length`
    
    `expr match "$string" '\($substring\)'`, `expr "$string" : '\($substring\)'`: Extracts `$substring` at beginning of `$string`, where `$substring` is a regular expression.

    `expr match "$string" '.*\($substring\)'`, `expr "$string" : '.*\($substring\)'`: Extracts `$substring` at end of `$string`, where `$substring` is a regular expression.

    Examples:

    ```bash
    #!/bin/bash

    var="Welcom to the geekstuff"
    echo ${var:15}  # geekstuff
    echo ${var:15:4}  # geek

    stringZ=abcABC123ABCabc
    echo ${stringZ:-4}  # abcABC123ABCabc
    echo ${stringZ:(-4)}  # Cabc
    echo ${stringZ: -4}  # Cabc

    echo ${*:2}  # the second and following positional parameters
    echo ${@:2}  # same as above
    echo ${*:2:3}  # three positional parameters, starting at the second

    echo `expr substr $stringZ 1 2`  # ab
    echo `expr substr $stringZ 4 3`  # ABC
    ```

1. Shortest substring match

    delete the shortest match of `$substring` from front of `$string`: `${string#substring}`

    delete the shortest match of `$substring` from back of `$string`: `${string%substring}`

    Examples:

    ```bash
    #!/bin/bash

    filename="bash.string.txt"
    echo ${filename#*.}  # string.txt
    echo ${filename%.*}  # bash.string
    ```

1. Longest substring match

    `${string##substring}`

    `${string%%substring}`

1. Find and replace string values

    Replace only first match: `${string/pattern/replacement}`

    Replace all the matches: `${string//pattern/replacement}`

    Replace beginning and end: `${string/#pattern/replacement}`, `${string/%pattern/replacement}`

    Examples:

    ```bash
    #!/bin/bash

    filename="bash.string.txt"
    echo ${filename/str*./operations.}  # bash.operations.txt

    filename="Path of the bash is /bin/bash"
    echo ${filename//bash/sh}  # Path of the sh is /bin/sh

    filename="/root/admin/monitoring/process.sh"

    echo ${filename/#\/root/\/tmp}  # /tmp/admin/monitoring/process.sh
    echo ${filename/%.*/.ksh}  # /root/admin/monitoring/process.ksh
    ```

1. Length of matching substring at beginning of string

    `expr match "$string" '$substring'`

    `expr "$string" : '$substring'`

    Note: `$substring` is a regular expression.

    Example:

    ```bash
    stringZ=abcABC123ABCabc
    #       |------|
    #       12345678

    echo `expr match "$stringZ" 'abc[A-Z]*.2'`   # 8
    echo `expr "$stringZ" : 'abc[A-Z]*.2'`       # 8
    ```

1. index

    `expr index $string $substring`

    Example:

    ```bash
    stringZ=abcABC123ABCabc
    #       123456 ...
    echo `expr index "$stringZ" C12`            # 6
                                                # C position.

    echo `expr index "$stringZ" 1c`             # 3
    # 'c' (in #3 position) matches before '1'.
    ```

1. convert a string to lower case in Bash

    posix:

    1. tr
    
        `echo "$a" | tr '[:upper:]' '[:lower:]'`

    1. awk
    
        `echo "$a" | awk '{print tolower($0)}'`

    Ref: <https://stackoverflow.com/questions/2264428/how-to-convert-a-string-to-lower-case-in-bash>

1. Converting a Bash array into a delimited string

    ```bash
    ids="1 2 3 4";echo ${ids// /|}
    ```

    output:

    ```
    1|2|3|4
    ```

    Ref: <https://stackoverflow.com/questions/13470413/converting-a-bash-array-into-a-delimited-string>

## Subshell

Materials:

1. <https://tldp.org/LDP/abs/html/subshells.html>

* parent shell create a new subshell and get the pid of the subshell

    Ref: <https://stackoverflow.com/questions/20573621/bash-get-process-id-of-a-process-started-in-subshell>


## File

bash 通常通过标准输入输出和文件进行交互。

按行读取文件并回显：

`content.txt`:

```
hello      world
nihao
zaijian
```

`test.sh`

```bash
#!/bin/bash

file_path=./content.txt
while read -r line
do
    echo "$line"
done < $file_path
```

输出：

```
hello      world
nihao
zaijian
```

可以看到，文件通过重定向的方式，被`read`函数捕获到，每次处理一行。

说明：

1. 在`echo "$line"`时，必须加上双引号，如果不加，bash 会首先把`$line`展开为带空格的字符串列表，然后按多个参数给`echo`输出。此时第一行就会变成`hello world`。

1. 如果`content.txt`的最后一行的末尾没有`\n`，那么`read`在读取完最后一行后，会返回 false，导致 while 循环退出，从而不会打印最后一行。但是此时最后一行的值已经存在了`$line`变量中，我们还可以在 while 外部将其打印出来。

    这种情况最好的解决办法就是让每旧文件的末尾都最好带有`\n`，使得 bash 能正常处理。

    如果遇到别人写的文本文档没有`\n`，自己写脚本处理时，可以检测跳出循环后的`$line`值是否为 empty。如果非空，说明最后一行仍有内容。

1. 如果`content.txt`文件里有反斜杠`\`，那么`read`会先将反斜杠后面的字符处理为转义字符，然后再将值存入到`$line`变量中。

    这显然不是我们所希望的，所以需要给`read`加上`-r`参数，保证所有字符都不会被转义。

1. 参考资料上的`read`写法是`IFS= read -r line`，不清楚`IFS`有什么用处。

    Ref: <https://www.cyberciti.biz/faq/unix-howto-read-line-by-line-from-file/>

## Miscellaneous

1. 有关标准输入输出，标准错误

    * stdin: `/proc/<PID>/fd/0`，`/proc/self/fd/0`，`/dev/stdin`，`0`，这几种都是等价的，下面的同理。

    * stdout: `/proc/<PID>/fd/1`，`/proc/self/fd/1`，`/dev/stdout`，`1`

    * stderr: `/proc/<PID>/fd/2`，`/proc/self/fd/2`，`/dev/stderr`，`2`

    程序可以通过 stdin 接收管道（pipe）传递的值：

    `recv.sh`:

    ```bash
    #!/bin/bash
    $input$(cat /dev/stdin)
    echo "recv: $input"
    ```

    `echo "hello" | ./recv.sh`

1. 有关进程和子进程

    当前进程的 PID：`echo "in current process, PID: $$"`

    可以用小括号打开一个子进程：`(echo "in child process, PID: $BASHPID")`（好像也可以用`$PPID`获得子进程（subshell）的 pid，但我没试过）

    注意子进程的 PID 不可以用`$$`获得。因为子进程会从父进程中继承一些环境变量。

    除了使用小括号可以打开一个子进程外，执行一个别的脚本`./xxx.sh`或`bash xxx.sh`也会打开一个子进程。

    将子进程放到后台执行：`(echo "hello") &`

    （似乎所有的后台命令都是开了个新进程，比如`echo hello &`）

    获得子进程的返回值（exit code）：

    1. 如果是前台的子进程，可以使用`$?`获得

        ```bash
        (echo "in child process"; exit 1)
        echo "the exit code of the child process: $?"
        ```

    1. 如果是后台的子进程，必须使用`wait`获得

        ```bash
        exit 1 &
        wait $!  # $! is the PID of the latest background process
        echo "the exit code of the background child process is $?"
        ```

        `wait`命令会等待子进程执行完成，并将其返回值传递到`$?`变量中。

    `$VAR=$(command)$`也会开启一个子进程：
    
    ```bash
    #!/bin/bash
    child_pid=$(echo $BASHPID)
    echo "child pid: $child_pid"
    echo "current pid: $$"
    ```

1. 有关字符串处理

* if statement

    <https://acloudguru.com/blog/engineering/conditions-in-bash-scripting-if-statements>

    <https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php>

* bash 中默认函数内外的变量都是全局变量。可以用`local xxxx`或`local xxx=xxxx`定义局部变量。

1. debug a bash script

    vscode 里有个插件叫 bash debug 可以对 bash 设置断点。但是如果是手动调试的话，可以看看下面这些链接：

    * <https://unix.stackexchange.com/questions/521775/how-to-debug-trace-bash-function>

    * <https://www.shell-tips.com/bash/debug-script/#gsc.tab=0>

* 有关返回值

    前台进程的返回值可以直接由`$?`获得：

    ```bash
    echo "hello" | grep hello
    echo $?  # 显示 0
    echo "hello" | grep world
    echo $?  # 字符串中没有 world，返回 1
    ```

    后台进程的返回值：

    后台进程的返回值不能直接获得，必须由`wait <PID>`才能获得。

    我们可以首先由`$!`得到最近后台进程的 pid，然后通过`wait $!`得到后台进程的返回值。

    Ref:

    1. <https://stackoverflow.com/questions/1570262/get-exit-code-of-a-background-process>

    1. <https://www.baeldung.com/linux/background-process-get-exit-code>

* `if`中的字符串在使用`-n`比较时，要加`""`

    Supposing `b` is an undefined variable，

    ```bash
    if [ -n "$b" ]; then echo "hello"; fi
    ```

    output： nothing

    ```bash
    if [ -n $b ]; then echo "hello"; fi
    ```

    output:

    ```
    hello
    ```

* EOF

    There is no method to echo an `EOF` directly. But there are some ways to trigger an `EOF`:

    1. reaching the end of a file

    1. pressing key bindings to `EOF` (`Ctrl + D` by default)

    1. `cat <<EOF`

        Ref: <https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash>

* `bash -c`

    <https://unix.stackexchange.com/questions/144514/add-arguments-to-bash-c>

* learning materials

    1. <https://askubuntu.com/questions/121866/why-does-bash-remove-n-in-cat-file>

    1. <https://stackoverflow.com/questions/10028820/bash-wait-with-timeout>

    1. <https://stackoverflow.com/questions/42615374/the-linux-timeout-command-and-exit-codes>

    1. <https://stackoverflow.com/questions/13296863/difference-between-wait-and-sleep>

    1. <https://superuser.com/questions/1699317/permission-denied-on-dev-stderr-after-sudo>

    1. <https://unix.stackexchange.com/questions/38538/bash-dev-stderr-permission-denied>

    1. <https://stackoverflow.com/questions/37981492/how-to-timeout-a-group-of-commands-in-bash>

    1. <https://linuxconfig.org/bash-scripting-tutorial>

    1. <https://www.javatpoint.com/bash-split-string>

    1. <https://www.tutorialkart.com/bash-shell-scripting/bash-split-string/>

    1. <https://www.freecodecamp.org/news/bash-array-how-to-declare-an-array-of-strings-in-a-bash-script/>

    1. <https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays>

    1. <https://linuxhint.com/bash_split_examples/>

    1. <https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash>

    1. <https://stackoverflow.com/questions/3601515/how-to-check-if-a-variable-is-set-in-bash>

    1. <https://stackoverflow.com/questions/71768999/how-to-merge-when-you-get-error-hint-you-have-divergent-branches-and-need-to-s>

    1. <https://stackoverflow.com/questions/9954794/execute-a-shell-function-with-timeout>

1. How to Create Bash Aliases

    <https://linuxize.com/post/how-to-create-bash-aliases/>



# grep note

## cache

* `grep -c`

    `grep -c "pattern" filename`

    只显示行数，不显示内容。

    如果统计多个文件，则分别显示行数：

    `grep -c "GET" access.log access.log.1`

    output:

    ```
    access.log: 1200
    access.log.1: 800
    ```

    说明：

    1. 单行多次匹配：`-c`只统计行数，即使一行中多次匹配模式，仍计为`1`。

        如需统计所有匹配次数（非行数），可用`grep -o "pattern" | wc -l`。

* `grep -n`可以显示行号。行数从 1 开始计数。

* `fgrep`与`grep -F`都表示 Fixed-string grep，`fgrep`是旧版 linux 的独立命令，不推荐使用。目前更推荐使用`grep -F`.

* 使用`grep`搜索一个文件中的`\|`字符串

    * `grep '\\|' example.txt`

        默认情况下，`grep`使用的模式是`-e`基本正则模式。在 bash 下`|`会被解释成管道，我们先使用单引号`'`绕开 bash 的解释。`-e`模式下，`|`在 grep 中是普通字符，不代表或运算，而`\`在 grep 中默认被解释为转义字符，我们需要用`\\`将其变为普通字符。

        output:

        ```
        ha\|ha
        ```

        其中`\|`为红色。

    * `grep -E '\\\|' example.txt`

        `-E`模式为扩展正则模式，此时`|`被解释为或运算，我们需要`\|`将其转义为普通字符。

    * `grep -F '\|' example.txt`

        `-F`表示禁用正则表达式，只使用普通字符匹配。

* `grep -v`命令

    `grep -v`指的是反向匹配（invert match）。

    example:

    `msg.txt`:

    ```
    hello
    world
    nihao
    zaijian
    ```

    run:

    `grep world msg.txt`

    output:

    ```
    world
    ```

    其中`world`被标红。

    run:

    `grep -v world msg.txt`

    output:

    ```
    hello
    nihao
    zaijian
    ```

    这三行都没有被标红。

* grep 搜索一行内同时出现多个关键字

    example:

    `msg.txt`:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    ```

    需要搜索同时出现`hello`, `workd`和`nihao`的行。

    * 方案一，使用多个管道

        `grep hello ./msg.txt | grep world | grep nihao`

        output:

        ```
        hello, world, nihao, zaijian
        ```

        其中，只有`nihao`是标红的。我们希望的是`hello`, `workd`, `nihao`这三个单词都标红。

        可以使用`--color=always`实现这个效果：

        `grep hello ./msg.txt --color=always | grep --color=always world | grep nihao`

        最后一个 grep 本身就有标红功能，可以不写`--color`参数。

        output:

        ```
        hello, world, nihao, zaijian
        ```

        其中,`hello`, `world`, `nihao`分别被标红。

    * 方案二，使用`.*`连接三个关键字

        `grep hello.*world.*nihao ./msg.txt`

        output:

        ```
        hello, world, nihao, zaijian
        ```

        这种方式，整个`hello, world, nihao`字符串都标红，也不是我们想要的。

        而且这种方式，要求`hello`, `world`, `nihao`这三个关键字的顺序不能乱，如果我们无法事先知道顺序，那么就需要把所有的顺序都试一遍。

    可见，grep 并没有很方便地实现搜索 pattern_1 AND pattern_2 AND pattern_3 AND ... 的命令，但是使用管道还是能实现的。如果有需求并且有时间的话，我们可以自己定义一个命令实现这个功能。

* grep 搜索一行内出现多个关键词的其中一个

    example:

    `msg.txt`:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    ```

    `grep -e hello -e world -e nihao ./msg.txt`

    output:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    ```

    输出中所有的`hello`, `world`, `nihao`都被标红。

* `grep`使用`-l`参数，可以只输出文件路径，不输出具体匹配了哪一行

* grep 搜索子目录下的指定文件

    `grep -r --include=<glob_exp> <reg_pattern> .`

    这里使用`--include`来指定要搜索的文件名，可以使用等号，也可以把等号替换成空格。

    注意`--include`使用的是通配符表达式，不是正则表达式。为了防止 bash 对输入内容进行转义，通常使用单引号`'<glob_exp>'`将通配符表达式包裹。

    example:

    `grep -r --include='hello*.txt' hello .`

    output:

    ```
    ./dir_2/hello.txt:hello, world
    ./dir_2/hello_w.txt:hello, world
    ```

    如果要指定多个通配符，那么可以指定多个`--include`参数。

## note
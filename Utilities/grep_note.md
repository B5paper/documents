# grep note

## cache

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
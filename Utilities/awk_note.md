# awk note

## cache

* awk

    命令基本结构：
    
    `awk '模式 {动作}' 文件名`

    * 模式：可选，用于筛选行（如 `/正则/`、`条件表达式`）。

    * 动作：对匹配的行执行的操作（如 `print`、计算）。

    常用内置变量:

    * `$0`：整行内容。

    * `$1, $2...`：第1、2...列字段。

    * `NF`：当前行的字段数。

    * `NR`：当前行号。

    * `FS`：输入字段分隔符（默认为空格/制表符）。

    * `OFS`：输出字段分隔符（默认为空格）。

    常见用法:

    * 打印指定列

        `msg.txt`:

        ```
        hello world nihao zaijian
        haha hehe haihai huaihuai
        1 2 3 4
        ```

        ```bash
        awk '{print $1, $3}' msg.txt      # 打印第1列和第3列
        ```

        output:

        ```
        hello nihao
        haha haihai
        1 3
        ```

    * 条件过滤

        `data.txt`:

        ```
        1 2 3
        4 5 6
        7 8 9
        10 11hehe 12
        ```

        ```bash
        awk '$2 >= 5 {print $0}' data.txt # 打印第 2 列大于等于 5 的行
        ```

        output:

        ```
        4 5 6
        7 8 9
        ```

        如果在比较时，发现有非数字项，那么会被当作`0`处理。所以`11hehe`没有被输出。数据中的小数也可以被正确处理。

        如果需要做“等于”比较，那么可以使用两个等号`==`。

        ```bash
        awk '/error/ {print NR, $0}' log.txt # 打印包含"error"的行及其行号
        ```

## topics

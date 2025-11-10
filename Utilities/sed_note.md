# Sed Note

## cache

* sed（Stream Editor）是 Linux/Unix 下的流编辑器，主要用于对文本进行过滤和转换

    * 替换操作

        ```py
        # 基本替换（每行第一个匹配）
        sed 's/old/new/' file.txt

        # 全局替换（所有匹配）
        sed 's/old/new/g' file.txt

        # 替换第N次出现
        sed 's/old/new/2' file.txt

        # 只替换匹配的行
        sed '/pattern/s/old/new/g' file.txt
        ```

        example:

        `test.txt`:

        ```
        hello, hello
        world, hello

        ```

        run: `set s/hello/nihao/ test.txt`

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
# tar note

## cache

* `tar -t`只查看顶层目录

    方法一：使用 --strip-components（推荐）

    ```bash
    tar -tf archive.tar.gz --strip-components=0

    --strip-components=0 会显示完整的路径，但配合其他命令可以筛选出顶层目录。
    ```

    方法二：使用管道组合命令

    ```bash
    # 只显示第一级目录/文件（不包含子目录内容）
    tar -tf archive.tar.gz | cut -d/ -f1 | sort -u

    # 或者使用 awk
    tar -tf archive.tar.gz | awk -F/ '{print $1}' | sort -u

    # 或者使用 sed
    tar -tf archive.tar.gz | sed 's|/.*||' | sort -u
    ```

    方法三：查看目录类型的顶层项目

    ```bash
    # 只显示顶层目录（排除文件）
    tar -tf archive.tar.gz | grep -E '^[^/]+/$' | head -20

    # 显示所有顶层项目（目录和文件）
    tar -tf archive.tar.gz | grep -v '/' | head -20
    ```

    方法四：使用 tar 的过滤选项（tar 1.28+）

    ```bash
    # 只显示目录（部分版本支持）
    tar -tf archive.tar.gz --no-wildcards-match-slash --wildcards "*/"
    ```

    实际示例

    假设有一个归档文件内容如下：

    ```text
    project/
    project/src/
    project/src/main.py
    project/docs/
    project/docs/readme.txt
    project/requirements.txt
    ```

    查看顶层目录：

    ```bash
    tar -tf project.tar.gz | cut -d/ -f1 | sort -u
    ```

    输出：

    ```text
    project
    ```

    查看顶层所有项目（包含文件和目录）：

    ```bash
    tar -tf project.tar.gz | awk -F/ 'NF==1 {print $0}' | sort -u
    ```

    常用快捷命令

    创建一个别名方便使用：

    ```bash
    # 添加到 ~/.bashrc
    alias tartop='tar -tf $1 | cut -d/ -f1 | sort -u'

    # 使用
    tartop archive.tar.gz
    ```

    这样就能快速查看归档文件的顶层结构了。

* tar

    tar 是 Linux/Unix 系统中用于打包和压缩文件的核心命令，名称源于 Tape Archive（磁带归档）。

    **基本语法**

    ```bash
    tar [选项] [输出文件] [输入文件/目录]
    ```

    **常用操作选项**

    1. 主要操作（必须选一个）

        * `-c`：创建新归档（create）。

        * `-x`：解压归档（extract）。

        * `-t`：查看归档内容（list）。

        * `-r`：向归档追加文件（不常用）。

    2. 辅助选项

        * `-f [文件名]`：指定归档文件名（file）。必须放在选项最后。

        * `-v`：显示处理过程（verbose）。

        * `-z`：通过 gzip 压缩/解压（后缀通常为 .tar.gz 或 .tgz）。

        * `-j`：通过 bzip2 压缩/解压（后缀 .tar.bz2）。

        * `-J`：通过 xz 压缩/解压（后缀 .tar.xz）。

        * `-C [目录]`：解压到指定目录（Change directory）。

        * `--exclude=`：排除特定文件/目录。

    **常用示例**

    1. 打包与压缩

        ```bash
        # 打包目录（不压缩）
        tar -cvf archive.tar /path/to/dir

        # 打包并用 gzip 压缩
        tar -czvf archive.tar.gz /path/to/dir

        # 打包并用 bzip2 压缩
        tar -cjvf archive.tar.bz2 /path/to/dir

        # 打包时排除某些文件
        tar -czvf backup.tar.gz --exclude="*.log" /path/to/dir
        ```

    2. 查看归档内容

        ```bash
        tar -tf archive.tar.gz      # 仅列表
        tar -tzvf archive.tar.gz    # 列表并显示详细信息
        ```

        注：

        1. tar 无法快速只查看顶层文件，必须配合正则表达式之类的命令来过滤才能看到。

    3. 解压

        ```bash
        # 解压到当前目录
        tar -xvf archive.tar
        tar -xzvf archive.tar.gz    # 解压 gzip
        tar -xjvf archive.tar.bz2   # 解压 bzip2

        # 解压到指定目录
        tar -xzvf archive.tar.gz -C /target/dir
        ```

    4. 直接操作压缩包内容

        ```bash
        # 不解压直接查看文件内容
        tar -O -xzf archive.tar.gz file.txt | less

        # 追加文件到压缩包（仅适用于未压缩的 tar 包）
        tar -rf archive.tar newfile.txt
        ```

    **常用组合速记**

    * 创建压缩包：`tar -czvf [输出文件.tar.gz] [目录]`

    * 解压压缩包：`tar -xzvf [文件.tar.gz]`

    * 查看内容：`tar -tzvf [文件.tar.gz]`

    **注意事项**

    * 选项顺序：-f 必须后接文件名，且通常放在最后。

    * 保留权限：默认保留文件权限（如需保留所有权用 -p，但需 root 权限）。

    * 现代简化写法：较新版本的 tar 支持省略 -，如 tar xvf archive.tar。

    * 压缩效率：xz > bzip2 > gzip（压缩率越高，耗时越长）。

    **实用技巧**

    ```bash
    # 1. 打包时保留软链接（默认行为）
    tar -czhf backup.tar.gz /path

    # 2. 仅打包比指定日期新的文件
    tar -czvf newfiles.tar.gz --newer="2023-01-01" /path

    # 3. 增量备份（结合 find）
    find /data -type f -mtime -7 | tar -czvf backup_week.tar.gz -T -
    ```
    
    掌握这些核心用法即可应对大多数场景，更多细节可通过 `man tar` 查看。

* tar 在打包目录时，要求目录中的文件不能有改动。如果还没来得及打包的文件被修改，那么就会报错。

    `tar: file changed as we read it`

## note

解压时指定目录：

`tar -xvf articles.tar --directory /tmp/my_articles/`

这个好像也行：

`tar -zvxf documents.tgz -C /tmp/tgz/ `

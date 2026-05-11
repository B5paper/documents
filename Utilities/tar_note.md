# tar note

## cache

* tar -t 我希望只查看顶层目录，怎么办

    在使用 `tar -t`（或 `tar --list`）命令时，默认会递归地列出压缩包内的所有文件。如果只想查看**顶层目录和文件**，可以通过以下几种方法实现：

    方法一：使用 `--exclude` 参数（最简单）

    这是最直接的方法，通过排除所有包含二级路径（即含有 `/`）的内容来实现。

    ```bash
    tar -tf your_archive.tar --exclude='*/*'
    ```

    * **原理**：`*/*` 匹配所有包含至少一个斜杠的路径。排除掉它们后，剩下的就是顶层的文件和文件夹。
    * **注意**：如果是 `.tar.gz` 文件，记得加上 `-z` 参数（即 `tar -ztf ...`）。

    方法二：使用 `awk` 进行过滤（最通用）

    如果你希望更精确地控制输出，可以使用 `awk` 根据路径深度进行过滤。

    ```bash
    tar -tf your_archive.tar | awk -F/ 'NF<=2 && (NF==1 || $2=="")'
    ```

    * **`-F/`**：将斜杠 `/` 作为分隔符。
    * **`NF<=2`**：限制段数。顶层文件段数为 1，顶层目录（如 `folder/`）段数为 2。
    * **`(NF==1 || $2=="")`**：确保如果是两段，第二段必须为空（即排除 `folder/subfile`）。

    方法三：使用 `grep` 正则表达式

    通过正则匹配不包含中间斜杠的行。

    ```bash
    tar -tf your_archive.tar | grep -E '^[^/]+/?$'
    ```

    * **`^[^/]+`**：从开头开始，匹配非斜杠字符。
    * **`/?$`**：结尾处可以有一个可选的斜杠（代表目录），然后结束。

    **总结与建议**

    | 方法 | 优点 | 缺点 |
    | --- | --- | --- |
    | **`--exclude`** | 命令简洁，无需管道 | 某些老旧版本的 tar 可能支持不佳 |
    | **`awk`** | 逻辑最严密，适合复杂场景 | 命令稍长 |
    | **`grep`** | 速度快，过滤方便 | 正则表达式对初学者稍显复杂 |

    **小贴士：**

    如果你的压缩包非常大，列出所有文件可能会很慢。通常建议配合 `head` 先看前几个，确认结构：
    `tar -tf your_archive.tar --exclude='*/*' | head -n 10`

* tar 在递归列出文件时，是按 top_a, top_a/sub_a, top_a/sub_b 的顺序，还是按 top_a, top_b, top_c, top_a/sub_a, top_a/sub_b ... 的顺序？

    简单直接的回答是：**`tar` 默认通常按照 `top_a, top_a/sub_a, top_a/sub_b` 的顺序（即深度优先遍历）排列。**

    这是由 `tar` 的本质和它创建压缩包的方式决定的。

    1. 为什么是这种顺序？

        `tar` 的全称是 **T**ape **Ar**chive（磁带归档）。在设计之初，它是为了向磁带这种**顺序存储介质**写入数据的。

        * **创建过程**：当你执行 `tar -cvf archive.tar directory/` 时，`tar` 会调用操作系统的目录遍历函数（类似于 `find` 命令）。
        * **遍历逻辑**：标准的文件系统遍历通常是**深度优先**的。它进入一个目录，立刻开始处理该目录下的子项，处理完所有子项后再转到下一个同级目录。
        * **流式写入**：`tar` 找到一个文件，就立刻把它追加到 `.tar` 文件的末尾。因此，压缩包内部的物理顺序就是它遍历文件系统的顺序。

    2. 结构对比

        如果你有如下目录结构：

        ```text
        top_a/
        ├── sub_a/
        │   └── file1
        └── sub_b/
        top_b/
        ```

        **`tar -t` 的典型输出顺序：**

        1. `top_a/` （父目录）
        2. `top_a/sub_a/` （进入第一个子目录）
        3. `top_a/sub_a/file1` （进入更深层）
        4. `top_a/sub_b/` （回溯并进入第二个子目录）
        5. `top_b/` （处理完 top_a，进入下一个顶层目录）

    3. 会有例外吗？

        虽然绝大多数情况下是上述顺序，但以下情况会改变顺序：

        * **手动指定参数顺序**：

            如果你手动输入：`tar -cvf archive.tar top_b top_a`，那么 `top_b` 及其所有内容会排在 `top_a` 之前。

        * **通过文件列表创建**：

            如果你使用 `find . -type f | sort | tar -cvf archive.tar -T -`，那么顺序将取决于 `sort` 的结果。

        * **追加文件 (`-r`)**：

            如果你在现有的 `tar` 包后追加文件，新加的文件永远出现在压缩包的最末尾，无论它在文件系统中的路径深度如何。

    4. 深度优先 vs 广度优先

        * **`tar` 采用的是深度优先（DFS）**：有利于流式处理，不需要在内存中缓存整个目录树。
        * **广度优先（BFS）**：即你提到的 `top_a, top_b, top_c...` 顺序。`tar` 默认**不采用**这种方式，因为这在创建大文件包时会频繁地在不同目录间切换，增加磁头寻道开销。

    **想进一步了解吗？**
    如果您是因为 `tar -t` 输出太乱，想要将其按照特定的深度或字母顺序重新排列显示，我可以为您提供相关的 `sort` 过滤技巧。需要我帮您写一个排序命令吗？
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

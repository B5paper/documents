# rsync note

## cache

* `rsync`本身不可实时同步目录，只能配合`inotifywait`实现实时同步

    (未验证)

    1. 使用 bash 实现

        ```bash
        #!/bin/bash

        # 定义源目录和目标服务器目录
        SOURCE_DIR="/path/to/source/dir/"
        TARGET_DIR="/path/to/target/dir/"
        REMOTE_USER="username"
        REMOTE_HOST="remote_server_ip"

        # 开始监控 inotifywait
        inotifywait -mrq --timefmt '%Y-%m-%d %H:%M:%S' --format '%T %w%f %e' \
          -e create,delete,modify,move,attrib \
          ${SOURCE_DIR} | while read time file event
        do
          echo "-----"
          echo "Event: $event"
          echo "File: $file"
          echo "Time: $time"

          # 触发 rsync 同步
          # 注意：这里使用了 --delete 选项，如果源端删除了文件，目标端也会删除。
          # 如果不希望这样，请移除 --delete 选项。
          rsync -avz --delete ${SOURCE_DIR} ${REMOTE_USER}@${REMOTE_HOST}:${TARGET_DIR}

          # 可选：输出同步完成时间
          echo "Sync completed at: $(date +'%Y-%m-%d %H:%M:%S')"
          echo "-----"
        done
        ```

        参数详解：

        inotifywait:

            -m: --monitor，持续监控，而不是触发一次就退出。

            -r: --recursive，递归监控目录。

            -q: --quiet，减少不必要的输出。

            --timefmt: 指定时间格式。

            --format: 指定事件输出格式。

            -e: 指定要监控的事件，最重要的一些事件包括：

                create (文件/目录创建)

                delete (文件/目录删除)

                modify (文件内容修改)

                move (文件/目录移动)

                attrib (文件属性变化，如权限、时间戳)

        rsync:

            -a: --archive，归档模式，保持所有文件属性，等同于 -rlptgoD。

            -v: --verbose，输出详细信息。

            -z: --compress，传输时压缩，节省带宽。

            --delete: 谨慎使用。让目标端和源端保持完全一致，源端删除的文件，目标端也会删除。

        运行：

        ```bash
        chmod +x live_sync.sh
        # 前台运行
        ./live_sync.sh

        # 或者放入后台运行，并将输出重定向到日志文件
        nohup ./live_sync.sh > live_sync.log 2>&1 &
        ```

    2. 使用`lsyncd`

    综合看来，`rsync`本身没法同步，自己写 bash 太麻烦，要不直接用`lsyncd`，要不直接用 nfs，或 sshfs。（有空可以调研下 nfs, sshfs, lsyncd 底层同步机制的区别）

* `rsync --exclude`

    `rsync --exclude=PATTERN`让 rsync 忽略匹配 PATTERN 的文件或目录，不进行同步。`PATTERN`可以是文件名、目录名（`PATTERN`需加`/`表示目录），或通配符模式（如`*.log`）。支持多次使用，排除多个不同规则。`PATTERN`作用于 src 目录。

    `rsync -av --exclude-from='exclude-list.txt' /source/ /destination/`

    `--exclude-from`表示从文件读取排除规则。

    example:

    `exclude-list.txt`:

    ```
    *.bak
    /logs/
    *.tmp
    ```

    排除隐藏文件：

    `rsync -av --exclude='.*' /source/ /destination/`

    优先级：

    `--include`的优先级高于`--exclude`，可组合使用（如先排除再包含部分文件）。

    * pattern 路径匹配细节

        * 相对路径：

            `PATTERN` 默认相对于 src 的路径。例如：

            `--exclude='temp/'` -> 排除`src/temp/`。

            `--exclude='/file.txt'` -> 排除`src/file.txt`（开头的 / 表示相对于 src 根）。

        * 绝对路径无效：

            若`PATTERN`是绝对路径（如`/home/user/file.txt`），rsync 会直接忽略（因为它始终相对于 src）。

* rsync 要求 local host 和 remote host 都安装有 rsync 才行。

* `rsync --delete`

    删除目标目录中多余的文件，使其与源目录完全一致。

* rsync 可以使用`--partial`断点续传

    `--append`用于给已经传输完成的文件只传输追加内容，比如 log。

    `--progress`可以显示进度。

    `-P`等同于`--partial --progress`

    `--timeout=30`可以自动重试，不需要人为重新开始。

    scp 无断点续传功能。

    `wget`可以使用`-c`断点续传。

* rsync 中的`.`等同于`./`，但是不等于`../<cur_dir>`

    `.`和`./`都是指当前文件夹下的所有内容。`../<cur_dir>`仅指当前文件夹。

* rsync 使用跳板机传输文件

    因为 rsync 底层依赖 ssh，所以 rsync 使用跳板机也需要修改 ssh 的配置。

    ```bash
    rsync -r -e 'ssh -J <jump_user>@<jump_addr>' <src_dir> <remote_user>@<remote_addr>:<dst_dir>
    ```

    如果既要改目标 ssh 端口，又要增加跳板机，那么可以：

    ```bash
    rsync -r -e 'ssh -p <remote_port> -J <jump_user>@<jump_addr>' <src_dir> <remote_user>@<remote_addr>:<dst_dir>
    ```

    但是不可以指定多个`-e`，如果有多个`-e`，那么以最后一个为准。

* rsync 默认使用的是 ssh 服务来传输文件，也有 873 端口的 rsync service，但是不常见。

    如果 remote host 的 ssh 使用的不是标准 22 端口，那么需要这样调用 rsync 来指定非标 ssh 端口：

    ```bash
    rsync -r -e 'ssh -p 4321' <src_dir> <user>@<remote_addr>:<dst_dir>
    ```

    其中`-e`用于指定 ssh 的配置。

* rsync

    `rsync -r local_dir <user>@<ip>:remote_parent_dir`

    这个命令会把本地的`local_dir`目录复制到远程的`remote_parent_dir/local_dir`目录。`-r`表示 recursively。如果 remote 目录中有 local 目录里没有的新文件/文件夹，那么 remote 目录中的内容不会被删除。如果使用`scp -R`复制目录，那么远程目录中的所有内容都会被替换。

    `rsync -r local_dir/ <user>@<ip>:remote_dir`

    这条命令会把本地的`local_dir`下的所有内容，复制到远程的`remote_dir`目录下。注意，这里的`/`代表目录中的内容，而不是目录本身。远程不会创建新文件夹。

    `rsync`默认不显示进度和已经复制的文件。

## note
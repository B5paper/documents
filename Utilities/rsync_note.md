# rsync note

## cache

* rsync 要求 local host 和 remote host 都安装有 rsync 才行。

* `rsync --delete`

    删除目标目录中多余的文件，使其与源目录完全一致。

* `rsync`默认开启了`-r`递归拷贝，虽然提供了`-r`参数，但大部分情况下无实际用处。

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
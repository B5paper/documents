# rsync note

## cache

* rsync

    `rsync -r local_dir <user>@<ip>:remote_parent_dir`

    这个命令会把本地的`local_dir`目录复制到远程的`remote_parent_dir/local_dir`目录。`-r`表示 recursively。如果 remote 目录中有 local 目录里没有的新文件/文件夹，那么 remote 目录中的内容不会被删除。如果使用`scp -R`复制目录，那么远程目录中的所有内容都会被替换。

    `rsync -r local_dir/ <user>@<ip>:remote_dir`

    这条命令会把本地的`local_dir`下的所有内容，复制到远程的`remote_dir`目录下。注意，这里的`/`代表目录中的内容，而不是目录本身。远程不会创建新文件夹。

    `rsync`默认不显示进度和已经复制的文件。

## note
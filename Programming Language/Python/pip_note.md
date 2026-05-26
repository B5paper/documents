# pip note

## cache

* `pip install --user <package>` 会把包安装到`~/.local/lib`下

    如果不指定`--user`，则会根据`which pip`找包的安装路径
* 手动指定 pip 的 cache 目录

    **方法：更改临时目录和缓存目录**

    这是最直接的解决办法。找一个空间较大的分区（比如 `/data` 或 `/mnt/your_name/`），手动指定目录：

    ```bash
    # 1. 创建一个新的临时文件夹和缓存文件夹
    mkdir -p /path/to/big_disk/pip_tmp
    mkdir -p /path/to/big_disk/pip_cache

    # 2. 使用环境变量并执行安装
    TMPDIR=/path/to/big_disk/pip_tmp \
    pip install --cache-dir=/path/to/big_disk/pip_cache torch torchvision
    ```

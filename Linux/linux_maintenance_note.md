# Linux Maintenance Note

这里主要记录 linux 相关的命令，配置，主题为运维。

这里不记录编程相关，操作系统原理相关。

## cache

* `rsync`中，src dir 的下面四种写法是等价的

    `.`, `./`, `*`, `./*`

* `rsync`中, `--progress`可以显示进度，`--partial`支持断点续传

    `-P`则表示同时 enable `--progress --partial`。

* grep 开启`-E`时，可以使用`|`匹配多个模式。

    `grep -E haha\|hehe msg.txt`

    `grep -E 'haha|hehe' msg.txt`

    bash 会将`|`默认解释为管道，如果希望 bash 将`|`解释为字符`|`，那么要么在之前加`\`，要么使用单引号`''`。
    
    注：

    1. 标准的正则表达式支持`|`，比如 python 的`re`模块。

    1. `|`的前后不能有空格，或者说，空格不会被忽略。

* `rsync -z`表示在传输过程中对要传输的文件进行压缩。如果传输过程文本文件比较多，可以使用`-z`大幅提高传输效率。

* `tee -a`表示在文件末尾追加

    如果文件不存在，则会创建文件。

    echo 本身会在行尾加`\n`，因此不需要额外考虑`\n`。

    ```bash
    echo 'heloo' | tee -a log.txt
    echo 'heloo' | tee -a log.txt
    cat log.txt
    ```

    output:

    ```
    heloo
    heloo
    heloo
    heloo
    ```

* grep 查看前后 n 行文本

    * 向前 n 行：`grep -B n`

    * 向后 n 行：`grep -A n`

    * 前后各 n 行：`grep -C n`

    其中 A 表示 after，B 表示 before，C 表示 context。

    example:

    `msg.txt`:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    haha
    hehesdf
    aaaaa
    bbb
    ```

    run: `grep -B 1 -A 2 345 msg.txt`

    output:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    haha
    hehesdf
    ```

    run : `grep -C 1 345 msg.txt`

    output:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    haha
    ```

* find 不输出没有权限的文件

    find 对没有权限的文件会输出类似

    ```
    ...
    find: ‘/proc/1188309/task/1188309/ns’: Permission denied
    find: ‘/proc/1188309/fd’: Permission denied
    find: ‘/proc/1188309/map_files’: Permission denied
    find: ‘/proc/1188309/fdinfo’: Permission denied
    find: ‘/proc/1188309/ns’: Permission denied
    ...
    ```

    的信息。这些信息其实都是 stderr。因此可以考虑过滤掉 stderr 的输出：

    `find <path> -name <pattern> 2>/dev/null`

    example:

    `find / -name hello 2>/dev/null`

    output:

    ```
    /home/hlc/miniconda3/pkgs/tk-8.6.14-h39e8969_0/lib/tk8.6/demos/hello
    /home/hlc/miniconda3/lib/tk8.6/demos/hello
    /home/hlc/miniconda3/envs/torch/lib/tk8.6/demos/hello
    /home/hlc/miniconda3/envs/vllm/lib/tk8.6/demos/hello
    /home/hlc/Documents/Projects/boost_1_87_0/tools/build/example/qt/qt4/hello
    /home/hlc/Documents/Projects/boost_1_87_0/tools/build/example/qt/qt3/hello
    /home/hlc/Documents/Projects/boost_1_87_0/tools/build/example/hello
    /home/hlc/Documents/Projects/chisel-tutorial/src/main/scala/hello
    /home/hlc/Documents/Projects/makefile_test/hello
    ```

* sudo 与环境变量的关系

    sudo 应该写在环境变量的前面。
    
    example:

    `sudo http_proxy=http://127.0.0.1:8822 https_proxy=http://127,0,0,1:8822 apt update`

* linux host name 相关

    * 显示当前的 host name: `hostname`

        example:

        run: `hostname`

        output: `hlc-VirtualBox`

    * 显示当前系统的基本信息：`hostnamectl`

        example:

        run: `hostnamectl`

        output:

        ```
         Static hostname: hlc-VirtualBox
               Icon name: computer-vm
                 Chassis: vm
              Machine ID: d3dcf00f11234838acfafd0a40493023
                 Boot ID: 5ce8f551956f4b14ab4c447a4a2ecbd0
          Virtualization: oracle
        Operating System: Ubuntu 22.04.4 LTS              
                  Kernel: Linux 6.8.0-52-generic
            Architecture: x86-64
         Hardware Vendor: innotek GmbH
          Hardware Model: VirtualBox
        ```

    * 修改 hostname `hostnamectl set-hostname <new-hostname>`

    * 可以通过修改`/etc/hostname`文件和`/etc/hosts`并重启系统来修改 hostname。

    * ubuntu 中，可以通过 settings -> about 修改 hostname。

    * 临时修改 hostname: `sudo hostname new-hostname` （未测试过）

## note
# qemu note

## cache

* 虚拟机 120G 磁盘不够用，150G 比较好

* qemu + gdb 调试 kernel module 代码

    1. 制作 qemu 虚拟机镜像，以 ubuntu 24.04 为例

        分配存储时，小于 30 GB 大概率没法用，所以我们直接选择 150 GB。

        cpu 核数和内存肯定越大越好。cpu 可以设置为 8 个，内存设置为 32768 MB。不够了再加。

        安装完成后，需要下载 linux 内核源码，编译内核：

        ```bash
        sudo apt update
        sudo apt install linux-source-6.8.0
        cd /usr/src
        sudo tar -xjvf linux-source-6.8.0.tar.bz2
        cd ./linux-source-6.8.0
        sudo apt install libncurses-dev 
        sudo make menuconfig
        ```

        选择 Save -> OK -> Exit

        `sudo vim .config`, 搜索`canonical`，将相关的`pem`字符串改成空字符串`""`。保存后退出。

        接下来编译内核：

        ```bash
        sudo apt install libelf-dev libssl-dev flex bison
        sudo make -j8
        sudo make modules_install
        ```

        将编译好的内核加入到 grub 中：

        ```bash
        sudo make install
        ```

        重启虚拟机，进入系统后执行`uname -r`，若输出为`6.8.12`，则说明内核安装成功。

        由于在编译内核时，`.config`文件中，debug 信息和符号表都是默认打开的，所以不需要额外设置，编译好的内核可以直接用于调试。

        此时 qemu 虚拟机的镜像制作成功。

    2. 在 host 上执行一遍和上一步一模一样的操作，复制一份完全相同的环境。

    3. 启动 qemu 虚拟机

        `qemu-system-x86_64 -accel kvm -m 8192 -smp 8 -hda ./ubuntu24.04.01.qcow2 -s -S`

        其中`-s`表示在`0.0.0.0:1234`端口开启 gdb server，`-S`表示在程序入口处停止。

    4. 在 host 上执行

        `gdb /usr/src/linux-source-6.8.0/vmlinux`

        `continue`

        此时 guest 会恢复运行。

    5. 在 guest vm 中`sudo insmod hello.ko`

        然后执行`sudo cat /sys/module/hello/sections/.text`，得到一个输出：`0xffffffffc05ef000`

        在 host gdb 中`add-symbol-file /home/test/Documents/Projects/gdb_module_test/hello.ko 0xffffffffc05ef000`

    6. 添加断点

        `b /home/test/Documents/Projects/gdb_module_test/hello.c:12`

        运行`continue`

    7. 触发断点

        在 guest vm 里，`sudo rmmod hello`。

        此时 host gdb 里会自动 hit 断点，并显示相关信息。

        至此说明 qemu + gdb 调试内核成功。

* ubuntu 24.04 下使用 apt 安装的 virt-manager （会自动安装 qemu），可以使用 QXL video card 成功启动 ubuntu 24.04 guest。

    版本号：

    `qemu-system-x86_64` -> `8.2.2 (Debian 1:8.2.2+ds-0ubuntu1)`

    `virt-manager` -> `4.1.0`

* qemu command line and net card

    使用`qemu-system-x86_64`启动的虚拟机，网卡是`ens3`，ip 是`10.0.2.15`, netmask `255.255.255.0`，但是 mac 地址和 host 不一样。

    虚拟机的网卡可以访问 host，也可以访问 internet。由于 guest 的 ip 和 host 相同，所以无法得知 host 是否能 ping 到 guest。

    尝试更改 guest ip 为`10.0.2.16/24`后，guest 失去所有网络访问能力。重新将 ip 改为`10.0.2.15/24`后，也不能恢复访问。

    此时执行`route -n`后发现路由表是空的。重启系统，网络恢复正常后，再次执行`route -n`可以看到路由表有 3 条 entry。说明修改 ip 后网络不通很有可能是路由表的问题。

    正常网络情况下，路由表中的`0.0.0.0`的 gateway 是`10.0.2.2`，在 guest 和 host 上均能 ping 通，但是在 host 上执行`ifconfig`和`ifconfig -a`均看不到这个网卡。

    不清楚这个 route 是什么机制。

## note
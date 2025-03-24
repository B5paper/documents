# Linux kernel note

此笔记与本目录中 linux maintaince，linux driver 区别开。linux maintaince 只有

## cache

* linux 删除自己编译的 kernel

    比如自己编译了`6.2.16`的内核，并且已经成功安装，现在要删除这个内核。

    1. 首先切换到其他内核上，确保当前正在用的内核不是要删除的内核

        check: `uname -r`, output: `6.8.0-52-generic`

    1. 到`/boot`目录下，输入`ls`

        output:

        ```
        config-5.19.0-50-generic      memtest86+_multiboot.bin
        config-6.2.16                 System.map-5.19.0-50-generic
        config-6.2.16.old             System.map-6.2.16
        config-6.8.0-52-generic       System.map-6.2.16.old
        efi                           System.map-6.8.0-52-generic
        grub                          vmlinuz
        initrd.img                    vmlinuz-5.19.0-50-generic
        initrd.img-5.19.0-50-generic  vmlinuz-6.2.16
        initrd.img-6.2.16             vmlinuz-6.2.16.old
        initrd.img-6.8.0-52-generic   vmlinuz-6.8.0-52-generic
        memtest86+.bin                vmlinuz.old
        memtest86+.elf
        ```

        删除三个文件：

        `rm vmlinuz-6.2.16 initrd.img-6.2.16 config-6.2.16 System.map-6.2.16`

    1. 进入`/lib/modules`，执行`ls`

        output:

        ```
        5.19.0-50-generic  6.5.0-44-generic  6.8.0-47-generic  6.8.0-52-generic
        6.2.16             6.8.0-40-generic  6.8.0-49-generic
        6.5.0-18-generic   6.8.0-45-generic  6.8.0-51-generic
        ```

        删除`6.2.16`文件夹：
        
        `rm -rf 6.2.16`

    1. 更新`grub`：`sudo update-grub`

        此时`grub`会自动重新搜索`/boot`中的可用内核，删除不存在的内核对应的菜单入口

    1. （可选）进入`/usr/src`，执行`ls`：

        output:

        ```
        linux-headers-5.19.0-50-generic   linux-source-6.2.0
        linux-headers-6.8.0-52-generic    linux-source-6.2.0.tar.bz2
        linux-hwe-5.19-headers-5.19.0-50  nvidia-530.30.02
        linux-hwe-6.8-headers-6.8.0-52
        ```

        如果要清空编译的结果，可以执行：

        ```bash
        cd linux-source-6.2.0
        make clean
        ```

        如果要彻底删除所有源代码，可以直接删除源代码目录：

        `rm -rf linux-source-6.2.0`

    ref: <https://www.cyberciti.biz/faq/debian-redhat-linux-delete-kernel-command/>

* linux kernel

    常见的 kernel 分两种，一种是 Microkernels，所有的函数都通过 interface 与操作系统进行交互。

    另外一种是 Monolithic Kernels，操作系统内核实现了大部分的功能，包括文件系统之类的。

    monolithic kernels 的效率比 microkernels 要高。linux 采用的是 monolithic kernels。
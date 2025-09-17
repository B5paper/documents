# qemu note

## cache

* `-append 'console=ttyS0'`不能单独使用，必须和`-kernel`同时使用。

    否则 qemu 会报错：

    ```
    qemu-system-x86_64: -append only allowed with -kernel option
    ```

    典型使用场景：

    ```bash
    qemu-system-x86_64 \
      -kernel /path/to/vmlinuz \
      -initrd /path/to/initrd \
      -append "root=/dev/vda1 ro console=ttyS0" \
      -nographic
    ```

    这里的 -append "console=ttyS0" 和 -nographic 选项配合，将虚拟机的控制台重定向到当前终端（模拟了一个串口）。

* qemu 中，先按`Ctrl + A`，再按`C`，可以切换到 qemu 后台（即 qemu monitor），如下：

    ```
    SeaBIOS (version 1.15.0-1)


    iPXE (https://ipxe.org) 00:03.0 CA00 PCI2.10 PnP PMM+BFF8B420+BFECB420 CA00
                                                                                   


    Booting from Hard Disk...
    QEMU 6.2.0 monitor - type 'help' for more information
    (qemu) 
    ```

    前面的`(qemu)` prompt 标志着已经进入了 qemu 后台，此时可以执行一些后台指令，比如`info registers`。

    qemu monitor 不是虚拟机的一部分，是 qemu 监视和管理虚拟机的后台程序。

* qemu 中`-serial mon:stdio`

    `-serial` 选项用来指定串口输出的重定向目标，比如 file:xxx、pty、stdio 等。

    `mon` 表示 QEMU Monitor。

    `stdio` 表示把输入输出绑定到 当前终端的标准输入输出。

    `mon:stdio`表示将虚拟机的串口和 qemu monitor 的输入输出，都绑定到`stdio`。

    如果不写`mon`，只写`stdio`，那么按快捷键`Ctrl + A`，`C`，将不会出现 qemu monitor。（未验证）

    如果只写`mon`，不写`stdio`，那么输出如下：

    ```
    QEMU 6.2.0 monitor - type 'help' for more information
    (qemu) qemu-system-x86_64: -serial mon: 'mon' is not a valid char driver
    qemu-system-x86_64: -serial mon: could not connect serial device to character backend 'mon'
    ```

    如果使用`-nographic -serial stdio -monitor none`启动 qemu，那么无法使用 ssh 登陆。不清楚原因。

    这个命令的意思是`-serial stdio -monitor stdio`，但是不能这么写，会报错，因为`stdio`是一种独占资源，不能被分配两次。`-serial mon:stdio`则是多路复用形式的重定向，所以没问题。

* qemu 无图形界面启动

    * 方案一
    
        `-nographic`

    * 方案二（未验证）

        `-display none`

* 强制关闭 qemu

    （未验证）

    强制退出：要强制终止 QEMU 进程，回到终端，可以先按下 Ctrl + A，松开后再按 X。

    * Ctrl + A 然后 X：立即终止 QEMU（相当于杀进程）。

    * Ctrl + A 然后 C：可以在 QEMU 监视器控制台和客户机串口控制台之间切换（用于更高级的操作）。

* qemu edu: `-device edu`

* qemu 虚拟机将内部的 22 端口映射到外部的 2222 端口

    `qemu-system-x86_64 -accel kvm -m 8192 -smp 8 -hda ./ccc.qcow2 -netdev user,id=mynet0,hostfwd=tcp::2222-:22 -device virtio-net-pci,netdev=mynet0`

* qemu 使用`-kernel`指定内核启动虚拟机

    `qemu-system-x86_64 -accel kvm -m 4096 -smp 4 -hda ./ubuntu22.04-for-pci.qcow2 -kernel /boot/vmlinuz-6.2.16 -initrd /boot/initrd.img-6.2.16 -append "root=/dev/sda3"`

    说明：

    * `-hda ./ubuntu22.04-for-pci.qcow2`表示加入磁盘，但是进入操作系统后，具体的磁盘设备不一定是`/dev/hda`，也有可能是`/dev/sda`，对应的分区为`/dev/sda0`，`/dev/sda1`，`/dev/sda2`，...。

    * `-kernel /boot/vmlinuz-6.2.16`表示使用 host 上的`/boot/vmlinuz-6.2.16`内核。
    
        看网上介绍说这个内核是使用 gzip 压缩过的，未压缩的版本应该是 vmlinux。不清楚具体是怎么压缩的。

        这个参数也可以替换为
        
        `-kernel /usr/src/linux-source-6.2.0/arch/x86/boot/bzImage`
        
        或者
        
        `-kernel /usr/src/linux-source-6.2.0/arch/x86/boot/compressed/vmlinux.bin`

    * `-initrd /boot/initrd.img-6.2.16`是可选参数，可以不写。

        据说是因为`-hda`中有了 initrd，所以才不需要指定`-initrd`，具体情况不太清楚。

    * `-append "root=/dev/sda3"`：`-append`表示添加 linux kernel command line，`root=`表示将磁盘的哪个分区挂载到`/`目录上。

    * 6.2.16 的 kernel 在 guest 中启动后，systemd 会读取`/etc`，`/usr/lib`等目录下的配置，继续 load module，比如`autofs4`等。因此 guest 的`/lib/modules`中也必须有编译好的`6.2.16`目录，用来加载这些额外的 module。

        否则 systemd 会报错找不到 module，guest 系统会进入 emergency 模式。

* 使用`qemu-system-x86_64 -accel kvm -m 4096 -smp 4 -hda ./xxx.qcow2 -kernel /boot/vmlinuz-6.8.0-52-generic -append "root=/dev/sda3"`可以启动系统，但是使用自己编译的内核`vmlinuz-6.2.16`无法启动系统

    使用自己编译的`vmlinuz-6.2.16`内核时，遇到的问题有：

    * `lp`, `ppdev`, `parport_pc` module 加载失败

        经搜索发现其在`/etc/modules-load.d/cups-filters.conf`中，将这三行注释掉即可

    * `msr` module 加载失败

        经搜索发现其在`/usr/lib/modules-load.d/fwupd-msr.conf`中，将其注释掉即可。

        以上这些模块都是由`modules-load.d` service 去启动的，可以使用`man modules-load.d`查看更多帮助。与此相关的 service 还有`systemd-modules-load`, `systemd-modules-load.service,`。

    * fail to mount `/boot/efi`

        不清空怎么解决。理论上使用自定义的`-kernel`和`-initrd`，就不再需要 efi 了，但是进入系统后仍要加载 efi，有可能是读到了`/etc`中的某些配置，所以才在内核启动完后，继续 modprob 一些 module。

* virtual box 中，在 libvirt 的 pool 中创建 qcow2 disk，如果 pool 的目录在 windows 的 shared folder 中，那么会在初始化时实际分配指定大小的磁盘空间。

    此时使用`qemu-img convert`进行转换，那么这个 qcow2 文件会被缩小到 200 KB 左右，并且不影响使用。在转换期间，磁盘占用并未有显著增加，说明转换过程中的 tmp 文件，可能和磁盘中的实际文件占用大小有关，并不是转换 150 GB 的磁盘就需要 150 GB 的 temp 空间。

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
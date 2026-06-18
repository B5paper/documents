# qemu note

## cache

* 详细讲讲 qemu 中的 tap

    在 QEMU/KVM 虚拟化环境中，**TAP** 是一种非常高效且常用的**网络虚拟化技术**。

    简单来说，TAP 是内核中的一个**虚拟网络设备**（属于 TUN/TAP 驱动的一部分）。不同于普通的物理网卡，TAP 设备没有真实的物理硬件对应，它完全是用软件模拟出来的。

    为了让你彻底搞懂它，我们从工作原理、数据流向、配置方法以及与其对比的 NAT/Bridge 模式来详细聊聊。

    ---

    ## 一、 TAP 的核心工作原理

    要理解 TAP，最直观的方式是把它想象成**一根虚拟网线**。这根网线有两端：

    * **一端（字符设备端 `/dev/net/tun`）：** 连着用户空间的程序（比如 QEMU）。QEMU 可以通过读写这个设备，直接把数据包塞进内核，或者从内核抓取数据包。
    * **一端（网络接口端 `tapX`）：** 连着宿主机的内核网络协议栈。在宿主机看来，它就像一块插在电脑上的“普通网卡”（比如 `eth0`），拥有 MAC 地址，也可以配置 IP。

    ### TAP 与 TUN 的区别（插播一个小知识）

    两者都由同一个内核驱动管理，但处理的数据层级不同：

    * **TUN（Network Tunnel）：** 工作在 **L3（网络层）**，只能处理 IP 数据包。常用于 VPN。
    * **TAP（Network Tap）：** 工作在 **L2（数据链路层）**，可以处理完整的 **Ethernet 帧**（包含 MAC 地址、ARP 请求等）。因为虚拟机需要像真实机器一样发送二层广播，所以 QEMU 必须使用 TAP。

    ---

    ## 二、 TAP 的典型使用场景：桥接模式 (Bridge)

    TAP 最经典的用法是配合宿主机的网桥（Bridge）使用。在这种模式下，虚拟机就像是直接插在宿主机所在的物理交换机上一样，拥有网络内独立的 IP 地址。

    ### 数据流向示意图

    当虚拟机（Guest）想要发送一个网络数据包到外网时，流量是这样走的：

    ```text
    +-----------------------------------------------------------------------+
    | 宿主机 (Host)                                                         |
    |                                                                       |
    |  +------------------+             +--------------------------------+  |
    |  | 虚拟机 (Guest)    |             | 宿主机内核 (Host Kernel)        |  |
    |  |                  |             |                                |  |
    |  |  +------------+  |             |  +----------+    +----------+  |  |
    |  |  | 虚拟网卡    |  |  (VirtIO)   |  | TAP 设备 |    | 网桥     |  |  |
    |  |  | vnet0      |==|=============|==| tap0     |----| br0      |  |  |
    |  |  +------------+  |             |  +----------+    +----------+  |  |
    |  +------------------+             |                       |        |  |
    |                                   |                  +----------+  |  |
    |                                   |                  | 物理网卡  |  |  |
    |                                   |                  | eth0     |  |  |
    |                                   +------------------+----------+--+  |
    +-------------------------------------------------------------------|---+
                                                                        |
                                                                    ( 物理交换机 )

    ```

    1. **虚拟机**内的应用发送数据，通过虚拟机内核到达虚拟网卡（如 `vnet0`）。
    2. QEMU 进程通过 **VirtIO** 等半虚拟化驱动，将这个二层以太网帧读取出来。
    3. QEMU 将数据写入宿主机的 `/dev/net/tun` 字符设备。
    4. 内核将该数据包装入宿主机侧的 **`tap0` 接口**。
    5. 因为 `tap0` 已经绑定到了宿主机的**网桥 `br0**` 上，网桥会根据 MAC 地址表，将数据转发给同样的绑定在 `br0` 上的**物理网卡 `eth0**`。
    6. 数据最终流向物理网络。

    ---

    ## 三、 如何在 QEMU 中配置和使用 TAP

    在 QEMU 中使用 TAP 通常需要两个步骤：在宿主机创建 TAP 设备并桥接，然后启动 QEMU。

    ### 步骤 1：在宿主机创建网桥和 TAP（以 Linux bridge 为例）

    ```bash
    # 1. 创建网桥 br0
    sudo ip link add name br0 type bridge
    sudo ip link set br0 up

    # 2. 将物理网卡 eth0 绑定到网桥（注意：这会导致 eth0 上的 IP 失效，需要把 IP 配置到 br0 上）
    sudo ip link set eth0 master br0

    # 3. 创建一个名为 tap0 的 TAP 设备
    sudo ip tuntap add dev tap0 mode tap

    # 4. 将 tap0 绑定到网桥 br0
    sudo ip link set tap0 master br0
    sudo ip link set tap0 up

    ```

    ### 步骤 2：启动 QEMU 虚拟机

    使用 `-netdev tap` 参数将刚才创建的 `tap0` 传递给虚拟机：

    ```bash
    qemu-system-x86_64 \
      -enable-kvm \
      -m 2G \
      -drive file=ubuntu.qcow2,if=virtio \
      -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
      -device virtio-net-pci,netdev=net0,mac=52:54:00:12:34:56

    ```

    > **参数详解：**
    > * `-netdev tap,id=net0...`：定义一个宿主机侧的网络后端，类型为 `tap`，名字叫 `net0`。
    > * `ifname=tap0`：指定使用宿主机上已经存在的 `tap0` 设备。
    > * `script=no,downscript=no`：禁用 QEMU 默认的启动/关闭脚本。如果不加这两个参数，QEMU 会尝试在启动时自动去执行 `/etc/qemu-ifup` 脚本来帮你创建 TAP，通常手动管理网络时我们倾向于关闭它。
    > * `-device virtio-net-pci,netdev=net0`：在虚拟机里创建一个 VirtIO 类型的虚拟网卡，并把它和后端的 `net0`（即 `tap0`）连起来。
    > 
    > 

    ---

    ## 四、 TAP 模式的优缺点分析

    为了让你更清楚什么时候该用 TAP，我们拿它和 QEMU 默认的 **User 模式 (SLIRP)** 进行对比：

    | 特性 | User 模式 (SLIRP) | TAP + Bridge 模式 |
    | --- | --- | --- |
    | **工作原理** | 完全由 QEMU 在用户态模拟一个网络栈和 NAT | 利用内核 TAP 设备直接和宿主机链路层打通 |
    | **性能** | **较差**（大量用户态与内核态的上下文切换） | **极高**（接近物理网络吞吐量，支持 vhost-net 硬件加速） |
    | **IP 分配** | 虚拟机获得内部私有 IP (如 10.0.2.15) | 虚拟机获得局域网独立 IP (与宿主机同网段) |
    | **外网访问** | 虚拟机可访问外网，但外网**无法直接访问**虚拟机 | **双向互通**，虚拟机像局域网里的一台真实电脑 |
    | **权限要求** | 不需要 root 权限 | 需要 root 权限来创建和操作内核网络设备 |
    | **配置复杂度** | 开箱即用，无需任何配置 | 配置较为复杂，需要管理宿主机网络 |

    ---

    ## 五、 进阶：TAP 的性能优化 (vhost-net)

    如果你在生产环境（比如 KVM + Libvirt）中使用 TAP，你经常会看到 `vhost=on` 这个参数。

    在默认的 TAP 模式下，数据包从 TAP 到虚拟机，需要穿过 **内核 (TAP) -> 用户态 (QEMU) -> 内核 (KVM) -> 虚拟机**。

    为了减少 QEMU 这一层的中转开销，Linux 引入了 **vhost-net** 技术。它是一个内核模块，允许内核中的 TAP 设备直接与 KVM 模块进行数据交换，完全绕过用户态的 QEMU 进程。

    **启用 vhost 后的 QEMU 命令：**

    ```bash
    -netdev tap,id=net0,ifname=tap0,script=no,downscript=no,vhost=on

    ```

    开启 `vhost=on` 后，虚拟机的网络吞吐量和延迟都会得到大幅改善，几乎可以压榨满物理网卡的带宽。
* 在 qemu 虚拟机中，我希望 03:00.0, 04:00.0, 05:00.0, 06:00.0 走 iommu passthrough，11:00.0 ~ 14:00.0 走正常 iommu，该如何配置jjjj

    ```
    02:03.0 PCI bridge: Texas Instruments XIO3130 PCI Express Switch (Downstream) (rev 01)
    03:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]
    04:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]
    05:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]
    06:00.0 Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]
    10:00.0 PCI bridge: Red Hat, Inc. QEMU PCIe Root port
    10:01.0 PCI bridge: Red Hat, Inc. QEMU PCIe Root port
    10:02.0 PCI bridge: Red Hat, Inc. QEMU PCIe Root port
    10:03.0 PCI bridge: Red Hat, Inc. QEMU PCIe Root port
    11:00.0 Processing accelerators: Device 205d:1100 (rev 01)
    12:00.0 Processing accelerators: Device 205d:1100 (rev 01)
    13:00.0 Processing accelerators: Device 205d:1100 (rev 01)
    14:00.0 Processing accelerators: Device 205d:1100 (rev 01)
    ```

    你这个需求可以拆成一句话来理解：

    > **同一个 QEMU 虚拟机里，一部分 PCIe 设备做“直通（VFIO passthrough，使用宿主 IOMMU）”，另一部分设备仍然走“虚拟 IOMMU（vIOMMU）/普通虚拟化路径”。**

    三、QEMU 命令行的**关键配置**

    1. **启用 vIOMMU（必须）**

        否则 11:00.0–14:00.0 在 guest 里就“没有 IOMMU”

        ```bash
        -device intel-iommu,intremap=on,caching-mode=on
        ```

        （AMD 平台用 `amd-iommu`）

    2. **VFIO 直通设备：关闭 iommu_platform**

        ```bash
        -device vfio-pci,host=03:00.0,iommu_platform=off
        -device vfio-pci,host=04:00.0,iommu_platform=off
        -device vfio-pci,host=05:00.0,iommu_platform=off
        -device vfio-pci,host=06:00.0,iommu_platform=off
        ```

        解释：

        * `iommu_platform=off`

          * **设备 DMA 直接走 host IOMMU**
          * 不走 guest vIOMMU
          * 这是 *bare-metal 风格直通*（你要的）

    3. 普通 / 虚拟设备：默认走 vIOMMU

        例如：

        ```bash
        -device virtio-net-pci,netdev=net0
        -device virtio-blk-pci,drive=vd0
        ```

        或者你那个：

        ```text
        11:00.0 ~ 14:00.0  Processing accelerators
        ```

        如果它们是：

        * virtio
        * emulated PCI
        * mediated device (mdev)

        👉 **什么都不用加**
        👉 它们会自动挂到 `intel-iommu`

    四、Guest OS 内部的表现（你应该看到什么）

    1. IOMMU 存在

        ```bash
        dmesg | grep -i iommu
        ```

        ```text
        DMAR: IOMMU enabled
        ```

    2. VFIO 设备 **不受 guest IOMMU 约束**

        ```bash
        lspci -vvv -s 03:00.0
        ```

        * 不会看到 guest IOMMU fault
        * DMA 地址是 guest 物理地址 → host IOVA

    3. 11:00.0–14:00.0 受 guest IOMMU 约束

        ```bash
        ls /sys/kernel/iommu_groups/
        ```

        能看到这些设备被分组

    **最小可用示例（总结版）**

    ```bash
    qemu-system-x86_64 \
      -enable-kvm \
      -machine q35,accel=kvm \
      -device intel-iommu,intremap=on \
      \
      -device vfio-pci,host=03:00.0,iommu_platform=off \
      -device vfio-pci,host=04:00.0,iommu_platform=off \
      -device vfio-pci,host=05:00.0,iommu_platform=off \
      -device vfio-pci,host=06:00.0,iommu_platform=off \
      \
      -device virtio-blk-pci,drive=vd0 \
      -device virtio-net-pci,netdev=net0
    ```

    注：

    1. 这里面的参数都没有实际测试过
* qemu edu dev spec: <https://www.qemu.org/docs/master/specs/edu.html>

* `qemu-img`的常见用法

    * 创建新的虚拟机磁盘

        `qemu-img create -f qcow2 vm_disk.qcow2 20G`

    * 转换磁盘格式

        `qemu-img convert -f vmdk -O qcow2 input.vmdk output.qcow2`

    * 检查磁盘信息

        `qemu-img info vm_disk.qcow2`

    * 快照管理

        `qemu-img snapshot -c snapshot1 disk.qcow2`

        QCOW2 使用 增量存储，快照仅记录变化部分。通过 元数据链 管理多个快照版本。

    * 镜像检查与修复

        `qemu-img check disk.qcow2`

        检查元数据一致性（如 QCOW2 的 L1/L2 表）。检测数据块是否损坏（可选修复）。

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

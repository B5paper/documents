# qemu note

## cache

* qemu-system-x86_64: -net tap,ifname=tap0,script=no,downscript=no: could not configure /dev/net/tun (tap0): Operation not permitted

    这是什么问题？

    这个错误表明 QEMU 没有权限访问 TUN/TAP 设备来创建虚拟网络接口 `tap0`。主要有以下几个原因和解决方案：

    ## 主要原因

    ### 1. **权限不足**
    - 当前用户不在 `tun` 或 `kvm` 组
    - 没有 `/dev/net/tun` 设备的读写权限

    ### 2. **设备不存在**
    - `/dev/net/tun` 设备没有创建
    - `tun` 内核模块未加载

    ## 解决方案

    ### 方案一：使用 sudo 运行 QEMU（最简单）
    ```bash
    sudo qemu-system-x86_64 ...（你的参数）...
    ```

    ### 方案二：将用户添加到相关组（推荐）
    ```bash
    # 检查当前用户所属组
    groups

    # 将用户添加到必要组
    sudo usermod -aG kvm $USER
    sudo usermod -aG tun $USER  # 如果 tun 组存在

    # 或者创建 tun 组并添加用户
    sudo groupadd tun
    sudo usermod -aG tun $USER

    # 重新登录或重启生效
    ```

    ### 方案三：配置设备权限
    ```bash
    # 检查 tun 设备权限
    ls -l /dev/net/tun

    # 设置永久权限（创建 udev 规则）
    sudo tee /etc/udev/rules.d/10-qemu.rules << EOF
    KERNEL=="tun", GROUP="kvm", MODE="0660"
    EOF

    # 重新加载 udev 规则
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```

    ### 方案四：检查内核模块
    ```bash
    # 检查 tun 模块是否加载
    lsmod | grep tun

    # 如果未加载，手动加载
    sudo modprobe tun

    # 确保开机自动加载
    echo "tun" | sudo tee -a /etc/modules-load.d/tun.conf
    ```

    ### 方案五：使用 libvirt 管理网络
    ```bash
    # 安装 libvirt 和相关工具
    sudo apt install libvirt-daemon-system libvirt-clients bridge-utils  # Debian/Ubuntu
    sudo dnf install libvirt bridge-utils  # Fedora/RHEL

    # 启动服务
    sudo systemctl enable --now libvirtd

    # 将用户添加到 libvirt 组
    sudo usermod -aG libvirt $USER
    ```

    ## 验证步骤

    1. **检查权限配置后：**
    ```bash
    # 退出当前会话重新登录
    logout
    # 重新登录

    # 验证用户组
    groups | grep -E "(kvm|tun|libvirt)"
    ```

    2. **测试 TUN/TAP 设备：**
    ```bash
    sudo ip tuntap add name tap0 mode tap user $USER
    sudo ip link delete tap0
    ```

    ## 临时解决方案（不推荐长期使用）

    如果你只是临时测试，可以使用：
    ```bash
    # 临时修改权限
    sudo chmod 0666 /dev/net/tun

    # 或者使用 capabilities
    sudo setcap cap_net_admin+ep $(which qemu-system-x86_64)
    ```

    ## 建议
    - **推荐使用方案二**（添加用户组），这是最安全持久的方法
    - 如果你是使用虚拟机管理器（如 virt-manager），使用方案五（libvirt）更方便
    - 确保在修改组后重新登录或重启才能生效

    选择适合你使用场景的解决方案即可解决此问题。

* 使用 sudo 运行后，报错如下：

    qemu-system-x86_64: -net tap,ifname=tap0,script=no,downscript=no: could not configure /dev/net/tun (tap0): Device or resource busy

    这个错误表示 `tap0` 设备已经被占用或存在。以下是解决方案：

    ## 原因分析
    1. **tap0 接口已存在**（最常见）
    2. **之前的 QEMU 进程未完全退出**
    3. **其他进程正在使用该接口**

    ## 解决方案

    ### 方案一：删除已存在的 tap0 接口
    ```bash
    # 检查 tap0 是否存在
    ip link show tap0

    # 删除 tap0 接口
    sudo ip link delete tap0

    # 或者使用 tunctl（如果已安装）
    sudo tunctl -d tap0
    ```

    ### 方案二：使用不同的接口名
    ```bash
    # 使用 tap1 或其他名称
    qemu-system-x86_64 -net tap,ifname=tap1,script=no,downscript=no ...

    # 或者让 QEMU 自动选择
    qemu-system-x86_64 -net tap,script=no,downscript=no ...
    ```

    ### 方案三：清理所有虚拟接口
    ```bash
    # 列出所有 TAP 接口
    ip link show type tap

    # 删除所有 TAP 接口
    for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep ^tap); do
        sudo ip link delete $iface
    done
    ```

    ### 方案四：检查并终止占用进程
    ```bash
    # 查看哪个进程在使用 tap0
    sudo lsof /dev/net/tun
    sudo fuser -v /dev/net/tun

    # 查看所有 QEMU 进程
    ps aux | grep qemu

    # 强制终止所有 QEMU 进程
    sudo pkill -9 qemu-system-x86_64
    ```

    ### 方案五：重启网络服务
    ```bash
    # 重启网络管理服务（谨慎使用）
    sudo systemctl restart NetworkManager

    # 或者重启网络命名空间
    sudo ip netns delete qemu  # 如果有的话
    ```

    ## 预防措施

    ### 1. **使用自动接口管理**
    ```bash
    # QEMU 会自动创建和删除接口
    -netdev tap,id=net0 -device e1000,netdev=net0
    ```

    ### 2. **使用脚本管理**
    创建一个启动脚本：
    ```bash
    #!/bin/bash
    # cleanup.sh
    sudo ip link delete tap0 2>/dev/null
    sudo ip tuntap add name tap0 mode tap user $(whoami)
    sudo ip link set tap0 up

    # 然后启动 QEMU
    qemu-system-x86_64 -net tap,ifname=tap0,script=no,downscript=no ...
    ```

    ### 3. **使用 libvirt 管理网络**
    ```bash
    # 创建虚拟网络
    sudo virsh net-define /etc/libvirt/qemu/networks/default.xml
    sudo virsh net-start default
    sudo virsh net-autostart default
    ```

    ### 4. **每次使用前清理**
    ```bash
    # 在启动 QEMU 前运行
    sudo ip link delete tap0 2>/dev/null; true
    ```

    ## 快速诊断命令
    ```bash
    # 1. 检查接口状态
    ip link show | grep tap

    # 2. 检查 TUN/TAP 设备
    ls -la /dev/net/tun

    # 3. 检查网络命名空间
    sudo ip netns list

    # 4. 检查进程
    sudo netstat -tulpn | grep tap
    ```

    ## 推荐做法
    **最简单的解决方案**是每次启动 QEMU 前运行：
    ```bash
    sudo ip link delete tap0 2>/dev/null; sudo qemu-system-x86_64 ...
    ```

    或者改用新的接口名：
    ```bash
    sudo qemu-system-x86_64 -net tap,ifname=tap$(date +%s),script=no,downscript=no ...
    ```

    选择适合你使用场景的方法即可解决此问题。

* Booting from Hard Disk...
    [    3.634164] Kernel panic - not syncing: VFS: Unable to mount root fs on unkn)
    [    3.637074] CPU: 9 UID: 0 PID: 1 Comm: swapper/0 Not tainted 6.14.0-37-generu
    [    3.639816] Hardware name: QEMU Standard PC (Q35 + ICH9, 2009), BIOS rel-1.14
    [    3.643232] Call Trace:
    [    3.644039]  <TASK>
    [    3.644784]  dump_stack_lvl+0x27/0xa0
    [    3.645986]  dump_stack+0x10/0x20
    [    3.647070]  panic+0x36f/0x400
    [    3.648086]  mount_root_generic+0x1a5/0x370
    [    3.649430]  mount_root+0x98/0x100
    [    3.650534]  ? __pfx_kernel_init+0x10/0x10
    [    3.651831]  prepare_namespace+0x6c/0x2f0
    [    3.653101]  kernel_init_freeable+0x1c8/0x210
    [    3.654513]  kernel_init+0x1b/0x200
    [    3.655626]  ret_from_fork+0x47/0x70
    [    3.656781]  ? __pfx_kernel_init+0x10/0x10
    [    3.658066]  ret_from_fork_asm+0x1a/0x30
    [    3.659358]  </TASK>
    [    3.667114] Kernel Offset: disabled
    [    3.668309] ---[ end Kernel panic - not syncing: VFS: Unable to mount root f

    qemu 报这个错是为什么？

    这个错误表示Linux内核无法挂载根文件系统(rootfs)，常见原因和解决方案如下：

    ## 主要可能原因：

    ### 1. **initramfs问题**
    ```bash
    # 最常见的解决方法：更新initramfs
    sudo update-initramfs -u -k all
    # 或者重建
    sudo mkinitcpio -P  # 适用于Arch
    ```

    ### 2. **引导参数问题**
    检查GRUB配置中的root参数：
    ```bash
    # 检查当前内核参数
    cat /proc/cmdline
    # 查看GRUB配置
    sudo grep GRUB_CMDLINE_LINUX /etc/default/grub
    ```

    确保包含正确的：
    - `root=UUID=...` 或 `root=/dev/sdXY`
    - `rootfstype=ext4`（根据实际文件系统）

    ### 3. **文件系统损坏**
    ```bash
    # 使用Live CD/USB启动后检查
    fsck /dev/sdXY -y
    ```

    ### 4. **QEMU配置问题**
    检查启动参数：
    ```bash
    # 确保正确指定了磁盘镜像和内核
    qemu-system-x86_64 \
      -kernel vmlinuz \
      -initrd initrd.img \
      -drive file=disk.img,format=raw \
      -append "root=/dev/sda1 console=ttyS0"
    ```

    ### 5. **内核模块缺失**
    缺少必要的文件系统驱动或存储控制器驱动：
    ```bash
    # 检查initramfs中是否包含需要的模块
    lsinitramfs /boot/initrd.img-* | grep -E "ext4|xfs|scsi|virtio"
    ```

    快速诊断步骤：

    1. 检查内核版本兼容性

        ```bash
        uname -r
        ```

    2. **检查/boot目录内容**

        ```bash
        ls -la /boot/
        ```

    3. **检查磁盘分区**

        ```bash
        lsblk
        blkid
        ```

    4. **检查dmesg完整输出**

        在QEMU中查看完整的启动日志

    针对QEMU的特别建议：

    ```bash
    # 1. 增加调试信息
    qemu-system-x86_64 -append "root=/dev/sda1 debug earlyprintk=vga"

    # 2. 确认磁盘驱动类型
    # 如果是virtio磁盘
    -append "root=/dev/vda1"

    # 3. 使用串口控制台查看详细错误
    -serial stdio -monitor none
    ```

    最可能的原因是initramfs没有正确生成或者GRUB配置中的root参数不正确。建议先从更新initramfs开始，然后检查GRUB配置。

* qemu 在启动时，如何将 pci 挂载到指定的 numa node 下？

    在 QEMU 中，将 PCI 设备挂载到指定的 NUMA 节点有以下几种方法：

    ## 1. **使用 `-device` 的 `numa_node` 参数**

    这是最直接的方法，为每个 PCI 设备指定 NUMA 节点：

    ```bash
    qemu-system-x86_64 \
      -m 8G,slots=2,maxmem=16G \
      -numa node,nodeid=0,cpus=0-3,mem=4G \
      -numa node,nodeid=1,cpus=4-7,mem=4G \
      -device vfio-pci,host=01:00.0,numa_node=0 \
      -device vfio-pci,host=01:00.1,numa_node=1
    ```

    ## 2. **完整示例配置**

    ```bash
    #!/bin/bash
    qemu-system-x86_64 \
      # 内存配置
      -m 16G,slots=4,maxmem=32G \
      
      # 定义 NUMA 节点
      -numa node,nodeid=0,cpus=0-7,mem=8G \
      -numa node,nodeid=1,cpus=8-15,mem=8G \
      
      # 指定 PCI 设备所属 NUMA 节点
      -device vfio-pci,host=0000:01:00.0,id=nic0,numa_node=0 \
      -device vfio-pci,host=0000:02:00.0,id=gpu0,numa_node=1 \
      
      # SCSI 控制器分配到特定节点
      -device virtio-scsi-pci,id=scsi0,numa_node=0 \
      
      # 网卡设备
      -device e1000,mac=52:54:00:12:34:56,numa_node=1
    ```

    ## 3. **使用 `-object` 创建内存后端（更精细控制）**

    对于需要大页内存的设备，可以指定内存后端：

    ```bash
    qemu-system-x86_64 \
      -m 16G \
      -numa node,nodeid=0,cpus=0-7,mem=8G \
      -numa node,nodeid=1,cpus=8-15,mem=8G \
      
      # 为设备创建内存后端
      -object memory-backend-ram,size=4G,id=ram-node0,prealloc=yes,host-nodes=0,policy=bind \
      -object memory-backend-ram,size=4G,id=ram-node1,prealloc=yes,host-nodes=1,policy=bind \
      
      # 设备使用指定内存后端
      -device vfio-pci,host=01:00.0,id=gpu0,memdev=ram-node0
    ```

    ## 4. **验证配置**

    启动后，在 guest OS 中验证：

    ```bash
    # 查看 NUMA 拓扑
    numactl -H

    # 查看 PCI 设备的 NUMA 关联
    lspci -v | grep -A10 "00:01.0"

    # 更详细的 NUMA 信息
    cat /sys/bus/pci/devices/0000:00:01.0/numa_node
    ```

    ## 5. **注意事项**

    ### **NUMA 节点编号**
    - `numa_node=0` 表示第一个 NUMA 节点
    - 节点编号从 0 开始
    - 确保指定的节点已用 `-numa node` 定义

    ### **内存分配**
    - PCI 设备使用的内存会从指定节点的内存中分配
    - 对于需要大量 DMA 内存的设备（如 GPU），正确设置 NUMA 节点非常重要

    ### **性能影响**
    - 将设备分配到离 CPU 最近的 NUMA 节点可减少访问延迟
    - 设备 DMA 缓冲区会使用指定节点的内存

    ### **限制**
    - 某些设备类型可能不支持 `numa_node` 参数
    - VFIO 直通设备通常支持此参数
    - 需要 QEMU 版本支持（较新版本都支持）

    通过以上方法，你可以将 PCI 设备精确地绑定到特定的 NUMA 节点，优化内存访问性能。

    * -device pxb-pcie,id=pcie.1,bus=pcie.0,bus_nr=16,numa_node=0,addr=5.0 \
        -device pxb-pcie,id=pcie.2,bus=pcie.0,bus_nr=32,numa_node=1,addr=6.0 \
        -device pxb-pcie,id=pcie.3,bus=pcie.0,bus_nr=48,numa_node=0,addr=7.0 \
        -device pxb-pcie,id=pcie.4,bus=pcie.0,bus_nr=64,numa_node=1,addr=8.0 \

        这几行做的对吗？

        这几行配置有一些问题，让我来详细分析：

        ## **主要问题：**

        ### 1. **bus_nr 值超出了合法范围**
        - PCI 总线号范围：0-255
        - 每个 PCI 域最多 256 条总线
        - `bus_nr=32,48,64` 可能不连续且可能导致问题

        ### 2. **addr 格式不正确**
        - `addr=5.0` 应该改为 `addr=0x5`（十六进制）
        - PCI 地址格式：`<总线>.<设备>.<功能>` 或简单的总线号

        ### 3. **缺少必要的 memory/io 参数**
        - pxb-pcie 需要指定内存和 I/O 窗口

        ## **正确的配置方法：**

        ```bash
        # 方法1：使用 pxb-pcie（PCI Express 扩展）
        qemu-system-x86_64 \
          -numa node,nodeid=0 \
          -numa node,nodeid=1 \
          
          # 创建连接到 node0 的 PCIe 总线
          -device pxb-pcie,id=pcie.1,bus_nr=0x10,numa_node=0 \
          -device ioh3420,id=root.1,bus=pcie.1,chassis=1,slot=0
          
          # 创建连接到 node1 的 PCIe 总线
          -device pxb-pcie,id=pcie.2,bus_nr=0x20,numa_node=1 \
          -device ioh3420,id=root.2,bus=pcie.2,chassis=2,slot=0
        ```

        ## **更完整的示例：**

        ```bash
        #!/bin/bash
        qemu-system-x86_64 \
          # NUMA 配置
          -m 32G \
          -smp 8,sockets=2,cores=4,threads=1 \
          -numa node,nodeid=0,cpus=0-3,mem=16G \
          -numa node,nodeid=1,cpus=4-7,mem=16G \
          
          # 主 PCIe 根端口
          -device pcie-root-port,id=rp0,bus=pcie.0,chassis=0,slot=0
          
          # 为 node0 创建 PCIe 交换机和端口
          -device pxb-pcie,id=pcie_node0,bus_nr=0x10,numa_node=0 \
          -device ioh3420,id=root_node0,bus=pcie_node0,chassis=1,slot=0
          
          # 为 node1 创建 PCIe 交换机和端口
          -device pxb-pcie,id=pcie_node1,bus_nr=0x20,numa_node=1 \
          -device ioh3420,id=root_node1,bus=pcie_node1,chassis=2,slot=0
          
          # 将设备挂载到 node0 的 PCIe 总线
          -device vfio-pci,host=01:00.0,bus=root_node0,numa_node=0
          
          # 将设备挂载到 node1 的 PCIe 总线
          -device vfio-pci,host=02:00.0,bus=root_node1,numa_node=1
        ```

        ## **注意事项：**

        1. **总线号规划**：
           - 主总线：0x00（默认）
           - 扩展总线：0x10, 0x20, 0x30...（建议用十六进制且连续）
           - 避免冲突

        2. **内存窗口分配**：
           - 如果需要手动控制，可以添加：
             ```bash
             -device pxb-pcie,id=pcie1,bus_nr=0x10,numa_node=0,\
                mem64=[0x100000000-0x13fffffff],\
                io=[0x2000-0x2fff]
             ```

        3. **验证配置**：
           - 启动后查看 PCI 拓扑：
             ```bash
             lspci -tv
             ```
           - 检查 NUMA 关联：
             ```bash
             cat /sys/bus/pci/devices/*/numa_node
             ```

        ## **推荐的简化配置：**

        如果不确定，建议使用更简单的方案：

        ```bash
        # 直接使用 pcie-root-port 并指定 numa_node
        qemu-system-x86_64 \
          -numa node,nodeid=0 \
          -numa node,nodeid=1 \
          
          # 为 node0 创建端口
          -device pcie-root-port,id=rp0,bus=pcie.0,chassis=1,slot=1,numa_node=0
          
          # 为 node1 创建端口
          -device pcie-root-port,id=rp1,bus=pcie.0,chassis=2,slot=2,numa_node=1
          
          # 设备挂载
          -device vfio-pci,host=01:00.0,bus=rp0,numa_node=0
          -device vfio-pci,host=02:00.0,bus=rp1,numa_node=1
        ```

        这样更简单且不容易出错。你原来的配置意图可能是想为每个 NUMA 节点创建独立的 PCIe 域，但需要更仔细地规划总线号和地址空间。

* 详细讲讲 qemu-system-x86_64 中 -net 的参数语法

    在 QEMU（以及 `qemu-system-x86_64`）中，网络的配置是新手最容易撞墙、也是老手最常查阅的地方。

    首先要明确一个极度重要的**时代背景**：

    > ⚠️ **重要警告**：早期的 QEMU 使用 `-net nic,model=... -net user,...` 这种语法。在现代 QEMU（包括 2026 年的当前版本）中，这种老旧的 `-net` 语法**已被废弃（Deprecated）**，甚至在很多发行版中被直接移除。
    > 现代 QEMU 推荐且**标准**的替代方案是：使用 `-netdev`（定义宿主机网络后端）和 `-device`（定义客户机虚拟网卡）**成对出现**的语法。

    下面我将以**现代标准语法**为主，并对参数、常见网络模式进行详细拆解。

    ---

    1. 核心逻辑：后端（-netdev）与前端（-device）的解耦

        现代 QEMU 配置网络就像组装电脑：

        1. **`-netdev ...`（后端）**：相当于把网线插到宿主机的哪个网络环境中（如：直接上网、桥接、点对点等）。
        2. **`-device ...`（前端）**：相当于在虚拟机里插一张什么型号的网卡（如：Intel e1000、Realtek rtl8139，或者高性能的 virtio-net-pci）。

        它们通过一个唯一的 **`id`** 绑定在一起。

    2. 核心语法：常用网络模式详解

        根据你的应用场景，QEMU 提供了几种最常用的网络模式：

        模式 A：用户模式网络（User / SLIRP）—— 最简单，开箱即用

        * **场景**：虚拟机只需要能上外网（查资料、下载包），不需要宿主机或局域网其他机器访问虚拟机。
        * **特点**：QEMU 内部实现了一个虚拟的路由器和 DHCP 服务器，不需要 root 权限。
        * **语法**：

        ```bash
        qemu-system-x86_64 \
          -netdev user,id=net0,hostfwd=tcp::2222-:22 \
          -device virtio-net-pci,netdev=net0
        ```

        关键参数（`user` 专属）：

        * `hostfwd=proto:[hostaddr]:hostport-[guestaddr]:guestport`：**端口映射**（极为常用）。上例中，把宿主机的 `2222` 端口映射到虚拟机的 `22` 端口。这样你在宿主机执行 `ssh -p 2222 user@localhost` 就能登录虚拟机。

        * `net=addr/mask`：自定义虚拟机所在的虚拟网段（默认是 `10.0.2.0/24`）。

        * `dhcpstart=addr`：DHCP 分配的起始 IP（默认从 `10.0.2.15` 开始）。

        ---

        模式 B：桥接模式（TAP）—— 性能最高，如同局域网独立主机

        * **场景**：你需要虚拟机拥有局域网独立 IP，宿主机、虚拟机、局域网其他物理机可以互相同步访问。常用于服务器虚拟化、复杂网络测试。

        * **特点**：需要配合宿主机的网桥（如 `br0`）和 `tap` 设备，通常需要 `sudo` 权限。

        * **语法**：

            ```bash
            qemu-system-x86_64 \
              -netdev tap,id=net1,ifname=tap0,script=no,downscript=no \
              -device virtio-net-pci,netdev=net1
            ```

        关键参数（`tap` 专属）：

        * `ifname=tap0`：指定宿主机上创建的虚拟网卡接口名称。

        * `script=no` / `downscript=no`：禁用 QEMU 默认的启动/关闭脚本。通常我们习惯自己用 `ip link` 或 `brctl` 提前把 `tap0` 建好并桥接到 `br0` 上，所以设为 `no`。

        * 如果你希望 QEMU 自动调用桥接辅助工具，可以使用：`-netdev bridge,id=net1,br=br0`（这会自动处理 tap 设备的创建和桥接）。

        ---

        模式 C：点对点/内部互连（Socket）—— 多台虚拟机组网

        * **场景**：你想启动两台虚拟机，让他们彼此能通信，但又不希望他们干扰宿主机网络。

        * **特点**：利用 UDP 或 TCP Socket 进行数据传输。

        * **语法（第一台，作为监听端）**：

            ```bash
            qemu-system-x86_64 -netdev socket,id=net2,listen=:1234 -device virtio-net-pci,netdev=net2
            ```

        * **语法（第二台，作为连接端）**：

            ```bash
            qemu-system-x86_64 -netdev socket,id=net2,connect=127.0.0.1:1234 -device virtio-net-pci,netdev=net2
            ```

    ---

    3. 前端设备（`-device`）的常用参数与网卡型号

        无论后端用什么网络模式，前端 `-device` 的配置决定了虚拟机的兼容性和性能。

        **常用网卡型号（`model`）**：

        * `virtio-net-pci`：**强烈推荐**。半虚拟化网卡，性能最高，CPU 消耗最低。现代 Linux 和 Windows 10/11（需装 VirtIO 驱动）都完美支持。
        * `e1000`：Intel Gigabit 傲腾网卡。兼容性极好，绝大多数老系统（如 Windows XP/7，旧版 Linux）都自带驱动。
        * `rtl8139`：百兆网卡。远古系统兼容神器。

        **配置 MAC 地址**：

        如果你多开虚拟机，强烈建议手动指定 MAC 地址，避免冲突：

        ```bash
        -device virtio-net-pci,netdev=net0,mac=52:54:00:12:34:56
        ```

        > 💡 *提示：QEMU 的默认 MAC 地址前缀通常是 `52:54:00`。*

    **附：如果你一定要看一眼“老旧的 `-net` 语法”**

    为了让你在看一些陈旧的博客或教案时不至于迷糊，以下是老语法与新语法的对比：

    * **旧语法（不推荐）**：

        ```bash
        -net nic,model=e1000,macaddr=52:54:00:12:34:56 -net user,hostfwd=tcp::2222-:22
        ```

        缺点：前端和后端混杂在相同的 `-net` 标签下，当有多块网卡时，容易出现语意模糊，难以维护。

    * **现代等价新语法（推荐）**：

        ```bash
        -netdev user,id=mynet0,hostfwd=tcp::2222-:22 \
        -device e1000,netdev=mynet0,mac=52:54:00:12:34:56
        ```

    你想为你这个虚拟机配置什么样的网络环境？如果是在做实验（比如搭建 K8s 集群或者测试软路由），告诉我具体需求，我可以直接帮你把整行 QEMU 启动命令拼出来。

* > qemu-system-x86_64: We need to set caching-mode=on for intel-iommu to enable de.

    这个报错是QEMU的警告信息，不是致命的错误。主要涉及以下几个方面：

    1. NUMA节点配置问题

        ```
        qemu-system-x86_64: warning: CPU(s) not present in any NUMA nodes: CPU 96 [sock]
        qemu-system-x86_64: warning: All CPU(s) up to maxcpus should be described in NUMA nodes
        ```

        原因： QEMU配置了多个CPU，但没有正确分配到NUMA（非统一内存访问）节点中

        影响： 性能警告，不会导致功能故障，但可能影响多核性能优化

    2. Intel IOMMU缓存模式警告

        ```
        qemu-system-x86_64: We need to set caching-mode=on for intel-iommu to enable de.
        ```

        原因： Intel IOMMU（输入输出内存管理单元）需要启用缓存模式

        影响： IOMMU功能可能受限，但对于基本PCIe设备模拟通常不影响

    建议解决方案：

    1. **修复NUMA警告**（如果需要）

        在QEMU启动参数中添加NUMA配置：

        ```bash
        -numa node,nodeid=0,cpus=0-95 -numa node,nodeid=1,cpus=96-127
        ```

        （根据你的CPU总数调整范围）

    2. **添加IOMMU缓存模式**

        在QEMU参数中添加：

        ```bash
        -machine q35,accel=kvm -device intel-iommu,caching-mode=on
        ```

* 我想让几个指定的 pci device 进行 iommu passthrough，剩下的正常走 iommu，怎么办

    实现 **部分 PCIe 设备使用 IOMMU passthrough，其他走普通 IOMMU** 在 QEMU 中有几种方法，主要取决于你的具体需求：

    方法一：使用 **VFIO 和 iommu=pt** 模式（推荐）

    1. **配置 GRUB 和内核参数**

        ```bash
        # 在 /etc/default/grub 中添加
        GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"
        # 对于 AMD 平台：amd_iommu=on iommu=pt
        ```

    2. **为不同设备使用不同的驱动**

        ```bash
        # 将需要 passthrough 的设备绑定到 vfio-pci
        # 假设要透传的设备是 00:02.0 和 00:03.0
        sudo lspci -n -s 00:02.0  # 获取 vendor:device ID
        # 输出示例：10de:1b80
        ```

        ```bash
        # 在 QEMU 启动参数中区分设备：
        -device vfio-pci,host=00:02.0,id=passthrough_dev1
        -device vfio-pci,host=00:03.0,id=passthrough_dev2
        # 其他设备会自动走标准 IOMMU
        ```

    方法二：在 QEMU 命令行中混合配置

    完整示例：

    ```bash
    qemu-system-x86_64 \
      -machine q35,accel=kvm \
      -device intel-iommu,caching-mode=on \
      # Passthrough 设备使用 VFIO
      -device vfio-pci,host=00:02.0,multifunction=on,id=sipu1 \
      -device vfio-pci,host=00:03.0,id=sipu2 \
      # 其他设备正常模拟（会走 IOMMU 重映射）
      -device e1000e,netdev=net0 \
      -device virtio-blk-pci,drive=disk0 \
      # VFIO 需要的内存锁定
      -object memory-backend-file,id=mem,size=4G,mem-path=/dev/hugepages,share=on \
      -numa node,memdev=mem \
      -mem-prealloc
    ```

    方法三：使用 **iommu_platform=on/off** 控制（对于 virtio 设备）

    ```bash
    # 某些设备支持直接控制 IOMMU 行为
    -device virtio-blk-pci,iommu_platform=on  # 走 IOMMU
    -device virtio-blk-pci,iommu_platform=off # 绕过 IOMMU
    ```

    方法四：通过 **ACS（Access Control Services）** 分组

    如果硬件支持 ACS，可以创建不同的 IOMMU 组：

    ```bash
    # 检查设备的 IOMMU 组
    ls -l /sys/kernel/iommu_groups/*/devices/

    # 在 QEMU 中可以为不同组设置不同策略
    -device ioh3420,id=root_port1 \
    -device x3130-upstream,id=upstream1,bus=root_port1 \
    -device xio3130-downstream,id=downstream1,bus=upstream1,acs=on
    ```

    SIPU 设备的具体示例：

    假设你想要：
    - sipu_id 0-1: IOMMU passthrough（直接访问物理设备）
    - sipu_id 2-3: 正常 IOMMU 重映射（虚拟化）

    ```bash
    qemu-system-x86_64 \
      -machine q35,accel=kvm \
      -device intel-iommu,caching-mode=on,intremap=on \
      # Passthrough 设备（直通物理 SIPU 0 和 1）
      -device vfio-pci,host=01:00.0,id=sipu0 \
      -device vfio-pci,host=01:00.1,id=sipu1 \
      # 虚拟设备（会通过 IOMMU 重映射）
      -device x-sipu-device,id=sipu2,bus=pcie.1 \
      -device x-sipu-device,id=sipu3,bus=pcie.1
    ```

    重要配置步骤：

    1. **加载必要的内核模块**

        ```bash
        sudo modprobe vfio
        sudo modprobe vfio-pci
        sudo modprobe vfio_iommu_type1
        ```

    2. **解除设备原驱动绑定**

        ```bash
        # 查找设备
        sudo lspci -nn | grep SIPU  # 或你的设备标识
        # 假设设备是 01:00.0 和 01:00.1
        sudo sh -c "echo 0000:01:00.0 > /sys/bus/pci/devices/0000:01:00.0/driver/unbind"
        sudo sh -c "echo 0000:01:00.1 > /sys/bus/pci/devices/0000:01:00.1/driver/unbind"
        ```

    3. **绑定到 vfio-pci**

        ```bash
        # 获取 vendor:device ID
        sudo lspci -n -s 01:00.0
        # 输出：1234:5678

        sudo sh -c "echo 1234 5678 > /sys/bus/pci/drivers/vfio-pci/new_id"
        ```

    验证配置：

    1. **检查 IOMMU 是否启用：**

        ```bash
        dmesg | grep -i iommu
        ```

    2. **检查设备分配：**

        ```bash
        # 查看哪些设备使用了 VFIO
        lspci -k
        # 查看 IOMMU 组
        find /sys/kernel/iommu_groups/ -type l | sort -n
        ```

    哪种场景更符合你的需求？是需要直通物理 SIPU 设备，还是在 QEMU 内混合虚拟和直通设备？
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

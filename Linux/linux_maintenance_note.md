# Linux Maintenance Note

这里主要记录 linux 相关的命令，配置，主题为运维。

这里不记录编程相关，操作系统原理相关。

## cache

* 详细讲讲 ip tuntap 命令的用法

    在 Linux 中，`ip tuntap` 是 `iproute2` 工具包中的一个子命令，用于**创建、删除和管理 TUN/TAP 虚拟网络接口**。

    在详细讲解命令之前，我们需要先简单理清 **TUN** 和 **TAP** 的区别，这决定了你在什么场景下该用哪种模式：

    * **TUN (Network Tunnel)**：工作在**网络层（Layer 3）**。它处理的是 **IP 数据包**。常用于 VPN（如 OpenVPN、WireGuard）、数据加密隧道等。

    * **TAP (Network Tap)**：工作在**数据链路层（Layer 2）**。它处理的是 **以太网帧（Ethernet Frames）**，就像一根虚拟的网线。常用于虚拟机网络（KVM/QEMU）、网桥（Bridge）等需要绑定 MAC 地址的场景。

    一、`ip tuntap` 基本语法

    核心语法结构如下：

    ```bash
    ip tuntap { add | del } [ dev NAME ] [ mode { tun | tap } ] [ user USER ] [ group GROUP ] [ one_queue ] [ pi ] [ vnet_hdr ] [ multi_queue ]
    ```

    常用参数解析：

    * `add` / `del`：添加（创建）或删除一个虚拟接口。
    * `dev NAME`：指定接口的名称（如 `tun0`, `tap1`）。如果不指定，系统会自动生成（如 `tun0`, `tap0`）。
    * `mode`：指定工作模式，可选 `tun` 或 `tap`。
    * `user USER`：指定哪个非 root 用户可以访问/控制该接口（极其实用，避免了让应用程序以 root 权限运行）。
    * `group GROUP`：指定哪个用户组可以访问该接口。
    * `multi_queue`：启用多队列支持，以提高多核 CPU 下的网络吞吐量。

    ---

    二、 核心操作实例

    1. 创建虚拟接口

        **示例 A：创建一个名为 `tun0` 的 TUN 接口，并允许普通用户 `vpnuser` 使用**

        ```bash
        sudo ip tuntap add dev tun0 mode tun user vpnuser
        ```

        **示例 B：创建一个名为 `tap0` 的 TAP 接口**

        ```bash
        sudo ip tuntap add dev tap0 mode tap
        ```

        > **注意**：刚创建好的接口默认是 **DOWN**（关闭）状态的。你需要将它启动并分配 IP 地址才能使用：
        > ```bash
        > sudo ip link set dev tun0 up
        > sudo ip addr add 10.0.0.1/24 dev tun0
        > 
        > ```
        > 

    2. 查看虚拟接口

        虽然 `ip tuntap` 没有专门的 `show` 子命令，但你可以通过以下两种方式查看系统中的 TUN/TAP 接口：

        * **方式一（推荐，专看 tuntap）**：

            直接查看系统的网络设备列表，或者使用 `ip link` 过滤：

            ```bash
            ip link show type tun
            ```

            注：在较新的 Linux 内核中，`tun` 和 `tap` 都可以通过 `type tun` 来查看

        * 方式二（看详细状态）：

            ```bash
            ip -d link show tun0
            ```

            加上 `-d` (details) 参数可以看到该接口是 `tun` 还是 `tap`，以及它所属的 `user` 和 `group`。

    3. 删除虚拟接口

        当你不再需要这个虚拟接口时，可以将其删除。

        **示例：删除 `tun0` 接口**

        ```bash
        sudo ip tuntap del dev tun0 mode tun
        ```

        > **提示**：删除时**必须**指定 `mode`（`tun` 或 `tap`），否则系统可能会报错或无法识别。

    ---

    三、 进阶与高级特性

    1. 免 Root 运行网络程序（User/Group 限制）

        这是 `ip tuntap` 最强大的地方之一。传统的 `tunctl` 或旧方法创建的接口通常需要 root 权限。而使用 `ip tuntap`，你可以提前为某个特定用户创建好接口：

        ```bash
        # 1. 由 root 创建接口并分配给普通用户 tom
        sudo ip tuntap add dev tun1 mode tun user tom

        # 2. 切换到 tom 用户，tom 可以直接用这个接口运行自己的 VPN 客户端，无需 sudo
        ```

    2. 多队列模式 (`multi_queue`)

        在高并发、高吞吐量的服务器场景下（比如作为高性能 KVM 宿主机），单队列的 TAP 接口可能会成为 CPU 0 的瓶颈。

        ```bash
        sudo ip tuntap add dev tap0 mode tap multi_queue
        ```

        开启 `multi_queue` 后，多个 CPU 核心可以同时处理该网卡的数据包，显著提升网络 I/O 性能。

    3. `pi` 参数 (Packet Information)

        默认情况下，TUN/TAP 接口会在传递给用户空间的数据包头部加上 4 个字节的额外信息（Flags 和 Protocol）。如果你不需要这些信息，或者你的应用程序明确要求不需要，可以在创建时加上 `pi` 参数（通常是配合特定的底层网络开发）：

        ```bash
        sudo ip tuntap add dev tun0 mode tun pi
        ```

    四、 常见工作流对比（老工具 vs 新工具）

    如果你在一些老的教程里看到 `tunctl` 或 `openvpn --mktun`，请自动将它们替换为 `ip tuntap`，因为前者在现代 Linux 中大已被淘汰。

    | 历史老命令 (已过时) | 现代标准命令 (`iproute2`) |
    | --- | --- |
    | `tunctl -t tap0` | `ip tuntap add dev tap0 mode tap` |
    | `tunctl -d tap0` | `ip tuntap del dev tap0 mode tap` |
    | `openvpn --mktun --dev tun0` | `ip tuntap add dev tun0 mode tun` |

    五、 快速总结清单

    * **创建 TUN**：`sudo ip tuntap add dev <名字> mode tun`
    * **创建 TAP**：`sudo ip tuntap add dev <名字> mode tap`
    * **删除**：`sudo ip tuntap del dev <名字> mode <tun|tap>`
    * **激活接口**：`sudo ip link set dev <名字> up`

* 详细讲讲 linux 中 brctl 命令的用法

    在 Linux 中，`brctl`（Bridge Control）是一个用于配置和管理 **网桥（Network Bridge）** 的经典命令行工具。网桥可以看作是一个虚拟的物理交换机，用来将多个物理或虚拟网卡连接在同一个局域网（LAN）中。

    虽然在较新的 Linux 发行版中，`brctl` 逐渐被 `ip link` 命令取代（属于 `iproute2` 工具包），但 `brctl`（属于 `bridge-utils` 工具包）因其简单直观，依然在很多自动化脚本和老系统中被广泛使用。

    1. 安装 brctl

        如果你的系统提示找不到 `brctl` 命令，可以通过以下方式安装：

        * **Ubuntu / Debian:** `sudo apt install bridge-utils`
        * **CentOS / RHEL / Fedora:** `sudo yum install bridge-utils` 或 `sudo dnf install bridge-utils`

    2. 核心操作分类详解

        `brctl` 的核心操作主要分为四类：**管理网桥**、**管理网口**、**查看状态** 以及 **配置 STP（生成树协议）**。

        **基础：管理网桥（创建与删除）**

        这类命令用于建立或销毁网桥实例本身。

        * **创建网桥**

            ```bash
            sudo brctl addbr br0
            ```

            解释：创建一个名为 `br0` 的虚拟网桥。

        * **删除网桥**

            ```bash
            sudo brctl delbr br0
            ```

            注意：在删除网桥之前，必须先让它处于 `down`（关闭）状态（使用 `sudo ip link set br0 down`）。

        ---

        **进阶：管理网口（绑定与解绑）**

        创建网桥后，它就像一个没有插网线的交换机。你需要把物理网卡（如 `eth0`）或虚拟网卡（如 `vnet0`）“插”到这个网桥上。

        * **将网卡添加到网桥（绑定）**

            ```bash
            sudo brctl addif br0 eth0
            ```

            解释：将 `eth0` 网卡绑定到 `br0` 网桥上。此时 `eth0` 将作为网桥的一个端口工作。

        * **将网卡从网桥中移除（解绑）**

            ```bash
            sudo brctl delif br0 eth0
            ```

        **监控：查看网桥状态与 MAC 地址表**

        * **查看当前系统中所有的网桥信息**

            ```bash
            brctl show
            ```


            *输出示例：*

            ```text
            bridge name    bridge id            STP enabled    interfaces
            br0            8000.000c29abcdef    no             eth0
                                                               eth1
            ```

        * **查看网桥的学习到的 MAC 地址表（FDB 表）**

            ```bash
            brctl showmacs br0
            ```

            解释：网桥像交换机一样会学习 MAC 地址。这个命令可以让你看到哪些 MAC 地址正连接在网桥的哪个端口上，以及它们是否是本地（local）地址。

        ---

        **高级：配置 STP（生成树协议）**

        当多个网桥连接形成环路时，会导致网络风暴。STP 可以防止环路。

        * **开启 / 关闭 STP**

            ```bash
            sudo brctl stp br0 on   # 开启
            sudo brctl stp br0 off  # 关闭
            ```

        * **修改网桥的各项时间参数（通常保持默认即可）**

            ```bash
            sudo brctl setfd br0 15       # 设置 Forward Delay（转发延迟）为 15 秒
            sudo brctl sethello br0 2     # 设置 Hello Time（问候时间）为 2 秒
            sudo brctl maxage br0 20      # 设置最大老化时间为 20 秒
            ```

    3. 一个完整的实战配置流程

        假设你要把物理网卡 `eth1` 变成一个网桥 `br0`，供虚拟机使用，标准的操作流程如下：

        ```bash
        # 1. 创建网桥
        sudo brctl addbr br0

        # 2. 把物理网卡绑定到网桥
        sudo brctl addif br0 eth1

        # 3. 清空物理网卡的 IP（网桥接管后，物理网卡不需要独立 IP）
        sudo ip addr flush dev eth1

        # 4. 启动物理网卡和网桥
        sudo ip link set eth1 up
        sudo ip link set br0 up

        # 5. 给网桥配置 IP 地址（可选：让宿主机可以通过这个网桥上网）
        sudo dhclient br0  # 或者使用 ip addr add 分配静态 IP
        ```

        > ⚠️ **重要提示：**
        > 
        > 使用 `brctl` 命令所做的所有配置都是**临时生效**的，系统重启后会丢失。如果需要永久生效，需要修改系统的网络配置文件（如 Ubuntu 的 `/etc/netplan/*.yaml` 或 CentOS 的 `/etc/sysconfig/network-scripts/ifcfg-*`）。

    ---

    **时代在变：`brctl` 与 `ip link` 的对比**

    由于 `brctl` 属于老旧工具，现代 Linux 更推荐使用 `iproute2` 工具链。你可以参考下表进行语法转换：

    | 操作目的 | 老命令 (`brctl`) | 新命令 (`ip link` / `bridge`) |
    | --- | --- | --- |
    | **创建网桥** | `brctl addbr br0` | `ip link add br0 type bridge` |
    | **删除网桥** | `brctl delbr br0` | `ip link del br0` |
    | **绑定网口** | `brctl addif br0 eth0` | `ip link set eth0 master br0` |
    | **解绑网口** | `brctl delif br0 eth0` | `ip link set eth0 nomaster` |
    | **查看网桥** | `brctl show` | `ip link show type bridge` 或 `bridge link` |
    | **查看 MAC 表** | `brctl showmacs br0` | `bridge fdb show br br0` |

    你是准备在特定的项目（比如 KVM 虚拟机、Docker 网络）中使用网桥，还是在排查现有的网络问题呢？如果有具体的应用场景，我可以为你提供更针对性的配置方案。

* 详细讲讲 linux 中 nmcli device 相关的命令的作用和用法

    在 Linux 中，`nmcli`（NetworkManager Client）是一个极其强大的命令行工具，用于控制 NetworkManager 并报告网络状态。

    在 `nmcli` 的世界里，有两个核心概念：

    1. **Device（设备）**：指系统中的**物理或虚拟网络接口**（如 `eth0`, `wlan0`, `br0`）。

    2. **Connection（连接）**：指一组**配置好的网络参数**（如 IP 地址、DNS、Wi-Fi 密码）。

    `nmcli device`（可简写为 `nmcli d`）主要负责**直接操作和查看这些网络硬件接口**。

    ---

    一、 核心基础命令

    1. 查看所有网络设备的状态

        ```bash
        nmcli device status
        # 或者简写
        nmcli d
        ```

        * 作用：列出系统识别到的所有网络接口，并显示它们当前的状态（已连接、已断开、未管理等）以及关联的连接名称。

        * 输出示例：

            | DEVICE | TYPE | STATE | CONNECTION |
            | --- | --- | --- | --- |
            | eth0 | ethernet | connected | Wired connection 1 |
            | wlan0 | wifi | disconnected | -- |
            | lo | loopback | unmanaged | -- |

    2. 查看设备的详细信息

        ```bash
        nmcli device show [interface]
        # 例如
        nmcli d show eth0
        ```

        * 作用：显示指定设备（如果不加设备名，则显示所有设备）的极其详细的信息，包括 MAC 地址、IP 地址、子网掩码、网关、DNS、路由信息等。**这相当于 NetworkManager 版的 `ifconfig` 或 `ip addr**`。

    ---

    二、 设备激活与断开（上下线）

    这些命令用于直接控制硬件接口的启用和禁用。

    1. 断开设备连接

        ```bash
        nmcli device disconnect <interface>
        # 例如
        nmcli d disconnect eth0
        ```

        * 作用：让指定设备断开当前的连接，并阻止该设备自动重新连接。
        * 注意：这**不会**关闭网卡硬件，只是让它处于 `disconnected`（空闲）状态。

    2. 激活/连接设备

        ```bash
        nmcli device connect <interface>
        # 例如
        nmcli d connect eth0
        ```

        * 作用：激活该网卡。NetworkManager 会尝试为该接口寻找并应用最合适的可用“连接配置”（Connection）。

    ---

    三、 Wi-Fi 设备专属命令（nmcli device wifi）

    如果你的网卡支持无线，`nmcli d wifi` 是管理无线网络的神器。

    1. 扫描周边的 Wi-Fi 热点

        ```bash
        nmcli device wifi list
        # 或者简写
        nmcli d wifi
        ```

        * **作用**：扫描并列出当前范围内所有可用的 Wi-Fi 网络，显示 SSID、信道、速率、信号强度（BARS）和加密方式。

    2. 连接到指定的 Wi-Fi

        ```bash
        nmcli device wifi connect "WiFi_Name" password "WiFi_Password"
        ```

        * **作用**：连接到指定的无线网络。如果连接成功，NetworkManager 会**自动创建一个同名的连接配置文件**，以后开机就会自动连接。

    3. 强制重新扫描

        ```bash
        nmcli device wifi rescan
        ```

        * **作用**：告诉网卡立刻刷新周围的 Wi-Fi 列表（默认情况下，NetworkManager 会在后台定期自动扫描）。

    ---

    四、 高级操作：修改与重新加载

    1. 实时重新加载配置（不重启网卡）

        ```bash
        nmcli device reapply <interface>
        ```

        * 作用：如果你手动修改了某个连接的配置（比如改了静态 IP），使用 `reapply` 可以让修改**立刻应用到当前运行的设备上**，而不需要先 `disconnect` 再 `connect`。

    2. 修改设备层面的属性

        ```bash
        nmcli device modify <interface> <setting>.<property> <value>
        # 例如临时修改 DNS
        nmcli d modify eth0 ipv4.dns "8.8.8.8"
        ```

        * 作用：对正在运行的设备进行**临时修改**。这种修改是临时的，设备重启或断开后会失效（不会写入持久化配置文件）。

    3. 开启/关闭网卡的自动连接属性

        ```bash
        nmcli device set <interface> autoconnect yes|no
        ```

        * **作用**：控制该硬件设备未来是否允许自动连接网络。

    五、 实用场景速查表

    > **安全提示**：在远程 SSH 连接服务器时，千万不要轻易对正在使用的网卡执行 `disconnect`，否则会导致连接瞬间切断且无法远程连回！

    | 想要实现的操作 | 命令 |
    | --- | --- |
    | **查 IP、网关、DNS 等全部网络信息** | `nmcli d show` |
    | **看看哪块网卡没插网线/没连上** | `nmcli d` |
    | **让某块网卡彻底罢工（断网）** | `nmcli d disconnect eth0` |
    | **让某块网卡重新工作（联网）** | `nmcli d connect eth0` |
    | **用命令行连公司的 Wi-Fi** | `nmcli d wifi connect "Office-Net" password "123456"` |
    | **临时换个 IP（测试用）** | `nmcli d modify eth0 ipv4.addresses 192.168.1.100/24` |

    你目前是在配置服务器的静态网络，还是在调试笔记本的无线网络呢？如果有具体的应用场景，我可以为你提供更针对性的命令示例。

* 详细讲讲 systemd-run 的用法

    `systemd-run` 是 systemd 工具家族中非常强大但常被低估的一个命令。它的核心作用是：**在临时（Transient）的 systemd 单元（Unit）中运行程序。**

    简单来说，平时你用 `&` 或 `nohup` 在后台运行任务，一旦终端断开或者父进程挂了，任务可能就会受影响。而使用 `systemd-run`，你是**让 systemd 进程（PID 1）直接接管并运行该任务**。它会获得和正式 systemd 服务一模一样的待遇：独立的 Cgroup、资源限制、日志收集等。

    以下是关于 `systemd-run` 用法的详细拆解：

    ---

    ## 核心应用场景

    * **真正的后台运行：** 替代 `nohup` 或 `screen`/`tmux` 运行临时脚本，不用担心终端关闭导致任务中断。
    * **资源限制测试：** 临时限制某个高负载命令的 CPU 或内存占用，防止系统卡死。
    * **定时任务：** 替代 `cron`，临时创建一个只执行一次（或周期执行）的定时任务。
    * **安全沙箱：** 利用 systemd 的安全特性，在隔离的环境中运行不信任的程序。

    ---

    ## 基础语法与常用参数

    基础结构非常简单：

    ```bash
    systemd-run [options] COMMAND [arguments...]

    ```

    ### 常用核心参数：

    * `--user`：以当前登录用户的身份（用户级 systemd 实例）运行，不需要 root 权限。如果不加，默认是 `--system`（需要 root 权限）。
    * `--unit=名称`：给这个临时服务起个名字。方便后面用 `systemctl` 控制。
    * `--scope`：默认情况下，命令会在一个独立的 **Service** 单元中异步运行（后台）。如果加上 `--scope`，它会在当前前台进程中运行，但受 systemd 的 **Scope** 单元管理（常用于限制当前 Shell 及其子进程的资源）。
    * `-p` 或 `--property=键=值`：这是最强大的参数，允许你直接注入任意的 systemd 配置项（如内存限制、环境变量等）。

    ---

    ## 经典使用场景与示例

    ### 1. 后台异步运行长耗时任务

    假设你有一个需要运行几小时的备份脚本 `backup.sh`：

    ```bash
    sudo systemd-run --unit=my-backup /usr/local/bin/backup.sh

    ```

    * **效果：** 任务立刻在后台运行，终端会返回一行提示：`Running as unit: my-backup.service`。
    * **如何查看日志？** ```bash
    journalctl -u my-backup.service -f
    ```

    ```


    * **如何停止它？** ```bash
    sudo systemctl stop my-backup.service
    ```


    ```



    ### 2. 限制命令的资源占用（防止系统宕机）

    如果你要编译一个超大项目，或者运行一个极度消耗内存的 Python 脚本，可以限制它最多使用 2 个 CPU 核心和 4G 内存：

    ```bash
    sudo systemd-run -p CPUQuota=200% -p MemoryMax=4G --unit=heavy-job ./run-heavy-simulation.sh

    ```

    > **注：** `CPUQuota=200%` 意味着最多压满 2 个 CPU 核心。如果程序超出了 4G 内存，systemd 会直接触发 OOM 杀死它，保护宿主机安全。

    ### 3. 作为定时任务（替代 `cron`）

    `systemd-run` 可以通过 `--on-active`（从现在起多久后）、`--on-calendar`（绝对时间）等参数直接生成临时的 **Timer** 单元。

    * **30分钟后执行一次：**
    ```bash
    sudo systemd-run --on-active="30m" --unit=delayed-task /usr/local/bin/clean.sh

    ```


    * **每周五凌晨 2 点执行：**
    ```bash
    sudo systemd-run --on-calendar="Fri *-*-* 02:00:00" --unit=weekly-clean /usr/local/bin/clean.sh

    ```


    *(查看排队的定时任务：`systemctl list-timers --all`)*

    ### 4. 普通用户身份运行（无需 root）

    普通用户想丢一个后台任务，加上 `--user` 即可：

    ```bash
    systemd-run --user --unit=my-crawler python3 spider.py

    ```

    * **注意：** 查看和管理时也要带上 `--user`：
    ```bash
    journalctl --user -u my-crawler.service
    systemctl --user status my-crawler.service

    ```

    ### 5. 限制当前终端（或后续命令）的资源

    如果你想限制**接下来的操作**，可以使用 `--scope`：

    ```bash
    sudo systemd-run --scope -p MemoryMax=1G stress --vm 1 --vm-bytes 1.5G

    ```

    * **效果：** `stress` 压测程序刚试图申请 1.5G 内存，就会立刻被 systemd 强行终止，因为整个 Scope 被限制在 1G 内存内。

    ---

    ## systemd-run 与传统工具的对比

    | 特性 | `nohup` / `&` | `screen` / `tmux` | `systemd-run` |
    | --- | --- | --- | --- |
    | **生命周期** | 绑定在当前用户会话（即使 nohup 偶尔也会因为 SIGHUP 异常退出） | 绑定在 tmux 后台进程 | **完全由 PID 1 (systemd) 接管**，与用户登录状态彻底脱钩 |
    | **日志收集** | 重定向到文件（如 `nohup.out`） | 屏幕缓存，断开就看不到了 | **自动整合进 `journalctl**`，支持动态轮转、时间筛选 |
    | **资源限制** | 极弱（依赖 `ulimit`） | 无 | **极强**（支持 cgroups 的 CPU、内存、I/O 动态限制） |
    | **状态查询** | 只能用 `ps` 查 PID | `tmux ls` | 标准的 `systemctl status` 观察，包含运行时间、Cgroup 树 |

    ---

    ## 总结与小贴士

    * **自动清理：** `systemd-run` 创建的 unit 是临时（Transient）的。当命令运行结束退出后，对应的 service 单元会自动销毁，不会在系统里留下垃圾配置文件。
    * **配合环境变量：** 如果你的命令依赖当前环境的环境变量，记得用 `-E KEY=VALUE` 传进去，因为由 systemd 启动的命令默认只有最基础的系统环境变量。

    你目前是有具体的后台任务需要迁移到 `systemd-run` 来管理，还是在设计某种自动化的运维脚本呢？
* 简述 glow, mdcat, catwalk 的作用

    好的，我们来简单说明一下 `glow`、`mdcat` 和 `catwalk` 这三个命令行工具的作用。

    它们的共同点是都用于在**终端**中更好地处理和展示文本内容，但各自的侧重点不同。

    ---

    1. Glow

        **核心作用：在终端中渲染并浏览 Markdown 文件。**

        `glow` 就像一个终端里的 Markdown 阅读器。它不仅能以纯文本方式显示 Markdown，还能对其进行**格式化渲染**，使其在终端中看起来更美观、易读，就像在图形化编辑器里看到的一样（支持加粗、斜体、颜色、表格、代码块高亮等）。

        **主要特点：**
        *   **本地文件渲染**：可以直接在终端里漂亮地查看本地的 `.md` 文件。
            ```bash
            glow README.md
            ```
        *   **暂存区管理**：它有一个独特的“暂存区”功能，可以让你将一些笔记或 Markdown 片段暂存起来，随时查阅。
        *   **浏览在线 Markdown**：可以直接查看 GitHub 等在线仓库的 Markdown 文件（需要配置 GitHub Token）。
        *   **交互式浏览**：不加任何参数直接运行 `glow`，会进入一个交互式界面，可以浏览和管理本地的 Markdown 文件以及暂存区内容。

        **一句话总结：想在终端里优雅地阅读 Markdown 文档？用 `glow`。**

    ---

    2. Mdcat

        **核心作用：在终端中**准确地**渲染 Markdown 文件。**

        `mdcat` 的功能和 `glow` 非常相似，也是用于在终端中渲染 Markdown。它是 [Rust 版本的 `bat`](https://github.com/sharkdp/bat) 生态的一部分。

        **主要特点：**
        *   **强调标准兼容**：`mdcat` 非常注重对 CommonMark 标准的遵循，力求渲染结果准确无误。
        *   **语法高亮**：它对代码块的语法高亮支持非常出色，通常比 `glow` 更强大和准确。
        *   **集成终端特性**：它能更好地处理终端特性，比如支持某些终端的**超链接点击**功能。
        *   **功能更纯粹**：它主要就是一个渲染器，没有像 `glow` 那样的暂存区等额外管理功能。

        **一句话总结：需要一个标准兼容、代码高亮精准的终端 Markdown 渲染器？用 `mdcat`。**

    ---

    3. Catwalk

        **核心作用：快速预览图像、视频等多媒体文件在终端中的显示效果。**

        `catwalk` 与前面两个工具完全不同。它利用现代终端（如 iTerm2, Kitty, WezTerm）的**图形协议**，直接在终端窗口中显示图像、GIF 和视频的缩略图。

        **主要特点：**
        *   **多媒体预览**：无需打开专门的图片查看器或视频播放器，就能在终端里快速看到文件内容。
        *   **批量查看**：可以一次性传入多个文件，它会将它们并排显示，方便你快速浏览和比较。
        *   **开发/设计工作流**：对于开发者或设计师，在命令行中处理图片资源时，可以快速确认文件内容，极大提升效率。
        *   **依赖终端支持**：它的效果取决于你使用的终端是否支持相应的图形协议。

        **一句话总结：想在终端里快速预览图片或视频文件？用 `catwalk`。**

    ---

    总结对比

    | 工具名 | 主要处理内容 | 核心作用 | 一句话场景 |
    | :--- | :--- | :--- | :--- |
    | **`glow`** | Markdown 文本 | **渲染和浏览** Markdown | “在终端里优雅地看我的笔记和文档。” |
    | **`mdcat`** | Markdown 文本 | **精准渲染** Markdown（尤其代码） | “我要在终端里准确无误地查看这个 API 文档。” |
    | **`catwalk`** | 图像、视频 | **预览多媒体**文件 | “我刚下载了一堆图片，在命令行里快速看看哪个是我要的。” |

    简单来说，`glow` 和 `mdcat` 是处理**文本（Markdown）**的，让你读得更舒服；而 `catwalk` 是处理**二进制（多媒体）**的，让你看得见内容。在 `glow` 和 `mdcat` 之间，`glow` 更像一个功能全面的阅读器，而 `mdcat` 则是一个精准的渲染器。
* 我该如何查看 fail2ban 的 filter 的 sshd.conf 中提取出来的 <HOST> 的值？

    `<HOST>` 是 Fail2ban **从日志里匹配出来的 IP 地址**，但 **Fail2ban 默认不会直接把它打印出来**。不过你有几种可靠的方法可以看到 Fail2ban 实际提取到的 `<HOST>` 值。

    下面给你最实用、最直接的方法。

    * fail2ban-regex 验证你的 filter 如何提取 `<HOST>`

        fail2ban-regex 会模拟 filter 并输出它提取到的 `<HOST>`。

        例如：

        ```bash
        fail2ban-regex /var/log/auth.log /etc/fail2ban/filter.d/sshd.conf
        ```

        输出中会列出所有匹配的条目，例如：

        ```
        Matched line(s):
          [1] 2025-12-02... Failed password for root from 45.9.20.11 port 12345 ssh2
            IP: 45.9.20.11
        ```

        这里的 `IP:` 就是 `<HOST>` 的实际值。

    * 方法 3：调试模式（只在临时测试时用）

        Fail2ban 可以带 `--debug` 输出所有匹配信息，包括 `<HOST>`：

        ```bash
        fail2ban-client start --debug
        ```

        但这个模式很啰嗦，不建议长期打开。

    fail2ban-regex 两种模式行为不同的根本原因

    * **单行模式（参数列表中直接放字符串）**
      → 不解析日期 → 正常匹配

    * **文件模式（使用 `-` 或文件路径）**
      → 要解析日期 → 中文日期无法被识别 → 导致 prefregex 失败 → failregex 也失败

    解决方法

    **方法 1 — 禁用日期解析（最简单）**

    在文件模式下加 `--datepattern=`，让 fail2ban 不解析日志日期：

    ```
    echo '12月 02 20:52:30 ...' | \
    fail2ban-regex --datepattern= - /etc/fail2ban/filter.d/sshd.conf
    ```

    这样就和模式 1 行为一致了。

    **方法 2 — 让 sshd 用英文月份（推荐）**

    修改 `/etc/ssh/sshd_config`：

    ```
    SyslogFacility AUTH
    LogLevel VERBOSE
    ```

    然后设置系统 locale：

    ```
    export LC_TIME=en_US.UTF-8
    ```

    或编辑 `/etc/locale.conf`：

    ```
    LC_TIME=en_US.UTF-8
    ```

    然后重启 sshd：

    ```
    systemctl restart sshd
    ```

    这样新日志会变成：

    ```
    Dec 02 20:52:30 ...
    ```

    fail2ban 就能识别。


    * ```<mdre-<mode>>```是什么意思？

        `<mdre-<mode>>` 是 Fail2ban **sshd filter 模板里的一个占位符（macro）**，用来根据 `mode` 参数动态加载不同的 failregex 子集。

        它的作用是 **根据用户设置的 `mode`，决定哪些额外的规则被启用**。

        1. 语法和展开规则

            在 `/etc/fail2ban/filter.d/sshd.conf` 中，你会看到类似：

            ```ini
            failregex = %(cmnfailre)s
                        <mdre-<mode>>
                        %(cfooterre)s
            ```

            解释：

            * `%(cmnfailre)s` → 引用 **公共失败规则**（common fail regex），比如常见的 “Failed password”
            * `<mdre-<mode>>` → **根据 mode 展开不同规则**
            * `%(cfooterre)s` → 引用 **公共尾部规则**（footer regex）

            `<mode>` 的可能值在同一个 filter 里定义：

            ```ini
            # Parameter "mode": normal (default), ddos, extra or aggressive
            mode = normal
            ```

        2. `<mdre-<mode>>` 的展开逻辑

            假设你设置了：

            ```ini
            mode = normal
            ```

            那么 `<mdre-<mode>>` 会展开为 `mdre-normal` 模块对应的规则块。

            * **normal** → 仅包含最基本的 failregex，例如普通登录失败
            * **ddos** → 添加针对大量尝试/端口扫描的规则
            * **extra** → 包含 “Connection closed/reset” 的规则
            * **aggressive** → 合并所有规则（normal + ddos + extra）

            也就是说：

            * 这条占位符本身不匹配任何日志
            * 它只是 **指向一组 failregex**，在加载 filter 时才会展开
            * 如果你 mode = normal，那么 “Connection closed by authenticating user” 这类规则 **不会启用**

        3. 实际效果

            举例：

            ```ini
            mode = normal
            failregex = %(cmnfailre)s
                        <mdre-<mode>>
                        %(cfooterre)s
            ```

            * 普通登录失败 → 匹配
            * Connection closed → **不会匹配**（因为 normal 模式不包含 extra 的规则）

            改成：

            ```ini
            mode = extra
            ```

            * `<mdre-<mode>>` 展开后包含 “Connection closed/reset by authenticating user” 的规则
            * 现在日志就能被匹配，<HOST> 能提取出来

        4. 小结

            * `<mdre-<mode>>` = Fail2ban filter 的宏，占位符，用于**动态加载 mode 对应的 failregex**
            * `<mode>` 可以是 `normal`、`ddos`、`extra` 或 `aggressive`
            * 默认 `normal` 模式不包含“Connection closed/reset”规则
            * 如果你想让这类规则生效，需要：

              1. 改 `mode = extra` 或 `mode = aggressive`
              2. 或直接把规则写在 failregex= 块里，不依赖 `<mdre-<mode>>`
* linux 中清空一个文件的方法

    * `> filename`

    * `truncate -s 0 filename`

    * `echo -n "" > filename`

        `-n`表示不输出最后的换行符

    如果文件需要 sudo 权限才能修改，那么需要配合 `tee` 命令：

    ```bash
    sudo tee filename < /dev/null
    # 或者用前面提到的 truncate
    sudo truncate -s 0 filename
    ```
* Zip 分卷压缩

    你遇到的这种文件列表，属于典型的 **Zip 分卷压缩（Split Zip Archive）**。

    在分卷压缩中，`data.zip` 是主索引文件（通常也是最后一个分卷），而 `.z01`, `.z02` 等是它的数据分卷。解压这种文件，**不能**直接去挨个解压 `.z01` 或 `.z02`，而是需要把它们视作一个整体。

    以下是不同系统下最简单、高效的解压方法：

    方法一：Linux / macOS 终端（最标准做法）

    在 Linux 或 macOS 的命令行中，标准的 `unzip` 命令有时对分卷压缩支持得不够完美（容易报错）。最稳妥的做法是先用 `zip` 命令将它们**合并恢复成一个完整的大 zip 文件**，然后再解压。

    打开终端，进入文件所在的目录，执行以下两步：

    ```bash
    # 1. 将所有分卷合并成一个完整的 single.zip
    zip -s 0 data.zip --out single.zip

    # 2. 直接解压合并后的 single.zip
    unzip single.zip
    ```

    **原理解析**：`-s 0` 参数的意思是“将分卷大小调整为 0（即不分卷）”，`--out` 后面跟着合并后的新文件名。

    为什么不需要手动指定列表？

    在分卷压缩的底层逻辑中，**主文件（也就是 `data.zip`）的末尾记录了整个分卷的核心目录（Central Directory）和所有分卷的信息**。

    1. `zip` 命令会先去读取 `data.zip` 的尾部数据。

    2. 它从里面读出：“哦！我其实是一个分卷压缩包的一部分，我的前一个兄弟叫 `data.z02`（或者根据你的命名，它会去寻找同目录下的前置分卷）。”

    3. **它会自动在当前目录下，按照编号顺序（`.z01`, `.z02`...）去寻找并拼接这些文件。**

    因为 `data.zip` 内部写死了寻找 `data.z01`、`data.z02`，所以我们用一个循环，把那些数字开头的分卷，全部软链接（或者重命名）成以 `data` 开头：

    ```bash
    for f in [0-1]*.z*; do ln -s "$f" "data.${f#*.}"; done
    ```

    > **这行命令在干嘛？** > 它会自动把 `01.z01` 映射成 `data.z01`，把 `14.z14` 映射成 `data.z14`。用软链接（`ln -s`）速度极快且不占额外空间。

    **用 `zip` 修复并合并**

    现在前缀统一了，再次召唤 `zip` 修复工具，它就能顺藤摸瓜找到所有分卷了：

    ```bash
    zip -s 0 data.zip --out correct_single.zip
    ```

    **最后：解压最终得到的完美文件**

    ```bash
    unzip -d data correct_single.zip
    ```
* unzip 的 -x  exclude files that follow (in xlist) 是什么含义

    在 `unzip` 命令中，`-x` 参数的作用是**在解压时「排除」指定的文件或文件夹**。

    example:

    1. 排除单个文件

        ```bash
        unzip project.zip -x notes.txt
        ```

    2. 按通配符匹配排除文件

        如果你不想解压任何 `.log` 结尾的日志文件：

        ```bash
        unzip project.zip -x "*.log"
        ```

        注意：在 Linux/macOS 终端中，通配符（如 `*`）建议**用双引号引起来**（`"*.log"`），防止终端提前把 `*` 解析成了当前目录下的文件。

    3. 排除整个文件夹

        如果你想跳过压缩包里的 `backup` 目录及其下的所有内容：

        ```bash
        unzip project.zip -x "backup/*"
        ```

    4. 同时排除多个不同的规则

        你可以直接在 `-x` 后面列出多个需要排除的内容，用空格隔开：

        ```bash
        unzip project.zip -x "*.md" "config/local.json" "test/*"
        ```

        结果：同时排除了所有 Markdown 文件、特定的本地配置文件以及整个 test 文件夹。

    **注意**

    * `-x` 参数后面跟的所有内容都会被当作「排除对象」。因此，如果你还要指定解压到某个特定目录（使用 `-d` 参数），**一定要把 `-d` 放在 `-x` 及其列表的前面**，或者明确区分开。

        * **正确做法**：`unzip target.zip -d /output/dir/ -x "secret/*"`

        * **错误做法**：`unzip target.zip -x "secret/*" -d /output/dir/` （这可能会让 unzip 误以为 `-d` 也是你想排除的文件名）
* linux 上查看磁盘配额

    * 最常用的命令：`quota`

        这是查看个人配额的标准工具。建议加上 `-s`（human-readable）参数：

        ```bash
        quota -s
        ```

        * **blocks (used):** 当前已使用的空间。
        * **quota (soft):** 软限制。达到此限制后你还能写入，但会收到警告。
        * **limit (hard):** 硬限制。**一旦达到这个值，就会触发你看到的 `Disk quota exceeded` 错误。**
        * **grace:** 如果超过了软限制，这里会显示你还有多长时间来清理文件。

        注：

        1. quota 命令需要使用 apt 安装

    * 如果发现配额确实满了，可以用 `du` 命令排查“大户”：

        ```bash
        # 查看当前目录下各一级子目录的大小，并按大小排序
        du -h --max-depth=1 | sort -hr
        ```
* 关于 ptrace 权限的修改

    在使用 GDB 调试 MPI 程序或进行多进程附加（Attach）调试时，你指的应该是 Linux 的 **`ptrace_scope`** 权限设置。

    这是 Linux 内核安全模块（Yama）的一项限制，旨在防止恶意程序随意通过 `ptrace` 系统调用来监控其他进程。

    **核心设置：调整 `ptrace_scope`**

    如果该值被设置为 `1`，GDB 将无法附加到非子进程的进程上（这在调试正在运行的 MPI 进程时经常发生）。

    1. 临时修改（立即生效，重启失效）

        在终端执行以下命令：

        ```bash
        sudo sysctl -w kernel.yama.ptrace_scope=0
        ```

        或者直接写入 proc 文件系统：

        ```bash
        echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
        ```

    2. 永久修改

        若希望重启后依然有效，需要修改配置文件：

        1.  编辑文件：`sudo nano /etc/sysctl.d/10-ptrace.conf`（或 `/etc/sysctl.conf`）。
        2.  找到 `kernel.yama.ptrace_scope = 1`。
        3.  将其改为 `0`。
        4.  保存退出后，执行 `sudo sysctl -p` 使其生效。

    -----

    **为什么 MPI 调试需要它？**

    当你使用 MPI 运行程序时（例如 `mpirun -np 4 ./my_app`），每个进程都是独立启动的。如果你尝试用 GDB 附加到已经在运行的某个 rank 进程，或者使用像 `tmpi` 这种自动拉起多个 xterm 窗口并在其中运行 GDB 的工具，GDB 需要对目标进程拥有完全的控制权。

    **权限等级说明：**

      * **0 (Classic)**: 允许进程附加到同 UID 的任何其他进程（调试 MPI 最常用）。
      * **1 (Restricted)**: 只能附加到由自己启动的子进程（默认值，通常会导致 MPI 附加失败）。
      * **2 (Admin-only)**: 只有拥有 `CAP_SYS_PTRACE` 权限（通常是 root）才能使用 ptrace。

    -----

    **进阶：调试 MPI 的推荐做法**

    除了设置权限，调试 MPI 通常还会配合以下参数：

      * **分窗口启动**：
        ```bash
        mpirun -np 2 xterm -e gdb ./my_app
        ```
        （这会为每个进程弹出一个独立的终端窗口进行调试）
      * **代码内设卡**：
        在代码中加入一个循环，等待调试器附加：
        ```c
        int i = 0;
        while (i == 0) sleep(5);
        ```
        运行后，手动用 `gdb -p <PID>` 附加，然后 `set var i = 1` 继续执行。

    需要我帮你写一段自动打印 PID 并等待 GDB 附加的 C/C++ 调试代码段吗？
* 详细讲讲 linux 中 pidstat, sy, si, perf, strace 这几个概念

    在 Linux 性能调优和故障排查中，你提到的这几个工具和指标构成了从**宏观系统监控**到**微观代码追踪**的完整链路。

    我们可以将它们分为三类：**系统指标（sy, si）**、**进程级统计（pidstat）**和**深度追踪工具（perf, strace）**。

    ---

    1. CPU 指标：sy 与 si
        
        这两个指标通常出现在 `top` 或 `vmstat` 命令的第一行，反映了 CPU 时间的消耗去向。

        *   **sy (System CPU):**
            *   **定义：** 内核态（Kernel Space）占用 CPU 的时间百分比。
            *   **含义：** 当进程执行**系统调用**（如读写文件 `read`/`write`、分配内存 `malloc`、创建线程等）时，CPU 会切换到内核态。
            *   **高 sy 的信号：** 如果 `sy` 过高（通常超过 20%），说明系统调用过于频繁，或者内核协议栈处理过于繁忙。
        *   **si (Soft IRQ):**
            *   **定义：** 软中断占用 CPU 的时间百分比。
            *   **含义：** 现代 Linux 处理硬件中断采取“上下半部”机制。`si` 代表下半部，通常与**网络 I/O** 密切相关。
            *   **高 si 的信号：** 如果 `si` 居高不下，通常意味着服务器正在处理海量的网络数据包（如 PPS 极高的 DDoS 攻击或高频小包转发）。

    ---

    2. pidstat：进程级的“监控探针”

        `pidstat` 是 `sysstat` 工具包中的利器，专门用于监控**特定进程**或线程的资源消耗。

        *   **核心功能：** 它可以实时查看某个 PID 的 CPU、内存、堆栈、I/O 以及上下文切换情况。
        *   **常用命令：**
            *   `pidstat -u 1`: 每秒显示一次各进程的 CPU 使用率。
            *   `pidstat -w`: 查看进程的**上下文切换**（Context Switch），这是排查高 `sy` 的关键指标。
            *   `pidstat -d`: 查看各进程的磁盘读写情况。

    ---

    3. strace：系统调用的“放大镜”
        
        如果说 `sy` 告诉你内核很忙，那么 `strace` 就能告诉你内核到底在忙什么。

        *   **功能：** 拦截并记录进程执行的所有**系统调用**（System Calls）以及收到的信号。
        *   **使用场景：**
            *   **程序报错：** 查找“File not found”到底是哪个路径拼写错了。
            *   **性能瓶颈：** 观察程序是否在死循环执行某个无效的 `read` 或 `stat`。
        *   **注意：** `strace` 会对目标进程产生巨大的性能损耗（通常会使程序慢一个数量级），**严禁直接在生产环境的高并发服务上长时间运行**。

    ---

    4. perf：全能的“性能手术刀”
        
        `perf` 是 Linux 内核原生的性能分析工具，基于事件采样（Sampling）。

        *   **功能：** 它不仅能看到系统调用，还能深入到**函数级别**，告诉你哪个函数（甚至哪行代码）最耗 CPU。
        *   **核心玩法：**
            1.  **perf top:** 类似于 `top`，但显示的是耗时最高的**函数名**。
            2.  **perf record & report:** 记录一段时间的数据，然后生成报告。
            3.  **火焰图 (Flame Graph):** 这是 `perf` 最出名的应用。通过采集数据并生成直观的火焰图，你可以一眼看出系统的性能瓶颈在哪里。

    ---

    **总结对比**

    | 工具/指标 | 层面 | 解决的核心问题 | 性能损耗 |
    | :--- | :--- | :--- | :--- |
    | **sy / si** | 系统级 | CPU 时间花在内核还是中断上？ | 极低 |
    | **pidstat** | 进程级 | 哪个进程在拖累系统？上下文切换多吗？ | 低 |
    | **perf** | 函数级 | 到底是代码里的哪个函数最慢？ | 中（采样模式） |
    | **strace** | 调用级 | 程序到底在和内核聊什么？为什么报错？ | **极高** |

    **排查建议：**
    先看 `top` 确定 **sy/si** 是否异常；若异常，用 `pidstat` 锁定可疑 **进程**；接着用 `perf` 分析该进程的 **函数耗时**；最后用 `strace` 确认具体的 **系统调用** 行为。
* ubuntu 如何设置开机自动运行某个basH脚本

    在 **Ubuntu** 下，你有好几种方式可以设置开机自动运行某个 `bash` 脚本，取决于你是想让它 **在用户登录时运行**，还是 **开机不需要登录就运行**（system-level）。我给你分情况列一下：

    ---

    方法一：使用 `systemd`（推荐，通用，开机即运行）

    1. 假设脚本路径是 `/home/username/myscript.sh`
       确保脚本可执行：

       ```bash
       chmod +x /home/username/myscript.sh
       ```

    2. 创建一个 `systemd` 服务文件：

       ```bash
       sudo nano /etc/systemd/system/myscript.service
       ```

    3. 写入以下内容（修改路径和用户）：

       ```ini
       [Unit]
       Description=My Startup Script
       After=network.target

       [Service]
       Type=simple
       ExecStart=/home/username/myscript.sh
       User=username
       WorkingDirectory=/home/username
       Restart=on-failure

       [Install]
       WantedBy=multi-user.target
       ```

    4. 启用并启动服务：

       ```bash
       sudo systemctl daemon-reload
       sudo systemctl enable myscript.service
       sudo systemctl start myscript.service
       ```

    5. 查看是否运行成功：

       ```bash
       systemctl status myscript.service
       ```

    ---

    方法二：使用 `crontab`（用户登录时运行）

    1. 编辑当前用户的 crontab：

       ```bash
       crontab -e
       ```

    2. 添加一行：

       ```bash
       @reboot /home/username/myscript.sh
       ```

    3. 确保脚本有执行权限：

       ```bash
       chmod +x /home/username/myscript.sh
       ```

    > ⚠️ 注意：`@reboot` 任务运行时环境比较简陋，可能缺少 PATH，你最好在脚本开头加上：

    ```bash
    #!/bin/bash
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    ```

    ---

    方法三：如果只是图形界面登录时运行

    把脚本添加到 **Startup Applications**（适合桌面环境）。

    1. 打开 `Startup Applications Preferences`
    2. 添加新任务，命令填入脚本路径即可。

    ---

    **总结**

    * 如果要 **系统启动就运行（不用登录）** → 用 **systemd**（方法一）。
    * 如果只要 **用户登录后运行** → 用 **crontab @reboot** 或 **Startup Applications**。

* 我记得不是有个类似 bashrc 的设置

    对的 👍，你记得的是 **用户登录时执行的脚本**，跟 `.bashrc`、`.profile` 之类有关：

    ---

    **区别几个常见文件**

    * **`~/.bashrc`**
      每次启动一个 *交互式 shell* 时执行（例如你开一个终端窗口）。
      ⚠️ 不会在纯开机时自动执行，只有开了终端才会跑。

    * **`~/.profile`** 或 **`~/.bash_profile`**
      在用户登录时（无论是图形界面还是 TTY）执行一次。
      适合放一些只需要在用户登录时跑的脚本。

    * **`/etc/profile`**
      系统范围的，所有用户登录时都会执行。

    ---

    **如果你想在登录时跑脚本**

    比如脚本 `/home/username/myscript.sh`：

    1. 编辑 `~/.profile`：

       ```bash
       nano ~/.profile
       ```

    2. 在文件末尾加：

       ```bash
       # 开机登录后自动执行
       /home/username/myscript.sh &
       ```

    3. 保存后退出。

        这样每次你 **登录这个用户** 时脚本就会跑了（包括图形界面和 ssh 登录）。

    ---

    **如果你想在开机但不登录时也能跑**

    那就不能只靠 `.bashrc` / `.profile`，要用我之前说的 **systemd** 或 **cron @reboot**。

* 我希望在用户登录时只启动一次，再登录就不启动了

    明白了。你要的是 **用户第一次登录时启动一次脚本**，但之后再登录同一会话时 **不再重复执行**。

    这种情况 `.bashrc` 不合适，因为它每开一个终端都会执行；
    `.profile` 虽然是登录时执行，但你可能还想限制为「只执行一次」。

    ---

    解决方法：加一个 **标记文件**

    思路就是：

    * 用户登录时，`.profile` 调用你的脚本
    * 但在执行前先检查一个「标记文件」是否存在
    * 如果不存在，就运行脚本并创建标记
    * 如果存在，就什么也不做

    ---

    **实现步骤**

    假设你的脚本是 `/home/username/myscript.sh`

    1. 编辑 `~/.profile`：

       ```bash
       nano ~/.profile
       ```

    2. 在文件末尾加上：

       ```bash
       # 仅第一次登录时执行脚本
       FLAG="$HOME/.myscript_ran"
       if [ ! -f "$FLAG" ]; then
           /home/username/myscript.sh &
           touch "$FLAG"
       fi
       ```

    3. 确保脚本可执行：

       ```bash
       chmod +x /home/username/myscript.sh
       ```

    ---

    **如果想每次重启后都再执行一次**

    那就在 `~/.profile` 里不变，但把「标记文件」放到 `/tmp`（因为 `/tmp` 每次开机都会清空）：

    ```bash
    FLAG="/tmp/.myscript_ran"
    ```

    ---

    这样，你就可以控制：

    * **同一次开机 → 登录一次就执行，后面不执行**
    * **下次重启 → 又会执行一次**

    ---

    要不要我给你写一个完整的 `myscript.sh` 示例（带日志输出），这样你可以确认它到底有没有只执行一次？

* /etc/profile 中的命令并不是开机运行，而是登录了用户才运行，有没有开机就运行的配置文件或脚本？

    你理解得对：

    * `/etc/profile` 和 `~/.bashrc` 之类的文件只会在**用户登录时**运行，根本不属于“开机自启”。
    * 如果想要 **开机（还没登录）就运行脚本**，有几种常见方式：

    ---

    ### 1. `systemd` 服务（推荐）

    适合所有现代 Linux 发行版（Ubuntu, CentOS, Debian…）。

    1. 写一个脚本，例如 `/usr/local/bin/myscript.sh`：

       ```bash
       #!/bin/bash
       echo "Boot script running at $(date)" >> /var/log/myscript.log
       ```

       记得赋予可执行权限：

       ```bash
       chmod +x /usr/local/bin/myscript.sh
       ```

    2. 新建一个 systemd unit：

       ```bash
       sudo nano /etc/systemd/system/myscript.service
       ```

       内容：

       ```ini
       [Unit]
       Description=Run my script at boot
       After=network.target

       [Service]
       Type=oneshot
       ExecStart=/usr/local/bin/myscript.sh
       RemainAfterExit=yes

       [Install]
       WantedBy=multi-user.target
       ```

    3. 启用并启动：

       ```bash
       sudo systemctl daemon-reload
       sudo systemctl enable myscript.service
       sudo systemctl start myscript.service
       ```

    这样脚本会在**系统启动**（到 multi-user.target）时自动运行，无需用户登录。

    ---

    ### 2. `/etc/rc.local`（旧方法）

    部分系统不再默认支持，但可以手动启用。

    * 在 `/etc/rc.local` 文件里加入命令，例如：

      ```bash
      /usr/local/bin/myscript.sh &
      exit 0
      ```
    * 确保文件可执行：

      ```bash
      sudo chmod +x /etc/rc.local
      ```

    然后 rc.local 就会在开机最后阶段运行。

    ---

    ### 3. `cron @reboot`

    使用 `crontab -e` 添加：

    ```bash
    @reboot /usr/local/bin/myscript.sh
    ```

    这样每次开机时都会执行。

    ---

    ✅ 总结：

    * 如果是现代系统（Ubuntu 16.04+），**systemd** 是最稳定、可控的方式。
    * 如果只想简单快速，可以用 `cron @reboot`。

    ---

    要不要我给你写一个最小的 **systemd service 文件**，你只需要改脚本路径就能直接用？

* @reboot /usr/local/bin/myscript.sh 这个方法根本无法用

    你说得对，在一些 Linux（特别是 Ubuntu 20.04+）上，`cron` 默认并不会在开机阶段正常运行 `@reboot`，需要额外条件满足（比如 `cron` 服务已经启动，并且脚本路径/权限没问题）。

    为了避免这种坑，最稳妥的就是 **用 systemd 服务**。这是现在 Linux 的主流方式。

    ---

    ### 最小可用的 systemd 服务示例

    假设你的脚本是 `/usr/local/bin/myscript.sh`：

    1. **检查脚本权限**

       ```bash
       chmod +x /usr/local/bin/myscript.sh
       ```

    2. **创建 service 文件**

       ```bash
       sudo nano /etc/systemd/system/myscript.service
       ```

       写入：

       ```ini
       [Unit]
       Description=Run my script at boot
       After=network.target

       [Service]
       Type=oneshot
       ExecStart=/usr/local/bin/myscript.sh
       RemainAfterExit=yes

       [Install]
       WantedBy=multi-user.target
       ```

    3. **启用并启动**

       ```bash
       sudo systemctl daemon-reload
       sudo systemctl enable myscript.service
       sudo systemctl start myscript.service
       ```

    4. **验证开机自启**
       重启后查看日志：

       ```bash
       journalctl -u myscript.service -b
       ```

    ---

    🔑 和 `@reboot` 不同，这种方式**百分百可靠**，不会依赖 cron。

    要不要我帮你写一个版本，支持后台常驻运行（比如脚本里有 while true 循环，systemd 会自动守护它）？

* 除了multi-user.target，是否还有其他 target？我希望图形界面和文字界面都能自动启动服务

    对的，`systemd` 里有很多 **target**，用来表示系统处于不同的运行级别（类似 SysV 的 runlevel）。

    ---

    ### 常见的几个 target

    1. **`multi-user.target`**

       * 文字界面（无图形），相当于传统的 runlevel 3。
       * 网络、多用户登录都可用，常用于服务器环境。

    2. **`graphical.target`**

       * 图形界面（桌面环境），相当于传统的 runlevel 5。
       * 一般桌面 Linux 在启动时会最终进入这个 target。

    3. **`basic.target`**

       * 提供基本的系统服务，但没有网络、多用户等。

    4. **`rescue.target`**

       * 单用户模式，类似 runlevel 1。

    5. **`emergency.target`**

       * 紧急维护模式，几乎只有 root shell 和最小挂载。

    6. **`network.target`**

       * 网络服务可用时的 target，常用作依赖。

    ---

    ### 你的需求：

    如果希望 **不论是文字界面还是图形界面** 都能自启动服务，直接把 `[Install]` 部分改成：

    ```ini
    [Install]
    WantedBy=multi-user.target graphical.target
    ```

    这样无论系统最终进入文字模式（multi-user）还是图形模式（graphical），你的服务都会被拉起来。

    ---

    ### 完整示例

    ```ini
    [Unit]
    Description=Run my script at boot
    After=network.target

    [Service]
    Type=oneshot
    ExecStart=/usr/local/bin/myscript.sh
    RemainAfterExit=yes

    [Install]
    WantedBy=multi-user.target graphical.target
    ```

    然后：

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable myscript.service
    ```

    这样就能保证：

    * 在服务器（只有文字界面）环境 → 会运行
    * 在桌面（进入图形界面）环境 → 也会运行

    ---

    要不要我帮你整理一个 **对照表：SysV runlevel ↔ systemd target**，这样你可以一眼看懂该挂到哪个 target？

* RemainAfterExit=yes 这个是干嘛用的？

    好问题 👍，这个参数在 `systemd` 里很容易被忽略。

    ---

    **`RemainAfterExit=yes` 的作用**

    * **默认情况下**，如果一个 service 是 `Type=oneshot`，那么 `ExecStart` 命令跑完进程退出后，`systemd` 会认为这个 service **已经 stopped**。
    * 加了 `RemainAfterExit=yes` 以后，即使 `ExecStart` 的进程退出了，`systemd` 依然会把这个服务标记为 **active (exited)** 状态。

    换句话说：

    * **不加** → 你的脚本跑完后，`systemctl status` 会显示 `inactive (dead)`。
    * **加了** → 你的脚本跑完后，`systemctl status` 会显示 `active (exited)`。

    ---

    **什么时候需要？**

    * 如果你只是想在开机时执行一次脚本（比如写个日志、调整系统参数），其实不需要 `RemainAfterExit=yes`，默认跑完就结束就行。
    * 如果你的脚本只执行一次，但你希望之后通过 `systemctl status` 还能看到它“执行过”，那么就加上 `RemainAfterExit=yes`。

    ---

    **举个例子**

    ```ini
    [Service]
    Type=oneshot
    ExecStart=/usr/local/bin/myscript.sh
    RemainAfterExit=yes
    ```

    执行结果：

    ```bash
    ● myscript.service - Run my script at boot
       Loaded: loaded (/etc/systemd/system/myscript.service; enabled)
       Active: active (exited) since Fri 2025-09-12 09:00:00 CST; 10s ago
    ```

    如果没写 `RemainAfterExit=yes`：

    ```bash
    ● myscript.service - Run my script at boot
       Loaded: loaded (/etc/systemd/system/myscript.service; enabled)
       Active: inactive (dead) since Fri 2025-09-12 09:00:00 CST; 10s ago
    ```

    ---

    **总结：**

    * `RemainAfterExit=yes` = **保持服务在 systemd 状态机里“激活”**，即使脚本已经退出。
    * 对“只执行一次就完事”的脚本不是必须，但有助于用 `systemctl status` 判断是否运行过。

* 报错：

    connect: network is unreachable. With loginFailExit enabled, no additional retries will b


    这个报错信息看起来像是 **systemd 在启动时跑了你的脚本，但脚本里依赖网络**，结果网络还没起来，于是直接失败了。

    关键点在这里：

    ```
    connect: network is unreachable
    With loginFailExit enabled, no additional retries will b...
    ```

    说明你的脚本在尝试连接网络的时候，网络接口还没 ready。

    ---

    ### 解决方法

    在你的 `.service` 文件里，**增加网络依赖**：

    ```ini
    [Unit]
    Description=Run my script at boot
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=oneshot
    ExecStart=/usr/local/bin/myscript.sh
    RemainAfterExit=yes

    [Install]
    WantedBy=multi-user.target graphical.target
    ```

    解释：

    * `After=network-online.target`
      确保脚本在“网络完全就绪”之后再执行。
    * `Wants=network-online.target`
      保证 systemd 会启动 `network-online.target` 这个服务。
    * 注意：`network.target` 只保证“网络服务启动”，但不保证网卡真正拿到 IP。很多脚本需要等到 **network-online.target**。

    ---

    ### 另外注意

    如果你在云环境（比如 AWS、Azure、KVM）或容器里跑，有些 Linux 没有默认启用 `systemd-networkd-wait-online.service`，可能还要手动启用：

    ```bash
    sudo systemctl enable systemd-networkd-wait-online.service
    ```

    ---

    要不要我帮你写一个 **同时支持联网前等待 + 重试机制** 的 service 配置？这样即使网络一开始不可用，脚本也能延迟或自动重试。

* systemd 开机启动脚本

    ```conf
    [Unit]
    Description=hlc script init frpc
    After=network.target
    After=network-online.target
    Wants=network-online.target
    ```

    可以这样写吗，两个 after 算重复吗？

    可以这样写，不算错误 👍。

    在 `systemd` 里：

    * `[Unit]` 段的 **多个 `After=`** 并不是互斥的，都会生效，表示“本服务要在这些 target 之后启动”。
    * 相当于把依赖条件加在一起 → 启动顺序约束更严格。

    所以你写成：

    ```ini
    [Unit]
    Description=hlc script init frpc
    After=network.target
    After=network-online.target
    Wants=network-online.target
    ```

    实际上等价于：

    ```ini
    [Unit]
    Description=hlc script init frpc
    After=network.target network-online.target
    Wants=network-online.target
    ```

    ---

    **推荐写法**

    一般直接写成一行更简洁：

    ```ini
    [Unit]
    Description=hlc script init frpc
    After=network-online.target
    Wants=network-online.target
    ```

    因为：

    * `network-online.target` 本身已经隐含了 `network.target`。
    * 所以单独写 `After=network.target` 没必要，但写了也不会有坏处。

    ---

    **总结：**

    * 你写的那种方式 **能正常用**，只是稍微重复。
    * 如果追求简洁，直接用 `After=network-online.target` 就够了。

* Top 命令的 cpu 使用量中，按下 t  后看到占用率有的线粗，有的线细，是什么含义

    在 `top` 命令界面按下 **`t`** 键，实际上是在切换 **CPU 占用率的显示模式**（由数据变为图形化柱状图）。

    你看到的“线粗”和“线细”，本质上是 `top` 用不同的字符来表示 **CPU 处理不同任务时的状态分布**。这是一种字符级的高低配比图：

    1. 粗线与细线的含义
        
        在默认的彩色模式或特定的字符模式下，柱状图通常由以下两部分组成：

        * **粗块/深色部分 (`#` 或 亮色块)：** 代表 **用户态 (User)** 和 **内核态 (System)** 的消耗。这部分是你运行的任务（如浏览器、编译任务）以及系统内核处理请求所占用的“实打实”的算力。
        * **细线/浅色部分 (`+` 或 暗色块)：** 代表 **低优先级用户态 (Nice)**。如果你手动调高了某些任务的 nice 值（让它们“礼让”其他进程），它们消耗的 CPU 就会以这种形态显示。

    2. 为什么会有这种区分？

        这种设计是为了让你一眼看出 CPU 的**负载质量**：

        * **全是粗块：** 说明系统正在全力处理高优先级的正常任务。
        * **粗块中混杂大量细线：** 说明虽然 CPU 很忙，但很大一部分资源是被那些“不着急”的后台任务占用的。如果此时有紧急任务进来，这些“细线”任务会被系统自动压后。

    3. 如何操作切换？
        
        按下 `t` 键会在以下四种状态间轮转：
        1.  **纯文字模式：** 传统的 `us, sy, ni, id...` 百分比数值。
        2.  **带分类的柱状图：** 用不同字符表示 `user`、`system` 和 `nice`。
        3.  **合并的柱状图：** 不区分具体类型，只显示总占用。
        4.  **关闭显示：** 直接隐藏 CPU 状态行。

    ---

    **小贴士：**
    如果你觉得 `top` 的界面太简陋，推荐尝试 **`htop`**。它的颜色区分更直观：**蓝色**代表低优先级 (nice)，**绿色**代表普通用户任务 (user)，**红色**代表内核任务 (system)。

    需要我帮你对比一下 `top` 里的其他快捷键（比如内存显示的 `m`）或者如何看具体的进程占用吗？

* top 中的 nice value

    在 top 或 ps 命令中，你可以查看 NI (Nice value) 这一列：

    * 范围： -20 到 19。

    * 越“Nice”越慢： 数值越高（比如 10 到 19），说明这个进程越“客气”，它会主动把 CPU 让给别人。这就是你在 top 里看到的那些细线。

    * 默认值： 普通任务的 Nice 值通常是 0。

        小知识：

        如果你想手动启动一个“不着急”的任务，可以在命令前加 nice。例如：
        nice -n 19 tar -czvf backup.tar.gz /large_data
        这样即使压缩几百 GB 的数据，你的电脑用起来依然会很流畅。


* 在执行 Linux 的 top 命令中，我看到那个进度条有虚线有实线，各代表什么意思？

    在 `top` 的 CPU 进度条里：
    - **实线** = 正在被使用的 CPU 时间（us、sy、ni、wa、hi、si、st）
    - **虚线** = **空闲（id）** 的 CPU 时间

    ### 一、快速看懂
    - 整条进度条 = 100% CPU
    - **实线部分**：用户态、内核态、IO等待、中断等已占用的时间
    - **虚线部分**：CPU 空闲、没干活的时间

    ### 二、切换显示方式
    按 `t` 键循环切换：
    1. 纯数字（默认）
    2. 进度条 + 数字
    3. 纯进度条
    4. 隐藏 CPU 行

    ### 三、小技巧
    - 按 `1` 可查看每个 CPU 核心的独立进度条
    - 按 `h` 打开帮助，查看所有交互快捷键

    要不要我给你整理一份 top 常用快捷键速查表，方便你快速切换显示和排序？

    * 我希望下方的任务列表按 Memory 的使用量倒序排列，该怎么办？

        ### 一、运行中按内存倒序（最常用）
        1. 先运行 `top` 进入界面
        2. 按 **Shift+M**（大写 M）
        3. 进程列表立刻按 **%MEM（内存使用率）** 从高到低倒序排列

        ### 二、启动时直接按内存排序
        ```bash
        top -o %MEM
        ```
        启动后直接按内存倒序，不用再按键

        ### 三、切换排序方向
        - 按 **R**：反转排序（升序 ↔ 降序）
        - 再按 **M**：切回内存倒序

        ### 四、小技巧
        - 按 **E**：切换内存单位（KB/MB/GB）
        - 按 **1**：看每个 CPU 核心的独立进度条
        - 按 **W**：保存当前排序设置，下次启动自动生效

        要不要我再给你一份 top 常用排序/显示快捷键速查表，方便你快速切换？

    * 那我如果想按 CPU 的使用量倒序排列呢？

        按 **CPU 使用率倒序** 就更简单了：

        ### 正在运行 top 时
        直接按：
        **Shift + P**（大写 P）

        进程立刻按 **%CPU** 从高到低排序。

        ---

        ### 启动时就按 CPU 排序
        ```bash
        top -o %CPU
        ```

        ---

        ### 小总结（超好记）
        - **Shift + M** → 按内存排序
        - **Shift + P** → 按 CPU 排序
        - **Shift + N** → 按 PID 排序
        - **R** → 反转排序（升序 ↔ 降序）

        需要我帮你把这些常用快捷键整理成**一行速记版**吗？

* tar -t 我希望只查看顶层目录，怎么办

    你希望使用 `tar -t` 命令只查看归档文件的顶层目录/文件（不递归显示子目录内容），这可以通过精准的参数组合和过滤方式实现，以下是两种可靠方案：

    方案一：使用 `tar` 内置 `--no-recursion` 参数（推荐，最简洁）

    `tar` 命令本身提供了 `--no-recursion`（可简写为 `--no-recurse`）参数，用于禁止递归遍历子目录，直接只显示归档的顶层内容，无需额外过滤工具。

    完整命令

    ```bash
    # 基本用法（查看顶层内容，显示详细信息）
    tar -tf 你的归档文件.tar --no-recursion

    # 示例（查看 test.tar 的顶层目录/文件）
    tar -tf test.tar --no-recursion
    ```

    参数说明

    - `-t`：列出归档文件中的内容（查看模式）
    - `-f`：指定归档文件（后面必须紧跟归档文件名，不可省略）
    - `--no-recursion`：核心参数，禁止 `tar` 递归进入子目录，仅展示顶层条目

    方案二：通过 `awk` 过滤路径（兼容无 `--no-recursion` 的老旧 `tar` 版本）

    如果你的 `tar` 版本较旧，不支持 `--no-recursion` 参数，可以通过 `awk` 过滤路径格式，只保留顶层内容（顶层内容的路径不含 `/`，或仅开头无嵌套 `/`）。

    完整命令

    ```bash
    # 用法1：过滤不含 "/" 的顶层条目（纯顶层文件/目录，无嵌套）
    tar -tf 你的归档文件.tar | awk -F '/' 'NF == 1'

    # 用法2：兼容部分特殊顶层目录（更严谨，确保仅第一层）
    tar -tf 你的归档文件.tar | awk '!index($0, "/") || (index($0, "/") == length($0))'

    # 示例（过滤 test.tar 的顶层内容）
    tar -tf test.tar | awk -F '/' 'NF == 1'
    ```

    原理说明

    - `awk -F '/' 'NF == 1'`：以 `/` 作为字段分隔符，仅保留字段数为 1 的行（即路径中不含 `/`，说明是顶层内容，无嵌套子目录）
    - 该方案完全依赖文本过滤，不依赖 `tar` 内置参数，兼容性更强

    补充说明

    1.  若归档文件是压缩格式（如 `.tar.gz`/`.tar.bz2`/`.tar.xz`），只需添加对应解压参数，核心逻辑不变：
        ```bash
        # .tar.gz 格式（加 -z 参数）
        tar -tzf test.tar.gz --no-recursion

        # .tar.bz2 格式（加 -j 参数）
        tar -tjf test.tar.bz2 --no-recursion

        # .tar.xz 格式（加 -J 参数）
        tar -tJf test.tar.xz --no-recursion
        ```
    2.  两种方案对比：`--no-recursion` 更简洁、效率更高（直接由 `tar` 内部控制，无需管道传输）；`awk` 过滤兼容性更强，适用于老旧系统。

    总结

    1.  优先使用 `tar -tf 归档文件 --no-recursion`，简洁高效，实现顶层内容查看；
    2.  老旧 `tar` 版本可备用 `tar -tf 归档文件 | awk -F '/' 'NF == 1'`；
    3.  压缩归档文件只需添加对应解压参数（`-z`/`-j`/`-J`），核心参数不变。

* apt 安装软件可以直接使用 http_proxy, https_proxy，但是如果使用 sudo apt 安装，需要添加 sudo -E apt install xxx

* `ps -xu mkeac | grep qemu`，报错：`error: must set personality to get -x option`，这个报错是为什么

    (ds 回答)

    这个错误是因为在 Linux 系统中，`ps` 命令的选项语法有 **标准语法** 和 **BSD 语法** 的区别，你混用了这两种语法导致的问题。

    **错误原因**

    你的命令 `ps -xu mkeac | grep qemu` 中：

    - `-x` 是 BSD 风格的选项
    - `-u` 是标准语法选项

    在 Linux 系统中，`ps` 命令不允许这样混合使用不同的语法风格。

    **解决方案**

    **方案1：使用纯 BSD 语法（推荐）**

    ```bash
    ps aux | grep qemu | grep mkeac
    # 或者直接过滤
    ps aux | grep qemu | grep -v grep
    ```

    **方案2：使用纯标准语法**

    ```bash
    # 显示所有进程并过滤
    ps -ef | grep qemu | grep mkeac

    # 或者指定用户
    ps -u mkeac -f | grep qemu
    ```

    **方案3：正确的混合使用方式**

    ```bash
    # 先指定用户，再用 BSD 风格显示
    ps -u mkeac --forest | grep qemu

    # 或者用 -U 指定用户
    ps -U mkeac -o pid,cmd | grep qemu
    ```

    **正确示例**

    根据你的需求（查看 mkeac 用户的 qemu 进程），这里有几个推荐的方法：

    ```bash
    # 方法1：最常用（推荐）
    ps -u mkeac -f | grep qemu

    # 方法2：查看所有进程后过滤
    ps aux | grep qemu | grep mkeac

    # 方法3：如果只想看 qemu 进程
    ps -C qemu -f -u mkeac

    # 方法4：更详细的进程信息
    ps -u mkeac -f --forest | grep -E "qemu|PID"
    ```

    **语法说明**

    **标准语法**（使用 `-` 前缀）：
    - `ps -ef`：显示所有进程
    - `ps -u username`：显示指定用户
    - `ps -C command`：显示指定命令

    **BSD 语法**（不使用 `-` 前缀）：
    - `ps aux`：显示所有进程（BSD风格）
    - `ps ax`：显示所有进程
    - `ps u`：显示用户格式

    **最简单的解决方法**

    对于你的需求，直接用这个命令就行：
    ```bash
    ps -fu mkeac | grep qemu
    # 或
    ps -u mkeac -f | grep qemu
    ```

    这两个命令都能正确显示 mkeac 用户下的 qemu 进程信息。

* 详细讲讲 linux 中 ps 的用法

    **Linux `ps` 命令详解**

    `ps` (process status) 是 Linux 中最基础的进程查看命令，用于显示当前系统的进程状态。

    1. **基础用法**

        ```bash
        ps
        ```
        默认显示当前终端下的进程，通常只显示两列：
        - PID：进程ID
        - TTY：终端类型
        - TIME：CPU使用时间
        - CMD：命令名称

    2. **常用选项组合**

        **查看所有进程**

        ```bash
        # 这三种方式都能查看所有进程
        ps -ef          # 标准语法
        ps aux          # BSD 语法  
        ps -eF          # 显示更多字段
        ```

        **常用参数详解**

        ```bash
        # 常用选项
        -e   # 显示所有进程
        -f   # 显示完整格式信息
        -l   # 长格式显示
        -u   # 按用户显示
        -p   # 指定PID显示

        # BSD风格参数
        a    # 显示所有终端进程
        u    # 显示面向用户的格式
        x    # 显示没有终端的进程
        ```

    3. **输出字段说明**

        ```bash
        # ps aux 的输出示例
        USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
        root         1  0.0  0.1 128116  6712 ?        Ss   Nov15   0:03 /usr/lib/systemd
        ```

        **字段含义：**
        - **USER**：进程所有者
        - **PID**：进程ID
        - **%CPU**：CPU使用率
        - **%MEM**：内存使用率
        - **VSZ**：虚拟内存大小(KB)
        - **RSS**：物理内存大小(KB)
        - **TTY**：终端类型
        - **STAT**：进程状态
        - **START**：启动时间
        - **TIME**：CPU占用时间
        - **COMMAND**：命令名称

    4. **进程状态(STAT)**

        ```
        R  # 运行 (Running)
        S  # 可中断睡眠 (Sleeping)
        D  # 不可中断睡眠 (Disk sleep)
        T  # 停止 (Stopped)
        Z  # 僵尸进程 (Zombie)
        X  # 死亡进程 (Dead)
        <  # 高优先级进程
        N  # 低优先级进程
        s  # 会话领导者
        l  # 多线程进程
        +  # 前台进程组
        ```

    5. **常用查询示例**

        **按用户过滤**
        
        ```bash
        ps -u username        # 查看指定用户的进程
        ps -U root -u root    # 查看root用户的进程
        ```

        **按进程名查找**

        ```bash
        ps -C nginx          # 显示nginx进程
        ps -C sshd -o pid,cmd  # 只显示PID和命令
        ```

        **自定义输出列**

        ```bash
        ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu
        # 自定义列并按CPU降序排列
        ```

        **显示进程树**

        ```bash
        ps -ef --forest      # 树形结构显示父子进程
        ps axjf              # BSD风格的进程树
        ```

    6. **实用组合命令**

        ```bash
        # 查找特定进程的PID
        ps aux | grep nginx

        # 显示CPU/内存占用前10的进程
        ps aux --sort=-%cpu | head -11
        ps aux --sort=-%mem | head -11

        # 显示进程的详细环境变量
        ps eww -p PID

        # 显示指定PID的进程
        ps -p 1234,5678 -f

        # 实时监控进程
        watch -n 1 'ps aux --sort=-%cpu | head -20'
        ```

    7. **性能相关用法**

        ```bash
        # 查看线程信息
        ps -eLf              # 显示所有线程
        ps -T -p PID         # 查看指定进程的线程

        # 查看进程的CPU亲和性
        ps -o pid,psr,cmd -p PID

        # 查看进程的内存映射
        ps -o pid,cmd,vsize,rssize -p PID
        ```

    8. **常见问题排查**

        **查找僵尸进程**

        ```bash
        ps aux | grep Z
        ps -eo pid,stat,cmd | grep Z
        ```

        **查找CPU占用过高的进程**

        ```bash
        ps aux --sort=-pcpu | head -5
        ```

        **查找内存占用过高的进程**

        ```bash
        ps aux --sort=-pmem | head -5
        ```

    9. **高级用法示例**

        ```bash
        # 显示所有进程的完整命令行，包括参数
        ps auxww

        # 结合其他命令处理进程
        kill -9 $(ps aux | grep 'process_name' | awk '{print $2}')

        # 按CPU使用率排序，并显示指定列
        ps -e -o pid,pcpu,pmem,args --sort=-pcpu | head -20

        # 显示进程的执行时间
        ps -eo pid,comm,etime,time
        ```

    10. **ps vs top/htop**

        | 命令 | 特点 | 适用场景 |
        |------|------|----------|
        | `ps` | 静态快照，适合脚本处理 | 一次性查看、脚本分析 |
        | `top` | 动态刷新，交互式 | 实时监控、交互操作 |
        | `htop` | 增强版top，界面友好 | 需要可视化操作的场景 |

    **小贴士**

    - `ps aux` 是最常用的组合，足够应付大多数场景
    - 结合 `grep` 可以快速过滤特定进程
    - 使用 `--sort` 参数可以方便地排序查看
    - 在脚本中建议使用完整的选项名（如 `--pid` 而不是 `-p`）以提高可读性

    掌握 `ps` 命令是 Linux 系统管理和问题排查的基础技能之一。

* 我想知道一个文件夹里，包括所有子文件夹，哪些文件最近被修改了

    在 Linux/macOS 系统中，可以使用以下几种方法查找文件夹（包含子文件夹）中最近被修改的文件：

    1. 使用 find 命令（最灵活）

        **查找最近 24 小时内修改的文件**

        ```bash
        find /path/to/folder -type f -mtime -1
        ```

        **查找最近 7 天内修改的文件**

        ```bash
        find /path/to/folder -type f -mtime -7
        ```

        **查找最近 1 小时内修改的文件（分钟级精度）**

        ```bash
        find /path/to/folder -type f -mmin -60
        ```

        **查找最近 10 分钟内修改的文件**

        ```bash
        find /path/to/folder -type f -mmin -10
        ```

    2. 按修改时间排序显示（最直观）

        **显示最近修改的 10 个文件**

        ```bash
        ls -lt /path/to/folder | head -10
        ```

        **递归显示所有文件，按时间排序**

        ```bash
        find /path/to/folder -type f -printf '%T@ %p\n' | sort -rn | head -20
        ```

        **更友好的格式（显示时间）**

        ```bash
        find /path/to/folder -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort -r | head -20
        ```

    3. 使用 stat 命令（详细信息）

        ```bash
        # 显示最近修改的 10 个文件及其修改时间
        find /path/to/folder -type f -exec stat -c '%y %n' {} \; | sort -r | head -10
        ```

    4. 使用 ls 递归（简单但可能较慢）

        ```bash
        # 递归显示所有文件，按时间排序
        ls -ltR /path/to/folder | head -20
        ```

    5. 实用脚本示例

        **查找并显示最近 N 分钟修改的文件**

        ```bash
        #!/bin/bash
        # 用法: ./recent_files.sh /path/to/folder 30

        FOLDER=${1:-.}
        MINUTES=${2:-60}

        find "$FOLDER" -type f -mmin -$MINUTES -exec ls -lh {} \;
        ```

        **带颜色输出的版本（显示最近 1 小时）**

        ```bash
        find . -type f -mmin -60 -exec ls -lh --color=auto {} \; 2>/dev/null
        ```

    6. 不同时间单位的参数

        | 参数 | 含义 | 示例 |
        |------|------|------|
        | `-mmin -n` | 最近 n 分钟内修改 | `-mmin -30` |
        | `-mmin +n` | n 分钟前修改 | `-mmin +60` |
        | `-mtime -n` | 最近 n 天内修改 | `-mtime -1` |
        | `-mtime +n` | n 天前修改 | `-mtime +7` |
        | `-newer file` | 比指定文件新 | `-newer reference.txt` |

    7. Windows 系统（PowerShell）

        ```powershell
        # 查找最近 1 天修改的文件
        Get-ChildItem -Path "C:\folder" -Recurse -File | Where-Object {$_.LastWriteTime -gt (Get-Date).AddDays(-1)} | Sort-Object LastWriteTime -Descending

        # 显示最近 10 个修改的文件
        Get-ChildItem -Path "C:\folder" -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 10
        ```

    8. 常用场景示例

        **查找当前目录最近修改的 5 个文件**

        ```bash
        find . -type f -printf '%T@ %p\n' | sort -rn | head -5 | cut -d' ' -f2-
        ```

        **查找并显示修改时间和文件大小**

        ```bash
        find . -type f -mtime -1 -exec ls -lh {} \; | awk '{print $6, $7, $8, $9}'
        ```

        **只显示文件名（不含路径）**

        ```bash
        find . -type f -mmin -30 -printf '%f\n'
        ```

    **推荐命令**

    **日常使用最方便**：
    ```bash
    # 显示最近修改的 20 个文件，带时间戳
    find . -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort -r | head -20
    ```

    **快速查看最近 1 小时**：
    ```bash
    find . -type f -mmin -60 -ls
    ```

    这些命令中的 `/path/to/folder` 替换为 `.（当前目录）` 或具体的文件夹路径即可。

* linux 如何查看 cpu 频率

    在 Linux 中查看 CPU 频率有多种方法，以下是最常用且实用的几种：

    1. **使用 `lscpu` 命令（最常用）**

        ```bash
        lscpu | grep MHz
        ```
        或者直接查看最大/最小频率：
        ```bash
        lscpu | grep -E "MHz|CPU max|CPU min"
        ```

    2. **查看 `/proc/cpuinfo`**

        ```bash
        cat /proc/cpuinfo | grep MHz
        ```
        查看所有核心：
        ```bash
        grep "MHz" /proc/cpuinfo
        ```

    3. **使用 `cpupower` 命令（需安装）**

        ```bash
        # 安装（Ubuntu/Debian）
        sudo apt install linux-tools-common

        # 查看频率
        cpupower frequency-info
        ```

    4. **实时监控频率变化**

        ```bash
        # 每秒刷新一次
        watch -n 1 "grep 'MHz' /proc/cpuinfo"

        # 或使用
        watch -n 1 "lscpu | grep MHz"
        ```

    5. **使用 `i7z` 工具（Intel CPU）**

        ```bash
        # 安装
        sudo apt install i7z

        # 运行
        sudo i7z
        ```

    6. **使用 `turbostat`（Intel/AMD）**
        
        ```bash
        # 系统自带，显示详细频率信息
        turbostat --show PkgWatt,Corwatt,GFXWatt
        ```

    7. **查看特定核心频率**

        ```bash
        # 查看所有核心的当前频率
        for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do echo "$i: $(cat $i)"; done

        # 转换为 MHz
        for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do echo "$i: $(($(cat $i) / 1000)) MHz"; done
        ```

    **推荐命令**

    最简单直接的方法：
    ```bash
    lscpu | grep MHz
    ```
    这会显示当前频率、最小和最大频率，适合大多数使用场景。

* 已经进入 less 后，如何显示行号

    在 `less` 界面内（已经打开文件的情况下），显示行号最快的方法是：

    1. 实时切换行号

        在 `less` 运行状态下，直接按下键盘上的：
        > **`-N`** (注意是大写的 N，或者先按 `-` 再按 `shift + n`)

        * **效果**：你会看到左下角出现 `line numbers enabled` 的提示，随后屏幕左侧会立即显示行号。
        * **关闭**：再次输入 `-N` 即可取消显示。

    ---

    2. 常用相关快捷键

        既然你已经进入了 `less`，这些操作配合行号会非常高效：

        * **跳到指定行**：直接输入 **行号** 然后按 **`G`** (大写)。例如想去第 500 行，输入 `500G`。
        * **当前百分比/行数信息**：按下 **`Ctrl + G`**，底部的状态栏会显示当前文件的行数范围和百分比。

    ---

    3. 一劳永逸的办法（下次使用）
        
        如果你希望以后每次用 `less` 都默认带行号，而不用进去再按键，可以采用以下两种方式：

        * **临时命令**：
            ```bash
            less -N filename
            ```
        * **永久环境变量**（推荐）：
            在你的 `~/.bashrc` 或 `~/.zshrc` 中添加一行：
            ```bash
            export LESS="-N"
            ```
            这样以后你只需要输入 `less filename`，行号就会自动在那儿等着你了。

* 如何在 linux terminal 中查看图片文件，不需要很精细，看个大概就行

    在 Linux 终端看图片，这听起来有点“硬核”，但实际上有很多有趣的工具可以满足你“看个大概”的需求。根据你使用的终端模拟器和对画质的要求，有以下几种方案：

    ---

    1. 字符画派：`timg` (推荐)
        
        `timg` 是目前最强大的终端图片查看器之一。它非常聪明，如果你的终端支持图形协议（如 Kitty 或 iTerm2），它会显示高像素图片；如果不支持，它会自动降级为高分辨率的 **24-bit ANSI 字符画**。

        * **安装：** `sudo apt install timg` (Ubuntu/Debian) 或 `brew install timg` (macOS)
        * **用法：** ```bash
            timg image.jpg
            ```
        * **优点：** 自动适配环境，彩色字符渲染效果在同类工具中出类拔萃。

    ---

    2. 极简像素派：`catimg`
        
        如果你只需要飞快地扫一眼轮廓，`catimg` 是个轻量级的好选择。它通过彩色方块字符来模拟像素。

        * **安装：** `sudo apt install catimg`
        * **用法：** ```bash
            catimg image.png
            ```
        * **特点：** 速度极快，虽然颗粒感重，但看个配色和构图绰绰有余。

    ---

    3. 老牌经典：`cacaview` (ASCII Art)
        
        如果你想要那种复古的、纯字符（字母、数字、标点）组成的艺术感，可以使用 `caca-utils`。

        * **安装：** `sudo apt install caca-utils`
        * **用法：** ```bash
            cacaview image.jpg
            ```
        * **特点：** 纯 ASCII 码实现，非常有黑客帝国的感觉。

    ---

    4. 现代终端专属：`icat` (Kitty) / `imgcat` (iTerm2)
        
        如果你使用的是一些现代化的终端，它们本身就支持在窗口内渲染真正的位图（不是字符拼凑的）。

        * **Kitty 用户：** 直接使用内置工具：

            ```bash
            kitten icat image.jpg
            ```
        * **iTerm2 (macOS) 用户：** 

            ```bash
            imgcat image.jpg
            ```

    ---

    **总结建议**

    | 工具 | 渲染方式 | 推荐理由 |
    | :--- | :--- | :--- |
    | **`timg`** | ANSI / 图形协议 | **首选**，画质平衡得最好，支持动图。 |
    | **`catimg`** | 彩色方块 | 够快、够简单。 |
    | **`cacaview`** | 纯 ASCII 字符 | 极致复古，不需要颜色支持。 |

    > **温馨提示：** 在使用这些工具时，建议**缩小终端字体**或者**拉大终端窗口**，这样“像素点”会更多，图片看起来会更清晰。

    你目前使用的是哪种终端（比如 GNOME Terminal, Alacritty, 还是 VS Code 的内置终端）？我可以帮你针对性地选一个效果最好的。


* 详细讲讲 find /some/dir -maxdepth 1

    在 Linux 终端中，`find` 是功能最强大的文件搜索工具，而 `-maxdepth` 参数则是控制搜索**“深度”**（即递归层级）的刹车。

    当你执行 `find /some/dir -maxdepth 1` 时，它的核心含义是：**“只在 `/some/dir` 这一层目录里找，不要钻进子目录里去。”**

    ---

    1. 参数详解

        **`-maxdepth 1`**
        * **层级定义**：
            * 层级 `0`：搜索起点本身（即 `/some/dir` 这个目录条目）。
            * 层级 `1`：搜索起点下的直接内容（文件和子文件夹）。
        * **作用**：它强制 `find` 表现得像 `ls` 一样平铺直叙，停止默认的递归搜索。

    ---

    2. 常用场景对比

        假设你的目录结构如下：
        ```text
        /home/hlc/project/
        ├── .env
        ├── main.py
        └── data/
            └── secret.txt
        ```

        **场景 A：不带 -maxdepth（默认行为）**
        ```bash
        find /home/hlc/project -name "*"
        ```
        **结果**：会列出所有文件，包括深度嵌套的 `data/secret.txt`。

        **场景 B：带 -maxdepth 1**
        ```bash
        find /home/hlc/project -maxdepth 1 -name "*"
        ```
        **结果**：只列出 `.env`、`main.py` 和 `data/` 文件夹本身，**绝不会**显示 `secret.txt`。

    ---

    3. 为什么在解决你的 `npm` 报错时很有用？

        在你之前的 `ENOTEMPTY` 错误中，`npm` 报错说目录不为空。如果你想看看到底是哪些“钉子户”文件阻止了重命名，你会用到：

        ```bash
        find /path/to/openai/codex -maxdepth 1
        ```

        **它的妙处在于：**
        1.  **可见性**：它会显示隐藏文件（以 `.` 开头的文件，比如 `.DS_Store` 或 `.git`），而普通的 `ls` 命令默认是不显示的。
        2.  **纯净度**：如果子目录里有成千上万个文件（比如 `node_modules` 内部），不加 `maxdepth` 会导致屏幕被刷屏，而 `maxdepth 1` 让你只关注当前层级的异常。

    ---

    4. 进阶用法示例

        1. 只看当前目录下的隐藏文件
        ```bash
        find . -maxdepth 1 -name ".*"
        ```

        2. 查找当前目录下最近 1 天修改过的文件
        ```bash
        find . -maxdepth 1 -mtime -1
        ```

        3. 统计当前目录下有多少个直接子项（不计深层文件）
        ```bash
        find . -maxdepth 1 | wc -l
        ```

    ---

    **一个重要的位置陷阱**
    在老版本的 `find` 或者某些特定的 Unix 系统中，**`-maxdepth` 必须紧跟在路径后面**。

    * **正确**：`find /dir -maxdepth 1 -name "*.js"`
    * **可能警告/报错**：`find /dir -name "*.js" -maxdepth 1`
        *(系统会提示：测试参数 `-name` 在全局参数 `-maxdepth` 之后，可能会影响效率)*。

    ---

    **总结**
    `find -maxdepth 1` 是你**精确打击**目标文件时的过滤器。在你升级 `codex` 的过程中，它能帮你快速确认那个出问题的全局目录里，到底塞进了什么不该存在的“私货”。

    你刚才在那个报错目录里，用这个命令抓到那个“作乱”的文件了吗？

* lnav (日志导航器 - 极度强大)

    这是目前最智能的命令行日志查看工具，它就像是一个带有 SQL 引擎的 dmesg。

        安装：sudo apt install lnav

        用法：lnav /var/log/

        核心特性：

            自动合并：如果你给它一个目录，它会按时间戳将所有日志文件合并成一个流显示。

            智能语义高亮：它能自动识别日志中的 IP、日期、错误码。

            热键导航：按 e 跳到下一个错误，按 w 跳到下一个警告。

            实时更新：默认就是实时追踪模式。

* grc (给普通命令加特技)

    如果你单纯觉得 tail -f 或 dmesg 太素了，看不清重点，可以用 grc (Generic Colouriser)。

        用法：grc tail -f /var/log/syslog

        效果：它会根据配置文件为输出内容上色，让原本混乱的日志流变得像代码编辑器一样清晰。

* journalctl -f

    如果你的目标是 Linux 系统服务日志（现在的发行版基本都是 Systemd），不要去翻 /var/log，直接用原生的工具：

        用法：journalctl -u nginx.service -f

        特点：这才是最接近 dmesg -w 体验的官方命令。-f 参数就是 follow。

* tee -a：这是一个“三通”命令。

    它的功能是将输入的内容打印在屏幕上，同时写入到文件中。

    -a (append) 参数表示追加。它不会覆盖文件原有的内容，而是把 drm 加在文件末尾。

* 在 Linux 系统中，存放在 modules-load.d 目录下的 .conf 文件会在系统启动阶段被读取，并加载其中列出的内核模块。

* ln 是否能创建文件夹的 hard link?

    简单直接的回答是：**不能。**（在绝大多数现代操作系统和文件系统中，这是被严格禁止的。）

    如果你尝试执行 `ln /path/to/dir /path/to/hardlink`，通常会收到如下报错：
    > `ln: /path/to/dir: hard link not allowed for directory`

    ---

    1. 为什么禁止文件夹硬链接？

        禁止文件夹硬链接主要有两个深层原因：

        * **避免循环引用（Cycles）**：

            硬链接允许一个目标有多个父节点。如果允许对文件夹建立硬链接，你可能会无意中创建一个死循环（例如：文件夹 A 链接到文件夹 B，而 B 又是 A 的子目录）。这会导致文件系统遍历工具（如 `find`、`du`、`ls -R`）陷入无限循环，甚至导致文件系统损坏。

        * **维护树状结构的完整性**：

            文件系统（如 Ext4, XFS, NTFS）依赖于清晰的树状结构来管理 `..`（父目录）指针。如果一个文件夹有两个硬链接，那么它的 `..` 应该指向哪一个父目录呢？这会破坏文件系统的逻辑一致性。

    ---

    2. 例外情况（历史与特殊场景）

        * **`.` 和 `..`**：
            实际上，每个文件夹内部都自带了两个“硬链接”：`.` 指向自己，`..` 指向父目录。这也是为什么一个空文件夹的链接计数（Link Count）通常是 **2** 的原因。
        * **超级用户特权**：
            在某些非常古老的 Unix 系统或特定的底层工具中，系统允许 root 用户强制创建文件夹硬链接（使用 `ln -d`），但这极度危险，现代 Linux 内核已经禁用了这一能力。
        * **macOS 的 Time Machine**：
            苹果的 HFS+ 和 APFS 文件系统在底层使用了一种特殊的“目录硬链接”技术来实现高效备份，但这由内核专门管理，用户层无法通过常规 `ln` 命令随意创建。

    ---

    3. 如果你想实现类似功能，该怎么办？

        如果你需要让一个文件夹出现在多个位置，有三种成熟的替代方案：

        方案 A：符号链接（Symbolic Link/Soft Link）—— **最常用**
        
        这是最标准的做法，类似于 Windows 的“快捷方式”。
        ```bash
        ln -s /source/path /target/link_name
        ```

        方案 B：挂载绑定（Bind Mount）—— **最接近硬链接效果**
        
        如果你希望在文件系统层面让两个路径“完全等价”（例如为了绕过 `chroot` 限制），可以使用 `mount --bind`。这在应用看来和原文件夹一模一样。
        ```bash
        sudo mount --bind /source/path /target/path
        ```

        方案 C：硬链接文件夹内的所有文件
        
        如果你只是想让文件同步，而不是文件夹本身，可以批量建立文件的硬链接：
        ```bash
        cp -al /source/path /target/path
        ```
        *注：`-l` 参数会让 `cp` 创建硬链接而不是复制数据内容。*

    ---

    **总结**

    **不支持。** 为了防止文件系统逻辑崩溃，Linux/Unix 限制了 `ln` 只能对文件创建硬链接。

    你想了解如何使用 **`mount --bind`** 来解决某些特定的目录同步问题吗？或是想深入了解文件系统中 **Inode** 的工作原理？

* 简述`ssh-add`，`ssh-agent`的作用

    在 SSH 体系中，这两个工具是为了解决**“频繁输入私钥密码”**和**“私钥安全管理”**而设计的协作搭档。

    简单来说：**`ssh-agent` 是保险柜，`ssh-add` 是往保险柜里放钥匙的动作。**

    ---

    1. `ssh-agent` (密钥代理服务)

        它是一个运行在后台的**守护进程（Daemon）**，专门用来在内存中暂存解密后的私钥。

        * **核心作用**：
            * **解密存储**：如果你为私钥设置了密码（Passphrase），通常每次连接服务器都要输入一遍。`ssh-agent` 会在你第一次输入密码后，将解密后的私钥存在内存里。
            * **单点登录体验**：一旦代理运行，后续所有的 `ssh`、`scp` 或 `git push` 命令都会自动询问代理，无需你再动手输密码。
            * **安全性**：私钥解密后存在内存中，不会以明文形式写回磁盘。

    ---

    2. `ssh-add` (密钥添加工具)
        
        它是一个用来与 `ssh-agent` 通信的命令行工具。

        * **核心作用**：
            * **加载密钥**：将你的私钥文件（如 `~/.ssh/id_rsa`）通过 `ssh-add` 告知 `ssh-agent`。
            * **解锁**：执行该命令时，系统会提示你输入一次私钥密码，随后该密钥便在代理中处于“激活”状态。
            
        * **常用命令**：
            * `ssh-add ~/.ssh/my_key`：添加指定私钥。
            * `ssh-add -l`：列出当前代理中已保存的所有密钥。
            * `ssh-add -D`：从代理中删除所有密钥（清空内存）。

    ---

    3. 两者如何配合工作？

        通常的工作流程如下：

        1.  **启动代理**：
            ```bash
            eval `ssh-agent -s`
            ```
            *(这一步会启动进程并设置环境变量 `SSH_AUTH_SOCK`，让 SSH 知道去哪里找代理。)*

        2.  **添加私钥**：
            ```bash
            ssh-add ~/.ssh/id_ed25519
            ```
            *(此时输入一次密码，之后这台机器再连对应的服务器就“秒进”了。)*

    ---

    4. 进阶神技：代理转发 (Agent Forwarding)
        
        这是它们最强大的场景。如果你需要从 **本地 A** 登录到 **跳板机 B**，再从 B 登录到 **目标机 C**，但你不想把私钥拷贝到 B 上：

        * 你在本地启动 `ssh-agent` 并执行 `ssh-add`。
        * 连接时使用 `ssh -A user@B`。
        * 当你从 B 连 C 时，B 会通过加密通道向 A 申请身份验证。**私钥始终留在你的本地 A 上**，既安全又方便。

    ---

    **温馨提示**：在现代 macOS 和 Linux 桌面版中，`ssh-agent` 通常在登录系统时就已自动启动并集成了系统的“钥匙串”（Keychain），所以你可能很少手动运行它。

    你想了解如何在 `.bashrc` 中配置自动启动 `ssh-agent`，还是想看看如何配置 `~/.ssh/config` 来自动管理不同的密钥？

* ifconfig eth0:0 192.168.1.200/24 这里的 eth0:0 中的 :0 是什么意思？

    **eth0:0 的含义**

    **eth0:0** 表示**虚拟网络接口**（也称为子接口、别名接口）。

    **详细解释**

    1. **eth0**：物理网卡名称
    2. **:0**：虚拟接口标识号（从0开始编号）
    3. **整体含义**：在物理网卡eth0上创建的**第一个虚拟子接口**

    **主要作用**

    1. **绑定多个IP地址**

        ```bash
        # 为同一块物理网卡配置多个IP
        ifconfig eth0 192.168.1.100/24          # 主IP
        ifconfig eth0:0 192.168.1.200/24        # 第二个IP
        ifconfig eth0:1 192.168.1.201/24        # 第三个IP
        ifconfig eth0:2 192.168.1.202/24        # 第四个IP
        ```

    2. **查看所有虚拟接口**

        ```bash
        ifconfig                    # 显示所有接口（包括虚拟接口）
        ifconfig -a                 # 显示所有接口（包括未激活的）
        ifconfig eth0:0             # 只显示指定虚拟接口

        ip addr show                # 使用ip命令查看（虚拟接口显示为eth0:0）
        ```

    **应用场景**

    1. **虚拟主机托管**

        ```bash
        # 为不同域名绑定不同IP
        ifconfig eth0:0 192.168.1.100/24   # www.site1.com
        ifconfig eth0:1 192.168.1.101/24   # www.site2.com
        ifconfig eth0:2 192.168.1.102/24   # www.site3.com
        ```

    2. **服务隔离**

        ```bash
        # 不同服务使用不同IP
        ifconfig eth0:0 192.168.1.200/24   # Web服务
        ifconfig eth0:1 192.168.1.201/24   # FTP服务
        ifconfig eth0:2 192.168.1.202/24   # 数据库服务
        ```

    3. **测试环境**
        
        ```bash
        # 模拟多台机器进行网络测试
        ifconfig eth0:0 10.0.0.1/24        # 模拟主机A
        ifconfig eth0:1 10.0.0.2/24        # 模拟主机B
        ifconfig eth0:2 10.0.0.3/24        # 模拟主机C
        ```

    **管理虚拟接口**

    **创建虚拟接口**

    ```bash
    # 方法1：ifconfig
    ifconfig eth0:0 192.168.1.200/24 up

    # 方法2：ip命令（推荐）
    ip addr add 192.168.1.200/24 dev eth0 label eth0:0
    ```

    **删除虚拟接口**

    ```bash
    # 方法1：ifconfig
    ifconfig eth0:0 down

    # 方法2：ip命令
    ip addr del 192.168.1.200/24 dev eth0 label eth0:0
    ```

    **永久配置（以CentOS/RHEL为例）**

    ```bash
    # /etc/sysconfig/network-scripts/ifcfg-eth0:0
    DEVICE=eth0:0
    IPADDR=192.168.1.200
    NETMASK=255.255.255.0
    ONBOOT=yes
    ```

    **注意事项**

    1. **传统 vs 现代**
       - `eth0:0` 是老式别名接口表示法
       - 现代Linux推荐使用 `ip` 命令的标签功能

    2. **与物理接口区别**
       - 虚拟接口共享物理网卡的带宽
       - 虚拟接口状态依赖于物理接口（eth0 down时所有eth0:x都down）

    3. **限制**
       - 不同Linux发行版支持数量可能不同
       - 某些网络服务可能需要特殊配置才能监听特定IP

    **现代替代方案（ip命令）**

    ```bash
    # 现代Linux使用ip命令管理多IP
    ip addr add 192.168.1.100/24 dev eth0        # 主IP
    ip addr add 192.168.1.200/24 dev eth0        # 添加第二个IP
    ip addr add 192.168.1.201/24 dev eth0        # 添加第三个IP

    # 查看
    ip addr show eth0
    ```

    这种方法更清晰，所有IP都属于eth0设备，没有虚拟接口的概念。

* 简述 ifconfig 的用法

    **ifconfig 命令简述**

    **ifconfig**（interface configuration）是 Linux/Unix 系统中用于配置和显示网络接口信息的命令。

    **基本用法**

    1. **查看网络接口信息**
        
        ```bash
        ifconfig                    # 查看所有已激活的网卡
        ifconfig -a                 # 查看所有网卡（包括未激活的）
        ifconfig eth0               # 查看指定网卡（如eth0）
        ```

    2. **配置网络接口**
        
        ```bash
        # 配置IP地址
        ifconfig eth0 192.168.1.100 netmask 255.255.255.0

        # 配置IP和子网掩码（简写）
        ifconfig eth0 192.168.1.100/24

        # 添加第二个IP地址
        ifconfig eth0:0 192.168.1.200/24
        ```

    3. **启动/关闭网卡**

        ```bash
        ifconfig eth0 up            # 激活网卡
        ifconfig eth0 down          # 关闭网卡
        ```

    4. **修改MAC地址**

        ```bash
        ifconfig eth0 hw ether 00:11:22:33:44:55
        ```

    **输出信息说明**

    - **eth0**：网卡名称
    - **inet addr**：IPv4地址
    - **inet6 addr**：IPv6地址
    - **HWaddr**：MAC地址
    - **Bcast**：广播地址
    - **Mask**：子网掩码
    - **MTU**：最大传输单元
    - **RX/TX**：接收/发送的数据包统计

    **注意事项**

    - 现代Linux系统推荐使用 **ip** 命令替代（`ip addr`、`ip link`等）
    - 需要root权限执行配置操作
    - 临时配置，重启网络服务或系统后失效

* 简述 ip 命令的用法

    **ip 命令简述**

    **ip** 是Linux系统中现代且强大的网络配置命令，用于替代ifconfig、route等传统网络工具。

    **基本语法**

    ```bash
    ip [选项] 对象 命令
    ```

    常用对象：link（网络设备）、addr（地址）、route（路由）、neigh（邻居）

    **常用用法**

    1. 网络接口管理（link）

        ```bash
        ip link show                    # 查看所有网络接口
        ip link show eth0               # 查看指定接口
        ip link set eth0 up             # 启动接口
        ip link set eth0 down           # 关闭接口
        ip link set eth0 mtu 1500       # 设置MTU值
        ip link set eth0 address 00:11:22:33:44:55  # 修改MAC地址
        ```

    2. IP地址管理（addr）
        
        ```bash
        ip addr show                    # 查看所有IP地址
        ip addr show eth0               # 查看指定接口IP
        ip addr add 192.168.1.100/24 dev eth0     # 添加IP地址
        ip addr del 192.168.1.100/24 dev eth0     # 删除IP地址
        ip addr flush dev eth0          # 清空接口所有IP
        ```

    3. 路由管理（route）

        ```bash
        ip route show                   # 查看路由表
        ip route add default via 192.168.1.1     # 添加默认网关
        ip route add 10.0.0.0/8 via 192.168.1.1  # 添加静态路由
        ip route del 10.0.0.0/8         # 删除路由
        ```

    4. ARP管理（neigh）
        
        ```bash
        ip neigh show                   # 查看ARP缓存
        ip neigh add 192.168.1.200 lladdr 00:11:22:33:44:55 dev eth0  # 添加静态ARP
        ip neigh del 192.168.1.200 dev eth0      # 删除ARP条目
        ```

    **常用选项**

    ```bash
    ip -4 addr show                 # 只显示IPv4地址
    ip -6 addr show                 # 只显示IPv6地址
    ip -s link show eth0            # 显示接口统计信息
    ip -br addr show                # 简洁显示（brief模式）
    ```

    **优势特点**

    - 功能更强大，整合了多个网络命令
    - 输出更规范，易于脚本处理
    - 支持网络命名空间等高级特性
    - 配置立即生效，无需重启服务

* 当没有颜色区分时，ls 看不出哪个是文件，哪个是文件夹。ls -l 又输出太多，我只想看哪个是文件，哪个是文件夹，不关心其他信息。如何解决这个问题？

    你这个问题很常见！有几种简单的方法可以只区分文件和文件夹，而不显示`ls -l`的详细信息：

    方法1：使用 `ls -F`（最推荐）

    ```bash
    ls -F
    ```

    这个命令会在文件名后面添加标识符：

    - `/` 表示目录（文件夹）
    - `*` 表示可执行文件
    - `@` 表示符号链接
    - `=` 表示套接字
    - `|` 表示命名管道

    示例输出：

    ```
    file1.txt  file2.sh*  folder1/  folder2/
    ```

    方法2：使用 `ls -p`

    ```bash
    ls -p
    ```

    只在目录后面添加 `/`，其他文件没有标记：

    ```
    file1.txt  file2.sh  folder1/  folder2/
    ```

    方法3：使用 `ls --classify`

    ```bash
    ls --classify
    ```

    这是 `-F` 的完整参数形式，功能相同。

    方法4：使用 `ls -l | grep`

    如果你想要更清晰的区分：

    ```bash
    ls -l | grep "^d"  # 只显示目录
    ls -l | grep "^-"  # 只显示普通文件
    ```

    方法5：创建别名

    如果你经常使用，可以在 `.bashrc` 或 `.zshrc` 中添加：

    ```bash
    alias lsf='ls -F'
    alias lsp='ls -p'
    ```

    然后就可以直接使用 `lsf` 或 `lsp` 了。

    我个人最推荐 **`ls -F`**，因为它不仅能区分目录，还能识别可执行文件等其他特殊类型的文件。

* 那如果一个符号链接 A 链到文件夹 B 上，我使用 ls -F 查看 A 时，看不出来它是文件还是文件夹

    你说得对！这是一个很好的观察。对于符号链接的情况，`ls -F` 确实有特殊处理：

    `ls -F` 对符号链接的显示规则：

    - 如果符号链接指向**目录**：显示为 `A@` （带 @ 符号）
    - 如果符号链接指向**文件**：显示为 `B@` （同样是带 @ 符号）

    所以 `ls -F` 只能识别出这是一个链接（通过 @），但无法直接看出它指向的是文件还是目录。

    解决方法：

    方法1：使用 `ls -l` 查看具体指向

    ```bash
    ls -l
    ```

    输出示例：

    ```
    lrwxrwxrwx 1 user user 5 Feb 23 10:00 A -> B/
    ```

    这里的 `B/` 表示指向目录，如果只是 `B` 则表示指向文件。

    方法2：使用 `ls -F` 加上 `-L` 选项

    ```bash
    ls -FL
    ```

    这会显示链接指向的目标的类型（会跟随链接）：

    - 如果 A 指向目录：显示为 `A/`
    - 如果 A 指向文件：显示为 `A`

    方法3：组合使用 `ls -F` 和 `file` 命令

    ```bash
    ls -F && file */ 2>/dev/null
    ```

    或者针对特定链接：

    ```bash
    file A
    ```

    输出示例：

    ```
    A: symbolic link to B/
    ```

    方法4：使用 `ls -l` 配合 grep 过滤

    ```bash
    ls -l | grep "^l"  # 查看所有链接
    ```

    所以，如果你主要关心是文件还是目录，并且需要处理符号链接的情况，可以：

    - 如果想**跟随链接看实际类型**：`ls -FL`
    - 如果想**保留链接信息但想知道指向什么**：`ls -l` 查看详情

    注：

    1. 最终选择了`alias ll='ls -pL'`

* 简述 linux 中创建磁盘文件，并制作为 swap，最后使用 swapon 开启的过程

    ## Linux 创建并启用 swap 的步骤

    ### 1. **创建用于 swap 的文件**
    ```bash
    # 方法1：使用 dd 命令（传统方式）
    sudo dd if=/dev/zero of=/swapfile bs=1M count=2048  # 创建 2GB 的 swap 文件

    # 方法2：使用 fallocate（更快，推荐）
    sudo fallocate -l 2G /swapfile  # 创建 2GB 的 swap 文件
    ```

    ### 2. **设置正确的权限**
    ```bash
    sudo chmod 600 /swapfile  # 仅 root 可读写
    ```

    ### 3. **格式化为 swap 格式**
    ```bash
    sudo mkswap /swapfile  # 将文件设置为 swap 格式
    ```

    ### 4. **启用 swap 文件**
    ```bash
    sudo swapon /swapfile  # 立即启用 swap
    ```

    ### 5. **验证 swap 已启用**
    ```bash
    # 查看 swap 使用情况
    swapon --show
    # 或
    free -h
    # 或
    cat /proc/swaps
    ```

    ### 6. **设置为开机自动挂载**
    编辑 `/etc/fstab` 文件：
    ```bash
    sudo vim /etc/fstab  # 或其他编辑器
    ```

    添加以下行：
    ```
    /swapfile none swap sw 0 0
    ```

    ### 完整示例
    ```bash
    # 完整命令序列
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    # 验证
    free -h

    # 设置开机启动
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    ```

    ### 其他相关操作

    #### 关闭 swap
    ```bash
    sudo swapoff /swapfile
    ```

    #### 调整 swappiness 值（控制使用 swap 的倾向）
    ```bash
    # 查看当前值
    cat /proc/sys/vm/swappiness

    # 临时修改（值范围 0-100，越小越少用 swap）
    sudo sysctl vm.swappiness=10

    # 永久修改
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    ```

    #### 删除 swap 文件
    ```bash
    sudo swapoff /swapfile
    sudo rm /swapfile
    # 并从 /etc/fstab 中删除对应行
    ```

    ### 注意事项
    1. **文件位置**：通常放在根目录 `/swapfile`，也可自定义位置
    2. **大小建议**：通常为物理内存的 1-2 倍，或根据实际需求确定
    3. **性能考虑**：swap 文件比 swap 分区稍慢，但更灵活
    4. **Btrfs 文件系统**：若使用 Btrfs，需额外配置才能支持 swap 文件

* @@ -1,4 +1,4 @@ 这个是什么意思？

    这是 `diff -u`（统一格式输出）中的**差异块头信息**，用来定位文件中发生变化的位置。

    基本格式

    ```
    @@ -行号范围 +行号范围 @@
    ```

    具体解释

    ```
    @@ -1,4 +1,4 @@
    ```

    左半部分 `-1,4`：

    - `-` 表示**原文件**（第一个文件）
    - `1` 表示从第1行开始
    - `4` 表示连续显示4行（即第1-4行）

    右半部分 `+1,4`：

    - `+` 表示**新文件**（第二个文件）
    - `1` 表示从第1行开始
    - `4` 表示连续显示4行（即第1-4行）

    实际含义

    这个头信息表示：

    - 原文件的第1行到第4行
    - 新文件的第1行到第4行
    - 这两段内容之间存在差异

    示例场景

    假设原文件 `old.txt`：

    ```
    1: 苹果
    2: 香蕉
    3: 橘子
    4: 葡萄
    ```

    新文件 `new.txt`：

    ```
    1: 苹果
    2: 香蕉
    3: 橙子      # 这里改了
    4: 葡萄
    ```

    执行 `diff -u old.txt new.txt` 输出：

    ```diff
    --- old.txt
    +++ new.txt
    @@ -1,4 +1,4 @@
     苹果
     香蕉
    -橘子
    +橙子
     葡萄
    ```

    不同情况的变化

    | 场景 | 头信息示例 | 说明 |
    |------|-----------|------|
    | 修改 | `@@ -1,4 +1,4 @@` | 第1-4行有修改 |
    | 添加 | `@@ -2,3 +2,4 @@` | 原文件第2-4行，新文件第2-5行（添加了一行） |
    | 删除 | `@@ -3,4 +2,3 @@` | 原文件第3-6行，新文件第2-4行（删除了一行） |

    记忆要点

    - **减号** `-` 代表原文件（被减数）
    - **加号** `+` 代表新文件（加数）
    - **两个数字**：起始行号和连续行数
    - 这个头信息帮助 `patch` 命令准确定位需要修改的位置

* 简述 linux 中 diff 命令的用法

    `diff` 命令用于比较两个文件或目录的差异，通常显示它们的不同之处。

    基本语法

    ```bash
    diff [选项] 文件1 文件2
    ```

    常用选项

    | 选项 | 说明 |
    |------|------|
    | `-i` | 忽略大小写差异 |
    | `-b` | 忽略空格数量的差异 |
    | `-w` | 忽略所有空格 |
    | `-B` | 忽略空行的差异 |
    | `-q` | 仅显示文件是否不同，不显示详细内容 |
    | `-r` | 递归比较目录 |
    | `-u` | 以统一格式输出（最常用） |
    | `-c` | 以上下文格式输出 |
    | `-y` | 并排显示比较结果 |

    输出格式说明

    **正常格式**（默认）：

    - `a`：添加 (add)
    - `c`：更改 (change)
    - `d`：删除 (delete)

    **统一格式**（`-u`）：

    ```
    --- 文件1
    +++ 文件2
    @@ -1,4 +1,4 @@
    -原内容
    +新内容
    ```

    使用示例

    ```bash
    # 比较两个文件
    diff file1.txt file2.txt

    # 统一格式显示差异（最常用）
    diff -u file1.txt file2.txt

    # 并排显示
    diff -y file1.txt file2.txt

    # 递归比较两个目录
    diff -r dir1 dir2

    # 仅显示哪些文件不同
    diff -q dir1 dir2
    ```

    实际应用

    ```bash
    # 生成补丁文件
    diff -u oldfile newfile > patch.diff

    # 应用补丁
    patch < patch.diff
    ```

    `diff` 命令是版本控制系统的基础，也是日常文件比较的常用工具。

* 简述 zcat 的用法

    `zcat` 是一个用于**查看压缩文件内容**的命令行工具，它可以在不解压文件的情况下，直接查看 `.gz` 格式压缩包内的文本内容。

    1. 核心作用

        `zcat` 本质上是 `gunzip -c` 的别名。它的功能是解压文件并将内容输出到标准输出（通常是屏幕），而原始文件保持不变。

    2. 基本语法

        ```bash
        zcat [选项] 文件名.gz
        ```

    3. 常见用法示例

        场景一：查看压缩的日志文件

        系统日志通常会被压缩归档（如 `/var/log/syslog.2.gz`），直接使用 `zcat` 可以查看而无需解压。

        ```bash
        zcat /var/log/syslog.2.gz
        ```

        为了更方便阅读，通常会配合 `less` 使用分页查看：

        ```bash
        zcat /var/log/syslog.2.gz | less
        ```

        场景二：配合 grep 搜索压缩文件内容

        当你需要在多个压缩日志中查找特定关键字（如 "error"）时，可以结合管道使用。

        ```bash
        zcat /var/log/*.gz | grep "error"
        ```

        场景三：同时查看多个文件

        ```bash
        zcat file1.gz file2.gz
        ```

        这会依次解压并输出两个文件的内容。

        场景四：重定向输出

        如果你想要解压并保存内容到新文件（类似于手动解压）：

        ```bash
        zcat archive.gz > newfile.txt
        ```

        这会将解压后的内容写入 `newfile.txt`，而原始的 `archive.gz` 依然存在。

    4. 注意事项

        -   **文件格式**：`zcat` 专门用于处理由 `gzip` 压缩的文件（通常后缀为 `.gz`）。

        -   **与 `gunzip -c` 等价**：以下两条命令效果完全相同：

            ```bash
            zcat file.gz
            gunzip -c file.gz
            ```

        -   **大小写区别**：注意区分 `zcat`（处理 `.gz`）和 `bzcat`（处理 `.bz2`）、`xzcat`（处理 `.xz`）。

    总结

    `zcat` 让你**无需解压就能读取 `.gz` 压缩包内的文本**，尤其适合快速查看系统日志压缩包或在不占用额外磁盘空间的情况下检索数据。

* xfreerdp 忽略证书：`/cert:ignore`

* xfreerdp 禁用 rfx: `-rfx`

    在 xfreerdp 中，`+xxx`表示开启某个功能，`-xxx`表示关闭某个功能

* 简述 strings 的用法

    strings 是一个用于从二进制文件中提取可打印字符串的命令行工具，常用于逆向工程、调试或分析二进制文件。

    1. 基本用法

        ```bash
        strings 文件名
        ```

        这会扫描文件，提取所有长度 ≥ 4 的可打印 ASCII 字符序列并输出。

    2. 常用选项

        * -n <长度>：设置最小字符串长度（默认 4）

            ```bash
            strings -n 6 文件名  # 只输出长度至少为 6 的字符串
            ```

        * -t <格式>：显示每个字符串在文件中的偏移位置

            ```bash
            strings -t d 文件名  # 十进制偏移
            strings -t x 文件名  # 十六进制偏移
            strings -t o 文件名  # 八进制偏移
            ```

        * -a：扫描整个文件（默认可能只扫描特定区域）

            ```bash
            strings -a 文件名
            ```

        * -e <编码>：指定字符编码（默认为 7 位 ASCII）

            ```bash
            strings -e l 文件名  # 16 位小端 Unicode
            strings -e b 文件名  # 16 位大端 Unicode
            strings -e L 文件名  # 32 位小端 Unicode
            strings -e B 文件名  # 32 位大端 Unicode
            ```

        * -f：打印文件名（在扫描多个文件时有用）

            ```bash
            strings -f *.bin
            ```

        * -o：等同于 -t o（八进制偏移，已过时，建议用 -t o）

    3. 典型应用场景

        3.1 逆向工程/CTF

        ```bash
        strings 可执行文件 | grep -i "flag{"
        ```

        3.2 查看嵌入式资源

        ```bash
        strings firmware.bin | head -50
        ```

        3.3 多文件扫描

        ```bash
        strings -f *.so
        ```

        3.4 查找特定长度的字符串

        ```bash
        strings -n 10 文件 | less
        ```

        3.5 带偏移输出，方便定位

        ```bash
        strings -t x 文件 | grep "password"
        ```

    4. 示例

        ```bash
        # 查看 /bin/ls 中的长字符串
        strings -n 10 /bin/ls

        # 带十六进制偏移查看，并搜索 "ELF"
        strings -t x /bin/bash | grep "ELF"

        # 扫描整个文件并输出到文本
        strings -a 可疑文件 > 输出.txt
        ```

    5. 注意事项

        * strings 可能误将二进制数据解释为字符串（假阳性）。

        * 对于 Unicode 文件，需正确使用 -e 选项。

        * 可以结合 grep、awk 等进行进一步过滤。

* 简述 linux 中 bear 的用法

    bear 是一个用于生成编译数据库（compile_commands.json）的工具，它通过拦截 make、cmake 等构建命令的编译过程，自动记录编译每个源文件时使用的编译命令和参数。

    主要功能

    * 生成 compile_commands.json 文件

    * 支持多种构建系统（Make、CMake、Autotools等）

    * 便于与代码分析工具（如clangd、clang-tidy）配合使用

    基本用法

    1. 安装 Bear

        ```bash
        # Ubuntu/Debian
        sudo apt install bear

        # Fedora/RHEL
        sudo dnf install bear

        # macOS
        brew install bear
        ```

    2. 基本命令格式

        ```bash
        bear -- <build-command>
        ```

    3. 常见使用场景

        配合 make 使用：

        ```bash
        bear -- make
        bear -- make -j4          # 支持并行编译
        ```

        配合 CMake 使用：

        ```bash
        mkdir build && cd build
        bear -- cmake ..
        bear -- make
        # 或者简化为
        bear -- cmake --build .
        ```

        配合 Autotools：

        ```bash
        bear -- ./configure
        bear -- make
        ```

    4. 输出文件

        默认在当前目录生成 compile_commands.json，包含每个编译单元的命令、参数、目录等信息。

    常用选项

    ```bash
    # 指定输出文件
    bear --output compile_commands.json -- make

    # 追加到现有文件（而不是覆盖）
    bear --append -- make

    # 使用特定编译器包装器
    bear --use-cc <compiler-wrapper> -- make

    # 详细输出模式
    bear --verbose -- make
    ```

    实际应用示例

    为现有项目生成编译数据库

    ```bash
    cd /path/to/project
    make clean                # 清理旧的构建
    bear -- make              # 重新构建并记录
    ```

    配合代码分析工具

    ```bash
    # 1. 生成编译数据库
    bear -- make

    # 2. 使用 clang-tidy 进行代码检查
    clang-tidy -p . src/*.cpp

    # 3. 为 clangd 等语言服务器提供编译信息
    # （clangd 会自动查找 compile_commands.json）
    ```

    处理复杂构建系统

    ```bash
    # 多层目录结构
    bear -- sh -c "cd src && make && cd ../lib && make"

    # 指定构建目录
    cd build
    bear -- make -C .. -j8
    ```

    注意事项

    * 完整构建：为了捕获所有编译命令，最好执行完整的构建（make clean 后重新构建）

    * 并行构建：Bear 支持 make -jN，但可能需要调整文件锁机制

        ```bash
        bear -- make -j4
        ```

    * 环境变量：某些构建系统可能依赖特定环境变量，确保在正确环境中运行

    * 输出位置：某些工具期望 compile_commands.json 在项目根目录或构建目录

    替代方案

    如果 bear 不适用，可以考虑：

    * CMake: 使用 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    * Bazel: 使用 bazel build --experimental_action_listener=...

    * compiledb: Python 实现的类似工具

    验证生成结果

    ```bash
    # 检查生成的 JSON 文件
    cat compile_commands.json | jq '.[0]'  # 需要安装 jq
    # 或直接查看
    head -20 compile_commands.json
    ```

    Bear 极大地简化了为现有项目配置语言服务器和静态分析工具的过程，是 C/C++ 开发工作流中很有用的工具。

* `grep -w`

    grep -w 是 grep 命令的一个常用选项，用于精确匹配整个单词，而不是单词的一部分。

    主要功能

    * 只匹配完整的单词，不会匹配单词中的一部分

    * 匹配的单词必须被非单词字符包围或位于行首/行尾

    * 非单词字符包括：空格、标点、特殊符号等

    基本语法

    ```bash
    grep -w "pattern" filename
    ```

    示例说明

    示例文件 (file.txt)

    ```text
    apple
    pineapple
    apples
    an apple a day
    apple123
    apple-pie
    ```

    使用示例

    ```bash
    # 1. 只匹配完整的 "apple"
    grep -w "apple" file.txt
    ```

    输出：

    ```text
    apple
    an apple a day
    ```

    解释：只匹配独立的单词 "apple"，不匹配 "pineapple"、"apples" 等包含 "apple" 的单词

    **对比不带 -w 的情况**

    ```bash
    # 不指定 -w 会匹配所有包含 "apple" 的字符串
    grep "apple" file.txt
    ```

    输出：

    ```text
    apple
    pineapple
    apples
    an apple a day
    apple123
    apple-pie
    ```

    更多实用例子

    ```bash
    # 在日志中查找特定错误代码（精确匹配）
    grep -w "ERROR" logfile.txt

    # 统计文件中某个单词出现的次数
    grep -w "success" file.txt | wc -l

    # 查找包含特定单词的行，忽略大小写
    grep -iw "error" file.txt

    # 在多个文件中查找
    grep -w "TODO" *.py
    ```

    常见组合选项

    * -i + -w: 忽略大小写并精确匹配单词

        ```bash
        grep -iw "error" file.txt  # 匹配 Error, ERROR, error 等
        ```

    * -n + -w: 显示匹配行的行号

        ```bash
        grep -nw "pattern" file.txt
        ```

    * -v + -w: 反向匹配（不包含指定单词的行）

        ```bash
        grep -vw "debug" file.txt
        ```

    注意事项

    * 下划线 (_) 被视为单词字符的一部分

    * 数字和字母的组合被视为单词

    * 连字符（-）在单词中间时，整个连字符字符串可能被视为一个单词

    简单来说，grep -w 相当于在模式两边自动加上了单词边界 \b 锚点，类似于正则表达式中的 \bpattern\b。

* linux 中`wc`是否可以统计汉字的字节数，单词数？

    是的，wc 命令可以统计包含汉字的文件，但理解其统计方式很重要：

    1. 基本功能

        ```bash
        wc 文件名
        ```

        输出：行数 单词数 字节数 文件名

    2. 统计汉字的相关选项

        字节数统计（-c）

        ```bash
        wc -c 文件名
        ```

        * 准确统计文件总字节数

        * 汉字通常占2-4字节（取决于编码）

        字符数统计（-m）

        ```bash
        wc -m 文件名
        ```

        * 统计字符数（包括汉字）

        * 一个汉字算作一个字符

        * 注意：需要确保环境设置正确（如LANG=C会影响结果）

        单词数统计（-w）

        ```bash
        wc -w 文件名
        ```

        * 以空格、制表符、换行符分隔的"单词"

        * 对于中英文混合文本：

            * 英文单词会被正确统计

            * 连续汉字会被视为一个"单词"

    3. 编码对统计的影响

        UTF-8编码文件示例：

        文件内容："你好 world"

        ```bash
        $ echo "你好 world" > test.txt
        $ wc test.txt
        1  2  12 test.txt
        # 1行，2个单词，12字节

        $ wc -m test.txt
        1  8  12 test.txt
        # 8个字符（2个汉字 + 5个英文字母 + 1个空格）
        ```

    4. 实用示例

        ```bash
        # 查看所有统计信息
        wc filename.txt

        # 只统计字节数
        wc -c filename.txt

        # 只统计字符数（包括汉字）
        wc -m filename.txt

        # 只统计单词数
        wc -w filename.txt

        # 统计行数
        wc -l filename.txt
        ```

    5. 注意事项

        * 编码敏感：wc 依赖系统区域设置，建议统一使用UTF-8编码

        * 单词定义：汉字之间没有空格时，多个汉字被计为一个"单词"

        * 组合使用：

            ```bash
            # 统计中文字符数量（估算）
            wc -m file.txt | awk '{print $1}'
            ```

    6. 替代方案

        如果需要更准确的中文分词统计：

        ```bash
        # 使用中文分词工具配合wc
        echo "你好世界" | jieba分词 | wc -w

        # 统计纯中文字符数
        grep -oP '[\p{Han}]' file.txt | wc -l
        ```

    总的来说，wc 可以统计汉字的字节数和作为整体的"单词数"，但对于中文的语义单词统计不够精确。

    注：

* wc 无法统计汉字的字数

    是的，你说得很对！wc 无法准确统计汉字的字数，原因如下：

    1. wc -w 的问题

        ```bash
        echo "你好世界今天天气很好" | wc -w
        # 输出：1
        ```

        * 所有连续汉字被视为一个"单词"

        * 不符合中文"词"的概念

    2. wc -c 和 wc -m 的局限

        ```bash
        echo "你好" | wc -m
        # 输出：3（2个汉字 + 1个换行符）
        echo "你好" | wc -c
        # 输出：7（UTF-8下每个汉字3字节 + 1个换行符）
        ```

        * -m 统计字符数，不是字数（中文里的"词"）

        * -c 统计字节数，更不相关

    3. 为什么 wc 不适合中文

        中文的"词" vs 英文的"word"

        * 英文：单词由空格分隔 → wc -w 有效

        * 中文：词由语义决定，无分隔符 → wc -w 无效

    4. 准确统计中文字数的替代方案

        方案1：使用专业中文分词工具

        ```bash
        # 安装结巴分词（Python）
        pip install jieba

        # 统计中文词数
        echo "你好世界今天天气很好" | python3 -c "
        import jieba
        import sys
        text = sys.stdin.read().strip()
        words = list(jieba.cut(text))
        print('词数:', len(words))
        print('分词结果:', '/'.join(words))"
        # 输出：你好/世界/今天/天气/很好 (5个词)
        ```

        方案2：统计汉字字符数

        ```bash
        # 只统计中文字符个数
        echo "Hello 你好世界！ABC" | grep -oP '[\p{Han}]' | wc -l
        # 输出：4（你、好、世、界）

        # 或使用Python
        echo "你好世界" | python3 -c "
        import sys
        text = sys.stdin.read()
        count = sum('\u4e00' <= ch <= '\u9fff' for ch in text)
        print(f'汉字数: {count}')"
        ```

        方案3：使用中文字数统计工具

        ```bash
        # 安装中文处理工具
        # pip install zhconv

        # 自定义统计脚本
        cat << 'EOF' > count_chinese.py
        import sys
        import jieba

        text = sys.stdin.read()
        # 统计汉字字符数
        char_count = sum('\u4e00' <= ch <= '\u9fff' for ch in text)
        # 统计中文词数
        word_count = len(list(jieba.cut(text)))

        print(f'汉字字符数: {char_count}')
        print(f'中文词数: {word_count}')
        EOF

        echo "我爱北京天安门" | python3 count_chinese.py
        ```

    5. 简单实用的临时方案

        纯汉字字符统计

        ```bash
        # 使用grep（需要支持PCRE）
        echo "测试文字123abc" | grep -oP '[\x{4e00}-\x{9fff}]' | wc -l

        # 使用sed
        echo "测试文字" | sed 's/[^[:punct:]]//g' | wc -m
        ```

        最简方案（仅统计字符）

        ```bash
        # 使用awk统计非空格字符
        echo "你好世界" | awk '{print gsub(/[^\s]/,"&")}'
        # 输出：4
        ```

    总结

    | 需求 | wc 能力 | 推荐方案 |
    | - | - | - |
    | 字节数 | ✅ 准确 | wc -c |
    | 字符数 | ✅ 基本准确 | wc -m |
    | 英文单词数 | ✅ 准确 | wc -w |
    | 中文字符数 | ⚠️ 需处理 | grep + wc |
    | 中文词数 | ❌ 不准确 | jieba分词 |

    结论：wc 不适合统计中文的字数/词数，需要借助中文分词工具才能准确统计。

* 简述 `/var/log/syslog`, `/var/log/messages` 的作用与区别

    /var/log/syslog 和 /var/log/messages 都是 Linux 系统中用于记录系统日志的重要文件，但它们的用途和内容因发行版而异。

    主要作用

    /var/log/syslog

    * 记录系统范围的日志信息

    * 包含内核、服务、应用程序等多种日志

    * 通常由 rsyslog 或 syslog-ng 管理

    * 是 syslog 服务的默认主日志文件

    /var/log/messages

    * 传统系统日志文件

    * 记录系统启动、运行状态、硬件检测等信息

    * 包含非关键性系统消息

    主要区别

    | 特性 | /var/log/syslog | /var/log/messages |
    | - | - | - |
    | 主要使用发行版 | Debian/Ubuntu 等 | RHEL/CentOS/Fedora 等 |
    | 日志详细程度 | 更详细，包含大多数日志 | 较简略，主要为系统级消息 |
    | 默认包含设施 | 所有设施（facility） | 通常排除 mail, authpriv, cron 等 |
    | 优先级 | 通常记录 info 及以上 | 通常记录 warning/err 及以上 |

    配置差异

    在 Debian/Ubuntu 系统中：

    * /var/log/syslog 是主要日志文件

    * 配置通常在 /etc/rsyslog.conf 或 /etc/rsyslog.d/50-default.conf

    在 RHEL/CentOS 系统中：

    * /var/log/messages 是主要系统日志

    * /var/log/syslog 可能不存在或为空

    * 配置在 /etc/rsyslog.conf

    查看建议

    * 快速查看系统问题：

        ```bash
        # Debian/Ubuntu
        tail -f /var/log/syslog

        # RHEL/CentOS
        tail -f /var/log/messages
        ```

    * 使用统一工具：

        ```bash
        # 使用 journalctl（systemd 系统）
        journalctl -f

        # 按优先级过滤
        grep -i error /var/log/syslog
        ```

    最佳实践

    * 不要手动删除：使用 logrotate 管理日志轮转

    * 监控关键日志：设置日志监控告警

    * 了解发行版差异：根据系统类型查看相应日志文件

    * 使用集中式日志：生产环境建议使用 ELK、Graylog 等集中日志系统

    在实际运维中，了解所用 Linux 发行版的日志配置习惯非常重要，这能帮助快速定位和解决系统问题。



* bash 中，有时候需要复制当前窗口中的内容，除了用鼠标，还有什么办法

    在 bash 中，除了鼠标操作，还有多种复制终端内容的方法：

    1. 使用键盘快捷键（取决于终端）

        * xclip / xsel（Linux）：

            ```bash
            # 复制命令输出
            ls -la | xclip -selection clipboard

            # 复制文件内容
            xclip -selection clipboard < file.txt
            ```

    2. 终端内置快捷键

        * tmux：

            ```bash
            # 进入复制模式
            Ctrl-b [  # 进入选择模式
            方向键选择文本
            Enter 复制

            # 粘贴
            Ctrl-b ]
            ```

        * screen：

            ```bash
            Ctrl-a [  # 进入选择模式
            空格开始选择，再次空格结束
            Ctrl-a ]  # 粘贴
            ```

    3. 通过管道复制到剪贴板

        ```bash
        # macOS
        ls | pbcopy
        cat file.txt | pbcopy

        # Linux (需要安装 xclip 或 xsel)
        sudo apt install xclip  # Debian/Ubuntu
        echo "文本" | xclip -selection clipboard

        # Windows WSL
        ls | clip.exe
        ```

    4. 使用脚本函数

        在 ~/.bashrc 中添加：

        ```bash
        # 复制到剪贴板
        copy() {
          if [[ "$(uname)" == "Darwin" ]]; then
            pbcopy
          elif [[ "$(uname)" == "Linux" ]]; then
            xclip -selection clipboard
          else
            cat > /dev/null
          fi
        }

        # 使用示例
        ls -la | copy
        echo "内容" | copy
        ```

    5. 选择并复制特定内容

        ```bash
        # 复制最近执行的命令
        history | tail -1 | cut -c 8- | xclip -selection clipboard

        # 复制当前目录路径
        pwd | xclip -selection clipboard

        # 复制文件的前几行
        head -n 10 file.txt | xclip -selection clipboard
        ```

    6. 使用 expect 脚本自动化

        ```bash
        #!/usr/bin/expect
        # 可用于自动化交互式复制
        ```

        实用技巧

        * 组合使用：

            ```bash
            # 复制当前 git 分支名
            git branch | grep "*" | cut -d " " -f2 | xclip -selection clipboard
            ```

        * 检查是否支持：

            ```bash
            # 测试剪贴板工具是否可用
            which pbcopy xclip clip.exe 2>/dev/null | head -1
            ```

    建议：在 ~/.bashrc 中设置别名，方便日常使用。不同系统的剪贴板工具不同，建议根据你的操作系统选择合适的方法。

* linux中，如何按进程显示实时的网络流量？

    在 Linux 中，有几种方法可以按进程显示实时的网络流量：

    1. iftop（按连接显示）

        ```bash
        sudo iftop -P
        ```

        * -P 显示端口和进程信息

        * 按 p 键切换显示端口/进程

        * 按 Shift + P 暂停显示更新

        * 按 Shift + H 显示帮助

    2. nethogs（专门按进程监控）

        ```bash
        sudo nethogs
        ```

        或者指定网卡：

        ```bash
        sudo nethogs eth0
        ```

        快捷键：

        * m：在 KB/s 和 KB 之间切换显示模式

        * r：按流量排序

        * s：按发送流量排序

        * q：退出

    3. bmon + bpfcc-tools

        安装必要工具：

        ```bash
        # Debian/Ubuntu
        sudo apt install bpfcc-tools linux-headers-$(uname -r)

        # RHEL/CentOS
        sudo yum install kernel-devel bcc-tools
        ```

        使用 tcptop 监控：

        ```bash
        sudo tcptop -C 1  # 每秒刷新一次
        ```

    4. ss + awk（组合命令）

        实时查看进程的网络连接：

        ```bash
        watch -n 1 "ss -tup | awk '{print \$1,\$2,\$5,\$6,\$7}' | column -t"
        ```

    5. bmon（带宽监控）

        ```bash
        sudo bmon -p eth0
        ```

        * 按 d 显示详细信息

        * 按 Shift + I 显示每个进程的流量

    6. iptraf-ng（综合网络监控）

        ```bash
        sudo iptraf-ng
        ```

        选择菜单中的相应选项查看进程级流量。

    7. 自定义脚本

        创建一个简单的监控脚本：

        ```bash
        #!/bin/bash
        while true; do
            clear
            echo "=== 进程网络连接统计 ==="
            ss -tup | grep -v "State" | awk '{print $6,$7}' | sed 's/.*pid=//' | sort | uniq -c | sort -rn
            echo -e "\n=== 按进程汇总 ==="
            ss -tup | awk '/pid=/ {split($7,a,"="); print a[2]}' | sort | uniq -c | while read count pid; do
                if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
                    name=$(ps -p $pid -o comm=)
                    echo "$count $pid ($name)"
                fi
            done | sort -rn
            sleep 2
        done
        ```

    推荐方案：

    * 简单易用：使用 nethogs

    * 详细信息：使用 iftop -P

    * 专业监控：使用 bcc-tools 中的工具

    注意事项：

    * 大多数工具需要 root 权限

    * 部分工具可能需要额外安装

    * 对于容器化环境（Docker/K8s），可能需要使用 nsenter 进入容器命名空间查看

* terminal 中，启动 ssh 进入远程机器，然后再退出 ssh 后，local terminal 会出现文本无法自动换行的情况，如何解决

    这是一个常见的终端窗口尺寸同步问题。当你在 SSH 会话中调整了终端窗口大小，退出后本地终端没有正确同步回来。

    解决方法：

    1. 最简单的方法

        直接输入以下命令重置终端：

        ```bash
        reset
        ```

        或者按 Ctrl+L 清屏有时也能解决问题。

    2. 调整窗口大小

        手动触发一次窗口大小调整：

        ```bash
        # 方法1：改变终端窗口大小（拖动边缘即可）
        # 方法2：按 Ctrl+C 中断当前行，然后回车
        # 方法3：输入：
        stty sane
        ```

    3. 在 SSH 会话中正确退出

        ```bash
        # 不要直接按 Ctrl+D 退出，先执行：
        exit
        ```

        或者确保在退出前执行：

        ```bash
        printf '\033[8;%d;%dt' $(stty size)
        ```

    4. 预防措施

        在 ~/.bashrc 或 ~/.bash_profile 中添加：

        ```bash
        # 设置 SSH 退出时重置终端
        trap 'echo -ne "\033]0;${USER}@${HOSTNAME}: ${PWD}\007"' DEBUG
        ```

    5. 如果以上都不行

        关闭当前终端标签/窗口，重新打开一个新的。

    6. 永久解决方案

        在 SSH 配置中启用终端窗口大小转发，编辑 ~/.ssh/config：

        ```text
        Host *
            ServerAliveInterval 60
            TCPKeepAlive yes
            EscapeChar none
        ```

    最常用的是方法1：直接输入 reset 回车即可恢复正常。

* 将 journactl 中的中文日期改成英文

    方法1：临时设置环境变量（当前会话有效）

    ```bash
    # 设置语言为英文
    export lang=en_us.utf-8
    export lc_time=en_us.utf-8

    # 查看日志
    journalctl -u sshd
    ```

    方法2：永久修改系统语言

    ```bash
    # 编辑 locale 配置文件
    sudo vim /etc/locale.conf

    # 添加或修改以下内容
    lang="en_us.utf-8"
    lc_time="en_us.utf-8"

    # 或者使用 sed 命令
    sudo sed -i 's/lang=.*/lang="en_us.utf-8"/' /etc/locale.conf
    echo 'lc_time="en_us.utf-8"' | sudo tee -a /etc/locale.conf

    # 重启系统或重新加载环境
    sudo systemctl restart systemd-journald
    ```

    方法3：journalctl 特定命令选项

    ```bash
    # 使用 --no-full 和特定输出格式
    journalctl -u sshd --since "2023-12-02 21:27:27" --output json

    # 或者使用特定字段显示
    journalctl -u sshd -o json-pretty | grep -e '(message|__realtime_timestamp)'
    ```
    
* less 中实现大小写不敏感的搜索

    1. 启动 less 时设置参数

        ```bash
        less -I filename
        ```

        或

        ```bash
        less -i filename
        ```

        区别：

        * -I：搜索时完全忽略大小写

        * -i：搜索时智能忽略大小写（如果搜索模式包含大写字母，则区分大小写）

    2. 在 less 内部设置

        进入 less 后，可以输入

        `-I`或`-i`   

        来切换大小写敏感设置。

    3. 使用环境变量

        可以在 shell 配置文件中设置默认选项：

        ```bash
        # 在 ~/.bashrc 或 ~/.zshrc 中添加
        export LESS="-I"
        ```

    4. 搜索时指定选项

        在 less 中使用 / 搜索时，可以：

        * 先输入 -i 再按回车，然后进行搜索

        * 或者直接在搜索模式前加 -i：

        ```text
        /-itext
        ```

        这将搜索 "text" 并忽略大小写

    5. 永久配置

        编辑 ~/.lesskey 文件（如果不存在则创建）：

        ```text
        # 设置默认忽略大小写
        -i
        ```

    提示：

    * 在 less 中，按 -i 可以切换大小写敏感模式

    * 当前设置状态会显示在左下角（如果有 -i 标志表示忽略大小写）

    * 按 = 可以查看当前 less 的设置状态

* 在 less 中，无法实现输入 /xx 后不按回车立即搜索。

* Here Document

    Here Document（文档内嵌）是一种在 Shell 脚本中直接嵌入多行文本输入的方法，通常用于命令的标准输入。

    syntax:

    ```bash
    命令 << 分隔符
        多行文本内容
    分隔符
    ```

    分隔符的两种形式

    1. 不带引号的分隔符 - 会进行变量替换和命令替换

        ```bash
        cat << EOF
        当前用户：$USER
        当前目录：$(pwd)
        EOF
        ```

    2. 带引号的分隔符 - 禁用变量和命令替换

        ```bash
        cat << 'EOF'
        当前用户：$USER  # 不会被替换
        当前目录：$(pwd)  # 不会被替换
        EOF
        ```

        (双引号的`"EOF"`可以吗？)

    常见用法示例

    1. 远程执行命令

        ```bash
        ssh user@hostname << 'EOF'
        cd /path/to/dir
        ./script.sh
        echo "任务完成"
        EOF
        ```

    2. 创建配置文件

        ```bash
        cat > /etc/config.conf << EOF
        server_ip=192.168.1.100
        port=8080
        timeout=30
        EOF
        ```

        注：

        1. 重定向`> /etc/config.conf`可以写到命令和参数的中间，记一下这个用法。

    3. 传递复杂命令参数

        ```bash
        mysql -u root -p << EOF
        USE database;
        SELECT * FROM users;
        EXIT;
        EOF
        ```

    4. 使用变量（不转义）

        ```bash
        name="Alice"
        cat << EOF
        Hello $name,
        Welcome to the system.
        EOF
        ```

    高级用法

    * 缩进 Here Document（使用 <<-）
    
        ```bash
        if true; then
            cat <<- EOF
            This line is indented.
            This too.
            EOF
        fi
        ```

        `<<-`允许每行的前面有制表符，在转换为文档时，会删除每行开头的制表符(Tab)。而`<<`会保持原样缩进。

        对于结束标记，`<<`要求必须顶格写，前后无空格。`<<-`则允许`EOF`前面有制表符。

        `<<-`的主要用途是在脚本中保持代码结构美观。

        注意：

        * `<<-`只删除制表符(Tab)，不删除空格，对每行开头的空格无效。

        * `<<-`只删除每行开头的连续制表符。
        
            这意味着如果制表符在空格后面，那么不会被删除。

    * 重定向到文件

        ```bash
        exec > output.log << EOF
        日志开始
        $(date)
        操作完成
        EOF
        ```

    注意事项

    * 分隔符可以是任意字符串，常见的有 EOF、END、STOP 等

    * 结束分隔符必须单独一行，前后不能有空格（除非使用 <<-）

    * 通常与 cat、ssh、mysql、ftp 等需要多行输入的命令配合使用

    * 使用 'EOF' 可以避免脚本中的特殊字符被解释

    这样可以使脚本更清晰，避免使用多个 echo 命令输出多行内容。

* autossh

    Autossh 是一个用于创建持久 SSH 隧道的工具，当连接断开时会自动重连。它通过监控 SSH 连接状态并在断开时重新启动 SSH 会话来确保隧道的稳定性。

    syntax:

    ```bash
    autossh [选项] -M <监控端口> <SSH 命令>
    ```

    * `-M <端口>`: 指定一个监控端口（用于检测连接状态），通常设为 0 让系统自动分配。

    * `-f`: 后台运行。

    * `-N`: 不执行远程命令（仅用于端口转发）。

    * `-L/-R/-D`: 与 SSH 相同的端口转发参数。

    **常用场景示例**

    1. 本地端口转发

        将本地端口 8080 转发到远程服务器的 80 端口：

        ```bash
        autossh -M 0 -f -N -L 8080:localhost:80 user@remote-host
        ```

    2. 远程端口转发

        将远程服务器的 3306 端口转发到本地的 3306 端口：

        ```bash
        autossh -M 0 -f -N -R 3306:localhost:3306 user@remote-host
        ```

    3. 动态 SOCKS 代理

        创建持久的 SOCKS5 代理（本地端口 1080）：

        ```bash
        autossh -M 0 -f -N -D 1080 user@remote-host
        ```

    **高级选项**

    * `-M 0`: 自动选择监控端口（推荐）。

    * `-o ServerAliveInterval=60`: 每 60 秒检测一次连接。

    * `-o ExitOnForwardFailure=yes`: 转发失败时退出。

    * `-o ServerAliveCountMax=3`: 最多重试 3 次。

    示例：

    ```bash
    autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -N -L 9000:localhost:22 user@remote-host
    ```

    **开机自启（Systemd）**

    创建服务文件 /etc/systemd/system/autossh-tunnel.service：

    ```conf
    [Unit]
    Description=AutoSSH Tunnel
    After=network.target

    [Service]
    User=your-username
    ExecStart=/usr/bin/autossh -M 0 -N -L 8080:localhost:80 user@remote-host
    Restart=always

    [Install]
    WantedBy=multi-user.target
    ```

    启用服务：

    ```bash
    sudo systemctl enable --now autossh-tunnel.service
    ```

    **注意事项**

    * 免密登录: 建议配置 SSH 密钥认证，避免输入密码。

    * 防火墙: 确保监控端口和转发端口未被阻塞。

    * 资源占用: 长时间运行时可结合 -o ConnectTimeout=30 优化超时设置。

* tail 的常用选项：

    ```bash
    # 从第N行开始显示
    tail -f -n +50 filename

    # 同时跟踪多个文件
    tail -f file1 file2 file3

    # 高亮显示关键字
    tail -f filename | grep --color=auto "keyword"
    ```

    对于日志文件的特殊处理：

    ```bash
    # 过滤包含特定关键词的行
    tail -f filename | grep "error"

    # 排除特定内容
    tail -f filename | grep -v "debug"

    # 彩色输出
    tail -f filename | ccze -A
    ```

* 实时追踪文本文件的最新内容

    * tail -f (最常用)

        ```bash
        # 基本用法
        tail -f filename

        # 显示行号
        tail -f -n 20 filename

        # 等同于 --follow=descriptor，文件被移动或重命名后仍能跟踪
        tail -F filename
        ```

    * less 的实时模式

        ```bash
        # 打开文件后，按 Shift+F 进入实时跟踪模式
        less filename
        # 然后按 Shift+F 开始跟踪，Ctrl+C 停止跟踪，回到普通浏览模式
        ```

        或者直接使用`less +F /var/log/syslog`

    * multitail (功能更强大)

        ```bash
        # 安装 multitail
        sudo apt install multitail  # Ubuntu/Debian
        sudo yum install multitail  # CentOS/RHEL

        # 使用 multitail
        multitail filename
        ```

        监控多个文件：

        `multitail -i log1.log -i log2.log`

        功能：

            分屏：在同一个终端窗口开多个子窗口监控。

            着色：自动识别日志等级（Error, Warn）并高亮。

            过滤：支持正则过滤，只看你想看的内容。

    * 使用 awk 实时处理

        ```bash
        # 结合 tail 和 awk 进行实时处理
        tail -f filename | awk '{print "New line:", $0}'
        ```

* linux 查看当前目录的大小

    1. du 命令（最常用）

        ```bash
        # 显示当前目录的总大小（人类可读格式）
        du -sh .

        # 显示详细的大小信息（包括子目录）
        du -sh *
        ```

        常用参数：

        * `-s`：汇总，只显示总大小

        * `-h`：人类可读格式（KB、MB、GB）

        * `-c`：显示总计

        * `--max-depth=N`：限制显示层级

    2. 显示当前目录的详细大小信息

        ```bash
        # 显示当前目录及所有子目录的大小
        du -h --max-depth=1

        # 按大小排序显示
        du -h --max-depth=1 | sort -hr
        ```

    3. 使用 ncdu（需要安装，但非常直观）
    
        ```bash
        # 安装ncdu
        sudo apt install ncdu    # Debian/Ubuntu
        sudo yum install ncdu    # CentOS/RHEL

        # 使用ncdu
        ncdu
        ```

    4. 显示磁盘使用情况

        ```bash
        # 查看整个文件系统的使用情况
        df -h .

        # 只显示当前目录所在分区的使用情况
        df -h $PWD
        ```

    5. 其他实用命令

        ```bash
        # 快速查看当前目录大小（以字节为单位）
        du -sb

        # 排除某些文件类型（如排除.log文件）
        du -sh --exclude="*.log" .

        # 仅显示超过特定大小的目录
        du -h --max-depth=1 | grep '[0-9]G\>'  # 显示GB级别的目录
        ```

    主要区别：

    * `du`：计算文件和目录占用的实际磁盘空间

    * `df`：显示文件系统的整体使用情况

    * `ncdu`：交互式磁盘使用分析器，更适合深入分析

    常用组合命令：

    ```bash
    # 查找当前目录下最大的10个文件/目录
    du -ah . | sort -rh | head -10

    # 只显示目录大小（不包括文件）
    du -h --max-depth=1 -t 1M .  # 只显示大于1MB的目录
    ```

    推荐使用：`du -sh .` 这是最简单直接的查看当前目录大小的方法。

* `watch "ps -aux | grep v2ray"`没输出, `watch bash -c "ps -aux | grep v2ray"`也没输出

    尝试了多种方法都未能解决，将这个作为疑难杂症问题长期保存吧

* 使用 bash 启动程序时，单行环境变量要写在脚本前面

    `run.sh`:

    ```bash
    ./$1
    ```

    ```bash
    LD_LIBRARY_PATH=xxx bash run.sh main  # OK

    bash LD_LIBRARY_PATH=xxx run.sh main  # error

    bash run.sh LD_LIBRARY_PATH=xxx main  # error
    ```

    环境变量 LD_LIBRARY_PATH 会传递给 bash 进程，然后在 bash 中执行的脚本（run_main.sh）及其子进程（包括 ./main）都会继承这个变量。

    其他传递环境变量的方法：

    * 使用 export

        ```bash
        export LD_LIBRARY_PATH=/path/to/libs
        bash run_main.sh
        ```

* systemd 与 ssh tunnel

    systemd 中启动 ssh tunnel 时，不要使用`ssh -f`，因为这会

    1. 创建一个 ssh 的前台程序，执行登陆认证等操作，假设其 pid 为 PID_1

    2. 成功登录后，fork 一份进程到后台，此时后台进程的 pid 为 PID_2

    3. 退出 PID_1 的 ssh 前台进程

    systemd 检测到 PID_1 退出，会认为 ssh 进程已经结束，从而导致 systemd 错误判断 service 的状态。

    因此我们直接使用`ssh -NL`或`ssh -NR`就可以。

    example:

    ```conf
    [Unit]
    Description=SSH Reverse Tunnel
    After=network.target

    [Service]
    Type=simple
    User=your_username
    # 使用密钥认证，避免交互
    ExecStart=/usr/bin/ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R 12345:localhost:22 user@remote-server
    Restart=always
    RestartSec=10
    # 密钥权限很重要
    Environment="HOME=/home/your_username"

    [Install]
    WantedBy=multi-user.target
    ```

    ```bash
    # 添加这些选项提高稳定性
    -o ExitOnForwardFailure=yes    # 端口转发失败时退出
    -o ServerAliveInterval=30      # 30秒发送一次保活包
    -o ServerAliveCountMax=3       # 3次无响应后断开
    -o TCPKeepAlive=yes
    -o BatchMode=yes               # 禁用交互提示
    ```

* `xrdp`

    这是一个 独立的、完整的 RDP 服务器 软件。它运行在 Linux 系统上，等待来自 RDP 客户端的连接请求。

    为了提供桌面，`xrdp` 需要依赖一个后端的图形会话管理器。最常见的是：

    * Xvnc：xrdp 会启动一个 VNC 服务器（如 TigerVNC、X11VNC）来承载桌面，然后 xrdp 在 RDP 和 VNC 协议之间进行转换。这是最常见的配置。

    * Xorg：较新的版本支持使用一个专门的 Xorg 会话作为后端（xrdp-xorg 模块），性能比 VNC 模式更好。

    与 FreeRDP 模块的核心区别：

    * xrdp 是一个常驻的系统服务（systemd 服务），监听 3389 端口，允许多个用户建立独立的、全新的桌面会话（登录屏幕 -> 输入用户名密码 -> 进入一个独立的桌面环境）。

    * freerdp2-shadow-x11 是一个临时工具，用于共享已经登录的、正在使用的现有桌面会话。它不提供登录管理器，不创建新会话。

* 查看当前 session 是 x11 还是 wayland

    `echo $XDG_SESSION_TYPE`

    output:

    `wayland`

* FreeRDP2

    FreeRDP 是一个开源的 RDP 客户端和服务器端库。`freerdp2-x11`和`freerdp2-wayland`是它的命令行工具`xfreerdp`的两个后端。

    * `freerdp2-x11`

        这是 RDP 客户端 在 X11 显示系统下的主程序。你用它来连接远程的 Windows 机器或其他 RDP 服务器。

    * `freerdp2-wayland`

        同样是 RDP 客户端，它使用 Wayland 原生接口进行渲染，而不是 X11。

    * `freerdp2-shadow-x11`

        这是 FreeRDP 的 “影子服务器” 或 “桌面共享” 组件。它用于将本机的 X11 桌面会话共享出去，供其他 RDP 客户端连接。

        它捕获当前 X11 显示器的输出，将其作为一个 RDP 会话对外提供。其他用户可以使用任意的 RDP 客户端（如 Windows 自带的 mstsc.exe、Android 客户端、或 xfreerdp 本身）来接入你的当前桌面。

        它不是客户端，而是一个服务端。但它不创建新的桌面会话，只是“投影”现有会话。

        通常通过命令行启动，例如 xfreerdp-shadow-subsystem。

    通常 xfreerdp 可以自动选择后端，但是我们也可手动指定后端：

    ```bash
    # 强制使用 X11 后端（即使在 Wayland 会话中）
    xfreerdp /b:x11 ...

    # 强制使用 Wayland 后端
    xfreerdp /b:wayland ...
    ```

* network.target vs network-online.target

    network.target：

        网络配置完成（接口已配置）

        不保证实际网络连通性

        启动较快

    network-online.target：

        网络真正连通（可以访问外部网络）

        等待 DHCP、DNS 等完全就绪

        启动较慢，可能超时

    所以这两个区别是，一个内网能访问通，一个能访问到公网？

* systemd 服务设置 ssh -R 反向隧道开机自启动

    1. 创建 systemd 服务文件

        ```bash
        sudo nano /etc/systemd/system/ssh-reverse-tunnel.service
        ```

    2. 编辑服务文件内容

        ```conf
        [Unit]
        Description=SSH Reverse Tunnel
        After=network.target

        [Service]
        Type=simple
        User=your_username
        ExecStart=/usr/bin/ssh -N -R remote_port:localhost:local_port username@remote_host -p remote_ssh_port -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new
        Restart=always
        RestartSec=10

        [Install]
        WantedBy=multi-user.target
        ```

    3. 设置和启动服务

        ```bash
        # 重新加载 systemd
        sudo systemctl daemon-reload

        # 设置开机自启动
        sudo systemctl enable ssh-reverse-tunnel.service

        # 立即启动服务
        sudo systemctl start ssh-reverse-tunnel.service

        # 检查服务状态
        sudo systemctl status ssh-reverse-tunnel.service
        ```

    注意事项

        确保网络连通性: 服务在网络就绪后启动

        使用非特权端口: 如果非root用户运行，remote_port通常需要大于1024

        监控日志: 使用 journalctl -u ssh-reverse-tunnel.service -f 查看日志

        安全考虑: 确保远程服务器是可信的，因为反向隧道会暴露本地服务

    注：

    1. `After=network.target`
    
        不写`After=network-online.target`是因为后面有无限重连做保证。

        但是我觉得直接写成`After=network-online.target`更好，直接一步到位。

    1. `ExitOnForwardFailure=yes`

        如果无法端口转发，那么 ssh 连接直接报错。如果远程的端口被其他程序占用，那么 ssh 报错退出。

        ssh 的默认设置是如果端口转发失败（比如远程端口已被占用），SSH 连接仍然会建立，端口转发功能实际上没有工作。

        这个设置配合 systemd 的自动重启服务，可以一定程序上解决远程端口被占用的问题。

    1. `Type=simple`

        SSH 命令会长期运行，保持连接和隧道。进程在前台持续运行，不会立即退出。systemd 会监控这个长期运行的进程。

        如果使用`Type=oneshot`，那么程序执行完就退出。这个配置配合`RemainAfterExit=yes`使用，检查 status 时的效果如下：

        ```
        active (exited)
        ```

        如果不设置`RemainAfterExit=yes`，则会变成

        ```
        inactive
        ```

        使用 simple 的好处：

        * systemd 直接监控主进程

        * 进程退出时自动重启

        * 完整的生命周期管理

        * 简单的日志收集

        * systemctl stop 能正确终止进程

* 把 rdp 的流量包在 ssh 流量里

    通常使用`ssh -L`进行转发：

    `ssh -N -L 本地端口:目标RDP服务器地址:3389 SSH用户名@SSH服务器地址`

    然后使用 xfreerdp 连接本地隧道端口：

    `xfreerdp /v:127.0.0.1:13389 /u:RDP用户名 /p:RDP密码`

    xfreerdp 本身没有原生支持 ssh tunnel 的方法

* `http://security.ubuntu.com/ubuntu/ jammy-security restricted multiverse universe main`的 ip 为`1.1.1.3`，属于 cloudflare 的机器，国内不一定能访问到。

    如果在`apt update`时无法访问这个 ip 的 80 端口，可以考虑在`/etc/apt/source.list`里把这一行注释掉。

* linux 下复制文件时显示进度

    `pv`（Pipe Viewer）可以显示复制进度、速度和估计时间。

    安装：`sudo apt install pv`

    * 复制单个文件
    
        `pv source_file > destination_file`

    * 复制整个目录

        `tar cf - source_dir | pv | tar xf - -C destination_dir`

    可以创建一个带进度复制的脚本：

    `/usr/local/bin/cpv`:

    ```bash
    #!/bin/bash
    tar cf - "$1" | pv | tar xf - -C "$2"
    ```

* ls 按访问时间排序

    ```bash
    ls -ltu
    ```

    -l：长格式显示

    -t：按时间排序（默认是 mtime，但配合 -u 就是按 atime 排序）

    -u：使用访问时间（atime）而不是修改时间（mtime）

* `ls`默认不支持直接按文件创建时间（birth time）排序

    ls 主要显示的是文件的修改时间（mtime）、访问时间（atime）和状态变更时间（ctime）

    较新的内核支持按创建时间排序：

    ```bash
    ls -lt --time=birth
    ```

    -l：长格式显示

    -t：按时间排序（默认是 mtime，但配合 --time=birth 就是按创建时间排序）

    --time=birth：显示并按创建时间排序

* `ls -l`命令将最近修改时间的文件放到最上面

    `ls -lt`

    * -t：按修改时间排序，最新的文件在最前面。

    ls 的其他选项：

    * `-r`：倒序

    * `-a`：显示隐藏文件

* `grep -E`

    主要特点：

    * 支持扩展正则语法：可以使用 |, +, ?, {} 等元字符而无需转义

    * 等同于 egrep：grep -E 与 egrep 命令功能相同

    * 更强大的模式匹配：相比基本正则表达式，提供更丰富的模式匹配能力

    `grep -E "keyword1|keyword2|keyword3" file.txt`: 在 file.txt 文件中搜索包含 keyword1 或 keyword2 或 keyword3 任意一个关键词的所有行。

    ```bash
    # 使用基本正则表达式（需要转义 |）
    grep "keyword1\|keyword2\|keyword3" file.txt

    # 使用扩展正则表达式（更简洁）
    grep -E "keyword1|keyword2|keyword3" file.txt
    ```

    `|`前后不能有空格，如果有空格，那么空格也会被匹配，是 keyword 的一部分。

* `ls -R`

    递归列出目录及其所有子目录中的内容。

    -R 是 “Recursive”（递归）的缩写。

* `ls -lS`

    以长格式列出文件，并按文件大小降序排序（从大到小）。

    -l 是 “long format” 的缩写，会显示详细信息（权限、所有者、大小、修改时间等）。

    -S 是 “Sort by size” 的缩写（注意是大写S）。

* `ls -lr`

    以反向（逆序） 方式列出文件和目录。

    -r 是 “reverse” 的缩写（注意是小写r）。

    组合使用后，它会将排序结果反转。默认情况下（没有其他排序选项），ls -l 是按文件名升序排序（a-z, 0-9），加上 -r 后就变成了降序（z-a, 9-0）。

    与其他排序选项结合时，用于反转排序顺序。例如，ls -lSr 会按文件大小升序排列（从小到大），因为 -S（大小降序）被 -r 反转了。

* 不默认保持中文输入法的另外一个原因

    有时候需要按`shift` + 鼠标滚轮横向滚动，但是`shift`又正好是切换中英文的，这样会导致每横向滚动一次，就在中英文中间切换一次。

* `ulimit -a`

    查看当前用户 Shell 进程及其子进程所能使用的系统资源限制情况。

    ulimit（User Limit）是一个 Shell 内建命令，用于控制和显示用户可用的资源限制。

    -a 是 --all 的缩写，表示 显示所有（All）当前的资源限制设置。

    常见和重要的限制项：

    | 限制项 | 参数 | 示例值 | 含义解释 |
    | - | - | - | - |
    | open files | -n | 1024 | 单个进程能同时打开的最大文件数量。这是最常需要调整的项，比如数据库服务器就需要很高的值。 |
    | max user processes | -u | 7873 | 该用户能同时运行的最大进程数（包括线程）。 |
    | stack size | -s | 8192 | 进程栈的最大大小（KB）。如果程序递归太深可能导致栈溢出，有时需要调整此项。 |
    | core file size | -c | 0 | 核心转储文件（core dump）的最大大小。0 表示禁止生成 core dump 文件。用于程序崩溃调试。 |
    | virtual memory | -v | unlimited | 进程可使用的最大虚拟内存大小。unlimited 表示无限制。 |
    | file size | -f | unlimited | Shell 创建的文件的最大大小。 |

    ulimit 命令本身也可以用来修改限制, 修改通常只对 当前 Shell 会话 有效，退出后即失效。永久修改需要在用户配置文件（如 ~/.bashrc、~/.bash_profile）或系统级配置文件（如 /etc/security/limits.conf）中设置

    example:

    ```bash
    # 将“打开文件数”限制临时改为 2048
    ulimit -n 2048

    # 将“核心文件大小”限制改为无限制
    ulimit -c unlimited
    ```

* `sudo lsof /dev/shm/nccl-AoFK4o`

    用于查看正在使用特定 NCCL 共享内存文件的进程信息。

    lsof: "list open files" - 列出打开文件的工具

    输出列的含义：

        COMMAND: 使用该文件的进程名称（如 python3, train.py 等）

        PID: 进程 ID

        USER: 进程所有者

        FD: 文件描述符（mem 表示内存映射文件）

        TYPE: 文件类型（REG 表示常规文件）

        SIZE/OFF: 文件大小

        NODE: 文件节点号

        NAME: 文件名

* 如果 linux 系统里安装了 systemd，那么可以使用`journalctl -k`查看历史日志

    如果想把新增的日志写入文件，可以使用`dmesg --follow-new | tee <log_file>`

* `sudo mount -t tmpfs -o size=2G tmpfs /dev/shm`

    将 /dev/shm 目录重新挂载为大小为 2GB 的 tmpfs（临时文件系统）。

    这个操作会：

    * 覆盖现有的 /dev/shm 挂载

    * 之前存储在 /dev/shm 中的所有数据都会丢失

    * 新的大小限制会影响所有使用共享内存的应用程序

    默认的 /dev/shm 大小通常是系统物理内存的 50%。

* `df -T /dev/shm`

    显示 /dev/shm 目录所在文件系统的磁盘空间使用情况和文件系统类型信息。

    * df：disk free 的缩写，用于显示文件系统的磁盘空间使用情况

    * -T：选项，显示文件系统类型

    example output:

    ```
    Filesystem     Type  1K-blocks  Used Available Use% Mounted on
    tmpfs          tmpfs   8180620 55444   8125176   1% /dev/shm
    ```

    * 文件系统：tmpfs - 临时文件系统

    * 类型：tmpfs - 基于内存的临时文件系统

    * 1K-块：总容量（以 1KB 为单位）

* `ipcs -m`

    列出当前系统中所有进程间通信（IPC）的资源中， specifically 关于共享内存（Shared Memory） segments 的详细信息。

    ipcs: 是 “Inter-Process Communication Status” 的缩写，即“进程间通信状态”。这是一个用于报告 IPC 设施状态的工具。

    -m: 指定 ipcs 命令只显示与共享内存（Shared Memory） 相关的信息。如果不加任何选项，ipcs 默认会显示消息队列、共享内存和信号量所有三类信息。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ipcs -m

    ------ Shared Memory Segments --------
    key        shmid      owner      perms      bytes      nattch     status      
    0x00000000 2          hlc        600        524288     2          dest         
    0x00000000 7          hlc        606        7881216    2          dest         
    0x00000000 8          hlc        606        7881216    2          dest         
    0x00000000 34         hlc        600        524288     2          dest         
    0x00000000 32811      hlc        606        7881216    2          dest         
    0x00000000 32812      hlc        606        7881216    2          dest 
    ```

    KEY: 创建共享内存段时指定的键值（key），用于进程间找到同一个内存段。0x00000000 通常是私有用途。

    SHMID: 共享内存段的唯一标识符（ID）。在程序中使用这个 ID 来操作特定的内存段。

    OWNER: 创建该内存段的用户。

    PERMS: 权限位（类似文件权限），如 644 表示所有者可读写，组和其他用户只可读。

    BYTES: 该共享内存段的大小（字节）。

    NATTCH: 当前关联（attach）到这个内存段的进程数量。如果为 0，表示没有进程在使用它，但它可能仍然存在系统中。

    STATUS: 状态信息（在某些系统上可能显示更多细节，如被锁定的内存段）。

* `objdump -p <文件名> | grep NEEDED`

    -p 选项：代表显示文件头信息。

* elf dynamic section lookup table

    | 标签名 (Tag) | 十六进制值 (Hex) | 含义简述 |
    | - | - | - |
    | DT_NULL | 0x0 | 标记动态段的结束 |
    | DT_NEEDED | 0x1 | 所需共享库的名称（字符串表偏移） |
    | DT_PLTGOT | 0x3 | 全局偏移表（GOT）和/或过程链接表（PLT）的地址 |
    | DT_HASH | 0x4 | 符号哈希表的地址 |
    | DT_STRTAB | 0x5 | 字符串表的地址 |
    | DT_SYMTAB | 0x6 | 符号表的地址 |
    | DT_INIT | 0xC | 初始化函数的地址 |
    | DT_FINI | 0xD | 终止函数的地址 |
    | DT_SONAME | 0xE | 共享库自身的SONAME（字符串表偏移） |
    | DT_RPATH | 0xF | 库搜索路径（已过时，被DT_RUNPATH取代） |
    | DT_SYMBOLIC | 0x10 | 提示链接器从该库本身开始符号解析 |
    | DT_DEBUG | 0x15 | 用于调试（运行时地址由调试器填充） |
    | DT_TEXTREL | 0x16 | 存在代码段重定位，表明非PIC代码 |
    | DT_JMPREL | 0x17 | PLT重定位条目的地址 |
    | DT_RUNPATH | 0x1D | 库搜索路径 |
    | DT_GNU_HASH | 0x6FFFFEF5 | GNU扩展的哈希表样式 |
    | DT_INIT_ARRAY | 0x19 | 初始化函数指针数组的地址 |
    | DT_FINI_ARRAY | 0x1A | 终止函数指针数组的地址 |

* `objdump -p <文件名> | grep NEEDED`

    * -p 选项： 代表 --private-headers，用于显示文件格式中特定于该文件的“私有”头信息。对于 ELF 格式的文件（Linux 和大多数Unix-like系统上的标准格式），这个选项会显示出程序头表（Program Headers） 和动态段（Dynamic Section） 等关键信息。

    ldd 会实际尝试加载库并模拟运行，在某些不安全的情况下可能执行恶意代码。而 objdump -p | grep NEEDED 是静态分析，只读取文件头信息，因此更安全，尤其是在分析来源不可信的二进制文件时。

    其他常见的标签：

    * 核心依赖与加载相关

        * `NEEDED`

            该文件运行时所依赖的共享库的名称（如 libc.so.6）。一个文件可以有多个 NEEDED 条目。

        * `SONAME`(Shared Object Name)

            仅存在于共享库（.so 文件）中。它包含了该库的共享对象名称。链接器在链接时会把这个名字（而不是文件名）记录到最终的可执行文件中。这就是实现库版本兼容性的关键机制。

            例如，你有一个文件名为 libxyz.so.1.2.3 的库，但其 SONAME 可能是 libxyz.so.1。可执行文件在运行时寻找的将是 libxyz.so.1，而不是具体的 libxyz.so.1.2.3。

        * `RPATH` / `RUNPATH`

            包含一个用冒号分隔的目录列表。动态链接器在查找 NEEDED 库时，会优先在这些目录中搜索，然后再去默认的系统库路径（如 /lib, /usr/lib）中查找。

            RPATH 是较老的属性，其优先级很高。

            RUNPATH 是新标准，其优先级规则不同（在 LD_LIBRARY_PATH 之后查找）。

    * 符号解析相关

        * HASH / GNU_HASH: 指向一个符号哈希表。动态链接器使用这个表来快速查找函数和变量（符号）在库中的地址，极大地加快了动态链接的过程。GNU_HASH 是现代 Linux 系统上更优的格式。

        * STRTAB: 指向字符串表的地址。该表存储了所有动态链接所需的字符串，如符号名、库名等。

        * SYMTAB: 指向符号表的地址。该表包含了所有需要被动态链接的符号（函数名、变量名）的详细信息（名称、值、大小等）。链接器通常结合 HASH 和 STRTAB 来使用它。

        * PLT (Procedure Linkage Table) / PLTGOT (通常显示为 JMPREL): 指向重定位表的地址。这个表包含了所有需要延迟绑定（Lazy Binding）的函数引用信息。这是实现“第一次调用函数时才进行链接”机制的关键。

    * 初始化与终止相关

        * INIT: 指向初始化函数的地址。这个函数（通常命名为 _init）会在该共享库被加载到内存后、任何其他代码执行之前，由动态链接器自动调用。用于完成该库的全局构造和初始化工作。

        * FINI: 指向终止函数的地址。这个函数（通常命名为 _fini）会在该共享库从内存中卸载之前，由动态链接器自动调用。用于完成清理工作（如释放资源）。

            注意：现代代码更推荐使用 `__attribute__((constructor))` 和 `__attribute__((destructor))` 函数属性来代替直接使用 _init 和 _fini 节。

    * 其他重要标签

        * TEXTREL: 这是一个标志。如果存在，表明链接器需要修改代码段（.text段）的权限（例如将其设为可写）以便进行重定位。这通常意味着共享库不是用 -fPIC 选项编译的（位置无关代码），会带来安全性和性能上的损失。看到这个标志通常不是好事情。

        * FLAGS / FLAGS_1: 一些特殊的标志位。例如 FLAGS_1 中的 PIE 标志表示该可执行文件是位置无关的可执行文件（Position-Independent Executable），这是现代Linux系统上ASLR（地址空间布局随机化）的基础。

        * DEBUG: 这是一个占位符，用于调试信息，通常没有运行时语义。

    ref: 
    
    1. <https://docs.oracle.com/cd/E53394_01/html/E54813/chapter6-42444.html>

    1. `man 5 elf`

    1. <https://www.gnu.org/software/binutils/>

* `od -A`

    od -A 选项用于指定输出偏移量（地址）的显示格式。这里的 -A 代表 "Address"。

    * `-A x`: 十六进制（hexadecimal）
    
        `0000000`, `0000010`
    
    * `-A d`: 十进制（decimal）
    
        `0000000`, `0000016`

    * `-A o`: 八进制（octal）, 默认情况
    
        `0000000`, `0000020`

    * `-A n`: 不显示偏移量（none）
    
        （左侧偏移量栏为空）

* 创建`tmpfs`类型的目录

    ```bash
    sudo mkdir /mnt/shm
    sudo mount -t tmpfs -o size=2G tmpfs /mnt/shm
    ```

    解释：

    * `-t tmpfs`: 挂载`tmpfs`类型的文件系统。`tmpfs`表示使用 ram，如果 ram 不够，则使用 swap

    * `-o size=2G`：限制目录最大大小为 2G

    * `tmpfs`: 这是“源设备”参数。对于像`tmpfs`这样的虚拟文件系统，这个位置通常就填写文件系统类型本身。

    * `/mnt/shm`: 挂载点（mount point）

    特点：

    * 存储在内存中：所有存放在 /mnt/shm 目录下的文件和目录都位于高速的 RAM 中，因此读写速度非常快。

    * 临时性：这是一个临时存储。当系统重启、崩溃或你手动卸载（umount /mnt/shm）这个文件系统时，其中的所有数据都会消失。

    * 动态分配：tmpfs 只会实际占用它已存储数据大小的内存。例如，如果你创建了一个 100MB 的文件，tmpfs 就只占用约 100MB 的 RAM（和少量元数据开销），而不是一开始就占满 2GB。它会根据存储内容的增加而动态增长，但最大不会超过 size 参数的限制（2GB）。

    * 可能使用交换空间（Swap）：如果系统内存不足，tmpfs 中不活跃的数据可能会被换出到硬盘的交换分区（swap）上，从而释放物理内存。这意味着它的大小可以超过物理 RAM，但性能会下降。

* `sudo mount -o remount /dev/shm`

    重新挂载 /dev/shm 文件系统，并在此过程中应用或刷新其挂载选项。

    如果没有在命令中指定新的挂载选项（例如 size=2G 或 noexec），那么它将使用系统默认的或之前在 /etc/fstab 文件中配置的选项来重新挂载。

    常见使用场景

    * 应用 /etc/fstab 中的新配置：

        如果你修改了 /etc/fstab 中关于 /dev/shm 的配置（比如改变了大小限制 size=512M），你可以运行此命令来立即应用新的配置，而无需重启系统。

    * 修复权限或属性问题：

        如果 /dev/shm 的权限意外被更改（例如，某个脚本错误地执行了 chmod 700 /dev/shm），导致某些程序无法正常使用共享内存，通过重新挂载可以将其恢复为正确的默认权限。

    * 清除所有内容（不常用）：

        虽然 remount 本身不是为了清除数据，但结合某些选项（如 noexec 然后再 remount 回默认值）可以间接达到目的。更直接的方法是直接重启（数据会丢失）或手动删除其中的文件。

* `truncate -s <bytes>`

    如果文件不存在, 创建指定大小的空文件，内容全部用空字节（\0）填充。如果文件已存在, 若文件原大小 > 指定大小，则截断文件，丢弃超出部分；若文件原大小 < 指定大小，则扩展文件，用空字节填充新增部分。

    example:

    `truncate -s 4096 /dev/shm/my_tmp_file`

    在`/dev/shm`中创建大小为 4096 字节的文件`my_tmp_file`。

* `od -t x<N>`

    按`<N>`字节一组，打印十六进制数据。

    其中`<N>`可以取值 1, 2, 4, 8，如果不指定`<N>`，则默认取`2`。

    注意多字节显示时，输出受字节序的影响，比如单字节显示的`01 02`，使用小端序 + `-x2`显示时可能变成`0201`。

* `objdump`

    主要用于反汇编和分析目标文件及可执行文件。

    常用功能：

    1. 反汇编

        将二进制可执行文件或目标文件（.o, .exe, .so, .dll 等）中的机器代码转换回汇编语言代码。

        `objdump -d ./my_program`

        * `-d`选项表示反汇编包含指令的节（section）。

    2. 查看目标文件结构

        显示文件的头部信息和各个节（Section）的详细信息。包括文件的格式（如ELF、PE）、入口地址、节的大小和位置等。

        `objdump -h ./my_program`
        
        * `-h`选项显示节的头部摘要。

    3. 查看符号表

        列出文件中定义和引用的所有符号（如函数名、全局变量名）。

        `objdump -t ./my_program`

        * `-t`选项显示符号表。

    4. 查看文件头信息

        显示二进制文件的元数据，例如目标架构（如x86-64、ARM）、操作系统ABI、文件类型（可执行、共享库等）和入口点地址。

        `objdump -f ./my_program`

        * `-f`选项显示文件头信息。

    5. 以十六进制格式查看文件内容

        除了反汇编，objdump 还可以直接显示文件的十六进制和ASCII表示，类似于 hexdump 或 xxd 命令。

        `objdump -s -j .text ./my_program`

        * `-s` 显示所有节的内容。

        * `-j` 指定只显示某个节（如 .text 节）的内容。

    6. 查看动态链接信息

        对于动态链接的可执行文件或共享库，可以显示其依赖的共享库（如Linux下的 .so 文件）以及动态符号表。

        `objdump -p ./my_program`

        * `-p`显示与动态链接相关的信息（在 ELF 文件中，这类似于`readelf -d`命令）。

* `readelf -l /bin/bash | grep interpreter`

    查找 /bin/bash 可执行文件所使用的动态链接器（interpreter）的路径。

    output:
    
    ```
          [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
    ```

    这表示：

    * `/bin/bash`依赖于动态链接器 /lib64/ld-linux-x86-64.so.2 来加载运行所需的共享库（如 libc.so）。

    * 系统内核在执行 /bin/bash 时，会先加载这个指定的动态链接器，再由它处理后续的库依赖和符号解析。

* `readelf`的用法

    用于显示关于 ELF (Executable and Linkable Format) 格式目标文件的信息。

    ELF 是现代 Linux 系统上的一种文件格式，用于：

    * 可执行文件 (例如编译生成的`a.out`)

    * 共享库 (例如：libc.so.6)

    * 目标文件 (例如：file.o)

    * 核心转储文件 (core dumps)

    常用选项：

    * 查看文件头 (`-h`)

        这是最常用的选项之一。它显示了 ELF 文件的概要信息，包括：

        * 文件类型（可执行文件、共享库、目标文件等）

        * 目标机器的体系结构（如 x86-64, ARM）

        * 程序的入口点地址（Entry point）

        * 程序头表（Program Headers）和节头表（Section Headers）的起始位置和大小。

    * 查看节头信息 (`-S`)

        显示文件中所有的 “节” (Sections) 的信息。节是 ELF 文件的重要组成部分，例如：

        * `.text`： 存放可执行代码。

        * `.data`： 存放已初始化的全局变量和静态变量。

        * `.bss`： 存放未初始化的全局变量和静态变量。

        * `.rodata`： 存放只读数据（如字符串常量）。

        * `.symtab`： 符号表。

        * `.strtab`： 字符串表。

    * 查看程序头信息 (`-l`)

        显示 “段” (Segments) 或称为 程序头 (Program Headers) 的信息。段告诉操作系统或动态链接器如何将文件加载到内存中并执行。这对于理解程序运行时布局至关重要。

    * 查看符号表 (`-s`)

        显示文件中定义和引用的所有符号，如函数名、变量名。这对于解决“未定义引用”等链接错误非常有用。

    * 查看动态段信息 (`-d`)

        对于动态链接的可执行文件或共享库，此选项显示其依赖的共享库（如 libc.so.6）以及动态链接器需要的其他信息（如重定位信息、符号表地址等）。

    * 查看重定位信息 (`-r`)

        显示文件中需要重定位的条目信息，这在分析目标文件(.o)时尤其有用。

    * 查看节的内容 (`-x` 或 `-p`)

        以十六进制或其他格式转储指定节的具体内容。

    example:

    `readelf -h main`

    output:

    ```
    ELF Header:
      Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
      Class:                             ELF64
      Data:                              2's complement, little endian
      Version:                           1 (current)
      OS/ABI:                            UNIX - System V
      ABI Version:                       0
      Type:                              DYN (Position-Independent Executable file)
      Machine:                           Advanced Micro Devices X86-64
      Version:                           0x1
      Entry point address:               0x1060
      Start of program headers:          64 (bytes into file)
      Start of section headers:          14048 (bytes into file)
      Flags:                             0x0
      Size of this header:               64 (bytes)
      Size of program headers:           56 (bytes)
      Number of program headers:         13
      Size of section headers:           64 (bytes)
      Number of section headers:         31
      Section header string table index: 30
    ```

* `at`

    安排一个任务在未来的某个特定时间点执行一次。

    ubuntu 系统中默认不安装`at`，需要手动安装：`sudo apt install at`

    syntax:

    ```bash
    at [选项] <时间设定>
    ```

    常用选项：

    * `-f <文件>`：从指定的文件中读取要执行的命令，而不是交互式输入。

        示例： `at -f /path/to/script.sh 10:00 AM`

    * `-l`：列出当前用户的所有待处理作业（等同于 atq 命令）。

        示例： at -l

    * `-r <作业ID>`或`-d <作业ID>`：删除一个指定的待处理作业（等同于 atrm 命令）。

        示例： at -r 5 (删除作业ID为5的任务)

    * `-m`：即使命令没有输出，也在任务完成后给用户发送邮件。

    * `-M`：即使命令有输出，也不给用户发送邮件。

    * `-t <时间>`：使用`[[CC]YY]MMDDhhmm[.ss]`格式指定时间（较不常用）。

        示例： `at -t 202507191430.00 (2025年7月19日14点30分00秒)`

    * `-v`：显示任务将被执行的时间，而不是仅仅将任务排入队列。

    * 时间设定

        * 简单时间格式：

            HH:MM - 在今天的指定时间执行。如果该时间已过，则顺延到明天。

                示例： at 14:30

            midnight, noon, teatime (通常是下午4点) - 使用预定义的单词。

        * 相对时间格式：

            now + <数量> <时间单位> - 从现在开始的一段时间后执行。

                时间单位： minutes, hours, days, weeks

                示例：

                    at now + 10 minutes (10分钟后)

                    at now + 2 hours (2小时后)

                    at now + 1 week (1周后)

        * 绝对时间格式：

            MMDDYY, MM/DD/YY, DD.MM.YY - 指定具体日期。

                示例：

                    at 11:00 PM 072025 (2025年7月20日晚上11点)

                    at 10am Jul 20 (7月20日上午10点)

                    at 2:00 tomorrow (明天凌晨2点)

        * 组合日期和时间：

            你可以将日期和时间组合起来。

                示例： at 3:30 PM next Friday (下周五下午3点半)

    example:

    * `at now + 10 minutes` (10分钟后)

    * `at 11:00 PM` (今晚11点)

    * `at 2:00 tomorrow` (明天凌晨2点)

    * `at 10:00 Jul 20` (7月20日上午10点)

    标准输入/交互式输入：通常你直接运行 at <TIME>，然后在其提示符下输入要执行的命令，按 Ctrl+D 结束输入。

    从文件读取：也可以使用 at -f script.sh <TIME> 来指定一个脚本文件在特定时间运行。

    examples:

    * 安排一个简单任务

        ```bash
        # 安排一个任务在下午2：30执行
        at 2:30 PM
        # 进入at提示符后，输入命令：
        at> echo "Hello from the past!" > ~/reminder.txt
        # 输入完毕后，按 Ctrl+D 来保存任务
        ```

    * 安排一个不久后的任务

        ```bash
        # 安排一个20分钟后的任务
        at now + 20 minutes
        at> shutdown -h now # 例如：20分钟后关机
        at> <EOT> # 按 Ctrl+D
        ```

    * 非交互模式

        ```bash
        # 将一个脚本文件中的命令安排在下午5点执行
        $ at -f /path/to/daily_cleanup.sh 5:00 PM
        job 7 at Tue Jul 18 17:00:00 2025
        ```

    * 使用管道（echo）

        ```bash
        # 通过管道将命令传递给at
        $ echo "shutdown -h now" | at now + 30 minutes
        job 8 at Tue Jul 18 15:45:00 2025
        ```

    `atq`：列出当前用户所有等待执行的 at 任务队列（root用户可以看到所有任务）。

    `atrm <job_id>`：删除一个已排队的 at 任务。例如 atrm 3 会删除作业编号为3的任务。

    注意事项:

    * 守护进程：at 命令的运行依赖于 atd 这个守护进程。请确保它正在运行（通常通过 systemctl status atd 检查）。

    * 执行环境：at 任务会在一个几乎最小化的 shell 环境中运行，只继承少量环境变量（如TERM, PATH, SHELL等）。如果你的命令依赖于特定的环境变量（如DISPLAY用于图形界面），可能需要在其内部重新设置。

    * 输出处理：任务的标准输出和标准错误默认会通过电子邮件发送给任务所有者（即安排任务的用户）。为了避免收邮件，最好在命令中明确重定向输出（如 >/dev/null 2>&1 或重定向到日志文件）。

    `at`命令中的时间指示符（如 AM, PM, tomorrow, midnight 等）是大小写不敏感的（case insensitive）。

    在交互环境中输入命令时，无法使用 tab 自动补全（比如文件名，目录下的文件等）。

    `at`指定的 command 执行后，无法随时中止，必须等 command 结束。

* `nsupdate`

    用于动态、增量地更新 DNS 域名。它允许你在不手动编辑 zone file（区域文件）和重启 DNS 服务的情况下，直接添加、修改或删除 DNS 记录。

    主要功能和用途

    * 动态 DNS (DDNS)

        这是 nsupdate 最经典的用途。很多家庭宽带或办公网络的公网 IP 地址是动态变化的。你可以在路由器或电脑上运行一个脚本，当检测到 IP 变化时，自动使用 nsupdate 命令向 DNS 服务器发送更新请求，将你的域名（如 home.example.com）快速指向新的 IP 地址。这样你始终可以通过一个固定的域名访问到动态 IP 的设备。

    * 自动化运维和脚本集成

        在自动化部署（CI/CD）、云基础设施管理中，经常需要批量创建或销毁服务器。这些流程可以通过脚本调用 nsupdate，自动为新服务器注册 DNS 记录，或者在下线时清理记录，实现全自动化管理。

    * 快速故障恢复

        如果需要将服务从一个 IP 迁移到另一个 IP，使用 nsupdate 可以几乎实时地更新 DNS 记录，大大缩短故障切换时间（RTO），比手动修改zone file并等待复制要快得多。

    工作原理

    nsupdate 并不直接操作 DNS 服务器上的配置文件。它的工作流程是：

    1. 连接：nsupdate 会连接到目标 DNS 服务器（通常是 BIND 9）的 53 端口，并使用 TSIG（Transaction SIGnature） 密钥进行身份验证。TSIG 确保了更新的安全性和合法性，防止任何人随意修改你的 DNS 记录。

    2. 发送更新指令：在交互式命令行或通过管道输入中，你发送一系列指令，例如：

        * `update add www.example.com 3600 A 192.0.2.1` （添加一条 A 记录）

        * `update delete oldhost.example.com A` （删除一条 A 记录）

    3. 服务器处理：DNS 服务器验证你的权限后，会在内存中直接更新它的 zone 数据，并递增该区域的序列号（SOA Serial）。这个更改会立即生效，并通知其他从服务器进行区域传输（AXFR/IXFR）以同步更新。

    example:

    假设我们有一个密钥key文件，用来向DNS服务器 dns.example.com 认证，并更新 example.com 域。

    ```bash
    # 启动 nsupdate 并指定服务器
    nsupdate -k /path/to/mykey.key

    # 在出现的交互提示符下输入命令
    > server dns.example.com
    > zone example.com
    > update add newserver.example.com 300 A 203.0.113.10
    > send
    > quit
    ```

    这条命令成功执行后，域名 newserver.example.com 就会立刻指向 203.0.113.10。

* `od -t x1`

    以十六进制（HEX）字节的形式，逐个字节地显示文件或输入流的内容。

    * `-t` (`--format`) : 用于指定输出数据的格式。它告诉 od 如何解释和显示文件中的字节。

    * `x1`: `x`表示 16 进制，`1`代表每个输出单元的大小是 1 个字节。

    example:

    ```
    0000000 7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
    0000020 03 00 3e 00 01 00 00 00 10 0a 00 00 00 00 00 00
    0000040 40 00 00 00 00 00 00 00 38 4e 00 00 00 00 00 00
    ...
    ```

* `/etc/ld.so.conf.d/myapp.conf`

    将一个自定义的库路径（例如`/opt/myapp/lib`）添加到系统中所有应用程序的共享库搜索路径中。`ld-linux.so`会搜索`/etc/ld.so.conf.d`这个目录下的所有配置。

    example:

    ```conf
    /opt/myapp/lib
    ```

    创建或修改配置文件后，必须运行 ldconfig 命令来重建共享库缓存，使更改立即生效:

    ```bash
    sudo ldconfig
    ```

    这个命令会读取所有`/etc/ld.so.conf.d/`下的配置文件和`/etc/ld.so.conf`，生成一个快速的缓存文件`/etc/ld.so.cache`。动态链接器实际使用的是这个缓存文件来加速查找。

    主配置文件 /etc/ld.so.conf 通常会包含一行：`include /etc/ld.so.conf.d/*.conf`。

* `host`

    一个 DNS 查询工具

    常见用法:

    * 查询域名对应的 IP

        `host www.example.com`

        输出该域名的 A/AAAA 记录。

        example:

        ```bash
        host www.google.com
        ```

        output:

        ```
        www.google.com has address 142.250.71.196
        www.google.com has IPv6 address 2404:6800:4005:816::2004
        ```

    * 查询 IP 对应的域名（反向解析）

        `host 8.8.8.8`

        输出 PTR 记录。

        example:

        ```bash
        host 8.8.8.8
        ```

        output:

        ```
        8.8.8.8.in-addr.arpa domain name pointer dns.google.
        ```

    * 指定查询记录类型

        ```bash
        host -t MX example.com   # 查询邮件服务器记录
        host -t NS example.com   # 查询域名服务器
        ```

    * 指定 DNS 服务器

        ```bash
        host www.example.com 8.8.8.8
        ```

        使用 Google DNS 来解析。

        (如果想取消这个映射，该怎么办？)

* `command_1 && command_2 &`里，`&`作用于`(command_1 && command_2)`。

 * `pkg-config --libs`

    获取链接一个特定软件库（Library）时，需要传递给编译器的所有链接器（Linker） flags（参数）。

    当你编译程序需要用到某个第三方库（比如 OpenCV、GTK、libpng 等）时，这个命令会告诉你应该加哪些 -l 和 -L 参数。

    example:

    ```bash
    pkg-config --libs mpi
    ```

    output:

    ```
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
    ```

    pkg-config 工具通过查询事先安装好的 .pc 配置文件（这些文件通常随库一起安装），自动为你生成正确的、适用于当前系统的编译参数。

    通常与 Makefile 结合起来使用：

    ```makefile
    # 编译 main.c 并链接 libcurl 库
    program: main.c
        gcc main.c -o program `pkg-config --libs --cflags libcurl`
    ```

    其中`--cflags`表示输出`-I<header_path>`

    * `pkg-config --libs <包名>`: 获取链接库所需的参数
    
        `-L/usr/lib -lcurl -lssl -lcrypto`

    * `pkg-config --cflags <包名>`: 获取编译时所需的参数（如头文件路径）
    
        `-I/usr/include/curl`

* `ldconfig`

    创建、更新和维护共享库的缓存和必要的链接，通过生成缓存文件 /etc/ld.so.cache 来加速程序启动时寻找共享库的过程，并管理共享库的符号链接，以便系统能够快速地找到并加载运行程序时所需的共享库（.so 文件）

    当你在Linux系统上运行一个程序时，动态链接器（ld.so 或 ld-linux.so）负责在程序启动前将它所依赖的所有共享库加载到内存中。

    动态链接器不会每次都去慢吞吞地扫描整个硬盘上的所有目录来寻找库文件，那样太慢了。相反，它直接去查询由 ldconfig 预先准备好的缓存文件 /etc/ld.so.cache。这个缓存就像一本书的索引，让动态链接器能瞬间找到所需的库。

    在执行了以下操作后，需要手动执行`sudo ldconfig`:

    * 安装了新的共享库（例如通过 make install 或手动拷贝 .so 文件到系统目录）。

    * 删除了已有的共享库。

    * 修改了共享库的搜索路径（例如编辑了 /etc/ld.so.conf 或在该目录下增加了新的 .conf 文件）。

    apt 等软件安装了新的包后，会自动调用`ldconfig`。

    更新缓存: `sudo ldconfig`

    显示缓存内容: `ldconfig -p`

* `ldd`

    List Dynamic Dependencies

    主要功能：

    * 列出依赖：显示运行某个程序需要哪些共享库（.so 文件）。

    * 定位库文件：显示这些依赖库在文件系统中的具体路径。

    * 检查兼容性：检查是否存在某个依赖库，以及库的版本是否兼容。如果某个库找不到，它会显示 not found，这通常是程序无法启动的原因。

    syntax:

    ```bash
    ldd [选项] <文件名>
    ```

    ldd 本身并不是一个可执行程序，而是一个Shell脚本

    它的工作原理是设置适当的环境变量（如 LD_TRACE_LOADED_OBJECTS），然后调用目标程序。目标程序在这种特殊环境下启动时，并不会真正运行其主逻辑，而是会列出其所有依赖的共享库信息后退出。

* `vncviewer`

    使用方法：

    * `vncviewer 192.168.1.100`

    * `vncviewer 192.168.1.100:5901`

* dig 命令包含在`bind-utils`软件包中

* 查看 glibc 版本：`ldd --version`

* `od -c <file>`

    以 字符形式（character）显示文件内容，不可打印字符会用转义符（如`\n`, `\t`, `\0`）或八进制表示。

* `od -x`

    od: 是 Octal Dump 的缩写。这个名字源于其最初的主要功能是以八进制（Octal）格式显示文件内容。虽然现在它支持多种格式，但名字保留了下来。

    以十六进制（Hex）格式显示文件的内容.

    `-x`代表输出为“十六进制。
    
    example:

    `od -x msg.txt`

    output:

    ```
    0000000 6568 6c6c 2c6f 7720 726f 646c 202c 696e
    0000020 6168 2c6f 7a20 6961 696a 6e61 202c 6568
    0000040 6568 202c 6168 6168 000a
    0000051
    ```

    * 最左边的一列: 偏移量地址. 表示当前行数据在文件中的起始位置（偏移量），默认以八进制数显示。

    * 中间的多列: 数据内容. -x 选项规定每2个字节（16位）为一组进行显示。
    
        由于计算机的字节序（Endianness）问题，od -x 在显示时使用的是主机本身的字节序（对于大多数x86架构的电脑是小端序）。

        如果文件中连续的2个字节是 0x61 和 0x62（即字符 'a' 和 'b' 的ASCII码），在小端序机器上，od -x 会将其显示为一组：6261（即 0x62 在高位，0x61 在低位）。

    * 如果使用 -x 的同时再加上 -c 选项（即 od -xc），输出还会在最右边增加一列，显示数据对应的ASCII字符。不可打印的字符会显示为转义序列（如 \n）或问号（?）。

        比较神奇的是 ascii 字符会按 2 个字节的间隔自动把顺序倒过来：

        `od -xc msg.txt`:

        ```
        0000000    6568    6c6c    2c6f    7720    726f    646c    202c    696e
                  h   e   l   l   o   ,       w   o   r   l   d   ,       n   i
        0000020    6168    2c6f    7a20    6961    696a    6e61    202c    6568
                  h   a   o   ,       z   a   i   j   i   a   n   ,       h   e
        0000040    6568    202c    6168    6168    000a
                  h   e   ,       h   a   h   a  \n
        0000051
        ```

        可以看到，本来`e = 0x65`，`h = 0x68`，但是在显示的时候 od 自动把顺序纠正了。

* `ps aux`

    ps: 进程状态（Process Status）

    * `a`： 显示所有用户的进程（而不仅仅是当前用户的）。

    * `u`： 以面向用户的格式显示，这会提供更详细的信息（如 CPU、内存占用率、用户等）。

    * `x`： 列出没有控制终端的进程。这很重要，因为很多系统守护进程（后台服务）是不依赖于终端的。加上 x 才能看到所有这些后台进程。

    输出的关键列的含义：

    | 列名 |全称 | 含义 |
    | - | - | - |
    | `USER` | User | 进程的所有者（是哪个用户启动的） |
    | `PID` | Process ID | 进程的唯一ID号，用于识别和管理进程 |
    | `%CPU` | CPU Percentage | 进程占用CPU的百分比 |
    | `%MEM` | Memory Percentage | 进程占用物理内存的百分比 |
    | `VSZ` | Virtual Set Size | 进程使用的虚拟内存大小（单位：KB） |
    | `RSS` | Resident Set Size | 进程使用的、未被换出的物理内存大小（单位：KB） |
    | `TTY` | Teletypewriter | 进程是在哪个终端上运行的。? 表示不是从终端启动的。 |
    | `STAT` | Process State | 进程状态码（非常重要），例如：<br>- `R`： 正在运行或可运行<br>- `S`： 可中断的睡眠状态（等待事件完成）<br>- `D`： 不可中断的睡眠（通常与IO有关）<br>- `Z`： 僵尸进程（已终止但未被父进程回收）<br>- `T`： 已停止（通常由信号控制） |
    | `START` | Start Time | 进程启动的时间 |
    | `TIME` | CPU Time | 进程实际使用CPU运行的总时间 |
    | `COMMAND` | Command | 启动该进程所用的命令行名称 |

* `timeout`

    如果计时器超时, ，timeout 会向子命令进程发送一个信号（默认是 TERM 信号），强制终止它, 并返回一个特定的退出状态码（默认为124）

    syntax:

    ```
    timeout [OPTIONS] DURATION COMMAND [ARG]...
    ```

    * DURATION：时间长度。这是必须指定的参数。它可以是一个整数，也可以是带单位的数字（例如 10s 表示10秒，2m 表示2分钟，1h 表示1小时）。如果只是数字，默认单位是秒。

    * COMMAND [ARG]...：你想要运行的实际命令及其参数。

    常用选项（OPTIONS）：

    * -s, --signal=SIGNAL：指定超时后要发送的信号。默认是 TERM (15)。

        例如，如果程序忽略 TERM 信号，你可以使用 -s KILL 来发送 KILL (9) 信号，强制杀死进程。

        查看所有信号：kill -l

    * -k, --kill-after=DURATION：双重保险。先发送 -s 指定的信号（默认TERM），如果再过 DURATION 时间后进程仍然存在，则发送 KILL 信号确保其被杀死。

    * --preserve-status：让 timeout 返回被它控制的命令的退出状态码。如果命令超时被杀死，仍然返回124。

    example:

    ```bash
    timeout 5s ping example.com
    timeout -s KILL 2m some-unstable-script.sh
    timeout -k 5s 10s some-command
    ```

    在脚本中使用并检查退出状态：

    ```bash
    #!/bin/bash
    timeout 30s long-running-task.sh

    case $? in
      0)  echo "Task completed successfully." ;;
      124) echo "Task timed out and was killed." ;;
      137) echo "Task was killed by a signal (e.g., KILL)." ;; # 如果用了 -s KILL，会返回137 (128+9)
      *)  echo "Task failed with exit code: $?" ;;
    esac
    ```

* `sync`

    `sync`可以作为一个命令使用，效果和调用`sync()`相同。

* `od`命令

    od（Octal Dump），用于以各种格式显示文件的内容，通常用于查看或诊断文件中那些不可打印的字符（如控制字符、换行符、空字符等）。

    * `od -c`: 将文件的每个字节（byte）解释为 ASCII 字符或转义序列，并以更可读的形式输出。

    example:

    `msg.txt`:

    ```
    hello, world
    nihao
    zaijian
    ```

    `od -c msg.txt` output:

    ```
    0000000   h   e   l   l   o   ,       w   o   r   l   d  \n   n   i   h
    0000020   a   o  \n   z   a   i   j   i   a   n  \n
    0000033
    ```

    前面的偏移是 8 进制。

* strace

    strace 可以记录一个进程在运行时与内核之间的所有系统调用（system calls）和接收到的信号（signals）

    example:

    ```bash
    strace ls
    ```

    常用方式（未验证）：

    * `strace <command>`：跟踪一个命令的执行。

    * `strace -p <pid>`：附着到一个正在运行的进程上进行跟踪。

    * `strace -e trace=<type>`：只跟踪特定类型的系统调用，如 strace -e trace=open,read ls 只跟踪 open 和 read 调用。

    * `strace -o output.txt`：将输出重定向到文件，便于分析。

    * `strace -c`：在程序结束后生成一个统计报告，显示各个系统调用的次数、时间和错误。

* `ss`

    Socket Statistics，可以查看当前的 tcp, udp 连接。

    ss 直接从内核空间获取信息，而 netstat 需要遍历许多 /proc 下的文件，效率较低。（未验证）

    netstat 在很多 Linux 发行版中已被标记为“已废弃”，官方推荐使用 ss 和 ip 命令（未验证）

    常用选项：

    * `-t`: 显示 tcp 连接

    * `-u`：显示 udp 连接

    * `-l`：显示 listening socket

    * `-a`：显示所有连接，包括监听和非监听的

    * `-p`：显示对应的进程

    * `-n`：以数字形式显示端口和地址，不尝试解析域名和服务名

    常用组合：

    `ss -tln`

    output:

    ```
    State   Recv-Q   Send-Q     Local Address:Port      Peer Address:Port  Process  
    LISTEN  0        64               0.0.0.0:43673          0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:35661          0.0.0.0:*              
    LISTEN  0        64               0.0.0.0:2049           0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:52865          0.0.0.0:*              
    LISTEN  0        128              0.0.0.0:22             0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:111            0.0.0.0:*              
    LISTEN  0        128            127.0.0.1:631            0.0.0.0:*              
    LISTEN  0        4096           127.0.0.1:39255          0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:45865          0.0.0.0:*              
    LISTEN  0        4096       127.0.0.53%lo:53             0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:46147          0.0.0.0:*              
    LISTEN  0        32         192.168.122.1:53             0.0.0.0:*              
    LISTEN  0        64                  [::]:2049              [::]:*              
    LISTEN  0        128                [::1]:631               [::]:*              
    LISTEN  0        64                  [::]:44879             [::]:*              
    LISTEN  0        128                 [::]:22                [::]:*              
    LISTEN  0        4096                [::]:111               [::]:*              
    LISTEN  0        4096                [::]:55635             [::]:*              
    LISTEN  0        4096                [::]:54891             [::]:*              
    LISTEN  0        4096                [::]:38721             [::]:*              
    LISTEN  0        4096                [::]:54351             [::]:*
    ```

    过滤连接（未验证）：

    * 查看所有已建立的 TCP 连接：

        `ss -t state established`

    * 查看所有处于 TIME-WAIT 状态的连接（通常是大量短连接后出现的状态）：

        `ss -t state time-wait`

    * 查看所有已建立连接的 IPv4 HTTP (端口 80) 连接：

        `ss -t '( dport = :http or sport = :http )'`

    * 或者使用数字端口：

        `ss -t '( dport = :80 or sport = :80 )'`

* `wc -l [file_1] [file_2] [file_3] ...`

    统计给定文件或输入流中的行数，即统计`\n`的数量。

    word count

    * `-w`: 统计单词数

    * `-c`：统计字节数

    * `-m`：统计字符数

* `read -n1`

    从标准输入中读取一个（且仅一个）字符，然后立即结束读取，无需用户按回车键

    注意这里`-n`后跟的是一，不是`l`。相当于`read -n 1`。

    `-n N`: 读取 N 个字符后立即返回，不再等待回车键

* `ed`

    `ed`是个很老的文本编辑器。

    example:

    `test.txt`:

    ```
    Hello world
    This is a test.
    Goodbye
    ```

    我们想将 “world” 替换为 “universe”。

    ```
    # 1. 用 ed 打开文件
    ed test.txt

    # 2. 此时 ed 在命令模式，没有任何提示。我们先显示所有内容（,p 表示从第一行到最后一行打印）
    ,p
    Hello world
    This is a test.
    Goodbye

    # 3. 使用替换命令。s/旧字符串/新字符串/
    # 首先搜索包含 ‘world’ 的行（/pattern/），然后对其执行替换命令
    /world/ s/world/universe/

    # 4. 再次查看修改后的内容（‘’代表当前行）
    p
    Hello universe

    # 5. 保存并退出
    w
    q
    ```

    example:

    `echo $'#!/bin/ed\n,w\nq' | ed test.txt > /dev/null`

    向 ed 发送命令 `,w`（写入文件）和 `q`（退出）

* `ip route get <dst_ip>`可以显示访问`dst_ip`是从本机的哪个路由表出去

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ip route get 223.5.5.5
    223.5.5.5 via 10.0.2.1 dev enp0s3 src 10.0.2.4 uid 1000 
        cache
    ```

    其中`via 10.0.2.1`表示网关（下一跳的地址），`dev enp0s3`表示使用的网卡设备，`src 10.0.2.4`表示源地址。

    `ip route get <dst_ip> from <src_ip>`可以指定源地址。

    ip route get 仅本地查询，不发送真实数据包。

* `ethtool`

    install: `sudo apt install ethtool`

    * 查看网卡基本信息: `ethtool <网卡名>`
    
        `ethtool enp0s3`

        output:

        ```
        Settings for enp0s3:
        	Supported ports: [ TP ]
        	Supported link modes:   10baseT/Half 10baseT/Full
        	                        100baseT/Half 100baseT/Full
        	                        1000baseT/Full
        	Supported pause frame use: No
        	Supports auto-negotiation: Yes
        	Supported FEC modes: Not reported
        	Advertised link modes:  10baseT/Half 10baseT/Full
        	                        100baseT/Half 100baseT/Full
        	                        1000baseT/Full
        	Advertised pause frame use: No
        	Advertised auto-negotiation: Yes
        	Advertised FEC modes: Not reported
        	Speed: 1000Mb/s
        	Duplex: Full
        	Auto-negotiation: on
        	Port: Twisted Pair
        	PHYAD: 0
        	Transceiver: internal
        	MDI-X: off (auto)
        netlink error: Operation not permitted
                Current message level: 0x00000007 (7)
                                       drv probe link
        	Link detected: yes
        ```

    * 查看驱动信息: `ethtool -i <网卡名>`

        `ethtool -i enp0s3`

        output:

        ```
        driver: e1000
        version: 6.8.0-65-generic
        firmware-version: 
        expansion-rom-version: 
        bus-info: 0000:00:03.0
        supports-statistics: yes
        supports-test: yes
        supports-eeprom-access: yes
        supports-register-dump: yes
        supports-priv-flags: no
        ```

    * 查看统计信息: `ethtool -S <网卡名>`

        `ethtool -S enp0s3`

        output:

        ```
        NIC statistics:
             rx_packets: 7599541
             tx_packets: 3763596
             rx_bytes: 7471543776
             tx_bytes: 5835620961
             rx_broadcast: 40
             tx_broadcast: 6
             rx_multicast: 0
             tx_multicast: 828
             rx_errors: 0
             tx_errors: 0
             tx_dropped: 0
             multicast: 0
             collisions: 0
             rx_length_errors: 0
             rx_over_errors: 0
             rx_crc_errors: 0
             rx_frame_errors: 0
             rx_no_buffer_count: 0
             rx_missed_errors: 0
             tx_aborted_errors: 0
             tx_carrier_errors: 0
             tx_fifo_errors: 0
             tx_heartbeat_errors: 0
             tx_window_errors: 0
             tx_abort_late_coll: 0
             tx_deferred_ok: 0
             tx_single_coll_ok: 0
             tx_multi_coll_ok: 0
             tx_timeout_count: 0
             tx_restart_queue: 0
             rx_long_length_errors: 0
             rx_short_length_errors: 0
             rx_align_errors: 0
             tx_tcp_seg_good: 1456528
             tx_tcp_seg_failed: 0
             rx_flow_control_xon: 0
             rx_flow_control_xoff: 0
             tx_flow_control_xon: 0
             tx_flow_control_xoff: 0
             rx_long_byte_count: 7471543776
             rx_csum_offload_good: 0
             rx_csum_offload_errors: 0
             alloc_rx_buff_failed: 0
             tx_smbus: 0
             rx_smbus: 0
             dropped_smbus: 0
        ```

    其他还有些功能，目前看上去用处不大。如果专业做网卡这块了再去了解。

* glob 匹配

    * 匹配任意字符（包括空字符）。

    ? 匹配单个字符。

    [abc] 匹配 a、b 或 c。

* `find`的匹配模式

    -name 和 -iname： 使用 glob 模式，按文件名匹配。`-iname`表示大小写不敏感。

    -regex 和 -iregex：使用正则表达式。正则匹配的是完整路径（如 ./dir/file.txt），而非仅文件名。

    其他匹配方式:

    * -path：类似 -name，但匹配完整路径（使用 glob 语法）。

    * -perm、-size 等：按权限、大小等属性匹配，与正则/glob 无关。

* `stat <file_path>`

    输出文件的基本信息。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ stat xml_log_4_gpu.txt
      File: xml_log_4_gpu.txt
      Size: 7311      	Blocks: 16         IO Block: 4096   regular file
    Device: 802h/2050d	Inode: 938640      Links: 1
    Access: (0664/-rw-rw-r--)  Uid: ( 1000/     hlc)   Gid: ( 1000/     hlc)
    Access: 2025-08-03 09:42:13.930462794 +0800
    Modify: 2025-06-06 09:55:17.893493034 +0800
    Change: 2025-06-06 09:55:17.893493034 +0800
     Birth: 2025-06-05 16:56:25.016703429 +0800
    ```

* `netstat`

    `netstat -a`: 显示所有连接

    `netstat -s`: 显示统计摘要

    `netstat -r`: 查看系统的路由表信息

    `netstat -i`: 显示网络接口的配置和流量统计

    `netstat -tuln`: 列出所有处于监听（LISTEN）状态的端口

    `netstat -tulnp`: 查看占用端口的进程ID（PID）和程序名（需管理员权限）

    常用参数：

    * `-a`: 显示所有连接和监听端口。

    * `-n`: 以数字形式显示地址和端口（禁用DNS解析）。

    * `-t/-u`: 仅显示TCP/UDP连接。

    * `-p`: 显示进程信息（Linux）。

    * `-o`: 显示进程ID（Windows）。

    * `-r`: 显示路由表。

    * `-s`: 显示协议统计信息。

* `disown`

    `disown`将指定作业从 Shell 的作业列表中删除，但进程仍继续运行。退出 shell 后作业仍继续运行。

    被 disown 的进程仍属于当前用户，但会变成孤儿进程（由 init/systemd 接管）（这一点存疑，在本机测试，无论是`sleep 10`, `sleep 10 &`, 还是`sleep 10 & disown`，`pstree -s <sleep_pid>`输出都是`systemd───systemd───gnome-terminal-───bash───sleep`）

    常见用法：

    ```bash
    disown <jobspec>      # 移除指定作业（如 %1）
    disown -a             # 移除所有作业
    disown -r             # 仅移除运行中的作业
    ```

    * `disown -h`

        将任务保留在 job list 但对其屏蔽 SIGHUP 信号。

        启动后台任务后使用 `disown -h`，即使退出 Shell 也不终止进程（类似 nohup 效果）。

        example:

        ```
        $ long_task &       # 启动后台任务
        $ disown -h %1      # 使任务屏蔽 SIGHUP，但是不从 job list 中移除
        $ exit              # 退出 Shell，任务继续运行
        ```

    * `./long_job & disown`

        在任务开始时就直接从当前 shell 的 job list 中清除，并保持后台执行。

    注意事项：

    1. `disown` 不会自动重定向输出。

    2. 被 `disown` 解除绑定的任务，可以使用`kill <PID>`的方式结束
    
        如果是 `disown -h` 解除绑定并屏蔽 `SIGHUP` 信号的任务，可以通过 `kill -kill <PID>` 的方式结束。

* 符号链接本身的大小是其指向的路径字符串的长度

    example:

    ```
    lrwxrwxrwx 1 hlc hlc    7  8月 17 13:34 msg_link.txt -> msg.txt
    ```

* `git-credential-libsecret`可以将 git 凭据存储在`libsecret`密钥管理服务中。

    需要系统安装 libsecret，还需要 git 去配置，略显复杂。有需求了再看。

* `less`

    less 用于分页查看。

    syntax:

    ```bash
    less [选项] 文件名
    ```

    常用选项：

        -N：显示行号。

        -i：忽略搜索时的大小写（除非搜索词包含大写字母）。

        -F：若文件可一屏显示，直接退出（类似 cat）。

        -S：禁用自动换行（超长行需左右滚动查看）。

        `less +F growing_file.log`: 类似 `tail -f`，按 `Ctrl+C` 退出跟踪模式

    交互式操作
    快捷键	功能
    空格 / f	向下翻一页
    b	向上翻一页
    Enter	向下翻一行
    /关键词	向前搜索（按 n 跳转到下一个）
    ?关键词	向后搜索（按 N 跳转到上一个）
    g	跳转到文件首行
    G	跳转到文件末行
    :n	查看下一个文件（多文件打开时）
    :p	查看上一个文件
    q	退出 less

    按 v 键可在当前光标位置启动默认编辑器（如 vi）编辑文件。

    使用 & 过滤显示匹配行（如 &error 只显示含 "error" 的行）。

* `stty -echo`

    关闭回显。

    ```bash
    stty -echo  # 关闭回显
    read -p "Enter password: " password
    stty echo   # 恢复回显
    echo        # 换行（避免密码输入后的提示符紧贴上一行）
    ```

* `stty`

    set teletype, or set terminal

    常用功能：

    * 查看当前终端设置: `stty -a`

    * 关闭回显（输入不显示，如输入密码时）: `stty -echo`

    * 恢复回显：`stty echo`

    * 禁用控制字符（如 Ctrl+C 中断信号）：`stty intr undef  # 取消 Ctrl+C 的中断功能`

    * 将退格键（Backspace）设为删除前一个字符: `stty erase ^H`

    * 如果终端因错误配置导致乱码或无响应，可通过重置恢复：`stty sane`

    * 禁止终端显示输入内容：`stty -echo; read -s; stty echo`

    stty 的配置是临时性的，仅对当前终端会话有效。

* `setsid`

    创建一个新的会话（session）并在此会话中运行指定的命令，使该进程完全脱离当前终端（terminal）的控制

    即使关闭当前终端窗口或退出登录，通过 setsid 启动的进程仍会继续在后台运行（不会被 SIGHUP 信号终止）。

    新进程会成为会话首进程（session leader），且拥有新的进程组 ID（PGID），与原有终端完全无关。

    example:

    * `setsid your_command &`

        后台运行守护进程（daemon）, 适用于需要长期运行的服务（如自定义脚本或服务）。

    * `setsid tail -f /var/log/syslog`

        即使终端关闭，进程也不会退出

    * 替代 nohup 的更彻底方案

        nohup 仅忽略 SIGHUP 信号，而 setsid 直接脱离终端会话。

        ```bash
        # 启动一个完全脱离终端的进程（日志重定向到文件）
        setsid your_command > /var/log/command.log 2>&1 &
        ```

* `<`, `<<`和`<<<`

    1. <（标准输入重定向）

        用于将文件内容作为命令的标准输入。

        ```bash
        command < file.txt
        ```

        （将 file.txt 的内容传递给 command 作为输入）

    1. <<（Here Document）

        用于在脚本中直接嵌入多行输入，直到遇到指定的结束标记（delimiter）。

        ```bash
        command << EOF
        line 1
        line 2
        EOF
        ```

    1. <<<（Here String）

        用于将单个字符串（而不是文件或多行文本）作为命令的标准输入。

        ```bash
        command <<< "string"
        ```

* `pstree`

    `pstree`可以以树状结构显示 ps 的内容。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ pstree 
    systemd─┬─ModemManager───2*[{ModemManager}]
            ├─NetworkManager───2*[{NetworkManager}]
            ├─accounts-daemon───2*[{accounts-daemon}]
            ├─acpid
            ├─avahi-daemon───avahi-daemon
            ├─blkmapd
            ├─colord───2*[{colord}]
    ```

    其中数字表示多个相同的进程/线程。

    常用参数：

    * `-c` 选项禁用合并

    * `-p`：显示进程的 PID。

    * `-n`：按 PID 数字排序（默认按进程名排序）。

    * `-a`：显示进程的完整命令行参数。

    * `pstree [username]`: 查看某用户启动的进程树

    * `-A`: 使用 ASCII 字符绘制树（兼容性更好）

* `finger`是一个早期的网络工具，用于查询系统上的用户信息，现代系统默认禁用

* inotify

    `inotifywait`默认不被安装，需要手动安装:

    ```
    Command 'inotifywait' not found, but can be installed with:
    sudo apt install inotify-tools
    ```

    example:

    `inotifywait -m test_1/`

    此时在`test_1`中执行`ls`，则`inotifywait`的输出为：

    ```
    Setting up watches.
    Watches established.
    test_1/ OPEN,ISDIR 
    test_1/ ACCESS,ISDIR 
    test_1/ CLOSE_NOWRITE,CLOSE,ISDIR
    ```

    继续执行`touch haha.txt`，则有输出：

    ```
    test_1/ CREATE haha.txt
    test_1/ OPEN haha.txt
    test_1/ ATTRIB haha.txt
    test_1/ CLOSE_WRITE,CLOSE haha.txt
    ```

    此时 cpu 几乎没有占用。

    未验证的限制：

    * 不适用于远程文件系统（如 NFS）。

    * 监控大量文件时可能耗尽 inotify 的 watch 句柄（需调整 /proc/sys/fs/inotify/max_user_watches）。

* `stat filename  # 查看文件详细信息`

* `mail`发送邮件

    直接输入`mail -s "主题" 收件人@example.com`，然后输入邮件正文，按 Ctrl+D 发送。

    接收邮件: 输入`mail`查看收件箱，按邮件编号阅读具体内容。

    `mail`命令默认不提供，需要`sudo apt install mailutils`。

    常用选项：

        -s "subject"：指定邮件主题。

        -c "抄送地址"：设置抄送。

        -b "密送地址"：设置密送。

    交互命令（接收邮件时）：

        d 编号：删除邮件。

        q：退出。

        h：重新显示邮件列表。

* `tail -f`

    用于动态追踪文件末尾的新内容, 默认显示文件最后 10 行

    其他选项：

    * `-n <行数>`：指定初始显示的行数（如 tail -f -n 20 file.log 显示最后 20 行）。

    * `-F`：比 -f 更健壮，会跟踪文件重命名或重建（如日志轮转场景）。

    使用 -F 时，tail 会定期检查文件的 inode 编号 是否变化（例如日志轮转后新文件的 inode 不同）。如果变化，则关闭旧文件描述符，重新打开新文件继续跟踪。

* 任何以`.`开头的文件或目录都会被系统视为隐藏文件

    `..xxx`并不是例外的隐藏文件。

* `ls -A`

    查看用户创建的隐藏文件

* 删除当前目录下的所有内容（包括隐藏文件）

    `rm -rf .[!.]* ..?* *`

    解析：

    * `.[!.]*`：匹配所有以`.`开头，且第二个字符不是`.`的文件/文件夹（如`.bashrc`）。

    * `..?*`：匹配所有以`..`开头，且第三个字符存在的文件/文件夹（如`..hidden`）。

    * `*`：匹配所有非隐藏文件/文件夹。

    这里对`.`和`..`进行特殊处理，是为了避开默认存在的两个文件夹，即`.`和`..`。

    如果文件名包含特殊字符（如 -、--、空格等），可以改用：

    `find . -mindepth 1 -delete`

    解析：

    find . -mindepth 1：查找当前目录下的所有文件/文件夹（不包括 . 本身）。

    -delete：直接删除（需谨慎）。

    仅删除隐藏文件和文件夹:

    `rm -rf .[!.]* .??*`

    解析：

    * `.[!.]*` → 匹配`.`开头，第二个字符不是`.`的文件/目录（如 .bashrc、.config）。

    .??* → 匹配 . 开头，且至少 3 个字符的文件/目录（如 .ssh、.gitignore）。

    `find . -maxdepth 1 -name ".*" -exec rm -rf {} +`

    解析：

    -maxdepth 1 → 仅当前目录（不递归子目录）。

    -name ".*" → 匹配所有以 . 开头的文件/目录。

    -exec rm -rf {} + → 批量删除匹配项。

    递归删除子目录中的隐藏文件：

    `find . -name ".*" -exec rm -rf {} +`

    仅删除隐藏文件（保留隐藏目录）:

    `find . -maxdepth 1 -type f -name ".*" -delete`

    -type f → 仅匹配文件（不包括目录）。

* `chfn`

    chfn（Change Finger）用于修改用户信息.

    syntax:

    ```
    chfn [选项] [用户名]
    ```

    * `-f`或`--full-name`: 修改全名

    * `-o`或`--office`: 设置办公室地址

    * `-p`或`--office-phone`: 设置办公电话

    * `-h`或`--home-phone`: 设置家庭电话

    example:

    * `chfn`

        交互式运行。

    * `chfn -f "Jane Smith" johndoe`

        将用户`johndoe`的 name 修改为`Jane Smith`.

    修改的信息存储在`/etc/passwd`的`GECOS`字段，可通过`finger`命令查看。

    `finger`需要使用 apt 安装：`apt install finger`

    `chfn`这个命令在现代 linux 中已经不怎么用了。

* crontab examples

    * `0 * * * * /path/script.sh`
        
        每小时整点执行

    * `*/10 * * * * date >> /tmp/log`
    
        每 10 分钟记录当前时间

    * `0 2 * * * /backup.sh`
    
        每天凌晨 2 点执行备份

    * `0 0 * * 1 tar -zcf /backup/weekly.tar.gz /data`
    
        每周一零点压缩备份

    * `@reboot /path/start_service.sh`
    
        系统启动时执行

    * `* * * * * /path/to/command`

        每分钟执行一次

    * `*/5 * * * * /path/to/command`

        每5分钟执行一次

    * `30 * * * * /path/to/command`

        每小时的第30分钟执行

    * `0 2 * * * /path/to/command`

        每天凌晨2点执行

    * `0 3 * * 1 /path/to/command`

        每周一凌晨3点执行

    * `0 12 1 * * /path/to/command`

        每月1号中午12点执行

    * `* * * * * /path/to/script.sh >> /var/log/script.log 2>&1`

        将输出追加到日志文件（避免邮件通知）

    * `* * * * * /path/to/command > /dev/null 2>&1`

        丢弃所有输出（静默执行）

    * `0 0 * * * rm -rf /tmp/*`

        每天凌晨清理临时文件

    * `0 2 * * 0 mysqldump -u root -pPASSWORD dbname > /backup/db.sql`

        每周日备份数据库

    * `*/30 * * * * /usr/sbin/ntpdate pool.ntp.org`

        每30分钟同步时间（需安装ntpdate）

    * `0 9 * * * echo "Daily meeting at 9:30!" | wall`

        每天9点提醒自己（写入终端）

    * `0 * * * * df -h > /home/user/disk_usage.log`

        每小时检查磁盘空间

    * `@reboot /path/to/startup_script.sh`

        系统重启后执行

    * `0 17 * * 1-5 echo "End of workday" | mail -s "Reminder" user@example.com`

        工作日（周一到周五）下午5点发邮件

    * `0 * * * 1-5 /path/to/command`

        工作日每小时执行一次

    * `*/10 * * * * sleep $((RANDOM \% 60)) && /path/to/command`

        随机延迟执行（避免任务集中触发）

    * `0 9-18 * * 1-5 /path/to/work_script.sh`

        每周一到周五，上午9点到下午6点，每小时执行一次

    * `59 23 28-31 * * [ "$(date +\%d -d tomorrow)" = "01" ] && /path/to/monthly_task.sh`

        每月最后一天23:59执行
    
    * `* * * * * /path/to/command | mail -s "Cron Debug" your@email.com`

        通过日志或临时邮件检查任务是否运行


* crontab 其他常用命令

    * `crontab -l`
        
        列出当前用户的所有定时任务

        其实就是列出 config 文件的所有内容，类似于 cat。

    * `crontab -r`
    
        删除当前用户的所有定时任务（谨慎使用！）

    * `crontab -u user`
    
        管理员专用：管理其他用户的任务（如`-e`, `-l`, `-r`）

* crontab

    `crontab`可以定时执行一些命令，在 ubuntu 系统上默认装有。

    `crontab -e`编辑定时任务的配置文件。

    配置文件的格式：

    ```
    * * * * * command_to_execute
    │ │ │ │ │
    │ │ │ │ └── 星期几 (0-7, 0和7均代表周日)
    │ │ │ └──── 月份 (1-12)
    │ │ └────── 日 (1-31)
    │ └──────── 小时 (0-23)
    └────────── 分钟 (0-59)
    ```

    * `*`：任意值（如 `* * * * *` 表示每分钟）

    * `,`：分隔多个时间（如 `0,15,30 * * * *` 表示每小时的 0、15、30 分）

    * `-`：范围（如 `0 9-17 * * *` 表示每天 9点到17点整点）

    * `/`：间隔（如 `*/5 * * * *` 表示每 5 分钟）

    example:

    ```
    # 每分钟执行一次
    * * * * * date
    ```

    使用`crontab -e`编辑任务后无需重启服务，修改会自动生效。

    output:

    `/var/log/syslog`:

    ```
    Jul 31 16:08:23 hlc-VirtualBox crontab[3437117]: (hlc) BEGIN EDIT (hlc)
    Jul 31 16:08:26 hlc-VirtualBox crontab[3437117]: (hlc) REPLACE (hlc)
    Jul 31 16:08:26 hlc-VirtualBox crontab[3437117]: (hlc) END EDIT (hlc)
    Jul 31 16:09:01 hlc-VirtualBox cron[689]: (hlc) RELOAD (crontabs/hlc)
    Jul 31 16:09:01 hlc-VirtualBox CRON[3437888]: (hlc) CMD (date)
    Jul 31 16:09:01 hlc-VirtualBox CRON[3437887]: (CRON) info (No MTA installed, discarding output)
    ```

    cron 默认会通过邮件发送命令的输出，但系统没有安装邮件传输代理（MTA）（如 sendmail、postfix 或 exim4），所以输出被丢弃。

    通常会让 cron 的输出重定向到日志文件：

    `* * * * * date >> /tmp/cron_date.log 2>&1`

* `pgrep`

    `pgrep`可根据程序的名称快速找到 pid.

    ```bash
    pgrep firefox
    ```

    output:

    ```
    2301052
    ```

    `pgrep`通常可以和`kill`等命令连用：

    `kill $(pgrep nginx)`

    如果有多个进程实例，则会列出多个 pid:

    ```bash
    pgrep bash
    ```

    output:

    ```
    2740
    3059
    2293259
    2294904
    2328484
    2529093
    2618992
    2754955
    2792818
    2823794
    ```

* `sudo -S`

    用于从标准输入（stdin）读取密码，而非交互式终端提示.

    `-S`等价于`--stdin`

    example:

    ```bash
    echo "你的密码" | sudo -S command

    cat password.txt | sudo -S apt update  # 从文件读取密码
    ```

    需要注意的是 echo 不会换行：

    ```bash
    echo xxx | sudo -S echo hello
    ```

    output:

    ```
    [sudo] password for hlc: hello
    ```

    似乎没有什么好的解决方案。

* `yum list`和`yum search`的区别

    * `yum list <package>`

        精确列出匹配名称的软件包（包括已安装、可安装或可升级的版本）。

        特点:

        * 严格匹配包名：默认按完整名称匹配（支持通配符 *）。

        * 显示详细信息：输出包含 包名、版本号、所属仓库（如是否已安装）。

        * 不搜索描述：仅检查包名，不涉及软件包的描述或关键字。

        example:

        `yum list "nginx*"  # 列出所有名称以 `nginx` 开头的软件包`

        output:

        ```
        nginx.x86_64    1.20.1-1.el7    @epel      # 已安装
        nginx-module.x86_64 1.20.1-1.el7 updates    # 可升级
        ```

    * `yum search <package>`

        通过关键字搜索软件包（匹配包名和描述信息）。

        特点:

        * 模糊搜索：同时匹配包名、描述、摘要中的关键字。

        * 返回摘要：显示包名和简要描述，但不显示版本或仓库信息。

        * 结果更广泛：可能包含名称不直接相关但描述匹配的包。

        example:

        `yum search "web server"  # 搜索描述或名称中包含 "web server" 的包`

        output:

        ```
        nginx.x86_64 : High performance web server
        httpd.x86_64 : Apache HTTP Server
        ```

* yum 简介

    Yum（Yellowdog Updater Modified）

    * 安装软件包

        `yum install <package_name>`

    * 安装本地 RPM 包（自动解决依赖）

        `yum localinstall <path_to_rpm>`

    * 升级单个软件包

        `yum update <package_name>`

    * 升级所有可升级软件包

        `yum update`

    * 检查可升级的软件包

        `yum check-update`

    * 搜索软件包（按名称/描述）

        `yum search <keyword>`

    * 列出已安装的软件包

        `yum list installed`

        `yum list`会显示仓库中所有可用的软件包（包括已安装和未安装的）。

        `yum list available`仅列出可安装但尚未安装的软件包

        `yum list updates`列出所有可升级的软件包

        `yum list extras`列出已安装但不在仓库中的包（如手动安装的 RPM）

        `yum list <package_name>`搜索特定软件包（支持通配符，如`yum list "nginx*"`）

    查看软件包信息
    yum info <package_name>

    列出软件包的依赖
    yum deplist <package_name>

    查找提供特定文件的软件包
    yum provides <file_path>

    删除软件包（保留依赖）
    yum remove <package_name>

    删除无用依赖
    yum autoremove

    清理旧内核
    yum remove $(rpm -q kernel | grep -v $(uname -r))

    删除软件包（保留依赖）
    yum remove <package_name>

    删除无用依赖
    yum autoremove

    清理旧内核
    yum remove $(rpm -q kernel | grep -v $(uname -r))

    列出所有仓库
    yum repolist all

    启用/禁用仓库
    yum-config-manager --enable/disablerepo <repo_name>

    添加新仓库
    yum-config-manager --add-repo <repo_url>

    清理缓存（保留元数据）
    yum clean packages

    清理所有缓存（包括元数据）
    yum clean all

    重建缓存
    yum makecache

    查看历史操作
    yum history

* `/etc/sudoers`中，`@includedir /etc/sudoers.d`用于 加载`/etc/sudoers.d`目录下的所有配置文件

    例如：

    `/etc/sudoers.d/web_admins` -> 存放 Web 管理员的 sudo 权限

    `/etc/sudoers.d/db_admins` -> 存放数据库管理员的 sudo 权限

    在软件包安装时（如 Docker、Nginx），它们可能会自动在`/etc/sudoers.d/`下添加自己的规则。

    修改`/etc/sudoers.d/`下的文件后，无需重启，sudo 会自动识别（但建议用`visudo -c`检查语法）。

    文件名：不能包含`.`或`~`（避免读取临时文件或备份文件）。

    权限：必须为 0440（root:root 可读，其他用户不可读），否则 sudo 会忽略它并报错.

    ```bash
    # 检查所有 sudoers 文件（包括 sudoers.d 下的）
    sudo visudo -c
    ```

* `sudo`与`/etc/sudoers`

    用户使用 sudo 执行的命令会被记录到`/var/log/auth.log`文件中。

    比如用户执行`sudo echo hello`，日志的记录为

    ```
    Jul 27 14:38:16 hlc-VirtualBox sudo:      hlc : TTY=pts/5 ; PWD=/home/hlc/Documents/documents/Personal/logs ; USER=root ; COMMAND=/usr/bin/echo hello
    Jul 27 14:38:16 hlc-VirtualBox sudo: pam_unix(sudo:session): session opened for user root(uid=0) by (uid=1000)
    Jul 27 14:38:16 hlc-VirtualBox sudo: pam_unix(sudo:session): session closed for user root
    ```

    `/etc/sudoers`规定了用户使用 sudo 可以执行哪些命令。

    通常使用`visudo`编辑这个文件，`visudo`可以在保存前进行语法检查，防止语法出错。

    `sudoers`文件的语法为：

    ```
    user    HOST=(RUNAS_USER)    COMMANDS
    %group  HOST=(RUNAS_USER)    COMMANDS
    ```

    其中，

    * `user/%group`：用户名或%组名。

    * `HOST`：允许使用 sudo 的主机名（通常设为`ALL`）。

    * `(RUNAS_USER)`：可以切换的目标用户（如 `(root)`、`(ALL)`）。

        通常设置为`(ALL:ALL)`或`(ALL)`。

        * `(ALL)`: 允许用户以 任意用户身份（包括 root）执行命令。

            ```bash
            sudo -u root apt update    # 以 root 身份运行
            sudo -u bob apt install xx # 以用户 bob 身份运行
            ```

        * `(root)`: 仅允许用户以 root 身份 执行命令（不能切换为其他用户）。

            ```bash
            sudo systemctl restart nginx      # 隐含 -u root
            sudo -u alice systemctl start xx  # 报错（不允许）
            ```

        * `(ALL:ALL)`: 允许用户以 任意用户和任意用户组 身份执行命令（用户和组均可切换）。

            `(RUNAS_USER:RUNAS_GROUP)`, 若省略`:GROUP`，默认使用目标用户的默认组。注意这个 group 的指定在后面，不在前面。

            ```bash
            sudo -u alice -g developers chmod 755 file
            sudo -u root -g root chmod 600 /etc/shadow
            ```

    `COMMANDS`：允许执行的命令（绝对路径，`ALL`表示全部）。

    example:

    ```conf
    alice   ALL=(root)    /usr/bin/apt      # alice 可以 root 身份运行 apt
    %admin  ALL=(ALL)     ALL                # admin 组成员可执行任何命令
    ```

    使用别名（变量）：

    ```conf
    User_Alias     ADMINS = alice, bob
    Host_Alias     SERVERS = 192.168.1.1
    Runas_Alias    DEVS = tom
    Cmnd_Alias     PKG_CMDS = /usr/bin/apt, /usr/bin/dpkg

    ADMINS SERVERS=(DEVS) PKG_CMDS
    ```

    其他配置：

    * `NOPASSWD`：执行命令无需密码

        ```conf
        bob    ALL=(root)    NOPASSWD: /usr/bin/systemctl
        ```

    * `!`排除命令：

        ```conf
        charlie ALL=(ALL) ALL, !/usr/bin/passwd root
        ```

    * 全局配置

        通过`Defaults`设置全局行为，如：

        ```conf
        Defaults    env_keep += "HTTP_PROXY"   # 保留环境变量
        Defaults    insults                    # 输错密码时显示“嘲讽”
        ```

    因为编辑这个文件可能会影响到当前虚拟机环境，所以上面的命令都没有验证过。

* `ssh-askpass`简介

    安装：`apt install ssh-askpass`

    直接运行`./ssh-askpass`时，会弹出一个窗口，输入密码并按回车后，输入的内容会出现在 terminal 里。

    `sudo`和`ssh`都可以使用`ssh-askpass`程序弹出 gui 输入密码。

    以`sudo`为例，首先配置环境变量，然后使用`sudo -A <command>`激活：

    ```bash
    export SUDO_ASKPASS=/usr/bin/ssh-askpass
    sudo -A -v
    ```

    此时在弹出的窗口中输入密码，terminal 不会回显。

    如果未检测到环境变量，sudo 会报错：

    `sudo: no askpass program specified, try setting SUDO_ASKPASS`

    `ssh`使用`ssh-askpass`同样需要使用环境变量：

    ```bash
    export SSH_ASKPASS="/usr/bin/ssh-askpass"
    ```

    经验证，ssh 很难启动 askpass 弹窗，目前不清楚原因。

* `sudo -v`用于延长 sudo 的密码缓存时间

    `sudo`第一次缓存密码的时间为 15 分钟，执行`sudo -v`会重置这个计时器。

    这里的`-v`代表 validate

    通常在脚本开头检查权限，防止脚本因为 sudo 密码问题中断：

    ```bash
    if ! sudo -v; then
        echo "Error: No sudo access or incorrect password."
        exit 1
    fi
    ```

    `sudo -k`可以清空密码缓存。执行`sudo -k`不需要 sudo 密码。

* `/proc/<PID>/cmdline` 是 Linux 系统 伪文件系统（procfs） 中的一个特殊文件，用于 获取指定进程（PID）的完整命令行启动参数，以`\0`（空字符）分隔各个参数。

    如果进程本身就是以相对路径启动的，比如`bash`，那么`cmdline`不会显示其绝对路径。

    `cmdline`的结尾没有`\n`。

    因为 cmdline 使用`\0`进行参数分隔，所以避免了很多字符转义的问题。

* `sudo -l`

    列出当前用户可以使用`sudo`执行哪些 root 操作。

    `sudo -ll`可以列出更详细的输出。

    `sudo -U user -l`可以查看其他用户的 root 权限。

* `ps -f`

    既显示完整信息，也显示完整 cmd（比如绝对路径）。否则只显示简略信息和 cmd 名称。

    example:

    ```bash
    ps -a
    ```

    output:

    ```
        PID TTY          TIME CMD
       2178 tty2     00:00:00 gnome-session-b
     295495 pts/9    00:01:03 ssh
     296484 pts/10   00:00:02 socat
     489812 pts/3    00:00:00 ps
    ```

    ```bash
    ps -af
    ```

    output:

    ```
    UID          PID    PPID  C STIME TTY          TIME CMD
    hlc         2178    2174  0 7月19 tty2    00:00:00 /usr/libexec/gnome-session-b
    hlc       295495  294700  0 7月19 pts/9   00:01:03 ssh -R 8825:127.0.0.1:8825 h
    hlc       296484  296052  0 7月19 pts/10  00:00:02 socat TCP-LISTEN:8825,reusea
    hlc       490526  487976  0 12:26 pts/3    00:00:00 ps -af
    ```

    可以看到，使用`-f`后，信息有截断，但是确实是完整路径。如果要显示未截断的完整信息，那么需要再加上`-l`参数。

* `ps -e --forest`

    * `-e`

        显示所有进程（包括其他用户的进程），等同于`-A`。

        `man ps`中的解释：

        > -e     Select all processes.  Identical to -A.

    * `--forest`

        以树状缩进形式显示进程层级，清晰体现父子关系。

        其输出中使用`\_`而不是`|_`，目前仍不清楚原因。

* `tr`的用法

    `tr`指的是 translate，通常用于字符替换

    example:

    ```bash
    echo hello | tr a-z A-Z
    ```

    output:

    ```
    HELLO
    ```

    也可以删除字符：

    ```bash
    echo "hello 123 world" | tr -d 0-9
    ```

    output:

    ```
    hello  world
    ```

    还可以去重：

    ```bash
    echo hello | tr -s l
    ```

    output:

    ```
    helo
    ```

    这里的`-s`可能是 squash 的意思。

    过滤（filter in，保留指定的字符）：

    ```bash
    echo "hello 123" | tr -cd 'a-z'
    ```

    output:

    ```
    hello
    ```

    output 末尾无换行符。`tr`只保留`a-z`小字字母字符。

    这里的`-c`可能是补集（complementary）的意思

    一一映射：

    ```bash
    echo abc | tr cba xzy
    ```

    output:

    ```
    yzx
    ```

    注：

    1. `tr`在处理`\n`时。需要给`\n`加上引号（单引号双引号都可以），否则会被 bash 转义。

        example:

        ```bash
        echo hello | tr '\n' N
        ```

        output:

        ```
        helloN
        ```

        output 后无换行。

    1. `tr`只能处理单个字符，不能处理字符串和正则表达式。

* `grep -F`表示不进行正则解析

    example:

    `grep -F "hello" content.txt`

    只查找`hello`字符串。

    `grep -F "a.*b" content.txt`

    匹配`a.*b`字符串。

    `grep -F`等价于`fgrep`。

* `/proc/<PID>/environ`是一个在内存中的文件，以只读的形式存储了指定 PID 进程在启用时的环境变量信息

    example:

    `cat /proc/3273/environ`

    output:

    ```
    SYSTEMD_EXEC_PID=2451SSH_AUTH_SOCK=/run/user/1000/keyring/sshSESSION_MANAGER=local/hlc-VirtualBox:@/tmp/.ICE-unix/2235,unix/hlc-VirtualBox:/tmp/.ICE-unix/2235GNOME_TERMINAL_SCREEN=/org/gnome/Terminal/screen/16e4c141_024c_4318_9398_96a803c31884LANG=en_US.UTF-8XDG_CURRENT_DESKTOP=ubuntu:GNOMEPWD=/home/hlcWAYLAND_DISPLAY=wayland-0LC_IDENTIFICATION=zh_CN.UTF-8IM_CONFIG_PHASE=1...
    ```

    其形式为`key=value`，环境变量之间使用`\0`间隔。

    此文件为只读属性，无法修改。

    可以将`\0`替换为`\n`，便于阅读：

    `cat /proc/<PID>/environ | tr '\0' '\n'`

    output:

    ```
    SYSTEMD_EXEC_PID=2451
    SSH_AUTH_SOCK=/run/user/1000/keyring/ssh
    SESSION_MANAGER=local/hlc-VirtualBox:@/tmp/.ICE-unix/2235,unix/hlc-VirtualBox:/tmp/.ICE-unix/2235
    GNOME_TERMINAL_SCREEN=/org/gnome/Terminal/screen/16e4c141_024c_4318_9398_96a803c31884
    LANG=en_US.UTF-8
    XDG_CURRENT_DESKTOP=ubuntu:GNOME
    PWD=/home/hlc
    WAYLAND_DISPLAY=wayland-0
    LC_IDENTIFICATION=zh_CN.UTF-8
    IM_CONFIG_PHASE=1
    ...
    ```

    说明：

    1. `/proc/<PID>/environ`不是实时的，如果在程序中运行`setenv()`，那么此文件内容不会被改变。

    1. 仅允许进程所有者或 root 用户读取（权限为`-r--------`）

        这样看来，这个文件的用处似乎不大？

* `sudo yum check-update`相当于`sudo apt update`

* centos 加入 sudo 权限

    `usermod -aG wheel 用户名`

    在 centos 中，`wheel`组有 sudo 权限。

* 解析`ps -ef`

    ps 指的是 process status

    `-e`：显示所有进程（包括其他用户的进程）。

    `-f`：以完整格式（full-format）输出详细信息。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ps -ef
    UID          PID    PPID  C STIME TTY          TIME CMD
    root           1       0  0 09:50 ?        00:00:01 /sbin/init splash
    root           2       0  0 09:50 ?        00:00:00 [kthreadd]
    root           3       2  0 09:50 ?        00:00:00 [pool_workqueue_release]
    root           4       2  0 09:50 ?        00:00:00 [kworker/R-rcu_g]
    root           5       2  0 09:50 ?        00:00:00 [kworker/R-rcu_p]
    root           6       2  0 09:50 ?        00:00:00 [kworker/R-slub_]
    root           7       2  0 09:50 ?        00:00:00 [kworker/R-netns]
    ...
    ```

    UID：进程所属用户。

    PID：进程的唯一ID。

    PPID：父进程ID。

    C：CPU占用率。

    STIME：进程启动时间。

    TTY：启动进程的终端（?表示与终端无关，如守护进程）。

    TIME：进程占用CPU总时间。

    CMD：进程对应的完整命令或程序路径。

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

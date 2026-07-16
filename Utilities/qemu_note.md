# qemu note

## cache

* qemu edu device 中，
0x20 (RW)status register
    Bitwise OR of:
    0x01
        computing factorial (RO)
    0x80
        raise interrupt after finishing factorial computation
详解解释一下 0x20 这个寄存器是如何使用的

    好的，我们来详细解释一下 QEMU edu 设备中偏移地址 `0x20` 这个状态/控制寄存器的用法。

    这个寄存器是一个多功能寄存器，它**融合了状态标志（只读）和控制位（读写）**。这是一个在硬件设备中很常见的设计模式。

    ### 寄存器概览

    *   **地址**: `0x20` (从设备的内存映射 I/O 基地址算起)
    *   **属性**: 读写 (RW)，但其中某些位是只读的。
    *   **功能**: 用于查询设备（阶乘计算）的当前状态，并控制设备在计算完成时是否发出中断。

    ---

    ### 位字段详解

    寄存器宽度是 32 位，但这里只定义了两个关键的位：

    #### 位 0 (0x01): 计算状态位

    *   **属性**: **只读 (RO)**
    *   **功能**: 指示阶乘计算单元当前的工作状态。
        *   **值 = 1**: 表示设备**正在计算**阶乘。
        *   **值 = 0**: 表示设备**处于空闲状态**，阶乘计算已经完成，可以读取结果或开始新的计算。

    **如何使用这个位：**
    驱动程序在向 `0x08` 寄存器写入要计算阶乘的数字后，可以轮询这个位来等待计算完成。

    **示例轮询代码（伪代码）：**
    ```c
    // 1. 将要计算的数字（例如 10）写入阶乘寄存器
    write_mmio(DEVICE_BASE + 0x08, 10);

    // 2. 轮询状态寄存器，等待计算完成
    while (read_mmio(DEVICE_BASE + 0x20) & 0x01) {
        // 位 0 为 1，说明仍在计算中，继续等待
        // 在实际驱动中，这里可能会加入短暂的延迟或让出 CPU
    }

    // 3. 当循环退出时，说明位 0 变为 0，计算已完成
    // 可以从 0x08 寄存器读取计算结果了
    uint32_t result = read_mmio(DEVICE_BASE + 0x08);
    ```

    #### 位 7 (0x80): 中断使能位

    *   **属性**: **读写 (RW)**
    *   **功能**: 控制设备在阶乘计算完成时是否产生一个中断信号。
        *   **值 = 1**: **使能中断**。当阶乘计算完成（即 `状态位` 从 1 变为 0）时，设备会向系统发出一个中断请求。
        *   **值 = 0**: **禁用中断**。计算完成后不会产生中断。

    **如何使用这个位：**
    驱动程序如果不想使用轮询这种低效的方式，可以采用中断驱动的方式。

    **示例中断驱动代码流程：**

    1.  **初始化阶段**:
        *   向操作系统注册 edu 设备的中断处理程序。
        *   **使能中断**：将 `0x80` 位写入 `0x20` 寄存器。
          ```c
          // 设置中断使能位 (0x80)，同时不影响其他位（如果存在的话）
          uint32_t current_status = read_mmio(DEVICE_BASE + 0x20);
          write_mmio(DEVICE_BASE + 0x20, current_status | 0x80);
          ```

    2.  **执行计算**:
        *   写入阶乘数字到 `0x08` 寄存器，设备开始计算。

    3.  **等待与响应**:
        *   CPU 可以转而执行其他任务，无需轮询。
        *   当计算完成时，设备产生中断。
        *   CPU 收到中断，调用你注册的**中断处理程序**。

    4.  **中断处理程序中**:
        *   确认中断来自 edu 设备（可能需要读取其他状态寄存器）。
        *   从 `0x08` 寄存器读取阶乘结果。
        *   进行必要的处理（例如，唤醒正在等待此结果的进程）。
        *   清除设备的中断挂起状态（通常通过向某个特定的寄存器写入值来实现，edu 设备的具体操作需参考其完整文档）。

    ---

    ### 综合使用场景

    假设你想在计算完成后使用中断，并且想知道设备的初始状态。

    1.  **读取初始状态**:
        ```c
        uint32_t status = read_mmio(DEVICE_BASE + 0x20);
        ```
        此时，`status` 的值可能是：
        *   `0x00`: 设备空闲，且中断被禁用。
        *   `0x80`: 设备空闲，但中断已被使能（可能是之前设置的）。
        *   `0x01`: 设备正在计算，中断被禁用。
        *   `0x81`: 设备正在计算，且中断被使能。

    2.  **配置设备**:
        如果你想确保中断是使能的，无论之前状态如何：
        ```c
        write_mmio(DEVICE_BASE + 0x20, status | 0x80); // 设置 bit 7
        ```

    3.  **启动任务并等待**:
        写入 `0x08` 寄存器启动计算后，驱动程序不再需要主动查询状态位。计算完成后，中断会自动触发，你的中断服务例程会处理后续事宜。

    ### 总结

    `0x20` 寄存器是 edu 设备与驱动程序交互的核心之一：

    *   **Bit 0 (只读)** 像一个**忙指示灯**，驱动程序可以通过轮询它来同步操作。
    *   **Bit 7 (读写)** 像一个**中断开关**，打开后，设备会在工作完成时“主动通知”驱动程序，这是一种更高效的处理方式。

    这种将状态和控制功能合并到一个寄存器的设计，节省了设备的地址空间，但要求驱动程序在写入控制位时必须小心，不要意外地改变状态位（虽然状态位是只读的，写入无效），或者在读取状态位时注意屏蔽掉控制位。通常的做法是，驱动程序在本地保存一个该寄存器的“影子副本”，只修改需要控制的位，然后将这个影子值写入硬件寄存器。

* 0x24 (RO)interrupt status register

    It contains values which raised the interrupt (see interrupt raise register below).
0x60 (WO)interrupt raise register

    Raise an interrupt. The value will be put to the interrupt status register (using bitwise OR).
那这两个呢，我看这两个也是管理中断的

    好的，这两个寄存器与 `0x20` 寄存器中的中断使能位协同工作，共同构成了 edu 设备的中断管理系统。让我们来详细解析它们。

    ### 整体中断流程

    首先，理解一个完整的中断流程：
    1. **触发条件**：某个事件发生（如阶乘计算完成）。
    2. **产生中断**：设备内部逻辑将中断状态寄存器的某个位设置为 1。
    3. **上报系统**：如果全局中断使能（`0x20` 的 bit 7）为 1，设备就会向 CPU 的中断控制器发送一个中断请求。
    4. **处理中断**：CPU 调用驱动程序的中断服务例程。
    5. **识别中断源**：驱动程序读取中断状态寄存器，查看是哪个事件引起的中断。
    6. **清除中断**：驱动程序通过某种方式告知设备它已处理完毕，设备随后会拉低中断信号线。

    `0x24` 和 `0x60` 寄存器正是在这个流程中扮演关键角色。

    ---

    ### 0x24 (RO) Interrupt Status Register - 中断状态寄存器

    *   **地址**: `0x24`
    *   **属性**: **只读 (RO)**
    *   **功能**: 这是一个**标志寄存器**，它的每一个位代表一个具体的中断源。当某个事件发生时，设备硬件会**自动**将该事件对应的位设置为 1。驱动程序通过读取这个寄存器，就能知道**究竟是什么原因导致了这次中断**。

    **如何工作：**
    - 当发生了一个可以引起中断的事件（比如阶乘计算完成），设备硬件会在这个寄存器的相应位上执行 **"位或"** 操作。
    - 即使全局中断使能（`0x20` 的 bit 7）被关闭，这个状态寄存器依然会被设置，它只是记录事件的发生，不管中断是否被最终上报给系统。
    - 这个寄存器是**只读**的，意味着**你不能通过向这个寄存器写入来清除中断**。清除中断需要通过其他机制（通常是向 `Interrupt Ack Register` 写入，在 edu 设备中可能是 `0x68` 或类似地址，需要查完整文档）。

    ---

    ### 0x60 (WO) Interrupt Raise Register - 中断触发寄存器

    *   **地址**: `0x60`
    *   **属性**: **只写 (WO)**
    *   **功能**: 这是一个**命令寄存器**。驱动程序通过向这个寄存器的特定位写入 1，可以**手动地、软件模拟地**触发一个中断。这是一种非常有用的调试和测试手段。

    **如何工作：**
    - 当你向这个寄存器的某一位写入 1 时，设备硬件会**在 `0x24` 中断状态寄存器的对应位上执行 "位或" 操作**。
    - 例如，如果你向 `0x60` 写入 `0x04`，那么 `0x24` 寄存器的 bit 2 就会被置 1。
    - 一旦 `0x24` 寄存器的值从 0 变为非零，并且全局中断使能（`0x20` 的 bit 7）是打开的，设备就会立即向系统上报一个中断。

    **重要特性：**
    - 它是**只写**的，读取它没有意义。
    - 它通常被用来**生成中断**，而不是清除中断。

    ---

    ### 综合工作流程与示例

    假设 edu 设备定义：**阶乘计算完成事件** 对应 `中断状态寄存器 (0x24)` 的 **Bit 0**。

    #### 场景一：硬件自动中断（正常操作）

    1.  **使能中断**:
        ```c
        // 设置 0x20 寄存器的 bit 7，开启中断
        write_mmio(DEVICE_BASE + 0x20, 0x80);
        ```

    2.  **启动计算**:
        ```c
        write_mmio(DEVICE_BASE + 0x08, 10); // 计算 10!
        ```

    3.  **计算完成，硬件自动操作**:
        - 设备内部逻辑将 `0x24` 寄存器的 **Bit 0** 置 1。现在 `0x24` 的值是 `0x01`。
        - 设备检查 `0x20` 的 bit 7，发现是 1（使能），于是向 CPU 发出中断信号。

    4.  **驱动中断处理程序**:
        ```c
        irq_handler() {
            // 1. 读取中断状态寄存器，识别中断源
            uint32_t status = read_mmio(DEVICE_BASE + 0x24);
            
            if (status & 0x01) { // 检查是否是阶乘计算完成中断
                // 2. 处理中断：读取结果
                uint32_t result = read_mmio(DEVICE_BASE + 0x08);
                // ... 处理结果 ...
                
                // 3. 【关键】清除中断状态！
                // 通常通过向一个“中断应答寄存器”写入来实现，比如 0x68。
                // 这个操作会将 0x24 寄存器中对应的位清零，并让设备拉低中断信号线。
                write_mmio(DEVICE_BASE + 0x68, 0x01); // 假设向 0x68 写入要清除的位模式
            }
        }
        ```

    #### 场景二：软件模拟中断（调试/测试）

    你想在不真正执行阶乘计算的情况下，测试你的中断处理程序是否能正确工作。

    1.  **使能中断** (同上)。
    2.  **手动触发中断**:
        ```c
        // 向中断触发寄存器(0x60)的 bit 0 写入 1，模拟阶乘完成事件
        write_mmio(DEVICE_BASE + 0x60, 0x01);
        ```
        - 这个操作会立即使设备：
            a. 将 `0x24` 寄存器的 Bit 0 置 1。
            b. 因为中断已使能，所以立即向 CPU 发出中断信号。
    3.  **驱动中断处理程序** (同上):
        - 读取 `0x24`，发现值是 `0x01`。
        - 执行相应的处理逻辑。
        - 清除中断。

    ### 总结：三个寄存器之间的关系

    | 寄存器地址 | 名称 | 核心功能 | 类比 |
    | :--- | :--- | :--- | :--- |
    | **`0x20` Bit 7** | 中断使能位 | **总开关**。控制设备是否允许向系统“大喊”有事情发生。 | 房间的**门铃开关**。 |
    | **`0x24`** | 中断状态寄存器 | **事件记录本**。只读，记录了所有已经发生但尚未处理的事件。 | 房间里的**待办事项清单**。 |
    | **`0x60`** | 中断触发寄存器 | **手动触发器**。只写，用于软件主动在“事件记录本”上添加一条记录。 | 你可以**自己**在待办清单上写一条事项。 |

    **完整中断链**:
    - **条件**：`中断状态寄存器 (0x24) != 0` **并且** `中断使能 (0x20 bit 7) == 1`
    - **结果**：设备向 CPU 发起中断请求。

    这种设计清晰地分离了**中断源状态**、**中断全局控制**和**软件调试接口**，是硬件设计中一个非常典型和优秀的模式。

* linux 如何拿到网卡的理由速率，比如 max_link_speed

    在 Linux 中获取网卡的理论速率（最大链路速度）有多种方法，以下是几种常用的方式：

    1. **使用 `ethtool` 命令（最常用）**

        ```bash
        # 查看指定网卡的信息
        ethtool eth0

        # 或直接查看速度和双工模式
        ethtool eth0 | grep -E "Speed|Duplex"
        ```

        输出示例：
        ```
        Settings for eth0:
            Speed: 1000Mb/s
            Duplex: Full
            Port: Twisted Pair
            ...
        ```

    2. **查看 `/sys/class/net/` 系统文件**

        ```bash
        # 查看当前实际速度
        cat /sys/class/net/eth0/speed

        # 查看支持的速度列表
        cat /sys/class/net/eth0/supported

        # 查看最大支持速度
        cat /sys/class/net/eth0/speed 2>/dev/null || \
        echo "查看支持的速率: $(cat /sys/class/net/eth0/supported)"
        ```

    3. **使用 `ip` 命令**

        ```bash
        # 查看网络接口状态（包含部分速率信息）
        ip -s link show eth0
        ```

    4. **查看 `/proc/net/dev`**

        ```bash
        # 显示所有网络接口统计信息
        cat /proc/net/dev

        # 或者
        less /proc/net/dev
        ```

    5. **使用 `lshw` 命令**

        ```bash
        # 查看详细的网络硬件信息
        sudo lshw -class network

        # 或针对特定网卡
        sudo lshw -class network | grep -A 20 "eth0"
        ```

    6. **使用 `dmidecode`（查看主板集成的网卡信息）**

        ```bash
        sudo dmidecode -t baseboard
        ```

    7. **查看 PCI 信息**

        ```bash
        # 找到网卡的PCI地址
        lspci | grep -i ethernet

        # 查看详细信息（替换为你的PCI地址）
        lspci -vvv -s 00:03.0 | grep -i speed
        ```

    8. **脚本示例：自动获取网卡最大速率**

        ```bash
        #!/bin/bash

        # 获取所有网卡名称
        INTERFACES=$(ls /sys/class/net/ | grep -v lo)

        for iface in $INTERFACES; do
            echo "=== 网卡: $iface ==="
            
            # 尝试获取当前速度
            if [ -f "/sys/class/net/$iface/speed" ]; then
                speed=$(cat /sys/class/net/$iface/speed)
                if [ "$speed" -ge 0 ]; then
                    echo "当前速率: ${speed}Mb/s"
                fi
            fi
            
            # 获取支持的速度列表
            if [ -f "/sys/class/net/$iface/supported" ]; then
                echo "支持速率: $(cat /sys/class/net/$iface/supported)"
            fi
            
            # 使用ethtool获取更多信息
            ethtool $iface 2>/dev/null | grep -E "Speed|Duplex|Supported link modes" | head -5
            echo ""
        done
        ```

    9. **使用 NetworkManager 工具（如果使用NetworkManager）**

        ```bash
        # 查看连接信息
        nmcli device show eth0 | grep -i speed
        ```

    注意事项：

    1. **权限需求**：大部分命令需要 root 权限或 sudo

    2. **速度单位**：

       - 大部分输出以 Mb/s（兆比特每秒）为单位
       - 注意区分 Mb（兆比特）和 MB（兆字节）：1 MB = 8 Mb

    3. **实际速度 vs 理论速度**：

       - 这些命令显示的是链路的协商速度
       - 实际传输速度受多种因素影响（协议开销、网络拥塞等）

    常见速率的对应关系：

    - 10 Mbps：标准以太网
    - 100 Mbps：快速以太网
    - 1000 Mbps（1 Gbps）：千兆以太网
    - 2500 Mbps（2.5 Gbps）
    - 5000 Mbps（5 Gbps）
    - 10000 Mbps（10 Gbps）：万兆以太网

    **推荐使用 `ethtool` 命令**，因为它提供的信息最全面，包括：

    - 当前协商的速度
    - 支持的速度模式
    - 双工模式
    - 自协商状态
    - 物理连接类型等

* qemu 中网卡的速率为 unknown

    ```
    (base) siorigin@q35:/sys/devices/pci0000:00/0000:00:02.0$ lspci | grep -i ether
    00:02.0 Ethernet controller: Intel Corporation 82574L Gigabit Network Connection
    (base) siorigin@q35:/sys/devices/pci0000:00/0000:00:02.0$ cat ./max_link_speed 
    Unknown
    ```

    这是为什么

    这个问题很典型，我来解释一下原因和解决方法：

    **虚拟环境下的正确查看方法**

    1. 通过ethtool查看虚拟网卡的能力

        ```bash
        # 在宿主机或虚拟机内执行
        ethtool eth0

        # 查看支持的模式
        ethtool eth0 | grep -i "supported"
        ```

    2. 查看虚拟机的配置

        ```bash
        # 检查虚拟机 XML 配置（在宿主机上）
        virsh dumpxml <虚拟机名称> | grep -i "model" -A 2 -B 2

        # 对于 QEMU 直接启动的情况
        ps aux | grep qemu | grep -o "net[^ ]*"
        ```

    3. 通过设备ID确认能力

        ```bash
        # 查看详细的PCI信息
        lspci -vvv -s 00:02.0

        # 查找Capabilities部分
        lspci -vvv -s 00:02.0 | grep -i "capability\|speed"
        ```

    4. 虚拟网卡速率的实际情况

        对于虚拟环境：

        1. **82574L是模拟的Intel千兆网卡**，理论上支持 1 Gbps

        2. **实际速率受限于**：

           - 宿主机的物理网卡速度
           - 虚拟化层的限制
           - 虚拟机的CPU和内存资源
           - 宿主机的负载

    5. 获取虚拟网卡最大速率的正确脚本

        ```bash
        #!/bin/bash

        # 尝试从多个来源获取网卡速度
        INTERFACE="eth0"  # 修改为你的网卡名

        echo "=== 检查网卡: $INTERFACE ==="

        # 1. 首先尝试ethtool
        echo "1. ethtool 信息:"
        if command -v ethtool &> /dev/null; then
            ethtool $INTERFACE 2>/dev/null | grep -E "Speed|Supported link modes|Advertised link modes"
        else
            echo "ethtool 未安装"
        fi

        # 2. 查看sysfs
        echo -e "\n2. sysfs 信息:"
        SYSFS_PATH="/sys/class/net/$INTERFACE"
        if [ -d "$SYSFS_PATH" ]; then
            if [ -f "$SYSFS_PATH/speed" ]; then
                speed=$(cat "$SYSFS_PATH/speed" 2>/dev/null)
                echo "速度: ${speed:-Unknown} Mb/s"
            fi
            
            if [ -f "$SYSFS_PATH/device/max_link_speed" ]; then
                max_speed=$(cat "$SYSFS_PATH/device/max_link_speed" 2>/dev/null)
                echo "最大链路速度: ${max_speed}"
            fi
        fi

        # 3. 查看PCI信息
        echo -e "\n3. PCI 信息:"
        if command -v lspci &> /dev/null; then
            PCI_ADDR=$(basename $(readlink -f "$SYSFS_PATH/device"))
            echo "PCI地址: $PCI_ADDR"
            lspci -s "$PCI_ADDR" -vvv 2>/dev/null | grep -i "ethernet\|controller\|gigabit"
        fi

        # 4. 检查是否是虚拟设备
        echo -e "\n4. 设备类型:"
        if [ -f "$SYSFS_PATH/device/driver/module" ]; then
            driver=$(basename $(readlink -f "$SYSFS_PATH/device/driver"))
            echo "驱动: $driver"
            
            # 检查是否是虚拟设备驱动
            if [[ "$driver" == *virtio* ]] || [[ "$driver" == *veth* ]] || [[ "$driver" == *vmxnet* ]]; then
                echo "这是虚拟网卡设备"
            fi
        fi
        ```

    6. **对于虚拟化环境的建议**

        1. **如果需要在虚拟机中获得确定的速率**：

            ```bash
            # 设置固定的链路速度（如果有权限）
            ethtool -s eth0 speed 1000 duplex full autoneg off

            # 或者通过虚拟机管理器配置
            ```

        2. **实际测试速率**：

           ```bash
           # 使用iperf3测试实际吞吐量
           # 在一台机器上运行服务器
           iperf3 -s
           
           # 在另一台机器上运行客户端
           iperf3 -c <服务器IP>
           ```

        3. **查看QEMU虚拟机的网络配置**：

           ```bash
           # 在宿主机上查看虚拟机的网卡类型
           virsh domiflist <虚拟机名称>
           ```

    **总结**

    你的`max_link_speed`显示"Unknown"是因为：

    1. **虚拟化环境**：QEMU模拟的82574L网卡可能没有完全实现所有硬件特性
    2. **驱动限制**：虚拟设备的驱动可能不提供这个信息
    3. **sysfs属性不完整**：虚拟设备通常只实现必要的功能

    **最可靠的方法是使用 `ethtool eth0`**，它会显示当前协商的速度，对于82574L这种模拟的千兆网卡，通常会显示1000Mb/s（如果网络环境支持）。

    * 我没法使用工具，是否有编程的方式获得网卡的最大速率

    有！以下是几种纯编程方式获取网卡最大速率的实现：

    1. **使用 sysfs 接口（C语言）**

        ```c
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <dirent.h>
        #include <unistd.h>

        // 从/sys/class/net/读取网卡最大速率
        int get_max_speed_sysfs(const char *ifname) {
            char path[256];
            FILE *fp;
            int speed = -1;
            
            // 方法1：尝试读取speed文件（当前协商速度）
            snprintf(path, sizeof(path), "/sys/class/net/%s/speed", ifname);
            fp = fopen(path, "r");
            if (fp) {
                fscanf(fp, "%d", &speed);
                fclose(fp);
                if (speed > 0) return speed;
            }
            
            // 方法2：尝试读取supported文件解析最大速度
            snprintf(path, sizeof(path), "/sys/class/net/%s/supported", ifname);
            fp = fopen(path, "r");
            if (fp) {
                char line[256];
                int max_speed = 0;
                
                while (fgets(line, sizeof(line), fp)) {
                    // 解析支持的速率，格式如: 1000baseT/Full
                    char *token = strtok(line, " \n");
                    while (token) {
                        if (strstr(token, "base")) {
                            char *base = strstr(token, "base");
                            char speed_str[32] = {0};
                            strncpy(speed_str, token, base - token);
                            int s = atoi(speed_str);
                            if (s > max_speed) max_speed = s;
                        }
                        token = strtok(NULL, " \n");
                    }
                }
                fclose(fp);
                if (max_speed > 0) return max_speed;
            }
            
            return -1; // 无法获取
        }

        int main() {
            const char *ifname = "eth0";
            int speed = get_max_speed_sysfs(ifname);
            if (speed > 0) {
                printf("%s max speed: %d Mbps\n", ifname, speed);
            } else {
                printf("Cannot get max speed for %s\n", ifname);
            }
            return 0;
        }
        ```

    2. **使用 ioctl 和 SIOCETHTOOL（C语言）**

        ```c
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <sys/ioctl.h>
        #include <net/if.h>
        #include <linux/ethtool.h>
        #include <linux/sockios.h>
        #include <unistd.h>
        #include <errno.h>

        // 获取网卡支持的最大速率
        int get_max_speed_ethtool(const char *ifname) {
            int sockfd;
            struct ifreq ifr;
            struct ethtool_cmd edata;
            int max_speed = 0;
            
            // 创建socket
            sockfd = socket(AF_INET, SOCK_DGRAM, 0);
            if (sockfd < 0) {
                perror("socket");
                return -1;
            }
            
            // 设置ifreq结构
            memset(&ifr, 0, sizeof(ifr));
            strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
            
            // 设置ethtool命令
            memset(&edata, 0, sizeof(edata));
            edata.cmd = ETHTOOL_GSET;
            ifr.ifr_data = (char *)&edata;
            
            // 执行ioctl调用
            if (ioctl(sockfd, SIOCETHTOOL, &ifr) < 0) {
                perror("ioctl");
                close(sockfd);
                return -1;
            }
            
            close(sockfd);
            
            // 从ethtool_cmd解析速度
            // 检查是否有有效速度
            if (edata.speed != SPEED_UNKNOWN && edata.speed != 0) {
                max_speed = edata.speed;
            }
            
            // 如果当前速度为0，尝试从支持的模式中获取
            if (max_speed <= 0) {
                // 从supported字段解析支持的速度
                // supported是位掩码，需要解析
                u32 supported = edata.supported;
                
                // 检查支持的速度（单位：Mb/s）
                if (supported & SUPPORTED_10000baseT_Full) max_speed = 10000;
                else if (supported & SUPPORTED_5000baseT_Full) max_speed = 5000;
                else if (supported & SUPPORTED_2500baseT_Full) max_speed = 2500;
                else if (supported & SUPPORTED_1000baseT_Full) max_speed = 1000;
                else if (supported & SUPPORTED_100baseT_Full) max_speed = 100;
                else if (supported & SUPPORTED_10baseT_Full) max_speed = 10;
            }
            
            return max_speed;
        }

        int main() {
            const char *ifname = "eth0";
            int speed = get_max_speed_ethtool(ifname);
            if (speed > 0) {
                printf("%s max speed: %d Mbps\n", ifname, speed);
            } else {
                printf("Cannot get max speed for %s\n", ifname);
            }
            return 0;
        }
        ```

    3. **使用 netlink 接口（C语言）**

        ```c
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <unistd.h>
        #include <sys/socket.h>
        #include <linux/netlink.h>
        #include <linux/rtnetlink.h>
        #include <linux/if_link.h>

        #define BUFFER_SIZE 8192

        int get_link_speed_netlink(const char *ifname) {
            int sockfd;
            struct sockaddr_nl sa;
            struct nlmsghdr *nlh;
            struct ifinfomsg *ifinfo;
            char buffer[BUFFER_SIZE];
            int len;
            
            // 创建netlink socket
            sockfd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
            if (sockfd < 0) {
                perror("socket");
                return -1;
            }
            
            memset(&sa, 0, sizeof(sa));
            sa.nl_family = AF_NETLINK;
            sa.nl_pid = getpid();
            
            if (bind(sockfd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
                perror("bind");
                close(sockfd);
                return -1;
            }
            
            // 准备netlink消息
            nlh = (struct nlmsghdr *)buffer;
            nlh->nlmsg_len = NLMSG_LENGTH(sizeof(struct ifinfomsg));
            nlh->nlmsg_type = RTM_GETLINK;
            nlh->nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
            nlh->nlmsg_seq = 1;
            nlh->nlmsg_pid = getpid();
            
            ifinfo = NLMSG_DATA(nlh);
            memset(ifinfo, 0, sizeof(struct ifinfomsg));
            
            // 发送请求
            if (send(sockfd, nlh, nlh->nlmsg_len, 0) < 0) {
                perror("send");
                close(sockfd);
                return -1;
            }
            
            // 接收响应
            len = recv(sockfd, buffer, BUFFER_SIZE, 0);
            if (len < 0) {
                perror("recv");
                close(sockfd);
                return -1;
            }
            
            // 解析响应
            for (nlh = (struct nlmsghdr *)buffer; NLMSG_OK(nlh, len); nlh = NLMSG_NEXT(nlh, len)) {
                struct rtattr *rta;
                int rtalen;
                
                ifinfo = NLMSG_DATA(nlh);
                
                // 查找IFLA_LINKINFO属性
                rta = IFLA_RTA(ifinfo);
                rtalen = IFLA_PAYLOAD(nlh);
                
                for (; RTA_OK(rta, rtalen); rta = RTA_NEXT(rta, rtalen)) {
                    if (rta->rta_type == IFLA_IFNAME) {
                        char *name = (char *)RTA_DATA(rta);
                        if (strcmp(name, ifname) == 0) {
                            // 找到目标网卡，查找速度属性
                            // 注意：标准netlink可能不直接提供最大速度
                            // 需要进一步解析IFLA_LINKINFO
                            close(sockfd);
                            // 这里需要更复杂的解析，通常需要结合ethtool
                            return -1; // 简化处理
                        }
                    }
                }
            }
            
            close(sockfd);
            return -1;
        }
        ```

    4. **使用 /proc/net/dev 解析（C语言）**

        ```c
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <ctype.h>

        // 从/proc/net/dev获取接口信息
        int parse_proc_net_dev(const char *ifname) {
            FILE *fp;
            char line[512];
            
            fp = fopen("/proc/net/dev", "r");
            if (!fp) {
                return -1;
            }
            
            // 跳过前两行表头
            fgets(line, sizeof(line), fp);
            fgets(line, sizeof(line), fp);
            
            while (fgets(line, sizeof(line), fp)) {
                char *name;
                char *ptr = line;
                
                // 跳过前导空格
                while (*ptr && isspace(*ptr)) ptr++;
                
                name = ptr;
                
                // 找到接口名结束位置
                while (*ptr && *ptr != ':') ptr++;
                if (*ptr == ':') {
                    *ptr = '\0';
                    
                    // 去掉接口名末尾的空格
                    char *end = ptr - 1;
                    while (end > name && isspace(*end)) {
                        *end = '\0';
                        end--;
                    }
                    
                    if (strcmp(name, ifname) == 0) {
                        fclose(fp);
                        // /proc/net/dev 不包含速度信息，只包含统计
                        // 这里只返回找到了接口
                        return 0;
                    }
                }
            }
            
            fclose(fp);
            return -1;
        }
        ```

    5. **综合方法：自动选择最佳途径（C语言）**

        ```c
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <unistd.h>
        #include <sys/ioctl.h>
        #include <net/if.h>
        #include <linux/ethtool.h>
        #include <linux/sockios.h>
        #include <dirent.h>

        // 尝试多种方法获取网卡最大速率
        int get_network_speed(const char *ifname) {
            int speed = -1;
            char path[256];
            FILE *fp;
            
            // 方法1：尝试sysfs speed文件
            snprintf(path, sizeof(path), "/sys/class/net/%s/speed", ifname);
            fp = fopen(path, "r");
            if (fp) {
                fscanf(fp, "%d", &speed);
                fclose(fp);
                if (speed > 0) return speed;
            }
            
            // 方法2：尝试ioctl+ethtool
            int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
            if (sockfd >= 0) {
                struct ifreq ifr;
                struct ethtool_cmd edata;
                
                memset(&ifr, 0, sizeof(ifr));
                strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
                
                memset(&edata, 0, sizeof(edata));
                edata.cmd = ETHTOOL_GSET;
                ifr.ifr_data = (char *)&edata;
                
                if (ioctl(sockfd, SIOCETHTOOL, &ifr) >= 0) {
                    if (edata.speed != SPEED_UNKNOWN && edata.speed != 0) {
                        speed = edata.speed;
                    }
                }
                close(sockfd);
            }
            
            return speed;
        }

        // 列出所有网络接口及其速度
        void list_network_interfaces() {
            DIR *dir;
            struct dirent *entry;
            
            dir = opendir("/sys/class/net");
            if (!dir) {
                return;
            }
            
            printf("Network Interfaces:\n");
            printf("===================\n");
            
            while ((entry = readdir(dir)) != NULL) {
                if (entry->d_name[0] == '.') continue;
                if (strcmp(entry->d_name, "lo") == 0) continue;
                
                int speed = get_network_speed(entry->d_name);
                if (speed > 0) {
                    printf("%-10s: %4d Mbps\n", entry->d_name, speed);
                } else {
                    printf("%-10s: Unknown speed\n", entry->d_name);
                }
            }
            
            closedir(dir);
        }

        int main(int argc, char *argv[]) {
            if (argc > 1) {
                // 获取指定接口的速度
                int speed = get_network_speed(argv[1]);
                if (speed > 0) {
                    printf("%s: %d Mbps\n", argv[1], speed);
                } else {
                    printf("Cannot determine speed for %s\n", argv[1]);
                }
            } else {
                // 列出所有接口
                list_network_interfaces();
            }
            
            return 0;
        }
        ```

    6. **Python实现（使用纯Python，无需外部工具）**

        ```python
        #!/usr/bin/env python3
        import os
        import struct
        import socket
        import fcntl
        from ctypes import *

        # Linux ioctl常量
        SIOCETHTOOL = 0x8946
        ETHTOOL_GSET = 0x00000001

        # ifreq结构
        class ifreq(Structure):
            _fields_ = [
                ("ifr_name", c_char * 16),
                ("ifr_data", c_void_p)
            ]

        # ethtool_cmd结构（简化）
        class ethtool_cmd(Structure):
            _fields_ = [
                ("cmd", c_uint32),
                ("supported", c_uint32),
                ("advertising", c_uint32),
                ("speed", c_uint16),
                ("duplex", c_uint8),
                ("port", c_uint8),
                ("phy_address", c_uint8),
                ("transceiver", c_uint8),
                ("autoneg", c_uint8),
                ("mdio_support", c_uint8),
                ("maxtxpkt", c_uint32),
                ("maxrxpkt", c_uint32),
                ("reserved", c_uint32 * 2)
            ]

        def get_ethtool_speed(interface):
            """使用ioctl获取网卡速度"""
            try:
                # 创建socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                # 准备ifreq结构
                ifr = ifreq()
                ifr.ifr_name = interface.encode()
                
                # 准备ethtool_cmd结构
                edata = ethtool_cmd()
                edata.cmd = ETHTOOL_GSET
                ifr.ifr_data = addressof(edata)
                
                # 调用ioctl
                fcntl.ioctl(sock.fileno(), SIOCETHTOOL, ifr)
                
                # 返回速度
                speed = edata.speed
                sock.close()
                
                if speed == 65535 or speed == 0:  # SPEED_UNKNOWN
                    return -1
                return speed
                
            except Exception as e:
                return -1

        def get_sysfs_speed(interface):
            """从sysfs获取网卡速度"""
            speed_path = f"/sys/class/net/{interface}/speed"
            if os.path.exists(speed_path):
                try:
                    with open(speed_path, 'r') as f:
                        speed = int(f.read().strip())
                        if speed > 0:
                            return speed
                except:
                    pass
            return -1

        def get_max_link_speed(interface="eth0"):
            """获取网卡最大链路速度"""
            # 尝试sysfs
            speed = get_sysfs_speed(interface)
            if speed > 0:
                return speed
            
            # 尝试ethtool ioctl
            speed = get_ethtool_speed(interface)
            if speed > 0:
                return speed
            
            return -1

        # 使用示例
        if __name__ == "__main__":
            interfaces = [d for d in os.listdir("/sys/class/net") 
                          if d != "lo" and not d.startswith('.')]
            
            for iface in interfaces:
                speed = get_max_link_speed(iface)
                if speed > 0:
                    print(f"{iface}: {speed} Mbps")
                else:
                    print(f"{iface}: Unknown speed")
        ```

    7. 最简化的 sysfs 读取（C语言）

        ```c
        #include <stdio.h>

        int main() {
            FILE *fp = fopen("/sys/class/net/eth0/speed", "r");
            if (fp) {
                int speed;
                fscanf(fp, "%d", &speed);
                fclose(fp);
                printf("Speed: %d Mbps\n", speed);
            } else {
                printf("Cannot read speed file\n");
            }
            return 0;
        }
        ```

    **编译和使用**

    ```bash
    # 编译C程序
    gcc -o get_speed get_speed.c

    # 运行（需要root权限读取ethtool信息）
    sudo ./get_speed eth0
    ```

    **注意事项**

    1. **权限问题**：

        - 读取 `/sys/class/net/eth0/speed` 通常不需要特殊权限
        - 使用 `ioctl` 需要root权限或CAP_NET_ADMIN能力

    2. **返回值说明**：

        - `-1` 或 `65535` 通常表示速度未知
        - `0` 可能表示链路断开
        - 正值表示速度（单位：Mb/s）

    3. **虚拟环境限制**：

        - 在虚拟化环境中，某些方法可能返回未知
        - 最可靠的是读取 `/sys/class/net/eth0/speed`

    **推荐使用第一种sysfs方法**，它最简单且不需要特殊权限。如果获取失败，再尝试使用ioctl方法（需要root权限）。

* qemu iommu no space left

    ```
    qemu-system-x86_64: VFIO_MAP_DMA failed: No space left on device
    qemu-system-x86_64: vfio_container_dma_map(0x55a5c3949a80,. 0x7c561000, 0x1000, 0x7f56e556d000) = -28 (No space lefton device)
    ```

    这个错误是因为 **IOMMU IOVA（Input/Output Virtual Address）空间不足**，导致VFIO无法为虚拟机分配DMA映射空间。

    **原因分析**

    1. **IOVA空间耗尽**：Linux内核分配给IOMMU的地址空间不足
    2. **过多的DMA映射**：可能因为虚拟机频繁的PCIe设备操作或内存热插拔
    3. **IOMMU表碎片化**：长期运行后IOMMU映射表出现碎片

    **解决方案**

    1. 立即解决方法

        ```bash
        # 重启宿主机 - 最直接的方法
        sudo reboot

        # 或者重启IOMMU子系统
        sudo sh -c "echo 1 > /sys/class/iommu/dmar*/reset"
        ```

    2. 增加 IOVA 空间
        
        修改 GRUB 配置增加 IOMMU 地址空间：

        ```bash
        # 编辑GRUB配置
        sudo nano /etc/default/grub
        ```

        在 GRUB_CMDLINE_LINUX 中添加：

        ```bash
        # iommu=pt,force  # 强制使用IOMMU
        # iommu.passthrough=1  # 直通模式
        # 增加IOVA空间（64GB示例）
        iommu=relaxable iommu.forcedac=0 iommu.strict=0
        ```

        更新 GRUB:

        ```bash
        sudo update-grub
        sudo reboot
        ```

        注：

        1. 没看明白这个

    3. 检查当前 IOMMU 状态
        
        ```bash
        # 查看IOMMU状态
        dmesg | grep -i iommu
        sudo cat /sys/kernel/debug/iommu/intel-iommu/registers  # Intel
        sudo cat /sys/kernel/debug/iommu/amd-iommu/registers    # AMD

        # 查看VFIO映射
        sudo cat /sys/kernel/debug/iommu/domain*/info
        ```

    4. QEMU 参数优化

        ```bash
        # 在启动QEMU时添加这些参数：
        -object iommu-ioas,id=ioa0,max-ioas=64
        -iommu iova=ioa0

        # 或者使用更大的IOMMU页面
        -machine q35,iommu=on,iommu-type=smmuv3
        ```

    5. **内核参数调优**
        
        ```bash
        # 临时调整
        sudo sh -c "echo 512 > /sys/module/vfio_iommu_type1/parameters/max_dma_mask"
        sudo sh -c "echo 65536 > /sys/module/vfio_iommu_type1/parameters/dma_entry_limit"

        # 永久配置
        sudo nano /etc/modprobe.d/vfio.conf
        # 添加：
        options vfio_iommu_type1 dma_entry_limit=65536 max_dma_mask=512
        ```

    6. 清理 IOMMU 映射

        ```bash
        # 卸载所有VFIO设备
        virsh nodedev-detach pci_xxxx_xx_xx_x

        # 重新加载VFIO模块
        sudo modprobe -r vfio_iommu_type1 vfio
        sudo modprobe vfio
        sudo modprobe vfio_iommu_type1
        ```

    7. 预防措施

        * 监控 IOMMU 使用情况

            ```bash
            watch -n 1 'cat /sys/kernel/debug/iommu/domain*/info 2>/dev/null'
            ```

        * 定期重启长时间运行的虚拟机

        * 减少PCIe设备的热插拔操作

    8. **如果使用 Libvirt**

        ```xml
        <!-- 在domain配置中添加 -->
        <domain type='kvm' xmlns:qemu='http://libvirt.org/schemas/domain/qemu/1.0'>
          <qemu:commandline>
            <qemu:arg value='-object'/>
            <qemu:arg value='iommu-ioas,id=ioa0,max-ioas=128'/>
            <qemu:arg value='-iommu'/>
            <qemu:arg value='iova=ioa0'/>
          </qemu:commandline>
        </domain>
        ```

    这个问题通常出现在：

    - 运行多个GPU直通的虚拟机
    - 虚拟机进行大量 DMA 操作
    - 宿主机长时间运行不重启
    - 使用老版本内核（<5.4）
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

# Linux Driver Note

Ref:

* <https://embetronicx.com/tutorials/linux/device-drivers/>

## cache

* `int m_open(struct inode *inode, struct file *file_ptr);`

    函数原型

    ```c
    int m_open(struct inode *inode, struct file *file_ptr);
    ```

    参数说明

    1. struct inode *inode

        * 作用：表示内核中文件的索引节点（inode）

        * 包含信息：

            * 设备号（主设备号、次设备号）

            * 文件权限（读/写/执行权限）

            * 文件所有者（UID/GID）

            * 文件大小

            * 时间戳（创建、修改、访问时间）

            * 指向文件操作函数的指针等

    2. struct file *file_ptr

        * 作用：表示打开的文件实例

        * 包含信息：

            * 文件操作函数指针（file_operations）

            * 文件当前位置（f_pos）

            * 打开模式（f_flags - O_RDONLY, O_WRONLY, O_RDWR等）

            * 私有数据指针（private_data）

            * 引用计数等

    返回值

    成功时

    * 返回 0：表示打开操作成功

    * 可以通过file_ptr->private_data存储设备特定数据

    失败时

    * 返回负的错误码：

        * -ENOMEM：内存不足

        * -EBUSY：设备忙

        * -EACCES：权限不足

        * -ENODEV：设备不存在

        * -EINVAL：无效参数

    函数作用

    主要功能：

    * 初始化设备：在第一次打开时初始化硬件设备

    * 验证权限：检查用户是否有权限访问设备

    * 分配资源：为文件操作分配必要的资源

    * 设置私有数据：在file_ptr->private_data中存储设备特定信息

    * 更新引用计数：管理设备的打开次数

    使用示例

    ```c
    static int mydev_open(struct inode *inode, struct file *file)
    {
        struct my_device *dev;
        
        // 1. 获取设备结构
        dev = container_of(inode->i_cdev, struct my_device, cdev);
        
        // 2. 检查设备是否可用
        if (!dev->ready)
            return -EBUSY;
        
        // 3. 存储私有数据
        file->private_data = dev;
        
        // 4. 更新设备使用计数
        atomic_inc(&dev->usage_count);
        
        return 0;  // 成功
    }
    ```

    调用时机

    * 当用户空间调用open()系统调用打开字符设备文件时

    * 由VFS（虚拟文件系统）调用，作为file_operations结构体中的成员

    * 通常对应驱动程序的.open函数指针

    这是Linux字符设备驱动开发中的核心函数之一，用于设备文件的初始化和准备工作。

* 简述 linux driver 开发中 ioctl 的用法

    在 Linux 驱动开发中，ioctl 是用于设备控制的重要接口。以下是其核心用法：

    1. 基本概念

        ioctl（输入/输出控制）允许用户空间程序与设备驱动进行复杂的交互，传递命令和数据，适用于无法用标准 read/write 操作实现的设备控制。

    2. 用户空间调用

        ```c
        #include <sys/ioctl.h>
        int ioctl(int fd, unsigned long cmd, ...);
        ```

        * fd：设备文件描述符

        * cmd：预定义的命令码

        * ...：可选参数（通常是指向数据的指针）

    3. 驱动端实现

        在驱动中定义 file_operations 结构体的 .unlocked_ioctl 或 .compat_ioctl 成员：

        ```c
        long my_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
        {
            switch (cmd) {
                case MY_CMD1:
                    // 处理命令1
                    break;
                case MY_CMD2:
                    // 处理命令2
                    break;
                default:
                    return -ENOTTY;  // 非法命令
            }
            return 0;
        }
        ```

    4. 命令码构造

        使用内核提供的宏构造命令码：

        ```c
        #include <linux/ioctl.h>

        #define MY_MAGIC 'x'  // 幻数（0-255）
        #define MY_CMD1 _IOR(MY_MAGIC, 1, int)
        #define MY_CMD2 _IOW(MY_MAGIC, 2, struct my_data)
        ```

        常用宏：

        * _IO(type, nr)：无数据传输

        * _IOR(type, nr, size)：从驱动读数据

        * _IOW(type, nr, size)：向驱动写数据

        * _IOWR(type, nr, size)：双向传输

    5. 数据交换

        通过 arg 参数传递数据地址，需要使用 copy_from_user() 和 copy_to_user() 在用户/内核空间之间安全复制数据：

        ```c
        struct my_data data;

        if (cmd == MY_CMD2) {
            if (copy_from_user(&data, (void __user *)arg, sizeof(data)))
                return -EFAULT;
            // 处理数据
        }
        ```

    6. 权限控制

        使用 _IOC_DIR(cmd) 检查命令方向，_IOC_SIZE(cmd) 获取数据大小，并使用 access_ok() 验证用户指针有效性。

    7. 注意事项

        * 确保命令码在驱动内唯一

        * 正确处理32/64位兼容性（使用 .compat_ioctl）

        * 避免使用已在内核中注册的幻数

        * 考虑使用替代方案如 sysfs 或 configfs 进行简单配置

    8. 替代方案

        对于新驱动，考虑：

        * sysfs：导出设备属性

        * debugfs：调试接口

        * netlink：网络相关配置

        * 专用设备文件：复杂数据交换

    ioctl 提供了最大的灵活性，但应谨慎使用以确保接口的稳定性和安全性。

* linux sysfs 中的 numa_node 数据是谁来填充的？

    在 Linux sysfs 中，numa_node 数据的填充是由设备驱动程序和内核的核心内存管理代码共同完成的。具体来说：

    主要填充机制：

    1. 设备驱动程序

        * 当设备驱动程序调用 pci_register_device() 或类似函数注册设备时

        * 驱动程序会设置设备的 NUMA 节点信息

        * 对于 PCI 设备，通常在 struct pci_dev 中设置 dev.numa_node 字段

    2. PCI 子系统

        * PCI 核心代码会自动从 ACPI（ACPI SRAT 表）或设备树（DT）中获取 NUMA 信息

        * 对于支持 NUMA 的平台，PCI 设备会继承其所属 PCI 主机桥的 NUMA 节点

    3. 内存分配路径

        * 当设备需要分配 DMA 缓冲区时

        * 内核的 DMA 映射层（dma_alloc_coherent() 等）会设置设备的 NUMA 节点

    4. 平台固件信息

        * 在系统启动时，内核从 ACPI SRAT（Static Resource Affinity Table）读取 NUMA 拓扑

        * 或从设备树中解析 NUMA 信息

    具体代码路径：

    ```c
    // 示例：PCI 设备 NUMA 节点设置
    pci_device_add()
      → pcibios_add_device()
        → acpi_pci_root_add()  // 从 ACPI 获取 NUMA 信息
          → acpi_get_node()    // 获取 NUMA 节点 ID

    // 在 sysfs 中的体现
    static ssize_t numa_node_show(struct device *dev,
                                 struct device_attribute *attr, char *buf)
    {
        return sprintf(buf, "%d\n", dev->numa_node);
    }
    static DEVICE_ATTR_RO(numa_node);
    ```

    查看示例：

    ```bash
    # 查看 PCI 设备的 NUMA 节点
    $ cat /sys/bus/pci/devices/0000:00:1f.0/numa_node
    0

    # 查看内存设备的 NUMA 节点
    $ cat /sys/bus/node/devices/node0/cpulist
    0-15
    ```

    特殊情况：

    * 如果设备不支持 NUMA 或无法确定节点，会显示 -1

    * 虚拟设备可能显示为 -1 或特定的节点

    * 有些架构可能需要在设备驱动中显式设置 NUMA 节点

    总之，numa_node 是由内核根据硬件拓扑和平台固件信息自动填充，设备驱动程序通常只需要使用标准 API 注册设备即可。

* 简述 linux 中 `pci_write_config_word()`, `pci_read_config_word()` 的原型、作用和用法

    在Linux内核中，pci_write_config_word() 和 pci_read_config_word() 是用于访问PCI设备配置空间的函数。

    函数原型

    ```c
    #include <linux/pci.h>

    int pci_read_config_word(struct pci_dev *dev, int offset, u16 *value);
    int pci_write_config_word(struct pci_dev *dev, int offset, u16 value);
    ```

    作用

    这些函数用于读取和写入PCI设备的配置空间寄存器：

    * PCI配置空间：每个PCI设备都有256字节（或4096字节）的配置空间

    * 配置寄存器：包含设备ID、厂商ID、中断引脚、基地址寄存器(BAR)等关键信息

    * 硬件抽象：提供跨平台的统一访问接口

    参数说明

    * dev：指向pci_dev结构的指针，表示目标PCI设备

    * offset：要访问的配置空间偏移量（0-255或0-4095）

    * value（读函数）：存储读取值的指针

    * value（写函数）：要写入的值

    用法示例

    1. 读取设备ID和厂商ID

        ```c
        u16 vendor_id, device_id;
        struct pci_dev *pdev;

        /* 获取pci_dev指针（通常通过PCI子系统获得） */

        /* 读取厂商ID（偏移0x00） */
        pci_read_config_word(pdev, PCI_VENDOR_ID, &vendor_id);

        /* 读取设备ID（偏移0x02） */
        pci_read_config_word(pdev, PCI_DEVICE_ID, &device_id);

        printk(KERN_INFO "Device %04x:%04x\n", vendor_id, device_id);
        ```

    2. 写入配置寄存器

        ```c
        u16 command_reg;

        /* 读取命令寄存器（偏移0x04） */
        pci_read_config_word(pdev, PCI_COMMAND, &command_reg);

        /* 设置总线主控位（bit 2） */
        command_reg |= PCI_COMMAND_MASTER;

        /* 写回命令寄存器 */
        pci_write_config_word(pdev, PCI_COMMAND, command_reg);
        ```

    3. 检查函数返回值

        ```c
        int ret;
        u16 data;

        ret = pci_read_config_word(pdev, PCI_STATUS, &data);
        if (ret) {
            /* 处理错误：PCI配置空间访问失败 */
            dev_err(&pdev->dev, "Failed to read PCI status\n");
            return ret;
        }
        ```

    相关函数家族

    Linux内核提供了不同宽度的配置空间访问函数：

    ```c
    /* 字节访问 */
    pci_read_config_byte()
    pci_write_config_byte()

    /* 字访问（16位） */
    pci_read_config_word()
    pci_write_config_word()

    /* 双字访问（32位） */
    pci_read_config_dword()
    pci_write_config_dword()
    ```

    重要注意事项

    * 偏移对齐：必须使用对齐的偏移（word访问需要2字节对齐）

    * 并发安全：确保在访问PCI配置空间时持有适当的锁

    * 错误处理：始终检查返回值（0表示成功）

    * 内核上下文：只能在内核上下文中使用

    * 设备使能：访问前设备必须已使能（pci_enable_device()）

    典型使用场景

    * 设备驱动初始化时识别硬件

    * 配置中断和DMA设置

    * 设置PCI电源管理状态

    * 调试和诊断PCI设备问题

    这些函数是Linux PCI子系统的基础API，为设备驱动程序提供了标准化的硬件访问接口。

    **那我用 mmio 也可以完成这个函数的功能吧**

    可以，但通常不应该这样做。让我详细解释：

    直接使用MMIO访问PCI配置空间的限制

    1. 架构依赖性

        ```c
        /* x86架构的传统方法（不推荐） */
        unsigned int pci_config_read(int bus, int slot, int func, int offset) {
            outl(0x80000000 | (bus << 16) | (slot << 11) | (func << 8) | offset, 0xCF8);
            return inl(0xCFC);
        }
        ```

        * x86专用：使用0xCF8/0xCFC I/O端口

        * 其他架构不同：ARM、PowerPC、RISC-V等有自己的方法

        * 违反内核抽象：绕过PCI子系统

    2. 正确的MMIO访问配置空间的方法

        实际上，PCI配置空间不是通过常规MMIO访问的，而是通过：

        * x86：CONFIG_ADDRESS/CONFIG_DATA I/O端口 (0xCF8/0xCFC)

        * PCIe/现代系统：ECAM (Enhanced Configuration Access Mechanism) 机制

        * 非x86架构：各自平台的特定方法

    比较两种方法

    | 特性 | pci_read/write_config_*() | 直接MMIO/I/O |
    | - | - | - |
    | 可移植性 | ✅ 跨架构通用 | ❌ 架构相关 |
    | 并发安全 | ✅ 内部处理锁 | ❌ 需要手动同步 |
    | 电源管理 | ✅ 处理电源状态 | ❌ 设备可能处于低功耗 |
    | 错误检查 | ✅ 验证设备状态 | ❌ 无错误检查 |
    | 内核ABI稳定性 | ✅ 稳定接口 | ❌ 可能变化 |

    实际使用场景的区别

    场景1：访问配置空间 → 使用PCI配置函数

    ```c
    /* 正确方式：访问PCI配置寄存器 */
    u16 vendor_id;
    pci_read_config_word(pdev, PCI_VENDOR_ID, &vendor_id);

    /* 错误方式：试图用MMIO访问配置空间 */
    void __iomem *config_base = ioremap(...); /* 错误！无法这样映射配置空间 */
    ```

    场景2：访问设备内存/寄存器空间 → 使用MMIO

    ```c
    /* 正确方式：通过BAR访问设备内存 */
    resource_size_t bar_addr = pci_resource_start(pdev, bar_num);
    void __iomem *regs = ioremap(bar_addr, size);

    /* 读写设备寄存器 */
    u32 status = readl(regs + STATUS_OFFSET);
    writel(command, regs + COMMAND_OFFSET);
    ```

    为什么不能互换使用？

    PCI配置空间 vs 设备内存空间

    ```text
    PCI设备地址空间：
    1. 配置空间 (Configuration Space)
       ├── 256字节（传统）或4096字节（PCIe）
       ├── 包含：设备ID、厂商ID、BAR、中断等
       ├── 访问方式：专用机制（I/O端口或ECAM）
       └── 使用函数：pci_read/write_config_*()

    2. 内存/IO空间 (通过BAR映射)
       ├── 设备寄存器、内存缓冲区等
       ├── 映射到系统物理地址空间
       ├── 访问方式：MMIO（内存映射I/O）
       └── 使用函数：readl()/writel(), ioremap()
    ```

    唯一可能使用直接访问的场景

    ```c
    /* 特殊情况：在用户空间工具或特殊内核模块中 */
    #ifdef CONFIG_X86
    /* 直接I/O端口访问（仅x86，需要root权限） */
    #include <sys/io.h>

    unsigned long read_pci_config(unsigned char bus, unsigned char slot,
                                  unsigned char func, unsigned char offset) {
        unsigned long v;
        unsigned int addr = 0x80000000 | (bus << 16) | (slot << 11) | (func << 8) | offset;
        
        iopl(3);  /* 获取I/O端口访问权限 */
        outl(addr, 0xCF8);
        v = inl(0xCFC);
        iopl(0);  /* 恢复权限 */
        return v;
    }
    #endif
    ```

    现代Linux的最佳实践

    * 驱动程序开发：始终使用pci_read/write_config_*()函数族

    * 用户空间工具：使用/sys/bus/pci/devices/或/proc/bus/pci/

    * 诊断调试：使用lspci、setpci等工具

    * 内核模块：通过PCI子系统API，不要绕过

    ```bash
    # 用户空间查看PCI配置
    $ lspci -xxxx  # 显示原始配置空间
    $ setpci -s 00:01.0 COMMAND=0x143  # 修改命令寄存器
    ```

    总结

    不，不能用常规MMIO完成pci_read/write_config_word()的功能：

    * PCI配置空间使用专用访问机制，不是普通内存

    * 直接访问会破坏可移植性和安全性

    * Linux PCI子系统已经提供了正确、稳定、跨平台的接口

    除非你在写非常底层的架构特定代码或诊断工具，否则应该始终使用内核提供的PCI配置空间访问函数。

* BAR 配置空间 都有哪些内容？

    好的，这是一个非常核心的计算机体系结构问题。BAR（Base Address Register，基址地址寄存器） 是PCI/PCIe设备配置空间中的关键部分，用于实现设备的内存映射和I/O端口映射。

    简单来说，BAR的作用就是告诉系统：“我的设备内部的功能模块（如寄存器、内存缓冲区）需要被映射到系统地址空间的哪个位置”，这样CPU或其它设备才能通过访问这些系统地址来与设备通信。

    BAR 在配置空间中的位置

    在标准的PCI/PCIe 256字节（或PCIe扩展的4KB）配置空间中，有6个BAR，每个BAR为32位宽（64位BAR需要占用两个连续的32位BAR位置）。它们位于配置空间头区域（Header Region）的偏移量 0x10 到 0x24 的位置。

    * BAR0: 偏移 0x10

    * BAR1: 偏移 0x14

    * BAR2: 偏移 0x18

    * BAR3: 偏移 0x1C

    * BAR4: 偏移 0x20

    * BAR5: 偏移 0x24

    BAR 寄存器的内容解析

    一个BAR寄存器的值由两部分组成：

    * 低有效位（LSB）： 表示类型和属性（只读）。

    * 高位部分： 存储系统分配给它的基地址（可写）。

    BAR 的位定义（32位示例）

    * Bit 0: 空间类型指示位

        * 0 = 映射到内存空间。

        * 1 = 映射到I/O空间（在x86-64架构中，I/O空间已较少使用，PCIe更推荐使用内存映射）。

    * Bits 2-1 (对于内存空间): 定位类型

        * 00 = 任何32位地址均可（可映射到32位内存空间任何位置）。

        * 10 = 地址必须在64位空间（表示这是一个64位BAR，需要占用两个BAR位置）。

        * 01 = 保留（未使用）。

    * Bits 3 (对于内存空间): 预取使能位

        * 0 = 该区域不可预取（例如，设备的寄存器，读取有副作用）。

        * 1 = 该区域可预取（例如，设备的帧缓冲区或DMA缓冲区，读取无副作用）。

    * Bits (N-1):4 及更高位: 基地址字段

        * 这是实际写入基地址的地方。N 取决于BAR请求的内存大小和对齐方式。系统软件通过向BAR写入全1，再读回，来探测设备请求的地址空间大小和对齐要求。设备会通过硬连线将低位（大小指示位）保持为0，从而“告诉”系统它需要多大的空间。

    BAR 配置空间的详细内容（总结为表格）

    | BAR 类型 | 关键位 | 含义与内容 |
    | - | - | - |
    | 内存空间 BAR | Bit 0 = 0 | 声明此BAR用于映射设备内存到系统内存地址空间。 |
    | | Bits 2:1 | 00: 32位地址请求。 <br/> 10: 64位地址请求（占用下一个BAR）。 |
    | | Bit 3 | 0: 不可预取。<br/> 1: 可预取。 |
    | | Bits 31:4 (或更高) | 基地址。实际映射的物理地址。低位（由大小决定）由硬件固定为0。 |
    | I/O 空间 BAR | Bit 0 = 1 | 声明此BAR用于映射到系统I/O地址空间（x86架构）。 |
    | | Bit 1 | 保留（必须为0）。 |
    | | Bits 31:2 | 基地址。I/O端口的起始地址。低位由硬件固定为0。 |

    BAR 的配置过程（系统软件视角）

    * 枚举与探测： 系统启动时，BIOS/UEFI或操作系统遍历PCIe总线，读取每个设备的配置空间。

    * 大小查询： 对于每个BAR，系统软件向其写入全1 (0xFFFFFFFF)。

    * 读回值： 然后立即读回BAR的值。设备逻辑会“屏蔽”掉表示大小的低位（例如，需要64KB空间的BAR，其低16位会硬连线为0）。因此，读回的值可能是 0xFFFF0000。

    * 解码需求： 系统软件对读回的值取反加一，即可得到该BAR请求的地址空间大小（上例中，~0xFFFF0000 + 1 = 0x00010000，即64KB）。同时，它也知道了所需的对齐（必须是其大小的整数倍）。

    * 分配地址： 系统软件根据所有设备的请求，在物理地址空间中分配一段空闲且符合对齐要求的地址区域。

    * 写入基地址： 将分配好的物理地址（高位部分）写入BAR的基地址字段。

    * 建立映射： 操作系统内核将此物理地址范围映射到内核的虚拟地址空间（对于内存空间BAR），驱动即可通过访问这些虚拟地址来操作设备。

    每个BAR的典型用途

    一个设备可以使用全部6个BAR，每个映射到设备内部不同的功能模块。例如，一个高性能网卡：

    * BAR0: 映射控制/状态寄存器组（CSR），用于配置设备、查询状态。（不可预取，32位）

    * BAR1: 映射接收描述符环和发送描述符环的头部。（可预取，64位，可能占用BAR1和BAR2）

    * BAR2: 作为BAR1的高位部分（如果BAR1是64位）。

    * BAR3: 映射设备的统计计数器区域。

    * BAR4/BAR5: 可能用于SR-IOV功能，为每个虚拟功能分配不同的BAR。

    总结

    BAR配置空间的核心内容就是：

    * 类型：是内存映射还是I/O映射。

    * 属性：是32位还是64位，是否可预取。

    * 请求的大小和对齐：通过写全1读回机制探测。

    * 最终分配的基地址：由系统软件写入，完成地址映射。

    正是通过BAR机制，PCI/PCIe设备才能以一种灵活、统一的方式“融入”系统的地址空间，实现高效的数据交换和控制。

* linux driver 中，`__iomem`有实际用处吗？还是只是个修饰？

    `__iomem` 在 Linux 内核中既有实际用处，也是个重要的修饰符，它的作用远不止于注释。以下是详细解释：

    1. 类型安全检查

        __iomem 实际上是一个 void __iomem * 类型，编译器会进行类型检查：

        ```c
        // 正确用法
        void __iomem *addr = ioremap(phys_addr, size);
        u32 value = readl(addr);  // OK

        // 错误用法 - 编译器会警告
        u32 value = *addr;  // 直接解引用会触发警告/错误
        char *ptr = addr;   // 类型不匹配警告
        ```

    2. 防止直接访问

        IO内存不能像普通内存那样直接访问，必须使用专门的访问函数：

        ```c
        // 错误：可能引起崩溃或未定义行为
        void __iomem *io_addr = ioremap(...);
        u32 wrong = *io_addr;  // 应该用 readl(io_addr)

        // 正确：使用专门的访问函数
        u32 value = readl(io_addr);     // 32位读
        writel(value, io_addr);         // 32位写
        u16 val16 = readw(io_addr);     // 16位读
        u8 val8 = readb(io_addr);       // 8位读
        ```

    3. Sparse 静态检查工具

        Linux 内核使用 Sparse 工具进行静态分析，__iomem 是 Sparse 能够识别的"上下文注解"：

        ```c
        #ifdef __CHECKER__
        # define __iomem    __attribute__((noderef, address_space(2)))
        #else
        # define __iomem
        #endif
        ```

        * noderef：告诉 Sparse 这个指针不应该被解引用

        * address_space(2)：标识这是地址空间#2（IO内存空间）

    4. 架构特定的优化

        在某些架构上，使用 __iomem 可以启用特定的优化或屏障：

        ```c
        // 某些架构可能需要内存屏障
        #define readl(c) ({ u32 __v = le32_to_cpu(__raw_readl(__iomem_addr(c))); __iomem_barrier(); __v; })
        ```

    5. 代码文档作用

        虽然不是主要功能，但 __iomem 作为文档也很重要：

        * 明确标识这是映射的IO内存

        * 提醒开发者使用正确的访问函数

        * 区分普通内存指针和IO内存指针

    实际示例

    ```c
    // 驱动中典型的用法
    static int my_driver_probe(struct platform_device *pdev)
    {
        struct resource *res;
        void __iomem *regs;
        
        // 获取设备资源
        res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
        if (!res)
            return -ENODEV;
        
        // 映射IO内存
        regs = ioremap(res->start, resource_size(res));
        if (!regs)
            return -ENOMEM;
        
        // 使用专用函数访问
        u32 status = readl(regs + REG_STATUS);
        if (status & STATUS_READY) {
            writel(CMD_START, regs + REG_COMMAND);
        }
        
        // 完成后取消映射
        iounmap(regs);
        
        return 0;
    }
    ```

    关键点总结

    * 不是空修饰符：有实际的类型检查作用

    * 强制使用正确的访问函数：防止直接解引用

    * 启用静态分析：Sparse工具可以检测错误使用

    * 架构相关优化：某些架构可能有特殊处理

    * 重要的文档价值：代码自说明性

    所以，__iomem 是 Linux 内核驱动开发中的重要机制，它确保了 IO 内存访问的安全性和正确性。

* dma 在 cpu 中，还是在 device 中？

    这是一个很好的问题，它触及了计算机系统的核心架构。简单来说：

    DMA控制器通常既不完全在“纯粹的CPU”中，也不完全在“纯粹的设备”中，它是一种位于两者之间的、共享的硬件模块。

    更详细的解释可以分为几个层面：

    1. 传统/经典架构（分立式）

        在早期的计算机系统中，DMA控制器是一个独立的芯片，安装在主板上。CPU通过系统总线对它进行编程（告诉它要传输什么数据，从哪里来，到哪里去），然后DMA控制器就会在设备和内存之间直接搬运数据，搬运完成后通知CPU。

        * 位置：在主板上，作为系统芯片组（北桥/南桥）的一部分。

        * 关系：它既是CPU的一个“外设”（CPU配置它），又是其他设备（如硬盘、网卡）的“代理”（替它们搬数据）。

    2. 现代集成架构

        在现代的SoC（片上系统）设计中，例如我们手机、平板电脑里的处理器：

        * DMA控制器被集成到SoC内部，但它仍然是一个独立于CPU核心的功能模块。它可能作为系统总线互连（如ARM的AMBA总线）的一个标准组件存在。

        * 许多复杂的外设（如GPU、高速网卡、SSD控制器）内部也集成了自己的专用DMA引擎。这些引擎可以看作是“在设备中”，但它们的工作机制和目的是相同的：减轻CPU的负担，实现高速数据传输。

    3. 如何工作？（关键原理）

        无论DMA控制器在哪里，其工作流程都体现了它的“中间人”角色：

        * 初始化（CPU负责）：CPU设置DMA控制器，告知源地址（设备缓冲区或内存）、目标地址（内存或设备缓冲区）、数据量等。

        * 启动传输（CPU下令）：CPU给设备和DMA控制器发送启动命令。

        * 数据搬运（DMA控制器负责）：DMA控制器接管系统总线的控制权，在设备和内存之间直接传输数据。在此期间，CPU可以继续执行其他任务（只要不访问被占用的总线）。

        * 完成通知（DMA控制器负责）：传输完成后，DMA控制器向CPU发送一个中断信号，告知CPU数据传输完毕。

    为什么这样设计？

    * 效率：让CPU这个“总经理”去干“搬运工”的活（通过循环读写I/O端口来移动每个字节）是极大的浪费。DMA作为一个“专职搬运工”，解放了CPU。

    * 并行性：CPU计算和DMA传输可以同时进行，提高了系统整体吞吐量。

    * 降低延迟：对于高速设备，只有DMA才能跟上其数据流的速度。

    总结与比喻

    | 角色 | 类比 | 在系统中的位置 |
    | - | - | - |
    | CPU | 总经理（负责核心计算和调度） | 中央处理器核心 |
    | DMA控制器 | 专职快递/物流部门（负责大批量物资搬运） | 在SoC/芯片组内部，作为一个共享的、独立的功能模块 |
    | 带DMA的设备 | 有自己专用车队的仓库（如高端网卡） | 设备内部（拥有自己的DMA引擎） |

    所以，最准确的回答是：

    DMA是一种能力/机制，其控制器硬件通常位于系统芯片组或SoC内部，作为连接CPU、内存和I/O设备的共享资源。同时，许多高性能设备自身也集成了DMA引擎以进一步提升性能。它本质上是CPU的一个高效协处理器，专门负责数据搬运任务。

* dma_map_page()

    原型

    ```c
    dma_addr_t dma_map_page(struct device *dev, 
                            struct page *page,
                            size_t offset,
                            size_t size,
                            enum dma_data_direction dir);
    ```

    参数说明：

    * dev：执行DMA操作的设备指针

    * page：要映射的物理页指针

    * offset：页内偏移量（字节）

    * size：映射区域大小（字节）

    * dir：数据传输方向

        * DMA_TO_DEVICE：数据从内存到设备

        * DMA_FROM_DEVICE：数据从设备到内存

        * DMA_BIDIRECTIONAL：双向传输

    返回值：

    * 成功：DMA总线地址（物理地址），设备可以直接访问

    * 失败：DMA_MAPPING_ERROR

    作用

    * 地址转换：将CPU可见的物理地址转换为设备可访问的DMA地址

    * 缓存一致性：确保CPU和设备的缓存一致性（对于coherent DMA架构）

    * 内存管理：建立页面的DMA映射，使外设可以直接访问内存

    典型用法示例

    ```c
    // 1. 分配内存页面
    struct page *page = alloc_page(GFP_KERNEL | GFP_DMA);
    if (!page) {
        return -ENOMEM;
    }

    // 2. 建立DMA映射
    dma_addr_t dma_handle;
    dma_handle = dma_map_page(&pdev->dev, page, 0, PAGE_SIZE, DMA_TO_DEVICE);
    if (dma_mapping_error(&pdev->dev, dma_handle)) {
        free_page(page);
        return -ENOMEM;
    }

    // 3. 将DMA地址传递给设备
    write_device_register(dev->reg_base, dma_handle);

    // 4. 启动DMA传输
    start_dma_transfer(dev);

    // 5. 等待传输完成（可选）
    wait_for_completion(&dev->done);

    // 6. 取消DMA映射
    dma_unmap_page(&pdev->dev, dma_handle, PAGE_SIZE, DMA_TO_DEVICE);

    // 7. 释放内存页面
    __free_page(page);
    ```

    使用注意事项

    * 配对使用：必须与 dma_unmap_page() 配对使用

    * 映射检查：使用 dma_mapping_error() 检查映射是否成功

    * 方向选择：正确选择DMA方向，影响缓存刷新操作

    * 大小对齐：建议使用DMA对齐的内存分配

    * 同步时机：映射后、启动DMA前可能需要缓存刷新

    相关函数

    ```c
    // 建立映射
    dma_map_single()      // 映射虚拟地址
    dma_map_sg()          // 映射散列表

    // 取消映射
    dma_unmap_page()
    dma_unmap_single()
    dma_unmap_sg()

    // 同步操作
    dma_sync_single_for_device()
    dma_sync_single_for_cpu()
    ```

    典型应用场景

    * 网络设备的数据包传输

    * 存储设备的块I/O操作

    * 多媒体设备的视频/音频数据传输

    * 任何需要外设直接访问内存的高性能数据传输

    该函数是Linux DMA API的核心部分，正确使用可确保内存与外设之间的高效、安全数据传输。

* dma_sync_single_for_device(), dma_sync_single_for_cpu()

    1. 函数原型

        ```c
        // 同步DMA缓冲区给设备使用
        void dma_sync_single_for_device(struct device *dev, 
                                       dma_addr_t addr,
                                       size_t size,
                                       enum dma_data_direction dir);

        // 同步DMA缓冲区给CPU使用
        void dma_sync_single_for_cpu(struct device *dev,
                                    dma_addr_t addr,
                                    size_t size,
                                    enum dma_data_direction dir);
        ```

    2. 作用

        dma_sync_single_for_device()

        * 目的：确保设备能看到CPU对DMA缓冲区的最新修改

        * 实际行为：

            * 将CPU的cache刷写到内存（如果CPU修改了数据）

            * 让设备访问的内存区域保持一致

            * 使设备能读取到CPU写入的数据

        dma_sync_single_for_cpu()

        * 目的：确保CPU能看到设备对DMA缓冲区的最新修改

        * 实际行为：

            * 使CPU的cache失效（如果设备修改了数据）

            * 确保CPU读取的是内存中的最新数据

            * 使CPU能读取到设备写入的数据

    3. 用法

        典型使用场景

        ```c
        #include <linux/dma-mapping.h>

        struct device *dev;      // 设备指针
        dma_addr_t dma_handle;   // DMA缓冲区总线地址
        void *cpu_addr;          // CPU访问的虚拟地址
        size_t size = 4096;      // 缓冲区大小

        // 场景1: CPU准备数据，设备读取
        void prepare_data_for_device(void)
        {
            // 1. CPU写入数据到缓冲区
            memset(cpu_addr, 0xAA, size);
            
            // 2. 同步给设备（确保设备能看到CPU的写入）
            dma_sync_single_for_device(dev, dma_handle, size, DMA_TO_DEVICE);
            
            // 3. 启动设备DMA读取操作
            start_device_dma_read(dma_handle);
        }

        // 场景2: 设备写入数据，CPU读取
        void read_data_from_device(void)
        {
            // 1. 启动设备DMA写入操作
            start_device_dma_write(dma_handle);
            
            // 2. 等待设备完成（中断或轮询）
            wait_for_device_completion();
            
            // 3. 同步给CPU（使CPU能看到设备的写入）
            dma_sync_single_for_cpu(dev, dma_handle, size, DMA_FROM_DEVICE);
            
            // 4. CPU读取设备写入的数据
            process_data(cpu_addr);
        }

        // 场景3: 双向传输（DMA_BIDIRECTIONAL）
        void bidirectional_transfer(void)
        {
            // 设备可能读写，CPU也可能读写
            dma_sync_single_for_device(dev, dma_handle, size, DMA_BIDIRECTIONAL);
            // ... 操作 ...
            dma_sync_single_for_cpu(dev, dma_handle, size, DMA_BIDIRECTIONAL);
        }
        ```

        数据方向参数

        ```c
        enum dma_data_direction {
            DMA_BIDIRECTIONAL = 0,    // 双向
            DMA_TO_DEVICE = 1,        // CPU → 设备
            DMA_FROM_DEVICE = 2,      // 设备 → CPU
            DMA_NONE = 3,             // 调试用
        };
        ```

    4. 重要注意事项

        * 对称调用：

            * 如果使用了 for_device()，之后必须使用对应的 for_cpu()

            * 方向参数必须一致

        * 性能影响：

            * 这些函数涉及cache操作，开销较大

            * 应尽量减少调用次数

        * 替代方案：

            ```c
            // 使用流式DMA映射（一次性的，通常性能更好）
            dma_map_single(dev, cpu_addr, size, dir);
            // ... DMA操作 ...
            dma_unmap_single(dev, dma_handle, size, dir);
            ```

        * 平台相关性：

            * 在x86等强一致架构上可能是空操作

            * 在ARM、PowerPC等弱一致架构上是必要的

    5. 典型驱动代码结构

        ```c
        static int my_driver_probe(struct device *dev)
        {
            // 分配DMA缓冲区
            cpu_addr = dma_alloc_coherent(dev, size, &dma_handle, GFP_KERNEL);
            
            // 或者使用流式映射
            // dma_handle = dma_map_single(dev, cpu_addr, size, direction);
        }

        static void my_driver_io(struct device *dev)
        {
            // 准备发送数据
            fill_transmit_buffer(cpu_addr);
            dma_sync_single_for_device(dev, dma_handle, size, DMA_TO_DEVICE);
            
            // 启动设备传输
            program_device_dma(dma_handle);
            
            // 接收数据时
            dma_sync_single_for_cpu(dev, dma_handle, size, DMA_FROM_DEVICE);
            process_received_data(cpu_addr);
        }
        ```

    这些函数是Linux内核DMA API的核心部分，确保在非一致性内存架构上，CPU和设备能看到彼此对共享内存的最新修改。

* qemu edu device 使用中断计算 facorial

    1. enable 中断

        `iowrite32(0x80, base_addr_bar0 + 0x20);`

    2. 读取中断寄存器的状态，每个中断源对应一个 bit，根据 bit 位判断中断源

        `int irq_reg_stat_val = ioread32(base_addr_bar0 + 0x24);`

    3. 读取计算出来的阶乘值

        `int fac_val = ioread32(base_addr_bar0 + 0x08);`

    4. 配置确认中断寄存器，告诉 device 此中断已被处理，防止 device 后续循环发送中断

        `iowrite32(irq_reg_stat_val, base_addr_bar0 + 0x64);`

    完整的 irq handler 如下：

    ```cpp
    irqreturn_t irq_handler(int irq, void *dev_id) {
        pr_info("in irq_handler()...\n");
        if (dev_id != hlc_pci_dev) {
            pr_warn("dev_id != hlc_pci_dev\n");
            return IRQ_NONE;
        }

        int irq_reg_stat_val = ioread32(base_addr_bar0 + 0x24);
        pr_info("reg status: %x\n", irq_reg_stat_val);  // 1

        int fac_val = ioread32(base_addr_bar0 + 0x08);
        pr_info("fac: %d\n", fac_val);

        iowrite32(irq_reg_stat_val, base_addr_bar0 + 0x64);

        pr_info("end of irq handler\n");
        return IRQ_HANDLED;
    }
    ```
    
    output:

    ```
    [ 5085.263385] in irq_handler()...
    [ 5085.263451] reg status: 1
    [ 5085.263471] fac: 120
    [ 5085.263504] end of irq handler
    ```

* qemu edu 使用 polling 方式计算阶乘

    在 bar0 偏移`0x20`处的寄存器的值`val`，如果`val & 0x01 != 0`，即`val`的最低位为 1，那么说明设备在忙计算。如果最低位为 0，那么说明设备计算结束。

    ```c
    // 0x08 (RW)factorial computation
    iowrite32(0x00, base_addr_bar0 + 0x20);
    iowrite32(5, base_addr_bar0 + 0x08);
    int cmp_cnt = 0;
    while (true) {
        val = ioread32(base_addr_bar0 + 0x20);
        if (val & 0x01) {
            pr_info("computing... %d\n", cmp_cnt++);
            continue;
        }
        break;
    }
    val = ioread32(base_addr_bar0 + 0x08);
    pr_info("fac: %d\n", val);
    ```

    output:

    ```
    [  364.548410] computing... 0
    [  364.548427] computing... 1
    [  364.548444] computing... 2
    [  364.548460] computing... 3
    [  364.548476] computing... 4
    [  364.548507] fac: 120
    ```

    注：

    * `iowrite32(0x00, base_addr_bar0 + 0x20);`

        禁用中断。

        如果一开始没有禁用中断，代码也没有正常处理中断，那么 device 会不停上报中断，即使卸载驱动后再重新加载驱动，device 也不会复位，依旧重复上报中断。只有重启 qemu 才能停止。

* `read`, `read_iter`, `splice_read`

    它们都是 VFS（虚拟文件系统）层定义的文件操作函数指针，由具体的文件系统（如 ext4）或设备驱动来实现，用于从文件或设备中读取数据。

    1. read (传统读取)

        作用：这是最经典、最通用的文件读取方法。它的任务是将数据从内核空间的文件（或设备）缓冲区拷贝到用户空间提供的缓冲区中。

        数据流：文件/设备 -> 内核缓冲区 -> 用户缓冲区

        特点：

            涉及两次数据拷贝：一次从文件到内核页缓存（如果使用了缓存），另一次从内核页缓存到用户缓冲区。

            通用性强，几乎所有文件类型都支持。

            对于大量数据的读写，这种拷贝会成为性能瓶颈。

        简单来说：read 是标准的“读取-拷贝”操作，稳定但效率不是最高。

    2. read_iter (迭代读取)

        作用：read 操作的进化版。它使用 struct iov_iter 来描述一个分散/聚集的缓冲区列表，可以一次性将数据读取到多个非连续的用户空间缓冲区，或者从一个非连续的内核缓冲区中读取数据。

        数据流：与 read 类似，也是拷贝，但源和目的可以是多个不连续的块。

        特点：

            支持向量化I/O（如 readv 系统调用），更灵活。

            仍然是数据拷贝的方式。

            在现代内核中，read 系统调用在底层通常会被转换成 read_iter 操作。可以认为 read_iter 是新的、更通用的读取接口。

        简单来说：read_iter 是更现代、更灵活的“读取-拷贝”操作，是 read 的替代者。

    3. splice_read (零拷贝读取)

        作用：实现“零拷贝”读取的高级操作。它的目标是将数据从一个文件描述符（比如一个真实文件）直接移动到到一个管道 中，而不需要经过用户空间。数据始终在内核空间内流动。

        数据流：文件/设备 -> 管道缓冲区 -> 另一个文件描述符（如socket）

        特点：

            零拷贝：避免了数据在内核和用户空间之间的昂贵拷贝，极大提升了性能。

            专用场景：通常用于将数据从文件“输送”到网络套接字（例如，一个文件下载服务器）。

            依赖管道：必须使用管道作为数据的中间载体。

            不是所有文件系统都支持。

        简单来说：splice_read 是高效的“数据搬运工”，它在内核里直接把数据从一个地方“倒”到另一个地方，省去了拷贝到用户层的步骤。

    总结与类比
    操作	数据流向	关键特点	适用场景
    read	内核 -> 用户空间	传统，两次数据拷贝	通用文件读取
    read_iter	内核 -> 用户空间	现代，支持分散/聚集IO	高效、灵活的文件读取（readv）
    splice_read	文件 -> 管道	零拷贝，高性能	高性能网络服务（如文件发送）

* `remap_pfn_range()`

    主要用于将物理内存映射到用户空间。

    将内核物理页帧（physical page frames）直接映射到用户进程的虚拟地址空间，使得用户态程序能够直接访问这些物理内存。

    syntax:

    ```c
    int remap_pfn_range(struct vm_area_struct *vma, unsigned long addr, 
                       unsigned long pfn, unsigned long size, pgprot_t prot);
    ```

    参数说明

    * vma：目标虚拟内存区域结构

    * addr：用户空间起始虚拟地址

    * pfn：物理页帧号（物理地址 >> PAGE_SHIFT）

    * size：映射区域大小

    * prot：页面保护标志

    典型应用场景

        设备驱动开发

            映射设备内存（如显存、DMA缓冲区）到用户空间

            实现零拷贝数据传输

        内核模块

            为特定应用提供直接硬件访问能力

            实现高性能内存共享机制

        嵌入式系统

            直接访问特定物理地址范围的硬件寄存器

    注意事项

        需要仔细管理内存安全和权限

        通常与 mmap() 系统调用配合使用

        必须确保映射的物理内存是合法且安全的

* `dma_sync_single_for_cpu()`

    在 DMA 传输完成后，让 CPU 能够安全地访问已被 DMA 设备写入数据的内存缓冲区，或者确保 CPU 对缓冲区的修改能被下一次 DMA 传输正确读取。

    在 CPU 和 DMA 设备之间同步一块共享内存的数据， 为了 CPU 即将进行的访问做准备。

    syntax:

    ```c
    #include <linux/dma-mapping.h>

    void dma_sync_single_for_cpu(
        struct device *dev,
        dma_addr_t addr,
        size_t size,
        enum dma_data_direction dir
    );
    ```

    * `dev`: 指向设备结构体的指针。这个指针通常在你初始化设备时获得（例如通过 &pdev->dev 获取平台设备指针）。它包含了 DMA 掩码等设备相关的 DMA 属性。

    * `addr`: DMA 总线的地址。这个地址不是内核的虚拟地址，而是之前通过 dma_map_single() 或类似映射函数返回的地址。它代表了设备能够直接访问的内存位置。

    * `size`: 需要同步的内存区域的大小（以字节为单位）。这个大小应该与你之前映射时使用的大小一致。

    * `dir`: DMA 数据传输的方向。这个方向必须与之前调用 dma_map_single() 时使用的方向一致。它有以下几种可能的值：

        * `DMA_BIDIRECTIONAL`

        * `DMA_TO_DEVICE`

        * `DMA_FROM_DEVICE`

    CPU 和 DMA 设备之间的 缓存一致性 问题:

    1. CPU 有缓存：CPU 访问内存时，会把数据缓存在自己的高速缓存（Cache）里，以加速后续访问。这意味着，CPU 看到的数据可能不是内存中的最新数据。

    2. DMA 直接访问内存：DMA 设备不经过 CPU，直接与内存交互。它不知道 CPU 的缓存里有什么。

    3. 数据不一致的风险：

        场景一（DMA 设备 -> CPU）：DMA 设备将新数据直接写入内存。但此时 CPU 缓存里的可能还是旧数据。如果 CPU 不去读取内存而直接读缓存，就会得到错误（过时）的数据。

        场景二（CPU -> DMA 设备）：CPU 准备了一些数据，打算让 DMA 设备来读取。CPU 可能只把数据写到了自己的缓存里，还没来得及写回内存。如果此时 DMA 设备直接从内存读取，得到的就是不完整/错误的数据。

    `dma_sync_single_for_cpu()` 就是为了解决上述 场景一 的问题，确保在 DMA 输入操作（设备到内存）后，CPU 能读到最新数据。同时，它也会为 场景二 的下一次传输做准备。

    这个函数会根据 DMA 的方向（由之前建立映射时指定）来执行不同的操作：

    * 如果方向是 DMA_FROM_DEVICE（数据从设备到内存）：

        作用：使 CPU 的缓存失效。

        解释：这相当于告诉 CPU：“你缓存里的这块内存数据已经过期了，下次读的时候必须去内存里拿最新的。” 这样 CPU 就能读到 DMA 设备刚刚传输过来的新数据。

    * 如果方向是 DMA_TO_DEVICE（数据从内存到设备）或 DMA_BIDIRECTIONAL（双向）：

        作用：通常在这个时机不做太多事，但它隐含着为下一次 DMA_TO_DEVICE 做准备。其主要同步工作是在它的“兄弟函数” dma_sync_single_for_device 中完成的（该函数在启动 DMA 前调用，用于刷写 CPU 缓存到内存）。

    一个典型的执行流程

    1. 分配缓冲区：驱动程序分配一块内存缓冲区。

    2. 映射 DMA：调用 dma_map_single()，获取一个设备可以理解的总线地址。（此时可能会刷写 CPU 缓存，确保设备能读到正确数据）

    3. 启动 DMA（设备读）：将总线地址告诉设备，并启动 DMA 传输（例如，从设备读取数据到内存）。

    4. DMA 传输中：设备独立工作，将数据写入内存。CPU 可以处理其他任务。

    5. DMA 传输完成：设备发出中断，通知 CPU 传输完成。

    6. 同步给 CPU：驱动程序中调用 dma_sync_single_for_cpu(...， DMA_FROM_DEVICE)。这会使得 CPU 缓存中对应这块缓冲区的部分失效。

    7. CPU 访问数据：现在，驱动程序可以安全地读取缓冲区，CPU 会从主内存中获取 DMA 设备写入的最新数据，而不是过时的缓存数据。

    8. 解除映射：当不再需要 DMA 访问时，调用 dma_unmap_single() 解除映射。

* Suspend（挂起）函数

    在 suspend 函数中，驱动通常需要做以下几件事：

    1. 保存设备状态：将设备当前关键的运行时状态（例如寄存器配置）保存起来，以便在 resume 时能恢复原样。

    2. 降低设备功耗：

        * 关闭设备时钟。

        * 将设备置于其硬件支持的睡眠模式。

        * 切断设备不需要的电源域（如果软件可控）。

    3. 处理未完成的操作：确保所有挂起的I/O操作被妥善处理，避免数据丢失。

    4. 释放共享资源：可能会释放一些系统级的共享资源，如中断、DMA通道等，以便整个系统能更彻底地休眠。

* Resume（恢复）函数

    在 resume 函数中，驱动通常需要做以下几件事：

    1. 恢复电源和时钟：重新开启设备的时钟和电源。

    2. 重新初始化设备：

        * 将从 suspend 中保存的设备状态重新写入设备的寄存器。

        * 执行一系列初始化序列，让设备回到它“睡着”之前的工作状态。

    3. 重新申请资源：重新申请中断、DMA通道等系统资源。

    4. 恢复正常操作：告知设备可以开始正常处理数据了。

* suspend 和 resume 的 example

    在现代 Linux 内核中，suspend 和 resume 函数通常被赋值给一个 struct dev_pm_ops 结构体的成员，或者更简单地，通过 DEFINE_SIMPLE_DEV_PM_OPS 宏来定义。

    ```c
    #include <linux/pm.h>

    static int my_driver_suspend(struct device *dev)
    {
        struct my_device_data *data = dev_get_drvdata(dev);

        /* 1. 保存设备状态到 data->saved_reg */
        /* 2. 将设备设置为睡眠模式 */
        /* 3. 关闭设备时钟 */
        return 0;
    }

    static int my_driver_resume(struct device *dev)
    {
        struct my_device_data *data = dev_get_drvdata(dev);

        /* 1. 打开设备时钟 */
        /* 2. 将设备从睡眠模式唤醒 */
        /* 3. 从 data->saved_reg 恢复设备状态 */
        return 0;
    }

    /* 定义电源操作结构 */
    static const struct dev_pm_ops my_driver_pm_ops = {
        .suspend = my_driver_suspend,
        .resume = my_driver_resume,
        /* 可能还有 .freeze, .thaw, .poweroff 等更细粒度的回调 */
    };

    /* 在驱动注册时，将这个 pm_ops 关联到设备 */
    static struct platform_driver my_driver = {
        .driver = {
            .name = "my_device",
            .pm = &my_driver_pm_ops, // 这里是关键
        },
        .probe = my_probe,
        .remove = my_remove,
    };
    ```

* S3（Standby/挂起到内存）和S4（Hibernate/挂起到硬盘）

    struct dev_pm_ops 的结构体中，主要包含以下几对函数:

    1. `.suspend` / `.resume`

        主要用于： 运行时电源管理（Runtime PM）和（在较老的内核中）系统睡眠的一部分。但在现代内核中，对于系统睡眠，它们通常被更具体的回调所取代。

    1. `.freeze` / `.thaw`

        主要用于： 为休眠（Hibernate）做准备。

        作用： 在创建休眠镜像之前，.freeze 会被调用来让设备进入一个安静、静止的状态，但它不一定要让设备进入低功耗模式。重点是停止所有I/O操作，并让设备处于一个已知的、稳定的状态，以便内核能安全地创建一个完整的内存镜像并保存到硬盘（S4状态）。

    1. `.poweroff` / `.restore` (或 `.suspend_noirq` / `.resume_noirq` 的特定阶段)

        主要用于： S3睡眠（挂起到内存） 和 S4睡眠（休眠）的最后阶段。

        作用： 这是真正让设备进入低功耗状态的函数。对于S3，它会切断设备电源，只保留内存供电；对于S4，在内存镜像保存完毕后，它会最终关闭设备电源。

    1. `.suspend_late` / `.resume_early`

        这些是在进程和中断被禁用/启用前后执行的，用于执行一些非常关键、需要原子上下文的操作。

* S3睡眠（挂起到内存）流程

    1. 冻结用户空间和进程。

    2. 调用驱动的`.suspend`或`.freeze`回调（让设备安静下来）。

    3. 执行系统核心的 suspend_late 操作。

    4. 禁用设备中断。

    5. 调用驱动的 .suspend_noirq 或 .poweroff 回调（这是最关键的一步，在这里驱动会真正切断设备时钟/电源，将其置于S3要求的低功耗状态）。

    6. 系统进入S3状态，仅内存保持刷新。

    恢复时，顺序相反。

* S4睡眠（休眠到硬盘）流程

    1. 冻结用户空间和进程。

    2. 调用驱动的`.freeze`回调（让设备进入静止状态，以便创建内存镜像）。

    3. 内核将整个内存内容写入硬盘的交换区（创建休眠镜像）。

    4. 调用驱动的`.poweroff`回调（既然内存镜像已保存，现在可以安全地关闭设备电源了）。

    5. 系统完全断电（S4状态）。

    恢复时，这是一个“重启”过程：BIOS/UEFI启动 -> 内核从休眠镜像重新加载 -> 调用驱动的`.restore`回调重新初始化设备 -> 解冻进程和用户空间。

* `devm_platform_ioremap_resource()`

    将平台设备（platform device）的指定内存资源映射到内核虚拟地址空间，并自动进行资源管理和错误处理。

    syntax:

    ```c
    void __iomem *devm_platform_ioremap_resource(struct platform_device *pdev, unsigned int index);
    ```

    example:

    ```c
    struct resource *res;
    void __iomem *base;

    // 传统方式需要手动管理
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    base = devm_ioremap_resource(&pdev->dev, res);

    // 使用devm_platform_ioremap_resource简化
    base = devm_platform_ioremap_resource(pdev, 0);
    ```

* MMIO 的缓存问题

    在 MMIO 区域，刚写入就读取不会读到未写入的值，因为在映射 MMIO 区域时，通常会标记为非缓存（Uncached）。另外编译器在处理 iowrite() / ioread() 相关的函数时，会不进行优化，防止乱序执行。

* qemu edu 写入状态寄存器`0x20`

    ```c
    int edu_open(struct inode *, struct file *) {
        pr_info("in edu_open()...\n");
        pr_info("base_addr_bar0: %p\n", base_addr_bar0);
        iowrite32(0x80, base_addr_bar0 + 0x20);
        uint32_t val = ioread32(base_addr_bar0 + 0x20);
        pr_info("val: %d\n", val);
        return 0;
    }
    ```

    output:

    ```
    [ 1275.443425] in edu_open()...
    [ 1275.443437] base_addr_bar0: 00000000a6ddd184
    [ 1275.443538] val: 128
    [ 1275.443577] in edu_read()...
    [ 1275.443614] in edu_release()...
    ```

* 如果只有 cdev，没有 device 设备文件节点，不可以调用 cdev 绑定的 fops 驱动

* `pci_alloc_dev()`

    分配并初始化一个 PCI 设备结构体。

    这个函数是内核内部使用的函数，不是给驱动开发者调用的。

* `pci_find_bus()`

    根据总线号查找并返回指定的 PCI 总线结构。

    syntax:

    ```c
    #include <linux/pci.h>

    struct pci_bus *pci_find_bus(int bus);
    ```

    在系统的 PCI 总线列表中搜索具有特定总线号的 PCI 总线, 接收一个总线号作为参数（通常为 0-255 范围内的整数）

    返回值:

    * 如果找到对应的总线，返回指向 struct pci_bus 的指针

    * 如果未找到，返回 NULL 指针

* `asm("int $0x3B");`

    触发一个软中断.

    作用解析

    * `int`：x86 架构的软中断指令（Interrupt）

    * `0x3B`：中断向量号（59号中断）

* 设备树源文件（.dts）

    设备树源文件（.dts）是一种类似 JSON 的层次化结构，主要包含以下元素：

    1. 节点：代表一个设备或一个总线。树状结构的根是 /。

    2. 属性：附着在节点上的键值对，用于描述设备的特性。

        * `compatible`：最重要的属性。它是一个字符串列表，内核通过它来匹配最适合的设备驱动程序。例如：`compatible = "ti,omap3-uart"`; 表示这个设备与德州仪器的 OMAP3 串口驱动兼容。

        * `reg`：定义设备在父总线地址空间内的内存映射地址和长度。

        * `interrupts`：定义设备使用的中断号。

        * `status`：指示设备状态，如 `"okay"`（启用）或 `"disabled"`（禁用）。

        * `model` & `compatible：描述板卡和` SoC 的型号。

    工作流程简述

    1. 编写：硬件工程师或驱动开发者根据实际硬件编写设备树源文件（.dts）。

    2. 编译：使用设备树编译器（dtc）将人类可读的 .dts 文件编译成机器可读的二进制格式——设备树 blob（.dtb 文件）。

    3. 传递：Bootloader 在启动内核时，将 .dtb 文件在内存中的地址传递给内核。

    4. 解析：内核启动初期，解析 .dtb 文件，根据 compatible 属性匹配驱动程序，并根据 reg、interrupts 等属性初始化并注册平台设备。

    example:

    `dts`:

    ```dts
    // 这是一个极其简化的示例
    /dts-v1/;

    / {
        model = "My Awesome Board";
        compatible = "my-company,my-board"; // 板卡兼容性

        cpus {
            cpu@0 {
                compatible = "arm,cortex-a53";
            };
        };

        memory@80000000 { // 内存节点
            device_type = "memory";
            reg = <0x80000000 0x20000000>; // 起始地址 0x80000000，大小 512MB
        };

        uart0: serial@ff000000 { // 串口设备，别名为 uart0
            compatible = "ns16550a"; // 匹配标准的 16550 串口驱动
            reg = <0xff000000 0x1000>; // 寄存器地址范围
            interrupts = <10>; // 使用中断号 10
            status = "okay";
        };
    };
    ```

* 设备树

    设备树 是一个描述硬件配置的数据结构，它以一种特定的格式（通常是一个.dts文件）来详细说明一个计算机系统里有哪些设备，它们的类型、地址、中断号以及各种可配置参数。

    设备树在系统启动时由 Bootloader（如 U-Boot）传递给操作系统内核（如 Linux）。内核通过解读这张“地图”，就知道自己运行在什么样的硬件之上，从而动态地加载正确的驱动程序并初始化对应的设备。

    在 x86 架构中，BIOS/UEFI 提供标准接口来枚举设备，内核可以询问“这里有什么硬件？”（即探测）。

    在 ARM 架构中，情况非常复杂，SoC 种类繁多，板卡设计多样。如果为每一种“SoC + 板卡”的组合都编译一个特定的内核，那将是一场维护噩梦。设备树就是为了解决这个问题而诞生的。

    设备树的核心作用是解耦，将硬件描述和操作系统内核分离开来，从而可以实现一个内核，多种硬件。

* 实现了 ioctl 读取与写入数据，见`ref_12`

    output:

    ```
    successfully write data by ioctl
    read value: 123
    ```

* kernel 中的内存管理感觉是个问题

    假如希望用户可以无限次乱序 read, write，并且遵循 fifo 的原则，那么可以把 buffer 设计成一个链表，每次调用 read 的时候减少一个节点，调用 write 的时候增加一个节点。

    如果在 read 的时候遇到链表为空，那么就输出 there is nothing to copy。

* AXI DMA MMIO

    MMIO 允许 CPU 通过读写特定内存地址来配置和控制 DMA 控制器

* 写了内核驱动的代码和用户态代码，成功从用户态向内核写入数据，并从内核读取数据。

    见`ref_11`。

    * user mode 的程序需要使用`sudo ./main`执行，不然没有权限打开 device 文件

    * kernel mode 的`h_write()`的返回值就是 user mode 的`write()`的返回值

        `read()`同理。在写 kernel code 的时候，按照约定俗成，返回写入/读取了多少个字节。

    * 如果使用`fopen()`，`fread()`，`fwrite()`等函数打开文件，那么`dmesg`中会报错。

    * `copy_to_user`, `copy_from_user`返回的是剩余的字节数，与`read()`，`write()`正好相反，需要注意。

* `sysfs_remove_file()`

    从 sysfs 文件系统中删除先前创建的文件。

    syntax:

    ```c
    void sysfs_remove_file(struct kobject *kobj, 
                          const struct attribute *attr);
    ```

    * 从指定的 kobject 对应的 sysfs 目录中删除属性文件

    * 清理相关的内核数据结构

    * 确保文件系统的一致性

    example:

    ```c
    // 示例：创建和删除 sysfs 文件
    static struct kobject *example_kobj;
    static struct attribute attr = {
        .name = "example_file",
        .mode = 0644,
    };

    // 创建设备时添加
    sysfs_create_file(example_kobj, &attr);

    // 清理设备时删除
    sysfs_remove_file(example_kobj, &attr);
    ```

    如果我们不手动调用这个函数，在 rmmod 时，操作系统不会帮我们清理。

    如果想让操作系统帮忙清理，可以使用`devm_sysfs_create_group();`。如果我们写的是模块级别的 sysfs 接口（无关联设备），那么就没法用这个方法了。

* `kobject_put()`

    减少 kobject 的引用计数，并在计数降至零时释放相关资源。

    syntax:

    ```c
    #include <linux/kobject.h>

    void kobject_put(struct kobject *kobj);
    ```

    驱动开发者必须在合适的业务逻辑中调用 kobject_put()，以表明“我不再使用这个内核对象了”。

    * 探测失败时：在 probe 函数中，如果设备初始化到一半失败了，需要回滚操作，对之前已经成功 kobject_add 或 kobject_init 的对象调用 kobject_put() 进行清理。

    * 设备移除时：在 remove 函数或 disconnect 函数中，当设备被拔出或驱动被卸载时，需要释放为该设备创建的所有内核对象。

    * 引用生命周期结束时：当你使用 kobject_get() 增加了一个对象的引用后，在完成操作后必须用 kobject_put() 来平衡。

    example:

    ```c
    static int my_driver_probe(struct device *dev)
    {
        struct my_device *my_dev = kzalloc(sizeof(*my_dev), GFP_KERNEL);
        
        // 初始化kobject
        kobject_init(&my_dev->kobj, &my_ktype);
        
        // 添加到sysfs
        if (kobject_add(&my_dev->kobj, &dev->kobj, "my_device")) {
            // 如果添加失败，需要清理
            kobject_put(&my_dev->kobj); // <-- 驱动开发者调用
            return -ENOMEM;
        }
        
        return 0;
    }

    static void my_driver_remove(struct device *dev)
    {
        struct my_device *my_dev = dev_get_drvdata(dev);
        
        // 移除设备时释放kobject
        kobject_put(&my_dev->kobj); // <-- 驱动开发者调用
    }
    ```

    与 kobject_put() 对应的是 kobject_get()，用于增加引用计数。

    其他常用的关联函数：

    ```c
    // 增加引用计数
    struct kobject *kobject_get(struct kobject *kobj);

    // 初始化 kobject
    void kobject_init(struct kobject *kobj, struct kobj_type *ktype);

    // 添加 kobject 到系统
    int kobject_add(struct kobject *kobj, struct kobject *parent, const char *fmt, ...);

    // 初始化并添加（常用组合）
    int kobject_init_and_add(struct kobject *kobj, struct kobj_type *ktype,
                             struct kobject *parent, const char *fmt, ...);
    ```

* `__user`表示是一个用户空间的指针，所以 kernel 不可以直接使用。

    ```c
    #ifdef __CHECKER__
    # define __user __attribute__((noderef, address_space(1)))
    # define __kernel /* default address space */
    #else
    # define __user
    # define __kernel
    #endif
    ```

* `inode`是文件的节点结构，用来存储文件静态信息。文件创建时，内核中就会有一个 inode 结构

    `lsmod`除了可以列出当前已经加载的模块，还可以显示模块之间的依赖关系。

    应用程序 app 先找到设备文件，设备文件通过设备号找到设备驱动，然后再调用相关的函数。设备号如何找到设备驱动？首先可以通过设备号找到`cdev`结构体，然后从`cdev`结构体找到`file_operations`结构体，再在这个结构体里找对应的驱动函数。

    每个静态文件都对应内核中一个`inode` struct，存放一些基本信息。而当文件被打开时，内核会创建一个`file` struct，记录一些信息。

* cdev

    cdev 指的是 char device，是 linux 中用于处理 char device 相关操作（比如 open, release, read, write, ioctl 等）的概念与结构体。

    相关函数：

    * `cdev_init()`：将 cdev 与 fops 绑定在一起

        Syntax:

        ```c
        void cdev_init(struct cdev *cdev, struct file_operations *fops);
        ```

    * `cdev_add`：将 cdev 添加到内核，并为 cdev 绑定设备号

        Syntax:

        ```c
        int cdev_add(struct cdev *p, dev_t dev, unsigned count);
        ```

        params:

        * `p`: 要添加的 cdev 结构

        * `dev`：起始设备号

        * `count`：设备号个数

        返回 0 表示成功，非 0 表示失败。

    * `cdev_del`：将 cdev 从内核中移除

        syntax:

        ```c
        void cdev_del(struct cdev *p)
        ```

    example:

    `hello_world.c`:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/cdev.h>
    #include <linux/fs.h>

    int m_open(struct inode *inode, struct file *file_ptr)
    {
        printk("in m_open function ...\n");
        return 0;
    }

    int m_release(struct inode *inode, struct file *file_ptr)
    {
        printk("in m_release function ...\n");
        return 0;
    }

    long int m_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset)
    {
        printk("in m_read function ...\n");
        return 0;
    }

    long int m_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
    {
        printk("in m_write function ...\n");
        return 0;
    }

    long m_ioctl(struct file *, unsigned int, unsigned long)
    {
        printk("in m_ioctl function ...\n");
        return 0;
    }

    dev_t dev_num;
    struct cdev m_dev;
    struct file_operations m_ops = {
        .owner = THIS_MODULE,
        .open = m_open,
        .read = m_read,
        .write = m_write,
        .release = m_release,
        .unlocked_ioctl = m_ioctl,
    };

    int hello_init(void)
    {
        printk("Insert my test module.\n");

        // allocate a device number
        dev_num = MKDEV(220,0);
        register_chrdev_region(dev_num, 1, "hlc_dev");

        cdev_init(&m_dev, &m_ops);
        cdev_add(&m_dev, dev_num, 1);
        return 0;
    }

    void hello_exit(void)
    {
        cdev_del(&m_dev);
        unregister_chrdev_region(dev_num, 1);
        printk("Exit my test module.\n");
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    加载完模块后，`/dev/`下并不会出现 dev 设备文件节点。如果需要显示这个文件节点，需要使用`struct device`相关的概念去创建。设备驱动需要和设备文件配合使用。

* `cdev_alloc()`

    除了使用`cdev_init()`静态绑定，还可以使用`cdev_alloc()`动态申请 cdev:

    ```c
    struct cdev *my_cdev = cdev_alloc( );
    my_cdev->ops = &my_fops;
    ``````

* 整理一下开发环境的搭建，因为发现只需要安装`build-essential`就可以自动安装 header 文件，那么其实可以简化流程

* 可以在函数声明后就直接构造`struct file_operations`，然后再在其他地方对函数进行定义。

* linux module 编译不出来，可能是因为`obj-m`写成了`odj-m`

* `pci_read_config_dword()`

    pci_read_config_dword() 是一个用于读取 PCI/PCIe 设备配置空间中一个双字（32 位）数据的函数。

    PCI/PCIe 设备的配置空间是一个标准化的寄存器区域，包含设备ID、厂商ID、资源分配（BAR）、设备类别等重要信息。

    syntax:

    ```c
    #include <linux/pci.h>

    int pci_read_config_dword(struct pci_dev *pdev, int offset, u32 *value);
    ```

    * pdev：指向 PCI 设备结构体 pci_dev 的指针

    * offset：要读取的配置空间偏移地址（字节偏移）

    * value：输出参数，用于存储读取到的32位数据

    返回值：

    成功时返回 0, 失败时返回错误码

    * example 1:

        ```c
        #include <linux/pci.h>

        struct pci_dev *pdev;
        u32 config_value;
        int ret;

        // 读取设备ID和厂商ID（偏移0x00）
        ret = pci_read_config_dword(pdev, 0x00, &config_value);
        if (ret == 0) {
            u16 vendor_id = config_value & 0xFFFF;
            u16 device_id = (config_value >> 16) & 0xFFFF;
            printk("Vendor: 0x%04x, Device: 0x%04x\n", vendor_id, device_id);
        }

        // 读取BAR0（偏移0x10）
        pci_read_config_dword(pdev, PCI_BASE_ADDRESS_0, &config_value);
        ```

    * example 2:

        ```c
        // 高级方式：使用专用函数
        int msi_cap = pci_find_capability(pdev, PCI_CAP_ID_MSI);
        if (msi_cap) {
            // 然后用 pci_read_config_dword 读取具体寄存器
            pci_read_config_dword(pdev, msi_cap + PCI_MSI_ADDRESS_LO, &address_lo);
        }

        // 底层方式：手动遍历（不推荐）
        u8 pos = PCI_CAPABILITY_LIST;
        pci_read_config_byte(pdev, pos, &pos);
        while (pos) {
            pci_read_config_dword(pdev, pos, &header);
            if ((header & 0xFF) == PCI_CAP_ID_MSI) {
                break; // 找到 MSI Capability
            }
            pci_read_config_byte(pdev, pos + 1, &pos); // 下一个
        }
        ```

    函数家族：

    ```c
    // 读取字节（8位）
    int pci_read_config_byte(struct pci_dev *pdev, int offset, u8 *value);

    // 读取字（16位）  
    int pci_read_config_word(struct pci_dev *pdev, int offset, u16 *value);

    // 写入双字（32位）
    int pci_write_config_dword(struct pci_dev *pdev, int offset, u32 value);
    ```

* `pci_find_ext_capability()`

    pci_find_ext_capability() 是用于查找 PCIe 设备扩展能力（Extended Capability） 的函数。

    syntax:

    ```c
    #include <linux/pci.h>

    int pci_find_ext_capability(struct pci_dev *dev, int cap);
    ```

    常见的 PCIe 特有的高级功能：

    * 高级错误报告（AER） - PCI_EXT_CAP_ID_AER

    * 虚拟通道（VC） - PCI_EXT_CAP_ID_VC

    * 设备序列号 - PCI_EXT_CAP_ID_DSN

    * 电源预算 - PCI_EXT_CAP_ID_PWR

    * 多功能 - PCI_EXT_CAP_ID_MFVC

    与 pci_find_capability() 的区别:

    | 特性 | pci_find_capability() | pci_find_ext_capability() |
    | - | - | - |
    | 能力类型 | 传统 PCI 能力 | PCIe 扩展能力 |
    | 搜索范围 | 0x40-0xFF | 0x100-0xFFF |
    | 链表结构 | 单字节 Next 指针 | 双字节 Next 指针 |
    | 典型能力 | MSI、MSI-X、电源管理 | AER、VC、设备序列号 |

* 基址寄存器（BARs）

    当电脑开机时，BIOS或操作系统会遍历所有PCIe设备，读取每个BAR的需求。然后，系统会统一规划，为每个设备的BAR分配一个实际的、唯一的起始内存地址（即“基址”），并把这个地址写回BAR中。此后，当CPU需要访问该设备时，只需从BAR指定的基址开始，进行读写操作即可。

    BAR中存储的值，是 CPU 视角下的总线地址（在x86体系结构中也常称为物理地址）。

    分配 bar 的过程：

    1. 初始值（设备声明）：在系统分配之前，BAR中存放的并不是一个有效的地址，而是一个特殊的值，用于声明该设备所需地址空间的类型（是内存空间还是I/O空间）和大小。例如，设备会将某些位写为只读的1，系统软件通过读取并操作这些值，可以计算出设备需要多大的窗口。

    2. 最终值（系统分配）：在系统启动（或设备热插拔）时，BIOS或操作系统会执行以下操作：

        * 读取所有PCIe设备的BAR初始值，了解每个设备的需求。

        * 在全局物理地址空间中，找出一段段未被占用的、大小合适的空闲区域。

        * 将每个空闲区域的起始地址（即基址）写入到对应设备的BAR中。

* `ssize_t`是`long`

* 报错：`insmod: ERROR: could not insert module ./hello.ko: Invalid module format`

    主要是因为编译时候使用的内核版本和当前系统的内核版本不一致。

* 对于 cpu 的 dma，cpu 内部的内存控制器里有 dma 引擎，dma 引擎设定一个计数器，要求dram把指定范围内的数据发送到总线上，如果目的地是网卡，那么网卡需要配合着读总线上的数据，并发送ack使计数器加一，直到数据全部传输完成。对于gpu和rdma网卡，则是网卡内部有dma引擎，向pcie发起read/write事务，读取或写入gpu显存的数据，gpu内则有专门的硬件单元来响应这个事务

* `/proc/iomem`

    /proc/iomem 是 Linux 内核提供的一个只读的虚拟文件，它直观地展示了整个系统物理内存地址空间的映射和分配情况。

    它不仅仅显示普通的 RAM（系统内存）, 还显示了其他所有被映射到物理地址空间的设备，例如：

    * 显卡的显存 (Video RAM)

    * BIOS 的 ROM

    * PCI/PCIe 设备的内存映射寄存器 (MMIO) - 这是非常重要的一部分，操作系统通过读写这些地址来与硬件（如网卡、显卡、硬盘控制器）通信。

    * 系统保留的内存区域。

    每一行代表一个地址范围，格式通常为：起始地址-结束地址 : 资源描述

    /proc/iomem 描述的就是 CPU 的总线地址空间（或叫物理地址空间）的映射图。，总线上面可以挂多个设备，包含 dram, pcie 等，每个设备又分别占用总线地址空间的一小段范围

    linux kernel 中的 va 指的并不是这里的总线地址。

* 内存排序模型（Memory Ordering Models）

    * 顺序一致性（Sequential Consistency, SC）

        最强模型。要求所有线程看到的整个程序的执行顺序是一致的，且每个线程内部的操作顺序就是其程序顺序。几乎在所有指令之间都隐式地插入了全屏障。性能最差，但最容易理解。

    * 宽松内存排序（Relaxed Memory Ordering / Weak Ordering）

        性能最优。允许大量的重排序，除非程序员显式地使用内存屏障指令来约束顺序。

        ARM、PowerPC 等架构是弱内存模型。

    * 介于两者之间（如 x86/x64-64）

        TSO（Total Store Order）模型：这是一种相对较强的模型。

        它只允许一种重排序：后来的读操作可以越过先前的写操作。因此，x86 不需要单独的 LoadLoad 或 LoadStore 屏障，但需要 StoreStore 和（尤其是）StoreLoad 屏障（这是开销最大的一种屏障）来防止这种重排序。

* `request_irq()` example

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/pci.h>

    static struct pci_device_id pci_id_table[] = {
        { PCI_DEVICE(0x1234, 0x11e8) },
        {0,}
    };

    static void *base_addr_bar0;
    static struct pci_dev *hlc_pci_dev = NULL;

    irqreturn_t irq_handler(int irq, void *dev_id) {
        pr_info("in irq_handler()...\n");
        if (dev_id != hlc_pci_dev) {
            pr_warn("dev_id != hlc_pci_dev\n");
            return IRQ_NONE;
        }

        return IRQ_HANDLED;
    }

    static int edu_probe(struct pci_dev *pci_dev, const struct pci_device_id *id) {
        pr_info("in edu_probe()...\n");
        hlc_pci_dev = pci_dev;
        
        int ret = pci_enable_device(pci_dev);
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to pci enable device, ret: %d\n", ret);
            goto ERR_PCI_ENABLE_DEVICE;
        }

        // mmio
        ret = pci_request_region(pci_dev, 0, "qemu_edu_drv");
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to pci request region\n");
            goto ERR_PCI_REQUEST_REGION;
        }

        resource_size_t res_len_bar0 = pci_resource_len(pci_dev, 0);
        base_addr_bar0 = pci_iomap(pci_dev, 0, res_len_bar0);
        if (base_addr_bar0 == NULL) {
            dev_err(&pci_dev->dev, "fail to pci iomap\n");
            goto ERR_PCI_IOMAP;
        }

        // dma
        ret = dma_set_mask_and_coherent(&pci_dev->dev, DMA_BIT_MASK(28));
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to set dma mask and conherent\n");
            goto ERR_DMA_SET_MASK_AND_COHERENT;
        }

        // irq
        ret = request_irq(pci_dev->irq, irq_handler, IRQF_SHARED, "qemu_edu_dev_irq_handler", pci_dev);
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to request irq\n");
            goto ERR_REQUEST_IRQ;
        }

        pr_info("successfully probe edu pci dev\n");

        return 0;

    ERR_REQUEST_IRQ:
    ERR_DMA_SET_MASK_AND_COHERENT:
        pci_iounmap(pci_dev, base_addr_bar0);
    ERR_PCI_IOMAP:
        pci_release_region(pci_dev, 0);
    ERR_PCI_REQUEST_REGION:
        pci_disable_device(pci_dev);
    ERR_PCI_ENABLE_DEVICE:
        return -1;
    }

    static void edu_remove(struct pci_dev *pci_dev) {
        pr_info("in edu_remove()...\n");
        free_irq(pci_dev->irq, pci_dev);
        pci_iounmap(pci_dev, base_addr_bar0);
        pci_release_region(pci_dev, 0);
        pci_disable_device(pci_dev);
    }

    static struct pci_driver edu_driver = {
        .name = "qemu_edu_drv",
        .id_table = pci_id_table,
        .probe = edu_probe,
        .remove = edu_remove
    };

    int init_mod(void) {
        pr_info("init hlc module...\n");
        int ret = pci_register_driver(&edu_driver);
        if (ret != 0) {
            pr_err("fail to register pci driver\n");
            goto ERR_PCI_REGISTER_DRIVER;
        }
        return 0;

    ERR_PCI_REGISTER_DRIVER:
        return -1;
    }

    void exit_mod(void) {
        pr_info("exit hlc module...\n");
        pci_unregister_driver(&edu_driver);
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 3040.252553] hello: loading out-of-tree module taints kernel.
    [ 3040.252692] hello: module verification failed: signature and/or required key missing - tainting kernel
    [ 3040.256257] init hlc module...
    [ 3040.256842] in edu_probe()...
    [ 3040.292191] ACPI: \_SB_.LNKD: Enabled at IRQ 10
    [ 3040.292492] successfully probe edu pci dev
    [ 3051.406874] exit hlc module...
    [ 3051.406976] in edu_remove()...
    ```

* `free_irq()`

    free an interrupt allocated with request_irq

    syntax:

    ```c
    void free_irq(unsigned int irq, void * dev_id);
    ```

* 一个 pci iomap 的 example

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/pci.h>

    static struct pci_device_id pci_id_table[] = {
        { PCI_DEVICE(0x1234, 0x11e8) },
        {0,}
    };

    static void *base_addr_bar0;

    static int edu_probe(struct pci_dev *pci_dev, const struct pci_device_id *id) {
        pr_info("in edu_probe()...\n");

        int ret = pci_enable_device(pci_dev);
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to pci enable device, ret: %d\n", ret);
            goto ERR_PCI_ENABLE_DEVICE;
        }

        ret = pci_request_region(pci_dev, 0, "qemu_edu_drv");
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to pci request region\n");
            goto ERR_PCI_REQUEST_REGION;
        }

        resource_size_t res_len_bar0 = pci_resource_len(pci_dev, 0);
        base_addr_bar0 = pci_iomap(pci_dev, 0, res_len_bar0);
        if (base_addr_bar0 == NULL) {
            dev_err(&pci_dev->dev, "fail to pci iomap\n");
            goto ERR_PCI_IOMAP;
        }
        
        return 0;

    ERR_PCI_IOMAP:
        pci_release_region(pci_dev, 0);
    ERR_PCI_REQUEST_REGION:
        pci_disable_device(pci_dev);
    ERR_PCI_ENABLE_DEVICE:
        return -1;
    }

    static void edu_remove(struct pci_dev *pci_dev) {
        pr_info("in edu_remove()...\n");
        pci_iounmap(pci_dev, base_addr_bar0);
        pci_release_region(pci_dev, 0);
        pci_disable_device(pci_dev);
    }

    static struct pci_driver edu_driver = {
        .name = "qemu_edu_drv",
        .id_table = pci_id_table,
        .probe = edu_probe,
        .remove = edu_remove
    };

    int init_mod(void) {
        pr_info("init hlc module...\n");
        int ret = pci_register_driver(&edu_driver);
        if (ret != 0) {
            pr_err("fail to register pci driver\n");
            goto ERR_PCI_REGISTER_DRIVER;
        }
        return 0;

    ERR_PCI_REGISTER_DRIVER:
        return -1;
    }

    void exit_mod(void) {
        pr_info("exit hlc module...\n");
        pci_unregister_driver(&edu_driver);
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 1514.274978] hello: loading out-of-tree module taints kernel.
    [ 1514.275048] hello: module verification failed: signature and/or required key missing - tainting kernel
    [ 1514.277452] init hlc module...
    [ 1514.277735] in edu_probe()...
    [ 1514.313214] ACPI: \_SB_.LNKD: Enabled at IRQ 10
    [ 1523.042644] exit hlc module...
    [ 1523.042691] in edu_remove()...
    ```

* `dma_set_mask_and_coherent()`主要影响的是`dma_alloc_coherent()`, `dma_map_single()`之类的函数，与`pci_iomap()`无关。

* `wmb()`

    写内存屏障 (Write Memory Barrier), 全称: smp_wmb(), 是 linux kernel 中定义的宏。

    确保所有在 wmb() 之前的写操作，一定在所有在 wmb() 之后的写操作之前执行完成。

    syntax:

    ```c
    #include <asm-generic/barrier.h>
    #include <arch/x86/include/asm/barrier.h>  // for x86 arch

    wmb()
    ```

    * 典型场景: 生产者-消费者模式

        1. 生产者准备数据（写操作到内存缓冲区）。

        2. 生产者设置一个标志位（例如 data_ready = 1）来通知消费者数据准备好了。

        如果没有 wmb()，CPU或编译器可能会重排序，先执行步骤2（写标志位），后执行步骤1（写数据）。这样消费者看到标志位后去读数据，读到的可能就是还没准备好的错误数据。
        在步骤1和步骤2之间插入 wmb()，就能保证数据一定先于标志位被写入。

* `rmb()`

    读内存屏障 (Read Memory Barrier), 全称: smp_rmb()

    确保所有在 rmb() 之前的读操作，一定在所有在 rmb() 之后的读操作之前执行完成。

    * 典型场景: 生产者-消费者模式（消费者侧）

        1. 消费者看到标志位 data_ready == 1（读操作）。

        2. 消费者去读取生产者准备好的数据（读操作）。

        如果没有 rmb()，CPU可能会为了效率先预读数据缓冲区，然后再检查标志位。如果标志位检查失败，它就读到了无效的旧数据。
        在步骤1和步骤2之间插入 rmb()，就能保证一定是先读完标志位，确认数据有效后，再去读数据。

* `wmb()`与`rmb()`的 ecample

    ```c
    // 共享数据结构
    struct shared_data {
        int data[100];
        int flag;
    };

    // 生产者线程
    void producer(void) {
        // ... 准备数据到 data_buffer ...
        for (int i = 0; i < 100; i++) {
            shared->data[i] = i;
        }

        // 写内存屏障！确保上面的数据写入一定先于下面的标志位写入
        wmb();

        // 发布数据：通知消费者数据准备好了
        shared->flag = 1;
    }

    // 消费者线程
    void consumer(void) {
        while (!shared->flag) { // 等待数据准备好
            // 等待...
        }

        // 读内存屏障！确保先读完标志位，再读数据
        rmb();

        // 现在可以安全地消费数据了
        for (int i = 0; i < 100; i++) {
            print("%d\n", shared->data[i]);
        }
    }
    ```

* 内存屏障

    内存屏障（Memory Barrier），也称为内存栅栏（Memory Fence），是一类低级别的指令，用于强制限制CPU和编译器对内存访问操作的重排序。

    它的主要作用是：确保屏障之前的特定内存操作一定在屏障之后的特定内存操作之前完成。

    如果没有内存屏障，为了提高性能，CPU和编译器可能会对指令进行重排序，这在单核时代没问题，但在多核多线程并发环境下，会导致其他线程看到错误的执行顺序，从而引发不可预知的错误。

* 在使用 stream dma 时，如果`dma_map_single()`指定的 flag 为`DMA_BIDIRECTIONAL`，那么每次映射/同步操作都可能同时执行“刷写”和“失效”，性能开销最大。通过这种方法保证缓存一致性。

* `iowrite32_rep()`

    将一块数据（由多个 32 位字组成）连续、高效地写入一个内存映射的 I/O（MMIO）设备寄存器或端口。

    它操作的对象不是普通的内存，而是被映射到内存空间的设备寄存器。这些寄存器控制着硬件设备（如网卡、显卡、磁盘控制器等）的行为或用于与设备交换数据。

    io： 代表输入/输出（Input/Output）。

    write： 代表“写”操作。

    32： 代表每次操作的数据单元是 32 位（即 4 字节的一个“字”）。

    _rep： 是 “repeat” 的缩写，代表这个操作是重复的、批量的。

    ```c
    #include <linux/io.h>

    void iowrite32_rep(const volatile void __iomem *addr, const void *buf, unsigned long count);
    ```

    addr: 要写入的设备寄存器的基础地址（内存映射地址）。

    buf: 源数据缓冲区的地址，即你要发送给设备的数据所在的内存位置。

    count: 要写入的 32 位字的数量，即重复的次数。

    example:

    ```c
    #include <linux/io.h>
    #include <linux/types.h>

    /* 假设我们有一个设备，其数据端口被映射到内存地址 my_device_port */
    static void __iomem *my_device_port;

    /* 我们要发送的数据 */
    u32 data_buffer[] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222};

    void send_data_to_device(void)
    {
        /* 将4个32位字从 data_buffer 连续写入到 my_device_port */
        iowrite32_rep(my_device_port, data_buffer, 4);

        /* 
         * 等效于（但效率远高于）：
         * iowrite32(data_buffer[0], my_device_port);
         * iowrite32(data_buffer[1], my_device_port + 4); // 地址通常按字节偏移
         * iowrite32(data_buffer[2], my_device_port + 8);
         * iowrite32(data_buffer[3], my_device_port + 12);
         */
    }
    ```

    工作流程：

    当你调用 iowrite32_rep(addr, buf, count) 时，它会：

    1. 从内存中的 buf 位置开始读取数据。

    2. 将第一个 32 位数据写入 addr 指定的设备地址。

    3. 根据设备的总线特性，可能会自动将地址递增到下一个相邻的寄存器位置（或者使用同一个地址，这取决于设备的设计）。

    4. 重复这个过程，直到成功写完了 count 个 32 位数据。

    使用 iowrite32_rep() 比在循环中多次调用 iowrite32() 效率更高。因为它允许内核或底层架构使用更优化的方式来完成批量传输，例如使用处理器的缓存预取或更高效的总线指令（如 x86 架构上的 MOVS 指令配合 REP 前缀）。

    在需要向设备传输大量数据时（例如，向网卡发送一个数据包，或向磁盘控制器发送一系列指令参数），使用 _rep 版本的函数可以显著提升性能并减少代码量。

    函数家族：

    ```c
    #include <asm/io.h>

    void iowrite8_rep(volatile void __iomem *addr, const void *buf, unsigned long count);
    void iowrite16_rep(volatile void __iomem *addr, const void *buf, unsigned long count);
    void iowrite32_rep(volatile void __iomem *addr, const void *buf, unsigned long count);

    void ioread8_rep(volatile void __iomem *addr, void *buf, unsigned long count);
    void ioread16_rep(volatile void __iomem *addr, void *buf, unsigned long count);
    void ioread32_rep(volatile void __iomem *addr, void *buf, unsigned long count);
    ```

* `BIT()`

    BIT() 是一个宏，它的核心作用是生成一个指定位为 1，其它位为 0 的掩码（mask）。它通常用于操作硬件寄存器中的单个位，例如设置、清除或检查某个特定的标志位。

    syntax:

    ```c
    #include <linux/bits.h>

    #define BIT(nr) (1UL << (nr))
    ```

    * `nr`： 是你想要设置的位的序号（从 0 开始计数）。

    * `1UL`： 表示一个无符号长整型（Unsigned Long）的数字 1。使用 UL 是为了确保位操作在 32 位或 64 位系统上都是安全的，避免符号扩展和位数不足的问题。

    examples:

    * 设置位（Set a Bit）

        使用按位或（|=） 操作来设置一个寄存器的特定位。
        
        ```c
        // 设置寄存器的第 3 位 (假设位序号从 0 开始)
        register |= BIT(3);
        ```

    * 清除位（Clear a Bit）

        使用按位与（&=） 和取反（~） 操作来清除一个寄存器的特定位。

        ```c
        // 清除寄存器的第 5 位
        register &= ~BIT(5);
        ```

    * 检查位（Check a Bit）

        使用按位与（&） 操作来检查一个寄存器的特定位是否被设置。

        ```c
        // 检查寄存器的第 2 位是否被设置
        if (register & BIT(2)) {
            // 第 2 位为 1，执行某些操作
        } else {
            // 第 2 位为 0
        }
        ```

    * 切换位（Toggle a Bit）

        使用按位异或（^=） 操作来切换一个寄存器的特定位（1 变 0，0 变 1）。

        ```c
        // 切换寄存器的第 4 位
        register ^= BIT(4);
        ```

* readq() 和 writeq()

    用于以原子性的、64 位（8 字节） 的宽度从设备的内存映射寄存器或硬件缓冲区中读取和写入数据。q 代表 "Quad Word"

    syntax:

    ```c
    #include <linux/io.h> // 需要包含这个头文件

    u64 readq(const volatile void __iomem *addr);
    void writeq(u64 value, volatile void __iomem *addr);
    ```

    与 readq_relaxed() 和 writeq_relaxed() 的区别

    * readq() / writeq()： 包含完整的内存屏障（rmb() / wmb()）。这意味着在执行该读/写操作之前，所有之前的内存访问必须完成；之后的所有内存访问必须在该操作完成后才能开始。这保证了严格的执行顺序，但性能开销稍大。

    * readq_relaxed() / writeq_relaxed()： “宽松”版本，不提供任何内存屏障保证。它们更快，但只适用于该特定访问的顺序对设备操作的正确性无关紧要的场景。使用时必须非常小心。

* `BUG()`

    用于主动引发 kernel panic。内核会立即停止正常运行，触发 panic。这会中止当前所有进程，并通常会导致系统重启或挂起（取决于配置）。

    当 BUG() 被触发时，内核会打印出大量的调试信息，其中最关键是：

    * Oops 消息：包含出错的指令地址、寄存器状态、调用栈回溯（stack trace）等。

    * 调用栈（Stack Trace）：这能帮助开发者精确地定位是哪个函数、哪一行代码触发了 BUG()。

    example:

    ```c
    void my_driver_function(unsigned int command) {
        switch (command) {
            case CMD_READ:
                // ... 处理读操作
                break;
            case CMD_WRITE:
                // ... 处理写操作
                break;
            // 假设我们只定义了上面两个命令，理论上传进来的 command 只能是其中之一
            default:
                // 如果执行到这里，说明调用者传入了非法参数，或者我们的假设错误
                BUG();
        }
    }
    ```

    * panic(): 也会导致内核崩溃，但它通常用于更上层的、可预测的灾难性错误（如无法挂载根文件系统）。而 BUG() 更侧重于指示内核内部的编程逻辑错误。

    * WARN(): 类似于 BUG()，但它不会导致系统崩溃。它只是打印一个完整的警告信息和堆栈跟踪，然后让系统继续运行。适用于那些严重但或许还能让系统勉强运行下去的错误。

* `mdev_get_drvdata()`

    用于 VFIO Mediated Device (mdev) 框架的函数。它的主要作用是从一个 mdev_device 结构体中，获取驱动之前存储的私有数据指针。

    VFIO (Virtual Function I/O)： 是一个内核框架，允许用户空间程序（如虚拟机）安全、高效地直接访问硬件设备（即设备直通，Device Passthrough）。

    mdev (Mediated Device)： 是 VFIO 框架的一部分，用于实现设备模拟或分区。它允许一个物理设备（如图形卡、网络卡）被虚拟化成多个独立的虚拟设备，每个虚拟设备可以被分配给不同的虚拟机。

    syntax:

    ```c
    void *mdev_get_drvdata(struct mdev_device *mdev);
    ```

    参数： `mdev` - 指向要获取数据的 mdev_device 的指针。

    返回值： 成功时返回之前通过 mdev_set_drvdata() 设置的 void * 类型的私有数据指针；如果从未设置过，则可能返回 NULL。

    example:

    ```c
    // 1. 在设备创建时（例如在 'probe' 回调中）分配并设置私有数据
    static int my_mdev_probe(struct mdev_device *mdev)
    {
        struct my_private_data *data;

        data = kzalloc(sizeof(*data), GFP_KERNEL);
        if (!data)
            return -ENOMEM;

        // 初始化私有数据结构...
        data->parent_dev = mdev->parent->dev;
        // ... 其他初始化

        // 关键步骤：将私有数据指针存储到 mdev_device 中
        mdev_set_drvdata(mdev, data);

        return 0;
    }

    // 2. 在设备的其他操作函数中（如 ioctl），获取并使用私有数据
    static long my_mdev_ioctl(struct mdev_device *mdev, unsigned int cmd, unsigned long arg)
    {
        // 关键步骤：获取之前存储的私有数据
        struct my_private_data *data = mdev_get_drvdata(mdev);

        if (!data) // 安全检查
            return -EINVAL;

        // 现在可以使用 data 来访问这个特定设备的上下文信息了
        switch (cmd) {
            case MY_CMD:
                // 操作 data->some_field
                break;
            // ... 其他命令
            default:
                return -ENOTTY;
        }
        return 0;
    }

    // 3. 在设备销毁时（例如在 'remove' 回调中），获取数据并释放内存
    static void my_mdev_remove(struct mdev_device *mdev)
    {
        struct my_private_data *data = mdev_get_drvdata(mdev);

        // 清理私有数据可能占用的资源...
        
        // 释放私有数据内存
        kfree(data);
        
        // 可选：将 mdev 的 drvdata 设为 NULL
        mdev_set_drvdata(mdev, NULL);
    }
    ```

    配套函数：`mdev_set_drvdata()`

* `copy_from_user()`

    将数据从用户空间安全地复制到内核空间。

    syntax:

    ```c
    #include <linux/uaccess.h>

    unsigned long copy_from_user(void *to, const void __user *from, unsigned long n);
    ```

    parameters:

    * `to`: Destination address, in the kernel space

    * `from`: The source address in the user space

    * `n`: Number of bytes to copy

    Returns number of bytes that could not be copied. On success, this will be zero.

* `devm_ioremap_resource()`

    将一个设备（通常是硬件外设）的物理内存地址区域（例如寄存器组）映射到内核的虚拟地址空间，并自动管理该映射的生命周期。

    syntax:

    ```c
    #include <linux/device.h>   // 提供 struct device 的定义
    #include <linux/io.h>       // 提供 __iomem、IORESOURCE_MEM 等定义和函数声明

    void __iomem *devm_ioremap_resource(struct device *dev, const struct resource *res);
    ```

    params:

    * `struct device *dev`: 指向申请该资源映射的设备结构体的指针。这通常由平台驱动框架自动提供（例如在 probe 函数中，&pdev->dev）。

    * `const struct resource *res`: 指向要映射的硬件资源（resource 结构体）的指针。这个资源通常通过 platform_get_resource() 等函数从平台设备信息中获取。

    返回值：

    * 成功: 返回一个指向映射后内核虚拟地址空间的指针（void __iomem * 类型）。驱动程序后续通过此指针（配合 readl/writel 等函数）来访问硬件寄存器。

    * 失败: 返回 ERR_PTR(...)，即一个封装了错误码的指针。需要使用 IS_ERR() 和 PTR_ERR() 来检查和获取错误信息。

    example:

    ```c
    #include <linux/device.h>
    #include <linux/io.h>
    #include <linux/platform_device.h> // 用于 platform_get_resource

    struct my_private_data {
        void __iomem *reg_base;
    };

    static int my_driver_probe(struct platform_device *pdev)
    {
        struct device *dev = &pdev->dev;
        struct resource *res;
        struct my_private_data *priv;

        // 从平台设备获取内存资源（索引0）
        res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
        if (!res) {
            dev_err(dev, "Failed to get MEM resource\n");
            return -EINVAL;
        }

        // 申请设备私有数据
        priv = devm_kzalloc(dev, sizeof(*priv), GFP_KERNEL);
        if (!priv)
            return -ENOMEM;

        // 核心步骤：映射设备的寄存器区域
        priv->reg_base = devm_ioremap_resource(dev, res);
        if (IS_ERR(priv->reg_base)) {
            dev_err(dev, "Failed to ioremap region\n");
            return PTR_ERR(priv->reg_base);
        }

        // 将私有数据存储到设备结构体中
        platform_set_drvdata(pdev, priv);

        // ... 后续的硬件初始化、中断注册等操作 ...
        // 可以通过 priv->reg_base 来访问寄存器，例如：
        // writel(0x12345678, priv->reg_base + SOME_REG_OFFSET);

        return 0;
    }
    // 注意：不需要在 remove 函数中手动 iounmap，devm_ 机制会自动处理。
    ```

* 缓存维护操作（Cache Flushing/Invalidating）

* 字节序转换宏

    ```c
    #include <linux/byteorder/generic.h>

    cpu_to_be16(x)	// CPU字节序 到 大端字节序 (16位)	htobe16(x)
    cpu_to_be32(x)	// CPU字节序 到 大端字节序 (32位)	htobe32(x)
    cpu_to_be64(x)	// CPU字节序 到 大端字节序 (64位)	htobe64(x)
    cpu_to_le16(x)	// CPU字节序 到 小端字节序 (16位)	htole16(x)
    cpu_to_le32(x)	// CPU字节序 到 小端字节序 (32位)	htole32(x)
    cpu_to_le64(x)	// CPU字节序 到 小端字节序 (64位)	htole64(x)
    be16_to_cpu(x)	// 大端字节序 到 CPU字节序 (16位)	be16toh(x)
    be32_to_cpu(x)	// 大端字节序 到 CPU字节序 (32位)	be32toh(x)
    be64_to_cpu(x)	// 大端字节序 到 CPU字节序 (64位)	be64toh(x)
    le16_to_cpu(x)	// 小端字节序 到 CPU字节序 (16位)	le16toh(x)
    le32_to_cpu(x)	// 小端字节序 到 CPU字节序 (32位)	le32toh(x)
    le64_to_cpu(x)	// 小端字节序 到 CPU字节序 (64位)	le64toh(x)
    ```

    `cpu`代表当前 CPU 的字节序（即主机字节序）。

* `readl()`, `writel()`

    用于访问内存映射 I/O (MMIO) 设备，读取/写入 32 位（`long`）的数据。

    syntax:

    ```c
    void writel(u32 value, volatile void __iomem *addr);
    u32 readl(const volatile void __iomem *addr);
    ```

    `writel()`: 它确保了写入操作是原子的（不会被其他操作打断），并且会处理不同 CPU 架构可能存在的字节序（Big-Endian vs Little-Endian）和内存访问顺序问题，保证驱动程序的跨平台兼容性。

    `readl()`: 除了处理字节序和内存屏障，它还能防止编译器对读取操作进行错误的优化（例如，认为该地址的值不会变化而将其缓存，导致重复读取同一个旧值）。volatile 关键字确保了每次都会从硬件地址实实在在地读取数据。

    我们不应该直接解引用访问内存映射的 I/O 寄存器。

    可能遇到的问题：
    
    1. 编译器优化问题：多次读取同一个寄存器。编译器可能会进行优化，认为第一次读取的值和第二次是一样的，于是省去第二次的实际读取操作，直接使用缓存的值。

    1. 内存访问顺序问题（内存屏障）：CPU 和编译器为了提升效率，可能会打乱指令的执行顺序（Out-of-Order Execution）。

        问题场景：

        1. 向 FIFO 数据寄存器 写入数据 writel(data, FIFO_DATA);

        2. 向 控制寄存器 写入一个启动命令 writel(CMD_START, FIFO_CTRL);
        硬件要求必须先写数据再发命令。

        如果没有屏障，CPU或编译器可能为了效率，先执行步骤2再执行步骤1，导致硬件收到错误的顺序，操作失败。

    1. 字节序问题: 不同的 CPU 架构（如 x86/Little-Endian 和 PowerPC/Big-Endian）和多字节数据在内存中的存储方式（字节序）不同。外围设备也有自己期望的字节序。

        要向一个设备寄存器写入一个32位值 0x12345678。在小端序的 CPU 上，这个值在内存中存储为 78 56 34 12。如果设备期望的是大端序（即希望收到 12 34 56 78），直接解引用写入就会发送错误的数据。

    1. 访问宽度和原子性问题: 对于某些寄存器，32位的写入操作必须是原子的（不可分割的）。

* `sparse`

    Linux 内核代码的静态代码分析器, 主要用于定位与类型安全和内存模型相关的问题。

    它通过扩展 GCC 并解析源代码来实现以下功能：

    * 类型检查： 强化内核中特定的数据类型属性。例如，检查地址空间（__user, __kernel）、内存一致性（__rcu）等注解是否正确使用，确保用户空间的指针不会被直接解引用。

    * 上下文检查： 确保某些函数在正确的上下文中调用（例如，可能睡眠的函数不能在原子上下文中调用）。

    * 锁机制检查： 帮助检查锁的初始化和使用是否正确，减少死锁风险。

    * 字节序检查： 帮助识别可能存在的字节序（Endianness）问题。

    通常在编译内核时使用，make C=1（检查重新编译的文件）或 make C=2（检查所有文件）。

* `outl`

    端口映射 I/O (PMIO)：使用一个与物理内存地址空间分离的、独立的地址空间（0x0000 到 0xFFFF）来访问设备。CPU 需要使用专门的 in 和 out 指令来与这个空间进行数据交换。

    `outl`: 从 CPU 向 外部设备的指定端口输出（写） 一个 32位（4字节） 的长字（Long Word） 数据。

    syntax (AT&T):

    ```asm
    outl %eax, port_number
    ```

    CPU 执行 outl 指令时，会将一个 32 位的值（通常存放在寄存器 eax 中）通过数据总线，发送到由指令指定的一个 16 位的 I/O 端口地址.

    当端口号大于 255 时，需要先放入 dx 寄存器：

    ```asm
    movw $0x1F7, %dx   # 将端口号 0x1F7 存入 dx 寄存器
    outl %eax, %dx     # 将 eax 中的 32 位数据输出到 dx 指定的端口
    ```

    example:

    告诉硬盘控制器要读取的扇区数:

    ```asm
    movl $0x100, %eax  # 设置要读取 256 个扇区
    movw $0x1F2, %dx   # 0x1F2 是硬盘扇区计数器的端口号
    outl %eax, %dx     # 将这个命令发送给硬盘
    ```

* `inb`

    从外部设备的指定端口向 CPU 输入（读） 一个 8位（1字节） 的数据。

    `b`：代表操作的数据长度是“字节”（8位）。

    syntax:

    ```asm
    inb port_number, %al
    ```

    CPU 执行 inb 指令时，会通过地址总线指定一个 16 位的 I/O 端口，然后该端口上的设备会通过数据总线返回一个 8 位的值，CPU 会将其接收（通常存入 al 寄存器）。

    或者当端口号大于 255 时：

    ```asm
    movw $0x3F8, %dx   # 将串口线路状态寄存器的端口号 0x3F8 存入 dx
    inb %dx, %al       # 从该端口读取一个字节的状态信息到 al 寄存器
    ```

    example:

    检查串口是否有数据到达:

    ```asm
    movw $0x3FD, %dx   # 0x3FD 是串口线路状态寄存器的端口
    inb %dx, %al       # 读取状态字节
    testb $0x01, %al   # 检查“数据就绪”位是否为 1
    jnz data_available # 如果为 1，跳转到数据处理代码
    ```

    `b`是字节（8位），`w`是字（16位），`l`是长字（32位）。

* `devm_ioremap_resource()`

    将 I/O 内存资源（通常是 MMIO 寄存器空间）映射到内核虚拟地址空间，并且由 设备管理器（device-managed, `devm_`）自动管理其生命周期.

    syntax:

    ```c
    void __iomem *devm_ioremap_resource(struct device *dev, struct resource *res);
    ```

    * res：指向已获取的 struct resource *，通常是 IORESOURCE_MEM 类型。

    返回值：

    成功：返回映射后的内核虚拟地址（void __iomem *）。

    失败：返回错误指针，使用 IS_ERR() 检查。

    作用总结:

    1. 获取资源：先调用 platform_get_resource() 或 pci_get_resource() 得到 struct resource *。

    2. 检查资源有效性：函数内部会调用 request_mem_region() 确保资源未被占用。

    3. 映射 I/O 内存：使用 ioremap() 将物理地址映射到内核虚拟地址。

    4. 自动释放：使用 devm 机制，设备卸载时自动调用 iounmap() 并释放资源。

* 可以使用`dmesg -e`显示消息的大致时间戳

* `ioremap()`与`pci_iomap()`的区别

    ```c
    #include <asm/io.h>

    void __iomem *ioremap(phys_addr_t offset, unsigned long size);
    void iounmap(volatile void __iomem *addr);


    #include <linux/pci.h>

    void __iomem *pci_iomap(struct pci_dev *dev, int bar, unsigned long maxlen);
    void pci_iounmap(struct pci_dev *dev, void __iomem *addr);
    ```

    `ioremap()`更底层，更通用。我们给定一个物理地址和大小，它请求内核在页表中建立映射，并返回一个可用的内核虚拟地址。它不关心这个物理地址来自哪里（可以是系统内存、设备内存、甚至是ISA总线上的端口）。

    `pci_iomap()`更高层、PCI 专用。它构建在`ioremap()`或其他类似机制之上，专门用于映射 PCI 设备 BAR（Base Address Register）中声明的内存区域。

    如果使用`ioremap()`，那么需要驱动程序开发者自己获取要映射的物理地址。对于 PCI 设备，需要先调用`pci_resource_start(pdev, bar)`来获取 BAR 对应的物理地址。

    `pci_iomap()`相当于帮你调用了`pci_resource_start()`, `pci_resource_len()`, `ioremap()`。

* `ioread32()`, `iowrite32()`

    用于安全、可移植地访问内存映射 I/O (MMIO) 寄存器.

    syntax:

    ```c
    #include <asm/io.h>

    void iowrite8(u8 value, volatile void __iomem *addr);
    void iowrite16(u16 value, volatile void __iomem *addr);
    void iowrite32(u32 value, volatile void __iomem *addr);
    #ifdef CONFIG_64BIT
    void iowrite64(u64 value, volatile void __iomem *addr);
    #endif
    ```

    参数:

    * `value`: 要写入的数据值。类型分别为`u8`, `u16`, `u32`, `u64`（无符号8/16/32/64位整数）。

    * `addr`: 目标内存映射I/O地址。这是一个指向`__iomem`空间的指针，通常由`ioremap()`, `pci_iomap()`等函数返回。`volatile`关键字告知编译器该地址的内容可能会被硬件意外修改，防止编译器进行激进的优化。

    ```c
    #include <asm/io.h>

    u8 ioread8(const volatile void __iomem *addr);
    u16 ioread16(const volatile void __iomem *addr);
    u32 ioread32(const volatile void __iomem *addr);
    #ifdef CONFIG_64BIT
    u64 ioread64(const volatile void __iomem *addr);
    #endif
    ```

    `addr`: 要读取的源内存映射I/O地址。同样是一个带有 __iomem 和 volatile 修饰符的指针。const 表明函数不会修改指针所指向的内容。

    任何时候你需要与通过 ioremap() 映射的设备寄存器进行交互，都必须使用 ioreadX()/iowriteX() 函数家族，而不是直接使用指针解引用（如 *reg = value）。

* `raw_copy_to_user()`

    将数据从内核空间安全地复制到用户空间。

    syntax:

    ```c
    unsigned long raw_copy_to_user(void __user *to, const void *from, unsigned long n);
    ```

    to： 目标地址（用户空间指针）。

    from： 源地址（内核空间指针）。

    n： 要拷贝的字节数。

    返回值： 如果成功，返回 0；如果失败，返回尚未成功拷贝的字节数。

    “raw”前缀： 这表明它是底层实现。相比 copy_to_user()，它假设调用者已经处理了可能引起睡眠的事情（如缺页中断），通常在知道上下文是安全的情况下使用，性能稍高。

* `int major = register_chrdev(0, "hlc_dev", &fops);`z失败时会返回负数，成功时返回 0 或正数

    常见的错误码：

    * `-ENOMEM`: 内存分配失败

    * `-EBUSY`: 设备号已被占用（当第一个参数不为0时）

    * `-EINVAL`: 无效参数

    * `-ENODEV`: 设备不存在或其他错误

    成功时的返回值：

    * 如果第一个参数为 0（如你的例子）：返回动态分配的正的主设备号

    * 如果第一个参数指定了具体设备号：返回0表示成功

* `ioremap_wc()`

    将一段物理内存地址映射到内核虚拟地址空间，并指明该映射支持“写合并”（Write-Combining）内存访问特性

    特点：

    * 合并（Combining）：如果CPU在很短的时间内连续发起多个针对相邻地址的写入操作，内存控制器可以将这些操作合并为一个更大的总线传输事务。这显著减少了访问次数，提高了效率。

    * 缓冲（Buffering）：写入操作可能不会立即到达最终设备，而是先被放入一个缓冲区，稍后再一起发送。这同样是为了优化性能。

    * 弱排序（Weak Ordering）：写合并操作不严格保证其相对于其他内存操作的顺序，这为硬件优化提供了更大灵活性。

    常用于: 显卡的帧缓冲区（Framebuffer）或显存.

    在现代内核（大约 v2.6 以后）中, `ioremap()`默认就等同于`ioremap_nocache()`.

* `kstrdup()`

    在内核空间（Kernel Space）中为指定的一个字符串（以 '\0' 结尾的 C 字符串）分配一块新的内存，并将原字符串的内容复制到这块新内存中。

    syntax:

    ```c
    #include <linux/string.h> // 需要包含的头文件

    char *kstrdup(const char *s, gfp_t gfp_mask);
    ```

    功能与`strdup()`类似。分配的内存在不再需要时，必须使用`kfree()`来释放。

* `gfp_mask`: Get Free Page mask

* `devm_kzalloc()`

    分配一块指定大小的内存，并将其初始化为零，同时将该内存的释放（free）操作与设备本身的生命周期进行绑定。

    `devm_`表示 device managed。

    syntax:

    ```c
    void *devm_kzalloc(struct device *dev, size_t size, gfp_t gfp);
    ```

    `flags`： 分配标志，用于控制分配行为（例如，GFP_KERNEL 表示在进程上下文中可睡眠等待，GFP_ATOMIC 表示在中断等原子上下文中分配）。

    当发生以下情况时，内核会自动释放这块内存：

    * 驱动被卸载（module unload）。

    * 设备被分离（device detach）。

    * 在分配过程中出现错误，导致设备探测（probe）失败。

* `ioremap_cache()`

    将设备的物理 I/O 内存（通常是 PCI/设备的寄存器或显存）映射到内核的虚拟地址空间，并且允许这段区域使用 CPU 的缓存（cache）机制。

    与其他 ioremap 的区别（未验证）：

    * `ioremap()`（或 ioremap_nocache()）：映射为 非缓存（uncached） 内存，CPU 访问时不走缓存，保证读写与硬件保持一致。

    * `ioremap_wc()`：写合并（write-combining），适合显存等带宽敏感区域。

    * `ioremap_cache()`：映射为 缓存（cached） 内存，CPU 访问时可以走缓存，性能高，但可能会导致和设备之间的数据不一致（需要显式刷新/失效 cache）。

    `ioremap_cache() `一般用于：

    * 显存（framebuffer）等只读/读多写少的区域，提高 CPU 访问效率；

    * 确定设备内存与 CPU 缓存一致性可控时。

    注意：不能随便对寄存器区使用 ioremap_cache()，因为缓存会导致寄存器读写失效或顺序错误。

* `resource_size()`

    计算一个`struct resource`所描述的硬件资源块的大小。

    syntax:

    ```c
    #include <linux/ioport.h> // 需要包含这个头文件

    resource_size(struct resource *res);
    ```

    参数:

        res: 指向 struct resource 的指针，通常是 platform_get_resource() 的返回值。

    返回值:

        返回该资源块的大小（以字节为单位）。

        如果传入的 res 是 NULL，行为是未定义的（通常会导致内核崩溃/Oops）。

    example:

    ```c
    static int my_driver_probe(struct platform_device *pdev)
    {
        struct resource *res;
        void __iomem *base_addr;

        // 1. 获取内存资源
        res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
        if (!res) {
            dev_err(&pdev->dev, "Failed to get MEM resource\n");
            return -EINVAL;
        }

        // 2. 使用 resource_size() 获取资源大小
        dev_info(&pdev->dev, "Resource start: 0x%llx, size: %lu bytes\n",
                 (unsigned long long)res->start, (unsigned long)resource_size(res));

        // 3. 申请并映射这段内存区域到内核的虚拟地址空间
        // 在调用 devm_ioremap_resource 时，内核内部也会使用 resource_size()
        base_addr = devm_ioremap_resource(&pdev->dev, res);
        if (IS_ERR(base_addr)) {
            return PTR_ERR(base_addr);
        }

        // 现在 base_addr 就是这段内存的起始虚拟地址，
        // 你可以通过 ioread32(base_addr + offset) 等方式访问硬件寄存器了。

        // ... 其他初始化操作 ...
        return 0;
    }
    ```

* `register_chrdev()`中填的`name`是设备号的名称，不是 dev 的名称。可以在`/proc/devices`中查看。

    主设备号数字从 1 开始编号。

* `register_chrdev()`的 example

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/pci.h>

    int hlc_open(struct inode *, struct file *) {
        pr_info("in hlc_open()...\n");
        return 0;
    }

    int hlc_release(struct inode *, struct file *) {
        pr_info("in hlc_release()...\n");
        return 0;
    }

    ssize_t hlc_read(struct file *, char __user *, size_t, loff_t *) {
        pr_info("in hlc_read()...\n");
        return 0;
    }

    ssize_t hlc_write(struct file *, const char __user *, size_t, loff_t *) {
        pr_info("in hlc_write()...\n");
        return 0;
    }

    long hlc_ioctl(struct file *, unsigned int, unsigned long) {
        pr_info("in hlc_ioctl()...\n");
        return 0;
    }

    const struct file_operations fops = {
        .open = hlc_open,
        .release = hlc_release,
        .read = hlc_read,
        .write = hlc_write,
        .unlocked_ioctl = hlc_ioctl
    };

    int dev_num_major;

    int init_mod(void) {
        pr_info("init hlc module...\n");
        dev_num_major = register_chrdev(0, "qemu_edu", &fops);
        if (dev_num_major <= 0) {
            pr_err("fail to register chrdev\n");
            goto ERR_REGISTER_CHRDEV;
        }
        return 0;

    ERR_REGISTER_CHRDEV:
        return -1;
    }

    void exit_mod(void) {
        pr_info("exit hlc module...\n");
        unregister_chrdev(dev_num_major, "hlc_dev");
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

* `	devm_ioremap()`

    将一段物理地址空间（通常是内存映射式设备的寄存器区域）映射到内核的虚拟地址空间，并且自动资源管理。

    syntax:

    ```c
    #include <linux/io.h>

    void __iomem *devm_ioremap(struct device *dev, resource_size_t offset, resource_size_t size);
    ```

    offset: 要映射的物理地址的起始偏移量（通常从设备资源（如 struct resource）中获取）。

    size: 要映射的地址区域的大小。

    资源管理（Managed - `devm_*`前缀）: 这是`devm_ioremap()`相对于传统 ioremap() 最关键的优势。devm_ 前缀意味着这个函数是“被托管的”或“自动资源管理的”。它将申请的资源（这里是指映射的虚拟地址区域）与特定的设备（struct device *dev）绑定。

    当该设备被卸载或者驱动模块被移除时，内核会自动调用反向操作（即 devm_iounmap()）来释放这个映射。程序员无需在代码的错误处理路径和驱动退出函数中显式地调用 iounmap()。

    `pci_ioremap()`专门用于 PCI/PCIe 设备, 通常与 pci_resource_start() 等PCI特定函数配对使用.  `devm_ioremap()`	通用于所有内存映射I/O设备, 通常与 `platform_get_resource()`等通用或平台函数配对使用.

* `irq_set_affinity_hint()`

    向系统提供一个“提示”（Hint），建议它将某个硬件的中断请求（IRQ）分配给特定的 CPU（或一组 CPU）来处理。

    它的目的通常是帮助用户空间的驱动程序或管理工具（而不是内核本身的驱动程序）来优化中断处理，尤其是在与支持 RSS (Receive Side Scaling) 或 RPS (Receive Packet Steering) 的高性能网卡配合使用时。

    中断亲和性是一种设置，允许管理员手动地将特定的中断绑定到特定的CPU核心上。这可以通过修改`/proc/irq/<IRQ_NUMBER>/smp_affinity`文件来实现。

    syntax:

    ```c
    #include <linux/interrupt.h>

    int irq_set_affinity_hint(unsigned int irq, const struct cpumask *m);
    ```

    * `const struct cpumask *m`

        含义: 一个指向 cpumask 的指针，该结构体指明了你希望提示中断在哪个（或哪些）CPU核心上运行。

        如何创建:

        * 使用 cpumask_of(cpu) 宏来创建一个指向特定CPU的掩码。

        * 例如：cpumask_of(0) 表示 CPU 0。

        也可以使用更高级的宏（如 cpumask_local_spread）来根据NUMA架构智能选择CPU，或者手动构建一个更复杂的掩码。

    返回值 (Return Value)

    * 0: 表示成功。

    * 非零值: 表示失败。错误码通常是 -EINVAL（无效参数）。

    example:

    ```c
    #include <linux/interrupt.h>
    #include <linux/cpu.h>

    /* 假设你的驱动已经获取到了这个中断号 */
    unsigned int my_irq_num;

    /* 设置亲和性提示到 CPU 0 */
    void setup_irq_hint(void)
    {
        int ret;
        const struct cpumask *cpu_mask = cpumask_of(0); // 创建指向 CPU 0 的掩码

        ret = irq_set_affinity_hint(my_irq_num, cpu_mask);
        if (ret) {
            pr_err("Failed to set affinity hint for IRQ %d, error: %d\n", my_irq_num, ret);
            /* 错误处理 */
        }
    }

    /* 在驱动卸载或设备移除时，通常需要清除提示 */
    void cleanup_irq_hint(void)
    {
        /* 
         * 通过传递一个全空的 cpumask 来清除之前的提示。
         * 这告诉内核“不再有特定提示”。
         * 注意：在某些旧内核中，传递 NULL 也可以，但使用 cpu_none_mask 是更明确的做法。
         */
        irq_set_affinity_hint(my_irq_num, cpu_none_mask);
    }
    ```

    已过时 (Deprecated)：在较新的 Linux 内核中（大约 5.11 版本之后），这个函数已被标记为过时。官方推荐使用更新的 irq_set_affinity() 接口以及 struct irq_affinity_desc 中的 flags 字段来提供更明确的管理策略（例如，设置 IRQ_AFFINITY_FLAG_MANAGED 标志），而不是使用这种模糊的“提示”。

* `/proc/ioports`

    列出当前系统中所有设备已注册（申请）的 I/O 端口地址范围。

    在 x86 体系架构中，CPU 与外部设备（如网卡、磁盘控制器、键盘、串口等）进行通信的一种主要方式是通过I/O 端口。CPU 通过 in 和 out 汇编指令来从端口读取数据或向端口写入数据，从而控制硬件。

    查看：`sudo cat /proc/ioports`

    output:

    ```
    0000-0cf7 : PCI Bus 0000:00
      0000-001f : dma1
      0020-0021 : pic1
      0040-0043 : timer0
      0050-0053 : timer1
      0060-0060 : keyboard
      0064-0064 : keyboard
      0070-0071 : rtc_cmos
        0070-0071 : rtc0
      0080-008f : dma page reg
      00a0-00a1 : pic2
    ...
    ```

    注：

    1. 必须要用`sudo`，否则显示出来的地址全是 0

* `unlocked_ioctl`与`compat_ioctl`有何不同

    syntax:

    ```c
	long (*unlocked_ioctl) (struct file *filp, unsigned int cmd, unsigned long arg);

    long (*compat_ioctl) (struct file *filp, unsigned int cmd, unsigned long arg);
    ```

    unlocked_ioctl：现代的标准方法，用于处理所有常规的 ioctl 调用。它的主要特点是不再持有 Big Kernel Lock (BKL)，因此得名 “unlocked”。

    compat_ioctl：兼容性方法，专门用于为32位应用程序在64位内核上提供兼容性支持。

* `dev_set_drvdata()`, `dev_get_drvdata()`

    从 struct device 对象中获取其关联的驱动私有数据（private driver data）的指针.

    syntax:

    ```c
    void dev_set_drvdata(struct device *dev, void *data);

    void *dev_get_drvdata(const struct device *dev);
    ```

    返回值:

    * `void *`: 成功则返回之前通过`dev_set_drvdata()`设置的指针。

    * 如果之前没有设置过，或者 dev 为 NULL，则返回 NULL。

    其实`dev_set_drvdata()`本质是把一个指针和一个 dev 关联起来，这样用户可以在`struct my_device_data`里定义自己需要管理的设备状态、数据（比如 dma 的地址和大小）、控制同步的锁等等。这些资源如果定义成全局的，不是不可以，但是最后还是得用户去关联到 dev，非常麻烦。

    example:

    ```c
    /* 1. 驱动自定义的私有数据结构 */
    struct my_device_data {
        int irq;
        void __iomem *regs;
        spinlock_t lock;
        // ... 其他设备特定字段
    };

    /* 2. 在 probe 函数中设置驱动数据 */
    static int my_driver_probe(struct device *dev)
    {
        struct my_device_data *priv;

        priv = devm_kzalloc(dev, sizeof(*priv), GFP_KERNEL);
        if (!priv)
            return -ENOMEM;

        // 初始化 priv 结构...
        priv->irq = platform_get_irq(...);
        priv->regs = devm_ioremap_resource(...);

        // !!! 关键步骤：将 priv 指针存储到 dev 中 !!!
        dev_set_drvdata(dev, priv);

        return 0;
    }

    /* 3. 在其它函数中获取驱动数据 */
    static void my_driver_do_something(struct device *dev)
    {
        // !!! 关键步骤：从 dev 中取回 priv 指针 !!!
        struct my_device_data *priv = dev_get_drvdata(dev);

        if (!priv) {
            dev_err(dev, "No driver data found!\n");
            return;
        }

        // 现在可以使用 priv 来访问设备特定数据了
        writel(0xAA55, priv->regs + SOME_REG_OFFSET);
    }

    /* 4. 在 remove 函数中获取并清理 */
    static int my_driver_remove(struct device *dev)
    {
        struct my_device_data *priv = dev_get_drvdata(dev);

        // 使用 priv 进行资源清理...
        // devm_* 函数会自动清理，所以这里可能不需要做太多

        // 通常不需要手动将 drvdata 设置为 NULL，因为 dev 本身即将被销毁
        // 但显式设置是个好习惯
        dev_set_drvdata(dev, NULL);

        return 0;
    }
    ```

    如果`struct my_device_data`指针由`dma_set_mask_and_coherent()`赋予，并且只存储 binary 数据，那么也完全可以把这里当作一个 buffer。binary 数据中，可能会有字节序（Endianness）的问题，需要注意。

* `ioremap()`

    将一段物理内存地址（通常是设备的内存或寄存器）映射到内核的虚拟地址空间。

    syntax:

    ```c
    #include <asm/io.h>

    void __iomem *ioremap(resource_size_t phys_addr, unsigned long size);

    void iounmap(volatile void __iomem *addr);
    ```

    example:

    1. 在驱动初始化时映射：

    ```c
    // 假设已知设备的寄存器物理地址是 0xFE000000，长度为 0x2000
    void __iomem *device_regs;
    device_regs = ioremap(0xFE000000, 0x2000);
    if (!device_regs) {
        // 映射失败，错误处理
    }
    ```

    2. 通过返回的虚拟地址访问设备：

        必须使用专门的读写函数（如 readl(), writel()），而不是直接解引用指针。

        ```c
        // 读取一个32位的寄存器
        u32 value = readl(device_regs + REG_OFFSET);

        // 向一个32位的寄存器写入
        writel(new_value, device_regs + ANOTHER_REG_OFFSET);
        ```

        为什么不能用`*`直接访问？ 因为设备寄存器的访问可能有副作用（比如读一次清零）或有严格的访问顺序要求，这些专用函数能保证正确的内存访问语义。

    配对函数：`iounmap()`

    如果你要映射的是非PCI设备的内存（例如，在设备树中定义的平台设备的内存区域），那么你必须使用`ioremap()`。

    如果硬要使用`ioremap()`映射 bar 0 寄存器，那么可能这样做：

    ```c
    // 1. 手动读取PCI配置空间，获取BAR的物理地址和大小
    resource_size_t start = pci_resource_start(pdev, 0); // 获取物理地址
    resource_size_t len   = pci_resource_len(pdev, 0);   // 获取长度
    // 2. 检查资源是否有效（好习惯）
    if (!request_mem_region(start, len, "my_driver")) {
        // 错误处理：内存区域可能已被其他驱动占用
        return -EBUSY;
    }
    // 3. 进行映射
    void __iomem *addr = ioremap(start, len);
    if (!addr) {
        release_mem_region(start, len);
        return -ENOMEM;
    }
    ```

* qemu edu 不支持 MSI-X （未验证）

* Bus Master

    在PCI/PCIe总线架构中，大多数设备是“从设备（Slaves）”，只能响应来自CPU的读写请求。

    而有些设备（如网卡、磁盘控制器、高性能显卡）需要能够主动地向系统内存大量、高速地传输数据。这类设备被称为“总线主（Bus Master）”。

    要成为总线主，设备必须能够在总线上驱动地址和数据信号，主动发起传输周期。

    * 如何成为总线主？

        * 每个PCI设备都有一个配置空间，其中包含一个叫做命令寄存器（Command Register） 的16位寄存器。

        * 该寄存器的第2位专门用于控制“总线主”功能：

            * 位2 = 0：禁用设备的Bus Master功能。

            * 位2 = 1：启用设备的Bus Master功能。

        * 默认情况下，在系统启动或设备复位后，这一位通常是清零（禁用） 的。

* `pci_set_master()`

    启动 pci device 的 bus master 功能。启用这个功能后，该 PCI 设备才能主动发起 DMA（直接内存访问）操作，即能够直接读写系统内存，而无需CPU的参与。

    syntax:

    ```c
    void pci_set_master(struct pci_dev *dev);
    ```

    example:

    ```c
    // 启用设备 (pci_enable_device())。

    // 请求和映射I/O资源（如 pci_request_regions(), pci_iomap()）。

    // 分配DMA缓冲区等资源。

    // pci_set_master();
    ```

    `pci_set_master()`所做的工作非常简单：它设置PCI设备配置空间中命令寄存器的第2位。

    在驱动程序的 remove 或 shutdown 路径中，通常不需要显式地禁用Bus Master，因为PCI核心在禁用设备时会自动处理。

* `spin_lock()`

    一种用于实现互斥（Mutual Exclusion）的同步原语。它是一种忙等待锁（Busy-Wait Lock），其核心特征在于：当一个线程尝试获取一个已被占用的锁时，它不会进入睡眠状态，而是会在一个循环中不断地“旋转”（Spinning），反复检查锁是否被释放，直到成功获取锁为止。

    syntax:

    ```c
    #include <linux/spinlock.h>

    void spin_lock(spinlock_t *lock);
    ```

    如果锁已被占用，函数不会返回，当前CPU核会在此自旋，直到成功获取锁。

    自旋锁设计的初衷是保护执行时间极短的临界区。它可以在中断上下文（Interrupt Context）中使用，而睡眠锁（如Mutex）绝对不可以，因为中断上下文中不允许调度（睡眠）。

    这里的中断上下文指的是中断处理程序 (ISR)、软中断 (softirq)、tasklet、工作队列（上半部）。

    example:

    ```c
    #include <linux/spinlock.h>

    spinlock_t my_lock;
    DEFINE_SPINLOCK(my_lock);  // 编译时静态初始化

    void my_func() {
        // spinlock_t my_lock;
        // spin_lock_init(&my_lock);  // 运行时动态初始化
        spin_lock(spinlock_t *lock);
        // critical area
        spin_unlock(spinlock_t *lock);
    }
    ```

    在持有自旋锁的同时再次请求获取同一个自旋锁（递归加锁），这会导致该CPU核永久自旋，系统卡死。

* `charp`定义在`<linux/moduleparam.h>`中

* `MODULE_AUTHOR()`, `MODULE_DESCRIPTION()`, `MODULE_VERSION()`定义在`<linux/module.h>`里

    `<linux/module.h>`专门为内核模块提供接口：

    * 模块信息宏：MODULE_AUTHOR(), MODULE_DESCRIPTION(), MODULE_LICENSE(), MODULE_VERSION()

    * 模块入口/出口：module_init(), module_exit()

    * 模块依赖：MODULE_ALIAS(), MODULE_FIRMWARE() 等

    example:

    ```c
    #include <linux/module.h>   // MODULE_* 宏和 module_init/exit 在这里
    #include <linux/kernel.h>   // printk/pr_info, min/max 等

    MODULE_AUTHOR("Liucheng Hu");
    MODULE_DESCRIPTION("My first Linux kernel module");
    MODULE_VERSION("1.0");
    MODULE_LICENSE("GPL");

    static int __init my_module_init(void)
    {
        pr_info("Hello, kernel!\n");
        return 0;
    }

    static void __exit my_module_exit(void)
    {
        pr_info("Goodbye, kernel!\n");
    }

    module_init(my_module_init);
    module_exit(my_module_exit);
    ```

* `linux/kernel.h`的作用

    1. 常用宏定义

        包含很多内核开发里会用到的 通用宏，比如：

        * min(), max()

        * clamp(), roundup(), rounddown()

        * DIV_ROUND_UP() 等

        这些宏在写驱动或内核代码时非常常见。

    2. 打印函数

        提供了内核日志输出接口：

        * printk()

        * pr_info(), pr_err(), pr_warn() 等封装

        用来在内核日志（dmesg）里打印调试信息。

    3. 类型转换辅助

        一些和内核类型、数据处理相关的工具宏，比如：

        * container_of()（通过结构体成员指针得到整个结构体指针）。

    4. 内核通用函数声明

        包含一些在内核里常用的函数声明，例如：

        * simple_strtol(), simple_strtoul()（字符串转数字，已逐渐被 kstrto*() 系列替代）。

        * print_hex_dump()（调试时打印十六进制数据）。

* `param_get_charp()`

    从一个内核模块参数中获取其字符串类型的值（即 char 指针类型的值），并返回这个指针的副本。

    syntax:

    ```c
    int param_get_charp(char *buffer, const struct kernel_param *kp);
    ```

    return val:

    返回成功写入到 buffer 中的字节数。如果发生错误，则返回一个负的错误代码。

    这个函数其实是一个回调函数：

    ```c
    struct kernel_param_ops param_ops_charp = {
        .set  = param_set_charp,
        .get  = param_get_charp,
        .free = param_free_charp,
    };
    ```

    * `.set = param_set_charp()`（设置参数时分配内存）

    * `.get = param_get_charp()`（读取参数时拷贝字符串）

    * `.free = param_free_charp()`（释放内存）

    该函数不会直接返回模块参数变量本身所存储的指针，而是会返回这个指针的一个 “副本”。它会为参数所指向的原始字符串重新分配一块内核内存，并将字符串内容复制到这块新内存中。因此在使用完成后，需要调用配对函数`param_free_charp()`释放内存。此函数本质就是调用了`kfree()`。

    `param_get_charp()`这个函数由内核自动调用，不需要用户手动调用。当用户从 sysfs 里读取参数（比如 cat /sys/module/mymodule/parameters/my_param）时，内核会自动调用。

    用户只需要写`module_param()`就可以了：

    ```c
    static char *my_param = "default";
    module_param(my_param, charp, 0644);
    ```

* auxiliary bus

    （未验证）

    一个父设备（Physical Device）可以创建多个 auxiliary devices（辅助设备），但这些 auxiliary devices 都注册在同一个、内核全局唯一的 auxiliary bus（即 auxiliary_bus_type）上。

* 平台设备 (Platform Device)

    直接映射到系统地址空间、通常集成在处理器芯片内部或直接挂在本地总线上的设备。

    平台总线（platform_bus_type）是一个虚拟的总线，这个总线是内核创建的一个抽象，用于统一管理这些无法被自动发现的设备。

    CPU 无法通过扫描总线来发现它们。系统需要预先知道它们的存在。

    平台总线设备的详细信息（地址、中断号等）必须静态地提供给内核。提供方式主要有两种：

    1. 设备树 (Device Tree)：在现代嵌入式系统中（如 ARM、RISC-V），这是主流方式。Bootloader 会将一个描述硬件拓扑结构的设备树二进制文件（.dtb）传递给内核。内核解析后，会自动创建对应的平台设备。

    2. 板级文件 (Board File)：在旧的内核或x86系统中，通常在架构相关的C代码中硬编码 (platform_device_register())。

    platform device examples:

    * 处理器内部的UART控制器（串口）。

    * 系统内部的硬件定时器。

    * GPIO控制器。

    * 集成在SoC上的I2C控制器、SPI控制器本身（注意：这些控制器是平台设备，而挂在其上的从设备是I2C/SPI设备）。

    * 内存映射的LED或按键。

    平台设备的驱动匹配过程依赖于名称：

    1. 设备源（设备树或板级文件）会指定一个设备的名称（例如 "serial8250"）。

    2. 平台驱动会声明自己支持的设备名称（例如 "serial8250"）。

    3. 平台总线核心负责将同名设备和驱动进行匹配。

    4.  匹配成功后，调用驱动的 probe 函数。

* 总线设备 (Bus Device)

    所有通过某种可枚举、有标准探测协议的总线连接到系统的设备。

    总线设备的总线都是有形的，这些设备通过物理上真实存在的、有明确标准的总线连接到处理器（CPU）。

    example:

    * PCI/PCIe：网卡、显卡、声卡、存储控制器等。

    * USB：U盘、键盘、鼠标、摄像头等。

    * I2C：各种传感器（温度、湿度）、EEPROM 存储器等。

    * SPI：Flash 存储器、显示屏控制器等。

    系统（内核）可以在启动时或运行时（热插拔）通过总线协议主动去枚举发现这些设备。例如，PCI 总线可以通过配置空间读取设备的厂商ID和设备ID。

    总线设备的驱动匹配模型：

    1. 总线驱动枚举设备，并创建 struct device（或其子结构，如 struct pci_dev, struct usb_device）。

    2. 设备注册到总线上。

    3. 内核将设备的标识符（如 PCI 的 VID/DID）与已注册的驱动提供的标识符列表进行比对。

    4. 找到匹配的驱动后，调用驱动的 probe 函数来初始化设备。

* 设备类 (struct class)

    在 Linux 设备模型中，设备被分类管理，例如所有的输入设备（键盘、鼠标）属于 input 类，所有的块设备（硬盘、U盘）属于 block 类，所有的网络设备属于 net 类。

    每个类都是一个 struct class 结构体，它包含了一个链表，链接着所有注册到该类的设备（struct device）。

* `class_find_device()`

    在一个指定的设备类（struct class）中，根据提供的匹配条件，遍历并找到第一个匹配的设备

    syntax:

    ```c
    struct device *class_find_device(struct class *class,
                                     struct device *start,
                                     const void *data,
                                     int (*match)(struct device *, const void *));
    ```

    start: 从哪个设备开始遍历（通常传入 NULL 表示从链表头开始）。

    data: 传递给匹配函数的数据，用于比较（例如一个设备名称字符串、一个ID号等）。

    match: 你自定义的匹配函数指针。

    其中`match`为回调函数，当 dev 的数据匹配 data 时，返回 true (1)，否则返回 false (0).

    example:

    假设你有一个自定义的类 my_class，并且你知道某个设备的名称（例如 "my_special_device"），你想根据这个名字找到对应的设备指针。

    ```c
    static int my_match_by_name(struct device *dev, const void *data) {
        const char *name = data;
        // 检查设备的名字是否与传入的name相同
        return (dev->kobj.name != NULL) && (strcmp(dev->kobj.name, name) == 0);
    }

    struct device *dev;
    dev = class_find_device(my_class, NULL, "my_special_device", my_match_by_name);
    if (dev) {
        // 找到设备，可以进行操作
        // ...
        // 使用完后，通常需要调用 put_device() 来减少引用计数
        put_device(dev);
    } else {
        // 设备未找到
    }
    ```

    引用计数：class_find_device() 在找到设备后，会增加该设备的引用计数。这意味着你在使用完返回的设备指针后，必须调用 put_device() 来减少引用计数，否则该设备将永远无法被正确卸载，导致内存泄漏。

* `pcie_get_readrq()`

    获取指定 PCI Express (PCIe) 设备的最大读请求大小（Maximum Read Request Size, MRRS）。

    设备读取 host 内存时，单个读请求数据包所能请求的最大数据量。

    该函数通过读取设备 PCI 配置空间中 “PCI Express 能力结构” 的特定字段来获取这个值。

    syntax:

    ```c
    #include <linux/pci.h>

    int pcie_get_readrq(struct pci_dev *dev);
    ```

    返回值：

    成功时，返回设备当前配置的 最大读请求大小 (MRRS)，单位为字节。这是一个离散值，通常是以下之一：128, 256, 512, 1024, 2048, 4096。

	如果发生错误（例如设备不支持 PCIe 能力），函数可能返回一个错误码（负值），但通常实现会返回一个安全的最小值（如 128）。

    配对函数：`pcie_set_readrq()`

* `pcie_set_readrq()`

    内核进行检查和钳制（Clamping）：

        内核首先会检查请求的 size 值是否合法（是否是 128, 256, 512, 1024, 2048, 4096 中的一个）。

        最关键的一步：内核会查询 PCIe 设备的 “Device Capabilities” 寄存器。这个寄存器由硬件定义，明确说明了该设备自身支持的 MRRS 有哪些。

        内核将比较软件的请求值和硬件支持的能力值，并最终选择一个不大于请求值且不超过硬件支持上限的值。这个过程就是“钳制”（Clamping）。

        例如：

            你请求设置 4096。

            但设备能力寄存器显示其最大只支持到 1024。

            内核最终会将该设备设置为 1024，而不是 4096。

        内核将钳制后得到的最终值写入设备的 “Device Control” 寄存器。

* `vmalloc()`

    分配一块虚拟地址连续，物理地址不一定连续的内存。

    syntax:

    ```c
    #include <linux/vmalloc.h>

    void *vmalloc(unsigned long size);
    ```

    配对函数：`vfree()`

    初始值清零版本：`vzalloc()`

* `kvzalloc()`

    首先尝试像`kzalloc()`一样分配物理连续的内存。如果分配失败，它会自动回退到 `vzalloc()`的方式，分配虚拟连续的内存。并将内存内容初始化为 0.

    syntax:

    ```c
    #include <linux/slab.h>

    void *kzalloc(size, flags);
    ```

    配对函数：`kvfree()`

    感觉这个函数没啥用。什么情况需要优先使用连续物理内存，如果物理内存不连续也无所谓？想不到。

* kmalloc() 在内核空间分配的内存，如果在模块退出函数（module_exit）中没有被显式地使用 kfree() 释放，那么这些内存将永远地泄露，直到系统重启。

    `vmalloc()`同理。

* `device_destroy()`是`device_create()`的反函数，`device_del()`是`device_add()`的反函数。

* `device_add()`

    将设备添加到内核中，底层函数。

    syntax:

    ```c
    device_add(struct device *dev)
    ```

    功能：

    1. 绑定驱动：将设备（dev）与可能匹配的驱动程序进行绑定（通过调用驱动的 probe 函数）。

    2. 创建 sysfs 条目：在 sysfs 文件系统（通常是 /sys）中为该设备创建对应的目录和属性文件。

    3. 发送 uevent：向用户空间发送一个 “add” 事件的 uevent。用户空间的守护进程（如 udev 或 mdev）会监听这些事件，并根据规则文件（如 udev 规则）来创建设备节点（/dev/下的文件）或执行其他操作。

* `device_create()`

    添加设备的 high-level 函数，通常用于添加字符设备。

    syntax:

    ```c
    device_create(struct class *cls, struct device *parent, dev_t devt, void *drvdata, const char *fmt, ...);
    ```

    功能：

    1. 内部创建 device 结构体：它首先调用 device_create_vargs()，该函数内部会分配并初始化一个 struct device。

    2. 设置关键属性：它会将这个新设备关联到指定的类 (cls)、父设备 (parent) 和设备号 (devt)。

    3. 调用 device_add()：在内部，它最终会调用 device_add() 来完成将设备添加到内核的核心工作。

    4. 触发设备节点创建：因为它将设备关联到了一个类（cls），而该类在创建时（class_create()）已经注册了一个 dev_uevent 回调函数。这个回调函数会在 uevent 中提供 DEVTYPE、MAJOR、MINOR 等关键信息。用户空间的 udev 看到这些信息后，就会自动在 /dev/ 下创建设备节点。通常节点名就是类名加上次设备号（例如 /dev/myclass0）。

* `register_chrdev()`与`register_chrdev_region()`

    `register_chrdev()`的目的是兼容旧驱动，标记为 deprecated。

    `register_chrdev_region()`的功能是申请设备号，是现代推荐使用的函数，在内核 2.6 版本引入。

* `register_chrdev_region()`后，可以在 /sys/dev/char/ 或 /sys/devices/ 下创建清晰的设备结构。（未验证）

* `device_add()`与`device_create()`

    `device_add()`是一个基本函数，`device_create()`是一个便捷的封装函数，其中包含了对`device_add()`的调用。

    头文件：`<linux/device.h>`

    `device_add()`仅仅是将 dev 对象添加到内核中，`device_create()`除此之外还在`/dev`目录下创建设备节点。

    如果我们需要手动在`/dev`下创建节点，需要调用`device_create_file()`。

    `device_add()`需要外部传递一个`struct device`对象，而`device_create()`会在函数内部创建一个`struct device`对象。

* `pci_register_driver()`

    `pci_register_driver()`是一个宏，用于向内核注册 pci 设备驱动。

    syntax:

    ```c
    #include <linux/pci.h>

    int pci_register_driver(struct pci_driver *driver);
    ```

    `struct pci_driver`中比较重要的字段如下：

    * `.name`： 驱动程序的名称。

    * `.id_table`： 它指向一个`pci_device_id`数组，这个数组列出了该驱动程序所能支持的所有PCI设备的厂商ID（Vendor ID）和设备ID（Device ID）。用来进行设备匹配。

    * `.probe`: 用于驱动初始化

    * `.remove`: probe 的逆过程

    * `.shutdown`: 不知道干嘛用的

    返回值： 成功时返回 0；失败时返回一个负的错误代码（负数）。

    在注册过程中，内核的PCI子系统会立刻遍历当前系统中所有已发现的PCI设备。

    注册成功后，内核会在`/sys/bus/pci/drivers/`目录下创建一个以驱动程序命名的新目录（例如`/sys/bus/pci/drivers/e1000e/`）。

    其逆函数为`pci_unregister_driver()`.

* `kzalloc()`

    申请内存，并将内存置 0.

* `file_operations`中的`.owner`主要用于引用计数，防止内核模块在仍被进程使用（即其代码正在执行）时被意外卸载

    `struct module *owner`通常被设置为`THIS_MODULE`

    当进程打开一个设备文件时，内核内部会调用 try_module_get(module) 来尝试增加该模块的引用计数。

    当进程关闭设备文件时，内核会调用 module_put(module) 来减少模块的引用计数。

    当执行 rmmod 命令时，内核会检查目标模块的引用计数。如果计数大于 0（表示还有进程正在使用该模块提供的功能），卸载操作会失败并提示 Module XXX is in use。只有当引用计数为 0 时，卸载才会成功进行。

    linux 内核中有些 module 是永久存在的，不需要被卸载，为了区分哪些需要计数，哪些不需要，内核通过`struct module*`指针来判断。对于内置的 module，`.owner`为`NULL`。

* `module_param_array()`中的数组长度参数只有在 write 数据的时候才会被改变

* gcc 12 要求所有函数必须有声明，不然会报 warning:

    ```
    make -C /usr/src/linux-headers-6.8.0-40-generic M=/home/hlc/Documents/Projects/linked_list_test modules
    make[1]: Entering directory '/usr/src/linux-headers-6.8.0-40-generic'
    warning: the compiler differs from the one used to build the kernel
      The kernel was built by: x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
      You are using:           gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
      CC [M]  /home/hlc/Documents/Projects/linked_list_test/hello.o
    /home/hlc/Documents/Projects/linked_list_test/hello.c:4:5: warning: no previous prototype for ‘hello_init’ [-Wmissing-prototypes]
        4 | int hello_init(void)
          |     ^~~~~~~~~~
    /home/hlc/Documents/Projects/linked_list_test/hello.c:10:6: warning: no previous prototype for ‘hello_exit’ [-Wmissing-prototypes]
       10 | void hello_exit(void)
          |      ^~~~~~~~~~
      MODPOST /home/hlc/Documents/Projects/linked_list_test/Module.symvers
      LD [M]  /home/hlc/Documents/Projects/linked_list_test/hello.ko
      BTF [M] /home/hlc/Documents/Projects/linked_list_test/hello.ko
    Skipping BTF generation for /home/hlc/Documents/Projects/linked_list_test/hello.ko due to unavailability of vmlinux
    make[1]: Leaving directory '/usr/src/linux-headers-6.8.0-40-generic'
    ```

* 为什么写`MKDEV()`时可以填`MKDEV(255, 0)`, `MKDEV(220, 0)`

    怎么保证 255, 220 这些数字不与其他冲突？

    答：可以用`cat /proc/devices`查看已经注册过的设备

* `static int __init edu_init(void)`中的`__init`

    （未验证）

    作用是标记此函数只执行一次，后续可以将此函数占用的资源释放。

    编译器会把`__init`函数的代码段放到`.init.text`中，把`__initdata`的变量放到`.init.data`中。当函数执行完成后，操作系统会调用`free_initmem()`，释放这两个 init 段中的内存，释放出空间。

    `__exit`标记的函数，如果 module 被静态编译进了内核，那么类似`edu_exit(void)`的函数将永不会被调用，这些函数会被直接丢弃。

* `MODULE_DEVICE_TABLE()`功能（未验证）

    一个宏，将 id table 的设备 ID 与 driver 注册到全局信息中。

    depmod 会读取所有已安装模块中的 MODULE_DEVICE_TABLE 信息，提取出每个模块所支持的设备ID。depmod 将这些信息生成一个全局的“设备-模块”映射数据库文件，通常是 /lib/modules/$(uname -r)/modules.alias 和 modules.dep

    udev 会查询由 depmod 生成的 modules.alias 数据库，根据设备的标识符查找与之匹配的模块名称，然后调用 modprobe 命令来自动加载对应的内核模块。

    相当于如果没有`MODULE_DEVICE_TABLE()`，那么只能手动 insmod 或 modprobe 来加载驱动，不能让系统自动加载驱动。

* kmd 中`__FUNCTION__`表示当前函数的字符串，可以辅助打印 log 信息

    ```c
    pr_info("in %s()...\n", __FUNCTION__);
    ```

* simple pci driver example

    下面是加载 pci driver 的一个最简示例，probe 和 remove 都是空函数，无实际功能。

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/pci.h>

    static struct pci_device_id pci_id_table[] = {
        { PCI_DEVICE(0x1234, 0x11e8) },
        {0,}
    };

    static int edu_probe(struct pci_dev *device, const struct pci_device_id *device_id) {
    	pr_info("in edu_probe()...\n");
    	return 0;
    }

    static void edu_remove(struct pci_dev *device) {
        pr_info("in edu_remove()...\n");
    }

    struct pci_driver pci_driver = {
    	.name = "edu",
    	.id_table = pci_id_table,
    	.probe = edu_probe,
    	.remove = edu_remove
    };

    int hello_init(void) {
        pr_info("int hello_init()...\n");

        int ret = pci_register_driver(&pci_driver);
        if (ret != 0) {
            pr_err("fail to register pci driver\n");
            goto PCI_REGISTER_DRIVER_FAILED;
        }
        
        return 0;

    PCI_REGISTER_DRIVER_FAILED:
        return -1;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        pci_unregister_driver(&pci_driver);
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg:

    ```
    [  473.423543] int hello_init()...
    [  473.423621] in edu_probe()...
    [  481.947618] in hello_exit()...
    [  481.947709] in edu_remove()...
    ```

    其中`PCI_DEVICE(0x1234, 0x11e8)`比较重要，qemu edu device 的 pci 号就是`1234:11e8`。如果`PCI_DEVICE()`中 pci id 填写错误，那么将不会加载 pci 驱动，dmesg 不会有`in edu_probe()...`和`in edu_remove()...`的输出。

* 内核代码只能操作虚拟地址，不能直接使用物理地址。（未验证）

* 操作 pci 寄存器的 routine

    1. `pci_enable_device()`

    2. `pci_resource_start()`, `pci_resource_len()`, `pci_resource_flags()`

    3. `ioremap()`, `devm_ioremap()`

    4. 读写寄存器

    5. `iounmap()`, 禁用设备。

* `pci_enable_device()`作用（未验证）

    主要是对操作系统内核和 pci 设备进行一些配置，使得后续可以正常使用 pci 设备。

    1. 管理PCI电源状态

        将设备从可能的低功耗状态（例如 D3hot）切换到全功能状态（D0）

    2. 声明驱动对 pci 设备的所有权
    
        主要是设备的I/O端口和内存区域，防止和其他驱动程序冲突

    3. 启用总线主控（Bus Mastering）

        在设备的PCI配置空间中设置“Bus Master Enable”位。它允许 device 通过 dma 访问 host 上的内存。

    4. 启用内存和I/O空间访问

        在设备的PCI配置空间中设置 Memory Space Enable 和 I/O Space Enable 位

        解除了PCI总线对设备响应地址访问的封锁, CPU 可以通过读写设备的 BAR 所定义的地址范围来与设备通信

    syntax:

    ```c
    int pci_enable_device(struct pci_dev *dev);
    ```

    成功: 返回 0。

    失败: 返回一个负的错误代码（负整数）。常见的错误包括：

    * `-EIO`: 无法使能设备。

    * `-ENODEV`: 设备未在系统中找到或设备不支持使能操作。

    * `-EINVAL`: 无效参数。

    example:

    ```c
    static int my_driver_probe(struct pci_dev *pdev, const struct pci_device_id *ent)
    {
        int err;

        // 首先，启用PCI设备
        err = pci_enable_device(pdev);
        if (err) {
            dev_err(&pdev->dev, "Failed to enable device\n");
            return err;
        }

        // 启用设备后，才能安全地执行以下操作：
        // 1. 设置DMA掩码 (pci_set_dma_mask)
        // 2. 获取资源地址 (pci_resource_start)
        // 3. 映射内存区域 (pci_ioremap_bar)
        // 4. 申请IRQ中断 (pci_request_irq)
        // ... 其他设备初始化操作 ...

        return 0; // 成功
    }
    ```

    可以使用`pci_disable_device()`进行反向操作，禁用 pci dev。

* cdev 和`cdev_init()`, `cdev_add()`相关，涉及到 open, release, read, write, ioctl 等操作；device 和`class_create()`, `device_create()`相关，涉及到`/dev/xxxx`设备文件的创建。

    可以看出，cdev 和 device 本身没有依赖关系，它们通过设备号`dev_t dev_num`关联到一起。

* `ssize_t`是`long`

* kernel parameter array 元素个数越界与不足

    如果提供的元素个数超过数组 size，那么会报错：

    `int m_arr[3];`

    ```
    test@Ubuntu22:/sys/module/hello/parameters$ echo "1,2,3,4" | sudo tee ./m_arr 
    1,2,3,4
    tee: ./m_arr: Invalid argument
    ```

    如果提供的元素个数不足，那么剩余元素保持原来的值：

    ```
    test@Ubuntu22:/sys/module/hello/parameters$ echo "4,5" | sudo tee ./m_arr 
    4,5
    test@Ubuntu22:/sys/module/hello/parameters$ sudo cat m_arr 
    4,5,3
    ```

* kernel parameter 由于 user 是 root，所以实际上权限设置为`S_IRUSR | S_IWUSR`，普通用户仍无法读写

    ```c
    module_param(m_int, int, S_IRUSR | S_IWUSR);
    module_param(m_str, charp, S_IRUSR | S_IWUSR);
    module_param_array(m_arr, int, NULL, 0755);
    ```

    ```
    test@Ubuntu22:/sys/module/hello/parameters$ ls -lh
    total 0
    -rw-r--r-- 1 root root 4.0K  8月 19 14:48 m_arr
    -rw------- 1 root root 4.0K  8月 19 14:48 m_int
    -rw------- 1 root root 4.0K  8月 19 14:48 m_str
    ```

* `major = register_chrdev(0, "edu", &file_ops);`

    `register_chrdev`的第一个参数填 0 时，会自动分配一个 region 号。比如 edu 设备就分到了`241`号：

    `cat /proc/devices`:

    ```
    ...
    202 cpu/msr
    204 ttyMAX
    226 drm
    241 edu
    242 ttyDBC
    243 bsg
    ...
    ```

* mmio

    mmio - Memory-Mapped I/O, 内存映射 I/O

    可能有用的资料集合：

    * ioremap 映射地址

        ```c
        #include <linux/io.h>

        void __iomem *mmio_base;

        // 映射物理地址到内核虚拟地址
        mmio_base = ioremap(phys_addr, size);
        if (!mmio_base) {
            pr_err("Failed to ioremap MMIO region\n");
            return -ENOMEM;
        }
        ```

        phys_addr：设备的物理基地址。

        size：要映射的区域大小（字节）。

        __iomem：标记指针指向的是 MMIO 空间（避免直接访问）。

    * 读写 MMIO 寄存器

        ```c
        #include <linux/io.h>

        // 读取 32 位寄存器
        u32 val = readl(mmio_base + offset);

        // 写入 32 位寄存器
        writel(new_val, mmio_base + offset);
        ```

        位宽	读操作	写操作
        8-bit	readb()	writeb()
        16-bit	readw()	writew()
        32-bit	readl()	writel()
        64-bit	readq()	writeq()

    * 解除映射

        ```c
        iounmap(mmio_base);
        ```

    * example

        ```c
        #include <linux/module.h>
        #include <linux/pci.h>
        #include <linux/io.h>

        void __iomem *mmio_base;

        static int my_pci_probe(struct pci_dev *dev, const struct pci_device_id *id) {
            // 获取 BAR0 的物理地址和长度
            phys_addr_t bar0_phys = pci_resource_start(dev, 0);
            size_t bar0_size = pci_resource_len(dev, 0);

            // 映射 MMIO
            mmio_base = ioremap(bar0_phys, bar0_size);
            if (!mmio_base) {
                pr_err("Failed to ioremap BAR0\n");
                return -ENOMEM;
            }

            // 示例：读取第一个寄存器
            u32 reg_val = readl(mmio_base);
            pr_info("MMIO Register 0: 0x%08X\n", reg_val);

            return 0;
        }

        static void my_pci_remove(struct pci_dev *dev) {
            if (mmio_base) {
                iounmap(mmio_base);
            }
        }

        static struct pci_driver my_pci_driver = {
            .name = "my_pci_driver",
            .id_table = my_pci_ids,  // 需定义 PCI 设备 ID 表
            .probe = my_pci_probe,
            .remove = my_pci_remove,
        };

        module_pci_driver(my_pci_driver);
        ```

    * 如果寄存器访问需要严格顺序，使用 rmb() / wmb() / mb() 或 readl_relaxed() / writel_relaxed()（无屏障版本）

* `/sys/class/pci_bus/0000:00/device/0000:00:00.0`中每个 attr 文件的 size 都是 4096

    比如这个目录下的`class`，`device`等，虽然有效内容也就几十个字节，但是使用

    ```c
    FILE *f = fopen(path, "r");
    fseek(f, 0, SEEK_END);
    int len = ftell(f);
    ```

    得到的`len`，值为`4906`。

* 如果需要对不同设备，函数做出不同的行为，一种方法增加一个`enum`类型的函数参数，判断调用者的情况。另一种方法是增加一个编译宏，然后使用`#ifdef xxx`来检测，这样可以在编译时判断调用函数的主体的情况。

    为了只编译一份 lib 就适用多种情况，目前采用的是`enum`方案。

* `gfp_t`指的是需要 alloc 的内存类型

    常用的是`GFP_KERNEL`。其他的用法在文档里有详细说明。

    参见`kmalloc()`文档。

* grub 未启动 iommu，在 vfio 里 bind 的时候会报错

    ```
    hlc@zjxj:~/Documents$ sudo bash -c "echo 0000:4b:00.1 > /sys/bus/pci/drivers/vfio-pci/bind"
    bash: line 1: echo: write error: No such device
    ```

* 关于将`kmalloc()`申请的内存设置成无 cache 的方法

    ```c
    char *buf = kmalloc(4096, GFP_KERNEL);
    set_memory_uc(buf, 1);
    // do something...
    set_memory_wb(buf, 1);
    kfree(buf);
    ```

    可能是上面这个过程，但是并不知道有没有生效。

    `set_memory_uc()`和`set_memory_wb()`在头文件`<asm/set_memory.h>`中。

* `pgprot_writecombine()`, `pgprot_noncached()`, `msync()`

    不知道这几个函数干嘛用的。

* mmap 映射物理内存地址的一个 example

    代码可以参考`ref_25`。

    期望的输出：

    dmesg:

    ```
    [  732.698329] in init_mod()...
    [  745.638702] in hlc_open()...
    [  745.638706] in hlc_mmap()...
    [  745.638706] phy_addr: 10a3d7000
    [  745.638707] pfn: 1090519
    [  745.638708] vma->vm_start: 7aa729d3d000, vma->vm_end: 7aa729d3e000
    [  745.638708] len: 4096
    [  745.638713] successfully map physical addr to virtual addr
    [  745.638762] in hlc_ioctl()...
    [  745.638762] buf: hello from hlc

    [  745.638788] in hlc_release()...
    ```

    app:

    ```
    mapped addr: 0x7aa729d3d000
    buf: buffer from kernel

    ```

    可以看到，app 中读出来的数据，直接就是内核里提前填的内容。app 给内核发送 ioctl 后，内核读出的内容也是 app 写进去的内容。

    说明：

    * `kmalloc()`申请的内存至少需要 4096 字节，即一页，不然会发生内核错误，系统崩溃。

        `mmap()`与`remap_pfn_range()`中的 len 似乎对长度没要求。

    * `virt_to_phys()`在头文件`<asm/io.h>`中，`remap_pfn_range()`在头文件`<linux/mm.h>`中

    * 使用物理内存地址算出页表的时候，可以直接除以 4096 得到，用二进制的方式就是右移 12 位，这个位数也可以直接由宏`PAGE_SHIFT`得到。

    * pfn 指的是 page frame number，即页表帧编号，可以通过物理地址除以 4096 得到的

    * `mmap()`函数在头文件`<sys/mman.h>`中

    * `mmap()`的第一个参数指的是需要映射的 app 虚拟地址的起始地址，这个地址和 app 的虚拟地址空间有关，挺复杂的。实际使用的时候，我们只需要填`NULL`就可以了，让操作系统自动分配这个虚拟地址的起始地址。

    * mmap 的原理只是做地址转换，因此并没有 cache, buffer 之类的中间过程。

        具体是否涉及到 cache, buffer，需要在内核里`kmalloc()`后对申请的内存进行属性设置。

* 为什么`sys/ioctl.h>`里的`ioctl()`的第 3 个参数要被设计成`...`？

    ioctl 函数原型是

    ```c
    extern int ioctl (int __fd, unsigned long int __request, ...) __THROW;
    ```

    驱动里的 ioctl 的原型是

    ```c
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    ```

    第 3 个参数写明了是`unsigned long`，为什么与用户态不同？
    

* ioctl 是阻塞执行的

* linux driver 中的 read 返回的是已经处理的字节数

    如果返回的未处理的字节数大于 0，那么`cat`会无限循环调用 read 读取数据。

    如果 write 函数返回的字节数小于需要处理的字节数，那么`echo xxx > /dev/hlc_dev`也会无限循环调用 write 写入数据。

* sysfs 编程

    使用`kobject_create_and_add()`在`/sys`中创建文件夹，使用`sysfs_create_file()`创建文件。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/sysfs.h>

    #define MAX_BUF_SIZE 20
    char my_sysfs_node_buf[MAX_BUF_SIZE] = "hello, sysfs dev\n";

    ssize_t my_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
    {
        return sprintf(buf, my_sysfs_node_buf);
    }

    ssize_t my_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
    {
        return snprintf(my_sysfs_node_buf, MAX_BUF_SIZE, "%s", buf);
    }

    struct kobject *kobj;
    struct kobj_attribute kobj_attr;

    int init_mod(void)
    {
        kobj = kobject_create_and_add("hlc_kobj", NULL);
        if (!kobj)
        {
            printk(KERN_INFO "fail to create and add kobject\n");
            goto KOBjECT_CREATE_AND_ADD_ERROR;
        }
        pr_info("successfully create and add kobject\n");

        kobj_attr.attr.name = "my_kobj_node";
        kobj_attr.attr.mode = 0666;
        kobj_attr.show = my_show;
        kobj_attr.store = my_store;
        int ret = sysfs_create_file(kobj, &kobj_attr.attr);
        if (ret != 0)
        {
            printk(KERN_INFO "fail to create sysfs file");
            goto SYSFS_CREATE_FILE_ERROR;
        }
        pr_info("successfully create sysfs file\n");
        return 0;

    SYSFS_CREATE_FILE_ERROR:
        kobject_put(kobj);
    KOBjECT_CREATE_AND_ADD_ERROR:
        return -1;
    }

    void exit_mod(void)
    {
        sysfs_remove_file(kobj, &kobj_attr.attr);
        kobject_put(kobj);
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

    运行：

    ```bash
    sudo insmod my_sysfs.ko
    ```

    测试：

    1. 进入`/sys`目录，可以看到一个`hlc_kobj`文件夹

    2. `cd /sys/hlc_kobj`，可以看到里面有个`my_kobj_node`文件。

    3. 读写测试

        ```
        hlc@hlc-VirtualBox:/sys/hlc_kobj$ cat my_kobj_node 
        hello, sysfs dev
        hlc@hlc-VirtualBox:/sys/hlc_kobj$ echo "hello, world" > my_kobj_node 
        hlc@hlc-VirtualBox:/sys/hlc_kobj$ cat my_kobj_node 
        hello, world
        hlc@hlc-VirtualBox:/sys/hlc_kobj$ echo "01234567890123456789012345" > my_kobj_node 
        hlc@hlc-VirtualBox:/sys/hlc_kobj$ cat my_kobj_node 
        0123456789012345678hlc@hlc-VirtualBox:/sys/hlc_kobj$ 
        ```

    说明：

    * 上述代码中，`struct kobj_attribute kobj_attr;`不能写成局部变量，否则会丢失函数指针信息，导致调用`show()`或`store()`时失败。

    * 使用`kobject_create_and_add()`创建文件夹，使用`kobject_put()`删除文件夹。

        使用`sysfs_create_file()`创建文件，使用`sysfs_remove_file()`删除文件。

        注意创建目录、文件和删除目录、文件都要遵循递归关系，不然会报错。

    * 可以使用`__ATTR()`宏快速创建 attribute:

        ```c
        struct kobj_attribute kobj_attr = __ATTR(my_kobj_node, 0664, my_show, my_sotre);
        ```

        但是有几个需要注意的地方：

        1. 第一个参数是字符串，但是不需要加双引号

        2. 第二个参数是权限，但是其他用户不允许有写权限。这个宏会做检查。如果其他用户有写权限，不能通过编译。

        3. 必须在结构体初始化的时候调用，不允许这样写：

            ```c
            struct kobj_attribute kobj_attr;
            kobj_attr = __ATTR(my_kobj_node, 0664, my_show, my_sotre);
            ```

    * 如果需要创建嵌套目录，只需要将`kobject_create_and_add()`的第二个参数写成 parent object 的指针就可以了。

* 如果 module param 的权限是`S_IWUSR | S_IRUSR`，那么我们使用`sudo insmod <module_file>`加载模块时，`/sys/module/<module_name>/parameters`下的文件的 owner 和 group 都是 root

    此时`S_IWUSR | S_IRUSR`指的是 root:root 具有读写权限，其它用户没有任何权限。

* chrdev 指的是 char device，不是 children device

* `linux/cdev.h`跟 char device 驱动相关

    `linux/device.h`跟`/dev/`下的设备文件相关

* 直接使用`mknod`创建设备文件时，没有看到`/sys/class`下生成对应的文件夹，说明可以不需要 class 直接创建 device 文件

    有时间试一试把`device_create()`中的 class 设置为 NULL，看能不能成功创建 device 文件并正常使用。

    * [2024.06.29] 似乎不可以，如果将`device_create()`中的 class 设置为`NULL`，那么函数会成功执行，但是`/dev`中没有生成设备文件

* 使用自己创建的 work queue 的一个 example

    见`ref_18`。

    `dmesg` output:

    ```
    [ 5794.009888] aaaaa: loading out-of-tree module taints kernel.
    [ 5804.635333] in m_open()...
    [ 5804.635340] in m_read()...
    [ 5804.635345] in m_release()...
    [ 8209.385918] in init_mod()...
    [ 8230.417934] in m_open()...
    [ 8230.417941] in m_read()...
    [ 8230.417947] in m_release()...
    [ 8230.417988] in work func
    [ 9099.408811] in exit_()...
    ```

    说明：

    1. 使用`struct workqueue_struct *wque;`创建一个 work queue 指针。

    2. 使用`create_workqueue()`宏创建一个 queue 指针。
    
        `wque = create_workqueue("hlc_wque");`

        显然，work queue 的内存是由操作系统负责管理的，没有显式交给用户管理。

    3. `INIT_WORK(&witem, work_func);`初始化一个 work item，即`struct work_struct`对象。

    4. 使用`queue_work(wque, &witem);`将 work item 放到 work queue 中

        work item 放入后将会被执行。

    5. 使用`destroy_workqueue(wque);`销毁一个 work queue。

    6. `queue_work()`用于将 work item 放入自定义的 work queue 中，`schedule_work()`将 work item 放到 global shared work queue 中。

    7. work item 的回调函数原型：

        ```c
        void work_callback_func(struct work_struct *work_item)
        {
            // do something...
        }
        ```

        要求返回值是`void`，函数参数是`struct work_struct *`。

        不然`INIT_WORK()`宏会报错。

* `struct work_struct`

    header: `#include <linux/workqueue.h>`

* `__this_cpu_write()`

    header: `asm/hw_irq.h`

* 一个可用的`INIT_WORK()`的代码，见`ref_17`

    测试：

    ```bash
    make
    sudo insmod wque.ko
    sudo cat /dev/hlc_dev
    ```

    `dmesg` output:

    ```
    [25350.311799] init hlc module done.
    [25366.856255] in m_open()...
    [25366.856262] in m_read()...
    [25366.856265] in irq_handler()...
    [25366.856272] in m_release()...
    [25366.856312] in work_queue_fn()...
    [25414.842921] in exit_mod()...
    [25414.843190] exit hlc module done.
    ```

    explanation:

    1. 在创建`work_struct`对象的时候，需要我们自己申请内存，要么就直接创建全局变量，不能只创建一个指针。

        代码中使用`struct work_struct work_item;`创建了个全局对象。

        这一点和`class_create()`，`device_create()`挺不一样的，这两个函数都是只返回指针，内存由操作系统管理。

    2. `INIT_WORK()`需要将`work_struct`的指针传进去：
    
        `INIT_WORK(&work_item, work_queue_fn);`

    3. `schedule_work()`传的也是指针：

        `schedule_work(&work_item);`

    4. 这份代码不包含函数返回值检测和异常处理，所以比较简洁。

* work queue 的一个 example

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include<linux/slab.h>                 //kmalloc()
    #include<linux/uaccess.h>              //copy_to/from_user()
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/interrupt.h>
    #include <asm/io.h>
    #include <linux/workqueue.h>            // Required for workqueues
    #include <linux/err.h>
    #include <asm/hw_irq.h>

    #define IRQ_NO 11


    void workqueue_fn(struct work_struct *work); 

    /*Creating work by Static Method */
    DECLARE_WORK(workqueue,workqueue_fn);

    /*Workqueue Function*/
    void workqueue_fn(struct work_struct *work)
    {
        printk(KERN_INFO "Executing Workqueue Function\n");
    }


    //Interrupt handler for IRQ 11. 
    static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
        bool ret = schedule_work(&workqueue);
        if (!ret)
        {
            pr_info("fail to schedule work\n");
        }
        else
        {
            pr_info("successfully schedule work\n");
        }
            
        return IRQ_HANDLED;
    }


    volatile int etx_value = 0;


    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;

    /*
    ** Function Prototypes
    */
    static int __init etx_driver_init(void);
    static void __exit etx_driver_exit(void);

    /*************** Driver Fuctions **********************/
    static int etx_open(struct inode *inode, struct file *file);
    static int etx_release(struct inode *inode, struct file *file);
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len,loff_t * off);
    static ssize_t etx_write(struct file *filp, 
                    const char *buf, size_t len, loff_t * off);

    /*************** Sysfs Fuctions **********************/
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf);
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count);

    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
    };

    /*
    ** This function will be called when we read the sysfs file
    */ 
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
    }

    /*
    ** This function will be called when we write the sysfsfs file
    */
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
    }

    /*
    ** This function will be called when we open the Device file
    */  
    static int etx_open(struct inode *inode, struct file *file)
    {
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */  
    static int etx_release(struct inode *inode, struct file *file)
    {
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read function\n");
        struct irq_desc *desc;
        desc = irq_to_desc(11);
        if (!desc)
                return -EINVAL;
        __this_cpu_write(vector_irq[59], desc);
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
        printk(KERN_INFO "Write Function\n");
        return len;
    }

    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create("etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }

        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }

        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
            goto irq;
        }
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;

    irq:
        free_irq(IRQ_NO,(void *)(irq_handler));

    r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);

    r_device:
        class_destroy(dev_class);
    r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
    }

    /*
    ** Module exit function
    */

    static void __exit etx_driver_exit(void)
    {
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
    }

    module_init(etx_driver_init);
    module_exit(etx_driver_exit);

    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Static method)");
    MODULE_VERSION("1.10");
    ```

    需要的头文件：

    `#include <linux/workqueue.h>`

    ```c
    void workqueue_fn(struct work_struct *work); 
    DECLARE_WORK(workqueue,workqueue_fn);
    ```

    用宏创建一个变量`workqueue`，让其与一个函数产生关联。

    `schedule_work(&workqueue);`

    让`workqueue`对应的线程函数从睡眠状态唤醒，并放入主队列进行执行。

    `dmesg` output:

    ```
    [ 1789.389643] Major = 240 Minor = 0
    [ 1789.390326] Device Driver Insert...Done!!!
    [ 1802.603002] Device File Opened...!!!
    [ 1802.603029] Read function
    [ 1802.603040] Shared IRQ: Interrupt Occurred
    [ 1802.603048] successfully schedule work
    [ 1802.603058] Executing Workqueue Function
    [ 1802.603085] Device File Closed...!!!
    ```

    可以看到，可以通过 work queue，控制线程的休眠和唤醒。

    疑问：

    1. 当 work queue 对应的函数执行完毕时，是否线程会自动进入休眠？

        猜想：应该会自动进入休眠。不然函数执行完了也没有其他什么事可做。

        不过还有一种可能，即线程死亡，每次 schedule work 都重新创建一个新的线程。

        休眠-唤醒，和死亡-重建，哪个是对的？

    2. work queue 和 wait event 有什么不同？

        work queue 可以用`schedule_work()`唤醒（或新建）一个线程，wait event 可以使用`wake_up()`唤醒一个线程，这两都有什么不一样？

* `flush_work()`可以阻塞等待指定的 work，直到 work 完成。

    syntax:

    `int flush_work( struct work_struct *work );`

    `flush_scheduled_work()`可以等待全局共享的 work queue 完成。

    example: 见`ref_16`

    执行`cat /dev/etx_device`后，可以看到`dmesg`的输出：

    ```
    [ 6195.143524] Major = 240 Minor = 0 
    [ 6195.143595] Device Driver Insert...Done!!!
    [ 6214.544080] Device File Opened...!!!
    [ 6214.544095] Read function
    [ 6214.544101] Shared IRQ: Interrupt Occurred
    [ 6214.544104] successfully schedule work
    [ 6214.544105] block flush scheduled work
    [ 6214.544107] BUG: scheduling while atomic: cat/12786/0x00010001
    [ 6214.544110] Modules linked in: hello(OE) tls(E) intel_rapl_msr(E) intel_rapl_common(E) intel_uncore_frequency_common(E) snd_intel8x0(E) binfmt_misc(E) snd_ac97_codec(E) ac97_bus(E) crct10dif_pclmul(E) polyval_clmulni(E) polyval_generic(E) ghash_clmulni_intel(E) nls_iso8859_1(E) snd_pcm(E) sha256_ssse3(E) sha1_ssse3(E) aesni_intel(E) crypto_simd(E) cryptd(E) joydev(E) snd_seq_midi(E) rapl(E) snd_seq_midi_event(E) snd_rawmidi(E) input_leds(E) vmwgfx(E) snd_seq(E) serio_raw(E) drm_ttm_helper(E) snd_seq_device(E) snd_timer(E) snd(E) ttm(E) soundcore(E) drm_kms_helper(E) vboxguest(E) mac_hid(E) sch_fq_codel(E) msr(E) parport_pc(E) ppdev(E) drm(E) lp(E) parport(E) efi_pstore(E) ip_tables(E) x_tables(E) autofs4(E) hid_generic(E) usbhid(E) crc32_pclmul(E) hid(E) psmouse(E) ahci(E) libahci(E) i2c_piix4(E) e1000(E) pata_acpi(E) video(E) wmi(E) [last unloaded: hello(OE)]
    [ 6214.544151] CPU: 2 PID: 12786 Comm: cat Tainted: G        W  OE      6.5.13 #4
    [ 6214.544154] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
    [ 6214.544155] Call Trace:
    [ 6214.544157]  <IRQ>
    [ 6214.544159]  dump_stack_lvl+0x48/0x70
    [ 6214.544165]  dump_stack+0x10/0x20
    [ 6214.544166]  __schedule_bug+0x64/0x80
    [ 6214.544169]  __schedule+0x100c/0x15f0
    [ 6214.544174]  schedule+0x68/0x110
    [ 6214.544176]  schedule_timeout+0x151/0x160
    [ 6214.544181]  __wait_for_common+0x92/0x190
    [ 6214.544183]  ? __pfx_schedule_timeout+0x10/0x10
    [ 6214.544185]  wait_for_completion+0x24/0x40
    [ 6214.544188]  __flush_workqueue+0x133/0x3e0
    [ 6214.544190]  ? vprintk_default+0x1d/0x30
    [ 6214.544194]  irq_handler+0x55/0x80 [hello]
    [ 6214.544199]  __handle_irq_event_percpu+0x4f/0x1b0
    [ 6214.544201]  handle_irq_event+0x39/0x80
    [ 6214.544204]  handle_edge_irq+0x8c/0x250
    [ 6214.544207]  __common_interrupt+0x52/0x110
    [ 6214.544209]  common_interrupt+0x9f/0xb0
    [ 6214.544212]  </IRQ>
    [ 6214.544212]  <TASK>
    [ 6214.544213]  asm_common_interrupt+0x27/0x40
    [ 6214.544217] RIP: 0010:etx_read+0x2e/0x50 [hello]
    [ 6214.544221] Code: 00 55 48 c7 c7 db 12 99 c0 48 89 e5 e8 0b 90 08 d8 bf 0b 00 00 00 e8 61 fa 08 d8 48 85 c0 74 14 65 48 89 05 fc de 70 3f cd 3b <31> c0 5d 31 ff c3 cc cc cc cc 48 c7 c0 ea ff ff ff 5d 31 ff c3 cc
    [ 6214.544223] RSP: 0018:ffffacfb01fe7d98 EFLAGS: 00000286
    [ 6214.544225] RAX: ffff8a8b00205200 RBX: 0000000000020000 RCX: 0000000000000000
    [ 6214.544226] RDX: 0000000000000000 RSI: 0000000000000000 RDI: 0000000000000000
    [ 6214.544227] RBP: ffffacfb01fe7d98 R08: 0000000000000000 R09: 0000000000000000
    [ 6214.544228] R10: 0000000000000000 R11: 0000000000000000 R12: 0000000000000000
    [ 6214.544229] R13: ffff8a8a52457700 R14: ffffacfb01fe7e50 R15: 000077e41adfe000
    [ 6214.544232]  ? etx_read+0x1f/0x50 [hello]
    [ 6214.544235]  vfs_read+0xb4/0x320
    [ 6214.544238]  ? __handle_mm_fault+0xb88/0xc70
    [ 6214.544242]  ksys_read+0x67/0xf0
    [ 6214.544244]  __x64_sys_read+0x19/0x30
    [ 6214.544245]  x64_sys_call+0x192c/0x2570
    [ 6214.544248]  do_syscall_64+0x56/0x90
    [ 6214.544250]  ? exit_to_user_mode_prepare+0x39/0x190
    [ 6214.544253]  ? irqentry_exit_to_user_mode+0x17/0x20
    [ 6214.544255]  ? irqentry_exit+0x43/0x50
    [ 6214.544256]  ? exc_page_fault+0x95/0x1b0
    [ 6214.544259]  entry_SYSCALL_64_after_hwframe+0x73/0xdd
    [ 6214.544261] RIP: 0033:0x77e41ab147e2
    [ 6214.544263] Code: c0 e9 b2 fe ff ff 50 48 8d 3d 8a b4 0c 00 e8 a5 1d 02 00 0f 1f 44 00 00 f3 0f 1e fa 64 8b 04 25 18 00 00 00 85 c0 75 10 0f 05 <48> 3d 00 f0 ff ff 77 56 c3 0f 1f 44 00 00 48 83 ec 28 48 89 54 24
    [ 6214.544264] RSP: 002b:00007ffdc7545928 EFLAGS: 00000246 ORIG_RAX: 0000000000000000
    [ 6214.544266] RAX: ffffffffffffffda RBX: 0000000000020000 RCX: 000077e41ab147e2
    [ 6214.544267] RDX: 0000000000020000 RSI: 000077e41adfe000 RDI: 0000000000000003
    [ 6214.544268] RBP: 000077e41adfe000 R08: 000077e41adfd010 R09: 000077e41adfd010
    [ 6214.544269] R10: 0000000000000022 R11: 0000000000000246 R12: 0000000000022000
    [ 6214.544270] R13: 0000000000000003 R14: 0000000000020000 R15: 0000000000020000
    [ 6214.544272]  </TASK>
    [ 6214.544277] Executing Workqueue Function
    [ 6214.544278] i: 1
    [ 6214.544278] i: 2
    [ 6214.544279] i: 3
    [ 6214.544279] i: 4
    [ 6214.544279] i: 5
    [ 6214.544280] i: 6
    [ 6214.544280] i: 7
    [ 6214.544281] i: 8
    [ 6214.544281] i: 9
    [ 6214.544282] i: 10
    [ 6214.544287] work queue flushed!
    [ 6214.544304] cat[12786]: segfault at 77e41aa89210 ip 000077e41aa89210 sp 00007ffdc7545a68 error 14 in libc.so.6[77e41aa28000+195000] likely on CPU 2 (core 2, socket 0)
    [ 6214.544313] Code: Unable to access opcode bytes at 0x77e41aa891e6.
    [ 6214.544325] BUG: scheduling while atomic: cat/12786/0x7fff0001
    [ 6214.544326] Modules linked in: hello(OE) tls(E) intel_rapl_msr(E) intel_rapl_common(E) intel_uncore_frequency_common(E) snd_intel8x0(E) binfmt_misc(E) snd_ac97_codec(E) ac97_bus(E) crct10dif_pclmul(E) polyval_clmulni(E) polyval_generic(E) ghash_clmulni_intel(E) nls_iso8859_1(E) snd_pcm(E) sha256_ssse3(E) sha1_ssse3(E) aesni_intel(E) crypto_simd(E) cryptd(E) joydev(E) snd_seq_midi(E) rapl(E) snd_seq_midi_event(E) snd_rawmidi(E) input_leds(E) vmwgfx(E) snd_seq(E) serio_raw(E) drm_ttm_helper(E) snd_seq_device(E) snd_timer(E) snd(E) ttm(E) soundcore(E) drm_kms_helper(E) vboxguest(E) mac_hid(E) sch_fq_codel(E) msr(E) parport_pc(E) ppdev(E) drm(E) lp(E) parport(E) efi_pstore(E) ip_tables(E) x_tables(E) autofs4(E) hid_generic(E) usbhid(E) crc32_pclmul(E) hid(E) psmouse(E) ahci(E) libahci(E) i2c_piix4(E) e1000(E) pata_acpi(E) video(E) wmi(E) [last unloaded: hello(OE)]
    [ 6214.544347] CPU: 2 PID: 12786 Comm: cat Tainted: G        W  OE      6.5.13 #4
    [ 6214.544349] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
    [ 6214.544349] Call Trace:
    [ 6214.544350]  <TASK>
    [ 6214.544351]  dump_stack_lvl+0x48/0x70
    [ 6214.544353]  dump_stack+0x10/0x20
    [ 6214.544354]  __schedule_bug+0x64/0x80
    [ 6214.544355]  __schedule+0x100c/0x15f0
    [ 6214.544357]  schedule+0x68/0x110
    [ 6214.544359]  schedule_timeout+0x151/0x160
    [ 6214.544361]  __wait_for_common+0x92/0x190
    [ 6214.544363]  ? __pfx_schedule_timeout+0x10/0x10
    [ 6214.544364]  wait_for_completion_state+0x21/0x50
    [ 6214.544366]  call_usermodehelper_exec+0x188/0x1c0
    [ 6214.544370]  do_coredump+0xa35/0x1680
    [ 6214.544374]  ? do_dec_rlimit_put_ucounts+0x6b/0xd0
    [ 6214.544377]  get_signal+0x97b/0xae0
    [ 6214.544379]  arch_do_signal_or_restart+0x2f/0x270
    [ 6214.544381]  ? __bad_area_nosemaphore+0x147/0x2e0
    [ 6214.544384]  exit_to_user_mode_prepare+0x11b/0x190
    [ 6214.544386]  irqentry_exit_to_user_mode+0x9/0x20
    [ 6214.544387]  irqentry_exit+0x43/0x50
    [ 6214.544388]  exc_page_fault+0x95/0x1b0
    [ 6214.544390]  asm_exc_page_fault+0x27/0x30
    [ 6214.544392] RIP: 0033:0x77e41aa89210
    [ 6214.544395] Code: Unable to access opcode bytes at 0x77e41aa891e6.
    [ 6214.544395] RSP: 002b:00007ffdc7545a68 EFLAGS: 00010246
    [ 6214.544396] RAX: 000077e41ac1b868 RBX: 0000000000000000 RCX: 0000000000000004
    [ 6214.544397] RDX: 0000000000000001 RSI: 0000000000000000 RDI: 000077e41ac1b780
    [ 6214.544398] RBP: 000077e41ac1b780 R08: 000077e41adfd000 R09: 000077e41adfd010
    [ 6214.544399] R10: 0000000000000022 R11: 0000000000000246 R12: 000077e41ac1a838
    [ 6214.544400] R13: 0000000000000000 R14: 000077e41ac1bee8 R15: 000077e41ac1bf00
    [ 6214.544401]  </TASK>
    [ 6214.552103] Device File Closed...!!!
    ```

    虽然有一些报错输出，但是可以看到这一段还是按照顺序执行的：

    ```
    [ 6214.544277] Executing Workqueue Function
    [ 6214.544278] i: 1
    [ 6214.544278] i: 2
    [ 6214.544279] i: 3
    [ 6214.544279] i: 4
    [ 6214.544279] i: 5
    [ 6214.544280] i: 6
    [ 6214.544280] i: 7
    [ 6214.544281] i: 8
    [ 6214.544281] i: 9
    [ 6214.544282] i: 10
    [ 6214.544287] work queue flushed!
    ```

    由于从 1 到 10 没有中断，所以确实发生了等待。

    将代码中的`flush_scheduled_work();`換成`flush_work(&workqueue);`后，同样适用。

    ref:

    1. <http://juniorprincewang.github.io/2018/11/20/Linux%E8%AE%BE%E5%A4%87%E9%A9%B1%E5%8A%A8%E4%B9%8Bworkqueue/>

    2. <https://embetronicx.com/tutorials/linux/device-drivers/workqueue-in-linux-kernel/>

* 页表用于将虚拟地址映射到物理地址

    首先将物理地址按照 4 KB 进行分区，然后对每个小区间进行编号。

    目前的 linux 指针是 8 个字节，即 64 位。但是虚拟地址并不需要 64 位，只用到了其中的 48 位。

    其中 4 KB = 4 * 1024 Bytes = 2^2 * 2^10 Bytes = 2^12 Bytes，为了索引到每个字节，至少需要 12 位。

    因此虚拟地址就分为两部分，一部分是前半段`48 - 12 = 36`位用于定位页表的 entry，一部分是后半段`12`位偏移，用于在 4KB 的 entry 中定位到具体的字节。

    在 linux 中可以用命令`getconf PAGESIZE`查看页表每个 entry 的字节数。

    ref:

    1. <https://blog.csdn.net/m0_51717456/article/details/124256870>

    2. <https://www.cnblogs.com/chaozhu/p/10191575.html>

    3. <https://www.oreilly.com/library/view/linux-device-drivers/0596000081/ch07s04.html>

    4. <https://medium.com/@aravindchetla/kmalloc-v-s-vmalloc-13cb60746bcc>

    5. <https://www.kernel.org/doc/html/v5.0/core-api/mm-api.html#c.vmalloc>

    6. <https://www.kernel.org/doc/html/v5.0/core-api/memory-allocation.html>


* `vmalloc()`可以将不连续的物理地址通过链表的形式映射成连续的虚拟内存地址。

    头文件：`#include <linux/vmalloc.h>`

    ```c
    void *vaddr = vmalloc(1024);
    vfree(vaddr);
    ```

* `kthread_run()`

    创建并唤醒该线程。
    
    等价于先调用`kthread_create()`，再调用`wake_up_process()`唤醒线程。

    `kthread_run()`不是一个函数，而是一个宏。

    syntax:

    ```c
    kthread_run(threadfn, data, namefmt, ...)
    ```

* `wait_event()`

    `wait_event()`是一个宏。

    syntax: `wait_event(wq, condition)`

    休眠，直到`condition`为真，无法被手动打断。

    队列中的 wait queue 被标记为`TASK_UNINTERRUPTIBLE`。

* `wait_event()`和`wake_up()`传入的都是 wait queue head，即使 entry 加入到了 queue 里，也是处理 head。

    猜想： wait queue 在 wait 时，会将与之相关联的 thread 休眠。

    证据：`DECLARE_WAITQUEUE(wait_task,current);`将 wait entry 和 thread 相关联。

* `close()`函数在 unistd 中

* 一个带错误处理的 udev 驱动，见`ref_10`

    暂时先不引入 ioctl，目前没什么用

* linux driver 的 ioctl 原型是

    ```c
    long (*unlocked_ioctl) (struct file *filp, unsigned int cmd, unsigned long data);
    ```

    可以看到，其与 read, write 的根本区别是，它的参数里没有指针，所以不能传递太多信息，只能传递单个指令。

    2024/05/07/00: 第三个参数`unsigned long`可以被类型转换为指针传递数据，这样一来，其实`cmd`用于解释类型，`data`用于传递指针，可以做很多事情。

* `file_operations`不填 ioctl 回调函数也是可以的

* 内核中指针了一些宏，用于判断指针是否出错。

    ```c
    IS_ERR(指针)  // 返回真表示出错
    IS_ERR_OR_NULL(指针)  // 
    PTR_ERR(指针)  // 将出错的指针转换成错误码
    ERR_PTR(错误码)  // 将错误码转换成指针
    ```

* linux kernel module 中的 error 处理

    kernel module 中通常采用`goto`的方式处理 error，清理现场。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/fs.h>
    #include <linux/device.h>

    dev_t dev_region;
    const char *dev_region_name = "hlc dev region";
    struct class *dev_cls;
    struct device *hlc_dev;

    int mod_init(void)
    {
        printk(KERN_INFO "in mod_init() ...\n");

        alloc_chrdev_region(&dev_region, 0, 1, dev_region_name);
        printk(KERN_INFO "allocated device region.\n");

        dev_cls = class_create("hlc dev cls");
        if (IS_ERR(dev_cls)) {
            printk(KERN_INFO "fail to create dev class.\n");
            goto r_class_create;
        }
        printk(KERN_INFO "created device class.\n");

        hlc_dev = device_create(dev_cls, NULL, dev_region, NULL, "hlc_dev");
        if (IS_ERR(hlc_dev)) {
            printk(KERN_INFO "fail to create device.\n");
            goto r_device_create;
        }
        printk(KERN_INFO "created device.\n");
        return 0;

    r_device_create:
        printk(KERN_INFO "clean device class...\n");
        class_destroy(dev_cls);
    r_class_create:
        printk(KERN_INFO "clean device region...\n");
        unregister_chrdev_region(dev_region, 1);
        return -1;
    }

    void mod_exit(void)
    {
        printk(KERN_INFO "in mod_exit() ...\n");
        device_destroy(dev_cls, dev_region);
        class_destroy(dev_cls);
        unregister_chrdev_region(dev_region, 1);
        printk(KERN_INFO "unregistered device region.\n");
    }

    module_init(mod_init);
    module_exit(mod_exit);
    MODULE_LICENSE("GPL");
    ```

    当`class_create()`失败时，会调用`goto r_class_create;`跳转到`r_class_create`处，执行`unregister_chrdev_region()`清理在调用`class_create()`之前申请的资源。

    `goto r_device_create;`也是同理。

    这样的写法把一个函数分隔成了栈的结构，可以很方便地选择性清理现场：

    ```c
    // step 1
    if (出错) {
        goto 标签1；
    }

    // step 2
    if（出错） {
        goto 标签2；
    }

    // step 3
    if (出错) {
        goto 标签3;
    }

    标签3:
        复原第2步
    标签2:
        复原第1步
    标签1:
        return 错误码;
    ```


    当`mod_init()`返回`-1`时，命令`insmod`会报错：

    ```
    hlc@virt-Ubuntu-2204:~/Documents/Projects/driver_test$ sudo insmod hello.ko
    insmod: ERROR: could not insert module hello.ko: Operation not permitted
    ```

    此时`mod_exit()`函数不会被执行，但是`mod_init()`函数中的内容会被执行。

* 可以在还没创建 cdev 驱动时就创建 udev 设备文件

    可以在`/dev`中看到新创建的设备文件，但是此时`cat`会报错：

    ```
    hlc@virt-Ubuntu-2204:/dev$ sudo bash -c "cat /dev/hlc_dev"
    cat: /dev/hlc_dev: No such device or address
    ```

* cdev 的 ops 函数原型中，`read`，`write`，`unloacked_ioctl`的返回值类型都是`ssize_t`，对应的是`long`。

* `pci_set_drvdata()`与`pci_get_drvdata()`用于获取/设置设备驱动私有数据

    syntax:

    ```c
    void *pci_get_drvdata(struct pci_dev *pdev);
    void pci_set_drvdata(struct pci_dev *pdev, void *data);
    ```

    It is a convenient way for example to save a pointer to a local dynamically allocated device context in the device probe callback and then retrieve it back with pci_get_drvdata in the device remove callback and do a proper cleanup of the context.

* `ktime_get_seconds()`可以获得系统启动后过去了多少时间

    `ktime_get_real_seconds()`可以获得 utc 时间，但是需要其他库/函数转换成人类可读时间。

    与时间相关的函数都在`linux/timekeeping.h`头文件中。

    如果需要 formatted output time，可以参考这篇：<https://www.kernel.org/doc/html/latest/core-api/printk-formats.html#time-and-date>

    与时间相关的函数与简要说明：<https://www.kernel.org/doc/html/latest/core-api/timekeeping.html>

    ref: <https://stackoverflow.com/questions/55566038/how-can-i-print-current-time-in-kernel>

* 对于 parameter 数组，在`cat`的时候，可以看到它是以`,`分隔的一些数字

    ```bash
    sudo bash -c "cat m_vec"
    ```

    output:

    ```
    1,2,3
    ```

    如果写入的数据多于数组的容量，会报错：

    ```bash
    sudo bash -c "echo 2,3,4,5,6 > m_vec"
    ```
    
    ```
    bash: line 1: echo: write error: Invalid argument
    ```

    导致写入失败。

    如果写入的数据少于数组的容量，则会自动在指针中写入具体有几个元素。


* 读取与写入 kernel module parameter 时，需要 root 权限的解决办法

    ```bash
    sudo bash -c "cat param_name"
    sudo bash -c "echo some_val > param_name"
    ```

## Topics

### platform

* `platform_get_resource()`

    在 Linux 设备模型中，那些直接连接在处理器总线上的、相对简单的设备（如 GPIO 控制器、I2C 控制器、内存映射的设备等）通常被抽象为“平台设备”（platform_device）。

    一个设备要工作，需要内核知道它的“资源”，比如：

    * 内存地址范围（IORESOURCE_MEM）：设备寄存器映射到的物理地址和长度。

    * 中断号（IORESOURCE_IRQ）：设备使用的中断线编号。

    * DMA 通道（IORESOURCE_DMA）：设备使用的 DMA 通道号。

    这些资源信息通常在设备树（Device Tree）或ACPI表中定义，在系统启动时由内核解析并填充到对应的 platform_device 结构体中。

    syntax:

    ```c
    struct resource *platform_get_resource(struct platform_device *pdev,
                                          unsigned int type,
                                          unsigned int num);
    ```

    * `pdev`: 指向对应的平台设备结构体的指针，通常会在驱动的 probe 函数中传入。

    * `type`: 要查找的资源类型。常见的有：

        * `IORESOURCE_MEM` - 内存资源

        * `IORESOURCE_IRQ` - 中断资源

        * `IORESOURCE_DMA` - DMA资源

    * `num`: 该类型资源的索引号（从 0 开始）。例如，一个设备可能有两块内存映射区域，第一块索引为 0，第二块索引为 1。

    返回值:

    * 成功：返回指向 struct resource 的指针。

    * 失败或指定的资源不存在：返回 NULL。

    example:

    ```c
    static int my_driver_probe(struct platform_device *pdev)
    {
        struct resource *res;

        // 1. 获取第一个内存资源（索引0）
        res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
        if (!res) {
            dev_err(&pdev->dev, "Failed to get MEM resource\n");
            return -EINVAL;
        }
        // 使用 res->start 和 res->end 获取地址范围

        // 2. 获取第一个中断资源（索引0）
        res = platform_get_resource(pdev, IORESOURCE_IRQ, 0);
        if (!res) {
            dev_err(&pdev->dev, "Failed to get IRQ resource\n");
            return -EINVAL;
        }
        int irq_num = res->start; // 中断号通常存放在 start 字段
        // 然后使用 request_irq() 申请这个中断

        // ... 其他初始化操作 ...
        return 0;
    }
    ```

    如果是中断，可以直接使用`platform_get_irq()`

* `platform_get_irq()`

    syntax:

    ```c
    platform_get_irq(struct platform_device *pdev, unsigned int num)
    ```

    从 platform 设备资源 中获取第 num 个中断号（IRQ number），供驱动程序后续调用 request_irq() 等接口使用。

    如果失败，会返回负数错误码（如 -ENXIO, -EINVAL 等）。

    example:

    ```c
    int irq;
    irq = platform_get_irq(pdev, 0);
    if (irq < 0)
        return irq;
    ret = devm_request_irq(&pdev->dev, irq, my_irq_handler, 0,
                           dev_name(&pdev->dev), dev);
    ```

### print and log

* `pr_info()`无论写不写`\n`都会换行，但是只有写`\n`才会刷新缓冲区

    也就是说，如果只写`pr_info("hello")`，那么`"hello"`可能会放在缓冲区里，不输出到屏幕上。

* 内核中日志等级的定义

    ```c
    #include <linux/kern_levels.h>

    #define KERN_SOH    "\001"  /* ASCII Start Of Header */
    #define KERN_SOH_ASCII  '\001'

    #define KERN_EMERG      KERN_SOH "0" /* system is unusable */
    #define KERN_ALERT      KERN_SOH "1" /* action must be taken immediately */ 
    #define KERN_CRIT       KERN_SOH "2" /* critical conditions */
    #define KERN_ERR        KERN_SOH "3" /* error conditions */
    #define KERN_WARNING    KERN_SOH "4" /* warning conditions */
    #define KERN_NOTICE     KERN_SOH "5" /* normal but significant condition */
    #define KERN_INFO       KERN_SOH "6" /* informational */
    #define KERN_DEBUG      KERN_SOH "7" /* debug-level messages */
    ```

    所以下面四种写法都是等价的：

    ```c
    pr_info("hello world\n");
    printk(KERN_INFO "hello, world\n");
    printk("\0016" "hello, world\n");
    printk("\0016hello, world\n");
    ```

    `printk("<6>" "hello, world\n");`这种写法目前已被淘汰，内核不支持。

* `pr_err_once()`

    如果多次调用到这个函数，那么它只输出一次，防止 dmesg 刷屏。

    其内部有一个 bool 标志位，标记此函数之前是否被执行。

    * `pr_emerg_once()`
        
        KERN_EMERG 系统可能即将崩溃

    * `pr_alert_once()`
        
        KERN_ALERT 需要立即行动

    * `pr_crit_once()`
    
        KERN_CRIT 临界条件

    * `pr_err_once()`
        
        KERN_ERR 错误条件

    * `pr_warn_once()`
        
        KERN_WARNING 警告条件

    * `pr_notice_once()`
    
        KERN_NOTICE 正常但重要的事件

    * `pr_info_once()`
    
        KERN_INFO 提示信息

    * `pr_debug_once()`
    
        KERN_DEBUG 调试信息（依赖配置）

* `fprintf(stdin, ...)`和`fscanf(stdout, ...)`一样，都是未定义行为。可能会导致程序崩溃。

* `dev_info()`

    输出一条附加了设备信息（如设备名称、地址等）的提示性消息.

    syntax:

    ```c
    #include <linux/device.h>

    int dev_info(const struct device *dev, const char *fmt, ...);
    ```

    它不仅仅打印用户提供的格式化字符串，还会自动前缀与设备相关的信息。

    对于 PCI 设备 (struct pci_dev *)，前缀通常是 [设备驱动名] 0000:03:00.0: 。

    对于 USB 设备 (struct usb_device *)，前缀可能包含制造商和产品信息。

    对于平台设备 (struct device *)，通常是设备树节点名或平台设备名。

    同类函数：

    dev_emerg()	KERN_EMERG	系统不可用，紧急消息
    dev_alert()	KERN_ALERT	需要立即采取行动
    dev_crit()	KERN_CRIT	临界状态，严重硬件错误
    dev_err()	KERN_ERR	错误状态，操作失败
    dev_warn()	KERN_WARNING	警告信息，可能有问题
    dev_info()	KERN_INFO	信息性消息，正常状态（最常用）
    dev_dbg()	KERN_DEBUG	调试消息，默认不打印，需开启动态调试

* `printk()`中指针地址的打印

    `%p`打印的并不是真实地址，而是经过处理的地址

    `%px`打印的是原始地址值，不经过处理。

    `%pK`是按配置文件打印值，更具体的用法可以参考这里：<https://blog.csdn.net/zqwone/article/details/127057245>

    <https://www.kernel.org/doc/Documentation/printk-formats.txt>

* 在使用`printk()`的`KERN_INFO`模式时，`dmesg`中显示的 log，第一个冒号之前的字体都为黄色，冒号以及之后的字体都是普通白色。

    如果字符串没有冒号，那么全部 log 都是白色。

### 开发环境

* 写 linux module 时，vscode 的 cpp 配置里，`KBUILD_MODNAME`要写成`KBUILD_MODNAME="hello"`，以前记的笔记是`KBUILD_MODNAME=\"hello\"`，似乎是不对的。

* insmod 时报错

    在`insmod`时报错`insmod: ERROR: could not insert module dkms_test.ko: Invalid module format`。
    
    经检查，`uname -r`查看的 kernel 版本与编译时 Makefile 里指定的 kernel 的版本相同，又查看`/boot`目录，`ls -lh`看到`initrd.img-xxxx`,`vmlinuz-xxxx`, `System.map-xxx`这三个文件的最后一次修改的日期都比较旧，说明最近没有被替换。

    最终发现是 gcc 的版本变了，默认版本的`gcc`在几分钟之内从`gcc-11`升级到了`gcc-12`。很有可能当前内核是`gcc-11`编译的，而编译新 ko 时，使用了`gcc-12`，导致版本不一致。

* 似乎在安装`apt install build-essential`的时候，就会安装 kernel 相关的 herders 和预编译库

* 在 insmod 时报错`module verification failed: signature and/or required key missing - tainting kernel`

    可以直接在 makefile 开头添加一行：`CONFIG_MODULE_SIG=n`解决。

    虽然在 insmod 时还会有提示，但是可以正常加载驱动。

    更完善的解决办法可以参考这个：<https://stackoverflow.com/questions/24975377/kvm-module-verification-failed-signature-and-or-required-key-missing-taintin>

    如果 kernel 不是 singed 的，那么也可以不用加`CONFIG_MODULE_SIG=n`这一行。

* kernel module 编译时出现 undefine symbol 是因为没有 export symbol

    ref: <https://blog.csdn.net/choumin/article/details/127094429>

* kbuild　添加自定义的 .o　文件

    ```makefile
    obj-m += haha.o
    haha-src := my_proc.c
    haha-objs := my_proc.o relative/path/to/hehe.o
    ```

    ref: <https://stackoverflow.com/questions/22150812/linking-kernel-module-with-a-static-lib>

    注意，在`xxx-objs`中使用的路径，都是相对于当前目录的路径。

* kbuild doc

    <https://docs.kernel.org/kbuild/makefiles.html>

* kbuild add extra flags to compiler

    <https://stackoverflow.com/questions/54118602/how-to-set-preprocessor-directives-in-makefile-for-kernel-module-build-target>

* vscode 中有时 tab 会变成 8 个空格，可以关闭这个设置：

    `Editor: Detect Indentation`

* kbuild system 相关

    * the most simple kbuild makefile

        ```makefile
        obj-y += foo.o
        ```

        This tells kbuild that there is one object in that directory, named `foo.o`. `foo.o` will be built from `foo.c` or `foo.S`.

    * build a module

        if `foo.o` shall be built as a module, the variable `obj-m` is used.

        ```makefile
        obj-m += foo.o
        ```

        Therefore the following pattern is often used:

        ```makefile
        obj-$(CONFIG_FOO) += foo.o
        ```

        `$(CONFIG_FOO)` evaluates to either `y` (for built-in) or `m` (for module).

        If `CONFIG_FOO` is neither `y` nor `m`, then the file will not be compiled nor linked.

    * The kbuild Makefile specified object files for vmlinux in the `$(obj-y)` lists.

* linux 6.8.0 vscode hello world 无静态报错的配置是这样的：

    ```json
    {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/**",
                    "/usr/src/linux-headers-6.8.0-49-generic/include",
                    "/usr/src/linux-headers-6.8.0-49-generic/arch/x86/include",
                    "/usr/src/linux-headers-6.8.0-49-generic/arch/x86/include/generated"
                ],
                "compilerPath": "/usr/bin/clang-11",
                "cStandard": "c17",
                "cppStandard": "c++14",
                "intelliSenseMode": "linux-clang-x64",
                "defines": [
                    "__KERNEL__",
                    "MODULE"
                ]
            }
        ],
        "version": 4
    }
    ```

    看起来是没有了 hwe 文件夹。

    `__KERNEL__`对应的宏是`module_init()`, `module_exit`, `MODULE_LICENSE()`。

    `MODULE`对应的宏是`MODULE_LICENSE()`。

    hello world 程序：

    ```c
    #include <linux/init.h>
    #include <linux/module.h>

    int init_mod(void)
    {
        printk("hello hlc module\n");
        return 0;
    }

    void exit_mod(void)
    {
        printk("exit hlc module\n");
        return;
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

* ubuntu 24.04 + 6.8.1 kernel 编译 hello world module 相对以前的变化

    * vscode 的 c/c++ json 配置如下

        ```json
        {
            "configurations": [
                {
                    "name": "Linux",
                    "includePath": [
                        "${workspaceFolder}/**",
                        "/usr/src/linux-source-6.8.0/include",
                        "/usr/src/linux-source-6.8.0/arch/x86/include/generated",
                        "/usr/src/linux-source-6.8.0/arch/x86/include"
                    ],
                    "defines": [
                        "__KERNEL__",
                        "MODULE"
                    ],
                    "compilerPath": "/usr/bin/gcc",
                    "cStandard": "c17",
                    "cppStandard": "gnu++17",
                    "intelliSenseMode": "linux-gcc-x64"
                }
            ],
            "version": 4
        }
        ```

        可以看到，好像没有 hwe 相关的文件夹了，另外 arch 文件夹使用的是`x86`，不是`x86_64`。

    * `init_module()`函数在`linux/module.h`中被定义了，但是`exit_module()`没有被定义。

### pci capability

* `pci_find_capability()`

    syntax:

    ```c
    #include <linux/pci.h>

    int pci_find_capability(struct pci_dev *dev, int cap);
    ```

    * `int cap_id`：要查找的能力类型的标识符（一个字节的 ID，如 0x10 代表 PCIe）。

    函数会从 PCI 配置空间的能力列表指针（Capabilities Pointer register，偏移量 `0x34`）开始，遍历整个能力链表。

    返回值：

    成功：如果找到了与 cap_id 匹配的能力项，则返回该能力结构在 PCI 配置空间中的偏移地址（例如，0x100）。驱动程序可以利用这个地址来读取或写入该能力结构中的具体寄存器（如配置 MSI 中断向量数、地址和数据）。

    失败：如果遍历完整个链表都没有找到指定的能力，或者设备根本不支持能力列表，则返回 0。

* pci capabilities list

    每个 node 代表一个 capabiiility，node 的大小并不固定，只有前两个字节是固定的。

    字节 0：Capability ID - 唯一标识能力的类型（如 0x01 是电源管理，0x10 是 PCIe）。

    字节 1：Next Capability Pointer - 指向下一个能力结构在配置空间中的偏移地址。这个指针将所有的能力节点链接在一起，形成链表。

    从字节 2 开始，驱动程序通过查询 Capability ID 来确定该如何解析后续的字节。详细的格式由 pci sig 维护。程序可以通过`pci_read_config_byte()`, `pci_read_config_word()`, `pci_read_config_dword()`来读取。开发者应该尽量使用更高级的接口，不要直接读取这些字节。

    常见的 pci capability id 与其对应的 struct:

    * `PCI_CAP_ID_MSI` (`0x05`): 消息信号中断

        MSI 结构有两种常见形式：32位和64位地址格式。

        * 最小形式 (32位地址, 1个向量)：

            偏移 0x00: Capability ID (0x05) + Next Pointer

            偏移 0x02: Message Control Register

            偏移 0x04: Message Address Register (低32位)

            偏移 0x08: Message Data Register

            总长度: 10 字节 (从链表指针开始算起的结构体大小)

        * 扩展形式 (64位地址, 多个向量)：

            包含最小形式的所有寄存器...

            偏移 0x08: Message Address Register (高32位)

            偏移 0x0C: Message Data Register

            偏移 0x10: Mask Bits Register (可选)

            偏移 0x14: Pending Bits Register (可选)

            总长度: 可达 24 字节或更多

    * `PCI_CAP_ID_MSIX` (`0x11`): 消息信号中断

    * `PCI_CAP_ID_PM` (`0x01`): 电源管理

    * `PCI_CAP_ID_EXP` (`0x10`): PCIe 特性

        PCI Express Capabilities Register

        Device Capabilities Register

        Device Status and Control Register

        Link Capabilities Register

        Link Status and Control Register

        Slot Capabilities Register (如果适用)

        ...

        总长度: 通常至少是 20 字节（对于端点设备），对于根端口或交换设备会更长

    * `PCI_CAP_ID_VNDR` (`0x09`): 虚拟通道

* 完整的 pci capability list

    可参考`uapi/linux/pci_regs.h`。

    | 常量 | 值 | 描述 |
    | - | - | - |
    | PCI_CAP_ID_PM | 0x01 | 电源管理 (Power Management) |
    | PCI_CAP_ID_AGP | 0x02 | 加速图形端口 (Accelerated Graphics Port) |
    | PCI_CAP_ID_VPD | 0x03 | 重要产品数据 (Vital Product Data) |
    | PCI_CAP_ID_SLOTID | 0x04 | 插槽识别 (Slot Identification) |
    | PCI_CAP_ID_MSI | 0x05 | 消息信号中断 (Message Signaled Interrupts) |
    | PCI_CAP_ID_CHSWP | 0x06 | 热插拔 (CompactPCI Hot-Swap) |
    | PCI_CAP_ID_PCIX | 0x07 | PCI-X |
    | PCI_CAP_ID_HT | 0x08 | HyperTransport |
    | PCI_CAP_ID_VNDR | 0x09 | 厂商特定信息 (Vendor-Specific) |
    | PCI_CAP_ID_DBG | 0x0A | 调试端口 (Debug port) |
    | PCI_CAP_ID_CCRC | 0x0B | 紧凑型PCI中央资源控制 (CompactPCI CRC) |
    | PCI_CAP_ID_SHPC | 0x0C | 标准热插拔控制器 (Standard Hot-Plug Controller) |
    | PCI_CAP_ID_SSVID | 0x0D | 子系统厂商ID (Subsystem Vendor/Device ID) |
    | PCI_CAP_ID_AGP3 | 0x0E | AGP 8x |
    | PCI_CAP_ID_SECDEV | 0x0F | 安全设备 (Secure Device) |
    | PCI_CAP_ID_EXP | 0x10 | PCI Express (这是最常用的之一) |
    | PCI_CAP_ID_MSIX | 0x11 | MSI-X 中断 (这是最常用的之一) |
    | PCI_CAP_ID_SATA | 0x12 | SATA 数据/配置索引 |
    | PCI_CAP_ID_AF | 0x13 | 高级功能 (Advanced Features) |

### dma

* AXI-DMA

    AXI（Advanced eXtensible Interface）

    axi 是 ARM 提出的片上互连协议，用于 fpga 和 soc。比如 FPGA 与处理器（如 ARM Cortex）之间的数据搬运。

* PCI-DMA

    PCI/PCIe（Peripheral Component Interconnect Express）

    需要 低延迟、芯片内通信 → AXI-DMA。

    需要 高带宽、跨设备通信 → PCI-DMA。

* `dma_set_mask_and_coherent()`

    设置 DMA 掩码。检查并告知内核：当前设备（通常是 PCIe、USB 等外设）能够访问的系统物理地址范围。确保 DMA 操作的安全性和正确性：防止设备尝试访问超出其寻址能力的物理地址，从而导致数据损坏或系统崩溃。

    并非所有硬件设备都支持 64 位物理地址寻址。一些较老或成本较低的设备可能只支持 32 位（即 4GB）甚至更小的地址空间。

    syntax:

    ```c
    int dma_set_mask_and_coherent(struct device *dev, u64 mask);
    ```

    * mask: 一个位掩码，表示设备支持的地址位。例如，0xFFFFFFFF 表示 32 位掩码（支持 4GB 以下地址），`DMA_BIT_MASK(64)` 表示 64 位掩码（支持全部 64 位地址）。

    返回值：

    * 成功时返回 0。

    * 失败时返回非零值（通常是`-EIO`），表示平台无法在该掩码下支持 DMA。例如，在一个不支持 64 位 DMA 的系统上尝试设置 64 位掩码可能会失败。

    example:

    ```c
    struct device *dev = &my_pci_dev->dev;
    u64 dma_mask = DMA_BIT_MASK(64); // 假设我们的设备支持 64 位 DMA

    if (dma_set_mask_and_coherent(dev, dma_mask)) {
        // 64 位 DMA 设置失败，尝试回退到 32 位
        dev_warn(dev, "64-bit DMA not supported, trying 32-bit\n");
        if (dma_set_mask_and_coherent(dev, DMA_BIT_MASK(32))) {
            dev_err(dev, "No usable DMA configuration found\n");
            return -EIO;
        }
    }
    ```

    dma_set_mask_and_coherent() 实际上一次性设置了两个掩码：

    * 流式 DMA 掩码 (DMA Mask)：

        用于“流式” DMA 映射（dma_map_single 等）。

        这种映射通常是短期、一次性的，缓存一致性通常由软件显式维护（如手动刷缓存）。

        内核会确保为流式 DMA 分配的内存地址落在设备声明的这个地址范围内。

    * 一致性 DMA 掩码 (Coherent DMA Mask)：

        用于“一致性” DMA 映射（dma_alloc_coherent 等）。

        这种映射是长期存在的，硬件和 CPU 都可以无障碍地访问，缓存一致性由硬件自动维护。

        内核会确保为一致性 DMA 分配的内存地址同样落在设备声明的这个地址范围内。

    `dma_set_mask_and_coherent()`确保了：

    * 内核不会为 DMA 操作分配设备无法访问的内存地址。

        （内核只能分指定范围内的地址）

    * 设备驱动能够安全、可靠地执行 DMA 数据传输，充分发挥设备性能。

        （设备只能访问只能范围内的地址）

    辅助宏：

    ```c
    #define DMA_BIT_MASK(n) (((n) == 64) ? ~0ULL : (1ULL<<(n))-1)
    ```

    * `DMA_BIT_MASK(32)` 生成 0x00000000FFFFFFFF

    * `DMA_BIT_MASK(64)` 生成 0xFFFFFFFFFFFFFFFF

* `dma_set_mask()`

    告知操作系统和设备驱动程序，某个硬件设备能够访问的系统物理内存地址范围（即DMA地址空间）有多大.

    syntax:

    ```c
    #include <linux/dma-mapping.h>

    int dma_set_mask(struct device *dev, u64 mask);
    int dma_set_coherent_mask(struct device *dev, u64 mask);
    int dma_set_mask_and_coherent(struct device *dev, u64 mask);
    ```

    返回 0 表示成功，返回一个非零的错误代码（通常是负数）表示失败。

    `dma_set_mask_and_coherent()`是其升级版，

    * streaming DMA：用于一次性的数据传输映射。CPU和设备对这块内存的访问可能不是同步的（非一致性）。

    * coherent DMA（或一致性DMA）：用于需要CPU和设备同时、一致地访问的内存（例如控制寄存器所在的内存）。这块内存在映射时会进行特殊处理以保证缓存一致性。

    `dma_set_mask_and_coherent()`的作用是同时为设备的两种DMA映射方式（流式和一致式）设置相同的地址掩码，这是最常见和推荐的做法，因为它确保了行为的一致性。

* `dma_alloc_coherent()`

    为设备与 CPU 之间进行直接内存访问（DMA）而分配一段“一致性”内存。解决了由于 CPU 缓存（Cache）的存在而引发的“缓存一致性问题”。它分配的内存区域被设置为 “无缓存”（Uncacheable） 的，或者内核会通过硬件机制（如 Cache Coherent Interconnect）自动维护这块内存的缓存一致性。

    syntax:

    ```c
    #include <linux/dma-mapping.h>

    void *dma_alloc_coherent(struct device *dev, size_t size,
                             dma_addr_t *dma_handle, gfp_t gfp);
    ```

    参数说明：

    * `dma_handle`: 一个指向 `dma_addr_t` 类型的指针。这是一个输出参数。函数成功返回后，`*dma_handle` 中存储的就是分配的内存区域的DMA总线地址。驱动程序需要将这个地址提供给设备，设备将使用这个地址来执行DMA操作。

    * `gfp`: gfp_t	分配内存时使用的标志位

        * `GFP_KERNEL`: 标准的内核内存分配，可能在分配时睡眠（阻塞）。

        * `GFP_ATOMIC`: 原子分配，不会睡眠，用于中断上下文等不能调度的地方。

    返回值：

    * `void *`: 如果分配成功，返回一个指向已分配内存区域的内核虚拟地址的指针。CPU使用这个指针来读写这块内存。

    * `NULL`: 如果分配失败，则返回 NULL。

    问题场景描述：

    * CPU 在处理数据时，数据可能缓存在 CPU 的高速缓存中，并未立即写回主内存。

    * 如果此时设备通过 DMA 直接从主内存读取数据，它读到的就是过时的、旧的数据。

    * 反之，如果设备通过 DMA 将数据写入主内存，而 CPU 的缓存中还有该地址的旧数据，那么 CPU 后续读取时可能会从缓存中得到过时的、旧的数据。

    常用的应用场景：

    * 网络设备驱动：分配用于接收和发送数据包的网络数据缓冲区。

    * 块设备驱动（如 SCSI）：分配传输命令和数据的“scatter-gather”列表。

    * USB 驱动：分配用于传输 USB 请求的数据缓冲区。

    配对函数：`dma_free_coherent()`

    syntax:

    ```c
    void dma_free_coherent(struct device *dev, size_t size,
                           void *cpu_addr, dma_addr_t dma_handle);
    ```

* `dma_addr_t`c

    dma_addr_t 保存的是设备能理解的物理地址, 而不是内核使用的虚拟地址。

    本质上是一个无符号整数类型（通常是 u64 或 u32）.

    驱动开发者应将其视为一个“黑盒”或令牌（token）或 handle。不需要关心它的具体数值是什么，只需要在DMA API函数中正确地传递和使用它。不能直接对它进行数学运算或解引用。

    地址转换的产生：

    * 简单系统（无IOMMU）： 设备直接访问物理内存。这时，dma_addr_t 通常就是CPU的物理地址（phys_addr_t）。驱动需要确保设备可以访问该物理地址范围。

    * 复杂系统（有IOMMU/SMMU）： IOMMU是一个位于设备和内存之间的硬件单元，类似于CPU的MMU。它可以将设备看到的“IO虚拟地址”（IOVA）翻译成真正的物理地址。

        在这种情况下，驱动通过DMA API申请的内存，其返回的 dma_addr_t 是一个IO虚拟地址（IOVA），而不是真正的物理地址。设备使用这个IOVA进行数据传输，IOMMU会透明地完成IOVA到物理地址的转换。

* `dma_mapping_error()`

    在使用 dma_map_single() 时，强烈建议使用 dma_mapping_error() 来检查映射是否成功，而不是直接判断返回值是否为0或NULL（因为DMA地址0可能是一个有效的物理地址）。

    ```c
    dma_addr_t dma_handle;
    dma_handle = dma_map_single(dev, ptr, size, dir);
    if (dma_mapping_error(dev, dma_handle)) {
        // 映射失败，处理错误
        return -ENOMEM; // 或其它错误码
    }
    // 映射成功，继续使用 dma_handle
    ```

* `dma_map_single()`

    为一次 DMA 传输做准备，将一块 CPU 可访问的内存区域映射到设备可以访问的 DMA 地址空间。

    syntax:

    ```c
    #include <linux/dma-mapping.h>

    dma_addr_t dma_map_single(struct device *dev, void *ptr, size_t size, enum dma_data_direction dir);

    void dma_unmap_single(struct device *dev, dma_addr_t dma_handle, size_t size, enum dma_data_direction dir);
    ```

    parameters:

    * `dev`: 指向设备结构体的指针。这个指针包含了DMA映射操作所需的硬件信息，例如设备所在的总线地址空间限制、是否具有IOMMU等。

    * `ptr`: 需要映射的内核虚拟地址。这通常是通过`kmalloc()`等内核函数分配的内存块的起始地址。

    * `size`: 需要映射的内存区域的大小（以字节为单位）。

    * `dir`: DMA 数据传输的方向。这是一个枚举类型，决定了内核如何处理缓存一致性。其取值通常为：

        * `DMA_TO_DEVICE`：数据从内存传输到设备（写操作）。

        * `DMA_FROM_DEVICE`：数据从设备传输到内存（读操作）。

        * `DMA_BIDIRECTIONAL`：数据可能双向传输。

        * `DMA_NONE`：仅用于调试，表明方向未知。

    返回值

    成功时返回映射后的`dma_addr_t` DMA 地址（总线地址）。

    如果映射失败（例如，参数无效或地址无法映射），函数可能会返回一个特殊的“错误”DMA地址（具体实现可能不同），或者在某些配置下触发BUG。

    CPU 访问内存使用的是虚拟地址（Virtual Address），经过 MMU（内存管理单元）转换后得到物理地址。而 DMA 设备通常工作在物理地址层面，但它看到的“物理地址”（我们称之为总线地址，Bus Address）有可能与 CPU 看到的物理地址不同（尤其是在有 IOMMU 的系统中）。

    `dma_map_single()`主要解决两个问题：

    1. 内地映射：将 cpu 使用的 va 和 dma handle 进行映射。cpu 使用的 va 由 kmalloc() 得到

    2. 维护缓存一致性

        * 对于设备要读取的内存（`DMA_FROM_DEVICE`），dma_map_single() 会刷洗（Flush）CPU Cache，确保设备读到的是内存中最新的数据。

        * 对于设备要写入的内存（`DMA_TO_DEVICE`），dma_map_single() 可能会作废（Invalidate）CPU Cache，确保设备写完后，CPU 下次读取时能从内存获取新数据，而不是旧的缓存数据。

    典型使用流程

    一个典型的 DMA 传输流程中，该函数的使用如下：

    1. 分配内存：使用 kmalloc() 等函数分配一块用于 DMA 缓冲区的内存。

    2. 准备数据：（如果是输出）CPU 将需要传输的数据填充到这块内存中。

    3. 映射：调用 dma_map_single(dev, addr, size, direction)。

        * dev：指向设备结构体的指针。

        * addr：第一步分配的内存的内核虚拟地址。

        * size：缓冲区大小。

        * direction：数据传输方向（如 DMA_TO_DEVICE, DMA_FROM_DEVICE, DMA_BIDIRECTIONAL）。

    4. 获取 DMA 地址：函数返回一个 dma_addr_t 类型的 DMA 地址。将这个地址写入设备的 DMA 控制器寄存器。

    5. 启动传输：通知设备可以从/向给定的 DMA 地址开始传输数据。

    6. 传输完成：设备产生中断，通知 CPU 传输完成。

    7. 解除映射：调用 dma_unmap_single() 解除映射。这会再次处理缓存一致性问题，并使得这块内存的映射关系失效。

    example:

    ```c
    // 必要的内核头文件
    #include <linux/kernel.h>
    #include <linux/module.h>
    #include <linux/init.h>

    // 设备模型相关
    #include <linux/device.h>       // 定义 struct device
    #include <linux/platform_device.h> // 如果是平台设备

    // DMA API 相关
    #include <linux/dma-mapping.h>  // 核心头文件，包含 dma_map_single 等
    // #include <linux/dma-direction.h> // 如果需要，但通常 dma-mapping.h 已包含

    // 其他可能需要的头文件，如用于内存分配的
    #include <linux/slab.h>         // 用于 kmalloc, kfree

    // ... 你的驱动代码 ...
    static int my_driver_dma_transfer(struct device *dev)
    {
        void *cpu_addr;
        dma_addr_t dma_handle;
        size_t size = 1024; // 1KB

        // 1. 分配内存
        cpu_addr = kmalloc(size, GFP_KERNEL);
        if (!cpu_addr)
            return -ENOMEM;

        // 2. 映射内存以用于DMA (传输到设备)
        dma_handle = dma_map_single(dev, cpu_addr, size, DMA_TO_DEVICE);
        if (dma_mapping_error(dev, dma_handle)) { // 推荐错误检查
            kfree(cpu_addr);
            return -ENOMEM;
        }

        // 3. 将 dma_handle 交给设备，启动DMA传输...
        // program_device_to_start_dma(dma_handle);

        // 4. (传输完成后) 解除映射
        // dma_unmap_single(dev, dma_handle, size, DMA_TO_DEVICE);

        // 5. 释放内存
        // kfree(cpu_addr);

        return 0;
    }
    ```

* `dma_alloc_coherent()`与`dma_map_single()`的区别

    dma_alloc_coherent() 用于“静态”或“长期”的共享缓冲区，而 dma_map_single() 用于“动态”或“短期”的流式DMA传输。

    `dma_alloc_coherent()`会配置这块内存为 uncacheable（不可缓存）来实现缓存一致性，代价是性能下降。通常只能分配较小的、物理连续的内存块（例如最多几个MB）（为什么？）
    
    dma_map_single() + kmalloc() 是流式映射，将一块普通的、由CPU高效使用的缓存内存，临时映射给设备用于一次DMA传输。传输完成后立即解除映射。一致性由软件（内核）在映射/解除映射时维护。适合大块内存。dma_map_single -> Flush Cache, dma_unmap_single -> Invalidate Cache.

    对于流式 dma，在 dma map 之后，dma unmap 之前，可能会有缓存不一致的情况。

### list

* 使用`list_for_each_entry_safe()`释放 list node 的内存

    list 依然是借 device 触发，关键代码如下：

    ```c
    #include <linux/list.h>

    struct my_node {
        struct list_head node_head;
        int val;
    };

    struct list_head lst_head;

    int m_open(struct inode *, struct file *) {
        pr_info("in m_open()...\n");
        INIT_LIST_HEAD(&lst_head);
        for (int i = 0; i < 3; ++i) {
            struct my_node *new_node = kmalloc(sizeof(struct my_node), GFP_KERNEL);
            new_node->val = i;
            list_add(&new_node->node_head, &lst_head);
        }
        return 0;
    }

    int m_release(struct inode *, struct file *) {
        pr_info("in m_release()...\n");
        struct my_node *cur_node, *tmp_node;
        int node_idx = 0;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, node_head) {
            pr_info("node %d, val: %d\n", node_idx++, cur_node->val);
            list_del(&cur_node->node_head);
            kfree(cur_node);
        }

        return 0;
    }
    ```

    注意这里不能使用`list_for_each_entry()`，必须使用`list_for_each_entry_safe()`，否则会运行时报错。

    `list_for_each_entry()`会直接使用当前节点`cur_node`访问到下一个节点，但是在我们的例子中，当前节点`cur_node`已经通过`kfree(cur_node);`释放掉了，所以会报错。`list_for_each_entry_safe()`则会使用`tmp_node`在`cur_node`被释放前，保存指向下个节点的指针，所以不会报错。

* `list_add_tail()`有可能是在 list 尾部添加新节点

    如果是，那么随着 list 长度增加，添加新节点会越来越费时。

* linux list

    linux list 添加新节点是倒序添加的：

    1. 初始状态

        ```
        head
        ```

    2. 添加第一个节点 0

        ```
        head -> 0
        ```

    3. 添加第二个节点 1

        ```
        head -> 1 -> 0
        ```

    4. 添加第三个节点 2

        ```
        head -> 2 -> 1 -> 0
        ```

    此时我们再使用`struct list_head`去遍历，得到的输出即为`2, 1, 0`。

* `struct list_head`是每次从尾部添加新节点，并不会每次遍历到最后一个节点才添加新节点

    example:

    ```c
    #include <linux/list.h>

    struct my_node {
        struct list_head node_head;
        int val;
    };

    struct list_head lst_head;

    int m_open(struct inode *, struct file *) {
        pr_info("in m_open()...\n");
        INIT_LIST_HEAD(&lst_head);
        for (int i = 0; i < 3; ++i) {
            struct my_node *new_node = kmalloc(sizeof(struct my_node), GFP_KERNEL);
            new_node->val = i;
            list_add(&new_node->node_head, &lst_head);
        }
        return 0;
    }

    int m_release(struct inode *, struct file *) {
        pr_info("in m_release()...\n");
        struct my_node *cur_node;
        int node_idx = 0;
        list_for_each_entry(cur_node, &lst_head, node_head) {
            pr_info("node %d, val: %d\n", node_idx++, cur_node->val);
        }

        return 0;
    }
    ```

    run: `sudo cat /dev/hlc_dev`

    dmesg output:

    ```
    [13574.844808] in m_open()...
    [13574.844916] in m_read()...
    [13574.844954] in m_release()...
    [13574.844956] node 0, val: 2
    [13574.844961] node 1, val: 1
    [13574.844964] node 2, val: 0
    ```

* `list_add()`两个参数都是`struct list_head*`

* `LIST_HEAD()`与`INIT_LIST_HEAD()`

    `LIST_HEAD()`是一个宏，在编译时展开，帮你定义变量，并做好初始化：

    ```c
    #include <linux/list.h>

    struct my_node {
        struct list_head node_head;
        int val;
    };

    static LIST_HEAD(lst_head);

    int m_open(struct inode *, struct file *) {
        pr_info("in m_open()...\n");
        for (int i = 0; i < 3; ++i) {
            struct my_node *new_node = kmalloc(sizeof(struct my_node), GFP_KERNEL);
            new_node->val = i;
            // ...
    ```

    其中，`static LIST_HEAD(lst_head);`做了如下几件事：
    
    1. 定义变量`struct list_head lst_head;`

    2. 将`lst_head`的`next`和`prev`都指向自己

    3. 由于是全局变量，所以声明为`static`的，防止和其他文件里的变量冲突。

    `INIT_LIST_HEAD()`是一个函数，其定义如下：

    ```c
    static inline void INIT_LIST_HEAD(struct list_head *list)
    {
    	WRITE_ONCE(list->next, list);
    	WRITE_ONCE(list->prev, list);
    }
    ```

    通常配合外部的`struct list_head xxx;`使用：

    ```c
    #include <linux/list.h>

    struct my_node {
        struct list_head node_head;
        int val;
    };

    struct list_head lst_head;

    int m_open(struct inode *, struct file *) {
        pr_info("in m_open()...\n");
        INIT_LIST_HEAD(&lst_head);
        for (int i = 0; i < 3; ++i) {
            struct my_node *new_node = kmalloc(sizeof(struct my_node), GFP_KERNEL);
            new_node->val = i;
            // ...
    ```

* `list_add()`是在指定 node 后添加 node

* `LIST_HEAD_INIT()`

    `LIST_HEAD_INIT()`展开为

    ```c
    #define LIST_HEAD_INIT(name) { &(name), &(name) }
    ```

    可以看到，主要是完成链表的静态初始化功能，将头节点指向自身。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head = LIST_HEAD_INIT(lst_head);

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add(&new_node->list_head, &lst_head);
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            pr_info("del node val: %d\n", cur_node->val);
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 2938.473855] in hello_init()...
    [ 2941.209560] in hello_exit()...
    [ 2941.209570] del node val: 456
    [ 2941.209575] del node val: 123
    ```

* `list_del_init()`

    删除完节点后，将此节点的 prev 和 next 指向自身。

    syntax:

    ```c
    #include <linux/list.h>

    void list_del_init(struct list_head *entry)
    ```

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head = LIST_HEAD_INIT(lst_head);
    LIST_HEAD(lst_head_2);

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add(&new_node->list_head, &lst_head);
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            list_del_init(&cur_node->list_head);
            list_add(&cur_node->list_head, &lst_head_2);
        }
        
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head_2, list_head) {
            pr_info("del node val: %d\n", cur_node->val);
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 4285.077823] in hello_init()...
    [ 4290.003627] in hello_exit()...
    [ 4290.003719] del node val: 123
    [ 4290.003754] del node val: 456
    ```

    这个函数通常用于将一个节点从一个链表中取出来，并准备好将它添加到另一个链表中。

    实测将上述代码中的`list_del_init()`替换成`list_del()`后，没有什么区别。

    `list_del()`将节点取出后，会将节点的`next`, `prev`设置为`LIST_POISON1`和`LIST_POISON2`；`list_del_init()`将节点取出后，会将节点的`next`, `prev`都指向自身。

* 将`struct list_head`内嵌到用户定义的`struct ListNode`中，可能是 linux c 没有泛型和模板的无奈之举

    如果内核直接实现

    ```c
    struct ListNode {
        struct ListNode *prev, *next;
        void *user_data;
        size_t user_data_len;
    }
    ```

    那么`kmalloc()`时无法把`user_data`的内存分出来，用户需要自己管理这块内存，而且还可能涉及到浅复制，深复制的问题。

    如果内核想实现

    ```cpp
    template<typename T>
    void list_add(T *list_head, T *new_node) {
        T *tmp_node = list_head->next;
        list_head->next = new_node;
        new_node->next = tmp_node;
        new_node->prev = list_head;
    }
    ```

    那么又没有模板机制。

    综合考虑，还是内嵌的方式最合适。

* `list_del_rcu()`

    RCU - Read-Copy-Update

    用于无锁删除链表中的节点。

    （未验证）

    多线程访问链表的情况下，`list_del()`在删除节点时，通常需要加锁保护。但是加锁会降低性能，如果想无锁删除节点，那么就需要用到`list_del_rcu()`。`list_del_rcu()`会先修改被删除节点前一个节点的 next 指针，然后调用 synchronize_rcu() 或 kfree_rcu() 等函数等待宽限期（Grace Period），确保所有在删除操作前开始的读临界区都结束后，才安全地释放该节点的内存。

    在宽限期结束前，可能仍有读者正在遍历链表并访问该节点的数据。由于 prev 指针未被修改，这些读者可以继续安全地向前遍历链表，而不会因为节点被删除而崩溃（不会遇到 LIST_POISON）。（为什么 reader 可以访问 prev 就不会崩溃？）

    适用于读多写少的链表。

    example:

    ```c
    // 假设一个RCU保护的链表
    struct my_data {
        int value;
        struct list_head list;
    };

    // 写者删除节点
    void delete_node(struct my_data *node)
    {
        spin_lock(&write_lock); // 写者之间仍需同步
        list_del_rcu(&node->list); // 1. 从链表逻辑删除
        spin_unlock(&write_lock);

        // 2. 等待所有读者离开宽限期
        synchronize_rcu(); 

        // 3. 现在安全地释放内存
        kfree(node);
    }

    // 读者遍历链表（无锁！）
    void reader(void)
    {
        struct my_data *node;

        rcu_read_lock(); // 标记进入RCU读临界区
        list_for_each_entry_rcu(node, &my_list, list) {
            // 安全地访问 node->value，即使它正被删除
            printk("%d\n", node->value);
        }
        rcu_read_unlock(); // 标记离开读临界区
    }
    ```

* `list_first_entry()`

    给定`struct list_head*`指针，拿到

    syntax:

    ```c
    #include <linux/list.h>

    list_first_entry(ptr, type, member)
    ```

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head;

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        INIT_LIST_HEAD(&lst_head);
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add(&new_node->list_head, &lst_head);
        struct my_list_node *node = list_first_entry(&lst_head, struct my_list_node, list_head);
        pr_info("first entry val: %d\n", node->val);
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [13557.594168] in hello_init()...
    [13557.594275] first entry val: 456
    [13564.272870] in hello_exit()...
    ```

    与其对应的宏为`list_last_entry()`。

* `list_next_entry()`

    根据当前 node 指针，拿到下一个 node 的指针。

    ```syntax
    ptr = list_next_entry(pos, member)
    ```

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head;

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        INIT_LIST_HEAD(&lst_head);
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add(&new_node->list_head, &lst_head);
        struct my_list_node *node = list_first_entry(&lst_head, struct my_list_node, list_head);
        pr_info("first entry val: %d\n", node->val);
        node = list_next_entry(node, list_head);
        pr_info("next entry val: %d\n", node->val);
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [16486.123036] in hello_init()...
    [16486.123155] first entry val: 456
    [16486.123160] next entry val: 123
    [16492.787828] in hello_exit()...
    ```

    可以看到，`list_first_entry()`传进去的是`list_head`，而`list_next_entry()`传进去的是我们自己的 node struct。

    与其对应的宏为`list_prev_entry()`。

* `list_add_tail()`

    在链表尾部添加元素。

    syntax:

    ```c
    static inline void list_add_tail(struct list_head *new, struct list_head *head)
    ```

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head;

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        INIT_LIST_HEAD(&lst_head);
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add_tail(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add_tail(&new_node->list_head, &lst_head);

        struct my_list_node *cur_node;
        list_for_each_entry(cur_node, &lst_head, list_head) {
            pr_info("node val: %d\n", cur_node->val);
        }
        
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [  488.597973] in hello_init()...
    [  488.597977] node val: 123
    [  488.597981] node val: 456
    [  495.047735] in hello_exit()...
    ```

    如果链表越来越长，`list_add_tail()`的速度不会变慢，因为 linux 中的链表是双向循环链表，head 节点的 prev 即指向尾节点。

* linux 内核没有提供方法直接得到链表的长度，需要我们自己实现。

    ```c
    #include <linux/list.h>

    int list_node_count(struct list_head *head)
    {
        int count = 0;
        struct list_head *pos;

        // 使用 list_for_each 遍历链表，每到一个节点计数器加一
        list_for_each(pos, head) {
            count++;
        }

        return count;
    }
    ```

* `list_empty()`

    判断链表是否为空。

    list_empty() 原理：它检查头节点的 next 指针是否指向它自己。

    ```c
    // list_empty 的典型实现
    #define list_empty(head) ((head)->next == (head))
    ```

    `list_empty()`只能作用于已经初始化的链表。

    一个被删除且未被重新 init 的节点调用 list_empty() 不会返回真。

* `list_entry()`

    `list_entry()`和`container_of()`完全等价，用于从成员成员拿到外部 struct 的指针。

    syntax:

    ```c
    list_entry(ptr, type, member)
    ```

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>
    #include <linux/pci.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head;

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        INIT_LIST_HEAD(&lst_head);
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add_tail(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add_tail(&new_node->list_head, &lst_head);

        struct my_list_node *cur_node;
        struct list_head *cur_head;
        list_for_each(cur_head, &lst_head) {
            cur_node = list_entry(cur_head, struct my_list_node, list_head);
            pr_info("node val: %d\n", cur_node->val);
        }

    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 4057.550524] in hello_init()...
    [ 4057.550644] node val: 123
    [ 4057.550650] node val: 456
    [ 4063.035152] in hello_exit()...
    ```

* `list_for_each()`

    对`struct list_head`进行遍历，如果需要拿到外部 struct 的指针，那么需要手动调用`container_of()`。

    syntax:

    ```c
    list_for_each(pos, head)
    ```

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/slab.h>
    #include <linux/pci.h>

    struct my_list_node {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_head;

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        INIT_LIST_HEAD(&lst_head);
        struct my_list_node *new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 123;
        list_add_tail(&new_node->list_head, &lst_head);
        new_node = kmalloc(sizeof(struct my_list_node), GFP_KERNEL);
        new_node->val = 456;
        list_add_tail(&new_node->list_head, &lst_head);

        struct list_head *cur_head;
        list_for_each(cur_head, &lst_head) {
            pr_info("node val: %d\n", container_of(cur_head, struct my_list_node, list_head)->val);
        }

    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        struct my_list_node *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_head, list_head) {
            list_del(&cur_node->list_head);
            kfree(cur_node);
        }
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 3736.286051] in hello_init()...
    [ 3736.286191] node val: 123
    [ 3736.286198] node val: 456
    [ 3743.204415] in hello_exit()...
    ```

    `list_for_each_safe()`为`list_for_each()`的删除节点版本。

* 没有`list_next()`这个函数。

* `list_splice()`

    通过修改指针，将一个链表合并到另一个链表。

    syntax:

    ```c
    void list_splice(struct list_head *list, struct list_head *head);

    void list_splice_init(struct list_head *list, struct list_head *head);
    ```

    将链表`list`整体添加到`head`链表的前面。

    比如

    ```
    list: 2 1 0
    head: 5 4 3

    after splice:
    list: empty
    head: 2 1 0 5 4 3
    ```

    `list_splice_init()`在修改完链表指针后，将`list`进行 init。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/list.h>
    #include <linux/vmalloc.h>

    struct ListNode {
        struct list_head list_head;
        int val;
    };

    struct list_head lst_1;
    struct list_head lst_2;

    int init_mod(void) {
        pr_info("init hlc module...\n");

        INIT_LIST_HEAD(&lst_1);
        INIT_LIST_HEAD(&lst_2);

        for (int i = 0; i < 3; ++i) {
            struct ListNode *new_node = vmalloc(sizeof(struct ListNode));
            new_node->val = i;
            list_add(&new_node->list_head, &lst_1);
        }

        for (int i = 3; i < 6; ++i) {
            struct ListNode *new_node = vmalloc(sizeof(struct ListNode));
            new_node->val = i;
            list_add(&new_node->list_head, &lst_2);
        }

        struct ListNode *cur_node;
        pr_info("list 1:\n");
        list_for_each_entry(cur_node, &lst_1, list_head) {
            pr_info("node val: %d\n", cur_node->val);
        }

        pr_info("list 2:\n");
        list_for_each_entry(cur_node, &lst_2, list_head) {
            pr_info("node val: %d\n", cur_node->val);
        }

        list_splice_init(&lst_1, &lst_2);
        pr_info("after list splice:\n");

        pr_info("list 1:\n");
        list_for_each_entry(cur_node, &lst_1, list_head) {
            pr_info("node val: %d\n", cur_node->val);
        }

        pr_info("list 2:\n");
        list_for_each_entry(cur_node, &lst_2, list_head) {
            pr_info("node val: %d\n", cur_node->val);
        }
        
        return 0;
    }

    void exit_mod(void) {
        pr_info("exit hlc module...\n");

        struct ListNode *cur_node, *tmp_node;
        list_for_each_entry_safe(cur_node, tmp_node, &lst_1, list_head) {
            list_del(&cur_node->list_head);
            vfree(cur_node);
        }

        list_for_each_entry_safe(cur_node, tmp_node, &lst_2, list_head) {
            list_del(&cur_node->list_head);
            vfree(cur_node);
        }
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [  354.895541] init hlc module...
    [  354.895569] list 1:
    [  354.895572] node val: 2
    [  354.895576] node val: 1
    [  354.895578] node val: 0
    [  354.895580] list 2:
    [  354.895582] node val: 5
    [  354.895583] node val: 4
    [  354.895585] node val: 3
    [  354.895587] after list splice:
    [  354.895589] list 1:
    [  354.895591] list 2:
    [  354.895592] node val: 2
    [  354.895594] node val: 1
    [  354.895596] node val: 0
    [  354.895598] node val: 5
    [  354.895600] node val: 4
    [  354.895602] node val: 3
    [  431.783673] exit hlc module...
    ```

    注：

    1. 如果不调用`list_splice_init()`，而调用`list_splice()`，那么 dmesg 输出会死循环

        ```
        [  784.259569] init hlc module...
        [  784.259599] list 1:
        [  784.259602] node val: 2
        [  784.259606] node val: 1
        [  784.259608] node val: 0
        [  784.259610] list 2:
        [  784.259612] node val: 5
        [  784.259614] node val: 4
        [  784.259616] node val: 3
        [  784.259618] after list splice:
        [  784.259620] list 1:
        [  784.259621] node val: 2
        [  784.259623] node val: 1
        [  784.259625] node val: 0
        [  784.259627] node val: 5
        [  784.259629] node val: 4
        [  784.259631] node val: 3
        [  784.259633] node val: 1074016256
        [  784.259636] node val: 2
        [  784.259638] node val: 1
        [  784.259640] node val: 0
        [  784.259642] node val: 5
        [  784.259644] node val: 4
        [  784.259646] node val: 3
        [  784.259647] node val: 1074016256
        ...
        ```

        从 list 1 的 head 出发，进入 list 2 的节点，遍历到 list 2 的结尾，发现 list 2 的结尾指向 list 2 的 head，并不是 list 1 的 head，程序以为没有回到 list 1 的开始，所以强行认为 list 2 head 是某个 struct Node 的成员，解析到了一个无效 val `1074016256`，并继续从头开始在环中遍历。

    1. 如果将`list_splice_init()`替换为`list_splice_tail_init()`，则输出为

        ```
        [ 1316.980085] init hlc module...
        [ 1316.980110] list 1:
        [ 1316.980113] node val: 2
        [ 1316.980117] node val: 1
        [ 1316.980120] node val: 0
        [ 1316.980122] list 2:
        [ 1316.980123] node val: 5
        [ 1316.980125] node val: 4
        [ 1316.980127] node val: 3
        [ 1316.980129] after list splice tail:
        [ 1316.980132] list 1:
        [ 1316.980133] list 2:
        [ 1316.980135] node val: 5
        [ 1316.980137] node val: 4
        [ 1316.980139] node val: 3
        [ 1316.980141] node val: 2
        [ 1316.980142] node val: 1
        [ 1316.980144] node val: 0
        [ 1363.336242] exit hlc module...
        ```

        如果使用`list_splice_tail()`，也会出现循环输出的情况。

* `list_is_singular()`

    判断当前链表是否只有一个有效节点。

    具体做两件事：

    1. 判断链表非空 (!list_empty(head))。即判断 head->next 是否指向自己。

    2. 判断链表的第一个节点的 next 指针指向头节点 (head->next->next == head)。

    syntax:

    ```c
    #include <linux/list.h>

    bool list_is_singular(const struct list_head *head);
    ```

    `list_is_singular()`要求链表是 init 过的。比如通过`list_del()`从一个链表上取下一个节点后，没有 init，那么无法使用`list_is_singular()`判断这个节点是否为单个节点。

* `init_llist_head()`

    init_llist_head() 用于无锁单向链表。初始化一个 struct llist_head 节点，将其 first 指针设置为 NULL，表示一个空的单向链表。

### irq, 中断与 msi-x

* 一个可用的 irq 软中断程序，见`ref_15`

    在编译完，`insmod`之后，可以使用`sudo cat /dev/etx_device`触发中断，然后可以看到`dmesg`里显示：

    ```
    [12575.759721] intrp: loading out-of-tree module taints kernel.
    [12575.759724] intrp: module verification failed: signature and/or required key missing - tainting kernel
    [12575.760032] Major = 240 Minor = 0 
    [12575.760356] Device Driver Insert...Done!!!
    [12715.415083] Device File Opened...!!!
    [12715.415103] Read function
    [12715.415107] __common_interrupt: 1.59 No irq handler for vector
    [12715.415119] Device File Closed...!!!
    ```

    11 号中断是保留中断，没有默认用途，因此用户可以去自定义。

    代码中比较难理解的是`asm("int $0x3B");  // Corresponding to irq 11`这一句。

    我们可以打开`/usr/src/linux-headers-6.5.0-28-generic/arch/x86/include/asm/irq_vectors.h`文件，查到

    `#define FIRST_EXTERNAL_VECTOR           0x20`

    不清楚`#define IRQ0_VECTOR (FIRST_EXTERNAL_VECTOR + 0x10)`这一步是怎么来的。

    最后还需要加上我们的中断号`11`，即`0x20 + 0x10 + 11 = 0x3B`，

    这诚是`asm("int $0x3B");`的由来。

* typical IRQ assignments for a PC

    | IRQ number | Device |
    | - | - |
    | 0 | System timer |
    | 1 | Keyboard (PS/2) |
    | 2 | Cascade from IRQ 9 |
    | 3 | COM port 2 or 4 |
    | 4 | COM port 1 or 3 |
    | 5 | Parallel (printer) port 2 or sound cards |
    | 6 | Floppy drive controller |
    | 7 | Parallel (printer) port 1 |
    | 8 | Real-time clock |
    | 9 | Video |
    | 10 | Open |
    | 11 | Open |
    | 12 | Mouse (PS/2) |
    | 13 | Coprocessor |
    | 14 | Primary IDE controller (hard drives) |
    | 15 | Secondary IDE controller (hard drives) |

    ref: <https://www.techtarget.com/whatis/definition/IRQ-interrupt-request>

* 常用 irq

    IRQ 分为两大类：标准硬件预设 IRQ 和 动态分配 IRQ。

    1. 标准硬件预设 IRQ (Legacy IRQs)

        * IRQ 0: 系统定时器 (System timer)
        
            由主板上的定时器芯片使用，不可更改。

            目前仍在使用。

        * IRQ 1: 键盘控制器 (Keyboard)
            
            专用于 PS/2 键盘。

            目前仍在使用。

        * IRQ 2: 级联中断 (Cascade to IRQ 9)
        
            用于连接第二个中断控制器 (PIC)，实际请求会重定向到 IRQ 9。

        * IRQ 3: 串行端口 COM2 (Serial port COM2/COM4)
        
            旧式串口。

            目前未在使用。

        * IRQ 4: 串行端口 COM1 (Serial port COM1/COM3)
        
            旧式串口。

            目前未在使用。

        * IRQ 5: 声卡 / 并行端口 LPT2 (Sound card / LPT2)
        
            在没有 LPT2 的系统中，常被 PCI 声卡等设备使用。

            目前未在使用。

        * IRQ 6: 软盘控制器 (Floppy disk controller)
        
            专用于软驱。

            目前未在使用。

        * IRQ 7:
        
            并行端口 LPT1 (Parallel port LPT1)
            
            打印机端口。

            目前未在使用。

        * IRQ 8: 实时时钟 (Real-time clock, RTC)
        
            由 CMOS 时钟使用。

            目前仍在使用。

        * IRQ 9: 重定向的 IRQ2 / ACPI
        
            接收来自 IRQ2 的级联中断，也常用于 ACPI 系统事件。

            目前正在使用。

        * IRQ 12: PS/2 鼠标 (PS/2 Mouse)
        
            专用于 PS/2 鼠标接口。

            目前仍在使用。

        * IRQ 13: 数学协处理器 (Math coprocessor)
        
            用于 x87 FPU 错误异常。

            目前未在使用。

        * IRQ 14: 主 IDE 通道 (Primary ATA channel)
        
            用于连接在主 IDE 通道上的硬盘或光驱。

            目前仍在使用。

        * IRQ 15
        
            次 IDE 通道 (Secondary ATA channel)	用于连接在次 IDE 通道上的硬盘或光驱。

            目前仍在使用。

    2. 动态分配 IRQ

        IRQ 号动态分配，中断共享非常普遍，多个 PCI 设备可以共享同一个 IRQ 线。

        从 irq 16 开始及以上的基本都是动态分配的。

    `cat /proc/interrupts`输出:

    ```
    (base) hlc@hlc-VirtualBox:~$ cat /proc/interrupts
               CPU0       CPU1       CPU2       CPU3       
      0:        121          0          0          0   IO-APIC   2-edge      timer
      1:          0          0          0      27426   IO-APIC   1-edge      i8042
      8:          0          0          0          0   IO-APIC   8-edge      rtc0
      9:          0          0          0          0   IO-APIC   9-fasteoi   acpi
     12:          0      90214          0          0   IO-APIC  12-edge      i8042
     14:          0          0          0          0   IO-APIC  14-edge      ata_piix
     15:          0          0          0      25200   IO-APIC  15-edge      ata_piix
     18:          0          0          0          1   IO-APIC  18-fasteoi   vmwgfx
     19:          0          0         57     529522   IO-APIC  19-fasteoi   ehci_hcd:usb2, enp0s3
     20:          0          0     157288          0   IO-APIC  20-fasteoi   vboxguest
     21:          0      15104     560519          0   IO-APIC  21-fasteoi   ahci[0000:00:0d.0], snd_intel8x0
     22:         27          0          0          0   IO-APIC  22-fasteoi   ohci_hcd:usb1
    NMI:          0          0          0          0   Non-maskable interrupts
    LOC:   19816261    6719703    6819970    6554434   Local timer interrupts
    SPU:          0          0          0          0   Spurious interrupts
    PMI:          0          0          0          0   Performance monitoring interrupts
    IWI:         35         22         15          9   IRQ work interrupts
    RTR:          0          0          0          0   APIC ICR read retries
    RES:     124747     132009     186545     193448   Rescheduling interrupts
    CAL:    5551069    6766045    5986573    5687906   Function call interrupts
    TLB:     891238     897723     909639     929551   TLB shootdowns
    TRM:          0          0          0          0   Thermal event interrupts
    THR:          0          0          0          0   Threshold APIC interrupts
    DFR:          0          0          0          0   Deferred Error APIC interrupts
    MCE:          0          0          0          0   Machine check exceptions
    MCP:         78         78         78         78   Machine check polls
    ERR:          0
    MIS:       4066
    PIN:          0          0          0          0   Posted-interrupt notification event
    NPI:          0          0          0          0   Nested posted-interrupt event
    PIW:          0          0          0          0   Posted-interrupt wakeup event
    ```

    其中 CPU 表示每个 CPU 核心对中断的处理次数，`IO-APIC`表示中断控制器类型，最后一栏`timer`之类的表示该中断对应的设备名。

* `request_irq()`

    内核注册一个中断处理程序（中断服务例程，ISR）

    中断号（IRQ）

    将一个具体的中断号（irq）与驱动程序提供的中断处理函数（handler）绑定起来

    syntax:

    ```c
    #include <linux/interrupt.h>

    int request_irq(unsigned int irq,
                    irq_handler_t handler,
                    unsigned long flags,
                    const char *name,
                    void *dev_id);
    ```

    * `flags`:

        * `IRQF_SHARED`：允许多个设备共享同一个中断线。

        * `IRQF_ONESHOT`：中断在处理完毕后需要重新显式启用（用于线程化中断）。

        * `IRQF_TIMER`：标记为定时器中断，以便系统在处理电源管理时特殊考虑。

        * `IRQF_IRQPOLL`：用于共享中断中的轮询处理。

    * `name`

        通常为设备名，用于在`/proc/interrupts`中标识这个中断的拥有者

    * `dev_id`

        提供一个唯一指针，用于区分是哪个设备触发了中断

        通常指向一个代表设备的结构体（如 struct device 或自定义的私有数据）

        在 irq handler 中，会被作为函数参数传入。

* 如果在注册 irq X 的 handler 1 时设置为 shared，那么后面注册 irq X 的 handler 2 时也设置为 shared，可以注册成功，有 irq X 的中断时，两个 handler 都会调用

    未验证。

    处理程序的责任：

        每个处理程序被调用时，会接收到 irq 号和注册时提供的 dev_id 作为参数。

        处理程序必须检查这个中断是否是发给它的设备的。它需要通过读取设备的某个状态寄存器来判断。

        如果中断是它的：它处理中断，然后返回 IRQ_HANDLED。

        如果中断不是它的：它什么都不做，立刻返回 IRQ_NONE。

        这种检查机制是共享中断能够工作的基础，确保每个驱动只处理属于自己的设备产生的中断。

* irq 1 在 request_irq 时是非 shared 的，所以再次注册 irq 1 的 handler 时，会失败

* MSI (Message Signaled Interrupts)

* irq 11 example

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/interrupt.h>

    irqreturn_t my_irq_handler(int irq, void *dev_id) {
        pr_info("in my_irq_handler()...\n");
        pr_info("irq: %d, dev_id: %p\n", irq, dev_id);
        return IRQ_HANDLED;
    }

    int hello_init(void) {
        pr_info("in hello_init()...\n");
        int ret = request_irq(11, my_irq_handler, IRQF_SHARED, "hlc irq", my_irq_handler);
        if (ret != 0) {
            pr_err("fail to request irq\n");
            return -1;
        }
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        free_irq(11, my_irq_handler);
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    `cat /proc/interrupts ` output:

    ```
    test@Ubuntu22:~/Documents/Projects/driver_test$ cat /proc/interrupts 
               CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       CPU6       CPU7       
      0:         33          0          0          0          0          0          0          0   IO-APIC   2-edge      timer
      1:         11          0          0          0          0          0          0          0   IO-APIC   1-edge      i8042
      6:          0          0          3          0          0          0          0          0   IO-APIC   6-edge      floppy
      7:          0          0          0          1          0          0          0          0   IO-APIC   7-edge      parport0
      8:          0          1          0          0          0          0          0          0   IO-APIC   8-edge      rtc0
      9:          0          0          0          0          0          0          0          0   IO-APIC   9-fasteoi   acpi
     11:          0          0          0          0          0          0          0          0   IO-APIC  11-fasteoi   hlc irq
     12:          0          0          0          0          0          0          0         15   IO-APIC  12-edge      i8042
    ```

* 操作系统中断与 pci msix 的交互流程

    1. module 调用`iv = pci_irq_vector(pdev, idx)`向操作系统申请到 msix 中第`idx`个中断的中断向量（由操作系统进行分配）

    2. module 调用`request_irq(iv, handler_func, ...)`将中断处理函数的地址写入到 IDT 中

    3. 向 msix 第`idx`号中断的配置空间中写入中断向量`iv`的数值

        操作系统将一个目标地址和一个数据值写入设备PCI配置空间中的MSI-X相关寄存器（每个中断向量都有自己的一套地址/数据寄存器）。

        目标地址：指向CPU中断控制器的端口（例如x86上的Local APIC的地址）。

        数据值：这个数据值就是分配给这个特定中断向量的CPU中断向量号（例如 0x41）。

    4. pci 每次发起中断时，由 msix 向特定的内存中写入中断向量`iv`，操作系统感知到后，由`iv`在 IDT 中找到对应的中断函数，对中断进行处理。

* 中断向量，IDT 与 irq

    操作系统维护一张全局的中断描述符表 (IDT, Interrupt Descriptor Table)，其本质为一个数组，存储的是中断处理函数的地址，比如

    ```
    ...
    IDT[0x40] = irq_handler_func_40()
    IDT[0x41] = irq_handler_func_41()
    IDT[0x42] = irq_handler_func_42()
    IDT[0X43] = irq_handler_func_43()
    ...
    ```
    
    其中数组的索引`0x40`, `0x41`, ... 即为中断向量 (Interrupt Vector)（号）。

    关于中断向量的解释（未验证）：
    
    1. 解释一：中断向量这个词最开始由 intel 这些 cpu 制造商引入，这里的“向量”是从硬件设计者的角度考虑的，倾向于表示这是一个矢量，一个路标，一个指引，告诉你中断处理函数在这里。

    2. 解释二：当一个PCI设备使用MSI或MSI-X时，它不再只有一个IRQ号，而是有一组IRQ号（一个“向量”）。

    由于 IDT 是全局的，但是每个驱动可能只想关心自己的中断向量，不想关心别人的，那么我们希望将 IDT 看作一个资源池，每个驱动向操作系统申请中断向量，并从 0 开始编号，得到中断向量索引(Vector Index)：

    ```
    ...
    IDT[0x40] = irq_handler_func_40(), driver xxx, vector idx 0
    IDT[0x41] = irq_handler_func_41(), driver yyy, vector idx 0
    IDT[0x42] = irq_handler_func_42(), driver yyy, vector idx 1
    IDT[0X43] = irq_handler_func_43(), driver xxx, vector idx 1
    ...
    ```

    其中的`vector idx 0`, `vector idx 1`, ... 这些即为中断向量索引。

    操作系统为了便于管理，将中断向量与全局资源 irq (Interrupt ReQuest) 进行一一映射，从而使得 module 可以基于 irq 注册中断处理函数。

    ```
    ...
    IDT[0x40] = irq_handler_func_40(), driver xxx, vector idx 0, irq 233
    IDT[0x41] = irq_handler_func_41(), driver yyy, vector idx 0, irq 234
    IDT[0x42] = irq_handler_func_42(), driver yyy, vector idx 1, irq 235
    IDT[0X43] = irq_handler_func_43(), driver xxx, vector idx 1, irq 236
    ...
    ```

* `pci_irq_vector()`

    获取一个 PCI 设备上某个特定 MSI-X 或 MSI 中断向量的 Linux 内核IRQ编号（中断号）。

    syntax:

    ```c
    #include <linux/pci.h>

    int pci_irq_vector(struct pci_dev *dev, unsigned int nr);
    ```

    * `nr`: MSI-X Table中中断条目的索引，索引从 0 开始

        `nr`的范围为`[0, dev->msix_cnt - 1]`或`[0, dev->msi_cnt - 1]`

    return value:

    返回一个大于等于 0 的值，表示PCI设备第 nr 个中断向量对应的 Linux 内核 IRQ 编号（中断号）

    如果失败，该函数返回一个错误码（负值）。

    调用 pci_irq_vector() 之前，必须已经成功地为PCI设备分配了MSI或MSI-X中断向量。这通常是通过 pci_alloc_irq_vectors() 函数完成的。

    example:

    假设你正在为一个支持MSI-X的PCI设备编写驱动程序。该设备申请了3个中断向量：

        索引0：用于接收数据包

        索引1：用于发送数据包完成

        索引2：用于报告错误

    ```c
    // 注册索引0（接收）的中断处理函数
    irq_rx = pci_irq_vector(pdev, 0);
    request_irq(irq_rx, my_rx_handler, 0, "my_driver_rx", my_data);

    // 注册索引1（发送）的中断处理函数
    irq_tx = pci_irq_vector(pdev, 1);
    request_irq(irq_tx, my_tx_handler, 0, "my_driver_tx", my_data);

    // 注册索引2（错误）的中断处理函数
    irq_err = pci_irq_vector(pdev, 2);
    request_irq(irq_err, my_err_handler, 0, "my_driver_err", my_data);
    ```

    对于传统的引脚中断（INTx），通常使用 pci_dev->irq 来获取设备唯一的IRQ编号。

    对于MSI/MSI-X，必须使用`pci_irq_vector(dev, n)`来获取第n个中断的 IRQ 编号。`pci_dev->irq`在这个场景下可能没有意义或已被重用。

* 中断服务程序（ISR）即中断处理函数

* 为什么使用`request_irq(irq, handler_func)`，而不是`request_vector_idx(idx, handler_func)`？

    假如 vector idx 对于每个设备来说从 0 开始编号，或者对于每个 module 来说从 0 开始编号，那么问题来了，如果一个 module 里有多个 pci 设备，每个设备的中断都是从 0 开始编号，那么`request_vector_idx(0, handler_func)`究竟指的是哪个设备的中断？如果对于 module 从 0 开始编号，那么如果多个 module 共同管理一个设备，那么应该以哪个 module 的编号为准？

    有一些 module 没有维护 device，注册的中断可能是软中断，或定时器中断，那么基于 device 编号就彻底不能用了。

    这样看来，将 irq 与中断向量作映射，作为全局资源进行管理，似乎是比较合理的解决方式。

* cpu 的中断向量

    向量 0-31 通常预留给异常（如除零错误、页故障）和不可屏蔽中断（NMI）。

    向量 32-255 留给用户定义的可屏蔽中断，也就是来自外部硬件设备的中断。

    irq 是操作系统抽象的概念，部分映射与 cpu 中断向量号相同，其余的不同。

    具体的 cpu 中断向量映射情况：

    * 32以下：CPU保留向量（0-31用于异常、NMI等）

    * 32-47：传统PIC映射区（IRQ0-15）

    * 现代 MSI-X：通常是一对一映射（每个Linux IRQ有独立CPU向量）

    * 现代 INTx：可能多对一映射（多个Linux IRQ共享CPU向量）

    由于单个 cpu 的中断向量只有 256 个，而 msix 最多可能有 2048 个中断向量，所以cpu 中的一个中断向量，可能会分时复用地处理多个 msix 的中断向量。

    如果 pc 上有多个 cpu 节点，那么 msix 可能将中断注册到多个 cpu 中断向量上以做负载均衡。

* MSI-X是MSI的增强版，其主要增强就是支持更多的中断向量（MSI最多32个，MSI-X可达2048个）。

* `cat /proc/interrupts`的最后一栏即`request_irq()`中填的 name

    ```c
    request_irq(11, irq_handler, IRQF_SHARED, "hlc irq", irq_handler);
    ```

    ```
     11:          0          0          0          0          0          0          0          0   IO-APIC  11-fasteoi   hlc irq
    ```

* MSI-X（Message Signaled Interrupts eXtended）是一种基于消息的中断机制。它允许设备（如网卡、GPU、NVMe SSD）通过向CPU写入一个特定的数据（消息）到特定的内存地址来请求中断，而不是使用传统的、基于专用引脚（IRQ线）的中断。

    系统软件（操作系统）会分配一块内存区域，称为 MSI-X Table。这个表中的每一项（Entry）都对应一个中断向量，包含：

    * 消息地址（Message Address）：中断消息要写入的目标地址。这个地址隐含了中断要发送给哪个CPU核心。

    * 消息数据（Message Data）：一个唯一的数据值，用来标识是哪个中断向量被触发。

    * 掩码位（Mask Bit）：用于临时禁用该中断。

    * Pending Bit：记录中断是否已发出但尚未被处理。

    虽然设备触发MSI-X的行为是一次内存写入，但整个系统（CPU、芯片组、内存控制器）被设计为能识别这次特殊的写入操作，并将其无缝地转换为一个中断信号送达CPU核心。

    MSI-X消息中指定的目标地址（Message Address） 并不指向普通的DRAM内存位置。它指向一个由CPU和芯片组预留的、特殊的物理地址范围。这个范围是专门用于处理消息信号中断的。

    芯片组通过内部的中断控制器（在现代x86系统中是APIC）架构，将一个“虚拟”的中断信号直接发送到步骤2a中确定的目标CPU核心。

    CPU核心收到这个信号后，会立即中断当前正在执行的指令流（当然，会在合理的边界处），并根据收到的向量号，去查找自己的中断描述符表（IDT），找到并执行对应的中断服务程序（ISR）。

    msix 中断并不是 cpu 轮询指定内存实现的，而是通过芯片组识别 pcie 事务类型实现的。
    
    msix 中断本质就是一次 pcie 事务（Transaction）。
    
    pcie 事务可能有很多类型：

    Memory Read

    Memory Write

    I/O Read

    I/O Write

    Message

    MSI 对应的事务为 Memory Write，而 msix 对应的事务为 Message (Message Signaled Interrupt)

    如果芯片组识别出来 pcie 事务是 message，又发现其写入的内存地址是为 MSI 中断信号预留的、特定的地址区域，那么就可以确认这是一个 msix 中断。

* `pci_alloc_irq_vectors()`

    为指定的 PCI 设备申请和分配一组中断向量（Interrupt Vectors）。

    syntax:

    ```c
    int pci_alloc_irq_vectors(struct pci_dev *dev, unsigned int min_vecs, unsigned int max_vecs, unsigned int flags);
    ```

    * `min_vecs`：驱动至少需要多少个中断向量。

    * `max_vecs`：驱动最多希望申请多少个中断向量。

    * `flags`：指定中断类型和行为的标志。最重要的标志是：

        * `PCI_IRQ_MSIX`：请求使用 MSI-X 中断。

        * `PCI_IRQ_MSI`：请求使用 MSI 中断。

        * `PCI_IRQ_LEGACY`：请求使用传统引脚中断（如 INTA#）。

        * `PCI_IRQ_ALL_TYPES`：尝试任何可用的类型（通常的用法）。

        * `PCI_IRQ_AFFINITY`：提示内核这些中断可以设置 CPU 亲和性（绑定到特定 CPU 核心）。

        各个 flag 可以使用`|`组合。你组合多个类型，内核会按性能从高到低的顺序（通常是 MSI-X -> MSI -> Legacy）自动选择可用的最佳类型。

    return value:

    分配成功则返回 实际分配到的中断数量。分配失败则返回一个负的错误码。

    配对函数：`pci_free_irq_vectors()`

    example:

    * 尝试所有中断类型，并提示亲和性：

        `int nvecs = pci_alloc_irq_vectors(pdev, 1, 32, PCI_IRQ_ALL_TYPES | PCI_IRQ_AFFINITY);`

        驱动表示它兼容任何硬件支持的中断模式，并且它设计为能够利用多CPU核心的优势。

    * 只尝试 MSI-X 或传统中断，不尝试 MSI

        `int nvecs = pci_alloc_irq_vectors(pdev, 1, 8, PCI_IRQ_MSIX | PCI_IRQ_LEGACY);`
        
        可能因为驱动在某些硬件上发现 MSI 实现有bug，或者功能需求上 MSI-X 是首选，如果不行则回退到最稳定的传统模式。

    * 只使用 MSI，并要求亲和性

        `int nvecs = pci_alloc_irq_vectors(pdev, 2, 2, PCI_IRQ_MSI | PCI_IRQ_AFFINITY);`

        只想使用 MSI 机制，并且希望中断能分布在不同的CPU上。

    example:

    ```c
    // 1. 分配中断向量（假设申请了4个）
    nvec = pci_alloc_irq_vectors(pdev, 4, 4, PCI_IRQ_MSIX);
    if (nvec < 0) {
        // 错误处理
    }

    // 2. 为每一个向量注册中断处理程序
    for (i = 0; i < nvec; i++) {
        // 注意：这里的 i 就是索引号
        ret = pci_request_irq(pdev, i, /* 索引i对应MSI-X表条目i */
                             my_interrupt_handler, // 中断处理函数
                             NULL, // dev_id
                             "my-driver:rx%d", i); // 中断名称
        if (ret) {
            // 错误处理
        }
    }
    ```

* `pci_enable_msix_range()`

    为一个 PCI 设备申请并启用一组 MSI-X 中断向量，并允许指定一个期望的数量范围。

    syntax:

    ```c
    #include <linux/pci.h>

    int pci_enable_msix_range(struct pci_dev *dev,
                              struct msix_entry *entries,
                              int min_vecs,
                              int max_vecs);
    ```

    * `entries`: 存储分配成功的中断向量信息。

    返回值：

    成功：返回一个正整数，表示实际分配的中断向量数量（这个值在 [min_vecs, max_vecs] 区间内）。

    失败：返回一个负的错误码（如 -ENOSPC 表示无足够中断资源）。

    example:

    ```c
    #define MIN_MSIX 2
    #define MAX_MSIX 8

    struct msix_entry entries[MAX_MSIX];
    int nvecs;

    // 尝试申请 MSI-X 中断，最少要2个，最多要8个。
    nvecs = pci_enable_msix_range(pdev, entries, MIN_MSIX, MAX_MSIX);

    if (nvecs < 0) {
        // 申请失败，连2个都没有，可能需要回退到传统的MSI或INTx中断
        dev_err(&pdev->dev, "Failed to enable MSI-X. Error: %d\n", nvecs);
        return nvecs;
    } else {
        // 申请成功，nvecs 是实际分配的数量
        dev_info(&pdev->dev, "Enabled %d MSI-X vectors\n", nvecs);

        // 接下来可以为每个 entries[i].vector 申请中断处理函数 (request_irq)
        for (int i = 0; i < nvecs; i++) {
            request_irq(entries[i].vector, my_handler, 0, dev_name(&pdev->dev), my_data);
        }
    }
    ```

    配对函数：`pci_disable_msix()`

    此函数目前已逐渐被`pci_alloc_irq_vectors()`（来自Linux 4.10）所取代。

* irq_handler

    syntax:

    `irqreturn_t irq_handler(int irq, void *dev_id);`

    返回值必须是`IRQ_HANDLED`或`IRQ_NONE`

    该函数运行在中断上下文中，因此其执行不能阻塞（不能睡眠、不能调用可能引起调度的函数如 kmalloc(..., GFP_KERNEL)、不能执行耗时操作）。

* 为什么中断上下文不能睡眠？

    1. 没有进程概念：中断上下文不属于任何进程。它只是“借道”执行，打断了当前正在运行的进程。如果它睡眠了，调度器不知道该唤醒哪个进程来继续执行它，因为它没有 struct task_struct 关联。这会导致系统彻底崩溃。

    2. 破坏原子性：中断是异步到来的，期望被快速处理。睡眠可能导致中断处理流程被无限期挂起，使得设备无法及时响应，其他中断也可能无法处理。

    3. 可能导致死锁：假设中断发生时，某个内核锁正被持有。如果中断处理程序尝试获取另一个锁，而这个操作又可能导致睡眠，那么系统就会死锁。持有锁的进程正在等待中断处理完成，而中断处理又在等待锁被释放。

* `device->irq`

    `dev->irq`存储了分配给该硬件设备的中断请求线（Interrupt ReQuest line）的编号。

    通常使用`request_irq(dev->irq, my_handler, ...)`去注册 ISR。配对函数为`free_irq(dev->irq, dev_id);`

    对于传统PCI设备，这个值通常在系统启动时由BIOS/UEFI或操作系统内核的PCI子系统自动分配。它通过读取设备的配置空间（具体是PCI_INTERRUPT_PIN和PCI_INTERRUPT_LINE寄存器）来分配和设置。对于现代设备（MSI/MSI-X），通常不直接使用`dev->irq`，而是使用`pci_alloc_irq_vectors()`等函数来管理 msi(x) 中断。

    在支持 MSI-X 的 PCI 设备上直接使用 dev->irq，其行为取决于内核版本和系统配置，但通常会导致性能低下、功能受限或完全无法工作。

* `request_irq()`

    内核注册一个中断处理程序（中断服务例程，ISR）

    syntax:

    ```c
    int request_irq(unsigned int irq,
                    irq_handler_t handler,
                    unsigned long flags,
                    const char *name,
                    void *dev_id);
    ```

    返回值：

    * 成功时返回 0。

    * 失败时返回一个错误码（负值），如 -EBUSY（中断线已被占用且不可共享）。

    中断标志设置: 通过 flags 参数设置中断的行为特性，常见的标志包括：

    * `IRQF_SHARED`：允许与其他驱动程序共享同一条中断线。这在中断资源紧张的系统（如基于 PCI 的系统）中很常见。

    * `IRQF_ONESHOT`：中断在处理完毕后需要重新显式启用（用于线程化中断）。

    * `IRQF_TIMER`：标记这是一个定时器中断，系统可能会对其进行特殊处理（如不会被暂停）。

    * `IRQF_IRQPOLL`：用于共享中断中的轮询处理。

    配对函数：`free_irq()`

### pci 设备：bar 与 iomap

* `pci_request_region()`

    声明和保留PCI设备的I/O端口或内存区域，防止其他驱动程序意外冲突。

    它向内核声明："这个PCI设备的这块BAR空间（I/O端口或内存）将由我这个驱动程序使用", 防止多个驱动程序同时访问同一硬件资源造成冲突.

    I/O 端口地址和内存地址都是操作系统的资源，当检测到 pci 设备时，pci 设备需要向操作系统申请这些资源，并映射到 pci 设备上，才能正常工作。

    这个申请资源的动作，即`pci_request_region()`。

    syntax:

    ```c
    int pci_request_region(struct pci_dev *pdev, int bar, const char *res_name);
    ```

    * `res_name`： 一个字符串标识符，通常为驱动名，用于在资源树中标识该资源的所有者（在 /proc/ioports 或 /proc/iomem 中可以看到）。

    返回值：

        0: 表示申请成功。

        非 0（错误码）: 表示申请失败（例如资源不存在或已被占用）。

    其逆操作函数为`pci_release_region()`。

    example:

    ```c
    static int my_driver_probe(struct pci_dev *pdev, const struct pci_device_id *id)
    {
        int ret;
        ...
        // 首先启用设备
        ret = pci_enable_device(pdev);
        if (ret) {
            dev_err(&pdev->dev, "Enable device failed\n");
            return ret;
        }

        // 然后请求设备的某个区域（例如 BAR 0）
        ret = pci_request_region(pdev, 0, "my_nic_driver");
        if (ret) {
            dev_err(&pdev->dev, "Request region for BAR0 failed\n");
            pci_disable_device(pdev); // 失败则禁用设备
            return ret;
        }

        // 请求成功，现在可以安全地映射和使用这个区域了
        my_priv->ioaddr = pci_iomap(pdev, 0, 0);
        ...
    }

    static void my_driver_remove(struct pci_dev *pdev)
    {
        ...
        // 在移除时，按相反顺序释放资源
        pci_iounmap(pdev, my_priv->ioaddr); // 取消映射
        pci_release_region(pdev, 0);        // 释放区域
        pci_disable_device(pdev);           // 禁用设备
    }
    ```

* `pci_msi_enabled()`

    检查一个 PCI 设备是否已经成功启用并配置了 MSI 或 MSI-X 中断模式.

    它通过检查该设备结构体中的内部标志位（例如 msi_enabled 或 msix_enabled）来判断状态。

    返回值：

        如果设备已经启用了 MSI 或 MSI-X 模式中的任何一种，则函数返回 true（非零值）。

        如果设备没有启用 MSI 或 MSI-X（即仍然在使用传统的引脚中断），则函数返回 false（0）。

    syntax:

    ```c
    #include <linux/pci.h>

    static inline bool pci_msi_enabled(struct pci_dev *pdev);
    ```

    example:

    ```c
    #include <linux/pci.h> // 必须包含这个头文件

    // 假设在你的驱动探测函数中
    static int my_driver_probe(struct pci_dev *pdev, const struct pci_device_id *id)
    {
        int ret;

        // ... 设备初始化、使能等操作 ...

        // 尝试启用MSI中断模式
        ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI);
        if (ret < 0) {
            dev_err(&pdev->dev, "Failed to enable MSI interrupts, using legacy.\n");
            // 通常这里会回退到传统中断
        }

        // 检查设备当前是否使用了MSI
        if (pci_msi_enabled(pdev)) {
            dev_info(&pdev->dev, "Device is using MSI interrupts.\n");
            // 进行MSI模式特有的设置
        } else {
            dev_info(&pdev->dev, "Device is using legacy INTx interrupts.\n");
            // 进行传统中断模式特有的设置
        }

        // ... 其他初始化代码 ...
        return 0;
    }
    ```

* `pci_ioremap_bar()`

    查找 PCI 设备上指定 BAR 的地址空间, 并将该物理地址空间映射到内核虚拟地址空间。这也是一个托管版本，无需驱动程序手动调用 iounmap().

    syntax:

    ```c
    void __iomem *pci_ioremap_bar(struct pci_dev *pdev, int bar);
    ```

    example:

    ```c
    static int my_pci_driver_probe(struct pci_dev *pdev, const struct pci_device_id *id)
    {
        void __iomem *bar0_base;
        int ret;

        // 1. 启用PCI设备（获取总线主控权、分配IRQ等）
        ret = pci_enable_device(pdev);
        if (ret) {
            dev_err(&pdev->dev, "Failed to enable device\n");
            return ret;
        }

        // 2. 请求设备的资源区域（如内存区域）
        ret = pci_request_regions(pdev, "My PCI Driver");
        if (ret) {
            dev_err(&pdev->dev, "Failed to request regions\n");
            goto err_disable;
        }

        // 3. 一键式映射 BAR0
        bar0_base = pci_ioremap_bar(pdev, 0);
        if (!bar0_base) {
            dev_err(&pdev->dev, "Failed to ioremap BAR0\n");
            ret = -ENOMEM;
            goto err_release;
        }

        // 4. 现在可以使用 bar0_base 指针来访问 BAR0 的寄存器了
        // 例如：writel(0xFFFFFFFF, bar0_base + CTRL_REG_OFFSET);

        // ... 驱动的其他初始化操作（如申请中断等）...

        return 0;

    // 错误处理路径
    err_release:
        pci_release_regions(pdev);
    err_disable:
        pci_disable_device(pdev);
        return ret;
    }

    static void my_pci_driver_remove(struct pci_dev *pdev)
    {
        // ... 其他清理工作（如释放中断）...

        // 注意：这里不需要 iounmap(bar0_base)！
        // 内核会自动清理由 pci_ioremap_bar() 创建的映射

        pci_release_regions(pdev);
        pci_disable_device(pdev);
    }
    ```

    在没有 pci_ioremap_bar() 时，你需要这样做：

    ```c
    // 传统繁琐的方法
    bar0_base = pci_resource_start(pdev, 0); // 1. 获取物理地址
    bar0_len = pci_resource_len(pdev, 0);     // 2. 获取长度
    bar0_base = ioremap(bar0_start, bar0_len); // 3. 手动映射
    // 并且在 remove 函数中必须记得： iounmap(bar0_base);
    ```

    该函数主要用于映射 内存空间 类型的 BAR（即 IORESOURCE_MEM）。虽然它也能处理 I/O 空间类型的 BAR（IORESOURCE_IO），但对于 I/O 端口，通常更推荐使用 pci_iomap() 系列函数，或者直接使用 inb()/outb() 等 I/O 端口操作函数。

* pci_iomap() 是一个历史遗留的、为了兼容两种不同IO方式而设计的通用接口，但在当今以MMIO为主流的开发中，更专用的 pci_ioremap_bar() 往往是更好的选择。

* `pci_iomap()`的 example

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/pci.h>

    static struct pci_device_id pci_id_table[] = {
        { PCI_DEVICE(0x1234, 0x11e8) },
        {0,}
    };

    static void *base_addr_bar0;

    static int edu_probe(struct pci_dev *pci_dev, const struct pci_device_id *id) {
        pr_info("in edu_probe()...\n");

        int ret = pci_enable_device(pci_dev);
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to pci enable device, ret: %d\n", ret);
            goto ERR_PCI_ENABLE_DEVICE;
        }

        ret = pci_request_region(pci_dev, 0, "qemu_edu_drv");
        if (ret != 0) {
            dev_err(&pci_dev->dev, "fail to pci request region\n");
            goto ERR_PCI_REQUEST_REGION;
        }

        resource_size_t res_len_bar0 = pci_resource_len(pci_dev, 0);
        base_addr_bar0 = pci_iomap(pci_dev, 0, res_len_bar0);
        if (base_addr_bar0 == NULL) {
            dev_err(&pci_dev->dev, "fail to pci iomap\n");
            goto ERR_PCI_IOMAP;
        }
        return 0;

    ERR_PCI_IOMAP:
        pci_release_region(pci_dev, 0);
    ERR_PCI_REQUEST_REGION:
        pci_disable_device(pci_dev);
    ERR_PCI_ENABLE_DEVICE:
        return -1;
    }

    static void edu_remove(struct pci_dev *pci_dev) {
        pr_info("in edu_remove()...\n");
        pci_iounmap(pci_dev, base_addr_bar0);
        pci_release_region(pci_dev, 0);
        pci_disable_device(pci_dev);
    }

    static struct pci_driver edu_driver = {
        .name = "qemu_edu_drv",
        .id_table = pci_id_table,
        .probe = edu_probe,
        .remove = edu_remove
    };

    int init_mod(void) {
        pr_info("init hlc module...\n");
        int ret = pci_register_driver(&edu_driver);
        if (ret != 0) {
            pr_err("fail to register pci driver\n");
            goto ERR_PCI_REGISTER_DRIVER;
        }
        return 0;

    ERR_PCI_REGISTER_DRIVER:
        return -1;
    }

    void exit_mod(void) {
        pr_info("exit hlc module...\n");
        pci_unregister_driver(&edu_driver);
    }

    module_init(init_mod);
    module_exit(exit_mod);
    MODULE_LICENSE("GPL");
    ```

    dmesg output:

    ```
    [ 9031.003646] init hlc module...
    [ 9031.003763] in edu_probe()...
    [ 9036.304856] exit hlc module...
    [ 9036.304988] in edu_remove()...
    ```

    加载完驱动后执行`sudo cat /proc/iomem | grep edu`, output:

    ```
        fea00000-feafffff : qemu_edu_drv
    ```

    比较关键的四个函数：`pci_enable_device()` -> `pci_request_region()` -> `pci_resource_len()` -> `pci_iomap()`

* `pci_request_region()`中，name 可以填`NULL`，此时`/proc/iomem`中 name 一栏为`??`（未验证）

* `pci_ioremap_wc_bar()`

    将一个 PCI 设备 BAR（基地址寄存器）所指定的 PCI 内存区域映射到内核的虚拟地址空间，并特别请求该映射为“写合并”（Write-Combining, WC）内存类型。

    syntax:

    ```c
    void __iomem *pci_ioremap_wc_bar(struct pci_dev *pdev, int bar);
    ```

    返回值：成功时返回一个 `__iomem` 类型的内核虚拟地址指针，指向映射区域的起始处。失败则返回 `NULL`。

    写合并（WC）: 这是一种弱内存序、高性能的映射模式。CPU 可能会将多个连续的写操作在缓存中“合并”成一个更大的写入事务，再一次性发送到总线上。

    对于帧缓冲区（Frame Buffer）或大量数据传输的设备（如高性能网卡、显卡），使用 WC 映射可以显著减少总线事务数量，极大提升写入带宽和性能。因为写入图像数据通常是连续的，合并后发送效率更高。

* `pci_resource_start()`

    （未验证）

    用于获取 PCI 设备某个资源（如内存地址空间或I/O端口空间）的起始地址。

    syntax:

    ```c
    #include <linux/pci.h>

    resource_size_t pci_resource_start(struct pci_dev *pdev, int bar);
    ```

    参数 pdev：指向目标PCI设备的指针。

    参数 bar：基址寄存器（BAR）的索引号，通常从0到5。

    返回值：一个 resource_size_t 类型（通常是64位或32位整数）的值，表示该资源区域的起始物理地址。

    example:

    ```c
    struct pci_dev *pdev; // 假设已初始化的设备结构体
    int bar = 0;          // 我们想使用第一个BAR
    resource_size_t start, len;
    void __iomem *io_addr; // 指向映射后虚拟地址的指针

    // 1. 获取资源的物理起始地址和长度
    start = pci_resource_start(pdev, bar);
    len = pci_resource_len(pdev, bar);

    // 2. 检查资源是否有效且存在
    if (!start || !len) {
        // 错误处理
    }

    // 3. 将物理地址映射到内核虚拟地址空间
    io_addr = ioremap(start, len);
    if (!io_addr) {
        // 映射失败处理
    }

    // 4. 现在可以通过 io_addr 来读写设备了
    // 例如：writel(0x12345678, io_addr + REG_OFFSET);
    //        value = readl(io_addr + STATUS_REG);

    // 5. (在驱动退出时) 取消映射
    iounmap(io_addr);
    ```

* `pci_resource_len()`

    获取指定 PCI 设备某个资源（如内存区域或 I/O 端口区域）的长度或大小。

    获取指定 bar 空间的字节数。bar 0 ~ bar 5

    syntax:

    ```c
    #include <linux/pci.h>

    resource_size_t pci_resource_len(struct pci_dev *dev, int bar);
    ```

    在调用 pci_resource_len() 或 pci_resource_start() 之前，必须成功调用 pci_enable_device()。否则，获取到的资源信息可能是无效的。

    在 qemu edu dev 上，bar 0 的 len size 为：`1048576`，即 1M。

* `pci_request_regions()`

    一次性申请 PCI 设备的所有有效资源区间（即所有有效的 BARs），是 `pci_request_region()`的“批处理”版本。

    syntax:

    ```c
    int pci_request_regions(struct pci_dev *pdev, const char *res_name);
    ```

    返回值：

    * `0`: 表示所有有效的资源区域都申请成功。

    * 非`0`（错误码）: 表示在申请任何一个BAR时失败。重要的是，如果失败，它会自动释放之前已经成功申请的所有BAR。这简化了错误处理。

    在大多数情况下，应优先使用 pci_request_regions()。除非你明确知道驱动只需要且应该只占用某一个特定BAR，否则使用批量申请更安全、更省事。

* `pci_release_regions()`是`pci_request_regions()`的逆函数。

    syntax:

    ```c
    void pci_release_regions(struct pci_dev *pdev);
    ```

* `pci_iomap()`

    将 pci 设备 bar 空间上的资源映射到内核的内存地址空间里 (内存映射 I/O (MMIO))。

    syntax:

    ```c
    void __iomem *pci_iomap(struct pci_dev *dev, int bar, unsigned long maxlen);
    ```

    * `bar`: 要映射的 BAR 的索引（0-5）。

    * `maxlen`: 想要映射的长度。如果为`0`，则映射整个 BAR 区域。

    这个函数将 pci 设备 bar 指定的物理地址空间映射到内核的虚拟地址空间。如果成功，函数返回一个内核虚拟地址。

    `__iomem`是一个修饰符，提醒程序员这个地址指向的是 I/O 内存，访问它需要使用专门的函数（如 ioread32, iowrite32），而不能直接解引用。

    example:

    ```c
    void __iomem *pci_iomap(struct pci_dev *dev, int bar, unsigned long maxlen);
    ```

    配对函数：`pci_iounmap()`

    `maxlen`有可能超过 bar 允许的长度，所以在映射前，我们最好调用`pci_resource_len(dev, bar)`函数来获取第 bar 个 BAR 的实际长度。如果不动态获取长度，直接映射指定长度的 bar 空间，那么，那么内核会读取 PCI 设备对应的`struct resource`里`->end - ->start + 1`的值，并以此为依据在 mmu 里创建页表。如果驱动访问到了超出合法长度的虚拟地址，那么 mmu 会报 page fault，整个内核有可能崩溃。

## Introduction

### 驱动开发环境的搭建

#### 基于 Ubuntu 和 apt 的驱动开发环境的搭建

如果我们使用 Ubuntu 系统，那么就可以在它编译好的内核库的基础上开发驱动。如果使用其他系统，那么有可能需要自己编译内核。

linux 驱动是以 module 的形式加载到内核中的。Ubuntu 的内核是 signed 版本，我们自己写的 module 是 unsigned 版本，没有办法直接加载（在执行`insmod`时会报错）。目前我的解决办法是，先更新一下内核，换成 unsigned 版本或者 hwe 版本（hwe 版本表示支持最新硬件）：

```bash
apt install linux-image-generic-hwe-22.04
```

可以直接使用

```bash
apt install linux-image-`uname -r`
```

来安装。

接下来我们把原来的 kernel 删掉，然后下载新的：

```bash
sudo apt update && sudo apt upgrade
sudo apt remove --purge linux-headers-*
sudo apt autoremove && sudo apt autoclean
sudo apt-get install linux-headers-`uname -r`
```

注：我们可以用`uname -r`查看当前内核的版本，`uname -a`查看系统的完整版本。

这时候应该可以看见`/lib/modules/xxxxx-generic`路径下（比如`/lib/modules/5.19.0-32-generic/`），有一个`build`文件夹。这个`build`是一个 symbolic link，指向`/usr/src`下对应的内核源码文件夹：

```
hlc@hlc-Ubuntu2204:~/Documents/Projects/kernel_test$ ls -lh /lib/modules/5.19.0-32-generic/
total 6.6M
lrwxrwxrwx  1 root root   40  1月 30 23:44 build -> /usr/src/linux-headers-5.19.0-32-generic
drwxr-xr-x  2 root root 4.0K  2月 18 09:33 initrd
drwxr-xr-x 16 root root 4.0K  2月 18 09:31 kernel
-rw-r--r--  1 root root 1.5M  2月 26 14:45 modules.alias
...
```

如果可以看到对应的内容（即`build`指向`/usr/src/`下的一个目录），那么就说明编译好的内核已经装好了。

#### 基于编译内核的驱动开发环境搭建

1. 下载内核源码

（这部分目前用不到，先不写了）

通常我们说的 linux kernel 可以在

如果使用的是 Ubuntu 系统，下载源码可以在`apt`里下载：

```bash
apt-cache search linux-source
```

使用`uname -r`查看当前内核的版本，找到对应的版本。

编译内核需要用到`flex`, `bison`, `libssh-dev`, `libelf-dev`

1. 编译内核（在 Ubuntu 22.04 下编译）

    一个写得还不错的 tutorial，可以参考一下：<https://phoenixnap.com/kb/build-linux-kernel>

    编译内核需要一些额外的工具，可以参考这个网页<https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel>装一下：

    ```bash
    sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm
    ```

    下载好源码后，首先需要创建一个`.config`文件。我们可以使用 Ubuntu 系统里现成的：

    如果使用这种方法后编译失败，通常是证书设置的问题，参考这个：<https://askubuntu.com/questions/1329538/compiling-the-kernel-5-11-11>

    在`.config`里把`CONFIG_SYSTEM_TRUSTED_KEYS`和`CONFIG_SYSTEM_REVOCATION_KEYS`都设置成空字符串。

    也可以创建一个默认的：

    ```bash
    make menuconfig
    ```

    这种方法会在内核源码目录创建一个`.config`文件。使用这个 config 编译起来比较快，可能是因为有点选项没选上吧。

    单核编译可以用`make`，多线程编译可以使用`make -j16`（使用 16 线程编译）。

    然后`sudo make modules_install`

    最后`sudo make install`

### hello, world 驱动

为了测试上面搭建的驱动开发环境是否成功，我们使用一个 hello world 项目测试一下。

首先，创建一个项目文件夹：`mkdir driver_test`，然后进入这个目录：`cd driver_test`。

接着，创建一个`hello_world.c`文件，然后写入以下内容：

```c
#include <linux/init.h>
#include <linux/module.h>

int hello_init(void)
{
    printk(KERN_INFO "hello my module\n");
    return 0;
}

void hello_exit(void)
{
    printk(KERN_INFO "bye bye!\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

然后我们创建一个`Makefile`文件：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-41-generic  # 这里要和我们下载的内核版本保持一致
obj-m  +=  hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

最后进行编译：`make`。编译完成后，我们可以在当前目录下看到`hello_world.ko`文件。

运行：`sudo insmod hello_world.ko`

调出 log，检查是否运行成功：`sudo dmesg`

如果输出的最后几行有`[ 2793.700004] <1>hello my module`，那么就说明驱动运行成功了。

最后卸载驱动：`sudo rmmod hello_world`。

### vscode config

* linux kernel module 开发 vscode 的配置

    `c_cpp_properties.json`:

    ```json
    {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/**",
                    "/usr/src/linux-headers-6.5.0-18-generic/include/",
                    "/usr/src/linux-headers-6.5.0-18-generic/arch/x86/include/generated/",
                    "/usr/src/linux-hwe-6.5-headers-6.5.0-18/arch/x86/include/",
                    "/usr/src/linux-hwe-6.5-headers-6.5.0-18/include"
                ],
                "defines": [
                    "KBUILD_MODNAME=\"hello\"",
                    "__GNUC__",
                    "__KERNEL__",
                    "MODULE"
                ],
                "compilerPath": "/usr/bin/gcc",
                "cStandard": "gnu17",
                "cppStandard": "c++17",
                "intelliSenseMode": "linux-gcc-x64"
            }
        ],
        "version": 4
    }
    ```

    `includePath`里新增的 include path 和`defines`里的四个宏，任何一个都不能少，不然 vscode 就会在代码里划红线报错。

    下面是一个没有报错的 example code:

    `hello.c`:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/ktime.h>

    int m_int = 5;
    module_param(m_int, int, S_IRUSR | S_IWUSR);

    int hello_init(void)
    {
        printk(KERN_INFO "hello my module\n");
        struct timespec64 ts64;
        ktime_get_ts64(&ts64);
        time64_t seconds = ktime_get_real_seconds();
        long nanoseconds = ts64.tv_nsec;
        printk(KERN_INFO "on init, current time: %ld seconds\n", seconds);
        return 0;
    }

    void hello_exit(void)
    {
        printk(KERN_INFO "bye bye!\n");
        struct timespec64 ts64;
        ktime_get_ts64(&ts64);
        time64_t seconds = ts64.tv_sec;
        long nanoseconds = ts64.tv_nsec;
        printk(KERN_INFO "on exit, current time: %ld seconds\n", seconds);
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

### 交叉编译

（有时间了填下这个坑）

交叉编译：`sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi-`

## 内核模块

* 内核模块编程注意事项：

    * 不能使用 C 库和 C 标准头文件

    * 使用 GNU C （在 ANSI C 上加了些语法）

    * 没有内存保护机制

    * 不能处理浮点运算

    * 注意并发互斥和可移植性

### 内核模块的加载与卸载

* 内核模块的加载与卸载

    example:

    `hello_world.c`：

    ```c
    #include <linux/init.h>
    #include <linux/module.h>

    int hello_init(void) {
        printk(KERN_INFO "hello my module\n");
        return 0;
    }

    void hello_exit(void) {
        printk(KERN_INFO "bye bye!\n");
    }

    module_init(hello_init);  // 这一行和下一行的分号都不是必要的，为什么？
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");  // 不加这行的话，无法通过编译。MODULE_LICENSE 必须大写，不然无法通过编译。这一行末尾的分号是必要的，为什么？
    ```

    其中加载函数（init function）与制裁函数（exit function）的原型为：

    ```c
    static int __init hello_init(void) {
        return 0;
    }

    void __exit hello_exit(void) {

    }

    module_init(hello_init);  // 使用宏来注册加载函数
    module_exit(hello_exit);  // 使用宏来注册卸载函数
    ```

    init function 返回 0 表示加载成功。

    说明：

    1. `hello_init()`与`hello_exit()`参数列表中的`void`不可省略，不然无法通过编译.

* 在内核模块中添加额外信息

    ```c
    #include<linux/kernel.h>  // 这个头文件有什么用？
    #include<linux/init.h>
    #include<linux/module.h>

    static int __init hello_world_init(void) {
        printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
        return 0;
    }

    static void __exit hello_world_exit(void) {
        printk(KERN_INFO "Kernel Module Removed Successfully...\n");
    }
     
    module_init(hello_world_init);
    module_exit(hello_world_exit);
     
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("A simple hello world driver");
    MODULE_VERSION("2:1.0");
    ```

* makefile 写法

    `Makefile`：

    ```Makefile
    KERNEL_DIR=/usr/src/linux-headers-5.19.0-32-generic
    obj-m += hello_world.o
    default:
    	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
    ```

    另外一个 makefile 的 example:

    ```Makefile
    obj-m += hello_world.o
     
    ifdef ARCH
      KDIR = /home/embetronicx/BBG/tmp/lib/modules/5.10.65/build
    else
      KDIR = /lib/modules/$(shell uname -r)/build  # 为什么要在前面加上 shell
    endif
     
    all:
      make -C $(KDIR)  M=$(shell pwd) modules  # $(PWD) 和 $(shell pwd) 有什么不同
     
    clean:
      make -C $(KDIR)  M=$(shell pwd) clean  # KDIR 是个变量，为什么要给它加上 $() ？
    ```

    然后在当前文件夹下运行`make`，会生成`hello_world.ko`文件。这个文件就是我们需要的内核模块文件，ko 代表 kernel object。

* 加载内核模块

    此时可以使用`sudo insmod hello_world.ko`插入模块，使用`sudo rmmod hello_world`移除模块，`sudo lsmod`查看已经加载的模块，`sudo dmesg`查看日志输出。

    （如果还是无法`insmod`，或许需要取消 secure boot：<https://askubuntu.com/questions/762254/why-do-i-get-required-key-not-available-when-install-3rd-party-kernel-modules>）

* 模块信息

    除了添加`MODULE_LICENSE()`外，可选的添加信息有：

    * `MODULE_AUTHOR`: 模块作者

        Example: `MODULE_AUTHOR("hlc")`

    * `MODULE_VERSION`: 模块版本
      
        Example: `MODULE_VERSION("1.0")`

    * `MODULE_DESCRIPTION`：模块描述

        Example: `MODULE_DESCRIPTION("this is my first module")`

    获得模块的一些信息：`modinfo hello_world_module.ko`

### 日志消息打印

* `printk()`

    内核模块中可以使用函数`printk()`将消息打印到日志中，用法和`printf()`几乎相同。

    syntax:

    ```c
    printk("hello my module\n");
    ```

    `printk()`如果不加`\n`，则不会在`dmesg`中立即刷新。

    `printk()`可打印的消息有不同的级别，我们可以在字符串前使用下面的宏字符串进行修饰：

    * `KERN_EMERG`: Used for emergency messages, usually those that precede a crash.

    * `KERN_ALERT`: A situation requiring immediate action.

    * `KERN_CRIT`: Critical conditions are often related to serious hardware or software failures.

    * `KERN_ERR`: Used to report error conditions; device drivers often use KERN_ERR to report hardware difficulties.

    * `KERN_WARNING`: Warnings about problematic situations that do not, in themselves, create serious problems with the system.

    * `KERN_NOTICE`: Situations that are normal, but still worthy of note. A number of security-related conditions are reported at this level.

    * `KERN_INFO`: Informational messages. Many drivers print information about the hardware they find at startup time at this level.

    * `KERN_DEBUG`: Used for debugging messages.

    Example:

    ```c
    printk(KERN_INFO "this is a info level log");
    printk(KERN_WARNING "this is a warning level log");
    printk("this is a non-level log")
    ```

    经实际测试，如果不加日志级别，那么为白色粗字，`KERN_NOTICE`及以下，全都是正常白字，`KERN_WARNING`的消息会白字加粗，`KERN_ERR`的消息会变成红字，`KERN_CRIT`消息为红字加粗，`KERN_ALERT`为红底黑字，`KERN_EMERG`又变成正常白字。（这个颜色可能和 terminal 配色有关）

* `pr_xxxx()`系列

    In the newer Linux kernels, you can use the APIs below instead of this `printk`.

    * `pr_info()` – Print an info-level message. (ex. `pr_info("test info message\n")`).
    * `pr_cont()` – Continues a previous log message in the same line.
    * `pr_debug()` – Print a debug-level message conditionally.
    * `pr_warn()` – Print a warning-level message.
    * `pr_err()` – Print an error-level message. (ex. `pr_err(“test error message\n”)`).

    经实测，`pr_info()`为正常白字，`pr_cont()`为白字加粗，`pr_debug()`没有输出，`pr_warn()`为白字加粗，`pr_err()`为红字。

* 默认打印级别

    上面的日志级别可以对应到数字 0 - 7，如果不指定日志级别，那么就是无级别。之所以映射到数字，似乎是因为可以使用数字控制 console 的输出级别。（但是经实际测试，好像不怎么有用）

    可以参考这几个网站的资料：

    1. <http://www.jyguagua.com/?p=708>

    2. <https://blog.csdn.net/qustDrJHJ/article/details/51382138>

    我们可以使用`cat /proc/sys/kernel/printk`查看当前日志的级别。

    第一个数为内核默认打印级别，只有当`printk`的打印级别高于内核默认打印级别时，`printk`打印的信息才能显示 。

    第二个数为`printk()`的默认打印级别。

    修改内核的默认打印级别（即修改第 1 个数）：

    `echo 5 > /proc/sys/kernel/printk`

    （现在好像已经升级了，不管 printk level 大于还是小于当前 console 的 level，都不会在 console 上输出）

### 模块参数（Module Parameters）

* a simple example

    ```c
    int param_int = 10;
    unsigned short param_ushort = 20;
    char *param_string = "hello";
    int param_arr[3] = {100, 200, 300};

    module_param(param_int, int, 0775);
    module_param(param_ushort, ushort, 0);
    module_param(param_string, charp, 0644);
    module_param_array(param_arr, int, NULL, 0755);

    int modparam_init(void)  // 这是一个普通函数，函数的名字可以随便改
    {
        pr_info("param_int = %d\n", param_int);
        pr_info("param_ushort = %hu\n", param_ushort);
        pr_info("param_string = %s\n", param_string);
        pr_info("param_arr = %d %d %d\n", param_arr[0], param_arr[1], param_arr[2]);
        return 0;
    }
    ```

    说明：

    1. 这里的`0775`并不是和 linux 文件权限一一对应。使用`0775`作为权限后，得到的参数文件的权限如下所示：

        ```
        -rw-rw-r-- 1 root root 4096  5月 16 10:52 /sys/module/hello_world/parameters/a
        ```

        因为这个数字和 linux 文件的权限并不是对应关系，所以使用`0776`，`0777`，`777`等作为参数时会编译报错。

        正常情况下还是使用`S_IWUSR`这些标志位吧。

    2. `0775`前面这个`0`必须加上，不然会编译报错。目前不清楚是为什么。

        * [2024.06.29] 可能是因为 755 是八进制，C 语言里表示八进制数需要在前加上 0

            不加 0 表示的是 10 进制数，那就表示七百七十五了，二进制肯定和 0775 不一样。

    3. `unsigned short`定义的变量，在`module_param()`中注册模块参数时，必须使用`ushort`作为类型。

        在`module_param()`中填`unsigned short`会编译报错。

        如果使用`typedef unsigned short us;`，然后在`module_param()`中填`us`，同样也会编译报错。

    4. 如果数组没有被初始化，或初始化的元素数量不够，那么元素的默认值都是 0。

* `module_param();`

    这种函数被称为 Module Parameters Macro。

    `module_param(name, type, perm);`

    `module_param(模块参数名，模块参数，访问权限);`

    This macro is used to initialize the arguments. `module_param` takes three parameters: the name of the variable, its type, and a permissions mask to be used for an accompanying sysfs entry.

    The macro should be placed outside of any function and is typically found near the head of the source file. `module_param()` macro, defined in `linux/moduleparam.h`.

    Parameters:

    * `type`

        可以是下面几个之一：`byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `charp`, `bool`, `invbool`;

        Numerous types are supported for module parameters:

        * `bool`

            A boolean (true or false) value (the associated variable should be of type int).

        * `invbool`

            The invbool type inverts the value, so that true values become false and vice versa.

        * `charp`

            A char pointer value. Memory is allocated for user-provided strings, and the pointer is set accordingly.

        * `int`, `long`, `short`, `uint`, `ulong`, `ushort`
            
            Basic integer values of various lengths. The versions starting with `u` are for unsigned values.

    * `perm` is the usual permissions value.

        There are several types of permissions:

        这些宏定义在`stat.h`中

        * `S_IWUSR`, `S_IRUSR`, `S_IXUSR`
        * `S_IRGRP`, `S_IWGRP`, `S_IXGRP`
        * `S_IROTH`, `S_IWOTH`, `S_IXOTH`

        可以看出来，`S_I`是一个 common prefix，R = read, W = write, X = Execute. USR = user, GRP = Group。

        Using `|` (OR operation) we can set multiple permissions at a time.

        在使用`S_IROTH`和`S_IWOTH`时会编译时报错，但是如果使用`0775`，可以顺序地给 others 加上`r`权限。不清楚为什么。

    Example:

    `module_param(valueETX, int, S_IWUSR | S_IRUSR);`

* `module_param_array();`

    `module_param_array(name, type, int *num, permissions);`

    `module_param_array(数组模块参数名，数组元素类型，NULL，访问权限);`

    This macro is used to send the array as an argument to the Linux device driver.

    Parameters:

    * `name`
    
        The name of the array (and of the parameter)

    * `type`
    
        The type of the array elements

    * `num`

        An integer variable (optional) otherwise `NULL`。

        在命令行中传递的数组的元素个数。

        比如`sudo insmod hello_world.ko m_arr=3,4,5`，那么`num`会被改写成`3`。

在代码中对模块参数的使用和普通变量没有区别。

Examples:

**在命令行中传递模块参数**

```bash
insmod hello_abc.ko param_int=50
insmod hello_abc.ko param_string="hello world"
insmod mod_param.ko param_arr=111,222,333
```

数组可以只传递一部分元素，此时会重新计算数组的长度，并重新赋值。比如在代码中定义了数组容量为`3`，并且初始化了所有元素，但是在命令行中只传递了 2 个元素的数组，那么`/sys/module/xxx/parameters`中的对应文件也只会显示 2 个元素。

如果命令行传递的元素的数量超出数组的容量，那么会报错。

当模块加载成功后，那些访问权限非 0 的模块会在以下路径下：

`/sys/module/模块名/parameters`

存在和模块参数名同名的文件，这些文件的权限来自于模块参数的权限。文件的内容与模块参数的值相同。因此可以通过修改文件中保存的数据，对模块参数进行修改：

```bash
sudo su
echo 1 > /sys/module/hello_world_module/parameters/my_param
```

**监测模块参数的改变**

`module_param_cb()`

This macro is used to register the callback. Whenever the argument (parameter) got changed, this callback function will be called.

为了注册 callback，我们首先需要填写一个结构体：

```c
struct kernel_param_ops 
{
    int (*set)(const char *val, const struct kernel_param *kp);
    int (*get)(char *buffer, const struct kernel_param *kp);
    void (*free)(void *arg);
};
```

（不明白这里的`free`有什么用）

Example:

```c
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/module.h>
#include<linux/moduleparam.h>
 
int valueETX, arr_valueETX[4];
char *nameETX;
int cb_valueETX = 0;
 
module_param(valueETX, int, S_IRUSR|S_IWUSR);                      // integer value
module_param(nameETX, charp, S_IRUSR|S_IWUSR);                     // String
module_param_array(arr_valueETX, int, NULL, S_IRUSR|S_IWUSR);      // Array of integers
 
/*----------------------Module_param_cb()--------------------------------*/
int notify_param(const char *val, const struct kernel_param *kp)
{
    // param_set_int 好像是把字符串转换成 int
    int res = param_set_int(val, kp); // Use helper for write variable
    if(res==0) {
        printk(KERN_INFO "Call back function called...\n");
        printk(KERN_INFO "New value of cb_valueETX = %d\n", cb_valueETX);
        return 0;
    }
    return -1;
}
 
const struct kernel_param_ops my_param_ops = 
{
    .set = &notify_param, // Use our setter ...
    .get = &param_get_int, // .. and standard getter
};
 
module_param_cb(cb_valueETX, &my_param_ops, &cb_valueETX, S_IRUGO|S_IWUSR );
/*-------------------------------------------------------------------------*/

/*
** Module init function
*/
static int __init hello_world_init(void)
{
    int i;
    printk(KERN_INFO "ValueETX = %d  \n", valueETX);
    printk(KERN_INFO "cb_valueETX = %d  \n", cb_valueETX);
    printk(KERN_INFO "NameETX = %s \n", nameETX);
    for (i = 0; i < (sizeof arr_valueETX / sizeof (int)); i++) {
            printk(KERN_INFO "Arr_value[%d] = %d\n", i, arr_valueETX[i]);
    }
    printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

/*
** Module Exit function
*/
static void __exit hello_world_exit(void)
{
    printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple hello world driver");
MODULE_VERSION("1.0");
```

加载模块：

```bash
sudo insmod hello_world_module.ko valueETX=14 nameETX="EmbeTronicX" arr_valueETX=100,102,104,106
```

改变模块参数值：

方法一：

```bash
sudo sh -c "echo 13 > /sys/module/hello_world_module/parameters/cb_valueETX"
```

方法二：

Type sudo su. Then enter the password if it asks. Then do `echo 13 > /sys/module/hello_world_module/parameters/cb_valueETX`

然后我们可以在`dmesg`里看到参数值变化的消息：

```
[ 1688.610775] ValueETX = 14  
[ 1688.610782] cb_valueETX = 0  
[ 1688.610784] NameETX = EmbeTronicX 
[ 1688.610785] Arr_value[0] = 100
[ 1688.610786] Arr_value[1] = 102
[ 1688.610787] Arr_value[2] = 104
[ 1688.610788] Arr_value[3] = 106
[ 1688.610789] Kernel Module Inserted Successfully...
[ 1849.370708] Call back function called...
[ 1849.370714] New value of cb_valueETX = 13
[ 1880.687099] Kernel Module Removed Successfully...
```

### 模块符号的导出

模块导出符号可以将模块中的变量/函数导出，供内核其他模块使用。

内核中提供了相应的宏来实现模块的导出:

```
EXPORT_SYMBOL
EXPORT_SYMBOL_GPL  (只有遵循 GPL 协议的代码才可以使用)
```

Example:

```c
#include <linux/init.h>
#include <linux/module.h>

int add(int a, int b)
{
    return a + b;
}

int mul(int a, int b)
{
    return a * b;
}

EXPORT_SYMBOL(add);
EXPORT_SYMBOL_GPL(mul);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("hlc");
MODULE_VERSION("1.0");
MODULE_DESCRIPTION("this is a module symbol!");
// ...
```

在别的模块中使用时，需要这样写：

```c
extern int add(int a, int b);
extern int mul(int a, int b);
```

`extern`表示函数是在外部实现的，不是在本文件中实现的。使用这些函数时，需要先加载他们所在的模块。

在 Makefile 中，应该把两个`.o`文件都写上：

```Makefile
obj-m += xxx.o xxx_2.o
```

在加载模块的时候，应该先加载导出符号的模块，再加载使用符号的模块。卸载时，顺序要相反。

## 设备驱动

### cache

* 通过代码测试设备文件

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>

    int main()
    {
        char ch = 0;
        char buf[10] = {};

        int fd = open("/dev/hello_dev", O_RDWR);
        if (fd == -1)
        {
            perror("open");
            exit(-1);
        }

        printf("open successed! fd = %d\n", fd);

        while (1) {
            ch = getchar();
            getchar();

            if (ch == 'q')
                break;
            switch(ch)
            {
                case 'r':
                    read(fd, buf, 0);
                    break;
                case 'w':
                    write(fd, buf, 0);
                    break;
                default:
                    printf("error input\n");
            }
        }
    }
    ```

* 合并注册设备号和注册 cdev:

    ```c
    int register_chrdev(unsigned int major, const char *name, const struct file_operations *fops);
    ```

    当打开一个设备文件时，kernel 会根据设备号遍历 cdev 数组，找到对应的 cdev 结构体对象，然后把里面的`file_operations`里面的函数指针赋值给文件结构体`struct file`的`file_operations`里对应的函数。

    如果`major`填 0，那么返回值为自动分配的 major。

### note

* linux 设备类型

    常见的有字符设备，块设备和网络设备。

    1. 字符设备

        按字节流访问，一般是按顺序访问

        绝大多数设备都是字符设备。比如 led，按键，串口，传感器，LCD

        字符设备的驱动通过字符设备文件来访问

    1. 块设备

        按数据块访问，块的大小固定，通常是 4k，具有随机访问能力

        内存，磁盘，SD卡，U盘

        块设备驱动通过块设备文件来访问

    1. 网络设备

        一般只代表网卡设备。

        驱动实现要结合网络协议栈（TCP/IP）

        访问网络设备不通过文件，通过套接字（网络通信地址）访问

    In fact, all device drivers that are neither storage nor network device drivers are some type of character driver.

    字符设备驱动的访问：

    驱动是沟通硬件和上层应用的媒介，字符设备驱动通过字符设备文件来访问，Linux 中所有的设备文件存放在`/dev`中，在用户层访问设备文件和普通文件的方法是没有区别的。Linux 操作系统实际上是通过设备号来找到对应的字符设备驱动（怎么找？）。

    一个设备文件需要实现和普通文件相同的方法：

    `open, close, read, write, lseek, ioctl, mmap, stat`

* 设备号

    设备号用 32 位的一个`dev_t`类型的变量来表示（无符号整型），高 12 位表示主设备号，后 20 位表示次设备号。

    The `dev_t` type (defined in `<linux/types.h>`) is used to hold device numbers—both the major and minor parts. `dev_t` is a 32-bit quantity with 12 bits set aside for the major number and 20 for the minor number.

    主设备号用来区分不同类型的设备，次设备号用于区分设备的实例。

    在`/proc/devices`文件中可以查找到设备号与对应的设备类型。

    内核中提供了操作设备号的宏：

    ```c
    MAJOR(设备号);  // 通过设备号获取主设备号  MAJOR(dev_t dev);
    MINOR(设备号);  // 通过设备号获取次设备号  MINOR(dev_t dev);
    MKDEV(主设备号, 次设备号);  // 通过主设备号和次设备号构造设备号  MKDEV(int major, int minor);
    ```

    这些宏都是位运算，有空可以看看。

* 申请设备号

    设备号在内核中属于资源，需要向内核申请。

    syntax:

    ```c
    #include <linux/fs.h>

    int alloc_chrdev_region(dev_t *dev, unsigned baseminor, unsigned count, const char *name);
    ```

    alloc_chrdev_region - register a range of char device numbers

    params:

    * `dev`: 设备号的地址

    * `baseminor`: 起始次设备号

    * `count`: 设备号个数

    * `name`: device region 的名称

    Return value:

    Returns zero or a negative error code.
    
    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/fs.h>

    dev_t dev_region;
    const char *dev_region_name = "hlc dev region";

    int mod_init(void) {
        printk(KERN_INFO "in mod_init() ...\n");
        int rtv = alloc_chrdev_region(&dev_region, 0, 1, dev_region_name);
        if (rtv != 0) {
            pr_info("fail to alloc_chrdev_region(), ret: %d\n", rtv);
        }
        printk(KERN_INFO "successfully allocate device region. major: %d, minor: %d\n",
            MAJOR(dev_region), MINOR(dev_region));
        return 0;
    }
    ```
    
* 静态申请设备号

    有两种申请方式，一种是静态申请（Statically allocating），一种是动态申请（Dynamically Allocating）。

    我们可以使用`register_chrdev_region()`静态申请设备号。

    Syntax:

    ```c
    #include <linux/fs.h>
    register_chrdev_region(dev_t from, unsigned count, const char *name);
    ```

    Params:

    * `from`: 要申请的起始设备号

    * `count`: 设备数量

        `count` is the total number of contiguous device numbers you are requesting. Note that, if the count is large, the range you request could spill over to the next major number; but everything will still work properly as long as the number range you request is available.

    * `name`: 设备号在内核中对应的名称

    Return value:

    返回 0 表示成功，返回非 0 表示失败。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/fs.h>

    dev_t dev = MKDEV(220, 0);

    int hlc_mod_init(void)
    {
        printk(KERN_INFO "load my module\n");

        // allocate a device number
        register_chrdev_region(dev, 1, "hlc_dev");
        printk(KERN_INFO "hlc dev, major = %d, minor = %d\n", MAJOR(dev), MINOR(dev));
        return 0;
    }
    ```

    上述代码中，使用`MKDEV(220, 0)`构造了一个设备号。构造方法是，首先选择一个内核中未被使用的主设备号（`cat /proc/devices`），比如`220`。然后根据设备个数分配次设备号，一般从`0`开始。

    `register_chrdev_region()`用于静态申请设备号。这个函数运行成功后，可以使用`cat /proc/devices`看到注册成功的设备号名称`220 hlc_dev`。

**注销设备号**

不再使用设备号需要注销：

```c
void unregister_chrdev_region(dev_t from, unsigned count);
```

header file: `<linux/fs.h>`

params:

* `from`: 要注销的起始设备号

* `count`: 设备号的个数

一般在卸载模块的时候释放设备号。The usual place to call `unregister_chrdev_region` would be in your module’s cleanup function (Exit Function).

Example:

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>

dev_t dev_region;
const char *dev_region_name = "hlc dev region";

int mod_init(void)
{
    printk(KERN_INFO "in mod_init() ...\n");
    alloc_chrdev_region(&dev_region, 0, 1, dev_region_name);
    printk(KERN_INFO "allocate device region.\n");
    return 0;
}

void mod_exit(void)
{
    printk(KERN_INFO "in mod_exit() ...\n");
    unregister_chrdev_region(dev_region, 1);
    printk(KERN_INFO "unregistered device region.\n");
}

module_init(mod_init);
module_exit(mod_exit);
MODULE_LICENSE("GPL");
```

output:

```
[ 6790.673894] in mod_init() ...
[ 6790.673896] allocate device region.
[ 6796.788756] in mod_exit() ...
[ 6796.788764] unregistered device region.
```

**Difference between static and dynamic method**

A static method is only really useful if you know in advance which major number you want to start with. With the Static method, you are telling the kernel that, what device numbers you want (the start major/minor number and count) and it either gives them to you or not (depending on availability).

With the Dynamic method, you are telling the kernel that how many device numbers you need (the starting minor number and count) and it will find a starting major number for you, if one is available, of course.

Partially to avoid conflict with other device drivers, it’s considered preferable to use the Dynamic method function, which will dynamically allocate the device numbers for you.

The disadvantage of dynamic assignment is that you can’t create the device nodes in advance, because the major number assigned to your module will vary. For normal use of the driver, this is hardly a problem, because once the number has been assigned, you can read it from /proc/devices.

注：

1. 这个资源的名字叫设备号（device number），但是相关的函数却都是 device region 相关。

    是不是 device number 有歧义，一方面表示设备号，一方面又表示设备的个数，所以把改了？

    然后设备号的类型还是`dev_t`，有点像 device type。既不含 number 信息，也不含 region 信息，还容易和后面的`device`类型弄混。不清楚为什么要这么起名，可能是为了向上兼容吧。

### cdev 设备驱动

cdev 在内核中代表一个字符设备驱动。(char device)

```c
struct cdev {
    struct kobject kobj;
    struct module *owner;
    const struct file_operations *ops;  // 驱动操作函数集合
    struct list_head list;
    dev_t dev;  // 设备号
    unsigned int count;
};
```

### 设备文件

**Manually Creating Device File**

We can create the device file manually by using `mknod`.

`mknod -m <permissions> <name> <device type> <major> <minor>`

* `<name>` – your device file name that should have a full path (/dev/name)

* `<device type>` – Put c or b

    * `c` – Character Device

    * `b` – Block Device

* `<major>` – major number of your driver

* `<minor>` – minor number of your driver

* `-m <permissions>` – optional argument that sets the permission bits of the new device file to permissions

Example: `sudo mknod -m 666 /dev/etx_device c 246 0`

然后我们创建一个设备文件：`sudo mknod /dev/hello_dev c 220 0`。这里的主设备号和次设备号要和前面代码中的保持一致。

最后我们进行测试：`cat /dev/hello_dev`，然后执行`sudo smesg`，可以看到一些输出：

```
[26495.940998] <1>hello my module
[26502.305035] in m_open function ...
[26502.305042] in m_read function ...
[26502.305048] in m_release function ...
```

可以看到驱动正常运行。

**Automatically Creating Device File**

The automatic creation of device files can be handled with `udev`. `udev` is the device manager for the Linux kernel that creates/removes device nodes in the `/dev` directory dynamically. Just follow the below steps.

1. Include the header file `linux/device.h` and `linux/kdev_t.h`

2. Create the struct `class`

3. Create `device` with the `class` which is created by the above step

example:

`hello.c`

```cpp
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>

dev_t dev_id;  // 设备号
struct cdev cdev;  // cdev handle
const char *cdev_name = "hlc_dev";
struct class *dev_cls;  // device class, registered at /sys/class
struct device *dev;  // device file handle

int m_open(struct inode *inode, struct file *file_ptr)
{
    printk(KERN_INFO "in m_open()...\n");
    return 0;
}

int m_release(struct inode *inod, struct file *file_ptr)
{
    printk(KERN_INFO "in m_release()...\n");
    return 0;
}

ssize_t m_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset)
{
    printk(KERN_INFO "in m_read()...\n");
    return 0;
}

ssize_t m_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    printk(KERN_INFO "in m_write()...\n");
    return 0;
}

ssize_t m_ioctl(struct file *file_ptr, unsigned int, unsigned long)
{
    printk(KERN_INFO "in m_ioctl()...\n");
    return 0;
}

struct file_operations m_ops = {
    .owner = THIS_MODULE,
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write,
    .unlocked_ioctl = m_ioctl
};

int hlc_module_init(void)
{
    printk(KERN_INFO "init hlc module\n");
    dev_id = MKDEV(255, 0);
    register_chrdev_region(dev_id, 1, "hlc cdev driver");
    cdev_init(&cdev, &m_ops);
    cdev_add(&cdev, dev_id, 1);
    dev_cls = class_create("hlc_dev_cls");
    if (IS_ERR(dev_cls)) {
        printk(KERN_INFO "fail to create device class.\n");
    }
    dev = device_create(dev_cls, NULL, dev_id, NULL, "hlc_dev");
    if (IS_ERR(dev)) {
        printk(KERN_INFO "fail to create device.\n");
    }
    return 0;
}

void hlc_module_exit(void)
{
    printk(KERN_INFO "exit hlc module!\n");
    device_destroy(dev_cls, dev_id);
    class_destroy(dev_cls);
    cdev_del(&cdev);
    unregister_chrdev_region(dev_id, 1);
}

module_init(hlc_module_init);
module_exit(hlc_module_exit);
MODULE_LICENSE("GPL");
```

`Makefile`:

```makefile
KERNEL_DIR=/usr/src/linux-headers-6.5.0-28-generic
obj-m  +=  hello.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	rm -f *.mod *.o *.order *.symvers *.cmd
```

compile:

```bash
make
```

run:

```bash
sudo insmod hello.ko
```

`dmesg` output:

```
[ 4976.473176] init hlc module
```

`ls /sys/class/ | grep hlc` output:

```
hlc_dev_cls
```

`ls /dev/ | grep hlc` output:

```
hlc_dev
```

run `sudo bash -c "cat /dev/hlc_dev"`, then `dmesg` output:

```
[ 5021.619227] in m_open()...
[ 5021.619251] in m_read()...
[ 5021.619269] in m_release()...
```

exit:

```bash
sudo rmmod hello
```

Explanation:

* Create the class

    This will create the struct class for our device driver. It will create a structure under `/sys/class/`. 创建的设备类在`/sys/class`目录下。

    `struct class * class_create(struct module *owner, const char *name);`

     ```c
    struct class *class_create(模块所有者, const char *name);
    ```

    * `owner` – pointer to the module that is to “own” this struct class

    * `name` – pointer to a string for the name of this class

    This is used to create a struct class pointer that can then be used in calls to class_device_create. The return value can be checked using IS_ERR() macro.

    销毁设备类：

    ```c
    void class_destroy(struct class *cls);
    ```

* `device_create`

    创建设备文件（设备节点）

    Syntax:

    ```c
    struct device *device_create(struct *class, struct device *parent, dev_t dev, void * drvdata, const char *fmt, ...);
    ```

    Parameters:

    * `class` – pointer to the struct class that this device should be registered to

    * `parent` – pointer to the parent struct device of this new device, if any

        父设备指针

    * `devt` – the dev_t for the char device to be added

        设备号

    * `drvdata` – the data to be added to the device for callbacks

        额外的数据

    * `fmt` – string for the device’s name

        设备文件名

    * `...` – variable arguments

    A “dev” file will be created, showing the `dev_t` for the device, if the `dev_t` is not `0,0`. If a pointer to a parent struct device is passed in, the newly created struct device will be a child of that device in sysfs. The pointer to the struct device will be returned from the call. Any further sysfs files that might be required can be created using this pointer. The return value can be checked using IS_ERR() macro.

    成功会在`/dev`目录下生成对应的设备文件，并返回设备指针

* `device_destroy`

    销毁设备文件

    Syntax:

    ```c
    void device_destroy(struct class *class, dev_t devt)
    ```

一个更加简洁的版本（没有添加错误处理，以及函数是否正常运行的判断）：

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>

dev_t dev_num;
struct cdev my_cdev;
struct class *dev_cls;
struct device *dev;

int m_open(struct inode *inode, struct file *file);
int m_release(struct inode *inode, struct file *file);
ssize_t m_read(struct file *file, char __user *buf, size_t size, loff_t *offset);
ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset);

struct file_operations fops = {
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write
};

int init_mod(void)
{
    pr_info("in init_mod()...\n");
    alloc_chrdev_region(&dev_num, 0, 1, "hlc_dev_num");
    cdev_init(&my_cdev, &fops);
    cdev_add(&my_cdev, dev_num, 1);
    dev_cls = class_create("hlc_dev_cls");
    dev = device_create(dev_cls, NULL, dev_num, NULL, "hlc_dev");
    pr_info("init hlc module done.\n");
    return 0;
}

void exit_mod(void)
{
    pr_info("in exit_mod()...\n");
    device_destroy(dev_cls, dev_num);
    class_destroy(dev_cls);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("exit hlc module done.\n");
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");

int m_open(struct inode *inode, struct file *file)
{
    pr_info("in m_open()...\n");
    return 0;
}

int m_release(struct inode *inode, struct file *file)
{
    pr_info("in m_release()...\n");
    return 0;
}

ssize_t m_read(struct file *file, char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in m_read()...\n");
    return 0;
}

ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in m_write()...\n");
    return 0;
}
```

### user mode program for device file

### cache

* ioctl 接口

    ioctl 是一个专用于硬件操作的接口，用于和实际数据传输相区分

    1. 用户层接口

        ```c
        #include <sys/ioctl.h>

        int ioctl(int fd, unsigned long request, ...);
        ```

        * `fd`: 文件描述符

    1. 内核接口

        头文件：

        ```c
        #include <linux/ioctl.h>
        ```

        对应`file_operations`中的成员：

        ```c
        long (*unlocked_ioctl) (struct file *filp, unsigned int cmd, unsigned long data);
        ```

    命令构造：

    Linux 内核提供构造 ioctl 命令的宏：

    ```c
    #define HELLO_ONE _IO('k',0)
    #define HELLO_TWO _IO('k',1)

    long cdd_ioctl(struct file *filp, xxxx)
    {
        printk("enter cdd_ioctl!\n");

        // 不同的命令对应不同的操作
        switch(cmd) {
            case HELLO_ONE:
                printk("hello one\n");
                break;
            case HELLO_TWO:
                printk("hello two\n");
                break;
            default:
                return -EINVAL;
        }
    }
    ```

我们可以使用`ls -l /dev`查看已经创建的设备文件。First of all, note that the first letter of the permissions field is denoted that driver type. Device files are denoted either by b, for block devices, or c, for character devices. Also, note that the size field in the ls -l listing is replaced by two numbers, separated by a comma. The first value is the major device number and the second is the minor device number.

file operations syntax:

header file: `<linux/fs.h>`

```c
struct file_operations {
    struct module *owner;
    loff_t (*llseek) (struct file *, loff_t, int);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    ssize_t (*read_iter) (struct kiocb *, struct iov_iter *);
    ssize_t (*write_iter) (struct kiocb *, struct iov_iter *);
    int (*iterate) (struct file *, struct dir_context *);
    int (*iterate_shared) (struct file *, struct dir_context *);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *, fl_owner_t id);
    int (*release) (struct inode *, struct file *);
    int (*fsync) (struct file *, loff_t, loff_t, int datasync);
    int (*fasync) (int, struct file *, int);
    int (*lock) (struct file *, int, struct file_lock *);
    ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long (*get_unmapped_area)(struct file *, unsigned long, unsigned long, unsigned long, unsigned long);
    int (*check_flags)(int);
    int (*flock) (struct file *, int, struct file_lock *);
    ssize_t (*splice_write)(struct pipe_inode_info *, struct file *, loff_t *, size_t, unsigned int);
    ssize_t (*splice_read)(struct file *, loff_t *, struct pipe_inode_info *, size_t, unsigned int);
    int (*setlease)(struct file *, long, struct file_lock **, void **);
    long (*fallocate)(struct file *file, int mode, loff_t offset,
              loff_t len);
    void (*show_fdinfo)(struct seq_file *m, struct file *f);
#ifndef CONFIG_MMU
    unsigned (*mmap_capabilities)(struct file *);
#endif
    ssize_t (*copy_file_range)(struct file *, loff_t, struct file *,
            loff_t, size_t, unsigned int);
    int (*clone_file_range)(struct file *, loff_t, struct file *, loff_t,
            u64);
    ssize_t (*dedupe_file_range)(struct file *, u64, u64, struct file *,
            u64);
};
```

fields:

* `struct module *owner`

    It is a pointer to the module that “owns” the structure. This field is used to prevent the module from being unloaded while its operations are in use. Almost all the time, it is simply initialized to `THIS_MODULE`, a macro defined in `<linux/module.h>`.

* `ssize_t (*read) (struct file *, char _ _user *, size_t, loff_t *);`

    This is used to retrieve data from the device. A null pointer in this position causes the read system call to fail with `-EINVAL` (“Invalid argument”). A non-negative return value represents the number of bytes successfully read (the return value is a “signed size” type, usually the native integer type for the target platform).

* `ssize_t (*write) (struct file *, const char _ _user *, size_t, loff_t *);`

    It is used to sends the data to the device. If NULL -EINVAL is returned to the program calling the write system call. The return value, if non-negative, represents the number of bytes successfully written.

* `int (*ioctl) (struct inode *, struct file *, unsigned int, unsigned long);`

    The ioctl system call offers a way to issue device-specific commands (such as formatting a track of a floppy disk, which is neither reading nor writing). Additionally, a few ioctl commands are recognized by the kernel without referring to the fops table. If the device doesn’t provide an ioctl method, the system call returns an error for any request that isn’t predefined (-ENOTTY, “No such ioctl for device”).

* `int (*open) (struct inode *, struct file *);`

    Though this is always the first operation performed on the device file, the driver is not required to declare a corresponding method. If this entry is NULL, opening the device always succeeds, but your driver isn’t notified.

* `int (*release) (struct inode *, struct file *);`

    This operation is invoked when the file structure is being released. Like open, release can be NULL.

## Data exchange between kernel space and user space

Using this driver we can send strings or data to the kernel device driver using the write function. It will store that string in the kernel space. Then when I read the device file, it will send the data which is written by write by function to the userspace application.

申请内存：

```c
#include <linux/slab.h>

void *kmalloc(size_t size, gfp_t flags);
```

The allocated region is contiguous in physical memory.

* `size` - how many bytes of memory are required.

* `flags`– the type of memory to allocate.

    The flags argument may be one of:

    * `GFP_USER` – Allocate memory on behalf of the user. May sleep.

    * `GFP_KERNEL` – Allocate normal kernel ram. May sleep.

    * `GFP_ATOMIC` – Allocation will not sleep. May use emergency pools. For example, use this inside interrupt handler.

    * `GFP_HIGHUSER` – Allocate pages from high memory.

    * `GFP_NOIO` – Do not do any I/O at all while trying to get memory.

    * `GFP_NOFS` – Do not make any fs calls while trying to get memory.

    * `GFP_NOWAIT` – Allocation will not sleep.

    * `__GFP_THISNODE` – Allocate node-local memory only.

    * `GFP_DMA` – Allocation is suitable for DMA. Should only be used for kmalloc caches. Otherwise, use a slab created with SLAB_DMA.

    Also, it is possible to set different flags by OR’ing in one or more of the following additional flags:

    * `__GFP_COLD` – Request cache-cold pages instead of trying to return cache-warm pages.


    * `__GFP_HIGH` – This allocation has high priority and may use emergency pools.

    * `__GFP_NOFAIL` – Indicate that this allocation is in no way allowed to fail (think twice before using).

    * `__GFP_NORETRY` – If memory is not immediately available, then give up at once.

    * `__GFP_NOWARN` – If allocation fails, don’t issue any warnings.

    * `__GFP_REPEAT` – If allocation fails initially, try once more before failing.

    更多的参数，可以参考`linux/gfp.h`

释放内存：

```c
void kfree(const void *objp)
```

Parameters:

* `*objp` – pointer returned by `kmalloc`

从内核态向用户态写入数据：

`copy_to_user()`

Syntax:

```c
unsigned long copy_to_user(const void __user *to, const void *from, unsigned long  n);
```

This function is used to Copy a block of data into userspace (Copy data from kernel space to user space).

Parameters:

* `to` – Destination address, in the user space

* `from` – The source address in the kernel space

* `n` – Number of bytes to copy

Returns number of bytes that could not be copied. On success, this will be zero.

example:

(empty)

Note: Instead of using user space application, you can use `echo` and `cat` command.

## ioctl

There are many ways to Communicate between the Userspace and Kernel Space, they are:

* IOCTL

* Procfs

* Sysfs

* Configfs

* Debugfs

* Sysctl

* UDP Sockets

* Netlink Sockets

IOCTL is referred to as Input and Output Control, which is used to talk to device drivers. This system call is available in most driver categories.

Some real-time applications of ioctl are Ejecting the media from a “cd” drive, changing the Baud Rate of Serial port, Adjusting the Volume, Reading or Writing device registers, etc. We already have the write and read function in our device driver. But it is not enough for all cases.

There are some steps involved to use IOCTL.

* Create IOCTL command in the driver

    ```c
    #include <linux/ioctl.h>

    #define WR_VALUE _IOW('a','a',int32_t*)
    #define RD_VALUE _IOR('a','b',int32_t*)

    #define "ioctl name" __IOX("magic number","command number","argument type")
    ```

    where IOX can be:

    * `IO`: an ioctl with no parameters

    * `IOW`: an ioctl with write parameters (copy_from_user)

    * `IOR`: an ioctl with read parameters (copy_to_user)

    * `IOWR`: an ioctl with both write and read parameters

    * The Magic Number is a unique number or character that will differentiate our set of ioctl calls from the other ioctl calls. some times the major number for the device is used here.
  
    * Command Number is the number that is assigned to the ioctl. This is used to differentiate the commands from one another.
  
    * The last is the type of data.

* Write IOCTL function in the driver

    ```c
    int ioctl(struct inode *inode, struct file *file, unsigned int cmd,unsigned long arg)
    ```

    * `inode`: is the inode number of the file being worked on.
    * `file`: is the file pointer to the file that was passed by the application.
    * `cmd`: is the ioctl command that was called from the userspace.
    * `arg`: are the arguments passed from the userspace

    Within the function “ioctl” we need to implement all the commands that we defined above (`WR_VALUE`, `RD_VALUE`). We need to use the same commands in the `switch` statement which is defined above.

    Then we need to inform the kernel that the ioctl calls are implemented in the function “etx_ioctl“. This is done by making the fops pointer “unlocked_ioctl” to point to “etx_ioctl” as shown below.

    ```c
    /*
    ** This function will be called when we write IOCTL on the Device file
    */
    static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
    {
        switch(cmd) {
            case WR_VALUE:
                if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
                {
                    pr_err("Data Write : Err!\n");
                }
                pr_info("Value = %d\n", value);
                break;
            case RD_VALUE:
                if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
                {
                    pr_err("Data Read : Err!\n");
                }
                break;
            default:
                pr_info("Default\n");
                break;
        }
        return 0;
    }

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
    };
    ```

* Create IOCTL command in a Userspace application

    ```c
    #define WR_VALUE _IOW('a','a',int32_t*)
    #define RD_VALUE _IOR('a','b',int32_t*)
    ```

* Use the IOCTL system call in a Userspace

    ```c
    #include <sys/ioctl.h>
    long ioctl( "file descriptor","ioctl command","Arguments");
    ```

    * `file descriptor`: This the open file on which the ioctl command needs to be executed, which would generally be device files.
  
    * `ioctl command`: ioctl command which is implemented to achieve the desired functionality
    
    * `arguments`: The arguments need to be passed to the ioctl command.

    Example:

    ```c
    ioctl(fd, WR_VALUE, (int32_t*) &number); 
    ioctl(fd, RD_VALUE, (int32_t*) &value);
    ```

Driver full code:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 // kmalloc()
#include<linux/uaccess.h>              // copy_to/from_user()
#include <linux/ioctl.h>
#include <linux/err.h>
 
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long     etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .read           = etx_read,
    .write          = etx_write,
    .open           = etx_open,
    .unlocked_ioctl = etx_ioctl,
    .release        = etx_release,
};

/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
    pr_info("Device File Opened...!!!\n");
    return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
    pr_info("Device File Closed...!!!\n");
    return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
    pr_info("Read Function\n");
    return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
    pr_info("Write function\n");
    return len;
}

/*
** This function will be called when we write IOCTL on the Device file
*/
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch(cmd) {
        case WR_VALUE:
            if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
            {
                pr_err("Data Write : Err!\n");
            }
            pr_info("Value = %d\n", value);
            break;
        case RD_VALUE:
            if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
            {
                pr_err("Data Read : Err!\n");
            }
            break;
        default:
            pr_info("Default\n");
            break;
    }
    return 0;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_err("Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_err("Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device 1\n");
            goto r_device;
        }
        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        return -1;
}

/*
** Module exit function
*/
static void __exit etx_driver_exit(void)
{
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    pr_info("Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (IOCTL)");
MODULE_VERSION("1.5");
```

User program full code:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int main()
{
    int fd;
    int32_t value, number;

    printf("\nOpening Driver\n");
    fd = open("/dev/etx_device", O_RDWR);
    if(fd < 0) {
        printf("Cannot open device file...\n");
        return 0;
    }

    printf("Enter the Value to send\n");
    scanf("%d",&number);
    printf("Writing Value to Driver\n");
    ioctl(fd, WR_VALUE, (int32_t*) &number);

    printf("Reading Value from Driver\n");
    ioctl(fd, RD_VALUE, (int32_t*) &value);
    printf("Value is %d\n", value);

    printf("Closing Driver\n");
    close(fd);
}
```

This is a simple example of using ioctl in a Linux device driver. If you want to send multiple arguments, put those variables into the structure, and pass the address of the structure.

## Procfs

`/proc` is a mount point for the procfs (Process Filesystem) which is a filesystem in memory. Many processes store information about themselves on this virtual filesystem. ProcFS also stores other system information.

Check memory info: `cat /proc/meminfo`

Check modules info: `cat /proc/modules`

* `/proc/devices` — registered character and block major numbers
* `/proc/iomem` — on-system physical RAM and bus device addresses
* `/proc/ioports` — on-system I/O port addresses (especially for x86 systems)
* `/proc/interrupts` — registered interrupt request numbers
* `/proc/softirqs` — registered soft IRQs
* `/proc/swaps` — currently active swaps
* `/proc/kallsyms` — running kernel symbols, including from loaded modules
* `/proc/partitions` — currently connected block devices and their partitions
* `/proc/filesystems` — currently active filesystem drivers
* `/proc/cpuinfo` — information about the CPU(s) on the system

Most proc files are read-only and only expose kernel information to user space programs.

proc files can also be used to control and modify kernel behavior on the fly. The proc files need to be writable in this case.

enable IP forwarding of iptable: `echo 1 > /proc/sys/net/ipv4/ip_forward`

The proc file system can also be used to debug a kernel module. Just create entries for every variable that we want to track.

Creating procfs directory:

```c
struct proc_dir_entry *proc_mkdir(const char *name, struct proc_dir_entry *parent)
```

`name`: The name of the directory that will be created under `/proc`.

`parent`: In case the folder needs to be created in a subfolder under `/proc` a pointer to the same is passed else it can be left as NULL.

create proc entries:

header file: `linux/proc_fs.h`

```c
struct proc_dir_entry *proc_create ( const char *name, umode_t mode, struct proc_dir_entry *parent, const struct file_operations *proc_fops )
```

* `name`: The name of the proc entry
* `mode`: The access mode for proc entry
* `parent`: The name of the parent directory under /proc. If NULL is passed as a parent, the /proc directory will be set as a parent.
* `proc_fops`: The structure in which the file operations for the proc entry will be created.

Note: The above proc_create is valid in the Linux Kernel v3.10 to v5.5. From v5.6, there is a change in this API. The fourth argument const struct file_operations *proc_fops is changed to const struct proc_ops *proc_ops.

Example:

```c
proc_create("etx_proc",0666,NULL,&proc_fops);
```

create `file_operations` structure `proc_fops` in which we can map the read and write functions for the proc entry:

```c
static struct file_operations proc_fops = {
    .open = open_proc,
    .read = read_proc,
    .write = write_proc,
    .release = release_proc
};
```

For linux kernel v5.6 and above, use this:

```c
static struct proc_ops proc_fops = {
        .proc_open = open_proc,
        .proc_read = read_proc,
        .proc_write = write_proc,
        .proc_release = release_proc
};
```

`open` and `release` functions are optional:

```c
static int open_proc(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "proc file opend.....\t");
    return 0;
}

static int release_proc(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "proc file released.....\n");
    return 0;
}
```

The write function will receive data from the user space using the function copy_from_user into an array “etx_array”:

```c
static ssize_t write_proc(struct file *filp, const char *buff, size_t len, loff_t * off)
{
    printk(KERN_INFO "proc file write.....\t");
    copy_from_user(etx_array,buff,len);
    return len;
}
```

Once data is written to the proc entry we can read from the proc entry using a read function, i.e transfer data to the user space using the function `copy_to_user` function:

```c
static ssize_t read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset)
{
    printk(KERN_INFO "proc file read.....\n");
    if(len)
        len=0;
    else{
        len=1;
        return 0;
    }
    copy_to_user(buffer,etx_array,20);

    return length;;
}
```

Proc entry should be removed in the Driver exit function using the below function:

```c
void remove_proc_entry(const char *name, struct proc_dir_entry *parent);
```

Example:

```c
remove_proc_entry("etx_proc",NULL);
```

And you can remove the complete parent directory using `proc_remove(struct proc_dir_entry *parent)`.

Complete Driver Source Code:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>                 //kmalloc()
#include <linux/uaccess.h>              //copy_to/from_user()
#include <linux/ioctl.h>
#include <linux/proc_fs.h>
#include <linux/err.h>

/* 
** I am using the kernel 5.10.27-v7l. So I have set this as 510.
** If you are using the kernel 3.10, then set this as 310,
** and for kernel 5.1, set this as 501. Because the API proc_create()
** changed in kernel above v5.5.
**
*/ 
#define LINUX_KERNEL_VERSION  510
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
char etx_array[20]="try_proc_array\n";
static int len = 1;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
static struct proc_dir_entry *parent;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);

/*************** Driver Functions **********************/
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long     etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
 
/***************** Procfs Functions *******************/
static int      open_proc(struct inode *inode, struct file *file);
static int      release_proc(struct inode *inode, struct file *file);
static ssize_t  read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset);
static ssize_t  write_proc(struct file *filp, const char *buff, size_t len, loff_t * off);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
};


#if ( LINUX_KERNEL_VERSION > 505 )

/*
** procfs operation sturcture
*/
static struct proc_ops proc_fops = {
        .proc_open = open_proc,
        .proc_read = read_proc,
        .proc_write = write_proc,
        .proc_release = release_proc
};

#else //LINUX_KERNEL_VERSION > 505

/*
** procfs operation sturcture
*/
static struct file_operations proc_fops = {
        .open = open_proc,
        .read = read_proc,
        .write = write_proc,
        .release = release_proc
};

#endif //LINUX_KERNEL_VERSION > 505

/*
** This function will be called when we open the procfs file
*/
static int open_proc(struct inode *inode, struct file *file)
{
    pr_info("proc file opend.....\t");
    return 0;
}

/*
** This function will be called when we close the procfs file
*/
static int release_proc(struct inode *inode, struct file *file)
{
    pr_info("proc file released.....\n");
    return 0;
}

/*
** This function will be called when we read the procfs file
*/
static ssize_t read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset)
{
    pr_info("proc file read.....\n");
    if(len)
    {
        len=0;
    }
    else
    {
        len=1;
        return 0;
    }
    
    if( copy_to_user(buffer,etx_array,20) )
    {
        pr_err("Data Send : Err!\n");
    }
 
    return length;;
}

/*
** This function will be called when we write the procfs file
*/
static ssize_t write_proc(struct file *filp, const char *buff, size_t len, loff_t * off)
{
    pr_info("proc file wrote.....\n");
    
    if( copy_from_user(etx_array,buff,len) )
    {
        pr_err("Data Write : Err!\n");
    }
    
    return len;
}

/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read function\n");
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write Function\n");
        return len;
}

/*
** This function will be called when we write IOCTL on the Device file
*/
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch(cmd) {
        case WR_VALUE:
            if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
            {
                pr_err("Data Write : Err!\n");
            }
            pr_info("Value = %d\n", value);
            break;
        case RD_VALUE:
            if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
            {
                pr_err("Data Read : Err!\n");
            }
            break;
        default:
            pr_info("Default\n");
            break;
    }
    return 0;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            pr_info("Cannot allocate major number\n");
            return -1;
    }
    pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0){
        pr_info("Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        pr_info("Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        pr_info("Cannot create the Device 1\n");
        goto r_device;
    }
    
    /*Create proc directory. It will create a directory under "/proc" */
    parent = proc_mkdir("etx",NULL);
    
    if( parent == NULL )
    {
        pr_info("Error creating proc entry");
        goto r_device;
    }
    
    /*Creating Proc entry under "/proc/etx/" */
    proc_create("etx_proc", 0666, parent, &proc_fops);

    pr_info("Device Driver Insert...Done!!!\n");
    return 0;
 
r_device:
    class_destroy(dev_class);
r_class:
    unregister_chrdev_region(dev,1);
    return -1;
}
 
/*
** Module exit function
*/
static void __exit etx_driver_exit(void)
{
    /* Removes single proc entry */
    //remove_proc_entry("etx/etx_proc", parent);
    
    /* remove complete /proc/etx */
    proc_remove(parent);
    
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    pr_info("Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (procfs)");
MODULE_VERSION("1.6");
```

As there are changes in the procfs file system in the Linux kernel 3.10 and 5.6, we have added a macro called `LINUX_KERNEL_VERSION`. You have to mention your Linux kernel version. Based on that, we will control the APIs in this source code.

Note:

You can follow this format for this `LINUX_KERNEL_VERSION`.

Example:

|Your Linux Kernel version | LINUX_KERNEL_VERSION |
| - | - |
| v3.10 | 310 |
| v5.6 | 506 |
| v5.10 | 510 |

test:

`cat /proc/etx/etx_proc`

`echo "device driver proc" > /proc/etx/etx_proc`

## waitqueue

Whenever a process must wait for an event (such as the arrival of data or the termination of a process), it should go to sleep. Sleeping causes the process to suspend execution, freeing the processor for other uses. After some time, the process will be woken up and will continue with its job when the event which we are waiting for has arrived.

Wait queue is a mechanism provided in the kernel to implement the wait. As the name itself suggests, waitqueue is the list of processes waiting for an event. In other words, A wait queue is used to wait for someone to wake you up when a certain condition is true. They must be used carefully to ensure there is no race condition.

There are 3 important steps in Waitqueue.

1. Initializing Waitqueue

    header file: `linux/wait.h`

    1. Static method

        ```c
        DECLARE_WAIT_QUEUE_HEAD(wq);
        ```

        Where the “wq” is the name of the queue on which task will be put to sleep.

    1. Dynamic method

        ```c
        wait_queue_head_t wq;
        init_waitqueue_head(&wq);
        ```

1. Queuing (Put the Task to sleep until the event comes)

    Once the wait queue is declared and initialized, a process may use it to go to sleep. There are several macros are available for different uses.

    * wait_event

        Syntax:

        ```c
        wait_event(wq, condition);
        ```

        sleep until a condition gets true.

        Parameters:

        * `wq` – the waitqueue to wait on

        * `condition` – a C expression for the event to wait for

        The process is put to sleep (`TASK_UNINTERRUPTIBLE`) until the condition evaluates to true. The `condition` is checked each time the waitqueue `wq` is woken up.

    * wait_event_timeout

        Syntax:

        ```c
        wait_event_timeout(wq, condition, timeout);
        ```

        sleep until a condition gets true or a timeout elapses

        Parameters:

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `timeout` –  timeout, in jiffies

        The process is put to sleep (TASK_UNINTERRUPTIBLE) until the condition evaluates to true or timeout elapses. The condition is checked each time the waitqueue wq is woken up.

        It returns 0 if the condition evaluated to false after the timeout elapsed, 1 if the condition evaluated to true after the timeout elapsed, or the remaining jiffies (at least 1) if the condition evaluated to true before the timeout elapsed.

    * wait_event_cmd

        Syntax:

        ```c
        wait_event_cmd(wq, condition, cmd1, cmd2);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `cmd1` – the command will be executed before sleep

        * `cmd2` – the command will be executed after sleep

        The process is put to sleep (TASK_UNINTERRUPTIBLE) until the condition evaluates to true. The condition is checked each time the waitqueue wq is woken up.

    * wait_event_interruptible

        Syntax:

        ```c
        wait_event_interruptible(wq, condition);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        The process is put to sleep (TASK_INTERRUPTIBLE) until the condition evaluates to true or a signal is received. The condition is checked each time the waitqueue wq is woken up.

        The function will return -ERESTARTSYS if it was interrupted by a signal and 0 if condition evaluated to true.

    * wait_event_interruptible_timeout

        Syntax:

        ```c
        wait_event_interruptible_timeout(wq, condition, timeout);
        ```

        sleep until a condition gets true or a timeout elapses

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `timeout` –  timeout, in jiffies

        The process is put to sleep (TASK_INTERRUPTIBLE) until the condition evaluates to true or a signal is received or timeout elapsed. The condition is checked each time the waitqueue wq is woken up.

        It returns, 0 if the condition evaluated to false after the timeout elapsed, 1 if the condition evaluated to true after the timeout elapsed, the remaining jiffies (at least 1) if the condition evaluated to true before the timeout elapsed, or -ERESTARTSYS if it was interrupted by a signal.

    * wait_event_killable

        Syntax:

        ```c
        wait_event_killable(wq, condition);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        The process is put to sleep (TASK_KILLABLE) until the condition evaluates to true or a signal is received. The condition is checked each time the waitqueue wq is woken up.

        The function will return -ERESTARTSYS if it was interrupted by a signal and 0 if condition evaluated to true.

    Whenever we use the above one of the macro, it will add that task to the waitqueue which is created by us. Then it will wait for the event.

    Note: Old kernel versions used the functions `sleep_on()` and `interruptible_sleep_on()`, but those two functions can introduce bad race conditions and should not be used.

1. Waking Up Queued Task

    When some Tasks are in sleep mode because of the waitqueue, then we can use the below function to wake up those tasks.

    * wake_up

        Syntax:

        ```c
        wake_up(&wq);
        ```

        wakes up only one process from the wait queue which is in non-interruptible sleep.

        Parameters:

        * `wq` – the waitqueue to wake up

    * wake_up_all

        Syntax:

        ```c
        wake_up_all(&wq);
        ```

        wakes up all the processes on the wait queue

    * wake_up_interruptible

        Syntax:

        ```c
        wake_up_interruptible(&wq);
        ```

        wakes up only one process from the wait queue that is in interruptible sleep

    * wake_up_sync and wake_up_interruptible_sync

        Syntax:

        ```c
        wake_up_sync(&wq);
        wake_up_interruptible_sync(&wq);
        ```

        Normally, a `wake_up` call can cause an immediate reschedule to happen, meaning that other processes might run before `wake_up` returns. The “synchronous” variants instead make any awakened processes runnable but do not reschedule the CPU. This is used to avoid rescheduling when the current process is known to be going to sleep, thus forcing a reschedule anyway. Note that awakened processes could run immediately on a different processor, so these functions should not be expected to provide mutual exclusion.

driver code:

* Waitqueue created by Static Method

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include <linux/slab.h>                 //kmalloc()
    #include <linux/uaccess.h>              //copy_to/from_user()
    #include <linux/kthread.h>
    #include <linux/wait.h>                 // Required for the wait queues
    #include <linux/err.h>
    
    
    uint32_t read_count = 0;
    static struct task_struct *wait_thread;
    
    DECLARE_WAIT_QUEUE_HEAD(wait_queue_etx);
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    int wait_queue_flag = 0;

    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
            .owner          = THIS_MODULE,
            .read           = etx_read,
            .write          = etx_write,
            .open           = etx_open,
            .release        = etx_release,
    };

    /*
    ** Thread function
    */
    static int wait_function(void *unused)
    {
        while(1) {
            pr_info("Waiting For Event...\n");
            wait_event_interruptible(wait_queue_etx, wait_queue_flag != 0 );
            if(wait_queue_flag == 2) {
                pr_info("Event Came From Exit Function\n");
                return 0;
            }
            pr_info("Event Came From Read Function - %d\n", ++read_count);
            wait_queue_flag = 0;
        }
        do_exit(0);
        return 0;
    }

    /*
    ** This function will be called when we open the Device file
    */
    static int etx_open(struct inode *inode, struct file *file)
    {
        pr_info("Device File Opened...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */
    static int etx_release(struct inode *inode, struct file *file)
    {
        pr_info("Device File Closed...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read Function\n");
        wait_queue_flag = 1;
        wake_up_interruptible(&wait_queue_etx);
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Write function\n");
        return len;
    }
    
    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            pr_info("Cannot allocate major number\n");
            return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
        etx_cdev.owner = THIS_MODULE;
        etx_cdev.ops = &fops;

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_info("Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_info("Cannot create the struct class\n");
            goto r_class;
        }

        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_info("Cannot create the Device 1\n");
            goto r_device;
        }

        //Create the kernel thread with name 'mythread'
        wait_thread = kthread_create(wait_function, NULL, "WaitThread");
        if (wait_thread) {
            pr_info("Thread Created successfully\n");
            wake_up_process(wait_thread);
        } else
            pr_info("Thread creation failed\n");

        pr_info("Device Driver Insert...Done!!!\n");
        return 0;

    r_device:
            class_destroy(dev_class);
    r_class:
            unregister_chrdev_region(dev,1);
            return -1;
    }

    /*
    ** Module exit function
    */ 
    static void __exit etx_driver_exit(void)
    {
        wait_queue_flag = 2;
        wake_up_interruptible(&wait_queue_etx);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple linux driver (Waitqueue Static method)");
    MODULE_VERSION("1.7");
    ```

* Waitqueue created by Dynamic Method

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include <linux/slab.h>                 //kmalloc()
    #include <linux/uaccess.h>              //copy_to/from_user()
    #include <linux/kthread.h>
    #include <linux/wait.h>                 // Required for the wait queues
    #include <linux/err.h>
    
    
    uint32_t read_count = 0;
    static struct task_struct *wait_thread;
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    wait_queue_head_t wait_queue_etx;
    int wait_queue_flag = 0;
    
    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
    };
    
    /*
    ** Thread function
    */
    static int wait_function(void *unused)
    { 
        while(1) {
            pr_info("Waiting For Event...\n");
            wait_event_interruptible(wait_queue_etx, wait_queue_flag != 0 );
            if(wait_queue_flag == 2) {
                pr_info("Event Came From Exit Function\n");
                return 0;
            }
            pr_info("Event Came From Read Function - %d\n", ++read_count);
            wait_queue_flag = 0;
        }
        return 0;
    }
    
    /*
    ** This function will be called when we open the Device file
    */ 
    static int etx_open(struct inode *inode, struct file *file)
    {
        pr_info("Device File Opened...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */
    static int etx_release(struct inode *inode, struct file *file)
    {
        pr_info("Device File Closed...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read Function\n");
        wait_queue_flag = 1;
        wake_up_interruptible(&wait_queue_etx);
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Write function\n");
        return len;
    }

    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_info("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_info("Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_info("Cannot create the struct class\n");
            goto r_class;
        }
    
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_info("Cannot create the Device 1\n");
            goto r_device;
        }
        
        //Initialize wait queue
        init_waitqueue_head(&wait_queue_etx);

        // Create the kernel thread with name 'mythread'
        wait_thread = kthread_create(wait_function, NULL, "WaitThread");
        if (wait_thread) {
            pr_info("Thread Created successfully\n");
            wake_up_process(wait_thread);
        } else
            pr_info("Thread creation failed\n");

        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
    
    r_device:
        class_destroy(dev_class);
    r_class:
        unregister_chrdev_region(dev,1);
        return -1;
    }

    /*
    ** Module exit function
    */
    static void __exit etx_driver_exit(void)
    {
        wait_queue_flag = 2;
        wake_up_interruptible(&wait_queue_etx);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple linux driver (Waitqueue Dynamic method)");
    MODULE_VERSION("1.8");
    ```

Makefile:

```Makefile
obj-m += driver.o
KDIR = /lib/modules/$(shell uname -r)/build
all:
    make -C $(KDIR)  M=$(shell pwd) modules
clean:
    make -C $(KDIR)  M=$(shell pwd) clean
```

（这个版本中的 Makefile 使用的并不是`$(PWD)`之类的，而是`$(shell pwd)`，这两者有什么不同？）

test: `sudo cat /dev/etx_device`

## sysfs

Sysfs is a virtual filesystem mounted on `/sys`. Sysfs contain information about devices and drivers.

**Kernel Objects**

The heart of the sysfs model is the kobject. Kobject is the glue that binds the sysfs and the kernel, which is represented by `struct kobject` and defined in `<linux/kobject.h>`. A struct kobject represents a kernel object, maybe a device or so, such as the things that show up as directory in the sysfs filesystem.

Kobjects are usually embedded in other structures.

Syntax:

```c
#define KOBJ_NAME_LEN 20 

struct kobject {
 char *k_name;
 char name[KOBJ_NAME_LEN];
 struct kref kref;
 struct list_head entry;
 struct kobject *parent;
 struct kset *kset;
 struct kobj_type *ktype;
 struct dentry *dentry;
};
```

Explanation:

* `struct kobject`

    * `name` (Name of the kobject. Current kobject is created with this name in sysfs.)

    * `parent` (This is kobject’s parent. When we create a directory in sysfs for the current kobject, it will create under this parent directory)

    * `ktype` (the type associated with a kobject)

    * `kset` (a group of kobjects all of which are embedded in structures of the same type)

    * `sd` (points to a sysfs_dirent structure that represents this kobject in sysfs.)

    * `kref` (provides reference counting)

    `kobject` is used to create kobject directory in /sys.

There are two steps to creating and using sysfs.

1. Create a directory in `/sys`

    We can use this function (`kobject_create_and_add`) to create a directory.

    `struct kobject * kobject_create_and_add ( const char * name, struct kobject * parent);`

    Where,

    * `name` – the name for the kobject

    * `parent` – the parent kobject of this kobject, if any.

        If you pass `kernel_kobj` to the second argument, it will create the directory under `/sys/kernel/`. If you pass `firmware_kobj` to the second argument, it will create the directory under `/sys/firmware/`. If you pass `fs_kobj` to the second argument, it will create the directory under `/sys/fs/`. If you pass NULL to the second argument, it will create the directory under `/sys/`.

    This function creates a kobject structure dynamically and registers it with sysfs. If the kobject was not able to be created, `NULL` will be returned.

    Call `kobject_put` and the structure `kobject` will be dynamically freed when it is no longer being used. (not clear. Does it mean free the memory immediately or wait for the last time that `struct object` was used?)

    Example:

    ```c
    struct kobject *kobj_ref;

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj); //sys/kernel/etx_sysfs

    /*Freeing Kobj*/
    kobject_put(kobj_ref);
    ```

1. Create Sysfs file

    sysfs file is used to interact user space with kernel space.

    We can create the sysfs file using sysfs attributes. Attributes are represented as regular files in sysfs with one value per file. There are loads of helper functions that can be used to create the kobject attributes. They can be found in the header file `sysfs.h`.

    * Create attribute

        Syntax:

        ```c
        struct kobj_attribute {
            struct attribute attr;
            ssize_t (*show)(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
            ssize_t (*store)(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
        };
        ```

        Parameters:

        * `attr` – the attribute representing the file to be created,

        * `show` – the pointer to the function that will be called when the file is read in sysfs,

        * `store` – the pointer to the function which will be called when the file is written in sysfs.

        We can create an attribute using `__ATTR` macro.

        `__ATTR(name, permission, show_ptr, store_ptr);`
        
    * Store and Show functions

        ```c
        ssize_t (*show)(struct kobject *kobj, struct kobj_attribute *attr, char *buf);

        ssize_t (*store)(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
        ```

        Store function will be called whenever we are writing something to the sysfs attribute.

        Show function will be called whenever we are reading the sysfs attribute.

    * Create sysfs file

        To create a single file attribute we are going to use ‘sysfs_create_file’.

        int sysfs_create_file ( struct kobject *  kobj, const struct attribute * attr);

        Where,

        `kobj` – object we’re creating for.

        `attr` – attribute descriptor.

        One can use another function `sysfs_create_group` to create a group of attributes.

        Once you have done with the sysfs file, you should delete this file using `sysfs_remove_file`。

        ```c
        void sysfs_remove_file ( struct kobject *  kobj, const struct attribute * attr);
        ```

        Where,

        `kobj` – object we’re creating for.

        `attr` – attribute descriptor.

    Example:

    ```c
    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
        return sprintf(buf, "%d", etx_value);
    }

    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            sscanf(buf,"%d",&etx_value);
            return count;
    }

    //This Function will be called from Init function
    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
        printk(KERN_INFO"Cannot create sysfs file......\n");
        goto r_sysfs;
    }
    //This should be called from exit function
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    ```

    driver:

    ```c
    /***************************************************************************//**
    *  \file       driver.c
    *
    *  \details    Simple Linux device driver (sysfs)
    *
    *  \author     EmbeTronicX
    *
    *  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
    *
    *******************************************************************************/
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include<linux/slab.h>                 //kmalloc()
    #include<linux/uaccess.h>              //copy_to/from_user()
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/err.h>
    
    volatile int etx_value = 0;
    
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;

    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, 
                            char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, 
                            const char *buf, size_t len, loff_t * off);
    
    /*************** Sysfs functions **********************/
    static ssize_t  sysfs_show(struct kobject *kobj, 
                            struct kobj_attribute *attr, char *buf);
    static ssize_t  sysfs_store(struct kobject *kobj, 
                            struct kobj_attribute *attr,const char *buf, size_t count);

    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
    };

    /*
    ** This function will be called when we read the sysfs file
    */
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
        pr_info("Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
    }

    /*
    ** This function will be called when we write the sysfsfs file
    */
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
        pr_info("Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
    }

    /*
    ** This function will be called when we open the Device file
    */ 
    static int etx_open(struct inode *inode, struct file *file)
    {
            pr_info("Device File Opened...!!!\n");
            return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */ 
    static int etx_release(struct inode *inode, struct file *file)
    {
        pr_info("Device File Closed...!!!\n");
        return 0;
    }
    
    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read function\n");
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Write Function\n");
        return len;
    }
    
    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_info("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_info("Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_info("Cannot create the struct class\n");
            goto r_class;
        }

        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_info("Cannot create the Device 1\n");
            goto r_device;
        }

        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                    pr_err("Cannot create sysfs file......\n");
                    goto r_sysfs;
        }

        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
    
    r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    
    r_device:
        class_destroy(dev_class);
    r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
    }

    /*
    ** Module exit function
    */
    static void __exit etx_driver_exit(void)
    {
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple Linux device driver (sysfs)");
    MODULE_VERSION("1.8");
    ```

    test: `ls -l /sys/kernel`, `ls -l /sys/kernel/etx_sysfs`

    read and modify the value:

    ```bash
    sudo su
    cat /sys/kernel/etx_sysfs/etx_value
    echo 123 > /sys/kernel/etx_sysfs/etx_value
    cat /sys/kernel/etx_sysfs/etx_value
    ```

## wait queue

wait queue 类似于用户态编程的 condition variable，用于控制同步。


核心的几个类型和函数：

```c
wait_queue_head_t
```

通常使用一个额外的线程来处理 wait queue，防止 init module 之类被阻塞。

```c
#include <linux/kthread.h>

struct task_struct *wait_thread = kthread_create(print_msg, NULL, "print_msg");
if (wait_thread) {
    pr_info("wake up process\n");
    wake_up_process(wait_thread);
}
```

初始化 wait queue:

```c
wait_queue_head_t wq;
int condi = 1;

void init_wq() {
    init_waitqueue_head(&wq);
}

int print_msg(void *unused)
{
    pr_info("wait condition variable...\n");
    wait_event_interruptible(wq, condi == 2);
    pr_info("condi is %d, hello, world\n", condi);
    condi = 1;
    pr_info("rechange condi to 1\n");
    return 0;
}
```

触发：

```c
ssize_t h_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in h_write()...\n");
    copy_from_user(&condi, buf, sizeof(condi));
    wake_up_interruptible(&wq);
    return sizeof(condi);
}
```

可以看到，当改变变量值后，还需要手动通知一下 wait queue，wait queue 才能判断是否往下走。

经实验，如果不满足`condi == 2`的条件，即使`wake_up_interruptible()`也无法让程序继续往下走。

详细的 example: 见`ref_13`

## Interrupts

### cache

* 调研 irq example

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/sched.h>
    #include <linux/workqueue.h>
    #include <linux/interrupt.h>
    #include <asm/io.h>

    #define MY_WORK_QUEUE_NAME "WQsched.c"

    static struct workqueue_struct *my_workqueue;

    static void got_char(void *scancode)
    {
    	printk(KERN_INFO "Scan Code %x %s.\n",
    	       (int)*((char *)scancode) & 0x7F,
    	       *((char *)scancode) & 0x80 ? "Released" : "Pressed");
    }

    irqreturn_t irq_handler(int irq, void *dev_id, struct pt_regs *regs)
    {
    	/* 
    	 * This variables are static because they need to be
    	 * accessible (through pointers) to the bottom half routine.
    	 */
    	static int initialised = 0;
    	static unsigned char scancode;
    	static struct work_struct task;
    	unsigned char status;

    	/* 
    	 * Read keyboard status
    	 */
    	status = inb(0x64);
    	scancode = inb(0x60);

    	if (initialised == 0) {
    		INIT_WORK(&task, (void(*)(struct work_struct *))got_char);
    		// INIT_WORK(&task, got_char, &scancode);
    		initialised = 1;
    	} else {
    		DECLARE_WORK(task, (void(*)(struct work_struct *))got_char);
    		// PREPARE_WORK(&task, got_char);
    		// PREPARE_WORK(&task, got_char, &scancode);
    	}

    	queue_work(my_workqueue, &task);

    	return IRQ_HANDLED;
    }

    int hello_init(void) {
        pr_info("int hello_init()...\n");
        my_workqueue = create_workqueue(MY_WORK_QUEUE_NAME);
        free_irq(1, NULL);
        int ret = request_irq(1, (void*) irq_handler, IRQF_SHARED,
            "test_keyboard_irq_handler", (void*) irq_handler);
        if (ret != 0) {
            pr_info("fail to request irq\n");
            return -1;
        }
    	return 0;
    }

    void hello_exit(void) {
        pr_info("in hello_exit()...\n");
        free_irq(1, NULL);
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

    打开`dmesg`日志，在 insmod 后，每次按下键盘都会打印中断函数的处理消息：

    ```
    [ 1234.381119] intrpt: loading out-of-tree module taints kernel.
    [ 1234.381123] intrpt: module verification failed: signature and/or required key missing - tainting kernel
    [ 1234.381522] ------------[ cut here ]------------
    [ 1234.381523] Trying to free already-free IRQ 1
    [ 1234.381527] WARNING: CPU: 0 PID: 2691 at kernel/irq/manage.c:1893 __free_irq+0x1a6/0x310
    [ 1234.381533] Modules linked in: intrpt(OE+) vboxsf intel_rapl_msr snd_intel8x0 intel_rapl_common snd_ac97_codec ac97_bus intel_uncore_frequency_common snd_pcm snd_seq_midi snd_seq_midi_event binfmt_misc snd_rawmidi crct10dif_pclmul polyval_clmulni polyval_generic ghash_clmulni_intel sha256_ssse3 sha1_ssse3 aesni_intel crypto_simd cryptd nls_iso8859_1 joydev snd_seq rapl input_leds snd_seq_device snd_timer vmwgfx drm_ttm_helper snd ttm serio_raw drm_kms_helper soundcore vboxguest mac_hid sch_fq_codel msr parport_pc ppdev lp parport drm efi_pstore ip_tables x_tables autofs4 hid_generic usbhid hid crc32_pclmul psmouse ahci libahci video i2c_piix4 e1000 wmi pata_acpi
    [ 1234.381561] CPU: 0 PID: 2691 Comm: insmod Tainted: G           OE      6.5.0-28-generic #29~22.04.1-Ubuntu
    [ 1234.381563] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
    [ 1234.381564] RIP: 0010:__free_irq+0x1a6/0x310
    [ 1234.381566] Code: 50 32 00 00 49 8b be 88 01 00 00 e8 74 ec 02 00 49 8b 7f 30 e8 0b 9c 22 00 eb 35 8b 75 d0 48 c7 c7 40 2d d6 88 e8 5a af f4 ff <0f> 0b 48 8b 75 c8 4c 89 e7 e8 5c d6 f8 00 49 8b 46 40 48 8b 40 78
    [ 1234.381568] RSP: 0018:ffffb0238399fac0 EFLAGS: 00010046
    [ 1234.381569] RAX: 0000000000000000 RBX: 0000000000000000 RCX: 0000000000000000
    [ 1234.381571] RDX: 0000000000000000 RSI: 0000000000000000 RDI: 0000000000000000
    [ 1234.381571] RBP: ffffb0238399faf8 R08: 0000000000000000 R09: 0000000000000000
    [ 1234.381572] R10: 0000000000000000 R11: 0000000000000000 R12: ffff95f3801688a4
    [ 1234.381573] R13: ffff95f380168960 R14: ffff95f380168800 R15: ffff95f382d72b00
    [ 1234.381574] FS:  00007d6eb84b3c40(0000) GS:ffff95f39bc00000(0000) knlGS:0000000000000000
    [ 1234.381576] CS:  0010 DS: 0000 ES: 0000 CR0: 0000000080050033
    [ 1234.381577] CR2: 00006172fb59f520 CR3: 000000002f81a000 CR4: 00000000000506f0
    [ 1234.381580] Call Trace:
    [ 1234.381581]  <TASK>
    [ 1234.381583]  ? show_regs+0x6d/0x80
    [ 1234.381587]  ? __warn+0x89/0x160
    [ 1234.381589]  ? __free_irq+0x1a6/0x310
    [ 1234.381591]  ? report_bug+0x17e/0x1b0
    [ 1234.381594]  ? handle_bug+0x46/0x90
    [ 1234.381597]  ? exc_invalid_op+0x18/0x80
    [ 1234.381598]  ? asm_exc_invalid_op+0x1b/0x20
    [ 1234.381602]  ? __free_irq+0x1a6/0x310
    [ 1234.381604]  free_irq+0x32/0x80
    [ 1234.381606]  ? __pfx_init_module+0x10/0x10 [intrpt]
    [ 1234.381609]  init_module+0x39/0x70 [intrpt]
    [ 1234.381612]  do_one_initcall+0x5e/0x340
    [ 1234.381616]  do_init_module+0x68/0x260
    [ 1234.381619]  load_module+0xb85/0xcd0
    [ 1234.381621]  ? security_kernel_post_read_file+0x75/0x90
    [ 1234.381624]  init_module_from_file+0x96/0x100
    [ 1234.381626]  ? init_module_from_file+0x96/0x100
    [ 1234.381629]  idempotent_init_module+0x11c/0x2b0
    [ 1234.381631]  __x64_sys_finit_module+0x64/0xd0
    [ 1234.381633]  do_syscall_64+0x5b/0x90
    [ 1234.381636]  ? ksys_mmap_pgoff+0x120/0x270
    [ 1234.381638]  ? exit_to_user_mode_prepare+0x30/0xb0
    [ 1234.381639]  ? syscall_exit_to_user_mode+0x37/0x60
    [ 1234.381641]  ? do_syscall_64+0x67/0x90
    [ 1234.381642]  ? exit_to_user_mode_prepare+0x30/0xb0
    [ 1234.381643]  ? syscall_exit_to_user_mode+0x37/0x60
    [ 1234.381645]  ? do_syscall_64+0x67/0x90
    [ 1234.381646]  entry_SYSCALL_64_after_hwframe+0x6e/0xd8
    [ 1234.381648] RIP: 0033:0x7d6eb7d1e88d
    [ 1234.381658] Code: 5b 41 5c c3 66 0f 1f 84 00 00 00 00 00 f3 0f 1e fa 48 89 f8 48 89 f7 48 89 d6 48 89 ca 4d 89 c2 4d 89 c8 4c 8b 4c 24 08 0f 05 <48> 3d 01 f0 ff ff 73 01 c3 48 8b 0d 73 b5 0f 00 f7 d8 64 89 01 48
    [ 1234.381659] RSP: 002b:00007fff0d23a4d8 EFLAGS: 00000246 ORIG_RAX: 0000000000000139
    [ 1234.381661] RAX: ffffffffffffffda RBX: 00006172fbf127a0 RCX: 00007d6eb7d1e88d
    [ 1234.381662] RDX: 0000000000000000 RSI: 00006172fb5aacd2 RDI: 0000000000000003
    [ 1234.381662] RBP: 0000000000000000 R08: 0000000000000000 R09: 0000000000000000
    [ 1234.381663] R10: 0000000000000003 R11: 0000000000000246 R12: 00006172fb5aacd2
    [ 1234.381664] R13: 00006172fbf12760 R14: 00006172fb5a9888 R15: 00006172fbf128b0
    [ 1234.381665]  </TASK>
    [ 1234.381666] ---[ end trace 0000000000000000 ]---
    [ 1234.468732] Scan Code 40 Released.
    [ 1237.372783] Scan Code 40 Released.
    [ 1237.428975] Scan Code 40 Released.
    [ 1237.429958] Scan Code 40 Released.
    [ 1237.501143] Scan Code 40 Released.
    [ 1237.548031] Scan Code 40 Released.
    [ 1237.596222] Scan Code 40 Released.
    [ 1237.612977] Scan Code 40 Released.
    [ 1237.700977] Scan Code 40 Released.
    [ 1237.716093] Scan Code 40 Released.
    [ 1237.780960] Scan Code 40 Released.
    [ 1237.851931] Scan Code 40 Released.
    [ 1237.924812] Scan Code 40 Released.
    [ 1237.947605] Scan Code 40 Released.
    [ 1238.028965] Scan Code 40 Released.
    [ 1238.029951] Scan Code 40 Released.
    ```

    在执行`sudo rmmod intrpt`后，系统会直接卡死。因为我们在代码里覆盖了系统原本处理键盘中断的 handler。

    具体的原因在程序注释里也有说明。

    feedback:

    1. 调研`INIT_WORK()`, `DECLARE_WORK()`

    1. 调研`queue_work()`, `create_workqueue()`

* Interrupts Flags

    * `IRQF_DISABLED`

    * `IRQF_SAMPLE_RANDOM`

    * `IRQF_TIMER`

    * `IRQF_SHARED`

        this flag enables one irq number matches multiple irq handler.

        (do the handlers run one by one, or run simultaneously, or under a random order?)

* `request_irq()` cannot be called from interrupt context (other situations where code cannot block), because it can block.

* irq 将一个整数和一个 handler function 相 map。

    有一些整数的含义是被预定的，比如`1`代表的是键盘中断。

* what should be cautioned when writing interrupt handler

    * Interrupt handlers can not enter sleep, so to avoid calls to some functions which has sleep.

    * When the interrupt handler has part of the code to enter the critical section, use spinlocks lock, rather than mutexes. Because if it can’t take mutex it will go to sleep until it takes the mute.

    * Interrupt handlers can not exchange data with the userspace.The interrupt handlers must be executed as soon as possible. To ensure this, it is best to split the implementation into two parts, the top half and the bottom half. The top half of the handler will get the job done as soon as possible and then work late on the bottom half, which can be done with softirq or tasklet or workqueue.

    * Interrupt handlers can not be called repeatedly. When a handler is already executing, its corresponding IRQ must be disabled until the handler is done.

        maybe this means the hander can't be recursively invoked.

    * Interrupt handlers can be interrupted by higher authority handlers. If you want to avoid being interrupted by a highly qualified handler, you can mark the interrupt handler as a fast handler. However, if too many are marked as fast handlers, the performance of the system will be degraded because the interrupt latency will be longer.

* handler function syntax:

    ```c
    irqreturn_t irq_handler(int irq, void *dev_id, struct pt_regs *regs)
    ```

    * `dev_id`

        this pointer is used to identify different devices.

        when the interruption occurs, one irq number may be conrresponding to multiple devices.

        Or, in other words, one irq number is shared by multiple devices.

        Thus there must be a unique identification value to distinct different devices.

        A common practice is to use the driver's structure pointer.

    * return a `irqreturn_t` type value

        the return value `IRQ_HANDLED` means it process irq successfully.

        the return value `IRQ_NONE` means the handler function fails to process the irq.

special functions called interrupt handlers (ISR)

In Linux, interrupt signals are the distraction that diverts the processor to a new activity outside of the normal flow of execution. This new activity is called interrupt handler or interrupt service routine (ISR) and that distraction is Interrupts.


**Polling vs Interrupts**

* Polling

    In polling the CPU keeps on checking all the hardwares of the availablilty of any request

    The polling method is like a salesperson. The salesman goes from door to door while requesting to buy a product or service. Similarly, the controller keeps monitoring the flags or signals one by one for all devices and provides service to whichever component that needs its service.

* Interrupt

    In interrupt the CPU takes care of the hardware only when the hardware requests for some service

    An interrupt is like a shopkeeper. If one needs a service or product, he goes to him and apprises him of his needs. In case of interrupts, when the flags or signals are received, they notify the controller that they need to be serviced.

**Interrupts and Exceptions**

Exceptions are often discussed at the same time as interrupts. Unlike interrupts, exceptions occur synchronously with respect to the processor clock; they are often called synchronous interrupts. Exceptions are produced by the processor while executing instructions either in response to a programming error (e.g. divide by zero) or abnormal conditions that must be handled by the kernel (e.g. a page fault).

Interrupts – asynchronous interrupts generated by hardware.

Exceptions – synchronous interrupts generated by the processor.

（这里的同步和异步指的是时序，如果在非时钟周期内到达了中断信号，那么就称其为异步。）

**Maskable and Non-maskable**

Maskable – All Interrupt Requests (IRQs) issued by I/O devices give rise to maskable interrupts. A maskable interrupt can be in two states: masked or unmasked; a masked interrupt is ignored by the control unit as long as it remains masked.

Non-maskable – Only a few critical events (such as hardware failures) give rise to nonmaskable interrupts. Non-maskable interrupts are always recognized by the CPU.

**Exception types**

* Falts – Like Divide by zero, Page Fault, Segmentation Fault.

* Traps – Reported immediately following the execution of the trapping instruction. Like Breakpoints.

* Aborts – Aborts are used to report severe errors, such as hardware failures and invalid or inconsistent values in system tables.

**Interrupt handler**

For a device’s each interrupt, its device driver must register an interrupt handler.

An interrupt handler or interrupt service routine (ISR) is the function that the kernel runs in response to a specific interrupt:

1. Each device that generates interrupts has an associated interrupt handler.

1. The interrupt handler for a device is part of the device’s driver (the kernel code that manages the device).


In Linux, interrupt handlers are normal C functions, which match a specific prototype and thus enable the kernel to pass the handler information in a standard way. What differentiates interrupt handlers from other kernel functions is that the kernel invokes them in response to interrupts and that they run in a special context called interrupt context. This special context is occasionally called atomic context because code executing in this context is unable to block.

Because an interrupt can occur at any time, an interrupt handler can be executed at any time. It is imperative that the handler runs quickly, to resume the execution of the interrupted code as soon as possible. It is important that

1. To the hardware: the operating system services the interrupt without delay.
1. To the rest of the system: the interrupt handler executes in as short a period as possible.

Top halves and Bottom halves
Top half
The interrupt handler is the top half. The top half will run immediately upon receipt of the interrupt and performs only the work that is time-critical, such as acknowledging receipt of the interrupt or resetting the hardware.

Bottom half
The bottom half is used to process data, letting the top half to deal with new incoming interrupts. Interrupts are enabled when a bottom half runs. The interrupt can be disabled if necessary, but generally, this should be avoided as this goes against the basic purpose of having a bottom half – processing data while listening to new interrupts. The bottom half runs in the future, at a more convenient time, with all interrupts enabled.

比如网卡接收数据，我们使用 top half 快速地把网络数据包从网卡的缓冲区复制到内存中，然后使用 bottom half 慢慢处理内存中的数据包就可以了。如果 top half 不够快，那么新来的数据就会覆盖掉旧数据，造成读写错误。

Intel processors handle interrupt using IDT (Interrupt Descriptor Table).  The IDT consists of 256 entries with each entry corresponding to a vector and of 8 bytes. All the entries are a pointer to the interrupt handling function. The CPU uses IDTR to point to IDT. The relation between those two can be depicted as below,

### An interrupt example with sysfs

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>                 //kmalloc()
#include <linux/uaccess.h>              //copy_to/from_user()
#include <linux/sysfs.h> 
#include <linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/err.h>
#define IRQ_NO 11

//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq, void *dev_id)
{
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    return IRQ_HANDLED;
}

volatile int etx_value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);

/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);

/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);

struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
 
static struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .read           = etx_read,
    .write          = etx_write,
    .open           = etx_open,
    .release        = etx_release,
};
 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
    printk(KERN_INFO "Sysfs - Read!!!\n");
    return sprintf(buf, "%d", etx_value);
}

static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
    printk(KERN_INFO "Sysfs - Write!!!\n");
    sscanf(buf,"%d",&etx_value);
    return count;
}

static int etx_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Opened...!!!\n");
    return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Closed...!!!\n");
    return 0;
}
 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
    printk(KERN_INFO "Read function\n");
    asm("int $0x3B");  // Corresponding to irq 11
    return 0;
}

static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
    printk(KERN_INFO "Write Function\n");
    return len;
}
 
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0) {
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d \n", MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0) {
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
    }

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs", kernel_kobj);

    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)) {
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
    }
    if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
    }
    printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;

irq:
    free_irq(IRQ_NO,(void *)(irq_handler));

r_sysfs:
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
    class_destroy(dev_class);
r_class:
    unregister_chrdev_region(dev,1);
    cdev_del(&etx_cdev);
    return -1;
}
 
static void __exit etx_driver_exit(void)
{
    free_irq(IRQ_NO,(void *)(irq_handler));
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Interrupts");
MODULE_VERSION("1.9");
```

期望输出：

```
linux@embetronicx-VirtualBox: dmesg

[19743.366386] Major = 246 Minor = 0
[19743.370707] Device Driver Insert...Done!!!
[19745.580487] Device File Opened...!!!
[19745.580507] Read function
[19745.580531] Shared IRQ: Interrupt Occurred
[19745.580540] Device File Closed...!!!
```

实际输出：

```
[162342.126355] Major = 238 Minor = 0 
[162342.138918] Device Driver Insert...Done!!!
[162359.827734] Device File Opened...!!!
[162359.827746] Read function
[162359.827955] __common_interrupt: 2.59 No irq handler for vector
[162359.827974] Device File Closed...!!!
```

If you are using the newer Linux kernel, then this may not work properly. You may get something like below.

`do_IRQ: 1.59 No irq handler for vector`

In order to solve that, you have to change the Linux kernel source code, Compile it, then install it.

build:

```bash
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.47.tar.xz
sudo tar -xvf ../linux-5.4.47.tar
cd linux-5.4.47/
cp -v /boot/config-$(uname -r) .confi
sudo apt install build-essential kernel-package fakeroot libncurses5-dev libssl-dev ccache flex libelf-dev bison libncurses-dev
```

Add the below line in the downloaded Linux kernel file `arch/x86/kernel/irq.c` right after all the include lines.

`EXPORT_SYMBOL(vector_irq);`

```bash
make oldconfig
make menuconfig
sudo make  （也可以并行编译：sudo make -j 4）
sudo su
make modules_install
sudo make install
sudo update-initramfs -c -k 5.4.47
sudo update-grub
reboot
uname -r
```

新版本的 kernel 应该使用的 driver 代码：

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Interrupt Example
*
*  \author     EmbeTronicX
*
*  \Tested with kernel 5.4.47
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <asm/hw_irq.h>
#include <linux/err.h>
#define IRQ_NO 11
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
        struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
        struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
 
static struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .read           = etx_read,
    .write          = etx_write,
    .open           = etx_open,
    .release        = etx_release,
};
 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
    printk(KERN_INFO "Sysfs - Read!!!\n");
    return sprintf(buf, "%d", etx_value);
}
 
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
    printk(KERN_INFO "Sysfs - Write!!!\n");
    sscanf(buf,"%d",&etx_value);
    return count;
}
 
static int etx_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Opened...!!!\n");
    return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Closed...!!!\n");
    return 0;
}
 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
    struct irq_desc *desc;

    printk(KERN_INFO "Read function\n");
    desc = irq_to_desc(11);
    if (!desc) 
    {
        return -EINVAL;
    }
    __this_cpu_write(vector_irq[59], desc);
    asm("int $0x3B");  // Corresponding to irq 11
    return 0;
}

static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
    printk(KERN_INFO "Write Function\n");
    return len;
}
 
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0){
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
    }

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
    }
    if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
    }
    printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;
 
irq:
    free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
    class_destroy(dev_class);
r_class:
    unregister_chrdev_region(dev,1);
    cdev_del(&etx_cdev);
    return -1;
}
 
static void __exit etx_driver_exit(void)
{
    free_irq(IRQ_NO,(void *)(irq_handler));
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Interrupts");
MODULE_VERSION("1.9");
```

### 一个可用的 interrupt example

这个 example 使用 read device file 触发中断。需要重新编译内核。

* 对于新版内核，由于我们不知道第一个 irq vector 的地址，所以只能重新编译内核

    `intrp.c`:

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include<linux/slab.h>                 //kmalloc()
    #include<linux/uaccess.h>              //copy_to/from_user()
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/interrupt.h>
    #include <linux/irqnr.h>
    #include <asm/io.h>
    #include <linux/err.h>
    #include <asm/hw_irq.h>
    #define IRQ_NO 11
    //Interrupt handler for IRQ 11. 
    static irqreturn_t irq_handler(int irq,void *dev_id) {
            printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
            return IRQ_HANDLED;
    }
    volatile int etx_value = 0;
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;
    
    static int __init etx_driver_init(void);
    static void __exit etx_driver_exit(void);
    /*************** Driver Fuctions **********************/
    static int etx_open(struct inode *inode, struct file *file);
    static int etx_release(struct inode *inode, struct file *file);
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len,loff_t * off);
    static ssize_t etx_write(struct file *filp, 
                    const char *buf, size_t len, loff_t * off);
    /*************** Sysfs Fuctions **********************/
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf);
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count);
    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
    
    static struct file_operations fops =
    {
            .owner          = THIS_MODULE,
            .read           = etx_read,
            .write          = etx_write,
            .open           = etx_open,
            .release        = etx_release,
    };
    
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
            printk(KERN_INFO "Sysfs - Read!!!\n");
            return sprintf(buf, "%d", etx_value);
    }
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            printk(KERN_INFO "Sysfs - Write!!!\n");
            sscanf(buf,"%d",&etx_value);
            return count;
    }
    static int etx_open(struct inode *inode, struct file *file)
    {
            printk(KERN_INFO "Device File Opened...!!!\n");
            return 0;
    }
    
    static int etx_release(struct inode *inode, struct file *file)
    {
            printk(KERN_INFO "Device File Closed...!!!\n");
            return 0;
    }

    // extern struct irq_desc* vector_irq;
    
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
         printk(KERN_INFO "Read function\n");
        struct irq_desc *desc;
        desc = irq_to_desc(11);
        if (!desc)
        {
                return -EINVAL;
        }
        __this_cpu_write(vector_irq[59], desc);
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
    }
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
            printk(KERN_INFO "Write Function\n");
            return len;
    }

    
    static int __init etx_driver_init(void)
    {
            /*Allocating Major number*/
            if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                    printk(KERN_INFO "Cannot allocate major number\n");
                    return -1;
            }
            printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
    
            /*Creating cdev structure*/
            cdev_init(&etx_cdev,&fops);
    
            /*Adding character device to the system*/
            if((cdev_add(&etx_cdev,dev,1)) < 0){
                printk(KERN_INFO "Cannot add the device to the system\n");
                goto r_class;
            }
    
            /*Creating struct class*/
            if(IS_ERR(dev_class = class_create("etx_class"))){
                printk(KERN_INFO "Cannot create the struct class\n");
                goto r_class;
            }
    
            /*Creating device*/
            if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
                printk(KERN_INFO "Cannot create the Device 1\n");
                goto r_device;
            }
    
            /*Creating a directory in /sys/kernel/ */
            kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
            /*Creating sysfs file for etx_value*/
            if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                    printk(KERN_INFO"Cannot create sysfs file......\n");
                    goto r_sysfs;
            }
            if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
                printk(KERN_INFO "my_device: cannot register IRQ ");
                        goto irq;
            }
            printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
    irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
    r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    
    r_device:
        class_destroy(dev_class);
    r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
    }
    
    static void __exit etx_driver_exit(void)
    {
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    MODULE_LICENSE("GPL");
    ```

    其中`#include <linux/irqnr.h>`用于提供`irq_to_desc()`函数的声明。

    如果此时直接编译，会看到 undefined symbol 的 error 输出。这时我们需要重新编译内核。

    首先是下载内核，可以直接使用`sudo apt-get install linux-source`，这个命令下载的内核版本通常会有问题。
    
    我们可以先搜索目前可下载的版本：

    `apt-cache search linux-source`

    如果使用`apt-get download linux-source`，那么会在当前目录下载一个`.deb`文件。

    ref: <https://askubuntu.com/questions/159833/how-do-i-get-the-kernel-source-code>

    源码会被下载到`/usr/src`中，分别是一个`.tar.bz2`的压缩包的 symbolic link，和一个已经解压缩的目录，比如`linux-source-6.5.0`。

    我们直接对这个`.tar.bz`的文件解压缩，就能在`/usr/src`下得到内核源码。

    进入目录后，执行`cp -v /boot/config-$(uname -r) .confi`（可能是备份一下配置文件？）

    编译内核前还需要安装几个依赖库：

    `apt-get install bison flex libssl-dev libelf-dev libncurses-dev`

    如果安装得不够，在 make 的时候会报错，按照提示安装就可以了。

    接下来找到`arch/x86/kernel/irq.c`文件，在最后添加一行：

    ```c
    EXPORT_SYMBOL (vector_irq);
    ```

    然后找到`kernel/irq/irqdesc.c`文件，注意这个文件中有两个`irq_to_desc()`函数的定义，我们在 379 行附近找到一个，添加上 export symbol:

    ```c
    struct irq_desc *irq_to_desc(unsigned int irq) 
    {
        return mtree_load(&sparse_irqs, irq);
    }
    EXPORT_SYMBOL(irq_to_desc);
    ```

    另外一个定义在 609 行附近，这个函数已经被 export symbol 过了，我们就不用管了。

    这两个函数定义是被宏控制的，实际编译的时候根据`#ifdef`之类的命令，只会生效一次。由于不知道具体是哪个生效，所以直接把两个处定义都 export 了。

    接下来执行：

    ```bash
    make oldconfig
    make menuconfig
    ```

    基本什么都不用改，保存后退出就可以了。

    然后`sudo vim .config`，把 system tructed keys 之类的清空，不然一会编译会报错：

    ```conf
    CONFIG_SYSTEM_TRUSTED_KEYRING=y
    CONFIG_SYSTEM_TRUSTED_KEYS=""
    ```

    ref:
    
    1. <https://blog.csdn.net/m0_47696151/article/details/121574718>

    2. <https://blog.csdn.net/qq_36393978/article/details/118157426>

    接下来就可以开始编译了：

    ```bash
    sudo make -j4
    ```

    4 线程编译大概要花 20 多分钟。

    编译好后执行：
    
    ```bash
    sudo make modules_install
    sudo make install
    ```

    然后更新引导：

    ```bash
    sudo update-initramfs -c -k 6.5.0
    sudo update-grub
    ```

    这里的`6.5.0`将来会变成`uname -r`的输出。

    最后重启系统：`reboot`，就大功告成了。

    接下来我们正常编译 kernel module，然后`insmode`，再进入`/dev`目录下，执行测试命令：

    ```bash
    sudo cat /dev/etx_device
    ```

    此时可以看到`dmesg` output:

    ```
    [   39.678202] intrp: loading out-of-tree module taints kernel.
    [   39.678390] Major = 240 Minor = 0
    [   39.678709] Device Driver Insert...Done!!!
    [   79.901307] Device File Opened...!!!
    [   79.901314] Read function
    [   79.901317] Shared IRQ: Interrupt Occurred
    [   79.901322] Device File Closed...!!!
    ```

    中断触发成功。

### Work queue

Work queues defer work into a kernel thread; this bottom half always runs in the process context.

Workqueue 将工作推迟到一个内核线程中，这个底层部分总是运行在进程的上下文环境中。

Workqueue is allowing users to create a kernel thread and bind work to the kernel thread.

So, this will run in the process context and the work queue can sleep.

（什么是 process context ? 为什么 kernel thread 会运行在 process context 中？）

Normally, it is easy to decide between using workqueue and softirq/tasklet:

If the deferred work needs to sleep, then workqueue is used.
If the deferred work need not sleep, then softirq or tasklet are used.

注：

* 什么是 softirq？

There are two ways to implement Workqueue in the Linux kernel.

Using global workqueue (Static / Dynamic)
Creating Own workqueue (We will see in the next tutorial)

**Initialize work using Static Method**

The below call creates a workqueue by the name and the function that we are passing in the second argument gets scheduled in the queue.

`DECLARE_WORK(name, void (*func)(void *))`

Where,

* `name`: The name of the “work_struct” structure that has to be created.

* `func`: The function to be scheduled in this workqueue.

Example:

```c
DECLARE_WORK(workqueue,workqueue_fn);
```

Schedule work to the Workqueue
The below functions are used to allocate the work to the queue.

`schedule_work`

This function puts a job in the kernel-global workqueue if it was not already queued and leaves it in the same position on the kernel-global workqueue otherwise.

int schedule_work( struct work_struct *work );

where,

work – job to be done

Returns zero if work was already on the kernel-global workqueue and non-zero otherwise.

Scheduled_delayed_work
After waiting for a given time this function puts a job in the kernel-global workqueue.

int scheduled_delayed_work( struct delayed_work *dwork, unsigned long delay );

where,

dwork – job to be done

delay– number of jiffies to wait or 0 for immediate execution

Schedule_work_on
This puts a job on a specific CPU.

int schedule_work_on( int cpu, struct work_struct *work );

where,

cpu– CPU to put the work task on

work– job to be done

Scheduled_delayed_work_on
After waiting for a given time this puts a job in the kernel-global workqueue on the specified CPU.

int scheduled_delayed_work_on(int cpu, struct delayed_work *dwork, unsigned long delay );
where,

cpu – CPU to put the work task on

dwork – job to be done

delay– number of jiffies to wait or 0 for immediate execution

Delete work from workqueue
There are also a number of helper functions that you can use to flush or cancel work on work queues. To flush a particular work item and block until the work is complete, you can make a call to flush_work. All work on a given work queue can be completed using a call to flush_work. In both cases, the caller blocks until the operation are complete. To flush the kernel-global work queue, call flush_scheduled_work.

`int flush_work( struct work_struct *work );`
`void flush_scheduled_work( void );`
Cancel Work from workqueue
You can cancel work if it is not already executing in a handler. A call to cancel_work_sync will terminate the work in the queue or block until the callback has finished (if the work is already in progress in the handler). If the work is delayed, you can use a call to cancel_delayed_work_sync.

`int cancel_work_sync( struct work_struct *work );`
`int cancel_delayed_work_sync( struct delayed_work *dwork );`
Check the workqueue
Finally, you can find out whether a work item is pending (not yet executed by the handler) with a call to work_pending or delayed_work_pending.

`work_pending( work );`

`delayed_work_pending( work );`

Example:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>                 //kmalloc()
#include <linux/uaccess.h>              //copy_to/from_user()
#include <linux/sysfs.h> 
#include <linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
 
void workqueue_fn(struct work_struct *work); 
 
/*Creating work by Static Method */
DECLARE_WORK(workqueue,workqueue_fn);
 
/*Workqueue Function*/
void workqueue_fn(struct work_struct *work)
{
    printk(KERN_INFO "Executing Workqueue Function\n");
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    schedule_work(&workqueue);
    
    return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This function will be called when we read the sysfs file
*/ 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This function will be called when we write the sysfsfs file
*/
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This function will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return len;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0){
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
    }

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
    }
    if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
    }
    printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Static method)");
MODULE_VERSION("1.10");
```

看起来似乎是`irq_handler()`会调用`schedule_work()`，把繁重的工作分配到 bottom half 去做。推测：`schedule_work()`一定是非阻塞的。

test:

```bash
sudo cat /dev/etx_device
sudo dmesg
```

#### Initialize work using Dynamic Method

The below call (`INIT_WORK`) creates a workqueue in Linux by the name work and the function that gets scheduled in the queue is work_fn.

`INIT_WORK(work,work_fn)`

Where,

`name`: The name of the “work_struct” structure that has to be created.
`func`: The function to be scheduled in this workqueue.

**Schedule work to the Workqueue**
The below functions used to allocate the work to the queue.

`Schedule_work`

This function puts a job in the kernel-global workqueue if it was not already queued and leaves it in the same position on the kernel-global workqueue otherwise.

`int schedule_work( struct work_struct *work );`

where,

`work` – job to be done

Returns zero if work was already on the kernel-global workqueue and non-zero otherwise.

`Scheduled_delayed_work`

After waiting for a given time this function puts a job in the kernel-global workqueue.

`int scheduled_delayed_work( struct delayed_work *dwork, unsigned long delay );`

where,

`dwork` – job to be done

`delay` – number of jiffies to wait or 0 for immediate execution

`Schedule_work_on`

This puts a job on a specific CPU.

`int schedule_work_on( int cpu, struct work_struct *work );`

where,

`cpu` – CPU to put the work task on

`work` – job to be done

`Scheduled_delayed_work_on`

After waiting for a given time this puts a job in the kernel-global workqueue on the specified CPU.

`int scheduled_delayed_work_on(int cpu, struct delayed_work *dwork, unsigned long delay );`

where,

`cpu` – CPU to put the work task on

`dwork` – job to be done

`delay` – number of jiffies to wait or 0 for immediate execution

**Delete work from workqueue**

There are also a number of helper functions that you can use to flush or cancel work on work queues. To flush a particular work item and block until the work is complete, you can make a call to flush_work. All work on a given work queue can be completed using a call to . In both cases, the caller blocks until the operation are complete. To flush the kernel-global work queue, call flush_scheduled_work.

`int flush_work( struct work_struct *work );`

`void flush_scheduled_work( void );`

**Cancel Work from workqueue**

You can cancel work if it is not already executing in a handler. A call to cancel_work_sync will terminate the work in the queue or block until the callback has finished (if the work is already in progress in the handler). If the work is delayed, you can use a call to cancel_delayed_work_sync.

`int cancel_work_sync( struct work_struct *work );`

`int cancel_delayed_work_sync( struct delayed_work *dwork );`

**Check workqueue**

Finally, you can find out whether a work item is pending (not yet executed by the handler) with a call to work_pending or delayed_work_pending.

`work_pending( work );`
`delayed_work_pending( work );`

Example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Global Workqueue - Dynamic method)
*
*  \author     EmbeTronicX
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
/* Work structure */
static struct work_struct workqueue;
 
void workqueue_fn(struct work_struct *work); 
 
/*Workqueue Function*/
void workqueue_fn(struct work_struct *work)
{
        printk(KERN_INFO "Executing Workqueue Function\n");
}
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
        /*Allocating work to queue*/
        schedule_work(&workqueue);
        
        return IRQ_HANDLED;
}
 
volatile int etx_value = 0;
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
*/ 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}
 
/*
** This fuction will be called when we write the sysfsfs file
*/
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This fuction will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return 0;
}
 
/*
** Module Init function
*/ 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
                    goto irq;
        }
 
        /*Creating work by Dynamic Method */
        INIT_WORK(&workqueue,workqueue_fn);
 
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Dynamic method)");
MODULE_VERSION("1.11");
```

The core workqueue is represented by structure struct workqueue_struct, which is the structure onto which work is placed. This work is added to the queue in the top half (Interrupt context) and the execution of this work happened in the bottom half (Kernel context).

The work is represented by structure struct `work_struct`, which identifies the work and the deferral function.

**Create and destroy workqueue structure**

Workqueues are created through a macro called create_workqueue, which returns a workqueue_struct reference. You can remove this workqueue later (if needed) through a call to the destroy_workqueue function.

`struct workqueue_struct *create_workqueue( name );`

`void destroy_workqueue( struct workqueue_struct * );`

You should use create_singlethread_workqueue() for creating a workqueue when you want to create only a single thread for all the processors.

Since create_workqueue and create_singlethread_workqueue() are macros. Both are using the alloc_workqueue function in the background.

```c
#define create_workqueue(name)                    
        alloc_workqueue("%s", WQ_MEM_RECLAIM, 1, (name))
#define create_singlethread_workqueue(name)       
        alloc_workqueue("%s", WQ_UNBOUND | WQ_MEM_RECLAIM, 1, (name))
```

alloc_workqueue
Allocate a workqueue with the specified parameters.

alloc_workqueue ( fmt, flags, max_active );

fmt– printf format for the name of the workqueue

flags – WQ_* flags

max_active – max in-flight work items, 0 for default

This will return Pointer to the allocated workqueue on success, NULL on failure.

WQ_* flags
This is the second argument of alloc_workqueue.

WQ_UNBOUND

Work items queued to an unbound wq are served by the special worker-pools which host workers who are not bound to any specific CPU. This makes the wq behave like a simple execution context provider without concurrency management. The unbound worker-pools try to start the execution of work items as soon as possible. Unbound wq sacrifices locality but is useful for the following cases.

* Wide fluctuation in the concurrency level requirement is expected and using bound wq may end up creating a large number of mostly unused workers across different CPUs as the issuer hops through different CPUs.

* Long-running CPU-intensive workloads which can be better managed by the system scheduler.

WQ_FREEZABLE

A freezable wq participates in the freeze phase of the system suspend operations. Work items on the wq are drained and no new work item starts execution until thawed.

WQ_MEM_RECLAIM

All wq which might be used in the memory reclaim paths MUST have this flag set. The wq is guaranteed to have at least one execution context regardless of memory pressure.

WQ_HIGHPRI

Work items of a highpri wq are queued to the highpri worker-pool of the target CPU. Highpri worker-pools are served by worker threads with elevated nice levels.

Note that normal and highpri worker-pools don’t interact with each other. Each maintains its separate pool of workers and implements concurrency management among its workers.

WQ_CPU_INTENSIVE

Work items of a CPU intensive wq do not contribute to the concurrency level. In other words, runnable CPU-intensive work items will not prevent other work items in the same worker pool from starting execution. This is useful for bound work items that are expected to hog CPU cycles so that their execution is regulated by the system scheduler.

Although CPU-intensive work items don’t contribute to the concurrency level, the start of their executions is still regulated by the concurrency management and runnable non-CPU-intensive work items can delay the execution of CPU-intensive work items.

This flag is meaningless for unbound wq.

Queuing Work to workqueue
With the work structure initialized, the next step is enqueuing the work on a workqueue. You can do this in a few ways.

queue_work
This will queue the work to the CPU on which it was submitted, but if the CPU dies it can be processed by another CPU.
int queue_work( struct workqueue_struct *wq, struct work_struct *work );

Where,

wq – workqueue to use

work – work to queue

It returns false if work was already on a queue, true otherwise.

queue_work_on
This puts work on a specific CPU.
int queue_work_on( int cpu, struct workqueue_struct *wq, struct work_struct *work );

Where,

cpu– cpu to put the work task on

wq – workqueue to use

work– job to be done

`queue_delayed_work`
After waiting for a given time this function puts work in the workqueue.

```c
int queue_delayed_work( struct workqueue_struct *wq,struct delayed_work *dwork, unsigned long delay );
```

Where,

`wq` – workqueue to use

`dwork` – work to queue

`delay`– number of jiffies to wait before queueing or 0 for immediate execution

`queue_delayed_work_on`

After waiting for a given time this puts a job in the workqueue on the specified CPU.

```c
int queue_delayed_work_on( int cpu, struct workqueue_struct *wq,struct delayed_work *dwork, unsigned long delay );
```

Where,

`cpu` – CPU to put the work task on

`wq` – workqueue to use

`dwork` – work to queue

`delay` – number of jiffies to wait before queueing or 0 for immediate execution

Full code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Own Workqueue)
*
*  \author     EmbeTronicX
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
static struct workqueue_struct *own_workqueue;
 
static void workqueue_fn(struct work_struct *work); 
 
static DECLARE_WORK(work, workqueue_fn);
 
 
/*Workqueue Function*/
static void workqueue_fn(struct work_struct *work)
{
    printk(KERN_INFO "Executing Workqueue Function\n");
    return;
        
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred\n");
        /*Allocating work to queue*/
        queue_work(own_workqueue, &work);
        
        return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/ 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/  
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This fuction will be called when we write the sysfsfs file
*/ 
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This fuction will be called when we open the Device file
*/ 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return 0;
}
 
/*
** Module Init function
*/ 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ \n");
                    goto irq;
        }
 
        /*Creating workqueue */
        own_workqueue = create_workqueue("own_wq");
        
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        /* Delete workqueue */
        destroy_workqueue(own_workqueue);
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Own Workqueue)");
MODULE_VERSION("1.12");
```

test:

```bash
sudo cat /dev/etx_device
dmesg
```

Use `ps -aef` command to see our workqueue. You can able to see our workqueue which is `own_wq`.

**Difference between Schedule_work and queue_work**

If you want to use your own dedicated workqueue you should create a workqueue using create_workqueue. At that time you need to put work on your workqueue by using queue_work function.

If you don’t want to create any own workqueue, you can use kernel global workqueue. In that condition, you can use schedule_work function to put your work to global workqueue.

## 常用数据结构

* cache

    * linked list 的一个 example

        ```c
        #include <linux/init.h>
        #include <linux/module.h>
        #include <linux/list.h>

        struct ListNode {
            struct list_head head;
            int my_data;
        };

        struct list_head lst_head;

        int init_mod(void)
        {
            pr_info("in init_mod()...\n");
            INIT_LIST_HEAD(&lst_head);
            struct ListNode node_1 = {
                .my_data = 1
            };
            struct ListNode node_2 = {
                .my_data = 2
            };
            struct ListNode node_3 = {
                .my_data = 3
            };
            struct ListNode node_4 = {
                .my_data = 4
            };
            list_add(&node_1.head, &lst_head);
            list_add(&node_2.head, &lst_head);
            list_add(&node_3.head, &lst_head);
            list_add_tail(&node_4.head, &lst_head);
            struct ListNode *cur;
            int len_count = 0;
            pr_info("traverse list:\n");
            list_for_each_entry(cur, &lst_head, head) {
                pr_info("%d\n", cur->my_data);
                ++len_count;
            }
            pr_info("list len: %d\n", len_count);
            return 0;
        }

        void exit_mod(void)
        {
            pr_info("in exit_mod()...\n");
        }

        module_init(init_mod);
        module_exit(exit_mod);
        MODULE_LICENSE("GPL");
        ```

        makefile:

        ```makefile
        KERN_DIR=/usr/src/linux-source-6.5.0/linux-source-6.5.0
        obj-m += lnk_lst.o
        default:
            $(MAKE) -C $(KERN_DIR) M=$(PWD) modules
        clean:
            rm -f *.o *.ko
        ```

        output:

        ```
        [24310.834664] in init_mod()...
        [24310.834666] traverse list:
        [24310.834667] 3
        [24310.834667] 2
        [24310.834668] 1
        [24310.834668] 4
        [24310.834668] list len: 4
        [24319.849530] in exit_mod()...
        ```

        说明：

        1. 头文件：`#include <linux/list.h>`

        2. `struct list_head`只包含两个成员，一个是`prev`，一个是`next`

            所以 linux kernel 中的链表其实是个双向链表。

        3. 我们在构建自定义的链表数据结构的时候，需要把`struct list_head`作为一个成员：

            ```c
            struct ListNode {
                struct list_head head;
                int my_data;
            };
            ```

        4. `INIT_LIST_HEAD(&lst_head);`用于初始化一个链表头，主要功能就是让 next 和 prev 都指向自己。

            注意这个 list head 不是定义在`ListNode`之中的，它只是一个 head，不存储数据。

            linux 想象中的链表是这样的：

            `list_head` -> `ListNode` -> `ListNode` -> `ListNode` -> `...`

            因此`list_head`只是一个链表头，本身不算一个完整的节点。在执行遍历等操作时，`list_head`都不会被访问到。

        5. `list_add(&node_1.head, &lst_head);`

            在`lst_head`后添加一个新节点`node_1`。

            对于后面的`node_2`，`node_3`等，由于每次都是在`lst_head`后面添加，所以其实是倒序添加的：

            ```
            lst_head
            lst_head -> 1
            lst_head -> 2 -> 1
            lst_head -> 3 -> 2 -> 1
            ```

        6. `list_add_tail(&node_4.head, &lst_head);`是在链表的末尾添加 node 4:

            ```
            lst_head -> 3 -> 2 -> 1 -> 4
            ```

        7. 遍历时，通过给定自定义类型的指针和`lsit_head`在自定义类型中的名称来进行遍历

            ```c
            struct ListNode {
                struct list_head head;
                int my_data;
            };

            // ...

            struct ListNode *cur;
            list_for_each_entry(cur, &lst_head, head) {
                pr_info("%d\n", cur->my_data);
            }
            ```

            可以看到，`cur`是用户自己提供的指针，用来访问当前遍历到的节点；`&lst_head`是链表头，`head`是`struct list_head`成员在用户自定义类型`ListNode`里的名字。

            `list_for_each_entry()`是一个宏，将宏展开可以看到它是如何确定每个`cur`的地址的。主要是根据`offset(head)`找到`head`的相对偏移，然后根据`head`的相对偏移再结合`head`的地址，找到`ListNode`的起始地址，这个地址就是`cur`的值。

        8. 遍历的输出为`3, 2, 1, 4`，链表长度为`4`，说明在遍历时并没有访问到`lst_head`。

### Linked list

Linux implement a Doubly Linked List, which is defined in `/lib/modules/$(uname -r)/build/include/linux/list.h`.

```c
#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD(name) \
    struct list_head name = LIST_HEAD_INIT(name)
struct list_head {
    struct list_head *next;
    struct list_head *prev;
};
```

Usage:

```c
#include <linux/list.h>

struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};
```

Create a linked list head:

```c
LIST_HEAD(linked_list);
```

This macro will create the head node structure in the name of “linked_list” and it will initialize that to its own address.

Example:

```c
struct list_head etx_linked_list = { &etx_linked_list , &etx_linked_list};
```

While creating the head node, it initializes the prev and next pointer to its own address. This means that prev and next pointer points to itself. The node is empty If the node’s prev and next pointer points to itself.

（不是很懂，创建了 head note 后，不应该已经有一个 node 了吗，为什么还说他是空的？猜测这里的 head 是一个 dummy head）

**Create Node in Linked List**

```c
INIT_LIST_HEAD(struct list_head *list);
```

可以用上面的宏进行初始化链表。

Example:

```c
struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};

struct my_list new_node;

INIT_LIST_HEAD(&new_node.list);
new_node.data = 10;
```

**Add Node to Linked List**

* Add after Head Node

    Insert a new entry after the specified head:

    `inline void list_add(struct list_head *new, struct list_head *head);`

    Where,

    `struct list_head * new` – the new entry to be added

    `struct list_head * head` – list head to add it after

    Example:

    ```c
    list_add(&new_node.list, &etx_linked_list);
    ```

* Add before Head Node

    Insert a new entry before the specified head：

    `inline void list_add_tail(struct list_head *new, struct list_head *head);`

    Where,

    `struct list_head * new` – a new entry to be added

    `struct list_head * head` – list head to add before the head

    Example:

    ```c
    list_add_tail(&new_node.list, &etx_linked_list);
    ```

* Delete Node from Linked List

    * `list_del`

        It will delete the entry node from the list. This function removes the entry node from the linked list by disconnecting prev and next pointers from the list, but it doesn’t free any memory space allocated for the entry node.

        `inline void list_del(struct list_head *entry);`

        Where,

        `struct list_head * entry` – the element to delete from the list.

    * `list_del_init`

        It will delete the entry node from the list and reinitialize it. This function removes the entry node from the linked list by disconnecting prev and next pointers from the list, but it doesn’t free any memory space allocated for the entry node.

        `inline void list_del_init(struct list_head *entry);`

        Where,

        `struct list_head * entry` – the element to delete from the list.

* Replace Node in Linked List

    * `list_replace`

        This function is used to replace the old node with the new node.

        `inline void list_replace(struct list_head *old, struct list_head *new);`

        Where,

        `struct list_head * old` – the element to be replaced

        `struct list_head * new` – the new element to insert

        If old was empty, it will be overwritten.

    * `list_replace_init`

        This function is used to replace the old node with the new node and reinitialize the old entry.

        `inline void list_replace_init(struct list_head *old, struct list_head *new);`

        Where,

       `struct list_head * old` – the element to be replaced

       `struct list_head * new` – the new element to insert

        If old was empty, it will be overwritten.

* Moving Node in Linked List

    * `list_move`

        This will delete one list from the linked list and again adds to it after the head node.

        inline void list_move(struct list_head *list, struct list_head *head);

        Where,

        `struct list_head * list` – the entry to move

        `struct list_head * head` – the head that will precede our entry

    * `list_move_tail`

        This will delete one list from the linked list and again adds it before the head node.

        inline void list_move_tail(struct list_head *list, struct list_head *head);

        Where,

        `struct list_head * list` – the entry to move

        `struct list_head * head` – the head that will precede our entry

* Rotate Node in Linked List

    This will rotate the list to the left.

    `inline void list_rotate_left(struct list_head *head);`

    Where,

    head – the head of the list

* Test the Linked List Entry

    * `list_is_last`

        This tests whether list is the last entry in the list head.

        inline int list_is_last(const struct list_head *list, const struct list_head *head);

            Where,

        const struct list_head * list – the entry to test

        const struct list_head * head – the head of the list

        It returns 1 if it is the last entry otherwise 0.

    * `list_empty`

        It tests whether a list is empty or not.

        `inline int list_empty(const struct list_head *head);`

        Where,

        `const struct list_head * head` – the head of the list

        It returns 1 if it is empty otherwise 0.

    * `list_is_singular`

        This will test whether a list has just one entry.

        `inline int list_is_singular(const struct list_head *head);`

        Where,

        const struct list_head * head – the head of the list

        It returns 1 if it has only one entry otherwise 0.

* Split Linked List into two parts

    This cut a list into two.
    This helper moves the initial part of head, up to and including entry, from head to list. You should pass on entry an element you know is on head. list should be an empty list or a list you do not care about losing its data.

    inline void list_cut_position(struct list_head *list, struct list_head *head, struct list_head *entry);

    Where,

    struct list_head * list – a new list to add all removed entries

    struct list_head * head– a list with entries

    struct list_head * entry– an entry within the head could be the head itself and if so we won’t cut the list

* Join Two Linked Lists

    This will join two lists, this is designed for stacks.
    inline void list_splice(const struct list_head *list, struct list_head *head);

    Where,

    const struct list_head * list – the new list to add.

    struct list_head * head – the place to add it in the first list.

* Traverse Linked List

    * `list_entry`

        This macro is used to get the struct for this entry.
        
        list_entry(ptr, type, member);

        ptr– the struct list_head pointer.

        type – the type of the struct this is embedded in.

        member – the name of the list_head within the struct.

    * `list_for_each`

        This macro is used to iterate over a list.
        
        list_for_each(pos, head);

        pos –  the &struct list_head to use as a loop cursor.

        head –  the head for your list.

    * `list_for_each_entry`

        This is used to iterate over a list of the given type.

        ```c
        list_for_each_entry(pos, head, member);
        ```

        pos – the type * to use as a loop cursor.

        head – the head for your list.

        member – the name of the list_head within the struct.

    * `list_for_each_entry_safe`

        This will iterate over the list of given type-safe against the removal of list entry.

        `list_for_each_entry_safe ( pos, n, head, member);`

        Where,

        pos – the type * to use as a loop cursor.

        n – another type * to use as temporary storage

        head – the head for your list.

        member – the name of the list_head within the struct.

    * `list_for_each_prev`

        This will be used to iterate over a list backward.

        list_for_each_prev(pos, head);

        pos – the &struct list_head to use as a loop cursor.

        head – the head for your list.

    * `list_for_each_entry_reverse`

        This macro is used to iterate backward over the list of the given type.
        
        list_for_each_entry_reverse(pos, head, member);

        pos – the type * to use as a loop cursor.

        head  the head for your list.

        member – the name of the list_head within the struct.

1. When we write the value to our device file using echo value > /dev/etx_value, it will invoke the interrupt. Because we configured the interrupt by using the software. If you don’t know how it works, please refer to this tutorial.

1. The interrupt will invoke the ISR function.

1. In ISR we are allocating work to the Workqueue.

1. Whenever Workqueue executes, we are creating the Linked List Node and adding the Node to the Linked List.

1. When we are reading the driver using cat /dev/etx_device, printing all the nodes which are present in the Linked List using traverse.

1. When we are removing the driver using rmmod, it will remove all the nodes in Linked List and free the memory.

Creating Head Node

```c
/*Declare and init the head node of the linked list*/
LIST_HEAD(Head_Node);
```

This will create the head node in the name of `Head_Node` and initialize that.

Creating Node and add that into Linked List

```c
/*Creating Node*/
temp_node = kmalloc(sizeof(struct my_list), GFP_KERNEL);

/*Assgin the data that is received*/
temp_node->data = etx_value;

/*Init the list within the struct*/
INIT_LIST_HEAD(&temp_node->list);

/*Add Node to Linked List*/
list_add_tail(&temp_node->list, &Head_Node);
```

Traversing Linked List:

```c
struct my_list *temp;
int count = 0;
printk(KERN_INFO "Read function\n");

/*Traversing Linked List and Print its Members*/
list_for_each_entry(temp, &Head_Node, list) {
    printk(KERN_INFO "Node %d data = %d\n", count++, temp->data);
}

printk(KERN_INFO "Total Nodes = %d\n", count);
```

Deleting Linked List:

```c
/* Go through the list and free the memory. */
struct my_list *cursor, *temp;
list_for_each_entry_safe(cursor, temp, &Head_Node, list) {
    list_del(&cursor->list);
    kfree(cursor);
}
```

source code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Kernel Linked List)
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
volatile int etx_value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
static struct workqueue_struct *own_workqueue;
 
 
static void workqueue_fn(struct work_struct *work); 
 
static DECLARE_WORK(work, workqueue_fn);
 
/*Linked List Node*/
struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};
 
/*Declare and init the head node of the linked list*/
LIST_HEAD(Head_Node);
 
/*
** Function Prototypes
*/ 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
/******************************************************/
 
 
/*Workqueue Function*/
static void workqueue_fn(struct work_struct *work)
{
        struct my_list *temp_node = NULL;
 
        printk(KERN_INFO "Executing Workqueue Function\n");
        
        /*Creating Node*/
        temp_node = kmalloc(sizeof(struct my_list), GFP_KERNEL);
 
        /*Assgin the data that is received*/
        temp_node->data = etx_value;
 
        /*Init the list within the struct*/
        INIT_LIST_HEAD(&temp_node->list);
 
        /*Add Node to Linked List*/
        list_add_tail(&temp_node->list, &Head_Node);
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred\n");
        /*Allocating work to queue*/
        queue_work(own_workqueue, &work);
        
        return IRQ_HANDLED;
}

/*
** File operation sturcture
*/ 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/  
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This fuction will be called when we write the sysfsfs file
*/  
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        return count;
}

/*
** This fuction will be called when we open the Device file
*/ 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/   
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        struct my_list *temp;
        int count = 0;
        printk(KERN_INFO "Read function\n");
 
        /*Traversing Linked List and Print its Members*/
        list_for_each_entry(temp, &Head_Node, list) {
            printk(KERN_INFO "Node %d data = %d\n", count++, temp->data);
        }
 
        printk(KERN_INFO "Total Nodes = %d\n", count);
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        /*Copying data from user space*/
        sscanf(buf,"%d",&etx_value);
        /* Triggering Interrupt */
        asm("int $0x3B");  // Corresponding to irq 11
        return len;
}
 
/*
** Module Init function
*/  
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device \n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ \n");
                    goto irq;
        }
 
        /*Creating workqueue */
        own_workqueue = create_workqueue("own_wq");
        
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module Exit function
*/ 
static void __exit etx_driver_exit(void)
{
 
        /* Go through the list and free the memory. */
        struct my_list *cursor, *temp;
        list_for_each_entry_safe(cursor, temp, &Head_Node, list) {
            list_del(&cursor->list);
            kfree(cursor);
        }
 
        /* Delete workqueue */
        destroy_workqueue(own_workqueue);
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Kernel Linked List");
MODULE_VERSION("1.13");
```

test:

```bash
cat /dev/etx_device
echo 10 > /dev/etx_device
```

### Radix Tree

## Thread

There are two types of threads.

1. User Level Thread

    In this type, the kernel is not aware of these threads. Everything is maintained by the user thread library. That thread library contains code for creating and destroying threads, for passing messages and data between threads, for scheduling thread execution, and for saving and restoring thread contexts. So all will be in User Space.

1. Kernel Level Thread

    Kernel level threads are managed by the OS, therefore, thread operations are implemented in the kernel code. There is no thread management code in the application area.

kernel level thread

header file: `linux/kthread.h`

Create Kernel Thread

* `kthread_create`

    Syntax:

    ```c
    struct task_struct * kthread_create (int (* threadfn(void *data), void *data, const char namefmt[], ...);
    ```

    Where,

    `threadfn` – the function to run until signal_pending(current).

    `data` – data ptr for threadfn.

    `namefmt[]` – printf-style name for the thread.

    `...` – variable arguments

    This helper function creates and names a kernel thread. But we need to wake up that thread manually. When woken, the thread will run `threadfn()` with data as its argument.

    `threadfn` can either call `do_exit` directly if it is a standalone thread for which no one will call `kthread_stop`, or return when ‘`kthread_should_stop`‘ is true (which means `kthread_stop` has been called). The return value should be zero or a negative error number; it will be passed to `kthread_stop`.

    It Returns `task_struct` or `ERR_PTR(-ENOMEM)`.

Start Kernel Thread

* `wake_up_process`

    Syntax:

    ```c
    int wake_up_process (struct task_struct * p);
    ```

    p – The process to be woken up.

    Attempt to wake up the nominated process and move it to the set of runnable processes.

    It returns 1 if the process was woken up, 0 if it was already running.

    It may be assumed that this function implies a write memory barrier before changing the task state if and only if any tasks are woken up.

Stop Kernel Thread

* `kthread_stop`

    Syntax:

    ```c
    int kthread_stop ( struct task_struct *k);
    ```

    Where,

    `k` – thread created by kthread_create.

    Sets kthread_should_stop for k to return true, wakes it and waits for it to exit. Your threadfn must not call do_exit itself, if you use this function! This can also be called after kthread_create instead of calling wake_up_process: the thread will exit without calling threadfn.

    It Returns the result of threadfn, or –EINTR if wake_up_process was never called.

Other functions in Kernel Thread

* `kthread_should_stop`

    ```c
    int kthread_should_stop (void);
    ```

    When someone calls `kthread_stop` on your kthread, it will be woken and this will return `true`. You should then return, and your return value will be passed through to `kthread_stop`.

    相当于由外部通知当前线程可以结束了。这种常见的场景，比如执行`ping`命令，默认一直发送 icmp 包，如果没有外部`Ctrl + C`信号，则不会主动停止。

* `kthread_bind`

    ```c
    void kthread_bind(struct task_struct *k, unsigned int cpu);
    ```

    `k` – thread created by kthread_create.

    `cpu` – CPU (might not be online, must be possible) for k to run on.


Thread Function

First, we have to create our thread that has the argument of void *  and should return int value.  We should follow some conditions in our thread function. It is advisable.

* If that thread is a long run thread, we need to check `kthread_should_stop()` every time, because any function may call kthread_stop. If any function called kthread_stop, that time `kthread_should_stop` will return true. We have to exit our thread function if true value been returned by `kthread_should_stop`.

* But if your thread function is not running long, then let that thread finish its task and kill itself using do_exit.

Example:

```c
int thread_function(void *pv) 
{
    int i=0;
    while(!kthread_should_stop())
    {
        printk(KERN_INFO "In EmbeTronicX Thread Function %d\n", i++);
        msleep(1000);
    } 
    return 0; 
}
```

```c
static struct task_struct *etx_thread; 

etx_thread = kthread_create(thread_function,NULL,"eTx Thread");

if (etx_thread) 
{
    wake_up_process(etx_thread); 
} 
else 
{
    printk(KERN_ERR "Cannot create kthread\n"); 
}
```

`kthread_run`:

Syntax:

```c
kthread_run (threadfn, data, namefmt, ...);
```

Where,

`threadfn` – the function to run until signal_pending(current).

`data` – data ptr for threadfn.

`namefmt` – printf-style name for the thread.

`...` – variable arguments

Convenient wrapper for `kthread_create` followed by `wake_up_process`.

It returns the `kthread` or `ERR_PTR(-ENOMEM)`.

Example:

```c
static struct task_struct *etx_thread;

etx_thread = kthread_run(thread_function,NULL,"eTx Thread"); 
if(etx_thread) 
{
 printk(KERN_ERR "Kthread Created Successfully...\n");
}
else 
{
 printk(KERN_ERR "Cannot create kthread\n"); 
}
```

Source code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Kernel Thread)
*
*  \author     EmbeTronicX
*
*  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include <linux/kthread.h>             //kernel threads
#include <linux/sched.h>               //task_struct 
#include <linux/delay.h>
#include <linux/err.h>
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
static struct task_struct *etx_thread;
 
/*
** Function Prototypes
*/
/*************** Driver functions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 /******************************************************/
 
int thread_function(void *pv);

/*
** Thread
*/
int thread_function(void *pv)
{
    int i=0;
    while(!kthread_should_stop()) {
        pr_info("In EmbeTronicX Thread Function %d\n", i++);
        msleep(1000);
    }
    return 0;
}

/*
** File operation sturcture
*/ 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This function will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/   
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read function\n");
 
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write Function\n");
        return len;
}

/*
** Module Init function
*/  
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_err("Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_err("Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device \n");
            goto r_device;
        }
 
        etx_thread = kthread_create(thread_function,NULL,"eTx Thread");
        if(etx_thread) {
            wake_up_process(etx_thread);
        } else {
            pr_err("Cannot create kthread\n");
            goto r_device;
        }
#if 0
        /* You can use this method also to create and run the thread */
        etx_thread = kthread_run(thread_function,NULL,"eTx Thread");
        if(etx_thread) {
            pr_info("Kthread Created Successfully...\n");
        } else {
            pr_err("Cannot create kthread\n");
             goto r_device;
        }
#endif
        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
 
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/  
static void __exit etx_driver_exit(void)
{
        kthread_stop(etx_thread);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Kernel Thread");
MODULE_VERSION("1.14");
```

Latest progress: <https://embetronicx.com/tutorials/linux/device-drivers/tasklet-static-method/>

## Miscellaneous

* modprobe 的作用

    在 linux 中，如果一些内核模块之间有依赖关系，那么必须按依赖关系进行`insmod`，否则会报错。

    `modprobe`会根据`depmod`所产生的相依关系，决定要载入哪些模块。若在载入过程中发生错误，在`modprobe`会卸载整组的模块。

    example:

    载入模块：

    ```bash
    sudo modprobe -v xdxgpu
    ```

    尝试制裁模块：

    ```bash
    sudo modprobe -vr xdxgpu
    ```

* depmod 命令

    `depmod`通常在`modprobe`之前运行，用于分析可载入模块的依赖关系。

    example:

    分析所有可用模块的依赖关系：

    ```bash
    sudo depmod -av
    ```

    注：

    1. 这里没有加路径，可能是会分析到当前目录下？

## cache

* 详细讲讲 qemu 中的 pxb-pcie

    `pxb-pcie` 是 QEMU 中为 PCIe 机器（如 q35 类型）设计的一种特殊设备，全称为 **PCI Express 扩展总线 (PCI Express Expander Bus)**。它的核心作用是**在虚拟的 PCIe 根总线上，再创建一个或多个独立的 PCIe 根总线 (Root Bus)**。你可以把它理解为一个“虚拟的、可扩展的 PCIe 根控制器”。

    ### 为什么需要 pxb-pcie？

    在 QEMU 中，默认的 PCIe 根总线 `pcie.0` 是设备挂载的主要位置，但存在一些限制。`pxb-pcie` 主要解决以下问题：

    1.  **设备数量与热插拔**：`pcie.0` 本身不直接支持热插拔，且可连接的设备类型有限。要连接多个设备或实现热插拔，必须在其下挂载 `pcie-root-port`。而 `pxb-pcie` 能创建一个全新的根总线域，你可以在此域下自由挂载更多的 `pcie-root-port` 来连接海量设备。
    2.  **资源隔离与 NUMA 亲和性**：通过创建独立的根总线，可以将不同组的 PCIe 设备与特定的 **NUMA 节点 (numa_node)** 绑定。这对于优化多路 CPU 系统的访存性能至关重要，可以避免设备 DMA 操作跨 NUMA 节点访问远端内存。
    3.  **IOMMU 分组**：在直通（Passthrough）场景中，`pxb-pcie` 创建的独立根总线可以作为独立的 IOMMU 组。这使得用户可以更精细地控制设备直通，例如，为新总线单独关联一个 SMMUv3 实例，实现设备间的 DMA 隔离。

    ### 核心特性

    *   **设备类型**：它模拟的是一个 PCIe 根复合体 (Root Complex) 的一部分，而不是一个 PCI-PCI 桥。因此，它向客户机呈现的是一个 PCIe 根总线，而不是传统 PCI 总线。
    *   **与总线号的注意点**：在 QEMU 命令行创建 `pxb-pcie` 时，必须显式指定其总线号范围（通过 `bus_nr` 参数）。若设置不当（如与现有根端口的总线号冲突），可能会导致客户机 BIOS/UEFI 枚举设备时出现问题，造成部分设备“丢失”。

    ### 使用示例

    以下是一个典型的 QEMU 命令行配置，展示了如何创建一个新的 PCIe 根总线并在其上挂载设备：

    ```bash
    # 创建一个 id 为 pcie.1 的新根总线，从总线号 8 开始
    -device pxb-pcie,id=pcie.1,bus_nr=8

    # 在新总线 pcie.1 下挂载一个根端口（用于支持热插拔和多设备）
    -device pcie-root-port,id=root_port1,bus=pcie.1,chassis=1,slot=1

    # 最后，将实际的 PCIe 设备（如 virtio-net）挂载到该根端口下
    -device virtio-net-pci,bus=root_port1
    ```

    ### 在复杂场景（如 SMMU）中的行为

    在 ARM 虚拟化等高级场景中，`pxb-pcie` 的行为涉及更底层的细节。QEMU 在实现上，`pxb-pcie` 的父设备指向默认根总线（如 `pcie.0`）。这在关联 IOMMU 时曾引起问题：为新总线 `pcie.1` 关联 SMMUv3 时，QEMU 可能会错误地回退到父总线 `pcie.0` 的 SMMU 配置。

    为了解决这个问题，QEMU 引入了 `pci_setup_iommu_per_bus()` 这类机制，明确标记 IOMMU 是“每总线 (per-bus)”的，从而确保设备能正确使用其所在根总线专属的 IOMMU，而不是错误地继承父总线的配置。这也体现了 `pxb-pcie` 在实现精细化资源管理时的底层复杂性。

    * -device intel-iommu,caching-mode=on
    详细讲讲这行参数的作用

        `-device intel-iommu,caching-mode=on` 这行参数的核心作用是**开启 vIOMMU（虚拟Intel IOMMU）的“缓存模式” (Caching Mode)**，其最主要且近乎强制的应用场景，是**为了让物理设备能够通过 VFIO 技术安全地直通 (Passthrough) 给虚拟机使用**。

        如果没有开启 `caching-mode=on`，当你尝试直通一个物理设备时，QEMU 会直接报错并阻止操作。这背后的原因和工作原理，可以从以下几个方面来理解：

        ### 📜 核心机制：为直通设备同步IOMMU页表

        1.  **基本原理**：当把一个物理设备直通给虚拟机时，这个设备使用的 **物理 IOMMU** 必须与虚拟机内部看到的 **虚拟 IOMMU (vIOMMU)** 状态保持同步。虚拟机的操作系统会操作 vIOMMU 的页表来为直通设备分配 DMA 内存地址。

        2.  **“缓存模式”的作用**：Intel 的规范定义了一种“缓存模式”能力。当一个 vIOMMU 开启了此模式，它就会告知 Guest 操作系统：“**在每次建立新的内存映射操作之后，都必须发出一次 TLB 刷新 (Invalidation) 指令**”。

        3.  **为何如此重要**：这个“必须刷新”的要求，就成为了 QEMU 监控和介入的“钩子”。当 Guest 发出刷新指令时，QEMU 就能捕获到这个事件，并趁机检查 vIOMMU 页表的变化，然后将这些变化**同步更新到物理 IOMMU 的页表中**。这样一来，直通的物理设备就能正确地访问虚拟机指定的内存了。

        ### ⚖️ 性能影响：一个权衡

        开启 `caching-mode=on` 会带来一定的性能开销。因为它要求 vIOMMU 和物理 IOMMU 的页表始终保持同步，这个同步过程本身是相对昂贵的。

        Linux 内核为了缓解这一问题，过去曾利用 `caching-mode` 的这一特性，选择使用开销更小的 **页级别刷新 (Page-specific Flush)**，而非成本高昂的 **全域刷新 (Domain/Global Flush)**。然而，在某些较新的内核版本中，当使用批量的 TLB 刷新机制时，可能会频繁触发全域刷新，这反而会加重 QEMU 同步页表的负担，导致性能下降。

        ### 💎 总结与最佳实践

        简而言之，`caching-mode=on` 是 **VFIO 设备直通的“强制通行证”**。它的存在是为了解决虚拟和物理两个世界之间 IOMMU 状态同步的根本问题。

        *   **何时必须开启**：只要你的 QEMU 命令行中包含了 `-device vfio-pci` 来直通任何物理设备，并且启用了 vIOMMU (`-device intel-iommu`)，就必须添加 `caching-mode=on`。
        *   **性能考量**：开启它确实会引入额外的开销。因此，一个实践中的优化思路是：如果虚拟机中**既需要直通设备，也需要高性能的虚拟设备**，可以考虑创建两个独立的 vIOMMU 实例——一个开启 `caching-mode` 专门服务于直通设备，另一个则关闭它以获得最佳性能。

* 我想在 qemu 里模拟一个 pci 设备，类似芯片架构设计的 archmodel 或 cmodel，该怎么办

    在 QEMU 中模拟 PCI 设备是一个很好的验证硬件设计的方法。以下是几种主要的实现方式：

    ## 1. 使用 QEMU 的 PCI 设备框架

    ### 基本步骤：

    ```c
    // 1. 定义 PCI 设备结构体
    typedef struct MyPCIDevice {
        PCIDevice pdev;
        MemoryRegion mmio;
        uint32_t regs[REGS_COUNT];
        // 其他设备特定字段
    } MyPCIDevice;

    // 2. 实现 PCI 配置空间操作
    static void mypci_realize(PCIDevice *pdev, Error **errp)
    {
        MyPCIDevice *d = MYPCI_DEVICE(pdev);
        
        // 设置 PCI 配置空间
        pci_config_set_interrupt_pin(pdev->config, 1);
        
        // 映射 MMIO 区域
        memory_region_init_io(&d->mmio, OBJECT(d), &mypci_mmio_ops, d,
                             "mypci-mmio", MYPCI_MMIO_SIZE);
        pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &d->mmio);
    }

    // 3. 实现 MMIO 操作
    static const MemoryRegionOps mypci_mmio_ops = {
        .read = mypci_mmio_read,
        .write = mypci_mmio_write,
        .endianness = DEVICE_LITTLE_ENDIAN,
        .valid = {
            .min_access_size = 4,
            .max_access_size = 4,
        },
    };
    ```

    ## 2. 基于现有框架的快速开发

    ### 使用 QEMU 的 PCI 设备模板：

    ```c
    // 在 hw/misc/ 目录下创建新设备
    #define TYPE_MYPCI_DEVICE "mypci-device"
    OBJECT_DECLARE_SIMPLE_TYPE(MyPCIDevice, MYPCI_DEVICE)

    struct MyPCIDevice {
        PCIDevice parent_obj;
        MemoryRegion bar0;
        QEMUTimer *timer;
        // 设备寄存器
        uint32_t control_reg;
        uint32_t status_reg;
        uint32_t data_reg;
    };
    ```

    ## 3. 集成到 QEMU 构建系统

    ### 修改编译配置：

    ```makefile
    # 在 hw/misc/meson.build 中添加
    softmmu_ss.add(when: 'CONFIG_MYPCI', if_true: files('mypci.c'))

    # 在 hw/misc/Kconfig 中添加
    config MYPCI
        bool
        select PCI
    ```

    ## 4. 使用 QEMU 的 QOM（QEMU Object Model）

    ```c
    // 定义设备类
    static void mypci_class_init(ObjectClass *klass, void *data)
    {
        DeviceClass *dc = DEVICE_CLASS(klass);
        PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
        
        k->realize = mypci_realize;
        k->vendor_id = 0x1234;
        k->device_id = 0x5678;
        k->revision = 0x01;
        k->class_id = PCI_CLASS_OTHERS;
        
        dc->desc = "My PCI Device";
    }

    static const TypeInfo mypci_info = {
        .name = TYPE_MYPCI_DEVICE,
        .parent = TYPE_PCI_DEVICE,
        .instance_size = sizeof(MyPCIDevice),
        .class_init = mypci_class_init,
        .interfaces = (InterfaceInfo[]) {
            { INTERFACE_CONVENTIONAL_PCI_DEVICE },
            { },
        },
    };
    ```

    ## 5. 启动和测试

    ### 启动命令：
    ```bash
    qemu-system-x86_64 \
        -machine q35 \
        -device mypci-device,id=mypci0 \
        -kernel your_kernel \
        -append "console=ttyS0"
    ```

    ### 在 guest 系统中检查：
    ```bash
    lspci -v
    dmesg | grep -i pci
    ```

    ## 6. 高级功能

    ### DMA 支持：
    ```c
    static void mypci_dma_setup(MyPCIDevice *d)
    {
        d->dma_as = pci_get_address_space(&d->pdev);
        // 设置 DMA 操作
    }
    ```

    ### 中断支持：
    ```c
    static void mypci_raise_irq(MyPCIDevice *d)
    {
        pci_irq_assert(&d->pdev);
    }

    static void mypci_lower_irq(MyPCIDevice *d)
    {
        pci_irq_deassert(&d->pdev);
    }
    ```

    ## 7. 调试技巧

    ### 添加调试输出：
    ```c
    #define MYPCI_DEBUG 1
    #if MYPCI_DEBUG
    #define DPRINTF(fmt, ...) \
        do { fprintf(stderr, "MYPCI: " fmt, ## __VA_ARGS__); } while (0)
    #else
    #define DPRINTF(fmt, ...) \
        do { } while (0)
    #endif
    ```

    ## 8. 参考现有实现

    学习 QEMU 源码中的 PCI 设备实现：
    - `hw/net/e1000e.c` - 网络设备
    - `hw/block/virtio-blk.c` - 块设备
    - `hw/misc/ivshmem.c` - 共享内存设备

    这种方法可以让你创建一个功能完整的 PCI 设备模型，用于架构验证和软件开发。



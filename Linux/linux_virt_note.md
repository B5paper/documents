# Linux Virtualization Note

## cache

* iommu 全称为 input output memory management unit

    mmu 的作用为将 cpu-visible virtual address map 到 physical address

    iommu 的作用为将 device-visible virtual address map 到 physical address

    这里的 physical address 指的都是 cpu memory address

    iommu 可以对 device 展示连续内存，而在实际申请内存时使用碎片内存。

    有些设备只有 32-bit 的寻址能力，iommu 可以扩展设备的寻址能力。

    因为 iommu 对设备提供的都是虚拟地址，所以可以在不同设备间做地址隔离。

    当虚拟设备透传给虚拟机时，虚拟机分配给 device 的是用户态的虚拟内存地址，device 拿用户态的地址和物理内存通信，肯定是有问题的。iommu 可以解决这个问题。

    iommu 实际上使得 dma 成为可能。

    ref: <https://michael2012z.medium.com/iommu-b59e2dc320bd>
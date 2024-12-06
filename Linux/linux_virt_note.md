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

* iommu 的新理解

    假如 device 希望向 host memory 读写数据，device 希望的方式是看到一块连续的内存，根据起始地址和偏移地址直接写就可以了。但是 host memory 通常是不连续的，因为内存是 user 使用`malloc()`申请的，在 linux kernel 里实际上给他分配的是不连续的页表（page），对应的是不连续的物理内存。linux kernel 做了一层封装，使得 user 看到的地址是连续的。

    现在 device 也希望看到连续的 host 内存地址，这样比较好处理。iommu 的需求就出现了：把不连续的 host memory 对 device 映射为连续的 memory。

    iommu 的基本映射任务完成后，由于给 device 的是虚拟地址，所以也顺带增加了内存访问保护，内存隔离等的功能。
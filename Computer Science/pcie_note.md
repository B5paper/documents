# PCIe Note

## cache

* pci 总线空间与处理器空间隔离。pci 设备具有独立的地址空间，即 pci 总线地址空间。

* host 主桥有许多缓冲，从而处理器总线和 pci 总线可以工作在各自的时钟频率中。

    cpu，memory 和 pci 都接在 host 主桥上。

* host 主桥还具有地址转换的功能

* host 主桥会管理一条 pci 总线，pci 总线可以继续扩展新的 pci 总线，形成一个总线树

    1 棵 pci 总线树上，最多只能挂载 256 个 pci 设备（包括 pci 桥）

* 同一条 pci 总线上的设备可以直接通信，不同 pci 总线上的设备可以通过 pci 桥转发进行通信

* pci 桥的配置空间有管理 pci 总线子树的配置寄存器

* pci 桥向上（向处理器方向）对接上游总线（primary bus），向下接一条新的下游总线（secondary bus）

* pci 总线提出了 MSI (Message Signal Interrupt) 机制

* x86 处理器使用南北桥结构，处理器内核在一个芯片中，host 主桥在北桥中

* host 主桥可能有多个，比如 host 主桥 x，host 主桥 y

* host 主桥下可以接 pci 设备，也可以接 pci 桥

* pci 总线频率越高，能挂载的负载越少（为什么？）

* pci 总线上可以挂载多个设备，其上的信号有地址/数据信号，控制信号，仲裁信号，中断信号

* pci 复用地址与数据信号，pci 总线启动后第一个时钟周期传送地址，这个地址是 pci 总线域的存储器地址或者 IO 地址，第二个时钟周期传送数据。

    什么是 pci 总线域的存储器地址，IO 地址？

* pci 总线也支持在一个地址周期后，紧跟多个数据周期

* pcie 的功能

    pcie 的全称是 peripheral component interconnect express

    pcie is a high-speed serial bus

    features:

    1. point-to-point

    2. bi-directional

    3. scalable

    4. backwards compatible

    5. widely adopted

    pcie 上的两个 device 可以直接通信，他们通信使用的概念叫 data link。

    一个 data link 由 1 or more lanes 组成.

    pcie 使用 dual simplex 来完成双向通信，其中 simplex 指的是 one-way communication on a physical connection。dual means two. 因此 dual complex 指的是 two physical connections。

    每个 dual simplex 叫做 one lane。

    throughput options: 1, 2, 4, 8, or 16 lanes

    technically the PCIe standard allows up to 32 lanes, but in practice the max is 16 lanes

    all the lanes used by two devices for communication consist of one **data link**.

    猜测：构成 lane 的 physical connection 应该在 pcie 芯片的内部，使用的是并行的电路。

    approximate data throughput rates table:

    | PCIe generation | one lane passthrouth rate |
    | - | - |
    | PCIe 2 | ~0.5 GB/s |
    | PCIe 3 | ~1 GB/s |
    | PCIe 4 | ~2 GB/s |
    | PCIe 5 | ~4 GB/s |
    | PCIe 6 | ~8 GB/s |

* pcie 每个 simplex path 的时钟信号采用 common clock，即外部的一个时钟，同时给两个 device 发送时钟信号。

    不同代的 pcie 有不同的 PLL，CDR （Clock & Data Recovery）。

    更高频率的时钟信号会导致更多的比特传输错误，纠错的工作就越多，这样又反过来制约了传输速度。

* PCIe 5+ 可以用示波器（Oscilloscope）或者相位噪声分析仪（phase noise analyer）来分析信号 jitter

* 有些单顆芯片可以产生多种不同频率的时钟信号，非常方便

* 相同频率的电磁波会产生干扰，这种现象叫做 EMI (eletromagnetic interference)

* 经过调制的时钟信号会产生更宽的频谱，从而减小 EMI 干扰
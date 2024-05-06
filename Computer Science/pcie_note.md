# PCIe Note

## cache

* pcie 的功能

    pcie 的全称是 peripheral component internet express

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
# NVIDIA Note

## cache

* 有关 A100

    A100 是 2020 年推出的芯片，前任是 V100，后继是 H100, H200

    A100 的规格：

    SMs 108, TPCs 54

* nvidia-smi

    nvidia-smi: NVIDIA System Management Interface

    NVML: NVIDIA Management Library

    1.1 可查询的状态

        ECC 错误计数
        GPU 利用率
        活动计算进程
        时钟和 PState
        温度和风扇速度
        电源管理
        硬件识别

    1.2 可修改的状态

        ECC 模式
        ECC 复位
        计算模式
        持久模式

    * `Persistence-M`: 	持久模式是否启用。On 表示启用, Off 表示关闭。启用时 GPU 将保持最大性能状态

    * `Disp.A`: 显示器是否连接到 GPU 的输出端口。On 表示连接,Off 表示没有连接

    * `Volatile Uncorr. ECC`: 未 corrected 错误的易失性 ECC 内存错误计数。用于检测内存错误

        这个数字正常情况下都是全 0.

    * `Perf`: 性能状态。P0 是最大性能状态, P8 是最小性能状态

    * `Compute M.`: 计算模式。Default 是默认模式

    * `MIG M.`: MIG(Multi-Instance GPU) 模式, 将一个物理 GPU 分成多个独立、隔离的实例。Disabled 表示未启用

* `nvidia-smi -lms 100`可以第隔 100 ms 输出一次结果

    类似地，`nvidia-smi -l 1`可以每隔 1 s 输出一次结果

    也可以`nvidia-smi -l`，这样的话，每隔 2 秒输出一次结果

## note
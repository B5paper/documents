# NCCL Note

## cache

* `comm->rank`是当前进程被分配到的 mpi rank （猜测）

* `comm->peerInfo[]`是第一轮 bootstrap all gather 后拿到的所有信息

* `comm->nRanks`这些 rank 不一定都在同一个 host 上，因此后面有`if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash)`保证每个 host 只处理当前 host 上的 rank

* 如果处理到了当前进程，那么为当前进程赋值`localRank`，这个 idx 统一根据`nLocalRanks`的计数来分配

    ```c
    if (i == comm->rank)
      localRank = nLocalRanks;
    ```

* `localRanks[nLocalRanks++] = i;`

    无论是否遇到当前进程，只要这个 rank 还在当前 host 上，那么就为`nLocalRanks`加 1，并且把`i`从`comm->nRanks`中拿出来。

* 整体看来，bootstrap all gather 交换完信息后，一共会有`nRanks`个 rank，并且得到一个数组`comm->peerInfo[]`，但是这些 rank 不一定在同一个 host 上，`peerInfo[]`也可能是乱序的。

    比如`nRanks`为`5`，`peerInfo[]`可能是下面这样的：

    ```
    peerInfo[0]
    host_hash_1

    peerInfo[1]
    host_hash_0

    peerInfo[2]
    host_hash_0

    peerInfo[3]
    host_hash_1

    peerInfo[4]
    host_hash_0
    ```

    目前还已知`comm->rank`表示当前进程的 rank，现在我希望得到这样的结果：

    ```
    host 0:
    nLocalRanks[3] = {1, 2, 4};

    host 1:
    nLocalRanks[2] = {0, 3};
    ```

    此时便需要写出 nccl 的代码：

    ```c
    NCCLCHECK(ncclCalloc(&localRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank)
          localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
    ```

    注意，`localRank`并不是从 0 ~ 4 中取值，而是只计算当前 host 上的 rank 数，取值范围为`host 0: 0 ~ 2`，`host 1: 0 ~ 1`。

* nccl 禁用 shm 后，会调用`AllReduce_Sum_f32_RING_SIMPLE`

* 如果数据横跨两个 cuda device，那么要么开启 p2p，要么使用 host mem 作中转

* nccl 中的 va 是 cuda malloc 时得到的

    使用 gdb 运行程序，则 cuda malloc 返回的指针总是固定的：

    ```
    dev 0, buf A: 0x7ffe60a00000, buf B: 0x7ffe60a00200
    dev 1, buf A: 0x7ffe60c00000, buf B: 0x7ffe60c00200
    ```

    正常运行程序，得到的指针则是随机的：

    ```
    dev 0, buf A: 0x7f7674a00000, buf B: 0x7f7674a00200
    dev 1, buf A: 0x7f7674c00000, buf B: 0x7f7674c00200
    ```

    ```
    dev 0, buf A: 0x7f1088a00000, buf B: 0x7f1088a00200
    dev 1, buf A: 0x7f1088c00000, buf B: 0x7f1088c00200
    ```

* nccl 很可能起了 46183 个 device 线程

* nccl 调试时的 temp 中间结果

    * c 为什么会从 0 循环到 1？

        因为`sendMask = 3`，只有低 2 位为 1.

        看不出来 sendMask，recvMask 有什么特别的二进制含义，可能只是为了省内存。

    * `ncclNvmlDevicePairs[0][1].p2pStatusRead`与`p2pStatusWrite`的值都为`NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED`

        `ncclNvmlDevicePairInfo ncclNvmlDevicePairs`是一个全局数组，专门记录 p2p 能力的。

* 224 机器如果不禁用 IB，那么`wrap_ibv_get_async_event()`会被调用。后面可以判断一下这个函数是否和 gpu direct rdma 有关。

    启动与禁用 IB 对测速影响不大，看起来 IB 应该没有被用到。

    * 224 机器，设置了`NCCL_IB_DISABLE＝1`后，确实没有了 ibv 相关函数的调用

* 实体机上可以跑通 p2p

    两种模式都可以跑通：

    1. P2P/CUMEM/CE

    2. P2P/direct pointer

    跑不通的模式：

    1. SHM/direct/direct

    在调用函数`pfn_nvmlDeviceGetP2PStatus()`时，得到 pcie p2p 不可用的结果。nvml 是 nvidia management library，是 nv 的一个库。显然这个函数是从其他 so 库中加载进来的。

* 在一个虚拟机 node 上透传两个 cuda device，运行 nccl 时，默认情况下走的是 shared memory 传输数据，并没有启用 pcie 的 p2p

* cuda 12.1 环境下，编译 nccl 使用 compute_90 编译时，无法跑通 nccl-test

    使用 compute_70 可以跑通。

* cuda 和 nccl 可以使用不同的 stream 异步执行 commands 队列

    ref: `ref_33`

    猜想：stream 有点像 vulkan 里的 queue。 queue 中每完成一项任务，device 就用中断上报一次 completion。 

* nccl 声称实现的通信原语

    * AllReduce

    * Broadcast

    * Reduce

    * AllGather

    * ReduceScatter

    * send/receive

    * scatter, gather

    * all-to-all

* 在`nvmlwrap.cc:156`这里，当`a = 0, b = 1`时，`ncclNvmlDevicePairs[0][1]`被修改。

    修改它调用的是`nvmlDeviceGetP2PStatus()`函数。

* 目前看起来是在`ncclTopoCheckP2p()`处失败的

* nccl 调试记录

    * `shmTransport`既包含在`struct ncclTransport* ncclTransports[NTRANSPORTS]`数组中，可以用 transport 索引直接调用到，对应的数组的索引是 1

        `p2pTransport`对应数组的索引是 0，`netTransport`对应 2，`collNetTransport`对应 3。

    * `ncclTransports`在五处地方被使用
    
        1. `proxyConnInit()`未被调用

        2. `proxyFree()`：未调用

        3. `ncclProxyConnect()`：未调用

        4. `selectTransport()`：调用

        5. `ncclTopoComputePaths()`

        说明全程没有用到 proxy。无法简单看代码看出逻辑，可能只要在同一台机器上就不需要创建 proxy。

        猜想：这个可能是在`groupLaunch()` -> `asyncJobLaunch()`阶段就判断出了不需要创建 proxy connect。

    * nccl 中`prims_ll.h`文件里有挺多 load, store 相关的函数，但是整个 nccl 中关于 atomic 的函数并不多。由此推断 nccl 很有可能不包含 load, store, atomic 的通信功能

* `all_reduce_perf` 2 gpu 的 log

    这个看起来稍微少一点

    ```
    root@3767c65d25c4:/home/hantian# NCCL_DEBUG=INFO all_reduce_perf -b 128M -e 512M -f 2 -g 2
    # nThread 1 nGpus 2 minBytes 134217728 maxBytes 536870912 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
    #
    # Using devices
    #  Rank  0 Group  0 Pid    880 on 3767c65d25c4 device  0 [0xa4] NVIDIA A100-SXM4-80GB
    #  Rank  1 Group  0 Pid    880 on 3767c65d25c4 device  1 [0xa9] NVIDIA A100-SXM4-80GB
    3767c65d25c4:880:880 [0] NCCL INFO Bootstrap : Using eth0:172.17.0.2<0>
    3767c65d25c4:880:880 [1] NCCL INFO cudaDriverVersion 12040
    NCCL version 2.20.5+cuda12.4
    3767c65d25c4:880:887 [0] NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
    3767c65d25c4:880:887 [0] NCCL INFO P2P plugin IBext_v8
    3767c65d25c4:880:887 [0] NCCL INFO NET/IB : No device found.
    3767c65d25c4:880:887 [0] NCCL INFO NET/IB : No device found.
    3767c65d25c4:880:887 [0] NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>
    3767c65d25c4:880:887 [0] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:880:887 [0] NCCL INFO Using network Socket
    3767c65d25c4:880:888 [1] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:880:888 [1] NCCL INFO Using network Socket
    3767c65d25c4:880:887 [0] NCCL INFO comm 0x55ba65abd9d0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId a4000 commId 0xab55a18c6f2a4343 - Init START
    3767c65d25c4:880:888 [1] NCCL INFO comm 0x55ba65ac2bd0 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId a9000 commId 0xab55a18c6f2a4343 - Init START
    3767c65d25c4:880:888 [1] NCCL INFO Setting affinity for GPU 1 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:880:887 [0] NCCL INFO Setting affinity for GPU 0 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:880:888 [1] NCCL INFO comm 0x55ba65ac2bd0 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
    3767c65d25c4:880:887 [0] NCCL INFO comm 0x55ba65abd9d0 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
    3767c65d25c4:880:888 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] -1/-1/-1->1->0 [5] -1/-1/-1->1->0 [6] 0/-1/-1->1->-1 [7] 0/-1/-1->1->-1 [8] 0/-1/-1->1->-1 [9] 0/-1/-1->1->-1 [10] 0/-1/-1->1->-1 [11] 0/-1/-1->1->-1 [12] -1/-1/-1->1->0 [13] -1/-1/-1->1->0 [14] -1/-1/-1->1->0 [15] -1/-1/-1->1->0 [16] -1/-1/-1->1->0 [17] -1/-1/-1->1->0 [18] 0/-1/-1->1->-1 [19] 0/-1/-1->1->-1 [20] 0/-1/-1->1->-1 [21] 0/-1/-1->1->-1 [22] 0/-1/-1->1->-1 [23] 0/-1/-1->1->-1
    3767c65d25c4:880:888 [1] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:880:887 [0] NCCL INFO Channel 00/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 01/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 02/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 03/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 04/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 05/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 06/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 07/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 08/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 09/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 10/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 11/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 12/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 13/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 14/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 15/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 16/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 17/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 18/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 19/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 20/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 21/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 22/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Channel 23/24 :    0   1
    3767c65d25c4:880:887 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] 1/-1/-1->0->-1 [5] 1/-1/-1->0->-1 [6] -1/-1/-1->0->1 [7] -1/-1/-1->0->1 [8] -1/-1/-1->0->1 [9] -1/-1/-1->0->1 [10] -1/-1/-1->0->1 [11] -1/-1/-1->0->1 [12] 1/-1/-1->0->-1 [13] 1/-1/-1->0->-1 [14] 1/-1/-1->0->-1 [15] 1/-1/-1->0->-1 [16] 1/-1/-1->0->-1 [17] 1/-1/-1->0->-1 [18] -1/-1/-1->0->1 [19] -1/-1/-1->0->1 [20] -1/-1/-1->0->1 [21] -1/-1/-1->0->1 [22] -1/-1/-1->0->1 [23] -1/-1/-1->0->1
    3767c65d25c4:880:887 [0] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:880:888 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 02/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 03/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 04/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 05/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 04/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 06/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 05/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 07/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 06/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 08/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 07/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 09/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 08/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 10/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 09/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 11/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 10/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 12/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 11/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 13/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 12/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 14/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 13/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 15/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 14/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 16/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 15/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 17/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 16/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 18/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 17/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 19/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 18/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 20/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 19/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 21/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 20/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 22/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 21/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Channel 23/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 22/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:887 [0] NCCL INFO Channel 23/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:880:888 [1] NCCL INFO Connected all rings
    3767c65d25c4:880:887 [0] NCCL INFO Connected all rings
    3767c65d25c4:880:887 [0] NCCL INFO Connected all trees
    3767c65d25c4:880:888 [1] NCCL INFO Connected all trees
    3767c65d25c4:880:888 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
    3767c65d25c4:880:888 [1] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:880:887 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
    3767c65d25c4:880:887 [0] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:880:887 [0] NCCL INFO NCCL_WORK_FIFO_DEPTH set by environment to 4194304.
    3767c65d25c4:880:887 [0] NCCL INFO comm 0x55ba65abd9d0 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId a4000 commId 0xab55a18c6f2a4343 - Init COMPLETE
    3767c65d25c4:880:888 [1] NCCL INFO comm 0x55ba65ac2bd0 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId a9000 commId 0xab55a18c6f2a4343 - Init COMPLETE
    #
    #                                                              out-of-place                       in-place          
    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
    134217728      33554432     float     sum      -1    857.6  156.51  156.51      0    819.6  163.76  163.76      0
    268435456      67108864     float     sum      -1   1445.2  185.74  185.74      0   1441.9  186.17  186.17      0
    536870912     134217728     float     sum      -1   2789.9  192.43  192.43      0   2793.6  192.18  192.18      0
    3767c65d25c4:880:880 [1] NCCL INFO comm 0x55ba65abd9d0 rank 0 nranks 2 cudaDev 0 busId a4000 - Destroy COMPLETE
    3767c65d25c4:880:880 [1] NCCL INFO comm 0x55ba65ac2bd0 rank 1 nranks 2 cudaDev 1 busId a9000 - Destroy COMPLETE
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 179.465 
    #

    ```

* `all_reduce_perf` 4 gpu 的 log

    ```
    root@3767c65d25c4:/home/hantian# NCCL_DEBUG=INFO all_reduce_perf -b 128M -e 512M -f 2 -g 4
    # nThread 1 nGpus 4 minBytes 134217728 maxBytes 536870912 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
    #
    # Using devices
    #  Rank  0 Group  0 Pid    855 on 3767c65d25c4 device  0 [0xa4] NVIDIA A100-SXM4-80GB
    #  Rank  1 Group  0 Pid    855 on 3767c65d25c4 device  1 [0xa9] NVIDIA A100-SXM4-80GB
    #  Rank  2 Group  0 Pid    855 on 3767c65d25c4 device  2 [0xe1] NVIDIA A100-SXM4-80GB
    #  Rank  3 Group  0 Pid    855 on 3767c65d25c4 device  3 [0xe7] NVIDIA A100-SXM4-80GB
    3767c65d25c4:855:855 [0] NCCL INFO Bootstrap : Using eth0:172.17.0.2<0>
    3767c65d25c4:855:855 [3] NCCL INFO cudaDriverVersion 12040
    NCCL version 2.20.5+cuda12.4
    3767c65d25c4:855:866 [2] NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
    3767c65d25c4:855:866 [2] NCCL INFO P2P plugin IBext_v8
    3767c65d25c4:855:866 [2] NCCL INFO NET/IB : No device found.
    3767c65d25c4:855:866 [2] NCCL INFO NET/IB : No device found.
    3767c65d25c4:855:866 [2] NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>
    3767c65d25c4:855:866 [2] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:855:866 [2] NCCL INFO Using network Socket
    3767c65d25c4:855:864 [0] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:855:864 [0] NCCL INFO Using network Socket
    3767c65d25c4:855:867 [3] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:855:867 [3] NCCL INFO Using network Socket
    3767c65d25c4:855:865 [1] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:855:865 [1] NCCL INFO Using network Socket
    3767c65d25c4:855:865 [1] NCCL INFO comm 0x562b860d2a20 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId a9000 commId 0x40531e6886adf963 - Init START
    3767c65d25c4:855:867 [3] NCCL INFO comm 0x562b860dc3d0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId e7000 commId 0x40531e6886adf963 - Init START
    3767c65d25c4:855:866 [2] NCCL INFO comm 0x562b860d7730 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId e1000 commId 0x40531e6886adf963 - Init START
    3767c65d25c4:855:864 [0] NCCL INFO comm 0x562b860cbf20 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId a4000 commId 0x40531e6886adf963 - Init START
    3767c65d25c4:855:865 [1] NCCL INFO Setting affinity for GPU 1 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:855:865 [1] NCCL INFO NVLS multicast support is not available on dev 1
    3767c65d25c4:855:866 [2] NCCL INFO Setting affinity for GPU 2 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:855:866 [2] NCCL INFO NVLS multicast support is not available on dev 2
    3767c65d25c4:855:867 [3] NCCL INFO Setting affinity for GPU 3 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:855:867 [3] NCCL INFO NVLS multicast support is not available on dev 3
    3767c65d25c4:855:864 [0] NCCL INFO Setting affinity for GPU 0 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:855:864 [0] NCCL INFO NVLS multicast support is not available on dev 0
    3767c65d25c4:855:867 [3] NCCL INFO comm 0x562b860dc3d0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
    3767c65d25c4:855:866 [2] NCCL INFO comm 0x562b860d7730 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
    3767c65d25c4:855:867 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2 [2] -1/-1/-1->3->2 [3] -1/-1/-1->3->2 [4] -1/-1/-1->3->2 [5] -1/-1/-1->3->2 [6] -1/-1/-1->3->2 [7] -1/-1/-1->3->2 [8] -1/-1/-1->3->2 [9] -1/-1/-1->3->2 [10] -1/-1/-1->3->2 [11] -1/-1/-1->3->2 [12] -1/-1/-1->3->2 [13] -1/-1/-1->3->2 [14] -1/-1/-1->3->2 [15] -1/-1/-1->3->2 [16] -1/-1/-1->3->2 [17] -1/-1/-1->3->2 [18] -1/-1/-1->3->2 [19] -1/-1/-1->3->2 [20] -1/-1/-1->3->2 [21] -1/-1/-1->3->2 [22] -1/-1/-1->3->2 [23] -1/-1/-1->3->2
    3767c65d25c4:855:865 [1] NCCL INFO comm 0x562b860d2a20 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
    3767c65d25c4:855:864 [0] NCCL INFO comm 0x562b860cbf20 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
    3767c65d25c4:855:866 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->1 [7] 3/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->1 [11] 3/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->1 [15] 3/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->1 [19] 3/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->1 [23] 3/-1/-1->2->1
    3767c65d25c4:855:865 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->0 [6] 2/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->0 [10] 2/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->0 [14] 2/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->0 [18] 2/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->0 [22] 2/-1/-1->1->0 [23] 2/-1/-1->1->0
    3767c65d25c4:855:865 [1] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:855:867 [3] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:855:864 [0] NCCL INFO Channel 00/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 01/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 02/24 :    0   1   2   3
    3767c65d25c4:855:866 [2] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:855:864 [0] NCCL INFO Channel 03/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 04/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 05/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 06/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 07/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 08/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 09/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 10/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 11/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 12/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 13/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 14/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 15/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 16/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 17/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 18/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 19/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 20/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 21/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 22/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Channel 23/24 :    0   1   2   3
    3767c65d25c4:855:864 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] 1/-1/-1->0->-1 [5] 1/-1/-1->0->-1 [6] 1/-1/-1->0->-1 [7] 1/-1/-1->0->-1 [8] 1/-1/-1->0->-1 [9] 1/-1/-1->0->-1 [10] 1/-1/-1->0->-1 [11] 1/-1/-1->0->-1 [12] 1/-1/-1->0->-1 [13] 1/-1/-1->0->-1 [14] 1/-1/-1->0->-1 [15] 1/-1/-1->0->-1 [16] 1/-1/-1->0->-1 [17] 1/-1/-1->0->-1 [18] 1/-1/-1->0->-1 [19] 1/-1/-1->0->-1 [20] 1/-1/-1->0->-1 [21] 1/-1/-1->0->-1 [22] 1/-1/-1->0->-1 [23] 1/-1/-1->0->-1
    3767c65d25c4:855:864 [0] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:855:865 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 01/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 02/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 01/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 03/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 02/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 04/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 03/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 05/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 04/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 06/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 05/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 07/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 06/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 08/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 07/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 09/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 08/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 10/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 09/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 11/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 10/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 12/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 11/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 13/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 02/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 12/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 03/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 14/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 13/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 04/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 15/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 14/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 05/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 15/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 16/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 16/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 06/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 17/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 17/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 07/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 18/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 18/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 19/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 08/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 19/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 20/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 20/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 09/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 21/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 21/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 10/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 22/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 22/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 11/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 23/0 : 1[1] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 23/0 : 2[2] -> 3[3] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 12/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 13/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 14/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 15/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 04/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 16/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 05/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 17/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 06/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 18/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 07/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 19/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 08/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 20/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 09/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 21/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 10/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 22/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 11/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 23/0 : 3[3] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 12/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 13/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 14/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 15/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 16/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 17/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 18/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 19/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 20/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 21/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 22/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Channel 23/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Connected all rings
    3767c65d25c4:855:865 [1] NCCL INFO Connected all rings
    3767c65d25c4:855:867 [3] NCCL INFO Connected all rings
    3767c65d25c4:855:864 [0] NCCL INFO Connected all rings
    3767c65d25c4:855:867 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 01/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 02/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 03/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 04/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 05/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 06/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 07/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 08/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 09/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 10/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 11/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 12/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 13/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 14/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 15/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 16/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 17/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 18/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 19/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 20/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 21/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 22/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:867 [3] NCCL INFO Channel 23/0 : 3[3] -> 2[2] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 02/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 03/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 04/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 05/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 06/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 07/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 00/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 08/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 01/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 09/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 02/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 10/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 03/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 11/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 04/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 12/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 05/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 13/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 06/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 14/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 07/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 15/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 08/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 16/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 09/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 17/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 10/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 18/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 11/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 19/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 12/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 20/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 13/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 21/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 14/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 22/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 15/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 16/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 17/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 18/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 19/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 20/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:865 [1] NCCL INFO Channel 23/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 21/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 22/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:866 [2] NCCL INFO Channel 23/0 : 2[2] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:855:864 [0] NCCL INFO Connected all trees
    3767c65d25c4:855:864 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
    3767c65d25c4:855:864 [0] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:855:865 [1] NCCL INFO Connected all trees
    3767c65d25c4:855:865 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
    3767c65d25c4:855:865 [1] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:855:867 [3] NCCL INFO Connected all trees
    3767c65d25c4:855:866 [2] NCCL INFO Connected all trees
    3767c65d25c4:855:867 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
    3767c65d25c4:855:867 [3] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:855:866 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
    3767c65d25c4:855:866 [2] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:855:864 [0] NCCL INFO NCCL_WORK_FIFO_DEPTH set by environment to 4194304.
    3767c65d25c4:855:864 [0] NCCL INFO comm 0x562b860cbf20 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId a4000 commId 0x40531e6886adf963 - Init COMPLETE
    3767c65d25c4:855:867 [3] NCCL INFO comm 0x562b860dc3d0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId e7000 commId 0x40531e6886adf963 - Init COMPLETE
    3767c65d25c4:855:865 [1] NCCL INFO comm 0x562b860d2a20 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId a9000 commId 0x40531e6886adf963 - Init COMPLETE
    3767c65d25c4:855:866 [2] NCCL INFO comm 0x562b860d7730 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId e1000 commId 0x40531e6886adf963 - Init COMPLETE
    #
    #                                                              out-of-place                       in-place          
    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
    134217728      33554432     float     sum      -1   1144.4  117.29  175.93      0   1116.5  120.21  180.32      0
    268435456      67108864     float     sum      -1   2157.0  124.45  186.67      0   2159.3  124.31  186.47      0
    536870912     134217728     float     sum      -1   4157.4  129.14  193.70      0   3704.9  144.91  217.36      0
    3767c65d25c4:855:855 [3] NCCL INFO comm 0x562b860cbf20 rank 0 nranks 4 cudaDev 0 busId a4000 - Destroy COMPLETE
    3767c65d25c4:855:855 [3] NCCL INFO comm 0x562b860d2a20 rank 1 nranks 4 cudaDev 1 busId a9000 - Destroy COMPLETE
    3767c65d25c4:855:855 [3] NCCL INFO comm 0x562b860d7730 rank 2 nranks 4 cudaDev 2 busId e1000 - Destroy COMPLETE
    3767c65d25c4:855:855 [3] NCCL INFO comm 0x562b860dc3d0 rank 3 nranks 4 cudaDev 3 busId e7000 - Destroy COMPLETE
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 190.076 
    #

    ```

* `all_reduce_perf`只用到了 coll channel （24 个），`sendrecv_perf`同时用到了 coll channel，p2p channel （32 个）

* `-g 2`和`-g 4`不会影响 channel 的数量，但会影响拓扑（ring, tree）的数量

    `-g 4`会起 4 ring, 4 tree

    `-g 2`会起 2 ring, 2 tree

* sendrecv -g 2 的 log

    ```
    root@3767c65d25c4:/home/hantian# NCCL_DEBUG=INFO sendrecv_perf -b 128M -e 512M -f 2 -g 2
    # nThread 1 nGpus 2 minBytes 134217728 maxBytes 536870912 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
    #
    # Using devices
    #  Rank  0 Group  0 Pid    706 on 3767c65d25c4 device  0 [0xa4] NVIDIA A100-SXM4-80GB
    #  Rank  1 Group  0 Pid    706 on 3767c65d25c4 device  1 [0xa9] NVIDIA A100-SXM4-80GB
    3767c65d25c4:706:706 [0] NCCL INFO Bootstrap : Using eth0:172.17.0.2<0>
    3767c65d25c4:706:706 [1] NCCL INFO cudaDriverVersion 12040
    NCCL version 2.20.5+cuda12.4
    3767c65d25c4:706:714 [1] NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
    3767c65d25c4:706:714 [1] NCCL INFO P2P plugin IBext_v8
    3767c65d25c4:706:714 [1] NCCL INFO NET/IB : No device found.
    3767c65d25c4:706:714 [1] NCCL INFO NET/IB : No device found.
    3767c65d25c4:706:714 [1] NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>
    3767c65d25c4:706:714 [1] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:706:714 [1] NCCL INFO Using network Socket
    3767c65d25c4:706:713 [0] NCCL INFO Using non-device net plugin version 0
    3767c65d25c4:706:713 [0] NCCL INFO Using network Socket
    3767c65d25c4:706:714 [1] NCCL INFO comm 0x556a7ff6ad20 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId a9000 commId 0xb7f271b6f0f05f4e - Init START
    3767c65d25c4:706:713 [0] NCCL INFO comm 0x556a7ff65b20 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId a4000 commId 0xb7f271b6f0f05f4e - Init START
    3767c65d25c4:706:714 [1] NCCL INFO Setting affinity for GPU 1 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:706:713 [0] NCCL INFO Setting affinity for GPU 0 to 7fffffff,ffffffff,00000000,00000000,ffffffff,ffffffff,00000000,00000000
    3767c65d25c4:706:714 [1] NCCL INFO comm 0x556a7ff6ad20 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
    3767c65d25c4:706:713 [0] NCCL INFO comm 0x556a7ff65b20 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
    3767c65d25c4:706:714 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] -1/-1/-1->1->0 [5] -1/-1/-1->1->0 [6] 0/-1/-1->1->-1 [7] 0/-1/-1->1->-1 [8] 0/-1/-1->1->-1 [9] 0/-1/-1->1->-1 [10] 0/-1/-1->1->-1 [11] 0/-1/-1->1->-1 [12] -1/-1/-1->1->0 [13] -1/-1/-1->1->0 [14] -1/-1/-1->1->0 [15] -1/-1/-1->1->0 [16] -1/-1/-1->1->0 [17] -1/-1/-1->1->0 [18] 0/-1/-1->1->-1 [19] 0/-1/-1->1->-1 [20] 0/-1/-1->1->-1 [21] 0/-1/-1->1->-1 [22] 0/-1/-1->1->-1 [23] 0/-1/-1->1->-1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 00/24 :    0   1
    3767c65d25c4:706:714 [1] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:706:713 [0] NCCL INFO Channel 01/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 02/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 03/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 04/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 05/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 06/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 07/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 08/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 09/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 10/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 11/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 12/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 13/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 14/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 15/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 16/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 17/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 18/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 19/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 20/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 21/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 22/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Channel 23/24 :    0   1
    3767c65d25c4:706:713 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] 1/-1/-1->0->-1 [5] 1/-1/-1->0->-1 [6] -1/-1/-1->0->1 [7] -1/-1/-1->0->1 [8] -1/-1/-1->0->1 [9] -1/-1/-1->0->1 [10] -1/-1/-1->0->1 [11] -1/-1/-1->0->1 [12] 1/-1/-1->0->-1 [13] 1/-1/-1->0->-1 [14] 1/-1/-1->0->-1 [15] 1/-1/-1->0->-1 [16] 1/-1/-1->0->-1 [17] 1/-1/-1->0->-1 [18] -1/-1/-1->0->1 [19] -1/-1/-1->0->1 [20] -1/-1/-1->0->1 [21] -1/-1/-1->0->1 [22] -1/-1/-1->0->1 [23] -1/-1/-1->0->1
    3767c65d25c4:706:713 [0] NCCL INFO P2P Chunksize set to 524288
    3767c65d25c4:706:714 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 02/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 03/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 04/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 04/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 05/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 05/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 06/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 06/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 07/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 07/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 08/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 08/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 09/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 09/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 10/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 10/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 11/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 11/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 12/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 12/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 13/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 13/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 14/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 14/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 15/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 15/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 16/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 16/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 17/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 17/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 18/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 18/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 19/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 19/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 20/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 20/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 21/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 21/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 22/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 22/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Channel 23/0 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:713 [0] NCCL INFO Channel 23/0 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:714 [1] NCCL INFO Connected all rings
    3767c65d25c4:706:714 [1] NCCL INFO Connected all trees
    3767c65d25c4:706:713 [0] NCCL INFO Connected all rings
    3767c65d25c4:706:714 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
    3767c65d25c4:706:714 [1] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:706:713 [0] NCCL INFO Connected all trees
    3767c65d25c4:706:713 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
    3767c65d25c4:706:713 [0] NCCL INFO 24 coll channels, 0 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
    3767c65d25c4:706:713 [0] NCCL INFO NCCL_WORK_FIFO_DEPTH set by environment to 4194304.
    3767c65d25c4:706:713 [0] NCCL INFO comm 0x556a7ff65b20 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId a4000 commId 0xb7f271b6f0f05f4e - Init COMPLETE
    3767c65d25c4:706:714 [1] NCCL INFO comm 0x556a7ff6ad20 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId a9000 commId 0xb7f271b6f0f05f4e - Init COMPLETE
    #
    #                                                              out-of-place                       in-place          
    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
    3767c65d25c4:706:722 [0] NCCL INFO Channel 00/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 01/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 00/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 02/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 01/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 03/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 02/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 04/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 03/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 05/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 04/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 06/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 05/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 07/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 06/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 08/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 07/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 09/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 08/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 10/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 09/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 11/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 10/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 12/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 11/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 13/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 12/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 14/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 13/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 15/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 14/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 16/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 15/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 17/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 16/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 18/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 17/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 19/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 18/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 20/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 19/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 21/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 20/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 22/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 21/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 23/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 22/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 24/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 23/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 25/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 24/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 26/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 25/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 27/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 26/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 28/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 27/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 29/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 28/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 30/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 29/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:722 [0] NCCL INFO Channel 31/1 : 0[0] -> 1[1] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 30/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    3767c65d25c4:706:721 [1] NCCL INFO Channel 31/1 : 1[1] -> 0[0] via P2P/direct pointer/read
    134217728      33554432     float     sum      -1    876.8  153.08  153.08      0    829.1  161.89  161.89    N/A
    268435456      67108864     float     sum      -1   1626.2  165.07  165.07      0   1594.1  168.39  168.39    N/A
    536870912     134217728     float     sum      -1   2541.2  211.27  211.27      0   2526.5  212.50  212.50    N/A
    3767c65d25c4:706:706 [1] NCCL INFO comm 0x556a7ff65b20 rank 0 nranks 2 cudaDev 0 busId a4000 - Destroy COMPLETE
    3767c65d25c4:706:706 [1] NCCL INFO comm 0x556a7ff6ad20 rank 1 nranks 2 cudaDev 1 busId a9000 - Destroy COMPLETE
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 178.7 
    #

    ```

* nccl 无法使用`CUDA_VISIBLE_DEVICES`指定具体使用哪块 gpu

    `NVIDIA_VISIBLE_DEVICES`也不行。

* nccl 有隐藏的环境变量`NCCL_LL_BUFFSIZE`, `NCCL_LL128_BUFFSIZE`，把这两个设置为`16384`，nccl 会找尽量满足这个 size 的 buffer size。将`NCCL_LL128_BUFFSIZE`设置为 16 KB 后，nccl 实际申请的内存是 20 KB，即使这样也是满足要求的。

    添加这两个环境变量后，可以在不跳过三种 protocol 注册 mr 的情况下，跑通所有的 test case。

## note

### compile and install

#### cache

* A100, cuda 12.4 对应的 nccl sm 为 80。编译 sm 90 无法跑通。

* cuda 12.1 环境下，编译 nccl 使用 compute_90 编译时，无法跑通 nccl-test

    使用 compute_70 可以跑通。

#### note

### app

#### cache

* 使用 unique id + init rank 的方式进行初始化

    ```cpp
    #include <nccl.h>
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        ncclUniqueId uni_id;
        ncclResult_t ret;
        ret = ncclGetUniqueId(&uni_id);
        if (ret != ncclSuccess)
        {
            printf("fail to get unique id\n");
            return -1;
        }
        printf("get unique id: %lu\n", uni_id);

        ncclComm_t comms[2];
        int dev_indices[2] = {0, 1};

        ncclGroupStart();
        for (int i = 0; i < 2; i++) {
            cudaSetDevice(dev_indices[i]);
            ncclCommInitRank(&comms[i], 2, uni_id, i);
        }
        ncclGroupEnd();

        for (int i = 0; i < 2; ++i)
        {
            cudaSetDevice(dev_indices[i]);
            cudaDeviceSynchronize();
        }

        for (int i = 0; i < 2; ++i)
        {
            printf("comms[%d]: %p\n", i, comms[i]);
        }

        return 0;
    }
    ```

    output:

    ```
    get unique id: 512
    comms[0]: 0x55fd1f29f6c0
    comms[1]: 0x55fd1f33cf20
    ```

* 使用`ncclCommInitRankConfig()`进行带参数的初始化

    ```cpp
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    config.minCTAs = 4;
    config.maxCTAs = 16;
    config.cgaClusterSize = 2;
    config.netName = "Socket";

    // ...

    ncclGroupStart();
    for (int i = 0; i < 2; i++)
    {
        cudaSetDevice(dev_indices[i]);
        ncclCommInitRankConfig(&comms[i], 2, uni_id, i, &config);
    }
    ncclGroupEnd();
    ```

    这里，`NCCL_CONFIG_INITIALIZER`并不是一个枚举，而是一个宏，写了 struct 初始化的一些字段。

* The `ncclCommInitRankScalable()` function enables the creation of a NCCL communicator using many ncclUniqueIds.

    看起来这个函数可以指定多个 unique id，但是 unique id 其实是标识 host　的，使用多个 host 又有什么用？

    试了下，这个功能在目前的版本里没有，好像是新加上去的。

* doc 里说使用这个函数判断一个 rank 是否应该产生一个 unique id

    ```cpp
    bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
      const int rmr = nRanks % nIds;
      const int rpr = nRanks / nIds;
      const int rlim = rmr * (rpr+1);
      if (rank < rlim) {
        return !(rank % (rpr + 1));
      } else {
        return !((rank - rlim) % rpr);
      }
    }
    ```

    For example, if 3 ncclUniqueIds are to be distributed accross 7 NCCL ranks, the first ncclUniqueId will be associated to ranks 0-2, while the others will be associated to ranks 3-4, and 5-6. This function will therefore return true on rank 0, 3, and 5, and false otherwise.

    Note: only the first ncclUniqueId will be used to create the communicator hash id, which is used to identify the communicator in the log file and in the replay tool.

    不清楚这个干嘛用的，创建这么多 unique id 有什么用。

* Using multiple NCCL communicators concurrently

    根据 doc 上的资料，在 cuda 12.3 之前，

    ```cpp
    cudaGraphLaunch(graph1, stream1); // all ranks do this first
    cudaGraphLaunch(graph2, stream2); // and this second
    ```

    是按顺序执行的，底层使用的是 cuda graph。

    从 cuda 12.3 开始，这两个语句是并行执行的，底层使用的是 completion events。

* nccl 提供的控制模式

    * single-threaded control of all GPUs

    * multi-threaded, for example, using one thread per GPU

    * multi-process, for example, MPI

* Each CUDA device is identified within the communication group by a zero-based index or rank. 

    看来每个 group 内的 rank 是从 0 开始分配，并且唯一的，但是 group 与 group 内的 rank 是互相独立的。

* When creating a communicator, a unique rank between 0 and n-1 has to be assigned to each of the n CUDA devices which are part of the communicator.

    猜测：在当前进程创建 communicator 后，当前 host 的所有 device 会被编号为 0 到 n - 1。

* Using the same CUDA device multiple times as different ranks of the same NCCL communicator is not supported and may lead to hangs.

    在不同的 communicator 中，同一个 deivce 是否会有不同的 rank？

* Given a static mapping of ranks to CUDA devices, the ncclCommInitRank(), ncclCommInitRankConfig() and ncclCommInitAll() functions will create communicator objects, each communicator object being associated to a fixed rank and CUDA device.

    cuda device，rank 和 communicator 是一一映射关系。

    每次 init 的时候，cuda dev 都会被映射到相同的 rank 上吗？

* Before calling ncclCommInitRank(), you need to first create a unique object which will be used by all processes and threads to synchronize and understand they are part of the same communicator. This is done by calling the ncclGetUniqueId() function.

    这个 unique id 其实就是 ip + bus id 了。

* The ncclGetUniqueId() function returns an ID which has to be broadcast to all participating threads and processes using any CPU communication system, for example, passing the ID pointer to multiple threads, or broadcasting it to other processes using MPI or another parallel environment using, for example, sockets.

    看他这个介绍，mpi 只是可选项之一，是否如果只调用 thread，那么就不会用到 mpi？

* You can also call the ncclCommInitAll operation to create n communicator objects at once within a single process. As it is limited to a single process, this function does not permit inter-node communication. ncclCommInitAll is equivalent to calling a combination of ncclGetUniqueId and ncclCommInitRank.

    猜想：inter-node 可能是跨 host 通信的意思。

    看起来`ncclCommInitAll()`是在一个进程上拿到所有 device 的 rank 的意思。

* The following sample code is a simplified implementation of ncclCommInitAll.

    ```cpp
    ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
      ncclUniqueId Id;
      ncclGetUniqueId(&Id);
      ncclGroupStart();
      for (int i=0; i<ndev; i++) {
        cudaSetDevice(devlist[i]);
        ncclCommInitRank(comm+i, ndev, Id, i);
      }
      ncclGroupEnd();
    }
    ```

    `ncclCommInitRank()`接收了个`ndev`参数，猜测可能是为每个`comm`，都创建长度为`ndev`的数组，保存 peer devs 的信息。而`Id`和`i`，则分别作为 bus id 和 dev rank 信息。

* 一条可以跑通的 nccl 命令：`mpirun -np 2 --host sw53,sw54 -mca btl_tcp_if_include ens9f0 $(pwd)/all_reduce_perf -b 8 -e 128M -f 2 -g 1`

    output:

    ```
    Authorization required, but no authorization protocol specified
    Authorization required, but no authorization protocol specified
    Authorization required, but no authorization protocol specified
    Authorization required, but no authorization protocol specified
    # nThread 1 nGpus 1 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
    #
    # Using devices
    #  Rank  0 Group  0 Pid  58680 on       sw53 device  0 [0xb1] Tesla V100-PCIE-32GB
    #  Rank  1 Group  0 Pid 928295 on       sw54 device  0 [0xb1] Tesla V100-PCIE-32GB
    #
    #                                                              out-of-place                       in-place          
    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
            8             2     float     sum      -1    17.39    0.00    0.00      0    16.26    0.00    0.00      0
            16             4     float     sum      -1    17.45    0.00    0.00      0    26.12    0.00    0.00      0
            32             8     float     sum      -1    17.09    0.00    0.00      0    16.83    0.00    0.00      0
            64            16     float     sum      -1    17.14    0.00    0.00      0    16.58    0.00    0.00      0
            128            32     float     sum      -1    17.08    0.01    0.01      0    16.93    0.01    0.01      0
            256            64     float     sum      -1    18.64    0.01    0.01      0    17.65    0.01    0.01      0
            512           128     float     sum      -1    19.45    0.03    0.03      0    18.65    0.03    0.03      0
            1024           256     float     sum      -1    20.68    0.05    0.05      0    21.43    0.05    0.05      0
            2048           512     float     sum      -1    24.26    0.08    0.08      0    23.42    0.09    0.09      0
            4096          1024     float     sum      -1    29.49    0.14    0.14      0    29.28    0.14    0.14      0
            8192          2048     float     sum      -1    39.72    0.21    0.21      0    39.90    0.21    0.21      0
        16384          4096     float     sum      -1    53.53    0.31    0.31      0    52.61    0.31    0.31      0
        32768          8192     float     sum      -1    90.02    0.36    0.36      0    90.05    0.36    0.36      0
        65536         16384     float     sum      -1    159.5    0.41    0.41      0    159.9    0.41    0.41      0
        131072         32768     float     sum      -1    178.5    0.73    0.73      0    177.4    0.74    0.74      0
        262144         65536     float     sum      -1    310.2    0.85    0.85      0    309.0    0.85    0.85      0
        524288        131072     float     sum      -1    583.2    0.90    0.90      0    582.7    0.90    0.90      0
        1048576        262144     float     sum      -1   1130.9    0.93    0.93      0   1130.6    0.93    0.93      0
        2097152        524288     float     sum      -1   2224.5    0.94    0.94      0   2223.9    0.94    0.94      0
        4194304       1048576     float     sum      -1   4402.7    0.95    0.95      0   4400.1    0.95    0.95      0
        8388608       2097152     float     sum      -1   8776.1    0.96    0.96      0   8776.5    0.96    0.96      0
        16777216       4194304     float     sum      -1    17371    0.97    0.97      0    17367    0.97    0.97      0
        33554432       8388608     float     sum      -1    34324    0.98    0.98      0    34302    0.98    0.98      0
        67108864      16777216     float     sum      -1    68192    0.98    0.98      0    68194    0.98    0.98      0
    134217728      33554432     float     sum      -1   136001    0.99    0.99      0   135982    0.99    0.99      0
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 0.471788 
    #
    ```

    参数说明：

    * `-np 2`：表示一共起 2 个进程。默认情况下会给每个 node 分 1 个进程，

    * `--host`：指定 node 的 hostname，使用逗号分隔

    * `-mca`：不知道有啥用，后面跟两个参数，好像是用来配置 mpi 走的网卡

    * `$(pwd)/all_reduce_perf`：因为指定绝对路径比较方便，所以这里使用了`$(pwd)`

    * `-b 8`, `-e 128M`, `-f 2`: 第 1 次先发 8 个字节，后面发送的数据量不断翻倍，直到 128M 为止，数据量每次翻 2 倍。

    * `-g 1`：在当前机器上使用 1 块 gpu

* 一个可以运行的 nccl test 命令

    ```bash
    nccl_tests_path=/home/hlc/Documents/Projects/nccl-tests
    mpirun -np 2 --host node1,node2 -mca btl_tcp_if_include enp0s3 -x NCCL_DEBUG=TRACE -x NCCL_BUFFSIZE=32768 ${nccl_tests_path}/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
    ```

    说明：

    * `CCL_DEBUG`必须设置成`TRACE`才能看到 nccl 运行时的详细的信息。设置成`INFO`不行。

    * `NCCL_BUFFSIZE`可以设置 cpu memory buffer 的大小

        这个默认为`4194304`,即 4MB.

        `32768`对应的是 32KB.

    * `-g 1`表示 nccl 在本 node 上只使用一块 gpu.

* 一个最简 nccl 程序

    `main.cu`:

    ```c
    #include <nccl.h>
    #include <stdio.h>

    int main()
    {
        ncclComm_t comm;
        int dev_id = 0;
        ncclResult_t ret = ncclCommInitAll(&comm, 1, &dev_id);
        if (ret != ncclSuccess)
        {
            printf("fail to init all comm\n");
            return -1;
        }
        printf("successfully init all comm\n");

        ret = ncclCommDestroy(comm);
        if (ret != ncclSuccess)
        {
            printf("fail to destroy comm\n");
            return -1;
        }
        printf("successfully destroy comm\n");
        return 0;
    }
    ```

    编译：

    `nvcc main.cu -lnccl -o main`

    运行：

    `./main`

    输出：

    ```
    successfully init all comm
    successfully destroy comm
    ```

    这个程序可以用来测试 nccl 编译和运行环境。

* 创建 communicator 时，nccl 会创建 rank，这个 rank 代表一个 device，不代表一个进程

    > Using the same CUDA device multiple times as different ranks of the same NCCL communicator is not supported and may lead to hangs.

* nccl all reduce example

    `main.c`:

    ```c
    #include <stdlib.h>
    #include <stdio.h>
    #include <nccl.h>
    #include <cuda_runtime.h>

    // resources on a cuda device
    typedef struct
    {
        float *cubuf_A;
        float *cubuf_B;
        cudaStream_t cu_stream;
    } CudevRes;

    void print_vec_cuf32(void *cubuf, int num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        cudaError_t cu_ret;
        cu_ret = cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        if (cu_ret != cudaSuccess)
        {
            printf("fail to cuda memcpy\n");
            exit(-1);
        }
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');
        free(buf);
    }

    void print_vec_f32(float *buf, int num_elm)
    {
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');
    }

    int main()
    {
        int cu_dev_cnt;
        cudaGetDeviceCount(&cu_dev_cnt);
        printf("there are totally %d cuda devices\n", cu_dev_cnt);

        // int cur_cu_dev_id;
        // cudaGetDevice(&cur_cu_dev_id);
        // printf("current cuda device: %d\n", cur_cu_dev_id);

        int num_cuda_devs = 2;

        ncclComm_t *nccl_comms = malloc(num_cuda_devs * sizeof(ncclComm_t));
        int *cuda_dev_ids = malloc(num_cuda_devs * sizeof(int));
        for (int i = 0; i < num_cuda_devs; ++i)
            cuda_dev_ids[i] = i;
        cudaError_t cu_ret = ncclCommInitAll(nccl_comms, num_cuda_devs, cuda_dev_ids);
        if (cu_ret != cudaSuccess)
        {
            printf("fail to comm init all\n");
            return -1;
        }
        printf("successfully comm init all\n");

        int num_elms = 8;

        float *buf_A_dev_0 = malloc(num_elms * sizeof(float));
        float *buf_A_dev_1 = malloc(num_elms * sizeof(float));
        float *buf_B = malloc(num_elms * sizeof(float));
        for (int i = 0; i < num_elms; ++i)
        {
            buf_A_dev_0[i] = rand() % 5;
            buf_A_dev_1[i] = rand() % 5;
            buf_B[i] = rand() % 5;
        }
        for (int i = 0; i < num_elms; ++i)
        {
            buf_B[i] = buf_A_dev_0[i] + buf_A_dev_1[i];
        }

        printf("buf_A_dev_0:\n");
        print_vec_f32(buf_A_dev_0, num_elms);

        printf("buf_A_dev_1:\n");
        print_vec_f32(buf_A_dev_1, num_elms);

        printf("buf_B:\n");
        print_vec_f32(buf_B, num_elms);

        putchar('\n');

        CudevRes *cudev_reses = malloc(num_cuda_devs * sizeof(CudevRes));
        for (int i = 0; i < num_cuda_devs; ++i)
        {
            CudevRes *cudev_res = &cudev_reses[i];

            cu_ret = cudaSetDevice(i);
            if (cu_ret != cudaSuccess)
                printf("fail to set cuda device %d\n", i);

            cu_ret = cudaMalloc((void**) &cudev_res->cubuf_A, num_elms * sizeof(float));
            if (cu_ret != cudaSuccess)
                printf("fail to malloc buf A on cuda dev %d\n", i);

            cu_ret = cudaMalloc((void**) &cudev_res->cubuf_B, num_elms * sizeof(float));
            if (cu_ret != cudaSuccess)
                printf("fail to malloc buf B on cuda dev %d\n", i);

            cu_ret = cudaStreamCreate(&cudev_res->cu_stream);
            if (cu_ret != cudaSuccess)
                printf("fail to create cuda stream on dev %d\n", i);

            printf("allocate resources from cuda device %d\n", i);

            if (i == 0)
                cu_ret = cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_0, num_elms * sizeof(float), cudaMemcpyHostToDevice);
            else if (i == 1)
                cu_ret = cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_1, num_elms * sizeof(float), cudaMemcpyHostToDevice);
            else
            {
                printf("error\n");
                return -1;
            }
            if (cu_ret != cudaSuccess)
            {
                printf("fail to cudaMemcpy buf A\n");
                return -1;
            }

            cu_ret = cudaMemset(cudev_res->cubuf_B, 0, num_elms * sizeof(float));
            if (cu_ret != cudaSuccess)
            {
                printf("fail to cudaMemset buf B\n");
                return -1;
            }

            printf("assign cuda mem data for dev %d\n", i);
        }

        for (int i = 0; i < num_cuda_devs; ++i)
        {
            cudaSetDevice(i);
            printf("cu dev %d:\n", i);
            printf("\tcubuf_A: ");
            print_vec_cuf32(cudev_reses[i].cubuf_A, num_elms);
            printf("\tcubuf_B: ");
            print_vec_cuf32(cudev_reses[i].cubuf_B, num_elms);
        }

        ncclGroupStart();
        for (int i = 0; i < num_cuda_devs; ++i)
        {
            CudevRes cudev_res = cudev_reses[i];
            cudaSetDevice(i);
            cu_ret = ncclAllReduce(cudev_res.cubuf_A, cudev_res.cubuf_B, num_elms, ncclFloat, ncclSum, nccl_comms[i], cudev_res.cu_stream);
            if (cu_ret != cudaSuccess)
            {
                printf("fail to all recude\n");
                return -1;
            }
        }
        cu_ret = ncclGroupEnd();
        if (cu_ret != cudaSuccess)
        {
            printf("fail to group end\n");
            return -1;
        }
        printf("nccl all reduce group ended\n");

        for (int i = 0; i < num_cuda_devs; ++i)
        {
            CudevRes cudev_res = cudev_reses[i];
            cudaSetDevice(i);
            cudaStreamSynchronize(cudev_res.cu_stream);
        }
        printf("cuda stream synchronized\n");

        for (int i = 0; i < num_cuda_devs; ++i)
        {
            cudaSetDevice(i);
            printf("cu dev %d:\n", i);
            printf("\tcubuf_A: ");
            print_vec_cuf32(cudev_reses[i].cubuf_A, num_elms);
            printf("\tcubuf_B: ");
            print_vec_cuf32(cudev_reses[i].cubuf_B, num_elms);
        }

        for (int i = 0; i < num_cuda_devs; ++i)
        {
            cudaSetDevice(i);
            CudevRes cudev_res = cudev_reses[i];
            cudaFree(cudev_res.cubuf_A);
            cudaFree(cudev_res.cubuf_B);
            cudaStreamDestroy(cudev_res.cu_stream);
        }
        printf("cuda dev resource free\n");

        for (int i = 0; i < num_cuda_devs; ++i)
        {
            ncclCommDestroy(nccl_comms[i]);
        }

        free(nccl_comms);
        free(cuda_dev_ids);
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.c
        nvcc -g -I/home/huliucheng/Documents/Projects/nccl/build/include main.c -L/home/huliucheng/Documents/Projects/nccl/build/lib -lnccl -o main

    clean:
        rm -f main
    ```

    `run.sh`:

    ```sh
    #!/bin/bash

    export LD_LIBRARY_PATH=/home/huliucheng/Documents/Projects/nccl/build/lib:${LD_LIBRARY_PATH}

    ./main
    ```

    compile: `make`

    run: `./run.sh`

    output:

    ```
    there are totally 2 cuda devices
    successfully comm init all
    buf_A_dev_0:
    3.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 
    buf_A_dev_1:
    1.0, 3.0, 2.0, 2.0, 4.0, 0.0, 1.0, 2.0, 
    buf_B:
    4.0, 3.0, 3.0, 3.0, 4.0, 1.0, 3.0, 5.0, 

    allocate resources from cuda device 0
    assign cuda mem data for dev 0
    allocate resources from cuda device 1
    assign cuda mem data for dev 1
    cu dev 0:
        cubuf_A: 3.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 
        cubuf_B: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    cu dev 1:
        cubuf_A: 1.0, 3.0, 2.0, 2.0, 4.0, 0.0, 1.0, 2.0, 
        cubuf_B: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    nccl all reduce group ended
    cuda stream synchronized
    cu dev 0:
        cubuf_A: 3.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 
        cubuf_B: 4.0, 3.0, 3.0, 3.0, 4.0, 1.0, 3.0, 5.0, 
    cu dev 1:
        cubuf_A: 1.0, 3.0, 2.0, 2.0, 4.0, 0.0, 1.0, 2.0, 
        cubuf_B: 4.0, 3.0, 3.0, 3.0, 4.0, 1.0, 3.0, 5.0, 
    cuda dev resource free
    ```

    可以看到，all reduce sum 把不同 device 上的 src 处的数据相加，然后把结果同步到两个 device 的 dst 显存上。

#### note

### topo

#### cache

* `TopoLinkList`、`TopoNodeList`与`TopoNodeSet`有什么不同？

* `ncclTopoSetPaths()`

    每个 node 都有多种 path，这里的 path 是个`ncclTopoLinkList`，其内容如下：

    ```cpp
    struct ncclTopoLinkList {
      struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
      int count;
      float bw;
      int type;
    };
    ```

    可以看到 path 中包含了一个 link list，还有个 type。这个 type 一共有如下几种：

    ```cpp
    // Local (myself)
    #define PATH_LOC 0

    // Connection traversing NVLink
    #define PATH_NVL 1

    // Connection through NVLink using an intermediate GPU
    #define PATH_NVB 2

    // Connection traversing at most a single PCIe bridge
    #define PATH_PIX 3

    // Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
    #define PATH_PXB 4

    // Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
    #define PATH_PXN 5

    // Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
    #define PATH_PHB 6

    // Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
    #define PATH_SYS 7

    // Connection through the network
    #define PATH_NET 8

    // Disconnected
    #define PATH_DIS 9
    ```

    每个 topo node 下有多种类型的 path，每种类型的 path 又有多条 path 实例，每条 path 其实是一个 topo path list。

    从`baseNode->paths+baseNode->type`可以看出，每个 node 只处理和当前 node 类型相同的 path。并且申请的内存大小是根据 topo system 中的数据确定的。

    因此每个 node 下的 path 可能是这样的：

    ```
    cpu 0 -> cpu 1 -> cpu 2 -> cpu 3
    ```

    这其中并没有 gpu，nic 相关的。

    2025053100: 上述代码表示“如果未初始化从 base node 出发，到 base node 类型的 path，那么将所有 base node 类型的 path 都初始化一下”。这个初始化实际上是为后面 bfs 的种子轮搜索服务的。前面的推理明显是完全错误的，错误原因一是将 base node 泛化为了所有 node，其实应该只能推断出在这里对 base node 类型的 path 做了申请内存和初始化，但是不知道拿来干嘛的；二是没有做实验验证`cpu 0 -> cpu 1 -> cpu 2 -> cpu 3`是否正确，如果验证了马上发现 path 中会有多种 node，那么前面的推理就被全部推翻了。由此我们得到的启示是：如果想要做出推断，那么必须做实验。

* getPath

    ```cpp
    static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id, struct ncclTopoLinkList** path) {
      for (int i=0; i<system->nodes[t].count; i++) {
        if (system->nodes[t].nodes[i].id == id) {
          *path = node->paths[t]+i;
          return ncclSuccess;
        }
      }
      WARN("Could not find node of type %d id %lx", t, id);
      return ncclInternalError;
    }
    ```

    这个函数的传入参数`t`应该是 type 的意思，比如`GPU`, `CPU`, `NIC`。

    返回从`node`出发，指向节点类型为`t`，节点 id 为`id`的 path。

* `ncclTopoConnectNodes()`

    ```cpp
    ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
      // Aggregate links into higher bw for NVLink
      struct ncclTopoLink* link;
      for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
        if (link->remNode == remNode && link->type == type) break;
      }

      // ...
    }
    ```

    `NCCL_TOPO_MAX_LINKS`宏展开是 128，这个遍历看起来是从头开始遍历，要么到达最大容量`NCCL_TOPO_MAX_LINKS`后停止，要么`link->remNode`为空时停止。因此前面使用数组已经分配了`link`的空间，所以在数组范围内`link`一定都不为空。

    往上走一层我们可以看到`ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw);`，说明 local link 是 cpu 1 上的 link，而 remote node 就是 cpu 2，link type 为`LINK_SYS`。

    ```cpp
    // Sort links in BW descending order
    struct ncclTopoLink linkSave;
    memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
    while (link != node->links) {
        if ((link-1)->bw >= linkSave.bw)
            break;
        memcpy(link, link-1, sizeof(struct ncclTopoLink));
        link--;
    }
    memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
    ```

    这个是从尾向前遍历，因为要用到`link - 1`，所以当遍历完第 2 个节点后停下，第 1 个节点单独处理。

    1, 2, 3, 4, 5

    `linkSave` -> 5

    `link - 1` -> 4

    `link` -> 4, `link - 1` -> 3

    1, 2, 3, 4, 4 => 5, 1, 2, 3, 4

    看起来比较像，对于`cpu_node_2`，让当前`cpu_node_1`的 link 向前找到一个合适的位置。

    5, 4, 2, 1, 3 => 5, 4, 3, 2, 1

    4, 5, 2, 1, 3 => 4, 5, 3, 2, 1

    由上面的例子可以看出，首先把要处理的数据`x`放在末尾，然后倒着向前找，直到找到第一个比`x`大的数`y`，最后将`x`放到`y`的后面，退出。再向前的数据则不去处理。

    边界条件：如果搜索整个数组都没有找到比`x`大的值，那么说明`x`是数组中最大的，将其放到最前面。

    * cpu node 的 links 是在什么时候被填充的？

        看起来比较像在`ncclTopoAddNvLinks()`里填充。这个是个递归调用的函数。

        * (2025.04.08,00) cpu node 的 link 并不是物理的 link，而是拓扑图里的 edge，因此并不是只在 add nvlinks 函数里填充。目前看来，应该是在`ncclTopoConnectNodes()`里被填充。

    * 为什么只有 cpu link 的 connect，没有 gpu link 的 connect？

    * link type

        ```cpp
        // We want link types and path types to match as much as possible
        #define LINK_LOC 0
        #define LINK_NVL 1
        // Skipping 2 for PATH_NVB
        #define LINK_PCI 3
        // Skipping 4 for PATH_PXB
        // Skipping 5 for PATH_PXN
        // Skipping 6 for PATH_PHB
        #define LINK_SYS 7
        #define LINK_NET 8
        ```

        实测每个 cpu 有 3 个 link，有两个 type 为`LINK_LOC`，还有一个 type 为`LINK_PCI`。

        看来 topo system 里的 link 指的不是 nvlink，而是 graph 里的 edge。

* nccl 中的 topo node 只有四种：gpu, net, cpu, pci

    目前不清楚 pci 中的`uint64_t device;`成员是干嘛用的，可能是 bus id？

    每个 node 上都可以对接 nvlink。（net 该怎么对接？）

    link 相关的结构体只有三个信息：

    ```cpp
    struct TopoLink
    {
        int type;
        float bw;
        struct TopoNode *peer_node;  // remNode
    };
    ```
    
    这里的`TopoLink`即为拓扑边，其实翻译成 edge 更恰当。`peer_node`, `remNode`就是从当前 node 出发的下一跳 node。

    这里的`int type;`类型指的是拓扑边的类型，不是下个节点的类型。

    每个节点都包含一个固定长度的`TopoLinkList*`数组，数组中的每个元素分别对应不同的类型：

    ```cpp
    enum {
        GPU = 0,
        PCI = 1,
        NVS = 2,
        CPU = 3, // Actually NUMA domains
        NIC = 4,
        NET = 5,
        NCCL_TOPO_NODE_TYPES
    };
    ```

    每个节点朝别的所有类型的节点都有 path，这里的 path 类型即 dst 节点的类型。

    `TopoLinkList`的 struct 为：

    ```cpp
    struct TopoLinkList
    {
        vector<TopoLink*> list;  // NCCL_TOPO_MAX_HOPS 256 * 7
        int count;
        float bw;
        int type;
    };
    ```

    这里的`list`是由一系列 topo link (edge) 组成的 path。

    其中的`int type;`指的是 path type。

    每个类型存储一个 path 指针，这个指针，其实是一个 path 数组，由于数组的长度需要在运行时动态确定，所以后面会有 malloc 填充这个指针。

* `ncclTopoFuseXml()`

    ```c
    ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
      struct ncclXmlNode* topNodeDst;
      NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

      if (topNodeDst == NULL) {
        xmlAddTree(dst, NULL, src->nodes);
        return ncclSuccess;
      }

      struct ncclXmlNode* topNodeSrc;
      NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

      return ncclSuccess;
    }
    ```

    如果`dst`是个空的 xml，那么把`src`的 xml 直接添加到`dst`里。

    如果`dst`非空，那么找到`src`中的`system` tag，

* nccl xml

    ```c
    if (comm->MNNVL) {
        // Ensure that we have enough room when fusing topos from multiple nodes.
        free(xml);
        NCCLCHECK(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES));
    } else {
        // In the intra-node case there's no need to enlarge the topo xml.
        xml->maxIndex = 0;
        free(localRanks);
    }
    ```

    不清楚这个`comm->MNNVL`是干嘛用的，疑似是发现了多个 host。如果有多个 host，那么就认为在 current node 上申请的 xml 的内存不够用（为什么不够用？前面在申请时，是按 max nodes 申请的内存吗？），全部释放掉再申请个新的。

    如果发现所有的 comm rank 都是在同一个 node 上的，那么认为为 xml 预留的空间够用，`xml->maxIndex = 0;`相当于移动了栈顶指针，效果上等价于释放数据。`free(localRanks);`不清楚干嘛用的，可能是如果发现 comm 都在同一个 host 上，那么 localRanks 就是无意义的。

* 修改环境变量`NCCL_P2P_LEVEL`, `NCCL_P2P_DIRECT_DISABLE`, `NCCL_P2P_DISABLE`都无法启动或禁止 p2p

* 发现本机资源的几个关键函数：`ncclTopoGetSystem()` -> `ncclTopoComputePaths()` -> `ncclTopoTrimSystem()`

    目前看来是在`ncclTopoComputePaths()`中判断了 pcie p2p 不可用。

    这里的不可用有可能是逻辑判断有问题，也有可能是上一个函数`ncclTopoGetSystem()`在获取资源时，获取的原始数据有误。

* nccl 中 xml 的内存申请

    nccl 中与 xml 相关的 struct 如下：

    ```cpp
    struct ncclXmlNode {
        char name[MAX_STR_LEN+1];
        struct {
            char key[MAX_STR_LEN+1];
            char value[MAX_STR_LEN+1];
        } attrs[MAX_ATTR_COUNT+1];  // Need an extra one to consume extra params
        int nAttrs;
        int type;
        struct ncclXmlNode* parent;
        struct ncclXmlNode* subs[MAX_SUBS];
        int nSubs;
    };

    struct ncclXml {
        int maxIndex, maxNodes;
        struct ncclXmlNode nodes[1];
    };
    ```

    可以看到`ncclXml`里记录了`maxNodes`，即`nodes`数组的最大长度，`maxIndex`即目前`nodes`的实际长度，而`nodes`数组其实是实际子 nodes 的起始地址。

    `struct ncclXmlNode nodes[1];`这样的写法，是想既申请一个节点，作为 xml 的 root node，又可以将`nodes`作为指针，增加偏移指向 child nodes。

    下面是 nccl 里内存申请相关的函数：

    ```cpp
    static size_t xmlMemSize(int maxNodes) {
      return offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode)*maxNodes;
    }

    static ncclResult_t xmlAlloc(struct ncclXml** xml, int maxNodes) {
      char* mem;
      NCCLCHECK(ncclCalloc(&mem, xmlMemSize(maxNodes)));
      *xml = (struct ncclXml*)mem;
      (*xml)->maxNodes = maxNodes;
      return ncclSuccess;
    }
    ```

    可以看到，`xmlMemSize()`在计算 size 时，`offsetof(struct ncclXml, nodes)`计算的是 header 部分，`sizeof(struct ncclXmlNode)*maxNodes`统计的是 child nodes 所占的 size。

    这样的写法，有点像：

    ```c
    struct MyXml
    {
        header member var 1;
        header member var 2;
        header member var 3;
        ...
        root node 0;
        ----------------------
        child node 1;
        child node 2;
        child node 3;
        ...
    };
    ```

    这样即可将一个大小可动态变化的 struct 写成静态的第 1 部分和动态的第 2 部分。

    是否有 feedback？未想到。

#### note

### communication operator

#### cache

* 基于 reduce copy 可以跑通的 all reduce

    见`ref_36`。

    output:

    ```
    the first 8 elms:
    cubuf_1: 1.0, 4.0, 3.0, 1.0, 5.0, 1.0, 4.0, 0.0, specialized as float
    cubuf_2: 3.0, 0.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, specialized as float
    after launch, the first 8 elms:
    cubuf_1: 4.0, 4.0, 5.0, 2.0, 6.0, 3.0, 9.0, 5.0, specialized as float
    cubuf_2: 4.0, 4.0, 5.0, 2.0, 6.0, 3.0, 9.0, 5.0, specialized as float
    in compare_buf_cubuf(), specialized as float
    in compare_buf_cubuf(), specialized as float
    all results are correct
    ```

    问题：

    1. 什么时候`Apply_Reduce::reduce()`中的 half 会被调用到？

    2. `MultimemSrcs`, `MultimemDsts`在何时被调用？

    3. 尝试将 Unroll 加上去，看是如何处理 hunk 的

    4. 16 字节的数据是如何被处理的？

    5. while 循环如何触发？while 中的 break 什么时候能删掉？

    6. `aligned`何时被触发？

    说明：

    1. `cvta_to_global()`目前看来没有对 va 地址作任何改变

* 在 ld_volatile_global() 处恢复运行，disable 断点后，nccl perf data 最后仍能正常输出。说明前面的数据 0 并不是真的 0，只是数据很小没有显示出来。

    同时还说明，前几轮的小数据量 perf 有可能没有调用到 ld_volatile_global()。

    很可能是 8 bytes - 1048576 bytes 这个范围内。

    * 2025/02/11/01: 确实没有用到，因为调用的是宏定义的 load 函数。`ld_volatile_global()`也很有可能不是加载数据用的，而是加载配置用的（step value）。

* nccl 在启用 ll128 协议时，调用`op128.h`中的函数。如果是 ll 协议，那么不会调用。simple 协议目前不清楚。

* `ld_volatile_global()`在两个地方被调用

    1. `Primitives::loadStepValue()`

        用于加载 peer connection 的 info

        * `connStepPtr = conn->head;`, `connStepPtr = conn->tail;`, 看起来`connStepPtr`是 conn 的链表, 这些都在`loadRecvConn()`被调用

        * 有可能 step 是异步处理，所以需要 volatile 加载数据

        * `st_relaxed_sys_global()`由`postPeer()`调用

    2. reduce copy

        用于取 payload 数据。

* `op128.h`中`ld_volatile_global()`会调用到（可以用 pritf 法证明）。其他`ld_volatile_global_xxx()`相关的函数都是使用宏定义的，覆盖了 128 bytes, 64 bytes, 32 bytes, 16 bytes 以及 8 bytes 的处理。

* 一个 unroll 中处理无法使用一个完整 warp 处理的数据的方式：

    unroll 为 1 时，因为每个线程是单独计算自己的任务进度，所以可以处理不完整的 warp 的任务

* `reduceCopyPacks()`是最底层负责干活的函数，每次起一个 warp，warp 里有 32 个线程，每个线程搬运 16 个字节，warp （线程）循环处理 Unroll 组数据，这叫一个 hunk。

    数据可能有多个 src，dst，此时需要做 reduce，把多个 src, dst 合到一处。

* nccl 数据传输的调用流程

    run work batch (dev func) -> run work coll -> run ring/tree -> prims -> send

    * `Primitives<> prims`由`RunWorkColl()` -> `runRing()`创建

#### note

### transport layer

#### cache

* 可以确认 socket 的 buffer 使用的是 cuda host alloc

    与 user 申请的数据对应的地址为`0x7fccafc00000`，查询 nccl 中所有的 buffer 的 addr，有一个`in ncclCudaHostCallocDebug(), host alloc size: 8388608, ptr: 0x7fccafa00000`，将`0x7fccafa00000`与`8388608`相加得到`7FCCB0200000`，由此可以看出`0x7fccafc00000`在这段内存里。

    这个用于 socket 的 buffer size 为 8M，nccl 运行的整个过程中，一共申请了 2 个 8M 的 buffer，猜测可能是因为每个 gpu 算一个 rank，每个 rank 上都要有一个 buffer。

    官网中的环境变量没有 8M 或 8388608 相关的字段，说明这个 buffer 大小在官网看来不受环境变量控制。

* nccl pcie 检查调用栈

    * `ncclAsyncJobMain()`

        * `commAlloc()`

            * `ncclNvmlDeviceGetHandleByPciBusId()`

                * `ncclNvmlEnsureInitialized()`

                    * `pfn_nvmlDeviceGetP2PStatus()`

        * `ncclCommInitRankFunc()`

            * `initTransportsRank()`

                * `ncclTopoComputePaths()`

                    * `ncclTopoCheckP2p()`

                    * `p2pCanConnect()`

                        * `ncclTopoCheckP2p()`

* nccl p2p 在开始数据传输任务后，host 端在 while 循环中进行 poll 检查 sockfd 状态，除此之外无显式的 send recv 操作，说明 p2p 确实是由 device 侧发起的数据搬运操作。

    p2p 模式下，cpu 单核仍会跑满，但是这个是主动循环 poll sockfd 事件导致的，并不是内核在读写 host memory。

    猜测可能是 host 申请一个 sockfd，然后把相关的 buffer 地址以及对应的权限交给 gpu. gpu 走 pcie p2p 和其他 gpu 进行通信，当通信完成后，gpu 会回填这个 sockfd 的 buffer，这个 buffer 里包含了 fd 的状态，代表事件已经完成。host 侧则通过轮询（也不是轮询，其实就是 poll）这个 fd 判断事件是否完成。

    （如果使用 poll，按道理 cpu 应该占用率为零才对，为什么仍会占用 100%？可能是 timeout 设置为 0？不清楚，有时间了看看。）

* nccl 调用了`p2pCanConnect()`和`shmCanConnect()`，但是后续会调用`shmSendConnect()`, `shmRecvConnect()`，并未调用 p2p 相关的函数，说明传输数据使用的是 shared host memory，并不是 pcie。

* 设置环境变量`NCCL_SHM_DISABLE=1`可以禁用 shared host memory，此时会使用 socket 进行通信

* 在建立 ring 连接时（`ncclTransportRingConnect()`），调用`ncclTransportP2pSetup()`建立 p2p 连接

    其中，会调用`selectTransport()` -> `transportComm->setup()`，最终调用到`shmRecvSetup()`。

    显然`setup()`函数指针在前面已经被替换成了`shmRecvSetup()`。

    目前看来，应该是用`struct ncclTransport shmTransport;`完成的替换，这个结构体里包含了 proxy 所需要用到的所有 shm 相关的函数。

* `shmTransport`既包含在`struct ncclTransport* ncclTransports[NTRANSPORTS]`数组中，可以用 transport 索引直接调用到，对应的数组的索引是 1

    `p2pTransport`对应数组的索引是 0，`netTransport`对应 2，`collNetTransport`对应 3。

* 为什么`p2pCanConnect()`会被执行多次？ 经 cnt 统计一共调用了 16 次。

    nccl 会起两个线程，每个线程独立扫描一遍本机资源，对于本机的两个 gpu，都判断一次 p2p can connect，即 0 - 1, 1 - 0， 因此`p2pCanConnect()`会被调用 4 次。

    1. thread 79, g = 0, p = 1

    2. thread 80, g = 0, p = 1

    3. thread 79, g = 1, p = 0

    4. thread 80, g = 1, p = 0

    5. thread 79, g = 0, p = 1

        这里开始第二次调用`ncclTopoComputePaths()`, recompute paths after triming

    6. thread 80, g = 0, p = 1

    7. thread 79, g = 1, p = 0

    8. thread 80, g = 1, p = 0

    9. thread 36, `ncclAsyncJobMain()` -> `ncclCollPreconnectFunc()` -> `ncclTransportRingConnect()` -> `ncclTransportP2pSetup()` -> `selectTransport()` -> `p2pCanConnect()`, c = 0

    10. thread 37, 

    11. thread 37, c = 1

    12. thread 36, c = 1

    13. thread 36, c = 0

        从这里开始，调用`selectTransport<1>()`

    14. thread 37, c = 0

    15. thread 36, c = 1

    16. thread 37, c = 1

* `ncclTransports`在五处地方被使用

    1. `proxyConnInit()`未被调用

    2. `proxyFree()`：未调用

    3. `ncclProxyConnect()`：未调用

    4. `selectTransport()`：调用

    5. `ncclTopoComputePaths()`

    说明全程没有用到 proxy。无法简单看代码看出逻辑，可能只要在同一台机器上就不需要创建 proxy。

    猜想：这个可能是在`groupLaunch()` -> `asyncJobLaunch()`阶段就判断出了不需要创建 proxy connect。

#### note
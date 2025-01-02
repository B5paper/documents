# NCCL Note

## cache

* 一个能跑通的`__shared__` example:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    __shared__ int val;

    __global__ void test_kern()
    {
        val = 123;
        printf("%d\n", val);
    }

    int main()
    {
        test_kern<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    ```

    output:

    ```
    123
    ```

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

* nccl 在启用 ll128 协议时，调用`op128.h`中的函数。如果是 ll 协议，那么不会调用。simple 协议目前不清楚。

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

* `ncclTransports`在五处地方被使用

    1. `proxyConnInit()`未被调用

    2. `proxyFree()`：未调用

    3. `ncclProxyConnect()`：未调用

    4. `selectTransport()`：调用

    5. `ncclTopoComputePaths()`

    说明全程没有用到 proxy。无法简单看代码看出逻辑，可能只要在同一台机器上就不需要创建 proxy。

    猜想：这个可能是在`groupLaunch()` -> `asyncJobLaunch()`阶段就判断出了不需要创建 proxy connect。

* 在一个虚拟机 node 上透传两个 cuda device，运行 nccl 时，默认情况下走的是 shared memory 传输数据，并没有启用 pcie 的 p2p

* cuda 12.1 环境下，编译 nccl 使用 compute_90 编译时，无法跑通 nccl-test

    使用 compute_70 可以跑通。

* cuda 和 nccl 可以使用不同的 stream 异步执行 commands 队列

    ref: `ref_33`

    猜想：stream 有点像 vulkan 里的 queue。 queue 中每完成一项任务，device 就用中断上报一次 completion。 

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

* 创建 communicator 时，nccl 会创建 rank，这个 rank 代表一个 device，不代表一个进程

    > Using the same CUDA device multiple times as different ranks of the same NCCL communicator is not supported and may lead to hangs.

* nccl 声称实现的通信原语

    * AllReduce

    * Broadcast

    * Reduce

    * AllGather

    * ReduceScatter

    * send/receive

    * scatter, gather

    * all-to-all

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

* 在`nvmlwrap.cc:156`这里，当`a = 0, b = 1`时，`ncclNvmlDevicePairs[0][1]`被修改。

    修改它调用的是`nvmlDeviceGetP2PStatus()`函数。

* `shmTransport`既包含在`struct ncclTransport* ncclTransports[NTRANSPORTS]`数组中，可以用 transport 索引直接调用到，对应的数组的索引是 1

    `p2pTransport`对应数组的索引是 0，`netTransport`对应 2，`collNetTransport`对应 3。

* 发现本机资源的几个关键函数：`ncclTopoGetSystem()` -> `ncclTopoComputePaths()` -> `ncclTopoTrimSystem()`

    目前看来是在`ncclTopoComputePaths()`中判断了 pcie p2p 不可用。

    这里的不可用有可能是逻辑判断有问题，也有可能是上一个函数`ncclTopoGetSystem()`在获取资源时，获取的原始数据有误。

* 在建立 ring 连接时（`ncclTransportRingConnect()`），调用`ncclTransportP2pSetup()`建立 p2p 连接

    其中，会调用`selectTransport()` -> `transportComm->setup()`，最终调用到`shmRecvSetup()`。

    显然`setup()`函数指针在前面已经被替换成了`shmRecvSetup()`。

    目前看来，应该是用`struct ncclTransport shmTransport;`完成的替换，这个结构体里包含了 proxy 所需要用到的所有 shm 相关的函数。

* 修改环境变量`NCCL_P2P_LEVEL`, `NCCL_P2P_DIRECT_DISABLE`, `NCCL_P2P_DISABLE`都无法启动或禁止 p2p

* 设置环境变量`NCCL_SHM_DISABLE=1`可以禁用 shared host memory，此时会使用 socket 进行通信

* nccl 调用了`p2pCanConnect()`和`shmCanConnect()`，但是后续会调用`shmSendConnect()`, `shmRecvConnect()`，并未调用 p2p 相关的函数，说明传输数据使用的是 shared host memory，并不是 pcie。

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

* nccl p2p 在开始数据传输任务后，host 端在 while 循环中进行 poll 检查 sockfd 状态，除此之外无显式的 send recv 操作，说明 p2p 确实是由 device 侧发起的数据搬运操作。

    p2p 模式下，cpu 单核仍会跑满，但是这个是主动循环 poll sockfd 事件导致的，并不是内核在读写 host memory。

    猜测可能是 host 申请一个 sockfd，然后把相关的 buffer 地址以及对应的权限交给 gpu. gpu 走 pcie p2p 和其他 gpu 进行通信，当通信完成后，gpu 会回填这个 sockfd 的 buffer，这个 buffer 里包含了 fd 的状态，代表事件已经完成。host 侧则通过轮询（也不是轮询，其实就是 poll）这个 fd 判断事件是否完成。

    （如果使用 poll，按道理 cpu 应该占用率为零才对，为什么仍会占用 100%？可能是 timeout 设置为 0？不清楚，有时间了看看。）

* nccl 有隐藏的环境变量`NCCL_LL_BUFFSIZE`, `NCCL_LL128_BUFFSIZE`，把这两个设置为`16384`，nccl 会找尽量满足这个 size 的 buffer size。将`NCCL_LL128_BUFFSIZE`设置为 16 KB 后，nccl 实际申请的内存是 20 KB，即使这样也是满足要求的。

    添加这两个环境变量后，可以在不跳过三种 protocol 注册 mr 的情况下，跑通所有的 test case。

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

* cuda 12.1 环境下，编译 nccl 使用 compute_90 编译时，无法跑通 nccl-test

    使用 compute_70 可以跑通。

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
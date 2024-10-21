# NCCL Note

## cache

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
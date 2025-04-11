* temp

    * `linux/wait.h`

    * `linux/delay.h`

    * 为什么注释了创建 wait queue head 的两行

    * `DECLARE_WAIT_QUEUE_HEAD()`, `DECLARE_WAITQUEUE()`

    * `add_wait_queue()`

    * `wait_event()`

    * `kthread_should_stop()`, `kthread_stop()`

    * `wake_up()`

    * `remove_wait_queue()`

* temp

    * `schedule_work`

    * `DECLARE_WORK(workqueue,workqueue_fn);`

* `virsh nodedev-dettach pci_0000_b1_00_0` e3

* python packages in use

    ·NumPy 1.9.1
    ·SciPy 0.14.0
    ·scikit-learn 0.15.2
    ·matplotlib 1.4.0
    ·pandas 0.15.2

* pytorch

    * `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`

    * `TorchText`, `TorchVision`, and `TorchAudio`, all of which include datasets. 

    * `transform` and `target_transform` to modify the samples and labels respectively.

    * `torch.cuda.is_available()`

    * `print(f"Using {device} device")`

    * torchvision datasets 的一个用法

        ```python
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        ```

        此时会自动下载数据集到当前文件夹。

        如果需要下载 test 部分的数据集，可以把`train=True`改成`train=False`。

    * dataloader 的一个构造方法：

        ```python
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        ```

        是按顺序的还是 shuffle 的？如果 batch size 无法正好被整除，那么最后一个 part 是 drop，还是合并入下一次，还是循环计数？

    * 从 dataloader 中拿数据的方法

        ```py
        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        ```

        看起来这个数据拿出来直接就是 tensor，，不知道 dataloader 是否支持拿出来是 list 或 dict，因为有时候 ground-truth 不一定是单个矩阵的形式。

    * 自己定义的 block 需要继承自`torch.nn.Module`

        `class NeuralNetwork(nn.Module):`

        在 self init 时需要调用父类的 init 函数：

        ```py
        def __init__(self):
            super().__init__()
        ```

        自定义 block 的两个必要函数：

        `__init__(self)`和`forward(self, x)`

    * `nn.Flatten()`和`nn.Sequential(`都可以被直接调用

        ```py
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        ```

    * 猜想：`nn.Flatten()`的作用是把任意维度的矩阵转换成一维向量

    * `model = MyBlock().to(device)`

        `.to()`并不是 in-place 修改。

    * `loss_fn = nn.CrossEntropyLoss()`

        loss 函数都放在 nn 模块中

    * `torch.optim.SGD(model.parameters(), lr=1e-3)`

        优化器都放在 optim 模块中

        `model.parameters()`可以拿到模型的参数。

        如果不在这里指定 lr，那么后面还能在哪里指定 lr？

    * `model.train()`

        这一行的作用是什么来着？好像是 drop out 层启动随机 drop，其他的还有什么，忘了。

        看来`.train()`是 in place 的行为。

    * 在训练的过程中将训练数据 to device，此时只有显存复制，没有计算，肯定会损失一部分效率。如果显存充足，有没有在做计算时 io 取放数据的方式？

    * `loss = loss_fn(pred, y)`

        loss 函数，第一个参数是 prediction，第二个参数是 gt

    * loss backward

        ```py
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ```

        看来是对一个 iter (或者说 batch) 的 loss 进行 backward，并不是对 epoch 的 loss 进行 backward。

        猜想：loss 本身是没有 grad 的，只有 loss 之前的 parameter 有 grad。

        loss 之前的 output 是否有 grad？

    * `loss, current = loss.item(), (batch + 1) * len(X)`

        loss 是 tensor，可以直接取`.item()`吗？以前只知道 numpy ndarray 可以取 item()。

    * `print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")`

        `:>`是个什么用法？

        下面还有个相似的：

        `print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")`

    * `size = len(dataloader.dataset)`

        dataloader 里竟然能直接拿到 dataset

    * `num_batches = len(dataloader)`

        只知道 dataset 支持 len()，没想到 dataloader 也支持 len。

        dataloader 的 len 是一共有多少个 batch。（还是那个问题，如果遇到了最后一个 batch 凑不够数量，该如何处理？）

    * `model.eval()`

        与 model.train() 对应。

    * `with torch.no_grad():`

        torch 默认是记录 grad 的。（是否有 example 对照实验来说明这一点？）

    * `pred = model(X)`

        `model(x)`就相当于`model.forward(x)`。调研 callable magic method

    * `correct += (pred.argmax(1) == y).type(torch.float).sum().item()`

        `pred`的 shape 为`(64, 10)`，`pred.argmax(1)`在 axis = 1 维度上取最大值对应的索引。

        `y`是 ground truth，shape 为`(64, )`.

        `type()`是否可以换成`.to()`？

    * `test_loss /= num_batches`

        `test_loss`是单个 batch 的 loss？还是单个 entry 的 loss？看起来像单个 batch 的 loss。

    * `training_data`和`test_data`直接就是 dataset，看来可以从`datasets`中直接拿到 dataset，不需要再自己适配了。

        如果不加`transform=`参数，拿到的数据是怎样的？

* cuda 可能有用的关键字

    `cudaMemcpyDefault`

    > pointers returned by cudaHostAlloc() can be used directly from within kernels running on these devices 

    `cudaHostGetDevicePointer()`

    Portable Memory

    Mapped Memory

    `unifiedAddressing`, Device Enumeration

* meeting log

    ```
    cudaEnablePeerAccess()


    __global__ void vec_add(float *A)
    {
        int x = threadIdx.x;
        A[x]++;
    }

    use p2p:

    1. nvlink

        load peer deivce memory ptr (need verify)   20s

        p2p cudaMemcpy(dev0, dev1)  concurrently?   20s

    2. pcie

    no p2p:

    1. shm: shared host memory

    2. **socket**, ib nic

        1. load device memory (current device)    LL / LL128 / SIMPLE (?)

        2. host memory buffer

        3. socket transfer buffer

        4. another device copy data from socket buffer to device buffer

    channel 0: p2p device 0 -> device 1
    channel 1: no p2p devcie 0 -> host -> device 1
    channel 2: 

    int main()
    {
        cudaSetDevice(0);
        float *buf_0 = cudaMalloc(8 * sizeof(float));

        cudaEnablePeerAccess(1, 0);

        cudaSetDevice(1);
        vec_add<<<1, 8>>>(buf_0);

        return 0;
    }
    ```

* nccl record

    * 禁用 shm 后，会调用`AllReduce_Sum_f32_RING_SIMPLE`

* ccl 面试

    * 之前接触过集合通信库吗？gather, scatter, all reduce 这些听说过吗？都分别有什么作用？

    * c++ 了解过吧，模板了解过吧

        * 使用递归对一个数组进行 reduce，不允许使用 if else，不能有 for 循环

            ```cpp
            #include <iostream>
            using namespace std;

            template<typename T, int ElmSize>
            struct SumReduce
            {
                static T sum_reduce(T *src)
                {
                    return SumReduce<T, ElmSize / 2>::sum_reduce(src) + 
                        SumReduce<T, ElmSize - ElmSize / 2>::sum_reduce(src + ElmSize / 2);
                }
            };

            template<typename T>
            struct SumReduce<T, 1>
            {
                static T sum_reduce(T *src)
                {
                    return *src;
                }
            };

            template<typename T>
            struct SumReduce<T, 2>
            {
                static T sum_reduce(T *src)
                {
                    return src[0] + src[1];
                }
            };

            template<typename T>
            struct SumReduce<T, 0>
            {
                static T sum_reduce(T *src)
                {
                    return 0;
                }
            };

            int main()
            {
                using my_type = int;
                my_type arr[] ={1, 2, 3, 4, 5, 6};
                const int num_elm = sizeof(arr) / sizeof(my_type);
                my_type s = SumReduce<my_type, num_elm>::sum_reduce(arr);
                printf("s: %d\n", s);
                return 0;
            }
            ```

        * nccl 中大量使用特化/偏特化，好处是什么？

    * reduce

        * （听说过），现在考虑一个简单的矩阵乘法的加速，目前已经实现了 send, recv, gather, scatter, sum reduce, 请你把这段代码改写成使用 ccl 加速的计算方式。如果这个矩阵不平衡，无法被整数切分，该如何处理？

            ```cpp
            #include <stdio.h>
            #include <stdlib.h>

            void broadcast(int *buf, int num_elms);
            void scatter(int *send_buf, int *recv_buf, int num_elms, int root);
            void gather(int *send_buf, int *recv_buf, int num_elms, int root);
            void sum_reduce(int *send_buf, int *recv_buf, int num_elms, int root);
            void all_gather(int *send_buf, int *recv_buf, int num_elms, int root);
            void all_sum_reduce(int *send_buf, int *recv_buf, int num_elms, int root);
            void all_to_all(int *send_buf, int *recv_buf, int num_elms);

            int main()
            {
                int num_rows_A = 1024, num_cols_A = 1024;
                int num_rows_B = 1024, num_cols_B = 1024;
                int num_rows_C = num_rows_A, num_cols_C = num_cols_B;
                int *mat_A = (int*) malloc(num_rows_A * num_cols_A * sizeof(int));
                int *mat_B = (int*) malloc(num_rows_B * num_cols_B * sizeof(int));
                int *mat_C = (int*) malloc(num_rows_C * num_cols_C * sizeof(int));

                int rank;
                int world_size;

                int span_len = 2;
                scatter(mat_A, mat_A + rank * span_len * num_cols_A,
                    span_len * num_cols_A, 0);
                broadcast(mat_B, mat_B, num_rows_B * num_cols_B);
                matmul(mat_A + rank * span_len * num_cols_A, mat_B,
                    mat_C + rank * span_len * num_cols_C);
                gather(mat_C, mat_C + rank * span_len * num_cols_C,
                    span_len * num_cols_C, 0);
                return 0;
            }
            ```

            如果无法整数切分，最后一个 block 稍大一些，使用 if else 额外处理。也可以用更小的切分粒度 + while 循环 + 抢占式任务，也可以将最后一个 block 单独再切分处理。

        * （顺利做出），现在我们考虑一个深度神经网络，对于这个网络，假如每个 block 放在一个 node 上，一共有 8 个 node，请你设计出尽可能高性能的拓扑方案。

    * cuda 了解过是吧，shared data 用过吗，有什么用？`#pragma unroll`了解过吗，大概可以节省多个条汇编指令？

        nccl 的 reduce copy 中，为什么按 hunk 处理？为什么 hunk 中一个线程处理 4 个元素？把这个数值设置得更大或者更小，有什么影响？bank conflict 听说过吗？

        基于 warp 的 ptx sync 指令都有哪些？为什么 ptx 更快？

        p2p 的通信和通过 host 中转，哪个更快，为什么？

    * 异步机制，现在有了一段 poll 实现的异步事件代码，请你分析下它可能在哪出现死锁

* nccl analysis tmp

    * `comm->rank`为当前进程被分配到的 mpi rank （猜测）

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

* nccl app

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

* driver 17 years interview

* [v] 准备面试题

    feedback:

    * linux kernel
    
        简述一下零拷贝原理。
        
        中断如何处理？（上下两部分）

        调试：
        
        简述一下如何使用 ftrace (查看函数上下文：function_graph)
        
        kgdb (`echo g > /proc/sysrq-trigger`, `gdb ./vmlinux 内核映像`, `break do_init_module`, `add-symbol-file内核模块名 .text段基地址 -s .data .data段基址 -s .bss .bss段基址`) 
        
        qemu gdb (-S -s)
        
        ebpf 了解吗？
        
        是否使用过 gdb 调试 coredump 文件？
        
        libdrm 显存管理了解吗？
    
        malloc 导致页表共用时，dma 的 iova 只接收 4K 对齐的地址，此时可能会有哪些问题？（1. 数据的覆盖；2. free 时不能全部释放，需查红黑树）

    * 指令集：riscv 中的 -fPIC 功能有什么实现的思路？
    
        arm 中的自旋锁和互拆锁，汇编是如何实现的？

    * rdma: 简述一下 rdma app 的写法；为什么使用 poll，不使用中断？mmap 后，如果用户态进程销毁，内核态如何清空资源？

    * gdb 原理是什么？gpu debug 搞过吗？

    * nccl: 做题。

    * 简述一下生产者消费者模型，简述一下订阅者，分发者模式

    项目：

    * 事件驱动，简述一下 poll, epoll 原理？协程听说过吗？调度器的 policy 是静态的还是动态的？如何实现 load balance？是否接触过 nvidia 的 nsight 软件，简述下可能的实现原理？

    * 简述一下 all reduce 的实现原理，简述 broadcast 的实现原理，简述 ReduceScatter 的作用？目前上层软件比如 vllm, sglang，都只调用到了 send / recv / all reduce 这三个函数，为什么？简述下什么是 AI 任务，如何分析 dependency？是否接触过计算算子，通信算子？numa 该如何优化？是否接触过 tvm，图优化？接口是 mlnx 写好的，为什么还要定义？rdma 网卡是否支持 roce v1/v2？ ABI 是干嘛用的？

    * 简介一下sev, sgx原理，各有什么优劣？是否听说过 libos? occlum, gramine? 简介下 fuzz 的原理？是否挖到了 cve？linux 中还有什么手段判断内存泄漏？(valgrind, gcc -ggdb3)

* nccl app

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

* topo system

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
        
        看起来这里设计的 link，可以从任意一种 topo node 走到另外任意一种 topo node。不清楚 net 与 net 之间是怎么走的。

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

        目前不知道这个干嘛用的。

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

* nccl `ncclTopoComputePaths()` 调研

    `ncclTopoSetPaths()`说是进行 bfs 搜索，但是只有两层循环，没有 queue 或者递归，看起来比较像从 cpu, gpu, nic 等出发，搜索第一层能到达的 node

    对于无法 p2p 直接的 gpu，寻找其共同的 cpu，并使用`addInterStep()`添加 cpu 中转。

    `ncclTransports` struct obj 提供了不同 transport 的统一 api。此时会调用`canConnect()`判断`srcInfo`和`dstInfo`能否连通。

    `ncclPeerInfo`如下：

    ```cpp
    struct ncclPeerInfo {
      int rank;
      int cudaDev;
      int nvmlDev;
      int gdrSupport;
      uint64_t hostHash;
      uint64_t pidHash;
      dev_t shmDev;
      int64_t busId;
      struct ncclComm* comm;
      int cudaCompCap;
      // MNNVL support
      nvmlGpuFabricInfoV_t fabricInfo;
      int cuMemSupport;
    };
    ```

    dst 由外层循环控制，src 由内层循环控制。

    接下来处理 nic，对于每个 nic，遍历 gpu，这个可能拿来判断是否使用 gpu direct rdma

    如果不能使用 gpu direct rdma，那么使用 nic - cpu - nic 做中转，同样地，使用`addInterStep()`添加中间节点。

* nccl tmp

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

    * sort link

        ```cpp
        static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
          // Shift all links to have upLink as last link
          if (upNode) {
            int l=0;
            while (node->links[l].remNode != upNode) l++;
            struct ncclTopoLink upLink;
            memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
            while (node->links[l+1].remNode) {
              memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
              l++;
            }
            memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
          }

          // ...
        }
        ```

        node 1 -> link 1 -> node 2 -> link 2

        目前看来，上述代码中的`upLink`指的就是 link 1，它相对于 node 2 来说是上游的 link，因此被称为 up link。

        代码中的`upNode`指的当然是 node 1。

        上述代码发现`upLink`后，将其移动到 node 2 所有 links 的最后一位。目前不知道这个是干嘛用的。

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

        对 system 中指定 type 的 nodes 进行搜索，如果 id 相符，那么将 path 赋值为 system 中所有 paths 对应索引的 path，并返回。

        由此可见，system 中的数据排布大概是这样的：

        ```cpp
        system:
            nodes:
                CPU:
                    nodes: 0, 1, 2, 3, 4, ...
                    CPU 0:
                        paths: GPU, PCI, NVS, CPU, ...
                    CPU 1:
                        paths: GPU, PCI, NVS, CPU, ...
                    ...
                GPU:
                    nodes: 0, 1, 2, 3, 4, ...
                    GPU 0:
                        paths: GPU, PCI, NVS, CPU, ...
                    GPU 1:
                        paths: GPU, PCI, NVS, CPU, ...
                    ...
                NIC:
                    0, 1, 2, 3, 4, ...
            paths:
                0, 1, 2, 3, 4, ...
        ```

        path 的种类一共这么多种：

        ```cpp
        #define NCCL_TOPO_NODE_TYPES 7
        #define GPU 0
        #define PCI 1
        #define NVS 2
        #define CPU 3 // Actually NUMA domains
        #define NIC 4
        #define NET 5
        ```

        这正好是 node 的类型。可见 node 的类型与 path 的类型是相同的。

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
        extern const char* topoLinkTypeStr[];

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

* nccl tmp

    * `getPath()`

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

        `node->paths[t]`是个`ncclTopoLinkList*`，但是每个 node 里 path 只有几种类型，每个类型存储一个 path 指针，这个指针，其实是一个 path 数组，由于数组的长度需要在运行时动态确定，所以后面会有 malloc 填充这个指针。

        实测过程中发现`i = 0`。如果非 0 的话，需要重新评估这个功能。

    * `TopoLinkList`、`TopoNodeList`与`TopoNodeSet`有什么不同？

* nccl tmp

    * path

        ```cpp
        static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
          if (baseNode->paths[baseNode->type] == NULL) {
            NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
          }

          // ...
        }
        ```

        每个 topo node 下有多种类型的 path，每种类型的 path 又有多条 path 实例，每条 path 其实是一个 topo node list。

        从`baseNode->paths+baseNode->type`可以看出，每个 node 只处理和当前 node 类型相同的 path。并且申请的内存大小是根据 topo system 中的数据确定的。

        因此每个 node 下的 path 可能是这样的：

        ```
        cpu 0 -> cpu 1 -> cpu 2 -> cpu 3
        ```

        这其中并没有 gpu，nic 相关的。

    * get path

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

        从`*path = node->paths[t]+i;`可以看出，每个 node 下，对应 node type 的 path 的数量是 system 中对应 node type 的 node 数量。并且每个 node 都可以映射到一条 path 上。

        比如 topo system 中，cpu 类型的 node 有 cpu_0, cpu_1, cpu_2, cpu_3，共 4 个 cpu。那么，在 cpu 0 中，其 cpu 类型的 path 一共有 4 条，并且每条 path 都可以映射到一个 cpu node 上：

        ```
        path_0 -> cpu_0
        path_1 -> cpu_1
        path_2 -> cpu_2
        path_3 -> cpu_3
        ```

        每条 path 又是一个 link list，其中有多个 link，每个 link 都有一个 target node 属性。

    * `ncclTopoCreateNode()`

        ```cpp
        ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
          if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
            WARN("Error : tried to create too many nodes of type %d", type);
            return ncclInternalError;
          }
          struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
          system->nodes[type].count++;
          n->type = type;
          n->id = id;
          if (type == GPU) {
            n->gpu.dev = NCCL_TOPO_UNDEF;
            n->gpu.rank = NCCL_TOPO_UNDEF;
            // ...
          }
          // ...
        }
        ```

        因为 nccl 中是提前把数组中元素申请好的，所以靠增加 count 计数来表示目前使用了多少个数组元素，模拟内存分配的过程。

    * `ncclTopoAddPci()`

        ```cpp
        ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId) {
          const char* str;

          int type;
          NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
          NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

          int64_t busId;
          NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
          NCCLCHECK(busIdToInt64(str, &busId));

          struct ncclTopoNode* node = NULL;
          struct ncclXmlNode* xmlGpu = NULL;
          NCCLCHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
          if (xmlGpu != NULL) {
            type = GPU;
            int index;
            // ...
          }
          // ...
        }
        ```

        这里的 add pci，实际上是从 xml 的 pci tag 中，解析出来此 pci 上具体是个什么设备，可能是 gpu，可能是 nic，也可能是另一个 pci。

        解析完后，将实际的 device 对接到 parent topo node 上（目前看来 parent topo node 是 cpu）。

    * `ncclTopoConnectNodes()`

        ```cpp
        ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
          // Aggregate links into higher bw for NVLink
          struct ncclTopoLink* link;
          for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
            if (link->remNode == remNode && link->type == type) break;
          }
          if (link - node->links == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            return ncclInternalError;
          }
          if (link->remNode == NULL) node->nlinks++;
          link->type = type;
          link->remNode = remNode;
          link->bw += bw;
          // ...
        }
        ```

        每个 topo node 中有默认创建的最大 128 条 link，每条 link 包含一个 remote topo node （或者可以理解为 target topo node, dst topo node, peer topo node 等），这些 link 并不构成单向链表（path 是 link 组成的 list，因此 path 构成单向链表）。

        上述代码对 node 的 link 数组进行遍历，若有 link 指向`remNode`，说明之已经添加过这条 link，现在又被要求添加这条 link，说明 node 和 remNode 之间不止有一条 link，这种情况只有可能是多条 nvlink。后续在处理这种情况时，我们看到它使用`link->bw += bw;`累加 bindwidth，聚合 nvlink 的带宽。

        若找不到 link 指向`remNode`，说明这条 link 还没被创建，此时使用`node->nlinks++;`使指针递增一位，模仿 malloc 的效果。然后使用

        ```cpp
        link->type = type;
        link->remNode = remNode;
        link->bw += bw;
        ```

        对这条新创建的 link 进行初始化。

    * system id 的含义

        一个 topo system 可能存了好多个 host hash，每个 host hash 对应一个 host。如果我想找到指定的 host，那么其对应的数组索引 idx 就是 system id。

    * `ncclTopoCreateNode()`

        这个函数模仿 malloc 的功能。

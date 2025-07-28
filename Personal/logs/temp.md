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

        `node->paths[t]`是个`ncclTopoLinkList*`，但是每个 node 里 path 只有几种类型，每个类型存储一个 path 指针，这个指针，其实是一个 path 数组，由于数组的长度需要在运行时动态确定，所以后面会有 malloc 填充这个指针。

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

        每个 topo node 下有多种类型的 path，每种类型的 path 又有多条 path 实例，每条 path 其实是一个 topo path list。

        从`baseNode->paths+baseNode->type`可以看出，每个 node 只处理和当前 node 类型相同的 path。并且申请的内存大小是根据 topo system 中的数据确定的。

        因此每个 node 下的 path 可能是这样的：

        ```
        cpu 0 -> cpu 1 -> cpu 2 -> cpu 3
        ```

        这其中并没有 gpu，nic 相关的。

        2025053100: 上述代码表示“如果未初始化从 base node 出发，到 base node 类型的 path，那么将所有 base node 类型的 path 都初始化一下”。这个初始化实际上是为后面 bfs 的种子轮搜索服务的。前面的推理明显是完全错误的，错误原因一是将 base node 泛化为了所有 node，其实应该只能推断出在这里对 base node 类型的 path 做了申请内存和初始化，但是不知道拿来干嘛的；二是没有做实验验证`cpu 0 -> cpu 1 -> cpu 2 -> cpu 3`是否正确，如果验证了马上发现 path 中会有多种 node，那么前面的推理就被全部推翻了。由此我们得到的启示是：如果想要做出推断，那么必须做实验。

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

        `*path = node->paths[t]+i;`: 这里的`i`指的是 nodes 中第 i 个 node，但是 i 也被用到了这里 path 上，说明 path 的排序也是按照 node 的顺序来的，那么很有可能每条 path 的第一个 node 是 path 起始的 node。这样的话，这行代码的含义就可以为：每个 node 只处理从自己开始的那条 path。

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

    * `ncclGetSystemId()`

        这个函数不只是搜索已有的 host hashes，而且还当 host hashed 不存在时，创建一个新的位置，并将当前要查询的 host hash 填入其中，并返回指向当前 host hash 的索引。

        功能有点像 c++ 中 unordered map 的`operator[]`查询。

    * `ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu);`

        这个函数参数中有`system`，在函数体实现中并没有用到这个，只是根据 xml tag 的相关字段，填充了 topo node 的相关数据。

    * topo system add nic

        ```cpp
        if (strcmp(node->name, "nic") == 0) {
          struct ncclTopoNode* nic = NULL;
          NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, 0));
          if (nic == NULL) {
            NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, NCCL_TOPO_ID(systemId, 0)));
            NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
            NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
          }
          NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
        }
        ```

        这个在`ncclTopoGetNode()`最后一个参数 id 处填了个 0，是表示一个 system 下只有一个 nic tag 吗？

        `ncclTopoCreateNode()`给出的 local id 也恒为 0。不清楚为什么。

    * system id and bit shift

        ```cpp
        // (uint64_t) is necessary, or shifing will occured on a 32 bit register
        // ((uint64_t) system_id << 56) is necessary, or expression 56) + numa_id will be calculated first
        system_id = 1;
        uint64_t topo_id = ((uint64_t) system_id << 56) + numa_id;
        ```

    * insert pci node

        ```cpp
        for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s-1];
        parent->subs[subIndex] = pciNode;
        ```

        reserve a placeholder for pci node.

        why use `subIndex` as the index of pci node? answer: nccl has to Keep PCI sub devices ordered by PCI Bus ID

* nccl tmp

    * get path

        ```cpp
        getPath(system, node, baseNode->type, baseNode->id, &path);
        ```

        可以看到，`getPath()`在被调用时，指定了 node，又指定了 base node，其含义为搜索 topo system，找到 base node 的位置，由于指定 node 开始的 path 的索引与 topo node 在 topo system 中的索引一致，所以我们便可以在 node 中定位以 base node 开头的 path 的索引。

        example:

        topo system 中 topo node 的排布如下：

        ```
        topo_system:

        cpu:
        cpu_node_0
        cpu_node_1

        gpu:
        gpu_node_0
        gpu_node_1
        gpu_node_2
        gpu_node_3

        ...
        ```

        `cpu_node_0`的 path 排布如下：

        ```
        path:

        cpu:
        cpu_node_0 -> gpu_node_0 -> gpu_node_1
        cpu_node_1 -> gpu_node_1 -> gpu_node_2

        gpu:
        gpu_node_0 -> cpu_node_0 -> cpu_node_1
        gpu_node_1 -> cpu_node_1 -> cpu_node_0
        gpu_node_2 -> cpu_node_0 -> cpu_node_1
        gpu_node_3 -> cpu_node_1 -> cpu_node_0

        ...
        ```

        假如我现在想在`cpu_node_0`中，找到以`cpu_node_1`开头的 path，但是我现在只有`cpu_node_1`的`topo_id`，那么我可以根据这个 id，在 topo system 中找到其对应的索引，然后根据这个索引在`cpu_node_0`的 paths 中找到对应的 path。

    * topo path 不是只搜索同类型的 node 之间的连接

        ```cpp
        NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
        ```

        这个 alloc 代码，size 使用的是`system->nodes[baseNode->type].count`，表示从各个 node 开始的 path。还是可以连接到其他类型的 node 的。

* reverse link

    ```cpp
    if ((remPath->bw == 0 || remPath->count > path->count) && remPath->bw < bw) {
      // Find reverse link
      for (int l=0; l<remNode->nlinks; l++) {
        if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
          remPath->list[0] = remNode->links+l;
          break;
        }
      }
      // ...
    }
    ```

    `path`表示从当前 node 走向 base node 的路径。

    `remNode`是当前 node 通过 link 指向的下一个 node，即 node --link--> remNode。

    `remPath`是`remNode`指向 base node 的路径。

    目前通过 base node 一路搜索过来的情况是这样的：base node -> ... -> node --link--> rem node。注意，这条链路是单向的，base node 能走到 node，走到 rem node，不代表 node / rem node 也能原路返回。

    现在我们开始分析 if 语句中条件的含义：

    * `remPath->bw == 0`: 表示 rem node 找不到返回 base node 的 path。

    * `remPath->count > path->count`: 表示 rem node 走到 base node 的路径比 node 走到 base node 的路径长。

        如果把这里的大于号改成小于号，即 rem node 到 base node 的路径比 node 到 base node 的路径更短，那么就没必要找 reverse link 了，我们直接走 rem path 返回就可以。事实上 nccl 也是这么做的。

    结合上下文，我们可以推测出，当 rem node 找不到返回 base node 的路径，或者说，找到了，但是路径比从 node 返回的路径长，那么就选择找到一条 rem node 到 node 的边，并通过 node 返回 base node，即 rem node --rem_link--> node -> ... -> base node。

    * `remPath->bw < bw`：这个比较好理解了，只有当 rem path 的带宽小于 path 的带宽，我们才尝试 reverse link -> path 的个组合的带宽是否有可能大于 rem path 的带宽。如果 rem path 的带宽本身比 path 的带宽大，那么就不考虑 path 和 rem path 上节点的数量了。

* nccl 是可以处理 pci 中的多个 child tag 的

    ```cpp
    ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId, int numaId) {
        // ...
        for (int s=0; s<xmlPci->nSubs; s++) {
          struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
          if (strcmp(xmlSubPci->name, "pcilink") != 0) { // PCI links will be added later
            NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
          }
        }
        // ...
    }
    ```

    `ncclTopoAddPci()`是 dfs 的结构，只有遇到 gpu 和 nic 时，才做 terminate 处理。由于 gpu/nic 可能和另一个 pci tag 并列，所以 child pci tag 并未放到函数末尾处理。

    这里先处理了 pci tag 的子节点，处理完后才 connect 当前 pci tag 创建出来的 pci node 和 parent node，说明这是一个树的后序遍历。理论上先序遍历也可以达成一样的效果，后面可以试一下。

* nccl 假设 pci 下要么只有一个 gpu 或一个 nic，要么只有另一个 pci，因此其递归终止的条件是解析当前节点，如果当前 pci 节点下有 gpu/nic，那么就停止递归。否则就认为当前节点下还有嵌套 pci，那么以当前节点为 parent pci 节点，遍历子节点，是非常常规的思路。

    但是 siccl 的 pci 下可能有多个 gpu 节点，这样我们在解析到 pci 后，并不能停止解析当前 pci，还要继续解析下去，这样就导致 gpu 的解析和子 pci 的解析被放到了同一个循环下。如果解析到了子节点是 pci 节点，我们以当前 pci 节点作为 parent 节点，再添加子 pci 时，相当于没有跳过当前 pci node 的创建，tag 也往下走了一层。

    按道理效果应该和直接递归调用是一样的。

    可以试试在写全排列时，对于 n = x 的情况单独展开，是否和统一写在 for 里一样。

    ```
    <parent-pci 0>
        <pci 1>
            <pci 2>
                <gpu>
            <pci 3>
                <gpu>
            
    ```

    按照目前的方案，pci 1 检测到 pci 2 是 child pci tag，转到小循环开始处理。pci 1 作为 parent node，重新扫描所有的子 tag。但是我们需要注意到，在 pci 1 的外层大循环中，仍然会处理 pci 2 和 pci 3。此时当处理 pci 3 时，又进入小循环，pci 2 又被处理了一遍。nccl 之所以能写小循环，是因为它没有外层的大循环。

* `ncclTopoGetLocal()`

    `locals`是 node idx 的数组，`localCount`是数组的长度。

    ```cpp
    ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType, int** locals, int* localCount, int* pathType) {
    ```

    参数含义为，从`type, index`的 node 出发，在指向所有`resultType`类型的 node 的 path 中，找到 bw 最大的那几条 path。

    `locals`中的内容为`0, 1`，对应`localNets`。

    第二次调用，`locals`中的内容为`0`，对应`localGpus`。

    `net`一直为 1，`channelId`是外部传进来的，为 0。

    `id`是个指针，刚传进来时是个未初始化的随机数，之后被赋值为 2。

    看起来`net`只是为了选择`localNets[]`数组中的哪个元素。

    `dev`是 net 的 dev，与 gpu 没有关系。外部传入的是 NULL，说明外部不需要这个参数，直接跳过不填。

    `ncclTopoGetLocalNet()`的作用，猜测可能是根据指定的 gpu node（只能是 gpu，不能是其他），在已有的 net node 中，找到一个带宽最大，路径约束最严的 net node。

    * 两条 if 的写法

        正常我们找最大值并保存，通常会这样写：

        ```cpp
        vector<int> max_vals;
        int max_val;
        for (int val : val_arr) {
            if (val > max_val) {
                max_vals.clear();
                max_vals.push_back(val);
                max_val = val;
                continue;
            }
            if (val == max_val) {
                max_vals.push_back(val);
            }
        }
        ```

        而 nccl 的写法是这样的：

        ```cpp
        if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
          maxBw = paths[i].bw;
          minType = paths[i].type;
          if (pathType)
            *pathType = minType;
          count = 0;  // 在这里清空临时存储的最大值
        }
        if (paths[i].bw == maxBw && paths[i].type == minType)
            (*locals)[count++] = i;  // 在这里添加最大值
        ```

        对应上面的写法，为：

        ```cpp
        vector<int> max_vals;
        int max_val;
        for (int val : val_arr) {
            if (val > max_val) {
                max_vals.clear();
                max_val = val;
            }
            if (val == max_val) {
                max_vals.push_back(val);
            }
        }
        ```

        这样两个 if 的逻辑关系改成了互相关联，还是挺有意思的。

    * `if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {`

        根据这行代码可以看出，如果有 bw 更大的 path，那么优先找 bw 更大的。如果 bw 相等，那么找 path type 更小的（即约束更严的，尽量 p2p 的）

    * `NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, &localNets, &localNetCount, NULL));`

        根据上面的分析，这行代码的作用就是，从`GPU`类型，index 为`gpu`的 node 出发，找到离`NET`类型节点 bw 最大的 path，或者 bw 相同时，paty type 最小的 path。

* `ncclTopoGetLocalNet()`

    反正要在函数内部调用`ncclTopoRankToIndex()`根据 rank 找到 gpu node idx，那么为什么不从一开始就传入 gpu node idx，或 gpu node 的指针？

    为什么传入参数要引入`channelId`，有什么用？

    `net % localNetCount = 1`, `localNets = [0, 1]`

    `system->nodes[NET][N]->id = {1, 2, 0}`，siccl, nccl 的值相同

    nccl `net->id = 2, 1`

    `nets = {1, 0}`, `localNets = {1, 0}`, `netCount = 2`

    * `ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {`

        根据前后的分析，`ncclTopoGetLocalNet()`的作用是给定 gpu rank 和 channel，找出一个合适的、最优的网卡（或 net interface 接口）。

    * 对下面代码的解析

        ```cpp
        int net = system->nodes[GPU][gpu]->gpu.dev;
        if (isPow2(localNetCount)) {
            net = mirrorBits(net, localNetCount);
        }
        // 假如网卡数量为 5，gpu 数量为 2，那么 div(5, 2) = 3
        // 即前面每 3 张网卡服务 1 个 gpu，最后 1 个 gpu 由剩余网卡（2 张）负责
        // 以 gpu 的视角来看，是这样的：
        //      gpu_0                  gpu_1
        //   /    |    \           /     |    \ 
        // net0  net1  net2       net3  net4  NULL
        // idx_0 idx_1 idx_2      idx_0  idx_1
        // 那么 channelId % 3 的含义即为，当前 channel 对应的 net 的 idx
        // net 前面被赋值的是 gpu 的 dev 号，现在 += 这个值，说明 gpu_1 由
        // net[1, 2, 3] 负责，并非由上图的 net[3, 4] 负责。
        net += channelId % div_up(localNetCount, localGpuCount);
        ```

        仿照这段逻辑写了个程序，代码和输出如下：

        ```cpp
        bool isPow2(int val) {
            return (val & (val-1)) == 0;
        }

        int mirrorBits(int val, int pow2) {
            int mirror = 0;
            // mb 表示最高位的 1
            for (int b = 1, mb = (pow2>>1); b < pow2; b <<= 1, mb >>= 1) {
                // 如果 val 最低位为 1，那么 mirror 的最高位置 1
                if (val & b) {
                    // |= 保护除了最高位之外的其他位不被改变
                    mirror |= mb;
                }
            }
            return mirror;  // 0b110;
        }

        int div_up(int x, int y) {
            return (x + y - 1) / y;
        }

        #include <stdio.h>

        int main() {
            int localNetCount = 8;
            int localGpuCount = 4;
            for (int channelId = 0; channelId < 6; ++channelId) {
                for (int i = 0; i < localGpuCount; ++i) {
                    int net = i;
                    if (isPow2(localNetCount)) {
                        net = mirrorBits(net, localNetCount);
                    }
                    net += channelId % div_up(localNetCount, localGpuCount);
                    printf("gpu idx: %d, channel: %d, net: %d\n", i, channelId, net);
                }
                putchar('\n');
            }
            return 0;
        }
        ```

        output:

        ```
        gpu idx: 0, channel: 0, net: 0
        gpu idx: 1, channel: 0, net: 4
        gpu idx: 2, channel: 0, net: 2
        gpu idx: 3, channel: 0, net: 6

        gpu idx: 0, channel: 1, net: 1
        gpu idx: 1, channel: 1, net: 5
        gpu idx: 2, channel: 1, net: 3
        gpu idx: 3, channel: 1, net: 7

        gpu idx: 0, channel: 2, net: 0
        gpu idx: 1, channel: 2, net: 4
        gpu idx: 2, channel: 2, net: 2
        gpu idx: 3, channel: 2, net: 6

        gpu idx: 0, channel: 3, net: 1
        gpu idx: 1, channel: 3, net: 5
        gpu idx: 2, channel: 3, net: 3
        gpu idx: 3, channel: 3, net: 7

        gpu idx: 0, channel: 4, net: 0
        gpu idx: 1, channel: 4, net: 4
        gpu idx: 2, channel: 4, net: 2
        gpu idx: 3, channel: 4, net: 6

        gpu idx: 0, channel: 5, net: 1
        gpu idx: 1, channel: 5, net: 5
        gpu idx: 2, channel: 5, net: 3
        gpu idx: 3, channel: 5, net: 7
        ```

        可以看到，当`localNetCount`为偶数时，在每个 channel 里分配的 net，先是偶数，再是奇数。

        如果`localNetCount`为奇数，为了更好地找规律，我们将程序修改如下：

        ```cpp
        bool isPow2(int val) {
            return (val & (val-1)) == 0;
        }

        int mirrorBits(int val, int pow2) {
            int mirror = 0;
            // mb 表示最高位的 1
            for (int b = 1, mb = (pow2>>1); b < pow2; b <<= 1, mb >>= 1) {
                // 如果 val 最低位为 1，那么 mirror 的最高位置 1
                if (val & b) {
                    // |= 保护除了最高位之外的其他位不被改变
                    mirror |= mb;
                }
            }
            return mirror;  // 0b110;
        }

        int div_up(int x, int y) {
            return (x + y - 1) / y;
        }

        #include <stdio.h>

        int main() {
            int localNetCount = 13;
            int localGpuCount = 3;
            for (int channelId = 0; channelId < 6; ++channelId) {
                for (int i = 0; i < localGpuCount; ++i) {
                    int net = i;
                    if (isPow2(localNetCount)) {
                        net = mirrorBits(net, localNetCount);
                    }
                    printf("gpu idx: %d, channel: %d, net before: %d, ", i, channelId, net);
                    net += channelId % div_up(localNetCount, localGpuCount);
                    printf("net after: %d\n", net);
                }
                putchar('\n');
            }
            return 0;
        }
        ```

        output:

        ```
        gpu idx: 0, channel: 0, net before: 0, net after: 0
        gpu idx: 1, channel: 0, net before: 1, net after: 1
        gpu idx: 2, channel: 0, net before: 2, net after: 2

        gpu idx: 0, channel: 1, net before: 0, net after: 1
        gpu idx: 1, channel: 1, net before: 1, net after: 2
        gpu idx: 2, channel: 1, net before: 2, net after: 3

        gpu idx: 0, channel: 2, net before: 0, net after: 2
        gpu idx: 1, channel: 2, net before: 1, net after: 3
        gpu idx: 2, channel: 2, net before: 2, net after: 4

        gpu idx: 0, channel: 3, net before: 0, net after: 3
        gpu idx: 1, channel: 3, net before: 1, net after: 4
        gpu idx: 2, channel: 3, net before: 2, net after: 5

        gpu idx: 0, channel: 4, net before: 0, net after: 4
        gpu idx: 1, channel: 4, net before: 1, net after: 5
        gpu idx: 2, channel: 4, net before: 2, net after: 6

        gpu idx: 0, channel: 5, net before: 0, net after: 0
        gpu idx: 1, channel: 5, net before: 1, net after: 1
        gpu idx: 2, channel: 5, net before: 2, net after: 2
        ```

        可以看到，net 在循环后移，增量为 0 ~ 4，正好 5 个，这里的 5 是从`13 / 3`并向上取整而来。

        总体上这段代码是 gpu 和 net 的负载均衡策略。

    * 这段代码比较奇怪，从 gpu 开始找 net 的时候，搜索到的是全部符合条件的 net，但是从 net 反向搜索 gpu 时，只使用了第 1 个 net。这样下来，那就不一定其他 net 也能连到这些 gpu 了。然而后面又使用了所有 net 的 count，说明其他的 net 也会参与其中。这样就矛盾了。

        目前只有两个解释：

        1. `net[0]`可以代表其他的 net 的情况，不需要其他 net 重复搜索了。

        2. `ncclTopoGetLocal()`函数不仅仅是为`ncclTopoGetLocalNet()`设计的，在其他地方也有调用到，所以输出才为数组的形式。如果只为`ncclTopoGetLocalNet()`设计，那么只给出第一条最佳 path 就可以了，不需要给出数组的形式。

    * 目前看起来，不进行静态负载均衡应该影响不大。只有到时候实际程序跑起来，又有多网卡、多显卡的情况，才能尝试测试这里的静态负载均衡是否合理。

* `ncclTopoSelectNets()`

    `localNets`是个数组，存放 net node 的 idx。数组长度为`localNetCount`。

    循环中调用`ncclTopoGetLocalNet()`，只是改变 channel，其他的都不变。

    第 1 次返回的`netId`为`2`，第 2 次返回的`netId`为`1`，第 3 次返回的`netId`为`2`。循环一共 3 次就停止，停止条件是`localNets`新添加的元素和第一个元素相等。退出循环时，`localNets`的长度`localNetCount`为 2.

    猜想：可能会循环或对称分配 net 资源，当 net 资源被分配完（新分配的数据又重新回到开头）时，则停止分配。

    看不懂

    ```cpp
      if (found == netCount)
        nets[netCount++] = n;
    ```

    这个是什么意思。

    ```cpp
      // Then add others satisfying typeInter
      for (int t=0; t <= typeInter; t++) {
    ```

    这里的`typeInter`为 3，是`graph->typeInter`传进来的。

    最终函数返回时，`netCount`为 2，并将其赋值给函数参数`*netCountRet`。

    * 对 gpu rank 与 channel 的遍历

        ```cpp
          for (int g=0; g<system->nodes[GPU].count; g++) {
            if (gpu != -1 && gpu != g) continue;
            localNetCount = 0;
            struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
            for (int c = 0; c<MAXCHANNELS; c++) {
              int64_t netId;
              print_path_1(system);
              NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
              // ...
        ```

        我们已知`ncclTopoGetLocalNet()`是根据 gpu rank 和 channel，找出最优的到 net 的 path，这里的两层 for，外层是对 gpu 进行遍历，内层是对 channel 进行遍历，刚好对应`ncclTopoGetLocalNet()`函数的作用。

    * 有关 channel 与 net

        ```cpp
        for (int c = 0; c<MAXCHANNELS; c++) {
          int64_t netId;
          print_path_1(system);
          NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
          NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, localNets+localNetCount));
          if (localNetCount > 0 && localNets[localNetCount] == localNets[0])
            break;
          localNetCount++;
        }
        // Append NICs to list
        for (int i=0; i<localNetCount; i++) {
          int n = localNets[i];
          int found = 0;
          while (nets[found] != n && found<netCount)
            found++;
          if (found == netCount)
            nets[netCount++] = n;
        }
        ```

        前面的`MAXCHANNELS`为 32，跳出第一个 for 走的是`break`语句，说明搜索到了重复的 local net，而`localNetCount`为 2，刚好对应 net node 为 2. 这似乎暗示每条 channel 都对应一个不同的 net，并不是一个 net 衍生出多条 channel。

    * `nets`是外部传入的，`ncclTopoSelectNets()`会尝试往这个数组里 append 新的 net node 的 idx。

    * `ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, int* nets, int* netCountRet) {`

        整体看来，`ncclTopoSelectNets()`似乎是往`nets`里添加新元素（即满足条件的 net node 的 idx）。

        不清楚`typeInter`是干嘛用的。

* nccl get env

    ```cpp
    const char *ncclGetEnv(const char *name) {
      static pthread_once_t once = PTHREAD_ONCE_INIT;
      pthread_once(&once, initEnv);
      return getenv(name);
    }
    ```

    nccl get env 使用 pthread 调用起一次`initEnv()`函数。

    其内容如下：

    ```cpp
    void initEnv() {
      char confFilePath[1024];
      const char * userDir = userHomeDir();
      if (userDir) {
        sprintf(confFilePath, "%s/.nccl.conf", userDir);
        setEnvFile(confFilePath);
      }
      sprintf(confFilePath, "/etc/nccl.conf");
      setEnvFile(confFilePath);
    }
    ```

    ```cpp
    const char* userHomeDir() {
      struct passwd *pwUser = getpwuid(getuid());
      return pwUser == NULL ? NULL : pwUser->pw_dir;
    }

    void setEnvFile(const char* fileName) {
      FILE * file = fopen(fileName, "r");
      if (file == NULL) return;

      char *line = NULL;
      char envVar[1024];
      char envValue[1024];
      size_t n = 0;
      ssize_t read;
      while ((read = getline(&line, &n, file)) != -1) {
        if (line[read-1] == '\n') line[read-1] = '\0';
        int s=0; // Env Var Size
        while (line[s] != '\0' && line[s] != '=') s++;
        if (line[s] == '\0') continue;
        strncpy(envVar, line, std::min(1023,s));
        envVar[s] = '\0';
        s++;
        strncpy(envValue, line+s, 1023);
        envValue[1023]='\0';
        setenv(envVar, envValue, 0);
        //printf("%s : %s->%s\n", fileName, envVar, envValue);
      }
      if (line) free(line);
      fclose(file);
    }
    ```
* `ncclTopoRemoveNode()`

    ```cpp
    ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
      struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
      for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
        free(delNode->paths[t]);
        for (int n=0; n<system->nodes[t].count; n++) {
          struct ncclTopoNode* node = system->nodes[t].nodes+n;
          if (node == delNode) continue;
          for (int l=0; l<node->nlinks; l++) {
            while (l<node->nlinks && node->links[l].remNode == delNode) {
              memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
              node->nlinks--;
            }
            if (l<node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
              node->links[l].remNode--;
            }
          }
        }
      }
      memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
      system->nodes[type].count--;
      return ncclSuccess;
    }
    ```

    * `free(delNode->paths[t]);`，每个 path 类型里的数据是单独 malloc 的，亏你还能想到这个，这么分配内存，正常人早就忘了

    * `struct ncclTopoNode* node = system->nodes[t].nodes+n;`，找到所有 node 的所有 edge，把里面指向 del node 的 edge 全都删了


    * `node->links[l].remNode--;`：由于 node 都在连续内存里，所以删掉一个 node 后，会空出来一块，我们需要把后面的补上，而 rem node 存储的也是指针数据，由于前面做了补全操作，所以这里的指针都指向了错误的数据，必须向前移动一位才对。

        siccl 里每个 node 都是单独 malloc 出来的，所以删除某个 node，不影响其他 node 的指针。

    * `memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));`：前面的准备工作做完了，现在删除 node。

* `generate_coll_graph()`

    * `crossNic`，不清楚干嘛用的，最终的值为 0，看起来像是没启用

    * `ccMin`: V100 上，`ccMin`为 70.

    * `nspeeds`: 目前为 19，`speedArray`目前为 48, 30, 28, ...

    * `tmpGraph.typeIntra`会在

        ```cpp
        tmpGraph.typeIntra = ngpus == 1 ? PATH_LOC : PATH_NVL;
        ```

        被清空为 0，然后从 0 开始搜索

    * `ncclTopoSearchRecNet()`几个关键点

        1. `ncclTopoSelectNets()`先找到可用的 nets

        2. 遍历所有 ntes，调用`ncclTopoSearchTryGpu()`找 gpu

        3. `ncclTopoFollowPath()`是主要找 gpu 的函数，但是经常返回`gpu`空指针

        4. 回退`bw`。

    * 常用流程

        1. `ncclTopoSearchRec()`开始搜索

        2. 通过`if (tmpGraph.sameChannels == 1`再找一遍，同样类型的 path，第二次不再进入此分支

        3. 进入`if (tmpGraph.typeIntra < maxTypeIntra`分支，使用`tmpGraph.typeIntra += 1;`放松对 path 的约束，并 goto search 继续搜索。

            `tmpGraph.typeIntra`增大到 4 时不再增大，因为`maxTypeIntra`就是 4.

            `maxTypeIntra`会随着`tmpGraph.typeInter`的增大而增大。

        4. 进入`if (system->nodes[NET].count > 0`分支，`tmpGraph.typeInter`被增大到 4，并清空前面状态，重新 goto search 开始搜索。

            `typeInter`依次变为 4, 5, 6

        5. `tmpGraph.typeInter = 6, tmpGraph.typeIntra = 0`时，退出 goto 循环。
        
            `tmpGraph.typeInter = PATH_PIX;`又将其设置为`3`。

            退出循环的关键改变时，`graph->nChannels = 2`，不再是 0.

            `ngpus`仍为 1，`tmpGraph.typeIntra`被设置为 0 (`PATH_LOC`)

        6. 在`if (crossNic == 2 && tmpGraph.crossNic == 0)`分支处重新 goto search.

            此后`tmpGraph.typeIntra`总是`0`，不再增加。`tmpGraph.typeInter`继续按`4, 5, 6`增加

            这个分支有点像前面的 same channel，先搜索一遍，然后改变条件再搜索一遍。

        7. `speedIndex = 16`, `tmpGraph.bwIntra`和`tmpGraph.bwInter`都被设置为`1.2`，然后进入 done 环节。

    done 环节：

    1. 在`if (pass == 2)`分支重新进入 goto search。

    整体看来，topo compute 整个函数，有点像不断改变搜索条件，去反复搜索。

    `int ngpus = system->nodes[GPU].count;`，可以看到 trim system 后，`ngpus`总为 1.
    
* `ncclTopoSearchRecGpu()`

    ```cpp
    ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
    ```

    这是一个递归函数，通常由`ncclTopoSearchTryGpu()`发起调用。

    * 所有`ncclTopoSearchTryGpu()`的调用，step 都是手动指定的 0.

    * 第 5 个参数`step`, 在外部传入的变量名为`ngpus`.
    
        search rec gpu 递归调用，step 不再为 0, 而是 ngpus 或 step

        search try gpu 调用 search rec gpu 时，step 是直接拿的外部传给 search try gpu 的值。

    * ngpus 在 search rec gpu 的`int ngpus = system->nodes[GPU].count;`处被赋值。

        既然这样，为什么有的 search rec gpu 递归调用时，step 仍为 0？

    * `backToNet`的值不是固定的，有时为 0，有时为 -1

    * 第一次进入 search rec gpu，传入的 step 为 0，第二次进入 search rec gpu，传入的 step 为 ngpus.

    * `graph->intra[graph->nChannels * ngpus + step] = gpu->gpu.rank;`

        这里的索引比较有意思。`nChannels`取值为 0，1，2，`ngpus`恒为`nodes[GPU]`的数量，即`1`，`step`显然就是中间路径了。

        猜想：每个 channel 都预留了 ngpus 个中间节点的位置，step 则表示目前在填充哪一个中间节点。比如假如 ngpus = 3, 那么`intra`数组可能为`[[x, x, x], [x, x, x], [x, x, x], ...]`，如果此时 nchannel = 1, step = 0，那么修改的就是`[[x, x, x], [o, x, x], [x, x, x], ...]`这个位置的数据。

    * 猜想`backToNet`是一个 border，如果`backToNet = -1`,那么在 path 里添加新 node

    * `if (step == ngpus) {`

        猜想这个可能是递归结束的条件，就像前面的数组，如果 ngpus 被填满，比如`[[o, o, o], [x, x, x], [x, x, x], ...]`，那么结束当前递归。

    * `if (graph->nChannels < graph->maxChannels) {`

        ```cpp
        if (graph->nChannels < graph->maxChannels) {
            NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
        }
        ```

        这里不是递归调用自己，而是通过`ncclTopoSearchRec()`调用最外层的函数，再由`ncclTopoSearchRec()`去调用 search rec get -> search rec gpu。为什么要这样设计？   

    * `nChannels`变成 1 后，`step`又从 0 开始

    * 后续会在`ncclTopoSearchRecNet()`函数中，因为

        ```cpp
        if (net->net.bw < bw)
            continue;
        ```

        而退出搜索。

    * 两个 channel ，占用了两个 net node，是否意味着 channel 都是带宽确定的 path？

* `ncclTopoSearchRec()`

   1. `ncclTopoSearchRecNet()`
   
        ```cpp
        ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
        ```

        `backToNet = 0`, `backToFirstRank = -1`

        1. `ncclTopoSearchTryGpu()`

            ```cpp
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0));
            ```

            `step = 0`, `backToNet = 0`, `backToFirstRank = -1`

    * `graph->intra`的值为`1, 1, 0, 0`

    * 主要是三个函数在调来调去`ncclTopoSearchRecNet()`, `ncclTopoSearchRec()`, `ncclTopoSearchRecGpu()`


* `ncclTopoFollowPath()`

    在`if (mult == 1 && (path->type > type))`处，`path->type = 6`，`type = 3`，直接退出函数。

    ```cpp
    int type = intra ? graph->typeIntra : graph->typeInter;
    ```

    `ncclTopoSearchRecGpu()`中，`graph->nChannels--;`又使得 channel 数变成 0.ssss

* `check_p2p()`

    当`vert_1`和`vert_2`相等时，`p2pLevel = PATH_PXB`,然后会在

    ```cpp
    if (path.type <= p2pLevel) {
        p2p_active = 1;
    }
    ```

    处，`p2p_active`变成 1. 此时`path.type = 0`。

    当`vert_1`不等于`vert_2`时，上述代码处，`path.type = 6` (`PATH_PHB`)。这里的`path`是从`vert_1`指向`vert_2`。

    之后在`check_p2p()`中的大部分的流程都被跳过。

    50 机器的 p2p 同样返回 0.

    只有当`p2p`为`1`时，才会检查`IGNORE_DISABLED_P2P`的值，否则这个环境变量没用。

    `NCCL_P2P_DISABLE`的值似乎会影响是否强制关闭 p2p。

* `ncclPxnDisable()`

    ```cpp
    // Net v4 plugins don't have non-blocking connect/accept. We can't therefore use
    // remote proxies without risking deadlocks
    int ncclPxnDisable(struct ncclComm* comm) {
      static int pxnDisable = -1;
      if (pxnDisable == -1) {
        if (comm && ncclNetVersion(comm) == 4) {
          INFO(NCCL_INIT, "PXN Disabled as plugin is v4");
          pxnDisable = 1;
        } else {
          pxnDisable = ncclParamPxnDisable();
        }
      }
      return pxnDisable;
    }

    int ncclNetVersion(struct ncclComm* comm) {
      return
        (comm->ncclNet == &ncclNet_v5_as_v8) ? 5 :
        (comm->ncclNet == &ncclNet_v6_as_v8) ? 6 :
        (comm->ncclNet == &ncclNet_v7_as_v8) ? 7 :
        8;
    }
    ```

    因为`ncclNetVersion()`不可能返回 4，所以`pxnDisable`总是由环境变量决定，目前这个值被设置为`0`，因此`ncclPxnDisable()`总是被设置为`0`。

    目前仍不知道 pxn 是干嘛用的。有时间可以调研下`ncclTopoGetPxnRanks()`。

    ```cpp
    NCCL_PARAM(PxnDisable, "PXN_DISABLE", 0);

    #define NCCL_PARAM(name, env, deftVal) \
      int64_t ncclParam##name() { \
        constexpr int64_t uninitialized = INT64_MIN; \
        static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
        static int64_t cache = uninitialized; \
        if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
          ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache); \
        } \
        return cache; \
      }

    void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
      static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
      pthread_mutex_lock(&mutex);
      if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
        const char* str = ncclGetEnv(env);
        int64_t value = deftVal;
        if (str && strlen(str) > 0) {
          errno = 0;
          value = strtoll(str, nullptr, 0);
          if (errno) {
            value = deftVal;
            INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
          } else {
            INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
          }
        }
        __atomic_store_n(cache, value, __ATOMIC_RELAXED);
      }
      pthread_mutex_unlock(&mutex);
    }

    const char *ncclGetEnv(const char *name) {
      static pthread_once_t once = PTHREAD_ONCE_INIT;
      pthread_once(&once, initEnv);
      return getenv(name);
    }

    void initEnv() {
      char confFilePath[1024];
      const char * userDir = userHomeDir();
      if (userDir) {
        sprintf(confFilePath, "%s/.nccl.conf", userDir);
        setEnvFile(confFilePath);
      }
      sprintf(confFilePath, "/etc/nccl.conf");
      setEnvFile(confFilePath);
    }
    ```

* trim system

    * nccl 中 node 中 path 的 dst node 与 system 中 path 的 dst node 是共享的，其中一个消失，另一个也会跟着消失。siccl 中 path 中的 edge list 是独占的，与 node 的变化无关，所以仅删除 node 无法影响到 path。

        推测：nccl 中 node 被删除，仅仅是 path 的 dst node 被删除，中间节点（intermediate note）依然存在，因此需要重新 compute path。

    * 目前适配 trim system 的结果没有问题，但是不清楚。后续可以看一下。

* `ncclTopoCompareGraphs()`

    ```cpp
    ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
    ```

    对比`graph`与`refGraph`哪个更好一点。

    1. 如果`graph`的`nChannels * bwIntra` (bandwidth) 比`refGraph`大，那么将`copy`置 1.

    2. 如果 bw 相等，那么如果`graph`的中转节点（`nHops`）比较少，那么将`copy`置 1.

    当`copy`被置 1 时，下一步马上将`graph`复制到`saveGraph`中。

    再接着，如果 channel 数搜索足够，那么将搜索时间设置为 -1，准备停止搜索：

    ```cpp
    if (graph->nChannels == graph->maxChannels)
        *time = -1;
    ```

    那么问题来了，什么是 channel？从上面代码看，nchannels 能和 bwIntra 相乘，说明每一条 channel 都是一条 intra 通路。那么，inter 通路算一条 channel 吗？这里的 bw intra，是所有 intra 通路中，bw 的最小值，还是统一值，还是平均值？

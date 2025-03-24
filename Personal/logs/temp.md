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


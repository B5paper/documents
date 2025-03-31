# Interview Questions

## cache

* 使用 c++ template partial specialization 实现 reduce

    ```cpp
    #include <iostream>
    using namespace std;

    template<typename T, int NumElm>
    struct GetSum
    {
        static T get_sum(T *arr)
        {
            return GetSum<T, NumElm / 2>::get_sum(arr) + GetSum<T, NumElm - NumElm / 2>::get_sum(arr + NumElm / 2);
        }
    };

    template<typename T>
    struct GetSum<T, 1>
    {
        static T get_sum(T *arr)
        {
            return *arr;
        }
    };

    int main()
    {
        using type = float;
        type arr[] = {1, 2, 3, 4, 5,};
        const int arr_len = sizeof(arr) / sizeof(type);
        type reduce_sum = GetSum<type, arr_len>::get_sum(arr);
        cout << "reduce sum: " << reduce_sum << endl;
        return 0;
    }
    ```

## nccl

1. 之前接触过 nccl 吗？看过源码吗？

2. c++ 模板：为什么 NCCL 中要这么写？

    ```cpp
    template<int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
    __device__ void genericOp() {}
    ```

3. `#pragma unroll Unroll`这个有什么用？为什么 Unroll 为 4？

4. 为什么 nccl 按 chunk, slice 来处理数据？有什么好处？ (ring 处理数据方便)

5. 为什么总是按 warp 来处理数据？bank conflict 听说过吗？是什么原因引起的，怎么避免？

6. 如果我想在一个 warp 里交换数据，又不想造成 bank conflict，有什么函数可以调用？

7. 异步事件相关的东西了解过吗？比如`poll()`之类？简述一下如何处理 socket fd 的？

8. 除了用异步事件通知数据传输完成，还有什么方式？可以主动 poll 数据吗？类似 rdma？

9. cuda 里处理的地址是 va 还是 pa？一个 device 可以直接处理另一个 device 中的地址吗？

10. device 访问 host mem 有哪些方式？（dma, pci controller）分别有什么好处坏处？NCCL 中的 LL 协议，LL 128 协议了解过吗？

11. rdma 相关的问题，app 怎么写？驱动接触过吗？查表怎么查，dma 引擎怎么配？

    假如现在功能是通的，标准 100Gb/s，但是只跑了 10 Gb/s，问题可能出在哪（mmap）

12. 集合通信的题目，cuda 的题目

13. 他的项目，流量是如何均衡的？
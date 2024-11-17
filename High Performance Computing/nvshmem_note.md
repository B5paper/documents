# NVSHMEM Note

## cache

* nvshmem 指定 transport 为 ucx 后不报错，但是看起来跑的结果也不对：

    ```
    huliucheng@zjxj:~/Documents/Projects/nvshmem_src_3.0.6-4/build/examples$ LD_LIBRARY_PATH=/usr/local/nvshmem/lib /home/huliucheng/Documents/Projects/openmpi-5.0.5/build/bin/oshrun -np 4 -x NVSHMEM_REMOTE_TRANSPORT=ucx ./on-stream 
    [0 of 1] run complete 
    [0 of 1] run complete 
    [0 of 1] run complete 
    [0 of 1] run complete
    ```

    ```
    huliucheng@zjxj:~/Documents/Projects/nvshmem_src_3.0.6-4/build/examples$ LD_LIBRARY_PATH=/usr/local/nvshmem/lib /home/huliucheng/Documents/Projects/openmpi-5.0.5/build/bin/mpirun -np 4 -x NVSHMEM_REMOTE_TRANSPORT=ucx ./dev-guide-ring
    0: received message 0
    0: received message 0
    0: received message 0
    0: received message 0
    ```

    正常的情况应该是 pe 编号分别为 0, 1, 2, 3，但是这个里面是全 0。第二个 example 里 recv msg 也全是 0，显然是有问题的。

* nvshmem 相关的项目有一些，但比较少，说明 nvshmem 并不是必选项，凡是能用 nvshmem 实现的，一定可以用其他方式实现

    nccl 与 nvshmem 是两套不同的通信机制。

    听说 nvshmem 的小包通信速率快，但是还未实际写程序验证过。

    目前看来，nv 主流的单机多卡通信方式仍然是 nccl + nvlink + p2p。

## note
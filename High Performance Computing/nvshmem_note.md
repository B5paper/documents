# NVSHMEM Note

## cache

* nvshmem 相关的项目有一些，但比较少，说明 nvshmem 并不是必选项，凡是能用 nvshmem 实现的，一定可以用其他方式实现

    nccl 与 nvshmem 是两套不同的通信机制。

    听说 nvshmem 的小包通信速率快，但是还未实际写程序验证过。

    目前看来，nv 主流的单机多卡通信方式仍然是 nccl + nvlink + p2p。

## note
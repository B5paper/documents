* get topo system output

    ```
    sipu nodes: num: 4
        0: sipu 1114112, chip id 0 uuid 43e0e946-3fc8-4df5-b042-a94000022010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> pci 65536
        1: sipu 1179648, chip id 1 uuid ac9b64ea-7000-4cef-a015-57c000024010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> pci 65552
        2: sipu 1245184, chip id 2 uuid 69131c28-f973-43f4-8548-0c0000026010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> pci 65568
        3: sipu 1310720, chip id 3 uuid 7f434194-03a5-4e91-993d-d0c000028010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> pci 65584
    pci nodes: num: 5
        0: pci 65536, bdf: 0000:10:00.0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1114112
            1: -- LINK_PCI, bw: 48.0 --> cpu 0
        1: pci 4096, bdf: 0000:01:00.0
            0: -- LINK_PCI, bw: 48.0 --> nic 20480
            1: -- LINK_PCI, bw: 48.0 --> nic 24576
            2: -- LINK_PCI, bw: 0.2 --> cpu 0
        2: pci 65552, bdf: 0000:10:01.0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1179648
            1: -- LINK_PCI, bw: 48.0 --> cpu 0
        3: pci 65568, bdf: 0000:10:02.0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1245184
            1: -- LINK_PCI, bw: 48.0 --> cpu 0
        4: pci 65584, bdf: 0000:10:03.0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1310720
            1: -- LINK_PCI, bw: 48.0 --> cpu 0
    eth_switch nodes: num: 1
        0: eth_switch 0
            0: -- LINK_SILINK, bw: 400.0 --> sipu 1114112
            1: -- LINK_SILINK, bw: 400.0 --> sipu 1179648
            2: -- LINK_SILINK, bw: 400.0 --> sipu 1245184
            3: -- LINK_SILINK, bw: 400.0 --> sipu 1310720
    cpu nodes: num: 1
        0: cpu 0
            0: -- LINK_PCI, bw: 48.0 --> pci 65536
            1: -- LINK_PCI, bw: 48.0 --> pci 65552
            2: -- LINK_PCI, bw: 48.0 --> pci 65568
            3: -- LINK_PCI, bw: 48.0 --> pci 65584
            4: -- LINK_PCI, bw: 0.2 --> pci 4096
    nic nodes: num: 2
        0: nic 20480
            0: -- LINK_NET, bw: 50.0 --> net 0
            1: -- LINK_PCI, bw: 48.0 --> pci 4096
        1: nic 24576
            0: -- LINK_NET, bw: 50.0 --> net 1
            1: -- LINK_PCI, bw: 48.0 --> pci 4096
    net nodes: num: 2
        0: net 0
            0: -- LINK_NET, bw: 50.0 --> nic 20480
        1: net 1
            0: -- LINK_NET, bw: 50.0 --> nic 24576
    ```

    nic 正常输出。

* compute path output

    这个看起来也没什么问题。

    ```
    sipu, num nodes: 4
        idx 0, id 1114112:
            path_sil: sipu 1114112  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1179648
            path_sil: sipu 1114112  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1245184
            path_sil: sipu 1114112  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1310720
            path_sil: sipu 1114112  --LINK_SILINK-->  eth_switch 0
            path_phb: sipu 1114112  --LINK_PCI-->  pci 65536  --LINK_PCI-->  cpu 0
            path_phb: sipu 1114112  --LINK_PCI-->  pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: sipu 1114112  --LINK_PCI-->  pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 1, id 1179648:
            path_sil: sipu 1179648  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1114112
            path_sil: sipu 1179648  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1245184
            path_sil: sipu 1179648  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1310720
            path_sil: sipu 1179648  --LINK_SILINK-->  eth_switch 0
            path_phb: sipu 1179648  --LINK_PCI-->  pci 65552  --LINK_PCI-->  cpu 0
            path_phb: sipu 1179648  --LINK_PCI-->  pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: sipu 1179648  --LINK_PCI-->  pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 2, id 1245184:
            path_sil: sipu 1245184  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1114112
            path_sil: sipu 1245184  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1179648
            path_sil: sipu 1245184  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1310720
            path_sil: sipu 1245184  --LINK_SILINK-->  eth_switch 0
            path_phb: sipu 1245184  --LINK_PCI-->  pci 65568  --LINK_PCI-->  cpu 0
            path_phb: sipu 1245184  --LINK_PCI-->  pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: sipu 1245184  --LINK_PCI-->  pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 3, id 1310720:
            path_sil: sipu 1310720  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1114112
            path_sil: sipu 1310720  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1179648
            path_sil: sipu 1310720  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1245184
            path_sil: sipu 1310720  --LINK_SILINK-->  eth_switch 0
            path_phb: sipu 1310720  --LINK_PCI-->  pci 65584  --LINK_PCI-->  cpu 0
            path_phb: sipu 1310720  --LINK_PCI-->  pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: sipu 1310720  --LINK_PCI-->  pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
    pci, num nodes: 5
        idx 0, id 65536:
            path_pix: pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: pci 65536  --LINK_PCI-->  cpu 0
            path_phb: pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 1, id 4096:
            path_phb: pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: pci 4096  --LINK_PCI-->  cpu 0
            path_pix: pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_pix: pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 2, id 65552:
            path_phb: pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_pix: pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: pci 65552  --LINK_PCI-->  cpu 0
            path_phb: pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 3, id 65568:
            path_phb: pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_pix: pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: pci 65568  --LINK_PCI-->  cpu 0
            path_phb: pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: pci 65568  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 4, id 65584:
            path_phb: pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_pix: pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: pci 65584  --LINK_PCI-->  cpu 0
            path_phb: pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: pci 65584  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
    eth_switch, num nodes: 1
        idx 0, id 0:
            path_sil: eth_switch 0  --LINK_SILINK-->  sipu 1114112
            path_sil: eth_switch 0  --LINK_SILINK-->  sipu 1179648
            path_sil: eth_switch 0  --LINK_SILINK-->  sipu 1245184
            path_sil: eth_switch 0  --LINK_SILINK-->  sipu 1310720
    cpu, num nodes: 1
        idx 0, id 0:
            path_phb: cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_phb: cpu 0  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
    nic, num nodes: 2
        idx 0, id 20480:
            path_phb: nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0
            path_loc: nic 20480  --LINK_NET-->  net 0
            path_pix: nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 1, id 24576:
            path_phb: nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0
            path_pix: nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
            path_loc: nic 24576  --LINK_NET-->  net 1
    net, num nodes: 2
        idx 0, id 0:
            path_phb: net 0  --LINK_NET-->  nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: net 0  --LINK_NET-->  nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: net 0  --LINK_NET-->  nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: net 0  --LINK_NET-->  nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: net 0  --LINK_NET-->  nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0
            path_pix: net 0  --LINK_NET-->  nic 20480  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 24576  --LINK_NET-->  net 1
        idx 1, id 1:
            path_phb: net 1  --LINK_NET-->  nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: net 1  --LINK_NET-->  nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: net 1  --LINK_NET-->  nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65568  --LINK_PCI-->  sipu 1245184
            path_phb: net 1  --LINK_NET-->  nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65584  --LINK_PCI-->  sipu 1310720
            path_phb: net 1  --LINK_NET-->  nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  cpu 0
            path_pix: net 1  --LINK_NET-->  nic 24576  --LINK_PCI-->  pci 4096  --LINK_PCI-->  nic 20480  --LINK_NET-->  net 0
    ```

* [ ] 文档增加 eth switch topo <-> smi <-> fm 数据交互的文档

* [ ] 2025 工作任务增加 sio_ncclXXX 兼容层 部分

* `./allreduce 0 1 2 3`

    似乎无法复现之前的 bug 了。

    output:

    ```
    set intra[30] to 0, nChannels: 7, ngpus: 4, step: 2
    graph->nChannels--: 6
    graph->nChannels--: 5
    graph->nChannels--: 4
    graph->nChannels--: 3
    graph->nChannels--: 2
    graph->nChannels--: 1
    graph->nChannels--: 0
    End initialize
    Allocating user buffers
    Rank 0: sendbuff=0xc007ffff3000, recvbuff=0xc007fffef000
    Rank 1: sendbuff=0xc007ffff3000, recvbuff=0xc007fffef000
    Rank 2: sendbuff=0xc007ffff3000, recvbuff=0xc007fffef000
    Rank 3: sendbuff=0xc007ffff3000, recvbuff=0xc007fffef000
    Verifying allreduce results
    AllReduce verification passed!
    Finalizing NCCL
    Success 
    [2026-03-06 15:09:12:932570][ArchModel] threading on: 0
    [2026-03-06 15:09:12:932655][ArchModel] thread num: 0
    [2026-03-06 15:09:12:932662][ArchModel] step cnt: 10470459
    [2026-03-06 15:09:12:932665][ArchModel] time elapsed: 11.0532s
    [2026-03-06 15:09:12:932776][ArchModel] simulation speed: 947281
    ```

* [ ] 
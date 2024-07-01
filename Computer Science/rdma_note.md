# RDMA Note

## cache

* 一开始一直 rdma read/write 不成功，主要是因为 send wr 的 opcode 没写对

* 在调用`getopt()`后，rdma qp 就无法创建成功了

* 即使 get cq event 也不代表数据发送完了

    在 get cq event 后突然结束程序，对方仍然有可能没有收到数据

* rdma server 有 server cm id，还有 client cm id

    rdma client 只有一个 cm id

* 如果一方没有 post recv wr，另一方就直接 post send wr，那么发送的数据会丢失，而且 send 方还会返回 cq event

* 不`rdma_connect()`就无法 post send，但是可以 post recv

* 不处理 cm event 也无法 post send

* 如果一个 mr 有 remote 操作，那么它的内存必须在堆上申请

* metadata 只需要 local write access 就可以了

    真正的 buffer 才需要 remote read/write access

* `ibv_req_notify_cq()`是必须的，不然真不通知了

* 启动 rdma 需要配置的内核和环境

    ```bash
    sudo modprobe ib_core
    sudo modprobe rdma_ucm
    sudo modprobe siw
    sudo rdma link add siw0 type siw netdev enp0s3
    sudo rdma link add siw0_lo type siw netdev lo
    ```

    可以使用`ibv_devices`命令列出可用的 ib device。

* rdma 会做一部分的内存管理，通常 get xxx event 的时候会内部申请内存，在 ack xxx event 的时候会释放内存

* sge 保存本地的一块内存地址，sge 随 wr 发送出去后，如果是 recv wr，remote 会将数据写到 sge 里。如果是 send wr，会把 sge 指定内存的数据发送到 remote

* `ibv_get_cq_event()`会从 completion channel 中 pick 一个 active cq

    这个看起来有点像 epoll

* reap v. 获得

* wr 只能操作 mr 中的数据，或者说 wr 操作的对象是 mr

* qp 放在 client cm id 中，但是 cq 不在

* `struct sockaddr`是 16 个字节，前 2 个字节表示 socket 类型（family），后面 14 个字节是具体地址信息，但是没有被定义

    `struct sockaddr_in`专指 internet 的 socket，同样是 16 个字节，但是只用到了前 8 个字节，后面 8 个字节用 0 补齐。前 8 个字节里，前 2 个字节是 socket 类型（family），接下来 2 个字节是 port，再接下来的 4 个字节是 addr。

    `sockaddr_in`的 family 类型通常是`AF_INET`，表示互联网的 socket 类型。

* `rdma_create_id()`会分配`struct rdma_cm_id`的内存，所以只返回一个指针。

* `rdma_buffer_alloc()`不是一个内置函数

    只有`ibv_reg_mr()`是 ib 的内置函数，具体的内存需要自己 malloc 管理。


* ib 完全没有 mac 的概念，roce 底层是 mac，在 mac 上又模拟了一套 rdma 的接口，ethernet 则完全走 tcp/ip 这套协议

* rdma server end flow

    1. `rdma_create_event_channel()`

        cm event channel

    1. `rdma_create_id()`

        server cm id

    1. `rdma_bind_addr()`

        bind server cm id with socket addr

    1. `rdma_listen()`

        specify server cm id and max client numbers

        不会在这里阻塞。

    1. `rdma_get_cm_event()`

        在这里阻塞，等待 client 连接事件的发生。

    1. `rdma_ack_cm_event()`

        客户端确认处理完 cm event。

        这个 ack 信息会发给 client 吗？

    1. `ibv_alloc_pd()`

        从这里开始，准备 server 端的缓冲区，用于接收 client 端的数据

    1. `ibv_create_comp_channel()`

        创建 cq channel，从 cq channel 中才能创建 cq

    1. `ibv_create_cq()`

        这里会指定一些 capacity 等属性，从 cq channel 中创建一个 cq。

    1. `ibv_req_notify_cq()`

        this function requests a notification when the next Work Completion of a requested type is added to the CQ.

    1. `rdma_create_qp()`

        创建 queue pair。

    1. `rdma_buffer_register()`

        把用 malloc 申请的内存注册到 pd 中，pd 会返回一个 mr。

        这里注册的是 receive buffer。

        目前看起来，这里接收的是 client metadata attr。

    1. `ibv_post_recv()`

        等待 client 发送消息。这里 client 会把 clinet 端准备的 buffer 信息发过来。

        post a linked list of wr to the receive queue.

        这个看起来是把一个 wr 放到 recv queue 里，如果任务完成，那么就在 cq 里生成一个事件？

    1. `rdma_accept()`

        正式接收 clinet 的 connection。

    1. `rdma_get_cm_event()`

        不清楚前面已经有了`rdma_accept()`，为啥这里还有 event。

        可能是为了等待 client 的 ack？

    1. `rdma_ack_cm_event()`

        client 发送一个`RDMA_CM_EVENT_ESTABLISHED` event 过来，这里 server 再返回一个 ack。

        到此为止，连接就算正式建立了。

        不过，ack 等于发送消息吗？还是说只需要确定处理事件就可以了？

    1. `ibv_get_cq_event()`

        建立连接后第一件事，等待 client 发送他的 metadata info。

        前面调用了`ibv_post_recv()`，这里拿到第一条收到的消息。

        这个消息应该是 client 要求 server 端准备的内存的需求信息。

    1. `ibv_req_notify_cq()`

        尝试处理更多的消息。

    1. `ibv_poll_cq()`

        应该是从 cq 中 pop 出来一些 item。

    1. `ibv_ack_cq_events()`

        确认。

    1. `rdma_buffer_alloc()`

        从 pd 中申请 mr。

        这里写了三个权限`IBV_ACCESS_LOCAL_WRITE`, `IBV_ACCESS_REMOTE_READ`, `IBV_ACCESS_REMOTE_WRITE`.

        看起来 pd 有两种使用方式，一种是自己申请内存，然后 register 到 pd 中，另一种就是直接调用`rdma_buffer_alloc()`，从 pd 中申请内存。

    1. `rdma_buffer_register()`

        将 server end buffer info 注册到 pd 中，过一会要把这个发给 client。

        这个信息叫做 server metadata。

    1. `ibv_post_send()`

        将 send 的 wr 发送到 send queue 中。

        这里发送的是 server metadata。

    1. `process_work_completion_events()`

        看起来`ibv_req_notify_cq()`只会 notify 一次？

        这个函数主要调用了

        `ibv_get_cq_event()`

        `ibv_req_notify_cq()`

        `ibv_poll_cq()`

        `ibv_ack_cq_events()`

        这些函数。

* 基于 iwarp 的 rdma 编程总结

    * pd (protection domain) 是一个内存保护区域，我们先手动 malloc 内存，然后把这块内存**注册**到这个 pd 里，这样就可以防止其他 channel 使用这块内存

    * 当需要使用内存时，再从 pd 中申请内存，指定内存的长度和读写权限，pd 会返回一个内存地址。

    * rdma 的异步机制是使用 3 个 queue，第一个 queue 是 cq (completion queue)，用于记录发生了什么事件，第二个、第三个 queue 分别是 send queue 和 receive queue，组成一个 qp (queue pair)，作为收发消息的缓冲区。

        这样我们先阻塞等待 cq 中产生事件，有了事件我们就去对应的缓冲区中处理数据就可以了。

    * post send 和 post receive 用于刚建立连接后，马上交换双方的缓冲区信息，知道双方接收数据的能力以及缓冲区地址等等。

* 有关 rdma 的资料调研

    * 三种主流 rdma 协议对比：

        * Infiniband
        
            支持 RDMA 的新一代网络协议。网卡（NIC）和交换机是定制的，贵。使用 udp 传输数据。在链路层上做了改动，从而保证了 udp 的可靠性，效率高。

        * RoCE
        
            一个允许在以太网上执行 RDMA 的网络协议。其较低的网络标头是以太网标头，其较高的网络标头（包括数据）是InfiniBand标头。 这支持在标准以太网基础设施（交换机）上使用RDMA。 只有网卡应该是特殊的，支持RoCE。

        * iWARP
        
            一个允许在 TCP 上执行 RDMA 的网络协议。 IB和RoCE中存在的功能在iWARP中不受支持。 这支持在标准以太网基础设施（交换机）上使用RDMA。 只有网卡应该是特殊的，并且支持iWARP（如果使用CPU卸载），否则所有iWARP堆栈都可以在SW中实现，并且丧失了大部分RDMA性能优势。

    * verbs 指的是 rdma 软件和硬件的接口

    * IB 代表厂商：Mellanox 40Gbps

    * iWARP 代表厂商：Chelsio 10Gbps

    * Mellanox 40Gbps, Emulex 10/40Gbps

    * RNIC: 指的是支持 rdma 的网卡

    * user 交互的对象与交互方法

        * connections

            user 通过 connection management 与之交互

        * queues

            user 通过`send`和`recv`与之交互。

        * keys

            user 通过 node, lkey, rkey, addr 与之交互。

        * memory space

            user 通过 memory management 与之交互

    * rdma 中有 cached page table entry (cached ptes)，页表用于将虚拟页面映射到相应的物理页面

* rdma client code note

    * `clinet_prepare_connection()`

        * `rdma_create_event_channel()`

            内置函数。为了创建 cm event channel.

        * `rdma_create_id()`

            内置函数。创建 cm id。

        * `rdma_resolve_addr()`

            内置函数。绑定 cm id 和 local device。

            那这个和`rdma_bind_addr()`有什么不一样？

            > The `rdma_resolve_addr()` function resolves the destination and optional source address from an IP address to an Remote Direct Memory Access (RDMA) address. If successful, the specified `rdma_cm_id` identifer is associated with a local device.

            这介函数根据 ip address 去找 remote rdma address，如果找到，那么将 cm id 绑定到 local device 上。

            如果 source address 没有被指定，只指定了 target address，那么 cm id 绑定的是一个 local device；如果 source address 被指定，那么 cm id 绑定的是 source address，此时的行为和`rdma_bind_addr()`相同。

        * `rdma_get_cm_event()`

            内置函数。这里已经用到前面创建的 channel 了。

        * 为什么在`process_rdma_cm_event()`中已经 ack event 了，出了函数又 ack 一次。

        * `rdma_resolve_route()`

            我的理解是先 ping 一下？

        * cm channel 不等同于 completion channel

            前者是用于建立连接的，后面是用于收发消息的。

            completion channel 中建立的 queue pair 才是 cq.

        * `ibv_alloc_pd()`

            内置函数。

        * `ibv_create_comp_channel()`

            内置函数。

        * `ibv_create_cq()`

            内置函数。

        * `ibv_req_notify_cq()`

            内置函数。

        * `rdma_create_qp()`

            内置函数，创建 queue pair。

        * `ibv_post_recv()`

            看来 clint 端在这里阻塞，server 端进行`ibv_post_send()`后才开继续执行程序。

    * `client_connect_to_server()`

        * `rdma_connect()`

            内置函数。根据 cm id 建立连接，建立完连接后，会填充一个`rdma_conn_param`的结构体。

    * `client_xchange_metadata_with_server()`

        `rdma_buffer_register`

    * `ibv_post_send()`发送的都是 wr


* rdma example note

    * `ibv_alloc_pd()`

        有关 PD 的描述：

        > Protection Domain (PD) is similar to a "process abstraction" in the operating system. All resources are tied to a particular PD. And accessing resources across PD will result in a protection fault.

        这个函数目前用于申请一个 PD 设备，这个 PD 设备就是 remote client 的 rdma device。

    * `ibv_create_comp_channel()`

        completion channel，主要用于异步 IO 判断是否有事件完成。

        > this is different from connection management (CM) event notifications.

        completion channel 同样会和一个 rdma device 相连。

    * `ibv_create_cq()`

        completion queue (CQ) 看来和 completion channel 还不太一样。

        这里用于存放 I/O completion 的 metadata

        metadata 其实是一个`ibv_wc`类型的结构体，wc 指的是 work completion

        其中 work 指的是 an I/O request in RDMA.

        `cm_client_id->verbs`指的就是 rdma device。用 verbs 来指代一个 device，有点奇怪。不明白为什么。

        cq 的创建依赖于 completion channel。

    * `ibv_req_notify_cq()`

        这其实相当于一个 filter 了，从 queue 中拿出指定的消息类型

        不过这个时候 cq 里应该是空的，这时能拿出什么东西呢？

    * `bzero()`

        看起来是给结构体置 0 的。

    * `rdma_create_qp()`

        注意这里开始用 rdma 开头了，不再使用 ibv 开头了。

        recv_cq 和 send_cq 都在同一个 cq 里。

        这里和 cq 一样，同样需要指定 pd

        `rdma_create_qp()`会填充`cm_client_id->qp`字段作为返回值。

    * `start_rdma_server()`

        * `rdma_create_event_channel()`

            这个说是也是一个 async comm event channel，但是不清楚这个和 cq 有什么区别。

        * `rdma_create_id()`

            `rdma_cm_id`的作用类似于一个 socket fd

        * `rdma_bind_addr()`

            不清楚为啥 rdma 的 addr 用的仍是 socket addr

        * `rdma_listen()`

            每当来一个新连接，cm event 会被发送到 cm event channel.

        * `process_rdma_cm_event()`

            这个就是我们自定义的函数了，用来 client 申请连接的事件。

        * `rdma_ack_cm_event()`

            给 client 应答。acknowledge

    * `accept_client_connection()`

        看注释这个函数也是接收 rdma client connection，不清楚和 cm 那个有啥不一样。

        * `rdma_buffer_register()`

            这个看起来更像是自己申请一块内存，然后放到 pd 里？

            确实是。

        * `ibv_post_recv()`

            看起来是先把要接收到 client 放到 qp 中，然后等 client 的消息来了就自动处理？

        * `rdma_accept()`

            不清楚 outstanding requests 是啥意思。看起来像是处理包的多少？

            似乎到这里，才真正接收了客户端的连接。

        * `process_rdma_cm_event()`

            这是第二次被调用了，看来每有一个 cm 消息就调用一次。

        * `rdma_ack_cm_event()`

            和前面一样。

        * `rdma_get_peer_addr()`

            根据`cm_client_id`就能得到 peer addr。

    * `send_server_metadata_to_client()`

        * `process_work_completion_events()`

            并不是回调函数，是直接被调用的函数。

            * `ibv_get_cq_event()`

                在这里阻塞等待 cq channel 的消息。

                看起来一个 cq channel 中会有很多的 cq。

            * `ibv_req_notify_cq()`

                尝试拿到更多的 notification

                不懂啥意思。

            * `ibv_poll_cq()`

                从 cq 中拿消息进行处理。

    * `rdma_buffer_attr`的常用字段

        * `address`

        * `length`

        * `stag`

            这个`stag`好像说的是权限。

    * `main()`

        server 在哪阻塞？又在哪循环处理消息？

        在`rdma_get_cm_event()`处阻塞接收 clinet 连接。

        这个程序不循环处理消息，只接收一次消息。

        在 server `ibv_post_send()`后，client 才开始发送数据。

    * 大部分`rmda`开头的函数都是作者自定义实现的，只有`ibv_`开头的函数才是内置的。

        也不是，一部分 rdma 开头的函数是作者自己写的，另一部分是内置的。

* rdma server source file note

    * `setup_client_resources()`

        预先配置好 client 的信息以及准备一个接收 client credentials 的 buffer。

* rdma-example relatives

    SGE 不知道什么意思     

    rdma 需要用到 port，这里的 port 指的可能是物理上的端口，绑定物理线路。

    `struct rdma_buffer_attr`

    `__attribute((packed))`表示不要让编译器尝试去 pad structure

    stag 不知道是什么意思，为什么要用 uint32_t 来表示？

    cm 与 channel, event type 相关。

    `rdma_event_channel`, `rdm_cm_event_type`, `rdma_cm_event`看起来都是内核中的类型。

    `rdma_buffer_alloc()`看起来像是申请有权限控制的内存。这个有点像 vulkan 里的 memory object。

    `rdma_buffer_free()`，释放内存，不需要多说了。

    看来每一块内存都是从 protection domain 里申请的，不清楚这个 pd 是一种权限机制，还是一个内存池。

    `rdma_buffer_register()`，猜测这个函数的作用是往 pd 里添加一块大内存？

    `rdma_buffer_deregister()`，不懂，可能是 pd 里拿出一块内存？但是这个函数接受一个`ibv_mr *mr`对象，但是`ibv_mr *`类型的对象是`rdma_buffer_alloc()`返回的，说明`rdma_buffer_alloc()`一定发生在`rdma_buffer_register()`之前。这样的话，“在 pd 里申请内存”的说法就不成立。

    `process_work_completion_events()`，这个看起来像是个回调函数，用来处理事件。

    `show_rdms_cmid()`根据`rdma_cm_id *id`显示一些详细信息。

* verbs 相关知识

    IB spec defines mandatory verbs and optional verbs.

    The mandatory verbs should be implemented forcely by CI.

    verbs includes two classes: privileged verbs and user-level verbs.

    privileged users can use all verbs, and user-level users can only use user-levle verbs.

    common used verbs:

    | verbs | mandatory / optional | privilege / user-level |
    | - | - | - |
    | Open HCA | m | p |
    | Query HCA | m | p |
    | Modify HCA Attributes | m | p |
    | Close HCA | m | p |
    | Allocate Protection Domain | m | p |
    | Deallocate Protection Domain | m | p |
    | Create Address Handle | m | u |
    | Modify Address Handle | m | u |
    | Query Address Handle | m | u |
    | Destroy Address Handle | m | u |
    | Create Shared Receive Queue | SRQ | p |
    | Modify Shared Receive Queue | SRQ | p |
    | Query Shared Receive Queue | SRQ | p |
    | Destroy Sahred Receive Queue | SRQ | p |
    | Create Queue Pair | m | p |
    | Modify Queue Pair | m | p |
    | Query Queue Pair | m | p |
    | Destroy Queue Pair | m | p |
    | Get Special QP | m | p |
    | Create Completion Queue | m | p |
    | Query Completion Queue | m | p |
    | Resize Completion Queue | m | p |
    | Destroy Completion Queue | m | p |
    | Allocate L_Key | 基本内存管理扩展 | p |
    | Register Memory Region | m | p |
    | Register Physical Memory Region | m | p |
    | Query Memory Region | m | p |
    | Deregister Memory Region | m | p |
    | Reregister Memory Region | m | p |
    | Reregister Physical Memory Region | m | p |
    | Register Shared Memory Region | m | p |
    | Allocate Memory Window | m | p |
    | Query Memory Window | m | p |
    | Bind Memory Window | m | u |
    | Deallocate Memory Window | m | p |
    | Post Send Request | m | u |
    | Post Receive Request | m | u |
    | Poll for completion | m | u |
    | Request Completion Notification | m | u |
    | Set Completion Event Handler | m | p |
    | Set Asynchronous Event Handler | m | p |

    ref: <https://zhuanlan.zhihu.com/p/114943081>

    这篇知乎文章还有很长，讲得挺详细的，有时间了看看。

* IB relatives

    * InfiniBand 链路层提供有序数据包传递和基于信用的流量控制

    * ref: <https://www.zhihu.com/question/422501188/answer/1488958744>

    * ib 协议栈

        * 物理层：帧。不怎么懂。

        * 链路层（data link layer）：定义数据包，主要是流控，路由选择，编码，解码

        * 网络层（network layer）：在数据包上添加一个 40 字节的全局路由报头（Global Route Header，GRH）来进行路由的选择，对数据进行转发

        * 传输层（transport layer）：主要和 queue pair (QP) 相关

    * queue pair (队列偶)

        QP 包含 send queue (SQ) 和 receive queue (RG) 两部分。

        SQ 中的每条数据被称为 send queue entry (SQE)，receive queue 中每条数据被称为 receive queue entry (RQE)。

        QP 底层由 rdma engine 负责传输数据，engine 与 engine 之间通过 fabric 相连。

    * IB 使用 fat tree 进行拓扑连接。但是接线的最高使用率只有一半，其他需要做冗余才能保证高性能。

    * v100 搭载的 nvlink 2 速率是 300GB/s。 pcie 3.0 速率最大是 32GB/s。

    * 常用的 IB 命令

        * `ibv_asyncwatch`：监听 IB 异步事件

        * `ibv_devices`, `ibv_devinfo`：枚举 IB 设备或设备信息

        * `ibstatus`：查询 IB 设备基本状态

        * `ibping`：验证 IB 节点之间的连通性

        * `ibtracert`：跟踪 IB 路径

        * `iblinkinfo`：查看 IB 交换模块的所有端口的连接状态。此命令会将集群内所有的 IB 交换模块都进行列举。

* OFED 相关调研

    * QoS 指的是 Quality of Service，主要是用来保证不佳网络下的传输质量。通常用到的方法是虚拟化。

    * IB 使用 Virtual Lanes (VL) 来实现 QoS。

        虛通道是一些共享一条物理链接的相互分立的逻辑通信链路。每条链路支持最多 15 条标准虚通道和一条管理通道（VL15）。

    * ref: <https://zhuanlan.zhihu.com/p/540388721>

    * 在`linux-rdma` github 用户下，有

        * `rdma-core`

            主力仓库，一直都在更新。

        * `perftest`

            次主力仓库，偶尔更新。

        * `qperf`

            不知道干嘛用的，6、7年都没更新了

        * `opensm`

            三年没更新了

        * `ibsim`

            两年没更新了。

## 常用术语缩写总结

PD 指的是 protection domain

CQ 指的是 completion queue

QP 指的是 queue pair

WR 指的是 work requests

CM 指的是 connection management

MR 指的是 memory region

## mellanox OFED

OFED 指的是 OpenFabrics Enterprise Distribution，这是一组 IB 的开源驱动，除了提供核心驱动，还实现了其他很多辅助功能库。底层实现了 IB 协议，往上又基于 IB 协议开发了别的协议，用于和上层 app 对接，这些 app 有 MPI，NFS 等。

OFED 由 OpenFabrics 维护。

ofed 的代码主要放在<https://github.com/linux-rdma>和<https://git.kernel.org>上。

## OFA

OFA 指的是 OpenFabrics Alliance，OFED 就是由这个组织发布的。

这个组织是专门研究高性能网络的。

## rdma introduction

RoCE v2：通过以太网实现RDMA, distinct from InfiniBand protocol.

RoCE:

拥塞管理：RoCE v2依赖于以太网交换机所支持的数据中心桥接（DCB）特性来有效应对网络拥塞状况。通过启用DCB，RoCE v2能够创建一个无损以太网环境，从而避免因拥塞导致的数据包丢失问题。

拥塞控制：RoCE v2本身并不具备内置的专门解决方案，而是主要依靠底层以太网基础设施所提供的功能来管理和缓解拥塞现象。

注：

1. 新名词 DCB，不知道有啥用。

2. 对于拥塞控制，是否用到了谷歌的 bbr 解决方案？

路由机制：RoCE v2通常采用传统的以太网路由协议进行路由决策，如路由信息协议（RIP）或开放最短路径优先（OSPF）。这意味着RoCE v2网络中的数据传输路径选择是基于这些成熟的标准路由协议实现的。

拓扑结构：RoCE v2普遍应用于标准以太网环境之中，其路由策略的制定和执行受到底层以太网基础设施的制约和影响。这意味着在设计和实施RoCE v2网络时，需要考虑现有的以太网架构，并根据该架构的特点来进行路由优化。

注：

1. 新名词 路由信息协议 RIP，开放最短路径优化 OSPF，有时间了调研一下

2. 看起来 RoCE 没有固定的拓扑结构，需要手动去定义。这一点可能有优化空间。

Ref:

1. <https://zhuanlan.zhihu.com/p/679909155>

2. <https://www.zhihu.com/people/fei-su-fs>
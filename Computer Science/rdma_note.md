# RDMA Note

## cache

* 在使用宏定义一个 ib uverbs ioctl cmd 的 request 和 response 的 struct 时，报错：it is not allowded to use an incomplete struct

    这个原因是在宏里填的参数正好和宏内部定义的 struct 的名字重名。

    这个例子说明最好还是不要用宏。

* 在创建 rdma kernel abi 的时候出错，主要是因为 rdma core 在 python 脚本里使用正则表达式进行匹配开发者定义的 struct 结构体，但是这个正则表达式只能匹配下面形式的：

    ```c
    struct my_struct {
        // some fileds
    };
    ```

    不能匹配这种格式的：

    ```c
    struct my_struct
    {
        // some fileds
    };
    ```

    非常扯。但这就是脚本，正则表达式和宏。

* `ib_uverbs_create_uapi()`逻辑

    1. `struct uverbs_api *uapi = uverbs_alloc_api();`

    2. `uverbs_dev->uapi = uapi;`

* `uverbs_alloc_api()`逻辑

    1. `struct uverbs_api *uapi = kzalloc();`

    2. `INIT_RADIX_TREE(&uapi->radix, GFP_KERNEL);`

        初始化 radix tree。radix tree 比较像一个哈希表，将一个 32 位整数映射到对象的指针上。这个 32 位整数由 user 自己构建，作为 key，映射的指针作为 value。

    3. `uapi_merge_def(uverbs_core_api)`

        这一步是添加 ib core 预设的 method，大部分常用的 verbs 都在这里了。

    4. `uapi_merge_def(ibdev->driver_def)`

        添加自定义的 verbs。可以为空，不影响 ib 的基本功能。

    5. `uapi_finalize_disable(uapi)`

        这一步会把所有属性为`disabled`的 method 从 radix tree 中清除。

    6. `uapi_finalize(uapi)`

* 在写 rdma umd 驱动时，只需要在`CMakeLists.txt`里适合的位置加一行`add_subdirectory(providers/sonc)`就可以了。

    `rdma-core/build/etc/libibverbs.d`文件夹下的`sonc.driver`会自动生成。

* auxiliary device bug

    ```
    [  196.940718] fail to register ib device
    [  196.940719] sonc_ib_aux.rdma: probe of mock_eth.rdma.0 failed with error -1
    ```

    解决方案：

    在 remove ib device 的时候加一个 dealloc ib dev:

    ```c
    void hlc_ib_aux_dev_remove(struct auxiliary_device *adev)
    {
        ib_unregister_device(&hlc_ib_dev->ib_dev);
        ib_dealloc_device(&hlc_ib_dev->ib_dev);
    }
    ```

* ib umd 中有些`ibv_`开头的函数是框架实现好的，有些是需要自己实现的。

* ib umd 中通过`ibv_cmd_xxx()`函数向 kmd 发送 ioctl 命令。

* 在`alloc_ucontext()`中，其函数参数的 udata 中的`inbuf`, `outbuf`指的就是用户自定义数据的起始地址。

    但是其中的`udata->inlen`和`udata->outlen`并不是和 struct 中的数据严格相同的。struct 很有可能是按照 8 字节对齐的。

    这个 struct 在 umd 里构造的时候，就已经是按 8 字节对齐的状态了。即使额外数据只有 1 个 int 值，使用 sizeof 通过减法计算得到的 int 占用的空间也是 8 个字节。

    有时间了做更多的实验，确认一下。

* mlnx sysfs 的创建过程

    * `ib_register_device())`

        这个函数调用了`ib_setup_port_attrs()`

    * `ib_setup_port_attrs()`

        这个函数会创建`"ports"`文件夹，然后对每个 port，调用`setup_port()`创建更多 sysfs 文件

    * `setup_port()`

        在这个函数中会创建 sysfs group

* 重启 ib 驱动

    `service openibd restart`

* 将 port 从 initializing 状态转变成 active 状态

    `service opensmd restart`

* 当 note 1 remote write node 2 的 buffer 时，node 2 无法使用`ibv_poll_cq()`拿到 wc。

    即使在 node 2 使用 post recv wr，也无法感知到 node 1 remote write 的操作。`ibv_poll_cq()`不会返回任何东西。

* remote write 时，只有将 wr 的 flags 设置成`IBV_SEND_SIGNALED`，才能在`ibv_poll_cq()`时收到 wc。

* qp 很有可能是通过 qp num 和 lid 建立连接通路的。

* 很长时间没跑通，最后终于跑通了，是因为 device 设置的不对

* max send wr 设置成 1 才可以成功创建 qp

* 无论是 rxe，还是 siw，都无法在不依赖 rdma_cma 的情况下创建 qp。因为他们没有真正的 qp。

    siw 目前是让 rdma_cma 维护一个 socket，伪装成 qp 的形态。

    在`ibv_modify_qp()`时，rdma cma 内部维护了一套专门针对 siw 设计的 qp attr 和 mask。这套 qp attr 和 mask 与标准的 ibv 的 qp attr 和 mask 不兼容，如果在 rdma cma 外部使用`ibv_modify_qp()`对 siw 的 qp 强行修改，那么可能会有内存读写错误导致系统直接崩溃。

    对于 rxe，不知道是 rdma cma 维护了什么机制，但是肯定也不是真正的 qp。

    只有硬件支持 qp 的网卡，才能使用`ibv_modify_qp()`进行属性修改。比如 mellanox 的网卡。

* rdma tutorial

    ref: <https://github.com/jcxue/RDMA-Tutorial>

    这个代码写得不算很好。

    这份代码使用 socket 交换 metadata 信息，并且使用 socket 做同步。

    这份代码完全没有用到`rdma_create_event_channel()`之类的函数，即没有使用 rdma connection management agent (rdma cma)。

* `struct rdma_id_private`这个是干嘛用的？

* restrack 指的是 resource tracking

* `struct ib_device`是一个实体结构体，不是一个类型占位。

* test:

    `rdma-core` project 中，`libverbs/examples`里有挺多测试程序的源代码。

    可以拿来测试 ib verbs 基本功能。

* `struct ib_device_ops`在`include/rdma/ib_verbs.h`中，里面定义了一些 rdma-core 所需的 callback function

* sq psn

    `sq_psn`: a 24 bits value of the Packet Sequence Number of the sent packets for any QP.

* rdma qp type

    In RDMA, there are several QP types. They can be represented by: XY

    X can be:

    **Reliable**: There is a guarantee that messages are delivered at most once, in order and without corruption.

    **Unreliable**: There isn't any guarantee that the messages will be delivered or about the order of the packets.

    In RDMA, every packet has a CRC and corrupted packets are being dropped (for any transport type). The Reliability of a QP transport type refers to the whole message reliability.

    Y can be:

    **Connected**: one QP send/receive with exactly one QP

    **Unconnected**: one QP send/receive with any QP

    因此 qp type 其实被分为 RC, UC, UD (Unreliable Datagram) 这三种。

    Ref: <https://www.rdmamojo.com/2013/06/01/which-queue-pair-type-to-use/>

* 一开始一直 rdma read/write 不成功，主要是因为 send wr 的 opcode 没写对

* 在调用`getopt()`后，rdma qp 就无法创建成功了



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

* rts 指的是 read to send

    qps 指的是 queue pair state

    rtr: ready to receive state

    sqd: send queue drain state

    sqe: send queue error state

    err: error state

    drain: 耗尽

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

## app

### cache

* 即使 get cq event 也不代表数据发送完了

    在 get cq event 后突然结束程序，对方仍然有可能没有收到数据

* rdma server 有 server cm id，还有 client cm id

    rdma client 只有一个 cm id

* 如果一方没有 post recv wr，另一方就直接 post send wr，那么发送的数据会丢失，而且 send 方还会返回 cq event

* 不`rdma_connect()`就无法 post send，但是可以 post recv

* 不处理 cm event 也无法 post send

* 如果一个 mr 有 remote 操作，那么它的内存必须在堆上申请

* `ibv_create_qp()`

    header file: `#include <infiniband/verbs.h>`

    syntax:

    ```c
    struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
        struct ibv_qp_init_attr *qp_init_attr);

    int ibv_destroy_qp(struct ibv_qp *qp);
    ```

    ```c
    struct ibv_qp_init_attr {
        void *qp_context;
        struct ibv_cq *send_cq;
        struct ibv_cq *recv_cq;
        struct ibv_srq *srq;
        struct ibv_qp_cap cap;
        enum ibv_qp_type qp_type;
        int sq_sig_all;
    };

    struct ibv_qp_cap {
        uint32_t max_send_wr;
        uint32_t max_recv_wr;
        uint32_t max_send_sge;
        uint32_t max_recv_sge;
        uint32_t max_inline_data;
    }
    ```

    Ref: <https://man7.org/linux/man-pages/man3/ibv_create_qp.3.html>

    可以看到`qp_type`是一个`enum ibv_qp_type`的枚举值。

    文档中常用的有三个值：`IBV_QPT_RC`, `IBV_QPT_UC`, `IBV_QPT_UD`

    Ref: <https://www.rdmamojo.com/2012/12/21/ibv_create_qp/>

### note

## kmd

### cache

* port 相关的数据放在`ib_device` -> `ib_port_data`中。

    大部分的数据都是参数，属性，状态相关的。并没有功能相关的数据。

* ib core 是通过下面的代码访问到 port 相关的属性/数据的

    `device->port_data[port_num].pkey_list`

* `ib_setup_device_attrs()`只有修改 device attr 函数指针的功能，并没有创建 sysfs 文件

    猜测：可以先创建 sysfs 文件，然后再修改 attr 函数指针，可以立即生效

* aux driver 是通过 id table 中的 name 与主 module 取得联系。

    证明：

    先执行主 module，aux driver 可以正常运行 init 函数和 exit 函数。

    不运行主 module，直接运行 aux driver，init 函数和 exit 函数不会被执行。

### note

## umd

### cache

* ib umd 里需要填充`static const struct verbs_context_ops mlx5_ctx_common_ops`这个结构体中的函数指针

    这个结构体被框架提供的`verbs_set_ops()`调用。

    `verbs_set_ops()`被`mlx5_set_context()`调用。

    `mlx5_alloc_context()`

    `static const struct verbs_device_ops mlx5_dev_ops = {`

    `PROVIDER_DRIVER(mlx5, mlx5_dev_ops);`

* 一些 ioctl 的返回数据的结构体放在`rdma-core/build/include/kernel-abi/mlx5-abi.h`中

* `ibv_open_device()`实际调用的是`verbs_open_device()`

    open device 是 core 直接实现好的，函数名是`verbs_open_device()`，直接去访问 cdev 文件。

* `ibv_get_device_list()`, `ibv_open_device()`这些函数都属于 umd 的内容

    通过与 sysfs 交互返回信息。

* `ibv_get_device_list()`

    这个函数定义在`rdma-core/libibverbs/device.c`文件中，函数的原型是：

    ```c
    LATEST_SYMVER_FUNC(ibv_get_device_list, 1_1, "IBVERBS_1.1",
            struct ibv_device **,
            int *num)
    {
        // ...
    }
    ```

    可以看到它用宏的方式定义了版本号之类的，直接就是一个 abi 接口。

    这个函数主要调用了`ibverbs_get_device_list()`，其定义在`rdma-core/libibverbs/init.c`文件中，函数原型是：

    ```c
    int ibverbs_get_device_list(struct list_head *device_list);
    ```

    这个函数的代码主要逻辑如下：

    1. 通过`find_sysfs_devs_nl()`去 sysfs 中找 device

        这个过程会获取 driver id

    2. 如果没有找到 device，那么调用`find_sysfs_devs()`继续去 sysfs 中找 device

    3. 调用`try_all_drivers()`去匹配合适的 umd driver

    4. 如果没有匹配成功，那么就调用`load_drivers()`加载所有已知的 driver
    
    5. 最后再次调用`try_all_drivers()`去匹配 driver

* `load_drivers()`的代码逻辑

    `load_drivers()`定义在`rdma-core/libibverbs/dynamic_driver.c`中，函数原型为：

    ```c
    void load_drivers(void);
    ```

    可以看到，它不接收任何参数。说明它只在固定路径找驱动。

    函数逻辑：

    1. 调用`read_config();`，在宏`IBV_CONFIG_DIR`指定的路径读取 config 文件列表

        `IBV_CONFIG_DIR`宏被定义在`rdma-core/build/include/config.h`中。

        默认的路径在`rdma-core/build/etc/libibverbs.d`下，文件列表为：

        ```
        hlc@hlc-VirtualBox:~/Documents/Projects/rdma-core/build/etc/libibverbs.d$ ls
        bnxt_re.driver    hns.driver         mlx5.driver    siw.driver
        cxgb4.driver      ipathverbs.driver  mthca.driver   sonc.driver
        efa.driver        irdma.driver       ocrdma.driver  vmw_pvrdma.driver
        erdma.driver      mana.driver        qedr.driver
        hfi1verbs.driver  mlx4.driver        rxe.driver
        ```

        文件内容也都比较简单：

        ```
        hlc@hlc-VirtualBox:~/Documents/Projects/rdma-core/build/etc/libibverbs.d$ cat mlx5.driver 
        driver /home/hlc/Documents/Projects/rdma-core/build/lib/libmlx5
        hlc@hlc-VirtualBox:~/Documents/Projects/rdma-core/build/etc/libibverbs.d$ cat mana.driver 
        driver /home/hlc/Documents/Projects/rdma-core/build/lib/libmana
        ```

        这些文件都是在`rdma-core`项目 build 的时候自动生成的。

        读取的结果会被放在全局链表`driver_name_list`中。

    2. 遍历 name list，对每个 entry 调用`load_driver()`，使用`dlopen()`加载各家驱动的`.so`文件

        这个函数定义在`rdma-core/libibverbs/dynamic_driver.c`中，目前的定义为

        ```c
        #define VERBS_PROVIDER_SUFFIX "-rdmav34.so"
        ```

        这里会使用宏`VERBS_PROVIDER_SUFFIX`对 so 文件名再做一次修饰。比如`libsonc`会变成`libsonc-rdmav34.so`

        宏定义在`rdma-core/build/include/config.h`中。

* `try_all_drivers()`的代码逻辑

    `try_all_drivers()`定义在`rdma-core/libibverbs/init.c`中，函数原型为

    ```c
    static void try_all_drivers(struct list_head *sysfs_list,
                    struct list_head *device_list,
                    unsigned int *num_devices);
    ```

    函数逻辑：

    1. 遍历所有的 sysfs 入口，对于每个 sysfs entry，都调用`try_drivers()`

    `try_drivers()`定义在`rdma-core/libibverbs/init.c`中，函数原型为：

    ```c
    static struct verbs_device *try_drivers(struct verbs_sysfs_dev *sysfs_dev);
    ```

    代码逻辑：

    1. 遍历所有的 driver，如果 sysfs dev 的 driver id 不为`RDMA_DRIVER_UNKNOWN`，那么先调用`match_driver_id()`去匹配 driver，如果匹配失败，才调用`try_driver()`去匹配

    2. 如果 sysfs dev 的 driver id 为`RDMA_DRIVER_UNKNOWN`，那么在遍历 driver 时，直接通过`try_driver()`进行匹配

    这里的 driver id 是之前在`find_sysfs_devs_nl()`里获取的。

* `try_driver()`的逻辑

    `try_driver()`定义在`dma-core/libibverbs/init.c`中，函数原型为：

    ```c
    static struct verbs_device *try_driver(const struct verbs_device_ops *ops, struct verbs_sysfs_dev *sysfs_dev);
    ```

    函数逻辑：
    
    1. `match_device()`

        查看 driver 和 device 是否匹配，如果匹配失败就直接退出

    2. 如果匹配成功，则调用 umd driver 中的`alloc_device()`为`struct verbs_device *vdev;`分配内存。并将 ops 对接到`vdev`上：

        ```c
        struct verbs_device *vdev;
        vdev = ops->alloc_device(sysfs_dev);
        vdev->ops = ops;
        ```

    后面的事基本不需要我们操心了，我们只需要知道到这里为止，我们的 ops 就能被调用就可以了。

* `ibv_poll_cq()`被定义在`verbs.h`头文件里

    因为它算是 umd，所以这个函数没有在 linux source code 里实现

* 猜测：`ibv_poll_cq()`是非阻塞式的，`ibv_get_cq_event()`是阻塞式的

* rdma-core 中，`libibverbs/examples/devinfo.c`使用的 ib verbs 主要有 2 个：

    * `ibv_get_device_list()`, `ibv_free_device_list()`

    * `ibv_get_device_name()`

    * `ibv_open_device()`, `ibv_close_device()`

    * `ibv_query_device_ex()`

    * `ibv_read_sysfs_file()`

    * `ibv_query_port()`

    * `ibv_query_gid()`

    * `ibv_query_gid_type()`

### note

所有`ibv_`开头的函数都是 ib 的 umd。这点需要和 ib 的 kmd 区分，ib 的 kmd 大部分函数都是`ib_`开头。

ib umd 被分成两部分，一部分是 libibverbs，由第三方组织完成，另一部分是 provider，由硬件厂商实现。这些代码可以在<https://githubp.com/linux-rdma/rdma-core>下载。

可以看到`rdma-core`项目中有一个`libibverbs`文件夹，有一个`providers`文件夹。如果是厂商开发驱动，只需要在`providers`里添加自己的代码就可以了。

#### ioctl cmd

ib umd 通过 ioctl 与 kmd 交换信息。

rdma core 预先定义了许多 ioctl cmd，放在`/usr/include/rdma/ib_user_verbs.h`里，这个文件中有个`enum ib_uverbs_write_cmds`枚举类型，里面有大量的 cmd。

这些枚举类型对应的具体的函数不需要硬件厂商去实现，都是 ib 驱动框架里实现好的。

ioctl 交换信息的设备是一个 cdev 设备，这个设备由 kmd 生成，路径为`/dev/infiniband/uverbs0`。

## rdma cma

RDMA Connection Management Agent 相关。

### cache

* `rdma_cm_id`中的`port_num`指的是 physical port 的编号（不是总数），因为这个赋值发生在`rdma_for_each_port()`中

* `rdma_bind_addr()`里比较重要的函数只有`rdma_bind_addr_dst()`这一个。

* `cma_acquire_dev_by_src_ip()`：

    这个函数先根据 ipv4 算出一个 gid （使用一个非随机的固定算法），然后再对`cma_device`进行枚举。
    
    `cma_device`是一个 list node，每个`cma_device`里都有一个`ib_device`。

    然后对`ib_device`的 port 进行枚举，从 port 拿到 gid 信息，比对是否和前面算出来的 gid 相同。

    如果相同，那么就记录下`ib_device`信息，port 信息。

    似乎是通过 ip 找到 net dev (可能是 ethernet dev)，然后对接到 aux driver，就算是结束了。

    一些重要的函数：

    * `rdma_ip2gid()`

    * `rdma_protocol_roce()`

    * `cma_validate_port()`

    * `cma_bind_sgid_attr()`

    * `cma_attach_to_dev()`

* `rdma_bind_addr_dst()`里比较重要的几个函数

    * `cma_comp_exch()`

    * `cma_check_linklocal()`

    * `cma_translate_addr()`

    * `cma_acquire_dev_by_src_ip()`

    * `cma_dst_addr()`

    * `cma_get_port()`

    * `rdma_restrack_add()`

* initiator depth

    > responder_resources
    >
        > The maximum number of outstanding RDMA read and atomic operations that the local side will accept from the remote side. Applies only to RDMA_PS_TCP. This value must be less than or equal to the local RDMA device attribute max_qp_rd_atom and remote RDMA device attribute max_qp_init_rd_atom. The remote endpoint can adjust this value when accepting the connection. 
        
    > initiator_depth
    >
        > The maximum number of outstanding RDMA read and atomic operations that the local side will have to the remote side. Applies only to RDMA_PS_TCP. This value must be less than or equal to the local RDMA device attribute max_qp_init_rd_atom and remote RDMA device attribute max_qp_rd_atom. The remote endpoint can adjust this value when accepting the connection. 

    ref: <https://linux.die.net/man/3/rdma_connect>

    这两个参数文档资料太少了，可能还得需要从代码或者书本里获取相关的信息。

### note

## bottom anchor

## problem shooting

* `failed status transport retry counter exceeded`

    报这种错就是数据包没发出去，原因可能有很多，比如 port 不是 active 状态，qp num 设置错误，对端机器配置有问题等等。

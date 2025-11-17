* main thread

    * 如果使用 host_1(gpu_1, rdma_1) + host_2(gpu_2, rdma_2) 进行通信，那么 gpu 向 host 写入数据时，使用的是 memcpy 还是 uva？

        如果是 uva，是否可以改成 memcpy？

    * 检测 rdma, 普通网卡设备

    * 解构 transport layer

    * h100 上编译 nccl，先跑通 2 卡、4 卡

    * 写一个 nccl c 语言 app，跑通 2 卡上的 all reduce，要求可以指定卡的索引号（比如 0, 1）和 data buffer 的大小（比如 256K, 4M, 16M 等）

* `vProps`

    ```c
    struct alignas(32) ncclIbNetCommBase {
      ncclNetVDeviceProps_t vProps;
    ```

    ```c
    typedef ncclNetVDeviceProps_v10_t ncclNetVDeviceProps_t;
    ```

    ```c
    typedef struct {
      int ndevs;
      int devs[NCCL_NET_MAX_DEVS_PER_NIC_V10];
    } ncclNetVDeviceProps_v10_t;
    ```

    ```c
    #define NCCL_NET_MAX_DEVS_PER_NIC_V10 4
    ```

    每张网卡最多负责 4 个 gpu，实际负责的 gpu 数由 ndevs 决定？

    `int devs[]`里的 int 有点像 gpu 的 rank 号。

    那这里的`vProps`该如何解释？当前这个网卡负责哪几个 gpu？

* `reqs`

    ```c
    struct alignas(32) ncclIbNetCommBase {
      ncclNetVDeviceProps_t vProps;
      bool isSend;
      struct ncclIbRequest reqs[MAX_REQUESTS];
    ```

    ```c
    #define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
    // We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive

    // Expands to:

    // (32*8)
    ```

    这里的 recvs 是 4 个 recv 队列吗？这里的 4 刚好可以和前面的每个网卡服务最多 4 个 gpu 对应上。有三种可能：

    1. 4 个 recv 队列分别对应 4 个 gpu

    2. 4 个 recv 队列共享 4 个 gpu

    3. 每个 gpu 分配 4 个 recv 队列

    ```c
    struct ncclIbRequest {
      struct ncclIbNetCommBase* base;
      int type;
      struct ncclSocket* sock;
    ```

    `base`是 parent。`type`不清楚是什么意思。

    ```c
    struct ncclSocket {
      int fd;
      int acceptFd;
      int errorRetries;
      union ncclSocketAddress addr;
      volatile uint32_t* abortFlag;
      int asyncFlag;
      enum ncclSocketState state;
      int salen;
      uint64_t magic;
      enum ncclSocketType type;
      int customRetry;
      int finalizeCounter; // Used to keep track of initial handshake for async sockets.
      char finalizeBuffer[sizeof(uint64_t)]; // Used to keep track of initial handshake for async sockets.
    };
    ```

    ```c
    union ncclSocketAddress {
      struct sockaddr sa;
      struct sockaddr_in sin;
      struct sockaddr_in6 sin6;
    };
    ```

    socket 这里比较熟悉了。`errorRetries`是重试的次数？`addr`是普通的 ipv4, ipv6。`abortFlag`这个不清楚是什么意思，`volatile`指的是这个值可能由 device 填写吗？

    `asyncFlag`不清楚什么意思，异步 socket，可能和 poll / epoll 有关系？如果不使用 epoll，支持同步模式吗，比如使用 recv 阻塞等待数据？

    `customRetry`与`errorRetries`有什么区别？error retries 指的是已经尝试过重连的次数，custom retry 指的是最大重连次数？

    ```c
    enum ncclSocketType {
      ncclSocketTypeUnknown = 0,
      ncclSocketTypeBootstrap = 1,
      ncclSocketTypeProxy = 2,
      ncclSocketTypeNetSocket = 3,
      ncclSocketTypeNetIb = 4,
      ncclSocketTypeRasNetwork = 5
    };
    ```

    看起来不同用途的网络都会起一个 socket。

    ```c
    enum ncclSocketState {
      ncclSocketStateNone = 0,
      ncclSocketStateInitialized = 1,
      ncclSocketStateAccepting = 2,
      ncclSocketStateAccepted = 3,
      ncclSocketStateConnecting = 4,
      ncclSocketStateConnectPolling = 5,
      ncclSocketStateConnected = 6,
      ncclSocketStateReady = 7,
      ncclSocketStateTerminating = 8,
      ncclSocketStateClosed = 9,
      ncclSocketStateError = 10,
      ncclSocketStateNum = 11
    };
    ```

    `ncclSocketStateConnectPolling` socket 也支持 polling 吗？

* `devCommSetup()`

    调用栈：

    ```c
    ncclAsyncJobMain()

    ncclCommInitRankFunc()

    initTransportsRank()

    devCommSetup()
    ```

    ```c
    static ncclResult_t devCommSetup(ncclComm_t comm) {
      ncclResult_t ret = ncclSuccess;
      int nRanks = comm->nRanks;
      struct ncclDevCommAndChannels tmpCommAndChans;
      struct ncclDevCommAndChannels *devCommAndChans = NULL;
      struct ncclNvmlCCStatus ccStatus;
      bool ccEnable;
      cudaStream_t deviceStream;

      memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));
      NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);
      NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream), ret, fail);
      ncclCommPushCudaFree(comm, devCommAndChans);
      NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
      ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, deviceStream), ret, fail);

      comm->devComm = &devCommAndChans->comm;
      tmpCommAndChans.comm.rank = comm->rank;
      tmpCommAndChans.comm.nRanks = nRanks;
      tmpCommAndChans.comm.node = comm->node;
      tmpCommAndChans.comm.nNodes = comm->nNodes;
      tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;
      tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;
      for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
        tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
      }
      tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
      tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];
    ```

* `tmpCommAndChans.comm.rankToLocalRank`

    这里的 tmp 的目的是先修改、再写回，防止原数据失效。

    确实，看到了

    ```c
    ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream)
    ```

* buffer sizes

    ```
    [0] =
    524288
    [1] =
    4915200
    [2] =
    4194304
    ```

    ```c
    #define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
    #define NCCL_PROTO_UNDEF -1
    #define NCCL_PROTO_LL 0
    #define NCCL_PROTO_LL128 1
    #define NCCL_PROTO_SIMPLE 2
    ```

* `comm->p2pChunkSize`, `524288`

* `ncclCommPushCudaHostFree()`

    管理CUDA主机内存的释放与NCCL通信的同步。确保在释放CUDA主机内存时，不会与正在进行的NCCL通信操作产生竞争条件。

    example:

    ```c
    // 伪代码示例
    void* cudaHostPtr;
    cudaMallocHost(&cudaHostPtr, size);  // 分配CUDA主机内存

    // 在NCCL通信中使用这个内存
    ncclSend(cudaHostPtr, count, datatype, peer, comm, stream);

    // 当需要释放内存时
    ncclCommPushCudaHostFree(comm, cudaHostPtr);  // 注册待释放的内存
    // 内存会在所有相关的NCCL操作完成后自动释放
    ```

    工作原理

    1. 延迟释放机制

        将需要释放的CUDA主机内存指针加入待释放队列

        NCCL确保所有使用该内存的通信操作完成后再实际释放

    2. 防止use-after-free

        ```cpp
        // 危险情况：
        // 操作1: ncclSend(hostPtr, ...)  // 异步发送
        // 操作2: cudaFreeHost(hostPtr)    // 立即释放 → 可能崩溃

        // 安全做法：
        // 操作1: ncclSend(hostPtr, ...)
        // 操作2: ncclCommPushCudaHostFree(comm, hostPtr)  // 安全延迟释放
        ```

    3. 与通信进度协调

        NCCL在推进通信进度的同时，检查待释放内存

        确认内存不再被任何通信操作使用后执行实际释放

* `devCommSetup`

    下面这段可能和拓扑有关系，有时间研究一下。

    ```c
      for (int c=0; c < MAXCHANNELS; c++) {
        tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
        tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
        tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
        tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
        tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
        tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
        tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

        if (comm->channels[c].ring.userRanks != nullptr) {
          NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, deviceStream), ret, fail);
        }
      }
    ```

    `#define MAXCHANNELS 64`

* `comm->allocP2pNetLLBuffers`

    ```c
      // Initialize num P2P LL buffers for this communicator
      comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;
    ```

    值是 false.

* `allGather3Data`

    allGather3Data是一个跨进程通信的数据结构，用于在所有参与计算的GPU/进程之间收集和交换网络拓扑信息。

    具体作用

        存储网络拓扑信息

            每个rank（进程）将自己的网络拓扑信息存入本地数组

            通过AllGather操作，所有rank都能获取其他rank的拓扑信息

        收集的关键信息包括：

            pattern: 通信模式

            nChannels: 通道数量

            sameChannels: 是否使用相同通道

            bwIntra/bwInter: 节点内/节点间带宽

            typeIntra/typeInter: 节点内/节点间连接类型

            crossNic: 是否跨NIC通信

    ```c
      // AllGather3 - begin
      NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
        allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
        allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
        allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
        allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
        allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
        allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
        allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
      }
    ```

    理论上各个 thread 计算出来的拓扑应该是一样的才对，为什么这里又同步一遍？

* 将不同GPU进程按所属计算节点进行分组，并收集节点的拓扑信息

    * 节点识别与分组

        ```c
        for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
        if (node == comm->nNodes) {
          comm->nNodes++;
          nodesFirstRank[node] = firstRank;
          // 记录每个节点的树形通信模式
          nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
        }
        comm->rankToNode[r] = node;
        ```

        * 通过firstRank（环接收的第一个rank）来识别不同的计算节点

        * 如果发现新节点，增加节点计数并记录该节点的特征

        * 建立rank到节点的映射关系 rankToNode

        这里的 node 指的是 host，每个 node 包含一个或多个GPU（通过PCIe/NVLink连接），firstRank 作为节点的"指纹"：同一个节点上的所有rank会有相同的firstRank值，comm->rankToNode[r] 建立了映射：哪个rank属于哪台物理服务器。

    * 硬件架构检测

        ```c
        if (comm->cpuArch != allGather3Data[r].cpuArch &&
            comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
          comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
        }
        ```

        检测集群中是否混合了不同CPU架构

        如果发现异构架构，标记为混合模式

    应用场景:

    * 跨节点通信需要知道哪些rank在同一个节点内（节点内通信通常更快）

    * 拓扑感知的通信算法需要根据节点边界优化通信模式

    * 异构环境处理需要识别混合硬件配置以选择最优通信策略

* 构建节点内rank映射关系

    * 构建节点内的rank列表

        ```c
        for (int r=0; r<comm->nRanks; r++) {
            int node = comm->rankToNode[r];
            comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
        }
        ```

        * 遍历所有rank，找到每个rank所属的节点

        * 将rank添加到对应节点的localRankToRank数组中

        * localRanks计数器记录每个节点内的rank数量

    * 设置当前进程的节点信息

        ```c
        comm->node = comm->rankToNode[rank];  // 当前rank所属的节点ID
        comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank; // 当前节点的rank映射表
        comm->localRank = comm->rankToLocalRank[rank];  // 当前rank在节点内的局部编号
        comm->localRanks = comm->nodeRanks[comm->node].localRanks;  // 当前节点内的rank总数
        ```

    实际效果示例

    假设有2个节点，每个节点4个GPU：

        Node 0: 全局rank 0,1,2,3

        Node 1: 全局rank 4,5,6,7

    执行后对于rank 5（在Node 1上）：

        comm->node = 1 （属于节点1）

        comm->localRankToRank = [4,5,6,7] （节点1的所有全局rank）

        comm->localRank = 1 （在节点1内的局部编号，因为5是节点1中的第1个rank）

        comm->localRanks = 4 （节点1共有4个rank）

* `ncclTopoNetState`

    ```cpp
    struct ncclTopoNetState {
      int nVirtualNics;
      int nPhysicalNics;
      const char* name;
    };
    ```

    ```cpp
    #define NCCL_NET_MAX_PLUGINS 16
    ```

    找到一个空位，然后插入新的 state。

    ```cpp
    ncclResult_t ncclTopoGetSharedState(ncclTopoNetState** state, const char* name, ncclTopoNetState* states) {
      INFO(NCCL_GRAPH, "Retrieving state for %s", name);
      for (int i = 0; i < NCCL_NET_MAX_PLUGINS; i++) {
        // Empty slot
        if (states[i].name == NULL) {
          states[i].nVirtualNics = -1;
          states[i].nPhysicalNics = -1;
          states[i].name = strdup(name);
          *state = states + i;
          INFO(NCCL_GRAPH, "Initialized state %d for %s", i, name);
          return ncclSuccess;
        // Found my slot
        } else if (strcmp(states[i].name, name) == 0) {
          *state = states + i;
          return ncclSuccess;
        }
      }
      WARN("NET/TOPO : Couldn't find net with name %s", name);
      return ncclInternalError;
    }
    ```

    `state`是个二级指针，表示把这个 entry 再拿出去，在外部进一步修改。

    如果当前 entry 不是空位，那么 name 能对得上也可以。感觉这个函数还可以写成：

    ```cpp
    // 在已有项中搜索
    while (states[i].name) {  // 判断当前 entry 是否为空
        if (strcmp(states[i].name, name) != 0) {
            ++i;
            continue;
        }
        *state = &states[i];
    }

    // 添加一个新项
    *state = &states[i];
    (*state)->name = strdup(name);
    // ...
    ```

* `ncclTopoProcessNet()`

    devices -> ncclIbDevices

* `ncclTopoPopulateNics()`

    ```cpp
    static ncclResult_t ncclTopoPopulateNics(ncclXml* xml, int startIndex, int endIndex, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), const char* netName, int coll, int virtualNics, bool dmaBufSupport) {
      for (int n = startIndex; n < endIndex; n++) {
        ncclNetProperties_t props;
        NCCLCHECK(getProperties(n, &props));
        struct ncclXmlNode* netNode = NULL;
        struct ncclXmlNode* parent = NULL;
        if (virtualNics) {
          struct ncclXmlNode* net = NULL;
          NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
          // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for this vNIC
          // Only run this if the net doesn't exist locally - this may alter the XML state
          if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, getProperties, &props.vProps, &parent));
        }

    ```

    * `endIndex`为`5`.

    * `getProperties`是`ncclIbGetProperties(int, ncclNetProperties_v10_t*)`

    ```cpp
    typedef ncclNetProperties_v10_t ncclNetProperties_t;

    typedef struct {
      char* name;                      // Used mostly for logging.
      char* pciPath;                   // Path to the PCI device in /sys.
      uint64_t guid;                   // Unique identifier for the NIC chip. Important for
                                       // cards with multiple PCI functions (Physical or virtual).
      int ptrSupport;                  // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
      int regIsGlobal;                 // regMr is not tied to a particular comm
      int forceFlush;                  // Force a flush on receives
      int speed;                       // Port speed in Mbps.
      int port;                        // Port number.
      float latency;                   // Network latency
      int maxComms;                    // Maximum number of comms we can create
      int maxRecvs;                    // Maximum number of grouped receives.
      ncclNetDeviceType netDeviceType; // Network offload type
      int netDeviceVersion;            // Version number for network offload
      ncclNetVDeviceProps_v10_t vProps;
      size_t maxP2pBytes;              // Max transfer size for point-to-point operations
      size_t maxCollBytes;             // Max transfer size for collective operations
    } ncclNetProperties_v10_t;
    ```

    这个结构是给 IB 网卡准备的，还是标卡也能用？

    后面有专门的`ncclIbDev`结构，那么这里应该是标卡、IB卡都能用。

    * `virtualNics`为 0

    原文片段 2:

    ```cpp
        NCCLCHECK(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode, parent));

        const char* colAttr;
        NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));

        NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
        int dev;
        xmlGetAttrIntDefault(netNode, "dev", &dev, -1);
        if (dev != -1 && dev != n) INFO(NCCL_GRAPH, "TOPO/NET : Changing %s dev index from %d to %d", netName, dev, n);
        NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
        NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
        NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
        NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
        NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
        NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
        bool gdrSupport = (props.ptrSupport & NCCL_PTR_CUDA) || (dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
        INFO(NCCL_NET,"NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netName, gdrSupport ? "Enabled" : "Disabled", n, props.name);
        NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));
        // Only set coll if it's not 0
        if (coll) NCCLCHECK(xmlInitAttrInt(netNode, "coll", coll));
    ```

    * `dev` -> `0`

        说明这里的 dev 是网卡设备的索引，不是 gpu dev

    * `latency` -> `0`

    * `speed` -> `400000`

    * `port`-> `1`

    * `guid` -> `1012277474802960544`

    * `maxconn` -> `131072`

    * `gdrSupport` -> true

    原⽂片段 3（完）：

    ```cpp
            const char* keepAttr;
            NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
            NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
            INFO(NCCL_GRAPH, "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s keep=%s coll=%s",
              props.name, props.pciPath, keepAttr, colAttr);
          }

          return ncclSuccess;
        }
    ```

    * `keepAttr` -> `"1"`

* `ncclIbGetProperties()`

    ```cpp
    ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
      if (dev >= ncclNMergedIbDevs) {
        WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
        return ncclInvalidUsage;
      }
      struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
      // Take the rest of the properties from an arbitrary sub-device (should be the same)
      NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
      props->name = mergedDev->devName;
      props->speed = mergedDev->speed;
      memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
      return ncclSuccess;
    }
    ```

    * `ncclNMergedIbDevs`为 5，正好和前面的`endIndex`相同。说明 end index 指的是 ib dev。

        不清楚这里的 merged 是什么意思。

* `ncclIbGetPhysProperties()`

    ```cpp
    ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
      struct ncclIbDev* ibDev = ncclIbDevs + dev;
      pthread_mutex_lock(&ibDev->lock);
      props->name = ibDev->devName;
      props->speed = ibDev->speed;
      props->pciPath = ibDev->pciPath;
      props->guid = ibDev->guid;
      props->ptrSupport = NCCL_PTR_HOST;
      if (ncclIbGdrSupport() == ncclSuccess) {
        props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
      }
      props->regIsGlobal = 1;
      if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
        props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
      }
      props->forceFlush = 0;
      if (ibDev->capsProvider.mlx5.dataDirect) {
        props->forceFlush = 1;
      }
      props->latency = 0; // Not set
      props->port = ibDev->portNum + ibDev->realPort;
      props->maxComms = ibDev->maxQp;
      props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
      props->netDeviceType    = NCCL_NET_DEVICE_HOST;
      props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
      props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
      pthread_mutex_unlock(&ibDev->lock);
      return ncclSuccess;
    }
    ```

    * `ncclIbDevs`是一个全局数组，

    ```cpp
    struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];

    #define MAX_IB_DEVS  32
    
    static int ncclNIbDevs = -1;
    struct alignas(64) ncclIbDev {
      pthread_mutex_t lock;
      int device;
      uint64_t guid;
      uint8_t portNum;
      uint8_t link;
      int speed;
      ibv_context* context;
      int pdRefs;
      ibv_pd* pd;
      char devName[MAXNAMESIZE];
      char* pciPath;
      char* virtualPciPath;
      int realPort;
      int maxQp;
      float latency;
      struct ncclIbMrCache mrCache;
      int ar; // ADAPTIVE_ROUTING
      struct ibv_port_attr portAttr;
      struct ncclIbStats stats;
      int dmaBufSupported;
      enum ncclIbProvider ibProvider;
      union {
        struct {
          int dataDirect;
        } mlx5;
      } capsProvider;
    };
    ```

    * 在进入函数时，`ncclIbDevs`中已经有内容了，这个函数明显是把 nccl ib devs 中的内容填到`props`内。

    ```cpp
    #define NCCL_PTR_HOST 0x1
    #define NCCL_PTR_CUDA 0x2
    #define NCCL_PTR_DMABUF 0x4
    ```

    * 不清楚这个`NCCL_PTR_DMABUF`是干嘛用的

    ```cpp
    typedef struct {
      int ndevs;
      int devs[NCCL_NET_MAX_DEVS_PER_NIC_V10];
    } ncclNetVDeviceProps_v10_t;

    #define NCCL_NET_MAX_DEVS_PER_NIC_V10 4
    ```

    * 看来`mergedDev`中的内容也是不准备要了，最终留下的就是这个`ncclNetProperties_t* props`

    * `ndevs`为 1，`devs`中的数据全是 0.

        不清楚这两个是什么意思。

* `ncclTopoFillNet()`

    ```cpp
    ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent) {
      NCCLCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));

      if (*netNode != NULL) return ncclSuccess;

      const char* pciSysPath = pciPath;
      if (pciSysPath) {
        char subSystem[PATH_MAX];
        NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
        // This is not a PCI device (virtual, usb, ...).
        if (strcmp(subSystem, "pci") != 0) {
          INFO(NCCL_NET|NCCL_GRAPH, "Topology detection: network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSystem);
          pciSysPath = NULL;
        }
      }
    ```

    * `pciPath` -> `0x7ffb1f5ca890 "/sys/devices/pci0000:37/0000:37:01.0/0000:38:00.0/0000:39:02.0/0000:3c:00.0"`

    * busid -> `$3 = "0000:3c:00.0\000\000\000"`

        这个地址是 ib 的还是 eth 的？

    原文片段 1（完）：

    ```cpp
      struct ncclXmlNode* parent = NULL;
      if (forceParent) {
        parent = forceParent;
      } else if (pciSysPath) {
        int offset;
        for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        strcpy(busId, pciSysPath+offset+1);
        NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
        NCCLCHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));
        NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
      } else {
        // Virtual NIC, no PCI device, attach to first CPU
        NCCLCHECK(xmlFindTag(xml, "cpu", &parent));
      }

      struct ncclXmlNode* nicNode = NULL;
      NCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
      if (nicNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
      }

      // We know that this net does not exist yet (we searched for it at the
      // beginning of this function), so we can add it.
      NCCLCHECK(xmlAddNode(xml, nicNode, "net", netNode));
      NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
      return ncclSuccess;
    }
    ```

    * 第一次走到这里时，`nicNode`为空。说明整个 xml 只能有一个 nic tag。

    * `netName` -> `0x7ffff7f89118 <ncclIbMergedDevs+24> "mlx5_0"`

* 原始的 nic device 列表从哪得到？
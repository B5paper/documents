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

* `ncclTopoGetSharedState()`

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

    * `state`是个二级指针，表示把这个 entry 再拿出去，在外部进一步修改。

    * 这个 get 函数只是匹配一个对应的 name，比如 "ib"、"ethernet"、或 coll-net ，如果匹配到了，那么返回数据项，如果没有匹配到，那么返回一个空的。

    * 感觉这个函数还可以写成：

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

    相关资料：

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

* `ncclTopoProcessNet()`

    原文片段 1：

    ```cpp
    // Calls to network plugin APIs should be protected. This function should be called inside a per-process lock.
    ncclResult_t ncclTopoProcessNet(ncclXml* xml, int coll, const char* dumpXmlFile, ncclTopoNetState* state, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*devices)(int*), const char* netName, bool dmaBufSupport) {
    ```

    * `coll` = `0`

    * `dumpXmlFile` = NULL

    * `state`是从 shared states 中取出的一个空元素或名字匹配的元素

    * devices -> ncclIbDevices

        ```cpp
        ncclResult_t ncclIbDevices(int* ndev) {
          *ndev = ncclNMergedIbDevs;
          return ncclSuccess;
        }
        ```

        `ncclNMergedIbDevs` -> 5

        说明在调用``ncclTopoProcessNet()`这一步之前，就已经处理好网卡搜索了。

    原文片段 2：

    ```cpp
      int usePhysicalDevices = (dumpXmlFile || makeVDevice == NULL);
      if (state->nPhysicalNics == -1)
        NCCLCHECK(devices(&state->nPhysicalNics));
      // Enumerate physical devices
      NCCLCHECK(ncclTopoPopulateNics(xml, 0, state->nPhysicalNics, getProperties, netName, coll, false, dmaBufSupport));
    ```

    * `devices` -> `0x7ffff72a2720 <ncclNetSocketDevices(int*)>`

        实参为`comm->ncclNet->devices`，对应函数为

        ```cpp
        ncclResult_t ncclIbDevices(int* ndev) {
          *ndev = ncclNMergedIbDevs;
          return ncclSuccess;
        }
        ```

    * `makeVDevice`是个函数指针，用于创建虚拟网卡

    * `usePhysicalDevices` -> `1`

    * `comm->ncclNet`从哪来？什么时候填的 name？什么时候填的 devices 函数指针？

    原文片段 3：

    ```cpp
      if (!usePhysicalDevices) {
        if (state->nVirtualNics == -1) {
          NCCLCHECK(ncclTopoMakeVNics(xml, makeVDevice, getProperties, state->nPhysicalNics));
          int nDevs;
          NCCLCHECK(devices(&nDevs));
          state->nVirtualNics = nDevs - state->nPhysicalNics;
        }
        if (state->nVirtualNics > 0) {
          // Populate new devices
          NCCLCHECK(ncclTopoPopulateNics(xml, state->nPhysicalNics, state->nPhysicalNics+state->nVirtualNics, getProperties, netName, coll, true, dmaBufSupport));
        }
      }

      return ncclSuccess;
    }
    ```

* `ncclTopoPopulateNics()`

    原文片段 1：

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

    * `virtualNics` -> `0`

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

* `ncclNetProperties_t`

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

    后面有专门的`ncclIbDev`结构，这里应该是 eth 卡、IB 卡都能用。

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

    * 开头`xmlFindTagKv()`先搜索一遍 xml 中是否已经有 net tag，

        然后通过`if (*netNode != NULL)`判断，如果已经有了 net tag，那么说明之前已经填写过信息，这里不再重复填写，直接返回。

    * `pciPath` -> `0x7ffb1f5ca890 "/sys/devices/pci0000:37/0000:37:01.0/0000:38:00.0/0000:39:02.0/0000:3c:00.0"`

        机器上一张 pci 网卡对应的 bdf 为：`3c:00.0`。

    * `netName` -> `0x7ffff7f89118 <ncclIbMergedDevs+24> "mlx5_0"`

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

    * `// Virtual NIC, no PCI device, attach to first CPU`，这里可以解答之前 xml 里 nic 在 cpu 外面的的疑问了。只是不知道什么时候会创建 virtual nic？

    * `ncclTopoGetXmlFromSys()`中，pci class 为`0x020700`，siccl 得到的 socket nic 的 class 为`0x020000`。

    * `ncclTopoGetXmlFromSys()`会递归找 parent pci，直到找到 cpu 节点为止

* 原始的 nic device 列表从哪得到？

* `ncclNetInit()`

    原文片段 1：

    ```cpp
    ncclResult_t ncclNetInit(struct ncclComm* comm) {
      bool ncclNetPluginInitialized = false;
      pthread_once(&initPluginLibsOnceControl, initPluginLibsOnceFunc);
      pthread_mutex_lock(&netPluginLock);
      for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
        if ((pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) && (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady)) {
          NCCLCHECK(ncclNetPluginLoad(&netPluginLibs[pluginIndex]));
        }
    ```

    * `pluginCount`: 3

        这个是啥意思，三种网络？标卡，IB, CollNet？

* `ncclNetPluginInit()`

    原文片段 1：

    ```cpp
    static ncclResult_t ncclNetPluginInit(netPluginLib_t* pluginLib) {
      int ndev;
      if (pluginLib->ncclNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclNet) {
        if (pluginLib->ncclNet->init(ncclDebugLog, ncclProfilerCallback) != ncclSuccess) goto fail;
        if (pluginLib->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0) goto fail;
      }
      pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;
      INFO(NCCL_INIT|NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);
    ```

    * 经过`pluginLib->ncclNet->init()`调用后，全局变量`ncclIbDevs()`中的内容被填写。

    * `pluginLib->ncclNet->init` -> `0x7fffb3aba3cd <ncclIbInit(void (*)(ncclDebugLogLevel, unsigned long, char const*, int, char const*, ...), ncclResult_t (*)(void**, int, void*, long, void*))>`

* `ncclNetIb`

    file: `net_ib.cc`

    原文：

    ```cpp
    ncclNet_t ncclNetIb = {
      "IB",
      ncclIbInit,
      ncclIbDevices,
      ncclIbGetProperties,
      ncclIbListen,
      ncclIbConnect,
      ncclIbAccept,
      ncclIbRegMr,
      ncclIbRegMrDmaBuf,
      ncclIbDeregMr,
      ncclIbIsend,
      ncclIbIrecv,
      ncclIbIflush,
      ncclIbTest,
      ncclIbCloseSend,
      ncclIbCloseRecv,
      ncclIbCloseListen,
      NULL /* getDeviceMr */,
      NULL /* irecvConsumed */,
      ncclIbMakeVDevice
    };
    ```

    * 所有与 ib 相关的函数都封装在这里面

* 既然调用`ncclIbInit()`就可以检测 ib 网卡，那么 topo system 是否只需要构造 xml tag 和把 xml tag 添加到 topo system 里就可以了？

* `netStates`

    ```cpp
    ncclTopoNetState netStates[NCCL_NET_MAX_PLUGINS] = {};
    ```

* `ncclTopoGetVNicParent()`

    ```cpp
    ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
      ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
      ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
      for (int i = 0; i < vProps->ndevs; i++) {
        NCCLCHECK(getProperties(vProps->devs[i], props + i));
        struct ncclXmlNode* physNetNode;
        NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name));
        physNetNodes[i] = physNetNode;
        TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i,  props[i].name);
      }

    ```

* nic node 在创建时应该清空为 0，防止 union 里有数据

* nccl 里的很多函数和变量的命名有误导性，这进一步增加了逆向的难度

* `xmlFindTagKv()`

    ```cpp
    static ncclResult_t xmlFindTagKv(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node, const char* attrName, const char* attrValue) {
      *node = NULL;
      for (int i=0; i<xml->maxIndex; i++) {
        struct ncclXmlNode* n = xml->nodes+i;
        if (strcmp(n->name, tagName) == 0) {
          const char* value;
          NCCLCHECK(xmlGetAttr(n, attrName, &value));
          if (value && strcmp(value, attrValue) == 0) {
            *node = n;
            return ncclSuccess;
          }
        }
      }
      return ncclSuccess;
    }
    ```

    这里只有线性搜索，有可能是`ncclXml`一次性把所有的 node ptr 都申请完，然后由线性数组 + parent ptr 构建树关系。因此搜索时没有递归。

* `ncclTopoGetSystem()`

    ```cpp
    // ...

      pthread_mutex_lock(&netLock);
      netLockHeld = 1;
      INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
      ncclTopoNetState* state;
      state = NULL;
      if (collNetSupport(comm)) {
        NCCLCHECKGOTO(ncclTopoGetSharedState(&state, comm->ncclCollNet->name, collNetStates), ret, fail);
        NCCLCHECKGOTO(ncclTopoProcessNet(xml, 1, dumpXmlFile, state,
          comm->ncclCollNet->getProperties, comm->ncclCollNet->makeVDevice, comm->ncclCollNet->devices, comm->ncclCollNet->name, comm->dmaBufSupport), ret, fail);
      }
      NCCLCHECKGOTO(ncclTopoGetSharedState(&state, comm->ncclNet->name, netStates), ret, fail);
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, 0, dumpXmlFile, state,
        comm->ncclNet->getProperties, comm->ncclNet->makeVDevice, comm->ncclNet->devices, comm->ncclNet->name, comm->dmaBufSupport), ret, fail);
      pthread_mutex_unlock(&netLock);
      netLockHeld = 0;

    // ...
    ```

    * `comm->ncclNet->makeVDevice` -> `0x7fffb3aba380 <ncclIbMakeVDevice(int*, ncclNetVDeviceProps_v10_t*)>`

        这个只有在 v10 里才有，在 v6 里没有。

* `fill_pci_attrs()`

    pcilink, numa_node

* `ncclTopoGetXmlFromSys()`

    原文片段 1:

    ```cpp
    ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
      // Fill info, then parent
      const char* busId;
      NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
      char* path = NULL;
      ncclDebugNoWarn = NCCL_GRAPH;
      getPciPath(busId, &path);
      ncclDebugNoWarn = 0;
    ```

    注：

    * `busId` -> `"0000:29:00.0"`

        这个就是网卡的 pci bdf

    * `path` -> `"/sys/devices/pci0000:26/0000:26:01.0/0000:27:00.0/0000:28:00.0/0000:29:00.0"`

    原文片段 2：

    ```cpp
      if (path) {
        NCCLCHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
      }
      int index;
      ncclDebugNoWarn = NCCL_GRAPH;
      NCCLCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
      if (index == -1) {
        if (path) ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
      }
      NCCLCHECK(xmlGetAttrIndex(pciNode, "device", &index));
      if (index == -1) {
        if (path) ncclTopoSetAttrFromSys(pciNode, path, "device", "device");
      }
      NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
      if (index == -1) {
        if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor");
      }
      NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
      if (index == -1) {
        if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device");
      }
      ncclDebugNoWarn = 0;
    ```

    注：

    * 这几个其实可以统一写成`set_attr_if_unset()`

    原文片段 3：

    ```cpp
      NCCLCHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
      if (index == -1) {
        if (path) {
          char deviceSpeedStr[MAX_STR_LEN];
          float deviceSpeed = FLT_MAX;
          NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));
          sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
          char portSpeedStr[MAX_STR_LEN];
          float portSpeed = FLT_MAX;
          NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
          sscanf(portSpeedStr, "%f GT/s", &portSpeed);
          NCCLCHECK(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
        } else {
          NCCLCHECK(xmlSetAttr(pciNode, "link_speed", ""));
        }
      }
    ```

    注：

    * 如果 xml 中 pci node 未设置`link_speed`，那么对比`max_link_speed`和`../max_link_speed`，哪个小选哪个

    * 因为`portSpeed`提前设置了`FLT_MAX`，所以如果`../max_link_speed`文件不存在，那么`sscanf(portSpeedStr, "%f GT/s", &portSpeed);`就不会生效，`portSpeed`依然保持`FLT_MAX`，而`max_link_speed`一定小于`FLT_MAX`，因此在`portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr`的比较中，总是会选`deviceSpeedStr`。

        这样就解决了`../max_link_speed`文件可能不存在的问题。

    原文片段 4：

    ```cpp
      NCCLCHECK(xmlGetAttrIndex(pciNode, "link_width", &index));
      if (index == -1) {
        if (path) {
          char strValue[MAX_STR_LEN];
          NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
          int deviceWidth = strtol(strValue, NULL, 0);
          NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
          int portWidth = strtol(strValue, NULL, 0);
          NCCLCHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
        } else {
          NCCLCHECK(xmlSetAttr(pciNode, "link_width", ""));
        }
      }
    ```

    * `link_width`处理方式与`link_speed`完全相同

    原文片段 5：

    ```cpp
      const char* vendor;
      NCCLCHECK(xmlGetAttr(pciNode, "vendor", &vendor));
      if (vendor != NULL && strcmp(vendor, "0x1000") == 0) { // BCM switch, look for P2P connections
        int nlinks;
        char* peers;
        NCCLCHECK(getBcmLinks(busId, &nlinks, &peers));
        for (int l=0; l<nlinks; l++) {
          char* target = peers+l*BUSID_SIZE;
          struct ncclXmlNode* linkNode;
          NCCLCHECK(xmlGetSubKv(pciNode, "pcilink", &linkNode, "target", target));
          if (linkNode == NULL) {
            NCCLCHECK(xmlAddNode(xml, pciNode, "pcilink", &linkNode));
            NCCLCHECK(xmlSetAttr(linkNode, "target", target));
          }
        }
      }
    ```

    * `vendor` -> `"0x15b3"`

    原文片段 6：

    ```cpp
      struct ncclXmlNode* parent = pciNode->parent;
      if (parent == NULL) {
        if (path) {
          // Save that for later in case next step is a CPU
          char numaIdStr[MAX_STR_LEN];
          NCCLCHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));

          // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
          // switch, or stop if we reach a CPU root complex.
          int slashCount = 0;
          int parentOffset;
          for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
            if (path[parentOffset] == '/') {
              slashCount++;
              path[parentOffset] = '\0';
              int start = parentOffset - 1;
              while (start>0 && path[start] != '/') start--;
              // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
              if (checkBDFFormat(path+start+1) == 0) {
                // This a CPU root complex. Create a CPU tag and stop there.
                struct ncclXmlNode* topNode;
                NCCLCHECK(xmlFindTag(xml, "system", &topNode));
                NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
                if (parent == NULL) {
                  NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
                  NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
                  NCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
                }
              } else if (slashCount == 2) {
                // Continue on the upper PCI switch
                for (int i = strlen(path)-1; i>0; i--) {
                  if (path[i] == '/') {
                    NCCLCHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));
                    if (parent == NULL) {
                      NCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                      NCCLCHECK(xmlSetAttr(parent, "busid", path+i+1));
                    }
                    break;
                  }
                }
              }
            }
            if (parent) break;
          }
    ```

    * `numaIdStr` -> `"0"`

    原文片段 7：

    ```cpp
        } else {
          // No information on /sys, attach GPU to unknown CPU
          NCCLCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
          if (parent == NULL) {
            struct ncclXmlNode* topNode;
            NCCLCHECK(xmlFindTag(xml, "system", &topNode));
            NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
            NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
            NCCLCHECK(xmlSetAttr(parent, "numaid", "-1"));
            NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
          }
        }
        pciNode->parent = parent;
        // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
        // Coverity complains about dereferenced parent being NULL
        // but this can never happen.
        // coverity[var_deref_op]
        int subIndex = parent->nSubs;
        const char* newBusId;
        NCCLCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
        for (int s=0; s<parent->nSubs; s++) {
          const char* busId;
          NCCLCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
          if (busId != NULL && strcmp(newBusId, busId) < 0) { subIndex = s; break; }
        }
        if (parent->nSubs == MAX_SUBS) {
          WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
          return ncclInternalError;
        }
        for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s-1];
        parent->subs[subIndex] = pciNode;
        parent->nSubs++;
      }
    ```

    原文片段 8（完）：

    ```cpp
      if (strcmp(parent->name, "pci") == 0) {
        NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
      } else if (strcmp(parent->name, "cpu") == 0) {
        NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
      }
      free(path);
      return ncclSuccess;
    }
    ```

* 真实机器上每个 pci 路径的 numa id 都是正常的，比如 0。virtual box 里所有的 numa id 都是 -1

* print_xml(), find_child_tag() 把改成类成员函数

* 网卡的枚举是否能放到 local res 中完成？

* 应该同时支持 nv gpu 和 sipu 地检测

* 网卡检测的两个问题

    * 在 top 层，不应该去处理上一级目录`../max_link_speed`相关的内容

        nv 那边是如何处理的？

    * 在当前层，如果检测到 unknown，如何处理？

        可以从`/sys/class/net/enp0s2`拿到正确的数据。那么 ifname 从哪来？populate nic 里有网卡的名称吗？

* `ncclTopoPostset()`

    * invoke

        parent: `initTransportsRank()`

        ```cpp
        NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs), ret, fail);
        ```

    * `connectRings()`

* compute path 时，未看到类似 sipu -> rdma -> sipu　的链路

    即使禁用了 p2p，得到的也是 sipu -> pci -> sipu，并不是经过网卡的方案。

* nccl 在 p2p 后的 log 文件，是否和 siccl 相同？

    不同。看到 nccl p2p 后 path_nvl 被设置成了 path_net，可能是开了 NCCL_P2P_DISABLE 环境变量的原因。

    禁用环境变量后恢复正常。siccl 强制 p2p = 0，因此也把 switch 节点删除了。设置 p2p = 1 后，是否可以恢复？

    可以，这时两者完全一致了。

* before return 前再比较一次是否相同

    相同。

* nv 环境中，网卡的 populate 过程是怎样的？

    只能检测到一张 eth 网卡。

* siccl 中`comm->ncclNet->devices`为`ncclNetSocketDevices(int*)`

    nccl 中，为`ncclIbDevices(int*)`。
    
    为什么？

    * nv 中`comm->ncclNet`为`ncclNetIb`，siccl 中为`ncclNetSocket`

* 先按 4 sipu p2p + 1 rdma init 成功，再跑 1 qemu + 4 sipu + 1 rdma，禁用 p2p, shm，强制使用网卡传输数据（或许要先试一把 socket 是否能成功？）

    * [v] 4 sipu p2p + 1 rdma init 成功

        ring_graph.n_channels 需要设置为 2 才行

* siccl output

    为什么 net 1 输出了 2 次？

    ```
    sipu nodes: num: 2
        0: sipu 1114112, chip id 0 uuid 8394d921-b848-45b4-b09f-5cc000022010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> cpu 0
        1: sipu 1179648, chip id 1 uuid ef5a08ec-12cd-43f8-bda0-e10000024010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> cpu 0
    pci nodes: num: 1
        0: pci 4096, bdf: 0000:01:00.0
            0: -- LINK_PCI, bw: 0.0 --> nic 20480
            1: -- LINK_PCI, bw: 0.0 --> nic 24576
            2: -- LINK_PCI, bw: 0.2 --> cpu 0
    eth_switch nodes: num: 1
        0: eth_switch 0
            0: -- LINK_SILINK, bw: 400.0 --> sipu 1114112
            1: -- LINK_SILINK, bw: 400.0 --> sipu 1179648
    cpu nodes: num: 1
        0: cpu 0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1114112
            1: -- LINK_PCI, bw: 0.2 --> pci 4096
            2: -- LINK_PCI, bw: 0.2 --> sipu 1179648
    nic nodes: num: 2
        0: nic 20480
            0: -- LINK_NET, bw: 50.0 --> net 0
            1: -- LINK_PCI, bw: 0.0 --> pci 4096
        1: nic 24576
            0: -- LINK_NET, bw: 50.0 --> net 1
            1: -- LINK_PCI, bw: 0.0 --> pci 4096
    net nodes: num: 3
        0: net 0
            0: -- LINK_NET, bw: 50.0 --> nic 20480
        1: net 1
            0: -- LINK_NET, bw: 50.0 --> nic 24576
        2: net 1
    ```

* siccl xml output

    这里是 collect local resources() 函数之后的输出，可以看到没有 nic 网卡。

    ```xml
    <system version="1">
        <cpu modelid="143" familyid="6" vendor="GenuineIntel" arch="x86_64" affinity="ffffffff,ffffffff,ffffffff,00000000,0000ffff,ffffffff" host_hash="0x34f963218b32f91c" numaid="0">
            <pci link_width="1" subsystem_vendor="0x205d" device="0x1100" link_speed="2.5 GT/s PCIe" subsystem_device="0x0000" vendor="0x205d" class="0x120000" busid="0000:11:00.0">
                <gpu uuid="8394d921-b848-45b4-b09f-5cc000022010" chip_id="0" rank="0" dev="0" micro_id="0">
                    <silink count="4" type="switch"></silink>
                </gpu>
            </pci>
        </cpu>
    </system>
    ```

* siccl xml output after populate nics, bootstrap 之后的 xml

    这下看起来是正常的了。

    ```xml
    <system version="1">
        <cpu numaid="0" host_hash="0x107f7b32ece0a131" affinity="ffffffff,ffffffff,ffffffff,00000000,0000ffff,ffffffff" arch="x86_64" vendor="GenuineIntel" familyid="6" modelid="143">
            <pci busid="0000:10:00.0" class="0x060400" vendor="0x1b36" subsystem_device="0x0000" link_speed="16.0 GT/s PCIe" device="0x000c" subsystem_vendor="0x1b36" link_width="32">
                <pci busid="0000:11:00.0" class="0x120000" vendor="0x205d" subsystem_device="0x0000" link_speed="2.5 GT/s PCIe" device="0x1100" subsystem_vendor="0x205d" link_width="1">
                    <gpu micro_id="0" dev="0" rank="0" chip_id="0" uuid="ba36c81b-1b71-4802-8a25-5b0000022010">
                        <silink type="switch" count="4"></silink>
                    </gpu>
                </pci>
            </pci>
            <pci busid="0000:01:00.0" class="0x060400" vendor="0x104c" subsystem_device="0x0000" link_speed="2.5 GT/s PCIe" device="0x8232" subsystem_vendor="0x0000" link_width="1">
                <pci busid="0000:05:00.0" class="0x02" vendor="0x15b3" subsystem_device="0x0023" link_speed="Unknown" device="0x1021" subsystem_vendor="0x15b3" link_width="0">
                    <nic>
                        <net name="mlx5_0" port="1" guid="2369990716664481952" gdr="0" keep="1" dev="0" latency="0.000000" speed="400000" maxconn="131072"></net>
                    </nic>
                </pci>
                <pci busid="0000:06:00.0" class="0x02" vendor="0x15b3" subsystem_device="0x0023" link_speed="Unknown" device="0x1021" subsystem_vendor="0x15b3" link_width="0">
                    <nic>
                        <net name="mlx5_1" port="1" guid="13577908673887504544" gdr="0" keep="1" dev="1" latency="0.000000" speed="400000" maxconn="131072"></net>
                    </nic>
                </pci>
            </pci>
            <pci busid="0000:10:01.0" class="0x060400" vendor="0x1b36" subsystem_device="0x0000" link_speed="16.0 GT/s PCIe" device="0x000c" subsystem_vendor="0x1b36" link_width="32">
                <pci busid="0000:12:00.0" class="0x120000" vendor="0x205d" subsystem_device="0x0000" link_speed="2.5 GT/s PCIe" device="0x1100" subsystem_vendor="0x205d" link_width="1">
                    <gpu micro_id="0" dev="1" rank="1" chip_id="1" uuid="d5fe127d-49da-4e7c-a255-cc0000024010">
                        <silink type="switch" count="4"></silink>
                    </gpu>
                </pci>
            </pci>
        </cpu>
    </system>
    ```

* topo system output

    nic 中  LINK_PCI, bw: 0.0 这个看起来有问题。

    ```
    sipu nodes: num: 2
        0: sipu 1114112, chip id 0 uuid ba36c81b-1b71-4802-8a25-5b0000022010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> pci 65536
        1: sipu 1179648, chip id 1 uuid d5fe127d-49da-4e7c-a255-cc0000024010 
            0: -- LINK_SILINK, bw: 400.0 --> eth_switch 0
            1: -- LINK_PCI, bw: 0.2 --> pci 65552
    pci nodes: num: 3
        0: pci 65536, bdf: 0000:10:00.0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1114112
            1: -- LINK_PCI, bw: 48.0 --> cpu 0
        1: pci 4096, bdf: 0000:01:00.0
            0: -- LINK_PCI, bw: 0.0 --> nic 20480
            1: -- LINK_PCI, bw: 0.0 --> nic 24576
            2: -- LINK_PCI, bw: 0.2 --> cpu 0
        2: pci 65552, bdf: 0000:10:01.0
            0: -- LINK_PCI, bw: 0.2 --> sipu 1179648
            1: -- LINK_PCI, bw: 48.0 --> cpu 0
    eth_switch nodes: num: 1
        0: eth_switch 0
            0: -- LINK_SILINK, bw: 400.0 --> sipu 1114112
            1: -- LINK_SILINK, bw: 400.0 --> sipu 1179648
    cpu nodes: num: 1
        0: cpu 0
            0: -- LINK_PCI, bw: 48.0 --> pci 65536
            1: -- LINK_PCI, bw: 48.0 --> pci 65552
            2: -- LINK_PCI, bw: 0.2 --> pci 4096
    nic nodes: num: 2
        0: nic 20480
            0: -- LINK_NET, bw: 50.0 --> net 0
            1: -- LINK_PCI, bw: 0.0 --> pci 4096
        1: nic 24576
            0: -- LINK_NET, bw: 50.0 --> net 1
            1: -- LINK_PCI, bw: 0.0 --> pci 4096
    net nodes: num: 2
        0: net 0
            0: -- LINK_NET, bw: 50.0 --> nic 20480
        1: net 1
            0: -- LINK_NET, bw: 50.0 --> nic 24576
    ```

* compute path output

    net 处没有 path，另外 cpu 处也没有输出，比较奇怪。

    ```
    compute path:
    sipu, num nodes: 2
        idx 0, id 1114112:
            path_sil: sipu 1114112  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1179648
            path_sil: sipu 1114112  --LINK_SILINK-->  eth_switch 0
            path_phb: sipu 1114112  --LINK_PCI-->  pci 65536  --LINK_PCI-->  cpu 0
        idx 1, id 1179648:
            path_sil: sipu 1179648  --LINK_SILINK-->  eth_switch 0  --LINK_SILINK-->  sipu 1114112
            path_sil: sipu 1179648  --LINK_SILINK-->  eth_switch 0
            path_phb: sipu 1179648  --LINK_PCI-->  pci 65552  --LINK_PCI-->  cpu 0
    pci, num nodes: 3
        idx 0, id 65536:
            path_pix: pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: pci 65536  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 65536  --LINK_PCI-->  cpu 0
        idx 1, id 4096:
            path_phb: pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: pci 4096  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 4096  --LINK_PCI-->  cpu 0
        idx 2, id 65552:
            path_phb: pci 65552  --LINK_PCI-->  cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_pix: pci 65552  --LINK_PCI-->  sipu 1179648
            path_phb: pci 65552  --LINK_PCI-->  cpu 0
    eth_switch, num nodes: 1
        idx 0, id 0:
            path_sil: eth_switch 0  --LINK_SILINK-->  sipu 1114112
            path_sil: eth_switch 0  --LINK_SILINK-->  sipu 1179648
    cpu, num nodes: 1
    nvs node or eth node, nothing todo
        idx 0, id 0:
            path_phb: cpu 0  --LINK_PCI-->  pci 65536  --LINK_PCI-->  sipu 1114112
            path_phb: cpu 0  --LINK_PCI-->  pci 65552  --LINK_PCI-->  sipu 1179648
    nic, num nodes: 2
        idx 0, id 20480:
            path_loc: nic 20480  --LINK_NET-->  net 0
        idx 1, id 24576:
            path_loc: nic 24576  --LINK_NET-->  net 1
    net, num nodes: 2
        idx 0, id 0:
        idx 1, id 1:
    ```

* ncclTopoSearchRec()

    这段代码是NCCL（NVIDIA Collective Communications Library）拓扑搜索算法的递归入口函数。主要作用是根据系统拓扑结构为集合通信操作寻找最优的通信路径和通道配置。

    核心功能：

    * 路径搜索入口：根据系统是否包含网络节点（NET）选择不同的搜索策略

    * 网络搜索模式：当系统有网络节点时，调用ncclTopoSearchRecNet进行跨节点搜索

    * 节点内搜索模式：

        * 对于NVLS模式：直接尝试GPU搜索

        * 对于新通道：先尝试PCI顺序，再尝试重放之前的通道

            进入这个分支的条件是`graph->nChannels == 0`, 表示这是第一次搜索，之前未搜索到有效 channel。

        * 最后尝试所有GPU可能性（除非要求相同通道）

    关键参数：

    * system：系统拓扑信息

    * graph：当前通信图结构

    * saveGraph：保存最优通信图

    * time：控制搜索时间的计数器

    该函数是NCCL自动优化通信路径的核心部分，通过递归搜索找到高效的GPU间通信方案。

* NVLS（NVIDIA Virtual Link Service）

    NVLS是什么？

    NVLS是NVIDIA GPU间的一种特殊的通信模式：

    * 虚拟化链路服务：通过虚拟化技术优化GPU间通信

    * 适用于特定拓扑：通常在特定GPU架构（如NVSwitch系统）中使用

    * 高效集合通信：针对AllReduce等操作进行优化

    在代码中的作用

    ```c
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
        // NVLS模式：直接进行GPU搜索，不尝试PCI顺序或重放
        ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels);
        return 0;
    }
    ```

    特殊处理的原因

    * 确定性路径：NVLS通常有固定的、最优的连接方式，不需要复杂的搜索策略

    * 架构特定：NVLS模式只在支持NVLink Switch的系统中有意义，拓扑相对固定

    * 参数传递：注意最后一个参数是 graph->nChannels，而不是具体的GPU索引

        * 这是为了批量搜索所有通道

        * 与普通模式（逐个GPU尝试）不同

    对比其他模式

    ```c
    // 普通模式（非NVLS）的搜索策略：
    // 1. 首次搜索（nChannels==0）：尝试PCI顺序
    // 2. 已有通道（nChannels>0）：尝试重放
    // 3. 最后：尝试所有GPU

    // NVLS模式：
    // 直接批量搜索，简化流程
    ```

    典型使用场景

    NVLS模式常见于：

    * DGX系统（NVSwitch架构）

    * 需要极低延迟的AllReduce操作

    * 大规模GPU集群中的分层通信

    这个条件体现了NCCL针对不同硬件架构和通信模式的优化策略，对于NVLS这种特殊且高效的通信模式，采用更直接、更简化的搜索方法。

* 为什么要 replay previous channel？

    Replay previous channel（重放之前的通道） 是NCCL的一个重要优化策略，主要基于以下几个关键原因：

    1. 保持通道一致性（Channel Consistency）

        ```c
        // 如果要求相同通道（graph->sameChannels != 0）
        // 或者已有通道（graph->nChannels > 0）
        // 就尝试重放之前的配置
        ```

        * 确保同一通信操作的所有通道具有相同的拓扑结构

        * 这对于负载均衡和性能可预测性至关重要

    2. 性能优化

        * 缓存友好：重用已证明高效的通信路径

        * 减少搜索时间：避免为每个通道重新搜索，直接从已有成功配置开始

        * 稳定性：使用已知能工作的配置，降低失败风险

    3. 搜索策略的演进

        代码中的搜索顺序体现了优化层次：

        ```c
        1. if (graph->nChannels == 0)       // 首次：尝试PCI顺序（简单启发式）
        2. else                            // 已有通道：尝试重放（重用成功配置）
        3. if (graph->sameChannels == 0)   // 最后：全面搜索（探索新可能）
        ```

    4. 实际应用场景

        假设有8个GPU需要4个通信通道：

        ```text
        第一次搜索（通道0）: 尝试多种路径，找到最佳方案A
        通道1-3: 直接"重放"方案A，而不是重新搜索
        ```

        这样做：

        * ✅ 保持所有通道结构一致

        * ✅ 大大减少搜索时间

        * ✅ 确保性能稳定性

    5. 参数传递的细节

        ```c
        ncclTopoReplayGetGpu(system, graph, -1, &g);
        // 获取之前通道使用的GPU作为起始点
        ncclTopoSearchTryGpu(..., FORCED_ORDER_REPLAY, ..., g);
        // 强制按重放顺序搜索
        ```

    6. 与sameChannels标志的关系

        * sameChannels == 1：必须使用相同通道结构

        * sameChannels == 0：可以尝试不同结构

        当sameChannels == 1时，重放是强制要求；当sameChannels == 0时，重放是优化建议。

    总结

    Replay previous channel是一种经验重用的优化策略：

    * 效率：避免重复搜索

    * 一致性：保证通信模式统一

    * 稳定性：使用已验证的方案

    * 渐进优化：先重用，再创新（后续的全面搜索）

    这种策略在保持性能的同时，显著降低了拓扑搜索的计算开销，对于大规模分布式训练尤其重要。

* ncclTopoSearchTryGpu()

    这段代码是 NCCL 拓扑搜索算法的一部分，用于在 GPU 集群中尝试连接特定 GPU 并递归探索通信路径。

    主要作用：

    * 路径尝试与回溯

        * 尝试通过指定路径（如 NVLink、PCIe）连接到目标 GPU（索引为 g）

        * 如果连接成功，递归探索该 GPU 的后续连接可能

        * 无论递归结果如何，都会恢复原始状态（回溯）

    * 标志位管理

        * 使用 flag = 1ULL << (graph->nChannels) 作为标识位

        * 通过 gpu->used ^= flag 切换该 GPU 在当前通道的使用状态

        * 确保递归探索不会重复使用同一 GPU

    * 递归搜索

        * 调用 ncclTopoSearchRecGpu 从当前 GPU 继续探索

        * 目标是找到最优的多 GPU 通信拓扑

    典型应用场景：

    * NCCL 集体通信优化：为 AllReduce、Broadcast 等操作寻找最佳通信路径

    * 多 GPU 拓扑发现：在复杂互联（NVLink、PCIe、网络）中找到低延迟、高带宽路径

    * 通信通道分配：为不同通信通道分配不同的物理路径以避免冲突

    关键特点：

    * 回溯机制：无论搜索成功与否，都会恢复系统状态

    * 递归探索：深度优先搜索可能的通信路径

    * 通道感知：考虑不同通信通道的独立路径需求

    这是 NCCL 实现高性能多 GPU 通信的核心算法之一，确保在复杂硬件拓扑中找到最优通信方案。

    结构简析：

    ```text
    ncclTopoSearchTryGpu()
        │
        ├── ncclTopoFollowPath(..., 1)   # 占用路径
        │    ├── 增加 GPU 使用标志
        │    ├── 增加中间节点使用计数
        │    └── 减少可用带宽
        │
        ├── ncclTopoSearchRecGpu()       # 递归探索
        │    │ (可能修改 graph 中的通道连接)
        │    │ (可能修改多个 GPU 的 used 标志)
        │    └── 递归返回时恢复所有修改
        │
        └── ncclTopoFollowPath(..., -1)  # 释放路径
             ├── 清除 GPU 使用标志
             ├── 减少中间节点使用计数
             └── 恢复可用带宽
    ```

    核心数据结构示例：

    ```c
    // 简化的数据结构
    struct TopoNode {
        int64_t used;       // 位图：每位表示一个通道是否使用此节点
        int bw;             // 可用带宽
        int paths[GPU];     // 到其他GPU的路径
        // ... 其他资源限制
    };

    struct CollGraph {
        int nChannels;      // 当前已分配的通道数
        int intra[MAX_GPUS][MAX_GPUS];  // 通道内连接矩阵
        int inter[MAX_NODES][MAX_NODES]; // 节点间连接
        // 每个通道的完整路径信息
    };
    ```

* saveGraph

    ```
    // (3) saveGraph 参数保存找到的可行解
    // 当找到更好的拓扑时，会复制到 saveGraph
    // 最终选择最优的 saveGraph 作为结果
    ```

* 搜索过程中的数据流：

    ```text
    搜索开始 (step=0)
        ↓
    尝试 GPU0 → GPU1 (标记 used[0], used[1])
        ↓
    递归尝试 GPU1 → GPU2
        ↓
    成功找到完整路径
        ↓ 保存到 saveGraph
    回溯：清除 used[2], used[1], used[0]
        ↓
    尝试 GPU0 → GPU3 (不同的路径)
        ↓
    ...继续搜索...
    ```

    关键理解：

    * used 是位图，不是简单数组，每位对应一个通道

    * -1 参数是关键，它触发资源释放

    * saveGraph 是结果容器，搜索过程中只修改 graph

    * 完全回溯确保每次尝试都在干净的状态开始

* ncclTopoCompute()

    这段代码是NCCL（NVIDIA Collective Communications Library）中用于计算最优通信拓扑图的核心函数。主要作用是根据硬件拓扑结构和通信模式，为多GPU/多节点通信寻找高效的通信路径。

    主要功能：

    * 拓扑分析

        * 分析系统中的GPU、网络设备等节点

        * 计算GPU间的最短/最长通信路径类型（PATH_LOC, PATH_PIX, PATH_SYS等）

    * 通信模式支持

        * 支持多种通信模式：环（RING）、树（TREE）、平衡树（BALANCED_TREE）、NVLS、COLLNET_DIRECT等

        * 根据GPU数量和系统配置选择合适的模式

    * 带宽优化

        * 在带宽约束下搜索最佳通信通道

        * 支持跨NIC（网络接口卡）通信

        * 考虑不同的带宽配置（根据GPU计算能力分SM90/SM100等不同档位）

    * 搜索策略

        * 两阶段搜索：先找可行解，再优化带宽

        * 支持从XML文件加载预定义图

        * 考虑超时机制，防止搜索时间过长

    * 容错处理

        * 如果找不到合适路径，回退到简单顺序通信

        * 考虑不同的CPU架构（AMD/Intel）的特殊处理

    关键特性：

    * 自适应：根据实际硬件拓扑调整通信策略

    * 可配置：支持环境变量NCCL_GRAPH_FILE指定外部拓扑图

    * 性能导向：在带宽、延迟、通道数等多个维度优化

    * 健壮性：在各种硬件配置下都能找到可行方案

    该函数是NCCL实现高效集体通信的基础，确保在多GPU和多节点环境下能够充分利用硬件带宽资源。

    逐段解释:

    ```c
    // 函数入口：计算拓扑通信图
    ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
      // 获取GPU数量，判断是否需要跨NIC通信
      int ngpus = system->nodes[GPU].count;
      int crossNic = (system->nodes[NET].count > 1) &&
         (graph->pattern == NCCL_TOPO_PATTERN_RING ||
          graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
          graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;
      // 设置crossNic标志，清除带宽和延迟数据
      graph->crossNic = crossNic == 1 ? 1 : 0;
      graph->bwIntra = graph->bwInter = 0;
      graph->latencyInter = 0;
    ```

    ```c
      // 初始化路径类型变量
      int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
      int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
      // 计算GPU间的最小/最大路径类型（节点内通信）
      if (ngpus > 1) {
        NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
        NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
      }
      // 计算GPU到网络的最小/最大路径类型（节点间通信）
      if (system->nodes[NET].count > 0) {
        NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
        NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
        maxTypeIntra = maxTypeInter;  // 如果存在网络，更新intra最大类型
      }
    ```

    ```c
      // 初始化图结构参数
      graph->typeIntra = minTypeIntra;
      graph->typeInter = minTypeInter;
      graph->nChannels = 0;
      // 决定是否使用相同通道（NVLS模式不使用相同通道）
      int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
      graph->sameChannels = trySameChannels;
    ```

    ```c
      // 获取CPU架构信息（用于特殊优化）
      int cpuArch, cpuVendor, cpuModel;
      NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));
    ```

    ```c
      // 检查环境变量，如果设置了XML图文件，从文件加载拓扑图
      const char* str = ncclGetEnv("NCCL_GRAPH_FILE");
      if (str) {
        INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
        struct ncclXml* xml;
        NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
        NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
        int nChannels;
        NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
        INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
        free(xml);
        if (graph->nChannels > 0) return ncclSuccess;  // 加载成功则直接返回
      }
    ```

    ```c
      // 检查计算能力，NVLS模式需要特定硬件支持
      int ccMin;
      NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));
      if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90)) 
        return ncclSuccess;  // 不支持NVLS则直接返回
    ```

    ```c
      // 设置不同模式的最大通道数限制
      if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) 
        graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);
      if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) 
        graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY+1, system->nodes[GPU].count);
    ```

    ```c
      // 单个GPU的特殊处理：非环模式改为树模式
      if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) 
        graph->pattern = NCCL_TOPO_PATTERN_TREE;
    ```

    ```c
      // NVLS在单节点内的特殊设置：确保从所有GPU均匀拉取数据
      if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
        graph->minChannels = graph->maxChannels;  // 最小通道数等于最大通道数
      }
    ```

    ```c
      // 检查是否跨NVLINK分裂（如双路CPU系统）
      int splitNvLink;
      NCCLCHECK(ncclTopoSplitNvLink(system, &splitNvLink));
      if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
        // 跨插槽通信较慢，强制使用至少2个通道
        if (graph->maxChannels >= 2 && graph->minChannels == 1) 
          graph->minChannels = 2;
      }
    ```

    ```c
      // 创建临时图用于搜索过程
      struct ncclTopoGraph tmpGraph;
      memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));
    ```

    ```c
      // 根据系统配置选择带宽数组（不同计算能力有不同的带宽配置）
      int nspeeds = 0;
      float* speedArray = NULL;
      if (system->nodes[NET].count == 0) {  // 节点内通信
        nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
        speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
      } else {  // 节点间通信
        nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
        speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
      }
    ```

    ```c
      // 初始化搜索参数
      int pass = 1;  // 第一阶段：寻找可行解
      int speedIndex = 0;
      float maxBw = system->maxBw;    // 系统最大带宽
      float totalBw = system->totalBw; // 系统总带宽
      // 非环模式调整总带宽估算
      if (ngpus > 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) 
        totalBw *= ngpus*1.0/(ngpus-1);
      // 找到合适的起始带宽值
      while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex]*graph->minChannels > totalBw) && speedIndex < nspeeds-1) 
        speedIndex++;
      tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
      int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;  // 全局超时控制
    ```

    ```c
    // 搜索标签，从这里开始递归搜索
    search:
      // 根据模式设置不同的超时时间
      int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
        tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
      tmpGraph.nChannels = 0;
      globalTimeout -= time;  // 更新剩余时间
    ```

    ```c
      // 调用核心搜索函数
      NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
      
      // 调试输出（被注释掉了）
    #if 0
      printf("Id %d Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", 
             tmpGraph.id, tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.bwInter, tmpGraph.bwIntra, 
             tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, 
             tmpGraph.sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, 
             time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
      // 输出每个通道的具体配置
      for (int c=0; c<graph->nChannels; c++) {
        printf("%2d : ", c);
        for (int g=0; g<ngpus; g++) {
          printf("%d ", graph->intra[c*ngpus+g]);
        }
        printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);
        printf("\n");
      }
    #endif
    ```

    ```c
      // 检查搜索结果
      if (time == -1) goto done;  // 找到完美解，结束
      if (graph->nChannels*graph->bwInter >= system->totalBw) goto done; // 带宽已达上限，结束
    ```

    ```c
      // 第一阶段搜索：尝试不同的优化策略
      if (pass == 1) {
        // 策略1：尝试不同的通道配置（AMD CPU+SYS路径除外）
        if (tmpGraph.sameChannels == 1 &&
            !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD && tmpGraph.typeIntra == PATH_SYS)) {
          tmpGraph.sameChannels = 0;
          goto search;  // 重新搜索
        }
        tmpGraph.sameChannels = trySameChannels;  // 恢复原始设置
    ```

    ```c
        // 更新时间并检查全局超时
        if (time != -1) globalTimeout += time;
        else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
        if (globalTimeout < 0 && graph->nChannels) goto done; // 超时但有解，结束
    ```

    ```c
        // 策略2：尝试更简单的树模式（计算能力≥90）
        if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
          goto search;
        }
        tmpGraph.pattern = graph->pattern;  // 恢复原始模式
    ```

    ```c
        // 策略3：尝试更差的节点内路径类型
        int maxIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : maxTypeIntra;
        if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
          tmpGraph.typeIntra += 1;  // 降低路径质量
          if (tmpGraph.typeIntra < PATH_DIS) goto search; // PATH_DIS是下限
        }
        tmpGraph.typeIntra = minTypeIntra;  // 恢复最佳路径
    ```

    ```c
        // 策略4：尝试更差的节点间路径类型
        if (system->nodes[NET].count > 0 && tmpGraph.typeInter < maxTypeInter && 
            (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
          tmpGraph.typeInter += 1;
          if (tmpGraph.typeInter < PATH_DIS) goto search;
        }
        tmpGraph.typeInter = minTypeInter;  // 恢复最佳路径
    ```

    ```c
        // 策略5：尝试跨NIC通信（如果允许）
        if (crossNic == 2 && tmpGraph.crossNic == 0
            && (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
          tmpGraph.crossNic = 2;
          goto search;
        }
        tmpGraph.crossNic = crossNic == 1 ? 1 : 0;  // 恢复原始设置
    ```

    ```c
        // 策略6：降低带宽要求
        if ((speedIndex < nspeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->bwInter > .49))) {
          tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];  // 使用更低的带宽
          goto search;
        }
        // 重置带宽索引
        speedIndex = 0;
        while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds-1) speedIndex++;
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
      }
    ```

    ```c
    // 第一阶段结束标签
    done:
      // 第一阶段找到解，进入第二阶段优化
      if (pass == 1) {
        time = -1;
        NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));  // 复制通道配置
        memcpy(&tmpGraph, graph, sizeof(tmpGraph));  // 复制结果到临时图
        
        // 重置带宽到找到的解的带宽
        speedIndex = 0;
        while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds-1) speedIndex++;
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
        
        tmpGraph.minChannels = graph->nChannels;  // 设置最小通道数为当前解
        pass = 2;  // 进入第二阶段
      }
    ```

    ```c
      // 第二阶段：在已有解的基础上优化带宽
      if (pass == 2) {
        // 尝试增加带宽（如果可能）
        if (time != 0 && speedIndex > 0) {
          if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
            // 环模式：直接增加带宽
            tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];
            goto search;
          } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph.bwInter == graph->bwInter && 
                     tmpGraph.bwInter < tmpGraph.bwIntra*2) {
            // NVLS模式：增加节点间带宽
            tmpGraph.minChannels = tmpGraph.maxChannels = graph->nChannels;
            tmpGraph.bwInter = speedArray[--speedIndex];
            goto search;
          } else if (tmpGraph.bwIntra == graph->bwIntra && tmpGraph.bwIntra < tmpGraph.bwInter*2) {
            // 树模式：增加节点内带宽
            tmpGraph.bwIntra = speedArray[--speedIndex];
            goto search;
          }
        }
        time = -1;
        memcpy(&tmpGraph, graph, sizeof(tmpGraph));  // 最终结果
      }
    ```

    ```c
      // 如果最终没有找到有效通道且不是特殊模式，使用简单回退方案
      if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
        INFO(NCCL_GRAPH, "Could not find a path for pattern %d, falling back to simple order", graph->pattern);
        // 按GPU顺序简单分配
        for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
        graph->inter[0] = graph->inter[1] = 0;
        graph->bwIntra = graph->bwInter = 0.1;  // 设置低带宽
        graph->typeIntra = graph->typeInter = PATH_SYS;  // 使用系统路径
        graph->nChannels = 1;  // 单通道
      }
      return ncclSuccess;  // 成功返回
    }
    ```

    这个函数的核心是一个两阶段的启发式搜索算法：

    * 第一阶段：降低要求寻找可行解（降低带宽、放宽路径限制）

    * 第二阶段：在可行解基础上优化带宽（提高带宽直至最优）

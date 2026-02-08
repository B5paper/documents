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

    该函数是 NCCL 自动优化通信路径的核心部分，通过递归搜索找到高效的GPU间通信方案。

    分段解释：

    1. 函数定义和参数

        ```c
        ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
        ```

        * system: 拓扑系统结构体，包含所有节点（GPU、NET、CPU等）

        * graph: 当前搜索的拓扑图

        * saveGraph: 保存最佳结果的拓扑图

        * time: 时间参数，可能用于超时控制或性能评估

    2. 获取搜索参数

        ```c
        int backToNet, backToFirstRank;
        NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
        ```

        * 根据通信模式（pattern）获取搜索参数

        * backToNet: 是否需要回到网络节点

        * backToFirstRank: 是否需要回到第一个rank

    3. 处理网络通信情况

        ```c
        if (system->nodes[NET].count) {
            // Start from NET
            ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
        }
        ```

        * 如果系统中有网络节点（跨节点通信）

        * 从网络节点开始递归搜索

    4. 处理节点内通信情况

        ```c
        } else {
            // Intra-node only.
        ```

        如果没有网络节点（单节点内通信）

        4.1 NVLS 模式处理

        ```c
        if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels));
            return ncclSuccess;
        }
        ```

        * 如果是NVLS（NVLink SHARP）模式

        * 直接尝试GPU搜索，传入当前通道数作为起始GPU

        4.2 首次搜索（无通道时）

        ```c
        else if (graph->nChannels == 0) {
            // Try PCI order first
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
        }
        ```

        * 如果还没有分配任何通道

        * 首先尝试按照PCI顺序搜索（从GPU 0开始）

        4.3 重放之前的最佳通道

        ```c
        } else {
            // Also try to replay previous channel
            int g;
            NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
        }
        ```

        * 如果已有分配的通道

        * 尝试重放（replay）之前的最佳配置

        * 获取之前使用的GPU作为起始点

        4.4 全面搜索（非强制相同通道时）

        ```c
        if (graph->sameChannels == 0 || graph->nChannels == 0) {
            // Finally, try all other possibilities unless we are forced to use the same channels
            for (int g=0; g<system->nodes[GPU].count; g++) {
                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
            }
        }
        ```

        * 如果不要求所有通道相同，或者还没有通道

        * 遍历所有GPU进行尝试

        * 这是最全面的搜索策略

    5. 返回成功

        ```c
        return ncclSuccess;
        }
        ```

    总结

    这个函数是NCCL拓扑搜索的核心，它根据不同的场景采用不同的搜索策略：

    * 跨节点通信：从网络节点开始搜索

    * 节点内通信：

        * NVLS模式：特殊处理

        * 首次搜索：优先PCI顺序

        * 已有配置：尝试重放优化

        * 全面搜索：遍历所有可能性

    函数通过递归和分支限界的方式寻找最优的通信路径，平衡了搜索效率和结果质量。

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

* `ncclTopoCompute()`

    这段代码是 NCCL（NVIDIA Collective Communications Library）中用于计算最优通信拓扑图的核心函数。主要作用是根据硬件拓扑结构和通信模式，为多 GPU/多节点通信寻找高效的通信路径。

    主要功能：

    * 拓扑分析

        * 分析系统中的 GPU、网络设备等节点

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

        * 支持从 XML 文件加载预定义图

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
      // 获取 GPU 数量，判断是否需要跨 NIC 通信
      int ngpus = system->nodes[GPU].count;
      int crossNic = (system->nodes[NET].count > 1) &&
         (graph->pattern == NCCL_TOPO_PATTERN_RING ||
          graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
          graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;
      // 设置 crossNic 标志，清除带宽和延迟数据
      graph->crossNic = crossNic == 1 ? 1 : 0;
      graph->bwIntra = graph->bwInter = 0;
      graph->latencyInter = 0;
    ```

    ```c
      // 初始化路径类型变量
      int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
      int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
      // 计算 GPU 间的最小/最大路径类型（节点内通信）
      if (ngpus > 1) {
        NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
        NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
      }
      // 计算 GPU 到网络的最小/最大路径类型（节点间通信）
      if (system->nodes[NET].count > 0) {
        NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
        NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
        maxTypeIntra = maxTypeInter;  // 如果存在网络，更新 intra 最大类型
      }
    ```

    注：

    1. 这里计算最小/最大路径类型有什么用？

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
      // 检查环境变量，如果设置了 XML 图文件，从文件加载拓扑图
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
      // 检查计算能力，NVLS 模式需要特定硬件支持
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
      // 单个 GPU 的特殊处理：非环模式改为树模式
      if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) 
        graph->pattern = NCCL_TOPO_PATTERN_TREE;
    ```

    注：

    1. 这个是为什么？没看懂。是因为树模式支持单节点，ring 模式不支持吗？

    ```c
      // NVLS 在单节点内的特殊设置：确保从所有 GPU 均匀拉取数据
      if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
        graph->minChannels = graph->maxChannels;  // 最小通道数等于最大通道数
      }
    ```

    ```c
      // 检查是否跨 NVLINK 分裂（如双路 CPU 系统）
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

    注：

    1. ccMin：系统中所有GPU的最小计算能力（Compute Capability）

    1. 三层判断逻辑：

        ccMin >= 100：SM100架构（如H100等最新GPU）

            使用 NSPEEDSINTRA_SM100

            使用 sm100SpeedArrayIntra

        ccMin >= 90：SM90架构（如A100等）

            使用 NSPEEDSINTRA_SM90

            使用 sm90SpeedArrayIntra

        其他：老架构

            使用 NSPEEDSINTRA

            使用 speedArrayIntra

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

    注：

    1. gpu 数量大于 1 且拓扑不是 ring，那么计算 

        totalBw *= ngpus*1.0/(ngpus-1);

        数学推导

        环状拓扑（Ring）：

        * 每个GPU同时发送和接收数据

        * 通信量在GPU间均匀分布

        * 总带宽 = 单个链路带宽

        树状拓扑（Tree）：

        * 有根节点（root）作为汇聚点

        * 根节点会成为瓶颈

        * 例如在 All-Reduce 操作中：

        对于 ngpus = 4 的情况：

        树状拓扑（二叉树）：

        ```text
            GPU0 (root)
            /    \
          GPU1   GPU2
                 |
               GPU3
        ```

        * GPU3 → GPU2 → GPU0（向上聚合）

        * GPU1 → GPU0（向上聚合）

        * GPU0 需要处理所有GPU的数据

        * 根节点带宽需求：来自3个子节点

        * 有效带宽需要放大：4/3 ≈ 1.33倍

        **公式解释**

        ```c
        totalBw *= ngpus*1.0/(ngpus-1);
        ```

        | ngpus | 乘数因子 | 含义 |
        | - | - | - |
        | 2 | 2.0 | 双GPU时树状拓扑的根要处理全部流量 |
        | 4 | 1.33 | 四GPU时根处理3/4的聚合流量 |
        | 8 | 1.14 | 八GPU时根处理7/8的聚合流量 |
        | n | n/(n-1) | 随着GPU数增加，瓶颈效应减弱 |

        **为什么要这样调整？**

        在 NCCL 的拓扑选择算法中：

        * 系统会评估不同拓扑的性能

        * 环状和树状拓扑有不同特性

        * 这个公式补偿树状拓扑的根节点瓶颈

        * 使得不同拓扑可以在相同基准下比较

    1. 在 All-Reduce 中的具体应用

        对于 n 个 GPU 的 All-Reduce：

        | 操作阶段 | 环状拓扑 | 树状拓扑 |
        | - | - | - |
        | Reduce-Scatter | n-1 步 | log₂(n) 步 |
        | All-Gather | n-1 步 | log₂(n) 步 |
        | 总步数 | 2(n-1) | 2log₂(n) |
        | 瓶颈 | 每个链路均匀负载 | 根节点集中负载 |

        所以树状拓扑虽然步数少（O(log n) vs O(n)），但根节点是瓶颈，需要更高的链路带宽来达到相同性能。

    ```c
    // 搜索标签，从这里开始递归搜索
    search:
      // 根据模式设置不同的超时时间
      int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
        tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
      tmpGraph.nChannels = 0;
      globalTimeout -= time;  // 更新剩余时间
    ```

    注：

    1. 这是一个三层的条件运算符，根据搜索情况设置不同的超时时间：

        第一种情况：sameChannels

        ```c
        tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS
        ```

        * 条件：所有通信通道使用相同的物理链路（Channel）

        * 场景：

            * 系统资源有限

            * 多个通信通道复用相同物理链路

            * 搜索空间较小（因为选择有限）

        * 超时较短：NCCL_SEARCH_TIMEOUT_SAMECHANNELS

            * 例如 5ms（因为搜索简单）

        第二种情况：树状拓扑

        ```c
        tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE
        ```

        * 条件：当前尝试构建树状拓扑

        * 场景：

            * 构建二叉树、平衡树等

            * 需要确定根节点和子树结构

        * 超时中等：NCCL_SEARCH_TIMEOUT_TREE

            * 例如 20ms（树构建相对复杂）

        第三种情况：默认（主要是环状拓扑）

        ```c
        NCCL_SEARCH_TIMEOUT
        ```

        * 条件：既不是sameChannels，也不是树状拓扑

        * 场景：

            * 通常是环状拓扑（Ring）的搜索

            * 或者一般的拓扑搜索

        * 超时最长：NCCL_SEARCH_TIMEOUT

            * 例如 100ms（环状拓扑搜索最复杂）

    1. 通道数重置

        ```c
        tmpGraph.nChannels = 0;
        ```

        * 重置找到的通道数为0

        * 表示开始新一轮的搜索

        * 每个搜索尝试都从空开始构建

    1. 全局超时更新

        ```c
        globalTimeout -= time;  // 更新剩余时间
        ```

        * globalTimeout：整个搜索过程的总体时间预算

        * 减去本次搜索尝试分配的时间

        * 实现分层时间管理

    1. 为什么需要这样的设计？

        1. 搜索复杂度不同

            环状拓扑：需要找到最优的GPU排列顺序

                对于n个GPU，有(n-1)!/2种可能环

                搜索空间巨大，需要更多时间

            树状拓扑：相对简单

                确定根节点后，结构相对固定

                搜索空间较小

            SameChannels：最简单

                选择有限，很快就能穷举

        2. 避免无限搜索

            实际部署中不能无限制搜索

            需要在启动延迟和性能间平衡

            典型配置：

                总超时：500ms

                环搜索：100ms × 可能2-3次

                树搜索：20ms × 可能几次

                SameChannels：5ms

    1. "渐进优化：允许失败后重试，但时间递减"，既然前面失败了，时间递减后不得更失败了？

        你提出了一个非常深刻的观察！这确实看起来有点矛盾。让我详细解释这背后的逻辑：

        核心洞察：这不是简单的"失败-重试"，而是搜索策略的转换

        实际执行流程更可能是这样的：

        场景：8个GPU的拓扑搜索

        ```c
        // 第1次尝试：完全搜索（100ms）
        time = NCCL_SEARCH_TIMEOUT;  // 100ms
        // 尝试所有可能的环排列，寻找最优解
        // 结果：找到解，但质量评估不高（比如带宽只有理论值的80%）
        globalTimeout -= 100;  // 剩余400ms

        // 第2次尝试：简化的启发式搜索（50ms）  
        time = NCCL_SEARCH_TIMEOUT / 2;  // 50ms
        // 使用启发式算法快速寻找可行解
        // 结果：找到解，质量一般（85%）
        globalTimeout -= 50;  // 剩余350ms

        // 第3次尝试：贪婪局部优化（25ms）
        time = NCCL_SEARCH_TIMEOUT / 4;  // 25ms
        // 基于前一次解进行局部优化
        // 结果：找到更好解（90%）
        ```

        为什么"减少时间"反而可能"找到更好解"？

        1. 不是简单重复，而是改变策略

            | 尝试次数 | 时间预算 | 搜索策略 | 目标 |
            | - | - | - | - |
            | 1 | 100ms | 穷举/深度搜索 | 找任何可行解 |
            | 2 | 50ms | 启发式算法 | 找快速可行解 |
            | 3 | 25ms | 局部优化 | 优化现有解 |

        2. 避免陷入局部最优的深度搜索

            ```c
            // 伪代码示例
            for (int attempt = 0; attempt < maxAttempts; attempt++) {
                int timeBudget = baseTimeout / (1 << attempt);  // 指数递减
                
                if (attempt == 0) {
                    // 深度优先搜索：可能陷入复杂分支
                    result = deepSearch(timeBudget);
                } else if (attempt == 1) {
                    // 广度优先搜索：探索更多可能性
                    result = heuristicSearch(timeBudget);
                } else {
                    // 基于前次结果的局部优化
                    result = localOptimize(previousResult, timeBudget);
                }
                
                if (result.quality > bestQuality) {
                    bestResult = result;
                    bestQuality = result.quality;
                }
            }
            ```

        3. 时间递减的实际效果

            ```text
            时间递减 → 迫使算法更聪明
                    ↓
            减少穷举，增加启发式
                    ↓  
            避免陷入复杂分支
                    ↓
            更快收敛到"足够好"的解
            ```

        具体算法模式

        模式A：多阶段搜索

        ```c
        // 阶段1：快速扫描（100ms）
        // - 评估简单方案
        // - 排除明显差的选择

        // 阶段2：重点搜索（50ms）  
        // - 只在前10%的候选解中深入
        // - 避免在差解上浪费时间

        // 阶段3：微调（25ms）
        // - 优化已找到的好解
        // - 调整参数达到局部最优
        ```

        模式B：模拟退火思想

        ```c
        // 高温阶段（时间多）：大胆探索
        // - 尝试随机排列
        // - 可能找到意外的好解

        // 低温阶段（时间少）：精细调优  
        // - 只做小幅调整
        // - 收敛到局部最优
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

* 为什么 NVLS 模式不使用相同通道？

    NVLS（NVLink SHARP）模式不使用相同通道的核心原因是：它需要同时从多个GPU收集数据，而不是点对点通信。

    让我详细解释：

    1. NVLS 的通信模式特点

        * 多对一聚合：NVLS 允许多个GPU同时向同一个GPU发送数据，进行数据聚合

        * 非点对点：不像环或树模式需要建立特定的点对点连接路径

        * 硬件加速：利用NVLink的SHARP（可扩展分层聚合和归约协议）硬件特性

    2. "相同通道"（sameChannels）的含义

        在NCCL中，sameChannels=1 意味着：

        * 所有通道（channels）使用相同的物理路径

        * GPU在通道间的连接关系保持一致

        * 适用于需要固定连接模式的通信（如环、树）

    3. 为什么NVLS不需要相同通道

        a) 数据流模式不同

        ```c
        // 环模式：固定顺序的点对点通信
        GPU0 → GPU1 → GPU2 → GPU3 → ... → GPU0

        // 树模式：分层聚合
        GPU0 ─┐
        GPU1 ┼→ GPU4 → 聚合结果
        GPU2 ┼┘
        GPU3 ┘

        // NVLS模式：多GPU同时写入
        GPU0 ──┐
        GPU1 ──┼→ GPU_X（聚合点）
        GPU2 ──┼
        GPU3 ──┘
        ```

        b) 硬件支持的特性

        * NVLS使用NVLink的原子操作：多个GPU可以同时写入同一内存区域

        * 不需要特定的路由路径：数据通过NVLink交换机直接到达目标

        * 通道独立性：每个GPU到聚合点的连接是独立的

    4. 代码中的具体体现

        ```c
        // NVLS模式强制设置 sameChannels = 0
        int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;

        // 节点内NVLS需要从所有GPU均匀拉取数据
        if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
            graph->minChannels = graph->maxChannels;  // 强制所有通道都使用
        }
        ```

    5. 实际影响

        使用相同通道（环/树模式）：

        ```python
        # 通道0：GPU0→GPU1→GPU2→GPU3
        # 通道1：GPU0→GPU1→GPU2→GPU3  # 相同路径
        # 所有通道复制相同的连接模式
        ```

        不使用相同通道（NVLS模式）：

        ```python
        # 通道0：GPU0→聚合点, GPU1→聚合点
        # 通道1：GPU2→聚合点, GPU3→聚合点  # 可能不同的源GPU
        # 通道可以有不同的源-目标映射
        ```

    6. 性能考虑

        * 带宽最大化：不同通道可以使用不同的NVLink路径，避免竞争

        * 并发性：多个GPU可以同时使用不同通道进行聚合

        * 负载均衡：数据可以更均匀地分布在所有可用链路上

    7. 系统架构对应

        ```bash
        # 典型的多GPU NVLink系统
        GPU0 ── NVLink ── GPU1
          │                  │
        NVLink           NVLink
          │                  │
        GPU2 ── NVLink ── GPU3
                │
            NVLink交换机
                │
            聚合点内存
        ```

    总结：NVLS不使用相同通道是因为它的通信模式本质上是多源到单目的地的并行写入，而不是需要固定连接拓扑的点对点通信。这种设计允许更好地利用NVLink的硬件聚合能力，提高带宽利用率。

* `ncclTopoGetGpuMinPath()`

    这段C/C++代码用于计算NCCL拓扑系统中GPU到指定类型设备之间的最小路径类型。以下是分段解释：

    函数签名

    ```c
    ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min)
    ```

    * 功能：获取所有GPU到type类型设备间的最小路径类型

    * 参数：

        * system: 拓扑系统结构体指针

        * type: 目标设备类型（如GPU/NIC/CPU等）

        * min: 输出参数，存储计算得到的最小路径类型

    * 返回值：ncclSuccess表示成功

    变量初始化

    ```c
    int minPath = PATH_SYS;
    ```

    * 初始化最小路径为PATH_SYS（系统默认最大值，表示最差路径）

    * PATH_SYS通常定义为最高的路径类型值，确保能被后续更小的值覆盖

    外层循环：遍历所有GPU

    ```c
    for (int i=0; i<system->nodes[GPU].count; i++) {
    ```

    * 遍历拓扑系统中所有的GPU节点

    * system->nodes[GPU]：GPU节点数组

    * count：GPU数量

    获取路径信息

    ```c
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    ```

    * 获取第i个GPU到type类型设备的所有路径

    * paths是一个二维数组：paths[j]表示从GPU i到type设备j的路径信息

    * 如果路径不存在则跳过此GPU

    内层循环：遍历目标类型设备

    ```c
    for (int j=0; j<system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;
      minPath = std::min(minPath, paths[j].type);
    }
    ```

    * 遍历所有type类型的设备

    * 特殊处理：如果目标类型也是GPU，跳过自己到自己的路径（i == j）

    * 更新最小路径：paths[j].type表示具体的路径类型值（如PCIE、NVLink等）

    * 使用std::min保持当前找到的最小值

    结果返回

    ```c
    *min = minPath;
    return ncclSuccess;
    ```

    * 将计算得到的最小路径类型赋值给输出参数min

    * 返回成功状态

    核心逻辑总结

    * 遍历所有GPU作为源设备

    * 遍历所有目标类型设备作为目标

    * 排除自环路径（当源和目标都是GPU且相同）

    * 收集所有路径类型中的最小值

    * 路径类型值越小通常表示连接质量越好（如NVLink < PCIe）

    示例场景

    如果系统有4个GPU，通过NVLink和PCIe混合连接：

    * GPU0↔GPU1: NVLink（类型值小）

    * GPU0↔GPU2: PCIe（类型值大）

    * 其他路径类似

    函数将返回所有GPU间路径中的最小类型值（即NVLink的类型值）。

* ncclTopoSplitNvLink()


    这个函数用于检测系统是否存在NVLink分割情况（即GPU被分成两个独立的NVLink域，域间无法通过NVLink直连）。

    核心作用

    判断系统是否处于分割的NVLink拓扑，通常发生在：

    * 两个独立的NUMA节点/CPU插槽

    * 每个CPU插槽下的GPU通过NVLink互连

    * 但两个插槽间的GPU只能通过QPI/UPI/PCIe等较慢链路通信

    算法步骤

    1. 初始化NVLink域标记

        ```c
        for (int g=0; g<system->nodes[GPU].count; g++) nvlDomain[g] = g;
        ```

        * 初始假设每个GPU都是独立的域

    2. 发现NVLink连接关系

        ```c
        if (gpu->paths[GPU][p].type == PATH_NVL) {
            nvlDomain[p] = domain;
        }
        ```

        * 遍历所有GPU对

        * 如果GPU g 和 GPU p 之间有NVLink连接（PATH_NVL）

        * 将GPU p 标记为与GPU g 同一域

    3. 统计每个域的GPU数量

        ```c
        for (int g=0; g<system->nodes[GPU].count; g++) {
            nvlDomainCount[nvlDomain[g]]++;
        }
        ```

    4. 判断是否为分割拓扑

        ```c
        // 计算包含多个GPU的域数量
        if (nvlDomainCount[g] > 1) nvlDomains++;
        // 判断：当且仅当有2个域时，认为是分割NVLink
        *splitNvLink = nvlDomains == 2 ? 1 : 0;
        ```

    典型场景示例

    场景1：单插槽8个GPU

    ```text
    GPU0-GPU7 全部通过NVLink互连
    ```

    * 所有GPU在同一NVLink域

    * nvlDomains = 1

    * splitNvLink = 0（不是分割拓扑）

    场景2：双插槽各4个GPU

    ```text
    插槽A: GPU0-GPU3 通过NVLink互连
    插槽B: GPU4-GPU7 通过NVLink互连
    但 A↔B 之间只能通过QPI连接
    ```

    * 形成2个NVLink域

    * nvlDomains = 2

    * splitNvLink = 1（是分割拓扑）

    为什么重要

    * 通信优化：在分割拓扑中，跨域通信需要特殊处理

    * 集合通信策略：可能需要分层通信（先域内NVLink，再域间QPI）

    * 性能考虑：避免跨域的频繁数据交换

    这个检测帮助NCCL选择最优的通信策略来应对复杂的多GPU系统拓扑。

* 那如果两个 gpu，各放在不同的 cpu 上，它们之间又用 nvlink 相连，是否可以直接通信？

    可以，但需要特定的硬件架构支持。

    实际情况分析

    1. 传统情况（分割拓扑）

        你描述的场景正是函数要检测的典型分割拓扑：

        * 每个CPU插槽上的GPU通过NVLink互连

        * 但两个CPU插槽间没有直接的NVLink连接

        * GPU跨插槽通信只能通过：

            * QPI/UPI（Intel CPU间连接）

            * Infinity Fabric（AMD CPU间连接）

            * PCIe交换（较慢）

    2. NVLink跨节点直连技术

        但NVIDIA确实提供了跨CPU的NVLink直连方案：

        A. NVLink Switch系统（NVIDIA DGX等）

        ```text
        CPU0-GPU0-GPU1     CPU1-GPU2-GPU3
            │    │            │    │
            └────┼────────────┼────┘
                 │            │
            NVLink Switch/NVSwitch
        ```

        * 使用NVSwitch作为中央交换设备

        * 所有GPU（无论属于哪个CPU）都连接到NVSwitch

        * 实现完全连接的NVLink网络

        B. NVLink Bridge（较旧方案）

        ```text
        CPU0-GPU0══GPU1-CPU1
                NVLink
        ```

        * 使用专门的NVLink桥接器

        * 直接连接两个GPU（跨CPU）

        C. Grace Hopper Superchip

        ```text
        Grace CPU ── NVLink-C2C ── Hopper GPU
            │                          │
            └───── 统一内存空间 ──────┘
        ```

        * CPU和GPU通过NVLink-C2C直连

        * 形成一致性内存空间

    函数检测逻辑的局限性

    ```c
    // 当前函数只能检测简单的"有无NVLink连接"
    if (gpu->paths[GPU][p].type == PATH_NVL) {
        nvlDomain[p] = domain;  // 标记为同一域
    }
    ```

    问题：

    * 如果所有GPU都通过NVSwitch互连，函数会认为它们都在同一个域

    * 但实际上可能存在NUMA拓扑差异（不同CPU内存控制器）

    实际通信能力

    即使有NVLink直连，仍需考虑：

    * NUMA效应

        * GPU访问"本地"CPU内存 vs "远程"CPU内存

        * 延迟和带宽可能不同

    * PCIe Root Complex

        * 每个CPU有自己的PCIe控制器

        * 跨CPU访问可能涉及额外跳转

    * 系统配置

        * BIOS设置（NUMA、PCIe归属）

        * 驱动和固件版本

    建议的增强检测

    更完整的检测应考虑：

    ```c
    // 伪代码：检测真正的通信拓扑
    bool canDirectNvLinkComm(int gpuA, int gpuB) {
        // 1. 检查NVLink物理连接
        if (!hasNvLinkPhysical(gpuA, gpuB)) return false;

        // 2. 检查是否跨NUMA节点
        if (getCpuOfGpu(gpuA) != getCpuOfGpu(gpuB)) {
            // 3. 检查是否有NVSwitch或跨CPU桥接
            return hasNvSwitchOrBridge();
        }
        return true;
    }
    ```

    结论：现代NVIDIA系统确实支持跨CPU的GPU通过NVLink直接通信，但需要特定的硬件架构（如NVSwitch）。常规的双路服务器通常还是分割拓扑。

* 既然都使用 nvlink 了，为什么还说跨插槽通信慢？

    这个问题非常好，触及了NCCL通信优化的核心细节。让我详细解释：

    关键点：即使有NVLink，跨插槽通信也可能变慢

    1. NVLink的层级结构

        ```text
        CPU0-PCIE-GPU0══GPU1-PCIE-CPU1
                NVLink         ↑
                            QPI/UPI瓶颈
        ```

        * GPU0↔GPU1：NVLink直连（快）

        * 但数据要到达对端CPU管理的GPU内存，可能还需要经过：

            * PCIe（GPU↔本地CPU）

            * QPI/UPI（CPU0↔CPU1）← 主要瓶颈

            * PCIe（远端CPU↔GPU）

    2. 内存访问模式差异

        在分割NVLink拓扑中：

        情况A：同插槽GPU通信

        ```text
        GPU0（插槽A）↔ GPU1（插槽A）
        路径：NVLink直连
        特点：无CPU介入，纯GPU间DMA
        ```

        情况B：跨插槽GPU通信

        ```text
        GPU0（插槽A）↔ GPU2（插槽B）
        可能路径：
        1. GPU0 → CPU0（PCIe）→ CPU1（QPI）→ GPU2（PCIe）
           - 涉及CPU间链接（QPI/UPI带宽较低）
           - 需要CPU参与内存拷贝

        2. 如果支持GPU直接访问远端内存：
           GPU0 → GPU2（NVLink？）
           - 但需要经过PCIe或专门的跨CPU NVLink桥
           - 仍可能涉及额外的协议转换
        ```

    3. QPI/UPI vs NVLink带宽对比

        ```text
        NVLink 4.0:   600 GB/s（双向）
        NVLink 3.0:   300 GB/s
        QPI/UPI:       20-40 GB/s  ← 差一个数量级！
        PCIe 5.0 x16:  64 GB/s（单向）
        ```

    4. 代码逻辑的深层含义

        ```c
        if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
            // 跨插槽通信较慢，强制使用至少2个通道
            if (graph->maxChannels >= 2 && graph->minChannels == 1)
                graph->minChannels = 2;
        }
        ```

    为什么Ring模式特别需要关注？

    ```text
    Ring通信模式：
    GPU0 → GPU1 → GPU2 → GPU3 → GPU0
    ```

    * 每个GPU都既是发送者也是接收者

    * 链条中任何一个环节慢都会拖慢整个环

    * 如果跨插槽链接是瓶颈，整个环的带宽受限于最慢的链路

    使用多个Channel的好处

    ```c
    graph->minChannels = 2;  // 强制至少2个通道
    ```

    双通道的作用：

    * 隐藏延迟

        * 当一个通道在等待跨插槽通信时，另一个通道可以继续工作

    * 提高带宽利用率

        ```text
        单通道：GPU0 → GPU1 → GPU2 → GPU3 （串行）
        双通道：GPU0→GPU2 和 GPU1→GPU3 可以并行
        ```

    * 平衡跨插槽流量

        * 避免单个通道承担所有跨插槽通信

    5. 实际系统示例

        假设8个GPU，4个在插槽A，4个在插槽B：

        ```text
        Ring单通道（问题）：
        A0 → A1 → A2 → A3 → B0 → B1 → B2 → B3 → A0
                            ↑
                        跨插槽瓶颈点
                        这里使用QPI/UPI
        ```

        ```text
        Ring双通道（改善）：
        通道1: A0 → A2 → B0 → B2 → A0
        通道2: A1 → A3 → B1 → B3 → A1
        跨插槽流量被分散到两个通道
        ```

    6. 现代硬件的改进

        最新系统如NVLink Switch/NVSwitch确实改善了这个问题：

        * 所有GPU通过中央交换机互连

        * 避免了CPU间链接的瓶颈

        * 但拓扑检测代码仍要考虑老系统

    总结

    核心原因：

    * 即使GPU间有NVLink，内存归属问题可能导致需要CPU间通信

    * CPU间链接（QPI/UPI）远慢于NVLink

    * Ring通信模式对最慢链路特别敏感

    解决方案：

    * 增加通道数以隐藏延迟和并行化跨插槽通信

    * 更智能的通信模式（如Tree、CollNet等）

    * 使用NVSwitch等现代互连技术

    这体现了NCCL对真实世界系统复杂性的深度优化，考虑了硬件拓扑的细微差异。

    注：

    1. 这个问题 ds 的回答存疑。逻辑不清。后续有时间了可以看看其他 ai 的回答。


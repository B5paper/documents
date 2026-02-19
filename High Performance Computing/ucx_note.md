# ucx note

## cache

* 简述 ucx 库是干嘛用的

    UCX（Unified Communication X）是一个开源的、用于高性能计算（HPC）和机器学习（ML）等领域的**跨平台通信框架**。它的核心目标是提供**统一、高效、可扩展的通信抽象层**，屏蔽底层不同网络硬件和通信协议的差异。

    **主要设计目标**

    1. **统一接口**  

       提供一致的API（C语言接口），让应用程序可以用相同的方式使用多种网络技术（如InfiniBand、RoCE、TCP、共享内存等）。

    2. **高性能**  

       直接暴露底层硬件的特性（如零拷贝、GPU Direct RDMA），最小化通信开销。

    3. **跨平台**  

       支持多种架构（x86、ARM、PowerPC）和操作系统（Linux、Windows）。

    4. **可扩展性**  

       支持大规模并行计算（如MPI应用）和分布式机器学习框架（如TensorFlow、PyTorch）。

    **核心功能**

    - **多网络支持**  

        自动检测并利用可用的高速网络硬件（如InfiniBand、RoCE），同时兼容标准TCP/IP。

    - **通信模式**  

        支持点对点通信、集合操作、远程内存访问（RMA）、原子操作等。

    - **内存与加速器集成**  

        支持GPU内存的直接通信（GPUDirect RDMA），减少CPU介入和数据拷贝。

    - **与上层库集成**  

        作为底层通信层，被许多高性能库使用：

        - **MPI实现**（如OpenMPI、MPICH）
        - **分布式深度学习框架**（如NCCL、RAPIDS）
        - **数据科学工具**（如Dask、Apache Spark的加速插件）

    **典型应用场景**

    1. **超算中心的大规模模拟**  

       在InfiniBand集群上为MPI应用提供低延迟通信。

    2. **分布式机器学习训练**  

       加速多GPU或多节点间的梯度同步（例如通过NCCL底层调用UCX）。

    3. **异构计算通信**  

       实现CPU与GPU、GPU与GPU之间的直接数据交换。

    4. **云环境高性能通信**  

       在云服务器（如AWS/Azure的RDMA实例）上构建低延迟应用。

    **简单例子**

    ```c
    // UCX的简易点对点通信流程（伪代码）
    ucp_init();                    // 初始化UCX上下文
    ucp_worker_create();           // 创建通信工作线程
    ucp_ep_create();               // 创建端点（对等连接）
    ucp_tag_send()/ucp_tag_recv(); // 发送/接收数据（基于标签通信）
    ```

    **与相关技术的关系**

    - **vs. MPI**  

        UCX不是MPI，但可以作为MPI的底层传输层（OpenMPI已集成UCX）。

    - **vs. NCCL**  

        NCCL是NVIDIA的多GPU集合通信库，其底层可通过UCX实现跨节点通信。

    - **vs. 直接使用RDMA**  

        UCX封装了RDMA编程的复杂性，提供更易用的API。

    **总结**

    UCX相当于高性能计算领域的“通信中间件”，让应用程序无需针对特定网络硬件编程，就能充分发挥高速网络的性能。它已成为现代HPC和AI基础设施中**关键的通信基础组件**。

    如果想进一步了解，可以查看其官网：[OpenUCX](https://www.openucx.org/)

* 简述 ucx 的安装和用法

    我将从**安装**和**基本用法**两个方面简述 UCX。

    **一、安装 UCX**

    **1. 前提条件**

    - **Linux 系统**（推荐 Ubuntu/CentOS）

    - GCC 编译器（≥ 4.9）

    - 可选但重要的硬件支持：

        - **InfiniBand/RoCE**：需要安装驱动（如 `libibverbs-dev`）

        - **GPU 支持**：需要 CUDA 和 GPU Direct RDMA 驱动

        - **NUMA 感知**：需要 `numactl-dev`

    **2. 安装方式**

        **方式一：包管理器安装（简单）**

        ```bash
        # Ubuntu/Debian
        sudo apt install libucx-dev ucx ucx-rdmacm

        # CentOS/RHEL
        sudo yum install ucx ucx-devel
        ```

        *注意：仓库版本可能较旧，缺少最新特性*

        **方式二：源码编译（推荐，功能完整）**

        ```bash
        # 1. 下载源码
        git clone https://github.com/openucx/ucx.git
        cd ucx

        # 2. 配置（典型选项）
        ./autogen.sh
        ./configure \
            --prefix=/usr/local \
            --with-cuda=/usr/local/cuda \      # 启用GPU支持
            --with-rdmacm \                    # 启用RDMA CM
            --enable-mt \                      # 多线程支持
            --enable-optimizations

        # 3. 编译安装
        make -j$(nproc)
        sudo make install

        # 4. 设置环境变量
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
        export PATH=/usr/local/bin:$PATH
        ```

        **方式三：通过 Spack 安装**

        ```bash
        spack install ucx +cuda +gdrcopy +knem +xpmem
        ```

    **二、验证安装**

    ```bash
    # 查看UCX信息
    ucx_info -v          # 版本信息
    ucx_info -d          # 检测到的设备
    ucx_info -b          # 构建配置
    ucx_info -c          # 配置参数

    # 测试性能
    ucx_perftest         # 性能测试工具
    ```

    **三、基本用法**

    **1. 作为应用程序的直接通信库**

    UCX 提供 C API，典型使用流程：

    ```c
    // 简化的UCX点对点通信示例
    #include <ucp/api/ucp.h>

    int main() {
        ucp_context_h context;
        ucp_worker_h worker;
        ucp_ep_h endpoint;
        
        // 1. 初始化配置
        ucp_config_t *config;
        ucp_config_read(NULL, NULL, &config);
        
        // 2. 创建上下文
        ucp_params_t params = {
            .field_mask = UCP_PARAM_FIELD_FEATURES,
            .features = UCP_FEATURE_TAG | UCP_FEATURE_RMA,
        };
        ucp_init(&params, config, &context);
        
        // 3. 创建工作线程
        ucp_worker_params_t worker_params = {
            .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
            .thread_mode = UCS_THREAD_MODE_SINGLE,
        };
        ucp_worker_create(context, &worker_params, &worker);
        
        // 4. 创建端点（连接远程）
        ucp_ep_params_t ep_params = {
            .field_mask = UCP_EP_PARAM_FIELD_SOCK_ADDR,
            .sockaddr = { /* 设置对端地址 */ },
        };
        ucp_ep_create(worker, &ep_params, &endpoint);
        
        // 5. 发送数据（标签通信）
        char send_buffer[100] = "Hello UCX";
        ucp_tag_send_nb(endpoint, send_buffer, 100,
                        ucp_dt_make_contig(1), 0,
                        (ucp_send_callback_t)send_callback);
        
        // 6. 接收数据
        char recv_buffer[100];
        ucp_tag_recv_nb(worker, recv_buffer, 100,
                        ucp_dt_make_contig(1), 0, 0,
                        (ucp_tag_recv_callback_t)recv_callback);
        
        // 7. 等待操作完成
        while (!completed) {
            ucp_worker_progress(worker);
        }
        
        // 8. 清理
        ucp_ep_destroy(endpoint);
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return 0;
    }
    ```

    **2. 作为 MPI 的底层传输层**

    ```bash
    # 以 OpenMPI 为例，编译时启用 UCX 支持
    ./configure --with-ucx=/usr/local

    # 运行 MPI 程序时自动使用 UCX
    mpirun -n 2 --mca pml ucx --mca osc ucx ./my_mpi_program

    # 显示使用的 UCX 传输
    export UCX_LOG_LEVEL=info
    mpirun -n 2 ./program
    ```

    **3. 用于分布式深度学习（通过 NCCL）**

    ```bash
    # NCCL 会自动检测 UCX 用于节点间通信
    export NCCL_IB_HCA=mlx5           # 指定 InfiniBand 设备
    export NCCL_SOCKET_IFNAME=eth0    # 指定 TCP 网络接口
    export NCCL_DEBUG=INFO

    # 运行 PyTorch 分布式训练
    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=4 \
        --node_rank=$RANK \
        train.py
    ```

    **4. 命令行工具使用**

    **性能测试工具**

    ```bash
    # 在两个终端中运行（服务器端和客户端）
    # 终端1（服务器）：
    ucx_perftest -s -t tag_bw -n 1000000

    # 终端2（客户端）：
    ucx_perftest server_hostname -t tag_bw -n 1000000
    ```

    **基准测试类型**

    ```bash
    # 带宽测试
    ucx_perftest -t tag_bw

    # 延迟测试
    ucx_perftest -t tag_lat

    # 流测试
    ucx_perftest -t stream_bw
    ```

    **四、关键环境变量配置**

    *网络选择**

    ```bash
    # 强制使用 TCP（覆盖自动选择）
    export UCX_NET_DEVICES=eth0
    export UCX_TLS=tcp

    # 使用 InfiniBand
    export UCX_NET_DEVICES=mlx5_0:1
    export UCX_TLS=rc

    # 使用共享内存（单节点内）
    export UCX_TLS=shm
    ```

    **性能调优**

    ```bash
    # 调整内存注册缓存
    export UCX_MAX_RNDV_RAILS=2

    # 启用 GPU Direct RDMA
    export UCX_IB_GPU_DIRECT_RDMA=yes

    # 设置传输大小阈值
    export UCX_TAG_RNDV_THRESH=8192
    ```

    **调试**

    ```bash
    # 日志级别
    export UCX_LOG_LEVEL=info    # debug/info/warn/error/fatal

    # 性能计数
    export UCX_STATS_DEST=stdout
    export UCX_STATS_TRIGGER=exit
    ```

    **五、与上层框架集成示例**

    **1. Dask + UCX**

    ```python
    # 启动 UCX 通信的 Dask 集群
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    cluster = LocalCUDACluster(
        protocol="ucx",           # 使用 UCX
        enable_tcp_over_ucx=True,
        enable_infiniband=True,
        enable_nvlink=True
    )
    client = Client(cluster)
    ```

    **2. Apache Spark 3.0+**

    ```bash
    # 使用 UCX 加速 Spark Shuffle
    export SPARK_SHUFFLE_UCX_ENABLE=true
    export SPARK_SHUFFLE_UCX_DRIVER_PORT=13337
    ```

    **常见问题**

    **Q: 如何确认 UCX 正在使用 RDMA？**

    ```bash
    UCX_LOG_LEVEL=info ucx_info -d | grep "transport: rc"
    # 如果看到 "rc"（Reliable Connected），说明正在使用 InfiniBand RDMA
    ```

    **Q: 多网卡环境下如何指定？**

    ```bash
    # 指定特定设备
    export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1

    # 或通过设备地址指定
    export UCX_NET_DEVICES=addr=192.168.1.10,addr=192.168.2.10
    ```

    **Q: 简单测试是否工作**

    ```bash
    # 在两台机器上测试
    # 机器1: ucx_perftest -s
    # 机器2: ucx_perftest <机器1IP>
    ```

    **总结**

    UCX 的安装和用法可以根据需求选择不同层次：

    1. **终端用户**：通过包管理器安装，通过环境变量配置
    2. **开发者**：源码编译，使用 C API 直接编程
    3. **框架用户**：配置 MPI/NCCL/Dask 等框架使用 UCX 作为后端

    其核心优势是**自动选择最优传输**，但在高性能场景下，手动调优环境变量可以进一步提升性能。

## topics

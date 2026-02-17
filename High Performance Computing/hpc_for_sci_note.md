# hpc for sci note

## cache

* mpi 是一个分布式计算工具，但是mpi的编程仍比较复杂，是否有基于 mpi 的库或工具可以实现一些算法，比如矩阵乘法之类的

    是的，有许多基于 MPI 的高级库和工具可以简化并行算法的实现，特别是像矩阵乘法这样的常见操作。以下是一些主要的库和工具：

    1. **并行线性代数库**

        * PETSc

            - **用途**：大规模科学计算，包含丰富的线性代数、非线性求解器、ODE 求解等

            - **特点**：支持稀疏和稠密矩阵，自动处理数据分布

            ```c
            // PETSc 矩阵乘法示例（简化）
            MatMatMult(A, B, MAT_INITIAL_MATRIX, 1.0, &C);
            ```

        * ScaLAPACK

            - **经典分布式线性代数库**
            - 专门为分布式内存系统设计
            - 包含矩阵乘法（PDGEMM）、线性系统求解、特征值计算等
            - 使用二维块循环数据分布

        * Elemental

            - **现代分布式线性代数库**
            - 更灵活的数据分布策略
            - 支持稠密和稀疏矩阵操作

    2. **张量计算库**

        * Cyclops Tensor Framework (CTF)

            - 支持任意维度的张量操作
            - 自动优化通信和数据分布
            - 包含矩阵乘法作为特例

            ```python
            # CTF Python 示例
            import ctf
            A = ctf.random.random([n, n])
            B = ctf.random.random([n, n])
            C = ctf.dot(A, B)  # 自动并行
            ```

    3. **基于 MPI 的框架**

        * DPLASMA

            - 用于多核集群的稠密线性代数库

            - 使用动态任务调度，提高资源利用率

        * HPX

            - 并行运行时系统，兼容 MPI

            - 支持异步任务和 future 模式

    4. **Python 高级接口**

        * mpi4py + NumPy/SciPy

            - 通过 MPI 通信 NumPy 数组

            - 可构建自定义并行算法

                ```python
                from mpi4py import MPI
                import numpy as np

                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()

                # 分布矩阵块，然后实现并行乘法
                local_A = np.random.rand(local_size, n)
                local_result = np.dot(local_A, B_part)
                ```

        * Dask + MPI

            - Dask 的分布式调度器支持 MPI 后端

            - 可自动并行化数组和矩阵运算

            ```python
            import dask.array as da
            from dask_mpi import initialize

            initialize()
            x = da.random.random((10000, 10000), chunks=(1000, 1000))
            y = da.random.random((10000, 10000), chunks=(1000, 1000))
            z = da.dot(x, y)  # 自动分布式计算
            ```

    5. **专用矩阵乘法工具**

        * COSMA

            - 专门优化的分布式矩阵乘法库

            - 最小化通信开销，理论最优通信量

        * SLATE

            - 下一代分布式线性代数库

            - 目标替代 ScaLAPACK，支持 GPU

    实用建议

    1. **对于大多数应用**：使用 **PETSc** 或 **ScaLAPACK** 最稳妥
    2. **需要现代特性**：考虑 **Elemental** 或 **SLATE**
    3. **Python 环境**：使用 **mpi4py + NumPy** 或 **Dask**
    4. **张量计算**：使用 **CTF**
    5. **教学和学习**：从 **mpi4py** 开始，理解基本概念后使用高级库

    简单示例：使用 mpi4py 实现矩阵乘法

    ```python
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 假设矩阵大小 N x N
    N = 1000
    rows_per_proc = N // size

    # 每个进程创建自己的数据块
    local_A = np.random.rand(rows_per_proc, N)
    B = np.random.rand(N, N) if rank == 0 else None
    B = comm.bcast(B, root=0)

    # 本地计算
    local_C = np.dot(local_A, B)

    # 收集结果（如果 root 需要完整结果）
    if rank == 0:
        C = np.empty((N, N), dtype=np.float64)
    else:
        C = None

    comm.Gather(local_C, C, root=0)
    ```

    这些工具大大降低了 MPI 编程的复杂性，让你能够专注于算法本身而不是通信细节。选择哪个工具取决于你的具体需求、编程语言偏好和计算规模。

* 基于 mpi 的排序、图算法、最优化、微分方程求解、流体力学仿真等任务的分布式计算框架

    对于这些不同领域的计算任务，都有专门的基于 MPI 的高级库和工具。让我为你详细介绍：

    1. **排序算法**

        **MPI-Sort 库**

        - **MPI-Sort**: 专门优化的并行排序库

        - **PSort**: 包含多种并行排序算法的实现

        **高效实现模式**

        ```c
        // 常见模式：采样 + 划分 + 局部排序
        MPI_Allgather(samples, ...);  // 收集样本确定划分点
        MPI_Alltoallv(data, ...);     // 重新划分数据
        qsort(local_data);            // 本地排序
        ```

        **外部库**

        - **Boost.MPI**: 包含并行排序算法

        - **Thrust + MPI** (CUDA 环境下): GPU 加速排序

    2. **图算法**

        **专用图计算框架**

        **分布式图数据库/框架**

        - **PGAS 模型库**:

            - **Sector/Sphere**: 支持图计算
            - **Giraph++**: 基于 MPI 的图处理系统

        **图算法库**

        - **PBGL** (Parallel BGL): Boost Graph Library 的并行版本

            ```cpp
            #include <boost/graph/distributed/mpi_process_group.hpp>
            // 支持并行 BFS、DFS、最短路径等
            ```

        - **ParMAT**: 多核/多节点图算法库

        - **GAP** (Graph Algorithm Platform): 包含多种并行图算法

        **通用算法实现**

        ```c
        // 并行 BFS 示例结构
        while (!global_empty) {
            process_local_frontier();  // 处理本地边界节点
            exchange_boundary_nodes(); // MPI 交换边界节点
            update_frontier();         // 更新边界集合
        }
        ```

    3. **最优化问题**

        **非线性优化**

        - **TAO** (Toolkit for Advanced Optimization):

          - 基于 PETSc，支持大规模非线性优化

          - 包含梯度法、牛顿法、内点法等

          ```c
          TaoCreate(MPI_COMM_WORLD, &tao);
          TaoSetObjectiveAndGradient(tao, FormFunctionGradient, NULL);
          TaoSolve(tao);
          ```

        - **IPOPT + MPI**: 大规模非线性优化

        **线性规划**

        - **PIPS** (Parallel Interior Point Solver):
          - 专门用于大规模线性规划
        - **COIN-OR**: 开源优化库，部分支持 MPI

        **全局优化**

        - **pagmo2**: 并行全局优化框架
        - **ParOpt**: 分布式优化框架

    4. **微分方程求解**

        **偏微分方程 (PDE)**

        **有限元法**

        - **deal.II**: 强大的有限元库

          ```cpp
          // 支持自适应网格、多物理场
          parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
          ```

        - **FEniCS**: Python/C++ 混合，自动并行

          ```python
          from dolfin import *
          mesh = Mesh(MPI_COMM_WORLD, "mesh.xml")
          # 自动分布式计算
          ```

        - **LibMesh**: 侧重于非结构网格

        - **MFEM**: 轻量高效，支持 GPU

        **有限差分/体积法**

        - **SAMRAI**: 结构化自适应网格

        - **OpenFOAM**: 计算流体力学，内置 MPI 并行

        **常微分方程 (ODE)**

        - **SUNDIALS**: 包含 CVODE、IDA 等求解器

          ```c
          CVodeInit(cvode_mem, f, t0, y, SUNMPI_COMM_WORLD);
          ```

        - **PETSc TS** (Time Stepping): 丰富的时步积分器

    5. **流体力学仿真**

        **专业 CFD 软件/库**

        **开源框架**

        - **OpenFOAM**:
          - 工业标准开源 CFD
          - 完全基于 MPI 并行

          ```
          decomposePar          # 划分网格
          mpirun -np 32 simpleFoam -parallel
          ```

        - **Nektar++**: 谱/hp 元方法

        - **Code_Saturne**: 工业级 CFD

        **专用求解器**

        - **LBM** (格子玻尔兹曼):
          - **Palabos**: 并行 LBM 框架
          - **waLBerla**: 多物理场 LBM

        - **SPH** (光滑粒子流体动力学):
          - **DualSPHysics**: GPU + MPI 混合并行

        **湍流模拟**

        - **Nek5000/NekRS**: 高保真湍流模拟

        - **ChannelFlow**: 专门用于槽道湍流

    6. **多物理场仿真**

        **耦合框架**

        - **preCICE**: 多物理场耦合库

          ```cpp
          precice::SolverInterface interface(participantName, configFile, rank, size);
          ```

        - **MUI** (Multiphysics Universal Interface)

        - **OpenPALM**: 动态耦合框架

    7. **机器学习/深度学习**

        **分布式训练**

        - **Horovod**:

          ```python
          import horovod.torch as hvd
          hvd.init()
          torch.cuda.set_device(hvd.local_rank())
          # 自动分布式优化
          ```

        - **DeepSpeed**: 微软的大规模训练框架

    选择建议指南

    | 任务类型 | 推荐工具 | 学习曲线 | 适用规模 |
    |---------|---------|---------|---------|
    | 通用排序 | MPI-Sort | 低 | 中小型 |
    | 图计算 | PBGL/GAP | 中 | 大型图 |
    | 优化问题 | TAO/IPOPT | 中 | 大规模 |
    | 有限元 | deal.II/FEniCS | 中高 | 超大网格 |
    | 流体力学 | OpenFOAM | 高 | 工业级 |
    | 多物理场 | preCICE | 中 | 耦合问题 |

    实用工作流程

    1. **先寻找现有库**：大多数问题都有成熟解决方案
    2. **评估复杂度**：简单问题用 MPI 原生，复杂问题用高级库
    3. **混合编程**：MPI + OpenMP + CUDA 混合并行
    4. **使用现代 C++/Python**：减少代码复杂度

    示例：FEniCS 并行求解 PDE

    ```python
    from dolfin import *
    import mpi4py.MPI as MPI

    # 自动 MPI 并行
    mesh = RectangleMesh(MPI_COMM_WORLD, 
                        Point(0, 0), Point(1, 1), 50, 50)
    V = FunctionSpace(mesh, 'P', 1)

    # 分布式定义问题
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = Constant(1.0) * v * dx

    # 并行求解
    u = Function(V)
    solve(a == L, u)

    # 每个进程处理自己的子域
    ```

    学习资源

    1. **书籍**:
       - "Using Advanced MPI"
       - "Parallel Programming with MPI"

    2. **课程**:
       - 瑞士国家超算中心 (CSCS) 培训
       - 德国于利希研究中心 MPI 课程

    3. **代码示例**:
       - **MPI 官方示例**
       - **各库的 tutorials 目录**

    这些工具极大地简化了并行编程，让你能专注于算法和物理模型，而不是通信细节。根据你的具体领域选择合适的库可以大大提高开发效率。

## topics

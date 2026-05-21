## cache


* 详细讲讲 mpi4py 的用法

    `mpi4py` 是 Python 中进行**并行计算**的核心库，它为消息传递接口（MPI, Message Passing Interface）提供了 Python 风格的封装。它非常适合用来在多核电脑或多台服务器组成的集群上加速计算。

    Usage:

    ```bash
    mpiexec -n 4 python script.py  # 启动 4 个进程并行运行
    ```

    * 基础用法：点对点通信 (Point-to-Point)

        点对点通信是指**一个进程向另一个进程发送数据**。

        `mpi4py` 提供两套方法：

        1. **小写字母方法 (`send`, `recv`)：** 使用 Python 默认的 `pickle` 序列化，可以发送**任意 Python 对象**（字典、列表、自定义类），但速度较慢。

        2. **大写字母方法 (`Send`, `Recv`)：** 传输底层的连续内存块（如 **NumPy 数组**），速度极快，接近 C 语言的性能。

        **示例：使用 `send` 和 `recv` 传输 Python 字典**

        ```python
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            # 进程 0 负责准备数据并发送
            data = {'key1': 'value1', 'key2': [1, 2, 3]}
            print(f"进程 {rank} 正在发送数据...")
            comm.send(data, dest=1, tag=11)
        elif rank == 1:
            # 进程 1 负责接收数据
            data = comm.recv(source=0, tag=11)
            print(f"进程 {rank} 成功接收到数据: {data}")
        ```

        **示例：使用 `Send` 和 `Recv` 高效传输 NumPy 数组**

        注意：使用大写方法时，接收方必须**提前初始化好一个同样大小和类型的空数组**来接收数据。

        ```python
        import numpy as np
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            # 创建一个 float64 类型的数组
            data = np.arange(10, dtype=np.float64)
            comm.Send(data, dest=1, tag=12)
            print(f"进程 0 已发送 NumPy 数组: {data}")
        elif rank == 1:
            # 接收方必须提前开辟好内存空间
            data = np.empty(10, dtype=np.float64)
            comm.Recv(data, source=0, tag=12)
            print(f"进程 1 已接收 NumPy 数组: {data}")
        ```

    * 进阶用法：集体通信 (Collective Communication)

        当需要所有进程一起参与数据交换时，点对点通信就显得太繁琐了。MPI 提供了集体通信函数，同样分为小写（任意对象）和大写（NumPy 数组）两套。

        常用的集体通信操作有四种：

        1. 广播 (Broadcast)

            将进程 0（Root 进程）的数据**复制发送给所有进程**。

            ```python
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if rank == 0:
                data = {'shared_config': True, 'threshold': 0.85}
            else:
                data = None

            # 所有进程都必须执行这行代码
            data = comm.bcast(data, root=0)
            print(f"进程 {rank} 拿到了广播数据: {data}")
            ```

        2. 散播 (Scatter)

            将进程 0 的一个列表/数组**切分成多份，均匀分发给各个进程**。

            ```python
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if rank == 0:
                # 假设有 4 个进程，列表长度也要是 4 的倍数
                data = [i**2 for i in range(size)]
                print(f"主进程准备散播的整份数据: {data}")
            else:
                data = None

            # 散播：每个进程分到列表中的一个元素
            my_subdata = comm.scatter(data, root=0)
            print(f"进程 {rank} 领到了属于自己的数据: {my_subdata}")
            ```

        3. 收集 (Gather)

            与 Scatter 相反，将各个进程独立计算出的结果**重新组合，打包汇总到进程 0**。

            ```python
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # 每个进程自己算一个数字
            my_result = rank * 10

            # 大家都把自己的数字交上去
            all_results = comm.gather(my_result, root=0)

            if rank == 0:
                print(f"主进程 0 收集到的最终汇总列表: {all_results}")
            ```

        4. 规约 (Reduce)

            将各个进程的数据通过某种**数学操作（如求和、求最大值、求乘积）进行压缩聚合**，最终把结果给进程 0。

            ```python
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            my_value = rank + 1  # 假设 4 个进程的值分别是 1, 2, 3, 4

            # 计算所有进程 value 的总和 (MPI.SUM)
            total_sum = comm.reduce(my_value, op=MPI.SUM, root=0)

            if rank == 0:
                print(f"所有进程的值相加的总和为: {total_sum}") # 1+2+3+4 = 10
            ```

            > **提示：** 常用的规约操作（`op`）包括：`MPI.SUM`（求和）、`MPI.PROD`（求乘积）、`MPI.MAX`（最大值）、`MPI.MIN`（最小值）。

    4. 常见坑点与注意事项

        * **死锁 (Deadlock)：**

            如果进程 0 在等待进程 1 发送数据，而进程 1 也在等待进程 0 发送数据，程序就会永远卡死。编写 `send`/`recv` 时，一定要确保发送和接收能够配对上。

        * **代码越界：**

            集体通信函数（如 `bcast`, `scatter`, `gather`）**必须被该通信子里面的所有进程同时调用**。不能把它写在诸如 `if rank == 0:` 的条件分支内部，否则会导致程序卡死。

        * **非阻塞通信：**

            如果想在传输数据的同时让 CPU 继续做其他计算，可以使用非阻塞版本的函数 `isend` 和 `irecv`。它们会立即返回一个 `Request` 对象，之后通过调用 `req.wait()` 来确保数据传输完成。

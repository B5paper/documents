# rpc note

## cache

* SSH (Paramiko/Fabric)

* 进阶小贴士：

    既然你已经在虚拟机里跑通了，如果你之后要处理超大规模的 numpy 数组（比如几百 MB 的矩阵），可以考虑以下优化方向：

    * 压缩传输：

        在 obtain 之前，在客户端先 zlib.compress(data.tobytes())，服务端再还原。这能显著减少网络 I/O 时间。

    * 共享内存 (Shared Memory)：
    
        如果 A 和 B 其实是在同一台物理机的不同容器或进程中，可以用 multiprocessing.shared_memory 配合 RPC。RPC 只传内存地址的名字，数据根本不走网络。

* pickle 是 Python 里的“核武器”。如果 RPyC 默认开启 allow_pickle 并在 0.0.0.0 监听，任何能访问你端口的人，都可以发送一个精心构造的字节流。你的服务器在执行 pickle.loads() 的瞬间，对方就能获得你机器的 Root 权限。

    如果这个服务暴露在公网（或者一个不安全的局域网），黑客只需要用 scapy 这种工具往你的 18861 端口发一段包含 os.system('rm -rf /') 的 Pickle 字节流，你的服务端进程会毫无怨言地执行它。

* protocol_config

    ```py
    if __name__ == "__main__":
    # 正确做法：把配置直接喂给 Server
    safe_config = {
        "allow_pickle": True,
        "allow_all_attrs": True
    }
    server = ThreadedServer(
        MyService, 
        port=18861, 
        protocol_config=safe_config  # 显式传入配置
    )
    server.start()
    ```

* RPyC 的“底层潜规则”

    1. exposed_ 前缀的真相（安全过滤）

        这是 RPyC 的一种命名空间保护机制。

        * 服务端规则：为了防止客户端通过 RPC 恶意调用服务端的内置方法（比如 __del__ 或 __init__），RPyC 规定：只有以 exposed_ 开头的函数，才允许被远程调用。

        * 客户端规则：客户端在调用时，必须省略这个前缀。当你调用 conn.root.process_array() 时，RPyC 的内核会自动在服务端寻找名为 exposed_process_array 的方法。

        如果你把服务端的 exposed_ 删掉，客户端就再也搜不到这个函数了。这相当于一种简单的 public/private 访问控制。

    2. root 到底是什么？

        root 是连接到远程服务的入口对象（也就是你定义的 MyService 实例）。

        在 RPyC 的协议里：

        * 当你执行 conn = rpyc.connect(...) 时，两端握手成功。

        * 服务端会实例化一个 MyService 类。

        * 客户端的 conn.root 实际上是一个代理对象（Proxy），它指向服务端那个 MyService 的实例。

        你可以把 conn.root 理解为**“那台机器上那个服务的根节点”**。所有你在服务端写的 exposed_xxx 函数，都会被挂载到这个 root 对象下面。

    3. 为什么不直接调，非要经过 root？

        这是为了支持多服务模型。

        虽然你现在只写了一个 MyService，但 RPyC 允许一个连接同时暴露多个不同的服务对象。root 默认指向主服务，确保路径清晰。

* 有哪些和 root 同级的概念？

    在 RPyC 的连接对象（conn）中，root 确实是最常用的入口，但从连接管理和协议控制的角度看，确实有几个和它同级（或者说在同一个层级由 conn 对象持有）的关键概念。

    理解了这些，你就能明白 RPyC 是怎么在底层“倒腾”数据的。

    1. conn.modules (最强大的同级属性)

        如果你开启了经典模式（Classic），conn.modules 是一个极其逆天的存在。它让你像操作本地模块一样操作远程模块。

        * 用法：conn.modules.os 或 conn.modules.numpy。

        * 对比：root 访问的是你自定义的业务逻辑（MyService），而 modules 访问的是远程机器整个 Python 环境。

        * 权限：它对权限要求极高，通常需要你在配置里显式允许。

    2. conn.builtins

        这也是一个代理对象，指向远程 Python 的内建命名空间（即 dir(__builtins__) 里的东西）。

        * 用法：如果你想在远程机器上调用 len()、open() 或 eval()，可以通过 conn.builtins.eval("1+1")。

        * 场景：当你想执行一些不属于任何模块的 Python 原生操作时。

    3. conn.root (深入理解)

        你已经知道了它是 Service 的实例。但要注意：root 是延迟加载的。
    当你第一次访问 conn.root 时，客户端才会请求服务端去创建那个 MyService 实例。
    4. conn._config (真正的配置所在地)

        这就是你刚才质问我的那个配置。

        在 conn 对象内部，确实有一个 _config 属性（注意带下划线）。这才是 RPyC 运行时真正叼的配置。

        * 真相：当你通过 rpyc.connect(..., config=my_config) 连接时，你的 my_config 会被合并进 conn._config。

        * 验证：你可以试着打印 print(conn._config)，你会看到一大堆默认值，包括你苦苦寻找的 allow_pickle、sync_request_timeout 等。

    5. conn.async_ (异步控制)

        如果你不想像现在这样等 process_array 执行完才返回（同步阻塞），你可以用 async_。

        * 用法：res = rpyc.async_(conn.root.process_array)(data)。

        * 状态：此时 res 会立即返回一个 AsyncResult 对象，你可以之后通过 res.ready 来判断算完没。

    总结对照表

    | 属性 | 作用范围 | 相当于 SSH 里的... |
    | - | - | - |
    | conn.root | 访问你写的 业务代码 (MyService) | 执行你写的远程脚本 |
    | conn.modules | 访问远程 第三方库 (numpy, os) | 远程库的交互式命令行 |
    | conn.builtins | 访问远程 原生函数 (print, eval) | 调用 Python 基础内核 |
    | conn._config | 控制这根 网线的属性 (超时、权限、序列化) | SSH 的 ssh_config 配置 |

* 使用 pickle

    `server.py`:

    ```py
    import rpyc
    from rpyc.utils.server import ThreadedServer
    from rpyc.utils.classic import obtain

    class MyService(rpyc.Service):
        def exposed_process_array(self, arr):
            print(arr)
            print(type(arr))
            try:
                print(f"收到数组，形状为: {arr.shape}")
            except Exception:
                print('fail to print arr.shape')

            arr = obtain(arr)
            print(arr)
            print(type(arr))
            print(f"收到数组，形状为: {arr.shape}")

            return arr * 2

    if __name__ == "__main__":
        config = {
            "allow_pickle": True,
        }
        server = ThreadedServer(MyService, hostname='localhost', port=18861, protocol_config=config)
        server.start()
    ```

    `client.py`

    ```py
    import rpyc
    import numpy as np
    from rpyc.utils.classic import obtain

    config = {
        "allow_pickle": True,
    }

    conn = rpyc.connect("localhost", 18861, config=config)
    data = np.array([1, 2, 3])
    result = conn.root.process_array(data)

    print(type(result))
    print(result)

    result = obtain(result)
    print(type(result))
    print(result)
    ```

    run:

    * server

        `python server.py`

    * client

        `python client.py`

    output:

    * server

        ```
        [1 2 3]
        <netref class 'rpyc.core.netref.numpy.ndarray'>
        fail to print arr.shape
        [1 2 3]
        <class 'numpy.ndarray'>
        收到数组，形状为: (3,)
        ```

    * client

        ```
        <netref class 'rpyc.core.netref.numpy.ndarray'>
        [2 4 6]
        <class 'numpy.ndarray'>
        [2 4 6]
        ```

    注：

    1. client 端指定`"allow_pickle": True`后，server 端可以正常 obtain。同样地，server 端指定`"allow_pickle": True`后，client 端可以正常 obtain。否则会报错

    1. server 端 try 块报错，说明不打开`allow_public_attrs`无法访问到 netref 的属性

    1. 如果`type(arr)`显示的是 ndarray，那么说明是一个本地对象，数据都在本地。如果显示的是 netref，那么要么需要 obtain，那么需要打开 allow_public_attrs，才能访问到其属性。

        如果不开权限，不影响使用索引访问，比如`arr[0]`。只不过每次索引都要通过网络传输一次数据。

* 使用 python rpyc 实现 rpc 调用

    install:

    `pip install rpyc`

    `server.py`:

    ```py
    import rpyc
    from rpyc.utils.server import ThreadedServer

    class MyService(rpyc.Service):
        def exposed_process_array(self, arr):
            print(arr)
            print(type(arr))
            print(f"收到数组，形状为: {arr.shape}")
            return arr * 2

    if __name__ == "__main__":
        # localhost, 127.0.0.1, 10.0.2.4
        # 对于不同的 ip，arr 对象的访问权限并无明显差别
        server = ThreadedServer(MyService, hostname='localhost', port=18861)
        server.start()
    ```

    `clinet.py`:

    ```py
    import rpyc
    import numpy as np

    config = {
        "allow_public_attrs": True,
    }

    conn = rpyc.connect("localhost", 18861, config=config)
    data = np.array([1, 2, 3])
    result = conn.root.process_array(data)

    print(result)
    print(type(result))
    ```

    run:

    start server: `python server.py`

    start client: `python client.py`

    output:

    * server

        ```
        (torch) hlc@hlc-VirtualBox:~/Documents/Projects/rpc_test$ python server.py 
        [1 2 3]
        <netref class 'rpyc.core.netref.numpy.ndarray'>
        收到数组，形状为: (3,)
        ```

    * client

        ```
        (torch) hlc@hlc-VirtualBox:~/Documents/Projects/rpc_test$ python client.py 
        [2 4 6]
        <class 'numpy.ndarray'>
        ```

* gRPC（推荐用于生产环境）

    gRPC 使用 Protocol Buffers (protobuf) 作为传输格式。它默认不支持 numpy，但你可以通过将数组转换为字节流（bytes）来传输。

    核心步骤：

    1. 定义 .proto 文件：

        ```Protocol Buffers
        syntax = "proto3";

        service DataService {
          rpc SendArray (ArrayRequest) returns (ArrayResponse);
        }

        message ArrayRequest {
          bytes data = 1;      // 序列化后的数组
          repeated int32 shape = 2; // 形状信息
          string dtype = 3;    // 数据类型
        }

        message ArrayResponse {
          string message = 1;
        }
        ```

    2. 序列化与反序列化：

        发送端： `data = arr.tobytes()`

        接收端： `arr = np.frombuffer(request.data, dtype=request.dtype).reshape(request.shape)`

* 压缩：如果网络带宽是瓶颈，可以在传输前使用 zlib 或 blosc 对 tobytes() 的结果进行压缩。

    ```py
    # 客户端发送
    import numpy as np
    import zlib

    def send_array(conn, arr):
        # 将数组转为字节流并压缩，避免 RPyC 的逐元素通信
        data = zlib.compress(arr.tobytes())
        shape = arr.shape
        dtype = str(arr.dtype)
        return conn.root.process_compressed_array(data, shape, dtype)
    ```

    * 内存视图：在处理超大型数组时，尽量使用 memoryview 以减少内存拷贝。

* 极简 HTTP 接口	FastAPI + Msgpack	虽然不是传统 RPC，但在 Web 背景下传输 Numpy 非常方便。

    ```py
    arr.to_bytes()
    np.frombuffer()
    ```

    bytes 是内置类型，不需要任何特殊权限，不需要 allow_pickle。

* Netref (网络引用)

    当你通过 RPyC 传递一个 numpy 数组时，对方拿到的其实不是一个真正的数组，而是一个代理对象 (Proxy Object)。

    * 占位符：服务端只收到了一个指向客户端内存中该数组的“地址标签”。

    * 按需获取：

        * 如果你访问 arr.shape，RPyC 会发起一个小的网络请求问：“那个数组的 shape 是多少？”

        * 如果你访问 arr[0]，它会发起请求问：“那个数组的第 0 个元素是多少？”

        * 这种“懒加载”模式在处理超大对象且你只需要其中一小部分信息时非常高效。

    **为什么这对 Numpy 很危险？（性能杀手）**

    虽然“按需传输”听起来很智能，但在 numpy 这种密集计算场景下，它是性能噩梦。

    假设你有一个 100×100 的矩阵，你想在服务端计算总和：

    ```Python
    # 如果这是个 RPyC 代理对象
    total = 0
    for row in remote_arr:
        for val in row:
            total += val
    ```

    后果： 这段代码会产生 10,000 次网络请求。因为每一次 val 的提取都是一次 Transfer-on-Read。即使在局域网内（延迟 0.1ms），这段循环也需要 1 秒钟才能跑完，而本地计算只需要微秒级。

    **如何规避“按需传输”？**

    如果你确定需要处理整个数组，你应该强制进行一次性深拷贝 (Deep Copy)，将数据从代理对象转为本地对象：

    方法 A：使用 rpyc.classic.obtain()

    ```Python
    import rpyc
    local_arr = rpyc.classic.obtain(remote_arr) # 一次性把数据全拉回来
    ```

    方法 B：手动转换（最推荐）

    在发送前将其转为 bytes，接收后再 np.frombuffer。这样就跳过了 Netref 机制，直接走的是值的传输。

    总结一下： RPyC 默认确实是 Transfer-on-Read，它让远程调用看起来像本地调用一样“优雅”，但如果你在循环里读它，网络延迟会让你非常痛苦。

* 那我如果想机器a向b发送命令，修改的是b环境里的数据呢，类似ssh那种感觉

    如果你追求的是像 SSH 那样远程操控 B 机器的环境、修改 B 机器内存中的数据或调用其本地资源，那么 RPyC 的 "Classic" 模式 或 Paramiko (纯 SSH 实现) 是最匹配的方案。

    在这种场景下，你不再是“传输数据”，而是在“操作远程上下文”。

    1. 方案一：RPyC Classic (最像“远程 Python 解释器”)

        RPyC 有一种模式叫 classic，它允许你直接在 A 机器上操作 B 机器的 内建模块、变量 和 内存。

        服务端 (机器 B)

        只需启动一个通用代理服务：

        ```Bash
        rpyc_classic.py --allow-all-attrs  # 启动经典模式服务
        ```

        客户端 (机器 A)

        你可以在 A 上写代码，直接修改 B 的环境变量或内存对象：

        ```Python
        import rpyc

        # 连接到机器 B
        conn = rpyc.classic.connect("machine_b_ip")

        # 1. 像在 B 本地一样导入模块
        remote_os = conn.modules.os
        remote_np = conn.modules.numpy

        # 2. 修改 B 机器的环境变量
        remote_os.environ["MY_VAR"] = "changed_by_machine_a"

        # 3. 在 B 机器的内存里创建一个 Numpy 数组
        # 注意：这个数组存在 B 的内存里，A 只拿着一个引用
        remote_arr = remote_np.array([10, 20, 30])
        remote_arr[0] = 99  # 直接修改 B 内存中的数据

        print(f"B 机器上的数组第一个元素现在是: {remote_arr[0]}")
        ```

        特点：这种方式是真正的 "Remote Control"。你在 A 写的每一行指令，执行逻辑和数据存储都在 B 上。

    2. 方案二：Fabric / Paramiko (真正的 SSH 协议)

            如果你不仅想改 Python 里的数据，还想改 B 机器的文件、重启服务或执行 Shell 命令，应该用标准的 SSH 库。Fabric 是基于 SSH 的高级封装，非常适合自动化运维。

        示例 (机器 A 控制 B)

        ```Python
        from fabric import Connection

        # 建立 SSH 连接
        c = Connection("user@machine_b_ip")

        # 1. 执行 Shell 命令修改系统环境
        c.run("export DATA_PATH='/tmp/data' && echo $DATA_PATH")

        # 2. 甚至可以远程运行一段 Python 脚本来处理 B 上的 Numpy 数据
        remote_code = """
        import numpy as np
        data = np.load('data.npy')
        data = data * 2
        np.save('data_processed.npy', data)
        """
        c.run(f"python3 -c \"{remote_code}\"")
        ```

    3. 关键区别：你要修改的是什么？

        | 维度 | RPyC Classic | SSH (Fabric/Paramiko) |
        | - | - | - |
        | 操作对象 | 运行中的 Python 对象/内存 | 系统命令、文件、进程 |
        | 交互感 | 像在本地写 Python，实际在远程跑 | 像在写脚本，发送一串命令去执行 |
        | 性能 | 极高（保持长连接，对象操作快） | 较低（每次执行可能有握手开销） |
        | 持久性 | 服务端 Python 进程结束，数据消失 | 修改的是文件或系统状态，永久保留 |

    4. 安全性重要提醒（必读）

        由于你提到这像 SSH：

        * RPyC Classic 没有任何安全防护：它默认允许远程执行任何 Python 代码。千万不要将其暴露在公网，只能在可信局域网或 VPN 下使用。

        * 身份验证：SSH 方案（Fabric）原生支持密钥登录，安全性极高。

* 处理 RPyC 中的中文字符：

    RPyC 在传输数据时，如果涉及中文，Python 2 的 str 和 unicode 转换经常报错。

    尝试在代码开头加入这几行（这是 Python 2 处理中文的“黑魔法”）：
    Python

    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')

* 针对 data 的处理（重中之重）

    你调用的 process_array(data)，如果 data 是 NumPy 数组或 PyTorch 张量：

    千万不要直接传对象：直接传过去，RPyC 会尝试在两端重建对象，Python 2 的 numpy 可能和 Python 3 的版本不兼容。

    改用二进制传参：

    客户端 (Py3): raw_data = data.tobytes() 或 pickle.dumps(data, protocol=2)

    服务端 (Py2): 在 exposed_process_array 内部用 np.frombuffer(data) 或 pickle.loads(data) 还原。

* rpyc 版本对齐

    1. 为什么会报这个错？

        RPyC 协议在版本 4.x 到 5.x 之间发生过重大变化。

        18 这个消息类型代码在较新版本的 RPyC 中存在，但在旧版本（通常是 2.7 常用的 3.x 或 4.x 版本）的解析器里根本不存在。

        当你的 Python 3 客户端发送了一个“版本 5”特有的指令，Python 2 的服务端解析器直接就“懵了”，回了一个无效消息，导致客户端崩溃。

    2. 核心解决方案：对齐 RPyC 版本

        你不需要改变 Python 版本，但你必须强行统一两端的 RPyC 库版本。对于跨 2.7 和 3.x 的情况，4.1.2 通常是最稳定的中介版本。

    `pip install rpyc==4.1.2`

    3. 代码层面的关键兼容补丁

        由于 Python 3 默认传字符串是 unicode，而 Python 2 的 RPyC 4.x 可能会在反序列化时卡死，请在 Python 3 客户端的连接代码里加入一个特殊的“桥接配置”：

        ```Python
        import rpyc

        # 针对 4.1.2 版本的经典跨版本配置
        config = {
            "allow_all_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 60,
            # 强制不进行自动编码转换，防止 Py3 的字符串传过去变成 Py2 无法识别的对象
            "instantiate_custom_exceptions": True,
            "instantiate_oldstyle_exceptions": True,
        }

        conn = rpyc.connect("localhost", 18861, config=config)
        ```

    4. 如果还是不行（最后的绝招）

        如果 4.1.2 依然报错，那是因为 Python 3.12 太新了，对旧版库的某些底层调用（如 inspect 模块）有改动。

        在这种极端情况下，科研环境最常用的“暴力桥接”方案是：不要传复杂的 data 对象，传 JSON 字符串或 Bytes。

            客户端 (Py3):
            Python

            import json
            # 把数组转成 list 再转成字符串传过去
            json_data = json.dumps(data.tolist() if hasattr(data, 'tolist') else data)
            result_raw = conn.root.process_array(json_data)

            服务端 (Py2):
            Python

            def exposed_process_array(self, json_str):
                data = json.loads(json_str) # 在 2.7 环境还原
                # ... 处理 ...
                return json.dumps(res)

* 如果你是做高性能传输，建议考虑：

    msgpack

    ZeroMQ, pyzmq

    或者直接 socket + pickle

    multiprocessing / shared memory

    * 如果是本机进程间通信：

        multiprocessing.shared_memory

        Pipe

        Queue

        比 rpyc 稳定太多。

    * 如果你想要“远程对象调用”的体验：

        Pyro5 （比 rpyc 维护好）

        比 rpyc 维护积极，Python 3.12 适配也更及时。

* rpyc 4.1.5 虽然支持 py2，但是只有 rpyc 6.0 才支持 py 3.12.12，而 rpyc 6.0 增加了新的消息类型，导致与 rpyc 4.1.5 不兼容。

    rpyc 5.x 与 py 3.12.12 也不兼容，因为 py 3.12.12 严格区别 bytes 和 str，而 rpyc 5.x 只接收 str （它认为 bytes 和 str 都是 str）。

    没有试 py 3.11 及以下的版本，因为 vscode py debugger 最低支持到 py 3.9，再往下的 py 版本可能和未来的软件不兼容了。

    因此只能手动把 py2 代码改到 py3。


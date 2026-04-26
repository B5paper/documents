# fnirs note

## cache

* 科研项目前后端分离方案

    这是一个非常典型的**计算密集型任务 vs. 资源受限型环境**的架构问题。将后端拆分为微服务不仅能解决编译环境复杂的痛点，还能实现“算力剥离”。

    以下是为你推荐的几种实现方案，按**从简到繁**排序：

    ---

    **方案一：Docker 镜像化（最推荐的“伪微服务”）**

    与其说是微服务，不如说是**环境解耦**。既然现场编译 Fortran 和 BLAS 环境极度痛苦，你可以直接在本地（或性能更好的机器）构建一个包含所有依赖的 Docker 镜像。

    * **操作思路：**
        1.  编写 `Dockerfile`，基础镜像选 `python:3.9-slim` 或 `ubuntu`。
        2.  在 Dockerfile 中预安装 `gfortran`, `libblas-dev`, `liblapack-dev`。
        3.  在镜像构建阶段完成代码的 `make` 或编译工作。
    * **优点：** 云服务器上只需运行 `docker pull` 和 `docker run`，**无需现场编译**，环境绝对一致。
    * **缺点：** 依然占用云服务器的计算资源，只是解决了“环境部署”难的问题。

    ---

    **方案二：计算离线化 + 异步任务队列 (Task Queue)**

    如果云服务器性能确实带不动计算任务，你需要将“Web 接收请求”和“计算执行”彻底分开。

    * **架构设计：**
        1.  **Web 前端/API（云服务器）：** 只负责接收用户上传的数据，将其存入数据库或 S3，并向**消息队列**（如 Redis 或 RabbitMQ）发送一个任务 ID。
        2.  **计算后端（Worker）：** 你可以在自己的本地高性能 PC、实验室服务器、甚至是另一台按量付费的高配实例上运行 Worker。
        3.  **流程：** Worker 监听队列 -> 取出任务 -> 本地高性能计算 -> 将结果传回数据库 -> Web 端轮询或通过 WebSocket 通知用户。
    * **工具推荐：** Python 生态下的 **Celery** 或 **Huey**。

    ---

    **方案三：Serverless 函数计算 (FaaS)**

    这是最符合“微服务”直觉且成本最低的方案。

    * **操作思路：**
        1.  使用阿里云 FC、腾讯云 SCF 或 AWS Lambda。
        2.  将你的 Fortran/BLAS 逻辑封装成一个函数。这些平台通常支持 **容器镜像部署**，你可以把复杂的依赖打包进去。
    * **优点：** * **按量计费：** 不计算时不花钱，不用扛着昂贵的高性能云服务器。
        * **弹性扩容：** 即使有 100 个人同时算，云厂商会自动开 100个 实例，不会卡死你的 Web 服务器。

    ---

    **方案四：简单的 REST/gRPC 独立服务**

    如果你想把后端做成一个长期运行的微服务，可以独立出一个 **Computation Service**。

    1.  **接口选择：** * 如果数据量小，用 **FastAPI** (Python) 封装一个 REST API 即可。
        * 如果需要传递大规模矩阵或二进制数据，建议用 **gRPC**，它的序列化效率更高。
    2.  **部署位置：** 找一台有公网 IP 且性能尚可的机器作为后端。
    3.  **安全注意：** 既然是私有微服务，记得在 Web 服务器与计算后端之间加入简单的 **Token 校验** 或设置防火墙白名单，防止算力被盗刷。

    ---

    **核心建议**

    * **如果只是部署难：** 选 **方案一 (Docker)**。这能解决 90% 的 Fortran 环境配置地狱。
    * **如果是计算慢：** 选 **方案二 (Celery + 异地 Worker)** 或 **方案三 (Serverless)**。
    * **关于 BLAS 的小 Tip：** 在编译时，推荐使用 **OpenBLAS** 或 **MKL**，并注意设置线程数限制（如 `export OMP_NUM_THREADS=1`），防止在弱鸡云服务器上因竞争多核资源导致系统直接 OOM 或假死。

    **你目前的计算量大概在什么量级？（比如单次运行几秒还是几分钟？）** 这决定了是否有必要引入复杂的异步队列。

* 详细讲讲方案二和方案四

    既然你已经动了“微服务”的念头，说明你不仅想解决**环境部署**（Fortran/BLAS）的问题，还想解决**资源隔离**的问题。

    以下是方案二（异步任务队列）和方案四（独立计算服务）的深度拆解：

    ---

    **方案二：异步任务队列（计算离线化）**

    这种模式的核心思想是：**“Web 服务器只接单，不干活”**。它非常适合耗时较长（超过 5 秒）的科研计算任务。

    1. 架构组件
        
        * **Producer (Web API):** 你的云服务器，用 Flask/FastAPI 编写，负责接收参数。
        * **Broker (消息中介):** 常用 **Redis**。它像一个置物架，Web 端把任务丢上去，计算端来取。
        * **Worker (计算节点):** 部署在你性能强劲的机器上。它通过网络连接 Redis，拿到参数后在本地调用 Fortran 编译好的程序。

    2. 执行流程
        
        1.  用户在网页点击“开始计算”。
        2.  Web 服务器生成一个唯一 `task_id`，把参数塞进 Redis，并立即向用户返回：“任务已提交，ID 为 123”。
        3.  **用户页面开始转圈。**
        4.  远程 Worker 监测到 Redis 有新任务，抓取参数，调用本地的科学计算后端。
        5.  Worker 计算完成后，将结果存入数据库（或写回 Redis）。
        6.  网页端通过 **长轮询 (Polling)** 或 **WebSocket** 发现任务状态变更为“已完成”，展示结果。

    3. 为什么适合你？
        
        * **解耦最彻底：** 你的计算节点甚至不需要公网 IP，只要它能访问到云服务器上的 Redis 即可。
        * **容错性强：** 如果计算过程崩溃，不会导致 Web 网页挂掉。

    ---

    **方案四：独立计算微服务 (REST/gRPC)**

    这种模式是把计算后端包装成一个**远程函数**，Web 服务器通过网络“同步”调用它。

    1. 实现方式
        
        你可以用 Python 的 **FastAPI** 包装你的 Fortran/C 代码。

        * **编译：** 使用 `f2py` (NumPy 自带) 或 `ctypes` 将 Fortran 代码编译成 `.so` 动态链接库。
        * **封装：** 在 Python 中 `import` 编译好的库。
        * **暴露接口：** 编写一个 API 接口，接收 JSON 格式的矩阵数据。

    2. 核心架构：gRPC vs REST
        
        * **REST (JSON over HTTP):** 编写简单，调试方便。但缺点是 JSON 传输大型矩阵（浮点数数组）效率极低，且占用大量内存。
        * **gRPC (Protobuf):** 强烈推荐用于科研项目。它传输的是**二进制流**，速度极快，且强制定义数据类型，非常适合 Fortran 这种对数据格式要求严苛的场景。

    3. 部署细节
        
        你需要将这个微服务部署在一台高性能机器上，并开放一个端口（如 `50051`）。
        * **Web 端代码片段：**
            ```python
            # 伪代码：Web 服务器调用计算服务
            response = requests.post("http://your-high-perf-node:8000/compute", json={"matrix": data})
            result = response.json()
            ```

    ---

    **方案对比：我该选哪个？**

    | 特性 | 方案二：异步任务队列 (Celery/Redis) | 方案四：独立微服务 (API) |
    | :--- | :--- | :--- |
    | **任务时长** | 适合**长耗时**（分钟级/小时级） | 适合**短耗时**（秒级） |
    | **实时性** | 用户需要等待/刷新，非实时 | 实时返回结果 |
    | **复杂度** | 较高（需维护 Redis 和多个 Worker） | 较低（就是一个带 API 的程序） |
    | **硬件要求** | 计算节点**不需要**公网 IP | 计算节点**需要**公网 IP 或内网穿透 |
    | **用户体验** | 适合“提交 -> 邮件/站内信通知”模式 | 适合“输入 -> 立即看结果”模式 |

    ---

    **落地建议**

    1.  **如果你想最快上手：**
        先尝试**方案四**。用 FastAPI 给你的 Fortran 程序套个壳，部署在高性能机器上。如果数据量大，直接传二进制文件或者用 gRPC。
    2.  **如果你打算长期运行，且计算任务很重：**
        选**方案二**。配合 **Docker** 封装 Worker 环境。这样你可以随时在你的笔记本、实验室服务器、云端之间横向扩展多个 Worker，而不需要改动 Web 端的一行代码。

    **你的计算任务单次运行大约需要多久？涉及的数据量（如矩阵大小）大概是多少？** 告诉我这些，我可以帮你定夺具体的协议选择。

* Rpc调用需要考虑多用户吗

    这是一个非常深刻的问题。答案是：**必须考虑，而且这是微服务架构中“最坑”的地方之一。**

    如果你只是简单地把一段 Fortran 代码封装成 RPC 接口，而不考虑多用户并发，你的服务很可能会在两个用户同时点击“计算”时**直接崩溃**或**算出错误的结果**。

    在 RPC 架构下，你需要从以下三个维度来应对多用户问题：

    ---

    1. 并发处理能力（Server 模式）
        
        RPC 服务端本身有不同的工作模式，这决定了它如何处理多个连接：

        * **同步单线程（最危险）：** 如果你的 RPC Server 是同步单线程的，当用户 A 在进行一个耗时 10 秒的计算时，用户 B 的请求会被**阻塞**，直到 A 算完。这在 Web 环境下是不可接受的。
        * **多线程/进程模式：** 像 `gRPC` 或 `FastAPI (uvicorn)` 默认支持多线程或异步。用户 A 和 B 的请求会进入不同的线程处理。
            * **风险：** Fortran 编写的底层库如果使用了全局静态变量或固定的临时文件名，多线程并行执行时会产生**竞态条件 (Race Condition)**，导致计算结果互相覆盖。

    2. 算力资源的抢占（计算瓶颈）
        
        即便你的 RPC 框架支持并发，你的**物理硬件（CPU/内存）**也是有限的。

        * **CPU 爆满：** 科学计算通常是满载运行。如果 4 个用户同时发起任务，而你的服务器只有 4 核，系统会因为频繁的上下文切换导致所有人变慢。
        * **内存崩溃 (OOM)：** 科学计算往往涉及大型矩阵。如果一个任务占用 2GB 内存，5 个用户并发请求可能直接撑爆服务器内存，导致进程被操作系统杀掉。
        * **解决方案：** * 在 RPC 服务端设置**最大并发数 (Max Workers)**。
            * 超过负载的请求直接返回 `429 Too Many Requests` 或 `503 Service Unavailable`。

    3. 数据隔离（最核心的科研逻辑问题）
        
        由于你的后端涉及 Fortran 和 BLAS，这些底层库的调用方式决定了隔离难度：

        * **线程安全问题：** 很多老旧的 Fortran 代码不是线程安全的。如果多个线程同时调用同一个 `.so` 库里的函数，可能会导致内存错误。
        * **解决办法：**
            1.  **进程隔离：** 每个 RPC 请求 Fork 出一个独立的子进程去运行计算任务。进程间内存独立，最安全。
            2.  **工作目录隔离：** 如果你的计算程序需要读写临时文件（如 `input.dat`, `output.txt`），**千万不要用固定文件名**。必须为每个请求创建唯一的临时文件夹（如使用 UUID 命名）。

    ---

    **总结：你应该怎么做？**

    如果你选择 **方案四（RPC/API）**，你需要：
    1.  **限制并发：** 明确你的服务器能承载几个并发任务，并在 RPC 配置中写死。
    2.  **动态路径：** 确保 Fortran 程序处理的所有文件路径都是由 Python 传入的动态参数。
    3.  **状态监控：** 增加一个 `/health` 或 `/status` 接口，让 Web 服务器知道计算后端是否已经过载。

    **如果你发现多用户并发非常严重，请果断转向“方案二（任务队列）”：**
    在方案二中，多用户问题会自动转化为**排队问题**。用户 A、B、C 的任务都会按顺序进入 Redis 队列，Worker 算完一个接一个，永远不会因为抢占资源而崩溃。

    **一句话建议：**
    如果计算耗时超过 10 秒，或者单次计算内存占用很高，**不要用 RPC 强拉，请用队列排队。**

    你现在的 Fortran 代码在运行时，单次会占用多少内存？是否有读写固定文件名的操作？

* 如果我在rpc中写死最多承载几个任务，那么多出的任务是被拒绝，还是drop，还是排队？

    这取决于你如何配置 RPC 服务器的**队列（Queue）**和**拒绝策略（Rejection Policy）**。

    在实际生产中，RPC 框架（如 gRPC 或基于 FastAPI 的 REST API）通常会提供一个“缓冲地带”。当并发请求超过你设定的 `Max Workers` 时，处理流程通常遵循以下三个阶段：

    1. 第一阶段：进入“等候区”（排队）
        
        大多数 RPC 框架都有一个内置的任务队列（Backlog）。
        * **行为：** 如果你有 4 个计算线程，但来了第 5 个请求，该请求不会立即失败，而是会在 TCP/应用层队列里排队。
        * **用户感受：** 用户的网页会一直显示“加载中”，直到前面的任务完成，空出一个 Worker 来处理它。
        * **风险：** 如果队列满了，或者等待时间超过了 Web 服务器配置的 **Timeout（超时时间）**，连接就会断开，用户看到 504 Gateway Timeout。

    2. 第二阶段：队列满后的“拒绝”（Rejected）
        
        当等候区也挤满了人（达到 `Max Connection` 或 `Backlog Limit`），服务器会采取主动拒绝措施。
        * **行为：** 服务器直接向客户端返回一个错误状态码（如 gRPC 的 `RESOURCE_EXHAUSTED` 或 HTTP 的 `429 Too Many Requests`）。
        * **用户感受：** 网页立即弹出提示：“服务器繁忙，请稍后再试”。
        * **优点：** 这种方式比“Drop”好，因为它明确告诉了客户端：我现在没空，你别等了。

    3. 第三阶段：丢弃（Drop）—— 极少见
        
        * **行为：** 服务器不回应，直接关掉 socket 连接。
        * **后果：** 客户端会报错 `Connection Reset`。这通常发生在系统底层崩溃、内核半连接队列溢出或防火墙拦截时。

    ---

    **如果你在 RPC 里“写死”限制，会发生什么？**

    以常用的两种实现方式为例：

    **A. 如果用 gRPC**

    gRPC 允许你配置 `max_concurrent_rpcs`。
    * **逻辑：** 当请求数超过这个值，gRPC 会将后续请求挂起（在流控窗口内等待）。如果等待太久，客户端会收到 `Deadline Exceeded`。
    * **建议：** 必须在客户端（Web 端）设置超时保护，否则 Web 服务器的进程会被这些挂起的 RPC 调用全部占满。

    **B. 如果用 FastAPI + Uvicorn (REST)**

    通过控制 Worker 数量（例如 `--workers 4`）或并发限制。
    * **逻辑：** 超过限制的请求会在 TCP 等待队列中。如果队列满了，新的连接会被拒绝。

    ---

    **核心痛点：RPC 的“假排队” vs 消息队列的“真排队”**

    虽然 RPC 可以排队，但它是一种**“不可见且脆弱”**的排队：
    1.  **连接占用：** 每个排队的请求都占着一个网络连接和 Web 服务器的一个等待进程。
    2.  **超时风险：** 科研计算往往很慢。如果前两个人在算大数据集，后面排队的人很容易因为超时而失败。
    3.  **无法持久化：** 如果此时计算服务器重启，所有在 RPC 队列里排队的任务全都**瞬间丢失**。

    **最终建议：**

    * **如果你的计算很快（< 5秒）：** 用 RPC，写死 `Max Workers`。多出的任务让它们在 TCP 队列里**小排一会儿**，体验最顺滑。
    * **如果你的计算很慢（> 10秒）：** **绝对不要靠 RPC 的内置队列排队**。
        * 你应该在 Web 端直接拒绝（返回 429），让用户过会再点。
        * 或者转向**方案二（Redis + Celery）**。在消息队列里，任务可以排队几小时甚至几天都不会丢，Worker 挂了重启也能继续接着算。

    **总结一句话：** RPC 的限制主要是为了**自我保护（防崩溃）**，而不是为了**任务管理**。如果你需要优雅地处理“多出来的任务”，消息队列才是正解。

* 详细讲讲Redis + Celery

    这就是科研计算场景下的“终极方案”。**Celery** 是一个异步任务队列（Task Queue），而 **Redis** 在这里扮演“中间人”（Broker）的角色。

    简单来说：**Web 服务器把“作业”扔进 Redis，Celery 像个勤奋的学生，守在 Redis 边上，有作业就拿去高性能服务器上算，算完把结果写回数据库。**

    以下是该方案的详细拆解：

    1. 核心组件及其职责

        * **Task (任务)**：你的科学计算逻辑（封装成 Python 函数，内部调用 Fortran/BLAS）。
        * **Broker (消息中间件 - Redis)**：存储任务队列的仓库。Web 端发出的计算指令会暂存在这里。
        * **Worker (执行者)**：运行在高性能服务器上的进程。它可以是 1 个，也可以是 10 个。它只负责从 Redis 领任务并执行。
        * **Backend (结果存储)**：计算完的结果存哪？通常可以用 Redis 或数据库（PostgreSQL/MySQL）。

    ---

    2. 工作流程

        1.  **提交任务**：用户在网页点“计算”，Web 后端执行 `compute_task.delay(params)`。
        2.  **入队**：这个请求被包装成一条消息，存入 Redis 列表。
        3.  **响应**：Web 后端**秒回**用户：“任务已收到，正在处理中...”，并返回一个 `task_id`。
        4.  **异步计算**：Worker 发现 Redis 有新作业，立刻下载参数，在本地启动计算。
        5.  **状态查询**：前端网页拿着 `task_id` 定时轮询 Web 后端：“我的任务 123 算完了吗？”
        6.  **结果展示**：Worker 算完后更新状态为 `SUCCESS`，用户下次轮询时，网页展示最终图表或数据。

    ---

    3. 如何解决你的痛点？

        A. 完美的多用户排队

        如果你的高性能服务器只能带两个重型计算，你就启动 2 个 Celery Worker。如果有 10 个用户同时提交，前 2 个立即开始，后 8 个在 Redis 里**静静排队**，既不会撑爆内存，也不会导致 Web 网页超时崩溃。

        B. 环境隔离

        你的 Web 服务器（云服务器）可以极其干净，不需要安装 `gfortran`、`blas` 或任何科研库。所有的复杂环境、动态库和编译过程全部封装在 **Worker 所在的服务器**（或 Docker 镜像）里。

        C. 计算弹性

        如果某天任务量剧增，你只需要在另一台高性能机器上安装 Celery 并连接同一个 Redis，计算能力立刻翻倍。

    ---

    4. 代码实现简述（以 Python 为例）

        **第一步：定义任务 (`tasks.py`)**
        ```python
        from celery import Celery
        import subprocess

        # 配置 Redis 作为 Broker
        app = Celery('science_project', broker='redis://:password@your-redis-ip:6379/0')

        @app.task
        def run_fortran_calc(params):
            # 这里可以调用你编译好的 Fortran 程序
            # 也可以使用 ctype/f2py 调用的 so 库
            result = subprocess.check_output(["./my_calc_core", params])
            return parse_result(result)
        ```

        **第二步：Web 端调用 (`views.py`)**
        ```python
        from tasks import run_fortran_calc

        def on_click_calculate(request):
            params = request.POST.get('data')
            # 异步调用，立即返回
            task = run_fortran_calc.delay(params) 
            return JsonResponse({"task_id": task.id})
        ```

        **第三步：启动 Worker (在高性能机器上)**
        ```bash
        celery -A tasks worker --loglevel=info --concurrency=2
        # --concurrency=2 表示最多同时跑 2 个任务，保护 CPU/内存
        ```

    ---

    5. 这个方案的代价（缺点）

        1.  **架构变复杂**：你多了一个 Redis 组件和一个 Celery 进程需要维护。
        2.  **非实时性**：用户不能通过一个 HTTP 请求直接拿到结果，必须通过“轮询”或者 WebSocket 来获取结果，前端代码量会增加。
        3.  **数据传输**：如果你的计算输入数据非常大（比如几个 GB 的矩阵），频繁通过 Redis 传输会有网络压力。这种情况通常建议 Web 服务器先存入分布式存储（或 NAS），Redis 只传文件路径。

    **总结建议**

    * 如果你的科研计算是**“秒开”**，选 **RPC**。
    * 如果你的计算是**“重量级”**（需要 BLAS 加速通常意味着计算量不小），**Redis + Celery 是最专业、最稳健的做法**。它让你的系统具备了生产环境级别的健壮性。

    **你目前的科研后端，单次计算大概需要跑多久？** 如果超过 10 秒，强烈建议直接上 Celery。

* toast++ github repo: <https://github.com/toastpp/toastpp>

* NIRS-Toolbox 是个基于 matlab 的工具箱

    相关网站：

    * <https://help.nirx.de/hc/en-us/articles/17400352078108-NIRS-Toolbox>

    * <https://github.com/huppertt/nirs-toolbox>

    * <https://www.artinis.com/blogpost-all/2021/fnirs-analysis-toolbox-series-brain-analyzir>

    * <https://nirx.net/fnirs-analysis>

* 我现在想加载 lumo 的数据后，使用 mne-nirs 进行预处理，然后使用 toast++ 进行 3D 重建，给我一份 example 代码，加上可视化功能 

    这是一个非常有挑战性且完整的 DOT（扩散光学断层扫描）流程。要实现从 Lumo 原始数据到 3D 重建的转换，我们需要将 Lumo 的几何结构映射到 MNI 空间，利用 `mne-nirs` 处理时间序列信号，最后调用 `toast++` 进行空间域的反演。

    由于 `toast++` 的 3D 重建通常针对静态或差分图像，下面的示例将演示如何将 **MNE 处理后的 OD（光密度）变化量** 映射到 **3D 吸收系数 $\Delta\mu_a$** 的重建。

    **核心流程设计**

    1. **数据转换**：将 Lumo 原始强度转换为 `mne.io.Raw` 对象。
    2. **预处理**：使用 `mne-nirs` 进行强度到光密度（OD）的转换及滤波。
    3. **几何对齐**：使用你提供的 `regi` 函数将探头坐标对齐到 MNI152 头部模型。
    4. **Toast++ 建模**：在 3D Mesh 上配置源 ($Q$) 和检测器 ($M$)。
    5. **3D 重建**：求解差分反问题。

    ---

    **Python 示例代码**

    ```python
    import numpy as np
    import mne
    import mne_nirs
    import toast
    import pyvista as pv
    from scipy.sparse import linalg
    import os

    # 假设你之前的工具代码保存在 lumo_utils.py 中
    # from lumo_utils import * ## 1. 加载 Lumo 数据并转为 MNE 对象
    def convert_lumo_to_mne(int_bin, rec_data, sd_layout):
        # Lumo 数据格式: [nframe, recordings_per_frame]
        data = int_bin.data.T  # 转置为 [channels, times]
        
        # 创建 MNE info
        ch_names = [f"CH{i+1}" for i in range(rec_data.n_chans)]
        # 这里简单处理，实际应根据 wavelength 分配类型
        ch_types = ['fnirs_cw_amplitude'] * rec_data.n_chans 
        info = mne.create_info(ch_names=ch_names, sfreq=rec_data.framerate, ch_types=ch_types)
        
        raw = mne.io.RawArray(data, info)
        return raw

    ## 2. 使用 MNE-NIRS 预处理
    def preprocess_nirs(raw):
        # 1. 强度转光密度 (Optical Density)
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        # 2. 带通滤波 (去除漂移和心跳噪声)
        raw_od.filter(0.01, 0.2, fir_design='firwin')
        # 3. 提取某一段的时间平均值作为重建目标 (差分重建)
        # 比如：delta_OD = mean(task) - mean(baseline)
        data_od = raw_od.get_data()
        delta_od = np.mean(data_od, axis=1) # 简化：取全过程均值作为演示
        return delta_od

    ## 3. Toast++ 3D 重建配置
    def setup_toast_3d(mshs, sd_layout, mni_landmarks):
        # a. 创建 Toast Mesh
        # 注意：toast 需要节点和单元。这里假设使用 headVolumeMesh 进行前向计算
        nodes = mshs.headVolumeMesh.node
        elems = mshs.headVolumeMesh.elem[:, :4] # 提取四面体索引
        
        # 写入临时文件或直接通过内存创建（取决于 toast 版本，通常建议写 .msh）
        # 这里演示逻辑：
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: elems}, nodes)
        # toast_mesh = toast.Mesh(nodes, elems) # 伪代码，根据实际 API 调整
        
        # b. 注册探头到 MNI 空间并投影到头皮
        src_mni = regi_coords_sd_to_mni(sd_layout.src_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        det_mni = regi_coords_sd_to_mni(sd_layout.det_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        
        # 投影到头皮表面最近节点
        src_on_scalp = regi_coords_to_nearest_neighbors(src_mni, mshs.scalpSurfaceMesh.node)
        det_on_scalp = regi_coords_to_nearest_neighbors(det_mni, mshs.scalpSurfaceMesh.node)
        
        return src_on_scalp, det_on_scalp

    ## 4. 主重建程序
    def run_3d_reconstruction():
        # --- 加载数据 ---
        # (此处省略文件路径定义，参考你之前的测试代码)
        int_bin = lumo_load_all_intensity_files(lumo_bin_fpaths)
        rec_data = lumo_load_recording_data(toml_path)
        sd_layout = lumo_load_sd_layout(layout_json)
        mshs = load_mni152_mshs(mshs_file)
        
        # --- 预处理 ---
        raw = convert_lumo_to_mne(int_bin, rec_data, sd_layout)
        delta_od = preprocess_nirs(raw) # 得到每个通道的测量值 y
        
        # --- Toast 准备 ---
        # 创建 Toast 3D 网格
        tm = toast.Mesh() 
        # 这里建议将 mshs.headVolumeMesh 保存为 .msh 后加载
        # tm.ReadModel('adult_mni.msh') 
        
        # 设置源探测器 (Qvec/Mvec)
        # 假设 Lumo 共有 Nsrc 个源和 Ndet 个检测器
        qvec = tm.Qvec(type='Neumann', shape='Gaussian', width=2, pos=src_on_scalp)
        mvec = tm.Mvec(shape='Gaussian', width=2, pos=det_on_scalp)
        
        # 初始化背景参数 (标准组织光学参数)
        nlen = tm.NodeCount()
        mua = np.ones(nlen) * 0.01  # 背景吸收
        mus = np.ones(nlen) * 1.0   # 背景散射
        ref = np.ones(nlen) * 1.4   # 折射率
        freq = 0 # Lumo 是连续波 (CW)
        
        # --- 计算雅可比矩阵 ---
        print("Computing Jacobian...")
        phi = tm.Fields(None, qvec, mua, mus, ref, freq)
        aphi = tm.Fields(None, mvec, mua, mus, ref, freq)
        # 对于 CW 数据，测量值是强度，这里计算灵敏度矩阵
        J = tm.Jacobian(None, phi, aphi, None) # 简化的 Jacobian 调用
        
        # --- 线性反演 (Tikhonov Regularization) ---
        # y = J * x  => x = (J'J + alpha*I)^-1 J'y
        # 其中 x 是 delta_mua
        alpha = 0.01 * np.max(np.diag(J.T @ J)) # 正则化参数
        delta_mua = linalg.lsqr(J, delta_od, damp=alpha)[0]
        
        # --- 可视化 ---
        visualize_result(mshs, delta_mua)

    def visualize_result(mshs, delta_mua):
        # 使用 PyVista 在脑模上显示吸收变化
        nodes = mshs.headVolumeMesh.node
        elems = mshs.headVolumeMesh.elem[:, :4] # 四面体单元
        
        # 创建网格对象
        cell_type = np.array([pv.CellType.TETRA] * elems.shape[0])
        cells = np.hstack(np.c_[np.full(elems.shape[0], 4), elems])
        grid = pv.UnstructuredGrid(cells, cell_type, nodes)
        
        # 将结果映射到网格节点
        grid.point_data["Delta_Mua"] = delta_mua
        
        # 绘图
        plotter = pv.Plotter()
        # 绘制半透明头皮
        plotter.add_mesh(pv.PolyData(mshs.scalpSurfaceMesh.node, 
                                    np.hstack(np.c_[np.full(mshs.scalpSurfaceMesh.face.shape[0], 3), 
                                                    mshs.scalpSurfaceMesh.face])), 
                        opacity=0.1, color='tan')
        
        # 绘制重建后的吸收变化切片
        slices = grid.slice_orthogonal()
        plotter.add_mesh(slices, cmap="jet", scalars="Delta_Mua")
        
        # 也可以绘制阈值以上的体积 (例如激活区)
        thres = grid.threshold(np.percentile(delta_mua, 95))
        plotter.add_mesh(thres, color="red")
        
        plotter.show()

    ```

    ---

    **关键点说明**

    1. CW（连续波） vs FD（频域）

        你提供的 `toast++` 例子中 `freq = 100`，那是针对频域设备的（有相位信息）。**Lumo 是连续波设备**，因此在 `tm.Fields` 和 `tm.Jacobian` 中，`freq` 必须设为 **0**。此时我们只重建吸收系数 $\mu_a$，因为 CW 很难区分散射和吸收。

    2. 坐标注册 (Registration)

        Lumo 的坐标是基于其 Dock 的局部坐标。

        * 使用你写的 `regi_coords_sd_to_mni` 非常关键。
        * **注意**：`mni_landmarks` 的 5 个点顺序（Nasion, Inion, LPA, RPA, Cz）必须与 Lumo 导出 JSON 中的 `landmark_coords` 顺序完全一致，否则旋转矩阵 $R$ 会出错。

    3. 雅可比矩阵与内存

        在 3D 重建中，`J` 矩阵可能会非常大（通道数 $\times$ 节点数）。

        * Lumo 的通道数通常很多（几百到上千）。
        * MNI152 节点数约为 21 万。
        * 如果内存不足，建议使用 `toast.Basis` 将重建空间降低到较低分辨率的网格（类似你例子中的 `basis_inv`）。

    4. MNE-NIRS 的作用

        * **清理数据**：Lumo 的原始 binary 包含噪声。通过 `mne-nirs` 转换成 OD 并滤波，可以显著提高重建质量。
        * **差分重建**：通常我们重建的是 $\Delta\mu_a$（即任务态 vs 基线的变化）。公式为：

        $$\Delta \text{OD} = J \cdot \Delta \mu_a$$

        这里的 `delta_od` 就是 `mne` 处理后的信号差值。

    **建议下一步**

    您可以先尝试运行 `setup_toast_3d` 部分，使用 `pyvista` 确认探头是否正确“贴”在了 MNI152 的头皮上。如果探头悬空或者掉进了脑子里，重建结果将完全不可信。

    **需要我针对某个特定部分（如如何把 Mshs 对象转换成 Toast 兼容格式）深入写一下代码吗？**

* 我可以确认探头经过 regi_xxx 函数配准后，一定贴在 mni scalp mesh 头皮上，我已经做了可视化驻

    太棒了！既然坐标配准（Registration）已经解决，且探头已精准“贴合”在头皮上，我们接下来的核心任务是**打通数据流**：将 MNE 处理后的物理信号（$\Delta OD$）映射到 Toast++ 的前向模型矩阵（Jacobian）中，完成 3D 空间的反演。

    由于 Lumo 是 **CW（连续波）** 设备，我们通常只重建吸收系数的变化 $\Delta\mu_a$。

    **核心实现逻辑**

    1. **物理量对齐**：MNE-NIRS 输出的是 $\Delta OD = -\ln(I/I_0)$。在 Toast++ 中，对于 CW 数据，前向模型对应的 Jacobian 元素 $J_{ij} = \frac{\partial \ln \Phi_i}{\partial \mu_{a,j}}$。
    2. **网格转换**：将 `mshs.headVolumeMesh` 的节点和四面体单元导入 Toast。
    3. **正则化反演**：使用 Tikhonov 正则化处理高度欠定（Underdetermined）的问题。

    ---

    **3D 重建完整示例代码**

    ```python
    import numpy as np
    import toast
    import mne
    import mne_nirs
    import pyvista as pv
    from scipy.sparse import linalg

    def run_lumo_toast_reconstruction(int_bin, rec_data, sd_layout, mshs, mni_landmarks):
        """
        基于已配准坐标的 3D 重建流程
        """
        
        # --- 1. MNE-NIRS 预处理获取 Delta OD ---
        # 构造 MNE 格式 (简化版，仅演示数据流)
        data = int_bin.data.T  # [channels, time]
        info = mne.create_info(ch_names=[f'CH{i}' for i in range(rec_data.n_chans)], 
                            sfreq=rec_data.framerate, ch_types='fnirs_cw_amplitude')
        raw = mne.io.RawArray(data, info)
        
        # 预处理：强度 -> 光密度 -> 滤波 -> 提取均值变化
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        raw_od.filter(0.01, 0.1)
        
        # 假设我们要重建全段相对于初始时刻的变化
        od_data = raw_od.get_data()
        delta_od = od_data[:, -1] - od_data[:, 0] # 最后一个时刻 vs 第一个时刻
        
        # --- 2. Toast++ 网格加载 ---
        # 将 MNI152 的头模型转为 Toast 可识别的 Mesh
        nodes = mshs.headVolumeMesh.node
        # 注意：matlab 加载的索引通常从1开始，需要确认是否需 -1
        # 且四面体在 toast 中通常是 [n, 4]
        elems = mshs.headVolumeMesh.elem[:, :4] 
        
        tm = toast.Mesh()
        tm.SetNodes(nodes)
        tm.SetElements(elems)
        
        # --- 3. 设置源探测器 (Qvec & Mvec) ---
        # 使用你已经验证过的配准坐标
        src_mni = regi_coords_sd_to_mni(sd_layout.src_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        det_mni = regi_coords_sd_to_mni(sd_layout.det_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        
        # 将源和探测器放置在头皮表面（Neumann 边界条件）
        qvec = tm.Qvec(type='Neumann', shape='Gaussian', width=2, pos=src_mni)
        mvec = tm.Mvec(shape='Gaussian', width=2, pos=det_mni)
        
        # --- 4. 配置光学参数 ---
        nlen = tm.NodeCount()
        # 典型生物组织背景值 (单位: mm^-1)
        mua = np.ones(nlen) * 0.01 
        mus = np.ones(nlen) * 1.0  
        ref = np.ones(nlen) * 1.4
        freq = 0  # CW 模式
        
        # --- 5. 计算 Jacobian ---
        print("Building Forward Model...")
        phi = tm.Fields(None, qvec, mua, mus, ref, freq)
        aphi = tm.Fields(None, mvec, mua, mus, ref, freq)
        
        # 计算灵敏度矩阵 (Jacobian)
        # 对于 CW，我们需要的是对 log 强度的导数
        proj = tm.Projection(phi, mvec) 
        # 获取 Jacobian (映射到节点空间)
        J = tm.Jacobian(None, phi, aphi, proj)
        
        # --- 6. 线性线性反演 ---
        # 因为通道数可能远小于节点数，这是一个极度病态的问题
        # 使用 Tikhonov 正则化: x = J^T * (J*J^T + alpha*I)^-1 * y
        print("Inverting...")
        # 选取正则化参数，通常根据最大奇异值的一定比例
        alpha = 1e-3 * np.linalg.norm(J @ J.T)
        
        # 求解: (J*J.T + alpha*I) * beta = delta_od
        A = J @ J.T + alpha * np.eye(J.shape[0])
        beta = np.linalg.solve(A, delta_od)
        # 映射回 3D 空间
        delta_mua = J.T @ beta

        return tm, delta_mua

    # --- 7. 可视化函数 ---
    def plot_3d_result(tm, delta_mua, mshs):
        nodes = tm.Nodes()
        elems = tm.Elements()
        
        # 创建 PyVista 对象
        # cells 数组格式: [n_points, p1, p2, p3, p4, n_points, ...]
        cells = np.hstack(np.c_[np.full(elems.shape[0], 4), elems])
        grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA]*elems.shape[0], nodes)
        
        grid.point_data["delta_mua"] = delta_mua
        
        plotter = pv.Plotter()
        
        # 添加半透明头皮作为背景
        scalp_poly = pv.PolyData(mshs.scalpSurfaceMesh.node, 
                                np.hstack(np.c_[np.full(mshs.scalpSurfaceMesh.face.shape[0], 3), 
                                                mshs.scalpSurfaceMesh.face]))
        plotter.add_mesh(scalp_poly, color="tan", opacity=0.15)
        
        # 关键：使用切片可视化脑内吸收变化
        slices = grid.slice_orthogonal(x=0, y=0, z=0)
        plotter.add_mesh(slices, cmap="RdBu_r", scalars="delta_mua", 
                        stitle="Absorption Change", clim=[-max(abs(delta_mua)), max(abs(delta_mua))])
        
        # 或者展示高吸收区域的等值面
        # threshold = grid.threshold(np.max(delta_mua) * 0.5)
        # plotter.add_mesh(threshold, color="red")
        
        plotter.add_scalar_bar()
        plotter.show()

    ```

    ---

    **几个关键细节提醒：**

    1. **Lumo 数据索引映射**：
        
        Lumo 的 `chans_list` 定义了哪些源-探测器对组成了通道。在 `tm.Qvec` 和 `tm.Mvec` 之后产生的 `phi` 和 `aphi` 矩阵，计算出的 `J` 矩阵行数默认是 `N_src * N_det`。你可能需要根据 `rec_data.chans_list_act` 提取出对应的行，使得 `J` 的行数与 `delta_od` 的长度一致。

        ```python
        # 假设 J 原始大小是 (Nsrc*Ndet, Nnodes)
        # 需要根据实际通道配置过滤
        J = J[active_channel_indices, :]

        ```

    2. **MNI152 组织分层**：

        你加载的 `headVolumeMesh` 包含 `labels` (Scalp, Skull, CSF, GM, WM)。
        * **进阶技巧**：在反演时，可以对不同的组织层设置不同的先验权重。例如，让 GM（灰质）层的正则化系数更小，从而引导重建结果更多地出现在脑皮层上。


    3. **单位一致性**：

        Toast 内部默认通常使用 `mm`。请确保你的 `mshs.node` 和 `sd_layout` 的坐标单位均为 `mm`。如果 MNI 坐标单位是厘米，请务必统一乘以 10。

    **下一步建议**

    你可以先尝试运行 `tm.Fields` 这一步。如果能顺利跑通前向计算并生成 `phi`（光通量分布），说明网格和源探测器配置完全正确。

    **要不要我帮你写一段代码，根据 Lumo 的 `chans_list` 自动从 Toast 的全采样 Jacobian 中提取出对应通道的子集？**

* `.nim` 文件

    ```nim
    NIM
    Mesh = ellips_tri10.msh
    SolutionType = N/A
    ImageSize = 32971
    EndHeader
    Image 0
    0.025 0.025 0.025 0.025 ...
    ```

* `ReadNim()`

    这段 C++ 代码展示了 `TOAST++` 底层库（FELIB）中 `ReadNim` 函数的实现逻辑。

    我们可以清晰地看到 `.nim` (Node Image) 或 `.rim` (Region Image) 文件的**内部结构**以及该函数的核心功能。

    1. 函数功能概述

        这段代码的主要作用是：**从一个文本格式的图像文件（.nim/ .rim）中提取特定的图像数据（标量场）以及关联的网格名称。**

        它并不直接读取网格的几何结构（那是 `Mesh::Read` 的工作），而是读取附加在网格节点或区域上的**数值数据**（例如吸收系数分布 $\mu_a$ 或散射系数分布 $\mu_s$）。

    ---

    2. 代码逻辑逐段解析

        **A. 文件头校验 (Header Check)**

        ```cpp
        if (!ifs.getline (cbuf, 256)) return false;
        if (strcmp (cbuf, "NIM") && strcmp (cbuf, "RIM")) return false;

        ```

        * **作用**：检查文件的第一行。
        * **关键点**：文件必须以关键字 `NIM` 或 `RIM` 开头。如果不是，函数立即返回 `false`（失败）。

        **B. 解析文件头信息 (Metadata Parsing)**

        ```cpp
        do {
            ifs.getline (cbuf, 256);
            if (!strncasecmp (cbuf, "ImageSize", 9))
                sscanf (cbuf+11, "%d", &imgsize);
            else if (!strncasecmp (cbuf, "Mesh", 4) && meshname)
                sscanf (cbuf+6, "%s", meshname);
        } while (strcasecmp (cbuf, "EndHeader"));

        ```

        * **作用**：在 `EndHeader` 标记出现前，循环查找关键参数。
        * **ImageSize**：读取数据的长度（通常对应网格的节点数或单元数）。
        * **Mesh**：读取该数据所依赖的网格文件名（例如 `mesh.msh`）。这解释了为什么函数参数里有一个 `meshname` 指针。

        **C. 索引定位与数据读取 (Data Extraction)**

        ```cpp
        for (;;) {
            do {
                ifs.getline (cbuf, 256);
            } while (ifs.good() && strncasecmp (cbuf, "Image", 5));
            // ...
            for (i = 0; i < imgsize; i++)
                ifs >> img[i];
            if (++j == idx) break;
        }

        ```

        * **作用**：`.nim` 文件可以包含多个数据集（Frames）。
        * **逻辑**：函数会查找以 `Image` 开头的标记。
        * **参数 `idx` 的意义**：如果你传入 `idx = 2`，它会跳过第一个图像块，读取文件中的第二个图像块。
        * **img 容器**：读取到的浮点数会被存储在 `RVector &img`（一个实数向量）中。

    ---

    3. 核心结论：它到底读了什么？

        基于源码，我们可以推断出一个 `.nim` 文件的典型结构如下：

        | 行/部分 | 内容示例 | 说明 |
        | --- | --- | --- |
        | **Header** | `NIM` | 文件类型标识 |
        |  | `ImageSize 1024` | 该网格有 1024 个节点 |
        |  | `Mesh mymesh.msh` | 关联的网格文件 |
        |  | `EndHeader` | 头部结束 |
        | **Data Block** | `Image 1` | 第一个数据块开始 |
        |  | `0.01 0.02 ...` | 1024 个浮点数值 |
        |  | `Image 2` | 第二个数据块开始 |
        |  | ... | ... |

## topics


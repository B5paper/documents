# Computer Network Note

## cache

* 一开始我希望有个内网穿透，使用了 frp，后来发现有黑客尝试弱密码攻击我的内网 sshd，于是我希望启用 fail2ban，但是我发现黑客的 source ip 全是 127.0.0.1，原来 frp 在转发后，会丢失原来的 tcp source ip。于是我希望找个代理工具，能在公网机器上转发流量时，带上一个 tcp source ip 的信息，然后在内网机器上再使用某个工具，把这个 tcp source ip 的信息写入到 tcp frame 里，这样内网的 sshd 就能看到真实的 source ip ，从而 fail2ban 可以正常工作了。目前我找到的代理工具为 haproxy，具体的流量链路如下：incoming connection -> haproxy on host A, add proxy protocol header -> frps on host A -> frpc on host B -> haproxy on host B, remove proxy protocol header, and modify tcp frame source ip -> app on host B, such as nc。

    这个过程是否可以实现？如果可以实现，该如何配置 haproxy？

    这是一个非常硬核且经典的网络架构问题。

    **简短回答：**

    是可以实现的。但是，你提到的“在内网机器上再使用某个工具，把这个 tcp source ip 的信息写入到 tcp frame 里”这一步，也就是**IP 地址欺骗（IP Spoofing）/ 透明代理（Transparent Proxy / TPROXY）**，配置起来相当复杂，且对操作系统内核配置有要求。

    如果你的核心目的是为了让 SSHD 看到真实 IP 从而让 Fail2ban 工作，我有一个**优化后的架构**，可以帮你省去 Host A 上的 HAProxy，并解决 Host B 上的配置难点。

    **方案分析与优化**

    **原有链路的问题**

    你设想的链路是：
    `User -> HAProxy(A) -> FRPS -> FRPC -> HAProxy(B) -> SSHD`

    这个链路有些冗余。**FRP 自身其实已经支持 Proxy Protocol**。你不需要在 Host A 上部署 HAProxy 来添加头信息，FRPC 可以直接在发给本地服务时带上这个头。

    **优化后的链路**

    `User -> FRPS (Host A) -> FRPC (Host B) -> HAProxy (Host B) -> SSHD (Host B)`

    在这个链路中：

    1. **FRPS/FRPC**: 负责将公网流量透传，并在输出流量给 HAProxy 时，添加 Proxy Protocol v2 头（包含真实源 IP）。
    2. **HAProxy (Host B)**: 接收带有 Proxy Protocol 头的流量，解析出真实 IP。
    3. **TPROXY (关键难点)**: HAProxy 使用 `usesrc clientip` 也就是 TPROXY 技术，伪造源 IP 连接本地的 SSHD。

    **详细配置指南**

    为了实现内网 SSHD 看到真实 IP（从而被 iptables/fail2ban 封禁），你需要配置 **Linux TPROXY**。

    第一步：配置 FRPC (Host B)

    修改 `frpc.ini`，在你的 ssh 穿透配置中开启 Proxy Protocol。

    ```ini
    [ssh]
    type = tcp
    local_ip = 127.0.0.1
    # 注意：这里不再直接填 22，而是填 HAProxy 监听的端口，比如 2222
    local_port = 2222
    remote_port = 6000
    # 关键配置：告诉 FRPC 在转发给 local_port 时，带上 Proxy Protocol 头
    proxy_protocol_version = v2

    ```

    第二步：配置 HAProxy (Host B)

    HAProxy 需要读取 Proxy Protocol，并使用 TPROXY 发送数据。

    1. **安装 HAProxy** (确保版本 >= 1.5)。
    2. **编辑配置文件** (`/etc/haproxy/haproxy.cfg`)：

    ```haproxy
    global
        log /dev/log local0
        user haproxy
        group haproxy
        # 必须以 root 运行或者拥有 CAP_NET_ADMIN 权限才能进行 IP 欺骗
        # 建议测试时先用 root，稳定后配置 capabilities
        # daemon

    defaults
        log global
        mode tcp
        option tcplog
        timeout connect 5000ms
        timeout client 50000ms
        timeout server 50000ms

    frontend frp_in
        # 监听 2222，对应 frpc 的 local_port
        # accept-proxy 表示只接受带有 Proxy Protocol 头的连接
        bind 127.0.0.1:2222 accept-proxy
        default_backend ssh_local

    backend ssh_local
        mode tcp
        # 关键点：source 0.0.0.0 usesrc clientip
        # 这行指令告诉 HAProxy 使用客户端的真实 IP 作为源 IP 发起连接
        source 0.0.0.0 usesrc clientip
        # 连接本地的 SSHD
        server ssh_server 127.0.0.1:22

    ```

    第三步：配置 Linux 内核路由与 Iptables (Host B) —— 最困难的一步

    因为 HAProxy 伪造了源 IP 向 SSHD 发起连接（例如黑客 IP 1.2.3.4），SSHD 收到包后，认为源 IP 是 1.2.3.4，回复数据包时会试图发给网关（走公网），而不是回给 HAProxy。这会导致连接建立失败（TCP 握手无法完成）。

    我们需要配置策略路由，把这些**回复包**强行抓回给本地的 HAProxy。

    **请在 Host B 上以 root 权限执行：**

    1. **启用 IP 转发（通常不需要，但在某些 TPROXY 场景下推荐）：**
    ```bash
    sysctl -w net.ipv4.ip_forward=1

    ```

    2. **配置路由规则 (把回复包截获)：**
    由于 SSHD 和 HAProxy 在同一台机器（Host B），且都监听 `127.0.0.1`，利用 TPROXY 处理 Loopback 流量非常棘手。
    **强烈建议的方法：** 让 SSHD 监听一个非 `127.0.0.1` 的 Loopback 地址（例如 `127.0.0.2`），或者让 HAProxy 和 SSHD 分离（例如 SSHD 在 Docker 里，或者 HAProxy 在 Docker 里）。
    **假设我们采用“SSHD 监听 127.0.0.2”的方案（最稳健）：**
    A. 修改 `/etc/ssh/sshd_config`：
    ```text
    ListenAddress 127.0.0.2

    ```

    重启 SSHD。

    B. 修改 HAProxy 配置：

    ```haproxy
    backend ssh_local
        source 0.0.0.0 usesrc clientip
        server ssh_server 127.0.0.2:22  # 改为 127.0.0.2

    ```


    C. 配置 iptables 规则（给数据包打标记）：
    我们需要把从 SSHD (`127.0.0.2`) 发出的、发往非本地（因为 HAProxy 伪造了外网 IP）的数据包拦截下来。
    ```bash
    # 创建路由表 100
    ip rule add fwmark 1 lookup 100
    # 将路由表 100 的流量导向本地 Loopback，这样 TPROXY 才能捕获它
    ip route add local 0.0.0.0/0 dev lo table 100

    # Iptables Mangle 表规则
    # 这里的逻辑是：如果是从 127.0.0.2:22 发出的包（SSHD 回包），打上标记 1
    # 只有打上标记 1，才会走上面的路由表 100，被送回 lo 接口，从而被 HAProxy 收到
    iptables -t mangle -A OUTPUT -p tcp -s 127.0.0.2 --sport 22 -j MARK --set-mark 1

    ```

    ---

    关于 Fail2ban 的特别说明

    当你完成上述配置后：

    1. 黑客连接 FRP。
    2. FRPC 传给 HAProxy。
    3. HAProxy 伪造黑客 IP 连接 `127.0.0.2:22`。
    4. SSHD 在 `/var/log/auth.log` 中记录下黑客的**真实公网 IP**。
    5. Fail2ban 扫描日志，发现攻击 IP。
    6. Fail2ban 调用 iptables 封禁该 IP。

    **存在的问题：**
    Fail2ban 默认的 `action` 是在 `INPUT` 链上封禁 IP。
    但是，因为数据包是通过 FRPC -> HAProxy 进来的，对于 Host B 的公网网卡来说，流量来源是 FRPS（或者是 FRPC 的连接）。
    真正到达 SSHD 的流量是**内部 Loopback 流量**，且是由 HAProxy **伪造**的。

    如果 Fail2ban 封禁了 IP（例如 `iptables -A INPUT -s 1.2.3.4 -j DROP`），这规则通常作用于物理网卡。对于 TPROXY 伪造的本地 Loopback 流量，你需要确保 Fail2ban 的封禁规则也能作用于 `lo` (Loopback) 接口或者 `INPUT` 链能够匹配到这些伪造包。

    通常，TPROXY 产生的包在进入 `INPUT` 链时会带有源 IP，所以 Fail2ban **理论上是可以工作的**。但务必检查 `iptables -vnL` 确认封禁规则的计数器在增加。

    ---

    更简单的替代方案（推荐）

    如果搞定 TPROXY 实在太麻烦（涉及到复杂的 iptables 和路由，重启后易失效），有一个极其简单的替代方案：**不要在 SSHD 层面封禁**。

    **方案：Fail2ban 监控 HAProxy 日志**

    1. **配置**：`User -> FRPS -> FRPC (Proxy Protocol) -> HAProxy (B) -> SSHD (B)`。
    2. **HAProxy**: 不需要 `usesrc clientip`（去掉 TPROXY），直接作为普通代理转发给 SSHD。
    * HAProxy 会知道真实 IP。
    * HAProxy 可以配置日志记录真实 IP。


    3. **Fail2ban**:
    * 创建一个新的 filter，去读取 **HAProxy 的日志**。
    * 当 HAProxy 记录到某个 IP 频繁发起连接（或者虽然无法判断密码错误，但可以限制单 IP 的连接频率）时，进行封禁。
    * **封禁动作**：Fail2ban 需要利用 HAProxy 的 Runtime API (socket) 来添加 ACL 拒绝该 IP，或者直接在 Host B 的防火墙拒绝该 IP（但这对 FRP 流量无效，除非用 TPROXY）。


    **最最推荐的方案（安全层面的降维打击）：**

    既然你都有公网 FRP 了，为什么不直接上 **WireGuard** 或 **Tailscale**？

    * 不要把 SSH 端口暴露在公网（哪怕是 FRP 转发的）。
    * 用 FRP 转发 WireGuard 的 UDP 流量（或者直接用 Tailscale 的 NAT 穿透）。
    * 连接 VPN 后，通过内网 IP 访问 SSH。
    * **结果**：黑客根本扫描不到你的 SSH 端口，Fail2ban 都不需要了。

    下一步建议

    如果你一定要死磕 SSHD 看到真实 IP 的方案，建议先尝试上面的 **"SSHD 监听 127.0.0.2" + "Iptables Mangle"** 的方法。

    **你需要我为你提供针对上述 TPROXY 方案的 Fail2ban 具体配置（jail.local 和 action），还是想了解一下 WireGuard/Tailscale 的方案？**

* 我看到 proxy_protocol_version = v2 这个配置已经在 frp conf legacy 目录里面了，这还有效吗？

    这是一个非常敏锐的问题。你看到的 "Legacy" 并不是指这个功能（Function）失效了，而是指 **INI 配置文件格式**在 frp 的新版本（v0.52.0+）中已经被标记为“遗留格式”了。

    **简单回答：**
    **依然有效。** 只要你的配置文件还是 `.ini` 格式，`proxy_protocol_version = v2` 依然可以正常工作。

    但是，frp 官方正在由 `.ini` 转向 `.toml` (或 `.yaml`/`.json`) 格式。为了保证你的配置在未来版本的兼容性，建议了解一下新旧格式的区别。

    以下是详细说明：

    1. 为什么会在 Legacy 目录？

        从 **frp v0.52.0** 开始，官方重构了配置文件的格式。

        * **旧格式 (Legacy)**: `frpc.ini` / `frps.ini`
        * **新格式 (Recommended)**: `frpc.toml` / `frps.toml`

        虽然新版本 frp 依然兼容旧的 `.ini` 格式（通常会自动识别，或者通过 `-c` 指定），但官方文档和源码结构中已经将其归类为 Legacy。

    2. 如何配置？

        无论你使用旧版还是新版配置格式，**Proxy Protocol 的功能都是核心支持的**，没有被移除。

        方式 A：如果你继续使用 `frpc.ini` (旧格式)

        这还是你目前查到的写法，完全有效：

        ```ini
        [ssh]
        type = tcp
        local_ip = 127.0.0.1
        local_port = 2222
        remote_port = 6000
        # 有效：告诉 frpc 在连接本地 local_port 时，发送 Proxy Protocol v2 头
        proxy_protocol_version = v2

        ```

        方式 B：如果你想迁移到 `frpc.toml` (新格式)

        在新的 TOML 格式中，这个配置项的位置发生了变化，它被归纳到了 `transport` 字段下：

        ```toml
        [[proxies]]
        name = "ssh"
        type = "tcp"
        localIP = "127.0.0.1"
        localPort = 2222
        remotePort = 6000

            [proxies.transport]
            # 对应旧版的 proxy_protocol_version
            proxyProtocolVersion = "v2" 

        ```

    3. 注意事项 (针对你的架构)

        在你的架构中：`Incoming -> FRPS -> FRPC -> HAProxy -> SSHD`

        你需要在 **Host B** 的 **`frpc`** 配置文件中设置这个参数。

        * 它的作用是：当 FRPC 收到来自 FRPS 的流量，准备转发给 `local_ip:local_port` (即 HAProxy) 时，会在 TCP 数据包的最前面加塞一段 Proxy Protocol v2 的元数据（包含 User 的真实 IP）。
        * 如果没有这一行，HAProxy 收到包时就会报错（因为它配置了 `accept-proxy`，期待有头，但实际没有头）或者只能看到 127.0.0.1。

    **总结：**

    放心使用 `proxy_protocol_version = v2`。如果你现在的 frp 版本比较新（v0.52.0+），虽然它会提示推荐使用 TOML，但它依然会完美解析 INI 文件中的这个参数。

    需要我帮你把之前的配置转换成最新的 TOML 格式吗？或者我们继续处理 HAProxy 的 TPROXY 配置？

* 我配置下，确实能看到 source ip 的信息了：

    ```
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ nc -kvl 6543
    Listening on 0.0.0.0 6543
    Connection received on localhost 48076
    PROXY TCP4 120.136.165.86 172.24.3.211 39604 4321
    hello
    byebye
    ```

    那么 sshd 可以直接处理这个 proxy protocol 吗？或者有什么办法处理？

    **简短回答：**
    **不行。** 原生的 OpenSSH `sshd` **无法** 识别或处理 `PROXY` 协议头。
    如果你直接把这串带有 `PROXY TCP4 ...` 的数据流发给 `sshd`（监听 22 端口），`sshd` 会认为协议不匹配或数据包损坏，直接断开连接。

    ---

    该如何处理？

    就像你最初设想的那样，你需要一个“中间人”来完成“剥离头信息 + 欺骗源 IP”的工作。
    HAProxy 正是完成这个工作的完美工具。它的工作流程如下：

    1. **接收（Bind）**：HAProxy 监听端口，通过 `accept-proxy` 指令接收流量，并**自动解析并剥离** `PROXY` 头。此时 HAProxy 内存里已经拿到了真实的 IP（`120.136...`）。
    2. **清洗（Strip）**：HAProxy 准备向后端的 SSHD 发起连接时，发送的是**不带 Proxy 头**的纯净 SSH 流量。
    3. **欺骗（Spoof/TProxy）**：配置 `usesrc clientip` 后，HAProxy 在向 SSHD 发起 TCP 握手时，会将数据包的 **Source IP** 伪造成刚才解析出来的 `120.136...`。

    这样，SSHD 收到的是：**纯净的 SSH 流量**，且**源 IP 是 120.136...**。

    ---

    **方案落地：HAProxy TProxy 配置回顾**

    既然你的 FRPC 已经配置好了（这是最关键的一步），现在的难点就在于 **Host B** 上的 HAProxy 和路由配置。

    我为你整理了最精简的实施步骤（假设你在 Host B 上操作）：

    1. 准备 SSHD

        让 SSHD 监听一个特定的别名 IP，防止和 HAProxy 冲突，也方便做路由拦截。
        修改 `/etc/ssh/sshd_config`：

        ```text
        # 不要监听 0.0.0.0，只监听这个特殊的本地 IP
        ListenAddress 127.0.0.2

        ```

        重启 sshd：`sudo systemctl restart ssh`

    2. 配置 HAProxy

        修改 `/etc/haproxy/haproxy.cfg`：

        ```haproxy
        global
            log /dev/log local0
            # 必须以 root 运行才能伪造 IP（或者配置复杂的 setcap）
            user root
            group root

        defaults
            mode tcp
            timeout connect 5s
            timeout client 30s
            timeout server 30s

        frontend frp_in
            # 对应 frpc.ini 里的 local_port
            # accept-proxy：接收并“吃掉” FRPC 发来的 Proxy 头
            bind 127.0.0.1:2222 accept-proxy
            default_backend ssh_local

        backend ssh_local
            # 关键：使用解析出来的客户端真实 IP 发起后端连接
            source 0.0.0.0 usesrc clientip
            # 连接我们刚才修改的 SSHD 监听地址
            server ssh_real 127.0.0.2:22

        ```

        重启 haproxy：`sudo systemctl restart haproxy`

    3. 配置路由守门员 (最关键的一步)

        因为 HAProxy 伪造了外网 IP（比如 `120.136...`）向 `127.0.0.2` 发起连接，SSHD 回包时会默认查路由表走外网网关，导致握手失败。我们需要强制把回包抓回来。

        执行以下脚本（重启会失效，测试成功后需写入开机启动）：

        ```bash
        # 1. 启用路由转发（TProxy 依赖）
        echo 1 > /proc/sys/net/ipv4/ip_forward

        # 2. 策略路由：凡是打上标记 1 的包，都去查路由表 100
        ip rule add fwmark 1 lookup 100

        # 3. 路由表 100：把所有流量都强行送回本地 (loopback)
        ip route add local 0.0.0.0/0 dev lo table 100

        # 4. Iptables 抓包打标记：
        # 只要是 SSHD (127.0.0.2:22) 发出的回包，就打上标记 1
        iptables -t mangle -A OUTPUT -p tcp -s 127.0.0.2 --sport 22 -j MARK --set-mark 1

        ```

    ---

    **验证方法**

    配置完成后，你在外网通过 FRP 连接 SSH：
    `ssh -p <frp_remote_port> user@frp_server_ip`

    然后在 Host B 上查看日志：

    1. **查看 HAProxy 日志** (`tail -f /var/log/haproxy.log`)：应该能看到源 IP。
    2. **查看 SSHD 日志** (`tail -f /var/log/auth.log` 或 `/var/log/secure`)：
    * 成功：`Accepted password for ... from 120.136.165.86 ...`
    * 失败（如果没配好 TProxy）：`Accepted ... from 127.0.0.1`



    **替代工具：mmproxy (更轻量)**

    如果你觉得 HAProxy 太重（或者配置起来太繁琐），Cloudflare 开源了一个专门做这个的小工具叫 **`mmproxy`**。

    * 它的作用就是：接收带 Proxy 头的流量 -> 拆包 -> 伪造 IP 发给本地目标。
    * 它依然需要上面的 **iptables/路由** 配置，但仅仅省去了 HAProxy 的配置过程。

    **建议：** 既然你已经比较熟悉 HAProxy 且它功能强大（自带重试、日志、监控），建议继续使用 HAProxy 方案。

* 我觉得这个有点复杂了，我的本质需求是防止黑客攻击内网的 sshd，目前我在公网服务器上可以成功配置 fail2ban 服务，并且成功阻止了大量的攻击。既然这样，是否有方法可以让所有的内网 app 流量都被公网的 ssh 包一层，这样我只需要把好公网 ssh 这一关就可以了

    你的直觉非常敏锐！你刚才提出的思路其实就是网络安全领域经典的 **“堡垒机” (Bastion Host) 或 “跳板机” (Jump Server)** 模式。

    你说得对，之前的 HAProxy + TProxy 方案虽然技术上很酷，但**过于复杂**，维护成本极高，容易出错。

    既然你已经把好了公网服务器（Host A）这一关，并且 Fail2ban 已经在那里工作了，那么**利用 SSH 隧道将内网流量“包裹”在公网 SSH 连接中**绝对是最佳实践。

    这样，黑客只能看到你公网服务器的 SSH 端口，而内网的一切对互联网都是**不可见**的。

    **核心方案：SSH 反向隧道 (Remote Port Forwarding)**

    你不需要 FRP，也不需要 HAProxy。你只需要利用 Linux 自带的 `ssh` 命令。

    1. 架构图

        * **原来的架构（危险）：**
        `黑客 -> 公网 IP:6000 (FRP暴露) -> 内网 SSHD`
        *(黑客直接攻击内网端口，需要内网自己做防御)*
        * **现在的架构（安全）：**
        `黑客 -> 公网 IP:22 (SSH)` -> **被公网 Fail2ban 拦截**
        `你 -> 公网 IP:22 (SSH 验证通过) -> 隧道 -> 内网 SSHD`
        *(内网 SSHD 对公网完全不可见，只监听本地回环地址)*

    2. 实施步骤

        **第一步：在内网机器 (Host B) 上建立反向隧道**

        在内网机器上执行一条命令，主动连接公网机器，建立一条“秘密通道”。

        ```bash
        # 语法：ssh -R [公网监听端口]:[目标IP]:[目标端口] user@公网IP
        # 这里的 127.0.0.1:22 是指内网机器自己的 SSH
        ssh -NfR 12345:127.0.0.1:22 user@host_a_ip

        ```

        * `-R 12345:127.0.0.1:22`: 告诉公网机器（Host A），请监听你自己的 `127.0.0.1:12345`。凡是发给这个端口的数据，都通过加密隧道传回来，发给我的 `22` 端口。
        * `-N`: 不执行远程命令（只做端口转发）。
        * `-f`: 后台运行。

        **关键点：** 默认情况下，SSH 的 `-R` 转发端口只会绑定在公网服务器的 `127.0.0.1` 上，**不会绑定到 0.0.0.0**。这意味着：**外网黑客扫描你的公网 IP，根本看不到 12345 这个端口！** 只有登录进公网服务器的人才能用。

        **第二步：如何连接回家？**

        现在，你要回家访问内网机器，有两种优雅的方式：

        **方式 A：分两步跳（最易理解）**

        1. 先 SSH 登录公网机器：`ssh user@host_a_ip`
        2. 在公网机器内部，连接刚才建立的隧道端口：`ssh -p 12345 local_user@127.0.0.1`

        **方式 B：一条命令 ProxyJump（推荐，像直连一样丝滑）**
        在你的个人电脑（笔记本）上，修改 `~/.ssh/config`：

        ```ssh
        # 定义公网机器（跳板机）
        Host bastion
            HostName 1.2.3.4  # 公网IP
            User root         # 公网用户名
            Port 22

        # 定义内网机器
        Host home-server
            User hlc          # 内网用户名
            # 关键：通过 bastion 跳转
            ProxyJump bastion
            # 这里的 HostName 和 Port 是相对于 bastion 而言的
            HostName 127.0.0.1
            Port 12345

        ```

        配置好后，你在任何地方只需要输入：
        `ssh home-server`
        SSH 客户端会自动先连公网，再通过隧道连内网。全过程加密，且**只需要公网机器的 Fail2ban 守住大门即可**。

    3. 如何让连接更稳定（替代 FRP 的守护进程）

        SSH 连接断了怎么办？你需要 `autossh` 或者 `systemd` 来自动重连。

        **推荐使用 Systemd 服务（在内网机器 Host B 上配置）：**

        创建文件 `/etc/systemd/system/ssh-tunnel.service`：

        ```ini
        [Unit]
        Description=SSH Reverse Tunnel
        After=network.target

        [Service]
        User=hlc
        # 关键参数：
        # -o ExitOnForwardFailure=yes : 端口被占用等错误直接退出，让systemd重启
        # -o ServerAliveInterval=60 : 每60秒发心跳包防断连
        ExecStart=/usr/bin/ssh -NT -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -R 12345:127.0.0.1:22 user@host_a_ip -i /home/hlc/.ssh/id_rsa
        Restart=always
        RestartSec=10

        [Install]
        WantedBy=multi-user.target

        ```

        启用并运行：
        `sudo systemctl enable --now ssh-tunnel`

    ---

    这种方案的优势总结

    1. **极简安全模型**：你只需要维护公网服务器（Host A）的一个 `Fail2ban`。只要黑客攻不破 Host A 的 SSH，内网机器就是绝对安全的（甚至都不知道它的存在）。
    2. **无需公网暴露端口**：Host A 上不需要开放额外的 6000 端口给互联网。
    3. **无中间人欺骗**：不需要搞复杂的 HAProxy TProxy 和 IP Spoofing，因为不需要记录源 IP，**所有合法的流量都来自你认证过的 SSH 隧道**。
    4. **通用性**：如果你还需要访问内网的 Web 服务（比如 8080），只需要把 `-R` 改一下或者加一条 `-R 8080:127.0.0.1:8080` 即可。

    这个方案是否更符合你的预期？不需要任何复杂的网络黑魔法，回归最纯粹的 SSH 隧道。
* 详细讲讲 mcast

    **Mcast** 通常是 **Multicast（组播/多播）** 的缩写。它是计算机网络中的一种通信模式，介于单播（Unicast）和广播（Broadcast）之间。

    为了让你彻底搞懂它，我们从它的核心原理、工作机制、应用场景以及优缺点来详细拆解。

    1. 什么是组播（Multicast）？

        在网络通信中，有三种基础的传输方式：

        * **单播 (Unicast):** 一对一。你要发给 100 个人，就要复制 100 份数据分别发送。带宽消耗极大。
        * **广播 (Broadcast):** 一对所有。不管别人想不想听，直接轰炸整个局域网。范围受限（不能跨路由器），且打扰不相关的设备。
        * **组播 (Multicast):** 一对一组。数据源只发送 **一份** 数据，网络中的路由器和交换机会自动复制并分发给“加入了该组”的特定用户。

        > 💡 **生动的比喻：**
        > * 单播是**私信**：一个人一个人地发。
        > * 广播是**大喇叭**：整个村子的人都能听到，不管愿不愿意。
        > * 组播是**微信群**：只有进群（加入组播组）的人才能收到消息，群外的人不受影响。
        > 

    2. Mcast 的工作原理

        组播的实现依赖于一套完整的网络协议体系，主要解决两个问题：**“谁想接收？”**（主机到路由器）和“怎么把数据运过去？”（路由器到路由器）。

        * 2.1 组播 IP 地址

            组播不使用普通的 A、B、C 类 IP 地址，而是使用专门的 **D 类 IP 地址**。

            * **范围：** `224.0.0.0` 到 `239.255.255.255`。
            * 这些地址不分配给具体的某台电脑，而是代表一个“俱乐部（组播组）”。
            * **特殊地址举例：** `224.0.0.1`（子网内的所有主机）、`224.0.0.2`（子网内的所有路由器）。

        * 2.2 核心协议

            1. **IGMP (Internet Group Management Protocol - 组管理协议):**
            * **作用：** 运行在**主机和最后一跳路由器**之间。
            * 当你的电脑想看某个组播视频时，会通过 IGMP 告诉路由器：“我要加入 224.1.1.1 组，有这个组的数据请转发给我”。


            2. **PIM (Protocol Independent Multicast - 组播路由协议):**
            * **作用：** 运行在**路由器与路由器**之间。
            * 它们负责在整个网络中建立一棵“组播分发树”（Multicast Distribution Tree），确保数据能用最少的复制次数、最优的路径送到所有需要它的路由器。


            3. **IGMP Snooping (交换机优化):**
            * **作用：** 运行在**二层交换机**上。
            * 普通的交换机不懂 IP 组播，会把组播当成广播到处乱发。IGMP Snooping 让交换机去“偷听”主机的 IGMP 报文，从而精准地只把数据从连接了“群成员”的端口发出去。

    3. Mcast 的优势与劣势

        **优点**

        * **极高的带宽利用率：** 无论有多少个接收者，服务器都只需要发送一份数据。降低了主干网络的拥堵。
        * **减轻服务器压力：** 服务器不需要为每个用户建立独立的连接和CPU消耗。
        * **高效的资源分发：** 极其适合大流量、高并发的同质化数据传输。

        **缺点**

        * **基于 UDP，不可靠：** 组播绝大多数基于 UDP 协议，没有 TCP 的三次握手、确认和重传机制。丢包了就丢了（通常需要在应用层自己做丢包补偿）。
        * **容易造成网络雪崩（组播风暴）：** 如果网络配置不当（如缺少 IGMP Snooping），组播可能退化为广播，瘫痪整个局域网。
        * **跨网段配置复杂：** 组播路由（PIM）的配置和维护比普通单播路由复杂得多，因此在公网上很难全网实现（公网多用 CDN 代替），主要活跃在企业内网和专网。

    4. 常见应用场景

        1. **IPTV / 直播电视：** 运营商（如电信、联通）的机顶盒看电视直播，家家户户看同一个频道，后台用的就是组播。

        2. **金融行情股票快讯：** 交易所需要将瞬息万变的股票价格**同时、无延迟**地推送到成千上万台交易终端。

        3. **企业内网系统部署/镜像克隆：** 网管同时给机房的 200 台电脑安装系统，用组播可以一次性全部装完，而不需要一台一台下镜像。

        4. **多播视频会议：** 一个人发言，多方同时接收画面。

        5. **路由协议内部通信：** 如 OSPF、RIPv2 等路由协议，路由器之间交换路由信息时也会频繁使用组播地址（例如 OSPF 使用 `224.0.0.5`）。

    5. 补充：如果是编程代码中的 `mcast`？

        如果你是在某些编程语言（如 C, Python, Go）或 Linux 工具中看到 `mcast`，它通常是指组播套接字（Multicast Socket）的相关操作。例如在 Python 中实现一个简单的组播接收端：

        ```python
        import socket
        import struct

        MCAST_GRP = '224.1.1.1'
        MCAST_PORT = 5007

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.bind(('', MCAST_PORT))

        # 核心步骤：加入组播组
        mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        while True:
            print(sock.recv(1024))

        ```
* 详细讲讲 VXLAN。你说把数据包封在 udp 中，那 tcp 的包该如何处理？

    这是一个非常敏锐的技术直觉。理解 VXLAN 的关键在于意识到它是一种**隧道（Tunneling）**技术。

    简单来说：**VXLAN 并不关心它里面装的是 TCP 还是 UDP，因为它是在“包装快递”而非“拆开信件”。**

    1. 核心原理：大包套小包 (MAC-in-UDP)

        VXLAN（Virtual Extensible LAN）的操作逻辑是将一个完整的**二层以太网帧（Ethernet Frame）**，塞进一个**三层 UDP 数据包**的“载荷”部分。

        当你的虚拟机发送一个 TCP 包时，会发生以下“套娃”过程：

        1.  **原始包裹（内层）**：虚拟机产生的 TCP 包（包含 TCP 头、IP 头、以太网头）。
        2.  **VXLAN 封装**：物理机上的 VTEP（VXLAN 隧道端点）拿到这个包，在它外面贴上 **VXLAN 头部**（包含关键的 VNI 网络 ID）。
        3.  **UDP 封装**：再外面套上一个 **UDP 头部**（默认目的端口是 4789）。
        4.  **外层运输（外层 IP）**：最外面贴上物理机 A 发往物理机 B 的 **IP 头部**和**物理 MAC 地址**。

        **结论**：TCP 的包在 VXLAN 看来只是“货物”。物理网络只看到物理机之间在跑 UDP 包，只有到了对端物理机，剥掉外壳，原来的 TCP 包才会被还原。

    2. 详细的通信全过程

        假设虚拟机 VM1 (IP: 192.168.1.1) 要发 TCP 请求给另一台物理机上的 VM2 (IP: 192.168.1.2)：

        **第一阶段：ARP 请求 (广播)**

        *   VM1 发现目的地是 1.2，但不知道 MAC 地址，于是发出 ARP 广播。
        *   物理机 A 捕获到这个广播，将其封装成 VXLAN 广播包，通过物理网络发送给其他物理机。
        *   物理机 B 拆包，发现是找 VM2，于是转交给 VM2。

        **第二阶段：封装发送 (Unicast)**

        *   VM1 开始发送 TCP 数据包。
        *   **物理机 A (VTEP)**：看到包的目的地 MAC 属于 VM2，查询自己的映射表（知道 VM2 在物理机 B 上）。
        *   **物理层传输**：物理机 A 构建 UDP 包：`Source: Host-A-IP, Dest: Host-B-IP, Port: 4789`。
        *   **中间网络**：物理路由器、交换机只负责把这个 UDP 包从 A 搬到 B。

        **第三阶段：拆封交付**

        *   **物理机 B (VTEP)**：收到 4789 端口的 UDP 包，拆开，露出原始的二层以太网帧。
        *   **交付**：物理机 B 将原始帧丢进自己的网桥，VM2 收到这个包，在它看来，这就像是从同一个交换机直接传过来的一样。

    3. VXLAN 的三大关键点

        1. VNI (VXLAN Network Identifier)

            VXLAN 头部有 24 位用于标识 VNI。这意味着它支持多达 **1600 万**个虚拟网络。相比之下，传统 VLAN 只有 12 位（4096 个 ID）。这就是为什么它是云巨头（如 AWS、阿里云）的标准方案。

        2. MTU 限制 (必须要考虑的问题)

            这是最容易踩坑的地方！
            *   标准的以太网帧是 **1500 字节**。
            *   VXLAN 的外壳（UDP+IP+VXLAN Header）额外增加了 **50 字节**。
            *   **结果**：如果你的虚拟机发出的包是 1500，封装后变成 1550，会被物理网络丢弃。
            *   **解决办法**：要么把虚拟机的网卡 MTU 调小（改成 **1450**），要么把物理网络的 MTU 调大（开启 **Jumbo Frames**，设为 1600+）。

        3. 性能开销

            虽然多了封包操作，但现在的网卡大都支持 **VXLAN Offloading**（硬件卸载）。也就是说，封装和拆封的过程是由网卡芯片完成的，几乎不占物理机的 CPU，性能非常接近线速。

    **总结**

    你提到的 **TCP 包**，被 VXLAN 完好无损地包裹在 UDP 里面。这种“隧道”技术让虚拟机感知不到物理网络的存在。对于 VM 来说，它依然认为自己在一个普通的局域网里跑 TCP，而对于物理机来说，它只是在跟另一台物理机打 UDP 电话。
* 使用`getifaddrs()`拿到当前 host 的 network interface 信息

    `main.c`:

    ```c
    #include <stdio.h>
    #include <ifaddrs.h>
    #include <arpa/inet.h>  // struct sockaddr_in

    int main()
    {
        struct ifaddrs *if_addrs;
        int ret = getifaddrs(&if_addrs);
        if (ret < 0)
        {
            printf("fail to get if addrs\n");
            return -1;
        }

        struct ifaddrs *ifa_node = if_addrs;
        while (ifa_node)
        {
            printf("ifa name: %s\n", ifa_node->ifa_name);

            struct sockaddr_in *addr = (struct sockaddr_in *) ifa_node->ifa_addr;
            printf("\tsin_family: %d\n", addr->sin_family);

            char addr_str[16] = {0};
            inet_ntop(addr->sin_family, &addr->sin_addr.s_addr, addr_str, 16);
            printf("\taddr: %s\n", addr_str);

            ifa_node = ifa_node->ifa_next;
        }

        freeifaddrs(if_addrs);
        
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    ifa name: lo
    	sin_family: 17
    	addr: 
    ifa name: enp0s3
    	sin_family: 17
    	addr: 
    ifa name: docker0
    	sin_family: 17
    	addr: 
    ifa name: lo
    	sin_family: 2
    	addr: 127.0.0.1
    ifa name: enp0s3
    	sin_family: 2
    	addr: 10.0.2.4
    ifa name: docker0
    	sin_family: 2
    	addr: 172.17.0.1
    ifa name: lo
    	sin_family: 10
    	addr: ::
    ifa name: enp0s3
    	sin_family: 10
    	addr: 
    ```

    `getifaddrs()`在内部创建一个链表，并返回链表头的指针`struct ifaddrs *`。我们可以用遍历链表的方式得到每个 interface 的信息。

    上面出现的 address family 数值与枚举值的对应关系：

    | enum | value |
    | - | - |
    | `AF_INET` | 2 |
    | `AF_INET6` | 10 |
    | `AF_PACKET` | 17 |

* shutdown 和 close 都无法立即重新将同一个 fd bind 到一个 address + port 上

* 假设 node 1 上有 vm 1，node 2 上有 vm 2。vm 1 无法 ping 通 vm 2 可能是因为只设置了 node 1 上的路由表，没有设置 node 2 上的路由表

    猜想：可能是 vm 1 缎带 vm 2 发送完 icmp 数据包后，vm 2 回复 icmp 包时，找不到 vm 1 所在的网段如何路由过去。

    可以将 node 1 上的路由表新添加一项：vm 2 所在网段的 gateway 为 node 2 的 ip；在 node 2 上的路由表上也新加一条：vm 1 所在网段的 gateway 为 node 1 的 ip。这样就能 ping 通了。

* linux socket programming

    `server.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <stdio.h>
    #include <unistd.h>

    int main()
    {
        int serv_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (serv_fd < 0)
        {
            printf("fail to create server sock fd\n");
            return -1;
        }
        printf("[OK] create server socket fd: %d\n", serv_fd);

        uint16_t listen_port = 6543;
        uint32_t listen_addr_ipv4 = INADDR_ANY;
        char ipv4_addr[16] = {0};
        const char *ret_ptr = inet_ntop(AF_INET, &listen_addr_ipv4, ipv4_addr, 16);
        if (ret_ptr == NULL)
        {
            printf("fail to convert u32 to ipv4 str\n");
            return -1;
        }

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = listen_addr_ipv4;
        serv_addr.sin_port = htons(listen_port);
        int ret = bind(serv_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret < 0)
        {
            printf("fail to bind serv fd %d, ret: %d\n", serv_fd, ret);
            return -1;
        }
        printf("[OK] bind fd %d to addr %s: %u\n", serv_fd, ipv4_addr, listen_port);

        ret = listen(serv_fd, 5);
        if (ret < 0)
        {
            printf("fail to listen\n");
            return -1;
        }
        printf("[OK] start to listen...\n");

        struct sockaddr_in cli_addr;
        socklen_t cli_addr_len = sizeof(cli_addr);
        int cli_fd = accept(serv_fd, (struct sockaddr*) &cli_addr, &cli_addr_len);
        if (cli_fd < 0)
        {
            printf("fail to accept, ret: %d\n", cli_fd);
            return -1;
        }
        printf("[OK] accept 1 incoming client.\n");

        ret_ptr = inet_ntop(AF_INET, &cli_addr.sin_addr.s_addr, ipv4_addr, 16);
        if (ret_ptr == NULL)
        {
            printf("fail to convert u32 ipv4 to string\n");
            return -1;
        }
        printf("\tincoming client: ip: %s, port: %u\n", ipv4_addr, cli_addr.sin_port);

        char *buf = "hello from server";
        size_t buf_len = strlen(buf) + 1;
        ssize_t bytes_send = send(cli_fd, buf, buf_len, 0);
        if (bytes_send != buf_len)
        {
            printf("fail to send, buf_len: %lu, bytes_send: %ld\n", buf_len, bytes_send);
            return -1;
        }
        printf("[OK] send buf: %s\n", buf);

        ret = shutdown(cli_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown client fd %d, ret: %d\n", cli_fd, ret);
            return -1;
        }
        printf("[OK] shutdown client fd %d.\n", cli_fd);

        ret = close(cli_fd);
        if (ret != 0)
        {
            printf("fail to close fd %d\n", cli_fd);
            return -1;
        }
        printf("[OK] close fd %d.\n", cli_fd);

        ret = shutdown(serv_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown server fd %d, ret: %d\n", serv_fd, ret);
            return -1;
        }
        printf("[OK] shutdown server fd %d.\n", serv_fd);

        ret = close(serv_fd);
        if (ret != 0)
        {
            printf("fail to close fd %d\n", serv_fd);
            return -1;
        }
        printf("[OK] close fd %d.\n", serv_fd);

        return 0;
    }
    ```

    `client.c`:

    ```c
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <stdio.h>
    #include <unistd.h>

    int main()
    {
        int cli_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (cli_fd < 0)
        {
            printf("fail to create client sock fd\n");
            return -1;
        }
        printf("[OK] create client socket fd: %d\n", cli_fd);

        uint16_t serv_port = 6543;
        const char serv_ipv4[16] = "127.0.0.1";
        struct in_addr ipv4_addr;
        int ret = inet_pton(AF_INET, serv_ipv4, &ipv4_addr);
        if (ret != 1)
        {
            printf("fail to convert ipv4 string to u32, ret: %d\n", ret);
            return -1;
        }

        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr = ipv4_addr;
        serv_addr.sin_port = htons(serv_port);
        ret = connect(cli_fd, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
        if (ret != 0)
        {
            printf("fail to connect to server, ret: %d\n", ret);
            return -1;
        }
        printf("[OK] connect to server %s: %u\n", serv_ipv4, serv_port);

        char buf[20] = {0};
        size_t buf_len = 20;
        ssize_t bytes_recv = recv(cli_fd, buf, buf_len, 0);
        if (bytes_recv <= 0)
        {
            printf("fail to recv, buf_len: %lu, bytes_recv: %ld\n", buf_len, bytes_recv);
            return -1;
        }
        printf("[OK] recv buf: %s\n", buf);

        ret = shutdown(cli_fd, SHUT_RDWR);
        if (ret != 0)
        {
            printf("fail to shutdown fd %d, ret: %d\n", cli_fd, ret);
            return -1;
        }
        printf("[OK] shutdown fd %d.\n", cli_fd);

        ret = close(cli_fd);
        if (ret != 0)
        {
            printf("fail to close fd %d, ret: %d\n", cli_fd, ret);
            return -1;
        }
        printf("[OK] close fd %d.\n", cli_fd);

        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    all: server client

    server: server.c
        gcc -g server.c -o server

    client: client.c
        gcc -g client.c -o client

    clean:
        rm -f server client
    ```

    compile: `make`

    run:

    `./server`

    `./client`

    output:

    * server end

        ```
        [OK] create server socket fd: 3
        [OK] bind fd 3 to addr 0.0.0.0: 6543
        [OK] start to listen...
        [OK] accept 1 incoming client.
            incoming client: ip: 127.0.0.1, port: 61611
        [OK] send buf: hello from server
        [OK] shutdown client fd 4.
        [OK] close fd 4.
        [OK] shutdown server fd 3.
        [OK] close fd 3.
        ```

    * client end

        ```
        [OK] create client socket fd: 3
        [OK] connect to server 127.0.0.1: 6543
        [OK] recv buf: hello from server
        [OK] shutdown fd 3.
        [OK] close fd 3.
        ```

    说明：

    * `sys/socket.h`文件中主要包含下面几个函数：`socket()`, `bind()`, `connect()`, `send()`, `recv()`, `listen()`, `accept()`, `shutdown()`

        这些函数组成了 socket 的基本功能。

        这里的 socket 并不完全是为 internet 设计的，有一些 unix domain 或者其他的 socket 也会用到这个库。

    * 如果有 Internet 相关的需要，还需要添加头文件`#include <arpa/inet.h>`

        宏`INADDR_ANY`，`htons()`, `inet_ntop()`等函数都包含在这个头文件内。

    * `shutdown()`和`close()`并不能使刚 bind ipv4 addr 的 fd 重新 bind 相同的 ipv4 addr。

    * `inet_pton()`第三个参数注意填的是长度的指针，不是长度的值

* `select()`会更新 timeout 的值，更新的值为`total timeout - blocking time`

* 将 net 大小端的 32 位 addr 转换成 string

    ```c
    #include <arpa/inet.h>

    char addr_str[16] = {0};
    inet_ntop(AF_INET, &client_addr.sin_addr.s_addr, addr_str, 16);
    ```

* 无法确定 client / server 身份时的一个解决方案

    我们让 node 0 交替地充当 client 和 server 的角色，让 node 1 也交替地充当 clinet 和 server 的角色，如下图所示：

    <div style='text-align:center'>
    <img height=400 src='/home/hlc/Documents/documents/Reference_resources/ref_29/pic_1.png'>
    </div>

    可以看到图中有红色虚线标注的间隙，只要有这种间隙存在，就一定会一方的角色是 client，另一方的角色是 server，此时便可成功建立 socket 连接。而这种间隙出现的概率很大。（具体概率为多少？是否可以用随机过程计算出来？）

* 获取本机的 ipv4 地址

    ```c
    #include <sys/types.h>
    #include <ifaddrs.h>
    #include <arpa/inet.h>
    #include <stdio.h>
    #include <string.h>

    char my_ipv4_addr[16] = {0};

    int get_my_ipv4_addr(char *my_ipv4_addr)
    {
        struct ifaddrs *if_addrs;
        int ret = getifaddrs(&if_addrs);
        if (ret < 0)
        {
            printf("fail to get if addrs\n");
            return -1;
        }

        char* if_name_black_list[] = {
            "lo",
            "docker"
        };
        int if_name_black_list_len = sizeof(if_name_black_list) / sizeof(char*);

        struct ifaddrs *ifa_node = if_addrs;
        int is_valid_ip_addr = 0;
        while (ifa_node)
        {
            int is_ifname_black_list_matched = 0;
            for (int i = 0; i < if_name_black_list_len; ++i)
            {
                char *if_name = if_name_black_list[i];
                int if_name_len = strlen(if_name);
                int ret = strncmp(ifa_node->ifa_name, if_name, if_name_len);
                if (ret == 0)
                {
                    is_ifname_black_list_matched = 1;
                    break;
                }
            }

            if (is_ifname_black_list_matched)
            {
                ifa_node = ifa_node->ifa_next;
                continue;
            }

            struct sockaddr_in *addr = (struct sockaddr_in *) ifa_node->ifa_addr;
            if (addr->sin_family != AF_INET)
            {
                ifa_node = ifa_node->ifa_next;
                continue;
            }

            char addr_str[16] = {0};
            inet_ntop(AF_INET, &addr->sin_addr.s_addr, addr_str, 16);
            printf("name: %s, addr: %s\n", ifa_node->ifa_name, addr_str);
            strncpy(my_ipv4_addr, addr_str, 16);
            is_valid_ip_addr = 1;
            break;
        }
        freeifaddrs(if_addrs);

        if (!is_valid_ip_addr)
            return -1;

        return 0;
    }

    int main()
    {
        char my_ipv4_addr[16];
        int ret = get_my_ipv4_addr(my_ipv4_addr);
        if (ret < 0)
        {
            printf("fail to get my ipv4 addr\n");
            return -1;
        }
        printf("my ip addr: %s\n", my_ipv4_addr);
        return 0;
    }
    ```

    编译：

    `gcc -g main.c -o main`

    运行：

    `./main`

    output:

    ```
    name: enp0s3, addr: 10.0.2.4
    my ip addr: 10.0.2.4
    ```

    说明：

    * `getifaddrs()`这个函数可以枚举当前系统中的所有 net interface

    * `inet_ntop()`可以将大端的 32 位整数转换成 ipv4 字符串

        看起来既可以将字符串作为函数值返回，也可以传入一段 buffer，将字符串写入到 buffer 内。

    * 一个 ipv4 字符串 buffer 只需要 16 字节就够用了

        `255.255.255.255`，`255`占 3 个字节，一共 4 组，占`3 * 4 = 12`个字节。再加上 3 个点，一共`12 + 3 = 15`个字节。额外再加一个`\0`表示字符串的末尾，所以一共 16 个字节。

* NIC 指网卡，Network interface controller

* max 地址表示为 12 个十六进制数

* MTU (maxmimum transmission unit，最大传输单元)指的是数据链路层上能通过的最大负载的大小，单位为字节

    标准心碎网的 MTU 为 1500。

    如果 IP 层有数据包要发送，而数据包的长度超过了 MTU，IP 层就要对数据包进行分片（fragmentation）操作。

* 缓存（cache）使用的是静态随机存储（static random access memory, SRAM）

* 路由表

    如果在路由表里只填网络接口，destination ip/network 和子网掩码，不填 gateway，那么说明对于指定的 ip/网段使用指定的 interface 进行收发包。

    `0.0.0.0`指任意 ip，其对应的子网掩码为`0.0.0.0`。

## note

网络字节序：

小端法：高位存高地址，低位存低地址

大端法：高位存低地址，低位存高地址

网络数据流采用大端字节序，但是 intel 的 cpu 通常采用小端字节序。

常用的一些函数：

```c++
#include <arpa/inet.h>

uint32_t htonl(uint32_t hostlong);
uint16_t htons(uint16_t hostshort);
uint32_t ntohl(uint32_t netlong);
uint16_t ntohs(uint16_t netshort);
```

h 表示 host，n 表示 network，l 表示 32 位整数，s 表示 16 位整数。

ip 字符串与二进制的转换：

```c++
#include <arpa/inet.h>
int inet_pton(int af, const char *src, void *dst);
const char *inet_ntop(int af, const void *src, char *dst, socklen_t size);
```

* `af`: `AF_INET`, `AF_INET6`

TCP 通信流程图：

三次握手：

1. 客户端：syn, 包号，每个包的最大长度
1. 服务端：syn, 包号，ack，每个包的最大长度
1. 客户端：ack 包号

四次挥手（因为每次都是半关闭）：

1. 客户端：fin, ack
1. 服务端：ack
1. 服务端：fin, ack
1. 客户端：ack

滑动窗口（TCP 流量控制）：保证数据不会丢失。

**多进程服务器**：使用`fork()`创建子进程，使用信号捕捉函数`SIGCHLD`回收子进程。

`sigaction()`

TCP 状态时序图：

1. 主动发起连接请求端：

    1. CLOSE
    1. 发送 SYN
    1. SEND_SYN
    1. 接收 ACK, SYN
    1. SEND_SYN
    1. 发送 ACK
    1. ESTABLISHED（数据通信态）

1. 主动关闭连接请求端

    1. ESTABLISHED
    1. 发送 FIN
    1. FIN_WAIT_1
    1. 接收 ACK
    1. FIN_WAIT_2（半关闭）
    1. 接收对端发送 FIN
    1. FIN_WAIT_2（半关闭）
    1. 回发 ACK
    1. TIME_WAIT（只有主动关闭连接方会有这个状态）
    1. 等 2MSL 时长
    1. CLOSE

1. 被动接收连接请求端

    1. CLOSE
    1. LISTEN
    1. 接收 SYN
    1. LISTEN
    1. 发送 ACK, SYN
    1. SYN_RCVD
    1. 接收 ack
    1. etablished

1. 被动关闭连接

    1. established
    1. 接收 fin
    1. extablished
    1. 发送 ack
    1. close_wait (说明对端处于半关闭状态)
    1. 发送 fin
    1. last_ack
    1. 接收 ack
    1. close

2MSL 意义：保证最后一个 ack 能成功被对端接收。（等待期间，对端没收到我发的 ack，对端会再发 fin）因此它一定出现在主动关闭连接请求端。

`shutdown(int sockfd, int how)`可以关闭读缓冲或写缓冲。

`shutdown()`在关闭多个文件描述符应用的文件时，采用全关闭方法。`close()`只关闭一个。

`dup2()`

## 多路 IO 转接

* select

    `int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *excepfds, struct timeval *timeout);`

    * `nfds`：最大文件描述符加 1。（`lfd+1`，每次有新连接时更新）
    * `writefds`和`exceptfds`通常设置为`NULL`。

    返回所有监听集合中，满足对应事件的总数。

    ```c++
    FD_SET();
    FD_CLR();
    FD_ZERO();
    FD_ISSET();
    ```

    `lfd`和`cfd`都可以放到`readfds`里。

    `cfd`从`lfd+1`开始，一直到最大文件描述符。

    ```c++
    for (int i = lfd + 1; i <= maxfd; ++i)
    {
        FD_ISSET(i, &rset);
        read();
        // ...
        write();
    }
    ```

    可以用一个数组来维护已经存在的文件描述符，这样可以避免轮循，提高效率。

    优点：

    1. 跨平台。

    缺点：

    1. 文件描述符最大为 1023（`FD_SETSIZE = 1024`）。

* poll

    `int poll(struct pollfd *fds, nfds_t nfds, int timeout);`

    * `fds`：要监听的文件描述符的数组

    返回满足条件的事件数。

    优点：

    1. 自带数组结构，可以将监听事件集合和返回事件集合分离。

    1. 拓展监听上限，超出 1024 限制。

    缺点：

    1. 不能跨平台。

    1. 无法直接定位满足监听条件的文件描述符。

* epoll

    `epoll_create();`

    `epoll_ctl();`

    * `EPOLL_CTL_ADD`
    * `EPOLL_CTL_MOD`
    * `EPOLL_CTL_DEL`

    `epoll_wait()`

    * `EPOLLIN`, `EPOLLOUT`, `EPOLLERR`

    `cat /proc/sys/fs/file-max`：当前计算机一个进程可以打开的文件描述符上限。默认为 1024。

    `sudo vim /etc/security/limits.conf`可修改打开的文件描述符上限。

    ```
    * soft nofile 65536
    * hard nofile 100000
    ```

    也可直接用`ulimit`来修改。

    `ulimit -a`：当前用户的进程默认打开的文件描述符上限。

    ET (Edge Triggered)：边缘触发。只有数据到来才触发，不管缓冲区中是否还有数据。

    LT (Level Triggered)：水平触发只要有数据都会触发。(默认模式)

    结论：

    epoll 的 et 模式是高效模式，但只支持非阻塞模式（非阻塞的 socket）。阻塞模式下使用`readn()`/`readline()`等函数时，可能会发生死锁。

    优点：

    高效。突破 1024 文件描述符。

    缺点：

    1. 不能跨平台，只支持 linux。

`offsetof()`

`unlink()`

## 查看 ssl 版本

查看 ssl 版本：

For TLS 1.2:

`openssl s_client -connect www.google.com:443 -tls1_2`

For TLS 1.1:

`openssl s_client -connect www.google.com:443 -tls1_1`

For TLS 1:

`openssl s_client -connect www.google.com:443 -tls1`

或者使用 nmap：`nmap --script ssl-enum-ciphers -p 443 www.google.com`

参考资料：<https://stackoverflow.com/questions/40557031/command-prompt-to-check-tls-version-required-by-a-host>

## Proxy

Materials:

1. <https://github.com/NOLFXceptMe/http>

1. <https://github.com/monkeyWie/proxyee>

1. 
        

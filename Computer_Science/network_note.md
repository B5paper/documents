# Computer Network Note

## cache

* 如果我在 socket 编程，如何找到一个可用的端口

    在 socket 编程中，有多种方法可以找到可用端口：

    1. 让系统自动分配（推荐）

        ```python
        import socket

        def get_free_port():
            """让系统自动分配一个可用端口"""
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', 0))  # 绑定到所有地址，端口设为0（系统自动分配）
            port = sock.getsockname()[1]  # 获取系统分配的端口
            sock.close()
            return port

        # 使用
        free_port = get_free_port()
        print(f"系统分配的可用端口: {free_port}")
        ```

        **优点**：最简单可靠，系统保证端口可用

    2. 在指定范围内查找可用端口

        ```python
        import socket

        def find_free_port(start_port=1024, end_port=65535, max_attempts=100):
            """在指定范围内查找可用端口"""
            import random
            
            # 随机尝试，避免总是从固定端口开始
            ports = list(range(start_port, end_port + 1))
            random.shuffle(ports)
            
            for i, port in enumerate(ports[:max_attempts]):
                try:
                    # 尝试绑定 TCP 端口
                    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    tcp_sock.bind(('localhost', port))
                    tcp_sock.close()
                    
                    # 尝试绑定 UDP 端口（可选）
                    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    udp_sock.bind(('localhost', port))
                    udp_sock.close()
                    
                    return port
                except OSError:
                    continue
            
            raise RuntimeError(f"在端口 {start_port}-{end_port} 范围内未找到可用端口")

        # 使用
        try:
            port = find_free_port(8000, 9000)
            print(f"找到可用端口: {port}")
        except RuntimeError as e:
            print(e)
        ```

    3. 检查并绑定特定端口

        ```python
        import socket

        def check_and_bind_port(port, host='localhost'):
            """检查并绑定到指定端口"""
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                sock.bind((host, port))
                return sock, port  # 返回 socket 和端口
            except OSError as e:
                sock.close()
                raise RuntimeError(f"端口 {port} 不可用: {e}")

        # 使用
        try:
            sock, port = check_and_bind_port(8080)
            print(f"成功绑定到端口: {port}")
        except RuntimeError as e:
            print(e)
        ```

    4. 完整的使用示例

        ```python
        import socket
        import random

        class PortFinder:
            def __init__(self, min_port=1024, max_port=65535):
                self.min_port = min_port
                self.max_port = max_port
            
            def is_port_available(self, port, host='0.0.0.0', socket_type='tcp'):
                """检查端口是否可用"""
                try:
                    if socket_type.lower() == 'tcp':
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((host, port))
                    sock.close()
                    return True
                except OSError:
                    return False
            
            def find_free_port(self, preferred_ports=None):
                """查找可用端口，可指定优先端口列表"""
                # 1. 先尝试优先端口
                if preferred_ports:
                    for port in preferred_ports:
                        if self.is_port_available(port):
                            return port
                
                # 2. 系统自动分配
                try:
                    return self.get_system_allocated_port()
                except:
                    pass
                
                # 3. 在范围内随机查找
                attempts = min(100, self.max_port - self.min_port)
                ports = random.sample(range(self.min_port, self.max_port + 1), attempts)
                
                for port in ports:
                    if self.is_port_available(port):
                        return port
                
                raise RuntimeError("未找到可用端口")
            
            def get_system_allocated_port(self):
                """让系统分配端口"""
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', 0))
                port = sock.getsockname()[1]
                sock.close()
                return port

        # 使用示例
        finder = PortFinder(min_port=8000, max_port=9000)

        # 方法1：系统自动分配
        auto_port = finder.get_system_allocated_port()
        print(f"系统分配端口: {auto_port}")

        # 方法2：查找可用端口
        free_port = finder.find_free_port(preferred_ports=[8080, 8888, 3000])
        print(f"找到的可用端口: {free_port}")

        # 方法3：检查特定端口
        if finder.is_port_available(8080):
            print("端口 8080 可用")
        else:
            print("端口 8080 已被占用")
        ```

    5. 结合上下文管理器使用

        ```python
        import socket
        from contextlib import contextmanager

        @contextmanager
        def acquire_free_port(host='localhost', socket_type='tcp'):
            """上下文管理器，自动获取并释放端口"""
            sock = None
            try:
                if socket_type.lower() == 'tcp':
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                sock.bind((host, 0))
                port = sock.getsockname()[1]
                yield sock, port
            finally:
                if sock:
                    sock.close()

        # 使用
        with acquire_free_port() as (sock, port):
            print(f"获取到端口: {port}")
            # 在这里使用 socket
            sock.listen(5)
            # ... 其他操作
        # 退出上下文后自动关闭
        ```

    6. 考虑竞态条件的版本

        ```python
        import socket
        import errno

        def find_and_reserve_port(start_port=1024, max_attempts=100):
            """查找并立即绑定端口，避免竞态条件"""
            for port in range(start_port, start_port + max_attempts):
                try:
                    # 创建 socket 并立即绑定
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(('localhost', port))
                    
                    # 可以在这里设置监听，防止其他进程占用
                    sock.listen(5)
                    
                    return sock, port  # 返回已绑定的 socket
                    
                except socket.error as e:
                    if e.errno == errno.EADDRINUSE:
                        continue  # 端口被占用，尝试下一个
                    else:
                        raise  # 其他错误
            
            raise RuntimeError(f"在 {max_attempts} 次尝试后未找到可用端口")

        # 使用
        try:
            sock, port = find_and_reserve_port(8000)
            print(f"已绑定并保留端口: {port}")
            # 使用 sock 进行后续操作
        finally:
            sock.close()
        ```

    最佳实践建议：

    1. **生产环境**：使用 `bind(('', 0))` 让系统自动分配
    2. **测试环境**：可以在固定范围内查找，便于调试
    3. **重要提示**：
       - 使用 `SO_REUSEADDR` 选项可以更快地重用端口
       - 检查端口后应立即绑定，避免竞态条件
       - 考虑同时检查 TCP 和 UDP 端口（如果需要）
       - 处理 `EADDRINUSE` 等特定错误

    **简单封装函数**

    ```python
    def get_available_port(host='localhost', default=0):
        """
        获取可用端口
        :param host: 绑定主机
        :param default: 0 表示系统分配，>0 表示检查该端口是否可用
        :return: 可用端口号
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, default))
                return s.getsockname()[1]
            except OSError:
                if default > 0:
                    # 如果指定端口不可用，让系统分配
                    s.bind((host, 0))
                    return s.getsockname()[1]
                raise

    # 使用
    port1 = get_available_port()  # 系统分配
    port2 = get_available_port(default=8080)  # 尝试8080，不行则系统分配
    ```

    **推荐使用第一种方法（系统自动分配）**，它最简单、最可靠，且避免了竞态条件问题。
* 我的 haproxy 配置如下：

    ```conf
    global
    	log /dev/log	local0
    	log /dev/log	local1 notice
    	chroot /var/lib/haproxy
    	stats socket /run/haproxy/admin.sock mode 660 level admin expose-fd listeners
    	stats timeout 30s
    	user haproxy
    	group haproxy
    	daemon

    	# Default SSL material locations
    	ca-base /etc/ssl/certs
    	crt-base /etc/ssl/private

    	# See: https://ssl-config.mozilla.org/#server=haproxy&server-version=2.0.3&config=intermediate
            ssl-default-bind-ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384
            ssl-default-bind-ciphersuites TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256
            ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets

    defaults
    	log	global
    	mode	tcp
    	option	httplog
    	option	dontlognull
            timeout connect 5000
            timeout client  50000
            timeout server  50000
    	errorfile 400 /etc/haproxy/errors/400.http
    	errorfile 403 /etc/haproxy/errors/403.http
    	errorfile 408 /etc/haproxy/errors/408.http
    	errorfile 500 /etc/haproxy/errors/500.http
    	errorfile 502 /etc/haproxy/errors/502.http
    	errorfile 503 /etc/haproxy/errors/503.http
    	errorfile 504 /etc/haproxy/errors/504.http

    frontend fe_pp
    	bind *:6543 accept-proxy
    	option tcplog
    	default_backend backend_apps

    backend backend_apps
        option tcplog
        server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
    ```

    haproxy 的 status 如下：

    ```bash
    ● haproxy.service - HAProxy Load Balancer
         Loaded: loaded (/lib/systemd/system/haproxy.service; enabled; vendor preset: enabled)
         Active: active (running) since Sun 2025-11-30 16:39:52 CST; 36min ago
           Docs: man:haproxy(1)
                 file:/usr/share/doc/haproxy/configuration.txt.gz
        Process: 60540 ExecReload=/usr/sbin/haproxy -Ws -f $CONFIG -c -q $EXTRAOPTS (code=exited, status=0/SUCCESS)
        Process: 60542 ExecReload=/bin/kill -USR2 $MAINPID (code=exited, status=0/SUCCESS)
       Main PID: 59301 (haproxy)
          Tasks: 5 (limit: 19071)
         Memory: 38.6M
            CPU: 110ms
         CGroup: /system.slice/haproxy.service
                 ├─59301 /usr/sbin/haproxy -sf 59303 -x /run/haproxy/admin.sock -Ws -f /etc/haproxy/haproxy.cfg -p /run/haproxy.pid -S /run/haproxy-master.sock
                 └─59303 /usr/sbin/haproxy -Ws -f /etc/haproxy/haproxy.cfg -p /run/haproxy.pid -S /run/haproxy-master.sock
    ```

    为什么 service haproxy start 后，无法访问 6543?

    ```
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ nc -zvn 127.0.0.1 6543
    nc: connect to 127.0.0.1 port 6543 (tcp) failed: Connection refused
    ```

    我仔细看了你的配置和状态，有几个关键点可能导致你 `6543` 无法访问：

    ---

    1. `frontend fe_pp` 的 `bind` 配置

        你写的是：

        ```haproxy
        frontend fe_pp
            bind *:6543 accept-proxy
        ```

        * `accept-proxy` **表示 HAProxy 期望客户端发送 PROXY 协议头**，否则直接连接会被拒绝。
        * 如果你用 `nc` 或普通 TCP 客户端测试，而客户端没有发送 PROXY 头，那么 HAProxy 会立即拒绝连接，从而出现 **Connection refused**。

        **解决方案：**

        * 如果不需要 PROXY 协议，可以改成：

        ```haproxy
        bind *:6543
        ```

        * 如果必须使用 PROXY 协议，测试时也要用支持 PROXY 协议的客户端，比如 `haproxy-proxy-protocol` 工具或者另一个 HAProxy 做源转发。

    ---

    2. 后端 `backend_apps` 的配置

        ```haproxy
        server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
        ```

        * `usesrc clientip` 会尝试把客户端 IP 用作源 IP 去连接后端。
        * 对于 `127.0.0.1` 的本地连接，Linux **通常不允许任意源 IP 发包到 127.0.0.1**，所以健康检查可能失败，连接被拒绝。
        * 如果只是本地测试，可以去掉 `usesrc clientip`：

        ```haproxy
        server app1 127.0.0.1:5432 check
        ```

    ---

    3. `mode tcp` 与日志

        你在 `defaults` 和 `frontend/backend` 都使用了 `mode tcp`，这是正确的，如果你是做 PostgreSQL 或其他 TCP 服务代理。

        * 日志显示没有错误，HAProxy 服务是启动状态，只是拒绝了 TCP 连接。
        * 所以问题核心就是 **`accept-proxy` + `usesrc clientip`** 这两个配置导致本地无法连接。

    ---

    ✅ 建议修改后的最简配置（本地测试可用）

    ```haproxy
    frontend fe_pp
        bind *:6543        # 去掉 accept-proxy
        option tcplog
        default_backend backend_apps

    backend backend_apps
        option tcplog
        server app1 127.0.0.1:5432 check  # 去掉 usesrc
    ```

    然后重载 HAProxy：

    ```bash
    sudo systemctl restart haproxy
    nc -zv 127.0.0.1 6543
    ```

    ---

    总结：

    1. `accept-proxy` 会拒绝普通 TCP 客户端。
    2. `usesrc clientip` 不能在 `127.0.0.1` 做源 IP 转发。

    这两个配置同时存在是你连接失败的根本原因。

    ---

    我可以帮你改成 **既支持普通 TCP 客户端访问，又可以保留 PROXY 协议选项** 的安全配置，这样本地测试和生产都可以用。你希望我直接给这个版本吗？

* 我使用的是 proxy protocol v1，accept-proxy这个配置正确吗

    明白了，如果你确实在客户端和 HAProxy 之间使用 **PROXY protocol v1**，那么 `accept-proxy` 是 **正确的配置**。

    但是结合你遇到的 `Connection refused` 现象，有几个容易被忽略的问题：

    ---

    1. 客户端是否真的发送了 PROXY 协议头

        * `accept-proxy` 的作用是：**HAProxy 只接受带 PROXY 协议头的连接**。
        * 如果你直接用 `nc 127.0.0.1 6543` 测试，这个客户端 **没有发送 PROXY 协议头**，HAProxy 会拒绝连接。

          * 这就是你看到 `Connection refused` 的原因。
        * 正确测试方法是使用支持 PROXY 协议的客户端，例如：

          * `haproxy-proxy-protocol` 工具
          * 另一个 HAProxy 或 Nginx 配置 `send-proxy` 去连接
          * PostgreSQL 等支持 PROXY 协议的客户端

    2. 后端 `usesrc clientip` 问题仍存在

        * 你的后端配置是：

        ```haproxy
        server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
        ```

        * 当 `clientip` 是来自 **127.0.0.1** 时，Linux 不允许任意源 IP 发包到本地 `127.0.0.1`。
        * 这可能导致健康检查失败或者后端连接失败，尤其是本地测试时。
        * 解决办法：

          * 本地测试去掉 `usesrc clientip`：

            ```haproxy
            server app1 127.0.0.1:5432 check
            ```
          * 生产环境需要保留客户端真实 IP，可用真实网卡 IP 或 loopback IP 替代 0.0.0.0。

    3. 总结

        * `accept-proxy` **是正确的**，前提是客户端真的发送 PROXY 协议头。
        * 如果只是用 `nc` 或普通 TCP 客户端测试，会拒绝。
        * 后端 `usesrc clientip` 对本地 `127.0.0.1` 连接不适用。

    如果你希望，我可以给你写一个 **正确使用 PROXY protocol v1 并且本地测试也能通的 HAProxy 配置示例**，保证既能用 `nc` 测试，又能保留客户端 IP 转发功能。

    你希望我帮你写吗？

* 但是我使用 netstat -nao 看到 haproxy 并没有监听 6543 端口，这对吗

    啊，这就是关键！如果 `netstat -nao` 看不到 HAProxy 在 `6543` 上监听，那么问题 **不在客户端或 PROXY 协议**，而在 HAProxy 本身 **根本没有绑定成功**。

    结合你的配置，我看到几个可能原因：

    ---

    1. `accept-proxy` 并不会阻止绑定

        * `accept-proxy` 只影响连接是否被接受，不会导致端口无法绑定。
        * 所以如果 HAProxy 没有监听，原因不是 `accept-proxy`。

    ---

    2. 检查 `frontend` 的 bind 是否生效

        你的配置：

        ```haproxy
        frontend fe_pp
            bind *:6543 accept-proxy
        ```

        * HAProxy 会尝试绑定 `0.0.0.0:6543`（所有网卡）
        * 如果 **端口被占用**，HAProxy 启动时会报错并退出绑定，但你的 `systemctl status haproxy` 显示 **Active: running**，说明 HAProxy 并没有直接崩溃，但可能 **frontend 没有激活**。
        * 可以查看 HAProxy 日志：

        ```bash
        sudo journalctl -u haproxy -e
        ```

        看是否有类似：

        ```
        [ALERT] 0000: bind(): Address already in use [0.0.0.0:6543]
        ```

        或者其他错误。

    ---

    3. systemd 的 socket 激活冲突

        * 你用的是 `/run/haproxy/admin.sock` 做 stats socket
        * 如果你使用了 **systemd socket activation** 的 HAProxy 服务，配置文件中的 `bind` 可能被忽略，需要确认 `/lib/systemd/system/haproxy.service` 是否使用 `socket` 激活模式。

    ---

    4. 权限问题

        * 你绑定的端口是 **6543 > 1024**，普通用户可以绑定，不是 root 端口，所以一般没问题
        * 但是你配置了 `chroot /var/lib/haproxy`，HAProxy 运行时 chroot 到该目录，如果目录权限不对，可能导致 bind 失败
        * 可以检查：

        ```bash
        ls -ld /var/lib/haproxy
        ```

        目录必须存在并可被 `haproxy` 用户访问。

    ---

    5. 快速排查方法

        1. 停止 HAProxy：

            ```bash
            sudo systemctl stop haproxy
            ```

        2. 手动启动并查看日志：

            ```bash
            sudo haproxy -f /etc/haproxy/haproxy.cfg -d
            ```

            * `-d` 会在前台打印调试信息
            * 看是否有报错或者 `bind *:6543` 成功信息

    ---

    💡 **结论**

    如果 HAProxy 没有监听 6543，`accept-proxy` 不是原因，最可能是：

    1. 端口被占用
    2. chroot 或权限问题
    3. HAProxy 配置错误导致 frontend 未激活

* haproxy 失败原因分析

    ```log
    11月 30 17:22:56 hlc-VirtualBox systemd[1]: Stopped HAProxy Load Balancer.
    ░░ Subject: A stop job for unit haproxy.service has finished
    ░░ Defined-By: systemd
    ░░ Support: http://www.ubuntu.com/support
    ░░ 
    ░░ A stop job for unit haproxy.service has finished.
    ░░ 
    ░░ The job identifier is 10394 and the job result is done.
    11月 30 17:22:56 hlc-VirtualBox systemd[1]: haproxy.service: Start request repeated too quickly.
    11月 30 17:22:56 hlc-VirtualBox systemd[1]: haproxy.service: Failed with result 'exit-code'.
    ░░ Subject: Unit failed
    ░░ Defined-By: systemd
    ░░ Support: http://www.ubuntu.com/support
    ░░ 
    ░░ The unit haproxy.service has entered the 'failed' state with result 'exit-code'.
    11月 30 17:22:56 hlc-VirtualBox systemd[1]: Failed to start HAProxy Load Balancer.
    ░░ Subject: A start job for unit haproxy.service has failed
    ░░ Defined-By: systemd
    ░░ Support: http://www.ubuntu.com/support
    ░░ 
    ░░ A start job for unit haproxy.service has finished with a failure.
    ░░ 
    ░░ The job identifier is 10394 and the job result is failed.
    ```

    谢谢你提供日志，这里就很清楚了：

    ```
    haproxy.service: Start request repeated too quickly.
    haproxy.service: Failed with result 'exit-code'.
    Failed to start HAProxy Load Balancer.
    ```

    这说明 HAProxy **启动失败了**，systemd 尝试重启几次都失败，于是标记服务为 failed。

    也就是说，之前你看到的 `Active: active` 很可能是 **systemd 的残留状态** 或者你是在不同时间查的状态。核心问题是 HAProxy **根本没有成功启动**，所以自然 **没有监听 6543**。

    ---

    可能原因

    1. **配置语法或逻辑错误**

        ```bash
        sudo haproxy -c -f /etc/haproxy/haproxy.cfg
        ```

        * 这个命令可以检查配置是否正确
        * 很可能你的 `bind *:6543 accept-proxy` 或 `usesrc clientip` 在当前环境下触发错误

    2. **端口占用或权限问题**

        * 使用 `sudo lsof -i :6543` 或 `sudo netstat -tlnp | grep 6543` 检查端口是否被占用

    3. **chroot 目录权限问题**

        * 你配置了 `chroot /var/lib/haproxy`
        * HAProxy 运行时需要在该目录下能访问日志、socket、证书等
        * 如果目录不存在或权限不对，启动会失败

    ---

    建议排查步骤

    1. **手动验证配置**

        ```bash
        sudo haproxy -f /etc/haproxy/haproxy.cfg -c
        ```

        * 如果报错，会显示具体行号和原因

    2. **前台调试启动**

        ```bash
        sudo haproxy -f /etc/haproxy/haproxy.cfg -d
        ```

        * `-d` 会打印详细调试信息，包括 bind 失败原因

    3. **检查日志**

        ```bash
        sudo journalctl -u haproxy -e
        ```

        * 查找 `[ALERT]` 或 `Error` 信息

    ---

    💡 **重点**

    你现在看到 `Connection refused` 和端口没监听，根本原因是 **HAProxy 启动失败**，必须先解决启动失败问题，再谈 6543 监听和 PROXY protocol。

    ---

    如果你愿意，我可以帮你分析你的配置，把 `accept-proxy` + `usesrc clientip` + `chroot` 这些潜在问题改成 **能成功启动并支持 PROXY v1 的配置**。

    你希望我直接给这个可用版本吗？

    ```log
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ sudo haproxy -c -f /etc/haproxy/haproxy.cfg
    [WARNING]  (61925) : parsing [/etc/haproxy/haproxy.cfg:42] : backend 'backend_apps' : 'option tcplog' directive is ignored in backends.
    Warnings were found.
    Configuration file is valid
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ sudo lsof -i :6543
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ sudo netstat -tlnp | grep 6543
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ ls -lh /var/lib/haproxy
    total 4.0K
    drwxr-xr-x 2 root root 4.0K 11月 30 16:39 dev
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ sudo haproxy -f /etc/haproxy/haproxy.cfg -d
    [WARNING]  (61980) : parsing [/etc/haproxy/haproxy.cfg:42] : backend 'backend_apps' : 'option tcplog' directive is ignored in backends.
    Note: setting global.maxconn to 524270.
    Available polling systems :
          epoll : pref=300,  test result OK
           poll : pref=200,  test result OK
         select : pref=150,  test result FAILED
    Total: 3 (2 usable), will use epoll.

    Available filters :
    	[SPOE] spoe
    	[CACHE] cache
    	[FCGI] fcgi-app
    	[COMP] compression
    	[TRACE] trace
    Using epoll() as the polling mechanism.
    [NOTICE]   (61980) : haproxy version is 2.4.29-0ubuntu0.22.04.1
    [NOTICE]   (61980) : path to executable is /usr/sbin/haproxy
    [ALERT]    (61980) : [haproxy.main()] Some configuration options require full privileges, so global.uid cannot be changed.
    ```

    很好，你的调试信息已经非常关键了。我们可以从中看到几个明确问题点：

    ---

    1. 配置语法本身没问题

        ```text
        Configuration file is valid
        ```

        * `-c` 检查配置通过，只有一个 **warning**：

        ```
        [WARNING] : backend 'backend_apps' : 'option tcplog' directive is ignored in backends.
        ```

        * 这个 warning 无关紧要，不会阻止 HAProxy 启动。
        * 说明配置文件语法本身没有致命错误。

    ---

    2. 关键的 ALERT 信息

        ```text
        [ALERT] : [haproxy.main()] Some configuration options require full privileges, so global.uid cannot be changed.
        ```

        * 你的 `global` 配置里有：

        ```haproxy
        user haproxy
        group haproxy
        chroot /var/lib/haproxy
        ```

        * **意思是**：HAProxy 需要 root 权限来绑定端口和使用某些配置，但 **尝试切换到 haproxy 用户失败**，因为当前执行环境权限不够或 chroot 目录权限不对。
        * 对应的症状就是 **HAProxy 启动失败，端口没有监听**。

    ---

    3. `chroot` 目录权限问题

        ```bash
        ls -lh /var/lib/haproxy
        total 4.0K
        drwxr-xr-x 2 root root 4.0K 11月 30 16:39 dev
        ```

        * HAProxy 在启动时会 chroot 到 `/var/lib/haproxy`。
        * 当前目录是 **root:root**，haproxy 用户无法写入或访问 socket/log 文件。
        * HAProxy 在 chroot 后找不到必需的文件，导致启动失败。
        * 你必须保证 chroot 目录至少可被 haproxy 用户读取和写入必要文件。

    ---

    4. `accept-proxy` 与端口 6543

        * 端口 6543 没被监听，是因为 HAProxy **根本没有成功启动**，跟 `accept-proxy` 无关。
        * 一旦解决了 **权限 + chroot** 问题，HAProxy 就会正常监听 6543，并处理 PROXY 协议。

    ---

    ✅ 解决方案

    1. 给 chroot 目录权限：

        ```bash
        sudo mkdir -p /var/lib/haproxy
        sudo chown haproxy:haproxy /var/lib/haproxy
        sudo chmod 755 /var/lib/haproxy
        ```

        * 确保 HAProxy 用户可以访问 chroot 目录。

    2. 使用 root 启动 HAProxy（systemd 默认就是 root），确保能切换用户：

        * systemd 配置通常是：

        ```text
        ExecStart=/usr/sbin/haproxy -Ws -f /etc/haproxy/haproxy.cfg -p /run/haproxy.pid
        ```

        * 启动时 root 可以 bind 高端口并执行 chroot，然后再切换到 haproxy 用户。

    3. 测试启动：

        ```bash
        sudo systemctl restart haproxy
        sudo systemctl status haproxy
        sudo netstat -tlnp | grep 6543
        ```

    4. 保留 `accept-proxy`，前提是客户端发送 PROXY v1。

    ---

    💡 **总结**

    * 核心问题：**HAProxy chroot 权限 + uid 切换失败** → 启动失败 → 没监听端口。
    * 配置语法是正确的，PROXY 协议 `accept-proxy` 是没问题的。
    * 修复 chroot 权限和确保以 root 启动即可。

    ---

    我可以帮你写一份 **修改后的最小可用 HAProxy 配置**，既支持 PROXY v1，又能在本地启动监听 6543，方便测试。

    你希望我直接写吗？

    ```bash
    (base) hlc@hlc-VirtualBox:/etc/haproxy$ sudo haproxy -f /etc/haproxy/haproxy.cfg -d
    [WARNING]  (62489) : parsing [/etc/haproxy/haproxy.cfg:42] : backend 'backend_apps' : 'option tcplog' directive is ignored in backends.
    Note: setting global.maxconn to 524270.
    Available polling systems :
          epoll : pref=300,  test result OK
           poll : pref=200,  test result OK
         select : pref=150,  test result FAILED
    Total: 3 (2 usable), will use epoll.

    Available filters :
    	[SPOE] spoe
    	[CACHE] cache
    	[FCGI] fcgi-app
    	[COMP] compression
    	[TRACE] trace
    Using epoll() as the polling mechanism.
    [NOTICE]   (62489) : haproxy version is 2.4.29-0ubuntu0.22.04.1
    [NOTICE]   (62489) : path to executable is /usr/sbin/haproxy
    [ALERT]    (62489) : [haproxy.main()] Some configuration options require full privileges, so global.uid cannot be changed.

    改了 /var/lib/haproxy 权限，好像还是不行呀
    ```

    明白了，你改了 `/var/lib/haproxy` 的权限，但 HAProxy 仍然报：

    ```
    [ALERT] : Some configuration options require full privileges, so global.uid cannot be changed.
    ```

    这说明 **问题不在 chroot 目录的权限**，而是 **HAProxy 启动时没有足够的特权去执行 `uid`/`gid` 切换或 chroot**。

    ---

    **分析原因**

    你配置里：

    ```haproxy
    global
        user haproxy
        group haproxy
        chroot /var/lib/haproxy
    ```

    * 当 HAProxy 启动时，如果 **不是以 root 用户运行**，它 **无法切换到 haproxy 用户**，也无法成功 chroot。
    * 而你的 `sudo haproxy -f ... -d` 实际上可能 **没有使用 systemd 的完整启动环境**，或者 `-d` 调试模式改变了权限处理。

    所以即便 `/var/lib/haproxy` 权限正确，也会报这个 alert。

    ---

    **解决方法**

    方法 1：保持 systemd 启动

    * 使用 systemd 启动 HAProxy（推荐）：

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl restart haproxy
    sudo systemctl status haproxy
    ```

    * systemd 会以 **root 启动**，HAProxy 能正常切换用户并 chroot。
    * 直接用 `haproxy -d` 调试有时会触发权限警告，这属于正常行为，不影响 systemd 正常启动。

    ---

    方法 2：临时绕过 `user` / `group` / `chroot`

    * 如果只是本地测试，可以先注释掉：

    ```haproxy
    # user haproxy
    # group haproxy
    # chroot /var/lib/haproxy
    ```

    * 然后再用：

    ```bash
    sudo haproxy -f /etc/haproxy/haproxy.cfg -d
    ```

    * 这样就不会报 `full privileges` 的 alert，并且可以确认 `6543` 端口能否监听。

    ---

    方法 3：确保 PROXY v1 可用

    * 保留 `accept-proxy`：

    ```haproxy
    frontend fe_pp
        bind *:6543 accept-proxy
    ```

    * 端口监听成功后，用支持 PROXY 协议的客户端测试。

    ---

    💡 **关键总结**

    1. `[ALERT] ... full privileges` 不是配置错误，而是 **运行 HAProxy 的用户不是 root**。
    2. systemd 启动 HAProxy 时默认以 root 运行，可以正确 chroot 和切换用户。
    3. 手动调试时可以暂时注释 `user` / `group` / `chroot`，确保能监听端口和调试 PROXY。

    ---

    如果你需要，我可以帮你写一份 **能用 systemd 启动、支持 PROXY v1、可监听 6543 的完整 HAProxy 配置示例**，保证本地测试可用。

    你希望我直接写吗？

* 解释下 source 0.0.0.0 usesrc clientip 这一行的作用

    好的，我来详细解释这一行配置的含义和作用：

    你的配置是：

    ```haproxy
    server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
    ```

    ---

    1. `source 0.0.0.0`

        * `source` 用来指定 **HAProxy 连接到后端服务器时使用的源 IP 地址**。
        * `0.0.0.0` 这里不是一个固定 IP，而是一个 **占位符**，表示 HAProxy 会根据 `usesrc` 自动选择源 IP。
        * 如果你写成 `source 192.168.1.10`，HAProxy 会强制用这个 IP 作为源 IP 发起 TCP 连接。

    ---

    2. `usesrc clientip`

        * `usesrc` 指定 **源 IP 的选择策略**。
        * `clientip` 的意思是：**HAProxy 会尝试把 TCP 连接的源 IP 设置为客户端的 IP**，也就是把客户端的真实 IP 传给后端服务器。
        * 配合 `source 0.0.0.0` 使用，HAProxy 会从本机的网卡中选一个 **可以用于模拟客户端 IP 的本地地址**。

    ---

    3. 联合效果

        * 当 HAProxy 收到客户端连接（假设客户端 IP 是 `203.0.113.5`）：

          1. HAProxy 向后端 127.0.0.1:5432 发起 TCP 连接
          2. 源 IP 不再是 HAProxy 的默认 IP（例如 127.0.0.1），而是 **客户端 IP**
          3. 后端可以看到原始客户端 IP，而不是 HAProxy 的 IP

    ---

    4. 限制和注意事项

        1. **必须有特权**

           * 要把 TCP 连接的源 IP 改成任意 IP，HAProxy 必须运行 **root** 或有 `CAP_NET_RAW` 权限，否则会失败。

        2. **本地测试问题**

           * 如果你后端是 `127.0.0.1`（loopback），Linux 不允许任意客户端 IP 作为源 IP 发送到 127.0.0.1。
           * 这就是你在本地测试时，`source 0.0.0.0 usesrc clientip` 会导致连接失败的原因。

        3. **生产环境**

           * 在真实网络接口上，`usesrc clientip` 可以让后端直接看到真实客户端 IP，非常适合 TCP 透明代理场景。

    ---

    💡 **总结**

    * `source 0.0.0.0 usesrc clientip` = “用客户端的真实 IP 作为 HAProxy 到后端的源 IP 发起连接”。
    * 优点：后端能获取真实客户端 IP
    * 限制：必须 root 或特权、不能在 127.0.0.1 上模拟客户端 IP

    ---

    如果你希望，我可以给你画一个 **HAProxy 客户端 → HAProxy → 后端 的源 IP变化示意图**，让这个概念一目了然。

    你希望我画吗？


    ```bash
     (base) hlc@hlc-VirtualBox:/etc/haproxy$ sudo haproxy -f /etc/haproxy/haproxy.cfg -d
    [WARNING]  (63518) : parsing [/etc/haproxy/haproxy.cfg:40] : backend 'backend_apps' : 'option tcplog' directive is ignored in backends.
    [WARNING]  (63518) : config : log format ignored for frontend 'fe_pp' since it has no log address.
    Note: setting global.maxconn to 524271.
    Available polling systems :
          epoll : pref=300,  test result OK
           poll : pref=200,  test result OK
         select : pref=150,  test result FAILED
    Total: 3 (2 usable), will use epoll.

    Available filters :
    	[SPOE] spoe
    	[CACHE] cache
    	[FCGI] fcgi-app
    	[COMP] compression
    	[TRACE] trace
    Using epoll() as the polling mechanism.
    00000000:fe_pp.accept(0004)=0012 from [120.136.165.86:10544] ALPN=<none>
    目前我看到 haproxy 可以正常运行，但是 nc 本地监听的 server 没有输出:
    (base) hlc@hlc-VirtualBox:~$ nc -kvl 5432
    Listening on 0.0.0.0 5432
    ```

    而在 nc client 端我发送了一个 hello:

    ```bash
    (base) hlc@hlc-VirtualBox:~$ nc <some_sensitive_ip> 5000
    hello
    ```

    明白了，你现在 HAProxy 已经 **正常运行**（前台 debug 日志显示 `fe_pp.accept` 有来自客户端的连接），但是你的后端 `nc` 监听 5432 没收到数据，这里有几个关键点需要理解：

    ---

    1. 当前配置回顾

        你前端和后端配置大概是这样：

        ```haproxy
        frontend fe_pp
            bind *:6543 accept-proxy
            default_backend backend_apps

        backend backend_apps
            server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
        ```

        * `accept-proxy` 表示 **HAProxy 期望客户端发送 PROXY 协议头**
        * `usesrc clientip` 表示 **HAProxy 连接后端时会用客户端的真实 IP 作为源 IP**
        * 后端是 `127.0.0.1:5432`，你用 `nc -kvl 5432` 监听。

    ---

    2. 为什么 nc 没收到数据

        （a）PROXY 协议头

        * 你的 frontend 使用 `accept-proxy`
        * 普通 `nc` 客户端发送的 `"hello"` **没有 PROXY v1 头**，所以 HAProxy **不会将连接转发到后端**
        * 解决办法：

          * 客户端必须发送 PROXY v1 头，格式类似：

            ```
            PROXY TCP4 192.0.2.10 198.51.100.2 12345 54321\r\n
            hello
            ```
          * 或者去掉 `accept-proxy` 测试普通 TCP 转发：

            ```haproxy
            bind *:6543
            ```

        （b）`usesrc clientip` 限制

        * HAProxy 试图用客户端 IP 作为源 IP 连接 `127.0.0.1`
        * Linux **不允许任意非 127.0.0.1 IP 发包到 127.0.0.1**
        * 因此即便 frontend 收到连接，HAProxy 到 127.0.0.1 的连接也会失败，后端 `nc` 看不到数据。

    ---

    3. 如何本地测试

        方法 1：关闭 `accept-proxy` 和 `usesrc clientip`

        ```haproxy
        frontend fe_pp
            bind *:6543
            default_backend backend_apps

        backend backend_apps
            server app1 127.0.0.1:5432 check
        ```

        * 这样可以直接用 `nc` 客户端测试：

          ```bash
          nc 127.0.0.1 6543
          ```
        * 后端 `nc -kvl 5432` 可以看到数据。

        方法 2：保留 PROXY 协议

        * 需要客户端支持发送 PROXY v1 头，例如：

        ```bash
        printf "PROXY TCP4 127.0.0.1 127.0.0.1 50000 5432\r\nhello\n" | nc 127.0.0.1 6543
        ```

        * 后端会收到 `"hello"`。

        方法 3：生产环境保持 `usesrc clientip`

        * 绑定后端到 **真实网卡 IP** 而不是 127.0.0.1
        * 这样 HAProxy 可以成功用客户端 IP 作为源 IP 发起连接

    ---

    4. 总结

        * 你现在看到的现象完全是 **PROXY 协议 + 使用 clientip 作为源 IP 的副作用**
        * **本地 127.0.0.1 测试无法同时使用 `usesrc clientip`**
        * **解决测试方法**：临时去掉 `accept-proxy` 和 `usesrc`，或者在发送端加上 PROXY 协议头。

    ---

    如果你希望，我可以帮你写一个 **本地测试版本 HAProxy 配置**，可以用 `nc` 正常测试，同时保留 PROXY 协议的选项用于生产。

    你希望我直接写吗？

* 其实是有协议头的，因为 <sensitive_ip>:5000 的 haproxy 的配置是这样的：

    ```
    frontend main
        bind *:5000
        acl url_static       path_beg       -i /static /images /javascript /stylesheets
        acl url_static       path_end       -i .jpg .gif .png .css .js

        default_backend             ssh_backend

    backend ssh_backend
        server ssh_server 127.0.0.1:4321 send-proxy
    ```


    明白了，你补充的配置说明问题的根源更清晰了：

    你前端 `main` 绑定 5000，客户端连接这个前端时，HAProxy **会转发到后端 127.0.0.1:4321 并发送 PROXY 协议头**：

    ```haproxy
    server ssh_server 127.0.0.1:4321 send-proxy
    ```

    而你的后端 `nc -kvl 5432` 并 **不是一个支持 PROXY 协议的服务**。

    ---

    关键点

    1. `send-proxy` 的作用

    * HAProxy 在向后端发送 TCP 流量前，会在数据前面加一行 PROXY 协议头，例如 v1：

    ```
    PROXY TCP4 203.0.113.5 198.51.100.2 50000 4321\r\n
    ```

    * 后端必须**能够识别并解析这个头**，否则会把它当作普通数据。

    2. `nc` 默认 **不解析 PROXY 协议**

    * 所以当你用 `nc -kvl 5432` 监听时，HAProxy 的连接发来的第一行是 PROXY 头
    * `nc` 并没有报错，但 **你发的数据可能被前置的 PROXY 头挡住**，导致你看到的 `"hello"` 没有按预期显示

    3. 端口不一致问题

    * 你前端是 5000 → 后端 4321
    * 你的本地测试是 6543 → 后端 5432
    * 确保端口对应正确，否则数据不会被转发到你监听的 nc。

    ---

    解决办法

    方法 1：去掉 PROXY 协议头（本地测试）

    * 如果只是本地测试 `nc`，可以把 backend 改成：

    ```haproxy
    server app1 127.0.0.1:5432
    ```

    * 不用 `send-proxy`，这样 `nc` 就能直接看到数据。

    方法 2：让后端支持 PROXY 协议

    * 对于像 SSH、PostgreSQL、Nginx 等支持 PROXY 协议的服务，可以使用 `send-proxy`
    * 对于 `nc` 这种简单工具，**需要手动处理 PROXY 头**：

    ```bash
    nc -kvl 5432 | while read line; do echo "recv: $line"; done
    ```

    * 或者使用 `socat` 支持 PROXY 协议解析：

    ```bash
    socat TCP-LISTEN:5432,fork PROXY-LISTEN:5432
    ```

    方法 3：发送测试数据时带 PROXY 头

    * 例如：

    ```bash
    printf "PROXY TCP4 127.0.0.1 127.0.0.1 50000 5432\r\nhello\n" | nc 127.0.0.1 6543
    ```

    * 后端会收到 `"hello"`。

    ---

    💡 **总结**

    * 你的问题不是 HAProxy 不工作，而是 **后端 nc 没有解析 PROXY 协议头**
    * 使用 `send-proxy` → 后端必须理解 PROXY 协议
    * 用 `nc` 测试时，要么去掉 `send-proxy`，要么模拟带 PROXY 头的数据流

    ---

    如果你希望，我可以写一个 **可以本地用 nc 测试，并模拟 PROXY 协议头的完整示例**，让你直接看到数据。

    你希望我写吗？

* 所以内网的 haproxy 即使这样配置，也无法去掉 proxy protocol 协议头吗？

    ```
    frontend fe_pp
            bind *:6543 accept-proxy
            option tcplog
            default_backend backend_apps

    backend backend_apps
        option tcplog
        server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
    ```

    明白你的疑问了。这里核心在于 **`accept-proxy` 和 `usesrc clientip` 的作用**，以及你能否“去掉 PROXY 协议头”。我们逐条分析：

    ---

    1. `accept-proxy` 的作用

        ```haproxy
        frontend fe_pp
            bind *:6543 accept-proxy
        ```

        * 这意味着 **HAProxy 前端期望客户端发送 PROXY v1/v2 头**。
        * 如果客户端没有发送这个头，HAProxy 会 **拒绝连接**（TCP reset 或直接 drop）。
        * `accept-proxy` **只影响 frontend** 接受连接时是否解析 PROXY 协议头。
        * 所以在这个 frontend 配置下，你 **无法去掉 PROXY 协议头**，除非客户端不发送它（否则 HAProxy 会报错）。

    ---

    2. `send-proxy` 或 `usesrc clientip` 的作用

        * `usesrc clientip`（结合 `source 0.0.0.0`）告诉 HAProxy：**向后端发起 TCP 连接时，用客户端真实 IP 作为源 IP**。
        * 这是 **TCP 透明代理**，本质上是另一种“源 IP 伪装”。
        * 为了实现这个功能，HAProxy **必须在 TCP 层知道客户端的 IP** → 通常通过 **PROXY 协议头** 或直连。
        * 如果你保留 `usesrc clientip`，HAProxy **必须解析 PROXY 头**才能知道客户端 IP，否则无法伪装源 IP。

    ---

    3. 为什么“无法去掉 PROXY 协议头”

        结合你的配置：

        ```haproxy
        frontend fe_pp
            bind *:6543 accept-proxy
        backend backend_apps
            server app1 127.0.0.1:5432 check source 0.0.0.0 usesrc clientip
        ```

        * `accept-proxy` = 前端必须解析 PROXY 头
        * `usesrc clientip` = 后端连接需要客户端真实 IP → HAProxy 必须知道客户端 IP
        * **所以 PROXY 协议头在这个场景下是必需的**，不能去掉。

        如果去掉 PROXY 协议头：

        1. 前端 `accept-proxy` 会拒绝连接
        2. 后端 `usesrc clientip` 无法获取客户端 IP → TCP 透明代理无法工作

    ---

    4. 如果你只是想本地测试或不需要真实客户端 IP

        * 可以改配置：

        ```haproxy
        frontend fe_pp
            bind *:6543
            default_backend backend_apps

        backend backend_apps
            server app1 127.0.0.1:5432 check
        ```

        * 这样就**不需要 PROXY 协议头**，普通 `nc` 或 TCP 客户端就能直接连上。
        * 代价：后端只能看到 HAProxy 的源 IP，无法看到真实客户端 IP。

    ---

    ✅ 总结

    | 功能                           | 是否需要 PROXY 协议                          |
    | ---------------------------- | -------------------------------------- |
    | accept-proxy（解析客户端 PROXY 头）  | 必须                                     |
    | usesrc clientip（透明代理，伪装源 IP） | 必须知道客户端 IP → 通常靠 PROXY 头               |
    | 本地测试或不关心客户端 IP               | 可去掉 `accept-proxy` + `usesrc clientip` |

    所以在你的内网配置下，**如果要保持真实客户端 IP（usesrc clientip）并让前端解析客户端 PROXY 头，PROXY 协议头是必须的**。

    ---

    我可以帮你画一个图，直观显示 **客户端 → HAProxy → 后端** 的 TCP/PROXY 协议流向和源 IP变化，让这个逻辑一目了然。

    你希望我画吗？

* 我希望让前端解析客户端 PROXY 头，但是希望 nc 收到的是 tcp 源 ip 已经被更改了的，没有 proxy header 的 payload


    明白了，你的目标是：

    * 前端 HAProxy **接收客户端 PROXY 协议头**（知道客户端真实 IP）
    * 后端 `nc` 或普通 TCP 服务 **收到的连接数据**里 **不包含 PROXY 头**
    * 后端连接使用 **客户端 IP作为源 IP**（透明代理），而不是 HAProxy 的 IP

    ---

    ### 可行性分析

    这就是典型的 **TCP 透明代理（TProxy / NAT + HAProxy）** 的场景：

    1. **accept-proxy**

       * 只是让 HAProxy 解析客户端的 PROXY 协议头
       * HAProxy 内部拿到客户端真实 IP，用于透明代理或者日志
       * **解析后并不会自动把 PROXY 头转发**，如果你用 `send-proxy`，它才会加到后端

    2. **usesrc clientip**

       * HAProxy 建立到后端的 TCP 连接时，源 IP 会伪装成客户端 IP
       * **前提条件**：必须在后端可用的网络接口上，内核允许使用非本机 IP 发包（IP_TRANSPARENT + CAP_NET_ADMIN 或 root）
       * 在 127.0.0.1 上测试 **不可能成功**，因为 Linux 不允许把非 127.0.0.1 的 IP 发到本地 loopback

    3. **send-proxy**

       * 如果你不想让 PROXY 头到后端，就不要在 `server` 配置里加 `send-proxy`
       * HAProxy 内部解析了 PROXY 头，**再用真实客户端 IP 发起 TCP**，数据 payload 是干净的

    ---

    ### 示例配置

    ```haproxy
    frontend fe_pp
        bind *:6543 accept-proxy
        default_backend backend_apps

    backend backend_apps
        server app1 192.168.1.10:5432 check source 0.0.0.0 usesrc clientip
    ```

    * 前端解析 PROXY 协议头
    * 后端连接到真实服务器 192.168.1.10，源 IP 是客户端 IP
    * **不要加 `send-proxy`** → 后端看到的 TCP payload **没有 PROXY 头**

    > ⚠️ 注意：
    >
    > * **127.0.0.1 无法做到真正的源 IP 伪装**
    > * 需要在真实网卡 IP 上，内核允许 IP_TRANSPARENT
    > * HAProxy 必须以 root 启动并有 CAP_NET_ADMIN

    ---

    ### 总结

    | 需求                | 配置要点                                                     |
    | ----------------- | -------------------------------------------------------- |
    | 解析客户端 PROXY 头     | frontend `accept-proxy`                                  |
    | 让后端收到干净 payload   | backend 不加 `send-proxy`                                  |
    | 让后端连接源 IP 是客户端 IP | backend `usesrc clientip` + 真实网卡 + root + IP_TRANSPARENT |

    ---

    如果你愿意，我可以给你画一张 **客户端 → HAProxy → 后端 TCP 流 + 源 IP变化示意图**，直观显示 PROXY 头解析和透明代理效果。

    你希望我画吗？
* 详细讲讲什么是流量反压

    在分布式系统、微服务架构以及流处理（如 Flink, Reactive Streams）中，流量反压（Backpressure，常简称为反压）是一个核心概念。

    简单用一句话概括：**反压是一种「下游处理不过来时，迫使上游降低发送速度」的反馈机制。**

    为了让你彻底理解，我们从生活中的类比开始，一步步深入到技术实现。

    ---

    **一个大白话的类比：奶茶店排队**

    想象一家奶茶店的流水线：

    1. **点单员（上游）**：疯狂接单，一秒钟出一张小票。
    2. **做茶员（下游）**：制作一杯茶需要 1 分钟。

    * **如果没有反压机制**：点单员不管后面做不做得到，只要有顾客来就疯狂打印小票。没多久，做茶员面前就堆了成百上千张小票（数据积压）。最后，小票多到从桌子上掉下来丢了（**数据丢失**），或者做茶员看着数不清的小票直接崩溃罢工了（**内存溢出/宕机**）。
    * **如果引入反压机制**：做茶员发现手里的订单超过 5 张时，就对点单员喊一句：“别打了，先停一下！”点单员于是暂停接单，等做茶员消化得差不多了，再继续接单。这就叫**反压**。

    ---

    **为什么需要反压？（不用的后果）**

    在计算机世界里，上游（生产者）和下游（消费者）的处理速度往往是不匹配的。当突发流量来袭（如秒杀活动、日志暴增），如果下游处理速度慢，而系统又没有反压机制，通常会导致以下两种灾难：

    1. **OOM（Out Of Memory，内存溢出）**：如果下游用一个无界队列（Queue）来接收数据，上游疯狂发送，队列就会无限膨胀，直到撑爆内存，导致服务宕机。
    2. **丢包/丢数据**：如果为了防止内存溢出，把接收队列设为有界队列，那么当队列满了之后，后续进来的数据就会被直接丢弃。

    ---

    **流量反压的常见实现方式**

    不同层级的技术架构，实现反压的方式也不同。主要有以下几种经典模式：

    1. 基于令牌/动态拉取（Pull 模式）

        这是最优雅、最主流的做法（如 Reactive Streams、Java 的 Flow API）。

        * **原理**：下游（消费者）主动向上游（生产者）申请数据。
        * **过程**：消费者对生产者说：“我现在有空，请给我发 5 条数据。”生产者就只发 5 条。发完之后，如果消费者不主动再次请求（`request(n)`），生产者就必须憋着，绝对不主动推送。

    2. 阻塞与有界队列（Push-Block 模式）

        常用于单机多线程或一些传统的中间件。

        * **原理**：在生产者和消费者之间放一个**容量固定**的队列（Bounded Queue）。
        * **过程**：当队列满了之后，生产者的“发送”动作会被直接**阻塞（Block）**（线程挂起）。直到消费者消费了数据，腾出了空间，生产者的线程才被唤醒继续发送。

    3. 滑动窗口与水位线（Watermark / Window）

        常用于网络传输层（如 TCP 的流量控制）。

        * **原理**：接收方会告诉发送方自己还有多少剩余缓存空间（Window Size）。
        * **过程**：发送方根据这个空间大小来决定发多少数据。如果接收方处理慢，它的缓存填满了，返回的窗口大小就会变成 0，发送方就会停止发送，直到窗口再次打开。

    ---

    **现代大数据/分布式系统中的反压机制**

    在复杂的分布式流处理系统中，反压的实现更具挑战性。我们以 **Apache Flink** 为例看看它是怎么做的：

    **Flink 的反压演进（基于 Netty 的有界流控）**

    在 Flink 1.5 之后，它采用了**基于 Credit 的流量控制机制**（类似于 TCP 滑动窗口）：

    * 每一个下游的算子（Task）在接收数据时，都会向对应的主机申请一组独享的 Buffer 块。
    * 下游会定期告诉上游：“我这里还有 3 个空闲的 Buffer（这个数量叫 Credit）”。
    * 上游收到后，**最多只会发送 3 个 Buffer 的数据**。如果没有 Credit 了，上游的 Netty 通道就会停止拉取数据，这种压力会一路向上游传导，最终让数据源（如 Kafka Source）降低读取速度。

    ---

    **总结与思考**

    流量反压不是一种单纯的“技术组件”，而是一种**架构设计思想**。它倡导的是：**在分布式系统中，宁可让上游暂时慢下来，也绝不要把下游压垮。**

    追求系统的吞吐量固然重要，但在面对不确定性的突发流量时，**系统的稳定性（Robustness）永远是第一位的**。反压就是那条保护系统不会在流量洪峰中窒息的“安全带”。

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
        

# Ssh Note

## cache

* disown 在 ssh 中的行为

    不同情况下`pstree -s <sleep_pid>`的输出：

    * `ssh user@host "sleep 10"`

        `systemd───sshd───sshd───sshd───sleep`

    * `ssh user@host "sleep 10 &"`

        `systemd───sleep`
    
    * `ssh user@host "sleep 10 & disown"`

        `systemd───sleep`

    上面这些情况都不会让 ssh 立即返回，本质原因都是 stdout 和 stderr 没有重定向到文件。

    ssh 立即返回的 example:

    * `ssh user@host "(sleep 10 &) > /dev/null 2>&1"`

    * `ssh user@host "(sleep 10 & disown) > /dev/null 2>&1"`

    * `ssh user@host "sleep 10 > /dev/null 2>&1 &"`

    上面三个的进程树都是`systemd───sleep`

    ssh 阻塞等待的 example:

    * `ssh 54 "sleep 20 > /dev/null 2>&1"`

    上面的进程树是`systemd───sshd───sshd───sshd───bash───sleep`

* nohup 在 ssh 中的行为

    方法 1：使用 nohup（推荐）

    ```bash
    ssh user@host "nohup sleep 5 > /dev/null 2>&1 &"
    ```

    方法 2：使用 setsid

    ```bash
    ssh user@host "setsid sleep 5 > /dev/null 2>&1 &"
    ```

    方法 3：重定向输出并完全脱离

    ```bash
    ssh user@host "sleep 5 > /dev/null 2>&1 < /dev/null &"
    ```

    方法 4：使用 screen 或 tmux（适合长时间运行的任务）

    ```bash
    ssh user@host "screen -dm sleep 5"
    # 或者
    ssh user@host "tmux new-session -d sleep 5"
    ```

    方法 5：使用 at 命令

    ```bash
    ssh user@host "echo 'sleep 5' | at now"
    ```

    推荐方案

    **对于简单的后台任务**，使用 `nohup` 是最简单可靠的方法：

    ```bash
    ssh user@host "nohup sleep 5 > /dev/null 2>&1 &"
    ```

    **对于需要更复杂管理的长时间运行任务**，建议使用 `screen` 或 `tmux`，这样可以随时重新连接到进程查看状态。

    这些方法都能确保 SSH 命令立即返回，同时在远程主机上继续执行 sleep 进程。

    **什么 ssh user@host "nohup sleep 5 &" 就可以， ssh user@host "sleep 5 &"  就不行?**

    - `nohup` 做了两件关键事情：
        1. **忽略 SIGHUP 信号**：防止进程在终端关闭时被终止
        2. **重定向输出**：默认重定向到 nohup.out，避免输出到已关闭的终端
    - `nohup` 让 sleep 进程与原始会话完全脱离关系
    - bash 退出后，SSH 不再关心后续进程

    **其他类似的解决方案**

    ```bash
    # 使用 setsid 创建新会话
    ssh user@host "setsid sleep 5"
    ```

    **总结**

    | 命令 | 效果 | 原因 |
    |------|------|------|
    | `sleep 5 &` | 不立即返回 | sleep 仍是SSH子进程，SSH会等待 |
    | `nohup sleep 5 &` | 立即返回 | nohup 使进程脱离会话，不受SIGHUP影响 |

    所以关键区别在于 `nohup` 让进程完全脱离了原来的会话层次结构，使得 SSH 不再需要等待该进程完成。

    `nohup` 的工作原理:

    ```bash
    ssh user@host "nohup sleep 5 &"
    ```

    - `nohup` 让 `sleep` 进程：
        1. 忽略 SIGHUP 信号
        2. 脱离控制终端的影响
        3. 完全独立于原始会话

    `setsid` 的工作原理:

    ```bash
    ssh user@host "setsid sleep 5"
    ```

    - `setsid` 创建一个全新的会话
    - `sleep` 进程成为新会话的领导者
    - 完全脱离原来的 SSH 会话树
* ssh user@host "for i in {0..5}; do echo $i >> ssh_test.log; sleep 1; done"

    我执行这个命令后，远程机器上只出现了 ssh_test.log 文件，但文件是空的。如何解决这个问题

    这个问题非常经典，原因是**双引号导致了变量提前解析**。

    当你使用双引号 `""` 包含命令时，你的**本地终端**在把命令发送给远程机器之前，就已经把 `$i` 给解析了。因为你本地当前并没有定义 `$i` 这个变量，所以 `$i` 被替换成了空字符串。

    实际上，远程机器执行的命令变成了：
    `for i in {0..5}; do echo  >> ssh_test.log; sleep 1; done` （`echo` 后面是个空的，所以只输出了换行符或者空内容）。

    ---

    ### 💡 解决方案

    要解决这个问题，你需要阻止本地终端解析 `$i`，让它原封不动地传给远程机器。有以下几种最简单的方法：

    #### 方法 1：将双引号改为单引号（最推荐）

    单引号会阻止本地终端解析任何变量，把包裹的内容当作纯文本发送。

    ```bash
    ssh user@host 'for i in {0..5}; do echo $i >> ssh_test.log; sleep 1; done'

    ```

    #### 方法 2：使用反斜杠转义 `$` 符号

    如果你因为其他原因必须用双引号（比如你需要在命令里传入本地的变量），你可以用 `\$` 来告诉本地终端：“放开那个美元符号，让远程机器来处理它！”

    ```bash
    ssh user@host "for i in {0..5}; do echo \$i >> ssh_test.log; sleep 1; done"

    ```

    #### 方法 3：通过标准输入（EOF）传递复杂脚本

    如果你以后要执行更复杂的循环或多行脚本，用 `<< 'EOF'` 是最不容易出错的方式：

    ```bash
    ssh user@host << 'EOF'
    for i in {0..5}; do
        echo $i >> ssh_test.log
        sleep 1
    done
    EOF

    ```

    > ⚠️ **注意**：这里的 `'EOF'` 带有单引号，也是为了防止本地解析变量。

    ---

    **总结**：下次在 `ssh` 后面写命令时，如果命令内部包含远程机器才懂的变量（如 `$i`, `$PATH`, `$USER`），记得**首选单引号**！
* 在远端用 shell wrapper／trap 把子进程绑在父进程上（更可控）——**推荐**（对常见非守护化程序最稳）

    * 思路：启动你的程序为子进程，父 shell `wait` 它，同时设置 `EXIT` trap 在父进程退出时 `kill` 子进程或其进程组。示例一行（把 `your_command` 换成真实命令）：

        ```bash
        ssh user@host 'bash -lc "trap \"kill 0\" EXIT; your_command & wait"'
        ```

    * 解释：父 shell 启动 `your_command` 并 `wait`，如果父 shell 因 ssh 断开/被 kill 而退出，`EXIT` trap 会执行 `kill 0`（向同一进程组的所有进程发信号），从而把子进程也清掉。**注意**：如果远端命令自己 double-fork、调用 `setsid` 或显式忽略 SIGTERM/SIGHUP，这个办法就无效。类似做法在社区常见并被讨论过。([Stack Overflow][4])

* 重定向 stdin 的方法（未测试）

    ```bash
    ssh -f -t user@host 'bash -lc "your_command & child=\$!; cat >/dev/null; kill \$child 2>/dev/null; pkill -P \$child 2>/dev/null; wait \$child"'
    ```

    解释：

    * `your_command & child=$!`：把你要跑的命令放后台，记下 pid。
    * `cat >/dev/null`：从远端 stdin（即那个 pty）持续读取，直到 EOF（也就是本地 SSH 连接断开）。
    * `kill $child` + `pkill -P $child`：在检测到 EOF 后尝试杀掉主进程和它的子进程。
    * `wait $child`：等清理完再退出 shell（保持 wrapper 生命周期完整）。

    **优点**：简单、常用、在大多数场景（包括 `-f -t`）下可靠。
    **缺点**：如果远端命令自行做了 `setsid`/double-fork并且完全脱离父子关系，`pkill -P` 可能找不到后代；这时需要更强的杀掉策略（见下面）。

    方法 C（更强）：把任务放到自己的进程组／session，然后按组杀掉

    如果你的命令会 fork 出很多子进程或做 daemonize，想一并杀掉所有进程，可以尝试给它创建一个新的进程组或 session，然后用 `kill -TERM -- -PGID` 去杀整组。示例如下（注意 quoting）：

    ```bash
    ssh -f -t user@host 'bash -lc "setsid bash -c '\''your_command'\'' & child=\$!; cat >/dev/null; kill -TERM -- -\$child 2>/dev/null; wait \$child"'
    ```

    说明：

    * `setsid bash -c 'your_command' & child=$!`：让 `your_command` 在新 session 中启动（通常其 PGID == PID），这样 `kill -- -$child` 会把该进程组全部杀掉。

    * caveat：`setsid` 在某些极端情况下会让命令完全脱离（这正是我们想处理的），但需要注意信号传播语义；不同命令对 `TERM`/`HUP` 的处理会不同。

* SSH multiplexing（ControlMaster），它会让多个 ssh 会话复用同一个 master 连接，杀掉一个客户端不一定会关闭底层连接：

   * 在本地 `~/.ssh/config` 看 `ControlMaster`/`ControlPath`，或试 `ssh -o ControlMaster=no ...` 强制不复用。

* 在远端查 surviving 进程的父子/pgid 信息，看看它是为什么还存活：

   ```bash
   ps -o pid,ppid,pgid,sid,cmd -p <surviving-pid>
   ls -l /proc/<surviving-pid>/fd | sed -n '1,120p'
   ```

* 若你不介意多行，可以用 heredoc，引用更直观：

    ```bash
    ssh user@host 'bash -s' <<'REMOTE'
    trap 'kill 0' EXIT
    your_command &
    wait
    REMOTE
    ```

* 方法 2：不用 `-f`，改用 shell 的后台符号 `&`

    让 ssh 保持前台分配 pty，再由本地 shell 把 ssh 放到后台：

    ```bash
    ssh -tt user@host 'bash -lc "sleep 1000 & child=$!; cat >/dev/null; kill $child; wait $child"' &
    ```

    效果和 `-f -tt` 类似，但逻辑更直观。

* 方法 3：避免伪终端，改用 stdin EOF 探测

    如果你不需要真正的 pty（只是想让 ssh 退出时杀掉远端进程），那就不用 `-t`，而是在远端脚本里用 `cat` 等待 stdin EOF：

    ```bash
    ssh -f user@host 'bash -lc "sleep 1000 & child=$!; cat >/dev/null; kill $child; wait $child"'
    ```

    这里 stdin 是 ssh 的 TCP channel，本地 kill 掉 ssh 后，远端 `cat` 也会得到 EOF，然后执行 kill。
* 实验验证 ssh 远程执行命令是否会随 ssh 本身的关闭而关闭

    1. 最简单情况

        ```bash
        ssh user@host "your_command"
        ```

        example:

        ```bash
        ssh user@host 'for i in {0..100}; do echo $i >> ssh_test.log; sleep 1; done'
        ```

        远程机器会执行命令，本地的 ssh 连接会阻塞，等命令执行完后，本地 ssh 会自动退出。

        如果在本地按 ctrl + c 强行结束 ssh 连接，那么远程命令会继续执行，直到自然结束。

    2. kill 本地 ssh 进程时，远程 bash 脚本接收到的是什么信号？`SIGHUP`?

        先是 SIGINT，然后是 EXIT，没有 SIGHUP

        验证: 
        
        我们使用 trap 捕捉信号，看看远程 bash 到底接收到的是什么。

        ```bash
        ssh -t user@host 'trap "echo exit >> ssh_test.log;" EXIT; trap "echo sighup >> ssh_test.log; exit" SIGHUP; trap "echo sigint >> ssh_test.log; exit" SIGINT; echo -n "" > ssh_test.log; for i in {0..5}; do echo $i >> ssh_test.log; sleep 1; done'
        ```

        `ssh_test.log`:

        ```
        0
        1
        sigint
        exit
        ```

        注：

        1. `-t`是必须的，否则就会像步骤 1 中，远程 bash 不会收到 signal，会在执行完后才自动结束。

    3. 如果远程命令里用了 `nohup`、`setsid`、`disown`，daemonize, 它是否会退出。

        * `nohup`

            即使指定了`nohup bash -c 'command'`，如果有`-t`，那么当 ctrl + c 结束本地 ssh 进程时，远程脚本仍然会收到 sigint。

    4. 使用`bash -c 'command'`又有什么不同？

        与直接执行 command 效果相同，无特殊效果。

    5. `-f -t`, `-f -tt` 有什么区别

        * `-t` 的作用

            * `-t` 强制分配伪终端。

            * 有了伪终端，远程的前台进程依赖这个 tty，当 ssh 连接断开/tty 收回时，内核会给前台进程组发 `SIGHUP`，通常就会退出。

            * 所以 **单用 `-t`**，往往能满足“kill ssh → 远程命令退出”。

        * `-f` 的作用

            * `-f` 意味着 ssh 在执行远程命令前就 fork 自己到后台。

            * 它常用于配合 `-N`（不执行命令，仅做端口转发）。

        * `-f -t`

            ```bash
            ssh -f -t user@host 'trap "echo exit >> ssh_test.log;" EXIT; trap "echo sighup >> ssh_test.log; exit" HUP; trap "echo sigint >> ssh_test.log; exit" SIGINT; echo -n "" > ssh_test.log; for i in {0..100}; do echo $i >> ssh_test.log; sleep 1; done'
            ```

            此时会提示：

            `Pseudo-terminal will not be allocated because stdin is not a terminal.`

            然后继续执行远程命令。
* 为什么 ssh 登陆时只能尝试三次输入密码？

    因为 **OpenSSH 默认把 “最大密码尝试次数” 限制为 3 次**，这是为了降低暴力破解的风险，同时避免用户在反复输错时占用连接资源。

    * 详细原因

        1. **OpenSSH 的默认设置**

            OpenSSH 服务器端（`sshd`）里有两个相关参数：

            | 参数名            | 含义                                  | 默认值   |
            | -------------- | ----------------------------------- | ----- |
            | `MaxAuthTries` | 一个 SSH 连接里允许的**最大认证失败次数**（包括密码、公钥等） | **6** |
            | `MaxSessions`  | 并发会话数量                              | 10    |

            那为什么你看到的是 **只能输错 3 次密码就被踢？**

            因为：

        2. **在 password 身份验证流程里，每次会话最多提供 3 次密码输入**

            OpenSSH 的认证结构是这样的：

            * SSH 建立连接
            * 进入 *password* 认证方式
            * **该认证方式内部限制最多尝试 3 次密码**
            * 如果 3 次都错，整个 "password" 方法失败并计入一次认证失败
            * 如果达到 `MaxAuthTries`（默认 6）会直接断链

            也就是说：

            💡 **你看到的“只能输 3 次”的限制实际上是 password 方法内部的固定行为 — 不是 MaxAuthTries。**

            这是 OpenSSH 的 hard-coded 设计。

    > 你不能单纯通过 sshd_config 修改，因为 3 次逻辑写死在源码里

    OpenSSH 源码里（auth.c / auth2-passwd.c）明确写死 3 次：

    ```c
    #define FAIL_PASSWD_FROM_STDIN 3
    ```

    能改的只有 **MaxAuthTries**

    例如：

    ```
    MaxAuthTries 10
    ```

    这会让你可以尝试密码认证方法更多次数，但**每次只有 3 次密码输入**。
    比如你可以失败两轮（3+3），第三轮开始失败到第 10 个认证失败才断。

* 假如 MaxAuthTries 为 6，我一共可以输错6次密码？

    不行。**不会是 6 次密码输入。**

    `MaxAuthTries` 控制的是：

    > **一次 SSH 连接允许的 “认证失败次数”**（不是密码输入次数）

    * **一次 password-method failure ＝ 1 次 auth failure**
    * MaxAuthTries=6 → 允许 **6 次 auth failure**
    * 但 password-method failure ≠ 重复启动 6 次

* 那如果达到 MaxAuthTries 会发生什么？

    当达到 **MaxAuthTries**（认证失败次数上限）时，OpenSSH 会立即：

    ❌ 立刻断开 SSH 连接

    并且输出类似日志：

    ```
    sshd[xxxx]: maximum authentication attempts exceeded for <user> from <ip> port <port>
    ```

    客户端会看到类似：

    ```
    Received disconnect from <ip> port <port>:2: Too many authentication failures
    Disconnected from <ip> port <port>
    ```
* ssh -R 是否稳定，抗得住巨大流量？比如我想暴露一个内网的 ftp 服务，上传和下载 TB 级别的数据（比如深度学习的数据集）？

    对于 TB 级别的数据传输（尤其是深度学习数据集这种大文件），`ssh -R` **完全扛不住**，且效率极低。

    如果你强行使用 `ssh -R` 跑 FTP 传输大数据，你将面临三大灾难：

    1. 性能灾难：TCP over TCP 问题

        SSH 隧道本身是基于 TCP 的，而 FTP 传输文件也是 TCP。
        当你在一个 TCP 连接（SSH 隧道）里面再跑一个 TCP 连接（FTP 数据流）时，一旦网络出现波动丢包：

        * 内层 TCP 认为丢包了，触发重传。
        * 外层 TCP 也认为丢包了，触发重传。
        * **结果**：重传风暴叠加，导致延迟瞬间飙升，吞吐量断崖式下跌，甚至连接直接断开（Broken pipe）。这在网络工程中被称为 **"TCP Meltdown"**。

    2. CPU 瓶颈：加密开销

        SSH 的设计初衷是安全，它会对**每一个字节**都进行高强度的加密/解密。

        * **SSH 隧道**：Host B 加密 -> Host A 解密 -> Host A 转发 -> 用户。
        * 传输几个 TB 的数据，意味着你的 CPU 要把这几个 TB 的数据完整地跑一遍加密算法。这会吃满单核 CPU，导致带宽跑不满（比如你有 1Gbps 带宽，但 SSH 只能跑到 100Mbps，瓶颈在 CPU）。

    3. FTP 协议本身的噩梦

        FTP 是一个很古老的协议，它分“控制端口”（21）和“数据端口”（随机端口）。

        * 单纯用 `ssh -R` 转发 21 端口是**无法传输文件**的。
        * 你必须配置 Passive Mode（被动模式），并把 FTP Server 的一大段被动端口（例如 50000-51000）全部转发出去。用 SSH 转发一千个端口？这在运维上是不可接受的。

    **最佳解决方案：WireGuard (或 Tailscale)**

    既然你的需求是：**安全（不暴露端口给黑客）** + **高性能（跑满带宽传输 TB 数据）**。

    **WireGuard** 是目前的版本答案。

    **为什么是 WireGuard？**

    1. **UDP 协议**：它基于 UDP，完美避开了 "TCP over TCP" 的性能熔断问题。
    2. **内核级性能**：WireGuard 运行在 Linux 内核空间，加密效率极高，几乎不损耗带宽，轻松跑满千兆甚至万兆网络。
    3. **隐身**：它不通过 TCP 端口握手。如果不持有私钥，黑客扫描你的 UDP 端口是**没有任何回应**的（Drop 包），在黑客看来你的机器就像断网了一样。
    4. **透明组网**：连上 WireGuard 后，你的公网机器 A 和内网机器 B 就像在同一个局域网里。你可以直接用 `ftp 10.100.0.2` 访问，完全不需要管端口映射。

    **架构图**

    ```mermaid
    graph LR
        User[你/公网用户] -- WireGuard VPN --> HostA[公网服务器]
        HostA -- WireGuard VPN (内网互联) --> HostB[内网 GPU 服务器]
        
        subgraph "VPN 虚拟局域网 (10.0.0.x)"
        HostA_IP[Host A: 10.0.0.1]
        HostB_IP[Host B: 10.0.0.2]
        end
    ```

    **实施步骤 (最简版)**

    如果不想手搓 WireGuard 配置，强烈建议直接使用基于 WireGuard 的封装工具 **Tailscale**（极度简单，无需公网 IP 配置，但你既然有公网 IP，也可以用原生的 WireGuard）。

    **这里以原生 WireGuard 为例（性能最强）：**

    **1. 公网机器 (Host A) 安装 WireGuard**

    ```bash
    apt install wireguard
    # 生成私钥和公网钥
    wg genkey | tee privatekey | wg pubkey > publickey

    ```

    配置 `/etc/wireguard/wg0.conf`:

    ```ini
    [Interface]
    Address = 10.0.0.1/24
    ListenPort = 51820
    PrivateKey = <Host A 的私钥>

    [Peer]
    # 内网机器 B
    PublicKey = <Host B 的公网钥>
    AllowedIPs = 10.0.0.2/32

    ```

    **2. 内网机器 (Host B) 安装 WireGuard**
    配置 `/etc/wireguard/wg0.conf`:

    ```ini
    [Interface]
    Address = 10.0.0.2/24
    PrivateKey = <Host B 的私钥>

    [Peer]
    # 公网机器 A
    PublicKey = <Host A 的公网钥>
    Endpoint = <Host A 的公网IP>:51820
    AllowedIPs = 10.0.0.1/32, 0.0.0.0/0  # 如果你想让内网机器也通过公网上网，或者只填 10.0.0.1/32 只互联
    PersistentKeepalive = 25 # 关键：每25秒发个包，保持内网穿透连接
    ```

    **3. 启动**
    两边都执行 `systemctl enable --now wg-quick@wg0`。

    **4. 传输 TB 数据**
    现在，你在公网机器 A 上，或者你的笔记本（也连上 WireGuard）上，直接访问：
    `sftp user@10.0.0.2`
    或者
    `ftp 10.0.0.2`

    * **黑客视角**：扫描公网 IP，只能看到 22 (SSH) 和一个无响应的 UDP 51820。Fail2ban 守好 22 即可。
    * **你的视角**：内网机器就在 `10.0.0.2`，速度飞快，极其稳定。

    **总结**

    * **SSH -R**: 适合传指令、改代码、轻量级 Web 访问。**传 TB 级文件会崩溃。**
    * **FRP**: 适合把服务公开给全网（建站）。
    * **WireGuard**: 适合**私有**的大数据传输和内网互联。这是你当前场景的最优解。
* 如果禁用了 ssh 密码登陆，那么`ssh-copy-id`就会无法使用，此时只能把 public key 写入到`~/.ssh/authorized_keys`中

* ssh 中 forward agent 是干嘛的

    `ForwardAgent`（ssh-agent forwarding）是 SSH 里一个 **非常重要但容易误解** 的功能，它的作用不是让远程机器获得你的私钥，而是：

    **代理转发（Agent Forwarding）= 把本地私钥“借给”远程主机使用，但私钥本身不离开本地**

    也就是说，**你的私钥仍然在 B 上，但 A 上可以替你使用它来登录别的机器**。

    **它是干嘛的？一句话解释**

    **允许你通过 A 登录 C，而不需要把私钥复制到 A。**

    实际流程：

    ```
    B (私钥在这里)
      ↓ SSH -A
    A（没有私钥，但可以通过 ssh-agent 调用 B 上的私钥）
      ↓ SSH
    C
    ```

    A 上并不知道你的私钥内容，但它能借助 ssh-agent 来使用该私钥完成登录。

    **这个功能解决什么问题？**

    比如你在 B（本地电脑）有私钥，但你要通过跳板机 A 登录到服务器 C：

    没有 Agent Forwarding 的情况：

    ```
    B --SSH--> A   （这里成功）
    A --SSH--> C   （失败：A 没有私钥）
    ```

    很多人会错误地把私钥复制到 A，造成安全风险。

    **用 agent forwarding：**

    ```
    ssh -A user@A
    ssh user@C   # 成功，因为A通过agent调用B上的私钥
    ```

    **重点理解：Agent forwarding 不会让你进入 A**

    Agent forwarding **完全不能**让你：

    * 从 B 登录 A 的 user_2
    * 在远程生成 authorized_keys
    * 替代 `ssh-copy-id`

    因为 **agent forwarding 的前提是你已经成功 SSH 到 A**。

    你现在的问题是：

    > A 禁止密码，而 B 不能登录 A 的 user_2 → 首次登录都没法建立
    > `-A` 根本派不上用场

    Agent forwarding 只能在 **你已经能登录的情况下** 扩展你的能力，“往外跳”，而不是解决“登录不进去”的问题。

    **用一个比喻解释**

    * 私钥 = 你的身份证
    * ssh-agent = 你家里一个拿着你身份证的助手
    * Agent forwarding = 你告诉酒店(A)：“如果有人问我的身份证，你打电话给我家里的助手(B)问就行，不要我把身份证交给你。”

    但是：

    如果你连酒店(A)的大门都进不去（即登录失败），那 assistant forwarding 完全用不上。

    **3 个要点总结**

    1. **ForwardAgent 是允许远程主机使用你本地的私钥去登录第三方机器。**
    2. **它不能帮助你第一次登录目标机器。**
    3. **它不会复制你的私钥，只是远程帮你调用。**
* 我通过 journalctl 看到有人在一直扫我的 ssh，有哪些防御方法

    1. 更改默认端口

    2. 禁用密码登录，只允许密钥登录

        彻底杜绝**暴力破解密码**的威胁。即使攻击者知道你的用户名，没有私钥也无法登录。
        
        在 `/etc/ssh/sshd_config` 中设置：

        ```conf
        PasswordAuthentication no
        PubkeyAuthentication yes
        ```

    3. 使用复杂密码（如果必须使用密码登录）

        * **操作：** 确保所有系统用户的密码都非常复杂，包含大小写字母、数字和符号，长度至少 $16$ 位。

    4. 禁用 Root 用户直接登录

        * **效果：** 攻击者无法直接针对最高权限用户进行破解。你应该使用普通用户登录后，再通过 `su` 或 `sudo` 提升权限。

        * **配置路径：** 在 `/etc/ssh/sshd_config` 中设置：
        
        * `PermitRootLogin no`

    5. 使用 Fail2Ban (强烈推荐)

        * **原理：** Fail2Ban 是一款基于日志扫描的入侵检测工具。它会监视 `/var/log/auth.log`（或类似文件），如果发现同一IP在短时间内（例如 $5$ 分钟内）尝试登录失败超过指定次数（例如 $3$ 次），它会自动调用防火墙（如 iptables/firewalld）封禁该IP一段时间（例如 $10$ 分钟）。

        * **安装和配置：** 大多数Linux发行版都可以通过包管理器安装。

        * **效果：** **这是对抗扫描最有效的工具**。它将不断尝试的恶意IP关在门外，显著减少日志中的噪声。

    6. 使用防火墙（iptables / firewalld / ufw）

        * **操作：** 显式地只允许特定或信任的IP地址范围访问你的SSH端口。
        * **效果：** 如果你的服务器只需要从少数几个固定地点访问，这是最彻底的防御。
        * **例如 (ufw)：** `sudo ufw allow from 192.168.1.0/24 to any port 2222`

    7. 限制允许登录的用户

        * **操作：** 明确指定哪些用户可以通过SSH登录，将其他不必要的系统用户禁用SSH登录权限。
        * **配置路径：** 在 `/etc/ssh/sshd_config` 中使用 `AllowUsers` 指令：
        * `AllowUsers username1 username2`

    8. 限制登录尝试次数和频率

        * **操作：** 在 SSH 配置中限制每个连接的密码尝试次数。
        * **配置路径：** 在 `/etc/ssh/sshd_config` 中设置：
        * `MaxAuthTries 3` (最大尝试次数)

* 如果我使用 MaxAuthTries 后，自己失败了3次以上，该怎么解封

    这是一个很好的问题。`MaxAuthTries` 这个设置是在 **SSH守护进程（sshd）** 层面进行限制的，它针对的是**单个连接尝试**。

    **MaxAuthTries 的工作机制**

    `MaxAuthTries` 的作用是：

    1. 用户A尝试连接SSH。
    2. 在**这个连接会话**中，如果密码输入错误达到设定的次数（例如 $3$ 次）。
    3. **sshd会立即断开这个连接。**
    4. 用户A可以**立即尝试发起一个新的连接**。

    **关键点：`MaxAuthTries` 不会封禁你的IP地址。**

    所以，如果你自己输错密码超过 $3$ 次，你不需要“解封”你的IP，你只需要：

    1. **断开当前的SSH客户端。**
    2. **重新发起一个新的SSH连接尝试。**
    3. 在新连接中输入正确的密码或使用正确的密钥。

* Preauth 是什么意思

    "Preauth" 是 **Pre-authentication** 的缩写，在安全和网络协议（尤其是 SSH）的上下文中，它的意思是**预认证**或**认证前阶段**。

    1. SSH中的 Preauth

        当你在 `journalctl` 中查看 SSH 日志时，看到 `Preauth` 通常指的是 **SSH 客户端和服务器开始建立连接，但尚未完成用户身份验证（User Authentication）的阶段**。

        这个阶段主要完成以下任务：

        * **协议版本协商 (Protocol Version Exchange):** 客户端和服务器确定他们将使用的 SSH 协议版本（例如 SSH-2.0）。
        * **密钥交换 (Key Exchange - KEX):** 双方协商并生成用于加密通信的会话密钥。这是确保后续数据（包括用户名和密码/密钥）传输安全的关键步骤。
        * **加密和MAC算法协商:** 确定使用哪种加密算法（如 AES, ChaCha20）和消息认证码（MAC）算法。

    2. 日志中的常见用途

        你在日志中看到包含 `Preauth` 的信息，通常是以下几种情况：

        | 日志内容 | 含义 |
        | --- | --- |
        | `Received disconnect from X.X.X.X: 11: Bye Bye [preauth]` | **攻击者/扫描仪断开连接。** 通常是因为客户端在密钥交换完成或开始用户认证之前就断开了。这在自动化扫描中非常常见。 |
        | `fatal: Read from remote host X.X.X.X: Connection reset by peer [preauth]` | **连接被重置。** 可能由于网络问题，但也可能是攻击者在发现无法继续攻击（例如目标端口已更改或协议不匹配）后立即中断连接。 |
        | `Did not receive identification string from X.X.X.X [preauth]` | **非SSH流量。** 客户端连接到了SSH端口，但没有发送标准的SSH协议识别字符串，这通常表明连接方不是一个正常的SSH客户端（可能是一个端口扫描工具或非SSH服务尝试连接）。 |

    **总结**

    当你在 SSH 日志中看到 `preauth` 时，你可以确信这个事件发生在**用户输入用户名和密码之前**。这意味着服务器已经成功地建立了加密通道，但还没有开始处理用户凭证。

    你的日志中是否有具体的 `preauth` 相关信息？我可以帮你进一步分析。

    * 所以 preauth 有可能是用户名输入错误？

        **不，Preauth 阶段通常发生在用户名输入之前。**

        * **Preauth 阶段** 是建立安全通道，进行密钥和协议协商的阶段。它涉及的是**机器与机器**之间的通信握手。
        * **用户认证（Authentication）阶段** 才是用户输入**用户名和密码/密钥**的阶段。它涉及的是**用户凭证**的验证。

        如果你在日志中看到连接在 `Preauth` 阶段断开或失败，那么**原因不是用户名输入错误**。

        **Preauth 阶段失败的常见原因**

        | 原因 | 描述 |
        | --- | --- |
        | **端口扫描** | 扫描工具连接到端口后，可能发现这不是他们想要的SSH版本，或只是为了记录端口开放状态，立即断开。 |
        | **Fail2Ban 封禁** | 如果攻击者在 `Preauth` 阶段（通常是因为发送了不符合协议的垃圾数据）被 Fail2Ban 规则捕获，连接可能会被防火墙重置。 |
        | **协议不匹配** | 客户端和服务器在协商协议版本、加密算法或密钥交换算法时，没有找到共同支持的选项。 |
        | **非SSH流量** | 某些服务或恶意软件错误地连接到SSH端口，没有发送SSH识别字符串，导致连接终止。 |
        | **网络中断** | 客户端或服务器之间的连接在中途被网络层中断。 |

        **区分日志中的阶段**

        以下是你在日志中如何区分这两个阶段：

        1. **Preauth 日志行：**

            ```log
            sshd[PID]: Did not receive identification string from X.X.X.X [preauth]
            ```

            (看到 `[preauth]` 说明未到认证阶段。)

        2. **认证阶段日志行：**

            ```log
            sshd[PID]: Failed password for invalid user <用户名> from X.X.X.X port YYYY ssh2
            ```

        (看到 `Failed password for...` 或 `Accepted password for...` 说明已经通过了 Preauth，进入了**用户认证阶段**。)

        **总结：** Preauth 失败意味着连接没有到达验证用户身份的步骤。

* Usepam 是什么意思

    `UsePAM` 是 SSH 服务器配置文件 (`/etc/ssh/sshd_config`) 中的一个指令，它的全称是 **Use Pluggable Authentication Modules**。

    这个设置是 **SSH 服务与 Linux/Unix 系统的核心认证机制 PAM (Pluggable Authentication Modules) 进行整合的关键**。

    **核心概念：PAM 是什么？**

    **PAM** 翻译过来是“可插拔认证模块”，它是 Linux/Unix 系统中一个非常灵活且重要的身份验证框架。

    你可以将 PAM 想象成一个**认证中介**：

    * **没有 PAM** (即 `UsePAM no`): 像 SSHD 这样的应用程序必须自己处理所有认证逻辑（例如，检查 `/etc/shadow` 文件）。
    * **使用 PAM** (即 `UsePAM yes`): SSHD 将认证工作外包给 PAM。当用户尝试登录时，SSHD 告诉 PAM：“请帮我验证这个用户”。PAM 接着会根据配置文件（通常在 `/etc/pam.d/` 目录下）中定义的一系列模块来执行验证步骤。

    **`UsePAM yes` 的作用和好处**

    将 `UsePAM` 设置为 `yes` 是现代 Linux 系统上的**标准做法**，它带来了巨大的灵活性和安全性：

    1. **集中管理认证策略：**

        * 所有系统服务（SSH、Sudo、登录控制台、FTP 等）都可以使用同一套认证规则。
        * 例如，如果你配置了一个 PAM 模块来强制要求 $16$ 位复杂密码，这个规则将自动应用于所有使用 PAM 的服务。

    2. **支持高级认证功能：**

        * PAM 允许集成 **二次认证/多因素认证 (MFA)** 模块（如 Google Authenticator）。
        * 它可以支持 LDAP、Kerberos 或生物识别等企业级认证系统。
        * 它可以执行密码过期检查、账户锁定、时间段限制等账户管理功能。

    3. **会话和环境管理：**

        * PAM 不仅处理认证，还会处理会话管理（**session management**）。例如，它负责设置用户登录后的环境变量、限制用户资源的使用（通过 `pam_limits.so` 模块）、或记录用户的最后一次登录时间。

    **总结**

    | 指令 | 作用 | 推荐设置 |
    | --- | --- | --- |
    | `UsePAM yes` | **启用** PAM 接口。SSHD 将认证、账户和会话管理交给系统的 PAM 框架来处理。 | **Yes (是)** |
    | `UsePAM no` | **禁用** PAM 接口。SSHD 必须使用自己的内部机制来处理认证，可能会导致一些系统级的安全策略失效。 | No (否) |

    **对于你的 SSH 安全防御来说：** 即使你使用更安全的密钥登录，也**应该保持 `UsePAM yes**`，因为它确保了系统级的账户和会话管理（例如，检查账户是否被锁定，或在登录时设置正确的环境）。

* 我想取消root的密码，不允许任何用户在bash直接登陆root，只能通过sudo su root切换到root，怎么办

    这是一个非常好的安全实践，因为它确保了所有的权限提升操作都有**审计记录**（通过 `sudo`），并且杜绝了直接通过 SSH 或控制台使用 `root` 密码登录的风险。

    要实现您的目标，您需要执行以下两个步骤：

    步骤一：禁用 Root 密码（阻止直接登录）

    要取消 `root` 用户的密码，使其无法通过密码进行登录或切换，可以使用 `passwd` 命令的 `-l` 或 `-d` 选项。

    **推荐方法：使用 `-l` (Lock) 锁定密码**

    锁定密码会保留原有的加密哈希，但在密码哈希前加上一个特殊字符（通常是 `!`），使其无法匹配任何输入，从而阻止密码登录。这是最安全且可逆的方法。

    ```bash
    sudo passwd -l root
    ```

    **执行后效果：**

    1. 当有人尝试使用 `root` 密码登录（包括 SSH 或在终端中使用 `su` 命令切换）时，会显示“Authentication failure”或类似错误。
    2. `root` 用户的账户信息仍然存在，`sudo` 机制可以正常工作。

    步骤二：确保 SSH 不允许 Root 直接登录

    虽然您已经取消了 `root` 密码，但为了彻底安全和规范，请确保 SSH 服务也配置为禁止 `root` 直接登录。

    1. **编辑 SSH 配置文件：**

        ```bash
        sudo nano /etc/ssh/sshd_config
        ```

    2. **查找并修改或添加以下指令：**

        ```conf
        PermitRootLogin no
        ```

    （如果您的服务器设置了密钥登录，并且想要允许 `root` **只通过密钥登录**，则可以设置为 `PermitRootLogin prohibit-password`。但您的目标是完全不允许直接登录，所以设置为 `no` 即可。）

    3. **重启 SSH 服务：**

        ```bash
        sudo systemctl restart sshd
        ```

    **最终效果总结**

    | 操作 | 结果 |
    | --- | --- |
    | **`ssh root@your-server`** | 失败。由于 `PermitRootLogin no`，连接将被拒绝。 |
    | **`su root` (输入密码)** | 失败。由于 `passwd -l root`，密码被锁定，认证失败。 |
    | **`sudo su -` (或 `sudo -i`)** | **成功。** 因为 `sudo` 机制验证的是**当前用户**的权限，而不是 `root` 用户的密码。这是唯一允许的权限提升路径。 |

    **重要提醒：**

    在执行 `passwd -l root` 之前，请务必确认您当前的用户已经在 `sudo` 组中，并且`sudo` 命令可以正常工作。否则，一旦锁定了 `root` 密码，您将失去最高的管理权限。

    注：

    1. 没试过这种方法，有空了试试
* > Host key for 127.0.0.1 has changed and you have requested strict checking.

    这个报错是 SSH 的一种安全保护机制。当你访问过一个服务器后，SSH 会记录它的“指纹”并存放在 `known_hosts` 文件中。如果服务器重装了系统、更新了 SSH 服务或者你重新生成了密钥，指纹就会对不上，SSH 为了防止**中间人攻击（MITM）**，会直接拒绝连接。

    由于你是连接 `127.0.0.1`（本机），通常是因为你重启后某些环境变化导致指纹重置。

    **快速解决方法**

    你可以通过以下几种方式清除旧的指纹：

    * 方法一：使用命令行工具（推荐）
    在你的终端（Cygwin、CMD 或 PowerShell）中运行以下命令：

    ```bash
    ssh-keygen -R 127.0.0.1
    ```
    这条命令会自动从你的 `known_hosts` 文件中删除关于 `127.0.0.1` 的旧记录。

    ---

    * 方法二：手动编辑文件
    根据你的报错信息，冲突的记录在第 6 行：
    > `Offending ED25519 key in /home/wsdlh/.ssh/known_hosts:6`

    1.  打开文件：`/home/wsdlh/.ssh/known_hosts`（在 Windows 资源管理器中通常对应 `C:\Users\你的用户名\.ssh\known_hosts`，或者在 Cygwin 下直接用 `vim`）。
    2.  找到第 **6** 行。
    3.  删掉整行并保存。

    ---

    * 方法三：忽略检查（仅建议本地开发使用）
    如果你频繁遇到这个问题且确定环境安全，可以在连接时跳过检查：

    ```bash
    ssh -o "StrictHostKeyChecking no" 127.0.0.1
    ```

    ---

    **为什么重启后总是报错？**

    如果**每次**重启都报这个错，说明你的 SSH 服务（SSHD）在重启后并没有保留之前的 Host Key。这通常有几个原因：

    1.  **临时环境**：如果你是在某种容器、沙盒或者未持久化配置的环境下运行 SSH 服务，重启后它会生成一套全新的密钥。
    2.  **权限问题**：SSH 服务可能无法读取已有的密钥文件（通常在 `/etc/ssh/ssh_host_*`），导致它每次启动都重新生成。
    3.  **Cygwin 路径映射**：你的 Cygwin 环境可能在重启后挂载点发生了变化，导致 SSH 找错了 `known_hosts` 文件的位置。

    **建议**：
    如果问题持续出现，请检查你的 SSH 服务端配置，确保它的主机密钥（Host Keys）存放在一个持久化的目录中，而不是 `/tmp` 或内存盘里。

* 如何查看 ssh id_rsa 是多少位的？

    查看 SSH RSA 密钥位数的几种方法：

    1. 使用 ssh-keygen 命令（推荐）

        ```bash
        # 检查默认位置的密钥
        ssh-keygen -l -f ~/.ssh/id_rsa

        # 或指定完整路径
        ssh-keygen -l -f /path/to/id_rsa
        ```

        输出示例：

        ```text
        3072 SHA256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx user@host (RSA)
        ```

    这里的 3072 就是密钥位数（3072位）

    2. 使用 openssl 命令

        ```bash
        openssl rsa -in ~/.ssh/id_rsa -text -noout | head -20
        ```

        在输出中查找：

        ```text
        RSA Private-Key: (3072 bit)
        ```

    3. 查看公钥文件

        ```bash
        ssh-keygen -l -f ~/.ssh/id_rsa.pub
        ```

    4. 使用 file 命令（不太准确）

        ```bash
        file ~/.ssh/id_rsa
        ```

    注意事项：

    * 需要私钥文件：方法1和方法2需要私钥文件（id_rsa），而方法3只需要公钥文件（id_rsa.pub）

    * 权限问题：私钥文件权限通常为 600 (-rw-------)，确保有读取权限

    * 现代建议：现在推荐使用至少 3072 位的 RSA 密钥，或者使用更现代的 Ed25519 算法

    查看其他类型密钥：

    ```bash
    # 查看 Ed25519 密钥
    ssh-keygen -l -f ~/.ssh/id_ed25519

    # 查看 ECDSA 密钥
    ssh-keygen -l -f ~/.ssh/id_ecdsa
    ```

    最简便的方法是使用 ssh-keygen -l -f ~/.ssh/id_rsa 命令，它会直接显示密钥位数和指纹信息。

* SSH -D 的作用

    ssh -D 用于创建 SOCKS 代理隧道，可以将本地端口与远程 SSH 服务器建立动态端口转发。通过这个代理，所有发送到本地指定端口的数据流量都会通过 SSH 加密隧道转发到远程服务器，再由远程服务器访问目标网络资源。

    主要作用：

    * 加密网络流量：保护本地应用程序的网络通信

    * 突破网络限制：绕过防火墙或访问受限内容

    * 身份伪装：使用远程服务器的 IP 地址访问网络

    * 安全浏览：在不安全的公共 Wi-Fi 中保护数据

    基本用法

    ```bash
    ssh -D [本地端口] [用户@]远程主机
    ```

    常见示例

    * 创建基本的 SOCKS5 代理（端口 1080）：

        ```bash
        ssh -D 1080 user@example.com
        ```

    * 指定本地接口（仅本机访问）：

        ```bash
        ssh -D localhost:1080 user@example.com
        ```

    * 允许局域网其他设备使用代理：

        ```bash
        ssh -D 0.0.0.0:1080 user@example.com
        ```

    * 后台运行：

        ```bash
        ssh -D 1080 -f -C -q -N user@example.com
        ```

    选项说明：

    * -f：后台运行

    * -C：启用压缩

    * -q：安静模式

    * -N：不执行远程命令

    客户端配置

    浏览器配置（以 Firefox 为例）

    * 设置 → 网络设置 → 手动代理配置

    * SOCKS 主机：127.0.0.1

    * 端口：1080

    * SOCKS v5，勾选“远程 DNS”

    命令行使用

    ```bash
    # 通过代理使用 curl
    curl --socks5 127.0.0.1:1080 http://example.com

    # 设置全局代理环境变量
    export ALL_PROXY=socks5://127.0.0.1:1080
    ```

    注意事项

    * 安全性：

        * 流量在本地到 SSH 服务器之间是加密的

        * 从 SSH 服务器到目标网站是明文的（除非目标支持 HTTPS）

        * 建议使用 -C 选项压缩数据提高速度

    * 性能：

        * 所有流量都经过远程服务器，速度受限于 SSH 服务器的带宽

        * 可能增加延迟

    * 连接管理：

        * 保持 SSH 连接稳定很重要

        * 可以使用 autossh 自动重连：

        ```bash
        autossh -M 0 -D 1080 -f -N user@example.com
        ```

    高级用法

    多级代理跳转：

    ```bash
    # 通过跳板机连接到最终代理服务器
    ssh -D 1080 -J jumpuser@jumpserver.com finaluser@finalserver.com
    ```

    指定加密算法（提高性能）：

    ```bash
    ssh -D 1080 -c aes128-gcm@openssh.com user@example.com
    ```

    这种代理方式特别适合需要临时安全访问、调试网络问题或绕过地域限制的场景。

* windows 中使用 nmap 的 ncat.exe 配置 ssh proxy

    ```ssh
    ProxyCommand ncat --proxy-type http --proxy 192.168.56.1:10809 %h %p
    ```

* sshd config 中

    * `AllowTcpForwarding`

        控制是否允许 TCP 端口转发

        默认值：通常是 yes
    
        作用：允许客户端使用 -L（本地转发）、-R（远程转发）、-D（动态转发）
    
        可选值：
    
        * `yes`：允许所有端口转发
    
        * `no`：禁止所有端口转发
    
        * `local`：只允许 -L（本地转发）
    
        * `remote`：只允许 -R（远程转发）

    * `GatewayPorts`

        控制远程转发的绑定地址

        默认值：通常是 no
        
        作用：控制 -R 远程转发的监听地址
        
        可选值：
        
        * `no` (默认值)

            * 远程转发 (-R) 只绑定到 localhost (127.0.0.1)

            * 只有 SSH 服务器本机可以访问转发的端口
        
        * `yes`

            * 远程转发 (-R) 绑定到所有接口 (0.0.0.0)

            * 网络中的任何机器都可以访问转发的端口
        
        * `clientspecified` (较新版本支持)

            * 允许客户端指定绑定地址

            * 使用格式： `ssh -R [bind_address:]port:host:hostport`
        
        example:
        
        ```bash
        # GatewayPorts no (默认)
        ssh -R 8080:localhost:80 user@server
        # 只在 server 的 127.0.0.1:8080 监听，只有 server 本地能访问
        
        # GatewayPorts yes
        ssh -R 8080:localhost:80 user@server  
        # 在 server 的 0.0.0.0:8080 监听，网络中的所有机器都能访问
        ```
        
        配置后需要重启：
        
        ```bash
        sudo systemctl restart sshd
        # 或
        sudo service ssh restart
        ```

* ssh 的`-t`或`-tt`不能与`-N`合用，因为它们的意义冲突，`-N`表示不分配终端，而`-tt`表示强制分配终端。如果确实合用了，那么会以`-N`为准。

    如果需要在远程机器上执行命令，那么就不要用`-N`，否则可能分配不出来终端。

* `ssh -R`中的 R 表示的是 remote，即远程端口转发。而`-L`中的 L 表示的是 local，即本地端口转发。

* `ssh -fNL`几乎等同于`ssh -NL &`，关闭当前 bash 对它们的存活都没有影响

    `ssh -f`会在完成认证后才进入后台，而`ssh -N &`会立即进入后台，可能会跳过一些认证环节。

* `ssh -N -L local_port:remote_host:remote_port user@server &`命令在当前 bash 退出后，job 不会被清空，即后台程序还在，因为 ssh 只处理 SIGINT （kill 命令） 和 SIGTERM 这两个 signal，不处理 SIGHUP

    ai 说这是为了在网络连接不稳定时保持连接。（网络质量不佳时，ssh 会收到 sighup signal 吗？）

    我们可以手动存储 ssh 的 pid，使用 trap 命令设置在 bash 退出时，手动 kill 掉 ssh 进程：

    ```bash
    ssh -NL 1234:127.0.0.1:1234 user@server &
    SSH_PID=$!
    trap "kill ${SSH_PID}" EXIT

    # run other commands that need ssh tunnel
    ```

    此时即可实现在当前 bash 退出时，自动关闭 ssh tunnel。

    AI 给了更详细的另一种写法：

    ```bash
    #!/bin/bash
    
    # 启动 SSH 隧道
    ssh -NL 33389:127.0.0.1:3389 user@ssh_server &
    SSH_PID=$!
    
    # 等待信号
    cleanup() {
        kill $SSH_PID 2>/dev/null
        exit 0
    }
    
    trap cleanup EXIT INT TERM
    
    # 等待后台进程（可选）
    wait $SSH_PID
    ```

    这种写法考虑到了 exit, int, term 三种信号。那么对比前面的写法，如果在执行 other commands 时，在外部对 bash 进行 kill 或发送 int, term 信号，"kill ${SSH_PID}"这个命令还会被执行吗？

* sshd 对`authorized_keys`文件的权限要求很严，必须是`600`（即`-rw-------`），并且 owner 和 group 都是当前用户，ssh server 才能正常 work。否则用户即使添加了 public key，也无法正常登陆。

    `.ssh`目录的权限必须是`700` (即`rwx------`)

* ssh 使用 nc 进行代理

    `ssh -o ProxyCommand="ssh jumpuser@bastion.example.com nc target.internal.com 22" targetuser@target.internal.com`

    命令分解：

    1. ssh targetuser@target.internal.com: 这是你想要执行的最终连接命令。

    2. -o ProxyCommand="...": 这个选项告诉 SSH 客户端，不要直接连接目标主机，而是执行 ProxyCommand 中的命令来建立连接。

    3. ssh jumpuser@bastion.example.com nc target.internal.com 22:

        * 先使用 SSH 登录到跳板机 bastion.example.com。

        * 登录成功后，在跳板机上执行 nc target.internal.com 22 命令。

        * nc (netcat) 会与目标机 target.internal.com 的 22 端口 (SSH 端口) 建立一个原始的 TCP 连接。

    4. 此时，你的本地 SSH 客户端与你本地 ssh 进程通信，你本地的 ssh 进程再与跳板机上的 nc 进程通信，跳板机上的 nc 进程再与目标机的 SSH 端口通信。这样，一条链就打通了，所有的 SSH 加密数据都通过这条“管道”进行传输。

* ssh 使用 `-W` 进行代理

    `ssh -o ProxyCommand="ssh -W %h:%p <跳板机用户名>@<跳板机IP>" <目标机用户名>@<目标机IP>`

    命令分解：

        -W %h:%p: 这是一个 SSH 的内置功能。

            %h 会被自动替换为最终目标主机名 (target.internal.com)。

            %p 会被自动替换为最终目标端口 (22)。

        这个选项命令 SSH 客户端连接到跳板机 (jumpuser@bastion.example.com)，然后请求它打开一个到 %h:%p 的 TCP 连接。之后，所有本地的 SSH 流量都会通过这个连接进行转发。

    现代更推荐使用 ssh -W 的方式进行代理，因为它更安全、更简洁，而且不需要安装 nc。

    配置 ssh config 进行代理：

    ```conf
    Host bastion  # 为跳板机起一个别名
        HostName bastion.example.com
        User jumpuser
        IdentityFile ~/.ssh/id_rsa_bastion_key  # 可选项，指定连接跳板机的私钥

    Host target  # 为目标机起一个别名
        HostName target.internal.com
        User targetuser
        IdentityFile ~/.ssh/id_rsa_target_key   # 可选项，指定连接目标机的私钥
        ProxyJump bastion  # 关键配置！表示通过 bastion 主机跳转
        # 或者使用旧的 ProxyCommand 语法也可以：
        # ProxyCommand ssh bastion -W %h:%p
    ```

    保存后，可以直接使用别名连接目标服务器：

    `ssh target`

    SSH 客户端会自动处理所有跳转逻辑。

* scp 使用跳板机

    如果在 ~/.ssh/config 中定义了跳板机，可以这样使用:

    ```bash
    scp -J myjump localfile.txt targetserver:/remote/path/
    ```

    其中 myjump 是在 SSH 配置中定义的跳板机别名。

    如果跳板机使用非标准端口，可以使用`-J user@host:port`格式.

    对于更复杂的跳转，可以使用逗号分隔多个跳板机: `-J user1@host1,user2@host2`.

    scp 也可以使用 proxy command:

    ```bash
    scp -o ProxyCommand="ssh -W %h:%p jumpuser@jumpserver.example.com" localfile.txt targetuser@targetserver.example.com:/remote/path/
    ```

    如果 target host 使用了非标准端口，即可以直接指定：

    ```bash
    scp -J jumpuser@jumpserver.example.com localfile.txt targetuser@targetserver.example.com:2345:/remote/path/
    ```

    也可以使用`-P`命令指定：

    ```bash
    scp -P 2345 -J jumpuser@jumpserver.example.com localfile.txt targetuser@targetserver.example.com:/remote/path/
    ```

    如果跳板机使用了非标准端口，那么只能直接指定：

    ```bash
    scp -J jumpuser@jumpserver.example.com:2222 localfile.txt targetuser@targetserver.example.com:/remote/path/
    ```

    如果使用 ssh config 文件，那么配置起来比较清晰：

    ```conf
    # ~/.ssh/config
    Host jumpserver
        HostName jumpserver.example.com
        User jumpuser
        Port 2222

    Host targetserver
        HostName targetserver.example.com
        User targetuser
        Port 2345
        ProxyJump jumpserver
    ```

    在连接的时候只需要执行：

    ```bash
    scp localfile.txt targetserver:/remote/path/
    ```

    如果使用 proxy command，那么可以使用：

    ```bash
    scp -P 2345 -o ProxyCommand="ssh -p 2222 -W %h:%p jumpuser@jumpserver.example.com" localfile.txt targetuser@targetserver.example.com:/remote/path/
    ```

* 如果 ssh 只指定`-N`，不指定`-f`，那么不会进入 remote host 的登陆界面

    输出如下：

    ```
    (base) hlc@hlc-VirtualBox:~$ ssh -N <user>@<host>


    ```

    正常情况下会有登陆提示：

    ```
    (base) hlc@hlc-VirtualBox:~$ ssh <user>@<host>
    Last login: Fri Sep 19 13:23:07 2025 from 10.129.8.115
    hlc@lab-sw1:~$ 

    ```

* ssh 远程执行 command，kill 本地 ssh 后，远程 command 被同步关闭

    可以使用`ssh -f -tt user@host "<command> > /dev/null 2>&1"`

    解释：

    * `-f`: fork 一份 ssh 在后台运行

    * `-tt`: 强制分配一个 psudo terminal，使得 local ssh 被 kill 时，远程的 psudo terminal 也被 kill，顺便清空 psudo terminal 里的所有 jobs

        `-t`只是建议分配 psudo terminal，和`-f`合用时会被抑制。`-tt`则是强制分配，不会被抑制。

    * `> /dev/null 2>&1`: 如果不加这个，远程的 stdout 会输出到当前 local terminal

* ssh 远程后台执行命令后直接返回

    核心要求有两点：

    1. 需要在命令后添加`&`

    2. stdout 和 stderr 必须重定向到文件

    example:

    * `ssh user@host "sleep 5 &"`

        不行，因为 stdout 和 stderr 仍在占用当前 shell session.

    * `ssh user@host "sleep 5 & disown"`

        不行，因为 ssh 执行的是非交互 shell，`disown`无法将 job 脱离当前 shell。

    * `ssh user@host "sleep 5 > /dev/null 2>&1 &"`

        OK. stdout 和 stderr 被重定向到`/dev/null`文件，并且命令的最后有`&`表示在后台执行。此时 ssh 会立即返回。

    * `ssh user@host "sleep 5 2>&1 > /dev/null &"`

        不行，stderr 指向 shell, stdout 指向`/dev/null`。

        从左到右解析命令，`2>&1`表示将 stderr 指向当前的 stdout (shell)，`> /dev/null`表示将 stdout 指向`/dev/null`，但是不改变 stderr。此时 stderr 仍指向 shell，而 stdout 指向`/dev/null`。

    * `ssh user@host "sleep 5 && echo hello > /dev/null 2>&1 &"`

        不行，首先`&`作用于`sleep 5 && echo hello`，即
        
        `(sleep 5 && echo hello > /dev/null 2>&1) &`

        其次`> /dev/null`和`2>&1`只作用于`echo hello`。

        此时`sleep 5`的 stdout / stderr 都定向到 shell。
        
        另外，`sleep 5 && echo hello > /dev/null 2>&1`可能会起一个子 shell，这个子 shell 的 stdout / stderr 都未重定向。
        
        因此 ssh 无法立即返回。

    * `ssh user@host"sleep 5 > /dev/null 2>&1 && echo hello > /dev/null 2>&1 &"`

        不行，很明显了，即使对两个子 command 都重定向，也不行。说明两个子 command 被合成为一个整体的 command。

    * `ssh host@user "(sleep 5 && echo hello) > /dev/null 2>&1 &"`

        OK。再次验证了上面的想法。

    * `ssh user@host "nohup (sleep 5 && echo hello) > /dev/null 2>&1 &"`

        不行。

        前面的 case 都是 ssh 可以立即返回，但是 ssh 在返回时会发送 sighup 信号，导致实际上任务会中断，nohup 可以解决这个问题，但是 nohup 会带来新的问题。

        语法错误，报错：

        ```
        bash: -c: line 0: syntax error near unexpected token `sleep'
        bash: -c: line 0: `nohup (sleep 5 && echo hello) > /dev/null 2>&1 &'
        ```

        `(sleep 5 && echo hello)`不是一个命令，是一个子进程命令组合，`nohup`只接收单个命令，不接收子进程命令。

    * `ssh user@host "nohup sleep 5 && echo hello > /dev/null 2>&1 &"`

        不行，`nohup`只作用于`sleep 5`。

    * `ssh user@host "nohup bash -c 'sleep 5 && echo hello' > /dev/null 2>&1 &"`

        OK。`nohup`只作用于`bash -c 'sleep 5 && echo hello'`，`> /dev/null`，`2>&1`以及`&`也作用于`bash -c`。这样既可以让 ssh 立即返回，也不会因为 ssh 的返回而中断任务。

* github 使用 ssh 协议时设置代理

    `~/.ssh/config`:

    ```conf
    Host github_ssh_proto
        Hostname github.com
        ForwardAgent yes
        ProxyCommand nc -X connect -x 127.0.0.1:10809 %h %p
    ```

    当`git remote -v`中使用的是 ssh 协议 (`git@xxxx:path/to/dir`) 时，设置`git config --global http.proxy http://xxxx:yyy`和`git config --global https.proxy http://xxxx:yyy`是无效的，这两个只对 http 协议生效。

    要想让 ssh 协议走代理，必须像上面一样配置`~/.ssh/config`。

    `nc`是使用`nc`程序开一个转发，`-X connect`表示代理协议是 https （为什么没有 http 协议，不清楚），`-x 127.0.0.1:10809`表示代理的 ip 和端口，`%h`表示 target host 的占位符，其实就是`github.com`，`%p`是 target host port 的占位符，上面没写`Port`字段，那就是默认的`22`。

* `ssh-copy-id -i`

    ssh-copy-id 命令会默认将本地用户的所有公钥文件（通常是 ~/.ssh/id_rsa.pub, ~/.ssh/id_ed25519.pub 等）都复制到远程服务器.

    `-i` (`--identity`) 表示只拷贝指定的 pub key。

    example:

    `ssh-copy-id -i ~/.ssh/my_key.pub user@remote_server`

* ssh config 中 HostName 和 Host

    Host 用于指定一个别名或模式，HostName 是实际的服务器地址。

    example:

    `~/.ssh/config`:

    ```conf
    Host myserver
        HostName example.com
        User alice
    ```

* `ssh -o PreferredAuthentications=password`

    强制使用 password 进行 ssh 登录。

    需确保目标服务器的 /etc/ssh/sshd_config 中启用了 PasswordAuthentication yes。

* `ssh -T`

    建立 ssh 连接后，不分配终端。

    此时 remote host 的 sshd 会启动一个 bash 程序，将 ssh client 的输入直接输入到 bash 进程，然后将 bash 的输出拿出来发送给 ssh client。

    在 remote host 上执行`ps -aux | grep ssh`，可以看到有 pty 为 ? 的一条输出：

    `hlc      4171601  0.0  0.0  12976  4212 ?        Ss   16:21   0:00 -bash`

    由于不分配终端，所以交互式程序无法正常运行，比如 vim，sudo, 上下左右键。

    下面是运行 sudo 的报错：

    ```
    sudo echo hello
    sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper
    sudo: a password is required
    ```

    如果`ssh`命令里既有`-t`，也有`-T`那么以后面一个为准：

    * `ssh -T -t`: 分配终端

    * `ssh -t -T`: 不分配终端

    ai 给出的 bash 启动时的具体行为（未验证）：

    有 PTY 时：bash 会以交互模式启动（加载 ~/.bashrc 等），支持作业控制、行编辑等。

    无 PTY 时（-T）：bash 以非交互模式启动（类似 bash -c "command"），仅读取 ~/.bash_profile 或 ~/.bash_login（依赖配置），且直接执行命令后退出。

    目前不知道这个有啥用，感觉没啥用。

* `ssh -f`

    如果`ssh -f`后边没有跟 command 命令，那么 ssh 会报错：

    > Cannot fork into background without a command to execute.

    但是有时候我们想让 ssh 放到后台，又希望它执行`-L`或`-R`之类的端口转发，又没有什么特别想执行的 command，此时可以配合`-N`选项实现这个目的.

    `ssh -fN -L 1234:127.0.0.1:1234 hlc@xx.xxx.xx.xx`

    目前不清楚这样做是否有 keep alive 的功能。

    如果只有`-f -L`，没有`-N`，那么同样会报错：

    > Cannot fork into background without a command to execute.

* `ssh -f`将 ssh 放到后台，并执行远程命令，输出到 stdout。

    `-f`表示 fork。

    example:

    ` ssh -f <user>@<addr> ls`

    output:

    ```
    (base) hlc@hlc-VirtualBox:~$ ssh -f hlc@xx.xxx.xx.xx ls
    (base) hlc@hlc-VirtualBox:~$ 04_local_res.cpp
    Data
    Desktop
    Documents
    Downloads
    link_mnt_hlc_to_Data.sh
    mlnx_perftest_log.txt
    Music
    nfs_shared
    nvshmem_srcs.tar.gz
    Pictures
    Public
    ...
    
    ```

    可以看到，ssh 先被挂到后台后，才输出的远程机器的`ls`内容。

    如果不加`-f`，则会先输出远程`ls`的内容后，才退出 ssh，如下：

    ```
    (base) hlc@hlc-VirtualBox:~$ ssh hlc@xx.xxx.xx.xxx ls
    04_local_res.cpp
    Data
    Desktop
    Documents
    Downloads
    link_mnt_hlc_to_Data.sh
    mlnx_perftest_log.txt
    Music
    nfs_shared
    ...
    (base) hlc@hlc-VirtualBox:~$ 
    ```

* ssh 使用跳板机连接到远程主机

    使用跳板机连接到主机，等价于`-J`。其 ssh config 文件写法为

    ```
    Host <目标主机别名>
        HostName <目标主机IP或域名>
        User <用户名>
        ProxyJump <跳板机用户名>@<跳板机IP或域名>
    ```

    也可以分开写：

    ```
    Host jump
        HostName jump.example.com
        User user_jump

    Host target
        HostName target.example.com
        User user_target
        ProxyJump jump  # 使用跳板机 "jump"
    ```

    多级跳板：

    ```bash
    ssh -J jump1,jump2,jump3 target
    ```

    ssh config:

    ```
    Host target
        ProxyJump jump1,jump2,jump3
    ```

    如果跳板机的端口不是默认的 22 端口，那么可以这样配置：

    ```bash
    ssh -J user@jump-host:2222 user@target-host
    ```

    ```
    # 配置跳板机
    Host jump-host
        HostName jump-host.example.com
        User user_jump
        Port 2222  # 指定非默认端口

    # 配置目标主机（通过跳板机连接）
    Host target-host
        HostName target-host.example.com
        User user_target
        ProxyJump jump-host  # 自动使用 jump-host 的端口 2222
    ```

* 跳板机 英语 Bastion Host

* ssh keep alive

    有几种可选方案

    1. 在 ssh client 的`~/.ssh/config`中配置

        ```
        Host *
            ServerAliveInterval 60      # 每60秒发送一次心跳包
            ServerAliveCountMax 3       # 连续3次无响应才断开
        ```

        其中`Host *`表示对所有主机生效，可替换为特定主机名（如`Host example.com`）

    1. ssh client 使用命令行参数

        ```bash
        ssh -o ServerAliveInterval=60 user@example.com
        ```

    1. 在 ssh server 的`/etc/ssh/sshd_config`中配置

        ```
        ClientAliveInterval 60         # 每60秒检查一次客户端活动
        ClientAliveCountMax 3          # 连续3次无响应后断开
        ```

        需要重启 ssh server。

* ssh 与伪终端

    `ssh user@addr`登陆后，会自动分配一个伪终端，进入交互式环境。`ssh user@addr command`则默认不会分配伪终端，因此不会进入交互式环境。但是有些命令需要伪终端的支持才能运行，比如`sudo`，此时我们可以使用`-t`强制分配伪终端：`ssh -t user@addr command`。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ssh hlc@10.133.1.54 "sudo ls"
    hlc@10.133.1.54's password: 
    sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper
    sudo: a password is required
    ```

    使用`-t`，则需要再输入一遍密码：

    ```
    (base) hlc@hlc-VirtualBox:~$ ssh -t hlc@10.133.1.54 "sudo ls"
    hlc@10.133.1.54's password: 
    [sudo] password for hlc: 
    add_proxy.sh  Documents   mpi_shared  perf_0924.tar.gz	Public	Softwares  work
    Data	      Downloads   Music       perftest		rdma	Templates
    Desktop       miniconda3  nfs_shared  Pictures		snap	Videos
    Connection to 10.133.1.54 closed.
    ```

    如果使用 public key 登陆，而非密码登陆，那么非交互环境可以使用`sudo`。

    所以说，sudo 无法运行，其实是因为输入 sudo 密码默认需要在交互环境里。如果 sudo 无需输入密码，或者经过配置，允许在非交互环境输入 sudo 密码，那么就不需要交互环境。

* ssh 直接执行远程命令

    `ssh <user>@<host> <command>`可以在登陆的时候直接在远程机器上执行命令，不显示欢迎信息。

    example:

    `ssh <user>@<host> ls`

    output:

    ```
    04_local_res.cpp
    Data
    Desktop
    Documents
    Downloads
    link_mnt_hlc_to_Data.sh
    mlnx_perftest_log.txt
    Music
    nfs_shared
    nvshmem_srcs.tar.gz
    Pictures
    Public
    snap
    Softwares
    Templates
    use-proxy.sh
    Videos
    Videos-link
    ```

    如果 command 里有空格，可以用双引号包起：

    `ssh <user>@<host> "ls /tmp"`

    如果使用**双引号**括起的命令里有变量`$VAR_NAME`，那么会被解析为本地变量。如果想将其解析为远程主机上的变量，那么需要使用**单引号**将 command 括起：

    `ssh <user>@<host> 'echo $VAR_NAME'`

    如果要执行多条命令，那么可以使用和 bash 相似的技巧：

    * 用分号分隔：`ssh user@host "cmd1; cmd2"`

    * 逻辑控制：`ssh user@host "ls /tmp && echo success || echo fail"`

* `ssh -W <target_ip>:<port> <user>@<jump_ip>`

    `ssh -W`表示登陆 jump host 的 ssh，并将当前窗口的所有 stdin, stdout 都转发到 target host 上。这个功能通常只能用于转发 ssh 数据。

    如果`<target_ip>:<port>`被设置为 target host 上的`nc -l <port>`，那么会报错：

    ```
    channel 0: open failed: connect failed: Connection refused
    stdio forwarding failed
    ```

    由此可见，这个命令可能并不是个通用命令，而是专门用于跳板机的 ssh 配套命令。

    `<target_ip>`通常使用`%h`表示，`<port>`通常使用`%p`表示，这样可以直接匹配到需要连接的主机：

    `ssh -o ProxyCommand="ssh -W %h:%p <user>@<jump_host>" <user>@<target_host>`

    此时`%h`会被自动替换成`<target_host>`，而`%p`会被自动替换成`22`。
    
    跳板机的 sshd 需要开启`AllowTcpForwarding yes`，默认情况下是开启的。

    `~/.ssh/config`的写法如下：

    ```
    Host target-host
        HostName 目标主机真实IP
        User 目标主机用户
        ProxyCommand ssh -W %h:%p 跳板机用户@跳板机IP
    ```

* `ssh -A`选项的全称为 Agent Forwarding，对应的 ssh command 为`ForwardAgent`，可以将本地的 ssh 私钥（比如`id_rsa`）临时放到远端机器上。

    假设我们当前的机器为`A`，需要登陆的机器为`C`，现在有个中间机器`B`。如果我们按通常方式`ssh <user>@<B_addr>`登陆到 B 机器，再使用`ssh <user>@<C_addr>`登陆到 C 机器，那么 C 会认为是 B 申请的登陆。假如 C 机器只存储了 A 的公钥，而不允许 B 登陆，那就只能使用`-A`选项了。

    先使用`ssh -A <user>@<B_addr>`登陆到 B，再使用`ssh <user>@<C_addr>`登陆到 C，此时 C 会认为是 A 申请的登陆。

    同理，使用跳板机参数`-J`时，也可以使用这个功能：`ssh -A -J <jump_user>@<jump_addr> <target_user>@<target_addr>`。

* 使用 scp + 跳板机传文件

    `scp -P <target_host_port> -J <jump_host_user>@<jump_host_ip> -r <local_src_dir> <target_host_user>@<target_host_ip>:<dst_path>`

* <https://www.ssh.com/academy/ssh/client>

    ssh 相关的概念、术语和解释，讲得比较系统，有时间了看看，了解下为什么 ssh 要这样设计。

* ssh 设置 socks 代理

    `ssh -X -C -p 39147 -D 4321 -o ServerAliveInterval=60 <user>@<ip_addr>`

    `-D 4321`表示监听本地`127.0.0.1:4321`，作为 socks server port

    `-o ServerAliveInterval=60`表示 keep alive，不然一段时间不操作，ssh 会自动退出

    socks 不对通信数据加密，比较适合局域网，不太适合公网。

    `-C`：压缩传输的数据，减小通信量。实测了下，对 cpu 的负荷较小，也不太影响延迟。

* ssh 反向代理

    * 下面这三条命令等价
    
        * `ssh -R 8822:127.0.0.1:8822 <user>@<host_addr>`

        * `ssh -R *:8822:127.0.0.1:8822 <user>@<host_addr>`

        * `ssh -R 0.0.0.0:8822:127.0.0.1:8822 <user>@<host_addr>｀

        这三条命令都是从远程 host 的任意地址的 8822 端口向本地的 127.0.0.1 的 8822 端口开一条 tunnel。

        如果在登陆时有提示：

        ```
        SIOCSIFADDR: Operation not permitted
        SIOCSIFFLAGS: Operation not permitted
        SIOCSIFNETMASK: Operation not permitted
        SIOCADDRT: Operation not permitted
        ```

        这些提示不影响反射代理的功能。

        如果`/etc/ssh/sshd_config`文件中的`GatewayPorts yes`没有打开，那么只能转发远程 host 的`127.0.0.1`的 8822 端口，无法 bind 其他地址。

    * 如果使用`-R`转发远程机器的端口，那么其他的远程机器也可以使用这个端口完成和外界的通信。

        比如现在有 host A, B, C 三台机器，先用 A 连接 B：

        host A: `ssh -R 8822:127.0.0.1:8822 <user>@<host_B_addr>`

        此时访问 B 任意 ipv4 的 8822 端口，都会被转发到 A 的 127.0.0.1:8822 端口。

        我们在 host A 上起一个 tcp server: `nc -l 8822`

        然后在 host C 上执行`nc <host_B_addr> 8822`，即可成功连接到 host A 上的 tcp server。输入 message 并按回车，可以看到 host A 上的回显。

* 命令行启动的 qemu 使用 X11 forward 时似乎不会产生内存泄漏

## note

Learning materials:

1. <https://www.ssh.com/academy/ssh/config>

1. <https://linuxize.com/post/using-the-ssh-config-file/>

正向转发：假如现在有两台机器 A 和 B，B 上装有 ssh server，A 想在访问本机的某个端口时，变成访问机器 B 上的某个端口，那么就称为正向代理。

此时在`A`机器上运行：

`ssh -L [A_addr:]<A_port>:<B_addr>:<B_port> user_name@addr`

登陆就可以了。

注意，`addr`不一定和`B_addr`相同。若，则通过`addr`转发到`B_addr`上。

反向代理：假如现在有机器`A`和`B`，`B`上装有 ssh server，目标是在`B`上访问`B_port`端口时，相当于访问`A_port`端口。

命令：

`ssh -R B_addr:B_port:A_addr:A_port user@addr`

此时在`B`上访问`B_addr:B_port`就当于访问`A_addr:A_port`。

同理，`addr`不一定和`B_addr`相同。

这种形式相当于内网穿透。

生成一个密钥对：`ssh-keygen -t rsa`

在远程机吕上把公钥写到`authorized_keys`文件里面：`cat ~/id_rsa.pub >> ~/.ssh/authorized_keys`

## Installation

安装 openssh-server:

`sudo apt install openssh-server`

查看服务列表：

`service --status-all`

output:

```
 [ + ]  acpid
 [ - ]  alsa-utils
 [ - ]  anacron
 [ + ]  apparmor
 [ + ]  apport
 [ + ]  avahi-daemon
 [ - ]  bluetooth
 [ - ]  console-setup.sh
 [ + ]  cron
 [ + ]  cups
 [ + ]  cups-browsed
 [ + ]  dbus
 [ + ]  gdm3
 [ - ]  grub-common
 [ - ]  hwclock.sh
 [ + ]  irqbalance
 [ + ]  kerneloops
 [ - ]  keyboard-setup.sh
 [ + ]  kmod
 [ - ]  nfs-common
 [ + ]  nfs-kernel-server
 [ + ]  openvpn
 [ - ]  plymouth
 [ + ]  plymouth-log
 [ + ]  procps
 [ - ]  pulseaudio-enable-autospawn
 [ + ]  rpcbind
 [ - ]  rsync
 [ - ]  saned
 [ - ]  speech-dispatcher
 [ - ]  spice-vdagent
 [ + ]  ssh
 [ + ]  ubuntu-fan
 [ + ]  udev
 [ + ]  ufw
 [ + ]  unattended-upgrades
 [ - ]  uuidd
 [ - ]  whoopsie
 [ - ]  x11-common
```

看到有` [ + ]  ssh`，说明 ssh 服务启动成功。

启动 ssh 服务：

`service ssh start`

## connect to a ssh server through a jump/intermediary server

Ref: <https://www.cyberciti.biz/faq/linux-unix-ssh-proxycommand-passing-through-one-host-gateway-server/>

(This artical did not mentioned the usage of ProxyCommand `nc xxx`.)

The simplest way to use a imtermediary to connect to another ssh server is

```bash
ssh -J <use_1@address_1[:port]> <user_2@address_2>
```

* What do the `%h` and `%p` mean in the proxy command?

    <https://unix.stackexchange.com/questions/183951/what-do-the-h-and-p-do-in-this-command>

## ssh config file

The `~/.ssh` directory is automatically created when the user runs the ssh command for the first time. If the directory doesn’t exist on your system, create it using the command below:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
```

By default, the SSH configuration file may not exist, so you may need to create it using the touch command :

```bash
touch ~/.ssh/config
```

This file must be readable and writable only by the user and not accessible by others:

```bash
chmod 600 ~/.ssh/config
```

The SSH Config File takes the following structure:

```ssh
Host hostname1
    SSH_OPTION value
    SSH_OPTION value

Host hostname2
    SSH_OPTION value

Host *
    SSH_OPTION value
```

Indentation is not required but is recommended since it makes the file easier to read.

The `Host` directive can contain one pattern or a whitespace-separated list of patterns. Each pattern can contain zero or more non-whitespace character or one of the following pattern specifiers:

* `*` - Matches zero or more characters. For example, Host * matches all hosts, while 192.168.0.* matches hosts in the 192.168.0.0/24 subnet.

* `?` - Matches exactly one character. The pattern, Host 10.10.0.? matches all hosts in 10.10.0.[0-9] range.

* `!` - When used at the start of a pattern, it negates the match. For example, Host 10.10.0.* !10.10.0.5 matches any host in the 10.10.0.0/24 subnet except 10.10.0.5.

The SSH client reads the configuration file stanza by stanza, and if more than one patterns match, the options from the first matching stanza take precedence. Therefore more host-specific declarations should be given at the beginning of the file, and more general overrides at the end of the file.

You can find a full list of available ssh options by typing `man ssh_config` in your terminal or visiting the `ssh_config` `man` page <http://man.openbsd.org/OpenBSD-current/man5/ssh_config.5>.

The SSH config file is also read by other programs such as scp , sftp , and rsync .

Example:

```ssh
Host dev
    HostName dev.example.com
    User john
    Port 2322
```

connect: `ssh dev`

Shared SSH Config File Example:

```ssh
Host targaryen
    HostName 192.168.1.10
    User daenerys
    Port 7654
    IdentityFile ~/.ssh/targaryen.key

Host tyrell
    HostName 192.168.10.20

Host martell
    HostName 192.168.10.50

Host *ell
    user oberyn

Host * !martell
    LogLevel INFO

Host *
    User root
    Compression yes
```

When you type ssh targaryen, the ssh client reads the file and apply the options from the first match, which is Host targaryen. Then it checks the next stanzas one by one for a matching pattern. The next matching one is Host * !martell (meaning all hosts except martell), and it will apply the connection option from this stanza. The last definition Host * also matches, but the ssh client will take only the Compression option because the User option is already defined in the Host targaryen stanza.

The full list of options used when you type ssh targaryen is as follows:

```ssh
HostName 192.168.1.10
User daenerys
Port 7654
IdentityFile ~/.ssh/targaryen.key
LogLevel INFO
Compression yes
```

When running ssh tyrell the matching host patterns are: Host tyrell, Host *ell, Host * !martell and Host *. The options used in this case are:

```ssh
HostName 192.168.10.20
User oberyn
LogLevel INFO
Compression yes
```

The ssh client reads its configuration in the following precedence order:

1. Options specified from the command line.
1. Options defined in the ~/.ssh/config.
1. Options defined in the /etc/ssh/ssh_config.

If you want to override a single option, you can specify it on the command line. For example, if you have the following definition:

```ssh
Host dev
    HostName dev.example.com
    User john
    Port 2322
```

and you want to use all other options but to connect as user root instead of john simply specify the user on the command line:

`ssh -o "User=root" dev`

The -F (configfile) option allows you to specify an alternative per-user configuration file.

To tell the ssh client to ignore all of the options specified in the ssh configuration file, use:

`ssh -F /dev/null user@example.com`

## Problem shooting

* `debug1: Local version string SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1`

    `Connection closed by UNKNOWN port 65535`

    Usually it's the network problem. Please check `~/.ssh/config` if there is any unexpected config.

    Also please check proxy configs.

* 启动服务`ssh`时，报错`sshd: no hostkeys available -- exiting`

    需要进入`/etc/ssh`文件夹下生成 rsa 的公钥：

    `ssh-keygen -A`

    然后再启动`ssh`就可以了：`sudo service ssh start`。

* 客户端登录`ssh`时，报错`no hostkeys available`

    需要在服务器端修改配置文件`/etc/ssh/sshd_config`，把`PasswordAuthentication`改成`yes`。

    此时用`sudo service ssh restart`重启服务好像也没用，可能需要重启系统才行。

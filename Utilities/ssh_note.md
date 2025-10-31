# Ssh Note

## cache

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

        OK，stdout 和 stderr 被重定向到`/dev/null`文件，并且命令的最后有`&`表示在后台执行。此时 ssh 会立即返回。

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
# Linux Maintenance Note

这里主要记录 linux 相关的命令，配置，主题为运维。

这里不记录编程相关，操作系统原理相关。

## cache

* terminal 中，启动 ssh 进入远程机器，然后再退出 ssh 后，local terminal 会出现文本无法自动换行的情况，如何解决

    这是一个常见的终端窗口尺寸同步问题。当你在 SSH 会话中调整了终端窗口大小，退出后本地终端没有正确同步回来。

    解决方法：

    1. 最简单的方法

        直接输入以下命令重置终端：

        ```bash
        reset
        ```

        或者按 Ctrl+L 清屏有时也能解决问题。

    2. 调整窗口大小

        手动触发一次窗口大小调整：

        ```bash
        # 方法1：改变终端窗口大小（拖动边缘即可）
        # 方法2：按 Ctrl+C 中断当前行，然后回车
        # 方法3：输入：
        stty sane
        ```

    3. 在 SSH 会话中正确退出

        ```bash
        # 不要直接按 Ctrl+D 退出，先执行：
        exit
        ```

        或者确保在退出前执行：

        ```bash
        printf '\033[8;%d;%dt' $(stty size)
        ```

    4. 预防措施

        在 ~/.bashrc 或 ~/.bash_profile 中添加：

        ```bash
        # 设置 SSH 退出时重置终端
        trap 'echo -ne "\033]0;${USER}@${HOSTNAME}: ${PWD}\007"' DEBUG
        ```

    5. 如果以上都不行

        关闭当前终端标签/窗口，重新打开一个新的。

    6. 永久解决方案

        在 SSH 配置中启用终端窗口大小转发，编辑 ~/.ssh/config：

        ```text
        Host *
            ServerAliveInterval 60
            TCPKeepAlive yes
            EscapeChar none
        ```

    最常用的是方法1：直接输入 reset 回车即可恢复正常。

* 将 journactl 中的中文日期改成英文

    方法1：临时设置环境变量（当前会话有效）

    ```bash
    # 设置语言为英文
    export lang=en_us.utf-8
    export lc_time=en_us.utf-8

    # 查看日志
    journalctl -u sshd
    ```

    方法2：永久修改系统语言

    ```bash
    # 编辑 locale 配置文件
    sudo vim /etc/locale.conf

    # 添加或修改以下内容
    lang="en_us.utf-8"
    lc_time="en_us.utf-8"

    # 或者使用 sed 命令
    sudo sed -i 's/lang=.*/lang="en_us.utf-8"/' /etc/locale.conf
    echo 'lc_time="en_us.utf-8"' | sudo tee -a /etc/locale.conf

    # 重启系统或重新加载环境
    sudo systemctl restart systemd-journald
    ```

    方法3：journalctl 特定命令选项

    ```bash
    # 使用 --no-full 和特定输出格式
    journalctl -u sshd --since "2023-12-02 21:27:27" --output json

    # 或者使用特定字段显示
    journalctl -u sshd -o json-pretty | grep -e '(message|__realtime_timestamp)'
    ```
    
* less 中实现大小写不敏感的搜索

    1. 启动 less 时设置参数

        ```bash
        less -I filename
        ```

        或

        ```bash
        less -i filename
        ```

        区别：

        * -I：搜索时完全忽略大小写

        * -i：搜索时智能忽略大小写（如果搜索模式包含大写字母，则区分大小写）

    2. 在 less 内部设置

        进入 less 后，可以输入

        `-I`或`-i`   

        来切换大小写敏感设置。

    3. 使用环境变量

        可以在 shell 配置文件中设置默认选项：

        ```bash
        # 在 ~/.bashrc 或 ~/.zshrc 中添加
        export LESS="-I"
        ```

    4. 搜索时指定选项

        在 less 中使用 / 搜索时，可以：

        * 先输入 -i 再按回车，然后进行搜索

        * 或者直接在搜索模式前加 -i：

        ```text
        /-itext
        ```

        这将搜索 "text" 并忽略大小写

    5. 永久配置

        编辑 ~/.lesskey 文件（如果不存在则创建）：

        ```text
        # 设置默认忽略大小写
        -i
        ```

    提示：

    * 在 less 中，按 -i 可以切换大小写敏感模式

    * 当前设置状态会显示在左下角（如果有 -i 标志表示忽略大小写）

    * 按 = 可以查看当前 less 的设置状态

* 在 less 中，无法实现输入 /xx 后不按回车立即搜索。

* Here Document

    Here Document（文档内嵌）是一种在 Shell 脚本中直接嵌入多行文本输入的方法，通常用于命令的标准输入。

    syntax:

    ```bash
    命令 << 分隔符
        多行文本内容
    分隔符
    ```

    分隔符的两种形式

    1. 不带引号的分隔符 - 会进行变量替换和命令替换

        ```bash
        cat << EOF
        当前用户：$USER
        当前目录：$(pwd)
        EOF
        ```

    2. 带引号的分隔符 - 禁用变量和命令替换

        ```bash
        cat << 'EOF'
        当前用户：$USER  # 不会被替换
        当前目录：$(pwd)  # 不会被替换
        EOF
        ```

        (双引号的`"EOF"`可以吗？)

    常见用法示例

    1. 远程执行命令

        ```bash
        ssh user@hostname << 'EOF'
        cd /path/to/dir
        ./script.sh
        echo "任务完成"
        EOF
        ```

    2. 创建配置文件

        ```bash
        cat > /etc/config.conf << EOF
        server_ip=192.168.1.100
        port=8080
        timeout=30
        EOF
        ```

        注：

        1. 重定向`> /etc/config.conf`可以写到命令和参数的中间，记一下这个用法。

    3. 传递复杂命令参数

        ```bash
        mysql -u root -p << EOF
        USE database;
        SELECT * FROM users;
        EXIT;
        EOF
        ```

    4. 使用变量（不转义）

        ```bash
        name="Alice"
        cat << EOF
        Hello $name,
        Welcome to the system.
        EOF
        ```

    高级用法

    * 缩进 Here Document（使用 <<-）
    
        ```bash
        if true; then
            cat <<- EOF
            This line is indented.
            This too.
            EOF
        fi
        ```

        `<<-`允许每行的前面有制表符，在转换为文档时，会删除每行开头的制表符(Tab)。而`<<`会保持原样缩进。

        对于结束标记，`<<`要求必须顶格写，前后无空格。`<<-`则允许`EOF`前面有制表符。

        `<<-`的主要用途是在脚本中保持代码结构美观。

        注意：

        * `<<-`只删除制表符(Tab)，不删除空格，对每行开头的空格无效。

        * `<<-`只删除每行开头的连续制表符。
        
            这意味着如果制表符在空格后面，那么不会被删除。

    * 重定向到文件

        ```bash
        exec > output.log << EOF
        日志开始
        $(date)
        操作完成
        EOF
        ```

    注意事项

    * 分隔符可以是任意字符串，常见的有 EOF、END、STOP 等

    * 结束分隔符必须单独一行，前后不能有空格（除非使用 <<-）

    * 通常与 cat、ssh、mysql、ftp 等需要多行输入的命令配合使用

    * 使用 'EOF' 可以避免脚本中的特殊字符被解释

    这样可以使脚本更清晰，避免使用多个 echo 命令输出多行内容。

* autossh

    Autossh 是一个用于创建持久 SSH 隧道的工具，当连接断开时会自动重连。它通过监控 SSH 连接状态并在断开时重新启动 SSH 会话来确保隧道的稳定性。

    syntax:

    ```bash
    autossh [选项] -M <监控端口> <SSH 命令>
    ```

    * `-M <端口>`: 指定一个监控端口（用于检测连接状态），通常设为 0 让系统自动分配。

    * `-f`: 后台运行。

    * `-N`: 不执行远程命令（仅用于端口转发）。

    * `-L/-R/-D`: 与 SSH 相同的端口转发参数。

    **常用场景示例**

    1. 本地端口转发

        将本地端口 8080 转发到远程服务器的 80 端口：

        ```bash
        autossh -M 0 -f -N -L 8080:localhost:80 user@remote-host
        ```

    2. 远程端口转发

        将远程服务器的 3306 端口转发到本地的 3306 端口：

        ```bash
        autossh -M 0 -f -N -R 3306:localhost:3306 user@remote-host
        ```

    3. 动态 SOCKS 代理

        创建持久的 SOCKS5 代理（本地端口 1080）：

        ```bash
        autossh -M 0 -f -N -D 1080 user@remote-host
        ```

    **高级选项**

    * `-M 0`: 自动选择监控端口（推荐）。

    * `-o ServerAliveInterval=60`: 每 60 秒检测一次连接。

    * `-o ExitOnForwardFailure=yes`: 转发失败时退出。

    * `-o ServerAliveCountMax=3`: 最多重试 3 次。

    示例：

    ```bash
    autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -N -L 9000:localhost:22 user@remote-host
    ```

    **开机自启（Systemd）**

    创建服务文件 /etc/systemd/system/autossh-tunnel.service：

    ```conf
    [Unit]
    Description=AutoSSH Tunnel
    After=network.target

    [Service]
    User=your-username
    ExecStart=/usr/bin/autossh -M 0 -N -L 8080:localhost:80 user@remote-host
    Restart=always

    [Install]
    WantedBy=multi-user.target
    ```

    启用服务：

    ```bash
    sudo systemctl enable --now autossh-tunnel.service
    ```

    **注意事项**

    * 免密登录: 建议配置 SSH 密钥认证，避免输入密码。

    * 防火墙: 确保监控端口和转发端口未被阻塞。

    * 资源占用: 长时间运行时可结合 -o ConnectTimeout=30 优化超时设置。

* tail 的常用选项：

    ```bash
    # 从第N行开始显示
    tail -f -n +50 filename

    # 同时跟踪多个文件
    tail -f file1 file2 file3

    # 高亮显示关键字
    tail -f filename | grep --color=auto "keyword"
    ```

    对于日志文件的特殊处理：

    ```bash
    # 过滤包含特定关键词的行
    tail -f filename | grep "error"

    # 排除特定内容
    tail -f filename | grep -v "debug"

    # 彩色输出
    tail -f filename | ccze -A
    ```

* 实时追踪文本文件的最新内容

    * tail -f (最常用)

        ```bash
        # 基本用法
        tail -f filename

        # 显示行号
        tail -f -n 20 filename

        # 等同于 --follow=descriptor，文件被移动或重命名后仍能跟踪
        tail -F filename
        ```

    * less 的实时模式

        ```bash
        # 打开文件后，按 Shift+F 进入实时跟踪模式
        less filename
        # 然后按 Shift+F 开始跟踪，Ctrl+C 停止跟踪，回到普通浏览模式
        ```

    * multitail (功能更强大)

        ```bash
        # 安装 multitail
        sudo apt install multitail  # Ubuntu/Debian
        sudo yum install multitail  # CentOS/RHEL

        # 使用 multitail
        multitail filename
        ```

    * 使用 awk 实时处理

        ```bash
        # 结合 tail 和 awk 进行实时处理
        tail -f filename | awk '{print "New line:", $0}'
        ```

* linux 查看当前目录的大小

    1. du 命令（最常用）

        ```bash
        # 显示当前目录的总大小（人类可读格式）
        du -sh .

        # 显示详细的大小信息（包括子目录）
        du -sh *
        ```

        常用参数：

        * `-s`：汇总，只显示总大小

        * `-h`：人类可读格式（KB、MB、GB）

        * `-c`：显示总计

        * `--max-depth=N`：限制显示层级

    2. 显示当前目录的详细大小信息

        ```bash
        # 显示当前目录及所有子目录的大小
        du -h --max-depth=1

        # 按大小排序显示
        du -h --max-depth=1 | sort -hr
        ```

    3. 使用 ncdu（需要安装，但非常直观）
    
        ```bash
        # 安装ncdu
        sudo apt install ncdu    # Debian/Ubuntu
        sudo yum install ncdu    # CentOS/RHEL

        # 使用ncdu
        ncdu
        ```

    4. 显示磁盘使用情况

        ```bash
        # 查看整个文件系统的使用情况
        df -h .

        # 只显示当前目录所在分区的使用情况
        df -h $PWD
        ```

    5. 其他实用命令

        ```bash
        # 快速查看当前目录大小（以字节为单位）
        du -sb

        # 排除某些文件类型（如排除.log文件）
        du -sh --exclude="*.log" .

        # 仅显示超过特定大小的目录
        du -h --max-depth=1 | grep '[0-9]G\>'  # 显示GB级别的目录
        ```

    主要区别：

    * `du`：计算文件和目录占用的实际磁盘空间

    * `df`：显示文件系统的整体使用情况

    * `ncdu`：交互式磁盘使用分析器，更适合深入分析

    常用组合命令：

    ```bash
    # 查找当前目录下最大的10个文件/目录
    du -ah . | sort -rh | head -10

    # 只显示目录大小（不包括文件）
    du -h --max-depth=1 -t 1M .  # 只显示大于1MB的目录
    ```

    推荐使用：`du -sh .` 这是最简单直接的查看当前目录大小的方法。

* `watch "ps -aux | grep v2ray"`没输出, `watch bash -c "ps -aux | grep v2ray"`也没输出

    尝试了多种方法都未能解决，将这个作为疑难杂症问题长期保存吧

* 使用 bash 启动程序时，单行环境变量要写在脚本前面

    `run.sh`:

    ```bash
    ./$1
    ```

    ```bash
    LD_LIBRARY_PATH=xxx bash run.sh main  # OK

    bash LD_LIBRARY_PATH=xxx run.sh main  # error

    bash run.sh LD_LIBRARY_PATH=xxx main  # error
    ```

    环境变量 LD_LIBRARY_PATH 会传递给 bash 进程，然后在 bash 中执行的脚本（run_main.sh）及其子进程（包括 ./main）都会继承这个变量。

    其他传递环境变量的方法：

    * 使用 export

        ```bash
        export LD_LIBRARY_PATH=/path/to/libs
        bash run_main.sh
        ```

* systemd 与 ssh tunnel

    systemd 中启动 ssh tunnel 时，不要使用`ssh -f`，因为这会

    1. 创建一个 ssh 的前台程序，执行登陆认证等操作，假设其 pid 为 PID_1

    2. 成功登录后，fork 一份进程到后台，此时后台进程的 pid 为 PID_2

    3. 退出 PID_1 的 ssh 前台进程

    systemd 检测到 PID_1 退出，会认为 ssh 进程已经结束，从而导致 systemd 错误判断 service 的状态。

    因此我们直接使用`ssh -NL`或`ssh -NR`就可以。

    example:

    ```conf
    [Unit]
    Description=SSH Reverse Tunnel
    After=network.target

    [Service]
    Type=simple
    User=your_username
    # 使用密钥认证，避免交互
    ExecStart=/usr/bin/ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R 12345:localhost:22 user@remote-server
    Restart=always
    RestartSec=10
    # 密钥权限很重要
    Environment="HOME=/home/your_username"

    [Install]
    WantedBy=multi-user.target
    ```

    ```bash
    # 添加这些选项提高稳定性
    -o ExitOnForwardFailure=yes    # 端口转发失败时退出
    -o ServerAliveInterval=30      # 30秒发送一次保活包
    -o ServerAliveCountMax=3       # 3次无响应后断开
    -o TCPKeepAlive=yes
    -o BatchMode=yes               # 禁用交互提示
    ```

* `xrdp`

    这是一个 独立的、完整的 RDP 服务器 软件。它运行在 Linux 系统上，等待来自 RDP 客户端的连接请求。

    为了提供桌面，`xrdp` 需要依赖一个后端的图形会话管理器。最常见的是：

    * Xvnc：xrdp 会启动一个 VNC 服务器（如 TigerVNC、X11VNC）来承载桌面，然后 xrdp 在 RDP 和 VNC 协议之间进行转换。这是最常见的配置。

    * Xorg：较新的版本支持使用一个专门的 Xorg 会话作为后端（xrdp-xorg 模块），性能比 VNC 模式更好。

    与 FreeRDP 模块的核心区别：

    * xrdp 是一个常驻的系统服务（systemd 服务），监听 3389 端口，允许多个用户建立独立的、全新的桌面会话（登录屏幕 -> 输入用户名密码 -> 进入一个独立的桌面环境）。

    * freerdp2-shadow-x11 是一个临时工具，用于共享已经登录的、正在使用的现有桌面会话。它不提供登录管理器，不创建新会话。

* 查看当前 session 是 x11 还是 wayland

    `echo $XDG_SESSION_TYPE`

    output:

    `wayland`

* FreeRDP2

    FreeRDP 是一个开源的 RDP 客户端和服务器端库。`freerdp2-x11`和`freerdp2-wayland`是它的命令行工具`xfreerdp`的两个后端。

    * `freerdp2-x11`

        这是 RDP 客户端 在 X11 显示系统下的主程序。你用它来连接远程的 Windows 机器或其他 RDP 服务器。

    * `freerdp2-wayland`

        同样是 RDP 客户端，它使用 Wayland 原生接口进行渲染，而不是 X11。

    * `freerdp2-shadow-x11`

        这是 FreeRDP 的 “影子服务器” 或 “桌面共享” 组件。它用于将本机的 X11 桌面会话共享出去，供其他 RDP 客户端连接。

        它捕获当前 X11 显示器的输出，将其作为一个 RDP 会话对外提供。其他用户可以使用任意的 RDP 客户端（如 Windows 自带的 mstsc.exe、Android 客户端、或 xfreerdp 本身）来接入你的当前桌面。

        它不是客户端，而是一个服务端。但它不创建新的桌面会话，只是“投影”现有会话。

        通常通过命令行启动，例如 xfreerdp-shadow-subsystem。

    通常 xfreerdp 可以自动选择后端，但是我们也可手动指定后端：

    ```bash
    # 强制使用 X11 后端（即使在 Wayland 会话中）
    xfreerdp /b:x11 ...

    # 强制使用 Wayland 后端
    xfreerdp /b:wayland ...
    ```

* network.target vs network-online.target

    network.target：

        网络配置完成（接口已配置）

        不保证实际网络连通性

        启动较快

    network-online.target：

        网络真正连通（可以访问外部网络）

        等待 DHCP、DNS 等完全就绪

        启动较慢，可能超时

    所以这两个区别是，一个内网能访问通，一个能访问到公网？

* systemd 服务设置 ssh -R 反向隧道开机自启动

    1. 创建 systemd 服务文件

        ```bash
        sudo nano /etc/systemd/system/ssh-reverse-tunnel.service
        ```

    2. 编辑服务文件内容

        ```conf
        [Unit]
        Description=SSH Reverse Tunnel
        After=network.target

        [Service]
        Type=simple
        User=your_username
        ExecStart=/usr/bin/ssh -N -R remote_port:localhost:local_port username@remote_host -p remote_ssh_port -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new
        Restart=always
        RestartSec=10

        [Install]
        WantedBy=multi-user.target
        ```

    3. 设置和启动服务

        ```bash
        # 重新加载 systemd
        sudo systemctl daemon-reload

        # 设置开机自启动
        sudo systemctl enable ssh-reverse-tunnel.service

        # 立即启动服务
        sudo systemctl start ssh-reverse-tunnel.service

        # 检查服务状态
        sudo systemctl status ssh-reverse-tunnel.service
        ```

    注意事项

        确保网络连通性: 服务在网络就绪后启动

        使用非特权端口: 如果非root用户运行，remote_port通常需要大于1024

        监控日志: 使用 journalctl -u ssh-reverse-tunnel.service -f 查看日志

        安全考虑: 确保远程服务器是可信的，因为反向隧道会暴露本地服务

    注：

    1. `After=network.target`
    
        不写`After=network-online.target`是因为后面有无限重连做保证。

        但是我觉得直接写成`After=network-online.target`更好，直接一步到位。

    1. `ExitOnForwardFailure=yes`

        如果无法端口转发，那么 ssh 连接直接报错。如果远程的端口被其他程序占用，那么 ssh 报错退出。

        ssh 的默认设置是如果端口转发失败（比如远程端口已被占用），SSH 连接仍然会建立，端口转发功能实际上没有工作。

        这个设置配合 systemd 的自动重启服务，可以一定程序上解决远程端口被占用的问题。

    1. `Type=simple`

        SSH 命令会长期运行，保持连接和隧道。进程在前台持续运行，不会立即退出。systemd 会监控这个长期运行的进程。

        如果使用`Type=oneshot`，那么程序执行完就退出。这个配置配合`RemainAfterExit=yes`使用，检查 status 时的效果如下：

        ```
        active (exited)
        ```

        如果不设置`RemainAfterExit=yes`，则会变成

        ```
        inactive
        ```

        使用 simple 的好处：

        * systemd 直接监控主进程

        * 进程退出时自动重启

        * 完整的生命周期管理

        * 简单的日志收集

        * systemctl stop 能正确终止进程

* 把 rdp 的流量包在 ssh 流量里

    通常使用`ssh -L`进行转发：

    `ssh -N -L 本地端口:目标RDP服务器地址:3389 SSH用户名@SSH服务器地址`

    然后使用 xfreerdp 连接本地隧道端口：

    `xfreerdp /v:127.0.0.1:13389 /u:RDP用户名 /p:RDP密码`

    xfreerdp 本身没有原生支持 ssh tunnel 的方法

* `http://security.ubuntu.com/ubuntu/ jammy-security restricted multiverse universe main`的 ip 为`1.1.1.3`，属于 cloudflare 的机器，国内不一定能访问到。

    如果在`apt update`时无法访问这个 ip 的 80 端口，可以考虑在`/etc/apt/source.list`里把这一行注释掉。

* linux 下复制文件时显示进度

    `pv`（Pipe Viewer）可以显示复制进度、速度和估计时间。

    安装：`sudo apt install pv`

    * 复制单个文件
    
        `pv source_file > destination_file`

    * 复制整个目录

        `tar cf - source_dir | pv | tar xf - -C destination_dir`

    可以创建一个带进度复制的脚本：

    `/usr/local/bin/cpv`:

    ```bash
    #!/bin/bash
    tar cf - "$1" | pv | tar xf - -C "$2"
    ```

* ls 按访问时间排序

    ```bash
    ls -ltu
    ```

    -l：长格式显示

    -t：按时间排序（默认是 mtime，但配合 -u 就是按 atime 排序）

    -u：使用访问时间（atime）而不是修改时间（mtime）

* `ls`默认不支持直接按文件创建时间（birth time）排序

    ls 主要显示的是文件的修改时间（mtime）、访问时间（atime）和状态变更时间（ctime）

    较新的内核支持按创建时间排序：

    ```bash
    ls -lt --time=birth
    ```

    -l：长格式显示

    -t：按时间排序（默认是 mtime，但配合 --time=birth 就是按创建时间排序）

    --time=birth：显示并按创建时间排序

* `ls -l`命令将最近修改时间的文件放到最上面

    `ls -lt`

    * -t：按修改时间排序，最新的文件在最前面。

    ls 的其他选项：

    * `-r`：倒序

    * `-a`：显示隐藏文件

* `grep -E`

    主要特点：

    * 支持扩展正则语法：可以使用 |, +, ?, {} 等元字符而无需转义

    * 等同于 egrep：grep -E 与 egrep 命令功能相同

    * 更强大的模式匹配：相比基本正则表达式，提供更丰富的模式匹配能力

    `grep -E "keyword1|keyword2|keyword3" file.txt`: 在 file.txt 文件中搜索包含 keyword1 或 keyword2 或 keyword3 任意一个关键词的所有行。

    ```bash
    # 使用基本正则表达式（需要转义 |）
    grep "keyword1\|keyword2\|keyword3" file.txt

    # 使用扩展正则表达式（更简洁）
    grep -E "keyword1|keyword2|keyword3" file.txt
    ```

    `|`前后不能有空格，如果有空格，那么空格也会被匹配，是 keyword 的一部分。

* `ls -R`

    递归列出目录及其所有子目录中的内容。

    -R 是 “Recursive”（递归）的缩写。

* `ls -lS`

    以长格式列出文件，并按文件大小降序排序（从大到小）。

    -l 是 “long format” 的缩写，会显示详细信息（权限、所有者、大小、修改时间等）。

    -S 是 “Sort by size” 的缩写（注意是大写S）。

* `ls -lr`

    以反向（逆序） 方式列出文件和目录。

    -r 是 “reverse” 的缩写（注意是小写r）。

    组合使用后，它会将排序结果反转。默认情况下（没有其他排序选项），ls -l 是按文件名升序排序（a-z, 0-9），加上 -r 后就变成了降序（z-a, 9-0）。

    与其他排序选项结合时，用于反转排序顺序。例如，ls -lSr 会按文件大小升序排列（从小到大），因为 -S（大小降序）被 -r 反转了。

* 不默认保持中文输入法的另外一个原因

    有时候需要按`shift` + 鼠标滚轮横向滚动，但是`shift`又正好是切换中英文的，这样会导致每横向滚动一次，就在中英文中间切换一次。

* `ulimit -a`

    查看当前用户 Shell 进程及其子进程所能使用的系统资源限制情况。

    ulimit（User Limit）是一个 Shell 内建命令，用于控制和显示用户可用的资源限制。

    -a 是 --all 的缩写，表示 显示所有（All）当前的资源限制设置。

    常见和重要的限制项：

    | 限制项 | 参数 | 示例值 | 含义解释 |
    | - | - | - | - |
    | open files | -n | 1024 | 单个进程能同时打开的最大文件数量。这是最常需要调整的项，比如数据库服务器就需要很高的值。 |
    | max user processes | -u | 7873 | 该用户能同时运行的最大进程数（包括线程）。 |
    | stack size | -s | 8192 | 进程栈的最大大小（KB）。如果程序递归太深可能导致栈溢出，有时需要调整此项。 |
    | core file size | -c | 0 | 核心转储文件（core dump）的最大大小。0 表示禁止生成 core dump 文件。用于程序崩溃调试。 |
    | virtual memory | -v | unlimited | 进程可使用的最大虚拟内存大小。unlimited 表示无限制。 |
    | file size | -f | unlimited | Shell 创建的文件的最大大小。 |

    ulimit 命令本身也可以用来修改限制, 修改通常只对 当前 Shell 会话 有效，退出后即失效。永久修改需要在用户配置文件（如 ~/.bashrc、~/.bash_profile）或系统级配置文件（如 /etc/security/limits.conf）中设置

    example:

    ```bash
    # 将“打开文件数”限制临时改为 2048
    ulimit -n 2048

    # 将“核心文件大小”限制改为无限制
    ulimit -c unlimited
    ```

* `sudo lsof /dev/shm/nccl-AoFK4o`

    用于查看正在使用特定 NCCL 共享内存文件的进程信息。

    lsof: "list open files" - 列出打开文件的工具

    输出列的含义：

        COMMAND: 使用该文件的进程名称（如 python3, train.py 等）

        PID: 进程 ID

        USER: 进程所有者

        FD: 文件描述符（mem 表示内存映射文件）

        TYPE: 文件类型（REG 表示常规文件）

        SIZE/OFF: 文件大小

        NODE: 文件节点号

        NAME: 文件名

* 如果 linux 系统里安装了 systemd，那么可以使用`journalctl -k`查看历史日志

    如果想把新增的日志写入文件，可以使用`dmesg --follow-new | tee <log_file>`

* `sudo mount -t tmpfs -o size=2G tmpfs /dev/shm`

    将 /dev/shm 目录重新挂载为大小为 2GB 的 tmpfs（临时文件系统）。

    这个操作会：

    * 覆盖现有的 /dev/shm 挂载

    * 之前存储在 /dev/shm 中的所有数据都会丢失

    * 新的大小限制会影响所有使用共享内存的应用程序

    默认的 /dev/shm 大小通常是系统物理内存的 50%。

* `df -T /dev/shm`

    显示 /dev/shm 目录所在文件系统的磁盘空间使用情况和文件系统类型信息。

    * df：disk free 的缩写，用于显示文件系统的磁盘空间使用情况

    * -T：选项，显示文件系统类型

    example output:

    ```
    Filesystem     Type  1K-blocks  Used Available Use% Mounted on
    tmpfs          tmpfs   8180620 55444   8125176   1% /dev/shm
    ```

    * 文件系统：tmpfs - 临时文件系统

    * 类型：tmpfs - 基于内存的临时文件系统

    * 1K-块：总容量（以 1KB 为单位）

* `ipcs -m`

    列出当前系统中所有进程间通信（IPC）的资源中， specifically 关于共享内存（Shared Memory） segments 的详细信息。

    ipcs: 是 “Inter-Process Communication Status” 的缩写，即“进程间通信状态”。这是一个用于报告 IPC 设施状态的工具。

    -m: 指定 ipcs 命令只显示与共享内存（Shared Memory） 相关的信息。如果不加任何选项，ipcs 默认会显示消息队列、共享内存和信号量所有三类信息。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ipcs -m

    ------ Shared Memory Segments --------
    key        shmid      owner      perms      bytes      nattch     status      
    0x00000000 2          hlc        600        524288     2          dest         
    0x00000000 7          hlc        606        7881216    2          dest         
    0x00000000 8          hlc        606        7881216    2          dest         
    0x00000000 34         hlc        600        524288     2          dest         
    0x00000000 32811      hlc        606        7881216    2          dest         
    0x00000000 32812      hlc        606        7881216    2          dest 
    ```

    KEY: 创建共享内存段时指定的键值（key），用于进程间找到同一个内存段。0x00000000 通常是私有用途。

    SHMID: 共享内存段的唯一标识符（ID）。在程序中使用这个 ID 来操作特定的内存段。

    OWNER: 创建该内存段的用户。

    PERMS: 权限位（类似文件权限），如 644 表示所有者可读写，组和其他用户只可读。

    BYTES: 该共享内存段的大小（字节）。

    NATTCH: 当前关联（attach）到这个内存段的进程数量。如果为 0，表示没有进程在使用它，但它可能仍然存在系统中。

    STATUS: 状态信息（在某些系统上可能显示更多细节，如被锁定的内存段）。

* `objdump -p <文件名> | grep NEEDED`

    -p 选项：代表显示文件头信息。

* elf dynamic section lookup table

    | 标签名 (Tag) | 十六进制值 (Hex) | 含义简述 |
    | - | - | - |
    | DT_NULL | 0x0 | 标记动态段的结束 |
    | DT_NEEDED | 0x1 | 所需共享库的名称（字符串表偏移） |
    | DT_PLTGOT | 0x3 | 全局偏移表（GOT）和/或过程链接表（PLT）的地址 |
    | DT_HASH | 0x4 | 符号哈希表的地址 |
    | DT_STRTAB | 0x5 | 字符串表的地址 |
    | DT_SYMTAB | 0x6 | 符号表的地址 |
    | DT_INIT | 0xC | 初始化函数的地址 |
    | DT_FINI | 0xD | 终止函数的地址 |
    | DT_SONAME | 0xE | 共享库自身的SONAME（字符串表偏移） |
    | DT_RPATH | 0xF | 库搜索路径（已过时，被DT_RUNPATH取代） |
    | DT_SYMBOLIC | 0x10 | 提示链接器从该库本身开始符号解析 |
    | DT_DEBUG | 0x15 | 用于调试（运行时地址由调试器填充） |
    | DT_TEXTREL | 0x16 | 存在代码段重定位，表明非PIC代码 |
    | DT_JMPREL | 0x17 | PLT重定位条目的地址 |
    | DT_RUNPATH | 0x1D | 库搜索路径 |
    | DT_GNU_HASH | 0x6FFFFEF5 | GNU扩展的哈希表样式 |
    | DT_INIT_ARRAY | 0x19 | 初始化函数指针数组的地址 |
    | DT_FINI_ARRAY | 0x1A | 终止函数指针数组的地址 |

* `objdump -p <文件名> | grep NEEDED`

    * -p 选项： 代表 --private-headers，用于显示文件格式中特定于该文件的“私有”头信息。对于 ELF 格式的文件（Linux 和大多数Unix-like系统上的标准格式），这个选项会显示出程序头表（Program Headers） 和动态段（Dynamic Section） 等关键信息。

    ldd 会实际尝试加载库并模拟运行，在某些不安全的情况下可能执行恶意代码。而 objdump -p | grep NEEDED 是静态分析，只读取文件头信息，因此更安全，尤其是在分析来源不可信的二进制文件时。

    其他常见的标签：

    * 核心依赖与加载相关

        * `NEEDED`

            该文件运行时所依赖的共享库的名称（如 libc.so.6）。一个文件可以有多个 NEEDED 条目。

        * `SONAME`(Shared Object Name)

            仅存在于共享库（.so 文件）中。它包含了该库的共享对象名称。链接器在链接时会把这个名字（而不是文件名）记录到最终的可执行文件中。这就是实现库版本兼容性的关键机制。

            例如，你有一个文件名为 libxyz.so.1.2.3 的库，但其 SONAME 可能是 libxyz.so.1。可执行文件在运行时寻找的将是 libxyz.so.1，而不是具体的 libxyz.so.1.2.3。

        * `RPATH` / `RUNPATH`

            包含一个用冒号分隔的目录列表。动态链接器在查找 NEEDED 库时，会优先在这些目录中搜索，然后再去默认的系统库路径（如 /lib, /usr/lib）中查找。

            RPATH 是较老的属性，其优先级很高。

            RUNPATH 是新标准，其优先级规则不同（在 LD_LIBRARY_PATH 之后查找）。

    * 符号解析相关

        * HASH / GNU_HASH: 指向一个符号哈希表。动态链接器使用这个表来快速查找函数和变量（符号）在库中的地址，极大地加快了动态链接的过程。GNU_HASH 是现代 Linux 系统上更优的格式。

        * STRTAB: 指向字符串表的地址。该表存储了所有动态链接所需的字符串，如符号名、库名等。

        * SYMTAB: 指向符号表的地址。该表包含了所有需要被动态链接的符号（函数名、变量名）的详细信息（名称、值、大小等）。链接器通常结合 HASH 和 STRTAB 来使用它。

        * PLT (Procedure Linkage Table) / PLTGOT (通常显示为 JMPREL): 指向重定位表的地址。这个表包含了所有需要延迟绑定（Lazy Binding）的函数引用信息。这是实现“第一次调用函数时才进行链接”机制的关键。

    * 初始化与终止相关

        * INIT: 指向初始化函数的地址。这个函数（通常命名为 _init）会在该共享库被加载到内存后、任何其他代码执行之前，由动态链接器自动调用。用于完成该库的全局构造和初始化工作。

        * FINI: 指向终止函数的地址。这个函数（通常命名为 _fini）会在该共享库从内存中卸载之前，由动态链接器自动调用。用于完成清理工作（如释放资源）。

            注意：现代代码更推荐使用 `__attribute__((constructor))` 和 `__attribute__((destructor))` 函数属性来代替直接使用 _init 和 _fini 节。

    * 其他重要标签

        * TEXTREL: 这是一个标志。如果存在，表明链接器需要修改代码段（.text段）的权限（例如将其设为可写）以便进行重定位。这通常意味着共享库不是用 -fPIC 选项编译的（位置无关代码），会带来安全性和性能上的损失。看到这个标志通常不是好事情。

        * FLAGS / FLAGS_1: 一些特殊的标志位。例如 FLAGS_1 中的 PIE 标志表示该可执行文件是位置无关的可执行文件（Position-Independent Executable），这是现代Linux系统上ASLR（地址空间布局随机化）的基础。

        * DEBUG: 这是一个占位符，用于调试信息，通常没有运行时语义。

    ref: 
    
    1. <https://docs.oracle.com/cd/E53394_01/html/E54813/chapter6-42444.html>

    1. `man 5 elf`

    1. <https://www.gnu.org/software/binutils/>

* `od -A`

    od -A 选项用于指定输出偏移量（地址）的显示格式。这里的 -A 代表 "Address"。

    * `-A x`: 十六进制（hexadecimal）
    
        `0000000`, `0000010`
    
    * `-A d`: 十进制（decimal）
    
        `0000000`, `0000016`

    * `-A o`: 八进制（octal）, 默认情况
    
        `0000000`, `0000020`

    * `-A n`: 不显示偏移量（none）
    
        （左侧偏移量栏为空）

* 创建`tmpfs`类型的目录

    ```bash
    sudo mkdir /mnt/shm
    sudo mount -t tmpfs -o size=2G tmpfs /mnt/shm
    ```

    解释：

    * `-t tmpfs`: 挂载`tmpfs`类型的文件系统。`tmpfs`表示使用 ram，如果 ram 不够，则使用 swap

    * `-o size=2G`：限制目录最大大小为 2G

    * `tmpfs`: 这是“源设备”参数。对于像`tmpfs`这样的虚拟文件系统，这个位置通常就填写文件系统类型本身。

    * `/mnt/shm`: 挂载点（mount point）

    特点：

    * 存储在内存中：所有存放在 /mnt/shm 目录下的文件和目录都位于高速的 RAM 中，因此读写速度非常快。

    * 临时性：这是一个临时存储。当系统重启、崩溃或你手动卸载（umount /mnt/shm）这个文件系统时，其中的所有数据都会消失。

    * 动态分配：tmpfs 只会实际占用它已存储数据大小的内存。例如，如果你创建了一个 100MB 的文件，tmpfs 就只占用约 100MB 的 RAM（和少量元数据开销），而不是一开始就占满 2GB。它会根据存储内容的增加而动态增长，但最大不会超过 size 参数的限制（2GB）。

    * 可能使用交换空间（Swap）：如果系统内存不足，tmpfs 中不活跃的数据可能会被换出到硬盘的交换分区（swap）上，从而释放物理内存。这意味着它的大小可以超过物理 RAM，但性能会下降。

* `sudo mount -o remount /dev/shm`

    重新挂载 /dev/shm 文件系统，并在此过程中应用或刷新其挂载选项。

    如果没有在命令中指定新的挂载选项（例如 size=2G 或 noexec），那么它将使用系统默认的或之前在 /etc/fstab 文件中配置的选项来重新挂载。

    常见使用场景

    * 应用 /etc/fstab 中的新配置：

        如果你修改了 /etc/fstab 中关于 /dev/shm 的配置（比如改变了大小限制 size=512M），你可以运行此命令来立即应用新的配置，而无需重启系统。

    * 修复权限或属性问题：

        如果 /dev/shm 的权限意外被更改（例如，某个脚本错误地执行了 chmod 700 /dev/shm），导致某些程序无法正常使用共享内存，通过重新挂载可以将其恢复为正确的默认权限。

    * 清除所有内容（不常用）：

        虽然 remount 本身不是为了清除数据，但结合某些选项（如 noexec 然后再 remount 回默认值）可以间接达到目的。更直接的方法是直接重启（数据会丢失）或手动删除其中的文件。

* `truncate -s <bytes>`

    如果文件不存在, 创建指定大小的空文件，内容全部用空字节（\0）填充。如果文件已存在, 若文件原大小 > 指定大小，则截断文件，丢弃超出部分；若文件原大小 < 指定大小，则扩展文件，用空字节填充新增部分。

    example:

    `truncate -s 4096 /dev/shm/my_tmp_file`

    在`/dev/shm`中创建大小为 4096 字节的文件`my_tmp_file`。

* `od -t x<N>`

    按`<N>`字节一组，打印十六进制数据。

    其中`<N>`可以取值 1, 2, 4, 8，如果不指定`<N>`，则默认取`2`。

    注意多字节显示时，输出受字节序的影响，比如单字节显示的`01 02`，使用小端序 + `-x2`显示时可能变成`0201`。

* `objdump`

    主要用于反汇编和分析目标文件及可执行文件。

    常用功能：

    1. 反汇编

        将二进制可执行文件或目标文件（.o, .exe, .so, .dll 等）中的机器代码转换回汇编语言代码。

        `objdump -d ./my_program`

        * `-d`选项表示反汇编包含指令的节（section）。

    2. 查看目标文件结构

        显示文件的头部信息和各个节（Section）的详细信息。包括文件的格式（如ELF、PE）、入口地址、节的大小和位置等。

        `objdump -h ./my_program`
        
        * `-h`选项显示节的头部摘要。

    3. 查看符号表

        列出文件中定义和引用的所有符号（如函数名、全局变量名）。

        `objdump -t ./my_program`

        * `-t`选项显示符号表。

    4. 查看文件头信息

        显示二进制文件的元数据，例如目标架构（如x86-64、ARM）、操作系统ABI、文件类型（可执行、共享库等）和入口点地址。

        `objdump -f ./my_program`

        * `-f`选项显示文件头信息。

    5. 以十六进制格式查看文件内容

        除了反汇编，objdump 还可以直接显示文件的十六进制和ASCII表示，类似于 hexdump 或 xxd 命令。

        `objdump -s -j .text ./my_program`

        * `-s` 显示所有节的内容。

        * `-j` 指定只显示某个节（如 .text 节）的内容。

    6. 查看动态链接信息

        对于动态链接的可执行文件或共享库，可以显示其依赖的共享库（如Linux下的 .so 文件）以及动态符号表。

        `objdump -p ./my_program`

        * `-p`显示与动态链接相关的信息（在 ELF 文件中，这类似于`readelf -d`命令）。

* `readelf -l /bin/bash | grep interpreter`

    查找 /bin/bash 可执行文件所使用的动态链接器（interpreter）的路径。

    output:
    
    ```
          [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
    ```

    这表示：

    * `/bin/bash`依赖于动态链接器 /lib64/ld-linux-x86-64.so.2 来加载运行所需的共享库（如 libc.so）。

    * 系统内核在执行 /bin/bash 时，会先加载这个指定的动态链接器，再由它处理后续的库依赖和符号解析。

* `readelf`的用法

    用于显示关于 ELF (Executable and Linkable Format) 格式目标文件的信息。

    ELF 是现代 Linux 系统上的一种文件格式，用于：

    * 可执行文件 (例如编译生成的`a.out`)

    * 共享库 (例如：libc.so.6)

    * 目标文件 (例如：file.o)

    * 核心转储文件 (core dumps)

    常用选项：

    * 查看文件头 (`-h`)

        这是最常用的选项之一。它显示了 ELF 文件的概要信息，包括：

        * 文件类型（可执行文件、共享库、目标文件等）

        * 目标机器的体系结构（如 x86-64, ARM）

        * 程序的入口点地址（Entry point）

        * 程序头表（Program Headers）和节头表（Section Headers）的起始位置和大小。

    * 查看节头信息 (`-S`)

        显示文件中所有的 “节” (Sections) 的信息。节是 ELF 文件的重要组成部分，例如：

        * `.text`： 存放可执行代码。

        * `.data`： 存放已初始化的全局变量和静态变量。

        * `.bss`： 存放未初始化的全局变量和静态变量。

        * `.rodata`： 存放只读数据（如字符串常量）。

        * `.symtab`： 符号表。

        * `.strtab`： 字符串表。

    * 查看程序头信息 (`-l`)

        显示 “段” (Segments) 或称为 程序头 (Program Headers) 的信息。段告诉操作系统或动态链接器如何将文件加载到内存中并执行。这对于理解程序运行时布局至关重要。

    * 查看符号表 (`-s`)

        显示文件中定义和引用的所有符号，如函数名、变量名。这对于解决“未定义引用”等链接错误非常有用。

    * 查看动态段信息 (`-d`)

        对于动态链接的可执行文件或共享库，此选项显示其依赖的共享库（如 libc.so.6）以及动态链接器需要的其他信息（如重定位信息、符号表地址等）。

    * 查看重定位信息 (`-r`)

        显示文件中需要重定位的条目信息，这在分析目标文件(.o)时尤其有用。

    * 查看节的内容 (`-x` 或 `-p`)

        以十六进制或其他格式转储指定节的具体内容。

    example:

    `readelf -h main`

    output:

    ```
    ELF Header:
      Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
      Class:                             ELF64
      Data:                              2's complement, little endian
      Version:                           1 (current)
      OS/ABI:                            UNIX - System V
      ABI Version:                       0
      Type:                              DYN (Position-Independent Executable file)
      Machine:                           Advanced Micro Devices X86-64
      Version:                           0x1
      Entry point address:               0x1060
      Start of program headers:          64 (bytes into file)
      Start of section headers:          14048 (bytes into file)
      Flags:                             0x0
      Size of this header:               64 (bytes)
      Size of program headers:           56 (bytes)
      Number of program headers:         13
      Size of section headers:           64 (bytes)
      Number of section headers:         31
      Section header string table index: 30
    ```

* `at`

    安排一个任务在未来的某个特定时间点执行一次。

    ubuntu 系统中默认不安装`at`，需要手动安装：`sudo apt install at`

    syntax:

    ```bash
    at [选项] <时间设定>
    ```

    常用选项：

    * `-f <文件>`：从指定的文件中读取要执行的命令，而不是交互式输入。

        示例： `at -f /path/to/script.sh 10:00 AM`

    * `-l`：列出当前用户的所有待处理作业（等同于 atq 命令）。

        示例： at -l

    * `-r <作业ID>`或`-d <作业ID>`：删除一个指定的待处理作业（等同于 atrm 命令）。

        示例： at -r 5 (删除作业ID为5的任务)

    * `-m`：即使命令没有输出，也在任务完成后给用户发送邮件。

    * `-M`：即使命令有输出，也不给用户发送邮件。

    * `-t <时间>`：使用`[[CC]YY]MMDDhhmm[.ss]`格式指定时间（较不常用）。

        示例： `at -t 202507191430.00 (2025年7月19日14点30分00秒)`

    * `-v`：显示任务将被执行的时间，而不是仅仅将任务排入队列。

    * 时间设定

        * 简单时间格式：

            HH:MM - 在今天的指定时间执行。如果该时间已过，则顺延到明天。

                示例： at 14:30

            midnight, noon, teatime (通常是下午4点) - 使用预定义的单词。

        * 相对时间格式：

            now + <数量> <时间单位> - 从现在开始的一段时间后执行。

                时间单位： minutes, hours, days, weeks

                示例：

                    at now + 10 minutes (10分钟后)

                    at now + 2 hours (2小时后)

                    at now + 1 week (1周后)

        * 绝对时间格式：

            MMDDYY, MM/DD/YY, DD.MM.YY - 指定具体日期。

                示例：

                    at 11:00 PM 072025 (2025年7月20日晚上11点)

                    at 10am Jul 20 (7月20日上午10点)

                    at 2:00 tomorrow (明天凌晨2点)

        * 组合日期和时间：

            你可以将日期和时间组合起来。

                示例： at 3:30 PM next Friday (下周五下午3点半)

    example:

    * `at now + 10 minutes` (10分钟后)

    * `at 11:00 PM` (今晚11点)

    * `at 2:00 tomorrow` (明天凌晨2点)

    * `at 10:00 Jul 20` (7月20日上午10点)

    标准输入/交互式输入：通常你直接运行 at <TIME>，然后在其提示符下输入要执行的命令，按 Ctrl+D 结束输入。

    从文件读取：也可以使用 at -f script.sh <TIME> 来指定一个脚本文件在特定时间运行。

    examples:

    * 安排一个简单任务

        ```bash
        # 安排一个任务在下午2：30执行
        at 2:30 PM
        # 进入at提示符后，输入命令：
        at> echo "Hello from the past!" > ~/reminder.txt
        # 输入完毕后，按 Ctrl+D 来保存任务
        ```

    * 安排一个不久后的任务

        ```bash
        # 安排一个20分钟后的任务
        at now + 20 minutes
        at> shutdown -h now # 例如：20分钟后关机
        at> <EOT> # 按 Ctrl+D
        ```

    * 非交互模式

        ```bash
        # 将一个脚本文件中的命令安排在下午5点执行
        $ at -f /path/to/daily_cleanup.sh 5:00 PM
        job 7 at Tue Jul 18 17:00:00 2025
        ```

    * 使用管道（echo）

        ```bash
        # 通过管道将命令传递给at
        $ echo "shutdown -h now" | at now + 30 minutes
        job 8 at Tue Jul 18 15:45:00 2025
        ```

    `atq`：列出当前用户所有等待执行的 at 任务队列（root用户可以看到所有任务）。

    `atrm <job_id>`：删除一个已排队的 at 任务。例如 atrm 3 会删除作业编号为3的任务。

    注意事项:

    * 守护进程：at 命令的运行依赖于 atd 这个守护进程。请确保它正在运行（通常通过 systemctl status atd 检查）。

    * 执行环境：at 任务会在一个几乎最小化的 shell 环境中运行，只继承少量环境变量（如TERM, PATH, SHELL等）。如果你的命令依赖于特定的环境变量（如DISPLAY用于图形界面），可能需要在其内部重新设置。

    * 输出处理：任务的标准输出和标准错误默认会通过电子邮件发送给任务所有者（即安排任务的用户）。为了避免收邮件，最好在命令中明确重定向输出（如 >/dev/null 2>&1 或重定向到日志文件）。

    `at`命令中的时间指示符（如 AM, PM, tomorrow, midnight 等）是大小写不敏感的（case insensitive）。

    在交互环境中输入命令时，无法使用 tab 自动补全（比如文件名，目录下的文件等）。

    `at`指定的 command 执行后，无法随时中止，必须等 command 结束。

* `nsupdate`

    用于动态、增量地更新 DNS 域名。它允许你在不手动编辑 zone file（区域文件）和重启 DNS 服务的情况下，直接添加、修改或删除 DNS 记录。

    主要功能和用途

    * 动态 DNS (DDNS)

        这是 nsupdate 最经典的用途。很多家庭宽带或办公网络的公网 IP 地址是动态变化的。你可以在路由器或电脑上运行一个脚本，当检测到 IP 变化时，自动使用 nsupdate 命令向 DNS 服务器发送更新请求，将你的域名（如 home.example.com）快速指向新的 IP 地址。这样你始终可以通过一个固定的域名访问到动态 IP 的设备。

    * 自动化运维和脚本集成

        在自动化部署（CI/CD）、云基础设施管理中，经常需要批量创建或销毁服务器。这些流程可以通过脚本调用 nsupdate，自动为新服务器注册 DNS 记录，或者在下线时清理记录，实现全自动化管理。

    * 快速故障恢复

        如果需要将服务从一个 IP 迁移到另一个 IP，使用 nsupdate 可以几乎实时地更新 DNS 记录，大大缩短故障切换时间（RTO），比手动修改zone file并等待复制要快得多。

    工作原理

    nsupdate 并不直接操作 DNS 服务器上的配置文件。它的工作流程是：

    1. 连接：nsupdate 会连接到目标 DNS 服务器（通常是 BIND 9）的 53 端口，并使用 TSIG（Transaction SIGnature） 密钥进行身份验证。TSIG 确保了更新的安全性和合法性，防止任何人随意修改你的 DNS 记录。

    2. 发送更新指令：在交互式命令行或通过管道输入中，你发送一系列指令，例如：

        * `update add www.example.com 3600 A 192.0.2.1` （添加一条 A 记录）

        * `update delete oldhost.example.com A` （删除一条 A 记录）

    3. 服务器处理：DNS 服务器验证你的权限后，会在内存中直接更新它的 zone 数据，并递增该区域的序列号（SOA Serial）。这个更改会立即生效，并通知其他从服务器进行区域传输（AXFR/IXFR）以同步更新。

    example:

    假设我们有一个密钥key文件，用来向DNS服务器 dns.example.com 认证，并更新 example.com 域。

    ```bash
    # 启动 nsupdate 并指定服务器
    nsupdate -k /path/to/mykey.key

    # 在出现的交互提示符下输入命令
    > server dns.example.com
    > zone example.com
    > update add newserver.example.com 300 A 203.0.113.10
    > send
    > quit
    ```

    这条命令成功执行后，域名 newserver.example.com 就会立刻指向 203.0.113.10。

* `od -t x1`

    以十六进制（HEX）字节的形式，逐个字节地显示文件或输入流的内容。

    * `-t` (`--format`) : 用于指定输出数据的格式。它告诉 od 如何解释和显示文件中的字节。

    * `x1`: `x`表示 16 进制，`1`代表每个输出单元的大小是 1 个字节。

    example:

    ```
    0000000 7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
    0000020 03 00 3e 00 01 00 00 00 10 0a 00 00 00 00 00 00
    0000040 40 00 00 00 00 00 00 00 38 4e 00 00 00 00 00 00
    ...
    ```

* `/etc/ld.so.conf.d/myapp.conf`

    将一个自定义的库路径（例如`/opt/myapp/lib`）添加到系统中所有应用程序的共享库搜索路径中。`ld-linux.so`会搜索`/etc/ld.so.conf.d`这个目录下的所有配置。

    example:

    ```conf
    /opt/myapp/lib
    ```

    创建或修改配置文件后，必须运行 ldconfig 命令来重建共享库缓存，使更改立即生效:

    ```bash
    sudo ldconfig
    ```

    这个命令会读取所有`/etc/ld.so.conf.d/`下的配置文件和`/etc/ld.so.conf`，生成一个快速的缓存文件`/etc/ld.so.cache`。动态链接器实际使用的是这个缓存文件来加速查找。

    主配置文件 /etc/ld.so.conf 通常会包含一行：`include /etc/ld.so.conf.d/*.conf`。

* `host`

    一个 DNS 查询工具

    常见用法:

    * 查询域名对应的 IP

        `host www.example.com`

        输出该域名的 A/AAAA 记录。

        example:

        ```bash
        host www.google.com
        ```

        output:

        ```
        www.google.com has address 142.250.71.196
        www.google.com has IPv6 address 2404:6800:4005:816::2004
        ```

    * 查询 IP 对应的域名（反向解析）

        `host 8.8.8.8`

        输出 PTR 记录。

        example:

        ```bash
        host 8.8.8.8
        ```

        output:

        ```
        8.8.8.8.in-addr.arpa domain name pointer dns.google.
        ```

    * 指定查询记录类型

        ```bash
        host -t MX example.com   # 查询邮件服务器记录
        host -t NS example.com   # 查询域名服务器
        ```

    * 指定 DNS 服务器

        ```bash
        host www.example.com 8.8.8.8
        ```

        使用 Google DNS 来解析。

        (如果想取消这个映射，该怎么办？)

* `command_1 && command_2 &`里，`&`作用于`(command_1 && command_2)`。

 * `pkg-config --libs`

    获取链接一个特定软件库（Library）时，需要传递给编译器的所有链接器（Linker） flags（参数）。

    当你编译程序需要用到某个第三方库（比如 OpenCV、GTK、libpng 等）时，这个命令会告诉你应该加哪些 -l 和 -L 参数。

    example:

    ```bash
    pkg-config --libs mpi
    ```

    output:

    ```
    -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
    ```

    pkg-config 工具通过查询事先安装好的 .pc 配置文件（这些文件通常随库一起安装），自动为你生成正确的、适用于当前系统的编译参数。

    通常与 Makefile 结合起来使用：

    ```makefile
    # 编译 main.c 并链接 libcurl 库
    program: main.c
        gcc main.c -o program `pkg-config --libs --cflags libcurl`
    ```

    其中`--cflags`表示输出`-I<header_path>`

    * `pkg-config --libs <包名>`: 获取链接库所需的参数
    
        `-L/usr/lib -lcurl -lssl -lcrypto`

    * `pkg-config --cflags <包名>`: 获取编译时所需的参数（如头文件路径）
    
        `-I/usr/include/curl`

* `ldconfig`

    创建、更新和维护共享库的缓存和必要的链接，通过生成缓存文件 /etc/ld.so.cache 来加速程序启动时寻找共享库的过程，并管理共享库的符号链接，以便系统能够快速地找到并加载运行程序时所需的共享库（.so 文件）

    当你在Linux系统上运行一个程序时，动态链接器（ld.so 或 ld-linux.so）负责在程序启动前将它所依赖的所有共享库加载到内存中。

    动态链接器不会每次都去慢吞吞地扫描整个硬盘上的所有目录来寻找库文件，那样太慢了。相反，它直接去查询由 ldconfig 预先准备好的缓存文件 /etc/ld.so.cache。这个缓存就像一本书的索引，让动态链接器能瞬间找到所需的库。

    在执行了以下操作后，需要手动执行`sudo ldconfig`:

    * 安装了新的共享库（例如通过 make install 或手动拷贝 .so 文件到系统目录）。

    * 删除了已有的共享库。

    * 修改了共享库的搜索路径（例如编辑了 /etc/ld.so.conf 或在该目录下增加了新的 .conf 文件）。

    apt 等软件安装了新的包后，会自动调用`ldconfig`。

    更新缓存: `sudo ldconfig`

    显示缓存内容: `ldconfig -p`

* `ldd`

    List Dynamic Dependencies

    主要功能：

    * 列出依赖：显示运行某个程序需要哪些共享库（.so 文件）。

    * 定位库文件：显示这些依赖库在文件系统中的具体路径。

    * 检查兼容性：检查是否存在某个依赖库，以及库的版本是否兼容。如果某个库找不到，它会显示 not found，这通常是程序无法启动的原因。

    syntax:

    ```bash
    ldd [选项] <文件名>
    ```

    ldd 本身并不是一个可执行程序，而是一个Shell脚本

    它的工作原理是设置适当的环境变量（如 LD_TRACE_LOADED_OBJECTS），然后调用目标程序。目标程序在这种特殊环境下启动时，并不会真正运行其主逻辑，而是会列出其所有依赖的共享库信息后退出。

* `vncviewer`

    使用方法：

    * `vncviewer 192.168.1.100`

    * `vncviewer 192.168.1.100:5901`

* dig 命令包含在`bind-utils`软件包中

* 查看 glibc 版本：`ldd --version`

* `od -c <file>`

    以 字符形式（character）显示文件内容，不可打印字符会用转义符（如`\n`, `\t`, `\0`）或八进制表示。

* `od -x`

    od: 是 Octal Dump 的缩写。这个名字源于其最初的主要功能是以八进制（Octal）格式显示文件内容。虽然现在它支持多种格式，但名字保留了下来。

    以十六进制（Hex）格式显示文件的内容.

    `-x`代表输出为“十六进制。
    
    example:

    `od -x msg.txt`

    output:

    ```
    0000000 6568 6c6c 2c6f 7720 726f 646c 202c 696e
    0000020 6168 2c6f 7a20 6961 696a 6e61 202c 6568
    0000040 6568 202c 6168 6168 000a
    0000051
    ```

    * 最左边的一列: 偏移量地址. 表示当前行数据在文件中的起始位置（偏移量），默认以八进制数显示。

    * 中间的多列: 数据内容. -x 选项规定每2个字节（16位）为一组进行显示。
    
        由于计算机的字节序（Endianness）问题，od -x 在显示时使用的是主机本身的字节序（对于大多数x86架构的电脑是小端序）。

        如果文件中连续的2个字节是 0x61 和 0x62（即字符 'a' 和 'b' 的ASCII码），在小端序机器上，od -x 会将其显示为一组：6261（即 0x62 在高位，0x61 在低位）。

    * 如果使用 -x 的同时再加上 -c 选项（即 od -xc），输出还会在最右边增加一列，显示数据对应的ASCII字符。不可打印的字符会显示为转义序列（如 \n）或问号（?）。

        比较神奇的是 ascii 字符会按 2 个字节的间隔自动把顺序倒过来：

        `od -xc msg.txt`:

        ```
        0000000    6568    6c6c    2c6f    7720    726f    646c    202c    696e
                  h   e   l   l   o   ,       w   o   r   l   d   ,       n   i
        0000020    6168    2c6f    7a20    6961    696a    6e61    202c    6568
                  h   a   o   ,       z   a   i   j   i   a   n   ,       h   e
        0000040    6568    202c    6168    6168    000a
                  h   e   ,       h   a   h   a  \n
        0000051
        ```

        可以看到，本来`e = 0x65`，`h = 0x68`，但是在显示的时候 od 自动把顺序纠正了。

* `ps aux`

    ps: 进程状态（Process Status）

    * `a`： 显示所有用户的进程（而不仅仅是当前用户的）。

    * `u`： 以面向用户的格式显示，这会提供更详细的信息（如 CPU、内存占用率、用户等）。

    * `x`： 列出没有控制终端的进程。这很重要，因为很多系统守护进程（后台服务）是不依赖于终端的。加上 x 才能看到所有这些后台进程。

    输出的关键列的含义：

    | 列名 |全称 | 含义 |
    | - | - | - |
    | `USER` | User | 进程的所有者（是哪个用户启动的） |
    | `PID` | Process ID | 进程的唯一ID号，用于识别和管理进程 |
    | `%CPU` | CPU Percentage | 进程占用CPU的百分比 |
    | `%MEM` | Memory Percentage | 进程占用物理内存的百分比 |
    | `VSZ` | Virtual Set Size | 进程使用的虚拟内存大小（单位：KB） |
    | `RSS` | Resident Set Size | 进程使用的、未被换出的物理内存大小（单位：KB） |
    | `TTY` | Teletypewriter | 进程是在哪个终端上运行的。? 表示不是从终端启动的。 |
    | `STAT` | Process State | 进程状态码（非常重要），例如：<br>- `R`： 正在运行或可运行<br>- `S`： 可中断的睡眠状态（等待事件完成）<br>- `D`： 不可中断的睡眠（通常与IO有关）<br>- `Z`： 僵尸进程（已终止但未被父进程回收）<br>- `T`： 已停止（通常由信号控制） |
    | `START` | Start Time | 进程启动的时间 |
    | `TIME` | CPU Time | 进程实际使用CPU运行的总时间 |
    | `COMMAND` | Command | 启动该进程所用的命令行名称 |

* `timeout`

    如果计时器超时, ，timeout 会向子命令进程发送一个信号（默认是 TERM 信号），强制终止它, 并返回一个特定的退出状态码（默认为124）

    syntax:

    ```
    timeout [OPTIONS] DURATION COMMAND [ARG]...
    ```

    * DURATION：时间长度。这是必须指定的参数。它可以是一个整数，也可以是带单位的数字（例如 10s 表示10秒，2m 表示2分钟，1h 表示1小时）。如果只是数字，默认单位是秒。

    * COMMAND [ARG]...：你想要运行的实际命令及其参数。

    常用选项（OPTIONS）：

    * -s, --signal=SIGNAL：指定超时后要发送的信号。默认是 TERM (15)。

        例如，如果程序忽略 TERM 信号，你可以使用 -s KILL 来发送 KILL (9) 信号，强制杀死进程。

        查看所有信号：kill -l

    * -k, --kill-after=DURATION：双重保险。先发送 -s 指定的信号（默认TERM），如果再过 DURATION 时间后进程仍然存在，则发送 KILL 信号确保其被杀死。

    * --preserve-status：让 timeout 返回被它控制的命令的退出状态码。如果命令超时被杀死，仍然返回124。

    example:

    ```bash
    timeout 5s ping example.com
    timeout -s KILL 2m some-unstable-script.sh
    timeout -k 5s 10s some-command
    ```

    在脚本中使用并检查退出状态：

    ```bash
    #!/bin/bash
    timeout 30s long-running-task.sh

    case $? in
      0)  echo "Task completed successfully." ;;
      124) echo "Task timed out and was killed." ;;
      137) echo "Task was killed by a signal (e.g., KILL)." ;; # 如果用了 -s KILL，会返回137 (128+9)
      *)  echo "Task failed with exit code: $?" ;;
    esac
    ```

* `sync`

    `sync`可以作为一个命令使用，效果和调用`sync()`相同。

* `od`命令

    od（Octal Dump），用于以各种格式显示文件的内容，通常用于查看或诊断文件中那些不可打印的字符（如控制字符、换行符、空字符等）。

    * `od -c`: 将文件的每个字节（byte）解释为 ASCII 字符或转义序列，并以更可读的形式输出。

    example:

    `msg.txt`:

    ```
    hello, world
    nihao
    zaijian
    ```

    `od -c msg.txt` output:

    ```
    0000000   h   e   l   l   o   ,       w   o   r   l   d  \n   n   i   h
    0000020   a   o  \n   z   a   i   j   i   a   n  \n
    0000033
    ```

    前面的偏移是 8 进制。

* strace

    strace 可以记录一个进程在运行时与内核之间的所有系统调用（system calls）和接收到的信号（signals）

    example:

    ```bash
    strace ls
    ```

    常用方式（未验证）：

    * `strace <command>`：跟踪一个命令的执行。

    * `strace -p <pid>`：附着到一个正在运行的进程上进行跟踪。

    * `strace -e trace=<type>`：只跟踪特定类型的系统调用，如 strace -e trace=open,read ls 只跟踪 open 和 read 调用。

    * `strace -o output.txt`：将输出重定向到文件，便于分析。

    * `strace -c`：在程序结束后生成一个统计报告，显示各个系统调用的次数、时间和错误。

* `ss`

    Socket Statistics，可以查看当前的 tcp, udp 连接。

    ss 直接从内核空间获取信息，而 netstat 需要遍历许多 /proc 下的文件，效率较低。（未验证）

    netstat 在很多 Linux 发行版中已被标记为“已废弃”，官方推荐使用 ss 和 ip 命令（未验证）

    常用选项：

    * `-t`: 显示 tcp 连接

    * `-u`：显示 udp 连接

    * `-l`：显示 listening socket

    * `-a`：显示所有连接，包括监听和非监听的

    * `-p`：显示对应的进程

    * `-n`：以数字形式显示端口和地址，不尝试解析域名和服务名

    常用组合：

    `ss -tln`

    output:

    ```
    State   Recv-Q   Send-Q     Local Address:Port      Peer Address:Port  Process  
    LISTEN  0        64               0.0.0.0:43673          0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:35661          0.0.0.0:*              
    LISTEN  0        64               0.0.0.0:2049           0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:52865          0.0.0.0:*              
    LISTEN  0        128              0.0.0.0:22             0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:111            0.0.0.0:*              
    LISTEN  0        128            127.0.0.1:631            0.0.0.0:*              
    LISTEN  0        4096           127.0.0.1:39255          0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:45865          0.0.0.0:*              
    LISTEN  0        4096       127.0.0.53%lo:53             0.0.0.0:*              
    LISTEN  0        4096             0.0.0.0:46147          0.0.0.0:*              
    LISTEN  0        32         192.168.122.1:53             0.0.0.0:*              
    LISTEN  0        64                  [::]:2049              [::]:*              
    LISTEN  0        128                [::1]:631               [::]:*              
    LISTEN  0        64                  [::]:44879             [::]:*              
    LISTEN  0        128                 [::]:22                [::]:*              
    LISTEN  0        4096                [::]:111               [::]:*              
    LISTEN  0        4096                [::]:55635             [::]:*              
    LISTEN  0        4096                [::]:54891             [::]:*              
    LISTEN  0        4096                [::]:38721             [::]:*              
    LISTEN  0        4096                [::]:54351             [::]:*
    ```

    过滤连接（未验证）：

    * 查看所有已建立的 TCP 连接：

        `ss -t state established`

    * 查看所有处于 TIME-WAIT 状态的连接（通常是大量短连接后出现的状态）：

        `ss -t state time-wait`

    * 查看所有已建立连接的 IPv4 HTTP (端口 80) 连接：

        `ss -t '( dport = :http or sport = :http )'`

    * 或者使用数字端口：

        `ss -t '( dport = :80 or sport = :80 )'`

* `wc -l [file_1] [file_2] [file_3] ...`

    统计给定文件或输入流中的行数，即统计`\n`的数量。

    word count

    * `-w`: 统计单词数

    * `-c`：统计字节数

    * `-m`：统计字符数

* `read -n1`

    从标准输入中读取一个（且仅一个）字符，然后立即结束读取，无需用户按回车键

    注意这里`-n`后跟的是一，不是`l`。相当于`read -n 1`。

    `-n N`: 读取 N 个字符后立即返回，不再等待回车键

* `ed`

    `ed`是个很老的文本编辑器。

    example:

    `test.txt`:

    ```
    Hello world
    This is a test.
    Goodbye
    ```

    我们想将 “world” 替换为 “universe”。

    ```
    # 1. 用 ed 打开文件
    ed test.txt

    # 2. 此时 ed 在命令模式，没有任何提示。我们先显示所有内容（,p 表示从第一行到最后一行打印）
    ,p
    Hello world
    This is a test.
    Goodbye

    # 3. 使用替换命令。s/旧字符串/新字符串/
    # 首先搜索包含 ‘world’ 的行（/pattern/），然后对其执行替换命令
    /world/ s/world/universe/

    # 4. 再次查看修改后的内容（‘’代表当前行）
    p
    Hello universe

    # 5. 保存并退出
    w
    q
    ```

    example:

    `echo $'#!/bin/ed\n,w\nq' | ed test.txt > /dev/null`

    向 ed 发送命令 `,w`（写入文件）和 `q`（退出）

* `ip route get <dst_ip>`可以显示访问`dst_ip`是从本机的哪个路由表出去

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ip route get 223.5.5.5
    223.5.5.5 via 10.0.2.1 dev enp0s3 src 10.0.2.4 uid 1000 
        cache
    ```

    其中`via 10.0.2.1`表示网关（下一跳的地址），`dev enp0s3`表示使用的网卡设备，`src 10.0.2.4`表示源地址。

    `ip route get <dst_ip> from <src_ip>`可以指定源地址。

    ip route get 仅本地查询，不发送真实数据包。

* `ethtool`

    install: `sudo apt install ethtool`

    * 查看网卡基本信息: `ethtool <网卡名>`
    
        `ethtool enp0s3`

        output:

        ```
        Settings for enp0s3:
        	Supported ports: [ TP ]
        	Supported link modes:   10baseT/Half 10baseT/Full
        	                        100baseT/Half 100baseT/Full
        	                        1000baseT/Full
        	Supported pause frame use: No
        	Supports auto-negotiation: Yes
        	Supported FEC modes: Not reported
        	Advertised link modes:  10baseT/Half 10baseT/Full
        	                        100baseT/Half 100baseT/Full
        	                        1000baseT/Full
        	Advertised pause frame use: No
        	Advertised auto-negotiation: Yes
        	Advertised FEC modes: Not reported
        	Speed: 1000Mb/s
        	Duplex: Full
        	Auto-negotiation: on
        	Port: Twisted Pair
        	PHYAD: 0
        	Transceiver: internal
        	MDI-X: off (auto)
        netlink error: Operation not permitted
                Current message level: 0x00000007 (7)
                                       drv probe link
        	Link detected: yes
        ```

    * 查看驱动信息: `ethtool -i <网卡名>`

        `ethtool -i enp0s3`

        output:

        ```
        driver: e1000
        version: 6.8.0-65-generic
        firmware-version: 
        expansion-rom-version: 
        bus-info: 0000:00:03.0
        supports-statistics: yes
        supports-test: yes
        supports-eeprom-access: yes
        supports-register-dump: yes
        supports-priv-flags: no
        ```

    * 查看统计信息: `ethtool -S <网卡名>`

        `ethtool -S enp0s3`

        output:

        ```
        NIC statistics:
             rx_packets: 7599541
             tx_packets: 3763596
             rx_bytes: 7471543776
             tx_bytes: 5835620961
             rx_broadcast: 40
             tx_broadcast: 6
             rx_multicast: 0
             tx_multicast: 828
             rx_errors: 0
             tx_errors: 0
             tx_dropped: 0
             multicast: 0
             collisions: 0
             rx_length_errors: 0
             rx_over_errors: 0
             rx_crc_errors: 0
             rx_frame_errors: 0
             rx_no_buffer_count: 0
             rx_missed_errors: 0
             tx_aborted_errors: 0
             tx_carrier_errors: 0
             tx_fifo_errors: 0
             tx_heartbeat_errors: 0
             tx_window_errors: 0
             tx_abort_late_coll: 0
             tx_deferred_ok: 0
             tx_single_coll_ok: 0
             tx_multi_coll_ok: 0
             tx_timeout_count: 0
             tx_restart_queue: 0
             rx_long_length_errors: 0
             rx_short_length_errors: 0
             rx_align_errors: 0
             tx_tcp_seg_good: 1456528
             tx_tcp_seg_failed: 0
             rx_flow_control_xon: 0
             rx_flow_control_xoff: 0
             tx_flow_control_xon: 0
             tx_flow_control_xoff: 0
             rx_long_byte_count: 7471543776
             rx_csum_offload_good: 0
             rx_csum_offload_errors: 0
             alloc_rx_buff_failed: 0
             tx_smbus: 0
             rx_smbus: 0
             dropped_smbus: 0
        ```

    其他还有些功能，目前看上去用处不大。如果专业做网卡这块了再去了解。

* glob 匹配

    * 匹配任意字符（包括空字符）。

    ? 匹配单个字符。

    [abc] 匹配 a、b 或 c。

* `find`的匹配模式

    -name 和 -iname： 使用 glob 模式，按文件名匹配。`-iname`表示大小写不敏感。

    -regex 和 -iregex：使用正则表达式。正则匹配的是完整路径（如 ./dir/file.txt），而非仅文件名。

    其他匹配方式:

    * -path：类似 -name，但匹配完整路径（使用 glob 语法）。

    * -perm、-size 等：按权限、大小等属性匹配，与正则/glob 无关。

* `stat <file_path>`

    输出文件的基本信息。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ stat xml_log_4_gpu.txt
      File: xml_log_4_gpu.txt
      Size: 7311      	Blocks: 16         IO Block: 4096   regular file
    Device: 802h/2050d	Inode: 938640      Links: 1
    Access: (0664/-rw-rw-r--)  Uid: ( 1000/     hlc)   Gid: ( 1000/     hlc)
    Access: 2025-08-03 09:42:13.930462794 +0800
    Modify: 2025-06-06 09:55:17.893493034 +0800
    Change: 2025-06-06 09:55:17.893493034 +0800
     Birth: 2025-06-05 16:56:25.016703429 +0800
    ```

* `netstat`

    `netstat -a`: 显示所有连接

    `netstat -s`: 显示统计摘要

    `netstat -r`: 查看系统的路由表信息

    `netstat -i`: 显示网络接口的配置和流量统计

    `netstat -tuln`: 列出所有处于监听（LISTEN）状态的端口

    `netstat -tulnp`: 查看占用端口的进程ID（PID）和程序名（需管理员权限）

    常用参数：

    * `-a`: 显示所有连接和监听端口。

    * `-n`: 以数字形式显示地址和端口（禁用DNS解析）。

    * `-t/-u`: 仅显示TCP/UDP连接。

    * `-p`: 显示进程信息（Linux）。

    * `-o`: 显示进程ID（Windows）。

    * `-r`: 显示路由表。

    * `-s`: 显示协议统计信息。

* `disown`

    `disown`将指定作业从 Shell 的作业列表中删除，但进程仍继续运行。

    常见用法：

    ```bash
    disown <jobspec>      # 移除指定作业（如 %1）
    disown -a             # 移除所有作业
    disown -r             # 仅移除运行中的作业
    disown -h             # 将任务保留在 job list 但对其屏蔽 SIGHUP 信号（推荐用法）
    ```

    启动后台任务后使用 disown -h，即使退出 Shell 也不终止进程（类似 nohup 效果）。

    disown 不会自动重定向输出

    被 disown 的进程仍属于当前用户，但会变成孤儿进程（由 init/systemd 接管）

    example:

    ```
    $ long_task &       # 启动后台任务
    $ disown -h %1      # 屏蔽 SIGHUP 并移除作业
    $ exit              # 退出 Shell，任务继续运行
    ```

    被 disown 解除绑定的任务，无法使用`kill <PID>`的方式结束（为什么？），但可以通过`kill -kill <PID>`结束任务。

* 符号链接本身的大小是其指向的路径字符串的长度

    example:

    ```
    lrwxrwxrwx 1 hlc hlc    7  8月 17 13:34 msg_link.txt -> msg.txt
    ```

* `git-credential-libsecret`可以将 git 凭据存储在`libsecret`密钥管理服务中。

    需要系统安装 libsecret，还需要 git 去配置，略显复杂。有需求了再看。

* `less`

    less 用于分页查看。

    syntax:

    ```bash
    less [选项] 文件名
    ```

    常用选项：

        -N：显示行号。

        -i：忽略搜索时的大小写（除非搜索词包含大写字母）。

        -F：若文件可一屏显示，直接退出（类似 cat）。

        -S：禁用自动换行（超长行需左右滚动查看）。

        `less +F growing_file.log`: 类似 `tail -f`，按 `Ctrl+C` 退出跟踪模式

    交互式操作
    快捷键	功能
    空格 / f	向下翻一页
    b	向上翻一页
    Enter	向下翻一行
    /关键词	向前搜索（按 n 跳转到下一个）
    ?关键词	向后搜索（按 N 跳转到上一个）
    g	跳转到文件首行
    G	跳转到文件末行
    :n	查看下一个文件（多文件打开时）
    :p	查看上一个文件
    q	退出 less

    按 v 键可在当前光标位置启动默认编辑器（如 vi）编辑文件。

    使用 & 过滤显示匹配行（如 &error 只显示含 "error" 的行）。

* `stty -echo`

    关闭回显。

    ```bash
    stty -echo  # 关闭回显
    read -p "Enter password: " password
    stty echo   # 恢复回显
    echo        # 换行（避免密码输入后的提示符紧贴上一行）
    ```

* `stty`

    set teletype, or set terminal

    常用功能：

    * 查看当前终端设置: `stty -a`

    * 关闭回显（输入不显示，如输入密码时）: `stty -echo`

    * 恢复回显：`stty echo`

    * 禁用控制字符（如 Ctrl+C 中断信号）：`stty intr undef  # 取消 Ctrl+C 的中断功能`

    * 将退格键（Backspace）设为删除前一个字符: `stty erase ^H`

    * 如果终端因错误配置导致乱码或无响应，可通过重置恢复：`stty sane`

    * 禁止终端显示输入内容：`stty -echo; read -s; stty echo`

    stty 的配置是临时性的，仅对当前终端会话有效。

* `setsid`

    创建一个新的会话（session）并在此会话中运行指定的命令，使该进程完全脱离当前终端（terminal）的控制

    即使关闭当前终端窗口或退出登录，通过 setsid 启动的进程仍会继续在后台运行（不会被 SIGHUP 信号终止）。

    新进程会成为会话首进程（session leader），且拥有新的进程组 ID（PGID），与原有终端完全无关。

    example:

    * `setsid your_command &`

        后台运行守护进程（daemon）, 适用于需要长期运行的服务（如自定义脚本或服务）。

    * `setsid tail -f /var/log/syslog`

        即使终端关闭，进程也不会退出

    * 替代 nohup 的更彻底方案

        nohup 仅忽略 SIGHUP 信号，而 setsid 直接脱离终端会话。

        ```bash
        # 启动一个完全脱离终端的进程（日志重定向到文件）
        setsid your_command > /var/log/command.log 2>&1 &
        ```

* `<`, `<<`和`<<<`

    1. <（标准输入重定向）

        用于将文件内容作为命令的标准输入。

        ```bash
        command < file.txt
        ```

        （将 file.txt 的内容传递给 command 作为输入）

    1. <<（Here Document）

        用于在脚本中直接嵌入多行输入，直到遇到指定的结束标记（delimiter）。

        ```bash
        command << EOF
        line 1
        line 2
        EOF
        ```

    1. <<<（Here String）

        用于将单个字符串（而不是文件或多行文本）作为命令的标准输入。

        ```bash
        command <<< "string"
        ```

* `pstree`

    `pstree`可以以树状结构显示 ps 的内容。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ pstree 
    systemd─┬─ModemManager───2*[{ModemManager}]
            ├─NetworkManager───2*[{NetworkManager}]
            ├─accounts-daemon───2*[{accounts-daemon}]
            ├─acpid
            ├─avahi-daemon───avahi-daemon
            ├─blkmapd
            ├─colord───2*[{colord}]
    ```

    其中数字表示多个相同的进程/线程。

    常用参数：

    * `-c` 选项禁用合并

    * `-p`：显示进程的 PID。

    * `-n`：按 PID 数字排序（默认按进程名排序）。

    * `-a`：显示进程的完整命令行参数。

    * `pstree [username]`: 查看某用户启动的进程树

    * `-A`: 使用 ASCII 字符绘制树（兼容性更好）

* `finger`是一个早期的网络工具，用于查询系统上的用户信息，现代系统默认禁用

* inotify

    `inotifywait`默认不被安装，需要手动安装:

    ```
    Command 'inotifywait' not found, but can be installed with:
    sudo apt install inotify-tools
    ```

    example:

    `inotifywait -m test_1/`

    此时在`test_1`中执行`ls`，则`inotifywait`的输出为：

    ```
    Setting up watches.
    Watches established.
    test_1/ OPEN,ISDIR 
    test_1/ ACCESS,ISDIR 
    test_1/ CLOSE_NOWRITE,CLOSE,ISDIR
    ```

    继续执行`touch haha.txt`，则有输出：

    ```
    test_1/ CREATE haha.txt
    test_1/ OPEN haha.txt
    test_1/ ATTRIB haha.txt
    test_1/ CLOSE_WRITE,CLOSE haha.txt
    ```

    此时 cpu 几乎没有占用。

    未验证的限制：

    * 不适用于远程文件系统（如 NFS）。

    * 监控大量文件时可能耗尽 inotify 的 watch 句柄（需调整 /proc/sys/fs/inotify/max_user_watches）。

* `stat filename  # 查看文件详细信息`

* `mail`发送邮件

    直接输入`mail -s "主题" 收件人@example.com`，然后输入邮件正文，按 Ctrl+D 发送。

    接收邮件: 输入`mail`查看收件箱，按邮件编号阅读具体内容。

    `mail`命令默认不提供，需要`sudo apt install mailutils`。

    常用选项：

        -s "subject"：指定邮件主题。

        -c "抄送地址"：设置抄送。

        -b "密送地址"：设置密送。

    交互命令（接收邮件时）：

        d 编号：删除邮件。

        q：退出。

        h：重新显示邮件列表。

* `tail -f`

    用于动态追踪文件末尾的新内容, 默认显示文件最后 10 行

    其他选项：

    * `-n <行数>`：指定初始显示的行数（如 tail -f -n 20 file.log 显示最后 20 行）。

    * `-F`：比 -f 更健壮，会跟踪文件重命名或重建（如日志轮转场景）。

    使用 -F 时，tail 会定期检查文件的 inode 编号 是否变化（例如日志轮转后新文件的 inode 不同）。如果变化，则关闭旧文件描述符，重新打开新文件继续跟踪。

* 任何以`.`开头的文件或目录都会被系统视为隐藏文件

    `..xxx`并不是例外的隐藏文件。

* `ls -A`

    查看用户创建的隐藏文件

* 删除当前目录下的所有内容（包括隐藏文件）

    `rm -rf .[!.]* ..?* *`

    解析：

    * `.[!.]*`：匹配所有以`.`开头，且第二个字符不是`.`的文件/文件夹（如`.bashrc`）。

    * `..?*`：匹配所有以`..`开头，且第三个字符存在的文件/文件夹（如`..hidden`）。

    * `*`：匹配所有非隐藏文件/文件夹。

    这里对`.`和`..`进行特殊处理，是为了避开默认存在的两个文件夹，即`.`和`..`。

    如果文件名包含特殊字符（如 -、--、空格等），可以改用：

    `find . -mindepth 1 -delete`

    解析：

    find . -mindepth 1：查找当前目录下的所有文件/文件夹（不包括 . 本身）。

    -delete：直接删除（需谨慎）。

    仅删除隐藏文件和文件夹:

    `rm -rf .[!.]* .??*`

    解析：

    * `.[!.]*` → 匹配`.`开头，第二个字符不是`.`的文件/目录（如 .bashrc、.config）。

    .??* → 匹配 . 开头，且至少 3 个字符的文件/目录（如 .ssh、.gitignore）。

    `find . -maxdepth 1 -name ".*" -exec rm -rf {} +`

    解析：

    -maxdepth 1 → 仅当前目录（不递归子目录）。

    -name ".*" → 匹配所有以 . 开头的文件/目录。

    -exec rm -rf {} + → 批量删除匹配项。

    递归删除子目录中的隐藏文件：

    `find . -name ".*" -exec rm -rf {} +`

    仅删除隐藏文件（保留隐藏目录）:

    `find . -maxdepth 1 -type f -name ".*" -delete`

    -type f → 仅匹配文件（不包括目录）。

* `chfn`

    chfn（Change Finger）用于修改用户信息.

    syntax:

    ```
    chfn [选项] [用户名]
    ```

    * `-f`或`--full-name`: 修改全名

    * `-o`或`--office`: 设置办公室地址

    * `-p`或`--office-phone`: 设置办公电话

    * `-h`或`--home-phone`: 设置家庭电话

    example:

    * `chfn`

        交互式运行。

    * `chfn -f "Jane Smith" johndoe`

        将用户`johndoe`的 name 修改为`Jane Smith`.

    修改的信息存储在`/etc/passwd`的`GECOS`字段，可通过`finger`命令查看。

    `finger`需要使用 apt 安装：`apt install finger`

    `chfn`这个命令在现代 linux 中已经不怎么用了。

* crontab examples

    * `0 * * * * /path/script.sh`
        
        每小时整点执行

    * `*/10 * * * * date >> /tmp/log`
    
        每 10 分钟记录当前时间

    * `0 2 * * * /backup.sh`
    
        每天凌晨 2 点执行备份

    * `0 0 * * 1 tar -zcf /backup/weekly.tar.gz /data`
    
        每周一零点压缩备份

    * `@reboot /path/start_service.sh`
    
        系统启动时执行

    * `* * * * * /path/to/command`

        每分钟执行一次

    * `*/5 * * * * /path/to/command`

        每5分钟执行一次

    * `30 * * * * /path/to/command`

        每小时的第30分钟执行

    * `0 2 * * * /path/to/command`

        每天凌晨2点执行

    * `0 3 * * 1 /path/to/command`

        每周一凌晨3点执行

    * `0 12 1 * * /path/to/command`

        每月1号中午12点执行

    * `* * * * * /path/to/script.sh >> /var/log/script.log 2>&1`

        将输出追加到日志文件（避免邮件通知）

    * `* * * * * /path/to/command > /dev/null 2>&1`

        丢弃所有输出（静默执行）

    * `0 0 * * * rm -rf /tmp/*`

        每天凌晨清理临时文件

    * `0 2 * * 0 mysqldump -u root -pPASSWORD dbname > /backup/db.sql`

        每周日备份数据库

    * `*/30 * * * * /usr/sbin/ntpdate pool.ntp.org`

        每30分钟同步时间（需安装ntpdate）

    * `0 9 * * * echo "Daily meeting at 9:30!" | wall`

        每天9点提醒自己（写入终端）

    * `0 * * * * df -h > /home/user/disk_usage.log`

        每小时检查磁盘空间

    * `@reboot /path/to/startup_script.sh`

        系统重启后执行

    * `0 17 * * 1-5 echo "End of workday" | mail -s "Reminder" user@example.com`

        工作日（周一到周五）下午5点发邮件

    * `0 * * * 1-5 /path/to/command`

        工作日每小时执行一次

    * `*/10 * * * * sleep $((RANDOM \% 60)) && /path/to/command`

        随机延迟执行（避免任务集中触发）

    * `0 9-18 * * 1-5 /path/to/work_script.sh`

        每周一到周五，上午9点到下午6点，每小时执行一次

    * `59 23 28-31 * * [ "$(date +\%d -d tomorrow)" = "01" ] && /path/to/monthly_task.sh`

        每月最后一天23:59执行
    
    * `* * * * * /path/to/command | mail -s "Cron Debug" your@email.com`

        通过日志或临时邮件检查任务是否运行


* crontab 其他常用命令

    * `crontab -l`
        
        列出当前用户的所有定时任务

        其实就是列出 config 文件的所有内容，类似于 cat。

    * `crontab -r`
    
        删除当前用户的所有定时任务（谨慎使用！）

    * `crontab -u user`
    
        管理员专用：管理其他用户的任务（如`-e`, `-l`, `-r`）

* crontab

    `crontab`可以定时执行一些命令，在 ubuntu 系统上默认装有。

    `crontab -e`编辑定时任务的配置文件。

    配置文件的格式：

    ```
    * * * * * command_to_execute
    │ │ │ │ │
    │ │ │ │ └── 星期几 (0-7, 0和7均代表周日)
    │ │ │ └──── 月份 (1-12)
    │ │ └────── 日 (1-31)
    │ └──────── 小时 (0-23)
    └────────── 分钟 (0-59)
    ```

    * `*`：任意值（如 `* * * * *` 表示每分钟）

    * `,`：分隔多个时间（如 `0,15,30 * * * *` 表示每小时的 0、15、30 分）

    * `-`：范围（如 `0 9-17 * * *` 表示每天 9点到17点整点）

    * `/`：间隔（如 `*/5 * * * *` 表示每 5 分钟）

    example:

    ```
    # 每分钟执行一次
    * * * * * date
    ```

    使用`crontab -e`编辑任务后无需重启服务，修改会自动生效。

    output:

    `/var/log/syslog`:

    ```
    Jul 31 16:08:23 hlc-VirtualBox crontab[3437117]: (hlc) BEGIN EDIT (hlc)
    Jul 31 16:08:26 hlc-VirtualBox crontab[3437117]: (hlc) REPLACE (hlc)
    Jul 31 16:08:26 hlc-VirtualBox crontab[3437117]: (hlc) END EDIT (hlc)
    Jul 31 16:09:01 hlc-VirtualBox cron[689]: (hlc) RELOAD (crontabs/hlc)
    Jul 31 16:09:01 hlc-VirtualBox CRON[3437888]: (hlc) CMD (date)
    Jul 31 16:09:01 hlc-VirtualBox CRON[3437887]: (CRON) info (No MTA installed, discarding output)
    ```

    cron 默认会通过邮件发送命令的输出，但系统没有安装邮件传输代理（MTA）（如 sendmail、postfix 或 exim4），所以输出被丢弃。

    通常会让 cron 的输出重定向到日志文件：

    `* * * * * date >> /tmp/cron_date.log 2>&1`

* `pgrep`

    `pgrep`可根据程序的名称快速找到 pid.

    ```bash
    pgrep firefox
    ```

    output:

    ```
    2301052
    ```

    `pgrep`通常可以和`kill`等命令连用：

    `kill $(pgrep nginx)`

    如果有多个进程实例，则会列出多个 pid:

    ```bash
    pgrep bash
    ```

    output:

    ```
    2740
    3059
    2293259
    2294904
    2328484
    2529093
    2618992
    2754955
    2792818
    2823794
    ```

* `sudo -S`

    用于从标准输入（stdin）读取密码，而非交互式终端提示.

    `-S`等价于`--stdin`

    example:

    ```bash
    echo "你的密码" | sudo -S command

    cat password.txt | sudo -S apt update  # 从文件读取密码
    ```

    需要注意的是 echo 不会换行：

    ```bash
    echo xxx | sudo -S echo hello
    ```

    output:

    ```
    [sudo] password for hlc: hello
    ```

    似乎没有什么好的解决方案。

* `yum list`和`yum search`的区别

    * `yum list <package>`

        精确列出匹配名称的软件包（包括已安装、可安装或可升级的版本）。

        特点:

        * 严格匹配包名：默认按完整名称匹配（支持通配符 *）。

        * 显示详细信息：输出包含 包名、版本号、所属仓库（如是否已安装）。

        * 不搜索描述：仅检查包名，不涉及软件包的描述或关键字。

        example:

        `yum list "nginx*"  # 列出所有名称以 `nginx` 开头的软件包`

        output:

        ```
        nginx.x86_64    1.20.1-1.el7    @epel      # 已安装
        nginx-module.x86_64 1.20.1-1.el7 updates    # 可升级
        ```

    * `yum search <package>`

        通过关键字搜索软件包（匹配包名和描述信息）。

        特点:

        * 模糊搜索：同时匹配包名、描述、摘要中的关键字。

        * 返回摘要：显示包名和简要描述，但不显示版本或仓库信息。

        * 结果更广泛：可能包含名称不直接相关但描述匹配的包。

        example:

        `yum search "web server"  # 搜索描述或名称中包含 "web server" 的包`

        output:

        ```
        nginx.x86_64 : High performance web server
        httpd.x86_64 : Apache HTTP Server
        ```

* yum 简介

    Yum（Yellowdog Updater Modified）

    * 安装软件包

        `yum install <package_name>`

    * 安装本地 RPM 包（自动解决依赖）

        `yum localinstall <path_to_rpm>`

    * 升级单个软件包

        `yum update <package_name>`

    * 升级所有可升级软件包

        `yum update`

    * 检查可升级的软件包

        `yum check-update`

    * 搜索软件包（按名称/描述）

        `yum search <keyword>`

    * 列出已安装的软件包

        `yum list installed`

        `yum list`会显示仓库中所有可用的软件包（包括已安装和未安装的）。

        `yum list available`仅列出可安装但尚未安装的软件包

        `yum list updates`列出所有可升级的软件包

        `yum list extras`列出已安装但不在仓库中的包（如手动安装的 RPM）

        `yum list <package_name>`搜索特定软件包（支持通配符，如`yum list "nginx*"`）

    查看软件包信息
    yum info <package_name>

    列出软件包的依赖
    yum deplist <package_name>

    查找提供特定文件的软件包
    yum provides <file_path>

    删除软件包（保留依赖）
    yum remove <package_name>

    删除无用依赖
    yum autoremove

    清理旧内核
    yum remove $(rpm -q kernel | grep -v $(uname -r))

    删除软件包（保留依赖）
    yum remove <package_name>

    删除无用依赖
    yum autoremove

    清理旧内核
    yum remove $(rpm -q kernel | grep -v $(uname -r))

    列出所有仓库
    yum repolist all

    启用/禁用仓库
    yum-config-manager --enable/disablerepo <repo_name>

    添加新仓库
    yum-config-manager --add-repo <repo_url>

    清理缓存（保留元数据）
    yum clean packages

    清理所有缓存（包括元数据）
    yum clean all

    重建缓存
    yum makecache

    查看历史操作
    yum history

* `/etc/sudoers`中，`@includedir /etc/sudoers.d`用于 加载`/etc/sudoers.d`目录下的所有配置文件

    例如：

    `/etc/sudoers.d/web_admins` -> 存放 Web 管理员的 sudo 权限

    `/etc/sudoers.d/db_admins` -> 存放数据库管理员的 sudo 权限

    在软件包安装时（如 Docker、Nginx），它们可能会自动在`/etc/sudoers.d/`下添加自己的规则。

    修改`/etc/sudoers.d/`下的文件后，无需重启，sudo 会自动识别（但建议用`visudo -c`检查语法）。

    文件名：不能包含`.`或`~`（避免读取临时文件或备份文件）。

    权限：必须为 0440（root:root 可读，其他用户不可读），否则 sudo 会忽略它并报错.

    ```bash
    # 检查所有 sudoers 文件（包括 sudoers.d 下的）
    sudo visudo -c
    ```

* `sudo`与`/etc/sudoers`

    用户使用 sudo 执行的命令会被记录到`/var/log/auth.log`文件中。

    比如用户执行`sudo echo hello`，日志的记录为

    ```
    Jul 27 14:38:16 hlc-VirtualBox sudo:      hlc : TTY=pts/5 ; PWD=/home/hlc/Documents/documents/Personal/logs ; USER=root ; COMMAND=/usr/bin/echo hello
    Jul 27 14:38:16 hlc-VirtualBox sudo: pam_unix(sudo:session): session opened for user root(uid=0) by (uid=1000)
    Jul 27 14:38:16 hlc-VirtualBox sudo: pam_unix(sudo:session): session closed for user root
    ```

    `/etc/sudoers`规定了用户使用 sudo 可以执行哪些命令。

    通常使用`visudo`编辑这个文件，`visudo`可以在保存前进行语法检查，防止语法出错。

    `sudoers`文件的语法为：

    ```
    user    HOST=(RUNAS_USER)    COMMANDS
    %group  HOST=(RUNAS_USER)    COMMANDS
    ```

    其中，

    * `user/%group`：用户名或%组名。

    * `HOST`：允许使用 sudo 的主机名（通常设为`ALL`）。

    * `(RUNAS_USER)`：可以切换的目标用户（如 `(root)`、`(ALL)`）。

        通常设置为`(ALL:ALL)`或`(ALL)`。

        * `(ALL)`: 允许用户以 任意用户身份（包括 root）执行命令。

            ```bash
            sudo -u root apt update    # 以 root 身份运行
            sudo -u bob apt install xx # 以用户 bob 身份运行
            ```

        * `(root)`: 仅允许用户以 root 身份 执行命令（不能切换为其他用户）。

            ```bash
            sudo systemctl restart nginx      # 隐含 -u root
            sudo -u alice systemctl start xx  # 报错（不允许）
            ```

        * `(ALL:ALL)`: 允许用户以 任意用户和任意用户组 身份执行命令（用户和组均可切换）。

            `(RUNAS_USER:RUNAS_GROUP)`, 若省略`:GROUP`，默认使用目标用户的默认组。注意这个 group 的指定在后面，不在前面。

            ```bash
            sudo -u alice -g developers chmod 755 file
            sudo -u root -g root chmod 600 /etc/shadow
            ```

    `COMMANDS`：允许执行的命令（绝对路径，`ALL`表示全部）。

    example:

    ```conf
    alice   ALL=(root)    /usr/bin/apt      # alice 可以 root 身份运行 apt
    %admin  ALL=(ALL)     ALL                # admin 组成员可执行任何命令
    ```

    使用别名（变量）：

    ```conf
    User_Alias     ADMINS = alice, bob
    Host_Alias     SERVERS = 192.168.1.1
    Runas_Alias    DEVS = tom
    Cmnd_Alias     PKG_CMDS = /usr/bin/apt, /usr/bin/dpkg

    ADMINS SERVERS=(DEVS) PKG_CMDS
    ```

    其他配置：

    * `NOPASSWD`：执行命令无需密码

        ```conf
        bob    ALL=(root)    NOPASSWD: /usr/bin/systemctl
        ```

    * `!`排除命令：

        ```conf
        charlie ALL=(ALL) ALL, !/usr/bin/passwd root
        ```

    * 全局配置

        通过`Defaults`设置全局行为，如：

        ```conf
        Defaults    env_keep += "HTTP_PROXY"   # 保留环境变量
        Defaults    insults                    # 输错密码时显示“嘲讽”
        ```

    因为编辑这个文件可能会影响到当前虚拟机环境，所以上面的命令都没有验证过。

* `ssh-askpass`简介

    安装：`apt install ssh-askpass`

    直接运行`./ssh-askpass`时，会弹出一个窗口，输入密码并按回车后，输入的内容会出现在 terminal 里。

    `sudo`和`ssh`都可以使用`ssh-askpass`程序弹出 gui 输入密码。

    以`sudo`为例，首先配置环境变量，然后使用`sudo -A <command>`激活：

    ```bash
    export SUDO_ASKPASS=/usr/bin/ssh-askpass
    sudo -A -v
    ```

    此时在弹出的窗口中输入密码，terminal 不会回显。

    如果未检测到环境变量，sudo 会报错：

    `sudo: no askpass program specified, try setting SUDO_ASKPASS`

    `ssh`使用`ssh-askpass`同样需要使用环境变量：

    ```bash
    export SSH_ASKPASS="/usr/bin/ssh-askpass"
    ```

    经验证，ssh 很难启动 askpass 弹窗，目前不清楚原因。

* `sudo -v`用于延长 sudo 的密码缓存时间

    `sudo`第一次缓存密码的时间为 15 分钟，执行`sudo -v`会重置这个计时器。

    这里的`-v`代表 validate

    通常在脚本开头检查权限，防止脚本因为 sudo 密码问题中断：

    ```bash
    if ! sudo -v; then
        echo "Error: No sudo access or incorrect password."
        exit 1
    fi
    ```

    `sudo -k`可以清空密码缓存。执行`sudo -k`不需要 sudo 密码。

* `/proc/<PID>/cmdline` 是 Linux 系统 伪文件系统（procfs） 中的一个特殊文件，用于 获取指定进程（PID）的完整命令行启动参数，以`\0`（空字符）分隔各个参数。

    如果进程本身就是以相对路径启动的，比如`bash`，那么`cmdline`不会显示其绝对路径。

    `cmdline`的结尾没有`\n`。

    因为 cmdline 使用`\0`进行参数分隔，所以避免了很多字符转义的问题。

* `sudo -l`

    列出当前用户可以使用`sudo`执行哪些 root 操作。

    `sudo -ll`可以列出更详细的输出。

    `sudo -U user -l`可以查看其他用户的 root 权限。

* `ps -f`

    既显示完整信息，也显示完整 cmd（比如绝对路径）。否则只显示简略信息和 cmd 名称。

    example:

    ```bash
    ps -a
    ```

    output:

    ```
        PID TTY          TIME CMD
       2178 tty2     00:00:00 gnome-session-b
     295495 pts/9    00:01:03 ssh
     296484 pts/10   00:00:02 socat
     489812 pts/3    00:00:00 ps
    ```

    ```bash
    ps -af
    ```

    output:

    ```
    UID          PID    PPID  C STIME TTY          TIME CMD
    hlc         2178    2174  0 7月19 tty2    00:00:00 /usr/libexec/gnome-session-b
    hlc       295495  294700  0 7月19 pts/9   00:01:03 ssh -R 8825:127.0.0.1:8825 h
    hlc       296484  296052  0 7月19 pts/10  00:00:02 socat TCP-LISTEN:8825,reusea
    hlc       490526  487976  0 12:26 pts/3    00:00:00 ps -af
    ```

    可以看到，使用`-f`后，信息有截断，但是确实是完整路径。如果要显示未截断的完整信息，那么需要再加上`-l`参数。

* `ps -e --forest`

    * `-e`

        显示所有进程（包括其他用户的进程），等同于`-A`。

        `man ps`中的解释：

        > -e     Select all processes.  Identical to -A.

    * `--forest`

        以树状缩进形式显示进程层级，清晰体现父子关系。

        其输出中使用`\_`而不是`|_`，目前仍不清楚原因。

* `tr`的用法

    `tr`指的是 translate，通常用于字符替换

    example:

    ```bash
    echo hello | tr a-z A-Z
    ```

    output:

    ```
    HELLO
    ```

    也可以删除字符：

    ```bash
    echo "hello 123 world" | tr -d 0-9
    ```

    output:

    ```
    hello  world
    ```

    还可以去重：

    ```bash
    echo hello | tr -s l
    ```

    output:

    ```
    helo
    ```

    这里的`-s`可能是 squash 的意思。

    过滤（filter in，保留指定的字符）：

    ```bash
    echo "hello 123" | tr -cd 'a-z'
    ```

    output:

    ```
    hello
    ```

    output 末尾无换行符。`tr`只保留`a-z`小字字母字符。

    这里的`-c`可能是补集（complementary）的意思

    一一映射：

    ```bash
    echo abc | tr cba xzy
    ```

    output:

    ```
    yzx
    ```

    注：

    1. `tr`在处理`\n`时。需要给`\n`加上引号（单引号双引号都可以），否则会被 bash 转义。

        example:

        ```bash
        echo hello | tr '\n' N
        ```

        output:

        ```
        helloN
        ```

        output 后无换行。

    1. `tr`只能处理单个字符，不能处理字符串和正则表达式。

* `grep -F`表示不进行正则解析

    example:

    `grep -F "hello" content.txt`

    只查找`hello`字符串。

    `grep -F "a.*b" content.txt`

    匹配`a.*b`字符串。

    `grep -F`等价于`fgrep`。

* `/proc/<PID>/environ`是一个在内存中的文件，以只读的形式存储了指定 PID 进程在启用时的环境变量信息

    example:

    `cat /proc/3273/environ`

    output:

    ```
    SYSTEMD_EXEC_PID=2451SSH_AUTH_SOCK=/run/user/1000/keyring/sshSESSION_MANAGER=local/hlc-VirtualBox:@/tmp/.ICE-unix/2235,unix/hlc-VirtualBox:/tmp/.ICE-unix/2235GNOME_TERMINAL_SCREEN=/org/gnome/Terminal/screen/16e4c141_024c_4318_9398_96a803c31884LANG=en_US.UTF-8XDG_CURRENT_DESKTOP=ubuntu:GNOMEPWD=/home/hlcWAYLAND_DISPLAY=wayland-0LC_IDENTIFICATION=zh_CN.UTF-8IM_CONFIG_PHASE=1...
    ```

    其形式为`key=value`，环境变量之间使用`\0`间隔。

    此文件为只读属性，无法修改。

    可以将`\0`替换为`\n`，便于阅读：

    `cat /proc/<PID>/environ | tr '\0' '\n'`

    output:

    ```
    SYSTEMD_EXEC_PID=2451
    SSH_AUTH_SOCK=/run/user/1000/keyring/ssh
    SESSION_MANAGER=local/hlc-VirtualBox:@/tmp/.ICE-unix/2235,unix/hlc-VirtualBox:/tmp/.ICE-unix/2235
    GNOME_TERMINAL_SCREEN=/org/gnome/Terminal/screen/16e4c141_024c_4318_9398_96a803c31884
    LANG=en_US.UTF-8
    XDG_CURRENT_DESKTOP=ubuntu:GNOME
    PWD=/home/hlc
    WAYLAND_DISPLAY=wayland-0
    LC_IDENTIFICATION=zh_CN.UTF-8
    IM_CONFIG_PHASE=1
    ...
    ```

    说明：

    1. `/proc/<PID>/environ`不是实时的，如果在程序中运行`setenv()`，那么此文件内容不会被改变。

    1. 仅允许进程所有者或 root 用户读取（权限为`-r--------`）

        这样看来，这个文件的用处似乎不大？

* `sudo yum check-update`相当于`sudo apt update`

* centos 加入 sudo 权限

    `usermod -aG wheel 用户名`

    在 centos 中，`wheel`组有 sudo 权限。

* 解析`ps -ef`

    ps 指的是 process status

    `-e`：显示所有进程（包括其他用户的进程）。

    `-f`：以完整格式（full-format）输出详细信息。

    example:

    ```
    (base) hlc@hlc-VirtualBox:~$ ps -ef
    UID          PID    PPID  C STIME TTY          TIME CMD
    root           1       0  0 09:50 ?        00:00:01 /sbin/init splash
    root           2       0  0 09:50 ?        00:00:00 [kthreadd]
    root           3       2  0 09:50 ?        00:00:00 [pool_workqueue_release]
    root           4       2  0 09:50 ?        00:00:00 [kworker/R-rcu_g]
    root           5       2  0 09:50 ?        00:00:00 [kworker/R-rcu_p]
    root           6       2  0 09:50 ?        00:00:00 [kworker/R-slub_]
    root           7       2  0 09:50 ?        00:00:00 [kworker/R-netns]
    ...
    ```

    UID：进程所属用户。

    PID：进程的唯一ID。

    PPID：父进程ID。

    C：CPU占用率。

    STIME：进程启动时间。

    TTY：启动进程的终端（?表示与终端无关，如守护进程）。

    TIME：进程占用CPU总时间。

    CMD：进程对应的完整命令或程序路径。

* `rsync`中，src dir 的下面四种写法是等价的

    `.`, `./`, `*`, `./*`

* `rsync`中, `--progress`可以显示进度，`--partial`支持断点续传

    `-P`则表示同时 enable `--progress --partial`。

* grep 开启`-E`时，可以使用`|`匹配多个模式。

    `grep -E haha\|hehe msg.txt`

    `grep -E 'haha|hehe' msg.txt`

    bash 会将`|`默认解释为管道，如果希望 bash 将`|`解释为字符`|`，那么要么在之前加`\`，要么使用单引号`''`。
    
    注：

    1. 标准的正则表达式支持`|`，比如 python 的`re`模块。

    1. `|`的前后不能有空格，或者说，空格不会被忽略。

* `rsync -z`表示在传输过程中对要传输的文件进行压缩。如果传输过程文本文件比较多，可以使用`-z`大幅提高传输效率。

* `tee -a`表示在文件末尾追加

    如果文件不存在，则会创建文件。

    echo 本身会在行尾加`\n`，因此不需要额外考虑`\n`。

    ```bash
    echo 'heloo' | tee -a log.txt
    echo 'heloo' | tee -a log.txt
    cat log.txt
    ```

    output:

    ```
    heloo
    heloo
    heloo
    heloo
    ```

* grep 查看前后 n 行文本

    * 向前 n 行：`grep -B n`

    * 向后 n 行：`grep -A n`

    * 前后各 n 行：`grep -C n`

    其中 A 表示 after，B 表示 before，C 表示 context。

    example:

    `msg.txt`:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    haha
    hehesdf
    aaaaa
    bbb
    ```

    run: `grep -B 1 -A 2 345 msg.txt`

    output:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    haha
    hehesdf
    ```

    run : `grep -C 1 345 msg.txt`

    output:

    ```
    hello, world, nihao, zaijian
    123, 234, 345, 456, nihao
    hello, 345
    haha
    ```

* find 不输出没有权限的文件

    find 对没有权限的文件会输出类似

    ```
    ...
    find: ‘/proc/1188309/task/1188309/ns’: Permission denied
    find: ‘/proc/1188309/fd’: Permission denied
    find: ‘/proc/1188309/map_files’: Permission denied
    find: ‘/proc/1188309/fdinfo’: Permission denied
    find: ‘/proc/1188309/ns’: Permission denied
    ...
    ```

    的信息。这些信息其实都是 stderr。因此可以考虑过滤掉 stderr 的输出：

    `find <path> -name <pattern> 2>/dev/null`

    example:

    `find / -name hello 2>/dev/null`

    output:

    ```
    /home/hlc/miniconda3/pkgs/tk-8.6.14-h39e8969_0/lib/tk8.6/demos/hello
    /home/hlc/miniconda3/lib/tk8.6/demos/hello
    /home/hlc/miniconda3/envs/torch/lib/tk8.6/demos/hello
    /home/hlc/miniconda3/envs/vllm/lib/tk8.6/demos/hello
    /home/hlc/Documents/Projects/boost_1_87_0/tools/build/example/qt/qt4/hello
    /home/hlc/Documents/Projects/boost_1_87_0/tools/build/example/qt/qt3/hello
    /home/hlc/Documents/Projects/boost_1_87_0/tools/build/example/hello
    /home/hlc/Documents/Projects/chisel-tutorial/src/main/scala/hello
    /home/hlc/Documents/Projects/makefile_test/hello
    ```

* sudo 与环境变量的关系

    sudo 应该写在环境变量的前面。
    
    example:

    `sudo http_proxy=http://127.0.0.1:8822 https_proxy=http://127,0,0,1:8822 apt update`

* linux host name 相关

    * 显示当前的 host name: `hostname`

        example:

        run: `hostname`

        output: `hlc-VirtualBox`

    * 显示当前系统的基本信息：`hostnamectl`

        example:

        run: `hostnamectl`

        output:

        ```
         Static hostname: hlc-VirtualBox
               Icon name: computer-vm
                 Chassis: vm
              Machine ID: d3dcf00f11234838acfafd0a40493023
                 Boot ID: 5ce8f551956f4b14ab4c447a4a2ecbd0
          Virtualization: oracle
        Operating System: Ubuntu 22.04.4 LTS              
                  Kernel: Linux 6.8.0-52-generic
            Architecture: x86-64
         Hardware Vendor: innotek GmbH
          Hardware Model: VirtualBox
        ```

    * 修改 hostname `hostnamectl set-hostname <new-hostname>`

    * 可以通过修改`/etc/hostname`文件和`/etc/hosts`并重启系统来修改 hostname。

    * ubuntu 中，可以通过 settings -> about 修改 hostname。

    * 临时修改 hostname: `sudo hostname new-hostname` （未测试过）

## note

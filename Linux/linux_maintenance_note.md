# Linux Maintenance Note

这里主要记录 linux 相关的命令，配置，主题为运维。

这里不记录编程相关，操作系统原理相关。

## cache

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
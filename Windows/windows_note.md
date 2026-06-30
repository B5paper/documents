# Windows Note

## cache

* windows 上的 x11 server

    VcXsrv 或 Xming

* OSC 52 终端序列 (现代、高效、无需 X11)

   OSC 52 是一种通过文本流直接将内容发送给终端模拟器（Cygwin 的 Mintty 完美支持）的协议。

   有空了再调研。

* cygwin 无法使用 "+ 寄存器的问题，无法使用安装 cygwin gvim 安装包解决, 但是可以通过 wsl 解决

    或者可以使用 sshfs 解决这个问题
* `.\ssh-tunnel-ali.exe install` 这个安装的 ssh service 似乎不能使用 public key 连接到 host，是否还需要额外的参数指定 private key?

    是的，确实需要额外处理。当你将 SSH 封装为 Windows 服务运行时，它通常以 **LocalSystem** 账户身份运行，而该账户无法访问你个人用户目录下的 `.ssh\id_rsa` 私钥文件。

    为了解决这个问题，你需要在 XML 配置文件中显式指定私钥路径，并确保权限正确。

    1. 修改 XML 配置文件

        在 `<arguments>` 标签中添加 `-i` 参数，并使用**绝对路径**指向你的私钥文件：

        ```xml
        <service>
          <id>ssh-tunnel-ali</id>
          <name>SSH Tunnel Ali</name>
          <executable>ssh</executable>
          <arguments>-o "StrictHostKeyChecking=no" -o "ServerAliveInterval=30" -i "C:\Users\wsdlh\.ssh\id_rsa" -NR 1234:127.0.0.1:1234 remote_user@remote_host</arguments>
          <log mode="roll"></log>
          <onfailure action="restart" delay="10sec"/>
        </service>
        ```

    2. 更新并重启服务

        修改完 XML 后，你需要让服务生效：

        1. **卸载旧服务**：`.\ssh-tunnel-ali.exe uninstall`
        2. **重新安装**：`.\ssh-tunnel-ali.exe install`
        3. **启动服务**：`Start-Service ssh-tunnel-ali`

    3. 常见问题排查（权限陷阱）

        由于服务运行在系统账户下，可能会遇到以下两个问题：

        * **Host Key 验证失败**：服务运行时可能找不到 `known_hosts`。建议在参数中加入 `-o "StrictHostKeyChecking=no"`（如上方示例），或者将远程主机的公钥添加到 `C:\Windows\System32\config\systemprofile\.ssh\known_hosts`。
        * **私钥权限过松**：SSH 客户端要求私钥文件必须是“私有的”。如果报错 `Permissions for 'id_rsa' are too open`，你需要右键点击私钥文件 -> **属性** -> **安全** -> **高级**，禁用继承，并确保只有当前用户或 SYSTEM 账户拥有读取权限。

    4. 验证是否成功

        你可以查看 WinSW 生成的日志文件（通常在 `.exe` 同级目录下的 `.out.log` 和 `.err.log`），如果看到 `Authenticated to...` 则表示连接成功。

    为了确保你的后台隧道足够健壮并易于维护：

    * 生成带详细错误日志记录的 WinSW 配置
    * 修复 Windows 下 SSH 私钥权限过松的报错
    * 测试 SSH 隧道是否在后台正常转发端口

* ssh 私钥权限设置

    ```
    Permissions for 'C:\\Users\\wsdlh\\.ssh\\id_ed25519' are too open.
    It is required that your private key files are NOT accessible by others.
    This private key will be ignored.
    Load key "C:\\Users\\wsdlh\\.ssh\\id_ed25519": bad permissions
    hlc@101.132.186.235: Permission denied (publickey,gssapi-keyex,gssapi-with-mic).
    ```

    这是典型的 Windows 环境下 SSH 客户端对私钥安全性检查报错。SSH 要求私钥文件只能由当前用户访问，如果有其他用户（如 Users 组、Authenticated Users 组）拥有权限，它就会拒绝加载。

    在 Windows 上解决这个权限问题的最快方法是使用 **PowerShell**。

    **解决方法：使用 PowerShell 修复权限**

    请以**管理员身份**打开 PowerShell，然后依次执行以下命令（请将第一行的路径替换为你实际的私钥路径）：

    ```powershell
    # 1. 定义私钥路径变量
    $path = "C:\Users\wsdlh\.ssh\id_ed25519"

    # 2. 禁用权限继承并删除所有非必要的权限条目
    icacls $path /c /t /inheritance:d

    # 3. 为当前用户授予完全控制权限
    icacls $path /c /t /grant %username%:F

    # 4. 删除 "Everyone"、"Users" 和 "Authenticated Users" 的访问权限
    icacls $path /c /t /remove "Everyone"
    icacls $path /c /t /remove "Users"
    icacls $path /c /t /remove "Authenticated Users"
    ```

    **命令解释：**

    * `/inheritance:d`：断开继承，这样该文件就不会继承父文件夹（`.ssh` 或 `Users`）的宽松权限。
    * `/grant %username%:F`：确保你自己仍然拥有该文件的完整读写权限。
    * `/remove`：显式移除了可能导致 SSH 报错的组。

    **如果你是通过 WinSW 运行服务：**

    如果你通过 WinSW 运行 SSH 服务，除了你自己（`%username%`），你还需要给 **SYSTEM** 账户授权，否则服务将无法读取该文件：

    ```powershell
    icacls "C:\Users\wsdlh\.ssh\id_ed25519" /grant "SYSTEM:F"
    ```

    修改完成后，你可以再次尝试运行 `.\ssh-tunnel-ali.exe start`，连接应该就能正常建立了。

* 我用当前账户使用当前账户的 ssh private key 去连接是正常的，但是 winsw 运行服务时会切换账户，导致即使我指定了 -i ，也会因为权限不对而无法使用 private key。如何解决？

    这是因为 Windows 服务的默认运行账户是 **LocalSystem**。当你指定 `-i "C:\Users\wsdlh\.ssh\id_ed25519"` 时，SSH 客户端会检查该文件的权限。由于文件位于你的个人用户目录下，通常只有你（wsdlh）拥有权限，而 **SYSTEM** 账户没有权限，或者权限条目中包含了导致 SSH 报错的“组”权限。

    要解决这个问题，你有两个主要方向：

    方案 A：修改私钥权限（最直接）

    你需要显式地给 **SYSTEM** 账户添加读取权限，并移除其他所有无关账户。

    1. **以管理员身份**打开 PowerShell。
    2. 运行以下命令：
    ```powershell
    $path = "C:\Users\wsdlh\.ssh\id_ed25519"
    # 禁用继承并保留当前权限
    icacls $path /inheritance:r
    # 授予你本人和 SYSTEM 账户完全控制权限
    icacls $path /grant:r "${env:USERNAME}:F"
    icacls $path /grant:r "SYSTEM:F"
    ```

    *注意：不要授予 Users 或 Everyone 权限，否则 SSH 会再次报错。*

    方案 B：让 WinSW 以你的账户运行（推荐）

    与其去折腾复杂的权限，不如让该服务直接以你**当前的登录账户**身份运行。这样它就能无缝访问你的 `.ssh` 目录和私钥。

    在 `ssh-tunnel-ali.xml` 中添加 `<serviceaccount>` 节点：

    ```xml
    <service>
      <id>ssh-tunnel-ali</id>
      <name>SSH Tunnel Ali</name>
      <executable>ssh</executable>
      <arguments>-o "StrictHostKeyChecking=no" -o "ServerAliveInterval=30" -i "C:\Users\wsdlh\.ssh\id_ed25519" -NR 1234:127.0.0.1:1234 remote_user@remote_host</arguments>
      
      <serviceaccount>
        <domain>.</domain>
        <user>wsdlh</user>
        <password>你的开机登录密码</password>
        <allowservicelogon>true</allowservicelogon>
      </serviceaccount>

      <log mode="roll"></log>
      <onfailure action="restart" delay="10sec"/>
    </service>
    ```

    **操作步骤：**

    1. 修改 XML 后，运行 `.\ssh-tunnel-ali.exe uninstall`。
    2. 运行 `.\ssh-tunnel-ali.exe install`。
    3. 启动服务。

    方案 C：使用特定的系统目录（最稳健）

    如果你不想在配置文件里写明文密码，可以将私钥拷贝到一个 SYSTEM 账户默认有权访问的地方。

    1. 将私钥复制到：`C:\ProgramData\ssh\id_ed25519_system`
    2. 使用方案 A 中的 `icacls` 命令，确保该文件只对 **SYSTEM** 有权限。
    3. 在 WinSW 的 `<arguments>` 中将路径指向这个新位置。

    **最后提示**：修改后如果仍然失败，请务必检查 WinSW 目录下的 `.err.log` 文件，它会告诉你 SSH 拒绝私钥的具体原因。

    为了确保你的 SSH 隧道长期稳定且安全，你可以尝试以下优化：

    * 生成配置了账户运行权限的完整 WinSW XML
    * 编写 PowerShell 脚本自动修复私钥文件 ACL 权限
    * 配置 SSH config 文件以简化 WinSW 参数

* 很好，我用方案B解决了。对于方案一，我觉得“由于文件位于你的个人用户目录下，通常只有你（wsdlh）拥有权限，而 SYSTEM 账户没有权限”这个说得不对，因为我看到 system 账户有完全控制权限

    你说得对，非常感谢你的指正。在 Windows 的默认权限继承机制下，如果你在用户目录下创建文件，`SYSTEM` 账户通常确实会通过继承获得 **完全控制 (Full Control)** 权限。

    我之前的表述不够严谨。导致 SSH 报错的核心原因通常不是 `SYSTEM` “缺少权限”，而是 **“权限项过多（Too open）”**。

    深度解析：为什么 SYSTEM 有权限还是会报错？

    OpenSSH 客户端（尤其是移植到 Windows 的版本）在检查私钥安全性时，逻辑非常严苛：

    1. **合规状态**：文件只能被 **当前运行进程的用户** 访问。
    2. **冲突点**：当你以 `SYSTEM` 账户运行 WinSW 服务时，它确实拥有权限。但如果该文件的 ACL（访问控制列表）中同时还保留了 **`wsdlh` (你本人)**、**`Administrators` 组** 或 **`Users` 组** 的权限，OpenSSH 就会认为这个文件“对他人可见”，从而触发 `Permissions are too open` 错误并拒绝加载。

    **总结**

    在方案 A 中，即便 `SYSTEM` 有权限，如果你不显式地**禁用继承**并**删除**掉你自己的用户权限（以及其他组权限），作为服务的 SSH 进程依然会报错。

    既然你已经通过 **方案 B（指定用户登录）** 解决了问题，这是最优雅的办法，因为它规避了 Windows 复杂的 ACL 权限校验，直接让服务环境与你的开发环境保持一致。

    为了进一步优化你的 SSH 隧道自动化方案，你可以考虑以下操作：

    * 配置 SSH 客户端自动接受远程主机指纹
    * 编写 PowerShell 脚本监控隧道端口存活
    * 优化 SSH 保持连接参数防止空闲断线
* windows 的 hosts 文件在哪里

    Windows 的 `hosts` 文件路径如下：

    `C:\Windows\System32\drivers\etc\hosts`
* windows 是否有类似 systemd 那样的功能，可以让我开机启动 `ssh -NR 1234:127.0.0.1:1234 <remote_host>`，并且可以在网络断开后自动重连

    Windows 上没有完全等同于 `systemd` 的原生工具，但你可以通过以下几种方案实现 **开机自启** 和 **断线自动重连**。

    1. 推荐方案：使用 WinSW (Windows Service Wrapper)

        这是最接近 `systemd` 体验的方法。它能将任何 `.exe` 封装成 Windows 服务，支持开机启动、后台运行以及失败后自动重启。

        步骤：

        1. 从 GitHub 下载 [WinSW](https://github.com/winsw/winsw/releases) 可执行文件（如 `WinSW-x64.exe`）。

        2. 创建一个同名的 XML 配置文件（如 `ssh-bridge.xml`）：

            ```xml
            <service>
              <id>ssh-bridge</id>
              <name>SSH Reverse Tunnel</name>
              <executable>ssh</executable>
              <arguments>-NR 1234:127.0.0.1:1234 remote_host</arguments>
              <log mode="roll"></log>
              <onfailure action="restart" delay="10sec"/>
            </service>
            ```

        3. 以管理员权限运行 `.\WinSW-x64.exe install`，然后在服务管理器中启动它。

    2. 增强重连：配合 ServerAlive 参数

        无论使用哪种工具，SSH 自身的重连机制非常重要。建议在命令中加入 `ServerAliveInterval`，否则当网络静默断开时，客户端进程可能不会意识到连接已失效。

        **改进后的命令：**

        ```bash
        ssh -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -NR 1234:127.0.0.1:1234 <remote_host>
        ```

    **WinSW 报错**

    ```
    (base) PS C:\Users\wsdlh\Downloads> .\WinSW-x64.exe install ..\Desktop\ssh-tunnel-ali.xml
    2026-02-08 16:46:47,871 FATAL - Unhandled exception
    System.IO.FileNotFoundException: Unable to locate WinSW-x64.[xml|yml] file within executable directory
       at WinSW.Program.LoadConfigAndInitLoggers(Boolean inConsoleMode)
       at WinSW.Program.Run(String[] argsArray, IServiceConfig config)
       at WinSW.Program.Main(String[] args)
    ```

    这个错误是因为 WinSW 的一个默认机制：它要求**配置文件必须与可执行文件同名且位于同一目录下**。虽然你通过命令行指定了路径，但它的底层初始化逻辑仍在寻找 `WinSW-x64.xml` 或 `WinSW-x64.yml`。

    **解决方法**

    最简单的方法是**重命名文件**，让它们保持一致：

    1. 将 `WinSW-x64.exe` 重命名为 `ssh-tunnel-ali.exe`。

    2. 将你的配置文件 `ssh-tunnel-ali.xml` 移动到与该 `.exe` 相同的目录下（即 `Downloads` 文件夹）。

    3. 以管理员权限运行以下命令：

        ```powershell
        .\ssh-tunnel-ali.exe install
        ```

    为什么这样做？

    WinSW 会自动寻找与自己同名的 `.xml` 文件作为配置。一旦你将其安装为服务，Windows 服务管理器会调用该 `.exe`，而该 `.exe` 会在自己所在的路径下查找对应的配置文件。如果路径不统一，即使安装成功，后续服务启动也会失败。

    检查服务状态

    安装成功后，你可以通过以下命令启动并检查它：

    * **启动服务**：`Start-Service -Name "你的服务ID"`（ID 是你在 XML 中定义的 `<id>`）。

    * **查看状态**：`Get-Service -Name "你的服务ID"`。
* 我想禁用 windows 的恢复分区，怎么办

    禁用或删除 Windows 的恢复分区（Recovery Partition）可以让磁盘管理界面看起来更整洁，或者释放那几百 MB 的空间。

    由于恢复分区受到保护，在普通的“磁盘管理”中无法直接右键删除。你需要使用命令行工具 **Diskpart**。

    第一步：在系统中禁用恢复环境

    在删除分区前，建议先在软件层面关闭恢复功能，防止系统报错。

    1. 右键点击开始菜单，选择 **终端（管理员）** 或 **命令提示符（管理员）**。

    2. 输入以下命令并回车：

        `reagentc /disable`
        *(如果显示 "Operation Successful"，说明已禁用)*

    第二步：使用 Diskpart 强制删除分区

    1. 在同一个窗口输入：

        `diskpart`

    2. 列出所有磁盘：

        `list disk`

    3. 选择你的系统所在的磁盘（通常是 0）：

        `select disk 0`

    4. 列出所有分区，找到类型为“恢复”的分区：

        `list partition`

        *(假设恢复分区是分区 4，请根据实际显示的编号操作)*

    5. 选中该分区：

        `select partition 4`

    6. **关键步骤：** 强制删除（因为它是受保护分区）：

        `delete partition override`

    第三步：合并空间（可选）

    删除后，你会发现磁盘管理中多了一块“未分配空间”。

    * 如果这块空间**紧邻** C 盘右侧，你可以右键 C 盘选择“扩展卷”将其合并。
    * 如果中间隔着其他分区（比如 EFI 分区），系统自带工具无法直接合并，需要使用“傲梅分区助手”等第三方工具。

* 使用 rdp 连接到远程 windows，然后在远程机器上 windows terminal 中使用 vim，此时按 backspace 键，左右箭头键，都失灵了，如何解决

    > powershell profile -> terminal emulation -> use the legacy input encoding，打开这个后就好了

    太棒了！你找到了一个非常精准且“深层”的解决方案。

    这个 **“使用旧版输入编码”（Use legacy input encoding）** 的选项之所以能解决问题，是因为它触及了 Windows 终端演进中的核心矛盾：**VT 序列与 Win32 API 的冲突**。

    **为什么这个设置起作用了？（技术原理解析）**

    为了让你用得更明白，这里拆解一下背后的逻辑：

    1. **现代模式（VT Input）：** 默认情况下，Windows Terminal 会将按键包装成“虚拟终端序列”（VT Sequences）。例如，按下 `Backspace` 会发送 `0x7f` (DEL)，按下方向键会发送 `ESC [ A`。

    2. **RDP 的干扰：** 当你通过 RDP 连接时，按键信号经过了一层远程传输协议的转换。在这种复杂的链路下，Vim（尤其是原生 Windows 版的 `vim.exe`）往往无法正确解析这些现代的 VT 序列，导致它不知道你按了什么。

    3. **旧版模式（Legacy Input）：** 打开这个开关后，终端会强迫输入流模拟旧版的 **Win32 控制台模式**。它不再发送复杂的 `ESC` 编码，而是直接发送传统的控制字符（如 `0x08` 代表 Backspace）。
* win 11 中的同名 sudo

    ```shell
    wsdlh@abchhh /cygdrive/c/home/wsdlh
    $ sudo vim 'C:\ProgramData\ssh\sshd_config'
    ```

    > 已在此计算机上禁用 Sudo。若要启用它，请转到“设置”应用中的 Developer Settings page

    这是一个非常有趣的“撞车”现象。你看到的这个错误提示**不是来自 Cygwin**，而是来自 **Windows 11 原生自带的 `sudo` 命令**。

    由于你的 `PATH` 变量中包含了 Windows 的系统目录，当你输入 `sudo` 时，系统优先调用了 Windows 最近新增的内置 `sudo.exe`，而不是你在 Cygwin 里设定的别名或脚本。

    要解决这个问题并成功编辑文件，你有两个选择：

    **方案一：启用 Windows 原生 Sudo（最快）**

    既然 Windows 已经提示你了，直接开启它即可：
    1.  打开 Windows **设置 (Settings)**。
    2.  前往 **系统 (System) > 开发者选项 (For developers)**。
    3.  找到 **启用 Sudo (Enable Sudo)** 开关并打开。
    4.  回到 Cygwin，再次运行你的命令。

    ---

    **方案二：绕过 Sudo，直接以管理员身份运行**

    在 Windows/Cygwin 环境下，修改 `sshd_config` 这种系统文件最稳妥的方法是不在终端里“提权”，而是直接开启一个“高权限终端”：

    1.  关闭当前的 Cygwin 窗口。
    2.  **右键点击** Cygwin 快捷方式（图标），选择 **“以管理员身份运行”**。
    3.  直接输入命令（不需要加 sudo）：
        ```bash
        vim /cygdrive/c/ProgramData/ssh/sshd_config
        ```

    ---

    **方案三：如果你坚持要在 Cygwin 模拟 Linux 风格**

    如果你想让 `sudo` 这个词指向你自定义的行为，而不是 Windows 自带的那个，你可以在 `.bashrc` 里强制指定别名。

    但在你目前的场景下，最简单且不报错的临时方法是使用 **PowerShell 提权调用**：
    ```bash
    powershell -Command "Start-Process vim -ArgumentList 'C:\ProgramData\ssh\sshd_config' -Verb RunAs"
    ```

    **一个小提示**

    `C:\ProgramData\ssh\sshd_config` 的权限非常严格。即使你用了 `sudo`，有时 Windows 仍然会因为文件所有权（Ownership）属于 `SYSTEM` 而阻止修改。

    如果修改失败，建议：
    * 先在管理员权限的窗口下，使用 `ls -l` 查看权限。
    * 或者直接在 Windows 资源管理器里右键该文件 -> 属性 -> 安全，确认你的账户有“写入”权限。

    你开启“开发者模式”中的 Sudo 选项了吗，还是更倾向于直接用管理员身份打开终端？

* win 11 的 ssh 登陆密码

    * 如果使用邮箱作为 win 11 的账号，那么 ssh 的密码与邮箱账号密码相同，ssh 的用户名可以通过`whoami`看到

    * 关闭“仅允许 hello 登录”的设置，并重启电脑 

        > 为了提高安全性，仅允许此设备上的 Microsoft 账户使用 Windows Hello 登录

        只重启 ssh service 没用。

* winsw: Windows Service Wrapper in a permissive license

    <https://github.com/winsw/winsw?tab=readme-ov-file>

    快速生成 windows server。

    usage example: <https://github.com/winsw/winsw/blob/v3/docs/cli-commands.md#install-command>

* WinFsp · Windows File System Proxy

    <https://github.com/winfsp/winfsp>

    在cygwin 里安装 sshfs 时需要这个库。

* windows在重启时有时会因为程序正在运行而卡住无法关机，如何保证能一定重启成功

    我来分享几种确保Windows能强制重启成功的方法，按推荐顺序排列：

    一、常规方法（先尝试）

    1. **任务管理器强制结束**

        - `Ctrl + Shift + Esc` 打开任务管理器
        - 查看“进程”选项卡，结束无响应的应用
        - 特别关注：用户进程、后台进程

    2. **使用重启命令**

        ```cmd
        shutdown /r /f /t 0
        ```

        - `/r`：重启
        - `/f`：强制关闭正在运行的应用程序
        - `/t 0`：0秒后执行

    二、快捷操作方案

    **创建强制重启快捷方式**

    1. 桌面右键 → 新建 → 快捷方式
    2. 输入位置：`shutdown /r /f /t 0`
    3. 命名后使用管理员权限运行

    **批处理脚本**

    创建 `force_restart.bat`：

    ```batch
    @echo off
    echo 正在强制重启...
    shutdown /r /f /t 5
    ```

    三、进阶解决方案

    1. **组策略设置（专业版以上）**

        ```
        gpedit.msc → 计算机配置 → 管理模板 → 系统 → 关机选项
        → "关闭会阻止或取消关机的应用程序的自动终止功能" → 禁用
        ```

    2. **修改注册表**

        **警告：修改前备份注册表**

        ```
        HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System
        新建 DWORD (32位) 值：FastReboot
        数值数据设为：1
        ```

    3. **检查关机脚本冲突**

        ```
        gpedit.msc → 计算机配置 → Windows设置 → 脚本(启动/关机)
        检查是否有错误的关机脚本
        ```

    四、物理方法（最后手段）

    1. **长按电源键**

        - 按住电源按钮4-8秒强制断电
        - **注意**：可能造成数据丢失，尽量避免

    2. **硬件重启按钮**

        - 有些机箱有专门的重启按钮

    五、预防措施

    1. **禁用快速启动**

        ```
        控制面板 → 电源选项 → 选择电源按钮的功能
        → 更改当前不可用的设置 → 取消“启用快速启动”
        ```

    2. **定期维护**

        ```powershell
        # 检查系统文件
        sfc /scannow

        # 检查磁盘错误
        chkdsk C: /f
        ```

    3. **软件优化**

        - 减少开机启动项（`任务管理器 → 启动`）
        - 卸载冲突软件
        - 更新驱动程序

    ## 推荐方案组合：

    1. **日常使用**：`shutdown /r /f /t 0` 命令
    2. **预防为主**：禁用快速启动 + 定期清理启动项
    3. **紧急情况**：长按电源键（万不得已时）

    **最佳实践**：先尝试软件层面的强制重启，如果频繁发生卡住问题，建议检查系统日志（`eventvwr.msc`）查看具体是哪个程序导致的问题。

* windows 如何运行 bash

    在 Windows 上运行 Bash 有以下几种主要方法：

    1. WSL (Windows Subsystem for Linux) - 推荐

        这是最集成、最强大的解决方案。

        WSL 2 (最新版本)

        ```bash
        # 安装步骤：
        1. 以管理员身份打开 PowerShell
        2. 运行：wsl --install
        3. 重启电脑
        4. 从 Microsoft Store 安装 Linux 发行版（如 Ubuntu）

        # 或者手动安装：
        wsl --install -d Ubuntu
        ```

        基本使用

        ```bash
        # 在 PowerShell 或 CMD 中输入
        wsl
        # 或直接运行具体命令
        wsl ls -la
        ```

    2. Git Bash

        适合 Git 用户，轻量级。

        ```bash
        # 下载地址：https://gitforwindows.org/
        # 安装后，可以在右键菜单找到 "Git Bash Here"
        ```

    3. Cygwin

        提供完整的 Linux 环境。

        ```bash
        # 下载地址：https://www.cygwin.com/
        # 安装时选择需要的包
        ```

    4. MSYS2

        类似 Cygwin，但使用 Arch Linux 的包管理。

        ```bash
        # 下载地址：https://www.msys2.org/
        ```

    5. Docker

        运行 Linux 容器。

        ```powershell
        # 安装 Docker Desktop for Windows
        docker run -it ubuntu bash
        ```

    6. Windows Terminal - 增强终端体验

        ```bash
        # Microsoft Store 安装 Windows Terminal
        # 支持 WSL、PowerShell、CMD 等多标签
        ```

    对比推荐

    | 方案 | 适合场景 | 性能 | 集成度 |
    | - | - | - | - |
    | WSL 2 | 开发、学习 Linux | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
    | Git Bash | Git 操作、简单脚本 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
    | Cygwin/MSYS2 | 需要兼容 POSIX | ⭐⭐⭐ | ⭐⭐⭐ |

    快速开始建议

    如果你是开发者，强烈推荐 WSL 2：

    * Windows 10/11 都支持

    * 几乎原生 Linux 体验

    * 可以直接访问 Windows 文件系统

    * 支持 systemd、Docker 等

        ```powershell
        # 检查 WSL 状态
        wsl --list --verbose

        # 设置默认版本
        wsl --set-default-version 2

        # 更新 WSL
        wsl --update
        ```

    选择哪种方式主要取决于你的具体需求和使用场景。


* bash.exe -i -l 这个是什么意思

    bash.exe -i -l 是一个启动 Bash shell 时常用的参数组合，让我详细解释每个部分：

    参数分解

    bash.exe

    ```bash
    # 基本的 Bash 可执行文件
    bash.exe          # 启动非交互式shell
    bash.exe script.sh # 执行脚本
    ```

    -i 参数 (interactive)

    ```bash
    # 交互模式
    bash -i
    ```

    作用：

    * 启用交互式 shell

    * 读取并执行 ~/.bashrc 配置文件

    * 启用命令历史、补全、别名等交互功能

    * 显示 PS1 提示符（如 user@host:~$）

    示例对比：

    ```bash
    # 非交互式 (默认)
    bash.exe           # 直接返回，没有提示符
    echo $PS1         # 空值

    # 交互式
    bash.exe -i        # 显示提示符，可以输入命令
    echo $PS1         # 显示提示符配置
    ```

    -l 参数 (login)

    ```bash
    # 登录模式
    bash -l
    ```

    作用：

    * 模拟登录 shell

    * 读取并执行 ~/.bash_profile、~/.profile 或 ~/.bash_login

    * 设置环境变量如 HOME, PATH, USER 等

    * 更完整的初始化流程

    重要区别：

    ```bash
    # 普通shell vs 登录shell
    ┌───────────────────┬─────────────────────┐
    │    bash -i        │    bash -l         │
    ├───────────────────┼─────────────────────┤
    │ 读取 .bashrc      │ 读取 .profile      │
    │ 继承当前环境      │ 全新环境初始化     │
    │ 适合子shell       │ 适合新会话         │
    └───────────────────┴─────────────────────┘
    ```

    组合使用：bash -i -l

    完整效果

    ```bash
    # 同时启用交互和登录模式
    bash -i -l
    # 等价于
    bash --login -i
    ```

    会执行：

    * 登录初始化：执行 ~/.bash_profile、~/.profile

    * 交互初始化：执行 ~/.bashrc

    * 环境设置：完整的登录环境变量

    * 交互功能：提示符、补全、历史等

    实际案例

    ```powershell
    # Windows Terminal 配置示例
    {
      "commandline": "C:/cygwin64/bin/bash.exe -i -l",
      "name": "Cygwin Bash",
      "startingDirectory": "C:/cygwin64/home/%USERNAME%"
    }
    ```

    为什么这样配置？

    ```bash
    # 如果没有 -i -l：
    1. 提示符可能显示为：bash-3.2$
    2. 别名不会生效：ll, la 等不可用
    3. PATH 可能不完整：找不到 cygwin 命令
    4. 颜色和主题不生效

    # 使用 -i -l 后：
    1. 显示完整提示符：user@host ~
    2. 所有别名生效
    3. 完整 PATH：/usr/local/bin:/usr/bin:/bin
    4. 彩色输出和主题
    ```

    参数顺序重要性

    ```bash
    # 顺序不同，效果相同
    bash -i -l    # ✓ 正确
    bash -l -i    # ✓ 也正确
    bash --login -i # ✓ 显式写法

    # 错误写法
    bash -il      # ✓ 也可以（短参数合并）
    bash -i -l script.sh # -l 对脚本影响有限
    ```

    常用组合对比

    | 命令 | 执行的文件 | 适用场景 |
    | - | - | - |
    | bash | 无 | 执行单个命令：bash -c "ls -la" |
    | bash -i | .bashrc | 交互但不重新初始化环境 |
    | bash -l | .profile .bashrc | 登录 shell，完整初始化 |
    | bash -i -l | .profile .bashrc | 最完整的交互式登录 shell |
    | bash --norc | 无 | 跳过所有初始化（调试用） |

    验证区别

    ```bash
    # 测试当前 shell 类型
    echo $0        # 显示 shell 名称
    shopt -q login_shell && echo "Login shell" || echo "Not login shell"
    echo $-        # 包含 'i' 表示交互式

    # 测试环境变量
    echo $PATH    # 比较不同模式的 PATH
    ```

    在 Windows 中的特殊考虑

    ```powershell
    # Cygwin/MSYS2 中尤其需要 -l
    # 因为需要设置：
    # 1. Cygwin 的根路径 (/cygdrive/c)
    # 2. 正确的 HOME 目录
    # 3. TMP/TEMP 变量转换
    # 4. Windows PATH 到 Unix PATH 的转换

    # 如果没有 -l，可能会遇到：
    # - 命令找不到（PATH 不对）
    # - 家目录错误
    # - 权限问题
    ```

    简洁写法

    ```bash
    # 等效的短写法
    bash -il      # 合并参数
    bash -li      # 顺序无关

    # 明确的写法（推荐）
    bash --login -i
    ```

    总结：bash -i -l 在 Windows 终端配置中非常常见，因为它确保了 Cygwin/MSYS2 的 Bash 能够以最完整、最兼容的方式启动，提供接近 Linux 终端的体验。

* windows office 替代

    🆓 免费开源替代品

        LibreOffice

            Writer（替代Word）

            Calc（替代Excel）

            Impress（替代PowerPoint）

            完全免费开源，兼容性优秀，支持多种格式

        Apache OpenOffice

            功能与LibreOffice类似

            经典的开源办公套件

        FreeOffice

            界面更接近MS Office

            个人使用免费

    💰 付费商业软件

        WPS Office

            个人版完全免费

            界面与Office高度相似

            云端同步功能

            对中文支持特别好

        SoftMaker Office

            专业级办公软件

            兼容性极佳

    ☁️ 在线办公套件

        Google Workspace

            Google Docs（在线Word）

            Google Sheets（在线Excel）

            Google Slides（在线PPT）

            需要网络，协作功能强大

        Microsoft 365 网页版

            免费使用基础功能

            需要微软账户

        Zoho Office Suite

            功能全面的在线办公套件

            有桌面客户端

    🎯 特色工具组合

        OnlyOffice - 开源协作平台

        Polaris Office - 移动端表现优秀

        Apple iWork（网页版） - 设计精美，可通过浏览器使用

    📊 专业领域工具

        LaTeX（学术论文排版）

        Markdown编辑器（Typora、Obsidian等）

        专业图表工具（Draw.io、yEd等）

    💡 选择建议

        追求完全免费 → LibreOffice

        需要高度兼容Office → WPS Office

        注重团队协作 → Google Workspace

        轻度用户 → 在线版Office 365

    大部分替代软件都支持打开和保存为Office格式（.docx, .xlsx, .pptx），但复杂格式可能有些差异，建议根据具体需求选择试用。

* windows 自带的 openssh-server，似乎只能把 public key 放到

    `c:/ProgramData/ssh/authorized_keys`中才能生效。

    放到`~/.ssh/authorized_keys`中不行。

## note

修改 cmd 的编码为 UTF-8：<https://blog.csdn.net/mantou_riji/article/details/122313517>

修改 powershell 的编码为 UTF-8：<https://blog.csdn.net/sxzlc/article/details/104880570>

插入 unicode 字符：<https://support.microsoft.com/en-us/office/insert-ascii-or-unicode-latin-based-symbols-and-characters-d13f58d3-7bcb-44a7-a4d5-972ee12e50e0>

unicode 编码查询网站：<https://www.qqxiuzi.cn/bianma/zifuji.php>，很好用

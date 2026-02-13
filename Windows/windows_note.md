# Windows Note

## cache

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

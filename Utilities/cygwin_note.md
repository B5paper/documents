# cygwin note

## cache

* cygwin 如何设置可以让 /d 或 /D 表示 d 盘

    在Cygwin中让 /d 或 /D 表示D盘，有几种方法可以实现：

    方法1：创建符号链接（推荐）

    在Cygwin终端中执行：

    ```bash
    # 创建链接到D盘
    ln -s /cygdrive/d /d
    ln -s /cygdrive/d /D
    ```

    方法2：修改 ~/.bashrc 或 ~/.bash_profile

    编辑 ~/.bashrc 文件：

    ```bash
    # 使用别名
    alias d:='cd /cygdrive/d'
    alias D:='cd /cygdrive/d'

    # 或者创建目录链接
    if [ ! -e /d ]; then
        ln -s /cygdrive/d /d 2>/dev/null
    fi
    if [ ! -e /D ]; then
        ln -s /cygdrive/d /D 2>/dev/null
    fi
    ```

    方法3：修改 /etc/fstab（需要管理员权限）

    编辑 /etc/fstab 文件，添加：

    ```text
    # 映射D盘
    C:/cygwin64/d /d none bind,posix=0,user
    C:/cygwin64/d /D none bind,posix=0,user
    ```

    或者更简单的方式：

    ```bash
    mount -b /cygdrive/d /d
    mount -b /cygdrive/d /D
    ```

    方法4：使用 cygpath 命令创建脚本

    创建 /usr/local/bin/d: 脚本：

    ```bash
    #!/bin/bash
    cd /cygdrive/d
    ```

    方法5：设置 CYGWIN 环境变量

    在Windows环境变量中添加：

    ```text
    CYGWIN=winsymlinks:nativestrict
    ```

    然后在Cygwin中创建链接：

    ```bash
    ln -s /cygdrive/d /d
    ```

    永久生效的方法

    创建 /etc/profile.d/mounts.sh：

    ```bash
    #!/bin/bash
    # 检查并创建链接
    [ ! -e /d ] && ln -s /cygdrive/d /d 2>/dev/null
    [ ! -e /D ] && ln -s /cygdrive/d /D 2>/dev/null

    # 或者使用别名
    alias d:='cd /d 2>/dev/null || cd /cygdrive/d'
    alias D:='cd /D 2>/dev/null || cd /cygdrive/d'
    ```

    推荐使用方法1，最简单直接。创建符号链接后，你就可以在Cygwin中使用 /d 或 /D 来访问D盘了。

* cygwin 的字体渲染没有 windows 自带的 terminal 渲染好看，为什么

    这是一个很好的观察！Cygwin 终端（通常使用 mintty）和 Windows Terminal 在字体渲染上确实有明显差异，主要原因如下：

    1. 不同的渲染引擎

        ```bash
        # Cygwin/mintty 使用：
        - 基于 X Window 系统的渲染方式
        - 传统的位图字体渲染
        - 缺少 Windows 现代抗锯齿技术

        # Windows Terminal 使用：
        - DirectWrite/DirectX 渲染引擎
        - 子像素抗锯齿 (ClearType)
        - 完整的字体 hinting 支持
        ```

    2. 技术架构差异

        ```bash
        Cygwin 终端:
        ┌─────────────────┐
        │    Cygwin DLL   │ ← POSIX 兼容层
        │   Xlib/mintty   │ ← X Window 风格渲染
        │    GDI 绘图     │ ← 老式 Windows GDI
        └─────────────────┘

        Windows Terminal:
        ┌─────────────────┐
        │  DirectWrite    │ ← 现代字体渲染
        │  DirectX 加速   │ ← GPU 加速
        │  WinUI 框架     │ ← 原生 Windows UI
        └─────────────────┘
        ```

    3. 字体 hinting 处理

        ```yaml
        # Cygwin/mintty:
        - 使用 FreeType 库的 hinting
        - 可能不兼容 Windows 的 hinting 指令
        - 尤其是中文字体表现差异明显

        # Windows Terminal:
        - 完全使用 Windows 字体系统
        - 支持所有 Windows 字体特性
        - 完美显示雅黑、等宽字体
        ```

    4. 解决方案

        方案一：优化 Cygwin 字体设置

        ```bash
        # 在 mintty 中：
        1. 右键标题栏 → Options → Text
        2. 尝试以下字体：
           - Consolas
           - Cascadia Code
           - DejaVu Sans Mono
           - Source Code Pro
        3. 开启抗锯齿 (Anti-aliasing)
        ```

        方案二：在 Windows Terminal 中使用 Cygwin

        ```json
        // Windows Terminal settings.json
        {
          "profiles": {
            "list": [
              {
                "name": "Cygwin",
                "commandline": "C:/cygwin64/bin/bash.exe -i -l",
                "startingDirectory": "C:/cygwin64/home/%USERNAME%",
                "fontFace": "Cascadia Code PL",
                "fontSize": 11,
                "useAcrylic": true
              }
            ]
          }
        }
        ```

        方案三：使用更好的终端

        ```bash
        # 替代 Cygwin 终端：
        1. Windows Terminal + Cygwin bash
        2. Tabby (前身 Terminus)
        3. ConEmu/Cmder
        4. Alacritty
        ```

        方案四：调整 Cygwin 显示设置

        ```bash
        # 创建或修改 ~/.minttyrc
        Font=Cascadia Code
        FontHeight=11
        FontSmoothing=full
        FontWeight=400
        CursorType=block
        ```

    5. 为什么会有这种差异

        ```markdown
        历史原因：
        • Cygwin 诞生于 1995 年，当时 Windows 字体渲染还很基础
        • mintty 最初为 Cygwin 设计，保留了 Unix 风格
        • Windows Terminal 是 2019 年的现代产物

        技术债务：
        • Cygwin 要兼容太多 Unix 特性
        • 不能完全利用 Windows 现代图形 API
        • 维护重点是功能兼容性而非 UI 美观
        ```

    6. 推荐做法

        如果你想要既美观又能运行 Cygwin：

        ```bash
        # 最佳组合：
        1. 安装 Windows Terminal (Microsoft Store)
        2. 安装 Cascadia Code 字体 (GitHub)
        3. 配置 Windows Terminal 使用 Cygwin bash
        4. 享受现代渲染 + Cygwin 功能
        ```

    简单来说：Cygwin 的重点是提供完整的 POSIX 环境，而渲染美观是 Windows Terminal 这种原生现代应用的优势。把它们结合起来使用是最佳选择！

* 如果你在 Cygwin 中使用 Git for Windows：

    ```bash
    # 确保使用正确的 SSH
    where ssh
    which ssh

    # 可能需要指定使用 Windows 的 SSH
    export GIT_SSH=/usr/bin/ssh
    ```

* cygwin 中，如何执行 sudo 命令？比如 sudo vim /etc/hosts

    在 Cygwin 中，没有原生的 sudo 命令，因为 Windows 没有 Unix/Linux 的权限系统。不过有几种替代方案：

    1. 直接以管理员身份运行 Cygwin

        这是最简单的方法：

        * 右键点击 Cygwin 终端图标

        * 选择 “以管理员身份运行”

        * 然后所有命令都拥有管理员权限

        * 此时可以直接编辑系统文件：vim /etc/hosts

    2. 安装替代的 sudo 工具

        方法 A：通过 Cygwin 安装包

        ```bash
        # 在 Cygwin 安装程序中搜索并安装：
        # - cygstart
        # - sudo (可能会有)
        ```

        方法 B：使用 cygstart

        ```bash
        # 编辑 hosts 文件
        cygstart --action=runas vim /etc/hosts

        # 或使用 notepad（Windows 方式）
        cygstart --action=runas notepad C:/Windows/System32/drivers/etc/hosts
        ```

    3. 创建 sudo 别名（推荐）

        在 ~/.bashrc 中添加：

        ```bash
        # 简化版 sudo
        sudo() {
            if [ "$1" = "vim" ] || [ "$1" = "vi" ]; then
                local file="$2"
                # 将 Cygwin 路径转换为 Windows 路径
                if [[ "$file" == /* ]]; then
                    file="$(cygpath -w "$file")"
                fi
                cygstart --action=runas vim "$file"
            else
                cygstart --action=runas "$@"
            fi
        }
        ```

        然后重新加载配置：

        ```bash
        source ~/.bashrc
        ```

    4. 手动编辑 hosts 文件

        由于权限问题，可以直接：

        ```bash
        # 使用 Windows 路径
        notepad C:/Windows/System32/drivers/etc/hosts

        # 或者以管理员身份启动编辑器
        runas /user:Administrator "notepad C:\Windows\System32\drivers\etc\hosts"
        ```

    5. 安装替代方案

        ```bash
        # 通过 apt-cyg 安装（如果有）
        apt-cyg install cygsudo

        # 或者从源代码编译安装
        ```

    建议方案

    对于你的具体需求，最简单的方法是：

    * 以管理员身份运行 Cygwin 终端

    * 直接使用：vim /etc/hosts

    或者在普通终端中使用：

    ```bash
    cygstart --action=runas vim /etc/hosts
    ```

    注意事项

    * Cygwin 的 /etc/hosts 实际上是 Windows hosts 文件的符号链接

    * 直接编辑 Windows 的 hosts 文件可能更方便：C:\Windows\System32\drivers\etc\hosts

    * 修改 hosts 文件后，可能需要刷新 DNS：ipconfig /flushdns（在 CMD 中运行）

* 简述 cygstart 的作用和用法

    cygstart 作用

    cygstart 是 Cygwin 中的一个命令行工具，主要用于：

    核心功能

    * 启动 Windows 应用程序

    * 打开文件（使用关联的应用程序）

    * 打开 URL（使用默认浏览器）

    * 以管理员权限运行程序（替代 sudo）

    基本用法

    1. 打开文件/URL

        ```bash
        # 打开文件（使用关联程序）
        cygstart filename.pdf
        cygstart image.jpg

        # 打开 URL
        cygstart https://www.example.com
        cygstart http://localhost:8080
        ```

    2. 启动应用程序

        ```bash
        # 启动 Windows 程序
        cygstart notepad
        cygstart calc              # 计算器
        cygstart cmd.exe           # CMD 命令行
        cygstart explorer .        # 打开当前目录的资源管理器
        ```

    3. 以管理员身份运行（关键用途）

        ```bash
        # 以管理员权限运行程序
        cygstart --action=runas program [args]

        # 示例
        cygstart --action=runas notepad C:/Windows/System32/drivers/etc/hosts
        cygstart --action=runas cmd
        ```

    常用参数

    | 参数 | 说明 |
    | - | - |
    | --action=runas | 以管理员身份运行（最重要） |
    | --action=edit | 编辑文件 |
    | --action=print | 打印文件 |
    | --action=open | 打开文件（默认） |
    | --working-directory=DIR | 设置工作目录 |
    | --show | 窗口显示状态（min/max/normal） |

    实用示例

    替代 sudo 的常用场景

    ```bash
    # 1. 编辑需要管理员权限的文件
    cygstart --action=runas vim /etc/hosts
    cygstart --action=runas "notepad C:/Windows/System32/drivers/etc/hosts"

    # 2. 运行需要权限的命令
    cygstart --action=runas cmd /k "ipconfig /flushdns"

    # 3. 安装软件
    cygstart --action=runas "msiexec /i package.msi"
    ```

    路径处理

    ```bash
    # Cygwin 路径自动转换
    cygstart ~/document.txt           # 自动转为 Windows 路径
    cygstart /cygdrive/c/Users/file   # 自动转换

    # 混合使用
    cygstart "$(cygpath -w ~/file.txt)"
    ```

    窗口控制

    ```bash
    # 最小化启动
    cygstart --show=min cmd

    # 最大化启动
    cygstart --show=max notepad
    ```

    与 sudo 对比的实用技巧

    创建 sudo 别名

    在 ~/.bashrc 中：

    ```bash
    # 简单 sudo 替代
    alias sudo='cygstart --action=runas'

    # 更智能的版本
    sudo() {
        if [ $# -eq 0 ]; then
            cygstart --action=runas bash
        else
            cygstart --action=runas "$@"
        fi
    }
    ```

    执行批处理

    ```bash
    # 运行 CMD 命令
    cygstart cmd /c "echo Hello & pause"

    # 运行 PowerShell
    cygstart powershell -Command "Get-Service"
    ```

    注意事项

    * 路径分隔符：可以使用 / 或 \，Cygwin 会自动处理

    * 引号处理：参数包含空格时需要引号

    * 返回值：cygstart 立即返回，不等待程序结束

    * 环境变量：会继承当前 Cygwin 环境的部分变量

    典型工作流示例

    ```bash
    # 1. 查看 hosts 文件
    cat /etc/hosts

    # 2. 需要修改时，用管理员权限编辑
    cygstart --action=runas vim /etc/hosts

    # 3. 完成后刷新 DNS（也需要管理员）
    cygstart --action=runas cmd /c "ipconfig /flushdns"
    ```

    cygstart --action=runas 是在 Cygwin 中获得管理员权限的最简单、最标准的方法。

* cygwin 下 git clone ssh-type repo 的问题

    * windows cygwin 下默认 ssh 使用的是 windows 自带的 ssh，因此使用的 ssh 配置为 `c:/users/<user_name>/.ssh/xxx`。
    
    * cygwin 下的 git 默认也是使用之前在 windows 上安装的 git (windows version)，但是这个 git 在 clone ssh 类型的 repo 时，会使用`~/.ssh/`下的配置。
    
    * `~`目录在 cygwin 下会被展开成`/home/<user>/`，这个目录并不是`c:/users/<user_name>/`

    * git 会认为 ssh 配置为空，或者不对

    * 用户此时使用 ssh 连接 remote server，发现连接是正常的

    * 解决办法为，指定 git 使用指定目录下的 ssh 配置

## topics

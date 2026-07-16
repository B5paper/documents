# Ubuntu Note

这个文件主要存放 ubuntu 特有的笔记。

## cache

* xtightvncviewer 适用于 vulkan 吗

    很好的问题！这是一个涉及技术细节的疑问。

    简单直接的回答是：**不直接适用，但可以有间接的、有限的配合。**

    下面为您详细解释一下：

    ### 核心原因：架构层级不同

    1.  **TightVNC / X11 的架构**

        * **xtightvncviewer** 是一个 VNC 客户端，它连接的是运行在远程 Linux 机器上的 **X VNC 服务器**。

        * 这个 VNC 服务器是 **X Window System** 的一个“虚拟”屏幕。所有在远程机器上运行的图形应用程序（无论是使用 OpenGL、X11 原生绘图还是其他工具包），最终都会将其图形输出渲染到这块 X11 的“虚拟屏幕”上。

        * VNC 服务器的工作是抓取这块虚拟屏幕的帧缓冲区，进行编码（如 Tight 编码），然后通过网络发送给 VNC 客户端。

        * 在这个过程中，VNC 对应用程序使用的是 **哪个 3D API（OpenGL, Vulkan）是完全不知情的**。它只看到最终的 2D 像素结果。

    2.  **Vulkan 的架构**

        * Vulkan 是一个低级的、跨平台的 3D 图形和计算 API。

        * 为了在屏幕上显示，Vulkan 需要一个“窗口系统集成”组件。在 Linux 上，这通常是通过 **X11** 或 **Wayland** 扩展来实现的（例如 `VK_KHR_xlib_surface` 或 `VK_KHR_wayland_surface`）。

        * 当 Vulkan 应用程序在带有 X11 的 VNC 会话中运行时，它会通过 X11 扩展创建表面，并将其内容渲染到 X11 的窗口中。这个渲染过程可能由 GPU 加速，但最终的合成和显示仍然由 X Server 管理。

    ### 间接的“适用”场景

    1.  **在 X11 VNC 会话中运行 Vulkan 应用程序**：
        *   这是完全可行的。你可以在远程机器上启动一个 VNC 服务器（如 `Xvnc`），然后在这个 VNC 会话中启动一个 Vulkan 程序（例如 `vulkan-smoketest`）。
        *   **发生了什么**：Vulkan 程序通过 X11 正常渲染，VNC 服务器抓取整个桌面（包括 Vulkan 程序窗口）的图像，发送给 `xtightvncviewer`。
        *   **局限性**：
            *   **性能**：虽然 Vulkan 渲染本身可能在远程 GPU 上高效运行，但最终的图像需要被 VNC 抓取、编码、传输，然后在客户端解码显示。这个过程会带来显著的延迟和性能损失，特别是对于高速变化的 3D 画面。它**远不如**在本地直接运行流畅。
            *   **功能**：VNC 协议本身并不理解 3D 或 Vulkan，它只是传输像素。因此无法利用任何针对 3D 应用的远程渲染优化。

    2.  **VirGL (虚拟化 GPU) **：
        *   在虚拟化环境中（例如 QEMU/KVM），有一个名为 **VirGL** 的项目。它允许虚拟机中的 Guest OS 使用 OpenGL，并将 OpenGL 命令通过网络传输给 Host OS 的 GPU 进行硬件加速渲染。
        *   虽然 VirGL 主要针对 OpenGL，但其思想与你的问题相关：**将图形 API 命令而非像素进行传输**。这是一种更高效的远程 3D 图形方案。
        *   目前，**没有与 VirGL 等效的、成熟且广泛支持的 “Vulkan over VNC” 解决方案**。

    ### 结论与建议

    *   **对于简单的测试或非实时应用**：你可以使用 `xtightvncviewer` 连接到一台运行 Vulkan 程序的 Linux VNC 服务器。它能工作，但体验不会好。
    *   **对于需要高性能的 3D/Vulkan 远程访问**：VNC（包括 TightVNC）不是正确的工具。你应该考虑以下替代方案：

        *   **Steam Remote Play / Sunshine + Moonlight**：即使是非游戏应用，这个组合也提供了极低的延迟和高效的视频流传输，对 Vulkan 和 DX11/12 支持良好。
        *   **Parsec**：另一个优秀的低延迟远程桌面工具，非常适合游戏和图形应用。
        *   **NoMachine**：对多媒体和 3D 图形支持比传统 VNC 好得多。
        *   **XRDP**：使用微软的 RDP 协议，其 RemoteFX 组件可以提供更好的 GPU 加速体验。

    **总结：**

    `xtightvncviewer` 本身不“适用于” Vulkan，它只是被动地显示 Vulkan 应用程序在 X11 会话中产生的最终图像。对于真正的硬件加速、低延迟的 Vulkan 远程访问，你需要使用基于视频流编码的现代远程桌面技术，而不是基于屏幕像素抓取的 VNC。

    * 那么 xtightvncviewer 适用于 wayland 吗

        这是一个非常好的问题，它触及了 Linux 图形系统演进的核心。

        直接的回答是：**不，标准的 `xtightvncviewer` 本身不适用于原生的 Wayland 环境。**

        原因比 Vulkan 的情况更根本，原因如下：

        ### 核心原因：架构的根本性冲突

        1.  **Wayland 没有“远程”协议**
            *   **X11** 从设计上就是一个**网络透明的**协议。它的核心就是一个客户端-服务器模型，应用程序（客户端）和显示服务器（服务器）可以轻松地运行在不同的机器上。VNC 服务器（如 `Xvnc`）可以伪装成一个 X11 服务器，接收所有图形指令，这是 X11 与生俱来的能力。
            *   **Wayland** 的设计哲学完全不同。它是一个**简单的、非网络的**显示服务器协议。Wayland 客户端（应用程序）和 Wayland 合成器（相当于服务器）通过本地 Unix Socket 进行通信，并且高度依赖共享内存等技术来传递像素数据。它**没有内置的网络支持**。

        2.  **Wayland 合成器角色转变**
            *   在 X11 中，VNC 服务器可以“抓取”整个屏幕的帧缓冲区。
            *   在 Wayland 中，**合成器就是显示服务器本身**。它直接管理窗口、渲染最终画面。没有一个独立的、统一的“帧缓冲区”供外部工具（如 VNC 服务器）来抓取。出于安全考虑（防止屏幕窃取），Wayland 也**不允许**任意程序直接抓取其他程序窗口或整个屏幕的内容。

        ### 如何实现 Wayland 下的 VNC 访问？

        既然原生不行，那解决方案就是让 **Wayland 合成器本身具备 VNC 服务器的功能**。这不再是运行一个独立的 `Xvnc` 进程，而是需要你的 Wayland 合成器**内置**一个 VNC 模块或插件。

        目前主流的 Wayland 合成器实现情况：

        *   **GNOME (使用 Mutter 合成器)**
            *   GNOME 的 Mutter 合成器**内置了 VNC 支持**。
            *   你可以在设置中（“共享”或“远程桌面”）直接**启用屏幕共享**，它会在后台启动一个 VNC 服务器。
            *   然后，你就可以使用 `xtightvncviewer` 或其他任何标准的 VNC 客户端连接到这台机器了。
            *   这是目前最成熟、最方便的解决方案。

        *   **KDE Plasma (使用 KWin 合成器)**
            *   KDE 的 KWin 也提供了类似的**远程桌面**功能，同样内置了 VNC 支持。
            *   你可以在系统设置中轻松开启它。

        *   **wlroots (Sway, Hyprland 等的基础库)**
            *   `wlroots` 提供了一个官方插件 `wlr-vnc`。
            *   对于使用基于 wlroots 的合成器（如 Sway），你可以在启动配置中加载这个插件，或者运行一个独立的 `wlr-vnc` 命令来启动 VNC 服务器。
            *   例如，启动 Sway 时加载 VNC：`sway --vnc 0.0.0.0:5900`

        *   **Weston (参考 Wayland 合成器)**
            *   Weston 本身也内置了 VNC 后端，启动时可以使用 `--backend vnc` 选项。

        ### 总结与工作流程

        所以，`xtightvncviewer` 在 Wayland 下的角色并没有变，它依然是一个优秀的 VNC **客户端**。变化的是 **VNC 服务器** 的来源：

        | 图形系统 | VNC 服务器来源 | 客户端 | 兼容性 |
        | :--- | :--- | :--- | :--- |
        | **X11** | 独立的 `Xvnc` 进程 | `xtightvncviewer` | **原生支持** |
        | **Wayland** | **Wayland 合成器内置的 VNC 模块** | `xtightvncviewer` | **通过合成器支持** |

        **结论：**

        你不能在纯粹的 Wayland 环境下运行一个独立的、传统的 VNC 服务器（如 `Xvnc`）来共享桌面。但是，你可以通过启用你的 **Wayland 桌面环境（GNOME, KDE）或合成器（Sway via wlroots）内置的 VNC 服务器功能**，然后使用 `xtightvncviewer` 这个客户端来顺利连接。

        因此，`xtightvncviewer` 作为客户端，可以用于连接到由 Wayland 合成器提供的 VNC 服务，但它本身并不“理解” Wayland 协议。

* linux上有哪些软件可以像手机上截图编辑或者微信截图编辑小工具那样，实现涂鸦、画框框、输入文字这些小功能

    当然有！Linux 上有非常多优秀的工具可以实现类似手机或微信截图编辑的功能，从轻量级到功能全面的都有。这些工具通常分为两类：**截图工具自带编辑功能** 和 **独立编辑工具配合截图工具使用**。

    下面我为你分类推荐一些主流和好用的选择：

    ### 一、 截图工具自带简单编辑功能

    这类工具最符合你的需求，截图后立刻可以进行涂鸦、画框、标注和文字输入。

    #### 1. Flameshot - **强烈推荐**
    这几乎是 Linux 上功能最全面、最受好评的截图工具，完美契合你的所有要求。
    - **功能特点**：
      - **涂鸦**：铅笔、画线（带箭头）、矩形、圆形。
      - **标注**：高亮区域、马赛克/模糊。
      - **文字**：添加文字标签。
      - **其他**：序号标注、撤销/重做、像素化、自定义颜色和粗细。
      - **工作流**：截图后界面底部和侧边有工具栏，编辑完成后可以直接保存到文件、复制到剪贴板或上传到图床。
    - **安装方式**：
      ```bash
      # Ubuntu/Debian
      sudo apt install flameshot

      # Fedora
      sudo dnf install flameshot

      # Arch Linux
      sudo pacman -S flameshot
      ```

    #### 2. Shutter（已停止维护，但有分支延续）
    曾经是 Linux 上的截图神器，虽然原版已停止维护，但有一些分支版本（如 `shutter-edit`）仍在更新。
    - **功能特点**：
      - 内置强大的编辑窗口，功能非常丰富。
      - 涂鸦、画箭头、矩形、椭圆。
      - 添加文字（并可设置阴影等效果）。
      - 高亮、马赛克等。
    - **安装方式**：建议从 PPA 或社区仓库安装其分支版本。

    #### 3. KDE Spectacle
    如果你是 KDE Plasma 用户，那么系统自带的 Spectacle 就是一个非常好的选择。
    - **功能特点**：
      - 截图后可以直接进行标注，包括画线、箭头、矩形、椭圆、高亮和文字。
      - 界面简洁，与 KDE 桌面完美集成。

    #### 4. GNOME Screenshot / 截图
    对于使用 GNOME 桌面的用户，新版 GNOME 自带的“截图”工具也具备了基础的标注功能。
    - **功能特点**：
      - 在截图后弹出的预览窗口中，可以直接进行画线、箭头、矩形、椭圆和文字输入。
      - 功能比 Flameshot 简单，但满足基本需求。

    ### 二、 截图后使用独立编辑工具

    这种方式更灵活，你可以用任何截图工具（甚至系统默认的 `PrtSc` 键）截图，然后使用专门的图片编辑软件进行精细处理。

    #### 1. GIMP - **功能强大**
    GIMP 是 Linux 上的 Photoshop，功能极其强大。用它来编辑截图属于“杀鸡用牛刀”，但绝对能完成所有任务。
    - **适用场景**：当你需要对截图进行非常复杂和精细的编辑时。
    - **基本操作**：截图后，打开 GIMP，粘贴图像，然后使用画笔、形状、文字工具进行编辑。

    #### 2. Pinta - **轻量级画图工具**
    Pinta 可以看作是 Windows 上“画图”工具的 Linux 版，非常轻量、快速，非常适合简单的图片标注。
    - **功能特点**：支持画笔、形状、文字、模糊、缩放等常用功能，界面直观。
    - **安装方式**：
      ```bash
      # Ubuntu/Debian
      sudo apt install pinta

      # 其他发行版也通常可以在官方仓库中找到
      ```

    #### 3. KolourPaint
    KDE 阵营的“画图”工具，功能与 Pinta 类似，轻巧易用。

    #### 4. mtPaint
    一个非常轻量级、快速的像素图像编辑软件，专注于易用性和速度，也适合快速标注截图。

    ### 三、 现代化与跨平台选择

    #### 1. Draw.io (diagrams.net) - **意想不到的好用**
    Draw.io 虽然是画流程图的，但其内置的图形、箭头、文字工具非常适合制作教学或说明类的截图标注。
    - **使用方法**：截图后，在 Draw.io 中插入背景图片，然后在其上进行标注。可以导出为图片或 PDF。

    #### 2. Krita
    一款专业的数字绘画软件，但它的画笔和矢量工具同样可以用来非常自由地标注截图，尤其适合需要手绘风格涂鸦的用户。

    ### 总结与建议

    - **首选推荐（一站式解决方案）**：**Flameshot**。它几乎是为这个场景量身定做的，功能强大，操作流畅，是大多数 Linux 用户的首选。
    - **GNOME 用户**：可以先用自带的**截图**工具，如果功能不够再安装 Flameshot。
    - **KDE 用户**：系统自带的 **Spectacle** 已经相当不错。
    - **需要重度编辑**：截图后，用 **GIMP** 或 **Pinta** 打开进行精细处理。

    **设置小技巧**：
    你可以将 Flameshot 设置为默认的截图工具。例如，在 GNOME 的“设置” -> “键盘” -> “键盘快捷键”中，将 `PrtSc` 键的自定义命令设置为：
    ```bash
    flameshot gui
    ```
    这样每次按 `PrtSc` 键，启动的就是功能强大的 Flameshot 了。

* gnome 如何用命令行重启 remote desktop 服务

    在 GNOME 中重启远程桌面服务，可以通过以下几种方法：

    ## 方法 1：使用 systemctl（如果作为系统服务运行）

    ```bash
    # 重启 GNOME Remote Desktop 服务
    sudo systemctl restart gnome-remote-desktop

    # 或者重启整个用户服务
    systemctl --user restart gnome-remote-desktop

    # 查看服务状态
    systemctl status gnome-remote-desktop
    ```

    ## 方法 2：使用 gsettings 禁用再启用

    ```bash
    # 禁用远程桌面
    gsettings set org.gnome.desktop.remote-desktop.rdp enable false

    # 启用远程桌面
    gsettings set org.gnome.desktop.remote-desktop.rdp enable true

    # 禁用 VNC
    gsettings set org.gnome.desktop.remote-desktop.vnc enable false

    # 启用 VNC
    gsettings set org.gnome.desktop.remote-desktop.vnc enable true
    ```

    ## 方法 3：杀死进程并自动重启

    ```bash
    # 杀死远程桌面相关进程
    pkill -f gnome-remote-desktop

    # 或者更具体地
    pkill -f "gnome-remote-desktop-daemon"

    # GNOME 通常会自动重新启动这些服务
    ```

    ## 方法 4：重新加载 GNOME Shell

    ```bash
    # 按 Alt+F2，然后输入 r 回车
    # 或者在终端中：
    gnome-shell --replace &
    ```

    ## 方法 5：检查远程桌面设置

    ```bash
    # 查看当前 RDP 设置
    gsettings get org.gnome.desktop.remote-desktop.rdp enable

    # 查看当前 VNC 设置
    gsettings get org.gnome.desktop.remote-desktop.vnc enable

    # 查看可用设置
    gsettings list-keys org.gnome.desktop.remote-desktop.rdp
    ```

    ## 推荐使用顺序：

    1. 首先尝试方法 2（gsettings 切换）
    2. 如果不行，使用方法 1（systemctl）
    3. 最后考虑方法 3（杀死进程）

    注意：某些 GNOME 版本中远程桌面服务可能集成在 GNOME Shell 中，重启可能需要重新登录或重启 GNOME Shell。

    * gsettings reset org.gnome.desktop.remote-desktop 
        怎么用 reset 写？

        使用 `gsettings reset` 重置远程桌面设置的几种写法：

        ## 重置整个远程桌面架构

        ```bash
        # 重置整个远程桌面设置（包括 RDP 和 VNC）
        gsettings reset org.gnome.desktop.remote-desktop
        ```

        ## 重置特定协议

        ```bash
        # 重置 RDP 所有设置
        gsettings reset org.gnome.desktop.remote-desktop.rdp

        # 重置 VNC 所有设置
        gsettings reset org.gnome.desktop.remote-desktop.vnc
        ```

        ## 重置单个键值

        ```bash
        # 重置 RDP 启用状态
        gsettings reset org.gnome.desktop.remote-desktop.rdp enable

        # 重置 VNC 启用状态
        gsettings reset org.gnome.desktop.remote-desktop.vnc enable

        # 重置 RDP 端口
        gsettings reset org.gnome.desktop.remote-desktop.rdp port

        # 重置 VNC 认证方式
        gsettings reset org.gnome.desktop.remote-desktop.vnc auth-method
        ```

        ## 重置后的效果

        执行 `reset` 命令后：
        - 设置会恢复为默认值
        - 通常 RDP/VNC 会被禁用（enable = false）
        - 端口和其他配置会恢复默认

        ## 验证重置结果

        ```bash
        # 检查重置后的值
        gsettings get org.gnome.desktop.remote-desktop.rdp enable
        gsettings get org.gnome.desktop.remote-desktop.vnc enable

        # 列出所有当前设置查看默认值
        gsettings list-recursively org.gnome.desktop.remote-desktop.rdp
        ```

        ## 完整重置流程示例

        ```bash
        # 1. 重置所有远程桌面设置
        gsettings reset org.gnome.desktop.remote-desktop.rdp
        gsettings reset org.gnome.desktop.remote-desktop.vnc

        # 2. 重启服务（如果需要）
        sudo systemctl restart gnome-remote-desktop

        # 3. 验证设置已重置
        gsettings list-recursively org.gnome.desktop.remote-desktop.rdp | head -5
        ```

        使用 `reset` 比手动设置 `false` 更彻底，会清除所有自定义配置。

    * (base) hlc@hlc-VirtualBox:/etc/systemd$ gsettings reset org.gnome.desktop.remote-desktop enable
        No such key “enable

        你说得对！`org.gnome.desktop.remote-desktop` 这个主架构下没有 `enable` 键。正确的重置方法应该是：

        ## 正确的重置命令

        ```bash
        # 重置 RDP 启用状态
        gsettings reset org.gnome.desktop.remote-desktop.rdp enable

        # 重置 VNC 启用状态
        gsettings reset org.gnome.desktop.remote-desktop.vnc enable
        ```

        ## 查看可用的键

        ```bash
        # 查看远程桌面主架构有哪些键
        gsettings list-keys org.gnome.desktop.remote-desktop

        # 查看 RDP 有哪些键
        gsettings list-keys org.gnome.desktop.remote-desktop.rdp

        # 查看 VNC 有哪些键
        gsettings list-keys org.gnome.desktop.remote-desktop.vnc
        ```

        ## 完整的重置方案

        ```bash
        # 重置 RDP 所有设置
        gsettings reset org.gnome.desktop.remote-desktop.rdp

        # 重置 VNC 所有设置
        gsettings reset org.gnome.desktop.remote-desktop.vnc

        # 或者只重置启用状态
        gsettings set org.gnome.desktop.remote-desktop.rdp enable false
        gsettings set org.gnome.desktop.remote-desktop.vnc enable false
        ```

        ## 验证重置

        ```bash
        # 检查重置后的状态
        gsettings get org.gnome.desktop.remote-desktop.rdp enable
        gsettings get org.gnome.desktop.remote-desktop.vnc enable
        ```

        主要记住：
        - `rdp` 和 `vnc` 是子架构
        - `enable` 键存在于这些子架构中，不在主架构中

    * 那么 gnome setting 的那个界面的命令秆命令是什么？

        GNOME Settings（设置）的界面可以通过以下命令打开：

        ## 打开 GNOME Settings 主界面

        ```bash
        # 打开 GNOME 设置主界面
        gnome-control-center
        ```

        ## 打开特定设置页面

        ```bash
        # 打开共享设置（包含远程桌面）
        gnome-control-center sharing

        # 打开隐私设置
        gnome-control-center privacy

        # 打开网络设置
        gnome-control-center network

        # 打开显示设置
        gnome-control-center display

        # 打开电源设置
        gnome-control-center power

        # 打开声音设置
        gnome-control-center sound
        ```

        ## 直接打开远程桌面相关设置

        ```bash
        # 打开共享页面（通常包含远程桌面选项）
        gnome-control-center sharing

        # 或者使用更具体的 URI
        gnome-control-center screen-sharing
        ```

        ## 其他相关界面命令

        ```bash
        # 打开区域和语言设置
        gnome-control-center region

        # 打开用户账户设置
        gnome-control-center user-accounts

        # 打开背景设置
        gnome-control-center background

        # 打开默认应用程序设置
        gnome-control-center default-apps
        ```

        ## 查看所有可用的设置面板

        ```bash
        # 查看 gnome-control-center 支持的所有面板
        gnome-control-center --list
        ```

        ## 在终端中保持打开状态

        ```bash
        # 在终端中运行并查看输出（调试用）
        gnome-control-center --verbose
        ```

        最常用的就是 `gnome-control-center sharing` 来打开共享设置，里面通常包含远程桌面的图形界面配置选项。

* wayland + gnome 桌面是否可以启动 vncserver?

    是的，Wayland + GNOME 桌面可以启动 VNC server，但有一些重要的注意事项和技术细节需要了解。

    ## 原生支持方式

    ### 1. GNOME 内置的 VNC 支持
    GNOME 在 Wayland 下提供了原生的远程桌面功能：

    ```bash
    # 启用 GNOME 远程桌面
    gsettings set org.gnome.desktop.remote-desktop.vnc enable true

    # 设置认证方式
    gsettings set org.gnome.desktop.remote-desktop.vnc auth-method 'password'

    # 设置查看权限
    gsettings set org.gnome.desktop.remote-desktop.vnc view-only false
    ```

    ### 2. 使用 GNOME 控制中心配置
    1. 打开「设置」→「共享」
    2. 开启「远程桌面」
    3. 点击配置，启用 VNC 协议

    ## 第三方解决方案

    ### 1. WayVNC
    专为 Wayland 设计的 VNC 服务器：

    ```bash
    # 安装 WayVNC
    sudo apt install wayvnc  # Ubuntu/Debian
    sudo dnf install wayvnc  # Fedora

    # 启动 WayVNC
    wayvnc -S /tmp/wayvnc.sock :0
    ```

    ### 2. 使用 X11 回退方案
    如果应用兼容性有问题，可以临时切换到 X11：

    ```bash
    # 在 GDM 登录界面选择 GNOME on Xorg
    # 然后使用传统的 VNC 服务器
    sudo apt install tigervnc-server
    vncserver :1
    ```

    ## 配置示例

    ### GNOME 远程桌面配置
    ```bash
    # 设置 VNC 密码
    gsettings set org.gnome.desktop.remote-desktop.vnc password '$(echo -n "yourpassword" | base64)'

    # 启用加密
    gsettings set org.gnome.desktop.remote-desktop.vnc enable-auth true
    ```

    ### WayVNC 配置文件
    创建 `~/.config/wayvnc/config`：
    ```ini
    [output]
    name=DP-1

    [server]
    bind=0.0.0.0
    port=5900

    [security]
    require_auth=true
    username=
    password=
    ```

    ## 连接方式

    使用 VNC 客户端连接：
    ```bash
    # 使用 vncviewer 连接
    vncviewer 192.168.1.100:5900

    # 或者使用 Remmina
    remmina
    ```

    ## 注意事项

    1. **性能考虑**：Wayland 下的 VNC 性能可能不如 X11
    2. **安全性**：确保使用防火墙限制访问
    3. **功能限制**：某些 VNC 功能在 Wayland 下可能不可用
    4. **多显示器**：WayVNC 支持指定输出显示器

    ## 故障排除

    如果遇到问题：
    ```bash
    # 检查 GNOME 远程桌面状态
    gsettings list-recursively org.gnome.desktop.remote-desktop

    # 查看 WayVNC 日志
    wayvnc --debug

    # 检查防火墙设置
    sudo ufw status
    ```

    总的来说，Wayland + GNOME 确实可以运行 VNC server，但推荐使用 GNOME 内置的远程桌面功能或 WayVNC 等专为 Wayland 设计的解决方案。

* 没带鼠标，只有触摸板，无法实现边选中边滚动。是否有工具可以辅助实现？

    轻量级软件工具（按需选择）

    如果系统自带功能不够用，这些工具可以显著提升效率：

    **1. DragScroll（Windows，免费开源）**
    - **特点**：按住设定的按键（如右键或中键）即可临时进入“拖动滚动模式”。
    - **适用场景**：浏览长网页、阅读PDF时特别有用。

    **2. AutoHotkey（Windows，免费）**
    - 可以自定义脚本，例如设置一个热键让鼠标悬停时自动滚动。
    - 示例简单脚本（保存为`.ahk`文件）：
      ```autohotkey
      #IfWinActive  ; 适用于所有窗口
      F1::  ; 按F1开启自动滚动
      ScrollMode := !ScrollMode
      If ScrollMode
          SetTimer, AutoScroll, 50
      Else
          SetTimer, AutoScroll, Off
      Return

      AutoScroll:
      MouseGetPos, , , , ControlUnderMouse
      ; 向上滚动
      ControlSend, %ControlUnderMouse%, {WheelUp}
      Return
      ```

    **3. ScrollAnywhere（Chrome/Firefox扩展）**
    - 在浏览器中，按住右键拖动即可滚动，完美替代鼠标滚轮。

    **最推荐**：先从**系统自带的触控板三指拖移（macOS）或双指/单击设置（Windows）** 开始尝试，通常能满足80%的需求。如果经常遇到此问题，再根据主要使用场景选择一款轻量工具辅助。

    Ubuntu/Linux 系统下同样有非常丰富的解决方案，很多比 Windows/macOS 更强大。

    一、系统级触控板增强（GNOME / KDE）

    **1. 启用“中间点击粘贴”模拟**

    - 在很多Linux桌面环境中，你可以用 **同时点击左键和右键** 来模拟中键点击。
    - 检查设置：`设置` → `鼠标和触控板` → 查看是否有“模拟中键点击”选项。

    **2. 安装并配置 `libinput-gestures`（强烈推荐）**

    这是目前Linux上最强大的触控板手势工具之一：

    ```bash
    # 1. 安装
    sudo gpasswd -a $USER input  # 将用户加入input组
    sudo apt install wmctrl xdotool libinput-tools

    # 2. 安装libinput-gestures
    git clone https://github.com/bulletmark/libinput-gestures.git
    cd libinput-gestures
    sudo make install

    # 3. 安装GUI配置工具（可选但方便）
    sudo apt install libinput-gestures-gui

    # 4. 启动并设为自启
    libinput-gestures-setup autostart
    libinput-gestures-setup start
    ```

    **配置示例**（编辑 `~/.config/libinput-gestures.conf`）：

    ```
    # 三指向上：显示所有窗口（类似macOS Mission Control）
    gesture swipe up 3 xdotool key super+s

    # 三指向下：显示桌面
    gesture swipe down 3 xdotool key super+d

    # 四指左右：切换工作区
    gesture swipe left 4 xdotool key ctrl+alt+Left
    gesture swipe right 4 xdotool key ctrl+alt+Right

    # 按住右键时双指滚动（解决你问题的核心！）
    gesture hold right 2 xdotool click --delay 100 3  # 按住右键模拟中键
    ```

    **3. 使用 `touchegg` 触控板手势守护进程**

    ```bash
    sudo apt install touchegg
    # 安装后需要重启或启动服务
    sudo systemctl start touchegg
    sudo systemctl enable touchegg
    ```
    - 图形化配置：`sudo apt install touchegg-gui`
    - 可以配置如“三指拖拽”等高级手势

    二、专用滚动/选择工具

    **1. `xautomation` 工具包（命令行神器）**
    ```bash
    sudo apt install xautomation
    # 查看鼠标位置
    xmousepos

    # 模拟鼠标滚动（向下滚动5个单位）
    xte 'mouseclick 5'

    # 编写脚本实现“按住某键时触摸板滑动变为滚动”
    ```

    **2. 简单脚本方案：滚动锁定模式**
    创建脚本 `~/.local/bin/scroll-toggle.sh`：
    ```bash
    #!/bin/bash
    # 切换滚动模式：按F6启动，触摸板滑动变为滚动，再按F6恢复

    TOGGLE_FILE="/tmp/scroll_mode.toggle"

    if [ -f "$TOGGLE_FILE" ]; then
        rm "$TOGGLE_FILE"
        notify-send "滚动模式" "已关闭 - 恢复正常选择"
        exit 0
    fi

    touch "$TOGGLE_FILE"
    notify-send "滚动模式" "已开启 - 触摸板滑动将滚动页面"

    # 监听触控板事件，当滚动模式开启时转换事件
    # 这里需要根据你的触控板设备进行调整
    ```

    **3. 使用 `imwheel` 增强滚轮行为**
    虽然主要针对鼠标，但也能配置触控板：
    ```bash
    sudo apt install imwheel
    # 编辑 ~/.imwheelrc
    ```

    三、桌面环境特定方案

    **GNOME 用户**：
    1. 安装扩展管理器：
    ```bash
    sudo apt install gnome-shell-extension-manager
    ```
    2. 推荐扩展：
       - **Gesture Improvements**：增强原生手势
       - **Touchpad Indicator**：快速切换触控板模式
       - **Two-finger Scroll to Top/Bottom**：双指快速跳转

    **KDE Plasma 用户**：
    1. 系统设置 → `输入设备` → `触控板`
    2. 开启“启用手势支持”
    3. 在`高级`选项卡中自定义手势

    **通用方法：使用 `xbindkeys` + `xdotool` 自定义热键**
    ```bash
    sudo apt install xbindkeys xdotool
    # 生成默认配置
    xbindkeys --defaults > ~/.xbindkeysrc
    ```
    在 `~/.xbindkeysrc` 中添加：
    ```
    # 按住Super键时，触摸板滑动变为滚动
    "xdotool click 5"
       m:0x40 + b:2  # 按住Super时移动鼠标
    ```

    四、应用层技巧

    **终端（Terminal）**：
    - `Shift + Page Up/Page Down`：滚动终端输出
    - `Ctrl + Shift + ↑/↓`：按行滚动
    - `tmux` 用户：`Ctrl-b` 然后 `[` 进入滚动模式

    **Vim/Neovim**：
    ```vimrc
    " 在可视模式下用触摸板滑动
    set mouse=a  " 允许所有鼠标模式
    " 或者使用插件：vim-smoothie
    ```

    **VS Code**：
    1. 设置中搜索 "Scroll"
    2. 开启 `Editor: Mouse Wheel Scroll Sensitivity`
    3. 安装扩展 **"Touchpad Scroll"**

    **Firefox/Chrome**：
    - 安装扩展 **Auto Scroll**（设置自定义触发键）
    - 或使用 `Alt + 左键拖动`（部分Linux版本已内置）

    五、进阶方案：修改内核级触控板参数

    查看触控板设备：
    ```bash
    libinput list-devices | grep -A5 "Touchpad"
    ```

    临时修改参数（重启后失效）：
    ```bash
    # 提高滚动速度
    xinput set-prop "ELAN Touchpad" "libinput Scrolling Pixel Distance" 30

    # 开启自然滚动
    xinput set-prop "ELAN Touchpad" "libinput Natural Scrolling Enabled" 1

    # 查看所有可用属性
    xinput list-props "设备名"
    ```

    永久修改：创建 `/etc/X11/xorg.conf.d/40-libinput.conf`（需根据设备调整）

    六、最简实践路线

    1. **立即生效**：
       ```bash
       # 安装xdotool临时解决问题
       sudo apt install xdotool
       # 按住Super键时模拟中键滚动
       xbindkeys -f ~/.xbindkeysrc
       ```

    2. **中期方案**：
       ```bash
       # 安装libinput-gestures
       # 配置三指拖拽用于选择，四指滑动用于滚动
       ```

    3. **长期最佳**：
       - 确定你的桌面环境（GNOME/KDE/Xfce等）
       - 安装对应手势扩展
       - 为不同应用（终端/编辑器/浏览器）分别优化

    七、特殊情况处理

    **如果你用的是Wayland**：
    ```bash
    # 查看手势支持
    gsettings list-keys org.gnome.desktop.peripherals.touchpad

    # 启用边缘滚动
    gsettings set org.gnome.desktop.peripherals.touchpad edge-scrolling-enabled true
    ```

    **触控板型号问题**：
    ```bash
    # 查看详细触控板信息
    sudo evtest  # 选择你的触控板设备
    # 或
    cat /proc/bus/input/devices | grep -A5 Touchpad
    ```

    **推荐组合方案**：
    1. **日常使用**：`libinput-gestures` + 浏览器扩展
    2. **编程工作**：终端使用 `tmux`，编辑器配置好键盘快捷键
    3. **文档处理**：LibreOffice中多用 `Ctrl+Shift+方向键`

    Ubuntu/Linux 的优势在于高度可定制性。从简单的 `xdotool` 脚本到完整的触控板手势套件，你可以构建完全符合自己工作流的解决方案。建议从 `libinput-gestures` 开始，它相对稳定且功能全面。

* seahouse 设置 keyring 为空密码

    说的是 **Seahorse**（GNOME Keyring 管理器）。这是最直接的方法：

    **使用 Seahorse 设置空密码/解锁 Keyring**

    **方法1：图形界面操作**

    ```bash
    # 1. 安装 Seahorse
    sudo apt update
    sudo apt install seahorse

    # 2. 打开 Seahorse
    seahorse
    # 或通过菜单：应用程序 → 附件 → 密码和密钥

    # 3. 操作步骤：
    #    - 左侧选择"登录"（Login）
    #    - 右键点击"登录"钥匙环 → "更改密码"
    #    - 输入当前密码
    #    - 新密码留空 → 点击"继续"
    #    - 提示警告时选择"使用不安全存储"或"确定"
    ```

    **方法2：命令行设置空密码**

    ```bash
    # 停止 keyring 服务
    pkill gnome-keyring-daemon

    # 备份现有 keyring
    cp -r ~/.local/share/keyrings ~/.local/share/keyrings.backup

    # 删除现有 keyring
    rm ~/.local/share/keyrings/login.keyring

    # 重启 GNOME Keyring（会在下次登录时创建空密码keyring）
    gnome-keyring-daemon --start --components=secrets

    # 或者重启系统，登录时不输入keyring密码
    ```

    **方法3：配置自动登录时解锁**

    ```bash
    # 编辑 GNOME 自动启动配置
    mkdir -p ~/.config/autostart
    cat > ~/.config/autostart/gnome-keyring-unlock.desktop << EOF
    [Desktop Entry]
    Type=Application
    Name=Unlock GNOME Keyring
    Exec=/usr/bin/gnome-keyring-daemon --start --components=secrets
    OnlyShowIn=GNOME;
    AutostartCondition=GSettings org.gnome.desktop.remote-desktop.rdp screen-share-enabled
    EOF
    ```

    **方法4：通过 PAM 配置自动解锁**

    ```bash
    # 编辑 PAM 配置
    sudo nano /etc/pam.d/gnome-remote-desktop

    # 添加以下内容（如果文件不存在则创建）：
    auth    optional    pam_gnome_keyring.so
    session optional    pam_gnome_keyring.so auto_start

    # 编辑 lightdm/gdm 的 PAM 配置
    sudo nano /etc/pam.d/lightdm-autologin
    # 或
    sudo nano /etc/pam.d/gdm-password

    # 在文件末尾添加：
    auth optional pam_gnome_keyring.so
    session optional pam_gnome_keyring.so auto_start
    ```

    **方法5：使用 expect 脚本自动解锁**

    ```bash
    # 创建解锁脚本
    cat > ~/unlock-keyring.sh << 'EOF'
    #!/usr/bin/expect -f
    set timeout 10
    spawn gnome-keyring-daemon --unlock
    expect "密码："
    send "\r"  # 发送空密码（回车）
    expect eof
    EOF

    chmod +x ~/unlock-keyring.sh

    # 添加到自动启动
    ln -s ~/unlock-keyring.sh ~/.config/autostart/
    ```
* ubuntu gnome 里 ctrl + alt + left/right 可以切换桌面，但是并没有年到 setting 里有设置的地方， 为什么

    这是因为：

    1. **历史原因**

        * GNOME 很多快捷键是从 *gnome-shell* 的内置 keybinding schema 里读取的。

        * “切换工作区”的快捷键在 `org.gnome.desktop.wm.keybindings` 这个 dconf schema 里，不在 Settings 应用直接管理的那部分。

    2. **如何查看 / 修改**

        你可以用 `gsettings` 或 `dconf-editor` 来查看：

        ```bash
        gsettings get org.gnome.desktop.wm.keybindings switch-to-workspace-left
        gsettings get org.gnome.desktop.wm.keybindings switch-to-workspace-right
        ```

        默认会输出：

        ```
        ['<Control><Alt>Left']
        ['<Control><Alt>Right']
        ```

        修改方式：

        ```bash
        gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-left "['<Super><Alt>Left']"
        gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-right "['<Super><Alt>Right']"
        ```

        这样就能改绑定。

    3. **为什么在 Settings 里看不到**

        GNOME 的 *Settings → Keyboard* 界面只展示了部分用户级别的常用 keybindings（比如 screenshot, launchers），
        而窗口管理器（Mutter/gnome-shell）的 workspace 切换快捷键是另一套 schema，所以 UI 没有合并展示。

* gnome 桌面被 lock 后，远程 rdp client 就无法连接，怎么办

    你这个情况是 **GNOME 桌面锁屏后，远程 RDP 就断开/无法连接**。这其实是 GNOME + systemd + gdm 的机制导致的：当会话被锁定，桌面 session 会进入受限状态，外部 RDP 客户端就拿不到图形帧。

    解决方案 ： 如果是 wayland，那么需要禁用 GNOME 锁屏
* 安装 xpra 或 VNC 等替代品

    Wayland 下 X11 forwarding 难搞，但：

    * **xpra**（推荐）
    * **NoVNC / TigerVNC**
    * **Waypipe**（Wayland 的 SSH forwarding 专用）

    都可以替代 SSH -X。

* 简述一下 waypipe

    **Waypipe 是什么？**

    Waypipe = **Wayland 版本的 SSH X11 Forwarding**

    * 作用类似 `ssh -X`
    * 但转发的不是 X11，而是 **Wayland 协议 + DMA-BUF（显存 buffer）**
    * 可以 **在远程服务器上运行 GUI 程序，并显示在本地的 Wayland 桌面**

    简单理解：

    > *X11 forwarding 的 Wayland 替代品。*

    ---

    **安装方式（Ubuntu/Debian）**

    **服务器 + 本地都需要安装：**

    ```
    sudo apt install waypipe
    ```

    **最常用用法**

    **正常转发（类似 ssh -X）**

    ```
    waypipe ssh user@remote gedit
    ```

    或登录后再跑：

    ```
    waypipe ssh user@remote
    #（进入 shell 后）
    gedit
    ```

    **实际做了什么？**

    * 它在本地启动一个 **Wayland 代理客户端**
    * SSH 加密传输远程的 Wayland 请求（类似 X11 forwarding）
    * 远程应用渲染输出通过 DMA-BUF 或像素复制传回来

    ---

    性能特点

    | 模式                | 性能                                       |
    | ----------------- | ---------------------------------------- |
    | **DMA-BUF 零拷贝**   | 远端 GPU → 本地 GPU，无需 CPU copy，非常快（但需要显卡支持） |
    | **像素流（fallback）** | 性能和 x11 forwarding 类似，较慢                 |

    Waypipe 会自动协商。

    最重要的两个选项（强烈推荐使用）

    1. 开启压缩（zstd）

        ```
        waypipe --compress=lz4 ssh user@remote app
        ```

    2. 开启零拷贝 DMA-BUF（默认自动）

        显卡支持的情况下，它自动启用，因此无需手动设置。

    如果你想“开一个完整桌面应用”

    比如想在远程打开 VS Code GUI：

    ```
    waypipe ssh user@remote code
    ```

    只要远程是 Wayland 环境即可（GNOME/KDE/Hyprland 都行）。

    ---

    注意点（非常重要）

    1. **服务器和客户端必须都支持 Wayland（至少客户端要 Wayland）**
    2. **不能转发 X11-only 程序**
       如果你运行 `xeyes` 这类纯 X11 应用 → 失败
       （可以试试 `WAYLAND_DISPLAY=""` 强制 XWayland，但意义不大）
    3. 性能好坏取决于：

       * 带宽
       * GPU DMA-BUF 支持
       * 应用是否本身 Wayland 原生

    如何确认 Wayland 代理成功？

    远程运行的应用里：

    ```
    echo $WAYLAND_DISPLAY
    ```

    应该输出：

    ```
    waypipe-wl
    ```

    表示正在被 waypipe 捕获。

    Waypipe vs SSH -X（对比表）

    | 项目             | SSH -X | Waypipe |
    | -------------- | ------ | ------- |
    | 协议             | X11    | Wayland |
    | 性能             | 很差     | 快很多     |
    | 支持 Wayland 应用  | ❌      | ✔       |
    | 是否依赖 X server  | ✔      | ❌       |
    | 适合现代桌面         | ❌      | ✔       |
    | 支持 GPU DMA-BUF | ❌      | ✔       |
* gnome 的 remote desktop 如何通过命令行重启？

    在 GNOME（特别是 GNOME 42+ 或 Ubuntu 22.04+ 等使用 GNOME Remote Desktop 的系统）中，远程桌面服务是由 **`gnome-remote-desktop`** systemd 服务提供的。
    你可以通过命令行用 `systemctl` 来重启它。

    ✅ 重启 GNOME Remote Desktop 服务的命令

    ```bash
    systemctl --user restart gnome-remote-desktop.service
    ```

    🔍 查看状态

    查看当前远程桌面服务是否运行：

    ```bash
    systemctl --user status gnome-remote-desktop.service
    ```

    🚀 如果你希望彻底重新加载配置

    比如修改了 VNC/RDP 设置，可以执行：

    ```bash
    systemctl --user daemon-reload
    systemctl --user restart gnome-remote-desktop.service
    ```

    **确认 GNOME 远程桌面在监听 RDP 端口（默认 3389）**

    执行：

    ```bash
    ss -lntp | grep 3389
    ```

    正常输出应类似：

    ```
    LISTEN 0 50 *:3389 *:* users:(("gnome-remote-de",pid=xxxx,fd=xx))
    ```

    如果没有，说明 GNOME RDP 没在监听，可能：

    * 设置被禁用了；
    * 或 GNOME Remote Desktop 崩溃了；
    * 或当前会话不是图形 GNOME session。

    **检查配置是否启用了 RDP**

    运行：

    ```bash
    gsettings get org.gnome.desktop.remote-desktop.rdp enable
    ```

    如果输出是 `false`，则开启：

    ```bash
    gsettings set org.gnome.desktop.remote-desktop.rdp enable true
    ```

    另外检查是否允许无提示登录：

    ```bash
    gsettings get org.gnome.desktop.remote-desktop.rdp view-only
    gsettings get org.gnome.desktop.remote-desktop.rdp auth-method
    ```

    总结修复思路

    1. 启用并重启服务：

       ```bash
       systemctl --user restart gnome-remote-desktop.service
       ```

    2. 检查 RDP 是否监听 3389：

       ```bash
       ss -lntp | grep 3389
       ```
    3. 确认配置已启用：

       ```bash
       gsettings get org.gnome.desktop.remote-desktop.rdp enable
       ```
    4. 若仍失败，查看日志：

       ```bash
       journalctl --user -u gnome-remote-desktop.service -n 50
       ```
* ubuntu + wayland 如何修改分辨率，特别是 gnome setting 中没显示的分辨率

    在 Ubuntu 的 Wayland 会话下，由于 `xrandr` 不再直接控制显示输出（它仅通过 XWayland 起作用），传统的 `xrandr --addmode` 方法不再生效。要强制开启 GNOME 设置中未列出的分辨率，目前最稳妥的方法是通过 **内核引导参数（Kernel Parameters）** 来修改。

    你可以按照以下步骤操作：

    1. 确认显示器接口名称

        首先需要知道系统是如何命名你的显示接口的（例如 `DP-1`, `HDMI-A-1`）。
        打开终端，运行以下命令查看：

        ```bash
        for p in /sys/class/drm/*/status; do con=${p%/status}; echo -n "${con#*/card?- }: "; cat $p; done

        ```

        记录下状态为 `connected` 的接口名（如 `DP-1`）。

    2. 修改 GRUB 配置文件

        通过修改内核参数，在系统加载驱动时就强制指定分辨率。

        1. 打开 GRUB 配置文件：

            ```bash
            sudo vim /etc/default/grub
            ```

        2. 找到 `GRUB_CMDLINE_LINUX_DEFAULT` 这一行，在引号内添加 `video=<接口名>:<分辨率>@<刷新率>`。

            * 例如，你想在 `DP-1` 接口设置 `2560x1440` 且刷新率为 `60Hz`：
            
            ```text
            GRUB_CMDLINE_LINUX_DEFAULT="quiet splash video=DP-1:2560x1440@60"
            ```

            * *提示：如果不确定接口名，也可以尝试省略接口名，直接写 `video=2560x1440@60`，但这在多显示器环境下可能不够稳定。*

    3. 更新并重启

        保存文件并退出后，更新 GRUB 配置：

        ```bash
        sudo update-grub
        ```

        重启电脑。重启后，该分辨率通常会出现在 GNOME 的“设置 -> 显示”选项中；如果设置得恰好是原生支持但被误判的分辨率，系统甚至可能直接以该分辨率启动。

    **其他进阶方法**

    * **清理旧配置**：如果修改后无效，尝试删除可能冲突的旧显示配置：

        ```bash
        rm ~/.config/monitors.xml
        ```

        然后重启，让系统重新生成显示布局。

    * **自定义 EDID**：如果是因为显示器 EDID 信息损坏导致分辨率识别错误，上述 `video=` 参数可能无法解决高频或非标比例问题。这时需要制作一个 `.bin` 格式的自定义 EDID 文件，并使用 `drm.edid_firmware` 参数加载（这通常涉及更复杂的 `cvt` 计算和固件放置，如果 `video=` 无效，我可以再为你详细展开）。

* 怎么感觉 wayland 更难用了

    理解这种挫败感是完全正常的。从成熟的 **X11** 切换到 **Wayland**，就像是从一套修修补补但“什么都能干”的老房子，搬进了一座结构先进但“规矩极多”的新公寓。

    Wayland 的设计初衷是为了解决 X11 架构陈旧、不安全且容易产生画面撕裂的问题，但这种“推倒重来”也带来了不少痛点：

    1. 权限与协议的“洁癖”

        在 X11 下，任何程序都能读取屏幕内容或模拟按键（所以 `xdotool` 和截图软件很好用）。Wayland 出于安全考虑，默认禁止了这种行为。

        * **代价**：录屏、截图、全局热键映射（如 `xbindkeys`）或颜色校准工具，都需要开发者针对不同的合成器（GNOME 的 Mutter, KDE 的 KWin）重新写协议（如 PipeWire 或 Portal）。

    2. 调试逻辑的变化

        你刚才遇到的“无法修改分辨率”就是典型例子。

        * **X11**：`xrandr` 是万能钥匙，用户层可以直接强行喂给显卡驱动任何参数。
        * **Wayland**：显示管理权收归到了 **Compositor（合成器）** 手里。如果 GNOME 没在 UI 里给你选项，你就得去动内核参数或者复杂的 XML 配置文件。这种“配置权下移”让习惯了应用层操作的用户感到非常束缚。

    3. Nvidia 用户的“阵痛期”

        如果你使用的是 Nvidia 显卡，Wayland 的体验会更糟糕一些。虽然近两年 Nvidia 开始支持 GBM 驱动，但在某些旧版本内核或特定应用中，依然存在闪烁、甚至在睡眠唤醒后直接黑屏的问题。

    4. 生产力工具的碎片化

        很多资深开发者依赖的工具链（比如高度定制化的 **Vim/Neovim** 剪贴板同步、**SSH 隧道**下的远程 X11 转发等），在 Wayland 下往往需要额外的配置层（比如 `wl-clipboard`）才能正常工作。

    **我们该怎么办？**

    如果你觉得 Wayland 严重影响了你的开发或投资调研效率：

    * **切回 X11**：在登录界面点击用户头像，右下角有一个小齿轮，选择 **"Ubuntu on Xorg"**。X11 依然是目前的“二等公民”避风港，非常稳定。
    * **拥抱新工具**：如果你想坚持使用 Wayland，建议寻找对应的现代替代品：
    * `xrandr` → `gnome-monitor-config` 或内核参数。
    * `xdotool` → `ydotool`。
    * `xclip` → `wl-copy` / `wl-paste`。

    **你目前主要是在哪些工作流（比如特定软件、多屏显示或是远程桌面）中感到了明显的不便？** 我可以帮你看看有没有针对性的优化方案。

* ubuntu 如何自定义屏幕分辨率

    在 Ubuntu 中自定义屏幕分辨率有多种方法，以下是几种常用的方式：

    方法一：使用xrandr命令（临时生效）

    1. 查看当前显示设备

        ```bash
        xrandr
        ```

        记录下你的显示设备名称（通常是 `Virtual1`、`HDMI-1`、`eDP-1` 等）

        比如：

        ```
        Screen 0: minimum 16 x 16, current 1536 x 960, maximum 32767 x 32767
        XWAYLAND0 connected primary 1536x960+0+0 (normal left inverted right x axis y axis) 0mm x 0mm
           1536x960      59.91*+
           1280x960      59.94  
           1152x864      59.78
        ```

        显示设备名称为`XWAYLAND0`。

    2. 使用cvt生成模式参数

        ```bash
        cvt 1920 1080 60
        # 1920x1080 是分辨率，60是刷新率
        ```

    3. 创建新的显示模式

        ```bash
        xrandr --newmode "1920x1080_60"  173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync
        ```

    4. 添加到显示设备
        
        ```bash
        xrandr --addmode Virtual1 1920x1080_60
        ```

    5. 应用新分辨率

        ```bash
        xrandr --output Virtual1 --mode 1920x1080_60
        ```

    方法三：永久保存配置

    创建 xorg.conf 配置文件

    1. 获取显卡信息：

        ```bash
        sudo lspci | grep VGA
        ```

    2. 创建配置文件：

        ```bash
        sudo gedit /etc/X11/xorg.conf
        ```

    3. 添加以下内容（根据实际情况修改）：

        ```conf
        Section "Monitor"
            Identifier "Monitor0"
            VendorName "Unknown"
            ModelName "Unknown"
            Option "PreferredMode" "1920x1080"
        EndSection

        Section "Screen"
            Identifier "Screen0"
            Monitor "Monitor0"
            DefaultDepth 24
            SubSection "Display"
                Depth 24
                Modes "1920x1080" "1600x900" "1280x720"
            EndSubSection
        EndSection
        ```

    方法四：使用 ARandR 图形工具

    ```bash
    # 安装ARandR
    sudo apt install arandr

    # 启动
    arandr
    ```

    这个工具提供图形界面来配置多显示器设置。

    方法五：使用启动脚本自动设置

    1. 创建脚本文件：

        ```bash
        gedit ~/.xprofile
        ```

    2. 添加xrandr命令：

        ```bash
        #!/bin/bash
        xrandr --newmode "1920x1080_60"  173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync
        xrandr --addmode Virtual1 1920x1080_60
        xrandr --output Virtual1 --mode 1920x1080_60
        ```

    3. 保存并赋予执行权限：

        ```bash
        chmod +x ~/.xprofile
        ```

    **常见问题解决**

    1. 分辨率设置后黑屏

        - 等待10秒自动恢复
        - 或者按 `Ctrl+Alt+F1` 进入终端，重置设置

    2. 检查显卡驱动

        ```bash
        # 查看显卡驱动
        ubuntu-drivers devices

        # 安装推荐驱动
        sudo ubuntu-drivers autoinstall
        ```

    3. 重置显示设置
        
        ```bash
        # 删除显示配置
        rm ~/.config/monitors.xml

        # 重启显示管理器
        sudo systemctl restart gdm3
        # 或
        sudo systemctl restart lightdm
        ```

    **虚拟机特殊设置**

    如果你在虚拟机（如VirtualBox）中：

    1. 安装增强功能：

        ```bash
        sudo apt install virtualbox-guest-dkms virtualbox-guest-utils virtualbox-guest-x11
        ```

    2. 重启系统：

        ```bash
        sudo reboot
        ```

    选择哪种方法取决于你的具体需求。如果只需要临时使用，方法一最简单；如果需要永久保存，推荐方法三或方法五。

* 如果坚持使用W ayland，可以使用 wlr-randr（如果支持）

    ```bash
    # 安装wlr-randr
    sudo apt install wlr-randr

    # 查看当前配置
    wlr-randr

    # 但自定义分辨率功能也有限
    ```

    ```bash
    # 查看系统显示服务器类型
    echo $XDG_SESSION_TYPE
    ```

* 在 Wayland 下无法直接使用xrandr修改分辨率。这是因为 Wayland 和 X11 的显示架构不同，xrandr 在 Wayland 下的功能受限

    注销当前会话:

    `gnome-session-quit --logout`

* xfreerdp 如何设置远程画面的缩放

    xfreerdp 可以通过以下方式设置远程画面的缩放：

    1. 使用 /scale 参数（推荐）

        ```bash
        xfreerdp /v:SERVER_IP /scale:100
        ```

        缩放比例值：

        * 100 = 100%（原始大小）

        * 150 = 150%

        * 200 = 200%

        * 等等，可以设置为任何整数值

        注：

        1. `/scale`只支持固定的几个数值，并不支持 80 等小于 100 的值

    2. 使用 /scale-desktop 参数

        ```bash
        xfreerdp /v:SERVER_IP /scale-desktop:125
        ```

        这是百分比缩放，更精确控制。

    3. 动态缩放（FreeRDP 3.0+）

        ```bash
        xfreerdp /v:SERVER_IP /dynamic-resolution
        ```

        这会根据窗口大小动态调整缩放。

    4. 自动缩放选项

        ```bash
        # 根据DPI自动缩放
        xfreerdp /v:SERVER_IP /auto-reconnect /auto-scale:130

        # 使用监视器DPI
        xfreerdp /v:SERVER_IP /scale:100 /monitor-dpi
        ```

    5. 组合示例

        ```bash
        # 150%缩放，启用智能缩放
        xfreerdp /v:192.168.1.100 /u:username /p:password \
            /scale:150 /dynamic-resolution +fonts +clipboard
        ```

    6. 配置文件设置

        在 ~/.config/freerdp/client.rc 中：

        ```ini
        scale=150
        scale_desktop=150
        use_scale_factor=true
        ```

    注意事项：

    * 某些服务器/协议版本可能不支持所有缩放选项

    * 缩放可能会影响性能

    * 使用 /gdi:sw（软件渲染）有时能改善缩放效果

    * 确保服务器端也支持高DPI设置

    要查看所有显示相关选项：

    ```bash
    xfreerdp /help | grep -i scale
    ```

* ubuntu 22.04 gnome 对 vnc 支持不好，目前即使端口 5900 打开也无法访问

* x11 与 wayland

    * X11 从设计上就是一个网络透明的协议。它的核心就是一个客户端-服务器模型，应用程序（客户端）和显示服务器（服务器）可以轻松地运行在不同的机器上。VNC 服务器（如 Xvnc）可以伪装成一个 X11 服务器，接收所有图形指令，这是 X11 与生俱来的能力。

    * Wayland 的设计哲学完全不同。它是一个简单的、非网络的显示服务器协议。Wayland 客户端（应用程序）和 Wayland 合成器（相当于服务器）通过本地 Unix Socket 进行通信，并且高度依赖共享内存等技术来传递像素数据。它没有内置的网络支持。

* ubuntu 中 apt update 的图形界面的程序是`software-properties-gtk`

    启动时记得加上`sudo`或`sudo -E`，否则设置不会被保存。

* ubuntu 屏幕锁定后，不能被 gnome remote desktop 正常连接

    最简单的解决办法是把 lock screen 相关的设置都禁掉。

* gnome 的图片查看器叫 eog

* ubuntu gnome 中 settings 里的 shortcuts 快捷键设置显示不全，可以使用命令行显示全部的快捷键

    example:

    * 向左/向右切换虚拟桌面

        gnome -> settings 中只列出了`super` + `PageUp` / `PageDown`，但是实际上还可以使用`Ctrl` + `Alt` + `ArrowLeft` / `ArrowRight` 切换，说明 settings 列出的信息不全。

        执行`gsettings get org.gnome.desktop.wm.keybindings switch-to-workspace-left`，输出为

        ```
        ['<Super>Page_Up', '<Super><Alt>Left', '<Control><Alt>Left']
        ```

        可以看到，这里的信息是全的。我们可以将其设置为只使用第一个：

        `gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-left "['<Super>Page_Up']"`

        此时便禁用掉了`Ctrl` + `Alt` + `LeftArrow`。

* 平时还是把输入法归位到英文语言比较好，因为如果默读使用中文输入法，而切换到其他窗口又是中文状态，那么按快捷键会失效，或者类似于在游戏界面弹出中文候选框输入一堆中文。

* ibus 中，ctrl + / 可以改变词组上屏方式（直接上屏 / 按空格上屏）

* gnome 远程桌面无法重启后直接登录

    首先需要设置开机自动登录：

    settings -> users -> unlock -> enable atomatic login

    然后需要取消 key ring:

    安装`seahourse`（GNOME 的密码和密钥管理器）: `sudo apt install seahorse`

    进入 seahorse, passwords -> login 右键 -> change password -> 输入当前密码 -> 新密码空白，直接点 continue -> 确认

    此时再重启系统，可直接进入桌面，使用远程 rdp client 连接当前 host，可顺利连接。

    除了这种方法外，还可以为远程桌面创建独立的密钥环，这种方法比较复杂，不研究了。

* x11vnc 只支持 X11，无法在 Wayland 环境下使用。

* ubuntu 22.04 上启动 rdp 远程桌面 service

    首先 server 端安装`sudo apt install gnome-remote-desktop`

    然后进入 settings -> sharing -> remote desktop，把用到的功能都打开，最下面有一个设置密码的地方，初始密码是随机密码，需要自己再设置一下。

    其中有一个 enable legacy vnc protocol，打开后简单试了下，似乎没法用。

    此时检查`nc -zvn 127.0.0.1 3389`，端口应该是开启的。

    在 client 端安装：`sudo apt install freerdp2-x11`

    连接 rdp server: `xfreerdp /v:<server_ip>`，然后会提示输入 Domain（用户名）和密码，输入后即可成功启动 rdm 远程桌面。

    可以使用`xfreerdp /v:<server_ip> /clipboard`开启共享剪贴板，还可以使用快捷键 Ctrl + Alt + Enter 进入全屏模式。

* gnome 的 remote desktop 几乎只能通过图形化设置完成，在使用命令行设置时，`gnome-keyring`的配置十分复杂

* ubuntu 的官方帮助网站

    <https://help.ubuntu.com/>

* ubuntu 可以使用 ctrl + alt + 左箭头／右箭头 切换虚拟桌面。也可以使用 win/super + page up / page down 切换虚拟桌面。

* 下载 ubuntu 时选择 Alternative downloads -> Other images and mirrors -> All Ubuntu mirrors 可以列出所有的镜像网站

    国内镜像南京大学的速度挺不错的。拉满可以到 5.5 MB/s。

* ubuntu 24.04 virtual box 虚拟机非常不稳定，经常用着用着就突然卡死

    大概率是 ubuntu 的问题。

* ubuntu 给用户设置短密码

    `sudo passwd <username>`

* ubuntu 24.04 无法直接安装`libpocl-dev`，可能是 gcc 的版本比较高，而`libpocl-dev`需要比较低的 gcc 版本

## note

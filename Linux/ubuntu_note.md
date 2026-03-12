# Ubuntu Note

这个文件主要存放 ubuntu 特有的笔记。

## cache

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

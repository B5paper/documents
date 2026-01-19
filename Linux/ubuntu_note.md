# Ubuntu Note

这个文件主要存放 ubuntu 特有的笔记。

## cache

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

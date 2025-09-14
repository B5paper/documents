# Ubuntu Note

这个文件主要存放 ubuntu 特有的笔记。

## cache

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
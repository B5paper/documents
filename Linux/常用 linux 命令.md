# Linux

* fdisk

* mkfs

* `gzip` - `.gz`

* `bzip2` - `.bz2`

* `tar`

* `zip`, `unzip`

* `rar` - `.rar`

    `sudo apt install rar`

## 双系统安装

在安装了 windows 的机器上安装 ubuntu 系统：

目前是 uefi + 单硬盘情形。

需要点我的电脑右键 -> 管理 -> 磁盘管理。然后选择硬盘，右键点压缩卷。压缩空间量写要给 Ubuntu 预留的空间。

接下来准备个新U盘，下载 ubuntu 的安装镜像，把里面的文件复制到U盘里就行。UEFI 的启动方式不需要写 mbr 分区表，只要有文件夹就行。

使用 U 盘启动 Ubuntu 安装程序后，在选择磁盘擦除和分区方式时，选择手动分区。

我们选择空闲空间，然后创建下面四个分区：

1. efi，设置为 200M 就差不多够了

1. swap: 相当于 windows 中的虚拟内存。设置成两倍内存大小就差不多了。

1. 根目录`/`。大小设置成 40G ~ 50G 吧，总之以后要装各种软件。格式为`ext4`。

1. `/home`。用户数据目录，剩下的空间全给它，格式为`ext4`。

在选择安装启动分区时，要选择 efi 对应的分区号，比如`/dev/sda1`。

剩下的正常安装就可以了。

## device 相关

查看网卡名：

* `ifconfig`

    需要事先安装`net-tools`

* `sudo vi /etc/network/interfaces`

**PCIE 网卡**

* 查看网卡型号

    * `lspci -k | grep -i -A2 net`

    * `lspci | grep -i net`

    * `lshw -class network`

    * `sudo lshw -class network -short`

    * `ip link show`, `ip a`

    * `ip -br -c link show`, `ip -br -c addr show`

查看无线网络：

`iwconfig`

查看 usb 网卡：

`lsusb`

## 用户相关

* `adduser`, `finger`, `passwd`, `deluser`

* `addgroup`, `groups`, `delgroup`

* `chmod`, `chown`

创建新用户：`sudo adduser hlc`

* 添加用户

    * `sudo useradd -m <user_name>`

        `-m`表示在`/home`中创建一个用户的文件夹。如果这个文件夹已经存在，那么会报错说一些模板文件没法复制过来（`useradd: Not copying any file from skel directory into it.`。

        如果想手动复制诸如`.bashrc`等模板文件，可以手动把复制过来：

        ```bash
        cp -r /etc/skel/. /<user_home_directpory>
        ```

        其他一些常用参数：

        * `-m`: 指定 home directory

            `sudo useradd -m -d /opt/username <user_name>`

            在创建时指定用户的 home 目录。

        * `-s`：指定 shell

            `sudo useradd -s /usr/bin/zsh username`

            如果不指定，则默认使用`sh`作为`shell`。

        Ref: <https://linuxize.com/post/how-to-create-users-in-linux-using-the-useradd-command/>

    * add a user to sudoers group

        ```bash
        sudo usermod -aG sudo username
        ```

        其中`-a`表示 add，`G`表示 group。

        Ref: <https://linuxize.com/post/how-to-add-user-to-sudoers-in-ubuntu/>

        当前用户需要重新登陆后才能生效。

    * change the login shell

        `chsh -s /usr/bin/bash <user_name>`

        Ref: <https://www.cyberciti.biz/faq/howto-change-linux-unix-freebsd-login-shell/>

        Ref: <https://www.tecmint.com/change-a-users-default-shell-in-linux/>

### `sudoers`

刚开始的时候我们新创建的用户不在`/etc/sudoers`文件中，没有`sudo`的权限。可以在使用`visudo`在`/etc/sudoers.d/`目录下创建个文件，对我们的权限进行配置。这个文件会被自动读入到`/etc/sudoers`文件中。

`sudo visudo -f /etc/sudoers.d/hlc`

写入这样一行：`hlc ALL=(ALL) ALL`，然后保存退出即可。

第一个`ALL`表示所有机器，第二个`ALL`表示所有权限，第三个`ALL`表示所有命令。具体可参照这篇文档中的说明：<https://help.ubuntu.com/community/Sudoers>

使用`service xxx restart`时，总是需要输入第一个用户的密码（比如`ubuntu`用户的密码），而不是`root`用户或当前用户的密码。这个似乎和`polkit`有关。相关的解释在这里：<https://askubuntu.com/questions/1114351/why-system-keep-ask-to-enter-password-for-the-first-member-of-sudo-group-instead>

转换到`root`账户：`sudo su`

### mount 共享文件夹

`sudo mount -t vboxsf E_DRIVE -o umask=0002 -o uid=1000 -o gid=1000 /mnt/e`

### Linux group

There are two types of groups in Linux operating systems:

* The primary group

    When a user creates a file, the file's group is set to the user's primary group. Usually, the name of the group is the same as the name of the user. The information about the user's primary group is stored in the `/etc/passwd` file.

* Secondary or supplementary group

    Useful when you want to grant certain file permissions to a set of users who are members of the group.

Each user can belong to eactly one primary group and zero or more secondary groups.

Only root or users with `sudo` access can add a user to a group.

将用户添加进一个组：

`sudo usermod -a -G groupname username`

其中`-a`表示 append，如果省略`-a`，那么用户会被移出除了指定组外其余所有的组。`-G`后跟组名，可以是多个组，用逗号分隔：sudo usermod -a -G group1,group2 username`。

将用户移出某个组：

`sudo gpasswd -d username groupname`

创建一个新组：

`sudo groupadd groupname`

删除一个组：

`sudo groupdel groupname`

更改用户的 primary group：

`sudo usermod -g groupname username`

创建用户的同时指定组：

`sudo useradd -g users -G wheel,developers nathan`

其中`-g`表示 primary group，`-G`表示 secondary groups。

显示一个用户所属的组：

`id username`

显示一个用户的 supplementary groups：

`groups username`

如果没有用户名参数，`groups`会显示 currently logged in user's groups。

显示所有的组：

`getent gorup`

（修改组后，要重新登录 shell 才会生效）

## grep

`grep hello test.txt`：显示`test.txt`文件中所有包含`hello`字符串的行

`grep -c hello test.txt`：显示包含`hello`的行数

`cat test.txt | grep hello`：与其他命令联合使用，过滤内容

`ps -ef | grep ssh`

`ps -ef | grep ssh | grep -v grep`：显示所有进程，但是不显示`grep`进程本身。`-v`表示不显示包含后续字符串的信息

## 文件相关

* `ln`

## make 相关

Makefile：

```Makefile
target: dep1.o dep2.o
    compile command (gcc -o target dep1.o dep2.o)
dep1.o: src1.c
    compile command (gcc -c src1.c)
dep2.o: src2.c
    compile command

clean:
    rm *.o
    rm main
```

命令以`tab`键开始，不能使用空格。

* `name = zzk`, `curname = $(name)`

    ```Makefile
    print:
        @echo curname: $(curname)
    ```

    执行：`make print`

* `curname := $(name)` 绑定

* `curname ?= zuozhongkai`：如果`curname`前面没有被赋值，那么此变量就是`zuozhongkai`；若前面已经赋过值了，则使用前面赋的值。

* `+=`追加字符串

常用的自动化变量：

|自动化变量|描述|
|-|-|
|`$@`|规则中的目标集合，在模式规则中，如果有多个目标的话，`$@`表示匹配模式中定义的目标集合|
|`$%`|当目标是函数库的时候表示规则中的目标成员名，如果目标不是函数库文件，那么其值为空|
|`$<`|依赖文件集合中的第一个文件，如果依赖文件是以模式（即`%`）定义的，那么`$<`就是符合模式的一系列的文件集合|
|`$?`|所有比目标新的依赖目标集合，以空格分开|
|`$^`|所有依赖文件的集合，使用空格分开，如果在依赖文件中有多个重复的文件，`$^`会去除重复的依赖文件，值保留一份|
|`$+`|和`$^`类似，但是当依赖文件存在重复的话不会去除重复的依赖文件|
|`$*`|这个变量表示目标模式中`%`及其之前的部分，如果目标是`test/a.test.c`，目标模式为`a.%.c`，那么`$*`就是`test/a.text`|

伪目标：

```Makefile
.PHONY: clean
clean:
    rm *.o
    rm main
```

## bash

`echo "string"`

`read -p "first num:" first`

`echo "$first"`

`test -e $filename`

`test $firststr == $secondstr`

`$0 ~ $n`：脚本的输入参数。`$0`表示脚本本身。

`$#`：最后一个参数的标号

`$@`

```bash
if 条件： then
    // do something
fi

if 条件: then
    // do something
else
    // do something
fi

if 条件: then
    # do something
elif [条件]: then
    # do something
else
    # do something
fi
```

```bash
case $变量 in
"第 1 个变量内容")
    程序段
    ;;  # 表示该程序块结束
"第 2 个变量内容")
    程序段;;
"第 n 个变量内容")
    程序段
    ;;
*)
    # xxx
    ;;
esac
```

函数：

```bash
function fname() {
    echo "param 1: $1"
    echo "param 2: $2"
}

print a b
```

循环：

```bash
while [条件]
do
    # ...
done

until [条件]  # 条件不成立时执行
do
    # loop code
done

for var in con1 con2 con3 ...
do
    # xxx
done

for((initialize; condition; exec))
do
    # xxx
done
```

## grub 中更改启动使用的内核

Ref: <https://gist.github.com/chaiyujin/c08e59752c3e238ff3b1a5098322b363>

显示当前使用的内核版本：

`grep submenu /boot/grub/grub.cfg`

显示所有可用的内核版本：

`grep gnulinux-4.15.0 /boot/grub/grub.cfg`

我们需要找到`$menuentry_id_option`后面的 kernel id 字符串。这个是我们需要的。

编辑`/etc/default/grub`文件中的`GRUB_DEFAULT`，网上说写成类似`GRUB_DEFAULT='gnulinux-advanced-4591a659-55e2-4bec-8dbe-d98bd9e489cf>gnulinux-4.15.0-126-generic-advanced-4591a659-55e2-4bec-8dbe-d98bd9e489cf'`这样的就可以，但是我试了试不行。写成`"1>2"`这样的可以（注意字符串要用引号括起来），其中`1`表示一级菜单的第二个选项，`2`表示二级菜单的第三个选项。

这些选项或许可以通过文件获得，但我是先实际看了看 grub 菜单，然后直接记住的。如果 grub 菜单等待时间过短，可以设置`GRUB_TIMEOUT`为`-1`或一个比较大的数。

最后需要执行`sudo update-grub`使改动生效。

More ref:

1. <https://docs.digitalocean.com/products/droplets/how-to/kernel/use-non-default/>

1. <https://unix.stackexchange.com/questions/694323/how-to-set-default-kernel-in-debian>

1. <https://unix.stackexchange.com/questions/198003/set-the-default-kernel-in-grub>

1. <https://askubuntu.com/questions/599208/how-to-list-grubs-menuentries-in-command-line/1022706#1022706>

    有时间可以看一下这个 script，调研一下 menu 是怎么抽出来的。

## deb package inspect

Ref: <https://blog.packagecloud.io/inspect-extract-contents-debian-packages/>

A debian package is a Unix `ar` archive that includes two tar archives: one containing the control information and another with the program data to be installed.

list its contents: `dpkg -c ./path/to/test.deb`

To extract data from a deb package to current dir: `ar -x ./test_2.0.0_amd64.deb`

Extract files from `control.tar.gz` and `data.tar.gz` using `tar`: `tar -xzf control.tar.gz`

To extract files from a debian package, using `dpkg-deb`:

`dpkg-deb -x ./path/to/test.deb ./path/to/destination`

Extract control information from a Debian package using `dpkg-deb`:

`dpkg -e ./test_2.0.0_amd64.deb`

The `preinst`, `postinst`, `prerm`, and `postrm` files are scripts that will automatically execute before or after a package is installed or removed. These scripts are part of the control section of a Debian package.

Using `apt-file` to view the contents of debian packages on remote repositories:

`apt-get install apt-file`

`apt-file update`

`apt-file list <packagename>`

## Tricks

* 使用`sudo`时保留用户的环境变量：`sudo -E <command>`

* 在一行中设置多个环境变量：

    `http_proxy=xxx https_proxy=xxx no_proxy=xxx <command>`

    用空格分隔开不同环境变量就可以。

* 查看某个端口`<port>`被哪个进程占用

    这个比较好用：`sudo ss -tulpn | grep :3306`

    其他的方法：<https://www.cyberciti.biz/faq/what-process-has-open-linux-port/>

* 查看环境变量

    `printenv`或者`env`

    注意`sudo`的环境变量与`root`并不一致，可以用`sudo printenv`查看`sudo`使用的环境变量。

    `sudo`默认加载的环境变量可以到`/etv/environment`，`/etc/profile`中设置。

* 挂载 iso 文件

    `sudo mount -o loop,ro -t iso9660 filename.iso test_folder`
    
    `sudo mount filename.iso test_folder`

* 清空 cache

    1. Clear PageCache only.

        `# sync; echo 1 > /proc/sys/vm/drop_caches`

    2. Clear dentries and inodes.

        `# sync; echo 2 > /proc/sys/vm/drop_caches`

    3. Clear pagecache, dentries, and inodes.

        `# sync; echo 3 > /proc/sys/vm/drop_caches`

* 修改 boot 文件，启动文字操作系统

    将`linux`行最后的`splash`替换成`text`即可。

    Ref: <https://superuser.com/questions/974797/how-to-boot-a-linux-system-without-graphical-server>

## 查看显存使用情况

`glxinfo | grep -E -i 'device|memory'`

注意`device`和`memory`中间不能有空格。

或者使用`glxinfo -B`，也能看到有显存使用情况的数据。

## amd gpu 的 usage

```bash
cat /sys/class/drm/card0/device/gpu_busy_percent  # gpu core usage.
cat /sys/class/drm/card0/device/mem_busy_percent  # gpu memory usage.
cat /sys/class/drm/card0/device/mem_info_vram_used  # memory used. 
```

ref: <https://unix.stackexchange.com/questions/576707/how-to-monitor-amd-gpu-on-linux-server>


## problem shooting

* Ubuntu 无法连接企业 Wifi

    解决方案：<https://ubuntuforums.org/showthread.php?t=2474436>

* 插上耳机没有声音

    `sudo apt install pavucontrol`

    `pavucontrol`

    然后在`configuration`中设置。

* 通过代理设置时间

    Ref: <https://superuser.com/questions/307158/how-to-use-ntpdate-behind-a-proxy/509620#509620?newreg=8c16b52e667c4208a9ae057d64a8195b>

    参考第二个回答，用 wget 获取时间。缺点是会在当前文件夹下生成一个`index.html`文件。

    `tlsdate`这个工具似乎好多年没更新过了，也不知道怎么编译。

* dpkg-deb: 错误: 子进程 粘贴 被信号(断开的管道) 终止了

    `sudo dpkg -i --force-overwrite " /var/cache/apt/archives/texlive-math-extra 2016.20160805.1 all.deb"`

    `sudo apt-get dist-upgrade`

* 查看系统版本

    1. 方法一：`lsb_release -a`

    1. 方法二：`cat /etc/lsb-release`

    1. 方法三：`cat /etc/os-release`

    1. 方法四：`/etc/issue`

    1. 方法五：`hostnamectl`

    1. 方法六：`uname -srn`

* 在換 ubuntu apt 源之前，需要先安装`apt install ca-certificates`，不然換完源无法`apt update`

* linux 在 mount u盘时出现 Read-only file system 问题

    原因是 U 盘是 ntfs 文件系统，而操作系统不是完全支持 ntfs。

    解决办法：

    ```bash
    apt install ntfs-3g fuse
    ```

    Ref: <https://linuxconfig.org/how-to-mount-partition-with-ntfs-file-system-and-read-write-access>

* 使用 dpkg 找到某个文件是属于哪个 package 的

    ```bash
    $ dpkg -S /bin/ls
    coreutils: /bin/ls
    ```

    ref: <https://askubuntu.com/questions/481/how-do-i-find-the-package-that-provides-a-file>

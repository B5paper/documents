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

查看网卡型号：

`lspci -k | grep -i -A2 net`

查看网卡设备信息：

`lspci | grep -i net `

查看无线网络：

`iwconfig`

查看 usb 网卡：

`lsusb`

## 用户相关

* `adduser`, `finger`, `passwd`, `deluser`

* `addgroup`, `groups`, `delgroup`

* `chmod`, `chown`

创建新用户：`sudo adduser hlc`

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

## problem shooting

1. Ubuntu 无法连接企业 Wifi

    解决方案：<https://ubuntuforums.org/showthread.php?t=2474436>


# Linux Driver Note

## Introduction



## 下载内核源码

通常我们说的 linux kernel 可以在

如果使用的是 Ubuntu 系统，下载源码可以在`apt`里下载：

```bash

```

## 编译内核

一个写得还不错的 tutorial，可以参考一下：<https://phoenixnap.com/kb/build-linux-kernel>

编译内核需要一些额外的工具，可以参考这个网页<https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel>装一下：

```bash
sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm
```

下载好源码后，首先需要创建一个`.config`文件。我们可以使用 Ubuntu 系统里现成的：

如果使用这种方法后编译失败，通常是证书设置的问题，参考这个：<https://askubuntu.com/questions/1329538/compiling-the-kernel-5-11-11>

在`.config`里把`CONFIG_SYSTEM_TRUSTED_KEYS`和`CONFIG_SYSTEM_REVOCATION_KEYS`都设置成空字符串。

也可以创建一个默认的：

```bash
make menuconfig
```

这种方法会在内核源码目录创建一个`.config`文件。使用这个 config 编译起来比较快，可能是因为有点选项没选上吧。

单核编译可以用`makd`，多线程编译可以使用`makd -j16`（使用 16 线程编译）。

## 内核模块

内核模块编程注意事项：

1. 不能使用 C 库和 C 标准头文件
1. 使用 GNU C （在 ANSI C 上加了些语法）
1. 没有内存保护机制
1. 不能处理浮点运算
1. 注意并发互斥和可移植性

**内核模块如何编写**

需要包含头文件

```cpp
#include <linux/init.h>
#include <linux/module.h>
```

必须实现加载函数和卸载函数

```c
// 加载函数
int xxx(void)
{
    // ...
    return 0;  //返回 0 表示加载成功
}

// 卸载函数
void yyy(void)
{
    // ...
}
```

使用以下两个宏来修饰加载函数和卸载函数

```c
module_init(xxx);  // 修饰加载函数
module_exit(yyy);  // 修饰卸载函数
```

（相当于把普通函数注册成加载函数/卸载函数）

注：内核中打印函数使用`printk`。用法和`printf`几乎一样。

Ubuntu 的内核是 signed 版本，我们要写的 module 是 unsigned 版本，没有办法直接`insmod`。目前我的解决办法是，先更新一下内核，换成 unsigned 版本或者 hwe 版本（hwe 版本表示支持最新硬件）：

```bash
apt install linux-image-generic-hwe-22.04
```

我们可以用`uname -r`查看当前内核的版本，`uname -a`查看系统的完整版本。

接下来我们把原来的 kernel 删掉，然后下载新的：

```bash
sudo apt update && sudo apt upgrade
sudo apt remove --purge linux-headers-*
sudo apt autoremove && sudo apt autoclean
sudo apt install linux-headers-generic
```

（也有可能是用下面这个命令装的，我不记是哪个了：

```bash
sudo apt-get update && sudo apt-get install linux-headers-`uname -r`
```
）

这时候应该可以看见`/lib/modules/xxxxx-generic`路径下（比如`/lib/modules/5.19.0-32-generic/`），有一个`build`文件夹。这个`build`是一个 symbolic link，指向`/usr/src`下对应的内核源码文件夹：

```
hlc@hlc-Ubuntu2204:~/Documents/Projects/kernel_test$ ls -lh /lib/modules/5.19.0-32-generic/
total 6.6M
lrwxrwxrwx  1 root root   40  1月 30 23:44 build -> /usr/src/linux-headers-5.19.0-32-generic
drwxr-xr-x  2 root root 4.0K  2月 18 09:33 initrd
drwxr-xr-x 16 root root 4.0K  2月 18 09:31 kernel
-rw-r--r--  1 root root 1.5M  2月 26 14:45 modules.alias
...
```

如果可以看到对应的内容，那么就说明编译好的内核已经装好了。

我们开始写我们的`hello_world.c`文件：

```c
#include <linux/init.h>
#include <linux/module.h>

int hello_init(void)
{
    printk("<1>""hello my module\n");
    return 0;
}

void hello_exit(void)
{
    printk("<1>""bye bye!\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");  // 不加这行的话，无法通过编译
```

接着写 Makefile：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-32-generic
obj-m  +=  hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

然后在当前文件夹下运行`make`，会生成`hello_world.ko`文件。这个文件就是我们需要的内核模块文件，ko 代表 kernel object。

此时我们可以使用`sudo insmod hello_world.ko`插入模块，使用`sudo rmmod hello_world`移除模块，`sudo lsmod`查看已经加载的模块，`sudo dmesg`查看日志输出。

（如果还是无法`insmod`，或许需要取消 secure boot：<https://askubuntu.com/questions/762254/why-do-i-get-required-key-not-available-when-install-3rd-party-kernel-modules>）

除了添加`MODULE_LICENSE()`外，可选的添加信息有：

`MODULE_AUTHOR("hlc")`

`MODULE_VERSION("1.0")`

`MODULE_DESCRIPTION("this is my first module")`

`printk()`的打印的日志级别有 0-7，还可以是无级别。如果无法看到输出内容，可以参考这几个网站的解决方式：

1. <http://www.jyguagua.com/?p=708>

1. <https://blog.csdn.net/qustDrJHJ/article/details/51382138>
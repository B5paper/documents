# Linux Driver Note

## Introduction



## 下载内核源码

通常我们说的 linux kernel 可以在

如果使用的是 Ubuntu 系统，下载源码可以在`apt`里下载：

```bash

```

## 编译内核

### 使用源码编译

### 在 Ubuntu 22.04 下编译

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

### 内核模块的加载与卸载

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

我们可以使用`cat /proc/sys/kernel/printk`查看当前日志的级别。

第一个数为内核默认打印级别，只有当`printk`的打印级别高于内核默认打印级别时，`printk`打印的信息才能显示 。

第二个数为`printk()`的默认打印级别。

修改内核的默认打印级别（即修改第 1 个数）：

`echo 5 > /proc/sys/kernel/printk`

（现在好像已经升级了，不管 printk level 大于还是小于当前 console 的 level，都不会在 console 上输出。 ）

### 内核模块参数

内核模块参数：

`module_param(模块参数名，模块参数，访问权限);`

`module_param_array(数组模块参数名，数组元素类型，NULL，访问权限);`

在代码中对模块参数的使用和普通变量没有区别。

Examples:

```c
int param_int = 10;
unsigned short param_ushort = 20;
char *param_string = "hello";
int param_arr[3] = {100, 200, 300};

module_param(param_int, int, 0775);
module_param(param_ushort, ushort, 0);
module_param(param_string, charp, 0644);
module_param_aray(param_arr, int, NULL, 0755);
```

打印模块参数：

```c
int modparam_init(void)
{
    printk("param_int = %d\n", param_int);
    printk("param_ushort = %hu\n", param_ushort);
    printk("param_string = %s\n", param_string);
    printk("param_arr = %d %d %d'n", param_arr[0], param_arr[1], param_arr[2]);
    return 0;
}
```

在命令行中传递模块参数：

```bash
insmod hello_abc.ko param_int=50
insmod hello_abc.ko param_string="hello world"
insmod mod_param.ko param_arr=111,222,333
```

数组可以只改一部分，但是参数不能给多。

当模块加载成功后，那些访问权限非0的模块会在以下路径下：

`/sys/module/模块名/parameters`

存在和模块参数名同名的文件，这些文件的权限来自于模块参数的权限。文件的内容与模块参数的值相同。因此可以通过修改文件中保存的数据，对模块参数进行修改。

### 模块符号的导出

模块导出符号可以将模块中的变量/函数导出，供内核其他代码/模块使用。

如何导出：

1. 内核中提供了相应的宏来实现模块的导出

    ```
    EXPORT_SYMBOL
    EXPORT_SYMBOL_GPL  (只有遵循 GPL 协议的代码才可以使用)
    ```

    Example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>

    int add(int a, int b)
    {
        return a + b;
    }

    int mul(int a, int b)
    {
        return a * b;
    }

    EXPORT_SYMBOL(add);
    EXPORT_SYMBOL_GPL(mul);

    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("hlc");
    MODULE_VERSION("1.0");
    MODULE_DESCRIPTION("this is module symbol!");
    // ...
    ```

    在别的模块中使用时，需要这样写：

    ```c
    extern int add(int a, int b);
    extern int mul(int a, int b);
    ```

    `extern`表示函数是在外部实现的，不是在本文件中实现的。使用这些函数时，需要先加载他们所在的模块。

    在 Makefile 中，应该把两个`.o`文件都写上：

    ```Makefile
    obj-m += xxx.o xxx_2.o
    ```

## 设备驱动

### 设备类型

linux 设备：

1. 字符设备

    按字节流访问，一般是按顺序访问

    绝大多数设备都是字符设备。比如 led 按键 串口 传感器 LCD

    字符设备的驱动通过字符设备文件来访问

1. 块设备

    按数据块访问，块的大小固定，通常是 4k，具有随机访问能力

    内存，磁盘，SD卡，U盘

    块设备驱动通过块设备文件来访问

1. 网络设备

    一般只代表网卡设备。

    驱动实现要结合网络协议栈（TCP/IP）

    访问网络设备不通过文件，通过套接字（网络通信地址）访问

1. 字符设备驱动的实现

    1. 概述

    驱动是沟通硬件和上层应用的媒介，字符设备驱动通过字符设备文件来访问，访问设备文件使用文件IO，在用户层访问设备文件和普通文件的方法是没有区别的

    `open, close, read, write, lseek, ioctl, mmap, stat`

1. 通过设备文件找到对应的驱动

    Linux 中所有的设备文件存放在`/dev`中。

    内核中有很多的字符设备驱动，这些字符设备驱动如何与对应的字符设备文件匹配，实际上是通过设备号来找到对应的字符设备驱动。

    设备号用 32 位的一个`dev_t`类型的变量来表示（无符号整型），高12位表示主设备号，后20位表示次设备号。

    主设备号用来区分不同类型的设备。

    在`/proc/devices`文件中可以查找到设备号与对应的设备类型。

    内核中提供了操作设备号的宏：

    ```c
    MAJOR(设备号);  // 通过设备号获取主设备号
    MINOR(设备号);  // 通过设备号获取次设备号
    MKDEV(主设备号,次设备号);  // 通过主设备号和次设备号构造设备号
    ```

### 设备号    

1. 如何向内核申请设备号

    设备号在内核中属于资源，需要向内核申请。

    需要包含头文件：

    ```c
    #include <linux/dev/h>
    #include <linux/fs.h>
    ```

    1. 静态申请

        首先选择一个内核中未被使用的主设备号（`/proc/devices`），比如 220。根据设备个数分配次设备号，一般从 0 开始。

        构造设备号：`MKDEV(220,0);`

        调用`register_chrdev_region(dev_t from, unsigned count, const char *name);`

        params:

        `from`: 要申请的起始设备号

        `count`: 设备数量

        `name`: 设备号在内核中对应的名称

        返回 0 表示成功，返回非 0 表示失败。

        不再使用设备号需要注销：

        ```c
        unregister_chrdev_region(dev_t from, unsigned count);
        ```

        params:

        `from`: 要注销的起始设备号

        `count`: 设备号的个数

        一般在卸载模块的时候释放设备号。

    1. 动态申请

        通过`alloc_chrdev_region`向内核申请

        ```c
        int alloc_chrdev_region(dev_t *dev, unsigned baseminor, unsigned count, const char *name);
        ```

        params:

        `dev`: 设备号的地址

        `baseminor`: 起始次设备号

        `count`: 设备号个数

        `name`：设备在内核中对应的名称

        释放的方法和静态申请一致。

### cdev

cdev 在内核中代表一个字符设备驱动。

```c
struct dev {
    struct kobject kjob;
    struct module *owner;
    const struct file_operations *ops;  // 驱动操作函数集合
    struct list_head list;
    dev_t dev;  // 设备号
    unsigned int count;
};
```

**如何往内核中添加一个 cdev**

`cdev_init`：初始化 cdev（为 cdev 提供操作函数集合）

```c
void cdev_init(struct *cdev, const struct file_operations *fops);
// cdev 事先声明，fops 也要事先写好
```

`cdev_add`：将 cdev 添加到内核（还会为 cdev 绑定设备号）

```c
int cdev_add(struct cdev *p, dev_t dev, unsigned count);
```

params:

`p`: 要添加的 cdev 结构

`dev`：起始设备号

`count`：设备号个数

返回 0 表示成功，非 0 表示失败。

* `cdev_del`：将 cdev 从内核中移除

    ```c
    void cdev_del(struct cdev *p)
    ```

Examples:

```c
dev_t dev;  // 设备号
struct cdev cdd_cdev;  // 声明 cdev

int cdd_open(struct inode *inode, struct file *flip)
{
    
}

struct file_operations cdd_fops = {  // GNU C 额外语法，选择性地初始化
    .owner = THIS_MODULE,
    .open = cdd_open,
    .read = cdd_read,
    .write = cdd_write,
    .unlocked_ioctl = cdd_ioctl,  // ioctl 接口
    .release = cdd_release,  // 对应用户 close 接口
}


```

`inode`是文件的节点结构，用来存储文件静态信息。文件创建时，内核中就会有一个 inode 结构
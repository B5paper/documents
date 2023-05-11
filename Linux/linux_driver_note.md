# Linux Driver Note

## Introduction

### 驱动开发环境的搭建

#### 基于 Ubuntu 和 apt 的驱动开发环境的搭建

如果我们使用 Ubuntu 系统，那么就可以使用它提供的

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
sudo apt-get install linux-headers-`uname -r`
```

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

如果可以看到对应的内容（即`build`指向`/usr/src/`下的一个目录），那么就说明编译好的内核已经装好了。

#### 基于编译内核的驱动开发环境搭建

**下载内核源码**

通常我们说的 linux kernel 可以在

如果使用的是 Ubuntu 系统，下载源码可以在`apt`里下载：

```bash

```

##### 编译内核

###### 使用源码编译

**在 Ubuntu 22.04 下编译**

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

### hello, world 驱动

为了测试上面搭建的驱动开发环境是否成功，我们使用一个 hello, world 项目测试一下。

首先，创建一个项目文件夹：`mkdir driver_test`，然后进入这个目录：`cd driver_test`。

接着，创建一个`hello_world.c`文件，然后写入以下内容：

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
MODULE_LICENSE("GPL");
```

然后我们创建一个`Makefile`文件：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-41-generic  # 这里要和我们下载的内核版本保持一致
obj-m  +=  hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

最后进行编译：`make`。编译完成后，我们可以在当前目录下看到`hello_world.ko`文件。

运行：`sudo insmod hello_world.ko`

调出 log，检查是否运行成功：`sudo dmesg`

如果输出的最后几行有`[ 2793.700004] <1>hello my module`，那么就说明驱动运行成功了。

最后卸载驱动：`sudo rmmod hello_world`。

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

Syntax:

1. Init function

    ```c
    static int __init hello_world_init(void) /* Constructor */
    {
        return 0;
    }
    module_init(hello_world_init);
    ```

1. Exit function

    ```c
    void __exit hello_world_exit(void)
    {

    }
    module_exit(hello_world_exit);
    ```

注：内核中打印函数使用`printk`。用法和`printf`几乎一样。

`printk()`使用的枚举消息以及例子：

* `KERN_EMERG`: Used for emergency messages, usually those that precede a crash.

* `KERN_ALERT`: A situation requiring immediate action.

* `KERN_CRIT`: Critical conditions are often related to serious hardware or software failures.

* `KERN_ERR`: Used to report error conditions; device drivers often use KERN_ERR to report hardware difficulties.

* `KERN_WARNING`: Warnings about problematic situations that do not, in themselves, create serious problems with the system.

* `KERN_NOTICE`: Situations that are normal, but still worthy of note. A number of security-related conditions are reported at this level.

* `KERN_INFO`: Informational messages. Many drivers print information about the hardware they find at startup time at this level.

* `KERN_DEBUG`: Used for debugging messages.

Example:

```c
printk(KERN_INFO "Welcome To EmbeTronicX");
```

Note: In the newer Linux kernels, you can use the APIs below instead of this printk.

pr_info – Print an info-level message. (ex. pr_info("test info message\n")).
pr_cont – Continues a previous log message in the same line.
pr_debug – Print a debug-level message conditionally.
pr_err – Print an error-level message. (ex. pr_err(“test error message\n”)).
pr_warn – Print a warning-level message.

我们开始写我们的`hello_world.c`文件：

```c
#include <linux/init.h>
#include <linux/module.h>

int hello_init(void)  // 参数列表中的 void 不可省略，不然无法通过编译.
{
    printk("<1>""hello my module\n");
    return 0;
}

void hello_exit(void)  // exit 不需要返回值
{
    printk("<1>""bye bye!\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");  // 不加这行的话，无法通过编译。MODULE_LICENSE 必须大写，不然无法通过编译
```

接着，我们写 Makefile：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-32-generic
obj-m += hello_world.o
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

（现在好像已经升级了，不管 printk level 大于还是小于当前 console 的 level，都不会在 console 上输出）

一个其他网站上提供的 hello world example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple hello world driver
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/module.h>
 
/*
** Module Init function
*/
static int __init hello_world_init(void)
{
    printk(KERN_INFO "Welcome to EmbeTronicX\n");
    printk(KERN_INFO "This is the Simple Module\n");
    printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

/*
** Module Exit function
*/
static void __exit hello_world_exit(void)
{
    printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple hello world driver");
MODULE_VERSION("2:1.0");
```

Makefile 的 example:

```Makefile
obj-m += hello_world.o
 
ifdef ARCH
  #You can update your Beaglebone path here.
  KDIR = /home/embetronicx/BBG/tmp/lib/modules/5.10.65/build
else
  KDIR = /lib/modules/$(shell uname -r)/build
endif
 
all:
  make -C $(KDIR)  M=$(shell pwd) modules
 
clean:
  make -C $(KDIR)  M=$(shell pwd) clean
```

交叉编译：`sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi-`

获得模块的一些信息：`modinfo hello_world_module.ko`

### 内核模块参数（Module Parameters Macros）

Module Parameters Macros：

* `module_param();`

    `module_param(name, type, perm);`

    `module_param(模块参数名，模块参数，访问权限);`

    This macro is used to initialize the arguments. module_param takes three parameters: the name of the variable, its type, and a permissions mask to be used for an accompanying sysfs entry.

    The macro should be placed outside of any function and is typically found near the head of the source file. module_param() macro, defined in linux/moduleparam.h.

    Example:

    `module_param(valueETX, int, S_IWUSR|S_IRUSR);`

* `module_param_array();`

    `module_param_array(name,type,num,perm);`

    `module_param_array(数组模块参数名，数组元素类型，NULL，访问权限);`

    This macro is used to send the array as an argument to the Linux device driver.

    Where,

    `name` is the name of your array (and of the parameter),

    `type` is the type of the array elements,

    `num` is an integer variable (optional) otherwise NULL,

    `perm` is the usual permissions value.

* `module_param_cb()`

    This macro is used to register the callback. Whenever the argument (parameter) got changed, this callback function will be called.

There are several types of permissions:

S_IWUSR
S_IRUSR
S_IXUSR
S_IRGRP
S_IWGRP
S_IXGRP

In this S_I is a common header.
R = read ,W =write ,X= Execute.
USR =user ,GRP =Group
Using OR ‘|’ (or operation) we can set multiple permissions at a time.

在代码中对模块参数的使用和普通变量没有区别。

Numerous types are supported for module parameters:

* `bool`

    A boolean (true or false) value (the associated variable should be of type int).

* `invbool`

    The invbool type inverts the value, so that true values become false and vice versa.

* `charp`

    A char pointer value. Memory is allocated for user-provided strings, and the pointer is set accordingly.

* `int`, `long`, `short`, `uint`, `ulong`, `ushort`
    
    Basic integer values of various lengths. The versions starting with u are for unsigned values.

Examples:

```c
int param_int = 10;
unsigned short param_ushort = 20;
char *param_string = "hello";
int param_arr[3] = {100, 200, 300};

module_param(param_int, int, 0775);
module_param(param_ushort, ushort, 0);
module_param(param_string, charp, 0644);
module_param_array(param_arr, int, NULL, 0755);
```

打印模块参数：

```c
int modparam_init(void)  // 这是一个普通函数，函数的名字可以随便改
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

当模块加载成功后，那些访问权限非 0 的模块会在以下路径下：

`/sys/module/模块名/parameters`

存在和模块参数名同名的文件，这些文件的权限来自于模块参数的权限。文件的内容与模块参数的值相同。因此可以通过修改文件中保存的数据，对模块参数进行修改：

```bash
sudo su
echo 1 > /sys/module/hello_world_module/parameters/my_param
```

如果我们需要监测模块变量的改变，那么可以用下面的代码：

```c
struct kernel_param_ops 
{
    int (*set)(const char *val, const struct kernel_param *kp);
    int (*get)(char *buffer, const struct kernel_param *kp);
    void (*free)(void *arg);
};
```

Example:

```c
/***************************************************************************//**
*  \file       hello_world.c
*
*  \details    Simple hello world driver
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/module.h>
#include<linux/moduleparam.h>
 
int valueETX, arr_valueETX[4];
char *nameETX;
int cb_valueETX = 0;
 
module_param(valueETX, int, S_IRUSR|S_IWUSR);                      // integer value
module_param(nameETX, charp, S_IRUSR|S_IWUSR);                     // String
module_param_array(arr_valueETX, int, NULL, S_IRUSR|S_IWUSR);      // Array of integers
 
/*----------------------Module_param_cb()--------------------------------*/
int notify_param(const char *val, const struct kernel_param *kp)
{
        // param_set_int 好像是把字符串转换成 int
        int res = param_set_int(val, kp); // Use helper for write variable
        if(res==0) {
                printk(KERN_INFO "Call back function called...\n");
                printk(KERN_INFO "New value of cb_valueETX = %d\n", cb_valueETX);
                return 0;
        }
        return -1;
}
 
const struct kernel_param_ops my_param_ops = 
{
        .set = &notify_param, // Use our setter ...
        .get = &param_get_int, // .. and standard getter
};
 
module_param_cb(cb_valueETX, &my_param_ops, &cb_valueETX, S_IRUGO|S_IWUSR );
/*-------------------------------------------------------------------------*/

/*
** Module init function
*/
static int __init hello_world_init(void)
{
        int i;
        printk(KERN_INFO "ValueETX = %d  \n", valueETX);
        printk(KERN_INFO "cb_valueETX = %d  \n", cb_valueETX);
        printk(KERN_INFO "NameETX = %s \n", nameETX);
        for (i = 0; i < (sizeof arr_valueETX / sizeof (int)); i++) {
                printk(KERN_INFO "Arr_value[%d] = %d\n", i, arr_valueETX[i]);
        }
        printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

/*
** Module Exit function
*/
static void __exit hello_world_exit(void)
{
    printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple hello world driver");
MODULE_VERSION("1.0");
```

编译：

```Makefile
obj-m += hello_world_module.o

KDIR = /lib/modules/$(shell uname -r)/build

all:
    make -C $(KDIR)  M=$(shell pwd) modules

clean:
    make -C $(KDIR)  M=$(shell pwd) clean
```

加载模块：

```bash
sudo insmod hello_world_module.ko valueETX=14 nameETX="EmbeTronicX" arr_valueETX=100,102,104,106
```

改变模块参数值：

方法一：

```bash
sudo sh -c "echo 13 > /sys/module/hello_world_module/parameters/cb_valueETX"
```

方法二：

Type sudo su. Then enter the password if it asks. Then do echo `13 > /sys/module/hello_world_module/parameters/cb_valueETX`

然后我们可以在`dmesg`里看到参数值变化的消息。

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
    MODULE_DESCRIPTION("this is a module symbol!");
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

    在加载模块的时候，应该先加载导出符号的模块，再加载使用符号的模块。卸载时，顺序要相反。

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

In fact, all device drivers that are neither storage nor network device drivers are some type of character driver.

1. 字符设备驱动的实现

    1. 概述

    驱动是沟通硬件和上层应用的媒介，字符设备驱动通过字符设备文件来访问，访问设备文件使用文件IO，在用户层访问设备文件和普通文件的方法是没有区别的

    `open, close, read, write, lseek, ioctl, mmap, stat`

1. 通过设备文件找到对应的驱动

    Linux 中所有的设备文件存放在`/dev`中。

    内核中有很多的字符设备驱动，这些字符设备驱动如何与对应的字符设备文件匹配，实际上是通过设备号来找到对应的字符设备驱动。

    设备号用 32 位的一个`dev_t`类型的变量来表示（无符号整型），高 12 位表示主设备号，后 20 位表示次设备号。

    The `dev_t` type (defined in `<linux/types.h>`) is used to hold device numbers—both the major and minor parts. `dev_t` is a 32-bit quantity with 12 bits set aside for the major number and 20 for the minor number.

    主设备号用来区分不同类型的设备。

    在`/proc/devices`文件中可以查找到设备号与对应的设备类型。

    内核中提供了操作设备号的宏：

    ```c
    MAJOR(设备号);  // 通过设备号获取主设备号  MAJOR(dev_t dev);
    MINOR(设备号);  // 通过设备号获取次设备号  MINOR(dev_t dev);
    MKDEV(主设备号, 次设备号);  // 通过主设备号和次设备号构造设备号  MKDEV(int major, int minor);
    ```

    这些宏都是位运算，有空可以看看。

    Example:

    ```c
    dev_t dev = MKDEV(235, 0);

    register_chrdev_region(dev, 1, "Embetronicx_Dev");
    ```

### 设备号

设备号在内核中属于资源，需要向内核申请。

需要包含头文件：

```c
#include <linux/cdev.h>
#include <linux/fs.h>
```

 1. 静态申请（Statically allocating）

    首先选择一个内核中未被使用的主设备号（`cat /proc/devices`），比如 220。根据设备个数分配次设备号，一般从 0 开始。

    构造设备号：`MKDEV(220,0);`

    调用`register_chrdev_region(dev_t from, unsigned count, const char *name);`

    params:

    * `from`: 要申请的起始设备号

    * `count`: 设备数量

        `count` is the total number of contiguous device numbers you are requesting. Note that, if the count is large, the range you request could spill over to the next major number; but everything will still work properly as long as the number range you request is available.

    * `name`: 设备号在内核中对应的名称

    返回 0 表示成功，返回非 0 表示失败。

    不再使用设备号需要注销：

    ```c
    unregister_chrdev_region(dev_t from, unsigned count);
    ```

    params:

    `from`: 要注销的起始设备号

    `count`: 设备号的个数

    一般在卸载模块的时候释放设备号。The usual place to call unregister_chrdev_region would be in your module’s cleanup function (Exit Function).

    Example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/cdev.h>
    #include <linux/fs.h>

    int hello_init(void)
    {
        printk("<1>""hello my module\n");

        // allocate a device number
        register_chrdev_region(MKDEV(220,0), 1, "hlc_dev");
        return 0;
    }

    void hello_exit(void)
    {
        unregister_chrdev_region(MKDEV(220,0), 1);
        printk("<1>""bye bye!\n");
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

 2. 动态申请（Dynamically Allocating）

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


**Difference between static and dynamic method**

A static method is only really useful if you know in advance which major number you want to start with. With the Static method, you are telling the kernel that, what device numbers you want (the start major/minor number and count) and it either gives them to you or not (depending on availability).

With the Dynamic method, you are telling the kernel that how many device numbers you need (the starting minor number and count) and it will find a starting major number for you, if one is available, of course.

Partially to avoid conflict with other device drivers, it’s considered preferable to use the Dynamic method function, which will dynamically allocate the device numbers for you.

The disadvantage of dynamic assignment is that you can’t create the device nodes in advance, because the major number assigned to your module will vary. For normal use of the driver, this is hardly a problem, because once the number has been assigned, you can read it from /proc/devices.

Examples:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple linux driver (Statically allocating the Major and Minor number)
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/module.h>
#include <linux/fs.h>

//creating the dev with our custom major and minor number
dev_t dev = MKDEV(235, 0);

/*
** Module Init function
*/
static int __init hello_world_init(void)
{
    register_chrdev_region(dev, 1, "Embetronicx_Dev");
    printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
    printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

/*
** Module exit function
*/
static void __exit hello_world_exit(void)
{
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="42272f202736302d2c2b213a02252f232b2e6c212d2f">[email protected]</a>>");
MODULE_DESCRIPTION("Simple linux driver (Statically allocating the Major and Minor number)");
MODULE_VERSION("1.0");
```

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple linux driver (Dynamically allocating the Major and Minor number)
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/module.h>
#include<linux/kdev_t.h>
#include<linux/fs.h>
 
dev_t dev = 0;

/*
** Module Init function
*/
static int __init hello_world_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "Embetronicx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number for device 1\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
        printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
        
        return 0;
}

/*
** Module exit function
*/
static void __exit hello_world_exit(void)
{
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="4a2f27282f3e3825242329320a2d272b232664292527">[email protected]</a>>");
MODULE_DESCRIPTION("Simple linux driver (Dynamically allocating the Major and Minor number)");
MODULE_VERSION("1.1");
```

### cdev

cdev 在内核中代表一个字符设备驱动。

```c
struct cdev {
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
struct cdev my_cdev;  // 声明 cdev

int my_cdev_open(struct inode *inode, struct file *flip)
{
    
}

int my_cdev_read()
{

}

struct file_operations cdd_fops = {  // GNU C 额外语法，选择性地初始化
    .owner = THIS_MODULE,
    .open = my_cdev_open,
    .read = my_cdev_read,
    .write = cdd_write,
    .unlocked_ioctl = cdd_ioctl,  // ioctl 接口
    .release = cdd_release,  // 对应用户 close 接口
};


```

`inode`是文件的节点结构，用来存储文件静态信息。文件创建时，内核中就会有一个 inode 结构

`lsmod`除了可以列出当前已经加载的模块，还可以显示模块之间的依赖关系。

应用程序 app 先找到设备文件，设备文件通过设备号找到设备驱动，然后再调用相关的函数。设备号如何找到设备驱动？首先可以通过设备号找到`cdev`结构体，然后从`cdev`结构体找到`file_operations`结构体，再在这个结构体里找对应的驱动函数。

每个文件都对应内核中一个`inode` struct。文件被打开时，内核会创建一个`file` struct，记录一些信息。

Example:

`hello_world.c`:

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/fs.h>

int m_open(struct inode *inode, struct file *file_ptr)
{
    printk("in m_open function ...\n");
    return 0;
}

int m_release(struct inode *inode, struct file *file_ptr)
{
    printk("in m_release function ...\n");
    return 0;
}

long int m_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset)
{
    printk("in m_read function ...\n");
    return 0;
}

long int m_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    printk("in m_write function ...\n");
    return 0;
}

long m_ioctl(struct file *, unsigned int, unsigned long)
{
    printk("in m_ioctl function ...\n");
    return 0;
}

dev_t dev_num;
struct cdev m_dev;
struct file_operations m_ops = {
    .owner = THIS_MODULE,
    .open = m_open,
    .read = m_read,
    .write = m_write,
    .release = m_release,
    .unlocked_ioctl = m_ioctl,
};

int hello_init(void)
{
    printk("<1>""hello my module\n");

    // allocate a device number
    dev_num = MKDEV(220,0);
    register_chrdev_region(dev_num, 1, "hlc_dev");

    cdev_init(&m_dev, &m_ops);
    cdev_add(&m_dev, dev_num, 1);
    return 0;
}

void hello_exit(void)
{
    cdev_del(&m_dev);
    unregister_chrdev_region(dev_num, 1);
    printk("<1>""bye bye!\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

对其编译后，加载模块：`sudo insmod hello_world.ko`。

Manually Creating Device File

We can create the device file manually by using mknod.

`mknod -m <permissions> <name> <device type> <major> <minor>`

* `<name>` – your device file name that should have a full path (/dev/name)

* `<device type>` – Put c or b

    * `c` – Character Device

    * `b` – Block Device

* `<major>` – major number of your driver

* `<minor>` – minor number of your driver

* `-m <permissions>` – optional argument that sets the permission bits of the new device file to permissions

Example: `sudo mknod -m 666 /dev/etx_device c 246 0`

然后我们创建一个设备文件：`sudo mknod /dev/hello_dev c 220 0`。这里的主设备号和次设备号要和前面代码中的保持一致。

最后我们进行测试：`cat /dev/hello_dev`，然后执行`sudo smesg`，可以看到一些输出：

```
[26495.940998] <1>hello my module
[26502.305035] in m_open function ...
[26502.305042] in m_read function ...
[26502.305048] in m_release function ...
```

可以看到驱动正常运行。

**Automatically Creating Device File**

The automatic creation of device files can be handled with `udev`. `Udev` is the device manager for the Linux kernel that creates/removes device nodes in the `/dev` directory dynamically. Just follow the below steps.

Include the header file `linux/device.h` and `linux/kdev_t.h`
Create the struct `Class`
Create `Device` with the `class` which is created by the above step

使用代码创建 cdev 设备文件：

需要头文件：

```c
#include <linux/device.h>
```

1. Create the class

    This will create the struct class for our device driver. It will create a structure under `/sys/class/`. 创建的设备类在`/sys/class`目录下。

    `struct class * class_create(struct module *owner, const char *name);`

     ```c
    struct class *class_create(模块所有者, const char *name);
    ```

    * `owner` – pointer to the module that is to “own” this struct class

    * `name` – pointer to a string for the name of this class

    This is used to create a struct class pointer that can then be used in calls to class_device_create. The return value can be checked using IS_ERR() macro.

销毁设备类：

```c
void class_destroy(struct class *cls);
```

创建设备文件（设备节点）：

`struct device *device_create(struct *class, struct device *parent, dev_t dev, void * drvdata, const char *fmt, ...);`

`class` – pointer to the struct class that this device should be registered to

`parent` – pointer to the parent struct device of this new device, if any

`devt` – the dev_t for the char device to be added

`drvdata` – the data to be added to the device for callbacks

`fmt` – string for the device’s name

`...` – variable arguments

A “dev” file will be created, showing the dev_t for the device, if the dev_t is not 0,0. If a pointer to a parent struct device is passed in, the newly created struct device will be a child of that device in sysfs. The pointer to the struct device will be returned from the call. Any further sysfs files that might be required can be created using this pointer. The return value can be checked using IS_ERR() macro.

```c
struct device *device_create(struct class *class, struct device *parent, dev_t devt, void *drvdata, const char *fmt, ...);
```

Params:

* `parent`: 父设备指针
* `devt`：设备号
* `drvdata`：额外的数据
* `fmt`：设备文件名

成功会在`/dev`目录下生成对应的设备文件，并返回设备指针

销毁设备文件：

`void device_destroy(struct class *class, dev_t devt)`

内核中指针了一些宏，用于判断指针是否出错。

```
IS_ERR(指针)  // 返回真表示出错
IS_ERR_OR_NULL(指针)  // 
PTR_ERR(指针)  // 将出错的指针转换成错误码
ERR_PTR(错误码)  // 将错误码转换成指针
```

Example：

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple linux driver (Automatically Creating a Device file)
*
*  \author     EmbeTronicX
*
*  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/err.h>
#include <linux/device.h>
 
dev_t dev = 0;
static struct class *dev_class;
 
/*
** Module init function
*/
static int __init hello_world_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number for device\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating struct class*/
        dev_class = class_create(THIS_MODULE,"etx_class");
        if(IS_ERR(dev_class)){
            pr_err("Cannot create the struct class for device\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device\n");
            goto r_device;
        }
        pr_info("Kernel Module Inserted Successfully...\n");
        return 0;
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        return -1;
}
 
/*
** Module exit function
*/
static void __exit hello_world_exit(void)
{
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        unregister_chrdev_region(dev, 1);
        pr_info("Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="4c29212e29383e2322252f340c2b212d2520622f2321">[email protected]</a>>");
MODULE_DESCRIPTION("Simple linux driver (Automatically Creating a Device file)");
MODULE_VERSION("1.2");
```

Linux 中的错误处理使用`goto`：

```c
// stop 1
if (出错) {
    goto 标签1；
}

// step 2
if（出错） {
    goto 标签2；
}

// step 3
if (出错) {
    goto 标签3;
}

标签3;
    复原第2步
标签2;
    复原第1步
标签1;
    return 错误码;
```

合并注册设备号和注册 cdev:

```c
int register_chrdev(unsigned int major, const char *name, const struct file_operations *fops);
```

当打开一个设备文件时，kernel 会根据设备号遍历 cdev 数组，找到对应的 cdev 结构体对象，然后把里面的`file_operatorions`里面的函数指针赋值给文件结构体`struct file`的`file_operations`里对应的函数。

通过代码测试设备文件：

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include<sys/types.h>
#include <unistd.h>

int main()
{
    char ch = 0;
    char buf[10] = {};

    int fd = open("/dev/hello_dev", O_RDWR);
    if (fd == -1)
    {
        perror("open");
        exit(-1);
    }

    printf("open successed! fd = %d\n", fd);

    while (1) {
        ch = getchar();
        getchar();

        if (ch == 'q')
            break;
        switch(ch)
        {
            case 'r':
                read(fd, buf, 0);
                break;
            case 'w':
                write(fd, buf, 0);
                break;
            default:
                printf("error input\n");
        }
    }
}
```

* ioctl 接口

    ioctl 是一个专用于硬件操作的接口，用于和实际数据传输相区分

    1. 用户层接口

        ```c
        #include <sys/ioctl.h>

        int ioctl(int fd, unsigned long request, ...);
        ```

        * `fd`: 文件描述符

    1. 内核接口

        头文件：

        ```c
        #include <linux/ioctl.h>
        ```

        对应`file_operations`中的成员：

        ```c
        long (*unlocked_ioctl) (struct file *filp, unsigned int cmd, unsigned long data);
        ```

    命令构造：

    Linux 内核提供构造 ioctl 命令的宏：

    ```c
    #define HELLO_ONE _IO('k',0)
    #define HELLO_TWO _IO('k',1)

    long cdd_ioctl(struct file *filp, xxxx)
    {
        printk("enter cdd_ioctl!\n");

        // 不同的命令对应不同的操作
        switch(cmd) {
            case HELLO_ONE:
                printk("hello one\n");
                break;
            case HELLO_TWO:
                printk("hello two\n");
                break;
            default:
                return -EINVAL;
        }
    }
    ```

我们可以使用`ls -l /dev`查看已经创建的设备文件。First of all, note that the first letter of the permissions field is denoted that driver type. Device files are denoted either by b, for block devices, or c, for character devices. Also, note that the size field in the ls -l listing is replaced by two numbers, separated by a comma. The first value is the major device number and the second is the minor device number.


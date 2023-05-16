# Linux Driver Note

Ref:

* <https://embetronicx.com/tutorials/linux/device-drivers/>

## Introduction

### 驱动开发环境的搭建

#### 基于 Ubuntu 和 apt 的驱动开发环境的搭建

如果我们使用 Ubuntu 系统，那么就可以在它编译好的内核库的基础上开发驱动。如果使用其他系统，那么有可能需要自己编译内核。

linux 驱动是以 module 的形式加载到内核中的。Ubuntu 的内核是 signed 版本，我们自己写的 module 是 unsigned 版本，没有办法直接加载（在执行`insmod`时会报错）。目前我的解决办法是，先更新一下内核，换成 unsigned 版本或者 hwe 版本（hwe 版本表示支持最新硬件）：

```bash
apt install linux-image-generic-hwe-22.04
```

接下来我们把原来的 kernel 删掉，然后下载新的：

```bash
sudo apt update && sudo apt upgrade
sudo apt remove --purge linux-headers-*
sudo apt autoremove && sudo apt autoclean
sudo apt-get install linux-headers-`uname -r`
```

注：我们可以用`uname -r`查看当前内核的版本，`uname -a`查看系统的完整版本。

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

1. 下载内核源码

（这部分目前用不到，先不写了）

通常我们说的 linux kernel 可以在

如果使用的是 Ubuntu 系统，下载源码可以在`apt`里下载：

```bash

```

1. 编译内核（在 Ubuntu 22.04 下编译）

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

为了测试上面搭建的驱动开发环境是否成功，我们使用一个 hello world 项目测试一下。

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

### 交叉编译

（有时间了填下这个坑）

交叉编译：`sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi-`

## 内核模块

内核模块编程注意事项：

1. 不能使用 C 库和 C 标准头文件
1. 使用 GNU C （在 ANSI C 上加了些语法）
1. 没有内存保护机制
1. 不能处理浮点运算
1. 注意并发互斥和可移植性

### 内核模块的加载与卸载

**内核模块的加载函数与卸载函数**

需要包含头文件

```cpp
#include <linux/init.h>
#include <linux/module.h>
```

Syntax:

1. 加载函数（Init function）

    ```c
    static int __init hello_world_init(void)  /* Constructor */
    {
        return 0;  //返回 0 表示加载成功
    }
    module_init(hello_world_init);  // 使用宏来注册加载函数
    ```

2. 卸载函数（Exit function）

    ```c
    void __exit hello_world_exit(void)
    {

    }
    module_exit(hello_world_exit);  // 使用宏来注册卸载函数
    ```

**`printk()`**

注：内核驱动中 IO 输出函数使用`printk()`，用法和`printf()`几乎一样。

`printk()`可打印的消息分为不同的类型，在字符串前使用宏字符串进行修饰。

消息类型如下：

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

Note: In the newer Linux kernels, you can use the APIs below instead of this `printk`.

* `pr_info()` – Print an info-level message. (ex. `pr_info("test info message\n")`).
* `pr_cont()` – Continues a previous log message in the same line.
* `pr_debug()` – Print a debug-level message conditionally.
* `pr_err()` – Print an error-level message. (ex. `pr_err(“test error message\n”)`).
* `pr_warn()` – Print a warning-level message.

`printk()`的打印的日志级别有 0-7，还可以是无级别。如果无法看到输出内容，可以参考这几个网站的解决方式：

1. <http://www.jyguagua.com/?p=708>

2. <https://blog.csdn.net/qustDrJHJ/article/details/51382138>

我们可以使用`cat /proc/sys/kernel/printk`查看当前日志的级别。

第一个数为内核默认打印级别，只有当`printk`的打印级别高于内核默认打印级别时，`printk`打印的信息才能显示 。

第二个数为`printk()`的默认打印级别。

修改内核的默认打印级别（即修改第 1 个数）：

`echo 5 > /proc/sys/kernel/printk`

（现在好像已经升级了，不管 printk level 大于还是小于当前 console 的 level，都不会在 console 上输出）

**有关模块加载与卸载的一个 Example**

`hello_world.c`：

```c
#include <linux/init.h>
#include <linux/module.h>

int hello_init(void)  // 参数列表中的 void 不可省略，不然无法通过编译.
{
    printk(KERN_INFO "hello my module\n");
    return 0;
}

void hello_exit(void)  // exit 不需要返回值
{
    printk(KERN_INFO "bye bye!\n");
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

此时可以使用`sudo insmod hello_world.ko`插入模块，使用`sudo rmmod hello_world`移除模块，`sudo lsmod`查看已经加载的模块，`sudo dmesg`查看日志输出。

（如果还是无法`insmod`，或许需要取消 secure boot：<https://askubuntu.com/questions/762254/why-do-i-get-required-key-not-available-when-install-3rd-party-kernel-modules>）

除了添加`MODULE_LICENSE()`外，可选的添加信息有：

* `MODULE_AUTHOR`: 模块作者

    Example: `MODULE_AUTHOR("hlc")`

* `MODULE_VERSION`: 模块版本
  
    Example: `MODULE_VERSION("1.0")`

* `MODULE_DESCRIPTION`：模块描述

    Example: `MODULE_DESCRIPTION("this is my first module")`

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

获得模块的一些信息：`modinfo hello_world_module.ko`

### 模块参数（Module Parameters Macros）

Module Parameters Macros：

* `module_param();`

    `module_param(name, type, perm);`

    `module_param(模块参数名，模块参数，访问权限);`

    This macro is used to initialize the arguments. module_param takes three parameters: the name of the variable, its type, and a permissions mask to be used for an accompanying sysfs entry.

    The macro should be placed outside of any function and is typically found near the head of the source file. `module_param()` macro, defined in `linux/moduleparam.h`.

    Parameters:

    * `type`

        可以是下面几个之一：`byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `charp`, `bool`, `invbool`;

    * `perm` is the usual permissions value.

        There are several types of permissions:

        这些宏定义在`stat.h`中

        * `S_IWUSR`, `S_IRUSR`, `S_IXUSR`
        * `S_IRGRP`, `S_IWGRP`, `S_IXGRP`
        * `S_IROTH`, `S_IWOTH`, `S_IXOTH`

        可以看出来，`S_I`是一个 common prefix，R = read, W = write, X = Execute. USR = user, GRP = Group。

        Using `|` (OR operation) we can set multiple permissions at a time.

        在使用`S_IROTH`和`S_IWOTH`时会编译时报错，但是如果使用`0775`，可以顺序地给 others 加上`r`权限。不清楚为什么。

    Example:

    `module_param(valueETX, int, S_IWUSR | S_IRUSR);`

* `module_param_array();`

    `module_param_array(name, type, int *num, permissions);`

    `module_param_array(数组模块参数名，数组元素类型，NULL，访问权限);`

    This macro is used to send the array as an argument to the Linux device driver.

    Parameters:

    * `name`
    
        The name of the array (and of the parameter)

    * `type`
    
        The type of the array elements

    * `num`

        An integer variable (optional) otherwise `NULL`。

        在命令行中传递的数组的元素个数。

        比如`sudo insmod hello_world.ko m_arr=3,4,5`，那么`num`会被改写成`3`。

* `module_param_cb()`

    This macro is used to register the callback. Whenever the argument (parameter) got changed, this callback function will be called.

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

说明：

1. 这里的`0775`并不是和 linux 文件权限一一对应。使用`0775`作为权限后，得到的参数文件的权限如下所示：

    ```
    -rw-rw-r-- 1 root root 4096  5月 16 10:52 /sys/module/hello_world/parameters/a
    ```

    因为这个数字和 linux 文件的权限并不是对应关系，所以使用`0776`，`0777`，`777`等作为参数时会编译报错。

    正常情况下还是使用`S_IWUSR`这些标志位吧。

2. `0775`前面这个`0`必须加上，不然会编译报错。目前不清楚是为什么。

3. `unsigned short`定义的变量，在`module_param()`中注册模块参数时，必须使用`ushort`作为类型。

    在`module_param()`中填`unsigned short`会编译报错。

    如果使用`typedef unsigned short us;`，然后在`module_param()`中填`us`，同样也会编译报错。

4. 如果数组没有被初始化，或初始化的元素数量不够，那么元素的默认值都是 0。

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

数组可以只传递一部分元素，此时会重新计算数组的长度，并重新赋值。比如在代码中定义了数组容量为`3`，并且初始化了所有元素，但是在命令行中只传递了 2 个元素的数组，那么`/sys/module/xxx/parameters`中的对应文件也只会显示 2 个元素。

如果命令行传递的元素的数量超出数组的容量，那么会报错。

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

然后我们可以在`dmesg`里看到参数值变化的消息：

```
[ 1688.610775] ValueETX = 14  
[ 1688.610782] cb_valueETX = 0  
[ 1688.610784] NameETX = EmbeTronicX 
[ 1688.610785] Arr_value[0] = 100
[ 1688.610786] Arr_value[1] = 102
[ 1688.610787] Arr_value[2] = 104
[ 1688.610788] Arr_value[3] = 106
[ 1688.610789] Kernel Module Inserted Successfully...
[ 1849.370708] Call back function called...
[ 1849.370714] New value of cb_valueETX = 13
[ 1880.687099] Kernel Module Removed Successfully...
```

### 模块符号的导出

模块导出符号可以将模块中的变量/函数导出，供内核其他模块使用。

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

two ways of allocating and initializing one of these structures:

1. Runtime Allocation

    ```c
    struct cdev *my_cdev = cdev_alloc( );
    my_cdev->ops = &my_fops;
    ``````

1. Own allocation

    ```c
    void cdev_init(struct cdev *cdev, struct file_operations *fops);
    ```

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

A “dev” file will be created, showing the `dev_t` for the device, if the `dev_t` is not `0,0`. If a pointer to a parent struct device is passed in, the newly created struct device will be a child of that device in sysfs. The pointer to the struct device will be returned from the call. Any further sysfs files that might be required can be created using this pointer. The return value can be checked using IS_ERR() macro.

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


file operations syntax:

```c
struct file_operations {
    struct module *owner;
    loff_t (*llseek) (struct file *, loff_t, int);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    ssize_t (*read_iter) (struct kiocb *, struct iov_iter *);
    ssize_t (*write_iter) (struct kiocb *, struct iov_iter *);
    int (*iterate) (struct file *, struct dir_context *);
    int (*iterate_shared) (struct file *, struct dir_context *);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *, fl_owner_t id);
    int (*release) (struct inode *, struct file *);
    int (*fsync) (struct file *, loff_t, loff_t, int datasync);
    int (*fasync) (int, struct file *, int);
    int (*lock) (struct file *, int, struct file_lock *);
    ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long (*get_unmapped_area)(struct file *, unsigned long, unsigned long, unsigned long, unsigned long);
    int (*check_flags)(int);
    int (*flock) (struct file *, int, struct file_lock *);
    ssize_t (*splice_write)(struct pipe_inode_info *, struct file *, loff_t *, size_t, unsigned int);
    ssize_t (*splice_read)(struct file *, loff_t *, struct pipe_inode_info *, size_t, unsigned int);
    int (*setlease)(struct file *, long, struct file_lock **, void **);
    long (*fallocate)(struct file *file, int mode, loff_t offset,
              loff_t len);
    void (*show_fdinfo)(struct seq_file *m, struct file *f);
#ifndef CONFIG_MMU
    unsigned (*mmap_capabilities)(struct file *);
#endif
    ssize_t (*copy_file_range)(struct file *, loff_t, struct file *,
            loff_t, size_t, unsigned int);
    int (*clone_file_range)(struct file *, loff_t, struct file *, loff_t,
            u64);
    ssize_t (*dedupe_file_range)(struct file *, u64, u64, struct file *,
            u64);
};
```

fields:

* `struct module *owner`

    It is a pointer to the module that “owns” the structure. This field is used to prevent the module from being unloaded while its operations are in use. Almost all the time, it is simply initialized to `THIS_MODULE`, a macro defined in `<linux/module.h>`.

* `ssize_t (*read) (struct file *, char _ _user *, size_t, loff_t *);`

    This is used to retrieve data from the device. A null pointer in this position causes the read system call to fail with `-EINVAL` (“Invalid argument”). A non-negative return value represents the number of bytes successfully read (the return value is a “signed size” type, usually the native integer type for the target platform).

* `ssize_t (*write) (struct file *, const char _ _user *, size_t, loff_t *);`

    It is used to sends the data to the device. If NULL -EINVAL is returned to the program calling the write system call. The return value, if non-negative, represents the number of bytes successfully written.

* `int (*ioctl) (struct inode *, struct file *, unsigned int, unsigned long);`

    The ioctl system call offers a way to issue device-specific commands (such as formatting a track of a floppy disk, which is neither reading nor writing). Additionally, a few ioctl commands are recognized by the kernel without referring to the fops table. If the device doesn’t provide an ioctl method, the system call returns an error for any request that isn’t predefined (-ENOTTY, “No such ioctl for device”).

* `int (*open) (struct inode *, struct file *);`

    Though this is always the first operation performed on the device file, the driver is not required to declare a corresponding method. If this entry is NULL, opening the device always succeeds, but your driver isn’t notified.

* `int (*release) (struct inode *, struct file *);`

    This operation is invoked when the file structure is being released. Like open, release can be NULL.

Examples:

```c
static struct file_operations fops =
{
.owner          = THIS_MODULE,
.read           = etx_read,
.write          = etx_write,
.open           = etx_open,
.release        = etx_release,
};
```

Complete example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (File Operations)
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
#include <linux/cdev.h>
#include <linux/device.h>

dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

static struct file_operations fops =
{
    .owner      = THIS_MODULE,
    .read       = etx_read,
    .write      = etx_write,
    .open       = etx_open,
    .release    = etx_release,
};

/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Driver Open Function Called...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Driver Release Function Called...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        pr_info("Driver Read Function Called...!!!\n");
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Driver Write Function Called...!!!\n");
        return len;
}

/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_err("Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_err("Cannot create the struct class\n");
            goto r_class;
        }

        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device 1\n");
            goto r_device;
        }
        pr_info("Device Driver Insert...Done!!!\n");
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
static void __exit etx_driver_exit(void)
{
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
}

module_init(etx_driver_init);
module_exit(etx_driver_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (File Operations)");
MODULE_VERSION("1.3");
```

执行`dmesg`的输出：

```

```

执行`cat > /dev/etx_device`：`cat `command will open the driver, read the driver, and close the driver. So if I do `cat` to our driver, it should call the open, read, and release functions. Just check.

此时执行`dmesg`的输出：

```

```

Instead of doing `echo` and `cat` command in the terminal you can also use `open()`, `read()`, `write()`, `close()` system calls from user-space applications.

## Data exchange between kernel space and user space

Using this driver we can send strings or data to the kernel device driver using the write function. It will store that string in the kernel space. Then when I read the device file, it will send the data which is written by write by function to the userspace application.

申请内存：

```c
#include <linux/slab.h>

void *kmalloc(size_t size, gfp_t flags);
```

The allocated region is contiguous in physical memory.

* `size` - how many bytes of memory are required.

* `flags`– the type of memory to allocate.

    The flags argument may be one of:

    * `GFP_USER` – Allocate memory on behalf of the user. May sleep.

    * `GFP_KERNEL` – Allocate normal kernel ram. May sleep.

    * `GFP_ATOMIC` – Allocation will not sleep. May use emergency pools. For example, use this inside interrupt handler.

    * `GFP_HIGHUSER` – Allocate pages from high memory.

    * `GFP_NOIO` – Do not do any I/O at all while trying to get memory.

    * `GFP_NOFS` – Do not make any fs calls while trying to get memory.

    * `GFP_NOWAIT` – Allocation will not sleep.

    * `__GFP_THISNODE` – Allocate node-local memory only.

    * `GFP_DMA` – Allocation is suitable for DMA. Should only be used for kmalloc caches. Otherwise, use a slab created with SLAB_DMA.

    Also, it is possible to set different flags by OR’ing in one or more of the following additional flags:

    * `__GFP_COLD` – Request cache-cold pages instead of trying to return cache-warm pages.


    * `__GFP_HIGH` – This allocation has high priority and may use emergency pools.

    * `__GFP_NOFAIL` – Indicate that this allocation is in no way allowed to fail (think twice before using).

    * `__GFP_NORETRY` – If memory is not immediately available, then give up at once.

    * `__GFP_NOWARN` – If allocation fails, don’t issue any warnings.

    * `__GFP_REPEAT` – If allocation fails initially, try once more before failing.

    更多的参数，可以参考`linux/gfp.h`

释放内存：

```c
void kfree(const void *objp)
```

Parameters:

* `*objp` – pointer returned by `kmalloc`

* `copy_from_user()`

    Syntax:

    ```c
    unsigned long copy_from_user(void *to, const void __user *from, unsigned long  n);
    ```

    to – Destination address, in the kernel space


    from – The source address in the user space

    n – Number of bytes to copy

    Returns number of bytes that could not be copied. On success, this will be zero.

* `copy_to_user()`

    Syntax:

    ```c
    unsigned long copy_to_user(const void __user *to, const void *from, unsigned long  n);
    ```

    This function is used to Copy a block of data into userspace (Copy data from kernel space to user space).

    Parameters:

    `to` – Destination address, in the user space

    `from` – The source address in the kernel space

    `n` – Number of bytes to copy

    Returns number of bytes that could not be copied. On success, this will be zero.

Example:

```c
static int etx_open(struct inode *inode, struct file *file)
{
        /*Creating Physical memory*/
        if((kernel_buffer = kmalloc(mem_size , GFP_KERNEL)) == 0){
            printk(KERN_INFO "Cannot allocate memory in kernel\n");
            return -1;
        }
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        copy_from_user(kernel_buffer, buf, len);
        printk(KERN_INFO "Data Write : Done!\n");
        return len;
}

static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        copy_to_user(buf, kernel_buffer, mem_size);
        printk(KERN_INFO "Data Read : Done!\n");
        return mem_size;
}

static int etx_release(struct inode *inode, struct file *file)
{
        kfree(kernel_buffer);
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}
```

Full driver code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Real Linux Device Driver)
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
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include <linux/err.h>
 

#define mem_size        1024           //Memory Size
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
uint8_t *kernel_buffer;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);


/*
** File Operations structure
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};
 
/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        //Copy the data from the kernel space to the user-space
        if( copy_to_user(buf, kernel_buffer, mem_size) )
        {
                pr_err("Data Read : Err!\n");
        }
        pr_info("Data Read : Done!\n");
        return mem_size;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        //Copy the data to kernel space from the user-space
        if( copy_from_user(kernel_buffer, buf, len) )
        {
                pr_err("Data Write : Err!\n");
        }
        pr_info("Data Write : Done!\n");
        return len;
}

/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_info("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_info("Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_info("Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_info("Cannot create the Device 1\n");
            goto r_device;
        }
        
        /*Creating Physical memory*/
        if((kernel_buffer = kmalloc(mem_size , GFP_KERNEL)) == 0){
            pr_info("Cannot allocate memory in kernel\n");
            goto r_device;
        }
        
        strcpy(kernel_buffer, "Hello_World");
        
        pr_info("Device Driver Insert...Done!!!\n");
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
static void __exit etx_driver_exit(void)
{
  kfree(kernel_buffer);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Real Linux Device Driver)");
MODULE_VERSION("1.4");
```

User space application code:

```c
/***************************************************************************//**
*  \file       test_app.c
*
*  \details    Userspace application to test the Device driver
*
*  \author     EmbeTronicX
*
*  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
*
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int8_t write_buf[1024];
int8_t read_buf[1024];

int main()
{
        int fd;
        char option;
        printf("*********************************\n");
        printf("*******WWW.EmbeTronicX.com*******\n");

        fd = open("/dev/etx_device", O_RDWR);
        if(fd < 0) {
                printf("Cannot open device file...\n");
                return 0;
        }

        while(1) {
                printf("****Please Enter the Option******\n");
                printf("        1. Write               \n");
                printf("        2. Read                 \n");
                printf("        3. Exit                 \n");
                printf("*********************************\n");
                scanf(" %c", &option);
                printf("Your Option = %c\n", option);
                
                switch(option) {
                        case '1':
                                printf("Enter the string to write into driver :");
                                scanf("  %[^\t\n]s", write_buf);
                                printf("Data Writing ...");
                                write(fd, write_buf, strlen(write_buf)+1);
                                printf("Done!\n");
                                break;
                        case '2':
                                printf("Data Reading ...");
                                read(fd, read_buf, 1024);
                                printf("Done!\n\n");
                                printf("Data = %s\n\n", read_buf);
                                break;
                        case '3':
                                close(fd);
                                exit(1);
                                break;
                        default:
                                printf("Enter Valid option = %c\n",option);
                                break;
                }
        }
        close(fd);
}
```

这里只需要正常编译就可以了：

```bash
gcc -o test_app test_app.c
```

Note: Instead of using user space application, you can use echo and cat command.

## ioctl

There are many ways to Communicate between the Userspace and Kernel Space, they are:
IOCTL
Procfs
Sysfs
Configfs
Debugfs
Sysctl
UDP Sockets
Netlink Sockets

IOCTL is referred to as Input and Output Control, which is used to talk to device drivers. This system call is available in most driver categories.  The major use of this is in case of handling some specific operations of a device for which the kernel does not have a system call by default.

Some real-time applications of ioctl are Ejecting the media from a “cd” drive, changing the Baud Rate of Serial port, Adjusting the Volume, Reading or Writing device registers, etc. We already have the write and read function in our device driver. But it is not enough for all cases.

There are some steps involved to use IOCTL.

* Create IOCTL command in the driver

    ```c
    #include <linux/ioctl.h>

    #define WR_VALUE _IOW('a','a',int32_t*)
    #define RD_VALUE _IOR('a','b',int32_t*)

    #define "ioctl name" __IOX("magic number","command number","argument type")
    ```

    where IOX can be :

    * `IO`: an ioctl with no parameters
    * `IOW`: an ioctl with write parameters (copy_from_user)
    * `IOR`: an ioctl with read parameters (copy_to_user)
    * `IOWR`: an ioctl with both write and read parameters

    * The Magic Number is a unique number or character that will differentiate our set of ioctl calls from the other ioctl calls. some times the major number for the device is used here.
  
    * Command Number is the number that is assigned to the ioctl. This is used to differentiate the commands from one another.
  
    * The last is the type of data.

* Write IOCTL function in the driver

    ```c
    int  ioctl(struct inode *inode,struct file *file,unsigned int cmd,unsigned long arg)
    ```

    * `inode`: is the inode number of the file being worked on.
    * `file`: is the file pointer to the file that was passed by the application.
    * `cmd`: is the ioctl command that was called from the userspace.
    * `arg`: are the arguments passed from the userspace

    Within the function “ioctl” we need to implement all the commands that we defined above (`WR_VALUE`, `RD_VALUE`). We need to use the same commands in the `switch` statement which is defined above.

    Then we need to inform the kernel that the ioctl calls are implemented in the function “etx_ioctl“. This is done by making the fops pointer “unlocked_ioctl” to point to “etx_ioctl” as shown below.

    ```c
    /*
    ** This function will be called when we write IOCTL on the Device file
    */
    static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
    {
            switch(cmd) {
                    case WR_VALUE:
                            if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
                            {
                                    pr_err("Data Write : Err!\n");
                            }
                            pr_info("Value = %d\n", value);
                            break;
                    case RD_VALUE:
                            if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
                            {
                                    pr_err("Data Read : Err!\n");
                            }
                            break;
                    default:
                            pr_info("Default\n");
                            break;
            }
            return 0;
    }

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
            .owner          = THIS_MODULE,
            .read           = etx_read,
            .write          = etx_write,
            .open           = etx_open,
            .unlocked_ioctl = etx_ioctl,
            .release        = etx_release,
    };
    ```

* Create IOCTL command in a Userspace application

    ```c
    #define WR_VALUE _IOW('a','a',int32_t*)
    #define RD_VALUE _IOR('a','b',int32_t*)
    ```  

* Use the IOCTL system call in a Userspace

    ```c
    #include <sys/ioctl.h>
    long ioctl( "file descriptor","ioctl command","Arguments");
    ```

    * `file descriptor`: This the open file on which the ioctl command needs to be executed, which would generally be device files.
  
    * `ioctl command`: ioctl command which is implemented to achieve the desired functionality
    
    * `arguments`: The arguments need to be passed to the ioctl command.

    Example:

    ```c
    ioctl(fd, WR_VALUE, (int32_t*) &number); 

    ioctl(fd, RD_VALUE, (int32_t*) &value);
    ```

Driver full code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (IOCTL)
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
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include <linux/ioctl.h>
#include <linux/err.h>
 
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long     etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
};

/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read Function\n");
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write function\n");
        return len;
}

/*
** This function will be called when we write IOCTL on the Device file
*/
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
         switch(cmd) {
                case WR_VALUE:
                        if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
                        {
                                pr_err("Data Write : Err!\n");
                        }
                        pr_info("Value = %d\n", value);
                        break;
                case RD_VALUE:
                        if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
                        {
                                pr_err("Data Read : Err!\n");
                        }
                        break;
                default:
                        pr_info("Default\n");
                        break;
        }
        return 0;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_err("Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_err("Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device 1\n");
            goto r_device;
        }
        pr_info("Device Driver Insert...Done!!!\n");
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
static void __exit etx_driver_exit(void)
{
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (IOCTL)");
MODULE_VERSION("1.5");
```

User program full code:

```c
/***************************************************************************//**
*  \file       test_app.c
*
*  \details    Userspace application to test the Device driver
*
*  \author     EmbeTronicX
*
*  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
*
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include<sys/ioctl.h>
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int main()
{
        int fd;
        int32_t value, number;
        printf("*********************************\n");
        printf("*******WWW.EmbeTronicX.com*******\n");
 
        printf("\nOpening Driver\n");
        fd = open("/dev/etx_device", O_RDWR);
        if(fd < 0) {
                printf("Cannot open device file...\n");
                return 0;
        }
 
        printf("Enter the Value to send\n");
        scanf("%d",&number);
        printf("Writing Value to Driver\n");
        ioctl(fd, WR_VALUE, (int32_t*) &number); 
 
        printf("Reading Value from Driver\n");
        ioctl(fd, RD_VALUE, (int32_t*) &value);
        printf("Value is %d\n", value);
 
        printf("Closing Driver\n");
        close(fd);
}
```

This is a simple example of using ioctl in a Linux device driver. If you want to send multiple arguments, put those variables into the structure, and pass the address of the structure.
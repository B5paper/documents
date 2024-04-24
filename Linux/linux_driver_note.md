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

可以直接使用

```bash
apt install linux-image-`uname -r`
```

来安装。

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
    printk(KERN_INFO "hello my module\n");
    return 0;
}

void hello_exit(void)
{
    printk(KERN_INFO "bye bye!\n");
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
    static int __init hello_world_init(void)  // __init 并不是必要的。它代表什么意思？
    {
        return 0;  // 返回 0 表示加载成功
    }
    module_init(hello_world_init);  // 使用宏来注册加载函数
    ```

2. 卸载函数（Exit function）

    ```c
    void __exit hello_world_exit(void)  // __exit 也不是必要的，它代表什么含义？
    {

    }
    module_exit(hello_world_exit);  // 使用宏来注册卸载函数
    ```

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

module_init(hello_init);  // 这一行和下一行的分号都不是必要的，为什么？
module_exit(hello_exit);
MODULE_LICENSE("GPL");  // 不加这行的话，无法通过编译。MODULE_LICENSE 必须大写，不然无法通过编译。这一行末尾的分号是必要的，为什么？
```

另外一个 example:

```c
#include<linux/kernel.h>  // 这个头文件有什么用？
#include<linux/init.h>
#include<linux/module.h>

static int __init hello_world_init(void)  // __init 是什么意思？
{
    printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

static void __exit hello_world_exit(void)  // __exit 是什么意思？
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

接着，我们写 Makefile：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-32-generic
obj-m += hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

另外一个 makefile 的 example:

```Makefile
obj-m += hello_world.o
 
ifdef ARCH
  KDIR = /home/embetronicx/BBG/tmp/lib/modules/5.10.65/build
else
  KDIR = /lib/modules/$(shell uname -r)/build  # 为什么要在前面加上 shell
endif
 
all:
  make -C $(KDIR)  M=$(shell pwd) modules  # $(PWD) 和 $(shell pwd) 有什么不同
 
clean:
  make -C $(KDIR)  M=$(shell pwd) clean  # KDIR 是个变量，为什么要给它加上 $() ？
```

然后在当前文件夹下运行`make`，会生成`hello_world.ko`文件。这个文件就是我们需要的内核模块文件，ko 代表 kernel object。

此时可以使用`sudo insmod hello_world.ko`插入模块，使用`sudo rmmod hello_world`移除模块，`sudo lsmod`查看已经加载的模块，`sudo dmesg`查看日志输出。

（如果还是无法`insmod`，或许需要取消 secure boot：<https://askubuntu.com/questions/762254/why-do-i-get-required-key-not-available-when-install-3rd-party-kernel-modules>）

### 日志消息打印

**`printk()`**

内核模块中可以使用函数`printk()`将消息打印到日志中，用法和`printf()`几乎相同。

```c
printk("hello my module\n");
```

`printk()`可打印的消息有不同的级别，我们可以在字符串前使用下面的宏字符串进行修饰：

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
printk(KERN_INFO "this is a info level log");
printk(KERN_WARNING "this is a warning level log");
printk("this is a non-level log")
```

经实际测试，如果不加日志级别，那么为白色粗字，`KERN_NOTICE`及以下，全都是正常白字，`KERN_WARNING`的消息会白字加粗，`KERN_ERR`的消息会变成红字，`KERN_CRIT`消息为红字加粗，`KERN_ALERT`为红底黑字，`KERN_EMERG`又变成正常白字。（这个颜色可能和 terminal 配色有关）

**默认打印级别**

上面的日志级别可以对应到数字 0 - 7，如果不指定日志级别，那么就是无级别。之所以映射到数字，似乎是因为可以使用数字控制 console 的输出级别。（但是经实际测试，好像不怎么有用）

可以参考这几个网站的资料：

1. <http://www.jyguagua.com/?p=708>

2. <https://blog.csdn.net/qustDrJHJ/article/details/51382138>

我们可以使用`cat /proc/sys/kernel/printk`查看当前日志的级别。

第一个数为内核默认打印级别，只有当`printk`的打印级别高于内核默认打印级别时，`printk`打印的信息才能显示 。

第二个数为`printk()`的默认打印级别。

修改内核的默认打印级别（即修改第 1 个数）：

`echo 5 > /proc/sys/kernel/printk`

（现在好像已经升级了，不管 printk level 大于还是小于当前 console 的 level，都不会在 console 上输出）

**`printk`的替代函数**

In the newer Linux kernels, you can use the APIs below instead of this `printk`.

* `pr_info()` – Print an info-level message. (ex. `pr_info("test info message\n")`).
* `pr_cont()` – Continues a previous log message in the same line.
* `pr_debug()` – Print a debug-level message conditionally.
* `pr_warn()` – Print a warning-level message.
* `pr_err()` – Print an error-level message. (ex. `pr_err(“test error message\n”)`).

经实测，`pr_info()`为正常白字，`pr_cont()`为白字加粗，`pr_debug()`没有输出，`pr_warn()`为白字加粗，`pr_err()`为红字。

### 模块信息

除了添加`MODULE_LICENSE()`外，可选的添加信息有：

* `MODULE_AUTHOR`: 模块作者

    Example: `MODULE_AUTHOR("hlc")`

* `MODULE_VERSION`: 模块版本
  
    Example: `MODULE_VERSION("1.0")`

* `MODULE_DESCRIPTION`：模块描述

    Example: `MODULE_DESCRIPTION("this is my first module")`

获得模块的一些信息：`modinfo hello_world_module.ko`

### 模块参数（Module Parameters Macros）

Module Parameters Macros：

* `module_param();`

    `module_param(name, type, perm);`

    `module_param(模块参数名，模块参数，访问权限);`

    This macro is used to initialize the arguments. `module_param` takes three parameters: the name of the variable, its type, and a permissions mask to be used for an accompanying sysfs entry.

    The macro should be placed outside of any function and is typically found near the head of the source file. `module_param()` macro, defined in `linux/moduleparam.h`.

    Parameters:

    * `type`

        可以是下面几个之一：`byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `charp`, `bool`, `invbool`;

        Numerous types are supported for module parameters:

        * `bool`

            A boolean (true or false) value (the associated variable should be of type int).

        * `invbool`

            The invbool type inverts the value, so that true values become false and vice versa.

        * `charp`

            A char pointer value. Memory is allocated for user-provided strings, and the pointer is set accordingly.

        * `int`, `long`, `short`, `uint`, `ulong`, `ushort`
            
            Basic integer values of various lengths. The versions starting with `u` are for unsigned values.

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

**在命令行中传递模块参数**

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

**监测模块参数的改变**

`module_param_cb()`

This macro is used to register the callback. Whenever the argument (parameter) got changed, this callback function will be called.

为了注册 callback，我们首先需要填写一个结构体：

```c
struct kernel_param_ops 
{
    int (*set)(const char *val, const struct kernel_param *kp);
    int (*get)(char *buffer, const struct kernel_param *kp);
    void (*free)(void *arg);
};
```

（不明白这里的`free`有什么用）

Example:

```c
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

Type sudo su. Then enter the password if it asks. Then do `echo 13 > /sys/module/hello_world_module/parameters/cb_valueETX`

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

内核中提供了相应的宏来实现模块的导出:

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

字符设备驱动的访问：

驱动是沟通硬件和上层应用的媒介，字符设备驱动通过字符设备文件来访问，Linux 中所有的设备文件存放在`/dev`中，在用户层访问设备文件和普通文件的方法是没有区别的。Linux 操作系统实际上是通过设备号来找到对应的字符设备驱动（怎么找？）。

一个设备文件需要实现和普通文件相同的方法：

`open, close, read, write, lseek, ioctl, mmap, stat`

### 设备号

**构造设备号**

设备号用 32 位的一个`dev_t`类型的变量来表示（无符号整型），高 12 位表示主设备号，后 20 位表示次设备号。

The `dev_t` type (defined in `<linux/types.h>`) is used to hold device numbers—both the major and minor parts. `dev_t` is a 32-bit quantity with 12 bits set aside for the major number and 20 for the minor number.

主设备号用来区分不同类型的设备，次设备号用于区分设备的实例。

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
dev_t m_dev_num = MKDEV(220,0);
```

**申请设备号**

设备号在内核中属于资源，需要向内核申请。有两种申请方式，一种是静态申请，一种是动态申请。

1. 静态申请（Statically allocating）

    首先选择一个内核中未被使用的主设备号（`cat /proc/devices`），比如 220。根据设备个数分配次设备号，一般从 0 开始。

    Syntax:

    ```c
    register_chrdev_region(dev_t from, unsigned count, const char *name);
    ```

    Header file: `<linux/fs.h>`

    Params:

    * `from`: 要申请的起始设备号

    * `count`: 设备数量

        `count` is the total number of contiguous device numbers you are requesting. Note that, if the count is large, the range you request could spill over to the next major number; but everything will still work properly as long as the number range you request is available.

    * `name`: 设备号在内核中对应的名称

    返回 0 表示成功，返回非 0 表示失败。

    Example:

    ```c
    dev_t dev = MKDEV(235, 0);
    register_chrdev_region(dev, 1, "my driver");
    ```

 2. 动态申请（Dynamically Allocating）

    通过`alloc_chrdev_region`向内核申请

    ```c
    int alloc_chrdev_region(dev_t *dev, unsigned baseminor, unsigned count, const char *name);
    ```

    header file: `<linux/fs.h>`

    params:

    * `dev`: 设备号的地址

    * `baseminor`: 起始次设备号

    * `count`: 设备号个数

**注销设备号**

不再使用设备号需要注销：

```c
unregister_chrdev_region(dev_t from, unsigned count);
```

header file: `<linux/fs.h>`

params:

* `from`: 要注销的起始设备号

* `count`: 设备号的个数

一般在卸载模块的时候释放设备号。The usual place to call `unregister_chrdev_region` would be in your module’s cleanup function (Exit Function).

动态申请得到的设备号，释放的方法和静态申请一致。

Example:

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>

dev_t dev = MKDEV(220, 0);

int hello_init(void)
{
    printk("load my module\n");

    // allocate a device number
    register_chrdev_region(dev, 1, "hlc_dev");
    printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
    return 0;
}

void hello_exit(void)
{
    unregister_chrdev_region(dev, 1);
    printk("unload my module\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

```c
// ...

dev_t dev = 0;
if ((alloc_chrdev_region(&dev, 0, 1, "Embetronicx_Dev")) < 0){
        printk(KERN_INFO "Cannot allocate major number for device 1\n");
        return -1;
}

// ...
```

**Difference between static and dynamic method**

A static method is only really useful if you know in advance which major number you want to start with. With the Static method, you are telling the kernel that, what device numbers you want (the start major/minor number and count) and it either gives them to you or not (depending on availability).

With the Dynamic method, you are telling the kernel that how many device numbers you need (the starting minor number and count) and it will find a starting major number for you, if one is available, of course.

Partially to avoid conflict with other device drivers, it’s considered preferable to use the Dynamic method function, which will dynamically allocate the device numbers for you.

The disadvantage of dynamic assignment is that you can’t create the device nodes in advance, because the major number assigned to your module will vary. For normal use of the driver, this is hardly a problem, because once the number has been assigned, you can read it from /proc/devices.

### cdev 设备驱动

cdev 在内核中代表一个字符设备驱动。

```c
struct cdev {
    struct kobject kobj;
    struct module *owner;
    const struct file_operations *ops;  // 驱动操作函数集合
    struct list_head list;
    dev_t dev;  // 设备号
    unsigned int count;
};
```

**向内核中添加一个 cdev**

two ways of allocating and initializing one of these structures:

1. Runtime allocation

    ```c
    struct cdev *my_cdev = cdev_alloc( );
    my_cdev->ops = &my_fops;
    ``````

1. Own allocation

    Syntax:

    ```c
    void cdev_init(struct cdev *cdev, struct file_operations *fops);
    ```

    （`cdev`事先声明，`fops`也要事先写好）

    初始化 cdev（为 cdev 提供操作函数集合）

`cdev_add`：将 cdev 添加到内核（还会为 cdev 绑定设备号）

Syntax:

```c
int cdev_add(struct cdev *p, dev_t dev, unsigned count);
```

params:

* `p`: 要添加的 cdev 结构

* `dev`：起始设备号

* `count`：设备号个数

返回 0 表示成功，非 0 表示失败。

`cdev_del`：将 cdev 从内核中移除

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

每个静态文件都对应内核中一个`inode` struct，存放一些基本信息。而当文件被打开时，内核会创建一个`file` struct，记录一些信息。

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
    printk("Insert my test module.\n");

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
    printk("Exit my test module.\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

对其编译后，加载模块：`sudo insmod hello_world.ko`。

设备驱动需要和设备文件配合使用。

`__user`表示是一个用户空间的指针，所以 kernel 不可以直接使用。

```c
#ifdef __CHECKER__
# define __user __attribute__((noderef, address_space(1)))
# define __kernel /* default address space */
#else
# define __user
# define __kernel
#endif
```

### 设备文件

**Manually Creating Device File**

We can create the device file manually by using `mknod`.

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

* `device_create`

    创建设备文件（设备节点）

    Syntax:

    ```c
    struct device *device_create(struct *class, struct device *parent, dev_t dev, void * drvdata, const char *fmt, ...);
    ```

    Parameters:

    * `class` – pointer to the struct class that this device should be registered to

    * `parent` – pointer to the parent struct device of this new device, if any

        父设备指针

    * `devt` – the dev_t for the char device to be added

        设备号

    * `drvdata` – the data to be added to the device for callbacks

        额外的数据

    * `fmt` – string for the device’s name

        设备文件名

    * `...` – variable arguments

    A “dev” file will be created, showing the `dev_t` for the device, if the `dev_t` is not `0,0`. If a pointer to a parent struct device is passed in, the newly created struct device will be a child of that device in sysfs. The pointer to the struct device will be returned from the call. Any further sysfs files that might be required can be created using this pointer. The return value can be checked using IS_ERR() macro.

    成功会在`/dev`目录下生成对应的设备文件，并返回设备指针

* `device_destroy`

    销毁设备文件

    Syntax:

    ```c
    void device_destroy(struct class *class, dev_t devt)
    ```

内核中指针了一些宏，用于判断指针是否出错。

```c
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
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0)
        {
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
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))) {
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

当打开一个设备文件时，kernel 会根据设备号遍历 cdev 数组，找到对应的 cdev 结构体对象，然后把里面的`file_operations`里面的函数指针赋值给文件结构体`struct file`的`file_operations`里对应的函数。

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

header file: `<linux/fs.h>`

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

## Procfs

`/proc` is a mount point for the procfs (Process Filesystem) which is a filesystem in memory. Many processes store information about themselves on this virtual filesystem. ProcFS also stores other system information.

Check memory info: `cat /proc/meminfo`

Check modules info: `cat /proc/modules`

* `/proc/devices` — registered character and block major numbers
* `/proc/iomem` — on-system physical RAM and bus device addresses
* `/proc/ioports` — on-system I/O port addresses (especially for x86 systems)
* `/proc/interrupts` — registered interrupt request numbers
* `/proc/softirqs` — registered soft IRQs
* `/proc/swaps` — currently active swaps
* `/proc/kallsyms` — running kernel symbols, including from loaded modules
* `/proc/partitions` — currently connected block devices and their partitions
* `/proc/filesystems` — currently active filesystem drivers
* `/proc/cpuinfo` — information about the CPU(s) on the system

Most proc files are read-only and only expose kernel information to user space programs.

proc files can also be used to control and modify kernel behavior on the fly. The proc files need to be writable in this case.

enable IP forwarding of iptable: `echo 1 > /proc/sys/net/ipv4/ip_forward`

The proc file system can also be used to debug a kernel module. Just create entries for every variable that we want to track.

Creating procfs directory:

```c
struct proc_dir_entry *proc_mkdir(const char *name, struct proc_dir_entry *parent)
```

`name`: The name of the directory that will be created under `/proc`.

`parent`: In case the folder needs to be created in a subfolder under `/proc` a pointer to the same is passed else it can be left as NULL.

create proc entries:

header file: `linux/proc_fs.h`

```c
struct proc_dir_entry *proc_create ( const char *name, umode_t mode, struct proc_dir_entry *parent, const struct file_operations *proc_fops )
```

* `name`: The name of the proc entry
* `mode`: The access mode for proc entry
* `parent`: The name of the parent directory under /proc. If NULL is passed as a parent, the /proc directory will be set as a parent.
* `proc_fops`: The structure in which the file operations for the proc entry will be created.

Note: The above proc_create is valid in the Linux Kernel v3.10 to v5.5. From v5.6, there is a change in this API. The fourth argument const struct file_operations *proc_fops is changed to const struct proc_ops *proc_ops.

Example:

```c
proc_create("etx_proc",0666,NULL,&proc_fops);
```

create `file_operations` structure `proc_fops` in which we can map the read and write functions for the proc entry:

```c
static struct file_operations proc_fops = {
    .open = open_proc,
    .read = read_proc,
    .write = write_proc,
    .release = release_proc
};
```

For linux kernel v5.6 and above, use this:

```c
static struct proc_ops proc_fops = {
        .proc_open = open_proc,
        .proc_read = read_proc,
        .proc_write = write_proc,
        .proc_release = release_proc
};
```

`open` and `release` functions are optional:

```c
static int open_proc(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "proc file opend.....\t");
    return 0;
}

static int release_proc(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "proc file released.....\n");
    return 0;
}
```

The write function will receive data from the user space using the function copy_from_user into an array “etx_array”:

```c
static ssize_t write_proc(struct file *filp, const char *buff, size_t len, loff_t * off)
{
    printk(KERN_INFO "proc file write.....\t");
    copy_from_user(etx_array,buff,len);
    return len;
}
```

Once data is written to the proc entry we can read from the proc entry using a read function, i.e transfer data to the user space using the function `copy_to_user` function:

```c
static ssize_t read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset)
{
    printk(KERN_INFO "proc file read.....\n");
    if(len)
        len=0;
    else{
        len=1;
        return 0;
    }
    copy_to_user(buffer,etx_array,20);

    return length;;
}
```

Proc entry should be removed in the Driver exit function using the below function:

```c
void remove_proc_entry(const char *name, struct proc_dir_entry *parent);
```

Example:

```c
remove_proc_entry("etx_proc",NULL);
```

And you can remove the complete parent directory using `proc_remove(struct proc_dir_entry *parent)`.

Complete Driver Source Code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (procfs)
*
*  \author     EmbeTronicX
* 
*  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
*
* *******************************************************************************/
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
#include<linux/proc_fs.h>
#include <linux/err.h>

/* 
** I am using the kernel 5.10.27-v7l. So I have set this as 510.
** If you are using the kernel 3.10, then set this as 310,
** and for kernel 5.1, set this as 501. Because the API proc_create()
** changed in kernel above v5.5.
**
*/ 
#define LINUX_KERNEL_VERSION  510
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
char etx_array[20]="try_proc_array\n";
static int len = 1;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
static struct proc_dir_entry *parent;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);

/*************** Driver Functions **********************/
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long     etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
 
/***************** Procfs Functions *******************/
static int      open_proc(struct inode *inode, struct file *file);
static int      release_proc(struct inode *inode, struct file *file);
static ssize_t  read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset);
static ssize_t  write_proc(struct file *filp, const char *buff, size_t len, loff_t * off);

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


#if ( LINUX_KERNEL_VERSION > 505 )

/*
** procfs operation sturcture
*/
static struct proc_ops proc_fops = {
        .proc_open = open_proc,
        .proc_read = read_proc,
        .proc_write = write_proc,
        .proc_release = release_proc
};

#else //LINUX_KERNEL_VERSION > 505

/*
** procfs operation sturcture
*/
static struct file_operations proc_fops = {
        .open = open_proc,
        .read = read_proc,
        .write = write_proc,
        .release = release_proc
};

#endif //LINUX_KERNEL_VERSION > 505

/*
** This function will be called when we open the procfs file
*/
static int open_proc(struct inode *inode, struct file *file)
{
    pr_info("proc file opend.....\t");
    return 0;
}

/*
** This function will be called when we close the procfs file
*/
static int release_proc(struct inode *inode, struct file *file)
{
    pr_info("proc file released.....\n");
    return 0;
}

/*
** This function will be called when we read the procfs file
*/
static ssize_t read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset)
{
    pr_info("proc file read.....\n");
    if(len)
    {
        len=0;
    }
    else
    {
        len=1;
        return 0;
    }
    
    if( copy_to_user(buffer,etx_array,20) )
    {
        pr_err("Data Send : Err!\n");
    }
 
    return length;;
}

/*
** This function will be called when we write the procfs file
*/
static ssize_t write_proc(struct file *filp, const char *buff, size_t len, loff_t * off)
{
    pr_info("proc file wrote.....\n");
    
    if( copy_from_user(etx_array,buff,len) )
    {
        pr_err("Data Write : Err!\n");
    }
    
    return len;
}

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
        pr_info("Read function\n");
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write Function\n");
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
        
        /*Create proc directory. It will create a directory under "/proc" */
        parent = proc_mkdir("etx",NULL);
        
        if( parent == NULL )
        {
            pr_info("Error creating proc entry");
            goto r_device;
        }
        
        /*Creating Proc entry under "/proc/etx/" */
        proc_create("etx_proc", 0666, parent, &proc_fops);
 
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
        /* Removes single proc entry */
        //remove_proc_entry("etx/etx_proc", parent);
        
        /* remove complete /proc/etx */
        proc_remove(parent);
        
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
MODULE_DESCRIPTION("Simple Linux device driver (procfs)");
MODULE_VERSION("1.6");
```

As there are changes in the procfs file system in the Linux kernel 3.10 and 5.6, we have added a macro called `LINUX_KERNEL_VERSION`. You have to mention your Linux kernel version. Based on that, we will control the APIs in this source code.

Note:

You can follow this format for this `LINUX_KERNEL_VERSION`.

Example:

|Your Linux Kernel version | LINUX_KERNEL_VERSION |
| - | - |
| v3.10 | 310 |
| v5.6 | 506 |
| v5.10 | 510 |

test:

`cat /proc/etx/etx_proc`

`echo "device driver proc" > /proc/etx/etx_proc`

## waitqueue

Whenever a process must wait for an event (such as the arrival of data or the termination of a process), it should go to sleep. Sleeping causes the process to suspend execution, freeing the processor for other uses. After some time, the process will be woken up and will continue with its job when the event which we are waiting for has arrived.

Wait queue is a mechanism provided in the kernel to implement the wait. As the name itself suggests, waitqueue is the list of processes waiting for an event. In other words, A wait queue is used to wait for someone to wake you up when a certain condition is true. They must be used carefully to ensure there is no race condition.

There are 3 important steps in Waitqueue.

1. Initializing Waitqueue

    header file: `linux/wait.h`

    1. Static method

        ```c
        DECLARE_WAIT_QUEUE_HEAD(wq);
        ```

        Where the “wq” is the name of the queue on which task will be put to sleep.

    1. Dynamic method

        ```c
        wait_queue_head_t wq;
        init_waitqueue_head (&wq);
        ```

1. Queuing (Put the Task to sleep until the event comes)

    Once the wait queue is declared and initialized, a process may use it to go to sleep. There are several macros are available for different uses.

    * wait_event

        Syntax:

        ```c
        wait_event(wq, condition);
        ```

        sleep until a condition gets true.

        Parameters:

        * `wq` – the waitqueue to wait on

        * `condition` – a C expression for the event to wait for

        The process is put to sleep (`TASK_UNINTERRUPTIBLE`) until the condition evaluates to true. The `condition` is checked each time the waitqueue `wq` is woken up.

    * wait_event_timeout

        Syntax:

        ```c
        wait_event_timeout(wq, condition, timeout);
        ```

        sleep until a condition gets true or a timeout elapses

        Parameters:

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `timeout` –  timeout, in jiffies

        The process is put to sleep (TASK_UNINTERRUPTIBLE) until the condition evaluates to true or timeout elapses. The condition is checked each time the waitqueue wq is woken up.

        It returns 0 if the condition evaluated to false after the timeout elapsed, 1 if the condition evaluated to true after the timeout elapsed, or the remaining jiffies (at least 1) if the condition evaluated to true before the timeout elapsed.

    * wait_event_cmd

        Syntax:

        ```c
        wait_event_cmd(wq, condition, cmd1, cmd2);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `cmd1` – the command will be executed before sleep

        * `cmd2` – the command will be executed after sleep

        The process is put to sleep (TASK_UNINTERRUPTIBLE) until the condition evaluates to true. The condition is checked each time the waitqueue wq is woken up.

    * wait_event_interruptible

        Syntax:

        ```c
        wait_event_interruptible(wq, condition);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        The process is put to sleep (TASK_INTERRUPTIBLE) until the condition evaluates to true or a signal is received. The condition is checked each time the waitqueue wq is woken up.

        The function will return -ERESTARTSYS if it was interrupted by a signal and 0 if condition evaluated to true.

    * wait_event_interruptible_timeout

        Syntax:

        ```c
        wait_event_interruptible_timeout(wq, condition, timeout);
        ```

        sleep until a condition gets true or a timeout elapses

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `timeout` –  timeout, in jiffies

        The process is put to sleep (TASK_INTERRUPTIBLE) until the condition evaluates to true or a signal is received or timeout elapsed. The condition is checked each time the waitqueue wq is woken up.

        It returns, 0 if the condition evaluated to false after the timeout elapsed, 1 if the condition evaluated to true after the timeout elapsed, the remaining jiffies (at least 1) if the condition evaluated to true before the timeout elapsed, or -ERESTARTSYS if it was interrupted by a signal.

    * wait_event_killable

        Syntax:

        ```c
        wait_event_killable(wq, condition);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        The process is put to sleep (TASK_KILLABLE) until the condition evaluates to true or a signal is received. The condition is checked each time the waitqueue wq is woken up.

        The function will return -ERESTARTSYS if it was interrupted by a signal and 0 if condition evaluated to true.

    Whenever we use the above one of the macro, it will add that task to the waitqueue which is created by us. Then it will wait for the event.

    Note: Old kernel versions used the functions `sleep_on()` and `interruptible_sleep_on()`, but those two functions can introduce bad race conditions and should not be used.

1. Waking Up Queued Task

    When some Tasks are in sleep mode because of the waitqueue, then we can use the below function to wake up those tasks.

    * wake_up

        Syntax:

        ```c
        wake_up(&wq);
        ```

        wakes up only one process from the wait queue which is in non-interruptible sleep.

        Parameters:

        * `wq` – the waitqueue to wake up

    * wake_up_all

        Syntax:

        ```c
        wake_up_all(&wq);
        ```

        wakes up all the processes on the wait queue

    * wake_up_interruptible

        Syntax:

        ```c
        wake_up_interruptible(&wq);
        ```

        wakes up only one process from the wait queue that is in interruptible sleep

    * wake_up_sync and wake_up_interruptible_sync

        Syntax:

        ```c
        wake_up_sync(&wq);
        wake_up_interruptible_sync(&wq);
        ```

        Normally, a `wake_up` call can cause an immediate reschedule to happen, meaning that other processes might run before `wake_up` returns. The “synchronous” variants instead make any awakened processes runnable but do not reschedule the CPU. This is used to avoid rescheduling when the current process is known to be going to sleep, thus forcing a reschedule anyway. Note that awakened processes could run immediately on a different processor, so these functions should not be expected to provide mutual exclusion.

driver code:

* Waitqueue created by Static Method

    ```c
    /***************************************************************************//**
    *  \file       driver.c
    *
    *  \details    Simple linux driver (Waitqueue Static method)
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
    #include <linux/slab.h>                 //kmalloc()
    #include <linux/uaccess.h>              //copy_to/from_user()
    #include <linux/kthread.h>
    #include <linux/wait.h>                 // Required for the wait queues
    #include <linux/err.h>
    
    
    uint32_t read_count = 0;
    static struct task_struct *wait_thread;
    
    DECLARE_WAIT_QUEUE_HEAD(wait_queue_etx);
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    int wait_queue_flag = 0;

    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

    /*
    ** File operation sturcture
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
    ** Thread function
    */
    static int wait_function(void *unused)
    {
            
            while(1) {
                    pr_info("Waiting For Event...\n");
                    wait_event_interruptible(wait_queue_etx, wait_queue_flag != 0 );
                    if(wait_queue_flag == 2) {
                            pr_info("Event Came From Exit Function\n");
                            return 0;
                    }
                    pr_info("Event Came From Read Function - %d\n", ++read_count);
                    wait_queue_flag = 0;
            }
            do_exit(0);
            return 0;
    }

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
            wait_queue_flag = 1;
            wake_up_interruptible(&wait_queue_etx);
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
            etx_cdev.owner = THIS_MODULE;
            etx_cdev.ops = &fops;
    
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
    
            //Create the kernel thread with name 'mythread'
            wait_thread = kthread_create(wait_function, NULL, "WaitThread");
            if (wait_thread) {
                    pr_info("Thread Created successfully\n");
                    wake_up_process(wait_thread);
            } else
                    pr_info("Thread creation failed\n");
    
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
            wait_queue_flag = 2;
            wake_up_interruptible(&wait_queue_etx);
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
    MODULE_DESCRIPTION("Simple linux driver (Waitqueue Static method)");
    MODULE_VERSION("1.7");
    ```

* Waitqueue created by Dynamic Method

    ```c
    /****************************************************************************//**
    *  \file       driver.c
    *
    *  \details    Simple linux driver (Waitqueue Dynamic method)
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
    #include <linux/slab.h>                 //kmalloc()
    #include <linux/uaccess.h>              //copy_to/from_user()
    #include <linux/kthread.h>
    #include <linux/wait.h>                 // Required for the wait queues
    #include <linux/err.h>
    
    
    uint32_t read_count = 0;
    static struct task_struct *wait_thread;
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    wait_queue_head_t wait_queue_etx;
    int wait_queue_flag = 0;
    
    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

    /*
    ** File operation sturcture
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
    ** Thread function
    */
    static int wait_function(void *unused)
    {
            
            while(1) {
                    pr_info("Waiting For Event...\n");
                    wait_event_interruptible(wait_queue_etx, wait_queue_flag != 0 );
                    if(wait_queue_flag == 2) {
                            pr_info("Event Came From Exit Function\n");
                            return 0;
                    }
                    pr_info("Event Came From Read Function - %d\n", ++read_count);
                    wait_queue_flag = 0;
            }
            return 0;
    }
    
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
            wait_queue_flag = 1;
            wake_up_interruptible(&wait_queue_etx);
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
            
            //Initialize wait queue
            init_waitqueue_head(&wait_queue_etx);
    
            //Create the kernel thread with name 'mythread'
            wait_thread = kthread_create(wait_function, NULL, "WaitThread");
            if (wait_thread) {
                    pr_info("Thread Created successfully\n");
                    wake_up_process(wait_thread);
            } else
                    pr_info("Thread creation failed\n");
    
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
            wait_queue_flag = 2;
            wake_up_interruptible(&wait_queue_etx);
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
    MODULE_DESCRIPTION("Simple linux driver (Waitqueue Dynamic method)");
    MODULE_VERSION("1.8");
    ```

Makefile:

```Makefile
obj-m += driver.o
KDIR = /lib/modules/$(shell uname -r)/build
all:
    make -C $(KDIR)  M=$(shell pwd) modules
clean:
    make -C $(KDIR)  M=$(shell pwd) clean
```

（这个版本中的 Makefile 使用的并不是`$(PWD)`之类的，而是`$(shell pwd)`，这两者有什么不同？）

test: `sudo cat /dev/etx_device`

## sysfs

Sysfs is a virtual filesystem mounted on `/sys`. Sysfs contain information about devices and drivers.

**Kernel Objects**

The heart of the sysfs model is the kobject. Kobject is the glue that binds the sysfs and the kernel, which is represented by `struct kobject` and defined in `<linux/kobject.h>`. A struct kobject represents a kernel object, maybe a device or so, such as the things that show up as directory in the sysfs filesystem.

Kobjects are usually embedded in other structures.

Syntax:

```c
#define KOBJ_NAME_LEN 20 

struct kobject {
 char *k_name;
 char name[KOBJ_NAME_LEN];
 struct kref kref;
 struct list_head entry;
 struct kobject *parent;
 struct kset *kset;
 struct kobj_type *ktype;
 struct dentry *dentry;
};
```

Explanation:

* `struct kobject`

    * `name` (Name of the kobject. Current kobject is created with this name in sysfs.)

    * `parent` (This is kobject’s parent. When we create a directory in sysfs for the current kobject, it will create under this parent directory)

    * `ktype` (the type associated with a kobject)

    * `kset` (a group of kobjects all of which are embedded in structures of the same type)

    * `sd` (points to a sysfs_dirent structure that represents this kobject in sysfs.)

    * `kref` (provides reference counting)

    `kobject` is used to create kobject directory in /sys.

There are two steps to creating and using sysfs.

1. Create a directory in `/sys`

    We can use this function (`kobject_create_and_add`) to create a directory.

    `struct kobject * kobject_create_and_add ( const char * name, struct kobject * parent);`

    Where,

    * `name` – the name for the kobject

    * `parent` – the parent kobject of this kobject, if any.

        If you pass `kernel_kobj` to the second argument, it will create the directory under `/sys/kernel/`. If you pass `firmware_kobj` to the second argument, it will create the directory under `/sys/firmware/`. If you pass `fs_kobj` to the second argument, it will create the directory under `/sys/fs/`. If you pass NULL to the second argument, it will create the directory under `/sys/`.

    This function creates a kobject structure dynamically and registers it with sysfs. If the kobject was not able to be created, `NULL` will be returned.

    Call `kobject_put` and the structure `kobject` will be dynamically freed when it is no longer being used. (not clear. Does it mean free the memory immediately or wait for the last time that `struct object` was used?)

    Example:

    ```c
    struct kobject *kobj_ref;

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj); //sys/kernel/etx_sysfs

    /*Freeing Kobj*/
    kobject_put(kobj_ref);
    ```

1. Create Sysfs file

    sysfs file is used to interact user space with kernel space.

    We can create the sysfs file using sysfs attributes. Attributes are represented as regular files in sysfs with one value per file. There are loads of helper functions that can be used to create the kobject attributes. They can be found in the header file `sysfs.h`.

    * Create attribute

        Syntax:

        ```c
        struct kobj_attribute {
            struct attribute attr;
            ssize_t (*show)(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
            ssize_t (*store)(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
        };
        ```

        Parameters:

        * `attr` – the attribute representing the file to be created,

        * `show` – the pointer to the function that will be called when the file is read in sysfs,

        * `store` – the pointer to the function which will be called when the file is written in sysfs.

        We can create an attribute using `__ATTR` macro.

        `__ATTR(name, permission, show_ptr, store_ptr);`
        
    * Store and Show functions

        ```c
        ssize_t (*show)(struct kobject *kobj, struct kobj_attribute *attr, char *buf);

        ssize_t (*store)(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
        ```

        Store function will be called whenever we are writing something to the sysfs attribute.

        Show function will be called whenever we are reading the sysfs attribute.

    * Create sysfs file

        To create a single file attribute we are going to use ‘sysfs_create_file’.

        int sysfs_create_file ( struct kobject *  kobj, const struct attribute * attr);

        Where,

        `kobj` – object we’re creating for.

        `attr` – attribute descriptor.

        One can use another function `sysfs_create_group` to create a group of attributes.

        Once you have done with the sysfs file, you should delete this file using `sysfs_remove_file`。

        ```c
        void sysfs_remove_file ( struct kobject *  kobj, const struct attribute * attr);
        ```

        Where,

        `kobj` – object we’re creating for.

        `attr` – attribute descriptor.

    Example:

    ```c
    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
        return sprintf(buf, "%d", etx_value);
    }

    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            sscanf(buf,"%d",&etx_value);
            return count;
    }

    //This Function will be called from Init function
    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
        printk(KERN_INFO"Cannot create sysfs file......\n");
        goto r_sysfs;
    }
    //This should be called from exit function
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    ```

    driver:

    ```c
    /***************************************************************************//**
    *  \file       driver.c
    *
    *  \details    Simple Linux device driver (sysfs)
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
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/err.h>
    
    volatile int etx_value = 0;
    
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;

    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, 
                            char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, 
                            const char *buf, size_t len, loff_t * off);
    
    /*************** Sysfs functions **********************/
    static ssize_t  sysfs_show(struct kobject *kobj, 
                            struct kobj_attribute *attr, char *buf);
    static ssize_t  sysfs_store(struct kobject *kobj, 
                            struct kobj_attribute *attr,const char *buf, size_t count);

    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    /*
    ** File operation sturcture
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
    ** This function will be called when we read the sysfs file
    */
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
            pr_info("Sysfs - Read!!!\n");
            return sprintf(buf, "%d", etx_value);
    }

    /*
    ** This function will be called when we write the sysfsfs file
    */
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            pr_info("Sysfs - Write!!!\n");
            sscanf(buf,"%d",&etx_value);
            return count;
    }

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
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
            pr_info("Read function\n");
            return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
            pr_info("Write Function\n");
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
    
            /*Creating a directory in /sys/kernel/ */
            kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
            /*Creating sysfs file for etx_value*/
            if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                    pr_err("Cannot create sysfs file......\n");
                    goto r_sysfs;
        }
            pr_info("Device Driver Insert...Done!!!\n");
            return 0;
    
    r_sysfs:
            kobject_put(kobj_ref); 
            sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    
    r_device:
            class_destroy(dev_class);
    r_class:
            unregister_chrdev_region(dev,1);
            cdev_del(&etx_cdev);
            return -1;
    }

    /*
    ** Module exit function
    */
    static void __exit etx_driver_exit(void)
    {
            kobject_put(kobj_ref); 
            sysfs_remove_file(kernel_kobj, &etx_attr.attr);
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
    MODULE_DESCRIPTION("Simple Linux device driver (sysfs)");
    MODULE_VERSION("1.8");
    ```

    test: `ls -l /sys/kernel`, `ls -l /sys/kernel/etx_sysfs`

    read and modify the value:

    ```bash
    sudo su
    cat /sys/kernel/etx_sysfs/etx_value
    echo 123 > /sys/kernel/etx_sysfs/etx_value
    cat /sys/kernel/etx_sysfs/etx_value
    ```

## Interrupts

special functions called interrupt handlers (ISR)

In Linux, interrupt signals are the distraction that diverts the processor to a new activity outside of the normal flow of execution. This new activity is called interrupt handler or interrupt service routine (ISR) and that distraction is Interrupts.

**Polling vs Interrupts**

* Polling

    In polling the CPU keeps on checking all the hardwares of the availablilty of any request

    The polling method is like a salesperson. The salesman goes from door to door while requesting to buy a product or service. Similarly, the controller keeps monitoring the flags or signals one by one for all devices and provides service to whichever component that needs its service.

* Interrupt

    In interrupt the CPU takes care of the hardware only when the hardware requests for some service

    An interrupt is like a shopkeeper. If one needs a service or product, he goes to him and apprises him of his needs. In case of interrupts, when the flags or signals are received, they notify the controller that they need to be serviced.

**Interrupts and Exceptions**

Exceptions are often discussed at the same time as interrupts. Unlike interrupts, exceptions occur synchronously with respect to the processor clock; they are often called synchronous interrupts. Exceptions are produced by the processor while executing instructions either in response to a programming error (e.g. divide by zero) or abnormal conditions that must be handled by the kernel (e.g. a page fault).

Interrupts – asynchronous interrupts generated by hardware.

Exceptions – synchronous interrupts generated by the processor.

（这里的同步和异步指的是时序，如果在非时钟周期内到达了中断信号，那么就称其为异步。）

**Maskable and Non-maskable**

Maskable – All Interrupt Requests (IRQs) issued by I/O devices give rise to maskable interrupts. A maskable interrupt can be in two states: masked or unmasked; a masked interrupt is ignored by the control unit as long as it remains masked.

Non-maskable – Only a few critical events (such as hardware failures) give rise to nonmaskable interrupts. Non-maskable interrupts are always recognized by the CPU.

**Exception types**

* Falts – Like Divide by zero, Page Fault, Segmentation Fault.

* Traps – Reported immediately following the execution of the trapping instruction. Like Breakpoints.

* Aborts – Aborts are used to report severe errors, such as hardware failures and invalid or inconsistent values in system tables.

**Interrupt handler**

For a device’s each interrupt, its device driver must register an interrupt handler.

An interrupt handler or interrupt service routine (ISR) is the function that the kernel runs in response to a specific interrupt:

1. Each device that generates interrupts has an associated interrupt handler.

1. The interrupt handler for a device is part of the device’s driver (the kernel code that manages the device).


In Linux, interrupt handlers are normal C functions, which match a specific prototype and thus enable the kernel to pass the handler information in a standard way. What differentiates interrupt handlers from other kernel functions is that the kernel invokes them in response to interrupts and that they run in a special context called interrupt context. This special context is occasionally called atomic context because code executing in this context is unable to block.

Because an interrupt can occur at any time, an interrupt handler can be executed at any time. It is imperative that the handler runs quickly, to resume the execution of the interrupted code as soon as possible. It is important that

1. To the hardware: the operating system services the interrupt without delay.
1. To the rest of the system: the interrupt handler executes in as short a period as possible.

Top halves and Bottom halves
Top half
The interrupt handler is the top half. The top half will run immediately upon receipt of the interrupt and performs only the work that is time-critical, such as acknowledging receipt of the interrupt or resetting the hardware.

Bottom half
The bottom half is used to process data, letting the top half to deal with new incoming interrupts. Interrupts are enabled when a bottom half runs. The interrupt can be disabled if necessary, but generally, this should be avoided as this goes against the basic purpose of having a bottom half – processing data while listening to new interrupts. The bottom half runs in the future, at a more convenient time, with all interrupts enabled.

比如网卡接收数据，我们使用 top half 快速地把网络数据包从网卡的缓冲区复制到内存中，然后使用 bottom half 慢慢处理内存中的数据包就可以了。如果 top half 不够快，那么新来的数据就会覆盖掉旧数据，造成读写错误。

Intel processors handle interrupt using IDT (Interrupt Descriptor Table).  The IDT consists of 256 entries with each entry corresponding to a vector and of 8 bytes. All the entries are a pointer to the interrupt handling function. The CPU uses IDTR to point to IDT. The relation between those two can be depicted as below,

Example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Interrupt Example
*
*  \author     EmbeTronicX
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
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/err.h>
#define IRQ_NO 11

//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
  printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
  return IRQ_HANDLED;
}

volatile int etx_value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);

/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);

/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);

struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};
 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}
 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return len;
}
 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
                    goto irq;
        }
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;

irq:
        free_irq(IRQ_NO,(void *)(irq_handler));

r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}
 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Interrupts");
MODULE_VERSION("1.9");
```

期望输出：

```
linux@embetronicx-VirtualBox: dmesg

[19743.366386] Major = 246 Minor = 0
[19743.370707] Device Driver Insert...Done!!!
[19745.580487] Device File Opened...!!!
[19745.580507] Read function
[19745.580531] Shared IRQ: Interrupt Occurred
[19745.580540] Device File Closed...!!!
```

实际输出：

```
[162342.126355] Major = 238 Minor = 0 
[162342.138918] Device Driver Insert...Done!!!
[162359.827734] Device File Opened...!!!
[162359.827746] Read function
[162359.827955] __common_interrupt: 2.59 No irq handler for vector
[162359.827974] Device File Closed...!!!
```

If you are using the newer Linux kernel, then this may not work properly. You may get something like below.

`do_IRQ: 1.59 No irq handler for vector`

In order to solve that, you have to change the Linux kernel source code, Compile it, then install it.

build:

```bash
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.47.tar.xz
sudo tar -xvf ../linux-5.4.47.tar
cd linux-5.4.47/
cp -v /boot/config-$(uname -r) .confi
sudo apt install build-essential kernel-package fakeroot libncurses5-dev libssl-dev ccache flex libelf-dev bison libncurses-dev
```

Add the below line in the downloaded Linux kernel file `arch/x86/kernel/irq.c` right after all the include lines.

`EXPORT_SYMBOL(vector_irq);`

```bash
make oldconfig
make menuconfig
sudo make  （也可以并行编译：sudo make -j 4）
sudo su
make modules_install
sudo make install
sudo update-initramfs -c -k 5.4.47
sudo update-grub
reboot
uname -r
```

新版本的 kernel 应该使用的 driver 代码：

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Interrupt Example
*
*  \author     EmbeTronicX
*
*  \Tested with kernel 5.4.47
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
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <asm/hw_irq.h>
#include <linux/err.h>
#define IRQ_NO 11
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
  printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
  return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};
 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}
 
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}
 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}
 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        struct irq_desc *desc;

        printk(KERN_INFO "Read function\n");
        desc = irq_to_desc(11);
        if (!desc) 
        {
            return -EINVAL;
        }
        __this_cpu_write(vector_irq[59], desc);
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return len;
}
 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
                    goto irq;
        }
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}
 
static void __exit etx_driver_exit(void)
{
    free_irq(IRQ_NO,(void *)(irq_handler));
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Interrupts");
MODULE_VERSION("1.9");
```

### Work queue

Work queues defer work into a kernel thread; this bottom half always runs in the process context.

Workqueue 将工作推迟到一个内核线程中，这个底层部分总是运行在进程的上下文环境中。

Workqueue is allowing users to create a kernel thread and bind work to the kernel thread.

So, this will run in the process context and the work queue can sleep.

（什么是 process context ? 为什么 kernel thread 会运行在 process context 中？）

Normally, it is easy to decide between using workqueue and softirq/tasklet:

If the deferred work needs to sleep, then workqueue is used.
If the deferred work need not sleep, then softirq or tasklet are used.

注：

* 什么是 softirq？

There are two ways to implement Workqueue in the Linux kernel.

Using global workqueue (Static / Dynamic)
Creating Own workqueue (We will see in the next tutorial)

**Initialize work using Static Method**

The below call creates a workqueue by the name and the function that we are passing in the second argument gets scheduled in the queue.

`DECLARE_WORK(name, void (*func)(void *))`

Where,

* `name`: The name of the “work_struct” structure that has to be created.

* `func`: The function to be scheduled in this workqueue.

Example:

```c
DECLARE_WORK(workqueue,workqueue_fn);
```

Schedule work to the Workqueue
The below functions are used to allocate the work to the queue.

Schedule_work
This function puts a job in the kernel-global workqueue if it was not already queued and leaves it in the same position on the kernel-global workqueue otherwise.

int schedule_work( struct work_struct *work );

where,

work – job to be done

Returns zero if work was already on the kernel-global workqueue and non-zero otherwise.

Scheduled_delayed_work
After waiting for a given time this function puts a job in the kernel-global workqueue.

int scheduled_delayed_work( struct delayed_work *dwork, unsigned long delay );

where,

dwork – job to be done

delay– number of jiffies to wait or 0 for immediate execution

Schedule_work_on
This puts a job on a specific CPU.

int schedule_work_on( int cpu, struct work_struct *work );

where,

cpu– CPU to put the work task on

work– job to be done

Scheduled_delayed_work_on
After waiting for a given time this puts a job in the kernel-global workqueue on the specified CPU.

int scheduled_delayed_work_on(int cpu, struct delayed_work *dwork, unsigned long delay );
where,

cpu – CPU to put the work task on

dwork – job to be done

delay– number of jiffies to wait or 0 for immediate execution

Delete work from workqueue
There are also a number of helper functions that you can use to flush or cancel work on work queues. To flush a particular work item and block until the work is complete, you can make a call to flush_work. All work on a given work queue can be completed using a call to flush_work. In both cases, the caller blocks until the operation are complete. To flush the kernel-global work queue, call flush_scheduled_work.

int flush_work( struct work_struct *work );
void flush_scheduled_work( void );
Cancel Work from workqueue
You can cancel work if it is not already executing in a handler. A call to cancel_work_sync will terminate the work in the queue or block until the callback has finished (if the work is already in progress in the handler). If the work is delayed, you can use a call to cancel_delayed_work_sync.

int cancel_work_sync( struct work_struct *work );
int cancel_delayed_work_sync( struct delayed_work *dwork );
Check the workqueue
Finally, you can find out whether a work item is pending (not yet executed by the handler) with a call to work_pending or delayed_work_pending.

`work_pending( work );`

`delayed_work_pending( work );`

Example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Global Workqueue - Static method)
*
*  \author     EmbeTronicX
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
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
 
void workqueue_fn(struct work_struct *work); 
 
/*Creating work by Static Method */
DECLARE_WORK(workqueue,workqueue_fn);
 
/*Workqueue Function*/
void workqueue_fn(struct work_struct *work)
{
        printk(KERN_INFO "Executing Workqueue Function\n");
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
        schedule_work(&workqueue);
        
        return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
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
** This function will be called when we read the sysfs file
*/ 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This function will be called when we write the sysfsfs file
*/
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This function will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return len;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
                    goto irq;
        }
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Static method)");
MODULE_VERSION("1.10");
```

看起来似乎是`irq_handler()`会调用`schedule_work()`，把繁重的工作分配到 bottom half 去做。推测：`schedule_work()`一定是非阻塞的。

test:

```bash
sudo cat /dev/etx_device
sudo dmesg
```

#### Initialize work using Dynamic Method

The below call (`INIT_WORK`) creates a workqueue in Linux by the name work and the function that gets scheduled in the queue is work_fn.

`INIT_WORK(work,work_fn)`

Where,

`name`: The name of the “work_struct” structure that has to be created.
`func`: The function to be scheduled in this workqueue.

**Schedule work to the Workqueue**
The below functions used to allocate the work to the queue.

`Schedule_work`

This function puts a job in the kernel-global workqueue if it was not already queued and leaves it in the same position on the kernel-global workqueue otherwise.

`int schedule_work( struct work_struct *work );`

where,

`work` – job to be done

Returns zero if work was already on the kernel-global workqueue and non-zero otherwise.

`Scheduled_delayed_work`

After waiting for a given time this function puts a job in the kernel-global workqueue.

`int scheduled_delayed_work( struct delayed_work *dwork, unsigned long delay );`

where,

`dwork` – job to be done

`delay` – number of jiffies to wait or 0 for immediate execution

`Schedule_work_on`

This puts a job on a specific CPU.

`int schedule_work_on( int cpu, struct work_struct *work );`

where,

`cpu` – CPU to put the work task on

`work` – job to be done

`Scheduled_delayed_work_on`

After waiting for a given time this puts a job in the kernel-global workqueue on the specified CPU.

`int scheduled_delayed_work_on(int cpu, struct delayed_work *dwork, unsigned long delay );`

where,

`cpu` – CPU to put the work task on

`dwork` – job to be done

`delay` – number of jiffies to wait or 0 for immediate execution

**Delete work from workqueue**

There are also a number of helper functions that you can use to flush or cancel work on work queues. To flush a particular work item and block until the work is complete, you can make a call to flush_work. All work on a given work queue can be completed using a call to . In both cases, the caller blocks until the operation are complete. To flush the kernel-global work queue, call flush_scheduled_work.

`int flush_work( struct work_struct *work );`

`void flush_scheduled_work( void );`

**Cancel Work from workqueue**

You can cancel work if it is not already executing in a handler. A call to cancel_work_sync will terminate the work in the queue or block until the callback has finished (if the work is already in progress in the handler). If the work is delayed, you can use a call to cancel_delayed_work_sync.

`int cancel_work_sync( struct work_struct *work );`

`int cancel_delayed_work_sync( struct delayed_work *dwork );`

**Check workqueue**

Finally, you can find out whether a work item is pending (not yet executed by the handler) with a call to work_pending or delayed_work_pending.

`work_pending( work );`
`delayed_work_pending( work );`

Example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Global Workqueue - Dynamic method)
*
*  \author     EmbeTronicX
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
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
/* Work structure */
static struct work_struct workqueue;
 
void workqueue_fn(struct work_struct *work); 
 
/*Workqueue Function*/
void workqueue_fn(struct work_struct *work)
{
        printk(KERN_INFO "Executing Workqueue Function\n");
}
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
        /*Allocating work to queue*/
        schedule_work(&workqueue);
        
        return IRQ_HANDLED;
}
 
volatile int etx_value = 0;
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
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
** This fuction will be called when we read the sysfs file
*/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}
 
/*
** This fuction will be called when we write the sysfsfs file
*/
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This fuction will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return 0;
}
 
/*
** Module Init function
*/ 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
                    goto irq;
        }
 
        /*Creating work by Dynamic Method */
        INIT_WORK(&workqueue,workqueue_fn);
 
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Dynamic method)");
MODULE_VERSION("1.11");
```

The core workqueue is represented by structure struct workqueue_struct, which is the structure onto which work is placed. This work is added to the queue in the top half (Interrupt context) and the execution of this work happened in the bottom half (Kernel context).

The work is represented by structure struct `work_struct`, which identifies the work and the deferral function.

**Create and destroy workqueue structure**

Workqueues are created through a macro called create_workqueue, which returns a workqueue_struct reference. You can remove this workqueue later (if needed) through a call to the destroy_workqueue function.

`struct workqueue_struct *create_workqueue( name );`

`void destroy_workqueue( struct workqueue_struct * );`

You should use create_singlethread_workqueue() for creating a workqueue when you want to create only a single thread for all the processors.

Since create_workqueue and create_singlethread_workqueue() are macros. Both are using the alloc_workqueue function in the background.

```c
#define create_workqueue(name)                    
        alloc_workqueue("%s", WQ_MEM_RECLAIM, 1, (name))
#define create_singlethread_workqueue(name)       
        alloc_workqueue("%s", WQ_UNBOUND | WQ_MEM_RECLAIM, 1, (name))
```

alloc_workqueue
Allocate a workqueue with the specified parameters.

alloc_workqueue ( fmt, flags, max_active );

fmt– printf format for the name of the workqueue

flags – WQ_* flags

max_active – max in-flight work items, 0 for default

This will return Pointer to the allocated workqueue on success, NULL on failure.

WQ_* flags
This is the second argument of alloc_workqueue.

WQ_UNBOUND

Work items queued to an unbound wq are served by the special worker-pools which host workers who are not bound to any specific CPU. This makes the wq behave like a simple execution context provider without concurrency management. The unbound worker-pools try to start the execution of work items as soon as possible. Unbound wq sacrifices locality but is useful for the following cases.

* Wide fluctuation in the concurrency level requirement is expected and using bound wq may end up creating a large number of mostly unused workers across different CPUs as the issuer hops through different CPUs.

* Long-running CPU-intensive workloads which can be better managed by the system scheduler.

WQ_FREEZABLE

A freezable wq participates in the freeze phase of the system suspend operations. Work items on the wq are drained and no new work item starts execution until thawed.

WQ_MEM_RECLAIM

All wq which might be used in the memory reclaim paths MUST have this flag set. The wq is guaranteed to have at least one execution context regardless of memory pressure.

WQ_HIGHPRI

Work items of a highpri wq are queued to the highpri worker-pool of the target CPU. Highpri worker-pools are served by worker threads with elevated nice levels.

Note that normal and highpri worker-pools don’t interact with each other. Each maintains its separate pool of workers and implements concurrency management among its workers.

WQ_CPU_INTENSIVE

Work items of a CPU intensive wq do not contribute to the concurrency level. In other words, runnable CPU-intensive work items will not prevent other work items in the same worker pool from starting execution. This is useful for bound work items that are expected to hog CPU cycles so that their execution is regulated by the system scheduler.

Although CPU-intensive work items don’t contribute to the concurrency level, the start of their executions is still regulated by the concurrency management and runnable non-CPU-intensive work items can delay the execution of CPU-intensive work items.

This flag is meaningless for unbound wq.

Queuing Work to workqueue
With the work structure initialized, the next step is enqueuing the work on a workqueue. You can do this in a few ways.

queue_work
This will queue the work to the CPU on which it was submitted, but if the CPU dies it can be processed by another CPU.
int queue_work( struct workqueue_struct *wq, struct work_struct *work );

Where,

wq – workqueue to use

work – work to queue

It returns false if work was already on a queue, true otherwise.

queue_work_on
This puts work on a specific CPU.
int queue_work_on( int cpu, struct workqueue_struct *wq, struct work_struct *work );

Where,

cpu– cpu to put the work task on

wq – workqueue to use

work– job to be done

`queue_delayed_work`
After waiting for a given time this function puts work in the workqueue.

```c
int queue_delayed_work( struct workqueue_struct *wq,struct delayed_work *dwork, unsigned long delay );
```

Where,

`wq` – workqueue to use

`dwork` – work to queue

`delay`– number of jiffies to wait before queueing or 0 for immediate execution

`queue_delayed_work_on`

After waiting for a given time this puts a job in the workqueue on the specified CPU.

```c
int queue_delayed_work_on( int cpu, struct workqueue_struct *wq,struct delayed_work *dwork, unsigned long delay );
```

Where,

`cpu` – CPU to put the work task on

`wq` – workqueue to use

`dwork` – work to queue

`delay` – number of jiffies to wait before queueing or 0 for immediate execution

Full code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Own Workqueue)
*
*  \author     EmbeTronicX
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
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
static struct workqueue_struct *own_workqueue;
 
static void workqueue_fn(struct work_struct *work); 
 
static DECLARE_WORK(work, workqueue_fn);
 
 
/*Workqueue Function*/
static void workqueue_fn(struct work_struct *work)
{
    printk(KERN_INFO "Executing Workqueue Function\n");
    return;
        
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred\n");
        /*Allocating work to queue*/
        queue_work(own_workqueue, &work);
        
        return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/ 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
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
** This fuction will be called when we read the sysfs file
*/  
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This fuction will be called when we write the sysfsfs file
*/ 
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This fuction will be called when we open the Device file
*/ 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return 0;
}
 
/*
** Module Init function
*/ 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ \n");
                    goto irq;
        }
 
        /*Creating workqueue */
        own_workqueue = create_workqueue("own_wq");
        
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        /* Delete workqueue */
        destroy_workqueue(own_workqueue);
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Own Workqueue)");
MODULE_VERSION("1.12");
```

test:

```bash
sudo cat /dev/etx_device
dmesg
```

Use `ps -aef` command to see our workqueue. You can able to see our workqueue which is `own_wq`.

**Difference between Schedule_work and queue_work**

If you want to use your own dedicated workqueue you should create a workqueue using create_workqueue. At that time you need to put work on your workqueue by using queue_work function.

If you don’t want to create any own workqueue, you can use kernel global workqueue. In that condition, you can use schedule_work function to put your work to global workqueue.

## Linked list

Linux implement a Doubly Linked List, which is defined in `/lib/modules/$(uname -r)/build/include/linux/list.h`.

```c
#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD(name) \
    struct list_head name = LIST_HEAD_INIT(name)
struct list_head {
    struct list_head *next;
    struct list_head *prev;
};
```

Usage:

```c
#include <linux/list.h>

struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};
```

Create a linked list head:

```c
LIST_HEAD(linked_list);
```

This macro will create the head node structure in the name of “linked_list” and it will initialize that to its own address.

Example:

```c
struct list_head etx_linked_list = { &etx_linked_list , &etx_linked_list};
```

While creating the head node, it initializes the prev and next pointer to its own address. This means that prev and next pointer points to itself. The node is empty If the node’s prev and next pointer points to itself.

（不是很懂，创建了 head note 后，不应该已经有一个 node 了吗，为什么还说他是空的？猜测这里的 head 是一个 dummy head）

**Create Node in Linked List**

```c
INIT_LIST_HEAD(struct list_head *list);
```

可以用上面的宏进行初始化链表。

Example:

```c
struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};

struct my_list new_node;

INIT_LIST_HEAD(&new_node.list);
new_node.data = 10;
```

**Add Node to Linked List**

* Add after Head Node

    Insert a new entry after the specified head:

    `inline void list_add(struct list_head *new, struct list_head *head);`

    Where,

    `struct list_head * new` – the new entry to be added

    `struct list_head * head` – list head to add it after

    Example:

    ```c
    list_add(&new_node.list, &etx_linked_list);
    ```

* Add before Head Node

    Insert a new entry before the specified head：

    `inline void list_add_tail(struct list_head *new, struct list_head *head);`

    Where,

    `struct list_head * new` – a new entry to be added

    `struct list_head * head` – list head to add before the head

    Example:

    ```c
    list_add_tail(&new_node.list, &etx_linked_list);
    ```

* Delete Node from Linked List

    * `list_del`

        It will delete the entry node from the list. This function removes the entry node from the linked list by disconnecting prev and next pointers from the list, but it doesn’t free any memory space allocated for the entry node.

        `inline void list_del(struct list_head *entry);`

        Where,

        `struct list_head * entry` – the element to delete from the list.

    * `list_del_init`

        It will delete the entry node from the list and reinitialize it. This function removes the entry node from the linked list by disconnecting prev and next pointers from the list, but it doesn’t free any memory space allocated for the entry node.

        `inline void list_del_init(struct list_head *entry);`

        Where,

        `struct list_head * entry` – the element to delete from the list.

* Replace Node in Linked List

    * `list_replace`

        This function is used to replace the old node with the new node.

        `inline void list_replace(struct list_head *old, struct list_head *new);`

        Where,

        `struct list_head * old` – the element to be replaced

        `struct list_head * new` – the new element to insert

        If old was empty, it will be overwritten.

    * `list_replace_init`

        This function is used to replace the old node with the new node and reinitialize the old entry.

        `inline void list_replace_init(struct list_head *old, struct list_head *new);`

        Where,

       `struct list_head * old` – the element to be replaced

       `struct list_head * new` – the new element to insert

        If old was empty, it will be overwritten.

* Moving Node in Linked List

    * `list_move`

        This will delete one list from the linked list and again adds to it after the head node.

        inline void list_move(struct list_head *list, struct list_head *head);

        Where,

        `struct list_head * list` – the entry to move

        `struct list_head * head` – the head that will precede our entry

    * `list_move_tail`

        This will delete one list from the linked list and again adds it before the head node.

        inline void list_move_tail(struct list_head *list, struct list_head *head);

        Where,

        `struct list_head * list` – the entry to move

        `struct list_head * head` – the head that will precede our entry

* Rotate Node in Linked List

    This will rotate the list to the left.

    `inline void list_rotate_left(struct list_head *head);`

    Where,

    head – the head of the list

* Test the Linked List Entry

    * `list_is_last`

        This tests whether list is the last entry in the list head.

        inline int list_is_last(const struct list_head *list, const struct list_head *head);

            Where,

        const struct list_head * list – the entry to test

        const struct list_head * head – the head of the list

        It returns 1 if it is the last entry otherwise 0.

    * `list_empty`

        It tests whether a list is empty or not.

        `inline int list_empty(const struct list_head *head);`

        Where,

        `const struct list_head * head` – the head of the list

        It returns 1 if it is empty otherwise 0.

    * `list_is_singular`

        This will test whether a list has just one entry.

        `inline int list_is_singular(const struct list_head *head);`

        Where,

        const struct list_head * head – the head of the list

        It returns 1 if it has only one entry otherwise 0.

* Split Linked List into two parts

    This cut a list into two.
    This helper moves the initial part of head, up to and including entry, from head to list. You should pass on entry an element you know is on head. list should be an empty list or a list you do not care about losing its data.

    inline void list_cut_position(struct list_head *list, struct list_head *head, struct list_head *entry);

    Where,

    struct list_head * list – a new list to add all removed entries

    struct list_head * head– a list with entries

    struct list_head * entry– an entry within the head could be the head itself and if so we won’t cut the list

* Join Two Linked Lists

    This will join two lists, this is designed for stacks.
    inline void list_splice(const struct list_head *list, struct list_head *head);

    Where,

    const struct list_head * list – the new list to add.

    struct list_head * head – the place to add it in the first list.

* Traverse Linked List

    * `list_entry`

        This macro is used to get the struct for this entry.
        
        list_entry(ptr, type, member);

        ptr– the struct list_head pointer.

        type – the type of the struct this is embedded in.

        member – the name of the list_head within the struct.

    * `list_for_each`

        This macro is used to iterate over a list.
        
        list_for_each(pos, head);

        pos –  the &struct list_head to use as a loop cursor.

        head –  the head for your list.

    * `list_for_each_entry`

        This is used to iterate over a list of the given type.

        ```c
        list_for_each_entry(pos, head, member);
        ```

        pos – the type * to use as a loop cursor.

        head – the head for your list.

        member – the name of the list_head within the struct.

    * `list_for_each_entry_safe`

        This will iterate over the list of given type-safe against the removal of list entry.

        `list_for_each_entry_safe ( pos, n, head, member);`

        Where,

        pos – the type * to use as a loop cursor.

        n – another type * to use as temporary storage

        head – the head for your list.

        member – the name of the list_head within the struct.

    * `list_for_each_prev`

        This will be used to iterate over a list backward.

        list_for_each_prev(pos, head);

        pos – the &struct list_head to use as a loop cursor.

        head – the head for your list.

    * `list_for_each_entry_reverse`

        This macro is used to iterate backward over the list of the given type.
        
        list_for_each_entry_reverse(pos, head, member);

        pos – the type * to use as a loop cursor.

        head  the head for your list.

        member – the name of the list_head within the struct.

1. When we write the value to our device file using echo value > /dev/etx_value, it will invoke the interrupt. Because we configured the interrupt by using the software. If you don’t know how it works, please refer to this tutorial.

1. The interrupt will invoke the ISR function.

1. In ISR we are allocating work to the Workqueue.

1. Whenever Workqueue executes, we are creating the Linked List Node and adding the Node to the Linked List.

1. When we are reading the driver using cat /dev/etx_device, printing all the nodes which are present in the Linked List using traverse.

1. When we are removing the driver using rmmod, it will remove all the nodes in Linked List and free the memory.

Creating Head Node

```c
/*Declare and init the head node of the linked list*/
LIST_HEAD(Head_Node);
```

This will create the head node in the name of `Head_Node` and initialize that.

Creating Node and add that into Linked List

```c
/*Creating Node*/
temp_node = kmalloc(sizeof(struct my_list), GFP_KERNEL);

/*Assgin the data that is received*/
temp_node->data = etx_value;

/*Init the list within the struct*/
INIT_LIST_HEAD(&temp_node->list);

/*Add Node to Linked List*/
list_add_tail(&temp_node->list, &Head_Node);
```

Traversing Linked List:

```c
struct my_list *temp;
int count = 0;
printk(KERN_INFO "Read function\n");

/*Traversing Linked List and Print its Members*/
list_for_each_entry(temp, &Head_Node, list) {
    printk(KERN_INFO "Node %d data = %d\n", count++, temp->data);
}

printk(KERN_INFO "Total Nodes = %d\n", count);
```

Deleting Linked List:

```c
/* Go through the list and free the memory. */
struct my_list *cursor, *temp;
list_for_each_entry_safe(cursor, temp, &Head_Node, list) {
    list_del(&cursor->list);
    kfree(cursor);
}
```

source code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Kernel Linked List)
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
volatile int etx_value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
static struct workqueue_struct *own_workqueue;
 
 
static void workqueue_fn(struct work_struct *work); 
 
static DECLARE_WORK(work, workqueue_fn);
 
/*Linked List Node*/
struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};
 
/*Declare and init the head node of the linked list*/
LIST_HEAD(Head_Node);
 
/*
** Function Prototypes
*/ 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
/******************************************************/
 
 
/*Workqueue Function*/
static void workqueue_fn(struct work_struct *work)
{
        struct my_list *temp_node = NULL;
 
        printk(KERN_INFO "Executing Workqueue Function\n");
        
        /*Creating Node*/
        temp_node = kmalloc(sizeof(struct my_list), GFP_KERNEL);
 
        /*Assgin the data that is received*/
        temp_node->data = etx_value;
 
        /*Init the list within the struct*/
        INIT_LIST_HEAD(&temp_node->list);
 
        /*Add Node to Linked List*/
        list_add_tail(&temp_node->list, &Head_Node);
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred\n");
        /*Allocating work to queue*/
        queue_work(own_workqueue, &work);
        
        return IRQ_HANDLED;
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
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/  
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This fuction will be called when we write the sysfsfs file
*/  
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        return count;
}

/*
** This fuction will be called when we open the Device file
*/ 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/   
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        struct my_list *temp;
        int count = 0;
        printk(KERN_INFO "Read function\n");
 
        /*Traversing Linked List and Print its Members*/
        list_for_each_entry(temp, &Head_Node, list) {
            printk(KERN_INFO "Node %d data = %d\n", count++, temp->data);
        }
 
        printk(KERN_INFO "Total Nodes = %d\n", count);
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        /*Copying data from user space*/
        sscanf(buf,"%d",&etx_value);
        /* Triggering Interrupt */
        asm("int $0x3B");  // Corresponding to irq 11
        return len;
}
 
/*
** Module Init function
*/  
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device \n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ \n");
                    goto irq;
        }
 
        /*Creating workqueue */
        own_workqueue = create_workqueue("own_wq");
        
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module Exit function
*/ 
static void __exit etx_driver_exit(void)
{
 
        /* Go through the list and free the memory. */
        struct my_list *cursor, *temp;
        list_for_each_entry_safe(cursor, temp, &Head_Node, list) {
            list_del(&cursor->list);
            kfree(cursor);
        }
 
        /* Delete workqueue */
        destroy_workqueue(own_workqueue);
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Kernel Linked List");
MODULE_VERSION("1.13");
```

test:

```bash
cat /dev/etx_device
echo 10 > /dev/etx_device
```

## Thread

There are two types of threads.

1. User Level Thread

    In this type, the kernel is not aware of these threads. Everything is maintained by the user thread library. That thread library contains code for creating and destroying threads, for passing messages and data between threads, for scheduling thread execution, and for saving and restoring thread contexts. So all will be in User Space.

1. Kernel Level Thread

    Kernel level threads are managed by the OS, therefore, thread operations are implemented in the kernel code. There is no thread management code in the application area.

kernel level thread

header file: `linux/kthread.h`

Create Kernel Thread

* `kthread_create`

    Syntax:

    ```c
    struct task_struct * kthread_create (int (* threadfn(void *data), void *data, const char namefmt[], ...);
    ```

    Where,

    `threadfn` – the function to run until signal_pending(current).

    `data` – data ptr for threadfn.

    `namefmt[]` – printf-style name for the thread.

    `...` – variable arguments

    This helper function creates and names a kernel thread. But we need to wake up that thread manually. When woken, the thread will run `threadfn()` with data as its argument.

    `threadfn` can either call `do_exit` directly if it is a standalone thread for which no one will call `kthread_stop`, or return when ‘`kthread_should_stop`‘ is true (which means `kthread_stop` has been called). The return value should be zero or a negative error number; it will be passed to `kthread_stop`.

    It Returns `task_struct` or `ERR_PTR(-ENOMEM)`.

Start Kernel Thread

* `wake_up_process`

    Syntax:

    ```c
    int wake_up_process (struct task_struct * p);
    ```

    p – The process to be woken up.

    Attempt to wake up the nominated process and move it to the set of runnable processes.

    It returns 1 if the process was woken up, 0 if it was already running.

    It may be assumed that this function implies a write memory barrier before changing the task state if and only if any tasks are woken up.

Stop Kernel Thread

* `kthread_stop`

    Syntax:

    ```c
    int kthread_stop ( struct task_struct *k);
    ```

    Where,

    `k` – thread created by kthread_create.

    Sets kthread_should_stop for k to return true, wakes it and waits for it to exit. Your threadfn must not call do_exit itself, if you use this function! This can also be called after kthread_create instead of calling wake_up_process: the thread will exit without calling threadfn.

    It Returns the result of threadfn, or –EINTR if wake_up_process was never called.

Other functions in Kernel Thread

* `kthread_should_stop`

    ```c
    int kthread_should_stop (void);
    ```

    When someone calls `kthread_stop` on your kthread, it will be woken and this will return `true`. You should then return, and your return value will be passed through to `kthread_stop`.

    相当于由外部通知当前线程可以结束了。这种常见的场景，比如执行`ping`命令，默认一直发送 icmp 包，如果没有外部`Ctrl + C`信号，则不会主动停止。

* `kthread_bind`

    ```c
    void kthread_bind(struct task_struct *k, unsigned int cpu);
    ```

    `k` – thread created by kthread_create.

    `cpu` – CPU (might not be online, must be possible) for k to run on.


Thread Function

First, we have to create our thread that has the argument of void *  and should return int value.  We should follow some conditions in our thread function. It is advisable.

* If that thread is a long run thread, we need to check `kthread_should_stop()` every time, because any function may call kthread_stop. If any function called kthread_stop, that time `kthread_should_stop` will return true. We have to exit our thread function if true value been returned by `kthread_should_stop`.

* But if your thread function is not running long, then let that thread finish its task and kill itself using do_exit.

Example:

```c
int thread_function(void *pv) 
{
    int i=0;
    while(!kthread_should_stop())
    {
        printk(KERN_INFO "In EmbeTronicX Thread Function %d\n", i++);
        msleep(1000);
    } 
    return 0; 
}
```

```c
static struct task_struct *etx_thread; 

etx_thread = kthread_create(thread_function,NULL,"eTx Thread");

if (etx_thread) 
{
    wake_up_process(etx_thread); 
} 
else 
{
    printk(KERN_ERR "Cannot create kthread\n"); 
}
```

`kthread_run`:

Syntax:

```c
kthread_run (threadfn, data, namefmt, ...);
```

Where,

`threadfn` – the function to run until signal_pending(current).

`data` – data ptr for threadfn.

`namefmt` – printf-style name for the thread.

`...` – variable arguments

Convenient wrapper for `kthread_create` followed by `wake_up_process`.

It returns the `kthread` or `ERR_PTR(-ENOMEM)`.

Example:

```c
static struct task_struct *etx_thread;

etx_thread = kthread_run(thread_function,NULL,"eTx Thread"); 
if(etx_thread) 
{
 printk(KERN_ERR "Kthread Created Successfully...\n");
}
else 
{
 printk(KERN_ERR "Cannot create kthread\n"); 
}
```

Source code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Kernel Thread)
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
#include <linux/kthread.h>             //kernel threads
#include <linux/sched.h>               //task_struct 
#include <linux/delay.h>
#include <linux/err.h>
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
static struct task_struct *etx_thread;
 
/*
** Function Prototypes
*/
/*************** Driver functions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 /******************************************************/
 
int thread_function(void *pv);

/*
** Thread
*/
int thread_function(void *pv)
{
    int i=0;
    while(!kthread_should_stop()) {
        pr_info("In EmbeTronicX Thread Function %d\n", i++);
        msleep(1000);
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
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read function\n");
 
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write Function\n");
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
            pr_err("Cannot create the Device \n");
            goto r_device;
        }
 
        etx_thread = kthread_create(thread_function,NULL,"eTx Thread");
        if(etx_thread) {
            wake_up_process(etx_thread);
        } else {
            pr_err("Cannot create kthread\n");
            goto r_device;
        }
#if 0
        /* You can use this method also to create and run the thread */
        etx_thread = kthread_run(thread_function,NULL,"eTx Thread");
        if(etx_thread) {
            pr_info("Kthread Created Successfully...\n");
        } else {
            pr_err("Cannot create kthread\n");
             goto r_device;
        }
#endif
        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
 
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/  
static void __exit etx_driver_exit(void)
{
        kthread_stop(etx_thread);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Kernel Thread");
MODULE_VERSION("1.14");
```

Latest progress: <https://embetronicx.com/tutorials/linux/device-drivers/tasklet-static-method/>

## Miscellaneous

* modprobe 的作用

    在 linux 中，如果一些内核模块之间有依赖关系，那么必须按依赖关系进行`insmod`，否则会报错。

    `modprobe`会根据`depmod`所产生的相依关系，决定要载入哪些模块。若在载入过程中发生错误，在`modprobe`会卸载整组的模块。

    example:

    载入模块：

    ```bash
    sudo modprobe -v xdxgpu
    ```

    尝试制裁模块：

    ```bash
    sudo modprobe -vr xdxgpu
    ```

* depmod 命令

    `depmod`通常在`modprobe`之前运行，用于分析可载入模块的依赖关系。

    example:

    分析所有可用模块的依赖关系：

    ```bash
    sudo depmod -av
    ```

    注：

    1. 这里没有加路径，可能是会分析到当前目录下？
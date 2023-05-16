# Linux Driver Note QA

[unit]
[u_0]
请写出一个最小版本的 linux 内核模块代码及 Makefile，对其进行编译，并检测是否能成功加载及退出。
[u_1]
`hello.c`:

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

`Makefile`:

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-41-generic
obj-m += hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

加载：`sudo insmod hello_world.ko`

查看输出：`sudo dmesg`

卸载：`sudo rmmod hello_world`

[unit]
[u_0]
分别注册整型，字符串，以及数组作为模块参数，并检验是否注册成功。使用回调函数对参数值进行监视，并检测其改变。
[u_1]
`hello.c`:

```c
#include <linux/init.h>
#include <linux/module.h>

int m_int = 5;
char *m_str = "hello, world";
int m_arr[3];

module_param(m_int, int, S_IRUSR | S_IWUSR);
module_param(m_str, charp, S_IRUSR | S_IWUSR);
module_param_array(m_arr, int, NULL, 0755);

int hello_init(void)
{
    printk("my hello world driver\n");
    return 0;
}

void hello_exit(void)
{
    printk("good bye my driver\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

检验是否注册成功：

`ls /sys/module/hello_world/parameters/`

修改参数值：

`echo 1 > /sys/module/hello_world/parameters/m_int`

[unit]
[u_0]
解释`module_param_array()`中各个参数的作用。
[u_1]
Syntax:

`module_param_array(name, type, int *num, permissions);`

其中，`name`为数组名，`type`为数组的元素类型，`num`是一个指针，当我们使用命令行传递模块参数时，`module_param_array()`会自动计算数组中元素的个数，并将元素数量存储到这个变量中，这个指针也可以直接填`NULL`，`permissions`指的是参数文件的权限。

[unit]
[u_0]
内核模块符号导出。（是否能导出变量，数组，字符串？）
[u_1]
(empty)

[unit]
[u_0]
请使用静态方式申请设备号。
[u_1]
(empty)

[unit]
[u_0]
请使用动态方法申请设备号。
[u_1]
(empty)

[unit]
[u_0]
请写出添加及删除 cdev 的最小代码。
[u_1]
(empty)

[unit]
[u_0]
如何动创建 cdev 设备文件？
[u_1]
(empty)

[unit]
[u_0]
如何使用代码创建 cdev 设备文件？
[u_1]
(empty)

[unit]
[u_0]
请使用代码测试 cdev 是否成功。
[u_1]
(empty)

[unit]
[u_0]
写出`printk()`的所有消息类型，并解释其含义。
[u_1]
(empty)

[unit]
[u_0]
写出`printk()`的常用替代函数。
[u_1]
(empty)

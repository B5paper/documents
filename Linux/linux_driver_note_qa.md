# Linux Driver Note QA

[unit]
[title]
内核模块的加载与卸载
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
obj-m += hello.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

加载：`sudo insmod hello_world.ko`

查看输出：`sudo dmesg`

卸载：`sudo rmmod hello_world`

[unit]
[title]
模块参数的使用
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
[title]
模块符号的导出
[u_0]
内核模块符号导出。（是否能导出变量，数组，字符串？）
[u_1]
(empty)

[unit]
[u_0]
请使用静态方式申请设备号。
[u_1]
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>

int hello(void)
{
    printk(KERN_WARNING "hello my module\n");
    dev_t dev_num = MKDEV(255, 0);
    register_chrdev_region(dev_num, 1, "my driver");
    return 0;
}

void bye(void)
{
    unregister_chrdev_region(MKDEV(255, 0), 1);
    printk("goodbye my module\n");
}

module_init(hello)
module_exit(bye)
MODULE_LICENSE("GPL");
```

[unit]
[u_0]
请使用动态方法申请设备号。
[u_1]
(empty)

[unit]
[u_0]
请写出添加及删除 cdev 的最小代码。
[u_1]
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

long m_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    printk("in m_write function ...\n");
    return 0;
}

long m_ioctl(struct file *file_ptr, unsigned int, unsigned long)
{
    printk("in m_ioctl function ...\n");
    return 0;
}

dev_t dev_num;
struct cdev m_dev;
const char *m_dev_name = "hlc_dev";
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
    printk("my hello world driver\n");
    alloc_chrdev_region(&dev_num, 0, 1, m_dev_name);
    
    cdev_init(&m_dev, &m_ops);
    cdev_add(&m_dev, dev_num, 1);
    return 0;
}

void hello_exit(void)
{
    printk("good bye my driver\n");
    cdev_del(&m_dev);
    unregister_chrdev_region(dev_num, 1);
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

[unit]
[deps]
内核模块的加载与卸载
[u_0]
如何手动创建 cdev 设备文件？
[u_1]
(empty)

[unit]
[u_0]
如何使用代码创建 cdev 设备文件？
[u_1]
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/device.h>
#include <linux/fs.h>

dev_t dev = 0;
static struct class *dev_class;

int load_my_driver(void)
{
    printk("load my driver\n");
    alloc_chrdev_region(&dev, 0, 1, "etx_dev");
    dev_class = class_create(THIS_MODULE, "etx_class");
    device_create(dev_class, NULL, dev, NULL, "etx_device");
    return 0;
}

void unload_my_driver(void)
{
    device_destroy(dev_class, dev);
    class_destroy(dev_class);
    unregister_chrdev_region(dev, 1);
    printk("unload my driver\n");
}

module_init(load_my_driver);
module_exit(unload_my_driver);
MODULE_LICENSE("GPL");
```

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
* `pr_info()` – Print an info-level message. (ex. `pr_info("test info message\n")`).
* `pr_cont()` – Continues a previous log message in the same line.
* `pr_debug()` – Print a debug-level message conditionally.
* `pr_err()` – Print an error-level message. (ex. `pr_err(“test error message\n”)`).
* `pr_warn()` – Print a warning-level message.

[unit]
[u_0]
实现通过用户空间的`read()`和`write()`函数与内核空间交换数据。
[u_1]
(empty)

[unit]
[u_0]
`struct file_operations`中`open`函数的原型是什么？解释各个参数的含义。
[u_1]
Syntax:

```c
int m_open(struct inode *inode, struct file *file_ptr);
```

（目前两个参数都用不到，所以不知道是啥意思）

[unit]
[u_0]
`struct file_operations`中`read`函数的原型是什么？解释各个参数的含义。
[u_1]
Syntax:

```c
long int m_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset);
```

（目前各个参数都用不到，所以不知道是什么意思）

[unit]
[u_0]
写出`struct file_operations`中的常用字段。
[u_1]
```c
struct file_operations
{
    struct module *owner;
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    int (*open) (struct inode *, struct file *);
    int (*release) (struct inode *, struct file *);
};
```
# Linux Driver Note QA

[unit]
[idx]
0
[id]
40735460710940864
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
[idx]
1
[id]
40741716463024864
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
[idx]
2
[id]
40743422786851680
[u_0]
解释`module_param_array()`中各个参数的作用。
[u_1]
Syntax:

`module_param_array(name, type, nump, perm);`

其中，`name`为数组名，`type`为数组的元素类型，`nump`是一个指针，当我们使用命令行传递模块参数时，`module_param_array()`会自动计算数组中元素的个数，并将元素数量存储到这个变量中，这个指针也可以直接填`NULL`，`perm`指的是参数文件的权限。

[unit]
[idx]
3
[id]
40746586403460328
[u_0]
请使用静态方式申请设备号。
[u_1]
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>

dev_t dev_num = MKDEV(255, 0);

int hello(void)
{
    printk(KERN_WARNING "hello my module\n");
    register_chrdev_region(dev_num, 1, "my driver");
    return 0;
}

void bye(void)
{
    unregister_chrdev_region(dev_num, 1);
    printk("goodbye my module\n");
}

module_init(hello)
module_exit(bye)
MODULE_LICENSE("GPL");
```

[unit]
[idx]
4
[id]
40749489459808928
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
[idx]
5
[id]
40752482444034880
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
    dev_class = class_create("etx_class");
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
[idx]
6
[id]
40756729806857856
[u_0]
写出`printk()`的常用替代函数。
[u_1]
* `pr_info()` – Print an info-level message. (ex. `pr_info("test info message\n")`).
* `pr_cont()` – Continues a previous log message in the same line.
* `pr_debug()` – Print a debug-level message conditionally.
* `pr_err()` – Print an error-level message. (ex. `pr_err(“test error message\n”)`).
* `pr_warn()` – Print a warning-level message.

[unit]
[idx]
7
[id]
40759699732653720
[u_0]
`struct file_operations`中`open`函数的原型是什么？解释各个参数的含义。
[u_1]
Syntax:

```c
int m_open(struct inode *inode, struct file *file_ptr);
```

（目前两个参数都用不到，所以不知道是啥意思）

[unit]
[idx]
8
[id]
40761094767674296
[u_0]
`struct file_operations`中`read`函数的原型是什么？解释各个参数的含义。
[u_1]
Syntax:

```c
long int m_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset);
```

（目前各个参数都用不到，所以不知道是什么意思）

[unit]
[idx]
9
[id]
40762499026066904
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

[unit]
[idx]
10
[id]
5737746927749468565
[u_0]
使用命令创建设备文件。
[u_1]
可以使用`mknod`创建设备文件：

```bash
sudo mknod -m 666 /dev/hlc_dev c 255 0
```

最后两个参数分别是主设备号和次设备号。

测试：

```bash
cat /dev/hlc_dev
```

查看输出：

```bash
sudo dmesg
```

output:

```
[ 7716.807868] in m_open()...
[ 7716.807889] in m_read()...
[ 7716.807905] in m_release()...
```

[unit]
[idx]
11
[id]
4188117425824722713
[u_0]
写出动态申请设备号并注销的完整代码。
[u_1]
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>

dev_t dev_region;
const char *dev_region_name = "hlc dev region";

int mod_init(void)
{
    printk(KERN_INFO "in mod_init() ...\n");
    alloc_chrdev_region(&dev_region, 0, 1, dev_region_name);
    printk(KERN_INFO "allocate device region.\n");
    return 0;
}

void mod_exit(void)
{
    printk(KERN_INFO "in mod_exit() ...\n");
    unregister_chrdev_region(dev_region, 1);
    printk(KERN_INFO "unregistered device region.\n");
}

module_init(mod_init);
module_exit(mod_exit);
MODULE_LICENSE("GPL");
```

[unit]
[idx]
12
[id]
199112764048893592
[u_0]
请写一个使用 linked list 的程序，在链表头部插入 1, 2, 3，并遍历输出。
[u_1]
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/list.h>

int init_mod(void);
void exit_mod(void);

typedef struct ListNode
{
    struct list_head head;
    int val;
} ListNode;

int init_mod(void)
{
    pr_info("in hello_init()\n");

    struct list_head lst_head;
    INIT_LIST_HEAD(&lst_head);
    ListNode node_1 = {
        .val = 1
    };
    struct ListNode node_2 = {
        .val = 2
    };
    ListNode node_3 = {
        .val = 3
    };
    list_add(&node_1.head, &lst_head);
    list_add(&node_2.head, &lst_head);
    list_add(&node_3.head, &lst_head);
    ListNode *cur;
    int len_cnt = 0;
    list_for_each_entry(cur, &lst_head, head)
    {
        printk(KERN_INFO "%d\n", cur->val);
        ++len_cnt;
    }
    pr_info("len cnt: %d\n", len_cnt);
    return 0;
}

void exit_mod(void)
{
    pr_info("in hello_exit()\n");
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");
```

dmesg output:

```
[ 6687.347700] in hello_init()
[ 6687.347706] 3
[ 6687.347708] 2
[ 6687.347709] 1
[ 6687.347710] len cnt: 3
[ 6692.841747] in hello_exit()
```

[unit]
[idx]
13
[id]
15625035695245153604
[u_0]
配置 vscode 的内核驱动开发环境, 使得 hello world 程序下无静态报错。
[u_1]
include paths:

```json
"/usr/src/linux-headers-6.5.0-18-generic/include/",
"/usr/src/linux-headers-6.5.0-18-generic/arch/x86/include/generated/",
"/usr/src/linux-hwe-6.5-headers-6.5.0-18/arch/x86/include/",
"/usr/src/linux-hwe-6.5-headers-6.5.0-18/include"
```

compiler macros:

```
KBUILD_MODNAME=\"hello\"
__GNUC__
__KERNEL__
MODULE
```

[unit]
[idx]
14
[id]
7941022488153863586
[u_0]
写一个程序实现 cdev 的基本 file operations，并且自动创建 dev file 节点。
[u_1]
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>

int m_open(struct inode *, struct file *) {
    pr_info("in m_open()...\n");
    return 0;
}

int m_release(struct inode *, struct file *) {
    pr_info("in m_release()...\n");
    return 0;
}

ssize_t m_read(struct file *, char __user *, size_t, loff_t *) {
    pr_info("in m_read()...\n");
    return 0;
}

ssize_t m_write(struct file *, const char __user *, size_t, loff_t *) {
    pr_info("in m_write()...\n");
    return 0;
}

long m_unlocked_ioctl(struct file *, unsigned int, unsigned long) {
    pr_info("in m_unlocked_ioctl()...\n");
    return 0;
}

dev_t dev_num;
const char *dev_region_name = "hlc dev";
struct cdev chdev;
const struct file_operations chdev_ops = {
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write,
    .unlocked_ioctl = m_unlocked_ioctl
};
struct class *dev_cls;
struct device *dev;

int hello_init(void) {
    pr_info("hello my module\n");
    int ret = alloc_chrdev_region(&dev_num, 0, 1, dev_region_name);
    if (ret != 0) {
        pr_info("fail to register chrdev region\n");
        return -1;
    }

    cdev_init(&chdev, &chdev_ops);
    ret = cdev_add(&chdev, dev_num, 1);
    if (ret != 0) {
        pr_info("fail to add cdev\n");
        goto CDEV_ADD_FAILED;
    }

    dev_cls = class_create("hlc dev cls");
    if (dev_cls == NULL) {
        pr_err("fail to create class\n");
        goto CLASS_CREATE_FAILED;
    }
    dev = device_create(dev_cls, NULL, dev_num, NULL, "hlc_dev");
    if (dev == NULL) {
        pr_err("fail to create device\n");
        goto DEVICE_CREATE_FAILED;
    }
    return 0;

DEVICE_CREATE_FAILED:
    class_destroy(dev_cls);
CLASS_CREATE_FAILED:
    cdev_del(&chdev);
CDEV_ADD_FAILED:
    unregister_chrdev_region(dev_num, 1);
    return -1;
}

void hello_exit(void) {
    pr_info("exit my module\n");
    device_destroy(dev_cls, dev_num);
    class_destroy(dev_cls);
    cdev_del(&chdev);
    unregister_chrdev_region(dev_num, 1);
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```


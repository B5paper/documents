#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/kthread.h>

dev_t dev_num;
const char *cls_name = "hlc_cls";
const char *dev_num_name = "hlc_dev_num";
struct cdev hlc_cdev;
struct class *hlc_dev_cls;
const char *hlc_dev_name = "hlc_dev";
struct device *hlc_dev;
void *allocated_mem = NULL;
size_t mem_len;

int h_open(struct inode *inod, struct file *file_ptr)
{
    pr_info("in h_open()...\n");
    return 0;
}

int h_release(struct inode *inod, struct file *file_ptr)
{
    pr_info("in h_release()...\n");
    return 0;
}

ssize_t h_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in h_read()...\n");
    return 0;
}

wait_queue_head_t wq;
int condi = 1;

ssize_t h_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in h_write()...\n");
    copy_from_user(&condi, buf, sizeof(condi));
    wake_up_interruptible(&wq);
    return sizeof(condi);
}

#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)

ssize_t h_ioctl(struct file *file_ptr, unsigned int cmd, unsigned long arg)
{
    unsigned long rtv;
    static int32_t value;
    switch (cmd)
    {
        case WR_VALUE:
            rtv = copy_from_user(&value, (int32_t*) arg, sizeof(value));
            if (rtv > 0) {
                pr_info("copy_from_user() remain %ld bytes\n", rtv);
            }
            pr_info("Value = %d\n", value);
            break;
        case RD_VALUE:
            rtv = copy_to_user((int32_t*) arg, &value, sizeof(value));
            if (rtv > 0) {
                pr_info("copy_to_user() remain %ld bytes\n", rtv);
            }
            break;
        default:
            pr_info("unrecognized ioctl command\n");
    }
    return 0;
}

struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = h_open,
    .release = h_release,
    .read = h_read,
    .write = h_write,
    .unlocked_ioctl = h_ioctl
};

int print_msg(void *unused)
{
    pr_info("wait condition variable...\n");
    wait_event_interruptible(wq, condi == 2);
    pr_info("condi is %d, hello, world\n", condi);
    condi = 1;
    pr_info("rechange condi to 1\n");
    return 0;
}

int mod_init(void)
{
    pr_info("in mod_init()...\n");
    int rtv;
    rtv = alloc_chrdev_region(&dev_num, 0, 1, dev_num_name);
    if (rtv != 0) {
        pr_info("fail to allocate device region\n");
        goto r_alloc_chrdev_region;
    }
    cdev_init(&hlc_cdev, &fops);
    if (IS_ERR(&hlc_cdev)) {
        pr_info("fail to init cdev\n");
        goto r_cdev_init;
    }
    rtv = cdev_add(&hlc_cdev, dev_num, 1);
    if (rtv != 0) {
        pr_info("fail to add cdev\n");
        goto r_cdev_add;
    }
    hlc_dev_cls = class_create("hlc_cls");
    if (IS_ERR(hlc_dev_cls)) {
        pr_info("fail to create class\n");
        goto r_class_create;
    }
    hlc_dev = device_create(hlc_dev_cls, NULL, dev_num, NULL, hlc_dev_name);
    if (IS_ERR(hlc_dev)) {
        pr_info("fail to create device\n");
        goto r_device_create;
    }
    pr_info("successfully create hlc dev\n");

    pr_info("init waitqueue\n");
    init_waitqueue_head(&wq);

    struct task_struct *wait_thread = kthread_create(print_msg, NULL, "print_msg");
    if (wait_thread) {
        pr_info("wake up process\n");
        wake_up_process(wait_thread);
    }
    pr_info("start print_msg thread\n");
    return 0;

r_device_create:
    class_destroy(hlc_dev_cls);
r_class_create:
    cdev_del(&hlc_cdev);
r_cdev_add:
r_cdev_init:
    unregister_chrdev_region(dev_num, 1);
r_alloc_chrdev_region:
    return -1;
}

void mod_exit(void)
{
    pr_info("in mod_exit()...\n");
    wake_up_interruptible(&wq);
    device_destroy(hlc_dev_cls, dev_num);
    class_destroy(hlc_dev_cls);
    cdev_del(&hlc_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("hlc dev removed.\n");
}

module_init(mod_init);
module_exit(mod_exit);
MODULE_LICENSE("GPL");
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/workqueue.h>

dev_t dev_num;
int m_open(struct inode *inode, struct file *file);
int m_release(struct inode *inode, struct file *file);
ssize_t m_read(struct file *file, char __user *buf, size_t size, loff_t *offset);
ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset);
struct cdev m_cdev;
struct class *dev_cls;
struct device *m_dev;
struct file_operations fops = {
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write
};

struct workqueue_struct *wque;
struct work_struct witem;
void work_func(struct work_struct *witem)
{
    pr_info("in work func\n");
}

int init_mod(void)
{
    pr_info("in init_mod()...\n");
    alloc_chrdev_region(&dev_num, 0, 1, "hlc_dev_num");
    cdev_init(&m_cdev, &fops);
    cdev_add(&m_cdev, dev_num, 1);
    dev_cls = class_create("hlc_dev_cls");
    m_dev = device_create(dev_cls, NULL, dev_num, NULL, "hlc_dev");
    wque = create_workqueue("hlc_wque");
    INIT_WORK(&witem, work_func);
    return 0;
}

void exit_mod(void)
{
    pr_info("in exit_()...\n");
    destroy_workqueue(wque);
    device_destroy(dev_cls, dev_num);
    class_destroy(dev_cls);
    cdev_del(&m_cdev);
    unregister_chrdev_region(dev_num, 1);
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");

int m_open(struct inode *inode, struct file *file)
{
    pr_info("in m_open()...\n");
    return 0;
}

int m_release(struct inode *inode, struct file *file)
{
    pr_info("in m_release()...\n");
    return 0;
}

ssize_t m_read(struct file *file, char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in m_read()...\n");
    queue_work(wque, &witem);
    return 0;
}

ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in m_write()...\n");
    return 0;
}
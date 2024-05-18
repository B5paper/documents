#include <linux/init.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/irqnr.h>
#include <linux/interrupt.h>
#include <linux/workqueue.h>
#include <asm/hw_irq.h>

dev_t dev_num;
struct cdev my_cdev;
struct class *dev_cls;
struct device *dev;

int m_open(struct inode *inode, struct file *file);
int m_release(struct inode *inode, struct file *file);
ssize_t m_read(struct file *file, char __user *buf, size_t size, loff_t *offset);
ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset);
irqreturn_t irq_handler(int irq, void *dev_id);
void work_queue_fn(struct work_struct *work_item);

struct file_operations fops = {
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write
};

struct work_struct work_item;

int init_mod(void)
{
    pr_info("in init_mod()...\n");
    alloc_chrdev_region(&dev_num, 0, 1, "hlc_dev_num");
    cdev_init(&my_cdev, &fops);
    cdev_add(&my_cdev, dev_num, 1);
    dev_cls = class_create("hlc_dev_cls");
    dev = device_create(dev_cls, NULL, dev_num, NULL, "hlc_dev");
    request_irq(11, irq_handler, IRQF_SHARED, "hlc_dev", irq_handler);
    INIT_WORK(&work_item, work_queue_fn);
    pr_info("init hlc module done.\n");
    return 0;
}

void exit_mod(void)
{
    pr_info("in exit_mod()...\n");
    free_irq(11, irq_handler);
    device_destroy(dev_cls, dev_num);
    class_destroy(dev_cls);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("exit hlc module done.\n");
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
    struct irq_desc *desc = irq_to_desc(11);
    __this_cpu_write(vector_irq[59], desc);
    asm("int $0x3B");
    return 0;
}

ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in m_write()...\n");
    return 0;
}

irqreturn_t irq_handler(int irq, void *dev_id)
{
    pr_info("in irq_handler()...\n");
    schedule_work(&work_item);
    return IRQ_HANDLED;
}

void work_queue_fn(struct work_struct *work_item)
{
    pr_info("in work_queue_fn()...\n");
}


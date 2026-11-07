#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>

int init_mod(void);
void exit_mod(void);

dev_t dev_num;
const char *my_dev_name = "hlc_dev";

int my_open(struct inode *inode, struct file *file_ptr) {
    pr_info("in my_open ...\n");
    return 0;
}

int my_release(struct inode *inode, struct file *file_ptr) {
    pr_info("in my_release...\n");
    return 0;
}

char hlc_dev_msg[] = "Hello, user! This is a greeting message sent from hlc_dev.\n";

ssize_t my_read(struct file *file_ptr, char __user *user_buf, size_t read_bytes, loff_t *offset) {
    pr_info("hlc cdev: my_read start...\n");

    const int hlc_dev_msg_len = sizeof(hlc_dev_msg) - 1;
    pr_info("hlc cdev msg str len: %d\n", hlc_dev_msg_len);

    pr_info("user requires: read from offset %llu, read bytes: %lu\n", *offset, read_bytes);

    if (*offset < 0) {
        pr_err("offset < 0, offset: %lld\n", *offset);
        return -EINVAL;
    }

    if (*offset >= hlc_dev_msg_len || read_bytes == 0) {
        return 0;
    }

    size_t real_read_bytes = min(read_bytes, (unsigned long) (hlc_dev_msg_len - *offset));
    
    unsigned long ret;
    ret = copy_to_user(user_buf, hlc_dev_msg + *offset, real_read_bytes);
    unsigned long bytes_not_copied = ret;

    unsigned long bytes_copied = real_read_bytes - bytes_not_copied;
    if (bytes_copied == 0) {
        pr_err("error: copied 0 bytes");
        return -EFAULT;
    }
    *offset += bytes_copied;
    pr_info("after read, real_read_bytes: %lu, new offset: %lld\n", real_read_bytes, *offset);

    pr_info("hlc cdev: my_read end.\n");
    return bytes_copied;
}

ssize_t my_write(struct file *pfile, const char __user *buf, size_t bytes_write, loff_t *offset) {
    pr_info("hlc cdev: my_write() start...\n");

    pr_info("hlc cdev: user requires write %lu bytes, offset: %lld\n", bytes_write, *offset);

    if (*offset < 0) {
        pr_err("offset < 0, offset: %lld\n", *offset);
        return -EINVAL;
    }

    size_t hlc_dev_msg_len = sizeof(hlc_dev_msg) - 1;
    if (bytes_write == 0 || *offset >= hlc_dev_msg_len) {
        return 0;
    }

    size_t bytes_left = hlc_dev_msg_len - *offset;
    size_t real_bytes_write = min(bytes_write, bytes_left);
    // 这里的 to 参数没有使用 hlc_dev_msg + *offset，所以每次都是从头开始覆盖
    unsigned long bytes_not_copied = copy_from_user(hlc_dev_msg, buf, real_bytes_write);
    size_t bytes_copied = real_bytes_write - bytes_not_copied;

    if (bytes_copied == 0) {
        pr_err("hlc cdev: copied 0 bytes\n");
        return -EFAULT;
    }

    *offset += bytes_copied;
    pr_info("hlc cdev: after write, %lu bytes copied, new offset: %lld", bytes_copied, *offset);

    pr_info("hlc cdev: my_write() finished.\n");
    return bytes_copied;
}

struct cdev my_cdev;
struct file_operations my_fops = {
    .open = my_open,
    .release = my_release,
    .read = my_read,
    .write = my_write
};

int init_mod(void) {
    printk(KERN_INFO "hlc cdev: start init mod\n");

    pr_info("alloc chrdev region...\n");
    int ret = alloc_chrdev_region(&dev_num, 0, 1, my_dev_name);
    if (ret != 0) {
        pr_err("fail to alloc chrdev region\n");
        goto ERR_ALLOC_CHRDEV_REGION;
    }
    pr_info("allocated dev num: %d:%d\n", MAJOR(dev_num), MINOR(dev_num));

    pr_info("cdev init...\n");
    cdev_init(&my_cdev, &my_fops);

    pr_info("cdev add...\n");
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret != 0) {
        pr_err("fail to cdev add");
        goto ERR_CDEV_ADD;
    }

    pr_info("hlc cdev: finished init mod.\n");
    return 0;

ERR_CDEV_ADD:
    unregister_chrdev_region(dev_num, 1);
ERR_ALLOC_CHRDEV_REGION:
    return -1;
}

void exit_mod(void) {
    pr_info("hlc cdev start exit mod...\n");

    pr_info("cdev del...\n");
    cdev_del(&my_cdev);

    pr_info("unregister chrdev region...\n");
    unregister_chrdev_region(dev_num, 1);

    printk(KERN_INFO "hlc cdev finished exit mod.\n");
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");


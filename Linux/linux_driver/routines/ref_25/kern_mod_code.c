#include <linux/init.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <asm/io.h>
#include <linux/mm.h>  // remap_pfn_range()

dev_t dev_num_start;
struct cdev hlc_cdev;
struct file_operations cdev_ops;
struct class *hlc_cls;
struct device *hlc_dev;

enum IOCTL_CMDS {
    aaa
};

int hlc_open(struct inode *inode, struct file *file_ptr)
{
    pr_info("in hlc_open()...\n");
    return 0;
}

int hlc_release(struct inode *, struct file *)
{
    pr_info("in hlc_release()...\n");
    return 0;
}

char *buf;

int hlc_mmap(struct file *file_ptr, struct vm_area_struct *vma)
{
    pr_info("in hlc_mmap()...\n");

    phys_addr_t phy_addr = virt_to_phys(buf);
    uint64_t pfn = phy_addr >> PAGE_SHIFT;
    unsigned long len = vma->vm_end - vma->vm_start;
    pr_info("phy_addr: %llx\n", phy_addr);
    pr_info("pfn: %llu\n", pfn);
    pr_info("vma->vm_start: %lx, vma->vm_end: %lx\n", vma->vm_start, vma->vm_end);
    pr_info("len: %lu\n", len);

    int ret = remap_pfn_range(vma, vma->vm_start, pfn, len, vma->vm_page_prot);
    if (ret < 0)
    {
        pr_info("fail to map physical addr to virtual addr\n");
        return -1;
    }
    pr_info("successfully map physical addr to virtual addr\n");
    return 0;
}

long hlc_ioctl(struct file *, unsigned int cmd, unsigned long arg)
{
    pr_info("in hlc_ioctl()...\n");
    switch (cmd)
    {
        case aaa:
            pr_info("buf: %s\n", buf);
            break;

        default:
            pr_info("unrecognized ioctl cmd: %d\n", cmd);
            return -1;
    }
    return 0;
}

int init_mod(void)
{
    pr_info("in init_mod()...\n");

    buf = kmalloc(4096, GFP_KERNEL);
    strcpy(buf, "buffer from kernel\n");

    int ret = alloc_chrdev_region(&dev_num_start, 0, 1, "hlc dev region");
    if (ret != 0)
    {
        pr_info("fail to alloc chrdev region, ret: %d\n", ret);
        return -1;
    }

    cdev_ops.open = hlc_open;
    cdev_ops.release = hlc_release;
    cdev_ops.unlocked_ioctl = hlc_ioctl;
    cdev_ops.mmap = hlc_mmap;
    cdev_init(&hlc_cdev, &cdev_ops);
    ret = cdev_add(&hlc_cdev, dev_num_start, 1);
    if (ret != 0)
    {
        pr_info("fail to add cdev, ret: %d\n", ret);
        return -1;
    }

    hlc_cls = class_create("hlc cls");
    if (hlc_cls == NULL)
    {
        pr_info("fail to create class\n");
        return -1;
    }

    hlc_dev = device_create(hlc_cls, NULL, dev_num_start, NULL, "hlc_dev");
    if (hlc_dev == NULL)
    {
        pr_info("fail to create device\n");
        return -1;
    }
    
    return 0;
}

void exit_mod(void)
{
    pr_info("in exit_mod()...\n");
    kfree(buf);
    device_destroy(hlc_cls, dev_num_start);
    class_destroy(hlc_cls);
    cdev_del(&hlc_cdev);
    unregister_chrdev_region(dev_num_start, 1);
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");

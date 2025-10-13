#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>

dev_t dev_num;
struct cdev edu_cdev;
struct class *dev_cls;
struct device *edu_dev;

int edu_open(struct inode *, struct file *) {
    pr_info("in edu_open()...\n");
    return 0;
}

int edu_release(struct inode *, struct file *) {
    pr_info("in edu_release()...\n");
    return 0;
}

ssize_t edu_read(struct file *, char __user *, size_t, loff_t *) {
    pr_info("in edu_read()...\n");
    return 0;
}

ssize_t edu_write(struct file *, const char __user *, size_t, loff_t *) {
    pr_info("in edu_write()...\n");
    return 0;
}

long edu_ioctl(struct file *, unsigned int, unsigned long) {
    pr_info("in edu_ioctl()...\n");
    return 0;
}

struct file_operations fops = {
    .open = edu_open,
    .release = edu_release,
    .read = edu_read,
    .write = edu_write,
    .unlocked_ioctl = edu_ioctl
};

static struct pci_device_id pci_id_table[] = {
    { PCI_DEVICE(0x1234, 0x11e8) },
    {0,}
};

static void *base_addr_bar0;
static struct pci_dev *hlc_pci_dev = NULL;

irqreturn_t irq_handler(int irq, void *dev_id) {
    pr_info("in irq_handler()...\n");
    if (dev_id != hlc_pci_dev) {
        pr_warn("dev_id != hlc_pci_dev\n");
        return IRQ_NONE;
    }

    return IRQ_HANDLED;
}

static int edu_probe(struct pci_dev *pci_dev, const struct pci_device_id *id) {
    pr_info("in edu_probe()...\n");
    hlc_pci_dev = pci_dev;
    
    int ret = pci_enable_device(pci_dev);
    if (ret != 0) {
        dev_err(&pci_dev->dev, "fail to pci enable device, ret: %d\n", ret);
        goto ERR_PCI_ENABLE_DEVICE;
    }

    // mmio
    ret = pci_request_region(pci_dev, 0, "qemu_edu_drv");
    if (ret != 0) {
        dev_err(&pci_dev->dev, "fail to pci request region\n");
        goto ERR_PCI_REQUEST_REGION;
    }

    resource_size_t res_len_bar0 = pci_resource_len(pci_dev, 0);
    base_addr_bar0 = pci_iomap(pci_dev, 0, res_len_bar0);
    if (base_addr_bar0 == NULL) {
        dev_err(&pci_dev->dev, "fail to pci iomap\n");
        goto ERR_PCI_IOMAP;
    }

    // dma
    ret = dma_set_mask_and_coherent(&pci_dev->dev, DMA_BIT_MASK(28));
    if (ret != 0) {
        dev_err(&pci_dev->dev, "fail to set dma mask and conherent\n");
        goto ERR_DMA_SET_MASK_AND_COHERENT;
    }

    // irq
    ret = request_irq(pci_dev->irq, irq_handler, IRQF_SHARED, "qemu_edu_dev_irq_handler", pci_dev);
    if (ret != 0) {
        dev_err(&pci_dev->dev, "fail to request irq\n");
        goto ERR_REQUEST_IRQ;
    }

    pr_info("successfully probe edu pci dev\n");

    return 0;

ERR_REQUEST_IRQ:
ERR_DMA_SET_MASK_AND_COHERENT:
    pci_iounmap(pci_dev, base_addr_bar0);
ERR_PCI_IOMAP:
    pci_release_region(pci_dev, 0);
ERR_PCI_REQUEST_REGION:
    pci_disable_device(pci_dev);
ERR_PCI_ENABLE_DEVICE:
    return -1;
}

static void edu_remove(struct pci_dev *pci_dev) {
    pr_info("in edu_remove()...\n");
    free_irq(pci_dev->irq, pci_dev);
    pci_iounmap(pci_dev, base_addr_bar0);
    pci_release_region(pci_dev, 0);
    pci_disable_device(pci_dev);
}

static struct pci_driver edu_driver = {
    .name = "qemu_edu_drv",
    .id_table = pci_id_table,
    .probe = edu_probe,
    .remove = edu_remove
};

int init_mod(void) {
    pr_info("init hlc module...\n");
    int ret = pci_register_driver(&edu_driver);
    if (ret != 0) {
        pr_err("fail to register pci driver\n");
        goto ERR_PCI_REGISTER_DRIVER;
    }

    ret = alloc_chrdev_region(&dev_num, 0, 1, "qemu_edu");
    if (ret != 0) {
        pr_err("fail to alloc chrdev region\n");
        goto ERR_ALLOC_CHRDEV_REGION;
    }

    cdev_init(&edu_cdev, &fops);
    ret = cdev_add(&edu_cdev, dev_num, 1);
    if (ret != 0) {
        pr_err("fail to add cdev\n");
        goto ERR_CDEV_ADD;
    }

    dev_cls = class_create("edu_cls");
    if (IS_ERR(dev_cls)) {
        pr_err("fail to create class\n");
        goto ERR_CLASS_CREATE;
    }

    edu_dev = device_create(dev_cls, NULL, dev_num, NULL, "edu_dev");
    if (IS_ERR(edu_dev)) {
        pr_err("fail to create device\n");
        goto ERR_DEVICE_CREATE;
    }
    return 0;

ERR_DEVICE_CREATE:
    class_destroy(dev_cls);
ERR_CLASS_CREATE:
    cdev_del(&edu_cdev);
ERR_CDEV_ADD:
    unregister_chrdev_region(dev_num, 1);
ERR_ALLOC_CHRDEV_REGION:
ERR_PCI_REGISTER_DRIVER:
    return -1;
}

void exit_mod(void) {
    pr_info("exit hlc module...\n");
    device_destroy(dev_cls, dev_num);
    class_destroy(dev_cls);
    cdev_del(&edu_cdev);
    pci_unregister_driver(&edu_driver);
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");


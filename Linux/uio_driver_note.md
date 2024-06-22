# UIO Driver

## cache

* uio 的目录`/sys/class/uio/uio0`

* umd driver work flow

    1. 内核空间 UIO 设备驱动程序必须在用户空间驱动程序启动之前加载（如果使用模块）
    
    2. 启动用户空间应用程序并打开 UIO 设备文件（/dev/uioX 其中 X 为 0、1、2 ...）

        从用户空间来看，UIO 设备是文件系统中的一个设备节点，就像任何其他设备一样

    3. 设备内存地址信息从相关的sysfs目录中找到，只需要大小
    
    4. 通过调用UIO驱动的mmap()函数将设备内存映射到进程地址空间
    
    5. 应用程序访问设备硬件来控制设备
    
    6. 通过调用 munmap() 取消映射设备内存
    
    7. UIO设备文件关闭

    ref: <https://zhuanlan.zhihu.com/p/556002063>

* uio umd driver example code:

    ```cpp
    #define UIO_SIZE "/sys/class/uio/uio0/maps/map0/size"

    int main( int argc, char ** argv)
    {
        int              uio_fd;
        无符号整数    uio_size;
        文件 *大小_fp;
        无效            *基地址；

        /*
        * 1. 打开 UIO 设备，使其可以使用
        */
        uio_fd = open( "/dev/uio0" , O_RDWR);

        /*
        * 2. 从大小 sysfs 文件中获取内存区域的大小
        *    属性
        */
        size_fp = fopen(UIO_SIZE, O_RDONLY);
        fscanf(size_fp, " 0x%08X " , & uio_size);

        /*
        * 3. 将设备寄存器映射到进程地址空间，以便它们
        *    可直接访问
        */
        base_address =mmap(NULL, uio_size,
                            PROT_READ| PROT_WRITE,
                            MAP_SHARED, uio_fd, 0 );

        // 现在可以访问硬件 ...

        /*
        * 4. 取消映射设备寄存器以完成
        */
        munmap(base_address, uio_size);

        ...
    }
    ```

* uio kmd example:

    ```cpp
    /*

    * This is simple demon of uio driver.

    * Version 1

    *Compile:
    *    Save this file name it simple.c
    *    #echo "obj -m := simple.o" > Makefile
    *    #make -Wall -C /lib/modules/'uname -r'/build M='pwd' modules
    *Load the module:
    *    #modprobe uio
    *    #insmod simple.ko
    */

    #include <linux/module.h>
    #include <linux/platform_device.h>
    #include <linux/uio_driver.h>
    #include <linux/slab.h>


    /*struct uio_info { 
        struct uio_device   *uio_dev; // 在__uio_register_device中初始化
        const char      *name; // 调用__uio_register_device之前必须初始化
        const char      *version; //调用__uio_register_device之前必须初始化
        struct uio_mem      mem[MAX_UIO_MAPS];
        struct uio_port     port[MAX_UIO_PORT_REGIONS];
        long            irq; //分配给uio设备的中断号，调用__uio_register_device之前必须初始化
        unsigned long       irq_flags;// 调用__uio_register_device之前必须初始化
        void            *priv; //
        irqreturn_t (*handler)(int irq, struct uio_info *dev_info); //uio_interrupt中调用，用于中断处理
        
        // 调用__uio_register_device之前必须初始化
        int (*mmap)(struct uio_info *info, struct vm_area_struct *vma); //在uio_mmap中被调用，
                                                            
        // 执行设备打开特定操作
        int (*open)(struct uio_info *info, struct inode *inode);//在uio_open中被调用，执行设备打开特定操作
        int (*release)(struct uio_info *info, struct inode *inode);//在uio_device中被调用，执行设备打开特定操作
        int (*irqcontrol)(struct uio_info *info, s32 irq_on);//在uio_write方法中被调用，执行用户驱动的
        
        //特定操作。
    };*/


    struct uio_info kpart_info = {  
            .name = "kpart",  
            .version = "0.1",  
            .irq = UIO_IRQ_NONE,  
    }; 
    static int drv_kpart_probe(struct device *dev);
    static int drv_kpart_remove(struct device *dev);
    static struct device_driver uio_dummy_driver = {
        .name = "kpart",
        .bus = &platform_bus_type,
        .probe = drv_kpart_probe,
        .remove = drv_kpart_remove,
    };

    static int drv_kpart_probe(struct device *dev)
    {
        printk("drv_kpart_probe(%p)\n",dev);
        kpart_info.mem[0].addr = (unsigned long) kmalloc(1024,GFP_KERNEL);
        
        if(kpart_info.mem[0].addr == 0)
            return -ENOMEM;
        kpart_info.mem[0].memtype = UIO_MEM_LOGICAL;
        kpart_info.mem[0].size = 1024;

        if(uio_register_device(dev,&kpart_info))
            return -ENODEV;
        return 0;
    }

    static int drv_kpart_remove(struct device *dev)
    {
        uio_unregister_device(&kpart_info);
        return 0;
    }

    static struct platform_device * uio_dummy_device;

    static int __init uio_kpart_init(void)
    {
        uio_dummy_device = platform_device_register_simple("kpart",-1,NULL,0);
        return driver_register(&uio_dummy_driver);
    }

    static void __exit uio_kpart_exit(void)
    {
        platform_device_unregister(uio_dummy_device);
        driver_unregister(&uio_dummy_driver);
    }

    module_init(uio_kpart_init);
    module_exit(uio_kpart_exit);

    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("IGB_UIO_TEST");
    MODULE_DESCRIPTION("UIO dummy driver");
    ```

    ref: <https://zhuanlan.zhihu.com/p/555333046>

* uio umd example code

    ```cpp
    #include <stdio.h>
    #include <fcntl.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <errno.h>

    #define UIO_DEV "/dev/uio0"
    #define UIO_ADDR "/sys/class/uio/uio0/maps/map0/addr"
    #define UIO_SIZE "/sys/class/uio/uio0/maps/map0/size"

    static char uio_addr_buf[16]={0};
    static char uio_size_buf[16]={0};

    int main(void)
    {
        int uio_fd,addr_fd,size_fd;
        int uio_size;
        void *uio_addr, *access_address;
        int n=0;
        uio_fd = open(UIO_DEV,O_RDWR);
        addr_fd = open(UIO_ADDR,O_RDONLY);
        size_fd = open(UIO_SIZE,O_RDONLY);
        if(addr_fd < 0 || size_fd < 0 || uio_fd < 0){
            fprintf(stderr,"mmap:%s\n",strerror(errno));
            exit(-1);
        }

        n=read(addr_fd,uio_addr_buf,sizeof(uio_addr_buf));
        if(n<0){
            fprintf(stderr, "%s\n", strerror(errno));
            exit(-1);
        }
        n=read(size_fd,uio_size_buf,sizeof(uio_size_buf));
        if(n<0){
            fprintf(stderr, "%s\n", strerror(errno));
            exit(-1);
        }
        uio_addr = (void*)strtoul(uio_addr_buf,NULL,0);
        uio_size = (int)strtol(uio_size_buf,NULL,0);

        access_address = mmap(NULL,uio_size,PROT_READ | PROT_WRITE,
                                MAP_SHARED,uio_fd,0);
        if(access_address == (void*)-1){
            fprintf(stderr,"mmap:%s\n",strerror(errno));
            exit(-1);
        }

        printf("The device address %p (lenth %d)\n"
            "can be accessed over\n"
            "logical address %p\n",uio_addr,uio_size,access_address);
    /*
        access_address = (void*)(long)mremap(access_address, getpagesize(),uio_size + getpagesize()+ 11111, MAP_SHARED);

        if(access_address == (void*)-1){
            fprintf(stderr,"mremap: %s\n",strerror(errno));
            exit(-1);
        }

        printf(">>>AFTER REMAP:""logical address %p\n",access_address);
    */
        return 0;
    }
    ```

* uio driver

    uio 指的是 user space IO，可以在用户态访问设备内存。

    uio 主要通过 sysfs 和`/dev/uioX`和设备进行交互。

    可能会用到的 struct:

    ```cpp
    /**
    * struct uio_info - UIO device capabilities
    * @uio_dev:		the UIO device this info belongs to
    * @name:		device name
    * @version:		device driver version
    * @mem:		list of mappable memory regions, size==0 for end of list
    * @port:		list of port regions, size==0 for end of list
    * @irq:		interrupt number or UIO_IRQ_CUSTOM
    * @irq_flags:		flags for request_irq()
    * @priv:		optional private data
    * @handler:		the device's irq handler
    * @mmap:		mmap operation for this uio device
    * @open:		open operation for this uio device
    * @release:		release operation for this uio device
    * @irqcontrol:		disable/enable irqs when 0/1 is written to /dev/uioX
    */
    struct uio_info {
        struct uio_device	*uio_dev;
        const char		*name;
        const char		*version;
        struct uio_mem		mem[MAX_UIO_MAPS];
        struct uio_port		port[MAX_UIO_PORT_REGIONS];
        long			irq;
        unsigned long		irq_flags;
        void			*priv;
        irqreturn_t (*handler)(int irq, struct uio_info *dev_info);
        int (*mmap)(struct uio_info *info, struct vm_area_struct *vma);
        int (*open)(struct uio_info *info, struct inode *inode);
        int (*release)(struct uio_info *info, struct inode *inode);
        int (*irqcontrol)(struct uio_info *info, s32 irq_on);
    };
    
    /**
    * struct uio_mem - description of a UIO memory region
    * @name:		name of the memory region for identification
    * @addr:               address of the device's memory rounded to page
    * 			size (phys_addr is used since addr can be
    * 			logical, virtual, or physical & phys_addr_t
    * 			should always be large enough to handle any of
    * 			the address types)
    * @offs:               offset of device memory within the page
    * @size:		size of IO (multiple of page size)
    * @memtype:		type of memory addr points to
    * @internal_addr:	ioremap-ped version of addr, for driver internal use
    * @map:		for use by the UIO core only.
    */
    struct uio_mem {
        const char		*name;
        phys_addr_t		addr;
        unsigned long		offs;
        resource_size_t		size;
        int			memtype;
        void __iomem		*internal_addr;
        struct uio_map		*map;
    };
    
    #define MAX_UIO_MAPS	5
    
    struct uio_portio;
    
    /**
    * struct uio_port - description of a UIO port region
    * @name:		name of the port region for identification
    * @start:		start of port region
    * @size:		size of port region
    * @porttype:		type of port (see UIO_PORT_* below)
    * @portio:		for use by the UIO core only.
    */
    struct uio_port {
        const char		*name;
        unsigned long		start;
        unsigned long		size;
        int			porttype;
        struct uio_portio	*portio;
    };
    
    /* defines for uio_mem->memtype */
    #define UIO_MEM_NONE	0
    #define UIO_MEM_PHYS	1
    #define UIO_MEM_LOGICAL	2
    #define UIO_MEM_VIRTUAL 3
    
    /* defines for uio_port->porttype */
    #define UIO_PORT_NONE	0
    #define UIO_PORT_X86	1
    #define UIO_PORT_GPIO	2
    #define UIO_PORT_OTHER	3
    ```

    kernel module driver code:

    ```cpp
    #include <linux/module.h>
    #include <linux/uio_driver.h>
    #include <linux/fs.h>
    #include <linux/miscdevice.h>
    #include <linux/slab.h>
    #include <linux/mm.h>
    #include <linux/vmalloc.h>
    #include <linux/platform_device.h>
    
    #define DRV_NAME "uio_test"
    #define MEM_SIZE 0x1000
    
    /*
    struct uio_info 是 UIO 驱动中定义设备资源的数据结构，也是 UIO 驱动设备的主要参数之一，包括设备名称、设备版本、中断类型、内存映射等信息。
    下面是 struct uio_info 结构体的具体字段：
        const char* name: 设备名称，字符串类型，比如 "uio_mydevice"
        const char* version: 设备版本，字符串类型，比如 "1.01"
        int irq: 设备使用的中断号，中断类型可以是UIO_IRQ_NONE, UIO_IRQ_EDGE, UIO_IRQ_LEVEL。
        int irq_flags: 中断的处理方法
        struct uio_mem* mem: 包含所需的内存资源的数组，可以是多个内存区域，每个内存块包括物理地址start、大小size、权限memtype等信息。
        int memtype: 位于进程地址空间的内存区域的类型。可以是UIO_MEM_PHYS (物理内存)，UIO_MEM_LOGICAL (逻辑内存)，UIO_MEM_VIRTUAL (虚拟内存)。
        void (*irqcontrol)(struct uio_info*, bool): 指向向文件操作提供中断的函数。
        int (*open)(struct uio_info*, struct inode*): 打开设备的函数。
        int (*release)(struct uio_info*, struct inode*): 关闭设备的函数。
        int (*mmap)(struct uio_info*, struct vm_area_struct*): 内存映射的函数。
        int (*ioctl)(struct uio_info*, unsigned int command, unsigned long argument): 设备控制函数。
        int (*irqhandler)(struct uio_info*, int irqs): 中断处理函数，对于需要驱动处理的中断使用。
    */
    static struct uio_info uio_test = {
        .name = "uio_device",
        .version = "0.0.1",
        .irq = UIO_IRQ_NONE,
    };
    
    static void uio_release(struct device *dev)
    {
        struct uio_device *uio_dev = dev_get_drvdata(dev);
        uio_unregister_device(uio_dev->info);
        kfree(uio_dev);
    }
    
    static int uio_mmap(struct file *filp, struct vm_area_struct * vma)
    {
        /*
        vm_area_struct 结构体的主要成员变量如下：
        vm_start：虚拟内存区域的起始地址。
        vm_end：虚拟内存区域的结束地址。
        vm_next：链表中下一个虚拟内存区域的指针。
        vm_flags：虚拟内存区域的标志，用于指定该区域的访问权限、映射方式等信息。
        vm_page_prot：虚拟内存区域对应的物理内存页的保护属性。
        vm_ops：虚拟内存区域的操作函数指针，用于操作该区域的相关操作。
        vm_file：指向该虚拟内存区域对应的文件对象，如果该内存区域没有对应的文件，则为NULL。
        vm_private_data：指向该虚拟内存区域私有数据的指针，可以用于存储和传递一些附加信息。
        vm_area_struct结构体的主要作用是表示进程的虚拟内存空间，并为操作系统内存管理提供了一些必要的信息，
        如虚拟地址范围、保护属性、映射方式等。它也为进程提供了一些操作虚拟内存的接口，如访问、分配、释放等。在进程创建、分配内存、映射文件等操作时，
        都需要使用vm_area_struct结构体来描述进程虚拟内存的状态。
        */
    
        struct uio_info *info = filp->private_data;
        /*virt_to_page()将虚拟地址转换为一个指向相应页面描述符的指针，并使用page_to_pfn()获取该页面描述符对应的页框号*/
        unsigned long pfn = page_to_pfn(virt_to_page(info->mem[0].addr));
        /*PFN_PHYS()将页框号转换为相应的物理地址*/
        unsigned long phys = PFN_PHYS(pfn);
        /*uio_info结构体中第一个内存区域的大小*/
        unsigned long size = info->mem[0].size;
    
        /*
        remap_pfn_range函数用于将一段物理地址空间映射到进程的虚拟地址空间，并返回映射后的虚拟地址。
        vma 是 vm_area_struct 结构体指针，表示进程的一段虚拟地址空间。
        vma->vm_start 表示用户空间地址的起始地址。
        phys >> PAGE_SHIFT 表示设备地址的起始页号。PAGE_SHIFT 表示的是系统页面大小的偏移量，通常为12位（2 ^ 12 = 4KB）。
        这是因为物理地址的低 12 位表示页面的偏移量，需要去除才能得到页面的编号。 
        size 表示映射空间的大小。
        vma->vm_page_prot 表示页保护标志，具体指定对应页的访问权限。
        */
        if (remap_pfn_range(vma, vma->vm_start, phys >> PAGE_SHIFT, size, vma->vm_page_prot)) {
            return -EAGAIN;
        }
    
        return 0;
    }
    
    static const struct file_operations uio_fops = {
        .owner = THIS_MODULE,
        .mmap = uio_mmap,
    };
    
    char test_arr[PAGE_SIZE] = {0};
    
    static ssize_t get_uio_info(struct device *dev, struct device_attribute *attr, char *buf)
    {
        return snprintf(buf, PAGE_SIZE, "%s\n", test_arr);
    }
    
    static ssize_t set_uio_info(struct device *dev, struct device_attribute *devattr, const char *buf, size_t count)
    {
        snprintf(test_arr, PAGE_SIZE, "%s\n", buf);
        return count;
    }
    
    static DEVICE_ATTR(uio_info, 0600, get_uio_info, set_uio_info);
    
    static struct attribute *uio_sysfs_attrs[] = {
        &dev_attr_uio_info.attr,
        NULL,
    };
    
    static struct attribute_group uio_attr_group = {
        .attrs = uio_sysfs_attrs,
    };
    
    static int uio_probe(struct platform_device *pdev)
    {
        struct uio_device *uio_dev;
        /*
        uio_device结构体是Linux内核中的一个结构体，用于表示用户空间IO设备
        struct uio_device {
        struct device dev; // 继承自struct device，表示内核中的设备
        struct uio_info *info; // 表示uio设备的信息
        struct list_head list; // 用于将uio_device结构体连接到uio设备链表中
        struct module *owner; // 表示该设备所属的内核模块
        int minor; // 表示uio设备的次设备号
        struct cdev cdev; // 表示该设备的字符设备描述符
        struct class *class; // 表示该设备所属的类别
        unsigned int event; // 表示该设备的事件标志
        int irq; // 表示该设备的中断号
        };
        */
        int err;
        void *p;
    
        uio_dev = kzalloc(sizeof(struct uio_device), GFP_KERNEL);
        if (uio_dev == NULL) {
            return -ENOMEM;
        }
    
        p = kmalloc(MEM_SIZE, GFP_KERNEL);
        strcpy(p, "123456");
        uio_test.mem[0].name = "uio_mem",
        uio_test.mem[0].addr = (unsigned long)p;
        uio_test.mem[0].memtype = UIO_MEM_LOGICAL;
        uio_test.mem[0].size = MEM_SIZE;
        uio_dev->info = &uio_test;
        uio_dev->dev.parent = &pdev->dev;
    
        err = uio_register_device(&pdev->dev, uio_dev->info);
        if (err) {
            kfree(uio_dev);
            return err;
        }
        if (sysfs_create_group(&pdev->dev.kobj, &uio_attr_group)) {
            printk(KERN_ERR "Cannot create sysfs for system uio\n");
            return err;
        }
    
        //dev_set_drvdata(pdev, uio_dev);
    
        return 0;
    }
    
    static int uio_remove(struct platform_device *pdev)
    {
        struct uio_device *uio_dev = platform_get_drvdata(pdev);
    
        sysfs_remove_group(&uio_dev->dev.kobj, &uio_attr_group);
        uio_unregister_device(uio_dev->info);
        //dev_set_drvdata(uio_dev, NULL);
        kfree(uio_dev);
    
        return 0;
    }
    
    static struct platform_device *uio_test_dev;
    static struct platform_driver uio_driver = {
        .probe = uio_probe,
        .remove = uio_remove,
        .driver = {
            .name = DRV_NAME,
        },
    
    };
    
    static int __init uio_init(void)
    {
        uio_test_dev = platform_device_register_simple(DRV_NAME, -1, NULL, 0);
        return platform_driver_register(&uio_driver);
    }
    
    static void __exit uio_exit(void)
    {
        platform_device_unregister(uio_test_dev);
        platform_driver_unregister(&uio_driver);
    }
    
    module_init(uio_init);
    module_exit(uio_exit);
    
    MODULE_AUTHOR("Arron Wu");
    MODULE_DESCRIPTION("UIO driver");
    MODULE_LICENSE("GPL");
    ```

    app code:

    ```cpp
    #include <stdio.h>
    #include <fcntl.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <errno.h>
    #include <string.h>
    
    #define UIO_DEV "/dev/uio0"  
    #define UIO_ADDR "/sys/class/uio/uio0/maps/map0/addr"  
    #define UIO_SIZE "/sys/class/uio/uio0/maps/map0/size"  
    
    static char uio_addr_buf[16], uio_size_buf[16];
    
    int main(void)
    {
        int uio_size;
        void* uio_addr, *access_address;
        
        int uio_fd = open(UIO_DEV, O_RDWR);
        int addr_fd = open(UIO_ADDR, O_RDONLY);
        int size_fd = open(UIO_SIZE, O_RDONLY);
        
        if( addr_fd < 0 || size_fd < 0 || uio_fd < 0) {  
            fprintf(stderr, "mmap: %s\n", strerror(errno));  
            exit(-1);  
        }
        read(addr_fd, uio_addr_buf, sizeof(uio_addr_buf));
        read(size_fd, uio_size_buf, sizeof(uio_size_buf));
        uio_addr = (void*)strtoul(uio_addr_buf, NULL, 0);
        uio_size = (int)strtol(uio_size_buf, NULL, 0);
        
        access_address = mmap(NULL, uio_size, PROT_READ | PROT_WRITE, MAP_SHARED, uio_fd, 0);
        if ( access_address == (void*) -1) {
            printf("mmap: %s\n", strerror(errno));
            exit(-1);
        }
        printf("The device address %p (lenth %d)\n" "logical address %p\n", uio_addr, uio_size, access_address);
        for(int i = 0; i<6; i++) {
            printf("%c", ((char *)access_address)[i]);
            ((char *)access_address)[i] += 1;
        }
        printf("\n");
        
        for(int i = 0; i<6; i++)
            printf("%c", ((char *)access_address)[i]);
        printf("\n");
        
        munmap(access_address, uio_size);
        return 0;
    }  
    ```

    ref: <https://blog.csdn.net/weixin_38452632/article/details/130947993>

* uio 的 kmd 主要做两件事

    1. 分配和记录设备需要的资源和注册UIO设备

    2. 实现必须在内核空间实现的中断处理函数

    其余的大部分操作都放到 umd 里实现。
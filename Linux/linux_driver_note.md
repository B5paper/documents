# Linux Driver Note

Ref:

* <https://embetronicx.com/tutorials/linux/device-drivers/>

## cache

* `request_irq()`

    header: `#include <linux/interrupt.h>`

* `struct work_struct`

    header: `#include <linux/workqueue.h>`

* `__this_cpu_write()`

    header: `asm/hw_irq.h`

* 一个可用的`INIT_WORK()`的代码，见`ref_17`

    测试：

    ```bash
    make
    sudo insmod wque.ko
    sudo cat /dev/hlc_dev
    ```

    `dmesg` output:

    ```
    [25350.311799] init hlc module done.
    [25366.856255] in m_open()...
    [25366.856262] in m_read()...
    [25366.856265] in irq_handler()...
    [25366.856272] in m_release()...
    [25366.856312] in work_queue_fn()...
    [25414.842921] in exit_mod()...
    [25414.843190] exit hlc module done.
    ```

    explanation:

    1. 在创建`work_struct`对象的时候，需要我们自己申请内存，要么就直接创建全局变量，不能只创建一个指针。

        代码中使用`struct work_struct work_item;`创建了个全局对象。

        这一点和`class_create()`，`device_create()`挺不一样的，这两个函数都是只返回指针，内存由操作系统管理。

    2. `INIT_WORK()`需要将`work_struct`的指针传进去：
    
        `INIT_WORK(&work_item, work_queue_fn);`

    3. `schedule_work()`传的也是指针：

        `schedule_work(&work_item);`

    4. 这份代码不包含函数返回值检测和异常处理，所以比较简洁。

* work queue 的一个 example

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include<linux/slab.h>                 //kmalloc()
    #include<linux/uaccess.h>              //copy_to/from_user()
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/interrupt.h>
    #include <asm/io.h>
    #include <linux/workqueue.h>            // Required for workqueues
    #include <linux/err.h>
    #include <asm/hw_irq.h>

    #define IRQ_NO 11


    void workqueue_fn(struct work_struct *work); 

    /*Creating work by Static Method */
    DECLARE_WORK(workqueue,workqueue_fn);

    /*Workqueue Function*/
    void workqueue_fn(struct work_struct *work)
    {
        printk(KERN_INFO "Executing Workqueue Function\n");
    }


    //Interrupt handler for IRQ 11. 
    static irqreturn_t irq_handler(int irq,void *dev_id) {
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    bool ret = schedule_work(&workqueue);
    if (!ret)
    {
        pr_info("fail to schedule work\n");
    }
    else
    {
        pr_info("successfully schedule work\n");
    }
        
    return IRQ_HANDLED;
    }


    volatile int etx_value = 0;


    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;

    /*
    ** Function Prototypes
    */
    static int __init etx_driver_init(void);
    static void __exit etx_driver_exit(void);

    /*************** Driver Fuctions **********************/
    static int etx_open(struct inode *inode, struct file *file);
    static int etx_release(struct inode *inode, struct file *file);
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len,loff_t * off);
    static ssize_t etx_write(struct file *filp, 
                    const char *buf, size_t len, loff_t * off);

    /*************** Sysfs Fuctions **********************/
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf);
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count);

    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
    };

    /*
    ** This function will be called when we read the sysfs file
    */ 
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
    }

    /*
    ** This function will be called when we write the sysfsfs file
    */
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
    }

    /*
    ** This function will be called when we open the Device file
    */  
    static int etx_open(struct inode *inode, struct file *file)
    {
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */  
    static int etx_release(struct inode *inode, struct file *file)
    {
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read function\n");
        struct irq_desc *desc;
        desc = irq_to_desc(11);
        if (!desc)
                return -EINVAL;
        __this_cpu_write(vector_irq[59], desc);
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
        printk(KERN_INFO "Write Function\n");
        return len;
    }

    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create("etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
        }

        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
        }

        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
        }
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;

    irq:
        free_irq(IRQ_NO,(void *)(irq_handler));

    r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);

    r_device:
        class_destroy(dev_class);
    r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
    }

    /*
    ** Module exit function
    */

    static void __exit etx_driver_exit(void)
    {
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
    }

    module_init(etx_driver_init);
    module_exit(etx_driver_exit);

    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Static method)");
    MODULE_VERSION("1.10");
    ```

    需要的头文件：

    `#include <linux/workqueue.h>`

    ```c
    void workqueue_fn(struct work_struct *work); 
    DECLARE_WORK(workqueue,workqueue_fn);
    ```

    用宏创建一个变量`workqueue`，让其与一个函数产生关联。

    `schedule_work(&workqueue);`

    让`workqueue`对应的线程函数从睡眠状态唤醒，并放入主队列进行执行。

    `dmesg` output:

    ```
    [ 1789.389643] Major = 240 Minor = 0
    [ 1789.390326] Device Driver Insert...Done!!!
    [ 1802.603002] Device File Opened...!!!
    [ 1802.603029] Read function
    [ 1802.603040] Shared IRQ: Interrupt Occurred
    [ 1802.603048] successfully schedule work
    [ 1802.603058] Executing Workqueue Function
    [ 1802.603085] Device File Closed...!!!
    ```

    可以看到，可以通过 work queue，控制线程的休眠和唤醒。

    疑问：

    1. 当 work queue 对应的函数执行完毕时，是否线程会自动进入休眠？

        猜想：应该会自动进入休眠。不然函数执行完了也没有其他什么事可做。

        不过还有一种可能，即线程死亡，每次 schedule work 都重新创建一个新的线程。

        休眠-唤醒，和死亡-重建，哪个是对的？

    2. work queue 和 wait event 有什么不同？

        work queue 可以用`schedule_work()`唤醒（或新建）一个线程，wait event 可以使用`wake_up()`唤醒一个线程，这两都有什么不一样？

* vscode 中有时 tab 会变成 8 个空格，可以关闭这个设置：

    `Editor: Detect Indentation`

* `flush_work()`可以阻塞等待指定的 work，直到 work 完成。

    syntax:

    `int flush_work( struct work_struct *work );`

    `flush_scheduled_work()`可以等待全局共享的 work queue 完成。

    example: 见`ref_16`

    执行`cat /dev/etx_device`后，可以看到`dmesg`的输出：

    ```
    [ 6195.143524] Major = 240 Minor = 0 
    [ 6195.143595] Device Driver Insert...Done!!!
    [ 6214.544080] Device File Opened...!!!
    [ 6214.544095] Read function
    [ 6214.544101] Shared IRQ: Interrupt Occurred
    [ 6214.544104] successfully schedule work
    [ 6214.544105] block flush scheduled work
    [ 6214.544107] BUG: scheduling while atomic: cat/12786/0x00010001
    [ 6214.544110] Modules linked in: hello(OE) tls(E) intel_rapl_msr(E) intel_rapl_common(E) intel_uncore_frequency_common(E) snd_intel8x0(E) binfmt_misc(E) snd_ac97_codec(E) ac97_bus(E) crct10dif_pclmul(E) polyval_clmulni(E) polyval_generic(E) ghash_clmulni_intel(E) nls_iso8859_1(E) snd_pcm(E) sha256_ssse3(E) sha1_ssse3(E) aesni_intel(E) crypto_simd(E) cryptd(E) joydev(E) snd_seq_midi(E) rapl(E) snd_seq_midi_event(E) snd_rawmidi(E) input_leds(E) vmwgfx(E) snd_seq(E) serio_raw(E) drm_ttm_helper(E) snd_seq_device(E) snd_timer(E) snd(E) ttm(E) soundcore(E) drm_kms_helper(E) vboxguest(E) mac_hid(E) sch_fq_codel(E) msr(E) parport_pc(E) ppdev(E) drm(E) lp(E) parport(E) efi_pstore(E) ip_tables(E) x_tables(E) autofs4(E) hid_generic(E) usbhid(E) crc32_pclmul(E) hid(E) psmouse(E) ahci(E) libahci(E) i2c_piix4(E) e1000(E) pata_acpi(E) video(E) wmi(E) [last unloaded: hello(OE)]
    [ 6214.544151] CPU: 2 PID: 12786 Comm: cat Tainted: G        W  OE      6.5.13 #4
    [ 6214.544154] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
    [ 6214.544155] Call Trace:
    [ 6214.544157]  <IRQ>
    [ 6214.544159]  dump_stack_lvl+0x48/0x70
    [ 6214.544165]  dump_stack+0x10/0x20
    [ 6214.544166]  __schedule_bug+0x64/0x80
    [ 6214.544169]  __schedule+0x100c/0x15f0
    [ 6214.544174]  schedule+0x68/0x110
    [ 6214.544176]  schedule_timeout+0x151/0x160
    [ 6214.544181]  __wait_for_common+0x92/0x190
    [ 6214.544183]  ? __pfx_schedule_timeout+0x10/0x10
    [ 6214.544185]  wait_for_completion+0x24/0x40
    [ 6214.544188]  __flush_workqueue+0x133/0x3e0
    [ 6214.544190]  ? vprintk_default+0x1d/0x30
    [ 6214.544194]  irq_handler+0x55/0x80 [hello]
    [ 6214.544199]  __handle_irq_event_percpu+0x4f/0x1b0
    [ 6214.544201]  handle_irq_event+0x39/0x80
    [ 6214.544204]  handle_edge_irq+0x8c/0x250
    [ 6214.544207]  __common_interrupt+0x52/0x110
    [ 6214.544209]  common_interrupt+0x9f/0xb0
    [ 6214.544212]  </IRQ>
    [ 6214.544212]  <TASK>
    [ 6214.544213]  asm_common_interrupt+0x27/0x40
    [ 6214.544217] RIP: 0010:etx_read+0x2e/0x50 [hello]
    [ 6214.544221] Code: 00 55 48 c7 c7 db 12 99 c0 48 89 e5 e8 0b 90 08 d8 bf 0b 00 00 00 e8 61 fa 08 d8 48 85 c0 74 14 65 48 89 05 fc de 70 3f cd 3b <31> c0 5d 31 ff c3 cc cc cc cc 48 c7 c0 ea ff ff ff 5d 31 ff c3 cc
    [ 6214.544223] RSP: 0018:ffffacfb01fe7d98 EFLAGS: 00000286
    [ 6214.544225] RAX: ffff8a8b00205200 RBX: 0000000000020000 RCX: 0000000000000000
    [ 6214.544226] RDX: 0000000000000000 RSI: 0000000000000000 RDI: 0000000000000000
    [ 6214.544227] RBP: ffffacfb01fe7d98 R08: 0000000000000000 R09: 0000000000000000
    [ 6214.544228] R10: 0000000000000000 R11: 0000000000000000 R12: 0000000000000000
    [ 6214.544229] R13: ffff8a8a52457700 R14: ffffacfb01fe7e50 R15: 000077e41adfe000
    [ 6214.544232]  ? etx_read+0x1f/0x50 [hello]
    [ 6214.544235]  vfs_read+0xb4/0x320
    [ 6214.544238]  ? __handle_mm_fault+0xb88/0xc70
    [ 6214.544242]  ksys_read+0x67/0xf0
    [ 6214.544244]  __x64_sys_read+0x19/0x30
    [ 6214.544245]  x64_sys_call+0x192c/0x2570
    [ 6214.544248]  do_syscall_64+0x56/0x90
    [ 6214.544250]  ? exit_to_user_mode_prepare+0x39/0x190
    [ 6214.544253]  ? irqentry_exit_to_user_mode+0x17/0x20
    [ 6214.544255]  ? irqentry_exit+0x43/0x50
    [ 6214.544256]  ? exc_page_fault+0x95/0x1b0
    [ 6214.544259]  entry_SYSCALL_64_after_hwframe+0x73/0xdd
    [ 6214.544261] RIP: 0033:0x77e41ab147e2
    [ 6214.544263] Code: c0 e9 b2 fe ff ff 50 48 8d 3d 8a b4 0c 00 e8 a5 1d 02 00 0f 1f 44 00 00 f3 0f 1e fa 64 8b 04 25 18 00 00 00 85 c0 75 10 0f 05 <48> 3d 00 f0 ff ff 77 56 c3 0f 1f 44 00 00 48 83 ec 28 48 89 54 24
    [ 6214.544264] RSP: 002b:00007ffdc7545928 EFLAGS: 00000246 ORIG_RAX: 0000000000000000
    [ 6214.544266] RAX: ffffffffffffffda RBX: 0000000000020000 RCX: 000077e41ab147e2
    [ 6214.544267] RDX: 0000000000020000 RSI: 000077e41adfe000 RDI: 0000000000000003
    [ 6214.544268] RBP: 000077e41adfe000 R08: 000077e41adfd010 R09: 000077e41adfd010
    [ 6214.544269] R10: 0000000000000022 R11: 0000000000000246 R12: 0000000000022000
    [ 6214.544270] R13: 0000000000000003 R14: 0000000000020000 R15: 0000000000020000
    [ 6214.544272]  </TASK>
    [ 6214.544277] Executing Workqueue Function
    [ 6214.544278] i: 1
    [ 6214.544278] i: 2
    [ 6214.544279] i: 3
    [ 6214.544279] i: 4
    [ 6214.544279] i: 5
    [ 6214.544280] i: 6
    [ 6214.544280] i: 7
    [ 6214.544281] i: 8
    [ 6214.544281] i: 9
    [ 6214.544282] i: 10
    [ 6214.544287] work queue flushed!
    [ 6214.544304] cat[12786]: segfault at 77e41aa89210 ip 000077e41aa89210 sp 00007ffdc7545a68 error 14 in libc.so.6[77e41aa28000+195000] likely on CPU 2 (core 2, socket 0)
    [ 6214.544313] Code: Unable to access opcode bytes at 0x77e41aa891e6.
    [ 6214.544325] BUG: scheduling while atomic: cat/12786/0x7fff0001
    [ 6214.544326] Modules linked in: hello(OE) tls(E) intel_rapl_msr(E) intel_rapl_common(E) intel_uncore_frequency_common(E) snd_intel8x0(E) binfmt_misc(E) snd_ac97_codec(E) ac97_bus(E) crct10dif_pclmul(E) polyval_clmulni(E) polyval_generic(E) ghash_clmulni_intel(E) nls_iso8859_1(E) snd_pcm(E) sha256_ssse3(E) sha1_ssse3(E) aesni_intel(E) crypto_simd(E) cryptd(E) joydev(E) snd_seq_midi(E) rapl(E) snd_seq_midi_event(E) snd_rawmidi(E) input_leds(E) vmwgfx(E) snd_seq(E) serio_raw(E) drm_ttm_helper(E) snd_seq_device(E) snd_timer(E) snd(E) ttm(E) soundcore(E) drm_kms_helper(E) vboxguest(E) mac_hid(E) sch_fq_codel(E) msr(E) parport_pc(E) ppdev(E) drm(E) lp(E) parport(E) efi_pstore(E) ip_tables(E) x_tables(E) autofs4(E) hid_generic(E) usbhid(E) crc32_pclmul(E) hid(E) psmouse(E) ahci(E) libahci(E) i2c_piix4(E) e1000(E) pata_acpi(E) video(E) wmi(E) [last unloaded: hello(OE)]
    [ 6214.544347] CPU: 2 PID: 12786 Comm: cat Tainted: G        W  OE      6.5.13 #4
    [ 6214.544349] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
    [ 6214.544349] Call Trace:
    [ 6214.544350]  <TASK>
    [ 6214.544351]  dump_stack_lvl+0x48/0x70
    [ 6214.544353]  dump_stack+0x10/0x20
    [ 6214.544354]  __schedule_bug+0x64/0x80
    [ 6214.544355]  __schedule+0x100c/0x15f0
    [ 6214.544357]  schedule+0x68/0x110
    [ 6214.544359]  schedule_timeout+0x151/0x160
    [ 6214.544361]  __wait_for_common+0x92/0x190
    [ 6214.544363]  ? __pfx_schedule_timeout+0x10/0x10
    [ 6214.544364]  wait_for_completion_state+0x21/0x50
    [ 6214.544366]  call_usermodehelper_exec+0x188/0x1c0
    [ 6214.544370]  do_coredump+0xa35/0x1680
    [ 6214.544374]  ? do_dec_rlimit_put_ucounts+0x6b/0xd0
    [ 6214.544377]  get_signal+0x97b/0xae0
    [ 6214.544379]  arch_do_signal_or_restart+0x2f/0x270
    [ 6214.544381]  ? __bad_area_nosemaphore+0x147/0x2e0
    [ 6214.544384]  exit_to_user_mode_prepare+0x11b/0x190
    [ 6214.544386]  irqentry_exit_to_user_mode+0x9/0x20
    [ 6214.544387]  irqentry_exit+0x43/0x50
    [ 6214.544388]  exc_page_fault+0x95/0x1b0
    [ 6214.544390]  asm_exc_page_fault+0x27/0x30
    [ 6214.544392] RIP: 0033:0x77e41aa89210
    [ 6214.544395] Code: Unable to access opcode bytes at 0x77e41aa891e6.
    [ 6214.544395] RSP: 002b:00007ffdc7545a68 EFLAGS: 00010246
    [ 6214.544396] RAX: 000077e41ac1b868 RBX: 0000000000000000 RCX: 0000000000000004
    [ 6214.544397] RDX: 0000000000000001 RSI: 0000000000000000 RDI: 000077e41ac1b780
    [ 6214.544398] RBP: 000077e41ac1b780 R08: 000077e41adfd000 R09: 000077e41adfd010
    [ 6214.544399] R10: 0000000000000022 R11: 0000000000000246 R12: 000077e41ac1a838
    [ 6214.544400] R13: 0000000000000000 R14: 000077e41ac1bee8 R15: 000077e41ac1bf00
    [ 6214.544401]  </TASK>
    [ 6214.552103] Device File Closed...!!!
    ```

    虽然有一些报错输出，但是可以看到这一段还是按照顺序执行的：

    ```
    [ 6214.544277] Executing Workqueue Function
    [ 6214.544278] i: 1
    [ 6214.544278] i: 2
    [ 6214.544279] i: 3
    [ 6214.544279] i: 4
    [ 6214.544279] i: 5
    [ 6214.544280] i: 6
    [ 6214.544280] i: 7
    [ 6214.544281] i: 8
    [ 6214.544281] i: 9
    [ 6214.544282] i: 10
    [ 6214.544287] work queue flushed!
    ```

    由于从 1 到 10 没有中断，所以确实发生了等待。

    将代码中的`flush_scheduled_work();`換成`flush_work(&workqueue);`后，同样适用。

    ref:

    1. <http://juniorprincewang.github.io/2018/11/20/Linux%E8%AE%BE%E5%A4%87%E9%A9%B1%E5%8A%A8%E4%B9%8Bworkqueue/>

    2. <https://embetronicx.com/tutorials/linux/device-drivers/workqueue-in-linux-kernel/>

* 对于新版内核，由于我们不知道第一个 irq vector 的地址，所以只能重新编译内核

    `intrp.c`:

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include<linux/slab.h>                 //kmalloc()
    #include<linux/uaccess.h>              //copy_to/from_user()
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/interrupt.h>
    #include <linux/irqnr.h>
    #include <asm/io.h>
    #include <linux/err.h>
    #include <asm/hw_irq.h>
    #define IRQ_NO 11
    //Interrupt handler for IRQ 11. 
    static irqreturn_t irq_handler(int irq,void *dev_id) {
            printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
            return IRQ_HANDLED;
    }
    volatile int etx_value = 0;
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;
    
    static int __init etx_driver_init(void);
    static void __exit etx_driver_exit(void);
    /*************** Driver Fuctions **********************/
    static int etx_open(struct inode *inode, struct file *file);
    static int etx_release(struct inode *inode, struct file *file);
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len,loff_t * off);
    static ssize_t etx_write(struct file *filp, 
                    const char *buf, size_t len, loff_t * off);
    /*************** Sysfs Fuctions **********************/
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf);
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count);
    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
    
    static struct file_operations fops =
    {
            .owner          = THIS_MODULE,
            .read           = etx_read,
            .write          = etx_write,
            .open           = etx_open,
            .release        = etx_release,
    };
    
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
            printk(KERN_INFO "Sysfs - Read!!!\n");
            return sprintf(buf, "%d", etx_value);
    }
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            printk(KERN_INFO "Sysfs - Write!!!\n");
            sscanf(buf,"%d",&etx_value);
            return count;
    }
    static int etx_open(struct inode *inode, struct file *file)
    {
            printk(KERN_INFO "Device File Opened...!!!\n");
            return 0;
    }
    
    static int etx_release(struct inode *inode, struct file *file)
    {
            printk(KERN_INFO "Device File Closed...!!!\n");
            return 0;
    }

    // extern struct irq_desc* vector_irq;
    
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
         printk(KERN_INFO "Read function\n");
        struct irq_desc *desc;
        desc = irq_to_desc(11);
        if (!desc)
        {
                return -EINVAL;
        }
        __this_cpu_write(vector_irq[59], desc);
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
    }
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
            printk(KERN_INFO "Write Function\n");
            return len;
    }

    
    static int __init etx_driver_init(void)
    {
            /*Allocating Major number*/
            if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                    printk(KERN_INFO "Cannot allocate major number\n");
                    return -1;
            }
            printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
    
            /*Creating cdev structure*/
            cdev_init(&etx_cdev,&fops);
    
            /*Adding character device to the system*/
            if((cdev_add(&etx_cdev,dev,1)) < 0){
                printk(KERN_INFO "Cannot add the device to the system\n");
                goto r_class;
            }
    
            /*Creating struct class*/
            if(IS_ERR(dev_class = class_create("etx_class"))){
                printk(KERN_INFO "Cannot create the struct class\n");
                goto r_class;
            }
    
            /*Creating device*/
            if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
                printk(KERN_INFO "Cannot create the Device 1\n");
                goto r_device;
            }
    
            /*Creating a directory in /sys/kernel/ */
            kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
            /*Creating sysfs file for etx_value*/
            if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                    printk(KERN_INFO"Cannot create sysfs file......\n");
                    goto r_sysfs;
            }
            if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
                printk(KERN_INFO "my_device: cannot register IRQ ");
                        goto irq;
            }
            printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
    irq:
            free_irq(IRQ_NO,(void *)(irq_handler));
    r_sysfs:
            kobject_put(kobj_ref); 
            sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    
    r_device:
            class_destroy(dev_class);
    r_class:
            unregister_chrdev_region(dev,1);
            cdev_del(&etx_cdev);
            return -1;
    }
    
    static void __exit etx_driver_exit(void)
    {
            free_irq(IRQ_NO,(void *)(irq_handler));
            kobject_put(kobj_ref); 
            sysfs_remove_file(kernel_kobj, &etx_attr.attr);
            device_destroy(dev_class,dev);
            class_destroy(dev_class);
            cdev_del(&etx_cdev);
            unregister_chrdev_region(dev, 1);
            printk(KERN_INFO "Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    MODULE_LICENSE("GPL");
    ```

    其中`#include <linux/irqnr.h>`用于提供`irq_to_desc()`函数的声明。

    如果此时直接编译，会看到 undefined symbol 的 error 输出。这时我们需要重新编译内核。

    首先是下载内核，可以直接使用`apt-get download linux-source`，这个命令下载的内核版本不一定是最新版，可能是刚装系统时的内核版本。
    
    还可以搜索目前可用的版本：

    `apt-cache search linux-source`

    ref: <https://askubuntu.com/questions/159833/how-do-i-get-the-kernel-source-code>

    源码会被下载到`/usr/src`中，分别是一个`.tar.bz2`的压缩包的 symbolic link，和一个已经解压缩的目录，比如`linux-source-6.5.0`。

    我们直接对这个`.tar.bz`的文件解压缩，就能在`/usr/src`下得到内核源码。

    进入目录后，执行`cp -v /boot/config-$(uname -r) .confi`（可能是备份一下配置文件？）

    编译内核前还需要安装几个依赖库：

    `apt-get install bison flex libssl-dev libelf-dev`

    如果安装得不够，在 make 的时候会报错，按照提示安装就可以了。

    接下来找到`arch/x86/kernel/irq.c`文件，在最后添加一行：

    ```c
    EXPORT_SYMBOL (vector_irq);
    ```

    然后找到`kernel/irq/irqdesc.c`文件，注意这个文件中有两个`irq_to_desc()`函数的定义，我们在 379 行附近找到一个，添加上 export symbol:

    ```c
    struct irq_desc *irq_to_desc(unsigned int irq) 
    {
            return mtree_load(&sparse_irqs, irq);
    }
    EXPORT_SYMBOL(irq_to_desc);
    ```

    另外一个定义在 609 行附近，这个函数已经被 export symbol 过了，我们就不用管了。

    这两个函数定义是被宏控制的，实际编译的时候根据`#ifdef`之类的命令，只会生效一次。由于不知道具体是哪个生效，所以直接把两个处定义都 export 了。

    接下来执行：

    ```bash
    make oldconfig
    make menuconfig
    ```

    基本什么都不用改，保存后退出就可以了。

    然后`sudo vim .config`，把 system tructed keys 之类的清空，不然一会编译会报错：

    ```conf
    CONFIG_SYSTEM_TRUSTED_KEYRING=y
    CONFIG_SYSTEM_TRUSTED_KEYS=""
    ```

    ref:
    
    1. <https://blog.csdn.net/m0_47696151/article/details/121574718>

    2. <https://blog.csdn.net/qq_36393978/article/details/118157426>

    接下来就可以开始编译了：

    ```bash
    sudo make -j4
    ```

    4 线程编译大概要花 20 多分钟。

    编译好后执行：
    
    ```bash
    sudo make modules_install
    sudo make install
    ```

    然后更新引导：

    ```bash
    sudo update-initramfs -c -k 6.5.0
    sudo update-grub
    ```

    这里的`6.5.0`将来会变成`uname -r`的输出。

    最后重启系统：`reboot`，就大功告成了。

    接下来我们正常编译 kernel module，然后`insmode`，再进入`/dev`目录下，执行测试命令：

    ```bash
    sudo cat /dev/etx_device
    ```

    此时可以看到`dmesg` output:

    ```
    [   39.678202] intrp: loading out-of-tree module taints kernel.
    [   39.678390] Major = 240 Minor = 0
    [   39.678709] Device Driver Insert...Done!!!
    [   79.901307] Device File Opened...!!!
    [   79.901314] Read function
    [   79.901317] Shared IRQ: Interrupt Occurred
    [   79.901322] Device File Closed...!!!
    ```

    中断触发成功。

* kernel module 编译时出现 undefine symbol 是因为没有 export symbol

    ref: <https://blog.csdn.net/choumin/article/details/127094429>

* kbuild　添加自定义的 .o　文件

    ```makefile
    obj-m += haha.o
    haha-src := my_proc.c
    haha-objs := my_proc.o relative/path/to/hehe.o
    ```

    ref: <https://stackoverflow.com/questions/22150812/linking-kernel-module-with-a-static-lib>

    注意，在`xxx-objs`中使用的路径，都是相对于当前目录的路径。

* kbuild doc

    <https://docs.kernel.org/kbuild/makefiles.html>

* kbuild add extra flags to compiler

    <https://stackoverflow.com/questions/54118602/how-to-set-preprocessor-directives-in-makefile-for-kernel-module-build-target>

* 一个可用的 irq 软中断程序，见`ref_15`

    在编译完，`insmod`之后，可以使用`sudo cat /dev/etx_device`触发中断，然后可以看到`dmesg`里显示：

    ```
    [12575.759721] intrp: loading out-of-tree module taints kernel.
    [12575.759724] intrp: module verification failed: signature and/or required key missing - tainting kernel
    [12575.760032] Major = 240 Minor = 0 
    [12575.760356] Device Driver Insert...Done!!!
    [12715.415083] Device File Opened...!!!
    [12715.415103] Read function
    [12715.415107] __common_interrupt: 1.59 No irq handler for vector
    [12715.415119] Device File Closed...!!!
    ```

    11 号中断是保留中断，没有默认用途，因此用户可以去自定义。

    代码中比较难理解的是`asm("int $0x3B");  // Corresponding to irq 11`这一句。

    我们可以打开`/usr/src/linux-headers-6.5.0-28-generic/arch/x86/include/asm/irq_vectors.h`文件，查到

    `#define FIRST_EXTERNAL_VECTOR           0x20`

    不清楚`#define IRQ0_VECTOR (FIRST_EXTERNAL_VECTOR + 0x10)`这一步是怎么来的。

    最后还需要加上我们的中断号`11`，即`0x20 + 0x10 + 11 = 0x3B`，

    这诚是`asm("int $0x3B");`的由来。

* typical IRQ assignments for a PC

    | IRQ number | Device |
    | - | - |
    | 0 | System timer |
    | 1 | Keyboard (PS/2) |
    | 2 | Cascade from IRQ 9 |
    | 3 | COM port 2 or 4 |
    | 4 | COM port 1 or 3 |
    | 5 | Parallel (printer) port 2 or sound cards |
    | 6 | Floppy drive controller |
    | 7 | Parallel (printer) port 1 |
    | 8 | Real-time clock |
    | 9 | Video |
    | 10 | Open |
    | 11 | Open |
    | 12 | Mouse (PS/2) |
    | 13 | Coprocessor |
    | 14 | Primary IDE controller (hard drives) |
    | 15 | Secondary IDE controller (hard drives) |

    ref: <https://www.techtarget.com/whatis/definition/IRQ-interrupt-request>

* 页表用于将虚拟地址映射到物理地址

    首先将物理地址按照 4 KB 进行分区，然后对每个小区间进行编号。

    目前的 linux 指针是 8 个字节，即 64 位。但是虚拟地址并不需要 64 位，只用到了其中的 48 位。

    其中 4 KB = 4 * 1024 Bytes = 2^2 * 2^10 Bytes = 2^12 Bytes，为了索引到每个字节，至少需要 12 位。

    因此虚拟地址就分为两部分，一部分是前半段`48 - 12 = 36`位用于定位页表的 entry，一部分是后半段`12`位偏移，用于在 4KB 的 entry 中定位到具体的字节。

    在 linux 中可以用命令`getconf PAGESIZE`查看页表每个 entry 的字节数。

    ref:

    1. <https://blog.csdn.net/m0_51717456/article/details/124256870>

    2. <https://www.cnblogs.com/chaozhu/p/10191575.html>

    3. <https://www.oreilly.com/library/view/linux-device-drivers/0596000081/ch07s04.html>

    4. <https://medium.com/@aravindchetla/kmalloc-v-s-vmalloc-13cb60746bcc>

    5. <https://www.kernel.org/doc/html/v5.0/core-api/mm-api.html#c.vmalloc>

    6. <https://www.kernel.org/doc/html/v5.0/core-api/memory-allocation.html>


* `vmalloc()`可以将不连续的物理地址通过链表的形式映射成连续的虚拟内存地址。

    头文件：`#include <linux/vmalloc.h>`

    ```c
    void *vaddr = vmalloc(1024);
    vfree(vaddr);
    ```

* `kthread_run()`

    创建并唤醒该线程。
    
    等价于先调用`kthread_create()`，再调用`wake_up_process()`唤醒线程。

    `kthread_run()`不是一个函数，而是一个宏。

    syntax:

    ```c
    kthread_run(threadfn, data, namefmt, ...)
    ```

* `wait_event()`

    `wait_event()`是一个宏。

    syntax: `wait_event(wq, condition)`

    休眠，直到`condition`为真，无法被手动打断。

    队列中的 wait queue 被标记为`TASK_UNINTERRUPTIBLE`。

* `wait_event()`和`wake_up()`传入的都是 wait queue head，即使 entry 加入到了 queue 里，也是处理 head。

    猜想： wait queue 在 wait 时，会将与之相关联的 thread 休眠。

    证据：`DECLARE_WAITQUEUE(wait_task,current);`将 wait entry 和 thread 相关联。

* `close()`函数在 unistd 中

* 一个带错误处理的 udev 驱动，见`ref_10`

    暂时先不引入 ioctl，目前没什么用

* linux driver 的 ioctl 原型是

    ```c
    long (*unlocked_ioctl) (struct file *filp, unsigned int cmd, unsigned long data);
    ```

    可以看到，其与 read, write 的根本区别是，它的参数里没有指针，所以不能传递太多信息，只能传递单个指令。

    2024/05/07/00: 第三个参数`unsigned long`可以被类型转换为指针传递数据，这样一来，其实`cmd`用于解释类型，`data`用于传递指针，可以做很多事情。

* `file_operations`不填 ioctl 回调函数也是可以的

* 内核中指针了一些宏，用于判断指针是否出错。

    ```c
    IS_ERR(指针)  // 返回真表示出错
    IS_ERR_OR_NULL(指针)  // 
    PTR_ERR(指针)  // 将出错的指针转换成错误码
    ERR_PTR(错误码)  // 将错误码转换成指针
    ```

* linux kernel module 中的 error 处理

    kernel module 中通常采用`goto`的方式处理 error，清理现场。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/fs.h>
    #include <linux/device.h>

    dev_t dev_region;
    const char *dev_region_name = "hlc dev region";
    struct class *dev_cls;
    struct device *hlc_dev;

    int mod_init(void)
    {
        printk(KERN_INFO "in mod_init() ...\n");

        alloc_chrdev_region(&dev_region, 0, 1, dev_region_name);
        printk(KERN_INFO "allocated device region.\n");

        dev_cls = class_create("hlc dev cls");
        if (IS_ERR(dev_cls)) {
            printk(KERN_INFO "fail to create dev class.\n");
            goto r_class_create;
        }
        printk(KERN_INFO "created device class.\n");

        hlc_dev = device_create(dev_cls, NULL, dev_region, NULL, "hlc_dev");
        if (IS_ERR(hlc_dev)) {
            printk(KERN_INFO "fail to create device.\n");
            goto r_device_create;
        }
        printk(KERN_INFO "created device.\n");
        return 0;

    r_device_create:
        printk(KERN_INFO "clean device class...\n");
        class_destroy(dev_cls);
    r_class_create:
        printk(KERN_INFO "clean device region...\n");
        unregister_chrdev_region(dev_region, 1);
        return -1;
    }

    void mod_exit(void)
    {
        printk(KERN_INFO "in mod_exit() ...\n");
        device_destroy(dev_cls, dev_region);
        class_destroy(dev_cls);
        unregister_chrdev_region(dev_region, 1);
        printk(KERN_INFO "unregistered device region.\n");
    }

    module_init(mod_init);
    module_exit(mod_exit);
    MODULE_LICENSE("GPL");
    ```

    当`class_create()`失败时，会调用`goto r_class_create;`跳转到`r_class_create`处，执行`unregister_chrdev_region()`清理在调用`class_create()`之前申请的资源。

    `goto r_device_create;`也是同理。

    这样的写法把一个函数分隔成了栈的结构，可以很方便地选择性清理现场：

    ```c
    // step 1
    if (出错) {
        goto 标签1；
    }

    // step 2
    if（出错） {
        goto 标签2；
    }

    // step 3
    if (出错) {
        goto 标签3;
    }

    标签3:
        复原第2步
    标签2:
        复原第1步
    标签1:
        return 错误码;
    ```


    当`mod_init()`返回`-1`时，命令`insmod`会报错：

    ```
    hlc@virt-Ubuntu-2204:~/Documents/Projects/driver_test$ sudo insmod hello.ko
    insmod: ERROR: could not insert module hello.ko: Operation not permitted
    ```

    此时`mod_exit()`函数不会被执行，但是`mod_init()`函数中的内容会被执行。

* 可以在还没创建 cdev 驱动时就创建 udev 设备文件

    可以在`/dev`中看到新创建的设备文件，但是此时`cat`会报错：

    ```
    hlc@virt-Ubuntu-2204:/dev$ sudo bash -c "cat /dev/hlc_dev"
    cat: /dev/hlc_dev: No such device or address
    ```

* 在使用`printk()`的`KERN_INFO`模式时，`dmesg`中显示的 log，第一个冒号之前的字体都为黄色，冒号以及之后的字体都是普通白色。

    如果字符串没有冒号，那么全部 log 都是白色。

* cdev 的 ops 函数原型中，`read`，`write`，`unloacked_ioctl`的返回值类型都是`ssize_t`，对应的是`long`。

* `pci_set_drvdata()`与`pci_get_drvdata()`用于获取/设置设备驱动私有数据

    syntax:

    ```c
    void *pci_get_drvdata(struct pci_dev *pdev);
    void pci_set_drvdata(struct pci_dev *pdev, void *data);
    ```

    It is a convenient way for example to save a pointer to a local dynamically allocated device context in the device probe callback and then retrieve it back with pci_get_drvdata in the device remove callback and do a proper cleanup of the context.

* `printk()`中指针地址的打印

    `%p`打印的并不是真实地址，而是经过处理的地址

    `%px`打印的是原始地址值，不经过处理。

    `%pK`是按配置文件打印值，更具体的用法可以参考这里：<https://blog.csdn.net/zqwone/article/details/127057245>

    <https://www.kernel.org/doc/Documentation/printk-formats.txt>

* 在 insmod 时报错`module verification failed: signature and/or required key missing - tainting kernel`

    可以直接在 makefile 开头添加一行：`CONFIG_MODULE_SIG=n`解决。

    虽然在 insmod 时还会有提示，但是可以正常加载驱动。

    更完善的解决办法可以参考这个：<https://stackoverflow.com/questions/24975377/kvm-module-verification-failed-signature-and-or-required-key-missing-taintin>

    如果 kernel 不是 singed 的，那么也可以不用加`CONFIG_MODULE_SIG=n`这一行。

* 似乎在安装`apt install build-essential`的时候，就会安装 kernel 相关的 herders 和预编译库

* `ktime_get_seconds()`可以获得系统启动后过去了多少时间

    `ktime_get_real_seconds()`可以获得 utc 时间，但是需要其他库/函数转换成人类可读时间。

    与时间相关的函数都在`linux/timekeeping.h`头文件中。

    如果需要 formatted output time，可以参考这篇：<https://www.kernel.org/doc/html/latest/core-api/printk-formats.html#time-and-date>

    与时间相关的函数与简要说明：<https://www.kernel.org/doc/html/latest/core-api/timekeeping.html>

    ref: <https://stackoverflow.com/questions/55566038/how-can-i-print-current-time-in-kernel>

* 对于 parameter 数组，在`cat`的时候，可以看到它是以`,`分隔的一些数字

    ```bash
    sudo bash -c "cat m_vec"
    ```

    output:

    ```
    1,2,3
    ```

    如果写入的数据多于数组的容量，会报错：

    ```bash
    sudo bash -c "echo 2,3,4,5,6 > m_vec"
    ```
    
    ```
    bash: line 1: echo: write error: Invalid argument
    ```

    导致写入失败。

    如果写入的数据少于数组的容量，则会自动在指针中写入具体有几个元素。


* 读取与写入 kernel module parameter 时，需要 root 权限的解决办法

    ```bash
    sudo bash -c "cat param_name"
    sudo bash -c "echo some_val > param_name"
    ```

## Introduction

### 驱动开发环境的搭建

#### 基于 Ubuntu 和 apt 的驱动开发环境的搭建

如果我们使用 Ubuntu 系统，那么就可以在它编译好的内核库的基础上开发驱动。如果使用其他系统，那么有可能需要自己编译内核。

linux 驱动是以 module 的形式加载到内核中的。Ubuntu 的内核是 signed 版本，我们自己写的 module 是 unsigned 版本，没有办法直接加载（在执行`insmod`时会报错）。目前我的解决办法是，先更新一下内核，换成 unsigned 版本或者 hwe 版本（hwe 版本表示支持最新硬件）：

```bash
apt install linux-image-generic-hwe-22.04
```

可以直接使用

```bash
apt install linux-image-`uname -r`
```

来安装。

接下来我们把原来的 kernel 删掉，然后下载新的：

```bash
sudo apt update && sudo apt upgrade
sudo apt remove --purge linux-headers-*
sudo apt autoremove && sudo apt autoclean
sudo apt-get install linux-headers-`uname -r`
```

注：我们可以用`uname -r`查看当前内核的版本，`uname -a`查看系统的完整版本。

这时候应该可以看见`/lib/modules/xxxxx-generic`路径下（比如`/lib/modules/5.19.0-32-generic/`），有一个`build`文件夹。这个`build`是一个 symbolic link，指向`/usr/src`下对应的内核源码文件夹：

```
hlc@hlc-Ubuntu2204:~/Documents/Projects/kernel_test$ ls -lh /lib/modules/5.19.0-32-generic/
total 6.6M
lrwxrwxrwx  1 root root   40  1月 30 23:44 build -> /usr/src/linux-headers-5.19.0-32-generic
drwxr-xr-x  2 root root 4.0K  2月 18 09:33 initrd
drwxr-xr-x 16 root root 4.0K  2月 18 09:31 kernel
-rw-r--r--  1 root root 1.5M  2月 26 14:45 modules.alias
...
```

如果可以看到对应的内容（即`build`指向`/usr/src/`下的一个目录），那么就说明编译好的内核已经装好了。

#### 基于编译内核的驱动开发环境搭建

1. 下载内核源码

（这部分目前用不到，先不写了）

通常我们说的 linux kernel 可以在

如果使用的是 Ubuntu 系统，下载源码可以在`apt`里下载：

```bash
apt-cache search linux-source
```

使用`uname -r`查看当前内核的版本，找到对应的版本。

编译内核需要用到`flex`, `bison`, `libssh-dev`, `libelf-dev`

1. 编译内核（在 Ubuntu 22.04 下编译）

    一个写得还不错的 tutorial，可以参考一下：<https://phoenixnap.com/kb/build-linux-kernel>

    编译内核需要一些额外的工具，可以参考这个网页<https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel>装一下：

    ```bash
    sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm
    ```

    下载好源码后，首先需要创建一个`.config`文件。我们可以使用 Ubuntu 系统里现成的：

    如果使用这种方法后编译失败，通常是证书设置的问题，参考这个：<https://askubuntu.com/questions/1329538/compiling-the-kernel-5-11-11>

    在`.config`里把`CONFIG_SYSTEM_TRUSTED_KEYS`和`CONFIG_SYSTEM_REVOCATION_KEYS`都设置成空字符串。

    也可以创建一个默认的：

    ```bash
    make menuconfig
    ```

    这种方法会在内核源码目录创建一个`.config`文件。使用这个 config 编译起来比较快，可能是因为有点选项没选上吧。

    单核编译可以用`makd`，多线程编译可以使用`makd -j16`（使用 16 线程编译）。

### hello, world 驱动

为了测试上面搭建的驱动开发环境是否成功，我们使用一个 hello world 项目测试一下。

首先，创建一个项目文件夹：`mkdir driver_test`，然后进入这个目录：`cd driver_test`。

接着，创建一个`hello_world.c`文件，然后写入以下内容：

```c
#include <linux/init.h>
#include <linux/module.h>

int hello_init(void)
{
    printk(KERN_INFO "hello my module\n");
    return 0;
}

void hello_exit(void)
{
    printk(KERN_INFO "bye bye!\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

然后我们创建一个`Makefile`文件：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-41-generic  # 这里要和我们下载的内核版本保持一致
obj-m  +=  hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

最后进行编译：`make`。编译完成后，我们可以在当前目录下看到`hello_world.ko`文件。

运行：`sudo insmod hello_world.ko`

调出 log，检查是否运行成功：`sudo dmesg`

如果输出的最后几行有`[ 2793.700004] <1>hello my module`，那么就说明驱动运行成功了。

最后卸载驱动：`sudo rmmod hello_world`。

### vscode config

* linux kernel module 开发 vscode 的配置

    `c_cpp_properties.json`:

    ```json
    {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/**",
                    "/usr/src/linux-headers-6.5.0-18-generic/include/",
                    "/usr/src/linux-headers-6.5.0-18-generic/arch/x86/include/generated/",
                    "/usr/src/linux-hwe-6.5-headers-6.5.0-18/arch/x86/include/",
                    "/usr/src/linux-hwe-6.5-headers-6.5.0-18/include"
                ],
                "defines": [
                    "KBUILD_MODNAME=\"hello\"",
                    "__GNUC__",
                    "__KERNEL__",
                    "MODULE"
                ],
                "compilerPath": "/usr/bin/gcc",
                "cStandard": "gnu17",
                "cppStandard": "c++17",
                "intelliSenseMode": "linux-gcc-x64"
            }
        ],
        "version": 4
    }
    ```

    `includePath`里新增的 include path 和`defines`里的四个宏，任何一个都不能少，不然 vscode 就会在代码里划红线报错。

    下面是一个没有报错的 example code:

    `hello.c`:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/ktime.h>

    int m_int = 5;
    module_param(m_int, int, S_IRUSR | S_IWUSR);

    int hello_init(void)
    {
        printk(KERN_INFO "hello my module\n");
        struct timespec64 ts64;
        ktime_get_ts64(&ts64);
        time64_t seconds = ktime_get_real_seconds();
        long nanoseconds = ts64.tv_nsec;
        printk(KERN_INFO "on init, current time: %ld seconds\n", seconds);
        return 0;
    }

    void hello_exit(void)
    {
        printk(KERN_INFO "bye bye!\n");
        struct timespec64 ts64;
        ktime_get_ts64(&ts64);
        time64_t seconds = ts64.tv_sec;
        long nanoseconds = ts64.tv_nsec;
        printk(KERN_INFO "on exit, current time: %ld seconds\n", seconds);
    }

    module_init(hello_init);
    module_exit(hello_exit);
    MODULE_LICENSE("GPL");
    ```

### 交叉编译

（有时间了填下这个坑）

交叉编译：`sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi-`

## 内核模块

内核模块编程注意事项：

1. 不能使用 C 库和 C 标准头文件
1. 使用 GNU C （在 ANSI C 上加了些语法）
1. 没有内存保护机制
1. 不能处理浮点运算
1. 注意并发互斥和可移植性

### 内核模块的加载与卸载

**内核模块的加载函数与卸载函数**

需要包含头文件

```cpp
#include <linux/init.h>
#include <linux/module.h>
```

Syntax:

1. 加载函数（Init function）

    ```c
    static int __init hello_world_init(void)  // __init 并不是必要的。它代表什么意思？
    {
        return 0;  // 返回 0 表示加载成功
    }
    module_init(hello_world_init);  // 使用宏来注册加载函数
    ```

2. 卸载函数（Exit function）

    ```c
    void __exit hello_world_exit(void)  // __exit 也不是必要的，它代表什么含义？
    {

    }
    module_exit(hello_world_exit);  // 使用宏来注册卸载函数
    ```

**有关模块加载与卸载的一个 Example**

`hello_world.c`：

```c
#include <linux/init.h>
#include <linux/module.h>

int hello_init(void)  // 参数列表中的 void 不可省略，不然无法通过编译.
{
    printk(KERN_INFO "hello my module\n");
    return 0;
}

void hello_exit(void)  // exit 不需要返回值
{
    printk(KERN_INFO "bye bye!\n");
}

module_init(hello_init);  // 这一行和下一行的分号都不是必要的，为什么？
module_exit(hello_exit);
MODULE_LICENSE("GPL");  // 不加这行的话，无法通过编译。MODULE_LICENSE 必须大写，不然无法通过编译。这一行末尾的分号是必要的，为什么？
```

另外一个 example:

```c
#include<linux/kernel.h>  // 这个头文件有什么用？
#include<linux/init.h>
#include<linux/module.h>

static int __init hello_world_init(void)  // __init 是什么意思？
{
    printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

static void __exit hello_world_exit(void)  // __exit 是什么意思？
{
    printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple hello world driver");
MODULE_VERSION("2:1.0");
```

接着，我们写 Makefile：

```Makefile
KERNEL_DIR=/usr/src/linux-headers-5.19.0-32-generic
obj-m += hello_world.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
```

另外一个 makefile 的 example:

```Makefile
obj-m += hello_world.o
 
ifdef ARCH
  KDIR = /home/embetronicx/BBG/tmp/lib/modules/5.10.65/build
else
  KDIR = /lib/modules/$(shell uname -r)/build  # 为什么要在前面加上 shell
endif
 
all:
  make -C $(KDIR)  M=$(shell pwd) modules  # $(PWD) 和 $(shell pwd) 有什么不同
 
clean:
  make -C $(KDIR)  M=$(shell pwd) clean  # KDIR 是个变量，为什么要给它加上 $() ？
```

然后在当前文件夹下运行`make`，会生成`hello_world.ko`文件。这个文件就是我们需要的内核模块文件，ko 代表 kernel object。

此时可以使用`sudo insmod hello_world.ko`插入模块，使用`sudo rmmod hello_world`移除模块，`sudo lsmod`查看已经加载的模块，`sudo dmesg`查看日志输出。

（如果还是无法`insmod`，或许需要取消 secure boot：<https://askubuntu.com/questions/762254/why-do-i-get-required-key-not-available-when-install-3rd-party-kernel-modules>）

### 日志消息打印

* `printk()`如果不加`\n`，那么不会在`dmesg`中立即刷新。

**`printk()`**

内核模块中可以使用函数`printk()`将消息打印到日志中，用法和`printf()`几乎相同。

```c
printk("hello my module\n");
```

`printk()`可打印的消息有不同的级别，我们可以在字符串前使用下面的宏字符串进行修饰：

* `KERN_EMERG`: Used for emergency messages, usually those that precede a crash.

* `KERN_ALERT`: A situation requiring immediate action.

* `KERN_CRIT`: Critical conditions are often related to serious hardware or software failures.

* `KERN_ERR`: Used to report error conditions; device drivers often use KERN_ERR to report hardware difficulties.

* `KERN_WARNING`: Warnings about problematic situations that do not, in themselves, create serious problems with the system.

* `KERN_NOTICE`: Situations that are normal, but still worthy of note. A number of security-related conditions are reported at this level.

* `KERN_INFO`: Informational messages. Many drivers print information about the hardware they find at startup time at this level.

* `KERN_DEBUG`: Used for debugging messages.

Example:

```c
printk(KERN_INFO "this is a info level log");
printk(KERN_WARNING "this is a warning level log");
printk("this is a non-level log")
```

经实际测试，如果不加日志级别，那么为白色粗字，`KERN_NOTICE`及以下，全都是正常白字，`KERN_WARNING`的消息会白字加粗，`KERN_ERR`的消息会变成红字，`KERN_CRIT`消息为红字加粗，`KERN_ALERT`为红底黑字，`KERN_EMERG`又变成正常白字。（这个颜色可能和 terminal 配色有关）

**默认打印级别**

上面的日志级别可以对应到数字 0 - 7，如果不指定日志级别，那么就是无级别。之所以映射到数字，似乎是因为可以使用数字控制 console 的输出级别。（但是经实际测试，好像不怎么有用）

可以参考这几个网站的资料：

1. <http://www.jyguagua.com/?p=708>

2. <https://blog.csdn.net/qustDrJHJ/article/details/51382138>

我们可以使用`cat /proc/sys/kernel/printk`查看当前日志的级别。

第一个数为内核默认打印级别，只有当`printk`的打印级别高于内核默认打印级别时，`printk`打印的信息才能显示 。

第二个数为`printk()`的默认打印级别。

修改内核的默认打印级别（即修改第 1 个数）：

`echo 5 > /proc/sys/kernel/printk`

（现在好像已经升级了，不管 printk level 大于还是小于当前 console 的 level，都不会在 console 上输出）

**`printk`的替代函数**

In the newer Linux kernels, you can use the APIs below instead of this `printk`.

* `pr_info()` – Print an info-level message. (ex. `pr_info("test info message\n")`).
* `pr_cont()` – Continues a previous log message in the same line.
* `pr_debug()` – Print a debug-level message conditionally.
* `pr_warn()` – Print a warning-level message.
* `pr_err()` – Print an error-level message. (ex. `pr_err(“test error message\n”)`).

经实测，`pr_info()`为正常白字，`pr_cont()`为白字加粗，`pr_debug()`没有输出，`pr_warn()`为白字加粗，`pr_err()`为红字。

### 模块信息

除了添加`MODULE_LICENSE()`外，可选的添加信息有：

* `MODULE_AUTHOR`: 模块作者

    Example: `MODULE_AUTHOR("hlc")`

* `MODULE_VERSION`: 模块版本
  
    Example: `MODULE_VERSION("1.0")`

* `MODULE_DESCRIPTION`：模块描述

    Example: `MODULE_DESCRIPTION("this is my first module")`

获得模块的一些信息：`modinfo hello_world_module.ko`

### 模块参数（Module Parameters Macros）

Module Parameters Macros：

* `module_param();`

    `module_param(name, type, perm);`

    `module_param(模块参数名，模块参数，访问权限);`

    This macro is used to initialize the arguments. `module_param` takes three parameters: the name of the variable, its type, and a permissions mask to be used for an accompanying sysfs entry.

    The macro should be placed outside of any function and is typically found near the head of the source file. `module_param()` macro, defined in `linux/moduleparam.h`.

    Parameters:

    * `type`

        可以是下面几个之一：`byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `charp`, `bool`, `invbool`;

        Numerous types are supported for module parameters:

        * `bool`

            A boolean (true or false) value (the associated variable should be of type int).

        * `invbool`

            The invbool type inverts the value, so that true values become false and vice versa.

        * `charp`

            A char pointer value. Memory is allocated for user-provided strings, and the pointer is set accordingly.

        * `int`, `long`, `short`, `uint`, `ulong`, `ushort`
            
            Basic integer values of various lengths. The versions starting with `u` are for unsigned values.

    * `perm` is the usual permissions value.

        There are several types of permissions:

        这些宏定义在`stat.h`中

        * `S_IWUSR`, `S_IRUSR`, `S_IXUSR`
        * `S_IRGRP`, `S_IWGRP`, `S_IXGRP`
        * `S_IROTH`, `S_IWOTH`, `S_IXOTH`

        可以看出来，`S_I`是一个 common prefix，R = read, W = write, X = Execute. USR = user, GRP = Group。

        Using `|` (OR operation) we can set multiple permissions at a time.

        在使用`S_IROTH`和`S_IWOTH`时会编译时报错，但是如果使用`0775`，可以顺序地给 others 加上`r`权限。不清楚为什么。

    Example:

    `module_param(valueETX, int, S_IWUSR | S_IRUSR);`

* `module_param_array();`

    `module_param_array(name, type, int *num, permissions);`

    `module_param_array(数组模块参数名，数组元素类型，NULL，访问权限);`

    This macro is used to send the array as an argument to the Linux device driver.

    Parameters:

    * `name`
    
        The name of the array (and of the parameter)

    * `type`
    
        The type of the array elements

    * `num`

        An integer variable (optional) otherwise `NULL`。

        在命令行中传递的数组的元素个数。

        比如`sudo insmod hello_world.ko m_arr=3,4,5`，那么`num`会被改写成`3`。

在代码中对模块参数的使用和普通变量没有区别。

Examples:

```c
int param_int = 10;
unsigned short param_ushort = 20;
char *param_string = "hello";
int param_arr[3] = {100, 200, 300};

module_param(param_int, int, 0775);
module_param(param_ushort, ushort, 0);
module_param(param_string, charp, 0644);
module_param_array(param_arr, int, NULL, 0755);
```

说明：

1. 这里的`0775`并不是和 linux 文件权限一一对应。使用`0775`作为权限后，得到的参数文件的权限如下所示：

    ```
    -rw-rw-r-- 1 root root 4096  5月 16 10:52 /sys/module/hello_world/parameters/a
    ```

    因为这个数字和 linux 文件的权限并不是对应关系，所以使用`0776`，`0777`，`777`等作为参数时会编译报错。

    正常情况下还是使用`S_IWUSR`这些标志位吧。

2. `0775`前面这个`0`必须加上，不然会编译报错。目前不清楚是为什么。

3. `unsigned short`定义的变量，在`module_param()`中注册模块参数时，必须使用`ushort`作为类型。

    在`module_param()`中填`unsigned short`会编译报错。

    如果使用`typedef unsigned short us;`，然后在`module_param()`中填`us`，同样也会编译报错。

4. 如果数组没有被初始化，或初始化的元素数量不够，那么元素的默认值都是 0。

打印模块参数：

```c
int modparam_init(void)  // 这是一个普通函数，函数的名字可以随便改
{
    printk("param_int = %d\n", param_int);
    printk("param_ushort = %hu\n", param_ushort);
    printk("param_string = %s\n", param_string);
    printk("param_arr = %d %d %d'n", param_arr[0], param_arr[1], param_arr[2]);
    return 0;
}
```

**在命令行中传递模块参数**

```bash
insmod hello_abc.ko param_int=50
insmod hello_abc.ko param_string="hello world"
insmod mod_param.ko param_arr=111,222,333
```

数组可以只传递一部分元素，此时会重新计算数组的长度，并重新赋值。比如在代码中定义了数组容量为`3`，并且初始化了所有元素，但是在命令行中只传递了 2 个元素的数组，那么`/sys/module/xxx/parameters`中的对应文件也只会显示 2 个元素。

如果命令行传递的元素的数量超出数组的容量，那么会报错。

当模块加载成功后，那些访问权限非 0 的模块会在以下路径下：

`/sys/module/模块名/parameters`

存在和模块参数名同名的文件，这些文件的权限来自于模块参数的权限。文件的内容与模块参数的值相同。因此可以通过修改文件中保存的数据，对模块参数进行修改：

```bash
sudo su
echo 1 > /sys/module/hello_world_module/parameters/my_param
```

**监测模块参数的改变**

`module_param_cb()`

This macro is used to register the callback. Whenever the argument (parameter) got changed, this callback function will be called.

为了注册 callback，我们首先需要填写一个结构体：

```c
struct kernel_param_ops 
{
    int (*set)(const char *val, const struct kernel_param *kp);
    int (*get)(char *buffer, const struct kernel_param *kp);
    void (*free)(void *arg);
};
```

（不明白这里的`free`有什么用）

Example:

```c
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/module.h>
#include<linux/moduleparam.h>
 
int valueETX, arr_valueETX[4];
char *nameETX;
int cb_valueETX = 0;
 
module_param(valueETX, int, S_IRUSR|S_IWUSR);                      // integer value
module_param(nameETX, charp, S_IRUSR|S_IWUSR);                     // String
module_param_array(arr_valueETX, int, NULL, S_IRUSR|S_IWUSR);      // Array of integers
 
/*----------------------Module_param_cb()--------------------------------*/
int notify_param(const char *val, const struct kernel_param *kp)
{
        // param_set_int 好像是把字符串转换成 int
        int res = param_set_int(val, kp); // Use helper for write variable
        if(res==0) {
                printk(KERN_INFO "Call back function called...\n");
                printk(KERN_INFO "New value of cb_valueETX = %d\n", cb_valueETX);
                return 0;
        }
        return -1;
}
 
const struct kernel_param_ops my_param_ops = 
{
        .set = &notify_param, // Use our setter ...
        .get = &param_get_int, // .. and standard getter
};
 
module_param_cb(cb_valueETX, &my_param_ops, &cb_valueETX, S_IRUGO|S_IWUSR );
/*-------------------------------------------------------------------------*/

/*
** Module init function
*/
static int __init hello_world_init(void)
{
        int i;
        printk(KERN_INFO "ValueETX = %d  \n", valueETX);
        printk(KERN_INFO "cb_valueETX = %d  \n", cb_valueETX);
        printk(KERN_INFO "NameETX = %s \n", nameETX);
        for (i = 0; i < (sizeof arr_valueETX / sizeof (int)); i++) {
                printk(KERN_INFO "Arr_value[%d] = %d\n", i, arr_valueETX[i]);
        }
        printk(KERN_INFO "Kernel Module Inserted Successfully...\n");
    return 0;
}

/*
** Module Exit function
*/
static void __exit hello_world_exit(void)
{
    printk(KERN_INFO "Kernel Module Removed Successfully...\n");
}
 
module_init(hello_world_init);
module_exit(hello_world_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple hello world driver");
MODULE_VERSION("1.0");
```

加载模块：

```bash
sudo insmod hello_world_module.ko valueETX=14 nameETX="EmbeTronicX" arr_valueETX=100,102,104,106
```

改变模块参数值：

方法一：

```bash
sudo sh -c "echo 13 > /sys/module/hello_world_module/parameters/cb_valueETX"
```

方法二：

Type sudo su. Then enter the password if it asks. Then do `echo 13 > /sys/module/hello_world_module/parameters/cb_valueETX`

然后我们可以在`dmesg`里看到参数值变化的消息：

```
[ 1688.610775] ValueETX = 14  
[ 1688.610782] cb_valueETX = 0  
[ 1688.610784] NameETX = EmbeTronicX 
[ 1688.610785] Arr_value[0] = 100
[ 1688.610786] Arr_value[1] = 102
[ 1688.610787] Arr_value[2] = 104
[ 1688.610788] Arr_value[3] = 106
[ 1688.610789] Kernel Module Inserted Successfully...
[ 1849.370708] Call back function called...
[ 1849.370714] New value of cb_valueETX = 13
[ 1880.687099] Kernel Module Removed Successfully...
```

### 模块符号的导出

模块导出符号可以将模块中的变量/函数导出，供内核其他模块使用。

内核中提供了相应的宏来实现模块的导出:

```
EXPORT_SYMBOL
EXPORT_SYMBOL_GPL  (只有遵循 GPL 协议的代码才可以使用)
```

Example:

```c
#include <linux/init.h>
#include <linux/module.h>

int add(int a, int b)
{
    return a + b;
}

int mul(int a, int b)
{
    return a * b;
}

EXPORT_SYMBOL(add);
EXPORT_SYMBOL_GPL(mul);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("hlc");
MODULE_VERSION("1.0");
MODULE_DESCRIPTION("this is a module symbol!");
// ...
```

在别的模块中使用时，需要这样写：

```c
extern int add(int a, int b);
extern int mul(int a, int b);
```

`extern`表示函数是在外部实现的，不是在本文件中实现的。使用这些函数时，需要先加载他们所在的模块。

在 Makefile 中，应该把两个`.o`文件都写上：

```Makefile
obj-m += xxx.o xxx_2.o
```

在加载模块的时候，应该先加载导出符号的模块，再加载使用符号的模块。卸载时，顺序要相反。

## 设备驱动

### cache

* 通过代码测试设备文件

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>

    int main()
    {
        char ch = 0;
        char buf[10] = {};

        int fd = open("/dev/hello_dev", O_RDWR);
        if (fd == -1)
        {
            perror("open");
            exit(-1);
        }

        printf("open successed! fd = %d\n", fd);

        while (1) {
            ch = getchar();
            getchar();

            if (ch == 'q')
                break;
            switch(ch)
            {
                case 'r':
                    read(fd, buf, 0);
                    break;
                case 'w':
                    write(fd, buf, 0);
                    break;
                default:
                    printf("error input\n");
            }
        }
    }
    ```

* 合并注册设备号和注册 cdev:

    ```c
    int register_chrdev(unsigned int major, const char *name, const struct file_operations *fops);
    ```

    当打开一个设备文件时，kernel 会根据设备号遍历 cdev 数组，找到对应的 cdev 结构体对象，然后把里面的`file_operations`里面的函数指针赋值给文件结构体`struct file`的`file_operations`里对应的函数。

### 设备类型

linux 设备：

1. 字符设备

    按字节流访问，一般是按顺序访问

    绝大多数设备都是字符设备。比如 led 按键 串口 传感器 LCD

    字符设备的驱动通过字符设备文件来访问

1. 块设备

    按数据块访问，块的大小固定，通常是 4k，具有随机访问能力

    内存，磁盘，SD卡，U盘

    块设备驱动通过块设备文件来访问

1. 网络设备

    一般只代表网卡设备。

    驱动实现要结合网络协议栈（TCP/IP）

    访问网络设备不通过文件，通过套接字（网络通信地址）访问

In fact, all device drivers that are neither storage nor network device drivers are some type of character driver.

字符设备驱动的访问：

驱动是沟通硬件和上层应用的媒介，字符设备驱动通过字符设备文件来访问，Linux 中所有的设备文件存放在`/dev`中，在用户层访问设备文件和普通文件的方法是没有区别的。Linux 操作系统实际上是通过设备号来找到对应的字符设备驱动（怎么找？）。

一个设备文件需要实现和普通文件相同的方法：

`open, close, read, write, lseek, ioctl, mmap, stat`

### 设备号

**构造设备号**

设备号用 32 位的一个`dev_t`类型的变量来表示（无符号整型），高 12 位表示主设备号，后 20 位表示次设备号。

The `dev_t` type (defined in `<linux/types.h>`) is used to hold device numbers—both the major and minor parts. `dev_t` is a 32-bit quantity with 12 bits set aside for the major number and 20 for the minor number.

主设备号用来区分不同类型的设备，次设备号用于区分设备的实例。

在`/proc/devices`文件中可以查找到设备号与对应的设备类型。

内核中提供了操作设备号的宏：

```c
MAJOR(设备号);  // 通过设备号获取主设备号  MAJOR(dev_t dev);
MINOR(设备号);  // 通过设备号获取次设备号  MINOR(dev_t dev);
MKDEV(主设备号, 次设备号);  // 通过主设备号和次设备号构造设备号  MKDEV(int major, int minor);
```

这些宏都是位运算，有空可以看看。

Example:

```c
dev_t m_dev_num = MKDEV(220,0);
```

**申请设备号**

设备号在内核中属于资源，需要向内核申请。有两种申请方式，一种是静态申请，一种是动态申请。

1. 静态申请（Statically allocating）

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/fs.h>

    dev_t dev = MKDEV(220, 0);

    int hlc_mod_init(void)
    {
        printk("load my module\n");

        // allocate a device number
        register_chrdev_region(dev, 1, "hlc_dev");
        printk(KERN_INFO "hlc dev, major = %d, minor = %d\n", MAJOR(dev), MINOR(dev));
        return 0;
    }
    ```

    上述代码中，使用`MKDEV(220, 0)`构造了一个设备号。构造方法是，首先选择一个内核中未被使用的主设备号（`cat /proc/devices`），比如`220`。然后根据设备个数分配次设备号，一般从`0`开始。

    `register_chrdev_region()`用于静态申请设备号。这个函数运行成功后，可以使用`cat /proc/devices`看到注册成功的设备号名称`220 hlc_dev`。

    Syntax:

    ```c
    register_chrdev_region(dev_t from, unsigned count, const char *name);
    ```

    Header file: `<linux/fs.h>`

    Params:

    * `from`: 要申请的起始设备号

    * `count`: 设备数量

        `count` is the total number of contiguous device numbers you are requesting. Note that, if the count is large, the range you request could spill over to the next major number; but everything will still work properly as long as the number range you request is available.

    * `name`: 设备号在内核中对应的名称

    Return value:

    返回 0 表示成功，返回非 0 表示失败。

2. 动态申请（Dynamically Allocating）

    动态申请指通过`alloc_chrdev_region()`向内核申请设备号。

    example:

    ```c
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/fs.h>

    dev_t dev_region;
    const char *dev_region_name = "hlc dev region";

    int mod_init(void)
    {
        printk(KERN_INFO "in mod_init() ...\n");
        int rtv = alloc_chrdev_region(&dev_region, 0, 1, dev_region_name);
        if (rtv != 0) {
            printk(KERN_INFO "alloc_chrdev_region() error code: %d\n", rtv);
        }
        printk(KERN_INFO "successfully allocate device region. major: %d, minor: %d\n",
            MAJOR(dev_region), MINOR(dev_region));
        return 0;
    }
    ```

    syntax:

    ```c
    int alloc_chrdev_region(dev_t *dev, unsigned baseminor, unsigned count, const char *name);
    ```

    alloc_chrdev_region - register a range of char device numbers

    header file: `<linux/fs.h>`

    params:

    * `dev`: 设备号的地址

    * `baseminor`: 起始次设备号

    * `count`: 设备号个数

    Return value:

    Returns zero or a negative error code.

**注销设备号**

不再使用设备号需要注销：

```c
void unregister_chrdev_region(dev_t from, unsigned count);
```

header file: `<linux/fs.h>`

params:

* `from`: 要注销的起始设备号

* `count`: 设备号的个数

一般在卸载模块的时候释放设备号。The usual place to call `unregister_chrdev_region` would be in your module’s cleanup function (Exit Function).

Example:

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

output:

```
[ 6790.673894] in mod_init() ...
[ 6790.673896] allocate device region.
[ 6796.788756] in mod_exit() ...
[ 6796.788764] unregistered device region.
```

**Difference between static and dynamic method**

A static method is only really useful if you know in advance which major number you want to start with. With the Static method, you are telling the kernel that, what device numbers you want (the start major/minor number and count) and it either gives them to you or not (depending on availability).

With the Dynamic method, you are telling the kernel that how many device numbers you need (the starting minor number and count) and it will find a starting major number for you, if one is available, of course.

Partially to avoid conflict with other device drivers, it’s considered preferable to use the Dynamic method function, which will dynamically allocate the device numbers for you.

The disadvantage of dynamic assignment is that you can’t create the device nodes in advance, because the major number assigned to your module will vary. For normal use of the driver, this is hardly a problem, because once the number has been assigned, you can read it from /proc/devices.

注：

1. 这个资源的名字叫设备号（device number），但是相关的函数却都是 device region 相关。

    是不是 device number 有歧义，一方面表示设备号，一方面又表示设备的个数，所以把改了？

    然后设备号的类型还是`dev_t`，有点像 device type。既不含 number 信息，也不含 region 信息，还容易和后面的`device`类型弄混。不清楚为什么要这么起名，可能是为了向上兼容吧。

### cdev 设备驱动

cdev 在内核中代表一个字符设备驱动。

```c
struct cdev {
    struct kobject kobj;
    struct module *owner;
    const struct file_operations *ops;  // 驱动操作函数集合
    struct list_head list;
    dev_t dev;  // 设备号
    unsigned int count;
};
```

**向内核中添加一个 cdev**

two ways of allocating and initializing one of these structures:

1. Runtime allocation

    ```c
    struct cdev *my_cdev = cdev_alloc( );
    my_cdev->ops = &my_fops;
    ``````

1. Own allocation

    Syntax:

    ```c
    void cdev_init(struct cdev *cdev, struct file_operations *fops);
    ```

    （`cdev`事先声明，`fops`也要事先写好）

    初始化 cdev（为 cdev 提供操作函数集合）

`cdev_add`：将 cdev 添加到内核（还会为 cdev 绑定设备号）

Syntax:

```c
int cdev_add(struct cdev *p, dev_t dev, unsigned count);
```

params:

* `p`: 要添加的 cdev 结构

* `dev`：起始设备号

* `count`：设备号个数

返回 0 表示成功，非 0 表示失败。

`cdev_del`：将 cdev 从内核中移除

```c
void cdev_del(struct cdev *p)
```

`inode`是文件的节点结构，用来存储文件静态信息。文件创建时，内核中就会有一个 inode 结构

`lsmod`除了可以列出当前已经加载的模块，还可以显示模块之间的依赖关系。

应用程序 app 先找到设备文件，设备文件通过设备号找到设备驱动，然后再调用相关的函数。设备号如何找到设备驱动？首先可以通过设备号找到`cdev`结构体，然后从`cdev`结构体找到`file_operations`结构体，再在这个结构体里找对应的驱动函数。

每个静态文件都对应内核中一个`inode` struct，存放一些基本信息。而当文件被打开时，内核会创建一个`file` struct，记录一些信息。

Example:

`hello_world.c`:

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

long int m_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    printk("in m_write function ...\n");
    return 0;
}

long m_ioctl(struct file *, unsigned int, unsigned long)
{
    printk("in m_ioctl function ...\n");
    return 0;
}

dev_t dev_num;
struct cdev m_dev;
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
    printk("Insert my test module.\n");

    // allocate a device number
    dev_num = MKDEV(220,0);
    register_chrdev_region(dev_num, 1, "hlc_dev");

    cdev_init(&m_dev, &m_ops);
    cdev_add(&m_dev, dev_num, 1);
    return 0;
}

void hello_exit(void)
{
    cdev_del(&m_dev);
    unregister_chrdev_region(dev_num, 1);
    printk("Exit my test module.\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

对其编译后，加载模块：`sudo insmod hello_world.ko`。

设备驱动需要和设备文件配合使用。

`__user`表示是一个用户空间的指针，所以 kernel 不可以直接使用。

```c
#ifdef __CHECKER__
# define __user __attribute__((noderef, address_space(1)))
# define __kernel /* default address space */
#else
# define __user
# define __kernel
#endif
```

### 设备文件

**Manually Creating Device File**

We can create the device file manually by using `mknod`.

`mknod -m <permissions> <name> <device type> <major> <minor>`

* `<name>` – your device file name that should have a full path (/dev/name)

* `<device type>` – Put c or b

    * `c` – Character Device

    * `b` – Block Device

* `<major>` – major number of your driver

* `<minor>` – minor number of your driver

* `-m <permissions>` – optional argument that sets the permission bits of the new device file to permissions

Example: `sudo mknod -m 666 /dev/etx_device c 246 0`

然后我们创建一个设备文件：`sudo mknod /dev/hello_dev c 220 0`。这里的主设备号和次设备号要和前面代码中的保持一致。

最后我们进行测试：`cat /dev/hello_dev`，然后执行`sudo smesg`，可以看到一些输出：

```
[26495.940998] <1>hello my module
[26502.305035] in m_open function ...
[26502.305042] in m_read function ...
[26502.305048] in m_release function ...
```

可以看到驱动正常运行。

**Automatically Creating Device File**

The automatic creation of device files can be handled with `udev`. `udev` is the device manager for the Linux kernel that creates/removes device nodes in the `/dev` directory dynamically. Just follow the below steps.

1. Include the header file `linux/device.h` and `linux/kdev_t.h`

2. Create the struct `class`

3. Create `device` with the `class` which is created by the above step

example:

`hello.c`

```cpp
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>

dev_t dev_id;  // 设备号
struct cdev cdev;  // cdev handle
const char *cdev_name = "hlc_dev";
struct class *dev_cls;  // device class, registered at /sys/class
struct device *dev;  // device file handle

int m_open(struct inode *inode, struct file *file_ptr)
{
    printk(KERN_INFO "in m_open()...\n");
    return 0;
}

int m_release(struct inode *inod, struct file *file_ptr)
{
    printk(KERN_INFO "in m_release()...\n");
    return 0;
}

ssize_t m_read(struct file *file_ptr, char __user *buf, size_t size, loff_t *offset)
{
    printk(KERN_INFO "in m_read()...\n");
    return 0;
}

ssize_t m_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    printk(KERN_INFO "in m_write()...\n");
    return 0;
}

ssize_t m_ioctl(struct file *file_ptr, unsigned int, unsigned long)
{
    printk(KERN_INFO "in m_ioctl()...\n");
    return 0;
}

struct file_operations m_ops = {
    .owner = THIS_MODULE,
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write,
    .unlocked_ioctl = m_ioctl
};

int hlc_module_init(void)
{
    printk(KERN_INFO "init hlc module\n");
    dev_id = MKDEV(255, 0);
    register_chrdev_region(dev_id, 1, "hlc cdev driver");
    cdev_init(&cdev, &m_ops);
    cdev_add(&cdev, dev_id, 1);
    dev_cls = class_create("hlc_dev_cls");
    if (IS_ERR(dev_cls)) {
        printk(KERN_INFO "fail to create device class.\n");
    }
    dev = device_create(dev_cls, NULL, dev_id, NULL, "hlc_dev");
    if (IS_ERR(dev)) {
        printk(KERN_INFO "fail to create device.\n");
    }
    return 0;
}

void hlc_module_exit(void)
{
    printk(KERN_INFO "exit hlc module!\n");
    device_destroy(dev_cls, dev_id);
    class_destroy(dev_cls);
    cdev_del(&cdev);
    unregister_chrdev_region(dev_id, 1);
}

module_init(hlc_module_init);
module_exit(hlc_module_exit);
MODULE_LICENSE("GPL");
```

`Makefile`:

```makefile
KERNEL_DIR=/usr/src/linux-headers-6.5.0-28-generic
obj-m  +=  hello.o
default:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	rm -f *.mod *.o *.order *.symvers *.cmd
```

compile:

```bash
make
```

run:

```bash
sudo insmod hello.ko
```

`dmesg` output:

```
[ 4976.473176] init hlc module
```

`ls /sys/class/ | grep hlc` output:

```
hlc_dev_cls
```

`ls /dev/ | grep hlc` output:

```
hlc_dev
```

run `sudo bash -c "cat /dev/hlc_dev"`, then `dmesg` output:

```
[ 5021.619227] in m_open()...
[ 5021.619251] in m_read()...
[ 5021.619269] in m_release()...
```

exit:

```bash
sudo rmmod hello
```

Explanation:

* Create the class

    This will create the struct class for our device driver. It will create a structure under `/sys/class/`. 创建的设备类在`/sys/class`目录下。

    `struct class * class_create(struct module *owner, const char *name);`

     ```c
    struct class *class_create(模块所有者, const char *name);
    ```

    * `owner` – pointer to the module that is to “own” this struct class

    * `name` – pointer to a string for the name of this class

    This is used to create a struct class pointer that can then be used in calls to class_device_create. The return value can be checked using IS_ERR() macro.

    销毁设备类：

    ```c
    void class_destroy(struct class *cls);
    ```

* `device_create`

    创建设备文件（设备节点）

    Syntax:

    ```c
    struct device *device_create(struct *class, struct device *parent, dev_t dev, void * drvdata, const char *fmt, ...);
    ```

    Parameters:

    * `class` – pointer to the struct class that this device should be registered to

    * `parent` – pointer to the parent struct device of this new device, if any

        父设备指针

    * `devt` – the dev_t for the char device to be added

        设备号

    * `drvdata` – the data to be added to the device for callbacks

        额外的数据

    * `fmt` – string for the device’s name

        设备文件名

    * `...` – variable arguments

    A “dev” file will be created, showing the `dev_t` for the device, if the `dev_t` is not `0,0`. If a pointer to a parent struct device is passed in, the newly created struct device will be a child of that device in sysfs. The pointer to the struct device will be returned from the call. Any further sysfs files that might be required can be created using this pointer. The return value can be checked using IS_ERR() macro.

    成功会在`/dev`目录下生成对应的设备文件，并返回设备指针

* `device_destroy`

    销毁设备文件

    Syntax:

    ```c
    void device_destroy(struct class *class, dev_t devt)
    ```

一个更加简洁的版本（没有添加错误处理，以及函数是否正常运行的判断）：

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>

dev_t dev_num;
struct cdev my_cdev;
struct class *dev_cls;
struct device *dev;

int m_open(struct inode *inode, struct file *file);
int m_release(struct inode *inode, struct file *file);
ssize_t m_read(struct file *file, char __user *buf, size_t size, loff_t *offset);
ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset);

struct file_operations fops = {
    .open = m_open,
    .release = m_release,
    .read = m_read,
    .write = m_write
};

int init_mod(void)
{
    pr_info("in init_mod()...\n");
    alloc_chrdev_region(&dev_num, 0, 1, "hlc_dev_num");
    cdev_init(&my_cdev, &fops);
    cdev_add(&my_cdev, dev_num, 1);
    dev_cls = class_create("hlc_dev_cls");
    dev = device_create(dev_cls, NULL, dev_num, NULL, "hlc_dev");
    pr_info("init hlc module done.\n");
    return 0;
}

void exit_mod(void)
{
    pr_info("in exit_mod()...\n");
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
    return 0;
}

ssize_t m_write(struct file *file, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in m_write()...\n");
    return 0;
}
```

### user mode program for device file

### cache

* ioctl 接口

    ioctl 是一个专用于硬件操作的接口，用于和实际数据传输相区分

    1. 用户层接口

        ```c
        #include <sys/ioctl.h>

        int ioctl(int fd, unsigned long request, ...);
        ```

        * `fd`: 文件描述符

    1. 内核接口

        头文件：

        ```c
        #include <linux/ioctl.h>
        ```

        对应`file_operations`中的成员：

        ```c
        long (*unlocked_ioctl) (struct file *filp, unsigned int cmd, unsigned long data);
        ```

    命令构造：

    Linux 内核提供构造 ioctl 命令的宏：

    ```c
    #define HELLO_ONE _IO('k',0)
    #define HELLO_TWO _IO('k',1)

    long cdd_ioctl(struct file *filp, xxxx)
    {
        printk("enter cdd_ioctl!\n");

        // 不同的命令对应不同的操作
        switch(cmd) {
            case HELLO_ONE:
                printk("hello one\n");
                break;
            case HELLO_TWO:
                printk("hello two\n");
                break;
            default:
                return -EINVAL;
        }
    }
    ```

我们可以使用`ls -l /dev`查看已经创建的设备文件。First of all, note that the first letter of the permissions field is denoted that driver type. Device files are denoted either by b, for block devices, or c, for character devices. Also, note that the size field in the ls -l listing is replaced by two numbers, separated by a comma. The first value is the major device number and the second is the minor device number.

file operations syntax:

header file: `<linux/fs.h>`

```c
struct file_operations {
    struct module *owner;
    loff_t (*llseek) (struct file *, loff_t, int);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    ssize_t (*read_iter) (struct kiocb *, struct iov_iter *);
    ssize_t (*write_iter) (struct kiocb *, struct iov_iter *);
    int (*iterate) (struct file *, struct dir_context *);
    int (*iterate_shared) (struct file *, struct dir_context *);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *, fl_owner_t id);
    int (*release) (struct inode *, struct file *);
    int (*fsync) (struct file *, loff_t, loff_t, int datasync);
    int (*fasync) (int, struct file *, int);
    int (*lock) (struct file *, int, struct file_lock *);
    ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long (*get_unmapped_area)(struct file *, unsigned long, unsigned long, unsigned long, unsigned long);
    int (*check_flags)(int);
    int (*flock) (struct file *, int, struct file_lock *);
    ssize_t (*splice_write)(struct pipe_inode_info *, struct file *, loff_t *, size_t, unsigned int);
    ssize_t (*splice_read)(struct file *, loff_t *, struct pipe_inode_info *, size_t, unsigned int);
    int (*setlease)(struct file *, long, struct file_lock **, void **);
    long (*fallocate)(struct file *file, int mode, loff_t offset,
              loff_t len);
    void (*show_fdinfo)(struct seq_file *m, struct file *f);
#ifndef CONFIG_MMU
    unsigned (*mmap_capabilities)(struct file *);
#endif
    ssize_t (*copy_file_range)(struct file *, loff_t, struct file *,
            loff_t, size_t, unsigned int);
    int (*clone_file_range)(struct file *, loff_t, struct file *, loff_t,
            u64);
    ssize_t (*dedupe_file_range)(struct file *, u64, u64, struct file *,
            u64);
};
```

fields:

* `struct module *owner`

    It is a pointer to the module that “owns” the structure. This field is used to prevent the module from being unloaded while its operations are in use. Almost all the time, it is simply initialized to `THIS_MODULE`, a macro defined in `<linux/module.h>`.

* `ssize_t (*read) (struct file *, char _ _user *, size_t, loff_t *);`

    This is used to retrieve data from the device. A null pointer in this position causes the read system call to fail with `-EINVAL` (“Invalid argument”). A non-negative return value represents the number of bytes successfully read (the return value is a “signed size” type, usually the native integer type for the target platform).

* `ssize_t (*write) (struct file *, const char _ _user *, size_t, loff_t *);`

    It is used to sends the data to the device. If NULL -EINVAL is returned to the program calling the write system call. The return value, if non-negative, represents the number of bytes successfully written.

* `int (*ioctl) (struct inode *, struct file *, unsigned int, unsigned long);`

    The ioctl system call offers a way to issue device-specific commands (such as formatting a track of a floppy disk, which is neither reading nor writing). Additionally, a few ioctl commands are recognized by the kernel without referring to the fops table. If the device doesn’t provide an ioctl method, the system call returns an error for any request that isn’t predefined (-ENOTTY, “No such ioctl for device”).

* `int (*open) (struct inode *, struct file *);`

    Though this is always the first operation performed on the device file, the driver is not required to declare a corresponding method. If this entry is NULL, opening the device always succeeds, but your driver isn’t notified.

* `int (*release) (struct inode *, struct file *);`

    This operation is invoked when the file structure is being released. Like open, release can be NULL.

## Data exchange between kernel space and user space

Using this driver we can send strings or data to the kernel device driver using the write function. It will store that string in the kernel space. Then when I read the device file, it will send the data which is written by write by function to the userspace application.

申请内存：

```c
#include <linux/slab.h>

void *kmalloc(size_t size, gfp_t flags);
```

The allocated region is contiguous in physical memory.

* `size` - how many bytes of memory are required.

* `flags`– the type of memory to allocate.

    The flags argument may be one of:

    * `GFP_USER` – Allocate memory on behalf of the user. May sleep.

    * `GFP_KERNEL` – Allocate normal kernel ram. May sleep.

    * `GFP_ATOMIC` – Allocation will not sleep. May use emergency pools. For example, use this inside interrupt handler.

    * `GFP_HIGHUSER` – Allocate pages from high memory.

    * `GFP_NOIO` – Do not do any I/O at all while trying to get memory.

    * `GFP_NOFS` – Do not make any fs calls while trying to get memory.

    * `GFP_NOWAIT` – Allocation will not sleep.

    * `__GFP_THISNODE` – Allocate node-local memory only.

    * `GFP_DMA` – Allocation is suitable for DMA. Should only be used for kmalloc caches. Otherwise, use a slab created with SLAB_DMA.

    Also, it is possible to set different flags by OR’ing in one or more of the following additional flags:

    * `__GFP_COLD` – Request cache-cold pages instead of trying to return cache-warm pages.


    * `__GFP_HIGH` – This allocation has high priority and may use emergency pools.

    * `__GFP_NOFAIL` – Indicate that this allocation is in no way allowed to fail (think twice before using).

    * `__GFP_NORETRY` – If memory is not immediately available, then give up at once.

    * `__GFP_NOWARN` – If allocation fails, don’t issue any warnings.

    * `__GFP_REPEAT` – If allocation fails initially, try once more before failing.

    更多的参数，可以参考`linux/gfp.h`

释放内存：

```c
void kfree(const void *objp)
```

Parameters:

* `*objp` – pointer returned by `kmalloc`

从用户态向内核态写入数据：

`copy_from_user()`

header file: `#include <linux/uaccess.h>`

syntax:

```c
unsigned long copy_from_user(void *to, const void __user *from, unsigned long n);
```

parameters:

* `to` – Destination address, in the kernel space

* `from` – The source address in the user space

* `n` – Number of bytes to copy

Returns number of bytes that could not be copied. On success, this will be zero.

从内核态向用户态写入数据：

`copy_to_user()`

Syntax:

```c
unsigned long copy_to_user(const void __user *to, const void *from, unsigned long  n);
```

This function is used to Copy a block of data into userspace (Copy data from kernel space to user space).

Parameters:

* `to` – Destination address, in the user space

* `from` – The source address in the kernel space

* `n` – Number of bytes to copy

Returns number of bytes that could not be copied. On success, this will be zero.

example:

(empty)

Note: Instead of using user space application, you can use echo and cat command.

## ioctl

There are many ways to Communicate between the Userspace and Kernel Space, they are:
IOCTL
Procfs
Sysfs
Configfs
Debugfs
Sysctl
UDP Sockets
Netlink Sockets

IOCTL is referred to as Input and Output Control, which is used to talk to device drivers. This system call is available in most driver categories.  The major use of this is in case of handling some specific operations of a device for which the kernel does not have a system call by default.

Some real-time applications of ioctl are Ejecting the media from a “cd” drive, changing the Baud Rate of Serial port, Adjusting the Volume, Reading or Writing device registers, etc. We already have the write and read function in our device driver. But it is not enough for all cases.

There are some steps involved to use IOCTL.

* Create IOCTL command in the driver

    ```c
    #include <linux/ioctl.h>

    #define WR_VALUE _IOW('a','a',int32_t*)
    #define RD_VALUE _IOR('a','b',int32_t*)

    #define "ioctl name" __IOX("magic number","command number","argument type")
    ```

    where IOX can be:

    * `IO`: an ioctl with no parameters
    * `IOW`: an ioctl with write parameters (copy_from_user)
    * `IOR`: an ioctl with read parameters (copy_to_user)
    * `IOWR`: an ioctl with both write and read parameters

    * The Magic Number is a unique number or character that will differentiate our set of ioctl calls from the other ioctl calls. some times the major number for the device is used here.
  
    * Command Number is the number that is assigned to the ioctl. This is used to differentiate the commands from one another.
  
    * The last is the type of data.

* Write IOCTL function in the driver

    ```c
    int ioctl(struct inode *inode,struct file *file,unsigned int cmd,unsigned long arg)
    ```

    * `inode`: is the inode number of the file being worked on.
    * `file`: is the file pointer to the file that was passed by the application.
    * `cmd`: is the ioctl command that was called from the userspace.
    * `arg`: are the arguments passed from the userspace

    Within the function “ioctl” we need to implement all the commands that we defined above (`WR_VALUE`, `RD_VALUE`). We need to use the same commands in the `switch` statement which is defined above.

    Then we need to inform the kernel that the ioctl calls are implemented in the function “etx_ioctl“. This is done by making the fops pointer “unlocked_ioctl” to point to “etx_ioctl” as shown below.

    ```c
    /*
    ** This function will be called when we write IOCTL on the Device file
    */
    static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
    {
        switch(cmd) {
            case WR_VALUE:
                if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
                {
                    pr_err("Data Write : Err!\n");
                }
                pr_info("Value = %d\n", value);
                break;
            case RD_VALUE:
                if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
                {
                    pr_err("Data Read : Err!\n");
                }
                break;
            default:
                pr_info("Default\n");
                break;
        }
        return 0;
    }

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
    };
    ```

* Create IOCTL command in a Userspace application

    ```c
    #define WR_VALUE _IOW('a','a',int32_t*)
    #define RD_VALUE _IOR('a','b',int32_t*)
    ```

* Use the IOCTL system call in a Userspace

    ```c
    #include <sys/ioctl.h>
    long ioctl( "file descriptor","ioctl command","Arguments");
    ```

    * `file descriptor`: This the open file on which the ioctl command needs to be executed, which would generally be device files.
  
    * `ioctl command`: ioctl command which is implemented to achieve the desired functionality
    
    * `arguments`: The arguments need to be passed to the ioctl command.

    Example:

    ```c
    ioctl(fd, WR_VALUE, (int32_t*) &number); 
    ioctl(fd, RD_VALUE, (int32_t*) &value);
    ```

Driver full code:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 // kmalloc()
#include<linux/uaccess.h>              // copy_to/from_user()
#include <linux/ioctl.h>
#include <linux/err.h>
 
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long     etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
};

/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read Function\n");
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write function\n");
        return len;
}

/*
** This function will be called when we write IOCTL on the Device file
*/
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch(cmd) {
        case WR_VALUE:
            if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
            {
                pr_err("Data Write : Err!\n");
            }
            pr_info("Value = %d\n", value);
            break;
        case RD_VALUE:
            if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
            {
                pr_err("Data Read : Err!\n");
            }
            break;
        default:
            pr_info("Default\n");
            break;
    }
    return 0;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_err("Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_err("Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device 1\n");
            goto r_device;
        }
        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        return -1;
}

/*
** Module exit function
*/
static void __exit etx_driver_exit(void)
{
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    pr_info("Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (IOCTL)");
MODULE_VERSION("1.5");
```

User program full code:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int main()
{
    int fd;
    int32_t value, number;

    printf("\nOpening Driver\n");
    fd = open("/dev/etx_device", O_RDWR);
    if(fd < 0) {
        printf("Cannot open device file...\n");
        return 0;
    }

    printf("Enter the Value to send\n");
    scanf("%d",&number);
    printf("Writing Value to Driver\n");
    ioctl(fd, WR_VALUE, (int32_t*) &number);

    printf("Reading Value from Driver\n");
    ioctl(fd, RD_VALUE, (int32_t*) &value);
    printf("Value is %d\n", value);

    printf("Closing Driver\n");
    close(fd);
}
```

This is a simple example of using ioctl in a Linux device driver. If you want to send multiple arguments, put those variables into the structure, and pass the address of the structure.

## Procfs

`/proc` is a mount point for the procfs (Process Filesystem) which is a filesystem in memory. Many processes store information about themselves on this virtual filesystem. ProcFS also stores other system information.

Check memory info: `cat /proc/meminfo`

Check modules info: `cat /proc/modules`

* `/proc/devices` — registered character and block major numbers
* `/proc/iomem` — on-system physical RAM and bus device addresses
* `/proc/ioports` — on-system I/O port addresses (especially for x86 systems)
* `/proc/interrupts` — registered interrupt request numbers
* `/proc/softirqs` — registered soft IRQs
* `/proc/swaps` — currently active swaps
* `/proc/kallsyms` — running kernel symbols, including from loaded modules
* `/proc/partitions` — currently connected block devices and their partitions
* `/proc/filesystems` — currently active filesystem drivers
* `/proc/cpuinfo` — information about the CPU(s) on the system

Most proc files are read-only and only expose kernel information to user space programs.

proc files can also be used to control and modify kernel behavior on the fly. The proc files need to be writable in this case.

enable IP forwarding of iptable: `echo 1 > /proc/sys/net/ipv4/ip_forward`

The proc file system can also be used to debug a kernel module. Just create entries for every variable that we want to track.

Creating procfs directory:

```c
struct proc_dir_entry *proc_mkdir(const char *name, struct proc_dir_entry *parent)
```

`name`: The name of the directory that will be created under `/proc`.

`parent`: In case the folder needs to be created in a subfolder under `/proc` a pointer to the same is passed else it can be left as NULL.

create proc entries:

header file: `linux/proc_fs.h`

```c
struct proc_dir_entry *proc_create ( const char *name, umode_t mode, struct proc_dir_entry *parent, const struct file_operations *proc_fops )
```

* `name`: The name of the proc entry
* `mode`: The access mode for proc entry
* `parent`: The name of the parent directory under /proc. If NULL is passed as a parent, the /proc directory will be set as a parent.
* `proc_fops`: The structure in which the file operations for the proc entry will be created.

Note: The above proc_create is valid in the Linux Kernel v3.10 to v5.5. From v5.6, there is a change in this API. The fourth argument const struct file_operations *proc_fops is changed to const struct proc_ops *proc_ops.

Example:

```c
proc_create("etx_proc",0666,NULL,&proc_fops);
```

create `file_operations` structure `proc_fops` in which we can map the read and write functions for the proc entry:

```c
static struct file_operations proc_fops = {
    .open = open_proc,
    .read = read_proc,
    .write = write_proc,
    .release = release_proc
};
```

For linux kernel v5.6 and above, use this:

```c
static struct proc_ops proc_fops = {
        .proc_open = open_proc,
        .proc_read = read_proc,
        .proc_write = write_proc,
        .proc_release = release_proc
};
```

`open` and `release` functions are optional:

```c
static int open_proc(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "proc file opend.....\t");
    return 0;
}

static int release_proc(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "proc file released.....\n");
    return 0;
}
```

The write function will receive data from the user space using the function copy_from_user into an array “etx_array”:

```c
static ssize_t write_proc(struct file *filp, const char *buff, size_t len, loff_t * off)
{
    printk(KERN_INFO "proc file write.....\t");
    copy_from_user(etx_array,buff,len);
    return len;
}
```

Once data is written to the proc entry we can read from the proc entry using a read function, i.e transfer data to the user space using the function `copy_to_user` function:

```c
static ssize_t read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset)
{
    printk(KERN_INFO "proc file read.....\n");
    if(len)
        len=0;
    else{
        len=1;
        return 0;
    }
    copy_to_user(buffer,etx_array,20);

    return length;;
}
```

Proc entry should be removed in the Driver exit function using the below function:

```c
void remove_proc_entry(const char *name, struct proc_dir_entry *parent);
```

Example:

```c
remove_proc_entry("etx_proc",NULL);
```

And you can remove the complete parent directory using `proc_remove(struct proc_dir_entry *parent)`.

Complete Driver Source Code:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>                 //kmalloc()
#include <linux/uaccess.h>              //copy_to/from_user()
#include <linux/ioctl.h>
#include <linux/proc_fs.h>
#include <linux/err.h>

/* 
** I am using the kernel 5.10.27-v7l. So I have set this as 510.
** If you are using the kernel 3.10, then set this as 310,
** and for kernel 5.1, set this as 501. Because the API proc_create()
** changed in kernel above v5.5.
**
*/ 
#define LINUX_KERNEL_VERSION  510
 
#define WR_VALUE _IOW('a','a',int32_t*)
#define RD_VALUE _IOR('a','b',int32_t*)
 
int32_t value = 0;
char etx_array[20]="try_proc_array\n";
static int len = 1;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
static struct proc_dir_entry *parent;

/*
** Function Prototypes
*/
static int      __init etx_driver_init(void);
static void     __exit etx_driver_exit(void);

/*************** Driver Functions **********************/
static int      etx_open(struct inode *inode, struct file *file);
static int      etx_release(struct inode *inode, struct file *file);
static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);
static long     etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
 
/***************** Procfs Functions *******************/
static int      open_proc(struct inode *inode, struct file *file);
static int      release_proc(struct inode *inode, struct file *file);
static ssize_t  read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset);
static ssize_t  write_proc(struct file *filp, const char *buff, size_t len, loff_t * off);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .unlocked_ioctl = etx_ioctl,
        .release        = etx_release,
};


#if ( LINUX_KERNEL_VERSION > 505 )

/*
** procfs operation sturcture
*/
static struct proc_ops proc_fops = {
        .proc_open = open_proc,
        .proc_read = read_proc,
        .proc_write = write_proc,
        .proc_release = release_proc
};

#else //LINUX_KERNEL_VERSION > 505

/*
** procfs operation sturcture
*/
static struct file_operations proc_fops = {
        .open = open_proc,
        .read = read_proc,
        .write = write_proc,
        .release = release_proc
};

#endif //LINUX_KERNEL_VERSION > 505

/*
** This function will be called when we open the procfs file
*/
static int open_proc(struct inode *inode, struct file *file)
{
    pr_info("proc file opend.....\t");
    return 0;
}

/*
** This function will be called when we close the procfs file
*/
static int release_proc(struct inode *inode, struct file *file)
{
    pr_info("proc file released.....\n");
    return 0;
}

/*
** This function will be called when we read the procfs file
*/
static ssize_t read_proc(struct file *filp, char __user *buffer, size_t length,loff_t * offset)
{
    pr_info("proc file read.....\n");
    if(len)
    {
        len=0;
    }
    else
    {
        len=1;
        return 0;
    }
    
    if( copy_to_user(buffer,etx_array,20) )
    {
        pr_err("Data Send : Err!\n");
    }
 
    return length;;
}

/*
** This function will be called when we write the procfs file
*/
static ssize_t write_proc(struct file *filp, const char *buff, size_t len, loff_t * off)
{
    pr_info("proc file wrote.....\n");
    
    if( copy_from_user(etx_array,buff,len) )
    {
        pr_err("Data Write : Err!\n");
    }
    
    return len;
}

/*
** This function will be called when we open the Device file
*/
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read function\n");
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write Function\n");
        return len;
}

/*
** This function will be called when we write IOCTL on the Device file
*/
static long etx_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch(cmd) {
        case WR_VALUE:
            if( copy_from_user(&value ,(int32_t*) arg, sizeof(value)) )
            {
                pr_err("Data Write : Err!\n");
            }
            pr_info("Value = %d\n", value);
            break;
        case RD_VALUE:
            if( copy_to_user((int32_t*) arg, &value, sizeof(value)) )
            {
                pr_err("Data Read : Err!\n");
            }
            break;
        default:
            pr_info("Default\n");
            break;
    }
    return 0;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            pr_info("Cannot allocate major number\n");
            return -1;
    }
    pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0){
        pr_info("Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        pr_info("Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        pr_info("Cannot create the Device 1\n");
        goto r_device;
    }
    
    /*Create proc directory. It will create a directory under "/proc" */
    parent = proc_mkdir("etx",NULL);
    
    if( parent == NULL )
    {
        pr_info("Error creating proc entry");
        goto r_device;
    }
    
    /*Creating Proc entry under "/proc/etx/" */
    proc_create("etx_proc", 0666, parent, &proc_fops);

    pr_info("Device Driver Insert...Done!!!\n");
    return 0;
 
r_device:
    class_destroy(dev_class);
r_class:
    unregister_chrdev_region(dev,1);
    return -1;
}
 
/*
** Module exit function
*/
static void __exit etx_driver_exit(void)
{
        /* Removes single proc entry */
        //remove_proc_entry("etx/etx_proc", parent);
        
        /* remove complete /proc/etx */
        proc_remove(parent);
        
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (procfs)");
MODULE_VERSION("1.6");
```

As there are changes in the procfs file system in the Linux kernel 3.10 and 5.6, we have added a macro called `LINUX_KERNEL_VERSION`. You have to mention your Linux kernel version. Based on that, we will control the APIs in this source code.

Note:

You can follow this format for this `LINUX_KERNEL_VERSION`.

Example:

|Your Linux Kernel version | LINUX_KERNEL_VERSION |
| - | - |
| v3.10 | 310 |
| v5.6 | 506 |
| v5.10 | 510 |

test:

`cat /proc/etx/etx_proc`

`echo "device driver proc" > /proc/etx/etx_proc`

## waitqueue

Whenever a process must wait for an event (such as the arrival of data or the termination of a process), it should go to sleep. Sleeping causes the process to suspend execution, freeing the processor for other uses. After some time, the process will be woken up and will continue with its job when the event which we are waiting for has arrived.

Wait queue is a mechanism provided in the kernel to implement the wait. As the name itself suggests, waitqueue is the list of processes waiting for an event. In other words, A wait queue is used to wait for someone to wake you up when a certain condition is true. They must be used carefully to ensure there is no race condition.

There are 3 important steps in Waitqueue.

1. Initializing Waitqueue

    header file: `linux/wait.h`

    1. Static method

        ```c
        DECLARE_WAIT_QUEUE_HEAD(wq);
        ```

        Where the “wq” is the name of the queue on which task will be put to sleep.

    1. Dynamic method

        ```c
        wait_queue_head_t wq;
        init_waitqueue_head(&wq);
        ```

1. Queuing (Put the Task to sleep until the event comes)

    Once the wait queue is declared and initialized, a process may use it to go to sleep. There are several macros are available for different uses.

    * wait_event

        Syntax:

        ```c
        wait_event(wq, condition);
        ```

        sleep until a condition gets true.

        Parameters:

        * `wq` – the waitqueue to wait on

        * `condition` – a C expression for the event to wait for

        The process is put to sleep (`TASK_UNINTERRUPTIBLE`) until the condition evaluates to true. The `condition` is checked each time the waitqueue `wq` is woken up.

    * wait_event_timeout

        Syntax:

        ```c
        wait_event_timeout(wq, condition, timeout);
        ```

        sleep until a condition gets true or a timeout elapses

        Parameters:

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `timeout` –  timeout, in jiffies

        The process is put to sleep (TASK_UNINTERRUPTIBLE) until the condition evaluates to true or timeout elapses. The condition is checked each time the waitqueue wq is woken up.

        It returns 0 if the condition evaluated to false after the timeout elapsed, 1 if the condition evaluated to true after the timeout elapsed, or the remaining jiffies (at least 1) if the condition evaluated to true before the timeout elapsed.

    * wait_event_cmd

        Syntax:

        ```c
        wait_event_cmd(wq, condition, cmd1, cmd2);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `cmd1` – the command will be executed before sleep

        * `cmd2` – the command will be executed after sleep

        The process is put to sleep (TASK_UNINTERRUPTIBLE) until the condition evaluates to true. The condition is checked each time the waitqueue wq is woken up.

    * wait_event_interruptible

        Syntax:

        ```c
        wait_event_interruptible(wq, condition);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        The process is put to sleep (TASK_INTERRUPTIBLE) until the condition evaluates to true or a signal is received. The condition is checked each time the waitqueue wq is woken up.

        The function will return -ERESTARTSYS if it was interrupted by a signal and 0 if condition evaluated to true.

    * wait_event_interruptible_timeout

        Syntax:

        ```c
        wait_event_interruptible_timeout(wq, condition, timeout);
        ```

        sleep until a condition gets true or a timeout elapses

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        * `timeout` –  timeout, in jiffies

        The process is put to sleep (TASK_INTERRUPTIBLE) until the condition evaluates to true or a signal is received or timeout elapsed. The condition is checked each time the waitqueue wq is woken up.

        It returns, 0 if the condition evaluated to false after the timeout elapsed, 1 if the condition evaluated to true after the timeout elapsed, the remaining jiffies (at least 1) if the condition evaluated to true before the timeout elapsed, or -ERESTARTSYS if it was interrupted by a signal.

    * wait_event_killable

        Syntax:

        ```c
        wait_event_killable(wq, condition);
        ```

        sleep until a condition gets true

        * `wq` –  the waitqueue to wait on

        * `condtion` – a C expression for the event to wait for

        The process is put to sleep (TASK_KILLABLE) until the condition evaluates to true or a signal is received. The condition is checked each time the waitqueue wq is woken up.

        The function will return -ERESTARTSYS if it was interrupted by a signal and 0 if condition evaluated to true.

    Whenever we use the above one of the macro, it will add that task to the waitqueue which is created by us. Then it will wait for the event.

    Note: Old kernel versions used the functions `sleep_on()` and `interruptible_sleep_on()`, but those two functions can introduce bad race conditions and should not be used.

1. Waking Up Queued Task

    When some Tasks are in sleep mode because of the waitqueue, then we can use the below function to wake up those tasks.

    * wake_up

        Syntax:

        ```c
        wake_up(&wq);
        ```

        wakes up only one process from the wait queue which is in non-interruptible sleep.

        Parameters:

        * `wq` – the waitqueue to wake up

    * wake_up_all

        Syntax:

        ```c
        wake_up_all(&wq);
        ```

        wakes up all the processes on the wait queue

    * wake_up_interruptible

        Syntax:

        ```c
        wake_up_interruptible(&wq);
        ```

        wakes up only one process from the wait queue that is in interruptible sleep

    * wake_up_sync and wake_up_interruptible_sync

        Syntax:

        ```c
        wake_up_sync(&wq);
        wake_up_interruptible_sync(&wq);
        ```

        Normally, a `wake_up` call can cause an immediate reschedule to happen, meaning that other processes might run before `wake_up` returns. The “synchronous” variants instead make any awakened processes runnable but do not reschedule the CPU. This is used to avoid rescheduling when the current process is known to be going to sleep, thus forcing a reschedule anyway. Note that awakened processes could run immediately on a different processor, so these functions should not be expected to provide mutual exclusion.

driver code:

* Waitqueue created by Static Method

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include <linux/slab.h>                 //kmalloc()
    #include <linux/uaccess.h>              //copy_to/from_user()
    #include <linux/kthread.h>
    #include <linux/wait.h>                 // Required for the wait queues
    #include <linux/err.h>
    
    
    uint32_t read_count = 0;
    static struct task_struct *wait_thread;
    
    DECLARE_WAIT_QUEUE_HEAD(wait_queue_etx);
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    int wait_queue_flag = 0;

    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
            .owner          = THIS_MODULE,
            .read           = etx_read,
            .write          = etx_write,
            .open           = etx_open,
            .release        = etx_release,
    };

    /*
    ** Thread function
    */
    static int wait_function(void *unused)
    {
        while(1) {
            pr_info("Waiting For Event...\n");
            wait_event_interruptible(wait_queue_etx, wait_queue_flag != 0 );
            if(wait_queue_flag == 2) {
                pr_info("Event Came From Exit Function\n");
                return 0;
            }
            pr_info("Event Came From Read Function - %d\n", ++read_count);
            wait_queue_flag = 0;
        }
        do_exit(0);
        return 0;
    }

    /*
    ** This function will be called when we open the Device file
    */
    static int etx_open(struct inode *inode, struct file *file)
    {
        pr_info("Device File Opened...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */
    static int etx_release(struct inode *inode, struct file *file)
    {
        pr_info("Device File Closed...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read Function\n");
        wait_queue_flag = 1;
        wake_up_interruptible(&wait_queue_etx);
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Write function\n");
        return len;
    }
    
    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            pr_info("Cannot allocate major number\n");
            return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
        etx_cdev.owner = THIS_MODULE;
        etx_cdev.ops = &fops;

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_info("Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_info("Cannot create the struct class\n");
            goto r_class;
        }

        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_info("Cannot create the Device 1\n");
            goto r_device;
        }

        //Create the kernel thread with name 'mythread'
        wait_thread = kthread_create(wait_function, NULL, "WaitThread");
        if (wait_thread) {
            pr_info("Thread Created successfully\n");
            wake_up_process(wait_thread);
        } else
            pr_info("Thread creation failed\n");

        pr_info("Device Driver Insert...Done!!!\n");
        return 0;

    r_device:
            class_destroy(dev_class);
    r_class:
            unregister_chrdev_region(dev,1);
            return -1;
    }

    /*
    ** Module exit function
    */ 
    static void __exit etx_driver_exit(void)
    {
        wait_queue_flag = 2;
        wake_up_interruptible(&wait_queue_etx);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple linux driver (Waitqueue Static method)");
    MODULE_VERSION("1.7");
    ```

* Waitqueue created by Dynamic Method

    ```c
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include <linux/slab.h>                 //kmalloc()
    #include <linux/uaccess.h>              //copy_to/from_user()
    #include <linux/kthread.h>
    #include <linux/wait.h>                 // Required for the wait queues
    #include <linux/err.h>
    
    
    uint32_t read_count = 0;
    static struct task_struct *wait_thread;
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    wait_queue_head_t wait_queue_etx;
    int wait_queue_flag = 0;
    
    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, const char *buf, size_t len, loff_t * off);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
    };
    
    /*
    ** Thread function
    */
    static int wait_function(void *unused)
    { 
        while(1) {
            pr_info("Waiting For Event...\n");
            wait_event_interruptible(wait_queue_etx, wait_queue_flag != 0 );
            if(wait_queue_flag == 2) {
                pr_info("Event Came From Exit Function\n");
                return 0;
            }
            pr_info("Event Came From Read Function - %d\n", ++read_count);
            wait_queue_flag = 0;
        }
        return 0;
    }
    
    /*
    ** This function will be called when we open the Device file
    */ 
    static int etx_open(struct inode *inode, struct file *file)
    {
        pr_info("Device File Opened...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */
    static int etx_release(struct inode *inode, struct file *file)
    {
        pr_info("Device File Closed...!!!\n");
        return 0;
    }

    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Read Function\n");
        wait_queue_flag = 1;
        wake_up_interruptible(&wait_queue_etx);
        return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
    {
        pr_info("Write function\n");
        return len;
    }

    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_info("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);

        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_info("Cannot add the device to the system\n");
            goto r_class;
        }

        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_info("Cannot create the struct class\n");
            goto r_class;
        }
    
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_info("Cannot create the Device 1\n");
            goto r_device;
        }
        
        //Initialize wait queue
        init_waitqueue_head(&wait_queue_etx);

        // Create the kernel thread with name 'mythread'
        wait_thread = kthread_create(wait_function, NULL, "WaitThread");
        if (wait_thread) {
            pr_info("Thread Created successfully\n");
            wake_up_process(wait_thread);
        } else
            pr_info("Thread creation failed\n");

        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
    
    r_device:
        class_destroy(dev_class);
    r_class:
        unregister_chrdev_region(dev,1);
        return -1;
    }

    /*
    ** Module exit function
    */
    static void __exit etx_driver_exit(void)
    {
        wait_queue_flag = 2;
        wake_up_interruptible(&wait_queue_etx);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple linux driver (Waitqueue Dynamic method)");
    MODULE_VERSION("1.8");
    ```

Makefile:

```Makefile
obj-m += driver.o
KDIR = /lib/modules/$(shell uname -r)/build
all:
    make -C $(KDIR)  M=$(shell pwd) modules
clean:
    make -C $(KDIR)  M=$(shell pwd) clean
```

（这个版本中的 Makefile 使用的并不是`$(PWD)`之类的，而是`$(shell pwd)`，这两者有什么不同？）

test: `sudo cat /dev/etx_device`

## sysfs

Sysfs is a virtual filesystem mounted on `/sys`. Sysfs contain information about devices and drivers.

**Kernel Objects**

The heart of the sysfs model is the kobject. Kobject is the glue that binds the sysfs and the kernel, which is represented by `struct kobject` and defined in `<linux/kobject.h>`. A struct kobject represents a kernel object, maybe a device or so, such as the things that show up as directory in the sysfs filesystem.

Kobjects are usually embedded in other structures.

Syntax:

```c
#define KOBJ_NAME_LEN 20 

struct kobject {
 char *k_name;
 char name[KOBJ_NAME_LEN];
 struct kref kref;
 struct list_head entry;
 struct kobject *parent;
 struct kset *kset;
 struct kobj_type *ktype;
 struct dentry *dentry;
};
```

Explanation:

* `struct kobject`

    * `name` (Name of the kobject. Current kobject is created with this name in sysfs.)

    * `parent` (This is kobject’s parent. When we create a directory in sysfs for the current kobject, it will create under this parent directory)

    * `ktype` (the type associated with a kobject)

    * `kset` (a group of kobjects all of which are embedded in structures of the same type)

    * `sd` (points to a sysfs_dirent structure that represents this kobject in sysfs.)

    * `kref` (provides reference counting)

    `kobject` is used to create kobject directory in /sys.

There are two steps to creating and using sysfs.

1. Create a directory in `/sys`

    We can use this function (`kobject_create_and_add`) to create a directory.

    `struct kobject * kobject_create_and_add ( const char * name, struct kobject * parent);`

    Where,

    * `name` – the name for the kobject

    * `parent` – the parent kobject of this kobject, if any.

        If you pass `kernel_kobj` to the second argument, it will create the directory under `/sys/kernel/`. If you pass `firmware_kobj` to the second argument, it will create the directory under `/sys/firmware/`. If you pass `fs_kobj` to the second argument, it will create the directory under `/sys/fs/`. If you pass NULL to the second argument, it will create the directory under `/sys/`.

    This function creates a kobject structure dynamically and registers it with sysfs. If the kobject was not able to be created, `NULL` will be returned.

    Call `kobject_put` and the structure `kobject` will be dynamically freed when it is no longer being used. (not clear. Does it mean free the memory immediately or wait for the last time that `struct object` was used?)

    Example:

    ```c
    struct kobject *kobj_ref;

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj); //sys/kernel/etx_sysfs

    /*Freeing Kobj*/
    kobject_put(kobj_ref);
    ```

1. Create Sysfs file

    sysfs file is used to interact user space with kernel space.

    We can create the sysfs file using sysfs attributes. Attributes are represented as regular files in sysfs with one value per file. There are loads of helper functions that can be used to create the kobject attributes. They can be found in the header file `sysfs.h`.

    * Create attribute

        Syntax:

        ```c
        struct kobj_attribute {
            struct attribute attr;
            ssize_t (*show)(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
            ssize_t (*store)(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
        };
        ```

        Parameters:

        * `attr` – the attribute representing the file to be created,

        * `show` – the pointer to the function that will be called when the file is read in sysfs,

        * `store` – the pointer to the function which will be called when the file is written in sysfs.

        We can create an attribute using `__ATTR` macro.

        `__ATTR(name, permission, show_ptr, store_ptr);`
        
    * Store and Show functions

        ```c
        ssize_t (*show)(struct kobject *kobj, struct kobj_attribute *attr, char *buf);

        ssize_t (*store)(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
        ```

        Store function will be called whenever we are writing something to the sysfs attribute.

        Show function will be called whenever we are reading the sysfs attribute.

    * Create sysfs file

        To create a single file attribute we are going to use ‘sysfs_create_file’.

        int sysfs_create_file ( struct kobject *  kobj, const struct attribute * attr);

        Where,

        `kobj` – object we’re creating for.

        `attr` – attribute descriptor.

        One can use another function `sysfs_create_group` to create a group of attributes.

        Once you have done with the sysfs file, you should delete this file using `sysfs_remove_file`。

        ```c
        void sysfs_remove_file ( struct kobject *  kobj, const struct attribute * attr);
        ```

        Where,

        `kobj` – object we’re creating for.

        `attr` – attribute descriptor.

    Example:

    ```c
    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
        return sprintf(buf, "%d", etx_value);
    }

    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            sscanf(buf,"%d",&etx_value);
            return count;
    }

    //This Function will be called from Init function
    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
        printk(KERN_INFO"Cannot create sysfs file......\n");
        goto r_sysfs;
    }
    //This should be called from exit function
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    ```

    driver:

    ```c
    /***************************************************************************//**
    *  \file       driver.c
    *
    *  \details    Simple Linux device driver (sysfs)
    *
    *  \author     EmbeTronicX
    *
    *  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
    *
    *******************************************************************************/
    #include <linux/kernel.h>
    #include <linux/init.h>
    #include <linux/module.h>
    #include <linux/kdev_t.h>
    #include <linux/fs.h>
    #include <linux/cdev.h>
    #include <linux/device.h>
    #include<linux/slab.h>                 //kmalloc()
    #include<linux/uaccess.h>              //copy_to/from_user()
    #include<linux/sysfs.h> 
    #include<linux/kobject.h> 
    #include <linux/err.h>
    
    volatile int etx_value = 0;
    
    
    dev_t dev = 0;
    static struct class *dev_class;
    static struct cdev etx_cdev;
    struct kobject *kobj_ref;

    /*
    ** Function Prototypes
    */
    static int      __init etx_driver_init(void);
    static void     __exit etx_driver_exit(void);
    
    /*************** Driver functions **********************/
    static int      etx_open(struct inode *inode, struct file *file);
    static int      etx_release(struct inode *inode, struct file *file);
    static ssize_t  etx_read(struct file *filp, 
                            char __user *buf, size_t len,loff_t * off);
    static ssize_t  etx_write(struct file *filp, 
                            const char *buf, size_t len, loff_t * off);
    
    /*************** Sysfs functions **********************/
    static ssize_t  sysfs_show(struct kobject *kobj, 
                            struct kobj_attribute *attr, char *buf);
    static ssize_t  sysfs_store(struct kobject *kobj, 
                            struct kobj_attribute *attr,const char *buf, size_t count);

    struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

    /*
    ** File operation sturcture
    */
    static struct file_operations fops =
    {
            .owner          = THIS_MODULE,
            .read           = etx_read,
            .write          = etx_write,
            .open           = etx_open,
            .release        = etx_release,
    };

    /*
    ** This function will be called when we read the sysfs file
    */
    static ssize_t sysfs_show(struct kobject *kobj, 
                    struct kobj_attribute *attr, char *buf)
    {
            pr_info("Sysfs - Read!!!\n");
            return sprintf(buf, "%d", etx_value);
    }

    /*
    ** This function will be called when we write the sysfsfs file
    */
    static ssize_t sysfs_store(struct kobject *kobj, 
                    struct kobj_attribute *attr,const char *buf, size_t count)
    {
            pr_info("Sysfs - Write!!!\n");
            sscanf(buf,"%d",&etx_value);
            return count;
    }

    /*
    ** This function will be called when we open the Device file
    */ 
    static int etx_open(struct inode *inode, struct file *file)
    {
            pr_info("Device File Opened...!!!\n");
            return 0;
    }

    /*
    ** This function will be called when we close the Device file
    */ 
    static int etx_release(struct inode *inode, struct file *file)
    {
            pr_info("Device File Closed...!!!\n");
            return 0;
    }
    
    /*
    ** This function will be called when we read the Device file
    */
    static ssize_t etx_read(struct file *filp, 
                    char __user *buf, size_t len, loff_t *off)
    {
            pr_info("Read function\n");
            return 0;
    }

    /*
    ** This function will be called when we write the Device file
    */
    static ssize_t etx_write(struct file *filp, 
                    const char __user *buf, size_t len, loff_t *off)
    {
            pr_info("Write Function\n");
            return len;
    }
    
    /*
    ** Module Init function
    */
    static int __init etx_driver_init(void)
    {
            /*Allocating Major number*/
            if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                    pr_info("Cannot allocate major number\n");
                    return -1;
            }
            pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
    
            /*Creating cdev structure*/
            cdev_init(&etx_cdev,&fops);
    
            /*Adding character device to the system*/
            if((cdev_add(&etx_cdev,dev,1)) < 0){
                pr_info("Cannot add the device to the system\n");
                goto r_class;
            }
    
            /*Creating struct class*/
            if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
                pr_info("Cannot create the struct class\n");
                goto r_class;
            }
    
            /*Creating device*/
            if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
                pr_info("Cannot create the Device 1\n");
                goto r_device;
            }
    
            /*Creating a directory in /sys/kernel/ */
            kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
    
            /*Creating sysfs file for etx_value*/
            if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                    pr_err("Cannot create sysfs file......\n");
                    goto r_sysfs;
        }
            pr_info("Device Driver Insert...Done!!!\n");
            return 0;
    
    r_sysfs:
            kobject_put(kobj_ref); 
            sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    
    r_device:
            class_destroy(dev_class);
    r_class:
            unregister_chrdev_region(dev,1);
            cdev_del(&etx_cdev);
            return -1;
    }

    /*
    ** Module exit function
    */
    static void __exit etx_driver_exit(void)
    {
            kobject_put(kobj_ref); 
            sysfs_remove_file(kernel_kobj, &etx_attr.attr);
            device_destroy(dev_class,dev);
            class_destroy(dev_class);
            cdev_del(&etx_cdev);
            unregister_chrdev_region(dev, 1);
            pr_info("Device Driver Remove...Done!!!\n");
    }
    
    module_init(etx_driver_init);
    module_exit(etx_driver_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
    MODULE_DESCRIPTION("Simple Linux device driver (sysfs)");
    MODULE_VERSION("1.8");
    ```

    test: `ls -l /sys/kernel`, `ls -l /sys/kernel/etx_sysfs`

    read and modify the value:

    ```bash
    sudo su
    cat /sys/kernel/etx_sysfs/etx_value
    echo 123 > /sys/kernel/etx_sysfs/etx_value
    cat /sys/kernel/etx_sysfs/etx_value
    ```

## wait queue

wait queue 类似于用户态编程的 condition variable，用于控制同步。


核心的几个类型和函数：

```c
wait_queue_head_t
```

通常使用一个额外的线程来处理 wait queue，防止 init module 之类被阻塞。

```c
#include <linux/kthread.h>

struct task_struct *wait_thread = kthread_create(print_msg, NULL, "print_msg");
if (wait_thread) {
    pr_info("wake up process\n");
    wake_up_process(wait_thread);
}
```

初始化 wait queue:

```c
wait_queue_head_t wq;
int condi = 1;

void init_wq() {
    init_waitqueue_head(&wq);
}

int print_msg(void *unused)
{
    pr_info("wait condition variable...\n");
    wait_event_interruptible(wq, condi == 2);
    pr_info("condi is %d, hello, world\n", condi);
    condi = 1;
    pr_info("rechange condi to 1\n");
    return 0;
}
```

触发：

```c
ssize_t h_write(struct file *file_ptr, const char __user *buf, size_t size, loff_t *offset)
{
    pr_info("in h_write()...\n");
    copy_from_user(&condi, buf, sizeof(condi));
    wake_up_interruptible(&wq);
    return sizeof(condi);
}
```

可以看到，当改变变量值后，还需要手动通知一下 wait queue，wait queue 才能判断是否往下走。

经实验，如果不满足`condi == 2`的条件，即使`wake_up_interruptible()`也无法让程序继续往下走。

详细的 example: 见`ref_13`

## Interrupts

### cache

* Interrupts Flags

    * `IRQF_DISABLED`

    * `IRQF_SAMPLE_RANDOM`

    * `IRQF_TIMER`

    * `IRQF_SHARED`

        this flag enables one irq number matches multiple irq handler.

        (do the handlers run one by one, or run simultaneously, or under a random order?)

* `request_irq()` cannot be called from interrupt context (other situations where code cannot block), because it can block.

* irq 将一个整数和一个 handler function 相 map。

    有一些整数的含义是被预定的，比如`1`代表的是键盘中断。

* what should be cautioned when writing interrupt handler

    * Interrupt handlers can not enter sleep, so to avoid calls to some functions which has sleep.

    * When the interrupt handler has part of the code to enter the critical section, use spinlocks lock, rather than mutexes. Because if it can’t take mutex it will go to sleep until it takes the mute.

    * Interrupt handlers can not exchange data with the userspace.The interrupt handlers must be executed as soon as possible. To ensure this, it is best to split the implementation into two parts, the top half and the bottom half. The top half of the handler will get the job done as soon as possible and then work late on the bottom half, which can be done with softirq or tasklet or workqueue.

    * Interrupt handlers can not be called repeatedly. When a handler is already executing, its corresponding IRQ must be disabled until the handler is done.

        maybe this means the hander can't be recursively invoked.

    * Interrupt handlers can be interrupted by higher authority handlers. If you want to avoid being interrupted by a highly qualified handler, you can mark the interrupt handler as a fast handler. However, if too many are marked as fast handlers, the performance of the system will be degraded because the interrupt latency will be longer.

* handler function syntax:

    ```c
    irqreturn_t irq_handler(int irq, void *dev_id, struct pt_regs *regs)
    ```

    * `dev_id`

        this pointer is used to identify different devices.

        when the interruption occurs, one irq number may be conrresponding to multiple devices.

        Or, in other words, one irq number is shared by multiple devices.

        Thus there must be a unique identification value to distinct different devices.

        A common practice is to use the driver's structure pointer.

    * return a `irqreturn_t` type value

        the return value `IRQ_HANDLED` means it process irq successfully.

        the return value `IRQ_NONE` means the handler function fails to process the irq.

special functions called interrupt handlers (ISR)

In Linux, interrupt signals are the distraction that diverts the processor to a new activity outside of the normal flow of execution. This new activity is called interrupt handler or interrupt service routine (ISR) and that distraction is Interrupts.

example:

a keyboard interrupter handler

```c
/*
 *  intrpt.c - An interrupt handler.
 *
 *  Copyright (C) 2001 by Peter Jay Salzman
 */

/* 
 * The necessary header files 
 */

/* 
 * Standard in kernel modules 
 */
#include <linux/kernel.h>	/* We're doing kernel work */
#include <linux/module.h>	/* Specifically, a module */
#include <linux/sched.h>
#include <linux/workqueue.h>
#include <linux/interrupt.h>	/* We want an interrupt */
#include <asm/io.h>

#define MY_WORK_QUEUE_NAME "WQsched.c"

static struct workqueue_struct *my_workqueue;

/* 
 * This will get called by the kernel as soon as it's safe
 * to do everything normally allowed by kernel modules.
 */
static void got_char(void *scancode)
{
	printk(KERN_INFO "Scan Code %x %s.\n",
	       (int)*((char *)scancode) & 0x7F,
	       *((char *)scancode) & 0x80 ? "Released" : "Pressed");
}

/* 
 * This function services keyboard interrupts. It reads the relevant
 * information from the keyboard and then puts the non time critical
 * part into the work queue. This will be run when the kernel considers it safe.
 */
irqreturn_t irq_handler(int irq, void *dev_id, struct pt_regs *regs)
{
	/* 
	 * This variables are static because they need to be
	 * accessible (through pointers) to the bottom half routine.
	 */
	static int initialised = 0;
	static unsigned char scancode;
	static struct work_struct task;
	unsigned char status;

	/* 
	 * Read keyboard status
	 */
	status = inb(0x64);
	scancode = inb(0x60);

	if (initialised == 0) {
		INIT_WORK(&task, (void(*)(struct work_struct *))got_char);
		// INIT_WORK(&task, got_char, &scancode);
		initialised = 1;
	} else {
		DECLARE_WORK(task, (void(*)(struct work_struct *))got_char);
		// PREPARE_WORK(&task, got_char);
		// PREPARE_WORK(&task, got_char, &scancode);
	}

	queue_work(my_workqueue, &task);

	return IRQ_HANDLED;
}

/* 
 * Initialize the module - register the IRQ handler 
 */
int init_module()
{
	my_workqueue = create_workqueue(MY_WORK_QUEUE_NAME);

	/* 
	 * Since the keyboard handler won't co-exist with another handler,
	 * such as us, we have to disable it (free its IRQ) before we do
	 * anything.  Since we don't know where it is, there's no way to
	 * reinstate it later - so the computer will have to be rebooted
	 * when we're done.
	 */
	free_irq(1, NULL);

	/* 
	 * Request IRQ 1, the keyboard IRQ, to go to our irq_handler.
	 * SA_SHIRQ means we're willing to have othe handlers on this IRQ.
	 * SA_INTERRUPT can be used to make the handler into a fast interrupt.
	 */
	return request_irq(1,	/* The number of the keyboard IRQ on PCs */
			   (void*)irq_handler,	/* our handler */
			   IRQF_SHARED, "test_keyboard_irq_handler",
			   (void *)(irq_handler));
}

/* 
 * Cleanup 
 */
void cleanup_module()
{
	/* 
	 * This is only here for completeness. It's totally irrelevant, since
	 * we don't have a way to restore the normal keyboard interrupt so the
	 * computer is completely useless and has to be rebooted.
	 */
	free_irq(1, NULL);
}

/* 
 * some work_queue related functions are just available to GPL licensed Modules
 */
MODULE_LICENSE("GPL");
```

打开`dmesg`日志，在 insmod 后，每次按下键盘都会打印中断函数的处理消息：

```
[ 1234.381119] intrpt: loading out-of-tree module taints kernel.
[ 1234.381123] intrpt: module verification failed: signature and/or required key missing - tainting kernel
[ 1234.381522] ------------[ cut here ]------------
[ 1234.381523] Trying to free already-free IRQ 1
[ 1234.381527] WARNING: CPU: 0 PID: 2691 at kernel/irq/manage.c:1893 __free_irq+0x1a6/0x310
[ 1234.381533] Modules linked in: intrpt(OE+) vboxsf intel_rapl_msr snd_intel8x0 intel_rapl_common snd_ac97_codec ac97_bus intel_uncore_frequency_common snd_pcm snd_seq_midi snd_seq_midi_event binfmt_misc snd_rawmidi crct10dif_pclmul polyval_clmulni polyval_generic ghash_clmulni_intel sha256_ssse3 sha1_ssse3 aesni_intel crypto_simd cryptd nls_iso8859_1 joydev snd_seq rapl input_leds snd_seq_device snd_timer vmwgfx drm_ttm_helper snd ttm serio_raw drm_kms_helper soundcore vboxguest mac_hid sch_fq_codel msr parport_pc ppdev lp parport drm efi_pstore ip_tables x_tables autofs4 hid_generic usbhid hid crc32_pclmul psmouse ahci libahci video i2c_piix4 e1000 wmi pata_acpi
[ 1234.381561] CPU: 0 PID: 2691 Comm: insmod Tainted: G           OE      6.5.0-28-generic #29~22.04.1-Ubuntu
[ 1234.381563] Hardware name: innotek GmbH VirtualBox/VirtualBox, BIOS VirtualBox 12/01/2006
[ 1234.381564] RIP: 0010:__free_irq+0x1a6/0x310
[ 1234.381566] Code: 50 32 00 00 49 8b be 88 01 00 00 e8 74 ec 02 00 49 8b 7f 30 e8 0b 9c 22 00 eb 35 8b 75 d0 48 c7 c7 40 2d d6 88 e8 5a af f4 ff <0f> 0b 48 8b 75 c8 4c 89 e7 e8 5c d6 f8 00 49 8b 46 40 48 8b 40 78
[ 1234.381568] RSP: 0018:ffffb0238399fac0 EFLAGS: 00010046
[ 1234.381569] RAX: 0000000000000000 RBX: 0000000000000000 RCX: 0000000000000000
[ 1234.381571] RDX: 0000000000000000 RSI: 0000000000000000 RDI: 0000000000000000
[ 1234.381571] RBP: ffffb0238399faf8 R08: 0000000000000000 R09: 0000000000000000
[ 1234.381572] R10: 0000000000000000 R11: 0000000000000000 R12: ffff95f3801688a4
[ 1234.381573] R13: ffff95f380168960 R14: ffff95f380168800 R15: ffff95f382d72b00
[ 1234.381574] FS:  00007d6eb84b3c40(0000) GS:ffff95f39bc00000(0000) knlGS:0000000000000000
[ 1234.381576] CS:  0010 DS: 0000 ES: 0000 CR0: 0000000080050033
[ 1234.381577] CR2: 00006172fb59f520 CR3: 000000002f81a000 CR4: 00000000000506f0
[ 1234.381580] Call Trace:
[ 1234.381581]  <TASK>
[ 1234.381583]  ? show_regs+0x6d/0x80
[ 1234.381587]  ? __warn+0x89/0x160
[ 1234.381589]  ? __free_irq+0x1a6/0x310
[ 1234.381591]  ? report_bug+0x17e/0x1b0
[ 1234.381594]  ? handle_bug+0x46/0x90
[ 1234.381597]  ? exc_invalid_op+0x18/0x80
[ 1234.381598]  ? asm_exc_invalid_op+0x1b/0x20
[ 1234.381602]  ? __free_irq+0x1a6/0x310
[ 1234.381604]  free_irq+0x32/0x80
[ 1234.381606]  ? __pfx_init_module+0x10/0x10 [intrpt]
[ 1234.381609]  init_module+0x39/0x70 [intrpt]
[ 1234.381612]  do_one_initcall+0x5e/0x340
[ 1234.381616]  do_init_module+0x68/0x260
[ 1234.381619]  load_module+0xb85/0xcd0
[ 1234.381621]  ? security_kernel_post_read_file+0x75/0x90
[ 1234.381624]  init_module_from_file+0x96/0x100
[ 1234.381626]  ? init_module_from_file+0x96/0x100
[ 1234.381629]  idempotent_init_module+0x11c/0x2b0
[ 1234.381631]  __x64_sys_finit_module+0x64/0xd0
[ 1234.381633]  do_syscall_64+0x5b/0x90
[ 1234.381636]  ? ksys_mmap_pgoff+0x120/0x270
[ 1234.381638]  ? exit_to_user_mode_prepare+0x30/0xb0
[ 1234.381639]  ? syscall_exit_to_user_mode+0x37/0x60
[ 1234.381641]  ? do_syscall_64+0x67/0x90
[ 1234.381642]  ? exit_to_user_mode_prepare+0x30/0xb0
[ 1234.381643]  ? syscall_exit_to_user_mode+0x37/0x60
[ 1234.381645]  ? do_syscall_64+0x67/0x90
[ 1234.381646]  entry_SYSCALL_64_after_hwframe+0x6e/0xd8
[ 1234.381648] RIP: 0033:0x7d6eb7d1e88d
[ 1234.381658] Code: 5b 41 5c c3 66 0f 1f 84 00 00 00 00 00 f3 0f 1e fa 48 89 f8 48 89 f7 48 89 d6 48 89 ca 4d 89 c2 4d 89 c8 4c 8b 4c 24 08 0f 05 <48> 3d 01 f0 ff ff 73 01 c3 48 8b 0d 73 b5 0f 00 f7 d8 64 89 01 48
[ 1234.381659] RSP: 002b:00007fff0d23a4d8 EFLAGS: 00000246 ORIG_RAX: 0000000000000139
[ 1234.381661] RAX: ffffffffffffffda RBX: 00006172fbf127a0 RCX: 00007d6eb7d1e88d
[ 1234.381662] RDX: 0000000000000000 RSI: 00006172fb5aacd2 RDI: 0000000000000003
[ 1234.381662] RBP: 0000000000000000 R08: 0000000000000000 R09: 0000000000000000
[ 1234.381663] R10: 0000000000000003 R11: 0000000000000246 R12: 00006172fb5aacd2
[ 1234.381664] R13: 00006172fbf12760 R14: 00006172fb5a9888 R15: 00006172fbf128b0
[ 1234.381665]  </TASK>
[ 1234.381666] ---[ end trace 0000000000000000 ]---
[ 1234.468732] Scan Code 40 Released.
[ 1237.372783] Scan Code 40 Released.
[ 1237.428975] Scan Code 40 Released.
[ 1237.429958] Scan Code 40 Released.
[ 1237.501143] Scan Code 40 Released.
[ 1237.548031] Scan Code 40 Released.
[ 1237.596222] Scan Code 40 Released.
[ 1237.612977] Scan Code 40 Released.
[ 1237.700977] Scan Code 40 Released.
[ 1237.716093] Scan Code 40 Released.
[ 1237.780960] Scan Code 40 Released.
[ 1237.851931] Scan Code 40 Released.
[ 1237.924812] Scan Code 40 Released.
[ 1237.947605] Scan Code 40 Released.
[ 1238.028965] Scan Code 40 Released.
[ 1238.029951] Scan Code 40 Released.
```

在执行`sudo rmmod intrpt`后，系统会直接卡死。因为我们在代码里覆盖了系统原本处理键盘中断的 handler。

具体的原因在程序注释里也有说明。

**Polling vs Interrupts**

* Polling

    In polling the CPU keeps on checking all the hardwares of the availablilty of any request

    The polling method is like a salesperson. The salesman goes from door to door while requesting to buy a product or service. Similarly, the controller keeps monitoring the flags or signals one by one for all devices and provides service to whichever component that needs its service.

* Interrupt

    In interrupt the CPU takes care of the hardware only when the hardware requests for some service

    An interrupt is like a shopkeeper. If one needs a service or product, he goes to him and apprises him of his needs. In case of interrupts, when the flags or signals are received, they notify the controller that they need to be serviced.

**Interrupts and Exceptions**

Exceptions are often discussed at the same time as interrupts. Unlike interrupts, exceptions occur synchronously with respect to the processor clock; they are often called synchronous interrupts. Exceptions are produced by the processor while executing instructions either in response to a programming error (e.g. divide by zero) or abnormal conditions that must be handled by the kernel (e.g. a page fault).

Interrupts – asynchronous interrupts generated by hardware.

Exceptions – synchronous interrupts generated by the processor.

（这里的同步和异步指的是时序，如果在非时钟周期内到达了中断信号，那么就称其为异步。）

**Maskable and Non-maskable**

Maskable – All Interrupt Requests (IRQs) issued by I/O devices give rise to maskable interrupts. A maskable interrupt can be in two states: masked or unmasked; a masked interrupt is ignored by the control unit as long as it remains masked.

Non-maskable – Only a few critical events (such as hardware failures) give rise to nonmaskable interrupts. Non-maskable interrupts are always recognized by the CPU.

**Exception types**

* Falts – Like Divide by zero, Page Fault, Segmentation Fault.

* Traps – Reported immediately following the execution of the trapping instruction. Like Breakpoints.

* Aborts – Aborts are used to report severe errors, such as hardware failures and invalid or inconsistent values in system tables.

**Interrupt handler**

For a device’s each interrupt, its device driver must register an interrupt handler.

An interrupt handler or interrupt service routine (ISR) is the function that the kernel runs in response to a specific interrupt:

1. Each device that generates interrupts has an associated interrupt handler.

1. The interrupt handler for a device is part of the device’s driver (the kernel code that manages the device).


In Linux, interrupt handlers are normal C functions, which match a specific prototype and thus enable the kernel to pass the handler information in a standard way. What differentiates interrupt handlers from other kernel functions is that the kernel invokes them in response to interrupts and that they run in a special context called interrupt context. This special context is occasionally called atomic context because code executing in this context is unable to block.

Because an interrupt can occur at any time, an interrupt handler can be executed at any time. It is imperative that the handler runs quickly, to resume the execution of the interrupted code as soon as possible. It is important that

1. To the hardware: the operating system services the interrupt without delay.
1. To the rest of the system: the interrupt handler executes in as short a period as possible.

Top halves and Bottom halves
Top half
The interrupt handler is the top half. The top half will run immediately upon receipt of the interrupt and performs only the work that is time-critical, such as acknowledging receipt of the interrupt or resetting the hardware.

Bottom half
The bottom half is used to process data, letting the top half to deal with new incoming interrupts. Interrupts are enabled when a bottom half runs. The interrupt can be disabled if necessary, but generally, this should be avoided as this goes against the basic purpose of having a bottom half – processing data while listening to new interrupts. The bottom half runs in the future, at a more convenient time, with all interrupts enabled.

比如网卡接收数据，我们使用 top half 快速地把网络数据包从网卡的缓冲区复制到内存中，然后使用 bottom half 慢慢处理内存中的数据包就可以了。如果 top half 不够快，那么新来的数据就会覆盖掉旧数据，造成读写错误。

Intel processors handle interrupt using IDT (Interrupt Descriptor Table).  The IDT consists of 256 entries with each entry corresponding to a vector and of 8 bytes. All the entries are a pointer to the interrupt handling function. The CPU uses IDTR to point to IDT. The relation between those two can be depicted as below,

Example:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>                 //kmalloc()
#include <linux/uaccess.h>              //copy_to/from_user()
#include <linux/sysfs.h> 
#include <linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/err.h>
#define IRQ_NO 11

//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq, void *dev_id)
{
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    return IRQ_HANDLED;
}

volatile int etx_value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);

/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);

/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);

struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
 
static struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .read           = etx_read,
    .write          = etx_write,
    .open           = etx_open,
    .release        = etx_release,
};
 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
    printk(KERN_INFO "Sysfs - Read!!!\n");
    return sprintf(buf, "%d", etx_value);
}

static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
    printk(KERN_INFO "Sysfs - Write!!!\n");
    sscanf(buf,"%d",&etx_value);
    return count;
}

static int etx_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Opened...!!!\n");
    return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Closed...!!!\n");
    return 0;
}
 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
    printk(KERN_INFO "Read function\n");
    asm("int $0x3B");  // Corresponding to irq 11
    return 0;
}

static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
    printk(KERN_INFO "Write Function\n");
    return len;
}
 
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0) {
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d \n", MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0) {
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
    }

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs", kernel_kobj);

    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)) {
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
    }
    if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
    }
    printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;

irq:
    free_irq(IRQ_NO,(void *)(irq_handler));

r_sysfs:
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
    class_destroy(dev_class);
r_class:
    unregister_chrdev_region(dev,1);
    cdev_del(&etx_cdev);
    return -1;
}
 
static void __exit etx_driver_exit(void)
{
    free_irq(IRQ_NO,(void *)(irq_handler));
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Interrupts");
MODULE_VERSION("1.9");
```

期望输出：

```
linux@embetronicx-VirtualBox: dmesg

[19743.366386] Major = 246 Minor = 0
[19743.370707] Device Driver Insert...Done!!!
[19745.580487] Device File Opened...!!!
[19745.580507] Read function
[19745.580531] Shared IRQ: Interrupt Occurred
[19745.580540] Device File Closed...!!!
```

实际输出：

```
[162342.126355] Major = 238 Minor = 0 
[162342.138918] Device Driver Insert...Done!!!
[162359.827734] Device File Opened...!!!
[162359.827746] Read function
[162359.827955] __common_interrupt: 2.59 No irq handler for vector
[162359.827974] Device File Closed...!!!
```

If you are using the newer Linux kernel, then this may not work properly. You may get something like below.

`do_IRQ: 1.59 No irq handler for vector`

In order to solve that, you have to change the Linux kernel source code, Compile it, then install it.

build:

```bash
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.47.tar.xz
sudo tar -xvf ../linux-5.4.47.tar
cd linux-5.4.47/
cp -v /boot/config-$(uname -r) .confi
sudo apt install build-essential kernel-package fakeroot libncurses5-dev libssl-dev ccache flex libelf-dev bison libncurses-dev
```

Add the below line in the downloaded Linux kernel file `arch/x86/kernel/irq.c` right after all the include lines.

`EXPORT_SYMBOL(vector_irq);`

```bash
make oldconfig
make menuconfig
sudo make  （也可以并行编译：sudo make -j 4）
sudo su
make modules_install
sudo make install
sudo update-initramfs -c -k 5.4.47
sudo update-grub
reboot
uname -r
```

新版本的 kernel 应该使用的 driver 代码：

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Interrupt Example
*
*  \author     EmbeTronicX
*
*  \Tested with kernel 5.4.47
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <asm/hw_irq.h>
#include <linux/err.h>
#define IRQ_NO 11
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
        struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
        struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
 
static struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .read           = etx_read,
    .write          = etx_write,
    .open           = etx_open,
    .release        = etx_release,
};
 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
    printk(KERN_INFO "Sysfs - Read!!!\n");
    return sprintf(buf, "%d", etx_value);
}
 
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
    printk(KERN_INFO "Sysfs - Write!!!\n");
    sscanf(buf,"%d",&etx_value);
    return count;
}
 
static int etx_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Opened...!!!\n");
    return 0;
}
 
static int etx_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Device File Closed...!!!\n");
    return 0;
}
 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
    struct irq_desc *desc;

    printk(KERN_INFO "Read function\n");
    desc = irq_to_desc(11);
    if (!desc) 
    {
        return -EINVAL;
    }
    __this_cpu_write(vector_irq[59], desc);
    asm("int $0x3B");  // Corresponding to irq 11
    return 0;
}

static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
    printk(KERN_INFO "Write Function\n");
    return len;
}
 
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0){
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
    }

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
    }
    if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
    }
    printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;
 
irq:
    free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
    class_destroy(dev_class);
r_class:
    unregister_chrdev_region(dev,1);
    cdev_del(&etx_cdev);
    return -1;
}
 
static void __exit etx_driver_exit(void)
{
    free_irq(IRQ_NO,(void *)(irq_handler));
    kobject_put(kobj_ref); 
    sysfs_remove_file(kernel_kobj, &etx_attr.attr);
    device_destroy(dev_class,dev);
    class_destroy(dev_class);
    cdev_del(&etx_cdev);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Interrupts");
MODULE_VERSION("1.9");
```

### Work queue

Work queues defer work into a kernel thread; this bottom half always runs in the process context.

Workqueue 将工作推迟到一个内核线程中，这个底层部分总是运行在进程的上下文环境中。

Workqueue is allowing users to create a kernel thread and bind work to the kernel thread.

So, this will run in the process context and the work queue can sleep.

（什么是 process context ? 为什么 kernel thread 会运行在 process context 中？）

Normally, it is easy to decide between using workqueue and softirq/tasklet:

If the deferred work needs to sleep, then workqueue is used.
If the deferred work need not sleep, then softirq or tasklet are used.

注：

* 什么是 softirq？

There are two ways to implement Workqueue in the Linux kernel.

Using global workqueue (Static / Dynamic)
Creating Own workqueue (We will see in the next tutorial)

**Initialize work using Static Method**

The below call creates a workqueue by the name and the function that we are passing in the second argument gets scheduled in the queue.

`DECLARE_WORK(name, void (*func)(void *))`

Where,

* `name`: The name of the “work_struct” structure that has to be created.

* `func`: The function to be scheduled in this workqueue.

Example:

```c
DECLARE_WORK(workqueue,workqueue_fn);
```

Schedule work to the Workqueue
The below functions are used to allocate the work to the queue.

`schedule_work`

This function puts a job in the kernel-global workqueue if it was not already queued and leaves it in the same position on the kernel-global workqueue otherwise.

int schedule_work( struct work_struct *work );

where,

work – job to be done

Returns zero if work was already on the kernel-global workqueue and non-zero otherwise.

Scheduled_delayed_work
After waiting for a given time this function puts a job in the kernel-global workqueue.

int scheduled_delayed_work( struct delayed_work *dwork, unsigned long delay );

where,

dwork – job to be done

delay– number of jiffies to wait or 0 for immediate execution

Schedule_work_on
This puts a job on a specific CPU.

int schedule_work_on( int cpu, struct work_struct *work );

where,

cpu– CPU to put the work task on

work– job to be done

Scheduled_delayed_work_on
After waiting for a given time this puts a job in the kernel-global workqueue on the specified CPU.

int scheduled_delayed_work_on(int cpu, struct delayed_work *dwork, unsigned long delay );
where,

cpu – CPU to put the work task on

dwork – job to be done

delay– number of jiffies to wait or 0 for immediate execution

Delete work from workqueue
There are also a number of helper functions that you can use to flush or cancel work on work queues. To flush a particular work item and block until the work is complete, you can make a call to flush_work. All work on a given work queue can be completed using a call to flush_work. In both cases, the caller blocks until the operation are complete. To flush the kernel-global work queue, call flush_scheduled_work.

`int flush_work( struct work_struct *work );`
`void flush_scheduled_work( void );`
Cancel Work from workqueue
You can cancel work if it is not already executing in a handler. A call to cancel_work_sync will terminate the work in the queue or block until the callback has finished (if the work is already in progress in the handler). If the work is delayed, you can use a call to cancel_delayed_work_sync.

`int cancel_work_sync( struct work_struct *work );`
`int cancel_delayed_work_sync( struct delayed_work *dwork );`
Check the workqueue
Finally, you can find out whether a work item is pending (not yet executed by the handler) with a call to work_pending or delayed_work_pending.

`work_pending( work );`

`delayed_work_pending( work );`

Example:

```c
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>                 //kmalloc()
#include <linux/uaccess.h>              //copy_to/from_user()
#include <linux/sysfs.h> 
#include <linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
 
void workqueue_fn(struct work_struct *work); 
 
/*Creating work by Static Method */
DECLARE_WORK(workqueue,workqueue_fn);
 
/*Workqueue Function*/
void workqueue_fn(struct work_struct *work)
{
    printk(KERN_INFO "Executing Workqueue Function\n");
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
    printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
    schedule_work(&workqueue);
    
    return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This function will be called when we read the sysfs file
*/ 
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This function will be called when we write the sysfsfs file
*/
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This function will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return len;
}
 
/*
** Module Init function
*/
static int __init etx_driver_init(void)
{
    /*Allocating Major number*/
    if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
            printk(KERN_INFO "Cannot allocate major number\n");
            return -1;
    }
    printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

    /*Creating cdev structure*/
    cdev_init(&etx_cdev,&fops);

    /*Adding character device to the system*/
    if((cdev_add(&etx_cdev,dev,1)) < 0){
        printk(KERN_INFO "Cannot add the device to the system\n");
        goto r_class;
    }

    /*Creating struct class*/
    if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
        printk(KERN_INFO "Cannot create the struct class\n");
        goto r_class;
    }

    /*Creating device*/
    if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
        printk(KERN_INFO "Cannot create the Device 1\n");
        goto r_device;
    }

    /*Creating a directory in /sys/kernel/ */
    kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);

    /*Creating sysfs file for etx_value*/
    if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
            printk(KERN_INFO"Cannot create sysfs file......\n");
            goto r_sysfs;
    }
    if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
        printk(KERN_INFO "my_device: cannot register IRQ ");
                goto irq;
    }
    printk(KERN_INFO "Device Driver Insert...Done!!!\n");
    return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Static method)");
MODULE_VERSION("1.10");
```

看起来似乎是`irq_handler()`会调用`schedule_work()`，把繁重的工作分配到 bottom half 去做。推测：`schedule_work()`一定是非阻塞的。

test:

```bash
sudo cat /dev/etx_device
sudo dmesg
```

#### Initialize work using Dynamic Method

The below call (`INIT_WORK`) creates a workqueue in Linux by the name work and the function that gets scheduled in the queue is work_fn.

`INIT_WORK(work,work_fn)`

Where,

`name`: The name of the “work_struct” structure that has to be created.
`func`: The function to be scheduled in this workqueue.

**Schedule work to the Workqueue**
The below functions used to allocate the work to the queue.

`Schedule_work`

This function puts a job in the kernel-global workqueue if it was not already queued and leaves it in the same position on the kernel-global workqueue otherwise.

`int schedule_work( struct work_struct *work );`

where,

`work` – job to be done

Returns zero if work was already on the kernel-global workqueue and non-zero otherwise.

`Scheduled_delayed_work`

After waiting for a given time this function puts a job in the kernel-global workqueue.

`int scheduled_delayed_work( struct delayed_work *dwork, unsigned long delay );`

where,

`dwork` – job to be done

`delay` – number of jiffies to wait or 0 for immediate execution

`Schedule_work_on`

This puts a job on a specific CPU.

`int schedule_work_on( int cpu, struct work_struct *work );`

where,

`cpu` – CPU to put the work task on

`work` – job to be done

`Scheduled_delayed_work_on`

After waiting for a given time this puts a job in the kernel-global workqueue on the specified CPU.

`int scheduled_delayed_work_on(int cpu, struct delayed_work *dwork, unsigned long delay );`

where,

`cpu` – CPU to put the work task on

`dwork` – job to be done

`delay` – number of jiffies to wait or 0 for immediate execution

**Delete work from workqueue**

There are also a number of helper functions that you can use to flush or cancel work on work queues. To flush a particular work item and block until the work is complete, you can make a call to flush_work. All work on a given work queue can be completed using a call to . In both cases, the caller blocks until the operation are complete. To flush the kernel-global work queue, call flush_scheduled_work.

`int flush_work( struct work_struct *work );`

`void flush_scheduled_work( void );`

**Cancel Work from workqueue**

You can cancel work if it is not already executing in a handler. A call to cancel_work_sync will terminate the work in the queue or block until the callback has finished (if the work is already in progress in the handler). If the work is delayed, you can use a call to cancel_delayed_work_sync.

`int cancel_work_sync( struct work_struct *work );`

`int cancel_delayed_work_sync( struct delayed_work *dwork );`

**Check workqueue**

Finally, you can find out whether a work item is pending (not yet executed by the handler) with a call to work_pending or delayed_work_pending.

`work_pending( work );`
`delayed_work_pending( work );`

Example:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Global Workqueue - Dynamic method)
*
*  \author     EmbeTronicX
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
/* Work structure */
static struct work_struct workqueue;
 
void workqueue_fn(struct work_struct *work); 
 
/*Workqueue Function*/
void workqueue_fn(struct work_struct *work)
{
        printk(KERN_INFO "Executing Workqueue Function\n");
}
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred");
        /*Allocating work to queue*/
        schedule_work(&workqueue);
        
        return IRQ_HANDLED;
}
 
volatile int etx_value = 0;
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
*/ 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}
 
/*
** This fuction will be called when we write the sysfsfs file
*/
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This fuction will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return 0;
}
 
/*
** Module Init function
*/ 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ ");
                    goto irq;
        }
 
        /*Creating work by Dynamic Method */
        INIT_WORK(&workqueue,workqueue_fn);
 
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Global Workqueue - Dynamic method)");
MODULE_VERSION("1.11");
```

The core workqueue is represented by structure struct workqueue_struct, which is the structure onto which work is placed. This work is added to the queue in the top half (Interrupt context) and the execution of this work happened in the bottom half (Kernel context).

The work is represented by structure struct `work_struct`, which identifies the work and the deferral function.

**Create and destroy workqueue structure**

Workqueues are created through a macro called create_workqueue, which returns a workqueue_struct reference. You can remove this workqueue later (if needed) through a call to the destroy_workqueue function.

`struct workqueue_struct *create_workqueue( name );`

`void destroy_workqueue( struct workqueue_struct * );`

You should use create_singlethread_workqueue() for creating a workqueue when you want to create only a single thread for all the processors.

Since create_workqueue and create_singlethread_workqueue() are macros. Both are using the alloc_workqueue function in the background.

```c
#define create_workqueue(name)                    
        alloc_workqueue("%s", WQ_MEM_RECLAIM, 1, (name))
#define create_singlethread_workqueue(name)       
        alloc_workqueue("%s", WQ_UNBOUND | WQ_MEM_RECLAIM, 1, (name))
```

alloc_workqueue
Allocate a workqueue with the specified parameters.

alloc_workqueue ( fmt, flags, max_active );

fmt– printf format for the name of the workqueue

flags – WQ_* flags

max_active – max in-flight work items, 0 for default

This will return Pointer to the allocated workqueue on success, NULL on failure.

WQ_* flags
This is the second argument of alloc_workqueue.

WQ_UNBOUND

Work items queued to an unbound wq are served by the special worker-pools which host workers who are not bound to any specific CPU. This makes the wq behave like a simple execution context provider without concurrency management. The unbound worker-pools try to start the execution of work items as soon as possible. Unbound wq sacrifices locality but is useful for the following cases.

* Wide fluctuation in the concurrency level requirement is expected and using bound wq may end up creating a large number of mostly unused workers across different CPUs as the issuer hops through different CPUs.

* Long-running CPU-intensive workloads which can be better managed by the system scheduler.

WQ_FREEZABLE

A freezable wq participates in the freeze phase of the system suspend operations. Work items on the wq are drained and no new work item starts execution until thawed.

WQ_MEM_RECLAIM

All wq which might be used in the memory reclaim paths MUST have this flag set. The wq is guaranteed to have at least one execution context regardless of memory pressure.

WQ_HIGHPRI

Work items of a highpri wq are queued to the highpri worker-pool of the target CPU. Highpri worker-pools are served by worker threads with elevated nice levels.

Note that normal and highpri worker-pools don’t interact with each other. Each maintains its separate pool of workers and implements concurrency management among its workers.

WQ_CPU_INTENSIVE

Work items of a CPU intensive wq do not contribute to the concurrency level. In other words, runnable CPU-intensive work items will not prevent other work items in the same worker pool from starting execution. This is useful for bound work items that are expected to hog CPU cycles so that their execution is regulated by the system scheduler.

Although CPU-intensive work items don’t contribute to the concurrency level, the start of their executions is still regulated by the concurrency management and runnable non-CPU-intensive work items can delay the execution of CPU-intensive work items.

This flag is meaningless for unbound wq.

Queuing Work to workqueue
With the work structure initialized, the next step is enqueuing the work on a workqueue. You can do this in a few ways.

queue_work
This will queue the work to the CPU on which it was submitted, but if the CPU dies it can be processed by another CPU.
int queue_work( struct workqueue_struct *wq, struct work_struct *work );

Where,

wq – workqueue to use

work – work to queue

It returns false if work was already on a queue, true otherwise.

queue_work_on
This puts work on a specific CPU.
int queue_work_on( int cpu, struct workqueue_struct *wq, struct work_struct *work );

Where,

cpu– cpu to put the work task on

wq – workqueue to use

work– job to be done

`queue_delayed_work`
After waiting for a given time this function puts work in the workqueue.

```c
int queue_delayed_work( struct workqueue_struct *wq,struct delayed_work *dwork, unsigned long delay );
```

Where,

`wq` – workqueue to use

`dwork` – work to queue

`delay`– number of jiffies to wait before queueing or 0 for immediate execution

`queue_delayed_work_on`

After waiting for a given time this puts a job in the workqueue on the specified CPU.

```c
int queue_delayed_work_on( int cpu, struct workqueue_struct *wq,struct delayed_work *dwork, unsigned long delay );
```

Where,

`cpu` – CPU to put the work task on

`wq` – workqueue to use

`dwork` – work to queue

`delay` – number of jiffies to wait before queueing or 0 for immediate execution

Full code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Own Workqueue)
*
*  \author     EmbeTronicX
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
static struct workqueue_struct *own_workqueue;
 
static void workqueue_fn(struct work_struct *work); 
 
static DECLARE_WORK(work, workqueue_fn);
 
 
/*Workqueue Function*/
static void workqueue_fn(struct work_struct *work)
{
    printk(KERN_INFO "Executing Workqueue Function\n");
    return;
        
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred\n");
        /*Allocating work to queue*/
        queue_work(own_workqueue, &work);
        
        return IRQ_HANDLED;
}
 
 
volatile int etx_value = 0;
 
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;

/*
** Function Prototypes
*/ 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);

/*
** File operation sturcture
*/
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/  
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This fuction will be called when we write the sysfsfs file
*/ 
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        sscanf(buf,"%d",&etx_value);
        return count;
}

/*
** This fuction will be called when we open the Device file
*/ 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/  
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Read function\n");
        asm("int $0x3B");  // Corresponding to irq 11
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        return 0;
}
 
/*
** Module Init function
*/ 
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device 1\n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file for etx_value*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ \n");
                    goto irq;
        }
 
        /*Creating workqueue */
        own_workqueue = create_workqueue("own_wq");
        
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/ 
static void __exit etx_driver_exit(void)
{
        /* Delete workqueue */
        destroy_workqueue(own_workqueue);
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("Simple Linux device driver (Own Workqueue)");
MODULE_VERSION("1.12");
```

test:

```bash
sudo cat /dev/etx_device
dmesg
```

Use `ps -aef` command to see our workqueue. You can able to see our workqueue which is `own_wq`.

**Difference between Schedule_work and queue_work**

If you want to use your own dedicated workqueue you should create a workqueue using create_workqueue. At that time you need to put work on your workqueue by using queue_work function.

If you don’t want to create any own workqueue, you can use kernel global workqueue. In that condition, you can use schedule_work function to put your work to global workqueue.

## Linked list

Linux implement a Doubly Linked List, which is defined in `/lib/modules/$(uname -r)/build/include/linux/list.h`.

```c
#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD(name) \
    struct list_head name = LIST_HEAD_INIT(name)
struct list_head {
    struct list_head *next;
    struct list_head *prev;
};
```

Usage:

```c
#include <linux/list.h>

struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};
```

Create a linked list head:

```c
LIST_HEAD(linked_list);
```

This macro will create the head node structure in the name of “linked_list” and it will initialize that to its own address.

Example:

```c
struct list_head etx_linked_list = { &etx_linked_list , &etx_linked_list};
```

While creating the head node, it initializes the prev and next pointer to its own address. This means that prev and next pointer points to itself. The node is empty If the node’s prev and next pointer points to itself.

（不是很懂，创建了 head note 后，不应该已经有一个 node 了吗，为什么还说他是空的？猜测这里的 head 是一个 dummy head）

**Create Node in Linked List**

```c
INIT_LIST_HEAD(struct list_head *list);
```

可以用上面的宏进行初始化链表。

Example:

```c
struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};

struct my_list new_node;

INIT_LIST_HEAD(&new_node.list);
new_node.data = 10;
```

**Add Node to Linked List**

* Add after Head Node

    Insert a new entry after the specified head:

    `inline void list_add(struct list_head *new, struct list_head *head);`

    Where,

    `struct list_head * new` – the new entry to be added

    `struct list_head * head` – list head to add it after

    Example:

    ```c
    list_add(&new_node.list, &etx_linked_list);
    ```

* Add before Head Node

    Insert a new entry before the specified head：

    `inline void list_add_tail(struct list_head *new, struct list_head *head);`

    Where,

    `struct list_head * new` – a new entry to be added

    `struct list_head * head` – list head to add before the head

    Example:

    ```c
    list_add_tail(&new_node.list, &etx_linked_list);
    ```

* Delete Node from Linked List

    * `list_del`

        It will delete the entry node from the list. This function removes the entry node from the linked list by disconnecting prev and next pointers from the list, but it doesn’t free any memory space allocated for the entry node.

        `inline void list_del(struct list_head *entry);`

        Where,

        `struct list_head * entry` – the element to delete from the list.

    * `list_del_init`

        It will delete the entry node from the list and reinitialize it. This function removes the entry node from the linked list by disconnecting prev and next pointers from the list, but it doesn’t free any memory space allocated for the entry node.

        `inline void list_del_init(struct list_head *entry);`

        Where,

        `struct list_head * entry` – the element to delete from the list.

* Replace Node in Linked List

    * `list_replace`

        This function is used to replace the old node with the new node.

        `inline void list_replace(struct list_head *old, struct list_head *new);`

        Where,

        `struct list_head * old` – the element to be replaced

        `struct list_head * new` – the new element to insert

        If old was empty, it will be overwritten.

    * `list_replace_init`

        This function is used to replace the old node with the new node and reinitialize the old entry.

        `inline void list_replace_init(struct list_head *old, struct list_head *new);`

        Where,

       `struct list_head * old` – the element to be replaced

       `struct list_head * new` – the new element to insert

        If old was empty, it will be overwritten.

* Moving Node in Linked List

    * `list_move`

        This will delete one list from the linked list and again adds to it after the head node.

        inline void list_move(struct list_head *list, struct list_head *head);

        Where,

        `struct list_head * list` – the entry to move

        `struct list_head * head` – the head that will precede our entry

    * `list_move_tail`

        This will delete one list from the linked list and again adds it before the head node.

        inline void list_move_tail(struct list_head *list, struct list_head *head);

        Where,

        `struct list_head * list` – the entry to move

        `struct list_head * head` – the head that will precede our entry

* Rotate Node in Linked List

    This will rotate the list to the left.

    `inline void list_rotate_left(struct list_head *head);`

    Where,

    head – the head of the list

* Test the Linked List Entry

    * `list_is_last`

        This tests whether list is the last entry in the list head.

        inline int list_is_last(const struct list_head *list, const struct list_head *head);

            Where,

        const struct list_head * list – the entry to test

        const struct list_head * head – the head of the list

        It returns 1 if it is the last entry otherwise 0.

    * `list_empty`

        It tests whether a list is empty or not.

        `inline int list_empty(const struct list_head *head);`

        Where,

        `const struct list_head * head` – the head of the list

        It returns 1 if it is empty otherwise 0.

    * `list_is_singular`

        This will test whether a list has just one entry.

        `inline int list_is_singular(const struct list_head *head);`

        Where,

        const struct list_head * head – the head of the list

        It returns 1 if it has only one entry otherwise 0.

* Split Linked List into two parts

    This cut a list into two.
    This helper moves the initial part of head, up to and including entry, from head to list. You should pass on entry an element you know is on head. list should be an empty list or a list you do not care about losing its data.

    inline void list_cut_position(struct list_head *list, struct list_head *head, struct list_head *entry);

    Where,

    struct list_head * list – a new list to add all removed entries

    struct list_head * head– a list with entries

    struct list_head * entry– an entry within the head could be the head itself and if so we won’t cut the list

* Join Two Linked Lists

    This will join two lists, this is designed for stacks.
    inline void list_splice(const struct list_head *list, struct list_head *head);

    Where,

    const struct list_head * list – the new list to add.

    struct list_head * head – the place to add it in the first list.

* Traverse Linked List

    * `list_entry`

        This macro is used to get the struct for this entry.
        
        list_entry(ptr, type, member);

        ptr– the struct list_head pointer.

        type – the type of the struct this is embedded in.

        member – the name of the list_head within the struct.

    * `list_for_each`

        This macro is used to iterate over a list.
        
        list_for_each(pos, head);

        pos –  the &struct list_head to use as a loop cursor.

        head –  the head for your list.

    * `list_for_each_entry`

        This is used to iterate over a list of the given type.

        ```c
        list_for_each_entry(pos, head, member);
        ```

        pos – the type * to use as a loop cursor.

        head – the head for your list.

        member – the name of the list_head within the struct.

    * `list_for_each_entry_safe`

        This will iterate over the list of given type-safe against the removal of list entry.

        `list_for_each_entry_safe ( pos, n, head, member);`

        Where,

        pos – the type * to use as a loop cursor.

        n – another type * to use as temporary storage

        head – the head for your list.

        member – the name of the list_head within the struct.

    * `list_for_each_prev`

        This will be used to iterate over a list backward.

        list_for_each_prev(pos, head);

        pos – the &struct list_head to use as a loop cursor.

        head – the head for your list.

    * `list_for_each_entry_reverse`

        This macro is used to iterate backward over the list of the given type.
        
        list_for_each_entry_reverse(pos, head, member);

        pos – the type * to use as a loop cursor.

        head  the head for your list.

        member – the name of the list_head within the struct.

1. When we write the value to our device file using echo value > /dev/etx_value, it will invoke the interrupt. Because we configured the interrupt by using the software. If you don’t know how it works, please refer to this tutorial.

1. The interrupt will invoke the ISR function.

1. In ISR we are allocating work to the Workqueue.

1. Whenever Workqueue executes, we are creating the Linked List Node and adding the Node to the Linked List.

1. When we are reading the driver using cat /dev/etx_device, printing all the nodes which are present in the Linked List using traverse.

1. When we are removing the driver using rmmod, it will remove all the nodes in Linked List and free the memory.

Creating Head Node

```c
/*Declare and init the head node of the linked list*/
LIST_HEAD(Head_Node);
```

This will create the head node in the name of `Head_Node` and initialize that.

Creating Node and add that into Linked List

```c
/*Creating Node*/
temp_node = kmalloc(sizeof(struct my_list), GFP_KERNEL);

/*Assgin the data that is received*/
temp_node->data = etx_value;

/*Init the list within the struct*/
INIT_LIST_HEAD(&temp_node->list);

/*Add Node to Linked List*/
list_add_tail(&temp_node->list, &Head_Node);
```

Traversing Linked List:

```c
struct my_list *temp;
int count = 0;
printk(KERN_INFO "Read function\n");

/*Traversing Linked List and Print its Members*/
list_for_each_entry(temp, &Head_Node, list) {
    printk(KERN_INFO "Node %d data = %d\n", count++, temp->data);
}

printk(KERN_INFO "Total Nodes = %d\n", count);
```

Deleting Linked List:

```c
/* Go through the list and free the memory. */
struct my_list *cursor, *temp;
list_for_each_entry_safe(cursor, temp, &Head_Node, list) {
    list_del(&cursor->list);
    kfree(cursor);
}
```

source code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Kernel Linked List)
*
*  \author     EmbeTronicX
*
* *******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include<linux/sysfs.h> 
#include<linux/kobject.h> 
#include <linux/interrupt.h>
#include <asm/io.h>
#include <linux/workqueue.h>            // Required for workqueues
#include <linux/err.h>
 
#define IRQ_NO 11
 
volatile int etx_value = 0;
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
struct kobject *kobj_ref;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
static struct workqueue_struct *own_workqueue;
 
 
static void workqueue_fn(struct work_struct *work); 
 
static DECLARE_WORK(work, workqueue_fn);
 
/*Linked List Node*/
struct my_list{
     struct list_head list;     //linux kernel list implementation
     int data;
};
 
/*Declare and init the head node of the linked list*/
LIST_HEAD(Head_Node);
 
/*
** Function Prototypes
*/ 
/*************** Driver Fuctions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 
/*************** Sysfs Fuctions **********************/
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf);
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count);
 
struct kobj_attribute etx_attr = __ATTR(etx_value, 0660, sysfs_show, sysfs_store);
/******************************************************/
 
 
/*Workqueue Function*/
static void workqueue_fn(struct work_struct *work)
{
        struct my_list *temp_node = NULL;
 
        printk(KERN_INFO "Executing Workqueue Function\n");
        
        /*Creating Node*/
        temp_node = kmalloc(sizeof(struct my_list), GFP_KERNEL);
 
        /*Assgin the data that is received*/
        temp_node->data = etx_value;
 
        /*Init the list within the struct*/
        INIT_LIST_HEAD(&temp_node->list);
 
        /*Add Node to Linked List*/
        list_add_tail(&temp_node->list, &Head_Node);
}
 
 
//Interrupt handler for IRQ 11. 
static irqreturn_t irq_handler(int irq,void *dev_id) {
        printk(KERN_INFO "Shared IRQ: Interrupt Occurred\n");
        /*Allocating work to queue*/
        queue_work(own_workqueue, &work);
        
        return IRQ_HANDLED;
}

/*
** File operation sturcture
*/ 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This fuction will be called when we read the sysfs file
*/  
static ssize_t sysfs_show(struct kobject *kobj, 
                struct kobj_attribute *attr, char *buf)
{
        printk(KERN_INFO "Sysfs - Read!!!\n");
        return sprintf(buf, "%d", etx_value);
}

/*
** This fuction will be called when we write the sysfsfs file
*/  
static ssize_t sysfs_store(struct kobject *kobj, 
                struct kobj_attribute *attr,const char *buf, size_t count)
{
        printk(KERN_INFO "Sysfs - Write!!!\n");
        return count;
}

/*
** This fuction will be called when we open the Device file
*/ 
static int etx_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Opened...!!!\n");
        return 0;
}

/*
** This fuction will be called when we close the Device file
*/   
static int etx_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "Device File Closed...!!!\n");
        return 0;
}

/*
** This fuction will be called when we read the Device file
*/ 
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        struct my_list *temp;
        int count = 0;
        printk(KERN_INFO "Read function\n");
 
        /*Traversing Linked List and Print its Members*/
        list_for_each_entry(temp, &Head_Node, list) {
            printk(KERN_INFO "Node %d data = %d\n", count++, temp->data);
        }
 
        printk(KERN_INFO "Total Nodes = %d\n", count);
        return 0;
}

/*
** This fuction will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        printk(KERN_INFO "Write Function\n");
        /*Copying data from user space*/
        sscanf(buf,"%d",&etx_value);
        /* Triggering Interrupt */
        asm("int $0x3B");  // Corresponding to irq 11
        return len;
}
 
/*
** Module Init function
*/  
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                printk(KERN_INFO "Cannot allocate major number\n");
                return -1;
        }
        printk(KERN_INFO "Major = %d Minor = %d n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            printk(KERN_INFO "Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            printk(KERN_INFO "Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            printk(KERN_INFO "Cannot create the Device \n");
            goto r_device;
        }
 
        /*Creating a directory in /sys/kernel/ */
        kobj_ref = kobject_create_and_add("etx_sysfs",kernel_kobj);
 
        /*Creating sysfs file*/
        if(sysfs_create_file(kobj_ref,&etx_attr.attr)){
                printk(KERN_INFO"Cannot create sysfs file......\n");
                goto r_sysfs;
        }
        if (request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "etx_device", (void *)(irq_handler))) {
            printk(KERN_INFO "my_device: cannot register IRQ \n");
                    goto irq;
        }
 
        /*Creating workqueue */
        own_workqueue = create_workqueue("own_wq");
        
        printk(KERN_INFO "Device Driver Insert...Done!!!\n");
        return 0;
 
irq:
        free_irq(IRQ_NO,(void *)(irq_handler));
 
r_sysfs:
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module Exit function
*/ 
static void __exit etx_driver_exit(void)
{
 
        /* Go through the list and free the memory. */
        struct my_list *cursor, *temp;
        list_for_each_entry_safe(cursor, temp, &Head_Node, list) {
            list_del(&cursor->list);
            kfree(cursor);
        }
 
        /* Delete workqueue */
        destroy_workqueue(own_workqueue);
        free_irq(IRQ_NO,(void *)(irq_handler));
        kobject_put(kobj_ref); 
        sysfs_remove_file(kernel_kobj, &etx_attr.attr);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_INFO "Device Driver Remove...Done!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Kernel Linked List");
MODULE_VERSION("1.13");
```

test:

```bash
cat /dev/etx_device
echo 10 > /dev/etx_device
```

## Thread

There are two types of threads.

1. User Level Thread

    In this type, the kernel is not aware of these threads. Everything is maintained by the user thread library. That thread library contains code for creating and destroying threads, for passing messages and data between threads, for scheduling thread execution, and for saving and restoring thread contexts. So all will be in User Space.

1. Kernel Level Thread

    Kernel level threads are managed by the OS, therefore, thread operations are implemented in the kernel code. There is no thread management code in the application area.

kernel level thread

header file: `linux/kthread.h`

Create Kernel Thread

* `kthread_create`

    Syntax:

    ```c
    struct task_struct * kthread_create (int (* threadfn(void *data), void *data, const char namefmt[], ...);
    ```

    Where,

    `threadfn` – the function to run until signal_pending(current).

    `data` – data ptr for threadfn.

    `namefmt[]` – printf-style name for the thread.

    `...` – variable arguments

    This helper function creates and names a kernel thread. But we need to wake up that thread manually. When woken, the thread will run `threadfn()` with data as its argument.

    `threadfn` can either call `do_exit` directly if it is a standalone thread for which no one will call `kthread_stop`, or return when ‘`kthread_should_stop`‘ is true (which means `kthread_stop` has been called). The return value should be zero or a negative error number; it will be passed to `kthread_stop`.

    It Returns `task_struct` or `ERR_PTR(-ENOMEM)`.

Start Kernel Thread

* `wake_up_process`

    Syntax:

    ```c
    int wake_up_process (struct task_struct * p);
    ```

    p – The process to be woken up.

    Attempt to wake up the nominated process and move it to the set of runnable processes.

    It returns 1 if the process was woken up, 0 if it was already running.

    It may be assumed that this function implies a write memory barrier before changing the task state if and only if any tasks are woken up.

Stop Kernel Thread

* `kthread_stop`

    Syntax:

    ```c
    int kthread_stop ( struct task_struct *k);
    ```

    Where,

    `k` – thread created by kthread_create.

    Sets kthread_should_stop for k to return true, wakes it and waits for it to exit. Your threadfn must not call do_exit itself, if you use this function! This can also be called after kthread_create instead of calling wake_up_process: the thread will exit without calling threadfn.

    It Returns the result of threadfn, or –EINTR if wake_up_process was never called.

Other functions in Kernel Thread

* `kthread_should_stop`

    ```c
    int kthread_should_stop (void);
    ```

    When someone calls `kthread_stop` on your kthread, it will be woken and this will return `true`. You should then return, and your return value will be passed through to `kthread_stop`.

    相当于由外部通知当前线程可以结束了。这种常见的场景，比如执行`ping`命令，默认一直发送 icmp 包，如果没有外部`Ctrl + C`信号，则不会主动停止。

* `kthread_bind`

    ```c
    void kthread_bind(struct task_struct *k, unsigned int cpu);
    ```

    `k` – thread created by kthread_create.

    `cpu` – CPU (might not be online, must be possible) for k to run on.


Thread Function

First, we have to create our thread that has the argument of void *  and should return int value.  We should follow some conditions in our thread function. It is advisable.

* If that thread is a long run thread, we need to check `kthread_should_stop()` every time, because any function may call kthread_stop. If any function called kthread_stop, that time `kthread_should_stop` will return true. We have to exit our thread function if true value been returned by `kthread_should_stop`.

* But if your thread function is not running long, then let that thread finish its task and kill itself using do_exit.

Example:

```c
int thread_function(void *pv) 
{
    int i=0;
    while(!kthread_should_stop())
    {
        printk(KERN_INFO "In EmbeTronicX Thread Function %d\n", i++);
        msleep(1000);
    } 
    return 0; 
}
```

```c
static struct task_struct *etx_thread; 

etx_thread = kthread_create(thread_function,NULL,"eTx Thread");

if (etx_thread) 
{
    wake_up_process(etx_thread); 
} 
else 
{
    printk(KERN_ERR "Cannot create kthread\n"); 
}
```

`kthread_run`:

Syntax:

```c
kthread_run (threadfn, data, namefmt, ...);
```

Where,

`threadfn` – the function to run until signal_pending(current).

`data` – data ptr for threadfn.

`namefmt` – printf-style name for the thread.

`...` – variable arguments

Convenient wrapper for `kthread_create` followed by `wake_up_process`.

It returns the `kthread` or `ERR_PTR(-ENOMEM)`.

Example:

```c
static struct task_struct *etx_thread;

etx_thread = kthread_run(thread_function,NULL,"eTx Thread"); 
if(etx_thread) 
{
 printk(KERN_ERR "Kthread Created Successfully...\n");
}
else 
{
 printk(KERN_ERR "Cannot create kthread\n"); 
}
```

Source code:

```c
/***************************************************************************//**
*  \file       driver.c
*
*  \details    Simple Linux device driver (Kernel Thread)
*
*  \author     EmbeTronicX
*
*  \Tested with Linux raspberrypi 5.10.27-v7l-embetronicx-custom+
*
*******************************************************************************/
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include<linux/slab.h>                 //kmalloc()
#include<linux/uaccess.h>              //copy_to/from_user()
#include <linux/kthread.h>             //kernel threads
#include <linux/sched.h>               //task_struct 
#include <linux/delay.h>
#include <linux/err.h>
 
dev_t dev = 0;
static struct class *dev_class;
static struct cdev etx_cdev;
 
static int __init etx_driver_init(void);
static void __exit etx_driver_exit(void);
 
static struct task_struct *etx_thread;
 
/*
** Function Prototypes
*/
/*************** Driver functions **********************/
static int etx_open(struct inode *inode, struct file *file);
static int etx_release(struct inode *inode, struct file *file);
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len,loff_t * off);
static ssize_t etx_write(struct file *filp, 
                const char *buf, size_t len, loff_t * off);
 /******************************************************/
 
int thread_function(void *pv);

/*
** Thread
*/
int thread_function(void *pv)
{
    int i=0;
    while(!kthread_should_stop()) {
        pr_info("In EmbeTronicX Thread Function %d\n", i++);
        msleep(1000);
    }
    return 0;
}

/*
** File operation sturcture
*/ 
static struct file_operations fops =
{
        .owner          = THIS_MODULE,
        .read           = etx_read,
        .write          = etx_write,
        .open           = etx_open,
        .release        = etx_release,
};

/*
** This function will be called when we open the Device file
*/  
static int etx_open(struct inode *inode, struct file *file)
{
        pr_info("Device File Opened...!!!\n");
        return 0;
}

/*
** This function will be called when we close the Device file
*/   
static int etx_release(struct inode *inode, struct file *file)
{
        pr_info("Device File Closed...!!!\n");
        return 0;
}

/*
** This function will be called when we read the Device file
*/
static ssize_t etx_read(struct file *filp, 
                char __user *buf, size_t len, loff_t *off)
{
        pr_info("Read function\n");
 
        return 0;
}

/*
** This function will be called when we write the Device file
*/
static ssize_t etx_write(struct file *filp, 
                const char __user *buf, size_t len, loff_t *off)
{
        pr_info("Write Function\n");
        return len;
}

/*
** Module Init function
*/  
static int __init etx_driver_init(void)
{
        /*Allocating Major number*/
        if((alloc_chrdev_region(&dev, 0, 1, "etx_Dev")) <0){
                pr_err("Cannot allocate major number\n");
                return -1;
        }
        pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));
 
        /*Creating cdev structure*/
        cdev_init(&etx_cdev,&fops);
 
        /*Adding character device to the system*/
        if((cdev_add(&etx_cdev,dev,1)) < 0){
            pr_err("Cannot add the device to the system\n");
            goto r_class;
        }
 
        /*Creating struct class*/
        if(IS_ERR(dev_class = class_create(THIS_MODULE,"etx_class"))){
            pr_err("Cannot create the struct class\n");
            goto r_class;
        }
 
        /*Creating device*/
        if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"etx_device"))){
            pr_err("Cannot create the Device \n");
            goto r_device;
        }
 
        etx_thread = kthread_create(thread_function,NULL,"eTx Thread");
        if(etx_thread) {
            wake_up_process(etx_thread);
        } else {
            pr_err("Cannot create kthread\n");
            goto r_device;
        }
#if 0
        /* You can use this method also to create and run the thread */
        etx_thread = kthread_run(thread_function,NULL,"eTx Thread");
        if(etx_thread) {
            pr_info("Kthread Created Successfully...\n");
        } else {
            pr_err("Cannot create kthread\n");
             goto r_device;
        }
#endif
        pr_info("Device Driver Insert...Done!!!\n");
        return 0;
 
 
r_device:
        class_destroy(dev_class);
r_class:
        unregister_chrdev_region(dev,1);
        cdev_del(&etx_cdev);
        return -1;
}

/*
** Module exit function
*/  
static void __exit etx_driver_exit(void)
{
        kthread_stop(etx_thread);
        device_destroy(dev_class,dev);
        class_destroy(dev_class);
        cdev_del(&etx_cdev);
        unregister_chrdev_region(dev, 1);
        pr_info("Device Driver Remove...Done!!\n");
}
 
module_init(etx_driver_init);
module_exit(etx_driver_exit);
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("EmbeTronicX <embetronicx@gmail.com>");
MODULE_DESCRIPTION("A simple device driver - Kernel Thread");
MODULE_VERSION("1.14");
```

Latest progress: <https://embetronicx.com/tutorials/linux/device-drivers/tasklet-static-method/>

## Miscellaneous

* modprobe 的作用

    在 linux 中，如果一些内核模块之间有依赖关系，那么必须按依赖关系进行`insmod`，否则会报错。

    `modprobe`会根据`depmod`所产生的相依关系，决定要载入哪些模块。若在载入过程中发生错误，在`modprobe`会卸载整组的模块。

    example:

    载入模块：

    ```bash
    sudo modprobe -v xdxgpu
    ```

    尝试制裁模块：

    ```bash
    sudo modprobe -vr xdxgpu
    ```

* depmod 命令

    `depmod`通常在`modprobe`之前运行，用于分析可载入模块的依赖关系。

    example:

    分析所有可用模块的依赖关系：

    ```bash
    sudo depmod -av
    ```

    注：

    1. 这里没有加路径，可能是会分析到当前目录下？
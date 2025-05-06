# dkms note

## cache

* dkms match

    如果你在`6.8.0-58-generic` kernel 上已经安装好了`dkms_test`这个 module，那么只需要执行`dkms match --templatekernel 6.8.0-58-generic -k 6.8.0-57-generic`，就可以将其安装到`6.8.0-57-generic` kernel 上。

    但是运行的时候报错了：

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/dkms_test$ dkms status
    dkms_test/1.0, 6.8.0-58-generic, x86_64: installed
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/dkms_test$ sudo dkms match --templatekernel 6.8.0-58-generic -k 6.8.0-57-generic

    Matching modules in kernel: 6.8.0-57-generic (x86_64)
    to the configuration of kernel: 6.8.0-58-generic (x86_64)

    Kernel preparation unnecessary for this kernel. Skipping...
    Module:  dkms_test/1.0
    Version: 6.8.0-58-generic

    Building & Installing this module/version:
    Error! Could not find module source directory.
    Directory: /usr/src/dkms_test/1.0-6.8.0-58-generic does not exist.
    ```

    dkms 可以通过`/usr/src/dkms_test/1.0-6.8.0-58-generic`这种方式找到对应的源代码目录吗？之前看到的都是`/usr/src/dkms_test-1.0`这样的。是 dkms 的代码长期没人维护，导致这个搜索方式变动？还是 dkms 有两套搜索源码的方式？

*  Kernel/DkmsDriverPackage

    dkms 创建 deb 包。

    <https://help.ubuntu.com/community/Kernel/DkmsDriverPackage>

* ubuntu dkms man page

    <https://manpages.ubuntu.com/manpages/trusty/man8/dkms.8.html>

* dkms 的 conf 文件中，各个变量的含义

    example:

    `dkms.conf`:

    ```conf
    MAKE="make -C src/ KERNELDIR=/lib/modules/${kernelver}/build"
    CLEAN="make -C src/ clean"
    BUILT_MODULE_NAME=awesome
    BUILT_MODULE_LOCATION=src/
    PACKAGE_NAME=awesome
    PACKAGE_VERSION=1.1
    REMAKE_INITRD=yes
    ```

    * `MAKE`: 如何构建此 module

    * `CLEAN`: 如何 clean 此 module

    * `BUILT_MODULE_NAME`: 指定需要 insmod 的`.ko`文件的不带后缀的名称。一个工程目录下可能会编译出来多个`.ko`或`.o`文件，这里指定需要加载的 module。

    * `BUILT_MODULE_LOCATION`: 指定在哪找编译生成的`.ko`文件。因此前面的`MAKE`命令不一定把`.ko`生成在当前文件夹。

    * `PACKAGE_NAME`, `PACKAGE_VERSION`: 不清楚这两个是干嘛用的。ubuntu help page 上的解释是

        > The name and version DKMS should associate with the module(s). 

        这个解释说了等于没说。

    * `REMAKE_INITRD`，官方解释为

        > To remake the initrd image after installing the module. 

* dkms 移除 module 的 example

    与 install 相反的是 uninstall:

    ```
    (base) hlc@hlc-VirtualBox:~$ sudo dkms uninstall -m dkms_test -v 1.0
    Module dkms_test-1.0 for kernel 6.8.0-58-generic (x86_64).
    Before uninstall, this module version was ACTIVE on this kernel.

    dkms_test.ko:
     - Uninstallation
       - Deleting from: /lib/modules/6.8.0-58-generic/updates/dkms/
     - Original module
       - No original module was found for this module on this kernel.
       - Use the dkms install command to reinstall any previous module version.

    depmod...
    ```

    此时的 status 为：

    ```
    (base) hlc@hlc-VirtualBox:~$ dkms status
    dkms_test/1.0, 6.8.0-58-generic, x86_64: built
    sipu/1.0.0, 6.8.0-58-generic, x86_64: installed
    ```

    如果需要在 dkms 中移除一个 module，则可以使用`remove`:

    ```
    (base) hlc@hlc-VirtualBox:~$ sudo dkms remove -m dkms_test -v 1.0
    Module dkms_test-1.0 for kernel 6.8.0-58-generic (x86_64).
    This module version was INACTIVE for this kernel.
    depmod...
    Deleting module dkms_test-1.0 completely from the DKMS tree.
    (base) hlc@hlc-VirtualBox:~$ dkms status
    (base) hlc@hlc-VirtualBox:~$ 
    ```

    在执行`dkms remove`时，会自动执行`dkms uninstall`。

    此时`dkms_test` module 完全从`/var/lib/dkms`中被移除，但是`/usr/src/dkms_test-1.0`仍保持存在。

* dkms example

    创建工程文件夹：`dkms_test`

    加入文件：

    * `dkms_test.c`:

        ```c
        #include <linux/init.h>
        #include <linux/module.h>

        int init_mod(void);
        void exit_mod(void);

        int init_mod(void) {
            printk(KERN_INFO "init hlc mod\n");
            return 0;
        }

        void exit_mod() {
            printk(KERN_INFO "exit hlc mod\n");
        }

        module_init(init_mod);
        module_exit(exit_mod);
        MODULE_LICENSE("GPL");
        ```

    * `Makefile`:

        ```makefile
        kern_dir := /usr/src/linux-headers-6.8.0-58-generic
        obj-m += dkms_test.o

        default:
        	make -C $(kern_dir) M=$(PWD) modules

        clean:
        	rm -f *.ko *.o *.mod *.mod.c *.cmd *.order *.symvers .*.cmd
        ```

    * `dkms.conf`:

        ```conf
        PACKAGE_NAME="dkms_test"
        PACKAGE_VERSION="1.0"
        BUILT_MODULE_NAME[0]="dkms_test"
        DEST_MODULE_LOCATION[0]="/kernel/extra"
        AUTOINSTALL="yes"
        ```

    回退到上一级文件夹，然后将`dkms_test`文件夹复制到`/usr/src`下：

    `sudo cp -r dkms_test /usr/src/dkms_test-1.0`

    向 dkms 数据库中添加新项目：

    `sudo dkms add -m dkms_test -v 1.0`

    output:

    ```
    Creating symlink /var/lib/dkms/dkms_test/1.0/source -> /usr/src/dkms_test-1.0
    ```

    如输出所示，此时`/var/lib/dkms`目录中会新建一个`dkms_test`文件夹，其目录结构如下所示：

    ```
    (base) hlc@hlc-VirtualBox:/var/lib/dkms/dkms_test$ tree .
    .
    └── 1.0
        ├── build
        └── source -> /usr/src/dkms_test-1.0

    3 directories, 0 files
    ```

    此时运行`dkms status`，输出如下：

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects$ dkms status 
    dkms_test/1.0: added
    ```

    执行`sudo dkms build -m dkms_test -v 1.0`进行编译，output 如下：

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/dkms_test$ sudo dkms build -m dkms_test -v 1.0

    Kernel preparation unnecessary for this kernel. Skipping...

    Building module:
    cleaning build area...
    make -j4 KERNELRELEASE=6.8.0-58-generic -C /lib/modules/6.8.0-58-generic/build M=/var/lib/dkms/dkms_test/1.0/build...
    Signing module:
     - /var/lib/dkms/dkms_test/1.0/6.8.0-58-generic/x86_64/module/dkms_test.ko
    Secure Boot not enabled on this system.
    cleaning build area...
    ```

    此时可以看到 dkms 目录已经准备好了`.ko`文件：

    ```
    (base) hlc@hlc-VirtualBox:/var/lib/dkms/dkms_test/1.0/6.8.0-58-generic/x86_64$ tree .
    .
    ├── log
    │   └── make.log
    └── module
        └── dkms_test.ko

    2 directories, 2 files
    ```

    此时即可执行 install 命令：

    ```bash
    sudo dkms install -m dkms_test -v 1.0
    ```

    output:

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/dkms_test$ sudo dkms install -m dkms_test -v 1.0 

    dkms_test.ko:
    Running module version sanity check.
     - Original module
       - No original module exists within this kernel
     - Installation
       - Installing to /lib/modules/6.8.0-58-generic/updates/dkms/

    depmod....
    ```

    此时可以看到`xxx.ko`已经被安装到了 kernel 目录的`updates/dkms`文件夹下：

    ```
    (base) hlc@hlc-VirtualBox:/lib/modules/6.8.0-58-generic/updates/dkms$ ls -lh dkms_test.ko 
    -rw-r--r-- 1 root root 6.2K  4月 29 17:08 dkms_test.ko
    ```

    此时的 dkms status:

    ```
    (base) hlc@hlc-VirtualBox:~$ dkms status
    dkms_test/1.0, 6.8.0-58-generic, x86_64: installed
    ```

    此时可以直接通过`modprobe`加载内核模块：

    ```
    (base) hlc@hlc-VirtualBox:~$ sudo modprobe dkms_test 
    [sudo] password for hlc: 
    (base) hlc@hlc-VirtualBox:~$ 
    ```

    dmesg output:

    ```
    [ 1243.508106] init hlc mod
    ```

    说明：

    * `dkms install`并不会执行 modprobe 或 insmod。

    * 在`dkms install`后，就可以在任意工作目录直接使用`modprobe`加载`xxxx.ko`文件了
    
        `dmesg`里仍会有这个提示：

        ```
        [  788.580510] dkms_test: loading out-of-tree module taints kernel.
        [  788.580519] dkms_test: module verification failed: signature and/or required key missing - tainting kernel
        ```

## note

install: `sudo apt install dkms`
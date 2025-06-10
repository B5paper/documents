# Libvirt Note

## cache

* `virt-sparsify`有可能会用到 root 权限，看 log 似乎会扫描 host 上的`/boot`目录下的 kernel 文件，并做一个复制操作。

* `virt-sparsify`的使用

    `virt-sparsify --tmp <tmp_dir_path> ./old_disk_img.qcow2 ./new_disk_img.qcow2`

    `virt-sparsify`在运行过程中，需要一个和原 disk img 差不多大的 tmp 空间保存临时数据，我们可以使用`--tmp`参数将 tmp 文件保存到指定的目录下。这个 tmp 文件的文件名是随机生成的，不用担心产生文件覆盖，因此也可以把 tmp 目录设置为当前目录。

    如果磁盘空间不够用，可以使用`--inplace`参数，这样不会产生 tmp 文件。

* 首次安装完`virt-manager`后，似乎不会直接自动启动`libvirt`。重启了系统后，`libvirt` daemon 才会启动。不清楚是否有安装完后，手动直接启动的方式。

* ubuntu 24.04.01 在 virt-manager 4.0.0, qemu-system-x86_64 6.2.0 (Debian 1:6.2-dfsg-2ubuntu6.21) 环境下，在 install 阶段指定 ubuntu iso 镜像，virt-manager 可以正确识别，也可以顺利安装，进入系统后也正常显示。

    video qxl 使用的是 qxl。

* ubuntu 24.04.01 在 qemu + virt manager 里安装时，需要

    1. video vga 选择 VGA

        选 QXL 和 Virtio 都会花屏

    2. iso installer 选 graphics safe

* pci 设备透传到 qemu 虚拟机内

    1. 在 bios 开启 VT-d 功能

    2. 在 grub 中打开 iommu

        `sudo vim /etc/default/grub`

        在`GRUB_CMDLIND_LINUX_DEFAULT`或`GRUB_CMDLINE_LINUX`中添加`intel_iommu=on`。这两个变量里只需要选择其中一个添加就可以了。

        假如选择的是`GRUB_CMDLINE_LINUX_DEFAULT`，那么最终的结果可能如下所示：

        ```
        GRUB_DEFAULT=0
        GRUB_TIMEOUT_STYLE=hidden
        GRUB_TIMEOUT=0
        GRUB_DISTRIBUTOR='lsb_release -i -s 2> /dev/null || echo Debian'
        GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on"
        GRUB_CMDLINE_LINUX=""
        ```

        使 grub 生效：`sudo update-grub`

        重启系统：`reboot`

    3. 启动`vfio-pci`内核态驱动

        `sudo modprobe vfio-pci`

        查看是否加载成功：

        `lsmod | grep vfio`

        如果没有任何输出，那么加载驱动失败。有输出说明加载成功。

    4. 解綁待透传设备的驱动

        以 nvidia 为例：

        首先查看设备是否在使用中：`lspci -v | less`，搜索`nvidia`，看到

        ```
        Kernel driver in use: nvidia
        ```

        说明`nvidia`驱动正占用该设备。

        对于非显卡的普通设备，比如网卡设备，可以进入对应的 driver 下，把 pcie 编号 unbind 一下。

        对于显卡设备，因为桌面图形环境总是占用显卡，所以要么把`nvidia`，`noveau`等驱动放到 blacklist 里，要么`sudo systemctl set-default multi-user.target`，进入文字系统。这两种方式都需要重启系统。

        此时重新执行`lspci -v | less`，看到对应的 pci 设备的 kernel driver in use 字段为空，说明设备未和任何驱动绑定。

    5. add vfio-pci new id

        这里以 nvidia 为例：

        `lspci -nn | grep -i nvidia`

        输出：

        ```
        b1:00.0 3D controller [0302]: NVIDIA Corporation GV100GL [Tesla V100 PCIe 32GB] [10de:1db6] (rev a1)
        ```

        进入`/sys/bus/pci/drivers/vfio-pci`文件夹，执行：

        `echo "10de 1db6" | sudo tee new_id`

        此时再使用`lspci -v`检查 nvidia 的 driver 占用情况，可以看到它变成了 `vfio-pci`。

    6. 在 qemu 中直接添加 pci device 即可

* qemu 启动虚拟机命令

    `qemu-system-x86_64 -name aaa -accel tcg -vga virtio -m 4096 -smp 8 -hda ./ccc.qcow2`

* `virt-sparsify`包含在工具包`guestfs-tools`中

    安装：`sudo apt install guestfs-tools`

* 将 qcow2 文件放在 virtual box shared folder 里，不太会影响启动的 qemu 虚拟机的性能。

* shrink the qemu `.qcow2` file size

    `qemu-img convert -p -O qcow2 /path/to/source.qcow2 /path/to/dest.qcow2`

    其中`-p`是显示进度条，最好不要省略。

* 在 virt-namanger 新建虚拟机时报错`Error: No active connection to Installed on`

    解决办法：

    查看下面文件的用户权限：

    `ls -lh /var/run/libvirt/libvirt-sock`

    可以看到其属于`libvirt`用户组。

    我们把当前用户加入到`libvirt`用户组：

    `sudo usermod -a -G libvirt hlc`

    然后重启系统即可。

    经测试，只运行`logout`是不够的。必须重启才行。

* libvirt / virt-manager 的一个文档

    <https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/virtualization_deployment_and_administration_guide/sect-virsh-delete>

* ubuntu server with libvirt

    <https://ubuntu.com/server/docs/libvirt>

* 可以用`qemu-img info <file>`查看 qcow2 文件的真实大小。

    `ls -lh <file>`只要看到 qcow2 文件虚拟的大小。

    `qcow2`只增大，不缩小，因此一个虚拟大小为 120G 的 qcow2 文件，实际占用磁盘可能有 40G，进入虚拟机系统后使用`df -h`看到的占用可能只有 20G。

* linux 5.19.x 的内核无法识别最新版 qemu (2024.06.20 测试) 提供的默认显卡 QXL, VGA，导致`/dev`目录下没有`dri`目录，自然也没有显卡设备。因此无法启动 wayland 桌面，也无法启动 x11 桌面。

    解决办法是使用 qemu 提供的 virtio 虚拟显卡。

* libvirt 默认的 disk image 存放位置是`/var/lib/libvirt/images`

* 5.19.0 的 kernel 在 qemu 下无法启动桌面，主要是因为`/dev/dri`没有被创建

    很有可能是这个版本的内核无法识别 qemu 提供的显卡。

* 关闭一个虚拟机：`virsh destroy aaa`

* 删除一个虚拟机，并且删除 virtual disk image: `virsh undefine aaa --remove-all-storage`

* console 模式下可以运行的安装命令：

    ```
    hlc@hlc-VirtualBox:/home/libvirt-qemu$ virt-install --name aaa --memory 2048 --vcpus 2 --disk size=20 --location https://mirror.sjtu.edu.cn/ubuntu/dists/focal/main/installer-amd64/ --os-variant ubuntu22.04 --nographics -x 'console=ttyS0'
    ```

    注：

    * `--cdrom`默认没有 console 输出，因此只能用于图形系统。`--location`可以设置 console 输出，因此可以在文字操作系统下安装一些 live server 版的系统。

    * 这个命令中的`--location`选用了 network url，目前可以正常安装。ubuntu focal 是最后一版可以通过 url 安装的 ubuntu 系统，下个版本 jammy 已经不支持了。

    * `-x 'console=ttyS0'`与`--location`搭配使用可以让输出重定向到 console。

    * `--location`的参数改成 iso 时，会报错目录错误。如果将 iso 先 mount 到本地目录，再使用 mount 的目录进行安装，那么会最终进入到 initramfs 中，无法找到可以挂载的磁盘 dev。

    * 网上的大部分方案，在使用`--location` + iso 时，都是用的 centos 镜像，所以很有可能 centos 的 iso 可以通过`--location`安装成功。但是为什么 ubuntu 的 iso 不行，不清楚。有时间了可以去看下 virt-manager 的开源代码实现：<https://github.com/virt-manager/virt-manager>

* virt-manager 在启动时需要`service libvirtd start`

* 如果 libvirt 没有 iso 权限，那么可以把`libvirt-qemu`加入到用户组里：`sudo usermod -a -G hlc libvirt-qemu`

* 文字操作系统模式下，无法安装操作系统的图形版本，比如 ubuntu desktop, linux mint 等。

* 列出正在运行的虚拟机：`virsh list`

    列出所有的虚拟机：`virsh list --all`

## note

使用 console 连接到虚拟机：`virsh console <domain>`

查看电脑是否支持虚拟化：`grep -E 'svm|vmx' /proc/cpuinfo`

（这个命令似乎是每个 cpu 核心都打印出来一个特性信息）

If the output contains a vmx entry, indicating an Intel processor with the Intel VT-x extension. （如果是一个 amd 处理器的话，可以看到`svm`关键字）

Ensure the KVM kernel modules are loaded: `lsmod | grep kvm`。如果输出包含`kvm_intel`或`kvm_amd`，那么就说明硬件是支持虚拟化的。

`virsh capabilities`也可以列出系统目前支持的虚拟化兼容性。（这个命令依赖`libvirt-client`包）

view a full list of the CPU models supported for an architecture type: `virsh cpu-models <architecture>`。Example: `virsh cpu-models x86_64`, `virsh cpu-models ppc64`。

The full list of supported CPU models and features is contained in the `/usr/share/libvirt/cpu_map/`。

A guest's CPU model and features can be changed in the <cpu> section of the domain XML file. （不知道这句话啥意思）

The host model can be configured to use a specified feature set as needed. （不知道啥意思）

一些必须要装的包：

* `qemu-kvm`: This package provides the user-level KVM emulator and facilitates communication between hosts and guest virtual machines.

    (这个是`yum`里的包。`apt`里对应的包不叫这个名字，我还不知道叫啥。)

* `qemu-img`: This package provides disk management for guest virtual machines.

    The qemu-img package is installed as a dependency of the qemu-kvm package.

    （这个包 apt 里找不到）

* `libvirt`: This package provides the server and host-side libraries for interacting with hypervisors and host systems, and the libvirtd daemon that handles the library calls, manages virtual machines, and controls the hypervisor.

    * `libvirt-client`: This package provides the client-side APIs and libraries for accessing libvirt servers. The libvirt-client package includes the virsh command-line tool to manage and control virtual machines and hypervisors from the command line or a special virtualization shell.

    apt 里搜不到`libvirt`，只能搜到`libvirt-clients`和`libvirt-daemon`。

Several additional virtualization management packages are also available and are recommended when using virtualization:

* `libvirt-python`: This package contains a module that permits applications written in the Python programming language to use the interface supplied by the libvirt API.

    （这个包 apt 里找不到）

* `virt-manager`: This package provides the virt-manager tool, also known as Virtual Machine Manager. This is a graphical tool for administering virtual machines. It uses the `libvirt-client` library as the management API.

    * `virt-install`: This command for creating virtual machines from the command line.

        在安装之前，需要把当前用户加入到`kvm`和`libvirt`的 group 中：

        ```
        sudo usermod -a -G kvm hlc
        sudo usermod -a -G libvirt hlc
        ```

        查看支持的系统类型：`virt-install --osinfo list`

        如果安装失败，那么 console 中会有对应的提示。通常都是`libvirt-qemu`没有 iso 文件的权限，改改权限就好了。

        Example:

        ```bash
        virt-install --name bbb --memory 2048 --vcpus 2 --disk size=8 --cdrom /home/libvirt-qemu/debian-11.7.0-amd64-netinst.iso --os-variant debian11
        ```

        执行后，会自动弹出来一个图形化的安装界面（virt viewer）。

        Importing a virtual machine image:

        The following example imports a virtual machine from a virtual disk image:

        ```bash
        virt-install \ 
            --name guest1-rhel7 \ 
            --memory 2048 \ 
            --vcpus 2 \ 
            --disk /path/to/imported/disk.qcow \ 
            --import \ 
            --os-variant rhel7 
        ```

        The `--import` option specifies that the virtual machine will be imported from the virtual disk image specified by the --disk /path/to/imported/disk.qcow option.

        installs a virtual machine from a network location: 

        ```bash
        virt-install \ 
            --name guest1-rhel7 \ 
            --memory 2048 \ 
            --vcpus 2 \ 
            --disk size=8 \ 
            --location http://example.com/path/to/os \ 
            --os-variant rhel7 
        ```

    （这个包 apt 里也找不到）

The main required options for virtual guest machine installations are:

* `--name`: The name of the virtual machine.

* `--memory`: The amount of memory (RAM) to allocate to the guest, in MiB.

* `--disk`: The storage configuration details for the virtual machine. If you use the `--disk none` option, the virtual machine is created with no disk space.

* `--filesystem`: The path to the file system for the virtual machine guest.

    这个参数和`--disk`参数只能选一个。

下面这些安装方式，选一个就行：

* `--location`: The location of the installation media.

* `--cdrom`: The file or device used as a virtual CD-ROM device. It can be path to an ISO image, or a URL from which to fetch or access a minimal boot ISO image. However, it can not be a physical host CD-ROM or DVD-ROM device.

* `--pxe`: Uses the PXE boot protocol to load the initial ramdisk and kernel for starting the guest installation process.

* `--import`: Skips the OS installation process and builds a guest around an existing disk image. The device used for booting is the first device specified by the disk or filesystem option.

* `--boot`: The post-install VM boot configuration. This option allows specifying a boot device order, permanently booting off kernel and initrd with optional kernel arguments and enabling a BIOS boot menu.

某个 option 的帮助：`virt-install --option=?`

Example:

```bash
virt-install \ 
  --name guest1-rhel7 \ 
  --memory 2048 \ 
  --vcpus 2 \ 
  --disk size=8 \ 
  --cdrom /path/to/rhel7.iso \ 
  --os-variant rhel7 
```

退出一个 console: `Ctrl` + `]`，（也有可能是`Ctrl` + `Shirt` + `]`）

## network

To configure a NAT network for the guest virtual machine, use the following option for `virt-install`:

`--network default`

If no `network` option is specified, the guest virtual machine is configured with a default network with NAT.

When configured for bridged networking, the guest uses an external DHCP server. This option should be used if the host has a static networking configuration and the guest requires full inbound and outbound connectivity with the local area network (LAN). It should be used if live migration will be performed with the guest virtual machine. To configure a bridged network with DHCP for the guest virtual machine, use the following option: `--network br0`

Bridged networking can also be used to configure the guest to use a static IP address. To configure a bridged network with a static IP address for the guest virtual machine, use the following options: 

```bash
--network br0 \
--extra-args "ip=192.168.1.2::192.168.1.1:255.255.255.0:test.example.com:eth0:none"
```

To configure a guest virtual machine with no network interface, use the following option:

`--network=none`

可以输入`virt-manager`进入图形管理界面。

Name the virtual machine. Virtual machine names can contain letters, numbers and the following characters: underscores (_), periods (.), and hyphens (-). Virtual machine names must be unique for migration and cannot consist only of numbers.

Most virt-install options are not required. The minimum requirements are --name, --memory, guest storage (--disk, --filesystem or --disk none), and an install method (--location, --cdrom, --pxe, --import, or boot).

## virt-install

```bash
virt-install --name ubuntu_2004 --memory 4096 --vcpus 4 --cpu host --cdrom /home/libvirt-qemu/ubuntu-20.04.6-live-server-amd64.iso --disk 40 -w network=default --graphics spice --input keyboard,bus=usb 
```

这个命令已经被验证可以正确执行。需要注意的是这个：`--cdrom /home/libvirt-qemu/ubuntu-20.04.6-live-server-amd64.iso`，因为权限的问题，iso 文件必须放在`libvirt-qemu`这个文件夹下，不然就会报错。
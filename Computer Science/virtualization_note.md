# Virtualization Note

## cache

* qemu tutorial

    * `qemu-img create -f qcow2 myVirtualDisk.qcow2 20G`

        创建磁盘镜像。

        output:

        ```
        Formatting 'myVirtualDisk.qcow2', fmt=qcow2 cluster_size=65536 extended_l2=off compression_type=zlib size=21474836480 lazy_refcounts=off refcount_bits=16
        ```

        check file format:

        `file myVirtualDisk.qcow2`

        output:

        ```
        myVirtualDisk.qcow2: QEMU QCOW2 Image (v3), 21474836480 bytes
        ```

    * start virtual machine

        ```bash
        qemu-system-x86_64 \
        -enable-kvm                                                    \
        -m 4G                                                          \
        -smp 2                                                         \
        -hda myVirtualDisk.qcow2                                       \
        -boot d                                                        \
        -cdrom linuxmint-21.1-cinnamon-64bit.iso                       \
        -netdev user,id=net0,net=192.168.0.0/24,dhcpstart=192.168.0.9  \
        -device virtio-net-pci,netdev=net0                             \
        -vga qxl                                                       \
        -device AC97
        ```

        parameter explanation:

        * `-enable-kvm` → KVM to boost performance
        * `-m 4G` → 4GB RAM
        * `-smp 2` → 2 CPUs
        * `-hda myVirtualDisk.qcow2` → our 20GB variable-size disk
        * `-boot d` → boots the first virtual CD drive
        * `-cdrom linuxmint-21.1-cinnamon-64bit.iso` → Linux Mint ISO
        * `-netdev user,id=net0,net=192.168.0.0/24,dhcpstart=192.168.0.9` → NAT with DHCP
        * `-device virtio-net-pci,netdev=net0` → network card
        * `-vga qxl` → powerful graphics card
        * `-device AC97` → sound card

* linux 系统可以通过配置控制 udev 设备加入时的权限属性

    qemu 可能会遇到权限问题无法访问 usb 设备，但通过配置权限后，qemu 就有了权限。

    为虚拟机加入 usb 设备的代码：

    ```bash
    -device usb-ehci,id=ehci \
    -usb \
    -device usb-host,bus=ehci.0,vendorid=0x0930,productid=0x6545
    ```

    查看 host 上的 usb：`lsusb`，`lsusb -v`

* iommu 其实就是 dma remapping，是 host bridge 上的一个设备，用于做地址转换使得 dma 可以访问超出自己地址空间能力的内存

    如果用不到 dma，就用不到 iommu

* vfio 用于在用户态提供给用户可交互的设备，做了一些地址转换，安全隔离的工作

    由于 qemu 实际工作在用户态，所以如果把虚拟设备透传给虚拟机，就必须要用到 vfio

* mdev 需要和 vfio 配合使用，实现对虚拟设备有相同的控制，但是数据分隔

    udev 指的是使用设备信息创建设备文件，从而将设备名固定下来，不再随设备插拔顺序的变动而变动。

* 查看 iommu 是否启用：

    `dmesg | grep -i iommu`

## note
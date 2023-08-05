# Libvirt Note

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
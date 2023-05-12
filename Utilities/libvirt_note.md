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

* `libvirt`: This package provides the server and host-side libraries for interacting with hypervisors and host systems, and the libvirtd daemon that handles the library calls, manages virtual machines, and controls the hypervisor.

Several additional virtualization management packages are also available and are recommended when using virtualization:

* `virt-install`: This package provides the virt-install command for creating virtual machines from the command line.

* `libvirt-python`: This package contains a module that permits applications written in the Python programming language to use the interface supplied by the libvirt API.

* `virt-manager`: This package provides the virt-manager tool, also known as Virtual Machine Manager. This is a graphical tool for administering virtual machines. It uses the `libvirt-client` library as the management API.

`libvirt-client`: This package provides the client-side APIs and libraries for accessing libvirt servers. The libvirt-client package includes the virsh command-line tool to manage and control virtual machines and hypervisors from the command line or a special virtualization shell.

The main required options for virtual guest machine installations are:

* `--name`: The name of the virtual machine.

* `--memory`: The amount of memory (RAM) to allocate to the guest, in MiB.

* `--disk`: The storage configuration details for the virtual machine. If you use the --disk none option, the virtual machine is created with no disk space.

* `--filesystem`: The path to the file system for the virtual machine guest.

    这个参数和`--disk`参数只能选一个。

下面这些安装方式，选一个就行：

* `--location`: The location of the installation media.

* `--cdrom`: The file or device used as a virtual CD-ROM device. It can be path to an ISO image, or a URL from which to fetch or access a minimal boot ISO image. However, it can not be a physical host CD-ROM or DVD-ROM device.

* `--pxe`: Uses the PXE boot protocol to load the initial ramdisk and kernel for starting the guest installation process.

* `--import`: Skips the OS installation process and builds a guest around an existing disk image. The device used for booting is the first device specified by the disk or filesystem option.

* `--boot`: The post-install VM boot configuration. This option allows specifying a boot device order, permanently booting off kernel and initrd with optional kernel arguments and enabling a BIOS boot menu.
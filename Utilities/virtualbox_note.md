# VirtualBox Note

## cache

* virtual box 的虚拟机默认使用的是 NAT 地址转换，并不是真正的 NAT，因此两台虚拟机之间无法连接

    可以在 virtual box 管理器 -> 工具 -> NAT网络 -> 创建，创建一个新的 net 网络，然后在虚拟机的控制 -> 设置 -> 网络 -> 连接方式里选择刚才创建的 NAT 网络，注意看名称是否对得上，点击确定。

    不到 1 分钟后，为防止 IP 冲突，新创建的 NAT 网络的 dhcp 服务器会重新配置各个虚拟机的 ip，等配置完成后，各个虚拟机之间就可以互相 ping 通了。

* 公司的电脑是因为启用了 windows hypervisor 虚拟化平台 feature，所以 virtual box 无法使用 kvm 特性。

    把 hypervisor 禁用就好了。

* virtual box enter BIOS

    select EFI option

    possible keys: F12, F2, Esc, F8, Del

* windows 上 virtual box 无法启动嵌套 kvm 解决方法

    ```powershell
    PS C:\Users\hlc\VirtualBox VMs> D:\Softwares\VirtualBox\VBoxManage.exe list vms
    ```

    output:

    ```
    "Ubuntu 20.04" {d8054d31-1c97-4ec1-946f-da7246cb03f4}
    "Ubuntu_2204" {86b4978b-45f8-40ff-848f-6094d1d89560}
    "Ubuntu_2204 _origin" {041fad56-26f1-4505-9f11-79098883357f}
    ```

    modify the vm nested virtualization feature:

    ```powershell
    PS C:\Users\hlc\VirtualBox VMs> D:\Softwares\VirtualBox\VBoxManage.exe modifyvm "Ubuntu_2204 _origin" --nested-hw-virt on
    ```

    运行后可以看到 vm 管理界面里嵌套虚拟化的对勾 v 已经开启了。

    此时进入虚拟机，运行`lsmod | grep kvm`可以看到

    ```
    kvm_intel       487424  0
    kvm            1409024  1 kvm_intel
    irqbypass        12288  1 kvm
    ```

    说明虚拟机中 kvm 启动成功。

* virtual box 中，开机时使用 esc 进入 bios。

* virtual box 使用 efi 启动的 iso 光盘

    virtual box 7.0.1 目前默认是从 MBR 启动引导，如果需要其他的方式启动引导，需要根据开机提示按键，或者在刚启动虚拟机时就一直按 del 键，进入 bios 和启动引导界面。

## Problem Shooting

1. win11 hyper-v compatibility

    virtual box 与 windows 11 的 hyper-v 存在兼容性问题。如果在 windows 11 上开了 hyper-v，那么 virtual box 会变得很慢很卡。

    解决办法是把 host 上的 hyper-v 禁用掉，重启电脑。

    Ref: <https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel>
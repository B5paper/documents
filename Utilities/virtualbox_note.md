# VirtualBox Note

## cache

* vbox + ubuntu 22.04.4 + vmsvga + 3d accel 在使用 xrdp client 时，远程桌面的矩形 update 有问题

    更新 vbox 增强功能后 rdp 仍有问题。

    改为 vboxsvga + 3d accel 后，桌面闪烁。

    改为只用 vboxsvga，不开 3d accel 后，桌面正常，rdp client 正常。但是在打开关闭普通程序时仍能观察到桌面不正常闪烁。

    注：

    1. 这个事件说明 vbox 在能用的情况下，不要随意更新，可能会出问题。

* ubuntu 22.04.4 + virtual box 7.2.4 r170995 不同虚拟显卡配置

    vboxvga: 开机后卡住。

    vboxsvga: 开机后卡住。未尝试开 3D 加速。

    vmsvga: 开机后闪屏。开 3d 加速后恢复正常。setting 也能正常打开。

* virtual box 中挂载 vboxsf 共享文件夹时，如果 -o umask=0002 失败，可以使用

    `sudo mount -t vboxsf -o uid=1000,gid=1000,dmode=755,fmode=644 D_DRIVE ~/d`

    还可以配置`/etc/fstab`以实现自动挂载：

    ```bash
    # 编辑 fstab 文件
    sudo nano /etc/fstab

    # 添加以下行（使用你的实际用户ID和组ID）
    D_DRIVE  /home/hlc/d  vboxsf  uid=1000,gid=1000,dmode=775,fmode=664  0  0

    # 然后测试挂载
    sudo mount -a
    ```

* i7-1360P + win10 + virtualbox 7.0 + host ubuntu 22.04 + virt-manager + vm ubuntu 22.04，虚拟机里的虚拟机，运行起来没有感觉到明显卡顿

    host ubuntu 22.04 运行在 VT-x/AMD-V 模式下，嵌套分页：活动，不受限执行：活动，运行峰值：100，半虚拟化接口：KVM，处理器：4

    host ubuntu 22.04 内：

    `lsmod | grep -i kvm`

    output:

    ```
    kvm_intel 487424 6
    kvm 1409024 1 kvm_intel
    irqbypass 12288 1 kvm
    ```

    guest ubuntu 22.04 info:

    Hypervisor: KVM

    `lsmod | grep -i kvm`

    output:

    ```
    kvm_intel 487424 0
    kvm 1409024 1 kvm_intel
    irqbypass 12288 2 vfio_pci_core,kvm
    ```

    在执行`sudo modprobe vfio-pci`后，`lsmod | grep -i vfio`有`vfio_pci`, `vfio_pci_core`, `vfio_iommu_type1`, `vfio`, `iommufd`, `irqbypass`相关的输出。

* virtual box 7.0 + ubuntu 24.04 必须关闭 EFI 才能正常启动

* ubuntu 24.04.01 在 virtual box 7.0 里安装时，需要

    1. 显示里选择 VBoxSVGA

    2. iso installer 里选 graphics safe 模式

    不然安装界面和启动界面都会花屏。

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

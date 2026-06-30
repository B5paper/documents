# iommu note

记录 iommu, vfio, qemu，虚拟化相关的笔记。

## cache

* qemu IOMMU 页面保护错误

    ```
    0x7c261000 [fault reason 0x05] PTE Write access is not set
    [1116255.522412] DMAR: DRHD: handling fault status reg 3
    [1116255.522994] DMAR: [DMA Write NO_PASID] Request device [3c:00.0] fault addr 0x3ff6e000 [fault reason 0x05] PTE Write access is not set
    [1116410.753322] DMAR: DRHD: handling fault status reg 2
    [1116410.753735] DMAR: [DMA Read NO_PASID] Request device [bc:00.0] fault addr 0x3ffc0000 [fault reason 0x06] PTE Read access is not set
    [1116427.782722] DMAR: DRHD: handling fault status reg 3
    [1116427.783133] DMAR: [DMA Write NO_PASID] Request device [dc:00.0] fault addr 0x3ff70000 [fault reason 0x05] PTE Write access is not set
    [1116456.387913] DMAR: DRHD: handling fault status reg 3
    [1116456.388347] DMAR: [DMA Write NO_PASID] Request device [dc:00.0] fault addr 0x7ab2f000 [fault reason 0x05] PTE Write access is not set
    ```

    这些是 **IOMMU 页面保护错误**，表示设备试图访问没有适当权限的内存页面。

    错误分析

    ```
    [fault reason 0x05] PTE Write access is not set
    [fault reason 0x06] PTE Read access is not set
    ```
    - **0x05**: 设备试图写入只读内存
    - **0x06**: 设备试图读取不可读内存

    **关键问题**: IOMMU页面表(PTE)权限配置错误，DMA访问被阻止。

    主要原因

    1. **内存页面权限不匹配**
    2. **IOMMU配置错误**
    3. **设备驱动或固件问题**
    4. **虚拟化配置问题**
    5. **内存热插拔或动态内存调整导致**

    解决方案

    1. **检查设备状态**

        ```bash
        # 查看故障设备
        lspci | grep -E "3c:00.0|bc:00.0|dc:00.0"

        # 查看设备驱动
        lspci -k -s 3c:00.0
        dmesg | grep -i "3c:00.0"

        # 检查设备是否在VFIO控制下
        ls -la /sys/bus/pci/devices/0000:3c:00.0/driver
        ```

    2. **更新内核参数（立即生效）**

        ```bash
        # 临时关闭IOMMU严格模式（允许更宽松的DMA）
        sudo sh -c "echo 0 > /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts"

        # 关闭IOMMU强制执行
        sudo sh -c "echo 1 > /sys/module/vfio_iommu_type1/parameters/disable_hugepages"
        sudo sh -c "echo 0 > /sys/module/vfio_iommu_type1/parameters/enforce_sysfs_direct"
        ```

    3. **永久内核参数配置**

        ```bash
        # 编辑GRUB配置
        sudo nano /etc/default/grub

        # 在GRUB_CMDLINE_LINUX中添加：
        # Intel平台
        intel_iommu=on iommu=pt iommu.passthrough=1 iommu.relaxable=1

        # AMD平台
        amd_iommu=on iommu=pt iommu.passthrough=1 iommu.relaxable=1

        # 强制允许所有DMA访问
        iommu.strict=0

        # 更新GRUB并重启
        sudo update-grub
        sudo reboot
        ```

    4. **VFIO模块配置**

        ```bash
        # 创建VFIO配置文件
        sudo nano /etc/modprobe.d/vfio.conf

        # 添加以下内容：
        options vfio_iommu_type1 allow_unsafe_interrupts=1
        options vfio_iommu_type1 disable_hugepages=1
        options vfio enable_unsafe_noiommu_mode=1

        # 重新加载模块
        sudo modprobe -r vfio_iommu_type1 vfio
        sudo modprobe vfio_iommu_type1
        sudo modprobe vfio
        ```

    5. **QEMU虚拟机配置优化**

        ```bash
        # 添加内存后端配置
        -object memory-backend-file,id=ram,size=16G,mem-path=/dev/hugepages,share=on,prealloc=on
        -numa node,memdev=ram

        # 或者使用显式内存锁定
        -object memory-backend-memfd,id=ram,size=16G,share=on,prealloc=yes
        -numa node,memdev=ram

        # 启用IOMMU模拟
        -device intel-iommu,intremap=on,caching-mode=on,aw-bits=48
        # 或
        -device virtio-iommu
        ```

    6. **Libvirt XML配置**

        ```xml
        <domain type='kvm'>
          <!-- 内存配置 -->
          <memoryBacking>
            <hugepages/>
            <locked/>
            <source type='memfd'/>
            <access mode='shared'/>
          </memoryBacking>
          
          <!-- IOMMU配置 -->
          <iommu model='intel'>
            <driver intremap='on' caching_mode='on' eim='on'/>
          </iommu>
          
          <!-- 设备配置 -->
          <devices>
            <!-- 为直通设备添加额外参数 -->
            <hostdev mode='subsystem' type='pci' managed='yes'>
              <source>
                <address domain='0x0000' bus='0x3c' slot='0x00' function='0x0'/>
              </source>
              <rom bar='off'/>
              <address type='pci' domain='0x0000' bus='0x00' slot='0x10' function='0x0'/>
            </hostdev>
          </devices>
        </domain>
        ```

    7. **特定设备解决方案**

        如果是GPU设备：

        ```bash
        # NVIDIA GPU可能需要额外的重置处理
        # 安装GPU重置脚本
        sudo apt install vendor-reset

        # 检查GPU是否支持FLR（Function Level Reset）
        lspci -vvv -s 3c:00.0 | grep -i flr

        # 如果没有FLR，可能需要ACS补丁
        pcie_acs_override=downstream,multifunction
        ```

        如果是网卡：

        ```bash
        # 关闭SR-IOV
        ethtool -i <interface> | grep driver
        # 重新绑定驱动
        echo 0000:3c:00.0 > /sys/bus/pci/drivers/vfio-pci/unbind
        echo 0000:3c:00.0 > /sys/bus/pci/drivers/ixgbe/bind
        ```

    8. **监控和调试**

        ```bash
        # 实时监控DMAR错误
        sudo dmesg -wH | grep -i dmar

        # 查看IOMMU映射
        sudo cat /sys/kernel/debug/iommu/intel-iommu/*/domain_translation_struct

        # 检查页表权限
        sudo cat /sys/kernel/debug/iommu/intel-iommu/*/pglvl
        ```

    9. **应急处理**

        如果问题突然出现：

        ```bash
        # 1. 暂停虚拟机
        virsh suspend <vm-name>

        # 2. 重置IOMMU
        sudo sh -c "echo 1 > /sys/class/iommu/dmar0/reset"
        sudo sh -c "echo 1 > /sys/class/iommu/dmar1/reset"

        # 3. 恢复虚拟机
        virsh resume <vm-name>

        # 4. 如果仍然失败，重启虚拟机
        virsh destroy <vm-name>
        virsh start <vm-name>
        ```

    10. **高级解决方案**

        如果以上方法无效，可能需要：

        编译自定义内核：

        ```bash
        # 启用CONFIG_INTEL_IOMMU_DEFAULT_ON
        # 启用CONFIG_INTEL_IOMMU_FLOPPY_WA
        # 禁用CONFIG_INTEL_IOMMU_STRICT
        ```

        使用ACS补丁：

        ```bash
        # 在GRUB中添加：
        pcie_acs_override=downstream,multifunction
        ```

        更新BIOS/UEFI：

        - 确保IOMMU相关设置正确
        - 启用ACS（Access Control Services）
        - 更新到最新固件版本

    诊断步骤

    1. **识别问题设备**：

        ```bash
        # 从错误信息中提取设备地址
        # [3c:00.0] -> 0000:3c:00.0
        ```

    2. **检查设备类型**：

        ```bash
        lspci -nn -s 3c:00.0
        ```

    3. **验证IOMMU分组**：

        ```bash
        dmesg | grep -i "DMAR:"
        find /sys/kernel/iommu_groups/ -type l | grep 3c:00.0
        ```

    4. **测试隔离模式**：

        ```bash
        # 临时切换到软IOMMU
        echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode
        ```

    推荐处理顺序

    1. **立即**：暂停并重启有问题的虚拟机
    2. **短期**：调整VFIO和IOMMU内核参数
    3. **中期**：优化虚拟机配置（内存后端、IOMMU设备）
    4. **长期**：更新内核/BIOS，考虑硬件兼容性

    **注意**：这些错误可能表示潜在的安全风险，在调整参数前请评估环境的安全要求。对于生产环境，建议先在测试环境中验证解决方案。

    一些建议的配置：

    1. 统一内核参数调整

        ```bash
        # 编辑GRUB配置，同时解决两个问题
        sudo nano /etc/default/grub

        # 在GRUB_CMDLINE_LINUX中添加：
        # 解决空间问题 + 解决权限问题
        intel_iommu=on iommu=pt 
        iommu.passthrough=1         # 直通模式减少映射
        iommu.relaxable=1           # 放宽IOMMU限制
        iommu.strict=0              # 关闭严格模式
        iommu.forcedac=0            # 允许64位DMA地址
        vfio_iommu_type1.allow_unsafe_interrupts=1
        vfio_iommu_type1.disable_hugepages=0

        # 更新并重启
        sudo update-grub
        sudo reboot
        ```

    2. 完整VFIO配置

        ```bash
        sudo nano /etc/modprobe.d/vfio-all.conf
        ```

        ```bash
        # 解决映射空间问题
        options vfio_iommu_type1 dma_entry_limit=131072
        options vfio_iommu_type1 max_dma_mask=48

        # 解决权限问题
        options vfio_iommu_type1 allow_unsafe_interrupts=1
        options vfio_iommu_type1 disable_hugepages=0
        options vfio enable_unsafe_noiommu_mode=0
        ```

    3. 虚拟机启动参数优化

        ```bash
        # 同时预防两个问题
        qemu-system-x86_64 \
          -object iommu-ioas,id=ioa0,max-ioas=256 \
          -iommu iova=ioa0 \
          -device intel-iommu,intremap=on,caching-mode=on \
          -object memory-backend-memfd,id=ram,size=16G,share=on,prealloc=on \
          -numa node,memdev=ram
        ```

    4. 监控脚本（同时检测两个问题）

        ```bash
        #!/bin/bash
        # monitor_iommu.sh

        while true; do
            echo "=== $(date) ==="
            
            # 检查IOVA空间
            echo "1. IOVA空间使用:"
            sudo cat /sys/kernel/debug/iommu/intel-iommu/*/iommu_resource 2>/dev/null
            
            # 检查DMAR错误
            echo "2. 最近DMAR错误:"
            sudo dmesg | tail -20 | grep -E "DMAR|IOMMU|VFIO"
            
            # 检查映射统计
            echo "3. VFIO映射统计:"
            sudo cat /sys/module/vfio_iommu_type1/parameters/* 2>/dev/null
            
            sleep 10
        done
        ```

    5. 完整修复流程

        **第一步：清理现有状态**

        ```bash
        # 停止所有使用VFIO的虚拟机
        virsh list --all | grep running | awk '{print $1}' | xargs -I{} virsh shutdown {}

        # 重置IOMMU
        sudo sh -c "echo 1 > /sys/class/iommu/dmar*/reset"

        # 重新加载VFIO
        sudo modprobe -r vfio_iommu_type1 vfio
        sudo modprobe vfio
        sudo modprobe vfio_iommu_type1
        ```

        **第二步：应用新配置**

        ```bash
        # 应用上述所有配置更改
        sudo nano /etc/default/grub      # 添加完整参数
        sudo nano /etc/modprobe.d/vfio.conf
        sudo update-grub
        sudo update-initramfs -u
        ```

        **第三步：重启并验证**

        ```bash
        sudo reboot

        # 启动后检查
        dmesg | grep -i "IOMMU.*enabled"
        cat /proc/cmdline | grep iommu
        lsmod | grep vfio
        ```
* `-device vfio-pci,host=03:00.0,iommu_platform=off` 再详细讲讲这个启动参数的作用

    这行参数是 QEMU/KVM 虚拟机中用于物理硬件直通（PCI Passthrough）的配置指令。

    **参数逐条拆解**

    1. `-device vfio-pci`

        * **作用**：调用 Linux 内核的 **VFIO (Virtual Function I/O)** 驱动框架，向虚拟机添加一个 PCI 直通设备。

        * **背景**：VFIO 是现代 Linux 上进行硬件直通的标准安全框架。在执行此 QEMU 命令前，宿主机的该硬件必须已经解绑了原有的驱动（如 `nvidia`、`ixgbe`），并成功绑定到了宿主机的 `vfio-pci` 内核模块上。

        注：

        1. 不写这个, 直接添加 device 会怎样？

    2. `host=03:00.0`

        * **作用**：指定要直通的**物理硬件的 PCI 地址**（Domain:Bus:Device.Function，这里省略了前面的 0000 域）。

    3. `iommu_platform=off`（关键性能优化项）

        * **深层作用**：这是一个**控制虚拟 IOMMU (vIOMMU) 边界**的开关。

        * **当 `iommu_platform=off` 时（默认行为）**：

            该直通设备在执行 DMA（直接内存访问）时，**直接使用虚拟机的物理内存地址（GPA）**，绕过虚拟机内部的 vIOMMU 驱动。

        * **当 `iommu_platform=on` 时**：

            该直通设备被强制纳入虚拟机内部 vIOMMU 的管辖范围。设备的所有 DMA 请求都必须先经过虚拟机内核的 IOMMU 页表翻译，才能访问内存。

        注：

        1. 这个应该是 qemu 本身的虚拟 iommu 对 dev 进行 pass through 了吧

    如果你的 QEMU 启动命令行同时包含了这两句：

    ```bash
    -device intel-iommu,intremap=on,caching-mode=on \
    -device vfio-pci,host=03:00.0,iommu_platform=off
    ```

    这是一种非常高级的架构，通常用于**云原生、DPDK 开发或嵌套虚拟化**：

    注：

    1. 未测试过，存疑, 不知道是否确实能跑通
* `-device intel-iommu,intremap=on,caching-mode=on` 详细讲讲这行 qemu 启动参数的作用

    这行参数是 QEMU/KVM 虚拟机中用于开启和配置虚拟 Intel IOMMU（vIOMMU）的核心指令。

    简单来说，它的主要作用是**在虚拟机内部模拟一个硬件级别的 IOMMU（Intel VT-d）**。这不仅能提升虚拟机的安全性（防止恶意 DMA 攻击），更是虚拟机内部实现**嵌套虚拟化（Nested Virtualization）**、**高效率 SR-IOV 直通**以及 **DPDK/VFIO 高性能网络驱动**的必备基石。

    下面我们逐一拆解这行参数的每个部分：

    **参数逐条拆解**

    1. `-device intel-iommu`

        * **字面意思**：向虚拟机添加一个型号为 `intel-iommu` 的虚拟设备。
        * **深层作用**：它会向虚拟机的 BIOS/UEFI 和操作系统宣告：“这台机器拥有 Intel VT-d 硬件支持”。虚拟机系统启动时，会加载 `intel_iommu=on` 驱动，并建立起一套属于虚拟机内部的 DMA 内存映射和 I/O 页表。

    2. `intremap=on` (Interrupt Remapping，中断重定向)

        * **字面意思**：开启虚拟中断重定向功能。

        * **深层作用**：
            
            * 在物理机中，中断重定向用于将外设的中断信号安全地路由到指定的 CPU 核心上，并能有效防止恶意的 MSI（消息触发中断）攻击。

            * 在虚拟机里（尤其是开启了 `-smp` 多核心时），**如果不开启 `intremap=on`，虚拟机的内核通常会直接拒绝启动 IOMMU 功能**（你会看到 `Intel-IOMMU: ioapic... missing` 之类的内核报错）。

            * **关键应用**：它是虚拟机内运行 **VFIO 驱动**（比如在虚拟机里把一个网卡虚拟 VF 直通给一个容器）的**硬性前提**。

        注：

        1. 这个还是没太看明白。不启动中断重定向会发生什么？

    3. `caching-mode=on` (缓存模式)

        * **字面意思**：开启 IOMMU 的上下文/IOTLB 缓存使能。

        * **深层作用**：
            
            * **这是专门为“软件模拟”优化的一个开关**。在物理硬件中，IOMMU 硬件可以直接扫描内存中的页表。但在虚拟机（QEMU）中，Guest OS（虚拟机系统）修改了它自己的 IOMMU 页表，QEMU 作为应用层是无法实时感知的。

            * 开启 `caching-mode=on` 后，Guest OS 每次修改 IOMMU 页表或使能某些映射时，都会**强制触发一个无效化（Invalidation）操作**。这个操作会被 QEMU 捕获（Trap），从而让 Host（宿主机）能够实时跟进并更新影子页表（Shadow Page Tables）或真实的硬件 IOMMU。

            * **关键应用**：如果要实现 **VFIO 直通设备的嵌套透传**（即：物理机直通给虚拟机 A，虚拟机 A 再直通给它内部的容器或 KVM 虚拟机 B），**必须**开启此参数，否则 Guest 内部的 DMA 映射无法同步到宿主机，设备无法正常工作。

    3**性能损耗提示**：

    开启 vIOMMU（尤其是 `caching-mode=on` 带来的 Trap）会引入额外的 CPU 弹跳和上下文切换损耗。**如果虚拟机不需要做内部的设备直通或高性能网络开发，不建议开启此参数**，纯粹的 CPU/内存虚拟化不需要它。

    注：

    1. 这行参数没有实际验证过，需要 qemu 环境验证一下

* `/sys/bus/pci/devices/<bdf>/iommu_group`

    绑定 vfio-pci（示例）

    ```bash
    echo 0000:03:00.0 > /sys/bus/pci/devices/0000:03:00.0/driver/unbind
    echo 15b3 1019 > /sys/bus/pci/drivers/vfio-pci/new_id
    ```

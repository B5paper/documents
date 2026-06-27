## cache

* zebu 中的 vhost 是否有 numa 参数？

    在 **zebu** 仿真平台的 **vhost** 配置中，通常 **没有专门的 NUMA 参数**。

    以下是详细说明和相关建议：

    1. **vhost 的基本作用**

       - 在 Zebu 中，vhost 是一种用于连接虚拟机和硬件仿真环境的虚拟化技术。
       - 它主要用于模拟硬件设备（如磁盘、网络设备）的虚拟化后端。

    2. **NUMA 相关的配置位置**

       - 在 ZebU 环境中，NUMA 配置通常在 **更高层次的配置文件中**，例如：
         - **SoC 或系统级配置文件**（如 `system.conf`、`topology.conf`）
         - **虚拟机配置**（如 QEMU/KVM 的启动参数）
         - **主机 BIOS/操作系统**的 NUMA 设置

    3. **如何为 Zebu 配置 NUMA**

       - **方法 1：通过 Zebu 配置文件**
         - 在 Zebu 的系统配置中，可以指定 CPU 和内存的 NUMA 节点布局。
         - 示例配置：
           ```ini
           [numa]
           node0_cpus = 0-3
           node0_mem = 2048
           node1_cpus = 4-7
           node1_mem = 2048
           ```

       - **方法 2：通过 QEMU/KVM 参数**
         - 如果 Zebu 与 QEMU/KVM 结合使用，可以在 QEMU 启动命令中指定 NUMA 配置：
           ```bash
           -numa node,nodeid=0,cpus=0-3,mem=2048 \
           -numa node,nodeid=1,cpus=4-7,mem=2048
           ```

       - **方法 3：主机操作系统设置**
         - 确保主机 BIOS 中启用 NUMA。
         - 使用 `numactl` 命令绑定进程到特定 NUMA 节点。

    4. **vhost 与 NUMA 的关联**

       - vhost 本身不直接管理 NUMA，但 **vhost 进程可以绑定到特定的 NUMA 节点**。
       - 通过将 vhost 进程与虚拟机 CPU/内存放在相同 NUMA 节点，可以提升性能。

    5. **最佳实践建议**

       - **统一配置**：在系统级配置中定义 NUMA，确保所有组件（包括 vhost）遵循同一拓扑。
       - **进程绑定**：使用 `numactl` 或 `taskset` 将 vhost 进程绑定到特定 CPU。
       - **性能监控**：使用 `numastat` 或 `perf` 工具监控 NUMA 相关的性能指标。

    6. **检查文档或支持**

       - 查阅 Zebu 用户手册中关于 **NUMA 配置** 的部分。
       - 联系 **Cadence 技术支持** 获取针对您 Zebu 版本的具体指导。

    总结

    虽然 **vhost 本身没有 NUMA 参数**，但您可以通过 **系统级配置、QEMU 参数或进程绑定** 来实现 NUMA 优化。建议在 Zebu 的系统配置文件中统一管理 NUMA 设置，以确保仿真环境的最佳性能。

* 我想让 zebu 仿真出的 pcie 设备挂载到 emu 虚拟机里指定的 numa 节点上，怎么办？

    在 Zebu 仿真环境中，让仿真出的 PCIe 设备挂载到虚拟机（如 QEMU/KVM）的指定 NUMA 节点上，需要进行 **系统级配置** 和 **设备绑定**。以下是具体步骤：

    ---

    **步骤 1：配置 Zebu 仿真环境的 NUMA 拓扑**

    在 Zebu 的 **系统配置文件**（如 `system.conf` 或 `topology.conf`）中定义 NUMA 节点：

    ```ini
    # 示例：配置两个 NUMA 节点
    [numa]
    node0_cpus = 0-3      # 节点 0 的 CPU
    node0_mem = 4096      # 节点 0 的内存（MB）
    node1_cpus = 4-7      # 节点 1 的 CPU
    node1_mem = 4096      # 节点 1 的内存（MB）
    ```

    ---

    **步骤 2：在仿真中定义 PCIe 设备**

    确保 PCIe 设备在 Zebu 仿真中被正确定义，并分配到特定的物理位置（如总线/设备号）。

    ```ini
    # 示例：定义一个 PCIe 设备
    [pcie_device_0]
    type = pcie_endpoint
    bus = 0
    device = 0
    function = 0
    # 可能需要的额外参数，如 BAR 大小等
    ```

    ---

    **步骤 3：配置虚拟机的 NUMA 拓扑**

    在启动虚拟机（如 QEMU/KVM）时，明确指定 NUMA 节点，并将 PCIe 设备绑定到特定节点。

    ### **方法 A：通过 QEMU 命令行配置**
    ```bash
    # 定义 NUMA 节点
    -numa node,nodeid=0,cpus=0-3,mem=4096 \
    -numa node,nodeid=1,cpus=4-7,mem=4096 \

    # 添加 Zebu 仿真的 PCIe 设备，并将其绑定到 NUMA 节点 1
    -device pcie-root-port,bus=pcie.0,id=rp1 \
    -device x3130-upstream,bus=rp1,id=us1 \
    -device xio3130-downstream,bus=us1,id=ds1 \
    -device pcie-pci-bridge,bus=ds1,id=bridge1 \
    -device pci-assign,host=01:00.0,bus=bridge1,addr=0x0,numa-node=1
    ```

    **关键参数**：
    - `numa-node=1`：将 PCIe 设备绑定到 NUMA 节点 1。

    ---

    **步骤 4：使用 VFIO 或 PCIe 直通**

    如果通过 PCIe 直通（如 VFIO）将 Zebu 仿真的 PCIe 设备传递给虚拟机：

    1. **在主机上绑定设备到 VFIO 驱动**：
       ```bash
       # 查找设备 BDF（总线:设备.功能）
       lspci -nn | grep "Cadence Zebu"  # 假设设备标识为 Zebu 仿真

       # 解绑当前驱动，绑定到 vfio-pci
       echo 0000:01:00.0 > /sys/bus/pci/devices/0000:01:00.0/driver/unbind
       echo vfio-pci > /sys/bus/pci/devices/0000:01:00.0/driver_override
       echo 0000:01:00.0 > /sys/bus/pci/drivers/vfio-pci/bind
       ```

    2. **在 QEMU 启动时指定设备 NUMA 节点**：
       ```bash
       -device vfio-pci,host=01:00.0,bus=bridge1,addr=0x0,numa-node=1
       ```

    ---

    **步骤 5：验证 NUMA 绑定**

    在虚拟机内部，验证 PCIe 设备是否位于正确的 NUMA 节点：

    ```bash
    # 1. 查找设备的总线地址
    lspci | grep "Cadence Zebu"

    # 2. 查看设备对应的 NUMA 节点
    cat /sys/bus/pci/devices/0000:01:00.0/numa_node
    # 输出应为 1（如果绑定到节点 1）

    # 3. 查看设备本地内存访问性能
    numactl --hardware
    numactl --cpubind=1 --membind=1 ./test_pcie_device
    ```

    ---

    **步骤 6：优化性能（可选）**

    1. **绑定 vCPU 到相同 NUMA 节点**：
       ```bash
       # 在 QEMU 中，将 vCPU 绑定到物理 CPU
       -vcpu vcpunum=0,affinity=4 \
       -vcpu vcpunum=1,affinity=5 \
       ...  # 对应 NUMA 节点 1 的 CPU（4-7）
       ```

    2. **配置大页内存**（如果使用大页）：
       ```bash
       # 在 NUMA 节点 1 上分配大页
       echo 1024 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
       ```

    ---

    **关键注意事项**

    1. **Zebu 版本差异**：不同 Zebu 版本对 NUMA 和 PCIe 的支持可能不同，请查阅对应版本的 **《Zebu User Guide》** 或 **《PCIe Emulation Guide》**。

    2. **QEMU/KVM 版本**：确保 QEMU/KVM 版本支持 `numa-node` 参数（较新版本通常支持）。

    3. **设备直通限制**：某些 PCIe 设备可能不支持 NUMA 绑定，需验证设备和驱动兼容性。

    4. **性能监控**：使用 `numastat`、`perf` 或 Zebu 自带的性能分析工具监控 NUMA 效果。

    ---

    **总结**

    要实现 Zebu 仿真的 PCIe 设备挂载到虚拟机指定 NUMA 节点：

    1. **配置 Zebu 的 NUMA 拓扑**。
    2. **在虚拟机启动时明确指定 NUMA 节点**，并使用 `numa-node` 参数绑定 PCIe 设备。
    3. **使用 VFIO 直通**时，确保主机驱动绑定正确。
    4. **在虚拟机内部验证** NUMA 节点绑定。

    如果遇到问题，建议：
    - 查阅 **Cadence Zebu 文档** 中关于 PCIe 和 NUMA 的章节。
    - 联系 **Cadence 技术支持** 获取针对您具体配置的指导。

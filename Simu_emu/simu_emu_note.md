# Simu Emu Note

这里主要记录 qemu 设备模拟，cmodel, arch model, emulation 相关的笔记。

## cache

* 搭建 qemu 环境

    * 60 机器上的 virt-manager 无法正常启动，qemu-system-x86_64 启动 qcow 图形界面太卡。如果使用无图形界面，速度应该会快一些。但是目前更好的办法是使用 54 机器开发。

        2025/07/01/00: 54 机器环境不完整，最终还是到 60 机器上搭建 qemu 了。

    * 编译时报错：

        ```
        [229/231] Linking target tests/qtest/qos-test
        [230/231] Linking target storage-daemon/qemu-storage-daemon
        [231/231] Linking target qemu-system-x86_64
        ert build fails
        build fails
        ```

        原因：

        依赖未安装完全。需要照着 arch 组的文档安装 apt 和 python 的依赖。
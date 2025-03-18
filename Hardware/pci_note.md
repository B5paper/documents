# PCI Note

## cache

* pci 相关

    * The full system path to the device might look like this: `<bus id>:<device id>:<function id>`

    * pci device 的 vender id 和 device id 在<https://pcisig.com/>注册

    * `lspci -nn`可以列出当前设备的 pci 信息

        example:

        ```
        00:03.0 Ethernet controller [0200]: Intel Corporation 82540EM Gigabit Ethernet Controller [8086:100e] (rev 02)
        ```

        其中，

        `00:03.0`: `<bus num>:<device num>.<function num>`

        `0200`: device class

        `8086:100e`: `<vendor id>:<product id>`

    * `lspci`中的 pci device 功能描述的信息是从`/usr/share/hwdata/pci.ids`中拿到的，这个路径是一个 symbolic link，真正的文件放在`/usr/share/misc/pci.ids`中。

        最新的 pci id 信息可以在<https://pci-ids.ucw.cz/>找到。

        最新的 vendor id 信息可以在<https://pcisig.com/membership/member-companies>找到。

    * “Base Address Registers” (BARs)

    * Bus enumeration is performed by attempting to read the vendor ID and device ID (VID/DID) register for each combination of the bus number and device number at the device’s function #0.

    * pci 的前 64 个字节为 configuration register，每个硬件厂商都要实现这些寄存器

        图：
        
        ![](reg_desc.png)

        * Please note that byte order is always little-endian. This might be important if you are working on some big-endian system.

        * command registers: 2 bytes

            command registers 占用 2 个字节，共 16 位，但是只用到了低 11 位。这些数值由操作系统写入。

            * 第 0 位： I/O Space Enable

            * 第 1 位： Memory Space Enable

            * 第 2 位： Bus Master Enable

            * 第 3 位： Special Cycles

            * 第 4 位： Memory Write and Invalidate Enable

            * 第 5 位： VGA Palette Snoop Enable

            * 第 6 位： Parity Error Response

            * 第 7 位： Stepping Control

            * 第 8 位： SERR# Enable

            * 第 9 位： Fast Back-to-Back Enable

            * 第 10 位： Interrupt Disable

        * Status registers: 2 bytes

            status registers 占用 2 个字节，共 16 位，实际只用到了高 13 位，[2:0] 位都没有被使用。这些数据由 device 填写，用于上报基本信息。

            （索引从 0 开始计数）

            第 3 位： Interrupt Status

            第 4 位： Capabilities List

            第 5 位： Reserved

            第 6 位： Reserved

            第 7 位： Fast Back-to-Back Capable

            第 8 位： Master Data Parity Error

            第 9 位，第 10 位： DEVSEL Timing

            第 11 位： Signaled Target-Abort

            第 12 位： Received Target-Abort

            第 13 位： Received Master-Abort

            第 14 位： Signaled System Error

            第 15 位： Detected Parity-Error

        * Revision ID: 1 byte

            不知道干嘛用的

        * Class Code： 3 bytes

            用于识别设备类型，比如 Network adapter

            <https://wiki.osdev.org/PCI#Class_Codes>这里有常用的 class code。

        * Base Address Registers

            filled by the Linux kernel and used for the IO operations.

## note

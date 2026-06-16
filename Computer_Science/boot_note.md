# Boot Note

这篇笔记主要记录和磁盘，分区，系统引导，启动等相关的知识。

Ref: 

* <http://jdebp.info/FGA/efi-boot-process.html>

* <https://www.happyassassin.net/posts/2014/01/25/uefi-boot-how-does-that-actually-work-then/>

* <https://www.scaler.com/topics/operating-system/master-boot-record/>

Master Boot Record (MBR) is the first sector of a hard disk and the place where all the information about the disk and booting can be found. It is the most essential part of the booting process. Along with the bootloader program, MBR also contains details regarding the partitions of the hard disk. The size of MBR is commonly less than or equal to 512 bytes.

As the name suggests, Master Boot Record, commonly known as MBR, is the first (main) sector of a hard disk and determines the location of the operating system (OS) to complete the execution of the booting process. MBR is often called the partition sector or master partition table because of its components. It consists of a table that holds details of the partitions of the hard disk and their locations. MBR also contains a record that helps in booting up the entire Operating System.

When a computer is first turned on, it runs a special program called Basic Input Output System (BIOS) that is stored in the Read-Only Memory (ROM). BIOS contains the code that locates and executes MBR. MBR contains a partition table holding locations of various hard disk partitions which further helps in loading the operating system.

The MBR is another de facto standard; basically, the very start of the disk describes the partitions on the disk in a particular format, and contains a 'boot loader', a very small piece of code that a BIOS firmware knows how to execute, whose job it is to boot the operating system(s). (Modern bootloaders frequently are much bigger than can be contained in the MBR space and have to use a multi-stage design where the bit in the MBR just knows how to load the next stage from somewhere else, but that's not important to us right now).

Both BIOS and UEFI are types of firmware for computers. BIOS-style firmware is (mostly) only ever found on IBM PC compatible computers. UEFI is meant to be more generic, and can be found on systems which are not in the 'IBM PC compatible' class.

EFI (Extensible Firmware Interface)

Bootstrapping on EFI involves a boot manager that is built in to the firmware. EFI systems do not rely upon bootstrap programs stored in boot records (VBRs or MBRs) at all. The firmware knows how to read a partition table and understands the FAT filesystem format. (IBM PC compatible firmware does neither.) A designated partition, that is formatted with the FAT filesystem format and identified with a specific well-known partition type, is known as the EFI System Partition. (The EFI System Partition is a true system volume that is identified by its type code in the partition table, not a Poor Man's equivalent that is guessed at like Microsoft's System Reserved Partition is.) It contains boot loader programs, which are EFI executable programs that are loaded and run by the EFI boot manager.

EFI executable programs are standalone programs, that use only machine firmware services and that do not require an underlying operating system in order to run. They can be either operating system boot loaders or "pre-boot" maintenance/diagnosis programs.


计算机启动流程：

1. 打开电源，BIOS 初始化硬件，并在磁盘的起始位置找到 MBR，并运行 MBR

1. MBR 从 active partition 的 bootsector 启动系统

1. bootsector 从文件系统中运行 bootloader

MBR 的结构：

Partition table 的 1st entry ~ 4th entry 结构如下：

1. Boot signature
1. Start head
1. Start sector
1. Start cylinder
1. System signature
1. End head
1. End sector
1. End cylinder
1. No. of sectors before the partition
1. No. of sectors in the partition

从 5th entry 开始，是 Identification code。

The MBR mainly consists of 3 parts:

1. Master Boot Routine (Bootstrap code area)

    Start address: `0x0000 (0)`

    Length: `446` bytes

    The master boot routine contains a variable loader code that passes control to the Operating System that has been registered in the partition table. The size of this component is 446 Bytes.

    格式：从 sector 1 开始，格式如下：

    ```
    00 00 Disk ID (4 bytes)
    Partition 1
    Partition 2
    Partition 3
    Partition 4
    55 AA
    ```

    Partitoin 1 structure example:

    1. Status `08 00`

    1. Start

        Cylinder, Head, Sector

    1. Partition Type

    1. End

        Cylinder, Head, Sector

    1. LBA

    1. LBA

    1. LBA

    1. LBA

    1. Partition Length

    1. Partition Length

    1. Partition Length

    1. Partition Length

1. Disk Partition Table (DPT)

    Start address: `0x01BE (446)`

    Length: `64` bytes (`16` bytes per partition, `4` primary partitions totally)

    The disk partition table is located at the first sector of each hard disk and contains locations of the partitions. The disk partition table is usually 64 bytes long. It contains a maximum of 4 partitions that can be 16 bytes each. If the user requires more partitions, they are free to create an extended partition.

    1. the first partition entry

        Start address: `0x01BE (446)`

        Length: `16` bytes

    1. the second partition entry

        Start address: `0x01CE (462)`

        Length: `16` bytes

    1. the third partition entry

        Start address: `0x01DE (478)`

        Length: `16` bytes

    1. the fourth partition entry

        Start address: `0x01EE (494)`

        Length: `16` bytes

1. Identification Code (Boot signature)

    Start address: `0x01FE (510)`

    Length: `2` bytes

    content: `0x55 0xAA`

    Identification code is used to identify an MBR and acts as a closing signature. Its value is AA55 H and may also be written as 55AA H. The identification code is 2 bytes long.

**BIOS 的功能**

The BIOS program first evaluates the system hardware and then checks available boot devices (devices that contain files which help a system to boot) in line with the boot order stored in CMOS (Complementary Metal Oxide Semiconductor technology that refers to the small amount of memory on the motherboard that stores settings for BIOS).

Following that, BIOS reads the first sector (the MBR sector) to 0000: 7C00H. The BIOS then checks the end of the sector at 0000: 7CFEH-0000 to determine if the final signature is 55AAH.

If it is 55AH, BIOS will transfer control to the MBR in order to boot the Operating System. If not, the BIOS will look for additional bootable devices. If no bootable device is present, we will see the warning "NO RAM BASIC," and OS will not boot.

**MBR 的命令行修复工具**

bootrec

**Features Of MBR**

1. Partitions in MBR can either be Primary(a partition that is needed to store and boot an operating system), Extended(partitions that are used to create logical partitions), or Logical(partitions that are created to extend the limitation of only 4 partitions).

1. The MBR partition table consists of details of only primary and extended partitions.

1. Commonly, MBR can have a maximum of 4 partitions with each partition having 16 bytes space and thus a total of 64 bytes for all the partitions. However, certain latest versions can support up to 16 partitions.

1. Since the maximum size of MBR is 512 bytes, disks that are formatted with MBR require a maximum of 2TB available for use.

**Limitations Of MBR**

1. The limit of 4 primary partitions can cause trouble when you need a higher number of primary partitions in the device.

1. An MBR partition cannot hold a hard drive that has a size bigger than 2TB.

1. MBR Drives aren't very reliable since they store all the data in a single location. If overwritten or corrupted, it will cause booting issues.

## GPT

GUID Partition Table (GPT) is a standard table that is used to store details of partition tables. GPT is considered to be the successor of MBR as it maintains all partition-related data and the boot code of the Operating System.

Unlike in MBR, you can create multiple (up to 128) partitions in GPT. Also, a supported hard disk drive size can be as big as 9.44 million TB. This means that GPT disk can provide much more storage than MBR.

Since in a GPT data is stored all across the drive, there is a guarantee that in case a partition is deleted or damaged, data will still be retrievable. GPT also makes use of CRC (Cyclic Redundancy Check - an error detection code that is based on the remainder of polynomial division) to ensure data security. This error check mechanism provides GPT with a higher factor of reliability.

Solid State Drive (SSD) is able to boot Windows far more quickly as compared to Hard disk drive (HDD). To make the best use of this benefit of speed, Unified Extensible Firmware Interface (UEFI) based systems are needed, which makes GPT a better choice.

Even though GPT has an edge over MBR, users might choose MBR depending on the OS that they are working with. MBR is a better choice for users who work on previous versions of Windows or with drives that are less than 2TB large. Also, in case a Windows system is being booted using BIOS, MBR will be preferred.
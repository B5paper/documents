# Linux Note

本篇笔记主要记录一些和 linux 相关的配置细节。

## cache

* 讲讲标准亲和性 (smp_affinity)

    在 Linux 内核的中断管理中，**标准亲和性（SMP Affinity）** 是一个控制“哪个 CPU 核心来处理哪个硬件中断”的机制。

    在多核（SMP, Symmetric Multi-Processing）系统中，如果所有硬件中断都扔给 CPU0 处理，CPU0 就会累死（陷入上下文切换和中断处理），而其他核心却在闲逛。`smp_affinity` 就是用来打破这种不公平，实现**中断负载均衡**的。

    ---

    1. 核心机制：掩码（Bitmask）

        中断亲和性是通过 Linux 的 `/proc` 文件系统来配置的，路径通常为：
        `/proc/irq/[IRQ_NUMBER]/smp_affinity`

        这个文件存储的是一个**十六进制的位掩码（Bitmask）**。每一位代表一个 CPU 核心：
        * **01 (二进制 0001)**：代表 CPU0
        * **02 (二进制 0010)**：代表 CPU1
        * **04 (二进制 0100)**：代表 CPU2
        * **08 (二进制 1000)**：代表 CPU3
        * **0f (二进制 1111)**：代表 CPU0-3 都可以处理该中断

        举例说明：

        如果你想让中断号为 44 的网卡中断只在 CPU0 和 CPU2 上处理：
        1.  计算掩码：$2^0 (CPU0) + $2^2 (CPU2) = 1 + 4 = 5$。
        2.  写入值：`echo 5 > /proc/irq/44/smp_affinity`

    ---

    2. smp_affinity vs smp_affinity_list

        除了十六进制掩码，Linux 还提供了一个更人性化的文件：
        `/proc/irq/[IRQ_NUMBER]/smp_affinity_list`

        * **`smp_affinity`**：使用十六进制（如 `f`）。
        * **`smp_affinity_list`**：使用十进制列表（如 `0-3` 或 `0,2`）。两者效果完全一样，改一个，另一个会自动同步。

    ---

    3. 为什么要手动设置亲和性？

        虽然内核有一个 `irqbalance` 服务会自动分配中断，但在高性能场景下，手动设置（Pinning）至关重要：

        1.  **减少缓存失效（Cache Miss）**：
            如果网卡中断一会儿在 CPU0，一会儿在 CPU3，那么处理数据包的内存缓存就会在不同核心间频繁同步，性能大打折扣。将中断固定在特定核心，可以提高 **Cache Locality**。
        2.  **避免“零号核心”瓶颈**：
            很多系统默认把所有中断都压在 CPU0 上，导致 CPU0 的 `softirq` 占用过高。手动将不同硬件（如磁盘、网卡）的中断分散到不同核心，可以显著提升吞吐量。
        3.  **实时性要求**：
            在实时系统中，我们会把某些核心“隔离”出来（使用 `isolcpus`），不让它们处理任何普通中断，只跑特定的核心任务。

    ---

    4. 操作步骤与验证

        第一步：查找中断号

        通过 `/proc/interrupts` 查看当前系统的中断分布：
        ```bash
        cat /proc/interrupts | grep eth0
        # 假设看到中断号是 19
        ```

        第二步：检查当前绑定

        ```bash
        cat /proc/irq/19/smp_affinity_list
        # 输出可能是 0-7，表示 8 个核都能处理
        ```

        第三步：修改绑定

        ```bash
        # 只允许 CPU4 处理
        echo 4 > /proc/irq/19/smp_affinity_list
        ```

        第四步：观察效果

        再次运行 `watch -n 1 cat /proc/interrupts`，你会发现 19 号中断的计数器只有 CPU4 在增加了。

    ---

    5. 注意事项

        * **硬件限制**：某些老的硬件或特定的中断控制器（如某些单队列网卡）不支持更改亲和性，此时写入可能会报错或无效。
        * **irqbalance 服务**：如果你的系统中运行着 `irqbalance` 守护进程，它可能会覆盖你的手动设置。在进行深度优化时，通常建议禁用它：`systemctl stop irqbalance`。
        * **多队列网卡**：现代万兆网卡通常有多个队列（Queue），每个队列对应一个独立的中断。最完美的方案是将 Queue0 绑定到 CPU0，Queue1 绑定到 CPU1，以此类推。

    **你想了解如何编写一个脚本来自动将多队列网卡的中断均匀分布到所有 CPU 核心上吗？**

* 简述 flush_cache_all() 的原型、作用、用法

    在 Linux 内核的内存管理与体系结构层（Arch-level）中，`flush_cache_all()` 是一个用于**强制同步物理缓存与主存**的底层函数。

    ---

    1. 函数原型
        
        该函数通常定义在体系结构相关的头文件中（如 `arch/arm/include/asm/cacheflush.h` 或 `arch/arm64/...`），其原型非常简单：

        ```c
        void flush_cache_all(void);
        ```

        * **输入**：无。
        * **返回值**：无。
        * **注意**：这是一个**体系结构相关（Architecture Specific）**的函数。在某些架构（如 x86）中，由于硬件保证了缓存一致性，该函数可能被定义为空操作（NOP）；而在 ARM、MIPS 等架构中，它是真实存在的。

    ---

    2. 核心作用
        
        `flush_cache_all()` 的主要作用是：**使系统中所有 CPU 的指令缓存（I-Cache）和数据缓存（D-Cache）全部失效（Invalidate）并将脏数据写回（Write-back）到主存。**

        * **同步数据**：确保内存中的内容是最新的，防止 CPU 读取到缓存中的旧数据。
        * **全局可见性**：这是一个“大杀器”级别的操作，它作用于系统范围内的所有缓存行，开销非常大。
        * **防止一致性问题**：在涉及非缓存一致性的操作（如修改内核页表、DMA 传输切换、系统挂起/恢复）时，确保状态一致。

    ---

    3. 典型用法与场景
        
        由于其性能损耗极大，普通驱动程序**极少**直接调用它。它通常出现在内核核心代码的以下场景：

        A. 修改地址空间/页表

        当内核大规模修改全局页表（特别是改变内存映射属性，如从“可缓存”改为“不可缓存”）时，需要刷掉旧的缓存内容。

        B. CPU 状态切换（如挂起/断电）

        在 CPU 进入深度睡眠（Deep Sleep）或彻底关机前，必须将缓存中的脏数据全部推回到 RAM 中，否则断电后数据将永久丢失。

        C. 系统初始化或 CPU 热插拔

        当一个新的 CPU 核心上线时，为了确保它看到的内存状态与当前系统一致，可能会执行该操作。

    ---

    4. 代码用法示例（内核片段）

        ```c
        #include <asm/cacheflush.h>

        void my_kernel_function(void) {
            // 假设进行了一些敏感的底层内存属性修改
            update_critical_page_tables();

            // 强制全局缓存刷写，确保修改对所有核心和外设生效
            flush_cache_all();
            
            // 通常后面还会紧跟一个屏障指令
            isb(); // Instruction Synchronization Barrier (ARM)
        }
        ```

    ---

    5. 注意事项（重要）

        1.  **性能杀手**：调用此函数会导致全系统的流水线停顿和内存带宽激增。在驱动开发中，应优先使用针对特定地址范围的函数，如 `flush_cache_range(vma, start, end)`。
        2.  **SMP 限制**：在多核（SMP）系统中，调用 `flush_cache_all()` 通常只会影响执行该命令的当前 CPU。如果需要刷写所有核心，内核通常会使用 `smp_call_function()` 来触发。
        3.  **替代方案**：
            * **DMA 场景**：请使用 `dma_map_single()` 或 `dma_sync_single_for_device()`。
            * **模块加载**：内核会自动处理模块指令的缓存同步（`flush_icache_range`）。

    **你想了解为什么 x86 架构通常不需要手动调用这个函数，而 ARM 架构却必须频繁处理缓存一致性吗？**

* 简述 get_user_pages() 的原型、作用、用法

    在 Linux 内核开发中，`get_user_pages()`（简称 **GUP**）是一个至关重要的函数。它搭建了**用户空间虚拟内存**与**内核空间物理内存**之间的桥梁。

    ---

    1. 函数原型
        
        随着内核版本的演进，该函数的参数略有变化。以下是现代内核（如 5.x/6.x）中常见的原型（定义在 `<linux/mm.h>`）：

        ```c
        long get_user_pages(unsigned long start, unsigned long nr_pages,
                            unsigned int gup_flags, struct page **pages,
                            struct vm_area_struct **vmas);
        ```

        * **`start`**: 待锁定的用户空间虚拟地址的起始值（必须按页对齐）。
        * **`nr_pages`**: 想要获取的页数。
        * **`gup_flags`**: 标志位（如 `FOLL_WRITE` 表示是否要写入，`FOLL_FORCE` 表示强制访问）。
        * **`pages`**: **核心输出**。一个数组，执行成功后，里面存放着指向每个物理页 `struct page` 的指针。
        * **`vmas`**: 可选输出。存放对应页所属的 VMA 结构指针（不需要可传 `NULL`）。
        * **返回值**: 成功返回实际获取并锁定的页数（正数），失败返回负数错误码。

    ---

    2. 核心作用：从“虚”到“实”
        
        在 Linux 中，用户申请的内存通常是“虚”的（只有虚拟地址，没有分配物理页）。`get_user_pages()` 的作用包括：

        1.  **物理页查找**：遍历进程页表，将虚拟地址转换为物理页结构 `struct page`。
        2.  **强制缺页中断**：如果目标地址尚未分配物理内存，GUP 会自动触发缺页异常（Page Fault），让内核分配物理内存。
        3.  **引用计数加一（Pinning）**：增加物理页的引用计数，防止这些页在内核操作期间被交换（Swap）出去或被系统回收。
        4.  **持久化访问**：允许内核在没有用户进程上下文的情况下（如在中断或异步 DMA 中）直接读写这些物理内存。

    ---

    3. 典型用法
        
        最常见的场景是 **DMA（直接内存访问）** 驱动开发：硬件设备需要直接往用户提供的 Buffer 里写数据。

        **基本步骤：**
        1.  **用户层**：通过 `malloc` 或 `mmap` 申请一块内存，并把地址传给驱动。
        2.  **内核层 (驱动)**：
            * 分配一个 `struct page*` 数组。
            * 调用 `get_user_pages()` 锁定这块内存。
            * 使用 `kmap()` 映射到内核空间访问，或者通过 `sg_set_page()` 交给 DMA 控制器。
            * **重要：** 处理完数据后，必须调用 `put_page()` 释放引用计数，否则会导致内存泄漏。

    ---

    4. 代码示例 (内核驱动片段)

        ```c
        // 假设用户传来了 addr 和 count
        unsigned long addr = ...; 
        int nr_pages = ...;
        struct page **pages;

        // 1. 分配指针数组
        pages = kmalloc_array(nr_pages, sizeof(struct page *), GFP_KERNEL);

        // 2. 锁定并获取物理页
        // FOLL_WRITE 表示我们要往里写数据
        int pinned = get_user_pages(addr, nr_pages, FOLL_WRITE, pages, NULL);

        if (pinned < nr_pages) {
            // 错误处理：释放已获取的页
            for (int i = 0; i < pinned; i++) put_page(pages[i]);
            return -EFAULT;
        }

        // 3. 此时可以直接通过物理页进行 DMA 操作 ...

        // 4. 使用完毕后的收尾工作
        for (int i = 0; i < pinned; i++) {
            if (!PageReserved(pages[i])) {
                set_page_dirty(pages[i]); // 如果改动了数据，标记为脏
            }
            put_page(pages[i]); // 释放页面
        }
        kfree(pages);
        ```

    ---

    5. 注意事项
        
        * **并发安全**：调用 GUP 时通常需要持有当前进程的内存信号量 `mmap_read_lock(mm)`。
        * **变体函数**：
            * 如果在进程上下文中，推荐用 `get_user_pages_fast()`，性能更高且会自动处理锁。
            * 如果要处理的是远程进程的内存，使用 `get_user_pages_remote()`。

    **你想深入了解如何在驱动程序中利用 GUP 配合 DMA 链表（Scatter-Gather）来传输大数据吗？**

* 简述 /proc/config.gz 的作用

    `/proc/config.gz` 是一个**压缩过的内核配置文件**，它提供了当前正在运行的 Linux 内核的编译选项信息。

    1. 核心作用

        -   **查看内核配置**：它允许用户查看当前内核是**如何编译**的，即启用了哪些功能（如文件系统支持、驱动模块、网络协议等），禁用了哪些功能。
        -   **调试与验证**：开发者和系统管理员可以通过它来确认内核是否支持某个特定功能，或在调试问题时了解内核的构建参数。
        -   **方便获取配置**：无需去 `/boot` 目录寻找配置文件，也无需知道内核版本，直接通过 `/proc` 文件系统即可获取。

    2. 使用方法

        由于文件是压缩格式，通常需要配合 `zcat` 来查看：

        **查看完整配置**

        ```bash
        zcat /proc/config.gz | less
        ```

        这会分页显示完整的配置选项，其中每个选项类似 `CONFIG_XXX=y`（编译进内核）、`CONFIG_XXX=m`（编译为模块）或 `# CONFIG_XXX is not set`（禁用）。

        **搜索特定配置项**

        例如，查找是否支持 `ext4` 文件系统：

        ```bash
        zcat /proc/config.gz | grep CONFIG_EXT4
        ```

        **保存为可用的配置文件**

        如果你想基于当前配置重新编译内核，可以将它保存下来：

        ```bash
        zcat /proc/config.gz > .config
        ```

        这样生成的 `.config` 文件可以直接用于内核编译。

    3. 存在的前提条件

        `/proc/config.gz` **并非所有系统都默认存在**。它的存在取决于内核编译时的配置：

        -   **启用**：内核必须启用了 `CONFIG_IKCONFIG` 选项。
        -   **提供 proc 访问**：同时还需要启用 `CONFIG_IKCONFIG_PROC` 选项。
        -   如果这两个选项未开启，`/proc/config.gz` 文件就不存在。

    4. 在容器或某些发行版中的情况

        -   **容器（如 Docker）**：容器内通常没有这个文件，因为容器共享宿主机的内核但拥有独立的文件系统。
        -   **发行版**：许多主流发行版（如 Ubuntu、CentOS）的内核包默认会开启这两个配置选项，因此通常可以找到。如果找不到，也可以去 `/boot/config-$(uname -r)` 查找，那里通常有一个未压缩的版本。

    总结

    `/proc/config.gz` 是当前运行内核的**配置快照**，为用户提供了一个便捷的接口，用于检查内核的编译特性。

* Linux获取文件完整路径

    是的，有几种更快捷的方法可以获取文件的完整路径：

    1. **使用 `realpath` 命令**

        ```bash
        realpath 文件名
        # 或
        realpath file_name
        ```

    2. **使用 `readlink -f`**

        ```bash
        readlink -f 文件名
        ```

    3. **使用 Tab 键自动补全**

        输入 `cat`、`ls -l` 或任何命令后：

        ```bash
        cat /home/username/Doc[按Tab键自动补全]
        ```

    4. **使用 `$PWD` 变量拼接**

        ```bash
        echo "$PWD/file_name"
        ```

    5. **自定义函数（添加到 `~/.bashrc` 或 `~/.zshrc`）**

        ```bash
        # 获取当前目录文件的完整路径
        fp() {
            if [ -z "$1" ]; then
                echo "$PWD"
            else
                echo "$PWD/$1"
            fi
        }

        # 获取多个文件的完整路径
        fps() {
            for file in "$@"; do
                echo "$PWD/$file"
            done
        }

        # 复制完整路径到剪贴板（需要xclip）
        cpfp() {
            echo "$PWD/$1" | xclip -selection clipboard
            echo "已复制: $PWD/$1"
        }
        ```

        然后运行 `source ~/.bashrc` 并使用方法：

        ```bash
        fp file_name       # 显示完整路径
        fps file1 file2    # 显示多个文件的完整路径
        ```

    6. **使用 `find` 命令**

        ```bash
        find "$PWD" -name "file_name"
        ```

    7. **使用 `ls` 的完整路径显示**

        ```bash
        ls -d "$PWD"/*       # 显示当前目录所有文件的完整路径
        ls -d "$PWD"/file*   # 显示匹配文件的完整路径
        ```

    8. **最推荐的快捷方式**：按 Ctrl+Alt+C（在大多数终端中）

        一些终端如 GNOME Terminal、Konsole 支持：

        - **Ctrl+Alt+C**：复制当前路径
        - **Ctrl+Alt+V**：粘贴路径

    **最方便的日常用法**：`realpath` 命令最简单直接，建议安装并设为别名：

    ```bash
    # 如果没有 realpath，先安装
    sudo apt install coreutils  # Debian/Ubuntu

    # 设为别名
    echo "alias fp='realpath'" >> ~/.bashrc
    source ~/.bashrc

    # 使用
    fp filename.txt
    ```

* 在 qemu 里编译内核，并`make install`后，新编译的内核并没有作为 grub 的默认选择，而是被放在了 grub 菜单的最下面。

    看起来`make install`并不能保证新编译的内核是默认选项。

* 编译内核时，CC 过程比较占 CPU，LD 过程不占 CPU。

* 编译 linux kernel 时，需要把 value 变成空字符串的两个 config

    `CONFIG_SYSTEM_TRUSTED_KEYS=""`, `CONFIG_SYSTEM_REVOCATION_KEYS=""`

* Linux filesystems don’t allow a slash (/) to be a part of a filename or directory name.

    In Linux, a directory path string often ends with a slash, such as “/tmp/dir/target/“. 

* sudo 与环境变量

    * 环境变量加到`sudo`前面，环境变量不生效：

        `http_proxy=xxx https_proxy=xxx sudo curl www.google.com`

    * 环境变量加到`sudo`后面才生效：

        `sudo http_proxy=xxx https_proxy=xxx curl www.google.com`

* kmd 添加 blacklist

    在`/etc/modprobe.d`目录下创建文件：

    `blacklist-<kmd_name>.conf`:

    ```conf
    blacklist <kmd_name>
    ```

## 解决中文的“门”字显示的问题

在`/etc/fonts/conf.d/64-language-selector-prefer.conf `文件中，把各个字体中`<family>Noto Sans CJK SC</family>`放到其它字体的最前面即可。

## speakers plugged into the front panel don't palay sound

1. <https://www.baidu.com/s?wd=%5B185018.045206%5D%20Lockdown%3A%20modprobe%3A%20unsigned%20module%20loading%20is%20restricted%3B%20s&rsv_spt=1&rsv_iqid=0xd2327c3800101709&issp=1&f=8&rsv_bp=1&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_dl=ib&rsv_enter=1&rsv_n=2&rsv_sug3=1&rsv_sug2=0&rsv_btype=i&inputT=698&rsv_sug4=699>

1. <https://blog.csdn.net/Lyncai/article/details/117777917>

1. <https://blog.csdn.net/qq_40212975/article/details/106542165>

1. <https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/User/CLI/>

1. <https://www.linuxquestions.org/questions/linux-newbie-8/bluetooth-not-working-in-pop-_os-20-10-a-4175686851/>

1. <https://linuxhint.com/pulse_audio_sounds_ubuntu/>

1. <https://flathub.org/apps/details/org.pulseaudio.pavucontrol>

1. <https://freedesktop.org/software/pulseaudio/pavucontrol/#documentation>

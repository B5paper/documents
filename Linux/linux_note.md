# Linux Note

本篇笔记主要记录一些和 linux 相关的配置细节。

## cache

* dmesg 的文件形式的log存在哪

    **`dmesg` 命令输出的日志主要存在于内核的环形缓冲区中，而不是一个普通的磁盘文件。** 
    
    但这个缓冲区的内容在启动早期和运行时，可以通过几个不同的文件或方式来查看和记录。

    1. `/var/log/dmesg` 或 `/var/log/boot.log`（系统启动日志）

        这个文件**保存了本次系统启动时的内核日志**。当系统完成启动过程后，初始化的内核环形缓冲区的内容会被复制保存到这个文件中。

        *   **目的**：专门用于记录本次启动的硬件检测、驱动初始化等信息。如果你的系统已经运行了很长时间，`dmesg` 命令可能会包含很多新的日志，但 `/var/log/dmesg` 文件仍然只保存着开机时的内容。

        *   **路径可能不同**：

            *   在 **Debian/Ubuntu** 等系统上，通常是 `/var/log/dmesg`。
            *   在 **RHEL/CentOS/Fedora** 等系统上，通常是 `/var/log/boot.log`。
            *   有些系统可能两个都存在，或者有类似 `/var/log/boot` 的日志。

    2. `/var/log/kern.log`（内核运行时日志）

        这个文件由 `rsyslog` 或 `syslog-ng` 等系统日志守护进程管理。内核会将重要的日志消息**持续地**转发给 `syslog`，然后 `syslog` 将它们写入到 `/var/log/kern.log` 文件中。

        *   **目的**：记录系统运行期间产生的所有内核消息，而不仅仅是启动时的。它是一个持续增长的真正的日志文件。

        *   **路径**：通常在 `/var/log/kern.log`。许多系统还会对其进行日志轮替（logrotate），生成如 `kern.log.1`, `kern.log.2.gz` 等归档文件。

    另一个重要的现代系统：Journal（Systemd）

    对于使用 **Systemd** 的现代 Linux 发行版（几乎所有主流发行版），所有日志（包括内核、应用、服务日志）都由 `journald` 统一管理。

    *   **存储位置**：二进制格式的日志文件通常存放在 `/var/log/journal/` 目录下。

    *   **查看工具**：使用 `journalctl` 命令来查看这些日志。

    **与内核日志相关的常用 `journalctl` 命令：**

    ```bash
    # 查看所有内核日志（相当于 dmesg）
    journalctl -k

    # 查看本次启动的内核日志
    journalctl -k -b

    # 实时追踪内核日志
    journalctl -k -f

    # 查看指定时间段的日志
    journalctl -k --since="2023-10-27 09:00:00" --until="2023-10-27 12:00:00"
    ```

    `journalctl` 功能非常强大，可以按时间、服务单元（Unit）、优先级等进行过滤和查询。

    ---

    总结与对比

    | 来源/命令 | 描述 | 数据位置 | 特点 |
    | :--- | :--- | :--- | :--- |
    | **`dmesg`** | 直接读取内核环形缓冲区 | **内存** | 实时，包含从开机到现在的所有记录，但重启后丢失或被覆盖 |
    | **`/var/log/dmesg`**<br>**`/var/log/boot.log`** | 系统启动后保存的启动日志快照 | **磁盘文件** | 只包含**本次启动**的初始化日志，重启后会被新日志覆盖 |
    | **`/var/log/kern.log`** | 由 syslog 记录的持续的内核日志 | **磁盘文件** | 包含**运行期间**的所有内核日志，会轮替和归档，是持久化的 |
    | **`journalctl -k`** | 查询 systemd journal 中的内核日志 | **磁盘文件（二进制）** | 统一日志系统，功能强大，可与其他系统日志关联查询 |

    **所以，如果你在寻找“文件形式的 dmesg log”，你最应该去检查的地方是：**

    1.  **`/var/log/dmesg`** 或 **`/var/log/boot.log`** （用于查看启动日志）
    2.  **`/var/log/kern.log`** （用于查看历史内核日志）
    3.  使用 **`journalctl -k`** 命令 （现代系统的首选查看方式）
* 详细讲讲 xclip 命令的用法

    `xclip` 是 Linux 系统中一个非常强大的命令行工具，它建立起了命令行（Terminal）**与**图形界面剪贴板（Clipboard）之间的桥梁。通过它，你可以直接用命令将文本、文件内容复制到剪贴板，或者将剪贴板的内容输出到终端、保存到文件。

    ---

    ## 核心概念：X11 的三大“剪贴板” (Selections)

    在深入命令之前，需要理解 Linux (X11) 系统中其实有三个不同的“剪贴板”，`xclip` 默认在它们之间切换：

    1. **`primary`（鼠标选中文本，默认值）**：只要你用鼠标在屏幕上高亮选中一段文本，它就自动进入了 `primary`。无需按 Ctrl+C，在别处点击鼠标中键（轮毂）即可粘贴。
    2. **`clipboard`（传统的复制粘贴）**：这就是我们熟知的快捷键剪贴板。通过 `Ctrl+C` / `Ctrl+V` 或右键复制粘贴的内容存放在这里。
    3. **`secondary`（极少使用）**：一个备用的剪贴板，通常可以忽略。

    `xclip` 通过 `-selection`（或简写为 `-sel`）参数来指定操作哪一个剪贴板。

    ---

    ## 常用基本语法

    ```bash
    xclip [参数] [文件]

    ```

    ### 1. 复制内容到剪贴板（写操作）

    * **将文件内容复制到“中键剪贴板” (PRIMARY)**
    ```bash
    xclip message.txt

    ```


    * **将文件内容复制到“常规剪贴板” (CLIPBOARD，最常用)**
    ```bash
    xclip -selection clipboard message.txt

    ```


    * **通过管道（Pipe）将命令的输出复制到剪贴板**
    ```bash
    # 复制当前目录路径
    pwd | xclip -sel c

    # 复制本机的公钥，方便去 GitHub 粘贴
    cat ~/.ssh/id_rsa.pub | xclip -sel clipboard

    ```


    *(注：`-sel c` 是 `-selection clipboard` 的简写，完全等价。)*

    ### 2. 从剪贴板读取内容（读操作）

    要输出剪贴板的内容，需要加上 `-o` (output) 参数。

    * **打印“中键剪贴板”的内容**
    ```bash
    xclip -o

    ```


    * **打印“常规剪贴板”的内容并保存到文件**
    ```bash
    # 打印到终端
    xclip -o -sel c

    # 将剪贴板里的代码或文本直接保存到本地文件
    xclip -o -sel c > code.py

    ```



    ---

    ## 高级进阶用法

    ### 1. 复制/粘贴非文本文件（如图片）

    `xclip` 不仅能处理文本，还能处理多媒体文件。你需要通过 `-t` (target) 参数指定媒体类型 (MIME type)。

    * **将一张图片复制到剪贴板**（比如想在网页或者聊天软件里直接 Ctrl+V 粘贴图片）：
    ```bash
    xclip -sel c -t image/png -i screenshot.png

    ```


    *(注：`-i` 表示输入文件)*
    * **将剪贴板里的图片保存为本地文件**：
    ```bash
    xclip -selection clipboard -t image/png -o > saved_image.png

    ```



    > **提示：如何查看当前剪贴板里支持什么格式？**
    > 使用 `xclip -sel c -t TARGETS -o` 可以列出当前剪贴板中内容的可用 MIME 类型。

    ### 2. 在终端里静默复制

    默认情况下，如果你直接运行 `xclip` 复制大段文本，它可能会在后台等待，直到下一次剪贴板被覆盖。如果你发现终端有些“卡住”，或者想让它完全在后台优雅运行，可以加上 `-quiet` 参数，或者直接重定向。

    ---

    ## 终极实用技巧：设置别名 (Alias)

    天天打 `xclip -selection clipboard` 实在太麻烦了。强烈建议你把它们写进你的 Shell 配置文件（如 `~/.bashrc` 或 `~/.zshrc`）中，伪装成类似 macOS 的 `pbcopy` 和 `pbpaste`：

    ```bash
    # 复制命令：vcopy
    alias vcopy='xclip -selection clipboard'

    # 粘贴命令：vpaste
    alias vpaste='xclip -selection clipboard -o'

    ```

    保存并 `source ~/.bashrc` 后，你的工作流就会变得极度丝滑：

    ```bash
    # 复制一个命令的输出
    echo "Hello Linux" | vcopy

    # 在另一个目录下把内容吐给文件
    vpaste > hello.txt

    ```

* 详细讲讲 xsel 命令的用法

    `xsel` 是 Linux/Unix 系统（基于 X11 窗口系统）中一个非常强大的命令行工具。它的主要作用是**在命令行和系统剪贴板（Clipboard）或选择区（Selection）之间进行数据交互**。

    简单来说，它可以让你用命令把文本复制到剪贴板，或者把剪贴板里的内容打印到终端，是自动化脚本和高效命令行操作的利器。

    ---

    ## 核心概念：X11 的三个“剪贴板”

    在深入命令之前，需要理解 X11 系统中有三个不同的“选择区”（Selections），`xsel` 可以自由操作它们：

    1. **PRIMARY（主选择区）**：这是最常用的。用鼠标左键**高亮选中**一段文本时，它就自动进入了 PRIMARY。通过点击鼠标中键（滚轮）可以将其粘贴。
    2. **CLIPBOARD（剪贴板）**：这就是我们熟知的传统剪贴板。通过 `Ctrl + C` 复制，`Ctrl + V` 粘贴的内容存放在这里。
    3. **SECONDARY（次选择区）**：极少使用，通常作为前两者的备用。

    > `xsel` 默认操作的是 **PRIMARY（主选择区）**。如果你想操作我们平时用的 `Ctrl+C` 剪贴板，必须加上 `-b` 参数。

    ---

    ## 常用参数速查表

    | 短参数 | 长参数 | 作用 |
    | --- | --- | --- |
    | `-p` | `--primary` | 操作 **PRIMARY** 选择区（**默认值**） |
    | `-b` | `--clipboard` | 操作 **CLIPBOARD** 剪贴板（最常用） |
    | `-s` | `--secondary` | 操作 **SECONDARY** 选择区 |
    | `-o` | `--output` | **输出**：将选择区的内容打印到标准输出（屏幕/管道） |
    | `-i` | `--input` | **输入**：将标准输入（来自键盘或管道）写入选择区 |
    | `-c` | `--clear` | **清空**：清空指定的选择区 |
    | `-a` | `--append` | **追加**：将标准输入追加到选择区内容的后面 |

    ---

    ## 实用命令示例（以 CLIPBOARD `-b` 为例）

    因为日常中最常用的是系统的 `Ctrl+C` / `Ctrl+V` 剪贴板，所以以下示例大多结合 `-b` 参数使用。

    ### 1. 复制内容到剪贴板 (Input)

    * **将命令的输出复制到剪贴板**：
    ```bash
    echo "Hello, World!" | xsel -b -i

    ```


    *或者简写为：*
    ```bash
    ls -la | xsel -b

    ```


    *(注：如果不指定 `-o`，默认输入流会自动触发 `-i` 的行为)*
    * **将文件的内容复制到剪贴板**：
    ```bash
    xsel -b -i < ~/.ssh/id_rsa.pub

    ```


    *(非常方便用于复制 SSH 公钥)*

    ### 2. 从剪贴板读取内容 (Output)

    * **在终端打印剪贴板里的内容**：
    ```bash
    xsel -b -o

    ```


    * **将剪贴板的内容保存到文件**：
    ```bash
    xsel -b -o > copied_text.txt

    ```



    ### 3. 清空剪贴板

    * **清空系统剪贴板**：
    ```bash
    xsel -b -c

    ```



    ### 4. 追加内容到剪贴板

    * 如果你想在剪贴板已有文字的后面再加点内容，可以用 `-a`：
    ```bash
    echo "这是第一行" | xsel -b -i
    echo "这是第二行" | xsel -b -a

    ```


    此时再通过 `Ctrl + V` 粘贴，内容会变成两行。

    ---

    ## 进阶技巧与日常联动

    ### 1. 操作鼠标选中的文本（PRIMARY）

    因为 `xsel` 默认就是 `-p`，所以不需要加参数：

    * **打印你刚刚用鼠标划选的文本**：
    ```bash
    xsel -o

    ```


    * **把鼠标划选的文本快速保存**：
    ```bash
    xsel -o >> research_notes.txt

    ```



    ### 2. 在别名（Alias）中简化使用

    天天打 `xsel -b -i` 太麻烦了？可以在你的 `~/.bashrc` 或 `~/.zshrc` 中设置别名，模仿 macOS 的 `pbcopy` 和 `pbpaste`：

    ```bash
    alias pbcopy='xsel -b -i'
    alias pbpaste='xsel -b -o'

    ```

    保存并刷新配置（`source ~/.bashrc`）后，你就可以像这样玩了：

    ```bash
    cat file.txt | pbcopy
    pbpaste > newfile.txt

    ```

    ---

    ## 避坑指南与常见问题

    1. **报错：`xsel: Can't open display: (null)**`
    * **原因**：`xsel` 依赖于 X11 窗口系统。如果你通过纯文本的 SSH 远程登录服务器，或者在没有运行图形界面的 TTY 控制台中运行，就会报这个错。
    * **解决**：如果是在 SSH 下，可以尝试使用 `ssh -X` 开启 X11 转发。


    2. **与 `xclip` 的区别**
    * `xclip` 是另一个非常类似的工具。`xsel` 的优势在于它的语法稍微简单、直观一些（比如 `-o` 和 `-i` 很容易让人联想到 output 和 input），并且在处理大文本或某些特定字符集时，`xsel` 的表现有时更稳定。两者功能基本可以互换。


    3. **Wayland 显示服务器下的兼容性**
    * 现代 Linux 发行版（如 Ubuntu 新版本、Fedora）默认使用 **Wayland** 而不是 X11。虽然许多系统内置了 XWayland 桥接，让 `xsel` 依然能工作，但在纯 Wayland 环境下，更推荐使用 `wl-clipboard` 工具（对应命令为 `wl-copy` 和 `wl-paste`）。
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

# Linux Note

本篇笔记主要记录一些和 linux 相关的配置细节。

## cache

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

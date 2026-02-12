# Linux Note

本篇笔记主要记录一些和 linux 相关的配置细节。

## cache

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

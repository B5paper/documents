# DPKG Note

* `dpkg`命令有时可以替换成`dpkg-deb`

* 显示某个 deb 包的摘要

    `dpkg --info <deb file>`，或`dpkg -I <deb file>`

    可以显示的信息有（摘录）：

    * deb 包按字节算的大小

    * Depends

    * Recommends

    * Architecture

* 列出 deb 包里的文件：`dpkg -c ./path/to/test.deb`

    ```
    -c, --contents archive
    List contents of a deb package.
    ```

* deb 安装包的创建方法

    见`ref_37`。构建方法：`dpkg-deb --build ./hello_world_hlc`，输出如下：

    ```
    dpkg-deb: building package 'hello-from-hlc' in 'hello_world_hlc.deb'.
    ```

    此时即可`sudo dpkg -i ./hello_world_hlc.deb`进行安装，输出如下：

    ```
    Selecting previously unselected package hello-from-hlc.
    (Reading database ... 277890 files and directories currently installed.)
    Preparing to unpack hello_world_hlc.deb ...
    Unpacking hello-from-hlc (0.2) ...
    Setting up hello-from-hlc (0.2) ...
    in post installation from hlc
    ```

    此时安装包中的文件已经复制到对应位置，我们可以直接执行：

    `echo_hello_hlc.sh`, output:

    ```
    hello world from hlc
    ```

    卸载：`sudo dpkg -r hello-from-hlc`，输出：

    ```
    (Reading database ... 277893 files and directories currently installed.)
    Removing hello-from-hlc (0.2) ...
    dpkg: warning: while removing hello-from-hlc, directory '/usr/local/bin' not empty so not removed
    ```

    卸载时，之前复制的文件会被删除。

    注：`control`文件中的 package name 不能有下划线`_`，但是可以有减号。

    debian 有官方文档详细描述安装包的格式：<https://www.debian.org/doc/debian-policy/ch-controlfields.html#s-binarycontrolfiles>
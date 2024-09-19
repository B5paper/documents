# Deb Note

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

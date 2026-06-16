# Linux driver note

一个 pipeline:

假如现在有一个 app 程序员，一个 driver 程序员。

app 程序员要开发一个应用，他只需要

1. 使用 C 语言中的`open()`函数把`/dev/xxx`作为一个普通文件打开

1. 然后使用 C 语言中的`read()`, `write()`等函数读取或写入数据。（盲猜这里的数据有格式要求）

1. 最后使用`close()`关闭文件即可。

C 语言的库会找到`open()`，`read()`，`write()`，`close()`等等的系统调用，比如`Open()`等。进行系统调用。

OS（比如 linux）的工作是：

1. 系统调用会去找驱动模块中的`file_operations`结构体，这个结构体中存了`Open()`等系统调用对应的具体函数。（这个结构类似于 c++ 的虚函数）

driver 程序员为了支持以上的功能，需要编写`file_operations`结构体，以及结构体中各个对应的函数。

## bootloader

计算机平台上用的 bootloader：

* linux 平台下是 grub
* windows 平台下是 bootmgr, ntboot

嵌入式平台上的 bootloader:

* uboot (官网：<https://www.denx.de/project/u-boot/>)

## Linux kernel

source code: <https://kernel.org/>
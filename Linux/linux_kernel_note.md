# Linux kernel note

## cache

* linux kernel

    常见的 kernel 分两种，一种是 Microkernels，所有的函数都通过 interface 与操作系统进行交互。

    另外一种是 Monolithic Kernels，操作系统内核实现了大部分的功能，包括文件系统之类的。

    monolithic kernels 的效率比 microkernels 要高。linux 采用的是 monolithic kernels。
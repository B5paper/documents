# VirtualBox Note

**cache**

* virtual box 使用 efi 启动的 iso 光盘

    virtual box 7.0.1 目前默认是从 MBR 启动引导，如果需要其他的方式启动引导，需要根据开机提示按键，或者在刚启动虚拟机时就一直按 del 键，进入 bios 和启动引导界面。

## Problem Shooting

1. win11 hyper-v compatibility

    virtual box 与 windows 11 的 hyper-v 存在兼容性问题。如果在 windows 11 上开了 hyper-v，那么 virtual box 会变得很慢很卡。

    解决办法是把 host 上的 hyper-v 禁用掉，重启电脑。

    Ref: <https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel>
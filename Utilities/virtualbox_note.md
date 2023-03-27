# VirtualBox Note

## Problem Shooting

1. win11 hyper-v compatibility

    virtual box 与 windows 11 的 hyper-v 存在兼容性问题。如果在 windows 11 上开了 hyper-v，那么 virtual box 会变得很慢很卡。

    解决办法是把 host 上的 hyper-v 禁用掉，重启电脑。

    Ref: <https://wiki.ubuntu.com/Kernel/BuildYourOwnKernel>
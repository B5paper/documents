# qemu note

## cache

* qemu command line and net card

    使用`qemu-system-x86_64`启动的虚拟机，网卡是`ens3`，ip 是`10.0.2.15`, netmask `255.255.255.0`，但是 mac 地址和 host 不一样。

    虚拟机的网卡可以访问 host，也可以访问 internet。由于 guest 的 ip 和 host 相同，所以无法得知 host 是否能 ping 到 guest。

    尝试更改 guest ip 为`10.0.2.16/24`后，guest 失去所有网络访问能力。重新将 ip 改为`10.0.2.15/24`后，也不能恢复访问。

    此时执行`route -n`后发现路由表是空的。重启系统，网络恢复正常后，再次执行`route -n`可以看到路由表有 3 条 entry。说明修改 ip 后网络不通很有可能是路由表的问题。

    正常网络情况下，路由表中的`0.0.0.0`的 gateway 是`10.0.2.2`，在 guest 和 host 上均能 ping 通，但是在 host 上执行`ifconfig`和`ifconfig -a`均看不到这个网卡。

    不清楚这个 route 是什么机制。

## note
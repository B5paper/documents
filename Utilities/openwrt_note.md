# Openwrt Note

interface: 指的是实际网卡和一系列配置，包括 ip 地址，默认网关，获取 ip 地址方式等等。一个物理网卡可以被复用多次，比如同时有 ipv4 和 ipv6 两种地址。

## OpenWrt as client device

指的是将当前路由器（即 openwrt 系统）接入上级路由器，在 lan 接口上，手动设置当前路由器的 ip 地址，子网掩码，使其与上级路由器在同一网段，然后将上一级服务器作为 dhcp 服务器和网关、防火墙。这样一来，我们将新的设备连接到当前路由器的 lan 口或无线网络时，由上级路由器分配 ip 地址，将上级路由器作为网关。

详细的设置步骤：<https://openwrt.org/docs/guide-user/network/openwrt_as_clientdevice>

## OpenWrt as router device

当前路由器工作在两个不同的网段，lan 口的 ip 地址需要设置成和 wan 口不同网段，然后将 wan 口设置成 dhcp client。

## Routed Client

官网教程：<https://openwrt.org/docs/guide-user/network/routedclient>

这里面的 MASQUERADE 指的是用 wifi 连到上一级路由器，然后在 lan 口连接下一级设备。

官方的步骤给得很详细，照着做就行。其中`config 'interface' 'wan' option 'proto' 'dhcp'`，在用图形界面创建的时候，要求必须选一个物理设备，我们可以选自定义，然后不填内容就行了。

在设置 wifi 的时候，图形界面中设置成 client 模式，配置文件就会自动把 mode 设置成 sta。

官网上说要创建一个桥接 bridge，但是目前并没有创建桥接，也能正常工作。不明白为什么。
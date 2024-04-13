# Openwrt Note

interface: 指的是实际网卡和一系列配置，包括 ip 地址，默认网关，获取 ip 地址方式等等。一个物理网卡可以被复用多次，比如同时有 ipv4 和 ipv6 两种地址。

## cache

* openwrt `mwan3`配置虚拟 wan 口

    如果使用正常的方式配置多个 wan 口，那么需要改变 vlan，将一些实体 lan 口划分成 wan 口。然后直接在`network`的配置文件里配置新的`interface`就可以了，此时的`device`肯定是 vlan 已经创建好的，直接把 intarface 对接到 device 上就可以了。

    如果需要用一根网线模拟出多个 wan 口，那么`mwan3`可以完成这个工作。`mwan3`会创建一些 virtual device，

    在 luci 界面“网络->虚拟WAN”里启动 mwan3，可以取消勾选“使用旧的macvlan创建方式”。

    然后在“网络->接口”里可以看到多出来的两个 vwan 接口，点修改，协议改为 dhcp（或者其他可以获得 ip 的协议）。此时如果在“物理设置”里接口选与 wan 相同的接口，那么保存应用后，会看到 vwan 和 wan 有着相同的 ip。这肯定不是我们想要的，我们希望每个 vwan 都有不同的 ip。

    此时我们需要手动设置一下 vwan 对应的 device 的 mac 地址，在“网络->负载均衡->高级->网络配置文件”里，可以看到一些类型为 device 的配置，很容易可以找到 vwan 对应的 device，然后在配置里加一行：

    `option macaddr 'xxxxxxxx'`，可以简单地把原 wan 的 mac 地址改一个数字就行。将`ifname`改成 wan 口的实体网卡，比如`eth0.2`。其实这个`eth0.2`也是前面用 vlan 对`eth0`进行划分虚拟出来的。

    提交后，再回到“网络->接口->选择一个 vwan 修改->物理设置”，此时可用的物理接口里并没有 mwan3 虚拟出来的 device，我们选择自定义，然后手动输入刚才手动指定 mac 的 device 的 name。保存应用。

    这样就可以看到不同的 vwan 接口分配到不同的 ip 了。

* openwrt 让 wifi 模拟一个 client，然后再模拟一个 master

    首先在无线里点“添加”，模式选择“客户端 client”，网络不要选已有的 wan，因为我们需要获得一个独立的 ip，选创建，随便输入一个名字。比如`wifi_wan`。

    ESSID 指的是要连接的别人的 wifi 名称，BSSID 是 MAC 地址，通常只需要填 ESSID 就行了，不需要填 BSSID。

    然后进无线安全页面，加密算法一般选 WPA-PSK/WPA2-PSK 混合加密，算法选自动，然后输入密码。

    保存应用后进入网络->接口页面，可以看到刚才创建成功的新 interface。

    点击修改，把协议改成 dhcp 客户端，然后点切换协议，这样就能拿到 ip 地址了。如果是其他获取 ip 地址的协议，也可以选其他的。

    物理设置里的东西不需要动，看看是否选中的是无线网络 clint。这里也不需要创建桥接，只有 lan 口之间才有创建桥接的需求。（一个从 wan 口进来的数据，没有连到其他 wan 口的需求）

    然后点防火墙，我们选择 wan 防火墙，使用和 wan 相同的策略就行。

    保存应用后，再回到网络->接口，可以看到我们创建的接口现在有了 ip 地址，基本表明已经创建成功了。

    如果有一台电脑是使用网线连着的路由器，那么现在已经可以测试上网了，`ping 223.5.5.5`看看通不通。

    接下来我们重新回到“无线”页面，找找看有没有一个虚拟网卡是 Master 模式，如果有，就直接点修改，如果没有，就创建一个新的。

    基本设置里，ESSID 就是我们要对外广播的 id，模式选择接入点 AP，网络选 lan，因为 lan 口统一分配 ip 就行了，只需要一个网关。

    然后选无线安全，加密方式 WPA-PSK/WPA-PSK2，方式自动，输入密码。最后保存应用就好了。

    此时可以使用手机，无线网卡等无线终端连接上新创建的 wifi 网络，并且正常上网。

* openwrt luci login 界面用户名密码错误

    使用 ssh 登陆路由器：`ssh -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedKeyTypes=+ssh-rsa root@192.168.1.1`

    然后执行：`/etc/init.d/rpcd restart`即可。

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
# tcpdump dump

## cache

* tcpdump 的使用

    `tcpdump -i <interface_name>`可以对指定网络接口（网卡）进行抓包。

    `-v`或`-vv`可以显示更详细的信息。

    `tcpdump -i <interface_name> host <ip_addr>`可以过滤只显示和 ip `<ip_addr>`有关的包。

## note

一个比较好的文档，有时间了可以看下：

<https://docs.netgate.com/pfsense/en/latest/diagnostics/packetcapture/tcpdump.html>

<https://www.cyberciti.biz/faq/linux-monitor-all-network-traffic-except-ssh-port-22/>

<https://danielmiessler.com/study/tcpdump/>
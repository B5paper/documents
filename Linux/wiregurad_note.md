# wireguard note

## cache

* wireguard 启动失败了，怎么办

    ```
    huliucheng@node13:~/wire_guard$ sudo wg-quick up wg0
    [#] ip link add wg0 type wireguard
    [#] wg setconf wg0 /dev/fd/63
    [#] ip -4 address add 10.0.0.1/24 dev wg0
    [#] ip link set mtu 1420 up dev wg0
    [#] iptables -A FORWARD -i wg0 -j ACCEPT; iptables -A FORWARD -o wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
    /usr/bin/wg-quick: line 295: iptables: command not found
    [#] ip link delete dev wg0
    ```

    这个错误是因为**系统没有安装 `iptables`**，而你的 WireGuard 配置中使用了 `iptables` 规则来做 NAT 转发。

    ---

    **解决方案**

    **方案一：安装 iptables（推荐）**

    ```bash
    sudo apt update
    sudo apt install iptables
    ```

    安装完成后，重新启动 WireGuard：

    ```bash
    sudo wg-quick up wg0
    ```

    **方案二：如果你的系统使用 nftables（Ubuntu 新版默认）**

    Ubuntu 新版默认用 `nftables` 替代了 `iptables`。你可以：

    1. **安装 iptables 兼容层**：
    ```bash
    sudo apt install iptables-nftables-compat
    ```

    2. 或者**修改 WireGuard 配置文件**，把 `iptables` 规则改为 `nftables` 规则。

    **方案三：临时禁用防火墙规则（不推荐，仅测试用）**

    编辑 `/etc/wireguard/wg0.conf`，找到并注释掉 `PostUp` 和 `PreDown` 中涉及 `iptables` 的行：

    ```ini
    [Interface]
    Address = 10.0.0.1/24
    PrivateKey = xxx
    # PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -A FORWARD -o wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
    # PreDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -D FORWARD -o wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
    ```

    然后重新启动。

    > ⚠️ 注意：这样做会导致客户端无法通过 WireGuard 访问外网（没有 NAT 转发），只能访问 WireGuard 同网段内的其他设备。

    **方案四：检查 iptables 是否真的缺失**

    ```bash
    which iptables
    ls -la /usr/sbin/iptables
    ```

    如果存在但路径不在 `$PATH` 中，可以尝试：

    ```bash
    sudo /usr/sbin/iptables -L
    ```

    **最可能的原因和推荐操作**

    你用的 Ubuntu 可能是较新版本（22.04+），默认不带 `iptables`。执行：

    ```bash
    sudo apt install iptables
    sudo wg-quick up wg0
    ```

    如果还有问题，检查一下网卡名称是否正确（配置里是 `eth0`，但你的网卡可能叫 `ens3`、`enp0s3` 等）：

    ```bash
    ip link show
    ```

    如果是网卡名不对，修改 `/etc/wireguard/wg0.conf` 中的 `eth0` 为实际网卡名。

    * wg0: flags=209<UP,POINTOPOINT,RUNNING,NOARP>  mtu 1420
            inet 10.0.0.2  netmask 255.255.255.0  destination 10.0.0.2
            unspec 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00  txqueuelen 1000  (UNSPEC)
            RX packets 2  bytes 124 (124.0 B)
            RX errors 0  dropped 0  overruns 0  frame 0
            TX packets 8  bytes 564 (564.0 B)
            TX errors 252  dropped 0 overruns 0  carrier 0  collisions 0
    这个 destination 配置对吗

        **这个配置有问题！** `destination 10.0.0.2` 是不正常的。

        ---

        **问题分析**

        正常情况下，`wg0` 的输出应该是：

        ```
        inet 10.0.0.1/24 scope global wg0
        ```

        或者点对点模式：

        ```
        inet 10.0.0.1 peer 10.0.0.2/32 scope global wg0
        ```

        但你的输出是：

        ```
        inet 10.0.0.2  netmask 255.255.255.0  destination 10.0.0.2
        ```

        这说明：

        - **`destination 10.0.0.2` 是错误的**，它和本机 IP 一样
        - 正常的 `destination` 应该是 `10.0.0.2`（对端 IP）或者根本不显示

        ---

        **原因**

        这个问题的根本原因是 **配置文件中 `Address` 设置错误**。你可能写成了：

        ```ini
        # 错误配置
        Address = 10.0.0.2/24
        ```

        但服务器端应该用 `10.0.0.1`，而客户端才用 `10.0.0.2`。

        ---

        **你需要确认**

        **这是服务器还是客户端？**

        | 角色 | 应该用的 IP |
        |------|-------------|
        | 服务器（VPN 网关） | `10.0.0.1/24` |
        | 客户端 | `10.0.0.2/24` 或 `10.0.0.x/24` |

        ---

        **解决方案**

        * 情况一：这是服务器

            编辑 `/etc/wireguard/wg0.conf`：

            ```ini
            [Interface]
            Address = 10.0.0.1/24   # 改成 .1
            PrivateKey = xxx
            ListenPort = 51820
            PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
            PreDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
            ```

            然后重启：

            ```bash
            sudo wg-quick down wg0
            sudo wg-quick up wg0
            ```

        * 情况二：这是客户端

            客户端用 `10.0.0.2` 是**正确的**，但需要检查是否有 `peer` 配置指向服务器：

            ```bash
            sudo wg show
            ```

            查看输出中是否有 `endpoint` 指向服务器 IP 和端口。

        **修复配置后，重新启动**

        ```bash
        sudo wg-quick down wg0
        sudo wg-quick up wg0
        ```

        正常情况应该看到：

        ```
        wg0: flags=209<UP,POINTOPOINT,RUNNING,NOARP>  mtu 1420
                inet 10.0.0.1/24 scope global wg0
                ...
        ```

        ---

        **还有一个问题**

        输出中 `TX errors 252` 也很高，通常表示：

        - 路由配置有问题
        - 对端（peer）没有正确响应
        - 防火墙阻止了 UDP 包

        修复 IP 配置后，再执行 `sudo wg show` 检查握手状态是否正常。

        ---

        **请先确认这台机器是服务器还是客户端，然后告诉我，我再给你具体的修复步骤。**

    * 这个应该 /32 还是 /24?

        ```conf
        ### 下面是客户端（Peer）的配置，有几个客户端就写几个 [Peer] ###
        [Peer]
        # 客户端的公钥（填入在客户端生成的 publickey 内容）
        PublicKey = fT9DPB5YW1UF5s70yQnwnpAOkVRZJRrzjxdKfmwKjx8=
        # 允许这个客户端使用的内网 IP
        AllowedIPs = 10.0.0.2/32

        [Peer]
        # 客户端的公钥（填入在客户端生成的 publickey 内容）
        PublicKey = ZV6SAwt+daUO7vu9zVmrSS0DQiMZ/sxzuIralpxoRnU=
        # 允许这个客户端使用的内网 IP
        AllowedIPs = 10.0.0.3/32
        ```

        **应该用 `/32`**，你现在的配置是**正确的**！

        ---

        **为什么是 `/32` 而不是 `/24`？**

        | 配置 | 含义 | 是否正确 |
        |------|------|----------|
        | `AllowedIPs = 10.0.0.2/32` | 只允许这个客户端使用 `10.0.0.2` 这一个 IP | ✅ **正确** |
        | `AllowedIPs = 10.0.0.2/24` | 允许这个客户端使用 `10.0.0.0~10.0.0.255` 所有 IP | ❌ 错误 |

        ---

        **详细解释**

        * `/32` 的作用

            - 每个 Peer 独占一个 IP
            - 服务器根据来源 IP 识别是哪个客户端
            - 客户端 A 只能用 `10.0.0.2`，客户端 B 只能用 `10.0.0.3`
            - **这是 WireGuard 的标准配置方式**

        * 如果用 `/24` 会怎样？

            - 客户端 A 可以冒充 `10.0.0.3`（客户端 B 的 IP）
            - 客户端 A 可以抢占用 `10.0.0.1`（服务器 IP）
            - 导致 IP 冲突、路由混乱、安全漏洞
            - **多个客户端无法正常工作**

        **完整的正确配置示例**

        * 服务器端 `/etc/wireguard/wg0.conf`

            ```ini
            [Interface]
            Address = 10.0.0.1/24    # 服务器自身用 /24
            PrivateKey = 服务器私钥
            ListenPort = 51820
            PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
            PreDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

            [Peer]
            PublicKey = fT9DPB5YW1UF5s70yQnwnpAOkVRZJRrzjxdKfmwKjx8=
            AllowedIPs = 10.0.0.2/32    # ✅ 客户端1只能用 .2

            [Peer]
            PublicKey = ZV6SAwt+daUO7vu9zVmrSS0DQiMZ/sxzuIralpxoRnU=
            AllowedIPs = 10.0.0.3/32    # ✅ 客户端2只能用 .3
            ```

        * 客户端1 `/etc/wireguard/wg0.conf`

            ```ini
            [Interface]
            Address = 10.0.0.2/24    # 客户端自己的 IP
            PrivateKey = 客户端1私钥
            DNS = 8.8.8.8

            [Peer]
            PublicKey = 服务器公钥
            Endpoint = 服务器公网IP:51820
            AllowedIPs = 0.0.0.0/0   # 所有流量走 VPN
            PersistentKeepalive = 25
            ```

        **总结**

        | 位置 | 配置 | 格式 |
        |------|------|------|
        | 服务器 `[Interface]` | `Address = 10.0.0.1/24` | 用 `/24` |
        | 服务器 `[Peer]` | `AllowedIPs = 10.0.0.x/32` | 用 `/32` |
        | 客户端 `[Interface]` | `Address = 10.0.0.x/24` | 用 `/24` |

        **你现在的配置完全正确，不需要修改！** 👍

        之前看到的 `destination 10.0.0.2` 问题，可能是客户端配置中 `Address` 设置为了 `10.0.0.2/24` 但没加 `/24`，或者是 `ip addr show` 显示的格式问题。你可以检查一下客户端的配置文件确认。

## note
